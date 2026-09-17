"""Volume-expansion core + second-condition candidates (research only).

Tests whether ``rvol_30 > 1.5`` alone, or paired with one anti-extension /
momentum filter, improves later +50%/+100% hit rates vs base rates.
Does not remove compression code; report may recommend reframing Model C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cmram.backtest.holdout import compute_time_split, membership_calendar_dates
from cmram.backtest.spike_backward import (
    DEFAULT_EXPLORE_FRAC,
    _eligible_membership,
)
from cmram.backtest.spike_detect import (
    DEFAULT_SPIKE_THRESHOLD,
    PRIMARY_WINDOW,
    SECONDARY_WINDOW,
    detect_spikes,
    split_spikes_by_cut,
)
from cmram.backtest.spike_indicators import compute_tech_indicators, snapshot_pre_spike
from cmram.backtest.spike_shared_setups import (
    SharedSetup,
    THRESH_100,
    THRESH_50,
    _control_flag_rows,
    _flags_at_signal_dates,
    _mask_for_flags,
    build_dual_threshold_panel,
    build_setup_flags,
    evaluate_setups_on_panel,
)

LAGOS = ZoneInfo("Africa/Lagos")
RVOL_CORE = "rvol_gt_1_5"
DEFAULT_MIN_FIRES = 5
DEFAULT_BEAT_LIFT = 1.10


@dataclass(frozen=True)
class VolumeCandidate:
    candidate_id: str
    label: str
    plain_english: str
    flag_ids: tuple[str, ...]


# Pre-registered candidates (locked before later-period look).
VOLUME_CANDIDATES: tuple[VolumeCandidate, ...] = (
    VolumeCandidate(
        "baseline_rvol",
        "Baseline: volume alone",
        "Relative volume > 1.5× 30d average",
        (RVOL_CORE,),
    ),
    VolumeCandidate(
        "A_not_mooned_10",
        "A: not already moon'd (≤10% above EMA20)",
        "Relative volume > 1.5× 30d average AND price not more than 10% above EMA20",
        (RVOL_CORE, "dist_ema20_le_10pct"),
    ),
    VolumeCandidate(
        "A_not_mooned_5",
        "A′: not already moon'd (≤5% above EMA20)",
        "Relative volume > 1.5× 30d average AND price not more than 5% above EMA20",
        (RVOL_CORE, "dist_ema20_le_5pct"),
    ),
    VolumeCandidate(
        "B_near_ema20",
        "B: reclaiming / near EMA20 (−5%…+10%)",
        "Relative volume > 1.5× 30d average AND price near EMA20 (between −5% and +10%)",
        (RVOL_CORE, "near_ema20_m5_p10"),
    ),
    VolumeCandidate(
        "C_macd_rising",
        "C: MACD hist > 0 and rising",
        "Relative volume > 1.5× 30d average AND MACD histogram positive and rising",
        (RVOL_CORE, "macd_hist_pos_rising"),
    ),
    VolumeCandidate(
        "D_rsi_45_65",
        "D: RSI 45–65 (not overbought blowoff)",
        "Relative volume > 1.5× 30d average AND RSI between 45 and 65",
        (RVOL_CORE, "rsi_45_65"),
    ),
)


@dataclass
class VolumeExpansionResult:
    spikes: pd.DataFrame
    discover: pd.DataFrame
    validate: pd.DataFrame
    flags_panel: pd.DataFrame
    candidates: list[SharedSetup] = field(default_factory=list)
    discover_support: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_50: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_100: pd.DataFrame = field(default_factory=pd.DataFrame)
    hot_vol_ret3d: dict[str, Any] = field(default_factory=dict)
    cut_date: pd.Timestamp | None = None
    split_info: dict[str, Any] = field(default_factory=dict)
    base_rate_50: float = float("nan")
    base_rate_100: float = float("nan")
    n_eligible_assets: int = 0
    summary: dict[str, Any] = field(default_factory=dict)


def candidates_as_shared_setups() -> list[SharedSetup]:
    """Build SharedSetup shells for evaluation (support filled later)."""
    out: list[SharedSetup] = []
    for c in VOLUME_CANDIDATES:
        out.append(
            SharedSetup(
                setup_id=c.candidate_id,
                plain_english=c.plain_english,
                flag_ids=c.flag_ids,
                spike_support=float("nan"),
                ctrl_support=float("nan"),
                lift=float("nan"),
                spike_count=0,
                spike_n=0,
                ctrl_count=0,
                ctrl_n=0,
                is_pair=len(c.flag_ids) > 1,
            )
        )
    return out


def compute_ret_3d(market: pd.DataFrame) -> pd.DataFrame:
    """Per-asset 3-day simple return: close / close.shift(3) - 1."""
    df = market.copy()
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    df["asset_id"] = df["asset_id"].astype(str)
    df = df.sort_values(["asset_id", "timestamp"], kind="mergesort")
    close = pd.to_numeric(df["close"], errors="coerce")
    prior = close.groupby(df["asset_id"], sort=False).shift(3)
    ret = close / prior - 1.0
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "asset_id": df["asset_id"].to_numpy(),
            "ret_3d": ret.to_numpy(),
        }
    )


def summarize_hot_vol_ret3d(
    spike_t1: pd.DataFrame,
    *,
    rvol_col: str = "rvol_30",
    rvol_min: float = 1.5,
    ret_col: str = "ret_3d",
) -> dict[str, Any]:
    """On spike T−1 rows: among hot-volume days, share with ret_3d≤0 vs >0.

    Documents Abdul's read that hot volume rarely sits with flat/down price —
    usually price is already ticking up.
    """
    empty = {
        "n_spike_t1": 0,
        "n_hot_vol": 0,
        "n_hot_with_ret": 0,
        "n_ret_le_0": 0,
        "n_ret_gt_0": 0,
        "share_ret_le_0": float("nan"),
        "share_ret_gt_0": float("nan"),
        "median_ret_3d_hot": float("nan"),
        "median_ret_3d_all_t1": float("nan"),
        "n_all_with_ret": 0,
        "share_all_ret_gt_0": float("nan"),
    }
    if spike_t1 is None or len(spike_t1) == 0:
        return empty

    df = spike_t1.copy()
    empty["n_spike_t1"] = int(len(df))
    if rvol_col not in df.columns or ret_col not in df.columns:
        return empty

    rvol = pd.to_numeric(df[rvol_col], errors="coerce")
    ret = pd.to_numeric(df[ret_col], errors="coerce")
    hot = rvol > float(rvol_min)
    n_hot = int(hot.fillna(False).sum())
    empty["n_hot_vol"] = n_hot

    all_ret = ret.dropna()
    empty["n_all_with_ret"] = int(len(all_ret))
    if len(all_ret):
        empty["median_ret_3d_all_t1"] = float(all_ret.median())
        empty["share_all_ret_gt_0"] = float((all_ret > 0).mean())

    hot_ret = ret.loc[hot.fillna(False)].dropna()
    empty["n_hot_with_ret"] = int(len(hot_ret))
    if len(hot_ret) == 0:
        return empty

    n_le = int((hot_ret <= 0).sum())
    n_gt = int((hot_ret > 0).sum())
    empty["n_ret_le_0"] = n_le
    empty["n_ret_gt_0"] = n_gt
    empty["share_ret_le_0"] = n_le / len(hot_ret)
    empty["share_ret_gt_0"] = n_gt / len(hot_ret)
    empty["median_ret_3d_hot"] = float(hot_ret.median())
    return empty


def _fill_candidate_support(
    setups: list[SharedSetup],
    spike_flags: pd.DataFrame,
    control_flags: pd.DataFrame,
) -> tuple[list[SharedSetup], pd.DataFrame]:
    """Attach discover-period spike vs control support onto candidate shells."""
    n_s = int(len(spike_flags))
    n_c = int(len(control_flags))
    filled: list[SharedSetup] = []
    rows: list[dict[str, Any]] = []
    for setup in setups:
        s_mask = _mask_for_flags(spike_flags, setup.flag_ids)
        c_mask = _mask_for_flags(control_flags, setup.flag_ids)
        s_cnt = int(s_mask.sum())
        c_cnt = int(c_mask.sum())
        s_sup = s_cnt / n_s if n_s else float("nan")
        c_sup = c_cnt / n_c if n_c else float("nan")
        lift = (
            s_sup / c_sup
            if (n_s and n_c and c_sup and c_sup > 0 and pd.notna(s_sup))
            else float("nan")
        )
        filled.append(
            SharedSetup(
                setup_id=setup.setup_id,
                plain_english=setup.plain_english,
                flag_ids=setup.flag_ids,
                spike_support=s_sup,
                ctrl_support=c_sup,
                lift=lift,
                spike_count=s_cnt,
                spike_n=n_s,
                ctrl_count=c_cnt,
                ctrl_n=n_c,
                is_pair=setup.is_pair,
            )
        )
        rows.append(
            {
                "candidate_id": setup.setup_id,
                "plain_english": setup.plain_english,
                "flag_ids": "|".join(setup.flag_ids),
                "spike_count": s_cnt,
                "spike_n": n_s,
                "spike_support": s_sup,
                "ctrl_count": c_cnt,
                "ctrl_n": n_c,
                "ctrl_support": c_sup,
                "lift": lift,
            }
        )
    return filled, pd.DataFrame(rows)


def _best_later_vs_baseline(
    val: pd.DataFrame,
    *,
    baseline_id: str = "baseline_rvol",
) -> dict[str, Any]:
    """Pick candidate with highest later lift among those with enough fires."""
    if val is None or len(val) == 0:
        return {
            "best_id": None,
            "best_lift": float("nan"),
            "baseline_lift": float("nan"),
            "improves_vs_baseline": False,
        }
    base_rows = val.loc[val["setup_id"] == baseline_id]
    base_lift = (
        float(base_rows["lift"].iloc[0]) if len(base_rows) else float("nan")
    )
    ok = val.loc[val["n_fire_ok"].fillna(False)].copy()
    if ok.empty:
        return {
            "best_id": None,
            "best_lift": float("nan"),
            "baseline_lift": base_lift,
            "improves_vs_baseline": False,
        }
    ok = ok.sort_values(
        ["lift", "n_hit", "n_fire"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    best = ok.iloc[0]
    best_id = str(best["setup_id"])
    best_lift = float(best["lift"])
    improves = (
        best_id != baseline_id
        and pd.notna(best_lift)
        and pd.notna(base_lift)
        and best_lift > base_lift
    )
    return {
        "best_id": best_id,
        "best_lift": best_lift,
        "baseline_lift": base_lift,
        "improves_vs_baseline": bool(improves),
    }



def _baseline_beats(val: pd.DataFrame, baseline_id: str = "baseline_rvol") -> bool:
    if val is None or len(val) == 0 or "setup_id" not in val.columns:
        return False
    rows = val.loc[val["setup_id"] == baseline_id]
    if rows.empty or "beats_base" not in rows.columns:
        return False
    return bool(rows["beats_base"].iloc[0])


def run_volume_expansion_study(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    features: pd.DataFrame | None = None,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    window: int = PRIMARY_WINDOW,
    secondary_window: int = SECONDARY_WINDOW,
    explore_frac: float = DEFAULT_EXPLORE_FRAC,
    feature_model: str = "C",
    control_seed: int = 42,
    max_control: int | None = 5000,
    min_fires: int = DEFAULT_MIN_FIRES,
    beat_lift: float = DEFAULT_BEAT_LIFT,
) -> VolumeExpansionResult:
    """Discover-descriptive + later validation for volume-expansion candidates."""
    eligible = _eligible_membership(membership)
    n_assets = int(eligible["asset_id"].nunique()) if len(eligible) else 0

    spikes = detect_spikes(
        market,
        threshold=threshold,
        window=window,
        secondary_window=secondary_window,
        eligible=eligible,
    )
    indicators = compute_tech_indicators(market)
    flags_panel = build_setup_flags(indicators, features, model=feature_model)
    spikes_ind = snapshot_pre_spike(spikes, indicators, features, model=feature_model)

    # Attach ret_3d at T−1 for Abdul hot-vol question
    ret3 = compute_ret_3d(market)
    ret3 = ret3.rename(columns={"timestamp": "signal_date"})
    spikes_ind = spikes_ind.merge(ret3, on=["signal_date", "asset_id"], how="left")

    dates = membership_calendar_dates(eligible if len(eligible) else membership)
    split_info = compute_time_split(dates, explore_frac=explore_frac)
    cut = split_info.get("cut_date")

    discover, validate = (
        split_spikes_by_cut(spikes_ind, cut)
        if cut is not None
        else (spikes_ind.copy(), spikes_ind.iloc[0:0].copy())
    )

    spike_flags_disc = _flags_at_signal_dates(discover, flags_panel)
    panel = build_dual_threshold_panel(market, eligible, window=window)
    control = _control_flag_rows(
        panel,
        flags_panel,
        before=cut,
        n=max_control,
        seed=control_seed,
    )

    shells = candidates_as_shared_setups()
    candidates, discover_support = _fill_candidate_support(
        shells, spike_flags_disc, control
    )

    # Hot vol × ret_3d on all spike T−1 (and discover-only for honesty)
    hot_all = summarize_hot_vol_ret3d(spikes_ind)
    hot_disc = summarize_hot_vol_ret3d(discover)
    hot_val = summarize_hot_vol_ret3d(validate)
    hot_vol_ret3d = {
        "all": hot_all,
        "discover": hot_disc,
        "validate": hot_val,
    }

    if cut is not None:
        panel_v = panel.loc[
            pd.to_datetime(panel["timestamp"]).dt.normalize()
            >= pd.Timestamp(cut).normalize()
        ].copy()
    else:
        panel_v = panel.iloc[0:0].copy()

    from cmram.backtest.spike_shared_setups import available_flag_ids

    flag_cols = available_flag_ids(flags_panel)
    fl = flags_panel.copy()
    fl["timestamp"] = pd.to_datetime(fl["timestamp"]).dt.normalize()
    fl["asset_id"] = fl["asset_id"].astype(str)
    panel_v["timestamp"] = pd.to_datetime(panel_v["timestamp"]).dt.normalize()
    panel_v["asset_id"] = panel_v["asset_id"].astype(str)
    panel_v_flags = panel_v.merge(
        fl[["timestamp", "asset_id", *flag_cols]],
        on=["timestamp", "asset_id"],
        how="left",
    )
    for c in flag_cols:
        panel_v_flags[c] = panel_v_flags[c].fillna(False).astype(bool)

    base_50 = (
        float(panel_v_flags["is_spike_50"].mean())
        if len(panel_v_flags)
        else float("nan")
    )
    base_100 = (
        float(panel_v_flags["is_spike_100"].mean())
        if len(panel_v_flags)
        else float("nan")
    )
    val_50 = evaluate_setups_on_panel(
        panel_v_flags,
        candidates,
        label_col="is_spike_50",
        min_fires=min_fires,
        beat_lift=beat_lift,
    )
    val_100 = evaluate_setups_on_panel(
        panel_v_flags,
        candidates,
        label_col="is_spike_100",
        min_fires=min_fires,
        beat_lift=beat_lift,
    )

    best_50 = _best_later_vs_baseline(val_50)
    best_100 = _best_later_vs_baseline(val_100)

    # Plain answer: which second condition (if any) improves later lift vs volume alone
    improvers_50 = []
    improvers_100 = []
    if len(val_50) and "baseline_rvol" in set(val_50["setup_id"]):
        b50 = float(
            val_50.loc[val_50["setup_id"] == "baseline_rvol", "lift"].iloc[0]
        )
        for _, r in val_50.iterrows():
            if r["setup_id"] == "baseline_rvol":
                continue
            if (
                bool(r["n_fire_ok"])
                and pd.notna(r["lift"])
                and pd.notna(b50)
                and float(r["lift"]) > b50
            ):
                improvers_50.append(str(r["setup_id"]))
    if len(val_100) and "baseline_rvol" in set(val_100["setup_id"]):
        b100 = float(
            val_100.loc[val_100["setup_id"] == "baseline_rvol", "lift"].iloc[0]
        )
        for _, r in val_100.iterrows():
            if r["setup_id"] == "baseline_rvol":
                continue
            if (
                bool(r["n_fire_ok"])
                and pd.notna(r["lift"])
                and pd.notna(b100)
                and float(r["lift"]) > b100
            ):
                improvers_100.append(str(r["setup_id"]))

    summary = {
        "n_eligible_assets": n_assets,
        "n_spikes_total": int(len(spikes)),
        "n_spikes_discover": int(len(discover)),
        "n_spikes_validate": int(len(validate)),
        "n_control_discover": int(len(control)),
        "cut_date": str(pd.Timestamp(cut).date()) if cut is not None else None,
        "threshold_discover": threshold,
        "window": window,
        "secondary_window": secondary_window,
        "base_rate_50": base_50,
        "base_rate_100": base_100,
        "best_50": best_50,
        "best_100": best_100,
        "improvers_50": improvers_50,
        "improvers_100": improvers_100,
        "any_improves_50": bool(improvers_50),
        "any_improves_100": bool(improvers_100),
        "baseline_beats_50": _baseline_beats(val_50),
        "baseline_beats_100": _baseline_beats(val_100),
    }

    return VolumeExpansionResult(
        spikes=spikes_ind,
        discover=discover,
        validate=validate,
        flags_panel=flags_panel,
        candidates=candidates,
        discover_support=discover_support,
        validation_50=val_50,
        validation_100=val_100,
        hot_vol_ret3d=hot_vol_ret3d,
        cut_date=pd.Timestamp(cut).normalize() if cut is not None else None,
        split_info=split_info,
        base_rate_50=base_50,
        base_rate_100=base_100,
        n_eligible_assets=n_assets,
        summary=summary,
    )


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _label_for_id(cid: str) -> str:
    for c in VOLUME_CANDIDATES:
        if c.candidate_id == cid:
            return c.label
    return cid


def write_volume_expansion_report(
    result: VolumeExpansionResult,
    path: Path | str,
) -> Path:
    """Plain-language markdown report — no alpha claims."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M %Z")
    s = result.summary
    split = result.split_info
    hot = result.hot_vol_ret3d.get("all", {})

    lines: list[str] = []
    lines.append("# Volume expansion (+ second condition) — Abdul check")
    lines.append("")
    lines.append(f"**When:** {now} (Africa/Lagos)")
    lines.append(
        "**Status:** research check only — **not alpha**, not a trading system, "
        "not a live bot"
    )
    lines.append(
        "Abdul's read: **volume expansion** matters; compression may be the wrong "
        "lead. Data often shows hot volume with price already ticking up — rarely "
        "hot volume with flat/down price."
    )
    lines.append("")

    lines.append("## Short answers")
    lines.append("")
    lines.append(
        f"- Spikes (≥ +50% in {s.get('window', 7)}d): **{s.get('n_spikes_total', 0)}** "
        f"total; **{s.get('n_spikes_discover', 0)}** earlier / "
        f"**{s.get('n_spikes_validate', 0)}** later "
        f"(eligible assets ≈ **{s.get('n_eligible_assets', 0)}**)."
    )
    lines.append(f"- Cut date (first later day): **{s.get('cut_date', 'n/a')}**.")
    lines.append(
        f"- Core flag: **rvol_30 > 1.5** (same as shared-setup `rvol_gt_1_5`)."
    )

    # Plain verdict on second conditions
    imp50 = s.get("improvers_50") or []
    imp100 = s.get("improvers_100") or []
    if not len(result.validation_50):
        verdict = (
            "Could not run later check (empty later panel). No conclusion on "
            "second conditions."
        )
        short_help = verdict
    elif not imp50 and not imp100:
        verdict = (
            "**No second condition improved later lift versus volume-alone** on "
            "+50% or +100% (among candidates with ≥5 later fires). "
            "Volume expansion alone remains the cleaner later reading in this "
            "sample — still **not alpha**."
        )
        short_help = verdict
    else:
        # Rank improvers by later +50% lift for a plain winner line
        ranked = []
        if len(result.validation_50):
            for cid in imp50:
                row = result.validation_50.loc[
                    result.validation_50["setup_id"] == cid
                ]
                if len(row):
                    ranked.append((cid, float(row["lift"].iloc[0])))
            ranked.sort(key=lambda x: x[1], reverse=True)
        winner = ranked[0][0] if ranked else (imp50[0] if imp50 else None)
        winner_lift = ranked[0][1] if ranked else float("nan")
        base_l = (s.get("best_50") or {}).get("baseline_lift", float("nan"))
        others = [c for c, _ in ranked[1:]] if ranked else []
        short_help = (
            f"**Best later add-on: {_label_for_id(winner)}** "
            f"(+50% lift {_fmt_num(winner_lift)}x vs volume-alone "
            f"{_fmt_num(base_l)}x)"
        )
        if others:
            short_help += (
                "; also above volume-alone: "
                + ", ".join(_label_for_id(x) for x in others)
            )
        if "C_macd_rising" not in imp50 and "C_macd_rising" not in imp100:
            short_help += (
                ". **C (MACD rising) did not improve** vs volume-alone"
            )
        short_help += ". Small sample — **no alpha claim**."
        verdict = short_help
    lines.append(f"- **Which second condition helps?** {short_help}")
    lines.append("")

    # Hot vol / ret_3d short answer
    if hot.get("n_hot_with_ret", 0):
        lines.append(
            f"- On spike T−1 days with hot volume (rvol>1.5): "
            f"**{_fmt_pct(hot.get('share_ret_gt_0'))}** already had "
            f"**ret_3d > 0** (price ticking up); "
            f"**{_fmt_pct(hot.get('share_ret_le_0'))}** had ret_3d ≤ 0 "
            f"(n={hot.get('n_hot_with_ret')} hot-vol T−1 days with ret_3d). "
            "That supports Abdul's read that hot volume rarely sits with flat price."
        )
    else:
        lines.append(
            "- Hot-volume × ret_3d on spike T−1: **n/a** (no overlapping rows)."
        )
    lines.append("")

    lines.append("### Overall")
    lines.append("")
    base_note = (
        f"Volume-alone later lift: +50% → {_fmt_num((s.get('best_50') or {}).get('baseline_lift'))}x; "
        f"+100% → {_fmt_num((s.get('best_100') or {}).get('baseline_lift'))}x "
        f"(base rates {_fmt_pct(result.base_rate_50)} / {_fmt_pct(result.base_rate_100)})."
    )
    lines.append(verdict)
    lines.append("")
    lines.append(base_note)
    lines.append("")
    lines.append(
        "**Model note:** consider reframing Model C toward **volume expansion** "
        "in future design. **Do not rip out compression code yet** — keep it "
        "available until a volume-led redesign is specified and re-tested."
    )
    lines.append("")

    lines.append("## What we did (plain)")
    lines.append("")
    lines.append(
        "1. Locked **volume expansion** as the core: `rvol_30 > 1.5`."
    )
    lines.append(
        "2. Pre-registered **one second condition at a time** (A ≤10% / A′ ≤5% "
        "above EMA20; B near EMA20 −5%…+10%; C MACD hist >0 rising; D RSI 45–65), "
        "plus volume-alone baseline — no fishing for extra combos."
    )
    lines.append(
        "3. On **earlier** spikes only, measured how often each candidate was true "
        "at T−1 vs normal days (descriptive support / lift)."
    )
    lines.append(
        "4. On the **later** period only (same cut ≈ 2026-08-21), measured hit rates "
        f"for **+50%** and **+100%** within {s.get('window', 7)}d when each "
        "candidate fires, vs everyday base rates and vs volume-alone lift."
    )
    lines.append(
        "5. On spike T−1 days, counted hot-volume rows with **ret_3d ≤ 0** vs "
        "**ret_3d > 0** (Abdul's flat-vs-ticking-up question)."
    )
    lines.append("")

    lines.append("## Date split")
    lines.append("")
    lines.append(f"- Membership calendar days: **{split.get('n_days', 0)}**")
    if split.get("cut_date") is not None:
        lines.append(
            f"- **Cut date:** `{pd.Timestamp(split['cut_date']).date()}` "
            f"(discover `{pd.Timestamp(split['explore_start']).date()}` → "
            f"`{pd.Timestamp(split['explore_end']).date()}`; "
            f"validate `{pd.Timestamp(split['holdout_start']).date()}` → "
            f"`{pd.Timestamp(split['holdout_end']).date()}`)"
        )
    lines.append(
        f"- Explore fraction: **{split.get('explore_frac', DEFAULT_EXPLORE_FRAC)}**"
    )
    lines.append(
        f"- Control non-spike days (earlier): **{s.get('n_control_discover', 0)}**"
    )
    lines.append("")

    lines.append("## Spike counts")
    lines.append("")
    lines.append("| set | n spikes | unique assets |")
    lines.append("|------|------|------|")
    for label, df in (
        ("all", result.spikes),
        ("discover", result.discover),
        ("validate", result.validate),
    ):
        n_a = int(df["asset_id"].nunique()) if len(df) else 0
        lines.append(f"| {label} | {len(df)} | {n_a} |")
    lines.append("")
    lines.append("- Sample sizes are **small**. Treat every percentage with caution.")
    lines.append("")

    lines.append("## Abdul check: hot volume vs ret_3d on spike T−1")
    lines.append("")
    lines.append(
        "Question: when volume is already hot the day before a spike, is price "
        "usually flat/down (`ret_3d ≤ 0`) or already ticking up (`ret_3d > 0`)?"
    )
    lines.append("")
    lines.append("| set | hot-vol T−1 (with ret_3d) | ret_3d > 0 | ret_3d ≤ 0 | median ret_3d (hot) |")
    lines.append("|------|------|------|------|------|")
    for label, h in (
        ("all spikes", result.hot_vol_ret3d.get("all", {})),
        ("discover", result.hot_vol_ret3d.get("discover", {})),
        ("validate", result.hot_vol_ret3d.get("validate", {})),
    ):
        lines.append(
            f"| {label} | {h.get('n_hot_with_ret', 0)} / hot={h.get('n_hot_vol', 0)} "
            f"(of {h.get('n_spike_t1', 0)} T−1) | "
            f"{_fmt_pct(h.get('share_ret_gt_0'))} ({h.get('n_ret_gt_0', 0)}) | "
            f"{_fmt_pct(h.get('share_ret_le_0'))} ({h.get('n_ret_le_0', 0)}) | "
            f"{_fmt_pct(h.get('median_ret_3d_hot'))} |"
        )
    lines.append("")
    if hot.get("n_all_with_ret"):
        lines.append(
            f"For context, among **all** spike T−1 days with ret_3d "
            f"(n={hot.get('n_all_with_ret')}): share ret_3d>0 = "
            f"{_fmt_pct(hot.get('share_all_ret_gt_0'))}, "
            f"median ret_3d = {_fmt_pct(hot.get('median_ret_3d_all_t1'))}."
        )
        lines.append("")

    lines.append("## Candidates (locked)")
    lines.append("")
    lines.append("| id | label | flags |")
    lines.append("|------|------|------|")
    for c in VOLUME_CANDIDATES:
        lines.append(
            f"| {c.candidate_id} | {c.label} | `{'` AND `'.join(c.flag_ids)}` |"
        )
    lines.append("")

    lines.append("## Discover support (earlier spikes only)")
    lines.append("")
    lines.append(
        "Support = share of earlier spike T−1 days where the candidate was true. "
        "Lift = spike support ÷ control support. Descriptive only — not validation."
    )
    lines.append("")
    if len(result.discover_support):
        lines.append(
            "| candidate | spike support | control support | lift | spike hits |"
        )
        lines.append("|------|------|------|------|------|")
        for _, r in result.discover_support.iterrows():
            lines.append(
                f"| {r['candidate_id']} | {_fmt_pct(r['spike_support'])} | "
                f"{_fmt_pct(r['ctrl_support'])} | {_fmt_num(r['lift'])}x | "
                f"{int(r['spike_count'])}/{int(r['spike_n'])} |"
            )
        lines.append("")
    else:
        lines.append("_No discover support rows._")
        lines.append("")

    lines.append(
        "**Tension to note:** on earlier *spike T−1* days, hot volume often "
        "already sat with price well above EMA20 (see Abdul ret_3d check — "
        "median hot-vol ret_3d was large). So A/B (“not extended / near EMA”) "
        "rarely co-occurred with hot volume on those historical pre-spike days "
        "(low discover support). The **later** panel instead asks: when the "
        "combo fires on ordinary eligible days, does the hit rate improve? "
        "Those are different questions."
    )
    lines.append("")

    lines.append("## Later check: +50% and +100% within 7d")
    lines.append("")
    lines.append(
        f"Base rate on later eligible days: "
        f"**+50%** → {_fmt_pct(result.base_rate_50)}; "
        f"**+100%** → {_fmt_pct(result.base_rate_100)}."
    )
    lines.append("")
    lines.append(
        "Compare each second condition's **lift** to **baseline_rvol**. "
        "We mark **beats base** when fires ≥ 5 and lift ≥ 1.10 vs the everyday "
        "base rate (same rule as shared setups)."
    )
    lines.append("")

    def _val_table(vt: pd.DataFrame, label: str, baseline_lift: float) -> None:
        lines.append(f"### {label}")
        lines.append("")
        if not len(vt):
            lines.append("_No validation rows._")
            lines.append("")
            return
        lines.append(
            "| candidate | fires | hits | hit rate | base | lift | vs vol-alone | beats base? |"
        )
        lines.append("|------|------|------|------|------|------|------|------|")
        for _, r in vt.iterrows():
            beat = (
                "yes"
                if r["beats_base"]
                else ("n/a (few fires)" if not r["n_fire_ok"] else "no")
            )
            if r["setup_id"] == "baseline_rvol":
                vs = "—"
            elif (
                bool(r["n_fire_ok"])
                and pd.notna(r["lift"])
                and pd.notna(baseline_lift)
            ):
                delta = float(r["lift"]) - float(baseline_lift)
                vs = f"{delta:+.2f}x lift"
            else:
                vs = "n/a"
            lines.append(
                f"| {r['setup_id']} | {int(r['n_fire'])} | {int(r['n_hit'])} | "
                f"{_fmt_pct(r['hit_rate'])} | {_fmt_pct(r['base_rate'])} | "
                f"{_fmt_num(r['lift'])}x | {vs} | {beat} |"
            )
        lines.append("")

    b50 = (s.get("best_50") or {}).get("baseline_lift", float("nan"))
    b100 = (s.get("best_100") or {}).get("baseline_lift", float("nan"))
    _val_table(result.validation_50, f"+50% within {s.get('window', 7)}d", b50)
    _val_table(result.validation_100, f"+100% within {s.get('window', 7)}d", b100)

    lines.append("## Recommendation (research process, not a trade)")
    lines.append("")
    lines.append(
        "- Treat **volume expansion (`rvol_30 > 1.5`)** as the primary lead to "
        "stress-test next — consistent with shared-setups later lift and Abdul's read."
    )
    if imp50 or imp100:
        winner = (s.get("best_50") or {}).get("best_id")
        if winner and winner != "baseline_rvol":
            lines.append(
                f"- Strongest later add-on in this cut: **{_label_for_id(winner)}**. "
                "Keep for a **repeat check** on a longer window / more spikes — "
                "do not promote yet."
            )
        else:
            lines.append(
                "- Keep the second condition(s) that improved later lift for a "
                "**repeat check** on a longer window / more spikes — do not promote yet."
            )
        lines.append(
            "- **C (MACD hist rising)** did not beat volume-alone later; do not "
            "treat it as a required second gate on this evidence."
        )
    else:
        lines.append(
            "- In this cut, **adding** EMA / MACD / RSI filters did **not** clearly "
            "raise later lift vs volume alone; prefer the simpler core until more data."
        )
    lines.append(
        "- **Future Model C design:** consider centering on volume expansion "
        "(and optionally a light “not already extended” gate) rather than "
        "compression-first. **Leave compression code in place** for now."
    )
    lines.append("")

    lines.append("## Honesty box")
    lines.append("")
    lines.append(
        f"- Discover spikes: **{s.get('n_spikes_discover', 0)}**. "
        f"Later spikes: **{s.get('n_spikes_validate', 0)}**. Thin sample."
    )
    lines.append(
        "- Candidates were locked before the later check; still one short history "
        "and overlapping crypto themes — easy to overfit."
    )
    lines.append(
        "- Hot-vol × ret_3d is **descriptive** on days that already preceded spikes "
        "(selected on the outcome). It does not prove causality."
    )
    lines.append("- **No alpha claim.** Do not treat this as a system.")
    lines.append(
        "- Compression feature code is unchanged; this report only recommends "
        "a future redesign discussion."
    )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_volume_expansion_outputs(
    result: VolumeExpansionResult,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Save candidate support / validation parquet under data/volume_expansion."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if len(result.discover_support):
        p = out_dir / "volume_expansion_discover_support.parquet"
        result.discover_support.to_parquet(p, index=False)
        paths["discover_support"] = p
    if len(result.validation_50):
        p = out_dir / "volume_expansion_val_50.parquet"
        result.validation_50.to_parquet(p, index=False)
        paths["val_50"] = p
    if len(result.validation_100):
        p = out_dir / "volume_expansion_val_100.parquet"
        result.validation_100.to_parquet(p, index=False)
        paths["val_100"] = p
    # Hot vol summary as tiny parquet
    rows = []
    for split_name, h in result.hot_vol_ret3d.items():
        rows.append({"split": split_name, **h})
    if rows:
        p = out_dir / "volume_expansion_hot_vol_ret3d.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        paths["hot_vol_ret3d"] = p
    return paths
