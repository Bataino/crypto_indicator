"""Fair-test Abdul's locked draft formula (no Bollinger).

Locked core (signal day when BOTH true):
  1. rvol_30 > 1.5
  2. dist_ema_20 <= 0.05

Optional soft filters reported WITH and WITHOUT:
  - RSI <= 70 (primary soft); RSI 45–65 as secondary band
  - Gap > -10 if available (optional; sentiment weak)

Compares locked core vs volume-alone and not-stretched-alone.
Primary metrics on later period (cut ~2026-08-21); alternate cut secondary.
Research only — no alpha claims.
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
    _control_flag_rows,
    _flags_at_signal_dates,
    available_flag_ids,
    build_dual_threshold_panel,
    build_setup_flags,
    evaluate_setups_on_panel,
)
from cmram.backtest.volume_expansion import (
    DEFAULT_BEAT_LIFT,
    DEFAULT_MIN_FIRES,
    RVOL_CORE,
    _fill_candidate_support,
)

LAGOS = ZoneInfo("Africa/Lagos")
LOCKED_DIST = "dist_ema20_le_5pct"
PRIMARY_EXPLORE_FRAC = 0.70  # yields cut ≈ 2026-08-21 on current membership
ALT_EXPLORE_FRAC = 0.60  # secondary robustness (earlier cut / longer later)


@dataclass(frozen=True)
class DraftCandidate:
    candidate_id: str
    label: str
    plain_english: str
    flag_ids: tuple[str, ...]
    role: str  # core | baseline | soft | soft_secondary | soft_optional


# Locked before later look — no fishing.
DRAFT_CANDIDATES: tuple[DraftCandidate, ...] = (
    DraftCandidate(
        "locked_core",
        "Locked draft (vol + not stretched ≤5%)",
        "Relative volume > 1.5× 30d average AND price not more than 5% above EMA20",
        (RVOL_CORE, LOCKED_DIST),
        "core",
    ),
    DraftCandidate(
        "volume_alone",
        "Volume alone",
        "Relative volume > 1.5× 30d average",
        (RVOL_CORE,),
        "baseline",
    ),
    DraftCandidate(
        "not_stretched_alone",
        "Not-stretched alone (≤5% above EMA20)",
        "Price not more than 5% above EMA20",
        (LOCKED_DIST,),
        "baseline",
    ),
    DraftCandidate(
        "locked_plus_rsi70",
        "Locked + RSI ≤ 70",
        "Locked draft AND RSI at or below 70",
        (RVOL_CORE, LOCKED_DIST, "rsi_le_70"),
        "soft",
    ),
    DraftCandidate(
        "locked_plus_rsi45_65",
        "Locked + RSI 45–65 (secondary)",
        "Locked draft AND RSI between 45 and 65",
        (RVOL_CORE, LOCKED_DIST, "rsi_45_65"),
        "soft_secondary",
    ),
    DraftCandidate(
        "locked_plus_gap_m10",
        "Locked + Gap > −10 (optional)",
        "Locked draft AND Rotation Gap > −10 (optional soft; sentiment weak)",
        (RVOL_CORE, LOCKED_DIST, "gap_gt_m10"),
        "soft_optional",
    ),
    DraftCandidate(
        "locked_plus_rsi70_gap",
        "Locked + RSI ≤ 70 + Gap > −10 (optional)",
        "Locked draft AND RSI ≤ 70 AND Gap > −10 (optional combo)",
        (RVOL_CORE, LOCKED_DIST, "rsi_le_70", "gap_gt_m10"),
        "soft_optional",
    ),
)


@dataclass
class CutSliceResult:
    explore_frac: float
    cut_date: pd.Timestamp | None
    split_info: dict[str, Any]
    discover_support: pd.DataFrame
    validation_50: pd.DataFrame
    validation_100: pd.DataFrame
    base_rate_50: float
    base_rate_100: float
    candidates: list[SharedSetup]
    summary: dict[str, Any] = field(default_factory=dict)
    label: str = "primary"


@dataclass
class DraftFormulaFairTestResult:
    spikes: pd.DataFrame
    flags_panel: pd.DataFrame
    primary: CutSliceResult
    alternate: CutSliceResult | None = None
    n_eligible_assets: int = 0
    summary: dict[str, Any] = field(default_factory=dict)


def candidates_as_shared_setups() -> list[SharedSetup]:
    out: list[SharedSetup] = []
    for c in DRAFT_CANDIDATES:
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


def _row_for(val: pd.DataFrame, setup_id: str) -> dict[str, Any] | None:
    if val is None or len(val) == 0 or "setup_id" not in val.columns:
        return None
    rows = val.loc[val["setup_id"] == setup_id]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _beats_chance(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    return bool(row.get("beats_base"))


def _lift_of(row: dict[str, Any] | None) -> float:
    if not row:
        return float("nan")
    v = row.get("lift")
    return float(v) if v is not None and pd.notna(v) else float("nan")


def _soft_helps_vs_core(
    val: pd.DataFrame,
    soft_id: str,
    *,
    core_id: str = "locked_core",
) -> bool:
    """True if soft filter has enough fires and higher later lift than locked core."""
    soft = _row_for(val, soft_id)
    core = _row_for(val, core_id)
    if not soft or not core:
        return False
    if not bool(soft.get("n_fire_ok")):
        return False
    s_l = _lift_of(soft)
    c_l = _lift_of(core)
    return bool(pd.notna(s_l) and pd.notna(c_l) and s_l > c_l)


def _core_beats_baselines(val: pd.DataFrame) -> dict[str, Any]:
    """Does locked core beat chance and both single-condition baselines?"""
    core = _row_for(val, "locked_core")
    vol = _row_for(val, "volume_alone")
    stretch = _row_for(val, "not_stretched_alone")
    c_l = _lift_of(core)
    v_l = _lift_of(vol)
    s_l = _lift_of(stretch)
    fire_ok = bool(core and core.get("n_fire_ok"))
    beats_chance = _beats_chance(core)
    beats_vol = (
        fire_ok and pd.notna(c_l) and pd.notna(v_l) and c_l > v_l
    )
    beats_stretch = (
        fire_ok and pd.notna(c_l) and pd.notna(s_l) and c_l > s_l
    )
    return {
        "beats_chance": beats_chance,
        "beats_volume_alone": bool(beats_vol),
        "beats_not_stretched_alone": bool(beats_stretch),
        "core_lift": c_l,
        "volume_lift": v_l,
        "not_stretched_lift": s_l,
        "n_fire": int(core["n_fire"]) if core else 0,
        "n_hit": int(core["n_hit"]) if core else 0,
        "hit_rate": float(core["hit_rate"]) if core and pd.notna(core.get("hit_rate")) else float("nan"),
        "n_fire_ok": fire_ok,
    }


def _evaluate_cut(
    *,
    market: pd.DataFrame,
    eligible: pd.DataFrame,
    spikes_ind: pd.DataFrame,
    flags_panel: pd.DataFrame,
    explore_frac: float,
    label: str,
    control_seed: int,
    max_control: int | None,
    min_fires: int,
    beat_lift: float,
    window: int,
) -> CutSliceResult:
    dates = membership_calendar_dates(eligible)
    split_info = compute_time_split(dates, explore_frac=explore_frac)
    cut = split_info.get("cut_date")

    discover, _validate_spikes = (
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
    # Rename candidate_id column already present from _fill_candidate_support

    if cut is not None:
        panel_v = panel.loc[
            pd.to_datetime(panel["timestamp"]).dt.normalize()
            >= pd.Timestamp(cut).normalize()
        ].copy()
    else:
        panel_v = panel.iloc[0:0].copy()

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

    core_50 = _core_beats_baselines(val_50)
    core_100 = _core_beats_baselines(val_100)

    soft_ids = [
        c.candidate_id
        for c in DRAFT_CANDIDATES
        if c.role in ("soft", "soft_secondary", "soft_optional")
    ]
    soft_help_50 = {
        sid: _soft_helps_vs_core(val_50, sid) for sid in soft_ids
    }
    soft_help_100 = {
        sid: _soft_helps_vs_core(val_100, sid) for sid in soft_ids
    }

    # Plain yes/no: locked combo beats chance later?
    beats_chance_later = bool(core_50["beats_chance"] or core_100["beats_chance"])
    # Soft filters help if any soft raises lift vs core on later +50 or +100
    any_soft_helps = bool(
        any(soft_help_50.values()) or any(soft_help_100.values())
    )

    summary = {
        "label": label,
        "explore_frac": explore_frac,
        "cut_date": str(pd.Timestamp(cut).date()) if cut is not None else None,
        "n_spikes_discover": int(len(discover)),
        "n_control_discover": int(len(control)),
        "base_rate_50": base_50,
        "base_rate_100": base_100,
        "core_50": core_50,
        "core_100": core_100,
        "soft_help_50": soft_help_50,
        "soft_help_100": soft_help_100,
        "beats_chance_later": beats_chance_later,
        "any_soft_helps": any_soft_helps,
        "beats_volume_alone_50": core_50["beats_volume_alone"],
        "beats_not_stretched_alone_50": core_50["beats_not_stretched_alone"],
    }

    return CutSliceResult(
        explore_frac=explore_frac,
        cut_date=pd.Timestamp(cut).normalize() if cut is not None else None,
        split_info=split_info,
        discover_support=discover_support,
        validation_50=val_50,
        validation_100=val_100,
        base_rate_50=base_50,
        base_rate_100=base_100,
        candidates=candidates,
        summary=summary,
        label=label,
    )


def run_draft_formula_fair_test(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    features: pd.DataFrame | None = None,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    window: int = PRIMARY_WINDOW,
    secondary_window: int = SECONDARY_WINDOW,
    explore_frac: float = PRIMARY_EXPLORE_FRAC,
    alt_explore_frac: float | None = ALT_EXPLORE_FRAC,
    feature_model: str = "C",
    control_seed: int = 42,
    max_control: int | None = 5000,
    min_fires: int = DEFAULT_MIN_FIRES,
    beat_lift: float = DEFAULT_BEAT_LIFT,
) -> DraftFormulaFairTestResult:
    """Fair-test locked draft formula on later period; optional alternate cut."""
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

    primary = _evaluate_cut(
        market=market,
        eligible=eligible,
        spikes_ind=spikes_ind,
        flags_panel=flags_panel,
        explore_frac=explore_frac,
        label="primary",
        control_seed=control_seed,
        max_control=max_control,
        min_fires=min_fires,
        beat_lift=beat_lift,
        window=window,
    )

    alternate = None
    if alt_explore_frac is not None and abs(float(alt_explore_frac) - float(explore_frac)) > 1e-9:
        alternate = _evaluate_cut(
            market=market,
            eligible=eligible,
            spikes_ind=spikes_ind,
            flags_panel=flags_panel,
            explore_frac=float(alt_explore_frac),
            label="alternate",
            control_seed=control_seed,
            max_control=max_control,
            min_fires=min_fires,
            beat_lift=beat_lift,
            window=window,
        )

    ps = primary.summary
    summary = {
        "n_eligible_assets": n_assets,
        "n_spikes_total": int(len(spikes)),
        "threshold": threshold,
        "window": window,
        "secondary_window": secondary_window,
        "primary_cut": ps.get("cut_date"),
        "primary_explore_frac": explore_frac,
        "alt_cut": (alternate.summary.get("cut_date") if alternate else None),
        "alt_explore_frac": alt_explore_frac,
        # Plain yes/no answers (primary later period)
        "locked_beats_chance_later": ps.get("beats_chance_later"),
        "locked_beats_volume_alone_50": ps.get("beats_volume_alone_50"),
        "locked_beats_not_stretched_alone_50": ps.get("beats_not_stretched_alone_50"),
        "soft_filters_help": ps.get("any_soft_helps"),
        "core_50": ps.get("core_50"),
        "core_100": ps.get("core_100"),
        "soft_help_50": ps.get("soft_help_50"),
        "soft_help_100": ps.get("soft_help_100"),
    }

    return DraftFormulaFairTestResult(
        spikes=spikes_ind,
        flags_panel=flags_panel,
        primary=primary,
        alternate=alternate,
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
    for c in DRAFT_CANDIDATES:
        if c.candidate_id == cid:
            return c.label
    return cid


def _yes_no(v: bool | None) -> str:
    if v is None:
        return "n/a"
    return "YES" if v else "NO"


def _val_table_lines(
    lines: list[str],
    vt: pd.DataFrame,
    title: str,
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if not len(vt):
        lines.append("_No validation rows._")
        lines.append("")
        return
    lines.append(
        "| rule | fires | hits | hit rate | base | lift | beats chance? |"
    )
    lines.append("|------|------|------|------|------|------|------|")
    for _, r in vt.iterrows():
        beat = (
            "yes"
            if r["beats_base"]
            else ("n/a (few fires)" if not r["n_fire_ok"] else "no")
        )
        lines.append(
            f"| {_label_for_id(str(r['setup_id']))} | {int(r['n_fire'])} | "
            f"{int(r['n_hit'])} | {_fmt_pct(r['hit_rate'])} | "
            f"{_fmt_pct(r['base_rate'])} | {_fmt_num(r['lift'])}x | {beat} |"
        )
    lines.append("")


def _write_cut_section(
    lines: list[str],
    slice_res: CutSliceResult,
    *,
    window: int,
    heading: str,
) -> None:
    s = slice_res.summary
    split = slice_res.split_info
    lines.append(f"## {heading}")
    lines.append("")
    lines.append(f"- Explore fraction: **{slice_res.explore_frac}**")
    if split.get("cut_date") is not None:
        lines.append(
            f"- **Cut date (first later day):** "
            f"`{pd.Timestamp(split['cut_date']).date()}` "
            f"(earlier `{pd.Timestamp(split['explore_start']).date()}` → "
            f"`{pd.Timestamp(split['explore_end']).date()}`; "
            f"later `{pd.Timestamp(split['holdout_start']).date()}` → "
            f"`{pd.Timestamp(split['holdout_end']).date()}`)"
        )
    lines.append(f"- Earlier spikes (descriptive only): **{s.get('n_spikes_discover', 0)}**")
    lines.append(f"- Control non-spike days (earlier): **{s.get('n_control_discover', 0)}**")
    lines.append(
        f"- Later everyday base rates: +50% → {_fmt_pct(slice_res.base_rate_50)}; "
        f"+100% → {_fmt_pct(slice_res.base_rate_100)}"
    )
    lines.append("")

    lines.append("### Discover support (earlier only — not the fair test)")
    lines.append("")
    lines.append(
        "How often each rule was true on earlier spike T−1 days vs normal days. "
        "Descriptive only."
    )
    lines.append("")
    if len(slice_res.discover_support):
        lines.append(
            "| rule | spike support | control support | lift | spike hits |"
        )
        lines.append("|------|------|------|------|------|")
        for _, r in slice_res.discover_support.iterrows():
            cid = str(r.get("candidate_id", r.get("setup_id", "")))
            lines.append(
                f"| {_label_for_id(cid)} | {_fmt_pct(r['spike_support'])} | "
                f"{_fmt_pct(r['ctrl_support'])} | {_fmt_num(r['lift'])}x | "
                f"{int(r['spike_count'])}/{int(r['spike_n'])} |"
            )
        lines.append("")
    else:
        lines.append("_No discover support rows._")
        lines.append("")

    lines.append(
        f"### Later fair check: +50% and +100% within {window}d"
    )
    lines.append("")
    lines.append(
        "Primary question: when the rule fires on ordinary eligible days **after** "
        "the cut, how often do we get +50% / +100% within 7 days, vs everyday chance?"
    )
    lines.append("")
    _val_table_lines(
        lines, slice_res.validation_50, f"+50% within {window}d (later)"
    )
    _val_table_lines(
        lines, slice_res.validation_100, f"+100% within {window}d (later)"
    )


def write_draft_formula_fair_test_report(
    result: DraftFormulaFairTestResult,
    path: Path | str,
) -> Path:
    """Plain-English markdown — clear yes/no, no alpha claims."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M %Z")
    s = result.summary
    ps = result.primary.summary
    window = int(s.get("window", PRIMARY_WINDOW) or PRIMARY_WINDOW)

    lines: list[str] = []
    lines.append("# Locked draft formula — fair test (Abdul)")
    lines.append("")
    lines.append(f"**When:** {now} (Africa/Lagos)")
    lines.append(
        "**Status:** research fair-test only — **not alpha**, not a trading system, "
        "not a live bot"
    )
    lines.append(
        "Abdul chose **A then B**: first fair-test this locked draft (no Bollinger); "
        "X comes later."
    )
    lines.append("")

    lines.append("## Locked formula (no Bollinger)")
    lines.append("")
    lines.append("Signal day when **BOTH** are true:")
    lines.append("1. `rvol_30 > 1.5` (volume expansion)")
    lines.append("2. `dist_ema_20 <= 0.05` (price not more than 5% above EMA20)")
    lines.append("")
    lines.append("Optional soft filters (reported WITH and WITHOUT):")
    lines.append("- RSI ≤ 70 (primary soft); RSI 45–65 band as secondary")
    lines.append(
        "- Gap > −10 if available — **optional**; sentiment evidence is weak"
    )
    lines.append("")

    lines.append("## Short answers (primary later period)")
    lines.append("")
    core50 = ps.get("core_50") or {}
    core100 = ps.get("core_100") or {}
    lines.append(
        f"- Primary cut date: **{ps.get('cut_date', 'n/a')}** "
        f"(explore_frac={result.primary.explore_frac})."
    )
    lines.append(
        f"- Spikes (≥ +50% in {window}d): **{s.get('n_spikes_total', 0)}** total; "
        f"eligible assets ≈ **{s.get('n_eligible_assets', 0)}**."
    )
    lines.append("")
    lines.append(
        f"- **Does the locked combo beat chance later?** "
        f"**{_yes_no(s.get('locked_beats_chance_later'))}** "
        f"(+50% lift {_fmt_num(core50.get('core_lift'))}x on "
        f"{core50.get('n_fire', 0)} fires / {core50.get('n_hit', 0)} hits; "
        f"+100% lift {_fmt_num(core100.get('core_lift'))}x on "
        f"{core100.get('n_fire', 0)} fires / {core100.get('n_hit', 0)} hits; "
        f"need ≥5 fires and lift ≥ 1.10× base to count as beating chance)."
    )
    lines.append(
        f"- **Better than volume alone later (+50%)?** "
        f"**{_yes_no(s.get('locked_beats_volume_alone_50'))}** "
        f"(locked {_fmt_num(core50.get('core_lift'))}x vs volume "
        f"{_fmt_num(core50.get('volume_lift'))}x)."
    )
    lines.append(
        f"- **Better than not-stretched alone later (+50%)?** "
        f"**{_yes_no(s.get('locked_beats_not_stretched_alone_50'))}** "
        f"(locked {_fmt_num(core50.get('core_lift'))}x vs not-stretched "
        f"{_fmt_num(core50.get('not_stretched_lift'))}x)."
    )

    soft50 = ps.get("soft_help_50") or {}
    soft100 = ps.get("soft_help_100") or {}
    # OR across horizons — do not let +100 overwrite +50 with False
    helping = sorted(
        {
            sid
            for sid in set(soft50) | set(soft100)
            if bool(soft50.get(sid)) or bool(soft100.get(sid))
        }
    )
    # Plain notes on each soft filter vs locked core (+50% later)
    soft_notes = []
    for cid in (
        "locked_plus_rsi70",
        "locked_plus_rsi45_65",
        "locked_plus_gap_m10",
        "locked_plus_rsi70_gap",
    ):
        row = _row_for(result.primary.validation_50, cid)
        core = _row_for(result.primary.validation_50, "locked_core")
        if not row or not core:
            continue
        same_fires = int(row["n_fire"]) == int(core["n_fire"]) and int(row["n_hit"]) == int(
            core["n_hit"]
        )
        if same_fires:
            soft_notes.append(
                f"{_label_for_id(cid)}: same fires/hits as locked core (no change)"
            )
        elif bool(soft50.get(cid)) or bool(soft100.get(cid)):
            soft_notes.append(
                f"{_label_for_id(cid)}: higher later lift "
                f"({_fmt_num(row.get('lift'))}x on +50% vs core "
                f"{_fmt_num(core.get('lift'))}x)"
            )
        else:
            soft_notes.append(
                f"{_label_for_id(cid)}: did not raise later lift vs core "
                f"({_fmt_num(row.get('lift'))}x on +50%)"
            )
    if s.get("soft_filters_help"):
        help_line = (
            "**YES** — soft filter(s) raised later lift vs locked core: "
            + ", ".join(_label_for_id(x) for x in helping)
        )
    else:
        help_line = (
            "**NO** — none of the soft filters clearly raised later lift vs "
            "the locked core alone (among soft rules with enough fires)."
        )
    lines.append(f"- **Do soft filters help?** {help_line}")
    for note in soft_notes:
        lines.append(f"  - {note}")
    lines.append("")
    lines.append(
        "Small sample. **No alpha claim.** Do not treat this as a trading system."
    )
    lines.append("")

    lines.append("## What we did (plain)")
    lines.append("")
    lines.append(
        "1. Locked the draft **before** looking at later numbers: volume expansion "
        "AND not more than 5% above EMA20 (no Bollinger)."
    )
    lines.append(
        "2. Also measured volume-alone and not-stretched-alone so we can see if "
        "the **combo** is doing real work."
    )
    lines.append(
        "3. Reported the same locked core **with** and **without** soft filters "
        "(RSI ≤ 70; RSI 45–65 secondary; Gap > −10 optional)."
    )
    lines.append(
        "4. Primary fair check uses the same cut style as prior studies "
        f"(~2026-08-21, explore_frac={result.primary.explore_frac}): hit rates for "
        f"+50% and +100% within {window}d on the **later** period only."
    )
    if result.alternate is not None:
        lines.append(
            f"5. Secondary robustness: alternate cut with explore_frac="
            f"{result.alternate.explore_frac} "
            f"(cut {result.alternate.summary.get('cut_date')})."
        )
    lines.append("")

    lines.append("## Candidates (locked)")
    lines.append("")
    lines.append("| id | role | flags |")
    lines.append("|------|------|------|")
    for c in DRAFT_CANDIDATES:
        lines.append(
            f"| {c.candidate_id} | {c.role} | `{'` AND `'.join(c.flag_ids)}` |"
        )
    lines.append("")

    _write_cut_section(
        lines,
        result.primary,
        window=window,
        heading="Primary later period (fair test)",
    )

    if result.alternate is not None:
        _write_cut_section(
            lines,
            result.alternate,
            window=window,
            heading="Secondary robustness (alternate cut)",
        )
        alt = result.alternate.summary
        lines.append("### Alternate cut — quick yes/no")
        lines.append("")
        lines.append(
            f"- Locked beats chance later? **{_yes_no(alt.get('beats_chance_later'))}**"
        )
        lines.append(
            f"- Soft filters help vs locked core? **{_yes_no(alt.get('any_soft_helps'))}**"
        )
        lines.append("")

    lines.append("## Recommendation (research process, not a trade)")
    lines.append("")
    if s.get("locked_beats_chance_later") and s.get("locked_beats_volume_alone_50"):
        lines.append(
            "- Locked draft **beats everyday chance** and **volume alone** on the "
            "primary later cut for +50% — keep it as the draft to stress-test next "
            "(still not promoted)."
        )
    elif s.get("locked_beats_chance_later"):
        lines.append(
            "- Locked draft **beats everyday chance** later, but did **not** clearly "
            "beat volume alone on +50% in this cut — treat the combo as tentative."
        )
    else:
        lines.append(
            "- Locked draft did **not** clearly beat chance later under the fair "
            "rules used here — do not lean on it until more data / another cut."
        )
    if s.get("soft_filters_help"):
        lines.append(
            "- Soft filters that helped later lift vs core: "
            + ", ".join(_label_for_id(x) for x in helping)
            + ". Keep them optional for a repeat check — do not harden yet."
        )
    else:
        lines.append(
            "- Soft filters (RSI / Gap) did **not** clearly help vs the locked core "
            "alone — prefer the simpler two-condition draft for now."
        )
    lines.append("- **X later** (Abdul path B) — not run in this report.")
    lines.append("- **No alpha claim.**")
    lines.append("")

    lines.append("## Honesty box")
    lines.append("")
    lines.append(
        f"- Total spikes in sample: **{s.get('n_spikes_total', 0)}**. Thin history."
    )
    lines.append(
        "- Candidates and soft filters were locked before the later look; still one "
        "short crypto window and overlapping themes — easy to overfit."
    )
    lines.append(
        "- Gap soft filter depends on Rotation Gap (sentiment/narrative); labeled "
        "**optional** because that evidence has been weak in prior checks."
    )
    lines.append("- **No alpha claim.** Do not treat this as a system.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_draft_formula_outputs(
    result: DraftFormulaFairTestResult,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Save validation / support parquet under data/draft_formula."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _dump(name: str, df: pd.DataFrame) -> None:
        if df is None or not len(df):
            return
        p = out_dir / name
        df.to_parquet(p, index=False)
        paths[name] = p

    _dump("draft_primary_discover_support.parquet", result.primary.discover_support)
    _dump("draft_primary_val_50.parquet", result.primary.validation_50)
    _dump("draft_primary_val_100.parquet", result.primary.validation_100)
    if result.alternate is not None:
        _dump(
            "draft_alt_discover_support.parquet",
            result.alternate.discover_support,
        )
        _dump("draft_alt_val_50.parquet", result.alternate.validation_50)
        _dump("draft_alt_val_100.parquet", result.alternate.validation_100)
    return paths
