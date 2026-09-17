"""Backward integration: find +50% spikes, learn pre-spike patterns, validate forward.

Research only — no alpha claims, no live trading bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cmram.backtest.holdout import compute_time_split, membership_calendar_dates
from cmram.backtest.spike_detect import (
    DEFAULT_SPIKE_THRESHOLD,
    PRIMARY_WINDOW,
    SECONDARY_WINDOW,
    detect_spikes,
    forward_max_return_by_asset,
    split_spikes_by_cut,
)
from cmram.backtest.spike_indicators import (
    INDICATOR_COLS,
    compute_tech_indicators,
    snapshot_pre_spike,
)
from cmram.features.prep import prepare_market

LAGOS = ZoneInfo("Africa/Lagos")
DEFAULT_EXPLORE_FRAC = 0.70
FEATURE_SCORE_COLS = (
    "MREI",
    "NSI",
    "Rotation_Gap",
    "score_C",
    "score_V",
    "score_N",
    "score_P",
)
COMPARE_COLS = tuple(INDICATOR_COLS) + FEATURE_SCORE_COLS


RuleFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class PatternRule:
    rule_id: str
    plain_english: str
    apply: RuleFn


@dataclass
class SpikeBackwardResult:
    spikes: pd.DataFrame
    spikes_with_indicators: pd.DataFrame
    discover: pd.DataFrame
    validate: pd.DataFrame
    control_discover: pd.DataFrame
    cut_date: pd.Timestamp | None
    split_info: dict[str, Any]
    compare_table: pd.DataFrame
    rules: list[PatternRule] = field(default_factory=list)
    validation_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    base_rate_validate: float = float("nan")
    n_eligible_assets: int = 0
    summary: dict[str, Any] = field(default_factory=dict)


def _eligible_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """Rows that count as eligible universe members (liquidity_pass if present)."""
    mem = membership.copy()
    mem["timestamp"] = (
        pd.to_datetime(mem["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    mem["asset_id"] = mem["asset_id"].astype(str)
    if "liquidity_pass" in mem.columns:
        # Keep PASS or True; also keep if null and no reason_excluded
        lp = mem["liquidity_pass"]
        if lp.dtype == bool or str(lp.dtype) == "boolean":
            mem = mem.loc[lp.fillna(False)]
        else:
            mem = mem.loc[lp.astype(str).str.upper().isin(["TRUE", "1", "PASS", "YES"])]
    if "reason_excluded" in mem.columns:
        rex = mem["reason_excluded"]
        keep = rex.isna() | (rex.astype(str).str.strip() == "")
        mem = mem.loc[keep]
    return mem


def build_panel_labels(
    market: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    window: int = PRIMARY_WINDOW,
) -> pd.DataFrame:
    """Eligible (date, asset) panel with forward-return label and is_spike_day.

    ``is_spike_day`` uses raw (non-deduped) threshold hit — used for base-rate
    and rule hit-rate on held-out days.
    """
    fwd = forward_max_return_by_asset(market, window)
    col = f"fwd_max_ret_{window}d"
    elig = eligible[["timestamp", "asset_id"]].drop_duplicates()
    if "band" in eligible.columns:
        elig = (
            eligible[["timestamp", "asset_id", "band"]]
            .drop_duplicates(["timestamp", "asset_id"], keep="first")
        )
    elig = elig.copy()
    elig["timestamp"] = (
        pd.to_datetime(elig["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    elig["asset_id"] = elig["asset_id"].astype(str)
    panel = elig.merge(fwd, on=["timestamp", "asset_id"], how="inner")
    panel["is_spike_day"] = panel[col] >= float(threshold)
    return panel


def sample_control_days(
    panel: pd.DataFrame,
    indicators: pd.DataFrame,
    features: pd.DataFrame | None,
    *,
    n: int,
    seed: int = 42,
    model: str = "C",
    before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Random eligible days that did NOT hit +50% forward (control set)."""
    ctrl = panel.loc[~panel["is_spike_day"].fillna(False)].copy()
    if before is not None:
        ctrl = ctrl.loc[
            pd.to_datetime(ctrl["timestamp"]).dt.normalize() < pd.Timestamp(before).normalize()
        ]
    if ctrl.empty or n <= 0:
        return pd.DataFrame()
    n_take = min(int(n), len(ctrl))
    rng = np.random.default_rng(seed)
    idx = rng.choice(ctrl.index.to_numpy(), size=n_take, replace=False)
    sampled = ctrl.loc[idx].copy()
    sampled = sampled.rename(columns={"timestamp": "signal_date"})
    # Fake spike_date = signal_date + 1 day placeholder for schema parity
    sampled["spike_date"] = pd.to_datetime(sampled["signal_date"]) + pd.Timedelta(days=1)
    sampled["close_t"] = sampled.get("close", np.nan)
    snap = snapshot_pre_spike(
        sampled.assign(signal_date=sampled["signal_date"]),
        indicators,
        features,
        model=model,
    )
    # Indicators are keyed to signal_date; for control, timestamp IS the snapshot day
    # Re-merge indicators on signal_date==timestamp of control day
    ind = indicators.copy()
    ind["timestamp"] = (
        pd.to_datetime(ind["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    ind["asset_id"] = ind["asset_id"].astype(str)
    ind_cols = ["timestamp", "asset_id"] + [c for c in INDICATOR_COLS if c in ind.columns]
    ind = ind[ind_cols]
    base = sampled[["signal_date", "asset_id", "spike_date", "close_t"]].copy()
    if "band" in sampled.columns:
        base["band"] = sampled["band"]
    base["signal_date"] = pd.to_datetime(base["signal_date"]).dt.normalize()
    base["asset_id"] = base["asset_id"].astype(str)
    out = base.merge(
        ind.rename(columns={"timestamp": "signal_date"}),
        on=["signal_date", "asset_id"],
        how="left",
    )
    if features is not None and len(features) > 0:
        from cmram.backtest.spike_indicators import attach_cmram_features

        out = attach_cmram_features(out, features, model=model)
    return out


def summarize_distributions(
    spike_snaps: pd.DataFrame,
    control_snaps: pd.DataFrame,
    cols: tuple[str, ...] = COMPARE_COLS,
) -> pd.DataFrame:
    """Median/mean/q25/q75 for spike vs control on discover set."""
    rows: list[dict[str, Any]] = []
    for col in cols:
        if col not in spike_snaps.columns and (
            control_snaps is None or col not in control_snaps.columns
        ):
            continue
        s = (
            pd.to_numeric(spike_snaps[col], errors="coerce")
            if col in spike_snaps.columns
            else pd.Series(dtype=float)
        )
        c = (
            pd.to_numeric(control_snaps[col], errors="coerce")
            if control_snaps is not None and col in control_snaps.columns
            else pd.Series(dtype=float)
        )
        s = s.dropna()
        c = c.dropna()
        rows.append(
            {
                "indicator": col,
                "spike_n": int(len(s)),
                "spike_median": float(s.median()) if len(s) else np.nan,
                "spike_mean": float(s.mean()) if len(s) else np.nan,
                "spike_q25": float(s.quantile(0.25)) if len(s) else np.nan,
                "spike_q75": float(s.quantile(0.75)) if len(s) else np.nan,
                "ctrl_n": int(len(c)),
                "ctrl_median": float(c.median()) if len(c) else np.nan,
                "ctrl_mean": float(c.mean()) if len(c) else np.nan,
                "ctrl_q25": float(c.quantile(0.25)) if len(c) else np.nan,
                "ctrl_q75": float(c.quantile(0.75)) if len(c) else np.nan,
                "median_diff": (
                    float(s.median() - c.median()) if len(s) and len(c) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _stat(med: dict, indicator: str, field: str) -> float | None:
    if indicator not in med:
        return None
    val = med[indicator].get(field) if hasattr(med[indicator], "get") else None
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def propose_rules_from_discover(
    compare: pd.DataFrame,
    spike_snaps: pd.DataFrame,
) -> list[PatternRule]:
    """Propose 2–4 simple plain-English rules from discover distributions.

    Rules are fixed from discover stats only (no peeking at validate).
    """
    rules: list[PatternRule] = []
    med = {
        str(r["indicator"]): r
        for _, r in compare.iterrows()
    } if compare is not None and len(compare) else {}

    # Rule 1: RSI mid-zone (not overbought) + price above EMA20
    # Use spike median RSI if available; else default 40-65
    rsi_hi = 65.0
    rsi_lo = 35.0
    q25 = _stat(med, "rsi_14", "spike_q25")
    q75 = _stat(med, "rsi_14", "spike_q75")
    if q25 is not None and q75 is not None:
        rsi_lo = max(30.0, q25 - 5.0)
        rsi_hi = min(75.0, q75 + 5.0)

    def rule_rsi_ema(df: pd.DataFrame) -> pd.Series:
        rsi = pd.to_numeric(df.get("rsi_14"), errors="coerce")
        dist = pd.to_numeric(df.get("dist_ema_20"), errors="coerce")
        return (rsi >= rsi_lo) & (rsi <= rsi_hi) & (dist > 0)

    rules.append(
        PatternRule(
            rule_id="R1_rsi_mid_above_ema20",
            plain_english=(
                f"RSI between {rsi_lo:.0f} and {rsi_hi:.0f} AND price above EMA20 "
                f"(dist_ema_20 > 0)"
            ),
            apply=rule_rsi_ema,
        )
    )

    # Rule 2: Elevated RVOL but not extreme + not deep drawdown
    rvol_lo = 1.0
    rvol_hi = 3.0
    sm = _stat(med, "rvol_30", "spike_median")
    if sm is not None:
        rvol_lo = max(0.8, sm * 0.6)
        rvol_hi = max(rvol_lo + 0.5, sm * 2.0)

    def rule_rvol(df: pd.DataFrame) -> pd.Series:
        rvol = pd.to_numeric(df.get("rvol_30"), errors="coerce")
        dd = pd.to_numeric(df.get("drawdown_60d"), errors="coerce")
        return (rvol >= rvol_lo) & (rvol <= rvol_hi) & (dd > -0.40)

    rules.append(
        PatternRule(
            rule_id="R2_rvol_elevated_not_crashed",
            plain_english=(
                f"Volume vs 30d average between {rvol_lo:.2f}x and {rvol_hi:.2f}x "
                f"AND drawdown from 60d high milder than -40%"
            ),
            apply=rule_rvol,
        )
    )

    # Rule 3: MACD hist positive + Bollinger %b mid/upper (expanding interest)
    def rule_macd_bb(df: pd.DataFrame) -> pd.Series:
        hist = pd.to_numeric(df.get("macd_hist"), errors="coerce")
        pctb = pd.to_numeric(df.get("bb_pct_b"), errors="coerce")
        return (hist > 0) & (pctb >= 0.4) & (pctb <= 1.05)

    rules.append(
        PatternRule(
            rule_id="R3_macd_hist_pos_bb_mid",
            plain_english=(
                "MACD histogram > 0 AND Bollinger %b between 0.40 and 1.05 "
                "(price in upper half of band, not a blow-off yet)"
            ),
            apply=rule_macd_bb,
        )
    )

    # Rule 4: CMRAM Gap / MREI if available and shows separation
    gap_med = _stat(med, "Rotation_Gap", "spike_median")
    mrei_med = _stat(med, "MREI", "spike_median")
    if gap_med is not None:
        gap_cut = float(gap_med)
        gap_ctrl = _stat(med, "Rotation_Gap", "ctrl_median")
        if gap_ctrl is not None:
            gap_thr = float(gap_ctrl + 0.5 * (gap_cut - gap_ctrl))
        else:
            gap_thr = gap_cut
        gap_thr = max(gap_thr, 0.0)

        def rule_gap(df: pd.DataFrame, thr: float = gap_thr) -> pd.Series:
            gap = pd.to_numeric(df.get("Rotation_Gap"), errors="coerce")
            dist = pd.to_numeric(df.get("dist_ema_20"), errors="coerce")
            return (gap >= thr) & (dist > -0.05)

        rules.append(
            PatternRule(
                rule_id="R4_gap_and_near_ema20",
                plain_english=(
                    f"Rotation Gap >= {gap_thr:.1f} AND price not far below EMA20 "
                    f"(dist_ema_20 > -5%)"
                ),
                apply=rule_gap,
            )
        )
    elif mrei_med is not None:
        mrei_thr = max(0.0, float(mrei_med) - 5.0)

        def rule_mrei(df: pd.DataFrame, thr: float = mrei_thr) -> pd.Series:
            mrei = pd.to_numeric(df.get("MREI"), errors="coerce")
            rsi = pd.to_numeric(df.get("rsi_14"), errors="coerce")
            return (mrei >= thr) & (rsi >= 40) & (rsi <= 70)

        rules.append(
            PatternRule(
                rule_id="R4_mrei_rsi",
                plain_english=(
                    f"MREI >= {mrei_thr:.1f} AND RSI between 40 and 70"
                ),
                apply=rule_mrei,
            )
        )
    else:
        # Fallback combo without CMRAM scores
        def rule_combo(df: pd.DataFrame) -> pd.Series:
            rsi = pd.to_numeric(df.get("rsi_14"), errors="coerce")
            dist = pd.to_numeric(df.get("dist_ema_20"), errors="coerce")
            rvol = pd.to_numeric(df.get("rvol_30"), errors="coerce")
            return (rsi >= 40) & (rsi <= 60) & (dist > 0) & (rvol >= 1.0)

        rules.append(
            PatternRule(
                rule_id="R4_rsi_ema_rvol",
                plain_english=(
                    "RSI between 40 and 60 AND price above EMA20 AND RVOL >= 1.0"
                ),
                apply=rule_combo,
            )
        )

    # Keep at most 4
    return rules[:4]


def evaluate_rules_on_panel(
    panel_snaps: pd.DataFrame,
    rules: list[PatternRule],
    *,
    label_col: str = "is_spike_day",
) -> pd.DataFrame:
    """Hit-rate of each rule vs base rate on a labeled panel with indicators."""
    if panel_snaps is None or len(panel_snaps) == 0:
        return pd.DataFrame()
    y = panel_snaps[label_col].fillna(False).astype(bool)
    base = float(y.mean()) if len(y) else float("nan")
    rows = []
    for rule in rules:
        mask = rule.apply(panel_snaps).fillna(False).astype(bool)
        n_fire = int(mask.sum())
        if n_fire == 0:
            hit = float("nan")
            n_hit = 0
        else:
            n_hit = int((mask & y).sum())
            hit = n_hit / n_fire
        rows.append(
            {
                "rule_id": rule.rule_id,
                "plain_english": rule.plain_english,
                "n_panel": int(len(panel_snaps)),
                "n_fire": n_fire,
                "n_hit": n_hit,
                "hit_rate": hit,
                "base_rate": base,
                "lift": (hit / base) if (base and base > 0 and pd.notna(hit)) else np.nan,
                "beats_base": bool(
                    n_fire >= 5
                    and pd.notna(hit)
                    and base > 0
                    and (hit / base) >= 1.10
                ),
                "n_fire_ok": n_fire >= 5,
            }
        )
    return pd.DataFrame(rows)


def _attach_indicators_to_panel(
    panel: pd.DataFrame,
    indicators: pd.DataFrame,
    features: pd.DataFrame | None,
    *,
    model: str = "C",
) -> pd.DataFrame:
    """Merge T-day indicators onto panel rows (timestamp = snapshot day)."""
    p = panel.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None).dt.normalize()
    p["asset_id"] = p["asset_id"].astype(str)
    ind = indicators.copy()
    ind["timestamp"] = (
        pd.to_datetime(ind["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    ind["asset_id"] = ind["asset_id"].astype(str)
    ind_cols = ["timestamp", "asset_id"] + [c for c in INDICATOR_COLS if c in ind.columns]
    out = p.merge(ind[ind_cols], on=["timestamp", "asset_id"], how="left")
    if features is not None and len(features) > 0:
        from cmram.backtest.spike_indicators import attach_cmram_features

        # attach expects signal_date
        tmp = out.rename(columns={"timestamp": "signal_date"})
        tmp = attach_cmram_features(tmp, features, model=model)
        out = tmp.rename(columns={"signal_date": "timestamp"})
    return out


def run_spike_backward(
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
) -> SpikeBackwardResult:
    """Full pipeline: detect → split → compare → propose rules → validate."""
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
    spikes_ind = snapshot_pre_spike(
        spikes, indicators, features, model=feature_model
    )

    dates = membership_calendar_dates(eligible if len(eligible) else membership)
    split_info = compute_time_split(dates, explore_frac=explore_frac)
    cut = split_info.get("cut_date")

    discover, validate = split_spikes_by_cut(spikes_ind, cut) if cut is not None else (
        spikes_ind.copy(),
        spikes_ind.iloc[0:0].copy(),
    )

    panel = build_panel_labels(
        market, eligible, threshold=threshold, window=window
    )
    control = sample_control_days(
        panel,
        indicators,
        features,
        n=max(len(discover) * 3, 50),
        seed=control_seed,
        model=feature_model,
        before=cut,
    )

    compare = summarize_distributions(discover, control)
    rules = propose_rules_from_discover(compare, discover)

    # Validate: rules on later panel days only
    if cut is not None:
        panel_v = panel.loc[
            pd.to_datetime(panel["timestamp"]).dt.normalize()
            >= pd.Timestamp(cut).normalize()
        ].copy()
    else:
        panel_v = panel.iloc[0:0].copy()

    panel_v_snap = _attach_indicators_to_panel(
        panel_v, indicators, features, model=feature_model
    )
    # For validation we evaluate whether the *day* is a raw spike day.
    # To be closer to "predict +50%", we use is_spike_day on the panel day.
    # Optionally also require signal-style: rule on day t predicts spike starting t+1 —
    # here we keep same-day label for simplicity (indicators known at close t).
    base_rate = (
        float(panel_v_snap["is_spike_day"].mean())
        if len(panel_v_snap)
        else float("nan")
    )
    val_table = evaluate_rules_on_panel(panel_v_snap, rules)

    pcol = f"fwd_max_ret_{window}d"
    scol = f"fwd_max_ret_{secondary_window}d"
    summary = {
        "n_eligible_assets": n_assets,
        "n_spikes_total": int(len(spikes)),
        "n_spikes_discover": int(len(discover)),
        "n_spikes_validate": int(len(validate)),
        "n_control_discover": int(len(control)),
        "cut_date": str(pd.Timestamp(cut).date()) if cut is not None else None,
        "threshold": threshold,
        "window": window,
        "secondary_window": secondary_window,
        "base_rate_validate": base_rate,
        "any_rule_beats_base": bool(
            val_table["beats_base"].any()
        )
        if len(val_table)
        else False,
        "median_fwd_7d": float(spikes[pcol].median())
        if len(spikes) and pcol in spikes.columns
        else np.nan,
        "median_fwd_14d": float(spikes[scol].median())
        if len(spikes) and scol in spikes.columns
        else np.nan,
    }

    return SpikeBackwardResult(
        spikes=spikes,
        spikes_with_indicators=spikes_ind,
        discover=discover,
        validate=validate,
        control_discover=control,
        cut_date=pd.Timestamp(cut).normalize() if cut is not None else None,
        split_info=split_info,
        compare_table=compare,
        rules=rules,
        validation_table=val_table,
        base_rate_validate=base_rate,
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


def write_spike_backward_report(
    result: SpikeBackwardResult,
    path: Path | str,
) -> Path:
    """Write plain-language markdown report for Abdul."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M %Z")
    s = result.summary
    split = result.split_info

    lines: list[str] = []
    lines.append("# Spike backward look — what came before +50% moves")
    lines.append("")
    lines.append(f"**When:** {now} (Africa/Lagos)")
    lines.append(
        "**Status:** research check only — **not alpha**, not a trading system, "
        "not a live bot"
    )
    lines.append(
        f"**Spike (locked):** forward max return ≥ **+{_fmt_pct(s.get('threshold', 0.5), 0)}** "
        f"within **{s.get('window', 7)} days** (daily closes). "
        f"Also report **{s.get('secondary_window', 14)} days**."
    )
    lines.append(
        "Overlapping spikes on the same coin are deduped (keep the first / "
        "non-overlapping window)."
    )
    lines.append("")
    lines.append("## Short answers")
    lines.append("")
    lines.append(f"- How many spikes found? **{s.get('n_spikes_total', 0)}** "
                 f"(eligible assets ≈ **{s.get('n_eligible_assets', 0)}**).")
    lines.append(
        f"- Discover (earlier) / validate (later): "
        f"**{s.get('n_spikes_discover', 0)}** / **{s.get('n_spikes_validate', 0)}**."
    )
    lines.append(
        f"- Cut date (first validate day): **{s.get('cut_date', 'n/a')}** "
        f"(explore ≈ {int(100 * float(split.get('explore_frac', 0.7)))}% of membership days)."
    )

    any_beat = bool(s.get("any_rule_beats_base"))
    if len(result.validation_table) == 0:
        lines.append(
            "- Did any rule beat chance on later data? **Could not test** "
            "(empty validate panel)."
        )
        overall = "Incomplete — no later-period test."
    elif any_beat:
        ok = result.validation_table.loc[result.validation_table["beats_base"]]
        ids = ", ".join(ok["rule_id"].tolist())
        lines.append(
            f"- Did any rule beat chance on later data? **Yes (weak check only)** — "
            f"{ids}. Still **not** a claim of alpha."
        )
        overall = (
            "Some rules edged above the later-period base rate (lift about 1.1–1.2x). "
            "That is a weak signal in a noisy sample — exploratory only, "
            "**no alpha claim**, do not trade this."
        )
    else:
        lines.append(
            "- Did any rule beat chance on later data? **No** "
            "(or not with enough fires)."
        )
        overall = (
            "No proposed rule clearly beat chance on the later period. "
            "Do **not** claim a predictive pattern."
        )
    lines.append("")
    lines.append(f"### Overall")
    lines.append("")
    lines.append(overall)
    lines.append("")

    lines.append("## What we did")
    lines.append("")
    lines.append(
        "1. Looked at every eligible coin-day in the universe (~150 names)."
    )
    lines.append(
        f"2. Marked a **spike start day T** when the best close in the next "
        f"{s.get('window', 7)} days was at least +50% above the close on T."
    )
    lines.append(
        "3. Removed overlapping spikes on the same coin (kept the earliest)."
    )
    lines.append(
        "4. On the day **before** the spike (T−1), snapped common indicators "
        "(EMA, MACD, RSI, Bollinger, volume vs 30d, plus MREI/NSI/Gap/scores if present)."
    )
    lines.append(
        "5. Used **earlier** spikes to see what looked different vs random "
        "non-spike days, and wrote 2–4 simple rules in plain English."
    )
    lines.append(
        "6. Tested those rules **only** on the **later** period: do days that "
        "match a rule hit +50% more often than the normal (base) rate?"
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
    lines.append("")

    lines.append("## Spike counts")
    lines.append("")
    lines.append(
        f"| set | n spikes | unique assets |"
    )
    lines.append("|------|------|------|")
    for label, df in (
        ("all", result.spikes),
        ("discover", result.discover),
        ("validate", result.validate),
    ):
        n_a = int(df["asset_id"].nunique()) if len(df) else 0
        lines.append(f"| {label} | {len(df)} | {n_a} |")
    lines.append("")
    pcol = f"fwd_max_ret_{s.get('window', 7)}d"
    scol = f"fwd_max_ret_{s.get('secondary_window', 14)}d"
    if len(result.spikes):
        lines.append(
            f"- Median forward max return at {s.get('window', 7)}d among spikes: "
            f"**{_fmt_pct(s.get('median_fwd_7d'))}**"
        )
        lines.append(
            f"- Median forward max return at {s.get('secondary_window', 14)}d among spikes: "
            f"**{_fmt_pct(s.get('median_fwd_14d'))}**"
        )
        lines.append("")

    lines.append("## What looked different before spikes (discover only)")
    lines.append("")
    lines.append(
        "Compared pre-spike (T−1) indicator medians vs a random control set of "
        f"non-spike days in the earlier period (n_control={s.get('n_control_discover', 0)})."
    )
    lines.append("")
    if len(result.compare_table):
        lines.append(
            "| indicator | spike median | control median | difference |"
        )
        lines.append("|------|------|------|------|")
        for _, r in result.compare_table.iterrows():
            lines.append(
                f"| {r['indicator']} | {_fmt_num(r['spike_median'])} | "
                f"{_fmt_num(r['ctrl_median'])} | {_fmt_num(r['median_diff'])} |"
            )
        lines.append("")
        # Plain takeaways: top absolute median diffs
        ct = result.compare_table.dropna(subset=["median_diff"]).copy()
        if len(ct):
            skip_abs = {"ema_20", "ema_50", "macd_line", "macd_signal", "macd_hist"}
            ct = ct.loc[~ct["indicator"].isin(skip_abs)].copy()
            ct["abs_diff"] = ct["median_diff"].abs()
            top = ct.sort_values("abs_diff", ascending=False).head(5)
            lines.append("### Plain takeaways (discover)")
            lines.append("")
            for _, r in top.iterrows():
                direction = "higher" if r["median_diff"] > 0 else "lower"
                lines.append(
                    f"- **{r['indicator']}** tended to be **{direction}** before spikes "
                    f"(spike median {_fmt_num(r['spike_median'])} vs control "
                    f"{_fmt_num(r['ctrl_median'])})."
                )
            lines.append("")
    else:
        lines.append("_No comparison table (not enough discover spikes)._")
        lines.append("")

    lines.append("## Pattern rules (written from discover only)")
    lines.append("")
    if not result.rules:
        lines.append("_No rules proposed._")
    else:
        for i, rule in enumerate(result.rules, 1):
            lines.append(f"{i}. **{rule.rule_id}:** {rule.plain_english}")
        lines.append("")
    lines.append(
        "These rules were **not** tuned on the later period."
    )
    lines.append("")

    lines.append("## Forward check (later period only)")
    lines.append("")
    lines.append(
        f"Base rate on later eligible days (share that hit +50% within "
        f"{s.get('window', 7)}d): **{_fmt_pct(result.base_rate_validate)}**."
    )
    lines.append("")
    if len(result.validation_table):
        lines.append(
            "| rule | fires | hits | hit rate | base rate | lift | beats base? |"
        )
        lines.append("|------|------|------|------|------|------|------|")
        for _, r in result.validation_table.iterrows():
            beat = "yes" if r["beats_base"] else (
                "n/a (few fires)" if not r["n_fire_ok"] else "no"
            )
            lines.append(
                f"| {r['rule_id']} | {int(r['n_fire'])} | {int(r['n_hit'])} | "
                f"{_fmt_pct(r['hit_rate'])} | {_fmt_pct(r['base_rate'])} | "
                f"{_fmt_num(r['lift'])}x | {beat} |"
            )
        lines.append("")
        lines.append(
            "Lift > 1 means the rule's hit rate is higher than the base rate. "
            "We only mark **beats base** when the rule fired at least 5 times "
            "**and** lift is at least 1.10 (10% above chance). Tiny lifts can be noise."
        )
    else:
        lines.append("_No validation rows._")
    lines.append("")

    lines.append("## Honesty box")
    lines.append("")
    lines.append("- This is **backward pattern hunting** plus a one-shot later check.")
    lines.append("- Small sample, short history, crypto noise — easy to overfit.")
    lines.append("- **No alpha claim.** Do not trade this as a system.")
    lines.append("- Spike events parquet is saved under `data/spikes/` for audit.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_spike_outputs(
    result: SpikeBackwardResult,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Save spike event table and comparison tables as parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    p1 = out_dir / "spike_events.parquet"
    result.spikes_with_indicators.to_parquet(p1, index=False)
    paths["spike_events"] = p1
    p2 = out_dir / "spike_compare_discover.parquet"
    result.compare_table.to_parquet(p2, index=False)
    paths["compare"] = p2
    if len(result.validation_table):
        p3 = out_dir / "spike_rule_validation.parquet"
        result.validation_table.to_parquet(p3, index=False)
        paths["validation"] = p3
    return paths
