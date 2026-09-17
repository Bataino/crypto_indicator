"""Cross-coin shared boolean setups before spikes (discover → validate).

Replaces median/quantile template rules with a library of yes/no setup flags.
Research only — no alpha claims, no live trading bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
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
    forward_max_return_by_asset,
    split_spikes_by_cut,
)
from cmram.backtest.spike_indicators import (
    attach_cmram_features,
    compute_tech_indicators,
    snapshot_pre_spike,
)

LAGOS = ZoneInfo("Africa/Lagos")
DEFAULT_LIFT_MIN = 1.2
DEFAULT_MIN_SPIKE_COUNT = 5
DEFAULT_TOP_SINGLES_FOR_PAIRS = 12
DEFAULT_TOP_SETUPS = 6
DEFAULT_MIN_SETUPS = 3
THRESH_50 = 0.50
THRESH_100 = 1.00

@dataclass(frozen=True)
class SetupFlagDef:
    flag_id: str
    plain_english: str
    # columns that must exist (skip flag if missing after merge)
    required_cols: tuple[str, ...] = ()


@dataclass
class SharedSetup:
    setup_id: str
    plain_english: str
    flag_ids: tuple[str, ...]
    spike_support: float
    ctrl_support: float
    lift: float
    spike_count: int
    spike_n: int
    ctrl_count: int
    ctrl_n: int
    is_pair: bool = False


@dataclass
class SharedSetupsResult:
    spikes: pd.DataFrame
    discover: pd.DataFrame
    validate: pd.DataFrame
    flags_panel: pd.DataFrame
    single_support: pd.DataFrame
    pair_support: pd.DataFrame
    selected: list[SharedSetup] = field(default_factory=list)
    validation_50: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_100: pd.DataFrame = field(default_factory=pd.DataFrame)
    cut_date: pd.Timestamp | None = None
    split_info: dict[str, Any] = field(default_factory=dict)
    base_rate_50: float = float("nan")
    base_rate_100: float = float("nan")
    n_eligible_assets: int = 0
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Flag library
# ---------------------------------------------------------------------------

SETUP_FLAG_DEFS: tuple[SetupFlagDef, ...] = (
    SetupFlagDef("price_above_ema20", "Price above EMA20", ("dist_ema_20",)),
    SetupFlagDef("price_above_ema50", "Price above EMA50", ("dist_ema_50",)),
    SetupFlagDef("ema20_gt_ema50", "EMA20 above EMA50 (uptrend)", ("ema_20", "ema_50")),
    SetupFlagDef("macd_hist_pos", "MACD histogram positive", ("macd_hist",)),
    SetupFlagDef(
        "macd_hist_pos_rising",
        "MACD histogram positive and rising vs prior day",
        ("macd_hist",),
    ),
    SetupFlagDef("rsi_40_60", "RSI between 40 and 60", ("rsi_14",)),
    SetupFlagDef("rsi_45_65", "RSI between 45 and 65", ("rsi_14",)),
    SetupFlagDef("rsi_le_70", "RSI at or below 70 (not blowoff overbought)", ("rsi_14",)),
    SetupFlagDef(
        "rsi_cross_up_50",
        "RSI crossed up through 50 (was below, now at/above)",
        ("rsi_14",),
    ),
    SetupFlagDef(
        "bb_pctb_mid_upper",
        "Bollinger %b between 0.4 and 0.9 (mid/upper band)",
        ("bb_pct_b",),
    ),
    SetupFlagDef(
        "bb_bandwidth_expanding",
        "Bollinger bandwidth expanding vs prior day",
        ("bb_bandwidth",),
    ),
    SetupFlagDef("rvol_gt_1", "Relative volume > 1× 30d average", ("rvol_30",)),
    SetupFlagDef("rvol_gt_1_5", "Relative volume > 1.5× 30d average", ("rvol_30",)),
    SetupFlagDef(
        "dist_ema20_le_10pct",
        "Price not more than 10% above EMA20 (not already extended)",
        ("dist_ema_20",),
    ),
    SetupFlagDef(
        "dist_ema20_le_5pct",
        "Price not more than 5% above EMA20 (not already extended)",
        ("dist_ema_20",),
    ),
    SetupFlagDef(
        "near_ema20_m5_p10",
        "Price near EMA20 (between −5% and +10%)",
        ("dist_ema_20",),
    ),
    SetupFlagDef(
        "reclaim_ema20_3d",
        "Price reclaimed EMA20 within the last 3 days",
        ("dist_ema_20",),
    ),
    SetupFlagDef(
        "quiet_rising_n",
        "Narrative score N was quieter then rising (score_N up, prior day lower)",
        ("score_N",),
    ),
    SetupFlagDef("gap_gt_0", "Rotation Gap > 0", ("Rotation_Gap",)),
    SetupFlagDef("gap_gt_10", "Rotation Gap > 10", ("Rotation_Gap",)),
    SetupFlagDef(
        "gap_gt_m10",
        "Rotation Gap > -10 (not terribly negative; optional soft filter)",
        ("Rotation_Gap",),
    ),
    SetupFlagDef(
        "compression_recovering",
        "Compression low then volume/price scores rising",
        ("score_C", "score_V", "score_P"),
    ),
)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _group_lag(df: pd.DataFrame, col: str, periods: int = 1) -> pd.Series:
    """Lag a column within each asset_id (assumes sorted by timestamp)."""
    s = _num(df, col)
    return s.groupby(df["asset_id"], sort=False).shift(periods)


def build_setup_flags(
    indicators: pd.DataFrame,
    features: pd.DataFrame | None = None,
    *,
    model: str = "C",
) -> pd.DataFrame:
    """Build boolean setup flags for every (timestamp, asset_id) row.

    Adds lagged comparisons within each asset. Optional CMRAM feature columns
    enable Gap / score_N / compression flags; otherwise those stay False/NaN-skipped.
    """
    if indicators is None or len(indicators) == 0:
        cols = ["timestamp", "asset_id"] + [d.flag_id for d in SETUP_FLAG_DEFS]
        return pd.DataFrame(columns=cols)

    df = indicators.copy()
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    df["asset_id"] = df["asset_id"].astype(str)
    df = df.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    if features is not None and len(features) > 0:
        tmp = df.rename(columns={"timestamp": "signal_date"})
        tmp = attach_cmram_features(tmp, features, model=model)
        df = tmp.rename(columns={"signal_date": "timestamp"})

    # Lags used by several flags
    macd_prev = _group_lag(df, "macd_hist", 1)
    rsi_prev = _group_lag(df, "rsi_14", 1)
    bbw_prev = _group_lag(df, "bb_bandwidth", 1)
    dist20 = _num(df, "dist_ema_20")
    dist20_prev = _group_lag(df, "dist_ema_20", 1)
    dist20_lag2 = _group_lag(df, "dist_ema_20", 2)
    dist20_lag3 = _group_lag(df, "dist_ema_20", 3)
    sn = _num(df, "score_N")
    sn_prev = _group_lag(df, "score_N", 1)
    sc = _num(df, "score_C")
    sv = _num(df, "score_V")
    sp = _num(df, "score_P")
    sv_prev = _group_lag(df, "score_V", 1)
    sp_prev = _group_lag(df, "score_P", 1)

    macd = _num(df, "macd_hist")
    rsi = _num(df, "rsi_14")
    bb_pct = _num(df, "bb_pct_b")
    bbw = _num(df, "bb_bandwidth")
    rvol = _num(df, "rvol_30")
    ema20 = _num(df, "ema_20")
    ema50 = _num(df, "ema_50")
    gap = _num(df, "Rotation_Gap")

    out = df[["timestamp", "asset_id"]].copy()

    out["price_above_ema20"] = dist20 > 0
    out["price_above_ema50"] = _num(df, "dist_ema_50") > 0
    out["ema20_gt_ema50"] = ema20 > ema50
    out["macd_hist_pos"] = macd > 0
    out["macd_hist_pos_rising"] = (macd > 0) & (macd > macd_prev)
    out["rsi_40_60"] = (rsi >= 40) & (rsi <= 60)
    out["rsi_45_65"] = (rsi >= 45) & (rsi <= 65)
    out["rsi_le_70"] = rsi <= 70
    out["rsi_cross_up_50"] = (rsi_prev < 50) & (rsi >= 50)
    out["bb_pctb_mid_upper"] = (bb_pct >= 0.4) & (bb_pct <= 0.9)
    out["bb_bandwidth_expanding"] = bbw > bbw_prev
    out["rvol_gt_1"] = rvol > 1.0
    out["rvol_gt_1_5"] = rvol > 1.5
    out["dist_ema20_le_10pct"] = dist20 <= 0.10
    out["dist_ema20_le_5pct"] = dist20 <= 0.05
    out["near_ema20_m5_p10"] = (dist20 >= -0.05) & (dist20 <= 0.10)

    # Reclaim: crossed from <=0 to >0 on any of today / yesterday / day-before
    cross_t = (dist20_prev <= 0) & (dist20 > 0)
    cross_1 = (dist20_lag2 <= 0) & (dist20_prev > 0)
    cross_2 = (dist20_lag3 <= 0) & (dist20_lag2 > 0)
    out["reclaim_ema20_3d"] = cross_t | cross_1.fillna(False) | cross_2.fillna(False)

    # Quiet→rising N: prior day below 50 (quieter), today higher than prior
    if "score_N" in df.columns and sn.notna().any():
        out["quiet_rising_n"] = (sn_prev < 50) & (sn > sn_prev)
    else:
        out["quiet_rising_n"] = False

    if "Rotation_Gap" in df.columns and gap.notna().any():
        out["gap_gt_0"] = gap > 0
        out["gap_gt_10"] = gap > 10
        out["gap_gt_m10"] = gap > -10
    else:
        out["gap_gt_0"] = False
        out["gap_gt_10"] = False
        out["gap_gt_m10"] = False

    # Compression recovering: score_C low (<45) and V or P rising
    if all(c in df.columns for c in ("score_C", "score_V", "score_P")) and sc.notna().any():
        rising_v = sv > sv_prev
        rising_p = sp > sp_prev
        out["compression_recovering"] = (sc < 45) & (rising_v | rising_p)
    else:
        out["compression_recovering"] = False

    # Normalize to nullable bool; NaN inputs → False for support counting simplicity
    for d in SETUP_FLAG_DEFS:
        col = d.flag_id
        if col not in out.columns:
            out[col] = False
        else:
            out[col] = out[col].fillna(False).astype(bool)

    return out


def available_flag_ids(flags: pd.DataFrame) -> list[str]:
    """Flag columns present in the frame (order matches SETUP_FLAG_DEFS)."""
    return [d.flag_id for d in SETUP_FLAG_DEFS if d.flag_id in flags.columns]


def plain_for_flag(flag_id: str) -> str:
    for d in SETUP_FLAG_DEFS:
        if d.flag_id == flag_id:
            return d.plain_english
    return flag_id


def plain_for_setup(flag_ids: tuple[str, ...] | list[str]) -> str:
    parts = [plain_for_flag(f) for f in flag_ids]
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(parts)


# ---------------------------------------------------------------------------
# Support / selection
# ---------------------------------------------------------------------------


def _mask_for_flags(df: pd.DataFrame, flag_ids: tuple[str, ...] | list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for fid in flag_ids:
        if fid not in df.columns:
            return pd.Series(False, index=df.index)
        mask = mask & df[fid].fillna(False).astype(bool)
    return mask


def compute_flag_support(
    spike_flags: pd.DataFrame,
    control_flags: pd.DataFrame,
    flag_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Support (% true) on spike T−1 vs control for each single flag."""
    ids = flag_ids or available_flag_ids(spike_flags)
    n_s = int(len(spike_flags))
    n_c = int(len(control_flags))
    rows: list[dict[str, Any]] = []
    for fid in ids:
        if fid not in spike_flags.columns and fid not in control_flags.columns:
            continue
        s_mask = (
            spike_flags[fid].fillna(False).astype(bool)
            if fid in spike_flags.columns
            else pd.Series(False, index=spike_flags.index)
        )
        c_mask = (
            control_flags[fid].fillna(False).astype(bool)
            if fid in control_flags.columns
            else pd.Series(False, index=control_flags.index)
        )
        s_cnt = int(s_mask.sum())
        c_cnt = int(c_mask.sum())
        s_sup = s_cnt / n_s if n_s else float("nan")
        c_sup = c_cnt / n_c if n_c else float("nan")
        lift = (
            s_sup / c_sup
            if (n_s and n_c and c_sup and c_sup > 0 and pd.notna(s_sup))
            else float("nan")
        )
        rows.append(
            {
                "setup_id": fid,
                "flag_ids": fid,
                "plain_english": plain_for_flag(fid),
                "is_pair": False,
                "spike_count": s_cnt,
                "spike_n": n_s,
                "spike_support": s_sup,
                "ctrl_count": c_cnt,
                "ctrl_n": n_c,
                "ctrl_support": c_sup,
                "lift": lift,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["lift", "spike_support", "spike_count"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
    return out


def compute_pair_support(
    spike_flags: pd.DataFrame,
    control_flags: pd.DataFrame,
    top_singles: list[str],
) -> pd.DataFrame:
    """Support for 2-flag combinations among top frequent singles."""
    n_s = int(len(spike_flags))
    n_c = int(len(control_flags))
    rows: list[dict[str, Any]] = []
    for a, b in combinations(top_singles, 2):
        if a not in spike_flags.columns or b not in spike_flags.columns:
            continue
        s_mask = _mask_for_flags(spike_flags, (a, b))
        c_mask = _mask_for_flags(control_flags, (a, b))
        s_cnt = int(s_mask.sum())
        c_cnt = int(c_mask.sum())
        s_sup = s_cnt / n_s if n_s else float("nan")
        c_sup = c_cnt / n_c if n_c else float("nan")
        lift = (
            s_sup / c_sup
            if (n_s and n_c and c_sup and c_sup > 0 and pd.notna(s_sup))
            else float("nan")
        )
        sid = f"{a}+{b}"
        rows.append(
            {
                "setup_id": sid,
                "flag_ids": f"{a}|{b}",
                "plain_english": plain_for_setup((a, b)),
                "is_pair": True,
                "spike_count": s_cnt,
                "spike_n": n_s,
                "spike_support": s_sup,
                "ctrl_count": c_cnt,
                "ctrl_n": n_c,
                "ctrl_support": c_sup,
                "lift": lift,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["lift", "spike_support", "spike_count"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
    return out


def select_shared_setups(
    single_support: pd.DataFrame,
    pair_support: pd.DataFrame | None = None,
    *,
    lift_min: float = DEFAULT_LIFT_MIN,
    min_spike_count: int = DEFAULT_MIN_SPIKE_COUNT,
    top_n: int = DEFAULT_TOP_SETUPS,
    min_n: int = DEFAULT_MIN_SETUPS,
) -> list[SharedSetup]:
    """Keep setups where spike support is meaningfully higher than control.

    Criteria: lift >= lift_min AND spike_count >= min_spike_count.
    Mix strong singles first (clear shared flags), then non-redundant pairs,
    ranked by lift. Caps how often one flag can dominate the list.
    """
    def _qualifying(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        ok = df.loc[
            (df["spike_count"] >= int(min_spike_count))
            & (df["lift"].notna())
            & (df["lift"] >= float(lift_min))
        ].copy()
        if ok.empty:
            return ok
        return ok.sort_values(
            ["lift", "spike_support", "spike_count"],
            ascending=[False, False, False],
            kind="mergesort",
        )

    singles_ok = _qualifying(single_support)
    pairs_ok = _qualifying(pair_support) if pair_support is not None else pd.DataFrame()

    selected_rows: list[pd.Series] = []
    flag_uses: dict[str, int] = {}
    max_flag_uses = 2  # avoid every combo repeating the same flag

    def _can_add(fids: tuple[str, ...], is_pair: bool) -> bool:
        if is_pair:
            # skip if both flags already appear in the list
            if all(flag_uses.get(f, 0) >= 1 for f in fids):
                return False
            # skip if any flag already used too often
            if any(flag_uses.get(f, 0) >= max_flag_uses for f in fids):
                return False
        return True

    def _add(row: pd.Series) -> None:
        fids = tuple(str(row["flag_ids"]).split("|"))
        selected_rows.append(row)
        for f in fids:
            flag_uses[f] = flag_uses.get(f, 0) + 1

    # 1) Take top qualifying singles (up to half of slots, at least 2 if available)
    n_single_slots = max(2, int(top_n) // 2)
    for _, row in singles_ok.iterrows():
        if len(selected_rows) >= n_single_slots or len(selected_rows) >= int(top_n):
            break
        _add(row)

    # 2) Fill with pairs that add diversity
    for _, row in pairs_ok.iterrows():
        if len(selected_rows) >= int(top_n):
            break
        fids = tuple(str(row["flag_ids"]).split("|"))
        if not _can_add(fids, is_pair=True):
            continue
        _add(row)

    # 3) If still short, add remaining singles by lift
    have = {str(r["setup_id"]) for r in selected_rows}
    for _, row in singles_ok.iterrows():
        if len(selected_rows) >= int(top_n):
            break
        if str(row["setup_id"]) in have:
            continue
        _add(row)
        have.add(str(row["setup_id"]))

    # 4) Soft fill to min_n with lift >= 1.0 (still min count)
    if len(selected_rows) < int(min_n):
        frames = [single_support]
        if pair_support is not None and len(pair_support):
            frames.append(pair_support)
        all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rest = all_df.loc[
            (all_df["spike_count"] >= int(min_spike_count)) & (all_df["lift"].notna())
        ].sort_values(
            ["lift", "spike_support", "spike_count"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        for _, row in rest.iterrows():
            if len(selected_rows) >= int(min_n):
                break
            if str(row["setup_id"]) in have:
                continue
            if float(row["lift"]) < 1.0:
                continue
            fids = tuple(str(row["flag_ids"]).split("|"))
            if bool(row.get("is_pair", False)) and not _can_add(fids, is_pair=True):
                continue
            _add(row)
            have.add(str(row["setup_id"]))

    # Re-rank selected by lift for the final reported order
    selected_rows.sort(
        key=lambda r: (float(r["lift"]), float(r["spike_support"]), int(r["spike_count"])),
        reverse=True,
    )

    out: list[SharedSetup] = []
    for row in selected_rows[: int(top_n)]:
        fids = tuple(str(row["flag_ids"]).split("|"))
        out.append(
            SharedSetup(
                setup_id=str(row["setup_id"]),
                plain_english=str(row["plain_english"]),
                flag_ids=fids,
                spike_support=float(row["spike_support"]),
                ctrl_support=float(row["ctrl_support"]),
                lift=float(row["lift"]),
                spike_count=int(row["spike_count"]),
                spike_n=int(row["spike_n"]),
                ctrl_count=int(row["ctrl_count"]),
                ctrl_n=int(row["ctrl_n"]),
                is_pair=bool(row.get("is_pair", len(fids) > 1)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _flags_at_signal_dates(
    spikes: pd.DataFrame,
    flags_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Join flag rows onto spike events at signal_date (T−1)."""
    if spikes is None or len(spikes) == 0:
        return pd.DataFrame()
    sp = spikes.copy()
    sp["signal_date"] = pd.to_datetime(sp["signal_date"]).dt.normalize()
    sp["asset_id"] = sp["asset_id"].astype(str)
    fl = flags_panel.copy()
    fl["timestamp"] = pd.to_datetime(fl["timestamp"]).dt.normalize()
    fl["asset_id"] = fl["asset_id"].astype(str)
    flag_cols = [c for c in available_flag_ids(fl)]
    merged = sp.merge(
        fl[["timestamp", "asset_id", *flag_cols]].rename(
            columns={"timestamp": "signal_date"}
        ),
        on=["signal_date", "asset_id"],
        how="left",
    )
    for c in flag_cols:
        merged[c] = merged[c].fillna(False).astype(bool)
    return merged


def _control_flag_rows(
    panel: pd.DataFrame,
    flags_panel: pd.DataFrame,
    *,
    before: pd.Timestamp | None,
    n: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Non-spike eligible days (discover window) with flags attached."""
    ctrl = panel.loc[~panel["is_spike_day"].fillna(False)].copy()
    if before is not None:
        ctrl = ctrl.loc[
            pd.to_datetime(ctrl["timestamp"]).dt.normalize()
            < pd.Timestamp(before).normalize()
        ]
    if ctrl.empty:
        return pd.DataFrame()
    if n is not None and n > 0 and len(ctrl) > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(ctrl.index.to_numpy(), size=int(n), replace=False)
        ctrl = ctrl.loc[idx].copy()
    fl = flags_panel.copy()
    fl["timestamp"] = pd.to_datetime(fl["timestamp"]).dt.normalize()
    fl["asset_id"] = fl["asset_id"].astype(str)
    ctrl["timestamp"] = pd.to_datetime(ctrl["timestamp"]).dt.normalize()
    ctrl["asset_id"] = ctrl["asset_id"].astype(str)
    flag_cols = available_flag_ids(fl)
    out = ctrl.merge(
        fl[["timestamp", "asset_id", *flag_cols]],
        on=["timestamp", "asset_id"],
        how="left",
    )
    for c in flag_cols:
        out[c] = out[c].fillna(False).astype(bool)
    return out


def evaluate_setups_on_panel(
    panel_flags: pd.DataFrame,
    setups: list[SharedSetup],
    *,
    label_col: str,
    min_fires: int = 5,
    beat_lift: float = 1.10,
) -> pd.DataFrame:
    """Hit-rate of each selected setup vs base rate on a labeled panel."""
    if panel_flags is None or len(panel_flags) == 0 or not setups:
        return pd.DataFrame()
    y = panel_flags[label_col].fillna(False).astype(bool)
    base = float(y.mean()) if len(y) else float("nan")
    rows = []
    for setup in setups:
        mask = _mask_for_flags(panel_flags, setup.flag_ids)
        n_fire = int(mask.sum())
        if n_fire == 0:
            hit = float("nan")
            n_hit = 0
        else:
            n_hit = int((mask & y).sum())
            hit = n_hit / n_fire
        lift = (
            hit / base if (base and base > 0 and pd.notna(hit)) else float("nan")
        )
        rows.append(
            {
                "setup_id": setup.setup_id,
                "plain_english": setup.plain_english,
                "n_panel": int(len(panel_flags)),
                "n_fire": n_fire,
                "n_hit": n_hit,
                "hit_rate": hit,
                "base_rate": base,
                "lift": lift,
                "beats_base": bool(
                    n_fire >= min_fires
                    and pd.notna(hit)
                    and base > 0
                    and lift >= beat_lift
                ),
                "n_fire_ok": n_fire >= min_fires,
                "label": label_col,
            }
        )
    return pd.DataFrame(rows)


def build_dual_threshold_panel(
    market: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    window: int = PRIMARY_WINDOW,
) -> pd.DataFrame:
    """Eligible panel with is_spike_50 and is_spike_100 labels (same window)."""
    fwd = forward_max_return_by_asset(market, window)
    col = f"fwd_max_ret_{window}d"
    elig = eligible.copy()
    keep_cols = ["timestamp", "asset_id"]
    if "band" in elig.columns:
        keep_cols.append("band")
    elig = elig[keep_cols].drop_duplicates(["timestamp", "asset_id"], keep="first")
    elig["timestamp"] = (
        pd.to_datetime(elig["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    elig["asset_id"] = elig["asset_id"].astype(str)
    panel = elig.merge(fwd, on=["timestamp", "asset_id"], how="inner")
    panel["is_spike_50"] = panel[col] >= THRESH_50
    panel["is_spike_100"] = panel[col] >= THRESH_100
    # alias used by control sampling
    panel["is_spike_day"] = panel["is_spike_50"]
    return panel


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_spike_shared_setups(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    features: pd.DataFrame | None = None,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    window: int = PRIMARY_WINDOW,
    secondary_window: int = SECONDARY_WINDOW,
    explore_frac: float = DEFAULT_EXPLORE_FRAC,
    feature_model: str = "C",
    lift_min: float = DEFAULT_LIFT_MIN,
    min_spike_count: int = DEFAULT_MIN_SPIKE_COUNT,
    top_singles_for_pairs: int = DEFAULT_TOP_SINGLES_FOR_PAIRS,
    top_setups: int = DEFAULT_TOP_SETUPS,
    control_seed: int = 42,
    max_control: int | None = 5000,
) -> SharedSetupsResult:
    """Discover shared boolean setups on early spikes; validate later for +50/+100."""
    eligible = _eligible_membership(membership)
    n_assets = int(eligible["asset_id"].nunique()) if len(eligible) else 0

    # Spikes at the discover threshold (default +50%); +100% is validation-only label
    spikes = detect_spikes(
        market,
        threshold=threshold,
        window=window,
        secondary_window=secondary_window,
        eligible=eligible,
    )

    indicators = compute_tech_indicators(market)
    flags_panel = build_setup_flags(indicators, features, model=feature_model)

    # Also attach indicator snapshots for audit parity (not used for rules)
    spikes_ind = snapshot_pre_spike(spikes, indicators, features, model=feature_model)

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

    single_support = compute_flag_support(spike_flags_disc, control)
    # Top singles by spike frequency (then lift) to bound pair explosion
    if len(single_support):
        by_freq = single_support.sort_values(
            ["spike_count", "lift"],
            ascending=[False, False],
            kind="mergesort",
        )
        top_singles = [
            str(x)
            for x in by_freq["setup_id"].head(int(top_singles_for_pairs)).tolist()
        ]
    else:
        top_singles = []
    pair_support = compute_pair_support(spike_flags_disc, control, top_singles)

    selected = select_shared_setups(
        single_support,
        pair_support,
        lift_min=lift_min,
        min_spike_count=min_spike_count,
        top_n=top_setups,
    )

    # Later-period panel with flags
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
        panel_v_flags, selected, label_col="is_spike_50"
    )
    val_100 = evaluate_setups_on_panel(
        panel_v_flags, selected, label_col="is_spike_100"
    )

    any_50 = bool(val_50["beats_base"].any()) if len(val_50) else False
    any_100 = bool(val_100["beats_base"].any()) if len(val_100) else False

    pcol = f"fwd_max_ret_{window}d"
    scol = f"fwd_max_ret_{secondary_window}d"
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
        "lift_min": lift_min,
        "min_spike_count": min_spike_count,
        "n_selected": len(selected),
        "base_rate_50": base_50,
        "base_rate_100": base_100,
        "any_setup_beats_50": any_50,
        "any_setup_beats_100": any_100,
        "median_fwd_7d": float(spikes[pcol].median())
        if len(spikes) and pcol in spikes.columns
        else np.nan,
        "median_fwd_14d": float(spikes[scol].median())
        if len(spikes) and scol in spikes.columns
        else np.nan,
    }

    return SharedSetupsResult(
        spikes=spikes_ind,
        discover=discover,
        validate=validate,
        flags_panel=flags_panel,
        single_support=single_support,
        pair_support=pair_support,
        selected=selected,
        validation_50=val_50,
        validation_100=val_100,
        cut_date=pd.Timestamp(cut).normalize() if cut is not None else None,
        split_info=split_info,
        base_rate_50=base_50,
        base_rate_100=base_100,
        n_eligible_assets=n_assets,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def write_shared_setups_report(
    result: SharedSetupsResult,
    path: Path | str,
) -> Path:
    """Plain-language markdown report (no jargon dump)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M %Z")
    s = result.summary
    split = result.split_info

    lines: list[str] = []
    lines.append("# Shared setups before spikes — cross-coin flags")
    lines.append("")
    lines.append(f"**When:** {now} (Africa/Lagos)")
    lines.append(
        "**Status:** research check only — **not alpha**, not a trading system, "
        "not a live bot"
    )
    lines.append(
        "This replaces the old “average indicator → hand template” step with "
        "**shared yes/no setups** that many coins showed the day before a spike."
    )
    lines.append("")

    lines.append("## Short answers")
    lines.append("")
    lines.append(
        f"- Spikes used for discovery (default ≥ +50% in {s.get('window', 7)}d): "
        f"**{s.get('n_spikes_total', 0)}** total; "
        f"**{s.get('n_spikes_discover', 0)}** earlier / "
        f"**{s.get('n_spikes_validate', 0)}** later "
        f"(eligible assets ≈ **{s.get('n_eligible_assets', 0)}**)."
    )
    lines.append(
        f"- Cut date (first later day): **{s.get('cut_date', 'n/a')}**."
    )
    lines.append(
        f"- Shared setups kept (lift ≥ {s.get('lift_min', DEFAULT_LIFT_MIN)} "
        f"vs normal days, min count {s.get('min_spike_count', DEFAULT_MIN_SPIKE_COUNT)}): "
        f"**{s.get('n_selected', 0)}**."
    )

    if len(result.validation_50) == 0:
        lines.append(
            "- Later check +50% / +100%: **Could not test** (empty later panel)."
        )
        overall = "Incomplete — no later-period test."
    else:
        y50 = "Yes (weak)" if s.get("any_setup_beats_50") else "No"
        y100 = "Yes (weak)" if s.get("any_setup_beats_100") else "No"
        lines.append(
            f"- Did any shared setup beat chance later for **+50%** in "
            f"{s.get('window', 7)}d? **{y50}**."
        )
        lines.append(
            f"- Did any shared setup beat chance later for **+100%** in "
            f"{s.get('window', 7)}d? **{y100}**."
        )
        if s.get("any_setup_beats_50") or s.get("any_setup_beats_100"):
            overall = (
                "At least one shared setup edged above the later-period base rate "
                "on one of the thresholds. That is a weak, noisy reading — "
                "**no alpha claim**, do not trade this."
            )
        else:
            overall = (
                "No selected shared setup clearly beat chance on the later period "
                "for +50% or +100%. Do **not** claim a predictive pattern."
            )
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append(overall)
    lines.append("")

    lines.append("## What we did (plain)")
    lines.append("")
    lines.append(
        "1. Built a library of **yes/no setup flags** on each coin-day "
        "(price vs EMA, trend, MACD rising, RSI bands, Bollinger mid/upper, "
        "volume heat, EMA reclaim, and Gap / N / compression when scores exist)."
    )
    lines.append(
        "2. For each earlier **spike**, looked at the day **before** (T−1) and "
        "counted how often each flag (and frequent 2-flag combos) was true — "
        "that is **spike support**."
    )
    lines.append(
        "3. Compared to **normal non-spike days** in the same earlier window "
        "(control). Kept setups where spikes showed the flag **meaningfully more "
        f"often** (lift ≥ {s.get('lift_min', DEFAULT_LIFT_MIN)}, "
        f"at least {s.get('min_spike_count', DEFAULT_MIN_SPIKE_COUNT)} spike hits)."
    )
    lines.append(
        "4. Locked the top shared setups from **earlier data only** "
        "(no peeking at later days)."
    )
    lines.append(
        f"5. On the **later** period only, asked: when a setup fires, how often "
        f"does the coin hit **+50%** and **+100%** within {s.get('window', 7)} days, "
        "versus the everyday base rates?"
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
    lines.append(
        f"- Sample sizes are **small**. Treat every percentage with caution."
    )
    lines.append("")

    lines.append("## Shared setups ranked (discover only)")
    lines.append("")
    lines.append(
        "Support = share of earlier spike T−1 days where the setup was true. "
        "Lift = spike support ÷ control support."
    )
    lines.append("")

    if result.selected:
        lines.append(
            "| rank | setup | spike support | control support | lift | spike hits |"
        )
        lines.append("|------|------|------|------|------|------|")
        for i, setup in enumerate(result.selected, 1):
            lines.append(
                f"| {i} | {setup.setup_id} | {_fmt_pct(setup.spike_support)} | "
                f"{_fmt_pct(setup.ctrl_support)} | {_fmt_num(setup.lift)}x | "
                f"{setup.spike_count}/{setup.spike_n} |"
            )
        lines.append("")
        lines.append("### In plain English")
        lines.append("")
        for i, setup in enumerate(result.selected, 1):
            kind = "combo" if setup.is_pair else "single"
            lines.append(
                f"{i}. **{setup.setup_id}** ({kind}): {setup.plain_english} "
                f"— true on {_fmt_pct(setup.spike_support)} of earlier pre-spike days "
                f"vs {_fmt_pct(setup.ctrl_support)} of normal days "
                f"(lift {_fmt_num(setup.lift)}x)."
            )
        lines.append("")
    else:
        lines.append(
            "_No setups passed the lift and minimum-count filter on discover._"
        )
        lines.append("")

    # Top singles table for honesty even if not selected
    if len(result.single_support):
        lines.append("### All single flags (discover support)")
        lines.append("")
        lines.append(
            "| flag | spike support | control support | lift | spike hits |"
        )
        lines.append("|------|------|------|------|------|")
        for _, r in result.single_support.head(20).iterrows():
            lines.append(
                f"| {r['setup_id']} | {_fmt_pct(r['spike_support'])} | "
                f"{_fmt_pct(r['ctrl_support'])} | {_fmt_num(r['lift'])}x | "
                f"{int(r['spike_count'])}/{int(r['spike_n'])} |"
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

    def _val_table(vt: pd.DataFrame, label: str) -> None:
        lines.append(f"### {label}")
        lines.append("")
        if not len(vt):
            lines.append("_No validation rows._")
            lines.append("")
            return
        lines.append(
            "| setup | fires | hits | hit rate | base | lift | beats base? |"
        )
        lines.append("|------|------|------|------|------|------|------|")
        for _, r in vt.iterrows():
            beat = (
                "yes"
                if r["beats_base"]
                else ("n/a (few fires)" if not r["n_fire_ok"] else "no")
            )
            lines.append(
                f"| {r['setup_id']} | {int(r['n_fire'])} | {int(r['n_hit'])} | "
                f"{_fmt_pct(r['hit_rate'])} | {_fmt_pct(r['base_rate'])} | "
                f"{_fmt_num(r['lift'])}x | {beat} |"
            )
        lines.append("")

    _val_table(result.validation_50, f"+50% within {s.get('window', 7)}d")
    _val_table(result.validation_100, f"+100% within {s.get('window', 7)}d")

    lines.append(
        "We mark **beats base** only when the setup fired at least 5 times "
        "on the later panel **and** lift ≥ 1.10. Small samples can look lucky."
    )
    lines.append("")

    lines.append("## Honesty box")
    lines.append("")
    lines.append(
        f"- Discover spikes: **{s.get('n_spikes_discover', 0)}**. "
        f"Later spikes: **{s.get('n_spikes_validate', 0)}**. "
        "That is a thin sample."
    )
    lines.append(
        "- Setups were chosen from earlier data only; later numbers are a "
        "one-shot check, not a license to trade."
    )
    lines.append("- Short history, crypto noise, overlapping themes — easy to overfit.")
    lines.append("- **No alpha claim.** Do not treat this as a system.")
    lines.append(
        "- Old +50/+100 spike-backward reports are left unchanged; this is a "
        "separate shared-setups note."
    )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_shared_setups_outputs(
    result: SharedSetupsResult,
    out_dir: Path | str,
) -> dict[str, Path]:
    """Save support tables and validation as parquet (separate from old spike dirs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    p1 = out_dir / "shared_setup_singles.parquet"
    result.single_support.to_parquet(p1, index=False)
    paths["singles"] = p1
    p2 = out_dir / "shared_setup_pairs.parquet"
    result.pair_support.to_parquet(p2, index=False)
    paths["pairs"] = p2
    if result.selected:
        sel = pd.DataFrame(
            [
                {
                    "setup_id": s.setup_id,
                    "plain_english": s.plain_english,
                    "flag_ids": "|".join(s.flag_ids),
                    "spike_support": s.spike_support,
                    "ctrl_support": s.ctrl_support,
                    "lift": s.lift,
                    "spike_count": s.spike_count,
                    "spike_n": s.spike_n,
                    "is_pair": s.is_pair,
                }
                for s in result.selected
            ]
        )
        p3 = out_dir / "shared_setups_selected.parquet"
        sel.to_parquet(p3, index=False)
        paths["selected"] = p3
    if len(result.validation_50):
        p4 = out_dir / "shared_setup_val_50.parquet"
        result.validation_50.to_parquet(p4, index=False)
        paths["val_50"] = p4
    if len(result.validation_100):
        p5 = out_dir / "shared_setup_val_100.parquet"
        result.validation_100.to_parquet(p5, index=False)
        paths["val_100"] = p5
    return paths
