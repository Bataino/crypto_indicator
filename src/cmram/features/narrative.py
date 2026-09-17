"""Narrative Acceleration Score (N) — attention velocity from social.

V0.1: combine social_daily mentions across sources → attention proxy →
velocity, acceleration, quiet→rising gates → CS percentiles → score_N.

Modes (config ``narrative_mode``):
  - ``quiet_rising`` (default): reward LOW lagged baseline then rising
    attention (Santiment social_volume and other sources). Strong damp /
    hard-zero when the lagged baseline is already crowded. Only positive
    velocity/acceleration from that quiet base contribute.
  - ``legacy``: prior soft 1/(1+log1p(baseline)) multiplier on raw vel/accel.

Does **not** invent social. Empty/missing social → null scores and
``n_available=false``. Attention proxy only — not alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmram.features.lookbacks import DEFAULT_LOOKBACKS
from cmram.features.prep import group_rolling, safe_div
from cmram.features.rank import mean_available

N_INPUT_COLS = (
    "n_velocity",
    "n_accel",
    "n_vel_lowbase",
    "n_accel_lowbase",
)

DEFAULT_NARRATIVE_MODE = "quiet_rising"
# Absolute mention-scale for "quiet" vs "crowded" (not attention_floor).
DEFAULT_N_CROWD_SCALE = 10.0
DEFAULT_N_CROWD_EXP = 3.0
# Hard-zero quiet weight when lagged baseline exceeds this (mention units).
DEFAULT_N_CROWD_HARD_MAX = 80.0


def combine_attention(
    social: pd.DataFrame,
    *,
    method: str = "mean",
    attention_floor: float = 0.1,
) -> pd.DataFrame:
    """Collapse multi-source social_daily to one attention series per day/asset.

    Returns columns: timestamp, asset_id, attention, sources_present (str).
    """
    if social is None or len(social) == 0:
        return pd.DataFrame(
            columns=["timestamp", "asset_id", "attention", "sources_present"]
        )

    df = social.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str)
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce")
    df = df.dropna(subset=["mentions"])
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "asset_id", "attention", "sources_present"]
        )

    per_src = (
        df.groupby(["timestamp", "asset_id", "source"], sort=False)["mentions"]
        .mean()
        .reset_index()
    )
    src_list = (
        per_src.groupby(["timestamp", "asset_id"], sort=False)["source"]
        .agg(lambda s: ",".join(sorted(set(s.astype(str)))))
        .rename("sources_present")
    )
    if method == "sum":
        att = per_src.groupby(["timestamp", "asset_id"], sort=False)["mentions"].sum()
    elif method == "max":
        att = per_src.groupby(["timestamp", "asset_id"], sort=False)["mentions"].max()
    else:
        att = per_src.groupby(["timestamp", "asset_id"], sort=False)["mentions"].mean()

    out = att.rename("attention").reset_index()
    out = out.merge(src_list.reset_index(), on=["timestamp", "asset_id"], how="left")
    out["attention"] = out["attention"].clip(lower=float(attention_floor))
    return out


def _legacy_lowbase_mult(baseline: pd.Series) -> pd.Series:
    """Prior soft boost: 1/(1+log1p(baseline)). Still non-zero when crowded."""
    return 1.0 / (1.0 + np.log1p(baseline.clip(lower=0)))


def _quiet_rising_weight(
    lagged_baseline: pd.Series,
    *,
    crowd_scale: float,
    crowd_exp: float,
    crowd_hard_max: float,
) -> pd.Series:
    """Near 1 when lagged baseline is quiet; steep damp → 0 when crowded.

    Uses absolute mention units (``crowd_scale``), not attention_floor.
    Hard-zeros when lagged baseline > ``crowd_hard_max``.
    """
    scale = max(float(crowd_scale), 1e-9)
    ratio = lagged_baseline.clip(lower=0) / scale
    exp = max(float(crowd_exp), 0.0)
    w = 1.0 / (1.0 + np.power(ratio, exp))
    hard = float(crowd_hard_max)
    if hard > 0:
        w = w.where(lagged_baseline.clip(lower=0) <= hard, 0.0)
    return w


def narrative_inputs(
    social: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
    combine: str = "mean",
    attention_floor: float = 0.1,
    narrative_mode: str | None = None,
) -> pd.DataFrame:
    """Raw N inputs + baseline attention, one row per day/asset.

    ``narrative_mode``: ``quiet_rising`` (default) or ``legacy``.
    """
    lb = dict(DEFAULT_LOOKBACKS)
    if lookbacks:
        lb.update(lookbacks)
    short = int(lb.get("n_vel_short_days") or lb.get("narrative_short_days") or 7)
    long = int(lb.get("n_vel_long_days") or lb.get("narrative_long_days") or 28)
    accel_shift = int(lb.get("n_accel_days") or short)

    mode = str(
        narrative_mode or lb.get("narrative_mode") or DEFAULT_NARRATIVE_MODE
    ).strip().lower()
    if mode not in ("quiet_rising", "legacy"):
        raise ValueError(
            f"narrative_mode must be 'quiet_rising' or 'legacy', got {mode!r}"
        )

    crowd_scale = float(
        lb["n_crowd_scale"] if lb.get("n_crowd_scale") is not None else DEFAULT_N_CROWD_SCALE
    )
    crowd_exp = float(
        lb["n_crowd_exp"] if lb.get("n_crowd_exp") is not None else DEFAULT_N_CROWD_EXP
    )
    crowd_hard_max = float(
        lb["n_crowd_hard_max"]
        if lb.get("n_crowd_hard_max") is not None
        else DEFAULT_N_CROWD_HARD_MAX
    )

    empty_cols = [
        "timestamp",
        "asset_id",
        "attention",
        "n_baseline",
        "n_velocity",
        "n_accel",
        "n_vel_lowbase",
        "n_accel_lowbase",
        "sources_present",
        "n_quiet_weight",
        "narrative_mode",
    ]

    att = combine_attention(social, method=combine, attention_floor=attention_floor)
    if att.empty:
        return pd.DataFrame(columns=empty_cols)

    df = att.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    frames = []
    for aid, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp")
        idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="D")
        daily = g.set_index("timestamp").reindex(idx)
        daily["attention"] = daily["attention"].ffill()
        daily["sources_present"] = daily["sources_present"].ffill()
        daily["asset_id"] = aid
        daily = daily.reset_index().rename(columns={"index": "timestamp"})
        frames.append(daily)
    dens = pd.concat(frames, ignore_index=True)
    dens["timestamp"] = pd.to_datetime(dens["timestamp"]).dt.tz_localize(None).dt.normalize()

    mean_s = group_rolling(
        dens, "attention", short, "mean", min_periods=max(2, short // 2)
    )
    mean_l = group_rolling(
        dens, "attention", long, "mean", min_periods=max(3, long // 2)
    )
    velocity = safe_div(mean_s, mean_l) - 1.0
    vel_prev = velocity.groupby(dens["asset_id"], sort=False).shift(accel_shift)
    accel = velocity - vel_prev
    baseline = mean_l

    dens["n_baseline"] = baseline
    dens["narrative_mode"] = mode

    if mode == "legacy":
        mult = _legacy_lowbase_mult(baseline)
        dens["n_quiet_weight"] = mult
        dens["n_velocity"] = velocity
        dens["n_accel"] = accel
        dens["n_vel_lowbase"] = velocity * mult
        dens["n_accel_lowbase"] = accel * mult
    else:
        # Lagged long baseline: "was quiet before the recent burst?"
        # so a fresh rise does not immediately inflate the quiet check.
        lagged_baseline = baseline.groupby(dens["asset_id"], sort=False).shift(short)
        quiet_w = _quiet_rising_weight(
            lagged_baseline.fillna(baseline),
            crowd_scale=crowd_scale,
            crowd_exp=crowd_exp,
            crowd_hard_max=crowd_hard_max,
        )
        rising_vel = velocity.clip(lower=0.0)
        rising_accel = accel.clip(lower=0.0)
        dens["n_quiet_weight"] = quiet_w
        dens["n_velocity"] = rising_vel * quiet_w
        dens["n_accel"] = rising_accel * quiet_w
        # Extra quiet emphasis on *_lowbase legs
        dens["n_vel_lowbase"] = rising_vel * quiet_w * quiet_w
        dens["n_accel_lowbase"] = rising_accel * quiet_w * quiet_w

    return dens[
        [
            "timestamp",
            "asset_id",
            "attention",
            "n_baseline",
            "n_velocity",
            "n_accel",
            "n_vel_lowbase",
            "n_accel_lowbase",
            "sources_present",
            "n_quiet_weight",
            "narrative_mode",
        ]
    ]


def score_N(
    social: pd.DataFrame | None,
    *,
    lookbacks: dict | None = None,
    index: pd.Index | None = None,
    combine: str = "mean",
    attention_floor: float = 0.1,
    narrative_mode: str | None = None,
) -> pd.Series:
    """Narrative score 0–100 (standalone CS within timestamp). Pipeline ranks in-band."""
    if social is None or len(social) == 0:
        if index is None:
            return pd.Series(dtype="float64", name="score_N")
        return pd.Series(np.nan, index=index, dtype="float64", name="score_N")

    raw = narrative_inputs(
        social,
        lookbacks=lookbacks,
        combine=combine,
        attention_floor=attention_floor,
        narrative_mode=narrative_mode,
    )
    if raw.empty:
        if index is None:
            return pd.Series(dtype="float64", name="score_N")
        return pd.Series(np.nan, index=index, dtype="float64", name="score_N")

    from cmram.features.rank import cs_percentile

    ranked = raw.copy()
    for col in N_INPUT_COLS:
        ranked[f"{col}_pct"] = ranked.groupby("timestamp", sort=False)[col].transform(
            cs_percentile
        )
    scores = mean_available(ranked, [f"{c}_pct" for c in N_INPUT_COLS])
    scores.name = "score_N"
    if index is not None:
        return pd.Series(np.nan, index=index, dtype="float64", name="score_N")
    return scores
