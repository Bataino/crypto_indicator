"""Volume Resurrection Score (V) — returning participation.

Formulas (PIT). Volume is floored at ``volume_floor_usd`` (default $1) before
ratios/logs so zeros do not explode. RVOL is clipped at ``rvol_clip`` (default
10) before ranking — a simple damp so climax outliers do not dominate V
(climax belongs in NSI volume exhaustion).

1. RVOL 3d / 30d
   ``rvol = mean(vol, 3) / mean(vol, 30)`` then ``min(rvol, rvol_clip)``

2. 7d vs 28d growth
   ``v_growth = mean(vol, 7) / mean(vol, 28) - 1``

3. 3d velocity
   ``vol_3 = mean(vol, 3)``
   ``v_velocity_3d = vol_3 / vol_3.shift(3) - 1``
   (3d mean vs the prior 3d mean; more robust than 1-day / 1-day)

4. ~7d log-volume acceleration
   ``logv = log(vol_floored)``
   ``log_vel_7 = logv_t - logv_{t-7}``
   ``v_log_accel_7 = log_vel_7 - log_vel_7.shift(7)``
   ``= logv_t - 2 logv_{t-7} + logv_{t-14}``

``score_V`` = equal-weight mean of the four CS-percentile columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmram.features.lookbacks import (
    DEFAULT_LOOKBACKS,
    DEFAULT_RVOL_CLIP,
    DEFAULT_VOLUME_FLOOR_USD,
)
from cmram.features.prep import group_rolling, prepare_market, safe_div
from cmram.features.rank import cs_percentile, mean_available

V_INPUT_COLS = (
    "v_rvol_3_30",
    "v_growth_7_28",
    "v_velocity_3d",
    "v_log_accel_7",
)


def volume_inputs(
    market: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
    volume_floor_usd: float = DEFAULT_VOLUME_FLOOR_USD,
    rvol_clip: float = DEFAULT_RVOL_CLIP,
) -> pd.DataFrame:
    """Raw V inputs, one row per (timestamp, asset_id)."""
    lb = dict(DEFAULT_LOOKBACKS)
    if lookbacks:
        lb.update(lookbacks)
    df = prepare_market(market)
    floor = float(volume_floor_usd)
    clip = float(rvol_clip)
    vol = df["volume_usd"].astype("float64").clip(lower=floor)
    df = df.assign(_vol=vol)

    r_s, r_l = int(lb["rvol_short_days"]), int(lb["rvol_long_days"])
    g_s, g_l = int(lb["vol_growth_short_days"]), int(lb["vol_growth_long_days"])
    vel_n = int(lb["velocity_days"])
    acc_n = int(lb["logvol_accel_days"])

    mean_s = group_rolling(df, "_vol", r_s, "mean")
    mean_l = group_rolling(df, "_vol", r_l, "mean")
    rvol = safe_div(mean_s, mean_l).clip(upper=clip)

    g_short = group_rolling(df, "_vol", g_s, "mean")
    g_long = group_rolling(df, "_vol", g_l, "mean")
    growth = safe_div(g_short, g_long) - 1.0

    vol_k = group_rolling(df, "_vol", vel_n, "mean")
    vol_k_prev = vol_k.groupby(df["asset_id"], sort=False).shift(vel_n)
    velocity = safe_div(vol_k, vol_k_prev) - 1.0

    logv = np.log(vol)
    logv_s = pd.Series(logv, index=df.index)
    log_vel = logv_s - logv_s.groupby(df["asset_id"], sort=False).shift(acc_n)
    log_acc = log_vel - log_vel.groupby(df["asset_id"], sort=False).shift(acc_n)

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "asset_id": df["asset_id"],
            "v_rvol_3_30": rvol,
            "v_growth_7_28": growth,
            "v_velocity_3d": velocity,
            "v_log_accel_7": log_acc,
        }
    )


def score_V(
    market: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
    volume_floor_usd: float = DEFAULT_VOLUME_FLOOR_USD,
    rvol_clip: float = DEFAULT_RVOL_CLIP,
) -> pd.Series:
    """Standalone V: PIT inputs + per-timestamp CS percentile + equal-weight mean."""
    raw = volume_inputs(
        market,
        lookbacks=lookbacks,
        volume_floor_usd=volume_floor_usd,
        rvol_clip=rvol_clip,
    )
    ranked = raw.copy()
    for col in V_INPUT_COLS:
        ranked[f"{col}_pct"] = ranked.groupby("timestamp", sort=False)[col].transform(
            cs_percentile
        )
    return mean_available(ranked, [f"{c}_pct" for c in V_INPUT_COLS])
