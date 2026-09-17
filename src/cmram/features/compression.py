"""Compression Score (C) — has the move *not* already completed?

Formulas (point-in-time, windows inclusive of t). Higher raw value → more
compression / more unused upside *before* CS percentile ranking:

1. Distance from 20d high
   ``c_dist_20d_high = (max(high, 20) - close) / max(high, 20)``
   0 at the high; larger = further below recent high (not already extended).

2. 10d / 40d range compression
   ``range_n = max(high, n) - min(low, n)``
   ``c_range_comp = 1 - range_10 / range_40``
   Nested windows ⇒ range_10 ≤ range_40; 1 = fully compressed 10d range.

3. Realized-vol ratio 7d / 30d
   ``rv_n = std(log(close_t / close_{t-1}), n)``  (sample std, ddof=1)
   ``c_vol_comp = 1 - rv_7 / rv_30``
   Positive = short vol contracted vs 30d. Can be negative if vol expanded.

4. Sell-pressure fade
   Down-day = ``close < close.shift(1)`` (uses close-to-close because chart-only
   ingest may set O=H=L=C).
   ``sell_frac_n = sum(volume | down-day, n) / sum(volume, n)``
   ``c_sell_fade = sell_frac_20 - sell_frac_5``
   Positive = recent down-volume share lighter than the 20d share.

5. Stabilization
   ``c_stabilize = (min(low, 5) - min(low, 20)) / (max(high, 20) - min(low, 20))``
   0 = still printing the 20d low; >0 = recent lows have lifted off the low.

``score_C`` is the equal-weight mean of the five CS-percentile columns.
These are research definitions, not assumed alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmram.features.lookbacks import DEFAULT_LOOKBACKS
from cmram.features.prep import (
    group_rolling,
    log_return,
    prepare_market,
    safe_div,
)
from cmram.features.rank import cs_percentile, mean_available

C_INPUT_COLS = (
    "c_dist_20d_high",
    "c_range_comp",
    "c_vol_comp",
    "c_sell_fade",
    "c_stabilize",
)


def compression_inputs(
    market: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
) -> pd.DataFrame:
    """Raw C inputs, one row per (timestamp, asset_id). PIT rolling windows."""
    lb = dict(DEFAULT_LOOKBACKS)
    if lookbacks:
        lb.update(lookbacks)
    df = prepare_market(market)
    high_n = int(lb["high_days"])
    r_s, r_l = int(lb["range_short_days"]), int(lb["range_long_days"])
    v_s, v_l = int(lb["vol_short_days"]), int(lb["vol_long_days"])
    s_s, s_l = int(lb["sell_short_days"]), int(lb["sell_long_days"])
    z_s, z_l = int(lb["stabilize_short_days"]), int(lb["stabilize_long_days"])

    high_20 = group_rolling(df, "high", high_n, "max")
    dist = safe_div(high_20 - df["close"], high_20)

    range_s = group_rolling(df, "high", r_s, "max") - group_rolling(df, "low", r_s, "min")
    range_l = group_rolling(df, "high", r_l, "max") - group_rolling(df, "low", r_l, "min")
    range_comp = 1.0 - safe_div(range_s, range_l)

    lr = log_return(df["close"], df["asset_id"])
    tmp = df.assign(_lr=lr)
    rv_s = group_rolling(tmp, "_lr", v_s, "std")
    rv_l = group_rolling(tmp, "_lr", v_l, "std")
    vol_comp = 1.0 - safe_div(rv_s, rv_l)

    prev_close = df.groupby("asset_id", sort=False)["close"].shift(1)
    down = df["close"] < prev_close
    down_vol = df["volume_usd"].where(down, 0.0)
    tmp2 = df.assign(_down_vol=down_vol)
    sell_s = safe_div(
        group_rolling(tmp2, "_down_vol", s_s, "sum"),
        group_rolling(df, "volume_usd", s_s, "sum"),
    )
    sell_l = safe_div(
        group_rolling(tmp2, "_down_vol", s_l, "sum"),
        group_rolling(df, "volume_usd", s_l, "sum"),
    )
    sell_fade = sell_l - sell_s

    min_s = group_rolling(df, "low", z_s, "min")
    min_l = group_rolling(df, "low", z_l, "min")
    max_l = group_rolling(df, "high", z_l, "max")
    stabilize = safe_div(min_s - min_l, max_l - min_l)

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "asset_id": df["asset_id"],
            "c_dist_20d_high": dist,
            "c_range_comp": range_comp,
            "c_vol_comp": vol_comp,
            "c_sell_fade": sell_fade,
            "c_stabilize": stabilize,
        }
    )
    return out


def score_C(market: pd.DataFrame, *, lookbacks: dict | None = None) -> pd.Series:
    """Standalone C: PIT inputs + per-timestamp CS percentile + equal-weight mean.

    Band-aware ranking lives in the pipeline; this helper ranks all assets
    present on each date (for unit tests / debugging).
    """
    raw = compression_inputs(market, lookbacks=lookbacks)
    ranked = raw.copy()
    for col in C_INPUT_COLS:
        ranked[f"{col}_pct"] = ranked.groupby("timestamp", sort=False)[col].transform(
            cs_percentile
        )
    return mean_available(ranked, [f"{c}_pct" for c in C_INPUT_COLS])
