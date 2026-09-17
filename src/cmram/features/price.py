"""Price Confirmation Score (P) — volume/attention translating into tape.

Formulas (PIT):

1. Higher-lows
   Two adjacent non-overlapping windows of length ``higher_lows_short_days``
   (default 10) inside ``higher_lows_long_days`` (default 20):
   ``ll_recent = min(low, 10)``
   ``ll_prior  = min(low.shift(10), 10)``
   ``p_higher_lows = (ll_recent - ll_prior) / ll_prior``
   Positive = recent trough above prior trough.

2. Short returns
   ``p_ret_3d = close / close.shift(3) - 1``
   ``p_ret_7d = close / close.shift(7) - 1``

3. Breakout vs 20d high
   Prior 20d high excludes today (known at t, no same-bar high lookahead):
   ``prior_high_20 = max(high.shift(1), 20)``
   ``p_breakout_20 = close / prior_high_20``
   > 1 means close broke the prior 20d high.

4. Relative strength 7d / 14d
   ``rs_n = (1 + ret_n_asset) / (1 + ret_n_bench) - 1``
   Benchmark (documented, PIT):
   - If ``bitcoin`` (or configured id) exists in ``market_daily`` on that date,
     use BTC's n-day return.
   - Else use the **band equal-weight** mean n-day return of assets in
     ``universe_membership`` for that ``(timestamp, band)``, including self.
     Only members as of t; their historical closes ≤ t. No future membership.

   Note: CS *ranks* of RS vs a common (or leave-one-out EW) benchmark equal
   CS ranks of own n-day return. ``p_rs_7d`` therefore duplicates ``p_ret_7d``
   in the ranked score; both are kept because they are specified inputs
   (absolute vs relative). ``p_rs_14d`` adds a 14d horizon not otherwise in P.

``score_P`` = equal-weight mean of the six CS-percentile columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmram.features.lookbacks import DEFAULT_BTC_ASSET_IDS, DEFAULT_LOOKBACKS
from cmram.features.prep import (
    group_rolling,
    prepare_market,
    safe_div,
    simple_return,
)
from cmram.features.rank import cs_percentile, mean_available

P_INPUT_COLS = (
    "p_higher_lows",
    "p_ret_3d",
    "p_ret_7d",
    "p_breakout_20",
    "p_rs_7d",
    "p_rs_14d",
)

RS_BENCHMARK_BTC = "bitcoin"
RS_BENCHMARK_BAND_EW = "band_equal_weight"


def _btc_close_series(
    market: pd.DataFrame,
    btc_asset_ids: tuple[str, ...],
) -> pd.Series | None:
    """Daily BTC close indexed by normalized timestamp, or None if absent."""
    ids = {x.lower() for x in btc_asset_ids}
    m = market.loc[market["asset_id"].str.lower().isin(ids)]
    if m.empty:
        return None
    # Prefer 'bitcoin' if several ids match.
    prefer = [i for i in btc_asset_ids if i.lower() == "bitcoin"]
    chosen = prefer[0] if prefer and (m["asset_id"].str.lower() == "bitcoin").any() else m["asset_id"].iloc[0]
    sub = m.loc[m["asset_id"].str.lower() == str(chosen).lower()]
    if sub.empty:
        sub = m
    s = sub.drop_duplicates("timestamp", keep="last").set_index("timestamp")["close"]
    return s.sort_index()


def price_inputs(
    market: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
) -> pd.DataFrame:
    """Asset-level P inputs (no RS). One row per (timestamp, asset_id)."""
    lb = dict(DEFAULT_LOOKBACKS)
    if lookbacks:
        lb.update(lookbacks)
    df = prepare_market(market)
    hl_s = int(lb["higher_lows_short_days"])
    ret_s = int(lb["ret_short_days"])
    ret_m = int(lb["ret_medium_days"])
    brk = int(lb["breakout_days"])

    ll_recent = group_rolling(df, "low", hl_s, "min")
    low_shifted = df.groupby("asset_id", sort=False)["low"].shift(hl_s)
    tmp = df.assign(_low_shift=low_shifted)
    ll_prior = group_rolling(tmp, "_low_shift", hl_s, "min")
    higher_lows = safe_div(ll_recent - ll_prior, ll_prior)

    ret_3 = simple_return(df["close"], df["asset_id"], ret_s)
    ret_7 = simple_return(df["close"], df["asset_id"], ret_m)

    high_shift = df.groupby("asset_id", sort=False)["high"].shift(1)
    tmp2 = df.assign(_high_shift=high_shift)
    prior_high = group_rolling(tmp2, "_high_shift", brk, "max")
    breakout = safe_div(df["close"], prior_high)

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "asset_id": df["asset_id"],
            "p_higher_lows": higher_lows,
            "p_ret_3d": ret_3,
            "p_ret_7d": ret_7,
            "p_breakout_20": breakout,
            "_ret_rs_short": simple_return(
                df["close"], df["asset_id"], int(lb["rs_short_days"])
            ),
            "_ret_rs_long": simple_return(
                df["close"], df["asset_id"], int(lb["rs_long_days"])
            ),
        }
    )


def attach_relative_strength(
    membership_frame: pd.DataFrame,
    price_raw: pd.DataFrame,
    market: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
    btc_asset_ids: tuple[str, ...] = DEFAULT_BTC_ASSET_IDS,
) -> tuple[pd.DataFrame, str]:
    """Add p_rs_7d / p_rs_14d onto membership rows. Returns (df, benchmark_name).

    ``membership_frame`` must contain timestamp, asset_id, band.
    Benchmark is global for the run: bitcoin if any BTC bars exist, else
    band equal-weight. Dates where BTC is missing fall back to band EW for
    those rows only; ``benchmark_name`` reports the primary choice.
    """
    lb = dict(DEFAULT_LOOKBACKS)
    if lookbacks:
        lb.update(lookbacks)
    n_s, n_l = int(lb["rs_short_days"]), int(lb["rs_long_days"])

    mkt = prepare_market(market)
    btc_close = _btc_close_series(mkt, btc_asset_ids)
    primary = RS_BENCHMARK_BTC if btc_close is not None else RS_BENCHMARK_BAND_EW

    mem = membership_frame.copy()
    mem["timestamp"] = pd.to_datetime(mem["timestamp"]).dt.tz_localize(None).dt.normalize()
    mem["asset_id"] = mem["asset_id"].astype(str)
    pr = price_raw.copy()
    pr["timestamp"] = pd.to_datetime(pr["timestamp"]).dt.tz_localize(None).dt.normalize()
    pr["asset_id"] = pr["asset_id"].astype(str)

    merged = mem.merge(
        pr,
        on=["timestamp", "asset_id"],
        how="left",
        sort=False,
    )

    # Band equal-weight returns (PIT membership at t).
    ew_s = merged.groupby(["timestamp", "band"], sort=False)["_ret_rs_short"].transform(
        "mean"
    )
    ew_l = merged.groupby(["timestamp", "band"], sort=False)["_ret_rs_long"].transform(
        "mean"
    )

    btc_ret_s = None
    btc_ret_l = None
    if btc_close is not None:
        btc_df = btc_close.rename("close").reset_index()
        btc_df["asset_id"] = "_btc_"
        btc_df = prepare_market(btc_df)
        btc_ret_s = (
            pd.DataFrame(
                {
                    "timestamp": btc_df["timestamp"],
                    "_btc_ret_s": simple_return(btc_df["close"], btc_df["asset_id"], n_s),
                    "_btc_ret_l": simple_return(btc_df["close"], btc_df["asset_id"], n_l),
                }
            )
        )
        merged = merged.merge(btc_ret_s, on="timestamp", how="left", sort=False)
        btc_ret_s = merged["_btc_ret_s"]
        btc_ret_l = merged["_btc_ret_l"]

    if btc_ret_s is not None:
        bench_s = btc_ret_s.where(btc_ret_s.notna(), ew_s)
        bench_l = btc_ret_l.where(btc_ret_l.notna(), ew_l)
    else:
        bench_s = ew_s
        bench_l = ew_l

    merged["p_rs_7d"] = safe_div(1.0 + merged["_ret_rs_short"], 1.0 + bench_s) - 1.0
    merged["p_rs_14d"] = safe_div(1.0 + merged["_ret_rs_long"], 1.0 + bench_l) - 1.0
    drop = [c for c in ("_ret_rs_short", "_ret_rs_long", "_btc_ret_s", "_btc_ret_l") if c in merged.columns]
    return merged.drop(columns=drop), primary


def score_P(market: pd.DataFrame, *, lookbacks: dict | None = None) -> pd.Series:
    """Standalone P without RS (no membership): 4 asset-level legs only."""
    raw = price_inputs(market, lookbacks=lookbacks)
    cols = ["p_higher_lows", "p_ret_3d", "p_ret_7d", "p_breakout_20"]
    ranked = raw.copy()
    for col in cols:
        ranked[f"{col}_pct"] = ranked.groupby("timestamp", sort=False)[col].transform(
            cs_percentile
        )
    return mean_available(ranked, [f"{c}_pct" for c in cols])
