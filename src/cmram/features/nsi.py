"""Narrative Saturation Index (NSI) and component legs.

NSI = social + price extension + volume exhaustion + momentum deceleration
(research §11). Social is **not** imputed: ``nsi_social`` is null and
``n_available=false`` for Phase 1.

Price extension (higher = more extended / more crowded on price):
  ``nsi_px_ret = close / close.shift(14) - 1``
  ``nsi_px_sma = close / sma(close, 20) - 1``
  ``nsi_price_ext`` = mean of their CS percentiles.

Volume exhaustion (higher = climax / fading participation):
  ``nsi_ve_rvol = RVOL 3/30`` (clipped, same as V)
  ``nsi_ve_accel_fade = -v_log_accel_7``  (deceleration of log-volume)
  ``nsi_vol_exh`` = mean of their CS percentiles.

Momentum deceleration (higher = growth rate slowing):
  ``ret_7 = close / close.shift(7) - 1``
  ``prev_ret_7 = close.shift(7) / close.shift(14) - 1``
  ``nsi_md_raw = prev_ret_7 - ret_7``
  ``nsi_mom_dec`` = CS percentile of that raw value.

Model A uses price + momentum legs only.
Model B uses price + volume + momentum (still no social).
"""

from __future__ import annotations

import pandas as pd

from cmram.features.lookbacks import DEFAULT_LOOKBACKS
from cmram.features.prep import group_rolling, prepare_market, safe_div, simple_return
from cmram.features.rank import mean_available

NSI_PRICE_COLS = ("nsi_px_ret", "nsi_px_sma")
NSI_VOL_COLS = ("nsi_ve_rvol", "nsi_ve_accel_fade")
NSI_MOM_COLS = ("nsi_md_raw",)


def nsi_inputs(
    market: pd.DataFrame,
    volume_raw: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
) -> pd.DataFrame:
    """Raw NSI legs, one row per (timestamp, asset_id).

    ``volume_raw`` is the output of ``volume_inputs`` (for RVOL + log-accel).
    """
    lb = dict(DEFAULT_LOOKBACKS)
    if lookbacks:
        lb.update(lookbacks)
    df = prepare_market(market)
    ext_n = int(lb["extension_days"])
    sma_n = int(lb["sma_days"])
    mom_n = int(lb["mom_dec_days"])

    px_ret = simple_return(df["close"], df["asset_id"], ext_n)
    sma = group_rolling(df, "close", sma_n, "mean")
    px_sma = safe_div(df["close"], sma) - 1.0

    ret_now = simple_return(df["close"], df["asset_id"], mom_n)
    close_t = df["close"]
    g = df["asset_id"]
    close_tm = close_t.groupby(g, sort=False).shift(mom_n)
    close_t2m = close_t.groupby(g, sort=False).shift(2 * mom_n)
    ret_prev = safe_div(close_tm, close_t2m) - 1.0
    mom_dec = ret_prev - ret_now

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "asset_id": df["asset_id"],
            "nsi_px_ret": px_ret,
            "nsi_px_sma": px_sma,
            "nsi_md_raw": mom_dec,
        }
    )
    vol = volume_raw[["timestamp", "asset_id", "v_rvol_3_30", "v_log_accel_7"]].copy()
    vol["timestamp"] = pd.to_datetime(vol["timestamp"]).dt.tz_localize(None).dt.normalize()
    out = out.merge(vol, on=["timestamp", "asset_id"], how="left", sort=False)
    out["nsi_ve_rvol"] = out["v_rvol_3_30"]
    out["nsi_ve_accel_fade"] = -out["v_log_accel_7"]
    return out.drop(columns=["v_rvol_3_30", "v_log_accel_7"])


def compute_nsi(
    *,
    nsi_social: pd.Series | None = None,
    nsi_price_ext: pd.Series | None = None,
    nsi_vol_exh: pd.Series | None = None,
    nsi_mom_dec: pd.Series | None = None,
) -> pd.Series:
    """Equal-weight mean of provided NSI legs; skip None / all-NaN.

    Does not invent a social leg. Pass social=None for Phase 1.
    """
    parts: dict[str, pd.Series] = {}
    index = None
    for name, s in (
        ("social", nsi_social),
        ("price_ext", nsi_price_ext),
        ("vol_exh", nsi_vol_exh),
        ("mom_dec", nsi_mom_dec),
    ):
        if s is None:
            continue
        parts[name] = s
        if index is None:
            index = s.index
    if not parts or index is None:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(parts, index=index)
    return mean_available(df, list(parts.keys()))
