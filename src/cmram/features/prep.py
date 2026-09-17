"""Shared market-frame prep for point-in-time feature inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_REQUIRED = ("timestamp", "asset_id", "close")
OHLCV_OPTIONAL = ("open", "high", "low", "volume_usd", "market_cap_usd")


def prepare_market(market: pd.DataFrame) -> pd.DataFrame:
    """Sort by (asset_id, timestamp), fill missing OHLC from close, drop dupes.

    Timestamps are normalized to midnight (tz-naive). Does not look ahead.
    """
    missing = [c for c in OHLCV_REQUIRED if c not in market.columns]
    if missing:
        raise ValueError(f"market missing columns: {missing}")

    df = market.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str)
    close = pd.to_numeric(df["close"], errors="coerce")
    df["close"] = close
    for col in ("open", "high", "low"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(close)
        else:
            df[col] = close
    if "volume_usd" in df.columns:
        df["volume_usd"] = pd.to_numeric(df["volume_usd"], errors="coerce")
    else:
        df["volume_usd"] = np.nan
    if "market_cap_usd" in df.columns:
        df["market_cap_usd"] = pd.to_numeric(df["market_cap_usd"], errors="coerce")

    df = df.sort_values(["asset_id", "timestamp"], kind="mergesort")
    df = df.drop_duplicates(["asset_id", "timestamp"], keep="last")
    return df.reset_index(drop=True)


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise divide; inf and 0-denom → NaN."""
    n = numer.astype("float64")
    d = denom.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = n / d
    return out.replace([np.inf, -np.inf], np.nan)


def group_rolling(
    df: pd.DataFrame,
    col: str,
    window: int,
    func: str,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Backward-looking rolling ``func`` within asset_id (inclusive of t)."""
    mp = window if min_periods is None else min_periods
    g = df.groupby("asset_id", sort=False)[col]
    rolled = g.rolling(window=window, min_periods=mp)
    if func == "mean":
        out = rolled.mean()
    elif func == "std":
        out = rolled.std(ddof=1)
    elif func == "min":
        out = rolled.min()
    elif func == "max":
        out = rolled.max()
    elif func == "sum":
        out = rolled.sum()
    elif func == "median":
        out = rolled.median()
    else:
        raise ValueError(f"unknown rolling func: {func}")
    return out.reset_index(level=0, drop=True)


def log_return(close: pd.Series, asset_id: pd.Series) -> pd.Series:
    """Close-to-close log return within asset; first bar / non-positive → NaN."""
    prev = close.groupby(asset_id, sort=False).shift(1)
    ratio = safe_div(close, prev)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(ratio)
    return pd.Series(lr, index=close.index, dtype="float64")


def simple_return(close: pd.Series, asset_id: pd.Series, n: int) -> pd.Series:
    """n-day simple return close_t / close_{t-n} - 1 (PIT)."""
    prev = close.groupby(asset_id, sort=False).shift(n)
    return safe_div(close, prev) - 1.0
