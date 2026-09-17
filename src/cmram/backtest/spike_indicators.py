"""Pre-spike technical indicators (snapshot at T-1). Research only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmram.features.prep import prepare_market, safe_div

INDICATOR_COLS = (
    "ema_20",
    "ema_50",
    "dist_ema_20",
    "dist_ema_50",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "bb_pct_b",
    "bb_bandwidth",
    "rvol_30",
    "days_since_local_low",
    "drawdown_60d",
)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype("float64").ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype("float64").diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = safe_div(avg_gain, avg_loss)
    return 100.0 - safe_div(pd.Series(100.0, index=close.index), 1.0 + rs)


def compute_tech_indicators(market: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA/MACD/RSI/Bollinger/RVOL/drawdown per (timestamp, asset_id).

    All rolling windows are backward-looking and inclusive of t (PIT at close t).
    """
    df = prepare_market(market)
    parts: list[pd.DataFrame] = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        close = g["close"].astype("float64")
        vol = g["volume_usd"].astype("float64")

        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        dist20 = safe_div(close - ema20, ema20)
        dist50 = safe_div(close - ema50, ema50)

        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = _ema(macd_line.fillna(0.0), 9)
        # Recompute signal with NaN-aware: only after macd_line has values
        macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_hist = macd_line - macd_signal

        rsi = _rsi(close, 14)

        mid = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std(ddof=1)
        upper = mid + 2.0 * std
        lower = mid - 2.0 * std
        bb_pct_b = safe_div(close - lower, upper - lower)
        bb_bw = safe_div(upper - lower, mid)

        vol_mean_30 = vol.rolling(30, min_periods=10).mean()
        rvol_30 = safe_div(vol, vol_mean_30)

        # Days since local low within trailing 20 closes
        roll_min = close.rolling(20, min_periods=5).min()
        is_low = close <= roll_min + 1e-12
        days_since = np.full(len(close), np.nan)
        last_low_i = None
        for i in range(len(close)):
            if bool(is_low.iloc[i]) and pd.notna(roll_min.iloc[i]):
                last_low_i = i
            if last_low_i is not None:
                days_since[i] = float(i - last_low_i)

        high_60 = close.rolling(60, min_periods=20).max()
        dd60 = safe_div(close, high_60) - 1.0

        parts.append(
            pd.DataFrame(
                {
                    "timestamp": g["timestamp"],
                    "asset_id": asset_id,
                    "close": close,
                    "ema_20": ema20,
                    "ema_50": ema50,
                    "dist_ema_20": dist20,
                    "dist_ema_50": dist50,
                    "macd_line": macd_line,
                    "macd_signal": macd_signal,
                    "macd_hist": macd_hist,
                    "rsi_14": rsi,
                    "bb_pct_b": bb_pct_b,
                    "bb_bandwidth": bb_bw,
                    "rvol_30": rvol_30,
                    "days_since_local_low": days_since,
                    "drawdown_60d": dd60,
                }
            )
        )

    if not parts:
        return pd.DataFrame(
            columns=["timestamp", "asset_id", "close", *INDICATOR_COLS]
        )
    return pd.concat(parts, ignore_index=True)


def attach_cmram_features(
    signal_rows: pd.DataFrame,
    features: pd.DataFrame,
    *,
    model: str = "C",
) -> pd.DataFrame:
    """Left-join CMRAM feature scores onto rows keyed by signal_date / asset_id.

    ``signal_rows`` must have signal_date (or timestamp) and asset_id.
    """
    rows = signal_rows.copy()
    date_col = "signal_date" if "signal_date" in rows.columns else "timestamp"
    rows["_join_ts"] = pd.to_datetime(rows[date_col]).dt.tz_localize(None).dt.normalize()
    rows["asset_id"] = rows["asset_id"].astype(str)

    feat = features.copy()
    feat = feat.loc[feat["model"].astype(str) == str(model)].copy()
    feat["_join_ts"] = (
        pd.to_datetime(feat["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    feat["asset_id"] = feat["asset_id"].astype(str)
    keep = [
        "_join_ts",
        "asset_id",
        "MREI",
        "NSI",
        "Rotation_Gap",
        "score_C",
        "score_V",
        "score_N",
        "score_P",
    ]
    keep = [c for c in keep if c in feat.columns]
    feat = feat[keep].drop_duplicates(["_join_ts", "asset_id"], keep="first")

    out = rows.merge(feat, on=["_join_ts", "asset_id"], how="left")
    return out.drop(columns=["_join_ts"])


def snapshot_pre_spike(
    spikes: pd.DataFrame,
    indicators: pd.DataFrame,
    features: pd.DataFrame | None = None,
    *,
    model: str = "C",
) -> pd.DataFrame:
    """Attach T-1 indicator (+ optional CMRAM feature) snapshot to each spike."""
    if spikes is None or len(spikes) == 0:
        return spikes.copy() if spikes is not None else pd.DataFrame()

    sp = spikes.copy()
    sp["signal_date"] = pd.to_datetime(sp["signal_date"])
    sp["asset_id"] = sp["asset_id"].astype(str)

    ind = indicators.copy()
    ind["timestamp"] = (
        pd.to_datetime(ind["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    ind["asset_id"] = ind["asset_id"].astype(str)
    ind_cols = ["timestamp", "asset_id"] + [
        c for c in INDICATOR_COLS if c in ind.columns
    ]
    ind = ind[ind_cols].rename(columns={"timestamp": "signal_date"})

    out = sp.merge(ind, on=["signal_date", "asset_id"], how="left")
    if features is not None and len(features) > 0:
        out = attach_cmram_features(out, features, model=model)
    return out
