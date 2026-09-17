"""Detect +50% forward-return spike events from daily closes.

Research helper only — not a trading system. Spike = forward max return
within a window (primary 7d, secondary 14d) >= threshold (default +50%).
Overlapping candidates per asset are deduplicated (keep earliest / non-overlap).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cmram.features.prep import prepare_market

DEFAULT_SPIKE_THRESHOLD = 0.50
PRIMARY_WINDOW = 7
SECONDARY_WINDOW = 14


def forward_max_return(
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Max close over the next ``window`` bars / close_t - 1 (excludes today).

    Uses only future closes t+1..t+window. Rows with no future bars are NaN;
    partial windows use whatever future bars exist.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    c = close.astype("float64").to_numpy()
    n = len(c)
    fut_max = np.full(n, np.nan, dtype="float64")
    for i in range(n):
        j1 = i + 1
        j2 = min(n, i + 1 + window)
        if j1 >= j2:
            continue
        fut_max[i] = np.nanmax(c[j1:j2])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = fut_max / close.astype("float64").to_numpy() - 1.0
    out = np.where(np.isfinite(out), out, np.nan)
    return pd.Series(out, index=close.index, dtype="float64")


def forward_max_return_by_asset(
    market: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute per-row forward max return within ``window`` days per asset.

    Returns columns: timestamp, asset_id, close, fwd_max_ret_{window}d
    """
    df = prepare_market(market)
    col = f"fwd_max_ret_{window}d"
    parts: list[pd.DataFrame] = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        out = g[["timestamp", "asset_id", "close"]].copy()
        out[col] = forward_max_return(g["close"], window).to_numpy()
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "asset_id", "close", col])
    return pd.concat(parts, ignore_index=True)


def dedupe_non_overlapping(
    spike_dates: list[pd.Timestamp],
    *,
    window: int,
) -> list[pd.Timestamp]:
    """Keep earliest spike; skip candidates until previous start + window days."""
    if not spike_dates:
        return []
    dates = sorted(pd.Timestamp(d).normalize() for d in spike_dates)
    kept: list[pd.Timestamp] = []
    block_until: pd.Timestamp | None = None
    for d in dates:
        if block_until is not None and d <= block_until:
            continue
        kept.append(d)
        block_until = d + pd.Timedelta(days=int(window))
    return kept


# Back-compat alias used internally
_dedupe_non_overlapping = dedupe_non_overlapping


def detect_spike_candidates(
    market: pd.DataFrame,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    window: int = PRIMARY_WINDOW,
) -> pd.DataFrame:
    """All (asset, date) rows where forward max return within window >= threshold.

    Does not dedupe. Adds ``is_spike_raw``.
    """
    fwd = forward_max_return_by_asset(market, window)
    col = f"fwd_max_ret_{window}d"
    fwd = fwd.copy()
    fwd["is_spike_raw"] = fwd[col] >= float(threshold)
    return fwd


def detect_spikes(
    market: pd.DataFrame,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    window: int = PRIMARY_WINDOW,
    secondary_window: int = SECONDARY_WINDOW,
    eligible: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect non-overlapping spike events for the eligible universe.

    Parameters
    ----------
    market :
        Daily OHLCV with timestamp, asset_id, close.
    threshold :
        Minimum forward max return (e.g. 0.50 = +50%).
    window :
        Primary forward window in days (default 7).
    secondary_window :
        Also attach fwd max ret over this window on kept events (default 14).
    eligible :
        Optional membership with timestamp, asset_id [, band]. If given, only
        spike starts on days the asset is in the universe are kept.

    Returns
    -------
    One row per deduplicated spike:
      asset_id, spike_date (T), signal_date (T-1), close_t,
      fwd_max_ret_{window}d, fwd_max_ret_{secondary}d, band, threshold, window
    """
    df = prepare_market(market)
    primary = forward_max_return_by_asset(df, window)
    secondary = forward_max_return_by_asset(df, secondary_window)
    pcol = f"fwd_max_ret_{window}d"
    scol = f"fwd_max_ret_{secondary_window}d"

    merged = primary.merge(
        secondary[["timestamp", "asset_id", scol]],
        on=["timestamp", "asset_id"],
        how="left",
    )

    if eligible is not None and len(eligible) > 0:
        cols = ["timestamp", "asset_id"]
        if "band" in eligible.columns:
            cols = ["timestamp", "asset_id", "band"]
        elig = eligible[cols].copy()
        elig["timestamp"] = (
            pd.to_datetime(elig["timestamp"]).dt.tz_localize(None).dt.normalize()
        )
        elig["asset_id"] = elig["asset_id"].astype(str)
        elig = elig.drop_duplicates(["timestamp", "asset_id"], keep="first")
        merged = merged.merge(elig, on=["timestamp", "asset_id"], how="inner")
    else:
        merged["band"] = None

    raw = merged.loc[merged[pcol] >= float(threshold)].copy()
    empty_cols = [
        "asset_id",
        "spike_date",
        "signal_date",
        "close_t",
        pcol,
        scol,
        "band",
        "threshold",
        "window",
    ]
    if raw.empty:
        return pd.DataFrame(columns=empty_cols)

    # Precompute T-1 map per asset from full market history
    prev_map: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
    for asset_id, g in df.groupby("asset_id", sort=False):
        ts_list = sorted(pd.Timestamp(t).normalize() for t in g["timestamp"].tolist())
        for i, t in enumerate(ts_list):
            if i > 0:
                prev_map[(str(asset_id), t)] = ts_list[i - 1]

    events: list[dict[str, Any]] = []
    for asset_id, g in raw.groupby("asset_id", sort=False):
        dates = [pd.Timestamp(t).normalize() for t in g["timestamp"].tolist()]
        kept = set(dedupe_non_overlapping(dates, window=window))
        g2 = g.copy()
        g2["_ts"] = pd.to_datetime(g2["timestamp"]).dt.normalize()
        g2 = g2.loc[g2["_ts"].isin(kept)]
        for _, row in g2.iterrows():
            spike_date = pd.Timestamp(row["timestamp"]).normalize()
            signal_date = prev_map.get((str(asset_id), spike_date), pd.NaT)
            events.append(
                {
                    "asset_id": str(asset_id),
                    "spike_date": spike_date,
                    "signal_date": signal_date,
                    "close_t": float(row["close"]),
                    pcol: float(row[pcol]) if pd.notna(row[pcol]) else np.nan,
                    scol: float(row[scol]) if pd.notna(row[scol]) else np.nan,
                    "band": row["band"] if "band" in row.index else None,
                    "threshold": float(threshold),
                    "window": int(window),
                }
            )

    out = pd.DataFrame(events)
    if out.empty:
        return pd.DataFrame(columns=empty_cols)
    return out.sort_values(["spike_date", "asset_id"], kind="mergesort").reset_index(
        drop=True
    )


def split_spikes_by_cut(
    spikes: pd.DataFrame,
    cut_date: Any,
    *,
    date_col: str = "spike_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split spike events into discover (< cut) and validate (>= cut)."""
    if spikes is None or len(spikes) == 0:
        empty = spikes.iloc[0:0].copy() if spikes is not None else pd.DataFrame()
        return empty, empty
    cut = pd.Timestamp(cut_date).normalize()
    ts = pd.to_datetime(spikes[date_col]).dt.normalize()
    discover = spikes.loc[ts < cut].copy()
    validate = spikes.loc[ts >= cut].copy()
    return discover, validate
