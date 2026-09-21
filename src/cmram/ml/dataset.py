"""Point-in-time labeled dataset for next-move prediction (Phase B).

Research only — no live trading, no model training in this module.

Labels (locked in docs/plans/AI_NEXT_MOVE_PLAN_V0.1.md):
  - up_7d: 1 if close[t+7] > close[t] (requires valid future bar)
  - spike_50_7d: 1 if max forward close return within 7d >= +50%
  - fwd_ret_7d: close[t+7] / close[t] - 1 (continuous)

Features v1 (social/Gap OFF):
  rvol_30, dist_ema_20, dist_ema_50, rsi_14,
  ret_1d, ret_3d, ret_7d, vol_14, log_mc

Universe: prefer Band C (C_1_100) membership days with liquidity_pass.
All features are backward-looking / PIT; labels use only future prices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from cmram.backtest.spike_detect import forward_max_return
from cmram.db.schema import init_db, migrate_schema
from cmram.features.prep import prepare_market, safe_div
from cmram.universe.membership import load_market_daily

logger = logging.getLogger(__name__)

BAND_C = "C_1_100"
HORIZON_D = 7
SPIKE_THRESH = 0.50
EXPLORE_FRAC_DEFAULT = 0.70

FEATURE_COLS = (
    "rvol_30",
    "dist_ema_20",
    "dist_ema_50",
    "rsi_14",
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "vol_14",
    "log_mc",
)

# Primary features required for "usable" row counts in smoke stats.
PRIMARY_FEATURES = (
    "rvol_30",
    "dist_ema_20",
    "rsi_14",
    "ret_1d",
    "ret_7d",
    "vol_14",
    "log_mc",
)

LABEL_COLS = ("up_7d", "spike_50_7d", "fwd_ret_7d")

SAMPLE_COLS = (
    "timestamp",
    "asset_id",
    *FEATURE_COLS,
    *LABEL_COLS,
    "band",
    "liquidity_pass",
)

AI_SAMPLES_DDL = """
CREATE TABLE IF NOT EXISTS ai_samples_daily (
    timestamp DATE,
    asset_id TEXT,
    rvol_30 DOUBLE,
    dist_ema_20 DOUBLE,
    dist_ema_50 DOUBLE,
    rsi_14 DOUBLE,
    ret_1d DOUBLE,
    ret_3d DOUBLE,
    ret_7d DOUBLE,
    vol_14 DOUBLE,
    log_mc DOUBLE,
    up_7d INTEGER,
    spike_50_7d INTEGER,
    fwd_ret_7d DOUBLE,
    band TEXT,
    liquidity_pass BOOLEAN,
    PRIMARY KEY (timestamp, asset_id, band)
);
"""


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


def compute_ai_features_labels(market: pd.DataFrame) -> pd.DataFrame:
    """Per (timestamp, asset_id): v1 features + 7d labels. PIT features only.

    Processes one asset at a time to keep peak RAM modest.
    """
    df = prepare_market(market)
    parts: list[pd.DataFrame] = []
    n_assets = df["asset_id"].nunique()
    for i, (asset_id, g) in enumerate(df.groupby("asset_id", sort=False), start=1):
        if i == 1 or i % 100 == 0 or i == n_assets:
            logger.info("features+labels asset %s/%s (%s)", i, n_assets, asset_id)
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        close = g["close"].astype("float64")
        vol = g["volume_usd"].astype("float64")
        mc = g["market_cap_usd"].astype("float64")

        # --- features (backward / inclusive of t) ---
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        dist20 = safe_div(close - ema20, ema20)
        dist50 = safe_div(close - ema50, ema50)
        rsi = _rsi(close, 14)

        vol_mean_30 = vol.rolling(30, min_periods=10).mean()
        rvol_30 = safe_div(vol, vol_mean_30)

        # simple returns (PIT)
        ret_1d = safe_div(close, close.shift(1)) - 1.0
        ret_3d = safe_div(close, close.shift(3)) - 1.0
        ret_7d = safe_div(close, close.shift(7)) - 1.0

        # realized vol: std of log returns over 14 bars (min 7)
        lr = np.log(safe_div(close, close.shift(1)))
        vol_14 = lr.rolling(14, min_periods=7).std(ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_mc = np.log(mc.to_numpy(dtype="float64"))
        log_mc = pd.Series(log_mc, index=g.index, dtype="float64")
        log_mc = log_mc.where(mc > 0)

        # --- labels (future only) ---
        close_fwd = close.shift(-HORIZON_D)
        valid_fwd = close_fwd.notna() & close.notna() & (close > 0)
        fwd_ret = safe_div(close_fwd, close) - 1.0
        up_7d = pd.Series(np.nan, index=g.index, dtype="float64")
        up_7d = up_7d.mask(valid_fwd, (close_fwd > close).astype("float64"))

        # Require full 7 future bars for spike label (same horizon as primary).
        fwd_max = forward_max_return(close, HORIZON_D)
        # forward_max_return allows partial windows; zero out when <7 future bars.
        n = len(close)
        has_full = pd.Series(
            [i + HORIZON_D < n for i in range(n)], index=g.index, dtype=bool
        )
        spike = pd.Series(np.nan, index=g.index, dtype="float64")
        spike = spike.mask(has_full, (fwd_max >= SPIKE_THRESH).astype("float64"))

        parts.append(
            pd.DataFrame(
                {
                    "timestamp": g["timestamp"],
                    "asset_id": asset_id,
                    "rvol_30": rvol_30,
                    "dist_ema_20": dist20,
                    "dist_ema_50": dist50,
                    "rsi_14": rsi,
                    "ret_1d": ret_1d,
                    "ret_3d": ret_3d,
                    "ret_7d": ret_7d,
                    "vol_14": vol_14,
                    "log_mc": log_mc,
                    "up_7d": up_7d,
                    "spike_50_7d": spike,
                    "fwd_ret_7d": fwd_ret.where(valid_fwd),
                }
            )
        )

    if not parts:
        cols = ["timestamp", "asset_id", *FEATURE_COLS, *LABEL_COLS]
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def load_band_c_membership(
    conn: duckdb.DuckDBPyConnection,
    *,
    band: str = BAND_C,
    require_liquidity_pass: bool = True,
) -> pd.DataFrame:
    """Load Band C membership rows (optionally liquidity_pass only)."""
    sql = """
        SELECT timestamp, asset_id, band, liquidity_pass, market_cap_usd
        FROM universe_membership
        WHERE band = ?
    """
    params: list[Any] = [band]
    if require_liquidity_pass:
        sql += " AND liquidity_pass = TRUE"
    sql += " ORDER BY timestamp, asset_id"
    return conn.execute(sql, params).df()


def build_ai_samples(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    require_labels: bool = True,
    require_primary_features: bool = True,
) -> pd.DataFrame:
    """Join PIT features/labels onto membership days; drop unusable rows.

    Parameters
    ----------
    market:
        Full ``market_daily`` (all assets) so rolling history is correct.
    membership:
        Band membership rows (timestamp, asset_id, band, liquidity_pass).
    require_labels:
        Drop rows missing ``up_7d`` / ``fwd_ret_7d`` (no valid t+7 bar).
    require_primary_features:
        Drop rows with any null among PRIMARY_FEATURES.
    """
    feat = compute_ai_features_labels(market)
    mem = membership.copy()
    mem["timestamp"] = pd.to_datetime(mem["timestamp"]).dt.tz_localize(None).dt.normalize()
    mem["asset_id"] = mem["asset_id"].astype(str)
    if "band" not in mem.columns:
        mem["band"] = BAND_C
    if "liquidity_pass" not in mem.columns:
        mem["liquidity_pass"] = True

    feat["timestamp"] = pd.to_datetime(feat["timestamp"]).dt.tz_localize(None).dt.normalize()
    feat["asset_id"] = feat["asset_id"].astype(str)

    keep_mem = ["timestamp", "asset_id", "band", "liquidity_pass"]
    merged = mem[keep_mem].merge(
        feat,
        on=["timestamp", "asset_id"],
        how="inner",
        sort=False,
    )

    if require_labels:
        merged = merged.loc[merged["up_7d"].notna() & merged["fwd_ret_7d"].notna()]
    if require_primary_features:
        mask = pd.Series(True, index=merged.index)
        for col in PRIMARY_FEATURES:
            mask &= merged[col].notna()
        merged = merged.loc[mask]

    # Cast labels to int where present
    for col in ("up_7d", "spike_50_7d"):
        if col in merged.columns:
            merged[col] = merged[col].astype("Int64")

    out = merged[[c for c in SAMPLE_COLS if c in merged.columns]].copy()
    out = out.sort_values(["timestamp", "asset_id"], kind="mergesort").reset_index(
        drop=True
    )
    return out


def time_split_cut_date(
    timestamps: pd.Series | pd.DatetimeIndex,
    *,
    explore_frac: float = EXPLORE_FRAC_DEFAULT,
) -> pd.Timestamp:
    """Cut date so ~explore_frac of unique calendar days are explore (earlier).

    Rows with timestamp <= cut_date are explore; later dates are holdout.
    Prints/returns the inclusive explore end date.
    """
    if not 0.0 < float(explore_frac) < 1.0:
        raise ValueError(f"explore_frac must be in (0,1), got {explore_frac}")
    days = (
        pd.to_datetime(pd.Series(timestamps))
        .dt.tz_localize(None)
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if len(days) == 0:
        raise ValueError("no timestamps for time split")
    # Inclusive cut: first ceil(n * frac) unique days → explore
    n_explore = max(1, int(np.ceil(len(days) * float(explore_frac))))
    n_explore = min(n_explore, len(days))
    cut = pd.Timestamp(days.iloc[n_explore - 1])
    return cut


def apply_time_split(
    samples: pd.DataFrame,
    *,
    explore_frac: float = EXPLORE_FRAC_DEFAULT,
    cut_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split samples into explore (earlier) / later by date."""
    cut = cut_date if cut_date is not None else time_split_cut_date(
        samples["timestamp"], explore_frac=explore_frac
    )
    ts = pd.to_datetime(samples["timestamp"]).dt.tz_localize(None).dt.normalize()
    explore = samples.loc[ts <= cut].copy()
    later = samples.loc[ts > cut].copy()
    return explore, later, pd.Timestamp(cut)


def ensure_ai_samples_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create ``ai_samples_daily`` if missing (idempotent)."""
    migrate_schema(conn)
    conn.execute(AI_SAMPLES_DDL)


def write_ai_samples_duckdb(
    conn: duckdb.DuckDBPyConnection,
    samples: pd.DataFrame,
    *,
    replace: bool = True,
) -> int:
    """Persist samples into DuckDB table ``ai_samples_daily``."""
    ensure_ai_samples_table(conn)
    if replace:
        conn.execute("DELETE FROM ai_samples_daily")
    if samples is None or len(samples) == 0:
        return 0
    df = samples.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    for col in ("up_7d", "spike_50_7d"):
        if col in df.columns:
            df[col] = df[col].astype("float64")  # duckdb accepts; Int64 → float ok
    cols = [c for c in SAMPLE_COLS if c in df.columns]
    df = df[cols]
    conn.register("_ai_samples_df", df)
    try:
        col_list = ", ".join(cols)
        conn.execute(
            f"INSERT INTO ai_samples_daily ({col_list}) SELECT {col_list} FROM _ai_samples_df"
        )
    finally:
        conn.unregister("_ai_samples_df")
    return len(df)


def write_ai_samples_parquet(samples: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(path, index=False)
    return path


def write_ai_samples(
    duckdb_path: str | Path,
    samples: pd.DataFrame,
    *,
    parquet_path: str | Path | None = None,
    replace: bool = True,
) -> dict[str, Any]:
    """Write parquet (optional) + DuckDB table; return paths/counts."""
    db_path = Path(duckdb_path)
    conn = init_db(db_path)
    try:
        n = write_ai_samples_duckdb(conn, samples, replace=replace)
    finally:
        conn.close()
    pq = None
    if parquet_path is not None:
        pq = write_ai_samples_parquet(samples, parquet_path)
    return {"rows": n, "duckdb": str(db_path), "parquet": str(pq) if pq else None}


def dataset_smoke_stats(
    samples: pd.DataFrame,
    *,
    explore_frac: float = EXPLORE_FRAC_DEFAULT,
) -> dict[str, Any]:
    """Plain-English-friendly smoke stats for Phase B report."""
    if samples is None or len(samples) == 0:
        return {
            "rows": 0,
            "assets": 0,
            "date_min": None,
            "date_max": None,
            "cut_date": None,
            "explore_rows": 0,
            "later_rows": 0,
            "up_7d_rate": None,
            "spike_50_7d_rate": None,
            "null_rates": {},
            "primary_non_null_rows": 0,
        }

    ts = pd.to_datetime(samples["timestamp"]).dt.tz_localize(None).dt.normalize()
    explore, later, cut = apply_time_split(samples, explore_frac=explore_frac)

    null_rates = {}
    for col in list(FEATURE_COLS) + list(LABEL_COLS):
        if col in samples.columns:
            null_rates[col] = float(samples[col].isna().mean())

    primary_ok = pd.Series(True, index=samples.index)
    for col in PRIMARY_FEATURES:
        if col in samples.columns:
            primary_ok &= samples[col].notna()

    up = samples["up_7d"].dropna()
    spike = samples["spike_50_7d"].dropna()

    return {
        "rows": int(len(samples)),
        "assets": int(samples["asset_id"].nunique()),
        "date_min": str(ts.min().date()),
        "date_max": str(ts.max().date()),
        "cut_date": str(cut.date()),
        "explore_frac": float(explore_frac),
        "explore_rows": int(len(explore)),
        "later_rows": int(len(later)),
        "up_7d_rate": float(up.mean()) if len(up) else None,
        "spike_50_7d_rate": float(spike.mean()) if len(spike) else None,
        "null_rates": null_rates,
        "primary_non_null_rows": int(primary_ok.sum()),
        "band": sorted(samples["band"].astype(str).unique().tolist())
        if "band" in samples.columns
        else [],
    }


def build_ai_dataset_from_db(
    duckdb_path: str | Path,
    *,
    band: str = BAND_C,
    require_liquidity_pass: bool = True,
    parquet_path: str | Path | None = None,
    explore_frac: float = EXPLORE_FRAC_DEFAULT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end: load market + Band C membership → samples → write → stats."""
    path = Path(duckdb_path)
    conn = init_db(path)
    try:
        market = load_market_daily(conn)
        membership = load_band_c_membership(
            conn, band=band, require_liquidity_pass=require_liquidity_pass
        )
        logger.info(
            "loaded market_daily=%s rows, membership=%s rows (band=%s, liq=%s)",
            len(market),
            len(membership),
            band,
            require_liquidity_pass,
        )
    finally:
        conn.close()

    samples = build_ai_samples(market, membership)
    # free large intermediates before write
    del market, membership

    write_meta = write_ai_samples(
        path,
        samples,
        parquet_path=parquet_path,
        replace=True,
    )
    stats = dataset_smoke_stats(samples, explore_frac=explore_frac)
    stats["write"] = write_meta
    return samples, stats
