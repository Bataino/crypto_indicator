"""Read/write features_daily in DuckDB (+ optional parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from cmram.db.schema import init_db, migrate_schema
from cmram.features.pipeline import FEATURE_COLUMNS, compute_features_daily
from cmram.ingest.social_persist import load_social_daily
from cmram.universe.membership import load_market_daily


def write_features_daily(
    conn: duckdb.DuckDBPyConnection,
    features: pd.DataFrame,
    *,
    replace: bool = True,
) -> int:
    """Write feature rows. ``replace=True`` rebuilds the table contents."""
    migrate_schema(conn)
    if replace:
        conn.execute("DELETE FROM features_daily")
    if features is None or len(features) == 0:
        return 0
    df = features[list(FEATURE_COLUMNS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    conn.register("_features_df", df)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO features_daily
            SELECT
                timestamp, asset_id, band, model,
                score_C, score_V, score_N, score_P,
                MREI,
                nsi_social, nsi_price_ext, nsi_vol_exh, nsi_mom_dec,
                NSI, Rotation_Gap,
                n_available, small_sample, n_in_band, model_version
            FROM _features_df
            """
        )
    finally:
        conn.unregister("_features_df")
    return len(df)


def load_universe_membership(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT timestamp, asset_id, band, market_cap_usd,
               history_days, liquidity_pass, amihud, reason_excluded
        FROM universe_membership
        ORDER BY timestamp, band, asset_id
        """
    ).df()


def write_features_parquet(features: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return path


def build_and_write_features(
    duckdb_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    parquet_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load market + membership + social, compute Models A/B/C, persist."""
    path = Path(duckdb_path)
    conn = init_db(path)
    try:
        market = load_market_daily(conn)
        membership = load_universe_membership(conn)
        try:
            social = load_social_daily(conn)
        except Exception:
            social = None
        features = compute_features_daily(
            market, membership, social_daily=social, config=config
        )
        write_features_daily(conn, features, replace=True)
        if parquet_path is not None:
            write_features_parquet(features, parquet_path)
        return features
    finally:
        conn.close()
