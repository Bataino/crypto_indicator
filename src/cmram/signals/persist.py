"""Read/write signals in DuckDB (+ optional parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from cmram.config import load_thresholds_config, load_universe_config
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.signals.generate import SIGNAL_COLUMNS, generate_signals
from cmram.universe.membership import load_market_daily


def load_features_daily(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT timestamp, asset_id, band, model,
               score_C, score_V, score_N, score_P,
               MREI, nsi_social, nsi_price_ext, nsi_vol_exh, nsi_mom_dec,
               NSI, Rotation_Gap, n_available, small_sample, n_in_band,
               model_version
        FROM features_daily
        ORDER BY timestamp, band, model, asset_id
        """
    ).df()


def write_signals(
    conn: duckdb.DuckDBPyConnection,
    signals: pd.DataFrame,
    *,
    replace: bool = True,
) -> int:
    """Persist signal rows. ``replace=True`` clears the table first."""
    if replace:
        conn.execute("DELETE FROM signals")
    if signals is None or len(signals) == 0:
        return 0
    df = signals[list(SIGNAL_COLUMNS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    conn.register("_signals_df", df)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO signals
            SELECT
                signal_id, timestamp, asset_id, band,
                entry_price, market_cap_usd, volume_usd,
                MREI, NSI, Rotation_Gap,
                tau_m, tau_g, model, liquidity_pass
            FROM _signals_df
            """
        )
    finally:
        conn.unregister("_signals_df")
    return len(df)


def write_signals_parquet(signals: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(path, index=False)
    return path


def build_and_write_signals(
    duckdb_path: str | Path,
    *,
    thresholds_config: dict[str, Any] | None = None,
    universe_config: dict[str, Any] | None = None,
    parquet_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load features + membership + market, generate signals, persist."""
    path = Path(duckdb_path)
    thr = thresholds_config or load_thresholds_config()
    uni = universe_config or load_universe_config()
    entry = str(uni.get("entry_convention") or "next_day_open")

    conn = init_db(path)
    try:
        features = load_features_daily(conn)
        membership = load_universe_membership(conn)
        market = load_market_daily(conn)
        signals = generate_signals(
            features,
            membership,
            thr,
            market_daily=market,
            entry_convention=entry,
        )
        write_signals(conn, signals, replace=True)
        if parquet_path is not None:
            write_signals_parquet(signals, parquet_path)
        return signals
    finally:
        conn.close()
