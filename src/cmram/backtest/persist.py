"""Read/write backtest_results in DuckDB (+ optional parquet/summary)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from cmram.backtest.engine import (
    RESULT_COLUMNS,
    aggregate_backtest,
    run_backtest,
)
from cmram.config import load_thresholds_config
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.universe.membership import load_market_daily


def load_signals(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT signal_id, timestamp, asset_id, band,
               entry_price, market_cap_usd, volume_usd,
               MREI, NSI, Rotation_Gap,
               tau_m, tau_g, model, liquidity_pass
        FROM signals
        ORDER BY timestamp, band, model, tau_m, tau_g, asset_id
        """
    ).df()


def write_backtest_results(
    conn: duckdb.DuckDBPyConnection,
    results: pd.DataFrame,
    *,
    replace: bool = True,
) -> int:
    if replace:
        conn.execute("DELETE FROM backtest_results")
    if results is None or len(results) == 0:
        return 0
    df = results[list(RESULT_COLUMNS)].copy()
    conn.register("_bt_df", df)
    try:
        conn.execute(
            """
            INSERT INTO backtest_results
            SELECT signal_id, horizon_d, ret, mfe, mae, benchmark_id, excess_ret
            FROM _bt_df
            """
        )
    finally:
        conn.unregister("_bt_df")
    return len(df)


def write_results_parquet(results: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(path, index=False)
    return path


def build_and_write_backtest(
    duckdb_path: str | Path,
    *,
    thresholds_config: dict[str, Any] | None = None,
    random_seed: int = 42,
    results_parquet: str | Path | None = None,
    aggregate_parquet: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load signals + market, run backtest, persist results + aggregate."""
    path = Path(duckdb_path)
    thr = thresholds_config or load_thresholds_config()
    horizons = list(thr.get("horizons_days") or [1, 3, 7, 14, 21, 30])

    conn = init_db(path)
    try:
        signals = load_signals(conn)
        market = load_market_daily(conn)
        membership = load_universe_membership(conn)
        results = run_backtest(
            signals,
            market,
            horizons_days=horizons,
            random_seed=random_seed,
            universe_membership=membership,
        )
        write_backtest_results(conn, results, replace=True)
        agg = aggregate_backtest(signals, results, benchmark_id="random")
        if results_parquet is not None:
            write_results_parquet(results, results_parquet)
        if aggregate_parquet is not None and len(agg):
            write_results_parquet(agg, aggregate_parquet)
        return results, agg
    finally:
        conn.close()
