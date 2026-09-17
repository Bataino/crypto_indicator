#!/usr/bin/env python3
"""CLI: forward-return backtest (research only; no orders)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.backtest.persist import build_and_write_backtest, load_signals
from cmram.config import get_backtests_dir, get_duckdb_path, load_thresholds_config
from cmram.db.schema import init_db
from cmram.report.backtest_report import write_backtest_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMRAM backtest: forward ret / MFE / MAE vs benchmarks."
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB path (default data/cmram.duckdb or CMRAM_DUCKDB_PATH).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random same-band benchmark (default 42).",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing parquet under data/backtests/.",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report under data/backtests/.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    thresholds = load_thresholds_config()
    db_path = Path(args.db) if args.db else get_duckdb_path()
    out_dir = get_backtests_dir()
    results_pq = None if args.no_parquet else out_dir / "backtest_results.parquet"
    agg_pq = None if args.no_parquet else out_dir / "backtest_aggregate.parquet"

    print("CMRAM run_backtest")
    print(f"  horizons_days: {thresholds.get('horizons_days')}")
    print("  metrics: ret, MFE, MAE, excess vs random/momentum/volume/market")
    print("  non-goals: live trading, orders, wallets, ML")
    print(f"  duckdb: {db_path}")
    print(f"  random_seed: {args.seed}")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    results, agg = build_and_write_backtest(
        db_path,
        thresholds_config=thresholds,
        random_seed=args.seed,
        results_parquet=results_pq,
        aggregate_parquet=agg_pq,
    )

    conn = init_db(db_path)
    try:
        signals = load_signals(conn)
    finally:
        conn.close()

    n_assets = int(getattr(results, "attrs", {}).get("n_assets") or 0)
    if n_assets == 0 and len(signals):
        n_assets = int(signals["asset_id"].nunique())
    chart_only = bool(getattr(results, "attrs", {}).get("chart_close_only"))
    small = bool(getattr(results, "attrs", {}).get("small_sample", n_assets < 50))

    print(f"  signals: {len(signals)}")
    print(f"  backtest_results rows: {len(results)}")
    print(f"  aggregate rows: {len(agg)}")
    print(f"  n_assets: {n_assets}")
    print(f"  chart_close_only: {chart_only}")
    if small:
        print(
            "  ⚠ SMALL SAMPLE / NOT CONCLUSIVE — do not claim alpha; "
            "metrics are exploratory only."
        )

    if not args.no_report:
        report_path = out_dir / "backtest_report.md"
        write_backtest_report(
            report_path,
            results=results,
            signals=signals,
            aggregate=agg,
            meta={
                "n_assets": n_assets,
                "small_sample": small,
                "chart_close_only": chart_only,
                "entry_mode": (
                    "next_day_close_chart_proxy"
                    if chart_only
                    else "next_day_open"
                ),
                "aggregate_benchmark": "random",
            },
        )
        print(f"  report: {report_path}")

    if results_pq is not None:
        print(f"  parquet: {results_pq}")
    if agg_pq is not None:
        print(f"  aggregate_parquet: {agg_pq}")
    print("  status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
