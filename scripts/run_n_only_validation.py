#!/usr/bin/env python3
"""CLI: Model E N-only validation (try to disprove). Research only — no alpha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.backtest.n_only_validation import (
    DEFAULT_RANDOM_SEEDS,
    PRIMARY_HORIZONS,
    run_n_only_validation,
    write_n_only_validation_report,
)
from cmram.backtest.persist import load_signals
from cmram.config import get_duckdb_path
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.universe.membership import load_market_daily


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate/disprove Model E (N-only) prereg 50/20 on bands A+B "
            "vs group_median/mean. No alpha claims."
        )
    )
    p.add_argument("--db", type=str, default=None, help="DuckDB path")
    p.add_argument(
        "--report",
        type=str,
        default=str(ROOT / "docs/reports/n_only_validation_abdul.md"),
        help="Markdown report path",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_RANDOM_SEEDS),
        help="Comma-separated random seeds for secondary random bench",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    db_path = Path(args.db) if args.db else get_duckdb_path()
    print("CMRAM run_n_only_validation")
    print("  intent: try to DISPROVE/validate N-only Model E sign — NOT alpha")
    print(f"  duckdb: {db_path}")
    print(f"  horizons: {PRIMARY_HORIZONS}")
    print("  primary benches: group_median, group_mean")
    print("  secondary: multi-seed random")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    print(f"  random_seeds: {seeds}")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    conn = init_db(db_path)
    try:
        signals = load_signals(conn)
        market = load_market_daily(conn)
        membership = load_universe_membership(conn)
    finally:
        conn.close()

    if len(signals) == 0 or len(membership) == 0:
        print("ERROR: empty signals or membership", file=sys.stderr)
        return 1

    evaluation = run_n_only_validation(
        signals,
        market,
        membership,
        random_seeds=seeds,
        horizons_days=list(PRIMARY_HORIZONS),
    )
    split = evaluation["split"]
    print(f"  cut_date: {split.get('cut_date')}")
    print(
        f"  holdout signals: {evaluation['n_signals_holdout']} "
        f"assets={evaluation['n_assets_holdout']}"
    )
    answers = evaluation.get("answers") or {}
    print(f"  overall: {str(answers.get('overall', ''))[:200]}")

    report_path = write_n_only_validation_report(args.report, evaluation)
    print(f"  report: {report_path}")
    print("  status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
