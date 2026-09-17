#!/usr/bin/env python3
"""CLI: time holdout evaluation (explore vs holdout). Research only — no alpha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.backtest.holdout import (
    DEFAULT_EXPLORE_FRAC,
    run_holdout_evaluation,
    write_holdout_report,
)
from cmram.backtest.persist import load_signals, write_results_parquet
from cmram.config import get_backtests_dir, get_duckdb_path
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.universe.membership import load_market_daily


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CMRAM time holdout: first 70% membership days = explore, "
            "last 30% = holdout. Report both. No alpha claims."
        )
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB path (default data/cmram.duckdb or CMRAM_DUCKDB_PATH).",
    )
    p.add_argument(
        "--explore-frac",
        type=float,
        default=DEFAULT_EXPLORE_FRAC,
        help=f"Fraction of calendar days for explore (default {DEFAULT_EXPLORE_FRAC}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random benchmark (default 42).",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing explore/holdout cell parquet summaries.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    db_path = Path(args.db) if args.db else get_duckdb_path()
    out_dir = get_backtests_dir()

    print("CMRAM run_holdout")
    print("  intent: explore (early) vs holdout (late) — NOT validated alpha")
    print(f"  explore_frac: {args.explore_frac}")
    print("  horizons_days: [7] (holdout peek metric; lean)")
    print("  benchmarks: [random]")
    print(f"  duckdb: {db_path}")
    print(f"  random_seed: {args.seed}")
    print("  non-goals: live trading, orders, wallets, ML, alpha claims")

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

    if len(membership) == 0:
        print("ERROR: universe_membership is empty", file=sys.stderr)
        return 1
    if len(signals) == 0:
        print("ERROR: signals table is empty — run signals first", file=sys.stderr)
        return 1

    # Lean by default: h=7 vs random (holdout peek metric). Full grid is slow.
    evaluation = run_holdout_evaluation(
        signals,
        market,
        membership,
        explore_frac=args.explore_frac,
        horizons_days=[7],
        benchmarks=["random"],
        random_seed=args.seed,
    )

    split = evaluation["split"]
    print(f"  membership_days: {split.get('n_days')}")
    print(f"  cut_date (first holdout day): {split.get('cut_date')}")
    print(
        f"  explore: {split.get('explore_start')} → {split.get('explore_end')} "
        f"({split.get('n_explore')} days, {evaluation['n_signals_explore']} signals)"
    )
    print(
        f"  holdout: {split.get('holdout_start')} → {split.get('holdout_end')} "
        f"({split.get('n_holdout')} days, {evaluation['n_signals_holdout']} signals)"
    )
    top = evaluation.get("explore_top")
    if top:
        print(
            f"  explore_top_peek: model={top['model']} "
            f"tau_m={top['tau_m']:g} tau_g={top['tau_g']:g} "
            f"median_excess={top['median_excess']:.4f} (NOT a winner claim)"
        )
    print(f"  support: {evaluation['support_statement'][:160]}...")
    print(
        "  ⚠ SMALL SAMPLE / NOT CONCLUSIVE — do not claim alpha; "
        "holdout is exploratory only."
    )

    report_path = out_dir / "holdout_report.md"
    write_holdout_report(report_path, evaluation)
    print(f"  report: {report_path}")

    if not args.no_parquet:
        he = evaluation.get("holdout_eval")
        ee = evaluation.get("explore_eval")
        eg = evaluation.get("explore_grid")
        if he is not None and len(he):
            write_results_parquet(he, out_dir / "holdout_cells.parquet")
            print(f"  holdout_cells_parquet: {out_dir / 'holdout_cells.parquet'}")
        if ee is not None and len(ee):
            write_results_parquet(ee, out_dir / "holdout_explore_cells.parquet")
        if eg is not None and len(eg):
            write_results_parquet(eg, out_dir / "holdout_explore_grid.parquet")

    print("  status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
