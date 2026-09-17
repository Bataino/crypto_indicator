#!/usr/bin/env python3
"""CLI: spike backward integration (+50% in 7d). Research only — no alpha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.backtest.spike_backward import (
    run_spike_backward,
    save_spike_outputs,
    write_spike_backward_report,
)
from cmram.config import DATA_DIR, get_duckdb_path
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.signals.persist import load_features_daily
from cmram.universe.membership import load_market_daily


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Find +50%/7d spikes, inspect pre-spike indicators, propose simple "
            "rules on early data, validate on later data. NOT alpha."
        )
    )
    p.add_argument("--db", type=str, default=None, help="DuckDB path")
    p.add_argument(
        "--report",
        type=str,
        default=str(ROOT / "docs/reports/spike_backward_abdul.md"),
        help="Markdown report path",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DATA_DIR / "spikes"),
        help="Parquet output directory",
    )
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--secondary-window", type=int, default=14)
    p.add_argument("--explore-frac", type=float, default=0.70)
    p.add_argument("--feature-model", type=str, default="C")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    db_path = Path(args.db) if args.db else get_duckdb_path()
    print("CMRAM run_spike_backward")
    print("  intent: backward spike patterns + later check — NOT alpha / not a bot")
    print(f"  duckdb: {db_path}")
    print(f"  spike: >= +{100 * args.threshold:.0f}% within {args.window}d "
          f"(also {args.secondary_window}d)")
    print(f"  explore_frac: {args.explore_frac}")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    conn = init_db(db_path)
    try:
        market = load_market_daily(conn)
        membership = load_universe_membership(conn)
        try:
            features = load_features_daily(conn)
        except Exception:
            features = None
    finally:
        conn.close()

    if len(market) == 0 or len(membership) == 0:
        print("ERROR: empty market or membership", file=sys.stderr)
        return 1

    result = run_spike_backward(
        market,
        membership,
        features,
        threshold=args.threshold,
        window=args.window,
        secondary_window=args.secondary_window,
        explore_frac=args.explore_frac,
        feature_model=args.feature_model,
    )

    out_dir = Path(args.out_dir)
    paths = save_spike_outputs(result, out_dir)
    report = write_spike_backward_report(result, args.report)

    print(f"  spikes: {result.summary.get('n_spikes_total')}")
    print(f"  discover/validate: {result.summary.get('n_spikes_discover')}/"
          f"{result.summary.get('n_spikes_validate')}")
    print(f"  cut_date: {result.summary.get('cut_date')}")
    print(f"  base_rate_validate: {result.base_rate_validate}")
    print(f"  any_rule_beats_base: {result.summary.get('any_rule_beats_base')}")
    for k, p in paths.items():
        print(f"  wrote {k}: {p}")
    print(f"  report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
