#!/usr/bin/env python3
"""CLI: cross-coin shared boolean setups before spikes. Research only — no alpha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.backtest.spike_shared_setups import (
    run_spike_shared_setups,
    save_shared_setups_outputs,
    write_shared_setups_report,
)
from cmram.config import DATA_DIR, get_duckdb_path
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.signals.persist import load_features_daily
from cmram.universe.membership import load_market_daily


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Find shared yes/no setups on earlier pre-spike days vs control, "
            "validate later for +50% and +100% within 7d. NOT alpha."
        )
    )
    p.add_argument("--db", type=str, default=None, help="DuckDB path")
    p.add_argument(
        "--report",
        type=str,
        default=str(ROOT / "docs/reports/spike_shared_setups_abdul.md"),
        help="Markdown report path (does not overwrite old spike_backward reports)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DATA_DIR / "spikes_shared"),
        help="Parquet output directory",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Spike threshold for discovery events (default +50%)",
    )
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--secondary-window", type=int, default=14)
    p.add_argument("--explore-frac", type=float, default=0.70)
    p.add_argument("--feature-model", type=str, default="C")
    p.add_argument("--lift-min", type=float, default=1.2)
    p.add_argument("--min-spike-count", type=int, default=5)
    p.add_argument("--top-setups", type=int, default=6)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    db_path = Path(args.db) if args.db else get_duckdb_path()
    print("CMRAM run_spike_shared_setups")
    print("  intent: shared boolean setups before spikes — NOT alpha / not a bot")
    print(f"  duckdb: {db_path}")
    print(
        f"  discover spikes: >= +{100 * args.threshold:.0f}% within {args.window}d; "
        f"later check +50% and +100%"
    )
    print(f"  lift_min: {args.lift_min}  min_spike_count: {args.min_spike_count}")

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

    result = run_spike_shared_setups(
        market,
        membership,
        features,
        threshold=args.threshold,
        window=args.window,
        secondary_window=args.secondary_window,
        explore_frac=args.explore_frac,
        feature_model=args.feature_model,
        lift_min=args.lift_min,
        min_spike_count=args.min_spike_count,
        top_setups=args.top_setups,
    )

    out_dir = Path(args.out_dir)
    paths = save_shared_setups_outputs(result, out_dir)
    report = write_shared_setups_report(result, args.report)

    print(f"  spikes: {result.summary.get('n_spikes_total')}")
    print(
        f"  discover/validate: {result.summary.get('n_spikes_discover')}/"
        f"{result.summary.get('n_spikes_validate')}"
    )
    print(f"  cut_date: {result.summary.get('cut_date')}")
    print(f"  selected setups: {result.summary.get('n_selected')}")
    for setup in result.selected:
        print(
            f"    - {setup.setup_id}: lift={setup.lift:.2f}x "
            f"spike_sup={setup.spike_support:.1%}"
        )
    print(f"  base_rate +50%/+100%: {result.base_rate_50:.4f} / {result.base_rate_100:.4f}")
    print(f"  any beats +50%: {result.summary.get('any_setup_beats_50')}")
    print(f"  any beats +100%: {result.summary.get('any_setup_beats_100')}")
    for k, p in paths.items():
        print(f"  wrote {k}: {p}")
    print(f"  report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
