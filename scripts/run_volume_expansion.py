#!/usr/bin/env python3
"""CLI: volume expansion core + second-condition candidates. Research only — no alpha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.backtest.volume_expansion import (
    run_volume_expansion_study,
    save_volume_expansion_outputs,
    write_volume_expansion_report,
)
from cmram.config import DATA_DIR, get_duckdb_path
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.signals.persist import load_features_daily
from cmram.universe.membership import load_market_daily


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Test rvol>1.5 alone and with one second condition each; "
            "later +50%/+100% hit rates. NOT alpha."
        )
    )
    p.add_argument("--db", type=str, default=None, help="DuckDB path")
    p.add_argument(
        "--report",
        type=str,
        default=str(ROOT / "docs/reports/volume_expansion_plus_abdul.md"),
        help="Markdown report path",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DATA_DIR / "volume_expansion"),
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
    print("CMRAM run_volume_expansion")
    print("  intent: volume expansion + one second condition — NOT alpha / not a bot")
    print(f"  duckdb: {db_path}")
    print(
        f"  core: rvol_30 > 1.5; later check +50% and +100% within {args.window}d"
    )

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

    result = run_volume_expansion_study(
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
    paths = save_volume_expansion_outputs(result, out_dir)
    report = write_volume_expansion_report(result, args.report)

    print(f"  spikes: {result.summary.get('n_spikes_total')}")
    print(
        f"  discover/validate: {result.summary.get('n_spikes_discover')}/"
        f"{result.summary.get('n_spikes_validate')}"
    )
    print(f"  cut_date: {result.summary.get('cut_date')}")
    print(
        f"  base_rate +50%/+100%: {result.base_rate_50:.4f} / {result.base_rate_100:.4f}"
    )
    print(f"  improvers +50%: {result.summary.get('improvers_50')}")
    print(f"  improvers +100%: {result.summary.get('improvers_100')}")
    hot = result.hot_vol_ret3d.get("all", {})
    print(
        f"  hot-vol T-1 ret_3d>0 share: {hot.get('share_ret_gt_0')} "
        f"(n={hot.get('n_hot_with_ret')})"
    )
    if len(result.validation_50):
        print("  later +50%:")
        for _, r in result.validation_50.iterrows():
            print(
                f"    - {r['setup_id']}: fires={int(r['n_fire'])} "
                f"hits={int(r['n_hit'])} lift={r['lift']:.2f}x "
                f"beats={r['beats_base']}"
            )
    for k, p in paths.items():
        print(f"  wrote {k}: {p}")
    print(f"  report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
