#!/usr/bin/env python3
"""CLI: fair-test locked draft formula (vol + not stretched). Research only — no alpha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.backtest.draft_formula_fair_test import (
    ALT_EXPLORE_FRAC,
    PRIMARY_EXPLORE_FRAC,
    run_draft_formula_fair_test,
    save_draft_formula_outputs,
    write_draft_formula_fair_test_report,
)
from cmram.config import DATA_DIR, get_duckdb_path
from cmram.db.schema import init_db
from cmram.features.persist import load_universe_membership
from cmram.signals.persist import load_features_daily
from cmram.universe.membership import load_market_daily


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fair-test locked draft: rvol>1.5 AND dist_ema20<=5%; "
            "soft filters WITH/WITHOUT. NOT alpha."
        )
    )
    p.add_argument("--db", type=str, default=None, help="DuckDB path")
    p.add_argument(
        "--report",
        type=str,
        default=str(ROOT / "docs/reports/draft_formula_fair_test_abdul.md"),
        help="Markdown report path",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DATA_DIR / "draft_formula"),
        help="Parquet output directory",
    )
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--secondary-window", type=int, default=14)
    p.add_argument("--explore-frac", type=float, default=PRIMARY_EXPLORE_FRAC)
    p.add_argument(
        "--alt-explore-frac",
        type=float,
        default=ALT_EXPLORE_FRAC,
        help="Secondary robustness cut; set <0 to skip",
    )
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
    print("CMRAM run_draft_formula_fair_test")
    print("  intent: locked draft fair-test — NOT alpha / not a bot")
    print(f"  duckdb: {db_path}")
    print(
        "  locked: rvol_30 > 1.5 AND dist_ema_20 <= 0.05; "
        f"later +50%/+100% within {args.window}d"
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

    alt = None if args.alt_explore_frac < 0 else float(args.alt_explore_frac)

    result = run_draft_formula_fair_test(
        market,
        membership,
        features,
        threshold=args.threshold,
        window=args.window,
        secondary_window=args.secondary_window,
        explore_frac=args.explore_frac,
        alt_explore_frac=alt,
        feature_model=args.feature_model,
    )

    out_dir = Path(args.out_dir)
    paths = save_draft_formula_outputs(result, out_dir)
    report = write_draft_formula_fair_test_report(result, args.report)

    s = result.summary
    print(f"  spikes: {s.get('n_spikes_total')}")
    print(f"  primary_cut: {s.get('primary_cut')}")
    print(f"  alt_cut: {s.get('alt_cut')}")
    print(
        f"  locked beats chance later? {s.get('locked_beats_chance_later')}"
    )
    print(
        f"  locked beats volume alone (+50%)? {s.get('locked_beats_volume_alone_50')}"
    )
    print(
        f"  locked beats not-stretched alone (+50%)? "
        f"{s.get('locked_beats_not_stretched_alone_50')}"
    )
    print(f"  soft filters help? {s.get('soft_filters_help')}")
    if len(result.primary.validation_50):
        print("  later +50%:")
        for _, r in result.primary.validation_50.iterrows():
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
