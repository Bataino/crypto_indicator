#!/usr/bin/env python3
"""CLI: build point-in-time universe_membership from DuckDB market_daily."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.config import (
    get_duckdb_path,
    load_eligibility_config,
    load_thresholds_config,
    load_universe_config,
)
from cmram.universe.membership import build_and_write_universe_membership


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMRAM universe membership (bands A/B + liquidity draft)."
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB path (default data/cmram.duckdb or CMRAM_DUCKDB_PATH).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return p.parse_args(argv)


def _report(membership) -> None:
    import pandas as pd

    elig_ex = dict(getattr(membership, "attrs", {}).get("eligibility_excluded") or {})
    if elig_ex:
        print(f"  eligibility_excluded: {len(elig_ex)} assets")
        for aid in sorted(elig_ex):
            print(f"    {aid}: {elig_ex[aid]}")

    if membership is None or len(membership) == 0:
        print("  rows: 0")
        return

    total = len(membership)
    print(f"  rows: {total}")
    by_band = membership.groupby("band", sort=True).agg(
        rows=("asset_id", "size"),
        assets=("asset_id", "nunique"),
        liq_pass=("liquidity_pass", "sum"),
    )
    for band, row in by_band.iterrows():
        n = int(row["rows"])
        n_pass = int(row["liq_pass"])
        pct = (100.0 * n_pass / n) if n else 0.0
        print(
            f"  band {band}: rows={n} assets={int(row['assets'])} "
            f"liquidity_pass={n_pass} ({pct:.1f}%)"
        )

    ts = pd.to_datetime(membership["timestamp"])
    print(f"  date_range: {ts.min().date()} → {ts.max().date()}")
    print(f"  membership_assets: {int(membership['asset_id'].nunique())}")
    excluded = membership.loc[~membership["liquidity_pass"], "reason_excluded"]
    if len(excluded):
        top = excluded.value_counts().head(5)
        print("  top reason_excluded:")
        for reason, cnt in top.items():
            print(f"    {reason}: {cnt}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    universe = load_universe_config()
    thresholds = load_thresholds_config()
    eligibility = load_eligibility_config()
    db_path = Path(args.db) if args.db else get_duckdb_path()

    print("CMRAM run_universe")
    print(f"  bands: {list(universe.get('bands', {}))}")
    print(f"  min_history_days: {universe.get('min_history_days')}")
    print(f"  v_min_usd candidates: {thresholds.get('v_min_usd')}")
    print(f"  draft V_min (first candidate): {(thresholds.get('v_min_usd') or [None])[0]}")
    amihud = thresholds.get("amihud") or {}
    print(
        f"  amihud lookback={amihud.get('lookback_days')} "
        f"illiq_max_percentile={amihud.get('illiq_max_percentile')}"
    )
    print(
        f"  eligibility: enabled={eligibility.get('enabled')} "
        f"deny_ids={len(eligibility.get('deny_ids') or [])} "
        f"deny_keywords={len(eligibility.get('deny_keywords') or [])}"
    )
    print(f"  duckdb: {db_path}")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    membership = build_and_write_universe_membership(
        db_path, universe, thresholds, eligibility_config=eligibility
    )
    _report(membership)
    print("  status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
