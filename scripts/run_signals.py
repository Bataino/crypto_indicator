#!/usr/bin/env python3
"""CLI: signal generation over τ_m / τ_g grids (research only)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.config import (
    get_duckdb_path,
    get_signals_dir,
    load_thresholds_config,
    load_universe_config,
)
from cmram.signals.persist import build_and_write_signals


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMRAM signals: liquidity_pass AND MREI>τ_m AND Gap>τ_g."
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB path (default data/cmram.duckdb or CMRAM_DUCKDB_PATH).",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing data/signals/signals.parquet.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def _report(signals) -> None:

    print(f"  rows: {0 if signals is None else len(signals)}")
    if signals is None or len(signals) == 0:
        print("  note: no signals — check features + liquidity + τ grids")
        return
    attrs = getattr(signals, "attrs", {}) or {}
    print(f"  entry_convention: {attrs.get('entry_convention')}")
    print(f"  entry_mode: {attrs.get('entry_mode')}")
    print(f"  chart_close_only: {attrs.get('chart_close_only')}")
    print(f"  skipped_no_entry: {attrs.get('n_skipped_no_entry')}")
    if attrs.get("chart_close_only"):
        print(
            "  limitation: O=H=L=C chart ingest → next-bar CLOSE used as open proxy"
        )
    g = (
        signals.groupby(["band", "model", "tau_m", "tau_g"], sort=True)
        .agg(n=("signal_id", "size"), assets=("asset_id", "nunique"))
        .reset_index()
    )
    print("  counts by band × model × τ_m × τ_g:")
    for r in g.itertuples(index=False):
        print(
            f"    {r.band} model={r.model} τ_m={r.tau_m:g} τ_g={r.tau_g:g}: "
            f"n={int(r.n)} assets={int(r.assets)}"
        )
    n_assets = int(signals["asset_id"].nunique())
    if n_assets < 50:
        print(
            f"  ⚠ HARD SMALL SAMPLE: n_assets={n_assets} (still tiny vs prior n=3) — "
            "signals are exploratory only; NOT conclusive; do not claim alpha."
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    universe = load_universe_config()
    thresholds = load_thresholds_config()
    db_path = Path(args.db) if args.db else get_duckdb_path()
    parquet = None if args.no_parquet else get_signals_dir() / "signals.parquet"

    print("CMRAM run_signals")
    print(f"  tau_m grid: {thresholds.get('tau_m')} (calibration TBD)")
    print(f"  tau_g grid: {thresholds.get('tau_g')} (calibration TBD)")
    print(f"  models: {thresholds.get('models')}")
    print(f"  entry_convention: {universe.get('entry_convention')}")
    print("  rule: liquidity_pass AND MREI > τ_m AND Gap > τ_g")
    print(f"  duckdb: {db_path}")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    signals = build_and_write_signals(
        db_path,
        thresholds_config=thresholds,
        universe_config=universe,
        parquet_path=parquet,
    )
    _report(signals)
    if parquet is not None:
        print(f"  parquet: {parquet}")
    print("  status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
