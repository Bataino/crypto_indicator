#!/usr/bin/env python3
"""CLI: Phase C LightGBM train + fair later test. Research only — no trading."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.config import DATA_DIR, get_duckdb_path
from cmram.ml.dataset import FEATURE_COLS
from cmram.ml.train import (
    DEFAULT_CUT_DATE,
    load_ai_samples,
    run_phase_c,
    save_result_json,
    write_phase_c_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CMRAM AI Phase C: train small LightGBM on explore only, "
            "fair-test once on later. NOT alpha / not a bot."
        )
    )
    p.add_argument(
        "--parquet",
        type=str,
        default=str(DATA_DIR / "ai" / "ai_samples_daily.parquet"),
        help="AI samples parquet (default data/ai/ai_samples_daily.parquet).",
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB fallback if parquet missing (default data/cmram.duckdb).",
    )
    p.add_argument(
        "--cut-date",
        type=str,
        default=DEFAULT_CUT_DATE,
        help=f"Explore inclusive cut date (default {DEFAULT_CUT_DATE}).",
    )
    p.add_argument(
        "--threshold-method",
        type=str,
        choices=("top_quintile", "f1"),
        default="top_quintile",
        help="How to pick high-prob threshold on explore only.",
    )
    p.add_argument(
        "--report",
        type=str,
        default=str(ROOT / "docs" / "reports" / "phase_c_lightgbm_fair_test.md"),
        help="Markdown report path.",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=str(DATA_DIR / "ai" / "phase_c_fair_test.json"),
        help="JSON metrics dump path.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parquet = Path(args.parquet)
    db = Path(args.db) if args.db else get_duckdb_path()

    print("CMRAM run_ai_train (Phase C)")
    print("  intent: research fair-test only — NOT alpha / not a bot")
    print(f"  parquet: {parquet}")
    print(f"  duckdb:  {db}")
    print(f"  cut:     explore <= {args.cut_date}; later > cut")
    print(f"  features: {', '.join(FEATURE_COLS)}")
    print(f"  label:   up_7d (secondary spike_50_7d)")

    try:
        samples = load_ai_samples(
            parquet_path=parquet if parquet.is_file() else None,
            duckdb_path=db if not parquet.is_file() else None,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if len(samples) == 0:
        print("ERROR: empty ai samples", file=sys.stderr)
        return 1

    result = run_phase_c(
        samples,
        cut_date=args.cut_date,
        threshold_method=args.threshold_method,
    )

    report = write_phase_c_report(
        result,
        args.report,
        parquet_path=str(parquet),
        script_path="scripts/run_ai_train.py",
    )
    jpath = save_result_json(result, args.json_out)

    print(f"  model:   {result.model_name}")
    print(f"  explore/train/val/later: {result.n_explore}/{result.n_train}/{result.n_val}/{result.n_later}")
    print(f"  later AUC: {_fmt(result.later_auc)}")
    print(f"  later accuracy: {result.later_accuracy:.4f} (majority {result.majority_baseline_acc:.4f})")
    print(
        f"  high-prob (thr={result.threshold:.4f}): fires={result.n_high_later} "
        f"up_hit={_fmt(result.high_up_hit_rate)} "
        f"lift={_fmt(result.high_up_lift_vs_base)}x "
        f"vs base {result.later_up_rate:.4f}"
    )
    print(
        f"  locked rule later: fires={result.locked_n_fire} "
        f"up_hit={_fmt(result.locked_up_hit_rate)} "
        f"lift={_fmt(result.locked_up_lift)}x"
    )
    print(f"  beats chance (AUC>0.5)? {result.beats_chance_auc}")
    print(f"  beats chance (high-prob)? {result.beats_chance_high_prob}")
    print(f"  beats locked (up)? {result.beats_locked_up}")
    print(f"  report: {report}")
    print(f"  json:   {jpath}")
    return 0


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
