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
from cmram.ml.dataset import time_split_cut_date
from cmram.ml.train import (
    DEFAULT_CUT_DATE,
    DEFAULT_TIGHTEN_BANDS,
    EXPLORE_QUANTILE_METHODS,
    PRIMARY_LABEL,
    SECONDARY_LABEL,
    VALID_TARGETS,
    load_ai_samples,
    run_phase_c,
    run_phase_c_multi_band,
    save_multi_band_result_json,
    save_result_json,
    write_phase_c_report,
    write_spike50_altcut_report,
    write_spike50_tighten_report,
)


def _parse_bands(raw: str | None) -> list[str] | None:
    """Parse comma-separated band keys (p80,p90,p95 / top_quintile / …)."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--bands must be non-empty")
    bad = [p for p in parts if p not in EXPLORE_QUANTILE_METHODS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown band(s) {bad}; "
            f"choose from {sorted(EXPLORE_QUANTILE_METHODS)}"
        )
    return parts


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
        default=None,
        help=(
            f"Explore inclusive cut date (default {DEFAULT_CUT_DATE} when "
            "--explore-frac is not set). Ignored when --explore-frac is set."
        ),
    )
    p.add_argument(
        "--explore-frac",
        type=float,
        default=None,
        help=(
            "Fraction of unique calendar days for explore (e.g. 0.6). "
            "When set, derives and prints the cut date from the sample dates "
            "(overrides --cut-date)."
        ),
    )
    p.add_argument(
        "--target",
        type=str,
        choices=list(VALID_TARGETS),
        default=PRIMARY_LABEL,
        help=(
            f"Training label (default {PRIMARY_LABEL}). "
            f"Use {SECONDARY_LABEL} for rare-event spike fair-test."
        ),
    )
    p.add_argument(
        "--threshold-method",
        type=str,
        choices=sorted(set(EXPLORE_QUANTILE_METHODS) | {"f1"}),
        default="top_quintile",
        help="How to pick high-prob threshold on explore only (single-band mode).",
    )
    p.add_argument(
        "--bands",
        type=str,
        default=None,
        help=(
            "Comma-separated explore-only high-prob cuts to fair-test together "
            f"(e.g. p80,p90,p95). Default for spike tighten: "
            f"{','.join(DEFAULT_TIGHTEN_BANDS)}. "
            "When set, trains once and scores each band on later."
        ),
    )
    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Markdown report path (default depends on --target / --bands).",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="JSON metrics dump path (default depends on --target / --bands).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _default_paths(
    target: str,
    bands: list[str] | None,
    *,
    explore_frac: float | None = None,
) -> tuple[str, str]:
    # Alt-cut robustness report when explore_frac is set away from primary 0.7
    if (
        bands is not None
        and target == SECONDARY_LABEL
        and explore_frac is not None
        and abs(float(explore_frac) - 0.7) > 1e-9
    ):
        return (
            str(ROOT / "docs" / "reports" / "phase_c_spike50_altcut_fair_test.md"),
            str(DATA_DIR / "ai" / "phase_c_spike50_altcut_fair_test.json"),
        )
    if bands is not None and target == SECONDARY_LABEL:
        return (
            str(ROOT / "docs" / "reports" / "phase_c_spike50_tighten_fair_test.md"),
            str(DATA_DIR / "ai" / "phase_c_spike50_tighten_fair_test.json"),
        )
    if target == SECONDARY_LABEL:
        return (
            str(ROOT / "docs" / "reports" / "phase_c_spike50_lightgbm_fair_test.md"),
            str(DATA_DIR / "ai" / "phase_c_spike50_fair_test.json"),
        )
    return (
        str(ROOT / "docs" / "reports" / "phase_c_lightgbm_fair_test.md"),
        str(DATA_DIR / "ai" / "phase_c_fair_test.json"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        bands = _parse_bands(args.bands)
    except argparse.ArgumentTypeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Spike tighten convenience: --bands with no value-ish default via explicit flag
    # If user passes --bands alone empty, argparse gives ""; we already reject.
    parquet = Path(args.parquet)
    db = Path(args.db) if args.db else get_duckdb_path()
    explore_frac = args.explore_frac
    if explore_frac is not None and not (0.0 < float(explore_frac) < 1.0):
        print(
            f"ERROR: --explore-frac must be in (0,1), got {explore_frac}",
            file=sys.stderr,
        )
        return 2
    def_report, def_json = _default_paths(
        args.target, bands, explore_frac=explore_frac
    )
    report_path = args.report or def_report
    json_out = args.json_out or def_json

    print("CMRAM run_ai_train (Phase C)")
    print("  intent: research fair-test only — NOT alpha / not a bot")
    print(f"  parquet: {parquet}")
    print(f"  duckdb:  {db}")
    print(f"  features: {', '.join(FEATURE_COLS)}")
    print(f"  label:   {args.target}")
    if bands is not None:
        print(f"  bands:   {', '.join(bands)} (explore-locked; later once each)")
    else:
        print(f"  threshold-method: {args.threshold_method}")

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

    # Resolve cut: explore_frac derives cut from sample dates; else cut-date / default.
    cut_date = args.cut_date
    if explore_frac is not None:
        cut_ts = time_split_cut_date(
            samples["timestamp"], explore_frac=float(explore_frac)
        )
        cut_date = str(cut_ts.date())
        print(
            f"  cut:     explore <= {cut_date} "
            f"(from explore_frac={float(explore_frac):.2f}); later > cut"
        )
    else:
        cut_date = cut_date or DEFAULT_CUT_DATE
        print(f"  cut:     explore <= {cut_date}; later > cut")

    if bands is not None:
        result = run_phase_c_multi_band(
            samples,
            cut_date=cut_date if explore_frac is None else None,
            explore_frac=float(explore_frac) if explore_frac is not None else None,
            band_methods=bands,
            target=args.target,
        )
        script_bits = [
            "scripts/run_ai_train.py",
            f"--target {args.target}",
            f"--bands {','.join(bands)}",
        ]
        if explore_frac is not None:
            script_bits.append(f"--explore-frac {float(explore_frac)}")
        else:
            script_bits.append(f"--cut-date {cut_date}")
        script = " ".join(script_bits)
        use_altcut = (
            explore_frac is not None
            and abs(float(explore_frac) - 0.7) > 1e-9
            and args.target == SECONDARY_LABEL
        )
        if use_altcut:
            report = write_spike50_altcut_report(
                result,
                report_path,
                parquet_path=str(parquet),
                script_path=script,
                json_path=str(json_out),
            )
        else:
            report = write_spike50_tighten_report(
                result,
                report_path,
                parquet_path=str(parquet),
                script_path=script,
                json_path=str(json_out),
            )
        jpath = save_multi_band_result_json(result, json_out)

        print(f"  model:   {result.model_name}")
        print(
            f"  explore/train/val/later: "
            f"{result.n_explore}/{result.n_train}/{result.n_val}/{result.n_later}"
        )
        print(f"  later AUC: {_fmt(result.later_auc)}")
        print(
            f"  locked rule later: fires={result.locked_n_fire} "
            f"spike_hit={_fmt(result.locked_spike_hit_rate)} "
            f"lift={_fmt(result.locked_spike_lift)}x"
        )
        for b in result.bands:
            print(
                f"  band {b.band} (thr={b.threshold:.4f}): fires={b.n_high_later} "
                f"({100 * b.later_fire_frac:.1f}% later) "
                f"spike_hit={_fmt(b.high_spike_hit_rate)} "
                f"lift={_fmt(b.high_spike_lift_vs_base)}x "
                f"beats_chance={b.beats_chance_high_prob} "
                f"beats_locked_spike={b.beats_locked_spike}"
            )
        print(f"  beats chance (AUC>0.5)? {result.beats_chance_auc}")
        print(f"  report: {report}")
        print(f"  json:   {jpath}")
        return 0

    result = run_phase_c(
        samples,
        cut_date=cut_date if explore_frac is None else None,
        explore_frac=float(explore_frac) if explore_frac is not None else None,
        threshold_method=args.threshold_method,
        target=args.target,
    )

    report = write_phase_c_report(
        result,
        report_path,
        parquet_path=str(parquet),
        script_path=f"scripts/run_ai_train.py --target {args.target}",
    )
    jpath = save_result_json(result, json_out)

    print(f"  model:   {result.model_name}")
    print(
        f"  explore/train/val/later: "
        f"{result.n_explore}/{result.n_train}/{result.n_val}/{result.n_later}"
    )
    print(f"  later AUC: {_fmt(result.later_auc)}")
    print(f"  later accuracy: {result.later_accuracy:.4f} (majority {result.majority_baseline_acc:.4f})")
    if args.target == SECONDARY_LABEL:
        print(
            f"  high-prob (thr={result.threshold:.4f}): fires={result.n_high_later} "
            f"spike_hit={_fmt(result.high_spike_hit_rate)} "
            f"lift={_fmt(result.high_spike_lift_vs_base)}x "
            f"vs base {result.later_spike_rate:.4f}"
        )
        print(
            f"  locked rule later: fires={result.locked_n_fire} "
            f"spike_hit={_fmt(result.locked_spike_hit_rate)} "
            f"lift={_fmt(result.locked_spike_lift)}x"
        )
        print(f"  beats chance (AUC>0.5)? {result.beats_chance_auc}")
        print(f"  beats chance (high-prob spike)? {result.beats_chance_high_prob}")
        print(f"  beats locked (spike)? {result.beats_locked_spike}")
    else:
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
