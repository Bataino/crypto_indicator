#!/usr/bin/env python3
"""CLI: compute Phase 1 Model A/B features into DuckDB features_daily.

Research only — no signals, no trading, no claimed alpha.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.config import get_duckdb_path, get_features_dir, load_features_config
from cmram.features.persist import build_and_write_features
from cmram.features.pipeline import features_summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMRAM features (Models A/B: C/V/P + partial NSI → MREI/Gap)."
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB path (default data/cmram.duckdb or CMRAM_DUCKDB_PATH).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Features YAML path (default config/features.yaml).",
    )
    p.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Optional parquet output path (default data/features/features_daily.parquet).",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing data/features/features_daily.parquet.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return p.parse_args(argv)


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def _report(features) -> None:
    summary = features_summary(features)
    print(f"  rows: {summary['total_rows']}")
    if summary["total_rows"] == 0:
        return
    dr = summary.get("date_range") or ["?", "?"]
    print(f"  date_range: {dr[0]} → {dr[1]}")
    bench = summary.get("rs_benchmark")
    print(f"  rs_benchmark: {bench}")
    print(f"  model_version: {summary.get('model_version')}")
    print("  limitation: small_sample flagged when n_in_band < 10; n=3 → coarse 0/50/100 ranks")
    n_true = sum(g.get("n_available_true", 0) for g in summary.get("groups", {}).values())
    print(f"  social: n_available_true rows (all models)={n_true} (not imputed when missing)")
    for key, g in summary["groups"].items():
        band, model = key.split("|", 1)
        print(
            f"  band {band} model {model}: rows={g['rows']} assets={g['assets']} "
            f"days={g['days']} n_in_band={g['n_in_band_min']}-{g['n_in_band_max']} "
            f"small_sample={g['small_sample_true']}/{g['rows']} "
            f"n_available_true={g['n_available_true']}"
        )
        for col in ("score_C", "score_V", "score_P", "MREI", "nsi_price_ext",
                    "nsi_vol_exh", "nsi_mom_dec", "NSI", "Rotation_Gap", "score_N"):
            st = g[col]
            print(
                f"    {col}: min={_fmt(st['min'])} max={_fmt(st['max'])} "
                f"mean={_fmt(st['mean'])} nulls={st['nulls']}"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_features_config(Path(args.config) if args.config else None)
    db_path = Path(args.db) if args.db else get_duckdb_path()
    if args.no_parquet:
        parquet = None
    elif args.parquet:
        parquet = Path(args.parquet)
    else:
        parquet = get_features_dir() / "features_daily.parquet"

    print("CMRAM run_features")
    print(f"  models: {cfg.get('models')}")
    print(f"  model_version: {cfg.get('model_version')}")
    print(f"  config: {args.config or 'config/features.yaml'}")
    print(f"  duckdb: {db_path}")
    print("  status: computing C/V/P/(N) + NSI (no signals; not alpha)")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    features = build_and_write_features(
        db_path, config=cfg, parquet_path=parquet
    )
    _report(features)
    if parquet is not None:
        print(f"  parquet: {parquet}")
    print("  status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
