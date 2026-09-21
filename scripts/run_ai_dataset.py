#!/usr/bin/env python3
"""CLI: build Phase B AI labeled dataset (features + labels) — no training.

Writes DuckDB table ``ai_samples_daily`` and parquet under data/ai/.
Research only — no trading, no X/social.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.config import DATA_DIR, get_duckdb_path
from cmram.ml.dataset import (
    BAND_C,
    EXPLORE_FRAC_DEFAULT,
    FEATURE_COLS,
    PRIMARY_FEATURES,
    apply_time_split,
    build_ai_dataset_from_db,
)

LAGOS = ZoneInfo("Africa/Lagos")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMRAM AI Phase B: point-in-time labels + feature matrix."
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB path (default data/cmram.duckdb).",
    )
    p.add_argument(
        "--band",
        type=str,
        default=BAND_C,
        help=f"Universe band (default {BAND_C}).",
    )
    p.add_argument(
        "--allow-failed-liquidity",
        action="store_true",
        help="Include Band C days that failed liquidity_pass (default: pass only).",
    )
    p.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Parquet output (default data/ai/ai_samples_daily.parquet).",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip parquet write.",
    )
    p.add_argument(
        "--explore-frac",
        type=float,
        default=EXPLORE_FRAC_DEFAULT,
        help="Fraction of unique days for explore split (default 0.7).",
    )
    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Markdown report path (default docs/reports/phase_b_ai_dataset.md).",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the phase_b markdown report.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.2f}%"


def _write_report(path: Path, stats: dict, *, parquet: str | None, db: str) -> None:
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M:%S WAT")
    null_lines = []
    for col, rate in (stats.get("null_rates") or {}).items():
        null_lines.append(f"| `{col}` | {_pct(rate)} |")
    null_table = "\n".join(null_lines) if null_lines else "| (none) | — |"

    bands = ", ".join(stats.get("band") or []) or "—"
    ok = stats.get("primary_non_null_rows", 0) >= 50_000
    verdict = (
        f"**PASS** — {stats['primary_non_null_rows']:,} usable rows "
        f"(≥50,000 target)."
        if ok
        else (
            f"**SHORT** — {stats.get('primary_non_null_rows', 0):,} usable rows "
            f"(target ≥50,000). See notes below."
        )
    )

    body = f"""# Phase B — AI labeled dataset (next-move)

**Zone:** Africa/Lagos (WAT / UTC+1)  
**Report time:** {now}  
**Scope:** Research only — labels + features. **No LightGBM training** (Phase C next).  
**Social / Gap / X:** OFF for v1.

## What we built

For each Band C coin-day we record:

1. **Inputs the model may see later (features, past/present only):**
   - `rvol_30` — today's volume ÷ 30-day average volume
   - `dist_ema_20` / `dist_ema_50` — how far price is from EMA20 / EMA50
   - `rsi_14` — 14-day RSI
   - `ret_1d`, `ret_3d`, `ret_7d` — recent simple returns
   - `vol_14` — 14-day realized volatility (std of log returns)
   - `log_mc` — log of point-in-time market cap (USD)
2. **What happened next (labels, future only):**
   - **`up_7d`** (primary) — 1 if close 7 days later is higher than today
   - **`spike_50_7d`** (secondary) — 1 if max close within the next 7 days is ≥ +50%
   - **`fwd_ret_7d`** — continuous 7-day return

No future prices leak into features. Rows need a real bar at t+7 and non-null primary features.

## Universe

| Knob | Value |
|------|-------|
| Band | `{bands}` (prefer liquidity_pass) |
| Explore fraction | {stats.get('explore_frac', 0.7):.0%} of unique calendar days |

## Counts (plain English)

| Metric | Value |
|--------|------:|
| Usable labeled rows | **{stats['rows']:,}** |
| Distinct assets | {stats['assets']:,} |
| Date span | {stats['date_min']} → {stats['date_max']} |
| Explore cut date (inclusive) | **{stats['cut_date']}** |
| Explore rows (≤ cut) | {stats['explore_rows']:,} |
| Later rows (> cut) | {stats['later_rows']:,} |
| `up_7d` rate (everyday chance up) | **{_pct(stats.get('up_7d_rate'))}** |
| `spike_50_7d` rate | {_pct(stats.get('spike_50_7d_rate'))} |
| Rows with all primary features non-null | {stats.get('primary_non_null_rows', 0):,} |

Primary features required: {", ".join(f"`{c}`" for c in PRIMARY_FEATURES)}.

### Success vs ≥50K usable rows

{verdict}

## Null rates (after filters)

| Column | Null rate |
|--------|----------:|
{null_table}

(After label + primary-feature filters, feature null rates should be ~0 for primary columns; `dist_ema_50` may still have a few nulls if listed as secondary.)

## Files written

| Artifact | Path |
|----------|------|
| DuckDB table | `{db}` → table `ai_samples_daily` |
| Parquet | `{parquet or "(skipped)"}` |
| This report | `{path}` |

## Time split (for Phase C train)

- **Explore (train / light tune):** timestamps ≤ **{stats['cut_date']}**
- **Later (fair test only):** timestamps > **{stats['cut_date']}**
- Printed once at explore_frac={stats.get('explore_frac', 0.7)}.

Do **not** peek at later rows while tuning.

## Feature columns (v1)

{", ".join(f"`{c}`" for c in FEATURE_COLS)}

## Next (Phase C — not done here)

1. Install `lightgbm` + `scikit-learn` if needed (`pip install 'cmram[ml]'` or deps in pyproject).
2. Train LightGBM classifier on explore rows → P(`up_7d`=1).
3. Score later window once; compare vs chance and vs locked draft  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.
4. Plain Abdul report — research only, no orders.

## Explicit non-goals (honored)

- No model training this phase
- No X / Santiment / social features
- No trading / wallets / live signals
- No GitHub push required for this step
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    db_path = Path(args.db) if args.db else get_duckdb_path()
    if args.no_parquet:
        parquet = None
    elif args.parquet:
        parquet = Path(args.parquet)
    else:
        parquet = DATA_DIR / "ai" / "ai_samples_daily.parquet"

    report_path = None
    if not args.no_report:
        report_path = (
            Path(args.report)
            if args.report
            else ROOT / "docs" / "reports" / "phase_b_ai_dataset.md"
        )

    print("CMRAM run_ai_dataset (Phase B)")
    print(f"  duckdb: {db_path}")
    print(f"  band: {args.band}")
    print(f"  liquidity_pass only: {not args.allow_failed_liquidity}")
    print(f"  explore_frac: {args.explore_frac}")
    print(f"  parquet: {parquet or '(skip)'}")
    print("  status: building labels + features (no train)")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    samples, stats = build_ai_dataset_from_db(
        db_path,
        band=args.band,
        require_liquidity_pass=not args.allow_failed_liquidity,
        parquet_path=parquet,
        explore_frac=args.explore_frac,
    )

    explore, later, cut = apply_time_split(samples, explore_frac=args.explore_frac)
    print(f"  rows: {stats['rows']:,}")
    print(f"  assets: {stats['assets']:,}")
    print(f"  date_range: {stats['date_min']} → {stats['date_max']}")
    print(f"  cut_date (explore ≤): {cut.date()}  [explore_frac={args.explore_frac}]")
    print(f"  explore_rows: {len(explore):,}  later_rows: {len(later):,}")
    print(f"  up_7d_rate: {_pct(stats.get('up_7d_rate'))}")
    print(f"  spike_50_7d_rate: {_pct(stats.get('spike_50_7d_rate'))}")
    print(f"  primary_non_null_rows: {stats.get('primary_non_null_rows', 0):,}")
    if parquet:
        print(f"  parquet: {parquet}")
    print("  duckdb_table: ai_samples_daily")

    if report_path is not None:
        _write_report(
            report_path,
            stats,
            parquet=str(parquet) if parquet else None,
            db=str(db_path),
        )
        print(f"  report: {report_path}")

    print("  status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
