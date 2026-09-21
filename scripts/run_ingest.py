#!/usr/bin/env python3
"""CLI: CoinGecko daily OHLCV + market cap ingest → data/raw + DuckDB."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.config import get_duckdb_path, get_raw_dir, load_universe_config
from cmram.ingest.coingecko import CoinGeckoError, RateLimitError
from cmram.ingest.market import ingest_coingecko
from cmram.secrets_env import ensure_env_secrets, secret_status


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMRAM CoinGecko ingest (research data only — no trading)."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max assets to ingest (default 5). Use with band discovery.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Wider markets pagination for full-universe candidate scrape "
        "(still respects --limit unless you pass a large --limit).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover candidates only; do not fetch charts or write DuckDB.",
    )
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated CoinGecko ids (skip band discovery).",
    )
    p.add_argument(
        "--days",
        type=int,
        default=180,
        help="History window for market_chart/OHLC (default 180; prefer ≥90).",
    )
    p.add_argument(
        "--ohlc",
        action="store_true",
        help="Also fetch /ohlc (true OHLC). Default uses market_chart close for O/H/L/C.",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-fetch assets even if DuckDB already has ≥min_history_days bars.",
    )
    p.add_argument(
        "--no-prefer-raw",
        action="store_true",
        help="Do not reuse on-disk market_chart.json; always hit the API.",
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="DuckDB path (default data/cmram.duckdb or CMRAM_DUCKDB_PATH).",
    )
    p.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Raw dump dir (default data/raw or CMRAM_RAW_DIR).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Load API keys from box-secrets into env before any CoinGecko calls.
    # Never print secret values — presence + length only.
    sec = ensure_env_secrets(("COINGECKO_API_KEY",))
    st = sec.get("COINGECKO_API_KEY") or secret_status("COINGECKO_API_KEY")
    print(
        f"  COINGECKO_API_KEY: present={st.get('present')} "
        f"length={st.get('length')}"
    )

    universe = load_universe_config()
    asset_ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None
    db_path = Path(args.db) if args.db else get_duckdb_path()
    raw_dir = Path(args.raw_dir) if args.raw_dir else get_raw_dir()

    print("CMRAM run_ingest")
    print(f"  bands: {list(universe.get('bands', {}))}")
    print(f"  min_history_days: {universe.get('min_history_days')}")
    print(f"  limit: {args.limit}  full={args.full}  dry_run={args.dry_run}")
    print(
        f"  skip_existing={not args.no_skip_existing}  "
        f"prefer_raw={not args.no_prefer_raw}"
    )
    print(f"  days: {args.days}")
    print(f"  duckdb: {db_path}")
    print(f"  raw_dir: {raw_dir}")
    if asset_ids:
        print(f"  ids: {asset_ids}")

    try:
        result = ingest_coingecko(
            asset_ids=asset_ids,
            limit=args.limit,
            full=args.full,
            dry_run=args.dry_run,
            days=args.days,
            include_ohlc=args.ohlc,
            duckdb_path=db_path,
            raw_dir=raw_dir,
            universe=universe,
            skip_existing=not args.no_skip_existing,
            prefer_raw=not args.no_prefer_raw,
        )
    except RateLimitError as exc:
        print(f"ERROR: rate limited by CoinGecko: {exc}", file=sys.stderr)
        print(
            "Hint: set COINGECKO_API_KEY (Demo/Pro), wait for Retry-After, "
            "or reduce --limit.",
            file=sys.stderr,
        )
        return 2
    except CoinGeckoError as exc:
        print(f"ERROR: CoinGecko ingest failed: {exc}", file=sys.stderr)
        return 1

    print(f"  candidates: {result.candidates}")
    print(f"  assets_upserted: {result.assets_upserted}")
    print(f"  market_rows_upserted: {result.market_rows_upserted}")
    if result.skipped_insufficient:
        print(f"  skipped: {result.skipped_insufficient}")
    print("  status: ok" if not args.dry_run else "  status: dry-run ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
