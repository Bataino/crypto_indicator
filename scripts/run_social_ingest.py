#!/usr/bin/env python3
"""CLI: ingest social attention proxies → DuckDB social_daily (+ coverage report).

V0.1 sources (priority): Google Trends → Reddit → X (if bearer) → Wikipedia.
Does not claim alpha. Never prints API secrets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmram.config import get_duckdb_path, get_raw_dir, load_social_config
from cmram.ingest.social import coverage_markdown, write_coverage_report
from cmram.ingest.social_persist import build_and_write_social
from cmram.secrets_env import ensure_env_secrets


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMRAM social ingest (Trends/Reddit/X/wiki/Santiment → social_daily)."
    )
    p.add_argument("--db", type=str, default=None, help="DuckDB path.")
    p.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma list: trends,reddit,x,wikipedia,santiment (default: all enabled).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit assets in this batch (smoke / rate-cap). Default: all eligible.",
    )
    p.add_argument(
        "--x-max-assets",
        type=int,
        default=None,
        help="Cap X API calls this run (workable batch if rate/plan caps).",
    )
    p.add_argument(
        "--x-mode",
        choices=("counts", "search"),
        default=None,
        help=(
            "X ingest mode. Default from config (counts). "
            "counts = cheap GET /2/tweets/counts/recent (mention totals). "
            "search = EXPENSIVE actual posts (max 1 page by default)."
        ),
    )
    p.add_argument(
        "--x-cost-cap-usd",
        type=float,
        default=None,
        help="Override estimated_cost_cap_usd for this run (default from config, $1.00).",
    )
    p.add_argument(
        "--x-allow-over-cap",
        action="store_true",
        help=(
            "Explicit override: run even if estimated X spend exceeds the cost cap. "
            "Still makes no calls if you have no credits."
        ),
    )
    p.add_argument(
        "--santiment-call-budget",
        type=int,
        default=None,
        help="Max Santiment GraphQL calls this run (default from config, e.g. 150).",
    )
    p.add_argument(
        "--santiment-max-assets",
        type=int,
        default=None,
        help="Cap Santiment assets after slug mapping (smoke / budget).",
    )
    p.add_argument(
        "--santiment-probe-only",
        action="store_true",
        help="Only map slugs + write coverage note; no timeseries ingest.",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Wipe all social_daily before write (default: wipe only when full ingest).",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing data/raw/social/social_daily.parquet.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Load secrets into os.environ for this process (+ children). Never log values.
    sec = ensure_env_secrets(
        (
            "X_BEARER_TOKEN",
            "TWITTER_BEARER_TOKEN",
            "COINGECKO_API_KEY",
            "SANTIMENT_API_KEY",
        )
    )
    x_st = sec.get("X_BEARER_TOKEN") or sec.get("TWITTER_BEARER_TOKEN") or {}
    s_st = sec.get("SANTIMENT_API_KEY") or {}
    print(
        "  secrets: X_BEARER present="
        f"{bool(x_st.get('present'))} length={x_st.get('length', 0)} "
        f"from_file={x_st.get('loaded_from_file')}"
    )
    print(
        "  secrets: SANTIMENT_API_KEY present="
        f"{bool(s_st.get('present'))} length={s_st.get('length', 0)} "
        f"from_file={s_st.get('loaded_from_file')}"
    )

    cfg = load_social_config()
    db_path = Path(args.db) if args.db else get_duckdb_path()
    raw = get_raw_dir() / "social"
    parquet = None if args.no_parquet else raw / "social_daily.parquet"
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None

    print("CMRAM run_social_ingest")
    print(f"  duckdb: {db_path}")
    print(f"  sources: {sources or 'config defaults (trends→reddit→x→wikipedia→santiment)'}")
    print(f"  santiment_call_budget: {args.santiment_call_budget}")
    print(f"  santiment_max_assets: {args.santiment_max_assets}")
    print(f"  santiment_probe_only: {args.santiment_probe_only}")
    print(f"  asset_limit: {args.limit}")
    print(f"  x_max_assets: {args.x_max_assets}")
    print(f"  x_mode: {args.x_mode or 'config default (counts)'}")
    print(f"  x_cost_cap_usd: {args.x_cost_cap_usd}")
    print(f"  x_allow_over_cap: {args.x_allow_over_cap}")
    print("  status: attention proxies only — not alpha")

    if not db_path.is_file():
        print(f"ERROR: DuckDB not found: {db_path}", file=sys.stderr)
        return 1

    if args.santiment_probe_only:
        from cmram.db.schema import init_db
        from cmram.ingest.santiment import coverage_markdown_santiment, probe_coverage
        from cmram.ingest.social_persist import load_eligible_assets

        conn = init_db(db_path)
        try:
            assets = load_eligible_assets(conn)
            if args.limit is not None:
                assets = assets.head(int(args.limit))
        finally:
            conn.close()
        budget = args.santiment_call_budget or 20
        src_cfg = cfg.get("sources") or cfg
        cov = probe_coverage(assets, config=src_cfg, call_budget=budget)
        raw.mkdir(parents=True, exist_ok=True)
        note = raw / "santiment_coverage.md"
        payload = {
            "coverage": cov,
            "metric": cov.get("metric"),
            "plan": cov.get("plan"),
            "calls_used": cov.get("calls_used"),
            "call_budget": budget,
            "assets": 0,
            "rows": 0,
            "skipped": [
                {"asset_id": m.get("asset_id"), "reason": m.get("map_reason")}
                for m in (cov.get("missing") or [])
            ],
        }
        note.write_text(coverage_markdown_santiment(payload), encoding="utf-8")
        write_coverage_report(cov, raw / "santiment_coverage.json")
        print(f"  santiment covered: {cov.get('n_covered')}/{cov.get('n_assets')}")
        print(f"  window: {cov.get('window_from')} → {cov.get('window_to')}")
        print(f"  calls_used: {cov.get('calls_used')}")
        print(f"  report: {note}")
        for m in cov.get("missing") or []:
            print(f"  missing: {m.get('asset_id')} ({m.get('map_reason')})")
        print("  status: probe_only (no timeseries ingest)")
        return 0

    replace = True if args.replace else None  # None → auto (full wipe only if all sources)
    social, report = build_and_write_social(
        db_path,
        config=cfg,
        parquet_path=parquet,
        sources=sources,
        replace=replace,
        x_max_assets=args.x_max_assets,
        asset_limit=args.limit,
        x_mode=args.x_mode,
        x_allow_over_cap=args.x_allow_over_cap,
        x_cost_cap_usd=args.x_cost_cap_usd,
        santiment_call_budget=args.santiment_call_budget,
        santiment_max_assets=args.santiment_max_assets,
    )

    report_json = raw / "n_coverage.json"
    report_md = raw / "n_coverage.md"
    write_coverage_report(report, report_json)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(coverage_markdown(report), encoding="utf-8")

    # Plain-language X note
    x_note = raw / "x_ingest_note.md"
    x_note.write_text(_x_plain_note(report), encoding="utf-8")

    print(f"  rows_in_db: {len(social)}")
    print(
        f"  coverage: {report.get('n_assets_with_any_source')}/"
        f"{report.get('n_assets_requested')} assets"
    )
    for src, st in (report.get("per_source") or {}).items():
        print(
            f"  {src}: assets={st.get('assets')} rows={st.get('rows')} "
            f"{st.get('date_min')}→{st.get('date_max')}"
            + (f" err={st.get('error')}" if st.get("error") else "")
        )
    estimate = report.get("x_cost_estimate") or (
        ((report.get("per_source") or {}).get("x") or {}).get("cost_estimate")
    )
    if estimate:
        print(
            "  x_cost_estimate: "
            f"mode={estimate.get('mode')} assets={estimate.get('n_assets')} "
            f"requests={estimate.get('n_requests')} × "
            f"${estimate.get('cost_per_request_usd')} = "
            f"${estimate.get('estimated_cost_usd')} "
            f"(cap ${estimate.get('cost_cap_usd')})"
        )
        print(
            "  x_cost_note: config price; X developer console wins if different"
        )
    blocker = report.get("x_plan_blocker")
    cost_block = report.get("x_cost_cap_blocker")
    if blocker:
        print("  X PLAN BLOCKER — stopped cleanly (no retry loop)")
        print(f"    error_class: {blocker.get('error_class')}")
        print(f"    status_code: {blocker.get('status_code')}")
        print(f"    upgrade: {blocker.get('upgrade_hint')}")
    if cost_block:
        print("  X COST CAP — stopped before any X HTTP calls")
        print(f"    estimated: ${cost_block.get('estimated_cost_usd')}")
        print(f"    cap: ${cost_block.get('cost_cap_usd')}")
        print(f"    hint: {cost_block.get('hint')}")

    san = (report.get("per_source") or {}).get("santiment") or {}
    if san.get("assets") or san.get("rows") or san.get("skipped") or san.get("coverage"):
        from cmram.ingest.santiment import coverage_markdown_santiment
        san_note = raw / "santiment_coverage.md"
        cov = dict(san.get("coverage") or {})
        if "covered" not in cov and san.get("ingested"):
            cov["covered"] = san.get("ingested")
        if "missing" not in cov:
            cov["missing"] = [
                s for s in (san.get("skipped") or [])
                if (s.get("reason") or "").startswith("slug_missing")
            ]
        san_payload = {
            "coverage": cov,
            "metric": "social_volume_total",
            "plan": san.get("plan") or {},
            "calls_used": san.get("calls_used"),
            "call_budget": san.get("call_budget"),
            "assets": san.get("assets"),
            "rows": san.get("rows"),
            "skipped": san.get("skipped") or [],
            "ingested": san.get("ingested") or [],
            "stopped_reason": san.get("stopped_reason"),
        }
        san_note.write_text(coverage_markdown_santiment(san_payload), encoding="utf-8")
        print(f"  santiment_note: {san_note}")
        print(
            f"  santiment_detail: assets={san.get('assets')} rows={san.get('rows')} "
            f"calls={san.get('calls_used')}/{san.get('call_budget')}"
        )

    print(f"  report: {report_md}")
    print(f"  x_note: {x_note}")
    print(f"  next_gap: {report.get('next_gap')}")
    if blocker:
        status = "blocked_on_x_plan"
    elif cost_block:
        status = "blocked_on_x_cost_cap"
    else:
        status = "ok"
    print(f"  status: {status}")
    return 0


def _x_plain_note(report: dict) -> str:
    flags = report.get("source_flags") or {}
    x = (report.get("per_source") or {}).get("x") or {}
    blocker = report.get("x_plan_blocker")
    cost_block = report.get("x_cost_cap_blocker")
    estimate = report.get("x_cost_estimate") or x.get("cost_estimate") or {}
    mode = x.get("mode") or estimate.get("mode") or "counts"
    lines = [
        "# X (Twitter) ingest note",
        "",
        "Attention proxy only — **not alpha**.",
        "",
        "Default X ingest uses **Counts: Recent** (`GET /2/tweets/counts/recent`, "
        "`granularity=day`, last 7 days, **one request per asset**). That is mention "
        "volume only — we do not download tweet text. Actual tweet **search** is an "
        "expensive opt-in (`--x-mode search`, 1 page cap by default).",
        "",
        f"- Credentials seen by process: **{flags.get('x_credentials')}** (presence/length only; token never printed)",
        f"- Mode this run: **{mode}**",
        f"- X rows this DB: assets={x.get('assets')} rows={x.get('rows')} "
        f"range={x.get('date_min')}→{x.get('date_max')}",
        "",
    ]
    if estimate:
        lines += [
            "## Cost estimate (before live calls)",
            "",
            f"- Assets × requests × configured price: "
            f"**{estimate.get('n_assets')} × {estimate.get('requests_per_asset')} × "
            f"${estimate.get('cost_per_request_usd')} = ${estimate.get('estimated_cost_usd')}**",
            f"- Hard cap: **${estimate.get('cost_cap_usd')}** "
            f"(over_cap={estimate.get('over_cap')}, allow_over_cap={estimate.get('allow_over_cap')})",
            f"- {estimate.get('cost_note') or 'X developer console price wins if different from config.'}",
            "",
        ]
    if cost_block:
        lines += [
            "## Cost cap — no X HTTP calls",
            "",
            f"Estimated spend **${cost_block.get('estimated_cost_usd')}** is above the "
            f"cap **${cost_block.get('cost_cap_usd')}**. Nothing was sent to X.",
            "",
            str(cost_block.get("hint") or ""),
            "",
            "Override only if you mean it: `--x-allow-over-cap` or a higher "
            "`--x-cost-cap-usd`.",
            "",
        ]
    if blocker:
        lines += [
            "## Plan / access blocker",
            "",
            f"The X API refused the request with **HTTP {blocker.get('status_code')}** "
            f"(`{blocker.get('error_class')}`).",
            "",
            str(blocker.get("upgrade_hint") or ""),
            "",
            "No further X calls were made after this error.",
            "",
        ]
        if blocker.get("body_excerpt"):
            lines += [
                "API excerpt (truncated):",
                "",
                "```",
                str(blocker["body_excerpt"]),
                "```",
                "",
            ]
    if x.get("assets", 0) and not blocker and not cost_block:
        ids = x.get("asset_ids") or []
        kind = (
            "Counts: Recent daily mention totals"
            if mode != "search"
            else "EXPENSIVE recent-search post buckets"
        )
        lines += [
            "## What came back",
            "",
            f"{kind} landed for **{x.get('assets')}** assets "
            f"over **{x.get('date_min')} → {x.get('date_max')}** (≈7-day window).",
            "",
            "Assets with X:",
            "",
        ]
        for a in ids:
            lines.append(f"- `{a}`")
        lines += [
            "",
            "These combine with existing Trends/Wikipedia series when Model C "
            "rebuilds `score_N` (mean attention across sources present that day).",
            "",
        ]
    elif not blocker and not cost_block:
        lines += [
            "## No X rows",
            "",
            "Credentials may be present but this run returned no mention buckets "
            "(empty queries, soft errors, skipped, or no credits yet). See `n_coverage.md`.",
            "",
        ]
    lines += [
        "## After adding credits ($5–$10 is plenty for counts)",
        "",
        "Counts: Recent is about **$0.005 per request** in the current console listing "
        "(re-check — console wins). One request per eligible asset. Example: 38 assets ≈ **$0.19**.",
        "",
        "```bash",
        "python scripts/run_social_ingest.py --sources x --x-mode counts -v",
        "```",
        "",
        "Optional small smoke first: `--x-max-assets 5`. Search mode is EXPENSIVE; do not use it unless you need post text.",
        "",
    ]
    return "\n".join(lines) + "\n"



if __name__ == "__main__":
    raise SystemExit(main())
