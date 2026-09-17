"""Orchestrate multi-source social ingest → social_daily rows + coverage.

Priority V0.1: Google Trends → Reddit → X (if bearer) → Wikipedia → Santiment.
N lands when ANY source has series for an asset; never block on missing X.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cmram.ingest.reddit_public import fetch_reddit_for_assets
from cmram.ingest.trends import fetch_trends_for_assets
from cmram.ingest.wikipedia_views import fetch_wikipedia_for_assets
from cmram.ingest.santiment import (
    fetch_santiment_for_assets,
    santiment_credentials_available,
)
from cmram.ingest.x_twitter import (
    XCostCapError,
    XPlanLimitError,
    enforce_x_cost_cap,
    fetch_x_for_assets,
    plan_x_batch,
    x_credentials_available,
)

logger = logging.getLogger(__name__)

SOCIAL_COLUMNS = (
    "timestamp",
    "asset_id",
    "source",
    "mentions",
    "unique_users",
    "engagement",
    "community_growth",
    "raw_payload_ref",
)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SOCIAL_COLUMNS))


def source_availability(config: dict[str, Any] | None = None) -> dict[str, bool]:
    """Per-source enabled + credential flags (no secret values)."""
    cfg = config or {}
    sources = cfg.get("sources") or cfg
    trends_on = bool((sources.get("trends") or {}).get("enabled", True))
    reddit_on = bool((sources.get("reddit") or {}).get("enabled", True))
    x_cfg = sources.get("x") or {}
    x_on = bool(x_cfg.get("enabled", True))
    wiki_on = bool((sources.get("wikipedia") or {}).get("enabled", False))
    san_on = bool((sources.get("santiment") or {}).get("enabled", False))
    return {
        "trends_enabled": trends_on,
        "reddit_enabled": reddit_on,
        "x_enabled": x_on,
        "x_credentials": x_credentials_available(),
        "wikipedia_enabled": wiki_on,
        "santiment_enabled": san_on,
        "santiment_credentials": santiment_credentials_available(),
        "reddit_praw_env": bool(
            __import__("os").environ.get("REDDIT_CLIENT_ID")
            and __import__("os").environ.get("REDDIT_CLIENT_SECRET")
        ),
    }


def ingest_social(
    assets: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    sources: list[str] | None = None,
    x_max_assets: int | None = None,
    x_mode: str | None = None,
    x_allow_over_cap: bool = False,
    x_cost_cap_usd: float | None = None,
    santiment_call_budget: int | None = None,
    santiment_max_assets: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run enabled social sources; return (social_daily frame, coverage report).

    ``assets`` needs columns asset_id, symbol, name.

    On X plan/access limits (402/403), stops X cleanly and records
    ``x_plan_blocker`` in the report — does not loop forever.
    On estimated spend above ``estimated_cost_cap_usd``, records
    ``x_cost_cap_blocker`` and makes no X HTTP calls unless
    ``x_allow_over_cap`` is set.
    """
    cfg = config or {}
    src_cfg = cfg.get("sources") or cfg
    want = sources or ["trends", "reddit", "x", "wikipedia", "santiment"]
    frames: list[pd.DataFrame] = []
    per_source: dict[str, Any] = {}
    x_plan_blocker: dict[str, Any] | None = None
    x_cost_cap_blocker: dict[str, Any] | None = None
    x_cost_estimate: dict[str, Any] | None = None

    avail = source_availability(cfg)
    logger.info("Social source flags: %s", avail)

    if "trends" in want and bool((src_cfg.get("trends") or {}).get("enabled", True)):
        try:
            t = fetch_trends_for_assets(assets, config=src_cfg)
            frames.append(t)
            per_source["trends"] = _source_stats(t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trends ingest error: %s", exc)
            per_source["trends"] = {"error": str(exc), "assets": 0, "rows": 0}

    if "reddit" in want and bool((src_cfg.get("reddit") or {}).get("enabled", True)):
        try:
            r = fetch_reddit_for_assets(assets, config=src_cfg)
            frames.append(r)
            per_source["reddit"] = _source_stats(r)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reddit ingest error: %s", exc)
            per_source["reddit"] = {"error": str(exc), "assets": 0, "rows": 0}

    if "x" in want and bool((src_cfg.get("x") or {}).get("enabled", True)):
        try:
            x_plan, _work = plan_x_batch(
                assets,
                config=src_cfg,
                max_assets=x_max_assets,
                mode=x_mode,
                allow_over_cap=x_allow_over_cap,
                cost_cap_usd=x_cost_cap_usd,
            )
            x_cost_estimate = x_plan.as_dict()
            logger.info(
                "X preflight: mode=%s assets=%d requests=%d est=$%.4f cap=$%.2f",
                x_plan.mode,
                x_plan.n_assets,
                x_plan.n_requests,
                x_plan.estimated_cost_usd,
                x_plan.cost_cap_usd,
            )
            enforce_x_cost_cap(x_plan)
            x = fetch_x_for_assets(
                assets,
                config=src_cfg,
                max_assets=x_max_assets,
                mode=x_mode,
                allow_over_cap=x_allow_over_cap,
                cost_cap_usd=x_cost_cap_usd,
            )
            frames.append(x)
            per_source["x"] = _source_stats(x)
            per_source["x"]["credentials"] = avail["x_credentials"]
            per_source["x"]["mode"] = x_plan.mode
            per_source["x"]["cost_estimate"] = x_cost_estimate
        except XCostCapError as exc:
            logger.error("X cost cap: estimated=$%.4f cap=$%.2f (no HTTP)",
                         exc.estimated_cost_usd, exc.cost_cap_usd)
            x_cost_cap_blocker = exc.as_report()
            x_cost_estimate = exc.plan or x_cost_estimate
            per_source["x"] = {
                "assets": 0,
                "rows": 0,
                "credentials": avail["x_credentials"],
                "error": str(exc),
                "mode": exc.mode,
                "cost_estimate": x_cost_estimate,
                "cost_cap_blocker": x_cost_cap_blocker,
            }
        except XPlanLimitError as exc:
            logger.error("X plan blocker: %s", exc.error_class)
            x_plan_blocker = exc.as_report()
            per_source["x"] = {
                "assets": 0,
                "rows": 0,
                "credentials": avail["x_credentials"],
                "error": str(exc),
                "plan_blocker": x_plan_blocker,
            }
            if x_cost_estimate:
                per_source["x"]["cost_estimate"] = x_cost_estimate
        except Exception as exc:  # noqa: BLE001
            logger.warning("X ingest error: %s", exc)
            per_source["x"] = {
                "error": str(exc),
                "assets": 0,
                "rows": 0,
                "credentials": avail["x_credentials"],
            }
            if x_cost_estimate:
                per_source["x"]["cost_estimate"] = x_cost_estimate

    if "wikipedia" in want and bool(
        (src_cfg.get("wikipedia") or {}).get("enabled", False)
    ):
        try:
            w = fetch_wikipedia_for_assets(assets, config=src_cfg)
            frames.append(w)
            per_source["wikipedia"] = _source_stats(w)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wikipedia ingest error: %s", exc)
            per_source["wikipedia"] = {"error": str(exc), "assets": 0, "rows": 0}

    if "santiment" in want and bool(
        (src_cfg.get("santiment") or {}).get("enabled", False)
    ):
        try:
            if not avail.get("santiment_credentials"):
                per_source["santiment"] = {
                    "assets": 0,
                    "rows": 0,
                    "credentials": False,
                    "error": "SANTIMENT_API_KEY missing",
                }
            else:
                s_df, s_rep = fetch_santiment_for_assets(
                    assets,
                    config=src_cfg,
                    call_budget=santiment_call_budget,
                    max_assets=santiment_max_assets,
                )
                frames.append(s_df)
                st = _source_stats(s_df)
                st["credentials"] = True
                st["calls_used"] = s_rep.get("calls_used")
                st["call_budget"] = s_rep.get("call_budget")
                st["skipped"] = s_rep.get("skipped")
                st["plan"] = s_rep.get("plan")
                st["coverage"] = s_rep.get("coverage")
                st["stopped_reason"] = s_rep.get("stopped_reason")
                if s_rep.get("ingested"):
                    st["ingested"] = s_rep.get("ingested")
                per_source["santiment"] = st
        except Exception as exc:  # noqa: BLE001
            logger.warning("Santiment ingest error: %s", exc)
            per_source["santiment"] = {
                "error": str(exc),
                "assets": 0,
                "rows": 0,
                "credentials": avail.get("santiment_credentials"),
            }

    nonempty = [f for f in frames if f is not None and len(f) > 0]
    social = pd.concat(nonempty, ignore_index=True) if nonempty else _empty()

    report = build_coverage_report(
        social,
        assets,
        avail=avail,
        per_source=per_source,
        x_plan_blocker=x_plan_blocker,
        x_cost_cap_blocker=x_cost_cap_blocker,
    )
    if x_cost_estimate:
        report["x_cost_estimate"] = x_cost_estimate
    return social, report


def build_coverage_report(
    social: pd.DataFrame,
    assets: pd.DataFrame | None,
    *,
    avail: dict[str, bool] | None = None,
    per_source: dict[str, Any] | None = None,
    x_plan_blocker: dict[str, Any] | None = None,
    x_cost_cap_blocker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build coverage dict from a social_daily frame (+ optional per_source extras)."""
    asset_ids = (
        set(assets["asset_id"].astype(str))
        if assets is not None and len(assets)
        else set()
    )
    covered: dict[str, set[str]] = {}
    if social is not None and len(social):
        for src, g in social.groupby("source"):
            covered[str(src)] = set(g["asset_id"].astype(str))
    any_covered = set().union(*covered.values()) if covered else set()

    stats = per_source
    if stats is None:
        stats = {}
        if social is not None and len(social):
            for src, g in social.groupby("source"):
                stats[str(src)] = _source_stats(g)

    flags = avail if avail is not None else {}
    report: dict[str, Any] = {
        "n_assets_requested": len(asset_ids),
        "n_assets_with_any_source": len(any_covered),
        "assets_with_n": sorted(any_covered),
        "assets_missing_n": sorted(asset_ids - any_covered),
        "per_source": stats,
        "source_flags": flags,
        "rows_total": int(len(social) if social is not None else 0),
        "next_gap": _next_gap(
            flags, stats, x_plan_blocker, x_cost_cap_blocker=x_cost_cap_blocker
        ),
        "note": (
            "Attention proxies only — not alpha. Optional later: LunarCrush / "
            "Santiment if a free tier is available."
        ),
    }
    if x_plan_blocker:
        report["x_plan_blocker"] = x_plan_blocker
    if x_cost_cap_blocker:
        report["x_cost_cap_blocker"] = x_cost_cap_blocker
    return report


def _source_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"assets": 0, "rows": 0, "date_min": None, "date_max": None}
    ts = pd.to_datetime(df["timestamp"])
    return {
        "assets": int(df["asset_id"].nunique()),
        "rows": int(len(df)),
        "date_min": str(ts.min().date()),
        "date_max": str(ts.max().date()),
        "asset_ids": sorted(df["asset_id"].astype(str).unique()),
    }


def _next_gap(
    avail: dict[str, bool],
    per_source: dict[str, Any],
    x_plan_blocker: dict[str, Any] | None = None,
    x_cost_cap_blocker: dict[str, Any] | None = None,
) -> str:
    gaps = []
    if x_plan_blocker:
        gaps.append(
            f"X PLAN BLOCKER [{x_plan_blocker.get('error_class')}]: "
            f"{x_plan_blocker.get('upgrade_hint')}"
        )
    elif x_cost_cap_blocker:
        gaps.append(
            f"X COST CAP [{x_cost_cap_blocker.get('error_class')}]: "
            f"{x_cost_cap_blocker.get('hint')}"
        )
    elif not avail.get("x_credentials"):
        gaps.append("X: set TWITTER_BEARER_TOKEN or X_BEARER_TOKEN")
    elif per_source.get("x", {}).get("assets", 0) == 0 and not per_source.get("x", {}).get(
        "plan_blocker"
    ) and not per_source.get("x", {}).get("cost_cap_blocker"):
        gaps.append("X: credentials present but no rows this run (query/rate/empty)")
    if per_source.get("reddit", {}).get("assets", 0) == 0:
        gaps.append(
            "Reddit: public JSON often blocked from datacenter IPs; "
            "set REDDIT_CLIENT_ID/SECRET (+ optional PRAW) or run locally"
        )
    if per_source.get("trends", {}).get("assets", 0) == 0:
        gaps.append("Trends: check rate limits / keyword mapping")
    san = per_source.get("santiment") or {}
    if not avail.get("santiment_credentials"):
        gaps.append("Santiment: set SANTIMENT_API_KEY (env or box-secrets)")
    elif san.get("assets", 0) == 0 and not san.get("error"):
        gaps.append(
            "Santiment: credentials present but no rows "
            "(slug map / plan window / call budget)"
        )
    gaps.append("Paid social extras: LunarCrush TBD")
    return "; ".join(gaps)


def write_coverage_report(report: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def coverage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CMRAM social / N coverage (V0.1)",
        "",
        "Attention proxies only — **not alpha**.",
        "",
        f"- Assets requested: **{report.get('n_assets_requested')}**",
        f"- Assets with any N source: **{report.get('n_assets_with_any_source')}**",
        f"- social_daily rows: **{report.get('rows_total')}**",
        "",
        "## Source flags",
        "",
    ]
    flags = report.get("source_flags") or {}
    for k, v in flags.items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Per source", ""]
    for src, st in (report.get("per_source") or {}).items():
        lines.append(
            f"- **{src}**: assets={st.get('assets')} rows={st.get('rows')} "
            f"range={st.get('date_min')}→{st.get('date_max')}"
            + (f" error={st.get('error')}" if st.get("error") else "")
        )
    blocker = report.get("x_plan_blocker")
    if blocker:
        lines += [
            "",
            "## X API plan blocker",
            "",
            f"- **error_class**: `{blocker.get('error_class')}`",
            f"- **status_code**: `{blocker.get('status_code')}`",
            f"- **message**: {blocker.get('message')}",
            f"- **upgrade**: {blocker.get('upgrade_hint')}",
        ]
        if blocker.get("body_excerpt"):
            lines.append(f"- **body_excerpt**: `{blocker.get('body_excerpt')}`")
    cost_block = report.get("x_cost_cap_blocker")
    estimate = report.get("x_cost_estimate") or (
        ((report.get("per_source") or {}).get("x") or {}).get("cost_estimate")
    )
    if estimate:
        lines += [
            "",
            "## X cost estimate (no live spend in this report unless rows landed)",
            "",
            f"- **mode**: `{estimate.get('mode')}`",
            f"- **assets**: {estimate.get('n_assets')}",
            f"- **requests**: {estimate.get('n_requests')} "
            f"({estimate.get('requests_per_asset')}/asset)",
            f"- **price/request (config)**: ${estimate.get('cost_per_request_usd')}",
            f"- **estimated**: ${estimate.get('estimated_cost_usd')}",
            f"- **cap**: ${estimate.get('cost_cap_usd')}",
            f"- **note**: {estimate.get('cost_note')}",
        ]
    if cost_block:
        lines += [
            "",
            "## X cost cap (no HTTP calls made)",
            "",
            f"- **error_class**: `{cost_block.get('error_class')}`",
            f"- **estimated**: ${cost_block.get('estimated_cost_usd')}",
            f"- **cap**: ${cost_block.get('cost_cap_usd')}",
            f"- **hint**: {cost_block.get('hint')}",
        ]
    lines += ["", "## Assets with N", ""]
    for a in report.get("assets_with_n") or []:
        lines.append(f"- `{a}`")
    missing = report.get("assets_missing_n") or []
    lines += ["", f"## Assets missing N ({len(missing)})", ""]
    for a in missing[:50]:
        lines.append(f"- `{a}`")
    if len(missing) > 50:
        lines.append(f"- … +{len(missing) - 50} more")
    lines += ["", "## Next gap", "", str(report.get("next_gap") or ""), ""]
    return "\n".join(lines) + "\n"
