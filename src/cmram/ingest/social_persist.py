"""Persist social_daily rows into DuckDB (+ optional parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from cmram.db.schema import init_db, migrate_schema
from cmram.ingest.social import (
    SOCIAL_COLUMNS,
    build_coverage_report,
    ingest_social,
    source_availability,
)


def write_social_daily(
    conn: duckdb.DuckDBPyConnection,
    social: pd.DataFrame,
    *,
    replace: bool = False,
    replace_sources: list[str] | None = None,
) -> int:
    """Upsert-ish write. ``replace`` clears all; else optional per-source delete."""
    migrate_schema(conn)
    if social is None or len(social) == 0:
        if replace:
            conn.execute("DELETE FROM social_daily")
        return 0
    df = social.copy()
    for col in SOCIAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(SOCIAL_COLUMNS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    df["asset_id"] = df["asset_id"].astype(str)
    df["source"] = df["source"].astype(str)

    if replace:
        conn.execute("DELETE FROM social_daily")
    elif replace_sources:
        for src in replace_sources:
            conn.execute("DELETE FROM social_daily WHERE source = ?", [src])

    conn.register("_social_df", df)
    try:
        conn.execute(
            """
            INSERT INTO social_daily
            SELECT timestamp, asset_id, source, mentions, unique_users,
                   engagement, community_growth, raw_payload_ref
            FROM _social_df
            """
        )
    finally:
        conn.unregister("_social_df")
    return len(df)


def load_social_daily(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT timestamp, asset_id, source, mentions, unique_users,
               engagement, community_growth, raw_payload_ref
        FROM social_daily
        ORDER BY timestamp, asset_id, source
        """
    ).df()


def load_eligible_assets(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Prefer universe_eligible; fall back to distinct membership assets."""
    cols = {r[0] for r in conn.execute("DESCRIBE assets").fetchall()}
    if "universe_eligible" in cols:
        df = conn.execute(
            """
            SELECT asset_id, symbol, name
            FROM assets
            WHERE coalesce(universe_eligible, true)
            ORDER BY asset_id
            """
        ).df()
        if len(df):
            return df
    return conn.execute(
        """
        SELECT DISTINCT a.asset_id, a.symbol, a.name
        FROM assets a
        INNER JOIN universe_membership u ON a.asset_id = u.asset_id
        ORDER BY a.asset_id
        """
    ).df()


def coverage_from_db(
    conn: duckdb.DuckDBPyConnection,
    *,
    config: dict[str, Any] | None = None,
    ingest_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild full N coverage from all social_daily rows currently in DuckDB."""
    assets = load_eligible_assets(conn)
    social = load_social_daily(conn)
    avail = source_availability(config)
    per_source: dict[str, Any] = {}
    if len(social):
        for src, g in social.groupby("source"):
            per_source[str(src)] = {
                "assets": int(g["asset_id"].nunique()),
                "rows": int(len(g)),
                "date_min": str(pd.to_datetime(g["timestamp"]).min().date()),
                "date_max": str(pd.to_datetime(g["timestamp"]).max().date()),
                "asset_ids": sorted(g["asset_id"].astype(str).unique()),
            }
    # Preserve per-source run notes (credentials, plan/budget blockers, etc.)
    if ingest_report:
        for src in ("trends", "reddit", "wikipedia", "santiment", "x"):
            run_st = (ingest_report.get("per_source") or {}).get(src) or {}
            if not run_st:
                continue
            if src not in per_source:
                per_source[src] = {
                    "assets": 0,
                    "rows": 0,
                    "date_min": None,
                    "date_max": None,
                }
            for k in (
                "credentials",
                "error",
                "calls_used",
                "call_budget",
                "skipped",
                "plan",
                "coverage",
                "stopped_reason",
                "ingested",
                "plan_blocker",
                "cost_cap_blocker",
                "cost_estimate",
                "mode",
            ):
                if k in run_st and k not in per_source[src]:
                    per_source[src][k] = run_st[k]
            if run_st.get("error") and "error" not in per_source[src]:
                per_source[src]["error"] = run_st["error"]

    report = build_coverage_report(
        social,
        assets,
        avail=avail,
        per_source=per_source,
        x_plan_blocker=(ingest_report or {}).get("x_plan_blocker"),
        x_cost_cap_blocker=(ingest_report or {}).get("x_cost_cap_blocker"),
    )
    if ingest_report and ingest_report.get("x_cost_estimate"):
        report["x_cost_estimate"] = ingest_report["x_cost_estimate"]
    return report


def build_and_write_social(
    duckdb_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    parquet_path: str | Path | None = None,
    sources: list[str] | None = None,
    replace: bool | None = None,
    x_max_assets: int | None = None,
    asset_limit: int | None = None,
    x_mode: str | None = None,
    x_allow_over_cap: bool = False,
    x_cost_cap_usd: float | None = None,
    santiment_call_budget: int | None = None,
    santiment_max_assets: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest social sources and write DuckDB (+ optional parquet).

    When ``sources`` is a partial list, default ``replace=False`` and only
    delete/replace those source names so Trends/Wikipedia are not wiped by an
    X-only refresh.
    """
    path = Path(duckdb_path)
    conn = init_db(path)
    try:
        assets = load_eligible_assets(conn)
        if asset_limit is not None:
            assets = assets.head(int(asset_limit))

        partial = sources is not None
        do_replace = (not partial) if replace is None else bool(replace)
        replace_sources = list(sources) if (partial and not do_replace) else None

        social, report = ingest_social(
            assets,
            config=config,
            sources=sources,
            x_max_assets=x_max_assets,
            x_mode=x_mode,
            x_allow_over_cap=x_allow_over_cap,
            x_cost_cap_usd=x_cost_cap_usd,
            santiment_call_budget=santiment_call_budget,
            santiment_max_assets=santiment_max_assets,
        )
        write_social_daily(
            conn,
            social,
            replace=do_replace,
            replace_sources=replace_sources,
        )
        # Full coverage from DB (includes prior Trends/wiki when partial X write)
        full_report = coverage_from_db(conn, config=config, ingest_report=report)
        all_social = load_social_daily(conn)

        if parquet_path is not None and len(all_social):
            p = Path(parquet_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            all_social.to_parquet(p, index=False)

        full_report["n_assets_requested"] = int(len(load_eligible_assets(conn)))
        full_report["ingest_batch_assets"] = int(len(assets))
        full_report["ingest_batch_rows"] = int(len(social))
        full_report["replace"] = do_replace
        full_report["replace_sources"] = replace_sources
        return all_social, full_report
    finally:
        conn.close()
