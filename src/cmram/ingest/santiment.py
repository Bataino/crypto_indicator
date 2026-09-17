"""Santiment GraphQL client — historical social_volume_total → social_daily.

Free-plan notes (document, do not claim alpha):
- ~1000 API calls/month typical free-tier budget (treat as soft; leave headroom).
- Restricted metrics often allow ~1y history with ~30d realtime lag
  (restrictedFrom / restrictedTo from getMetric metadata).
- Auth: ``Authorization: Apikey <key>``. Never log the key (bool + length only).

Attention proxy only — not alpha.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Callable

import httpx
import pandas as pd

from cmram.secrets_env import ensure_env_secrets, secret_status

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.santiment.net/graphql"
DEFAULT_METRIC = "social_volume_total"
SOURCE_NAME = "santiment"

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


class SantimentError(Exception):
    """Base Santiment client error."""


class SantimentAuthError(SantimentError):
    """Missing or rejected API key."""


class SantimentPlanError(SantimentError):
    """Plan / access restriction (date window, inaccessible metric)."""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.detail = detail or {}


class SantimentCallBudgetError(SantimentError):
    """Session call budget exhausted — stop before further HTTP."""

    def __init__(self, message: str, *, calls_used: int, call_budget: int):
        super().__init__(message)
        self.calls_used = calls_used
        self.call_budget = call_budget


def santiment_credentials_available() -> bool:
    """True if SANTIMENT_API_KEY is present (loads from box-secrets if needed)."""
    ensure_env_secrets(("SANTIMENT_API_KEY",))
    return bool((os.environ.get("SANTIMENT_API_KEY") or "").strip())


def _api_key() -> str:
    ensure_env_secrets(("SANTIMENT_API_KEY",))
    key = (os.environ.get("SANTIMENT_API_KEY") or "").strip()
    if not key:
        raise SantimentAuthError(
            "SANTIMENT_API_KEY missing (env or box-secrets card). "
            f"status={secret_status('SANTIMENT_API_KEY')}"
        )
    return key


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SOCIAL_COLUMNS))


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    # 2025-09-17T14:51:39Z or date-only
    try:
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


class SantimentClient:
    """Thin GraphQL client with call counter and safe logging."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        call_budget: int = 200,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        url: str = GRAPHQL_URL,
    ):
        self._owns_client = client is None
        self._key = (api_key if api_key is not None else _api_key()).strip()
        self.call_budget = int(call_budget)
        self.calls_used = 0
        self.url = url
        self._client = client or httpx.Client(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Apikey {self._key}",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
        st = secret_status("SANTIMENT_API_KEY")
        logger.info(
            "Santiment client ready: key_present=%s key_length=%s call_budget=%s",
            st["present"] or bool(self._key),
            st["length"] or len(self._key),
            self.call_budget,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SantimentClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.calls_used >= self.call_budget:
            raise SantimentCallBudgetError(
                f"Santiment call budget exhausted ({self.calls_used}/{self.call_budget})",
                calls_used=self.calls_used,
                call_budget=self.call_budget,
            )
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        self.calls_used += 1
        r = self._client.post(self.url, json=payload)
        if r.status_code in (401, 403):
            raise SantimentAuthError(
                f"Santiment HTTP {r.status_code} (auth/plan). "
                f"key_status={secret_status('SANTIMENT_API_KEY')}"
            )
        if r.status_code >= 400:
            raise SantimentError(f"Santiment HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        errors = body.get("errors") or []
        if errors:
            msg = str(errors[0].get("message") or errors[0])
            low = msg.lower()
            if "subscription" in low or "allowed interval" in low or "upgrade" in low:
                raise SantimentPlanError(msg, detail={"errors": errors})
            raise SantimentError(msg)
        return body.get("data") or {}

    def metric_metadata(self, metric: str = DEFAULT_METRIC) -> dict[str, Any]:
        q = """
        query ($metric: String!) {
          getMetric(metric: $metric) {
            metadata {
              humanReadableName
              isAccessible
              isRestricted
              restrictedFrom
              restrictedTo
              availableSlugs
            }
          }
        }
        """
        data = self.graphql(q, {"metric": metric})
        meta = ((data.get("getMetric") or {}).get("metadata")) or {}
        return meta

    def all_projects(self) -> list[dict[str, Any]]:
        q = "{ allProjects { slug name ticker } }"
        data = self.graphql(q)
        return list(data.get("allProjects") or [])

    def social_volume_timeseries(
        self,
        slug: str,
        *,
        from_dt: date | datetime | str,
        to_dt: date | datetime | str,
        metric: str = DEFAULT_METRIC,
        interval: str = "1d",
    ) -> list[dict[str, Any]]:
        def _fmt(v: date | datetime | str) -> str:
            if isinstance(v, datetime):
                if v.tzinfo is None:
                    v = v.replace(tzinfo=timezone.utc)
                return v.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if isinstance(v, date):
                return f"{v.isoformat()}T00:00:00Z"
            return str(v)

        q = """
        query (
          $metric: String!
          $slug: String!
          $from: DateTime!
          $to: DateTime!
          $interval: interval!
        ) {
          getMetric(metric: $metric) {
            timeseriesData(
              slug: $slug
              from: $from
              to: $to
              interval: $interval
            ) {
              datetime
              value
            }
          }
        }
        """
        data = self.graphql(
            q,
            {
                "metric": metric,
                "slug": slug,
                "from": _fmt(from_dt),
                "to": _fmt(to_dt),
                "interval": interval,
            },
        )
        return list(((data.get("getMetric") or {}).get("timeseriesData")) or [])


def resolve_slug(
    asset: dict[str, Any] | pd.Series,
    *,
    available_slugs: set[str],
    projects: list[dict[str, Any]] | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """Map CoinGecko asset_id / symbol / name → Santiment slug.

    Returns (slug_or_None, reason) where reason is one of:
    override | asset_id | ticker | name | slug_missing
    """
    aid = str(asset.get("asset_id") or "").strip()
    sym = str(asset.get("symbol") or "").strip().upper()
    name = str(asset.get("name") or "").strip().lower()
    ov = dict(overrides or {})

    if aid in ov:
        cand = ov[aid].strip()
        if cand in available_slugs:
            return cand, "override"
        return None, "slug_missing_override"

    if aid and aid in available_slugs:
        return aid, "asset_id"

    projects = projects or []
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for p in projects:
        slug = (p.get("slug") or "").strip()
        if slug not in available_slugs:
            continue
        t = (p.get("ticker") or "").strip().upper()
        n = (p.get("name") or "").strip().lower()
        if t:
            by_ticker.setdefault(t, []).append(p)
        if n:
            by_name.setdefault(n, []).append(p)

    if sym and sym in by_ticker and len(by_ticker[sym]) == 1:
        return by_ticker[sym][0]["slug"], "ticker"

    if name and name in by_name and len(by_name[name]) == 1:
        return by_name[name][0]["slug"], "name"

    return None, "slug_missing"


def free_plan_window(meta: dict[str, Any]) -> tuple[date | None, date | None, dict[str, Any]]:
    """Extract restrictedFrom/To as dates + note dict for coverage docs."""
    rf = _parse_iso_date(meta.get("restrictedFrom"))
    rt = _parse_iso_date(meta.get("restrictedTo"))
    note = {
        "metric": DEFAULT_METRIC,
        "is_accessible": meta.get("isAccessible"),
        "is_restricted": meta.get("isRestricted"),
        "restricted_from": meta.get("restrictedFrom"),
        "restricted_to": meta.get("restrictedTo"),
        "free_plan_note": (
            "Santiment FREE / restricted metrics: typically ~1 year of history "
            "with ~30 days realtime lag (restrictedTo often ~30d before today). "
            "~1000 API calls/month — leave headroom. Not alpha."
        ),
    }
    return rf, rt, note


def probe_coverage(
    assets: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    client: SantimentClient | None = None,
    call_budget: int | None = None,
) -> dict[str, Any]:
    """Map eligible assets → Santiment slugs; report coverage without timeseries pulls.

    Uses 1–2 GraphQL calls (metric metadata + optional allProjects).
    """
    cfg = dict((config or {}).get("santiment") or config or {})
    metric = str(cfg.get("metric") or DEFAULT_METRIC)
    overrides = dict(cfg.get("slug_overrides") or {})
    budget = int(call_budget if call_budget is not None else cfg.get("call_budget", 200))

    owns = client is None
    client = client or SantimentClient(call_budget=budget)
    try:
        if assets is None or len(assets) == 0:
            return {
                "n_assets": 0,
                "n_covered": 0,
                "covered": [],
                "missing": [],
                "plan": {},
                "calls_used": client.calls_used,
            }

        meta = client.metric_metadata(metric)
        available = set(meta.get("availableSlugs") or [])
        rf, rt, plan_note = free_plan_window(meta)
        if meta.get("isAccessible") is False:
            raise SantimentPlanError(
                f"Metric {metric} not accessible on current plan",
                detail=plan_note,
            )

        need_projects = False
        for _, row in assets.iterrows():
            aid = str(row["asset_id"])
            if aid in overrides and overrides[aid] in available:
                continue
            if aid in available:
                continue
            need_projects = True
            break

        projects: list[dict[str, Any]] = []
        if need_projects:
            projects = client.all_projects()

        covered: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for _, row in assets.iterrows():
            slug, reason = resolve_slug(
                row, available_slugs=available, projects=projects, overrides=overrides
            )
            entry = {
                "asset_id": str(row["asset_id"]),
                "symbol": str(row.get("symbol") or ""),
                "name": str(row.get("name") or ""),
                "slug": slug,
                "map_reason": reason,
            }
            if slug:
                covered.append(entry)
            else:
                missing.append(entry)

        return {
            "metric": metric,
            "n_assets": int(len(assets)),
            "n_covered": len(covered),
            "n_missing": len(missing),
            "covered": covered,
            "missing": missing,
            "plan": plan_note,
            "window_from": rf.isoformat() if rf else None,
            "window_to": rt.isoformat() if rt else None,
            "n_available_slugs": len(available),
            "calls_used": client.calls_used,
            "note": "Attention proxy only — not alpha. Free-plan lag/history apply.",
        }
    finally:
        if owns:
            client.close()


def fetch_social_volume_series(
    slug: str,
    *,
    from_dt: date,
    to_dt: date,
    metric: str = DEFAULT_METRIC,
    client: SantimentClient | None = None,
    graphql_fn: Callable[..., dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Fetch daily social volume for one Santiment slug → timestamp/mentions."""
    if graphql_fn is not None:
        # Test hook: graphql_fn returns timeseries list or full data dict
        raw = graphql_fn(slug=slug, from_dt=from_dt, to_dt=to_dt, metric=metric)
        if isinstance(raw, list):
            points = raw
        else:
            points = list(
                (((raw or {}).get("getMetric") or {}).get("timeseriesData")) or []
            )
    else:
        owns = client is None
        client = client or SantimentClient()
        try:
            points = client.social_volume_timeseries(
                slug, from_dt=from_dt, to_dt=to_dt, metric=metric
            )
        finally:
            if owns:
                client.close()

    rows = []
    for pt in points:
        dt = _parse_iso_date(pt.get("datetime"))
        if dt is None:
            continue
        val = pt.get("value")
        if val is None:
            continue
        rows.append({"timestamp": dt, "mentions": float(val)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp", "mentions"])


def fetch_santiment_for_assets(
    assets: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    sleep_s: float | None = None,
    call_budget: int | None = None,
    max_assets: int | None = None,
    client: SantimentClient | None = None,
    coverage: dict[str, Any] | None = None,
    fetch_fn: Callable[..., pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest social_volume_total for mapped assets within call budget.

    Returns (social_daily frame with source=santiment, ingest report).
    """
    cfg = dict((config or {}).get("santiment") or {})
    # Allow passing full social.yaml (sources.santiment) or flat
    if not cfg and config and "metric" in (config or {}):
        cfg = dict(config)
    metric = str(cfg.get("metric") or DEFAULT_METRIC)
    sleep = float(sleep_s if sleep_s is not None else cfg.get("sleep_s", 0.35))
    budget = int(call_budget if call_budget is not None else cfg.get("call_budget", 150))
    max_a = max_assets if max_assets is not None else cfg.get("max_assets")
    overrides = dict(cfg.get("slug_overrides") or {})

    if assets is None or assets.empty:
        return _empty(), {
            "assets": 0,
            "rows": 0,
            "calls_used": 0,
            "skipped": [],
            "reason": "no_assets",
        }

    owns = client is None
    client = client or SantimentClient(call_budget=budget)
    report: dict[str, Any] = {
        "metric": metric,
        "assets": 0,
        "rows": 0,
        "skipped": [],
        "ingested": [],
        "plan": {},
        "calls_used": 0,
        "call_budget": budget,
        "stopped_reason": None,
    }
    try:
        cov = coverage
        if cov is None:
            cov = probe_coverage(assets, config={"santiment": cfg}, client=client)
        report["coverage"] = {
            "n_covered": cov.get("n_covered"),
            "n_missing": cov.get("n_missing"),
            "window_from": cov.get("window_from"),
            "window_to": cov.get("window_to"),
        }
        report["plan"] = cov.get("plan") or {}

        rf = _parse_iso_date(cov.get("window_from") or (cov.get("plan") or {}).get("restricted_from"))
        rt = _parse_iso_date(cov.get("window_to") or (cov.get("plan") or {}).get("restricted_to"))
        if rf is None or rt is None:
            # Fallback: ~1y ago to ~30d ago if metadata missing
            today = date.today()
            rt = rt or date.fromordinal(today.toordinal() - 30)
            rf = rf or date(rt.year - 1, rt.month, rt.day)
            report["plan_window_fallback"] = True

        covered = list(cov.get("covered") or [])
        for m in cov.get("missing") or []:
            report["skipped"].append(
                {
                    "asset_id": m.get("asset_id"),
                    "reason": m.get("map_reason") or "slug_missing",
                    "slug": None,
                }
            )

        if max_a is not None:
            covered = covered[: int(max_a)]

        frames: list[pd.DataFrame] = []
        for i, item in enumerate(covered):
            aid = str(item["asset_id"])
            slug = str(item["slug"])
            remaining = budget - client.calls_used
            if remaining <= 0:
                report["stopped_reason"] = "call_budget"
                report["skipped"].append(
                    {"asset_id": aid, "reason": "call_budget", "slug": slug}
                )
                # mark rest as budget-skipped
                for rest in covered[i + 1 :]:
                    report["skipped"].append(
                        {
                            "asset_id": rest["asset_id"],
                            "reason": "call_budget",
                            "slug": rest.get("slug"),
                        }
                    )
                break

            logger.info(
                "Santiment [%d/%d] %s → slug=%s window=%s→%s calls=%s/%s",
                i + 1,
                len(covered),
                aid,
                slug,
                rf,
                rt,
                client.calls_used,
                budget,
            )
            try:
                if fetch_fn is not None:
                    series = fetch_fn(slug, from_dt=rf, to_dt=rt, metric=metric)
                else:
                    series = fetch_social_volume_series(
                        slug, from_dt=rf, to_dt=rt, metric=metric, client=client
                    )
            except SantimentCallBudgetError:
                report["stopped_reason"] = "call_budget"
                report["skipped"].append(
                    {"asset_id": aid, "reason": "call_budget", "slug": slug}
                )
                for rest in covered[i + 1 :]:
                    report["skipped"].append(
                        {
                            "asset_id": rest["asset_id"],
                            "reason": "call_budget",
                            "slug": rest.get("slug"),
                        }
                    )
                break
            except SantimentPlanError as exc:
                logger.warning("Santiment plan restriction for %s: %s", aid, exc)
                report["skipped"].append(
                    {
                        "asset_id": aid,
                        "reason": "plan_restriction",
                        "slug": slug,
                        "detail": str(exc)[:240],
                    }
                )
                if sleep > 0:
                    time.sleep(sleep)
                continue
            except SantimentError as exc:
                logger.warning("Santiment fetch failed for %s (%s): %s", aid, slug, exc)
                report["skipped"].append(
                    {
                        "asset_id": aid,
                        "reason": "fetch_error",
                        "slug": slug,
                        "detail": str(exc)[:240],
                    }
                )
                if sleep > 0:
                    time.sleep(sleep)
                continue

            if series is None or series.empty:
                report["skipped"].append(
                    {"asset_id": aid, "reason": "empty_series", "slug": slug}
                )
            else:
                part = series.copy()
                part["asset_id"] = aid
                part["source"] = SOURCE_NAME
                part["unique_users"] = pd.NA
                part["engagement"] = pd.NA
                part["community_growth"] = pd.NA
                part["raw_payload_ref"] = f"santiment:{metric}:{slug}"
                frames.append(part)
                report["ingested"].append(
                    {
                        "asset_id": aid,
                        "slug": slug,
                        "rows": int(len(part)),
                        "map_reason": item.get("map_reason"),
                    }
                )

            if sleep > 0:
                time.sleep(sleep)

        out = pd.concat(frames, ignore_index=True) if frames else _empty()
        if len(out):
            out = out[list(SOCIAL_COLUMNS)]
        report["assets"] = int(out["asset_id"].nunique()) if len(out) else 0
        report["rows"] = int(len(out))
        report["calls_used"] = client.calls_used
        report["date_min"] = (
            str(pd.to_datetime(out["timestamp"]).min().date()) if len(out) else None
        )
        report["date_max"] = (
            str(pd.to_datetime(out["timestamp"]).max().date()) if len(out) else None
        )
        report["asset_ids"] = (
            sorted(out["asset_id"].astype(str).unique()) if len(out) else []
        )
        return out, report
    finally:
        report["calls_used"] = client.calls_used
        if owns:
            client.close()


def coverage_markdown_santiment(report: dict[str, Any]) -> str:
    """Plain-language Santiment coverage / free-plan note for Abdul."""
    cov = report.get("coverage") or report
    plan = report.get("plan") or cov.get("plan") or {}
    lines = [
        "# Santiment social volume coverage",
        "",
        "Attention proxy only — **not alpha**.",
        "",
        f"- Metric: `{report.get('metric') or cov.get('metric') or DEFAULT_METRIC}`",
        f"- Eligible assets requested: **{cov.get('n_assets') or report.get('n_assets')}**",
        f"- Slug-mapped (covered): **{cov.get('n_covered')}**",
        f"- Missing slug: **{cov.get('n_missing')}**",
        f"- Ingested this run: assets={report.get('assets')} rows={report.get('rows')}",
        f"- API calls used: **{report.get('calls_used')}** / budget **{report.get('call_budget')}**",
        "",
        "## Free-plan limits",
        "",
        str(plan.get("free_plan_note") or ""),
        "",
        f"- restrictedFrom: `{plan.get('restricted_from')}`",
        f"- restrictedTo: `{plan.get('restricted_to')}` "
        f"(~30d lag — no last-month realtime on FREE for restricted metrics)",
        f"- Window used: `{cov.get('window_from')}` → `{cov.get('window_to')}`",
        "",
        "## Covered (asset_id → slug)",
        "",
    ]
    for c in (cov.get("covered") or report.get("ingested") or [])[:80]:
        lines.append(
            f"- `{c.get('asset_id')}` → `{c.get('slug')}` "
            f"({c.get('map_reason') or c.get('reason') or 'ok'})"
        )
    missing = cov.get("missing") or [
        s for s in (report.get("skipped") or []) if s.get("reason") == "slug_missing"
    ]
    lines += ["", f"## Missing / skipped ({len(missing)}+)", ""]
    for m in (report.get("skipped") or missing)[:80]:
        lines.append(
            f"- `{m.get('asset_id')}`: {m.get('reason')}"
            + (f" slug=`{m.get('slug')}`" if m.get("slug") else "")
        )
    if report.get("stopped_reason"):
        lines += ["", f"**Stopped:** `{report.get('stopped_reason')}`", ""]
    lines += ["", "No alpha claims. Combine with Trends/wiki in Model C `score_N`.", ""]
    return "\n".join(lines) + "\n"
