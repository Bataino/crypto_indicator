"""X (Twitter) attention proxy — default cheap Recent Counts.

Narrative N only needs mention volume. Default ingest is
``GET /2/tweets/counts/recent`` (granularity=day, last 7 days, one request
per asset/query). Actual tweet search is opt-in and expensive.

Requires TWITTER_BEARER_TOKEN or X_BEARER_TOKEN. If missing, returns empty
frames and documents the gap — do not block Trends/Reddit on X.
Never prints the bearer token.

Pay-per-use: ``cost_per_request_usd`` is a documented estimate
(Counts: Recent listed ~$0.005/request). The live X developer console price
always wins if it differs — update the config; do not hard-code spend.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import httpx
import pandas as pd

from cmram.secrets_env import ensure_env_secrets

logger = logging.getLogger(__name__)

X_RECENT_COUNTS = "https://api.x.com/2/tweets/counts/recent"
TWITTER_RECENT_COUNTS = "https://api.twitter.com/2/tweets/counts/recent"
X_RECENT_SEARCH = "https://api.x.com/2/tweets/search/recent"
TWITTER_RECENT_SEARCH = "https://api.twitter.com/2/tweets/search/recent"

DEFAULT_MODE = "counts"
DEFAULT_COST_PER_REQUEST_USD = 0.005
DEFAULT_COST_CAP_USD = 1.00
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_SEARCH_MAX_PAGES = 1
DEFAULT_SEARCH_MAX_RESULTS = 10  # X recent-search minimum
COUNTS_REQUESTS_PER_ASSET = 1

# HTTP statuses that mean the developer plan cannot use this endpoint.
_PLAN_LIMIT_STATUSES = frozenset({402, 403})

_COST_NOTE = (
    "cost_per_request_usd is configurable. Counts: Recent was listed around "
    "$0.005/request; the X developer console price always wins if different."
)


class XPlanLimitError(Exception):
    """X API plan / access limit — stop the batch; do not retry forever."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_class: str,
        upgrade_hint: str,
        body_excerpt: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_class = error_class
        self.upgrade_hint = upgrade_hint
        self.body_excerpt = body_excerpt

    def as_report(self) -> dict[str, Any]:
        return {
            "error_class": self.error_class,
            "status_code": self.status_code,
            "message": str(self),
            "upgrade_hint": self.upgrade_hint,
            "body_excerpt": self.body_excerpt,
        }


class XCostCapError(Exception):
    """Estimated X spend exceeds configured cap — no HTTP calls made."""

    def __init__(
        self,
        message: str,
        *,
        estimated_cost_usd: float,
        cost_cap_usd: float,
        n_assets: int,
        n_requests: int,
        mode: str,
        hint: str,
        plan: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.estimated_cost_usd = float(estimated_cost_usd)
        self.cost_cap_usd = float(cost_cap_usd)
        self.n_assets = int(n_assets)
        self.n_requests = int(n_requests)
        self.mode = mode
        self.hint = hint
        self.plan = plan or {}
        self.error_class = "x_cost_cap_exceeded"

    def as_report(self) -> dict[str, Any]:
        return {
            "error_class": self.error_class,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_cap_usd": self.cost_cap_usd,
            "n_assets": self.n_assets,
            "n_requests": self.n_requests,
            "mode": self.mode,
            "message": str(self),
            "hint": self.hint,
            "plan": self.plan,
        }


@dataclass(frozen=True)
class XBatchPlan:
    """Pre-flight spend plan. Built before any live X HTTP call."""

    mode: str
    n_assets: int
    requests_per_asset: int
    n_requests: int
    cost_per_request_usd: float
    estimated_cost_usd: float
    cost_cap_usd: float
    allow_over_cap: bool
    over_cap: bool
    lookback_days: int
    max_pages: int
    max_results: int
    granularity: str
    sleep_s: float
    query_template: str | None
    max_assets: int | None
    cost_note: str = _COST_NOTE

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_assets": self.n_assets,
            "requests_per_asset": self.requests_per_asset,
            "n_requests": self.n_requests,
            "cost_per_request_usd": self.cost_per_request_usd,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_cap_usd": self.cost_cap_usd,
            "allow_over_cap": self.allow_over_cap,
            "over_cap": self.over_cap,
            "lookback_days": self.lookback_days,
            "max_pages": self.max_pages,
            "max_results": self.max_results,
            "granularity": self.granularity,
            "sleep_s": self.sleep_s,
            "max_assets": self.max_assets,
            "cost_note": self.cost_note,
        }


def _ensure_bearer_env() -> None:
    """Load X bearer from box-secrets into os.environ if process env empty."""
    ensure_env_secrets(("X_BEARER_TOKEN", "TWITTER_BEARER_TOKEN"))


def x_bearer_token() -> str | None:
    """Return bearer from env if set (stripped); never log the value."""
    _ensure_bearer_env()
    for key in ("TWITTER_BEARER_TOKEN", "X_BEARER_TOKEN", "TWITTER_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def x_credentials_available() -> bool:
    return x_bearer_token() is not None


def default_query(asset: dict[str, Any], *, template: str | None = None) -> str:
    symbol = str(asset.get("symbol") or "").strip()
    name = str(asset.get("name") or symbol or "").strip()
    asset_id = str(asset.get("asset_id") or "").strip()
    tmpl = template or "({name} OR ${symbol}) (crypto OR coin OR token) -is:retweet"
    try:
        return tmpl.format(symbol=symbol, name=name, asset_id=asset_id).strip()
    except KeyError:
        return f"({name} OR ${symbol}) crypto -is:retweet"


def _x_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = config or {}
    if "x" in cfg and isinstance(cfg.get("x"), dict):
        return dict(cfg["x"])
    return dict(cfg)


def _as_float(val: Any, default: float) -> float:
    if val is None or val == "":
        return float(default)
    return float(val)


def _as_int(val: Any, default: int) -> int:
    if val is None or val == "":
        return int(default)
    return int(val)


def estimate_x_batch_cost(
    n_assets: int,
    *,
    mode: str = DEFAULT_MODE,
    requests_per_asset: int | None = None,
    cost_per_request_usd: float = DEFAULT_COST_PER_REQUEST_USD,
    max_pages: int = DEFAULT_SEARCH_MAX_PAGES,
) -> dict[str, Any]:
    """estimated_cost = n_assets × expected_requests_per_asset × price.

    Counts mode: 1 request per asset. Search mode: ``max_pages`` per asset
    (default 1). Price is whatever is configured — console wins.
    """
    mode_n = (mode or DEFAULT_MODE).strip().lower()
    if requests_per_asset is None:
        requests_per_asset = (
            COUNTS_REQUESTS_PER_ASSET if mode_n != "search" else max(1, int(max_pages))
        )
    n_assets_i = max(0, int(n_assets))
    rpa = max(0, int(requests_per_asset))
    price = float(cost_per_request_usd)
    n_requests = n_assets_i * rpa
    estimated = round(n_assets_i * rpa * price, 6)
    return {
        "mode": mode_n,
        "n_assets": n_assets_i,
        "requests_per_asset": rpa,
        "n_requests": n_requests,
        "cost_per_request_usd": price,
        "estimated_cost_usd": estimated,
        "cost_note": _COST_NOTE,
    }


def plan_x_batch(
    assets: pd.DataFrame | None,
    *,
    config: dict[str, Any] | None = None,
    max_assets: int | None = None,
    mode: str | None = None,
    allow_over_cap: bool = False,
    cost_cap_usd: float | None = None,
) -> tuple[XBatchPlan, pd.DataFrame]:
    """Resolve mode/caps and estimate spend. Does not touch the network."""
    cfg = _x_cfg(config)
    search_cfg = dict(cfg.get("search") or {})

    mode_n = str(mode or cfg.get("mode") or DEFAULT_MODE).strip().lower()
    if mode_n not in {"counts", "search"}:
        raise ValueError("x.mode must be 'counts' or 'search', got %r" % (mode_n,))

    lookback = min(_as_int(cfg.get("lookback_days"), DEFAULT_LOOKBACK_DAYS), 7)
    sleep_s = _as_float(cfg.get("sleep_s"), 3.0)
    granularity = str(cfg.get("granularity") or "day").strip().lower() or "day"
    template = cfg.get("query_template")

    max_pages = _as_int(
        search_cfg.get("max_pages", cfg.get("max_pages")), DEFAULT_SEARCH_MAX_PAGES
    )
    max_pages = max(1, min(max_pages, 5))
    max_results = _as_int(
        search_cfg.get("max_results", cfg.get("max_results")),
        DEFAULT_SEARCH_MAX_RESULTS,
    )
    max_results = max(10, min(int(max_results), 100))

    if mode_n == "search":
        price = _as_float(
            search_cfg.get("cost_per_request_usd", cfg.get("cost_per_request_usd")),
            DEFAULT_COST_PER_REQUEST_USD,
        )
        rpa = max_pages
    else:
        price = _as_float(cfg.get("cost_per_request_usd"), DEFAULT_COST_PER_REQUEST_USD)
        rpa = COUNTS_REQUESTS_PER_ASSET
        max_pages = 1

    cap_usd = (
        float(cost_cap_usd)
        if cost_cap_usd is not None
        else _as_float(cfg.get("estimated_cost_cap_usd"), DEFAULT_COST_CAP_USD)
    )

    cap_assets = max_assets if max_assets is not None else cfg.get("max_assets")
    cap_i = int(cap_assets) if cap_assets is not None and cap_assets != "" else None

    if assets is None or assets.empty:
        work = _empty_assets()
    elif cap_i is not None:
        if cap_i <= 0:
            work = assets.iloc[0:0].copy()
        else:
            work = assets.head(cap_i).copy()
    else:
        work = assets.copy()

    n = int(len(work))
    est = estimate_x_batch_cost(
        n,
        mode=mode_n,
        requests_per_asset=rpa,
        cost_per_request_usd=price,
        max_pages=max_pages,
    )
    estimated = float(est["estimated_cost_usd"])
    over = estimated > cap_usd + 1e-12

    plan = XBatchPlan(
        mode=mode_n,
        n_assets=n,
        requests_per_asset=int(est["requests_per_asset"]),
        n_requests=int(est["n_requests"]),
        cost_per_request_usd=price,
        estimated_cost_usd=estimated,
        cost_cap_usd=cap_usd,
        allow_over_cap=bool(allow_over_cap),
        over_cap=over,
        lookback_days=lookback,
        max_pages=max_pages,
        max_results=max_results,
        granularity=granularity,
        sleep_s=sleep_s,
        query_template=str(template) if template else None,
        max_assets=cap_i,
    )
    return plan, work


def enforce_x_cost_cap(plan: XBatchPlan) -> None:
    """Raise ``XCostCapError`` when estimated spend is above cap (no HTTP)."""
    if plan.over_cap and not plan.allow_over_cap:
        hint = (
            "Abdul: estimated X spend $%.4f exceeds cap $%.2f "
            "(mode=%s, assets=%d, requests=%d × $%.4f). Lower max_assets, "
            "raise estimated_cost_cap_usd, or pass --x-allow-over-cap. %s"
            % (
                plan.estimated_cost_usd,
                plan.cost_cap_usd,
                plan.mode,
                plan.n_assets,
                plan.n_requests,
                plan.cost_per_request_usd,
                _COST_NOTE,
            )
        )
        logger.error(
            "X cost cap: estimated=$%.4f cap=$%.2f mode=%s assets=%d requests=%d "
            "(no HTTP calls)",
            plan.estimated_cost_usd,
            plan.cost_cap_usd,
            plan.mode,
            plan.n_assets,
            plan.n_requests,
        )
        raise XCostCapError(
            "X estimated cost $%.4f exceeds cap $%.2f"
            % (plan.estimated_cost_usd, plan.cost_cap_usd),
            estimated_cost_usd=plan.estimated_cost_usd,
            cost_cap_usd=plan.cost_cap_usd,
            n_assets=plan.n_assets,
            n_requests=plan.n_requests,
            mode=plan.mode,
            hint=hint,
            plan=plan.as_dict(),
        )


def _classify_plan_error(status_code: int, body: str) -> tuple[str, str]:
    """Return (error_class, upgrade_hint) from HTTP status + body text."""
    low = (body or "").lower()
    if "credits" in low and ("deplet" in low or "exhausted" in low or status_code == 402):
        return (
            "x_api_credits_depleted_402",
            "Abdul: X API credits are depleted (HTTP 402, type credits-depleted). "
            "Add $5–$10 credits on the X developer console so Counts: Recent "
            "(/2/tweets/counts/recent) has quota — then re-run "
            "python scripts/run_social_ingest.py --sources x --x-mode counts -v",
        )
    if status_code == 402 or "payment" in low:
        return (
            "x_api_payment_required_402",
            "Abdul: X API returned 402 Payment Required — add credits or check "
            "the X developer plan. Default ingest uses cheap Counts: Recent "
            "(/2/tweets/counts/recent), not full tweet search.",
        )
    if "client-not-enrolled" in low or "not enrolled" in low:
        return (
            "x_api_client_not_enrolled_403",
            "Abdul: app not enrolled for tweet counts/search — enable "
            "Counts: Recent on the X developer project / add pay-per-use credits.",
        )
    if "access" in low and ("level" in low or "plan" in low or "product" in low):
        return (
            "x_api_plan_forbidden_403",
            "Abdul: current X API access level cannot call tweets/counts/recent — "
            "add credits or enable Counts: Recent on the developer project.",
        )
    if status_code == 403:
        return (
            "x_api_forbidden_403",
            "Abdul: X API 403 Forbidden — confirm bearer is valid and the app "
            "can call Counts: Recent (/2/tweets/counts/recent).",
        )
    return (
        f"x_api_plan_limit_{status_code}",
        f"Abdul: X API HTTP {status_code} — check developer portal plan / "
        "product access for counts/recent (default) or search/recent (opt-in).",
    )


def _raise_plan_limit(status_code: int, body: str) -> None:
    excerpt = (body or "").replace("\n", " ").strip()[:240]
    error_class, hint = _classify_plan_error(status_code, body)
    # Never include Authorization headers; body is API JSON (safe-ish) but truncated.
    logger.error(
        "X plan/access limit: status=%s class=%s excerpt=%r",
        status_code,
        error_class,
        excerpt,
    )
    raise XPlanLimitError(
        f"X API plan/access limit HTTP {status_code} ({error_class})",
        status_code=status_code,
        error_class=error_class,
        upgrade_hint=hint,
        body_excerpt=excerpt or None,
    )


def _raise_unauthorized(body: str) -> None:
    excerpt = (body or "").replace("\n", " ").strip()[:240]
    logger.error("X unauthorized 401 (bad/missing bearer); stopping")
    raise XPlanLimitError(
        "X API unauthorized HTTP 401",
        status_code=401,
        error_class="x_api_unauthorized_401",
        upgrade_hint=(
            "Abdul: bearer token rejected (401). Re-check X_BEARER_TOKEN "
            "in box-secrets / the X developer portal app."
        ),
        body_excerpt=excerpt or None,
    )


def _x_get(
    client: httpx.Client,
    *,
    url: str,
    fallback_url: str,
    params: dict[str, Any],
    max_retries_429: int,
) -> httpx.Response:
    """GET with twitter.com fallback, plan-limit raise, limited 429 retries.

    Never logs Authorization / bearer.
    """
    r = client.get(url, params=params)
    if r.status_code == 404:
        r = client.get(fallback_url, params=params)
        url = fallback_url

    if r.status_code in _PLAN_LIMIT_STATUSES:
        _raise_plan_limit(r.status_code, r.text)
    if r.status_code == 401:
        _raise_unauthorized(r.text)

    if r.status_code == 429:
        retries = 0
        while r.status_code == 429 and retries < max_retries_429:
            retry_after = r.headers.get("retry-after")
            wait = float(retry_after) if retry_after else (15.0 * (retries + 1))
            wait = min(wait, 90.0)
            logger.warning(
                "X rate limit 429; sleeping %.1fs (retry %d/%d)",
                wait,
                retries + 1,
                max_retries_429,
            )
            time.sleep(wait)
            r = client.get(url, params=params)
            retries += 1
        if r.status_code in _PLAN_LIMIT_STATUSES:
            _raise_plan_limit(r.status_code, r.text)
        if r.status_code == 401:
            _raise_unauthorized(r.text)
    return r


def fetch_x_recent_counts(
    query: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    granularity: str = "day",
    client: httpx.Client | None = None,
    max_retries_429: int = 2,
    **kwargs: Any,
) -> pd.DataFrame:
    """Daily mention counts from Counts: Recent. One HTTP request (no paging).

    Raises:
        XPlanLimitError: on 402/403 plan or product access limits (caller must stop).
    """
    token = x_bearer_token()
    if not token:
        logger.info("X bearer not set; skipping X ingest for this run")
        return pd.DataFrame(columns=["timestamp", "mentions"])

    query = (query or "").strip()
    if not query:
        return pd.DataFrame(columns=["timestamp", "mentions"])

    lookback_days = min(int(lookback_days), 7)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    start_s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    gran = (granularity or "day").strip().lower() or "day"

    owns = client is None
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "cmram-research/0.1",
    }
    client = client or httpx.Client(headers=headers, timeout=30.0)
    try:
        params: dict[str, Any] = {
            "query": query,
            "granularity": gran,
            "start_time": start_s,
        }
        r = _x_get(
            client,
            url=X_RECENT_COUNTS,
            fallback_url=TWITTER_RECENT_COUNTS,
            params=params,
            max_retries_429=max_retries_429,
        )
        if r.status_code == 429:
            logger.warning("X counts still 429 after retries; stopping query")
            return pd.DataFrame(columns=["timestamp", "mentions"])
        if r.status_code != 200:
            logger.warning("X recent counts HTTP %s", r.status_code)
            return pd.DataFrame(columns=["timestamp", "mentions"])

        payload = r.json()
        if (payload.get("meta") or {}).get("next_token"):
            logger.info(
                "X counts next_token present; not following (1 request/asset cap)"
            )
        return _counts_payload_to_daily(payload)
    except XPlanLimitError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("X counts fetch failed: %s", type(exc).__name__)
        return pd.DataFrame(columns=["timestamp", "mentions"])
    finally:
        if owns:
            client.close()


def _counts_payload_to_daily(payload: dict[str, Any]) -> pd.DataFrame:
    counts: dict[date, int] = {}
    for row in payload.get("data") or []:
        n = row.get("tweet_count")
        if n is None:
            continue
        start = row.get("start")
        if start:
            d = pd.to_datetime(start, utc=True).date()
        else:
            end = row.get("end")
            if not end:
                continue
            d = (pd.to_datetime(end, utc=True) - pd.Timedelta(days=1)).date()
        counts[d] = counts.get(d, 0) + int(n)
    if not counts:
        return pd.DataFrame(columns=["timestamp", "mentions"])
    rows = sorted(counts.items())
    return pd.DataFrame(
        {"timestamp": [d for d, _ in rows], "mentions": [float(c) for _, c in rows]}
    )


def fetch_x_recent_mentions(
    query: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    max_pages: int = DEFAULT_SEARCH_MAX_PAGES,
    client: httpx.Client | None = None,
    max_retries_429: int = 2,
    granularity: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bucket recent-search tweets into daily counts.

    EXPENSIVE opt-in: pulls actual posts. Default ``max_pages=1``.
    Prefer ``fetch_x_recent_counts`` for Narrative N.

    Raises:
        XPlanLimitError: on 402/403 plan or product access limits (caller must stop).
    """
    token = x_bearer_token()
    if not token:
        logger.info("X bearer not set; skipping X ingest for this run")
        return pd.DataFrame(columns=["timestamp", "mentions"])

    query = (query or "").strip()
    if not query:
        return pd.DataFrame(columns=["timestamp", "mentions"])

    lookback_days = min(int(lookback_days), 7)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    start_s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    page_cap = max(1, min(int(max_pages), 5))
    per_page = max(10, min(int(max_results), 100))

    owns = client is None
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "cmram-research/0.1",
    }
    client = client or httpx.Client(headers=headers, timeout=30.0)
    try:
        counts: dict[date, int] = {}
        next_token = None
        pages = 0
        while pages < page_cap:
            params: dict[str, Any] = {
                "query": query,
                "max_results": per_page,
                "start_time": start_s,
                "tweet.fields": "created_at",
            }
            if next_token:
                params["next_token"] = next_token
            r = _x_get(
                client,
                url=X_RECENT_SEARCH,
                fallback_url=TWITTER_RECENT_SEARCH,
                params=params,
                max_retries_429=max_retries_429,
            )
            if r.status_code == 429:
                logger.warning("X search still 429 after retries; stopping page")
                break
            if r.status_code != 200:
                logger.warning("X recent search HTTP %s", r.status_code)
                return pd.DataFrame(columns=["timestamp", "mentions"])

            payload = r.json()
            for tw in payload.get("data") or []:
                created = tw.get("created_at")
                if not created:
                    continue
                ts = pd.to_datetime(created, utc=True).to_pydatetime()
                d = ts.date()
                counts[d] = counts.get(d, 0) + 1
            next_token = (payload.get("meta") or {}).get("next_token")
            pages += 1
            if not next_token:
                break

        if not counts:
            return pd.DataFrame(columns=["timestamp", "mentions"])
        rows = sorted(counts.items())
        return pd.DataFrame(
            {"timestamp": [d for d, _ in rows], "mentions": [float(c) for _, c in rows]}
        )
    except XPlanLimitError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("X fetch failed: %s", type(exc).__name__)
        return pd.DataFrame(columns=["timestamp", "mentions"])
    finally:
        if owns:
            client.close()


def fetch_x_for_assets(
    assets: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    sleep_s: float | None = None,
    fetch_fn: Callable[..., pd.DataFrame] | None = None,
    max_assets: int | None = None,
    mode: str | None = None,
    allow_over_cap: bool = False,
    cost_cap_usd: float | None = None,
) -> pd.DataFrame:
    """Fetch X daily mention buckets per asset. Empty if no credentials.

    Default mode is ``counts`` (one Counts: Recent request per asset).
    Estimates cost before any live HTTP and stops if above cap unless
    ``allow_over_cap`` is set.

    Stops the whole batch on ``XPlanLimitError`` / ``XCostCapError``.
    """
    cfg = _x_cfg(config)
    if not x_credentials_available():
        logger.info(
            "X skipped: set TWITTER_BEARER_TOKEN or X_BEARER_TOKEN to enable"
        )
        return _empty()

    plan, work = plan_x_batch(
        assets,
        config=cfg,
        max_assets=max_assets,
        mode=mode,
        allow_over_cap=allow_over_cap,
        cost_cap_usd=cost_cap_usd,
    )
    logger.info(
        "X cost estimate: mode=%s assets=%d requests=%d × $%.4f = $%.4f "
        "(cap $%.2f, over_cap=%s, allow_over_cap=%s). %s",
        plan.mode,
        plan.n_assets,
        plan.n_requests,
        plan.cost_per_request_usd,
        plan.estimated_cost_usd,
        plan.cost_cap_usd,
        plan.over_cap,
        plan.allow_over_cap,
        _COST_NOTE,
    )
    if plan.mode == "search":
        logger.warning(
            "X mode=search is EXPENSIVE (pulls actual posts, max_pages=%d, "
            "max_results=%d). Prefer mode=counts for Narrative N.",
            plan.max_pages,
            plan.max_results,
        )
    enforce_x_cost_cap(plan)

    if work is None or work.empty:
        return _empty()

    if plan.max_assets is not None:
        logger.info(
            "X batch capped at %d of %d eligible assets",
            len(work),
            0 if assets is None else len(assets),
        )

    sleep = float(sleep_s if sleep_s is not None else plan.sleep_s)
    template = plan.query_template
    lookback = plan.lookback_days

    if fetch_fn is not None:
        fetcher = fetch_fn
        raw_ref = "x:recent_search" if plan.mode == "search" else "x:recent_counts"
    elif plan.mode == "search":
        fetcher = fetch_x_recent_mentions
        raw_ref = "x:recent_search"
    else:
        fetcher = fetch_x_recent_counts
        raw_ref = "x:recent_counts"

    rows: list[pd.DataFrame] = []
    for i, (_, asset) in enumerate(work.iterrows()):
        aid = str(asset["asset_id"])
        q = default_query(asset.to_dict(), template=template)
        logger.info("X [%d/%d] %s mode=%s", i + 1, len(work), aid, plan.mode)
        try:
            try:
                series = fetcher(
                    q,
                    lookback_days=lookback,
                    max_results=plan.max_results,
                    max_pages=plan.max_pages,
                    granularity=plan.granularity,
                )
            except TypeError:
                # Test fetch_fn that only accepts query / lookback_days.
                series = fetcher(q, lookback_days=lookback)
        except XPlanLimitError:
            logger.error(
                "X plan limit hit after %d asset(s); stopping cleanly (no further calls)",
                i,
            )
            raise
        if series.empty:
            if i + 1 < len(work) and sleep > 0:
                time.sleep(sleep)
            continue
        part = series.copy()
        part["asset_id"] = aid
        part["source"] = "x"
        part["unique_users"] = pd.NA
        part["engagement"] = pd.NA
        part["community_growth"] = pd.NA
        part["raw_payload_ref"] = raw_ref
        rows.append(part)
        if i + 1 < len(work) and sleep > 0:
            time.sleep(sleep)

    if not rows:
        return _empty()
    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "timestamp",
            "asset_id",
            "source",
            "mentions",
            "unique_users",
            "engagement",
            "community_growth",
            "raw_payload_ref",
        ]
    ]


def _empty_assets() -> pd.DataFrame:
    return pd.DataFrame(columns=["asset_id", "symbol", "name"])


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "asset_id",
            "source",
            "mentions",
            "unique_users",
            "engagement",
            "community_growth",
            "raw_payload_ref",
        ]
    )
