"""Reddit mention counts — public JSON first, optional PRAW if creds exist.

Never prints secrets. On 403 / block, returns empty (honest missingness).
Public search is recent-biased; V0.1 buckets post created_utc into daily counts.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_UA = "cmram-research/0.1 (academic research; no trading)"


def _reddit_creds_present() -> bool:
    cid = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    secret = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    return bool(cid and secret)


def default_query(asset: dict[str, Any], *, template: str = "{name} OR {symbol}") -> str:
    symbol = str(asset.get("symbol") or "").strip()
    name = str(asset.get("name") or symbol or "").strip()
    asset_id = str(asset.get("asset_id") or "").strip()
    try:
        return template.format(symbol=symbol, name=name, asset_id=asset_id).strip()
    except KeyError:
        return f"{name} OR {symbol}".strip()


def fetch_reddit_public_mentions(
    query: str,
    *,
    lookback_days: int = 90,
    search_limit: int = 100,
    sleep_s: float = 0.0,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Bucket public search hits into daily mention counts.

    Returns timestamp/mentions. Empty on block or empty results.
    """
    query = (query or "").strip()
    if not query:
        return pd.DataFrame(columns=["timestamp", "mentions"])

    owns = client is None
    headers = {"User-Agent": DEFAULT_UA, "Accept": "application/json"}
    client = client or httpx.Client(headers=headers, timeout=30.0, follow_redirects=True)
    try:
        # Prefer old.reddit JSON; fall back to www
        urls = (
            "https://old.reddit.com/search.json",
            "https://www.reddit.com/search.json",
        )
        children: list[dict[str, Any]] = []
        last_status = None
        for url in urls:
            try:
                r = client.get(
                    url,
                    params={
                        "q": query,
                        "sort": "new",
                        "limit": min(int(search_limit), 100),
                        "t": "year",
                        "type": "link",
                    },
                )
                last_status = r.status_code
                if r.status_code == 200:
                    payload = r.json()
                    children = list(payload.get("data", {}).get("children") or [])
                    break
                logger.info("Reddit public %s → HTTP %s", url, r.status_code)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reddit public error on %s: %s", url, exc)
        if sleep_s > 0:
            time.sleep(sleep_s)
        if not children:
            if last_status and last_status >= 400:
                logger.warning(
                    "Reddit public returned no data (status=%s); try PRAW creds later",
                    last_status,
                )
            return pd.DataFrame(columns=["timestamp", "mentions"])

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
        counts: dict[date, int] = {}
        for child in children:
            data = child.get("data") or {}
            created = data.get("created_utc")
            if created is None:
                continue
            ts = datetime.fromtimestamp(float(created), tz=timezone.utc)
            if ts < cutoff:
                continue
            d = ts.date()
            counts[d] = counts.get(d, 0) + 1

        if not counts:
            return pd.DataFrame(columns=["timestamp", "mentions"])
        rows = sorted(counts.items())
        return pd.DataFrame(
            {"timestamp": [d for d, _ in rows], "mentions": [float(c) for _, c in rows]}
        )
    finally:
        if owns:
            client.close()


def fetch_reddit_praw_mentions(
    query: str,
    *,
    lookback_days: int = 90,
    search_limit: int = 100,
) -> pd.DataFrame:
    """Optional PRAW path when REDDIT_CLIENT_ID/SECRET are set. Never logs secrets."""
    if not _reddit_creds_present():
        return pd.DataFrame(columns=["timestamp", "mentions"])
    try:
        import praw  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PRAW not installed; skipping authenticated Reddit")
        return pd.DataFrame(columns=["timestamp", "mentions"])

    client_id = os.environ["REDDIT_CLIENT_ID"].strip()
    client_secret = os.environ["REDDIT_CLIENT_SECRET"].strip()
    ua = (os.environ.get("REDDIT_USER_AGENT") or DEFAULT_UA).strip()
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=ua,
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
    counts: dict[date, int] = {}
    try:
        for submission in reddit.subreddit("all").search(
            query, sort="new", time_filter="year", limit=int(search_limit)
        ):
            ts = datetime.fromtimestamp(float(submission.created_utc), tz=timezone.utc)
            if ts < cutoff:
                continue
            d = ts.date()
            counts[d] = counts.get(d, 0) + 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("PRAW search failed: %s", exc)
        return pd.DataFrame(columns=["timestamp", "mentions"])

    if not counts:
        return pd.DataFrame(columns=["timestamp", "mentions"])
    rows = sorted(counts.items())
    return pd.DataFrame(
        {"timestamp": [d for d, _ in rows], "mentions": [float(c) for _, c in rows]}
    )


def fetch_reddit_series(
    query: str,
    *,
    lookback_days: int = 90,
    search_limit: int = 100,
    sleep_s: float = 0.0,
    client: httpx.Client | None = None,
    public_fn: Callable[..., pd.DataFrame] | None = None,
    praw_fn: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Public JSON first; PRAW if public empty and creds present."""
    public = (public_fn or fetch_reddit_public_mentions)(
        query,
        lookback_days=lookback_days,
        search_limit=search_limit,
        sleep_s=sleep_s,
        client=client,
    )
    if not public.empty:
        return public
    return (praw_fn or fetch_reddit_praw_mentions)(
        query, lookback_days=lookback_days, search_limit=search_limit
    )


def fetch_reddit_for_assets(
    assets: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    sleep_s: float | None = None,
    fetch_fn: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fetch Reddit daily mentions for each asset → social_daily rows."""
    cfg = dict((config or {}).get("reddit") or {})
    if config and "reddit" not in config and "sleep_s" in (config or {}):
        cfg = dict(config or {})
    sleep = float(sleep_s if sleep_s is not None else cfg.get("sleep_s", 2.0))
    template = str(cfg.get("query_template") or "{name} OR {symbol}")
    lookback = int(cfg.get("lookback_days") or 90)
    limit = int(cfg.get("search_limit") or 100)
    fetcher = fetch_fn or fetch_reddit_series

    if assets is None or assets.empty:
        return _empty("reddit")

    rows: list[pd.DataFrame] = []
    for i, (_, asset) in enumerate(assets.iterrows()):
        aid = str(asset["asset_id"])
        q = default_query(asset.to_dict(), template=template)
        logger.info("Reddit [%d/%d] %s → %r", i + 1, len(assets), aid, q)
        series = fetcher(
            q, lookback_days=lookback, search_limit=limit, sleep_s=0.0
        )
        if series.empty:
            if i + 1 < len(assets) and sleep > 0:
                time.sleep(sleep)
            continue
        part = series.copy()
        part["asset_id"] = aid
        part["source"] = "reddit"
        part["unique_users"] = pd.NA
        part["engagement"] = pd.NA
        part["community_growth"] = pd.NA
        part["raw_payload_ref"] = f"reddit:{q}"
        rows.append(part)
        if i + 1 < len(assets) and sleep > 0:
            time.sleep(sleep)

    if not rows:
        return _empty("reddit")
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


def _empty(source: str) -> pd.DataFrame:
    del source
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
