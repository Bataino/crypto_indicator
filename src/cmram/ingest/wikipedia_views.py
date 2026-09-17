"""Optional Wikimedia pageview proxy (free, daily, historical).

Useful when Trends/Reddit are sparse for an asset that has a Wikipedia article.
Respects WMF robot policy via identifying User-Agent.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Callable

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

UA = "cmram-research/0.1 (https://local/cmram; research; educational)"


def fetch_wikipedia_pageviews(
    title: str,
    *,
    start: date | None = None,
    end: date | None = None,
    project: str = "en.wikipedia",
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Daily pageviews for an article title (underscores ok). Empty on 404."""
    title = (title or "").strip().replace(" ", "_")
    if not title:
        return pd.DataFrame(columns=["timestamp", "mentions"])
    end = end or date.today()
    start = start or (end - timedelta(days=90))
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{project}/all-access/user/{title}/daily/{start_s}/{end_s}"
    )
    owns = client is None
    client = client or httpx.Client(
        headers={"User-Agent": UA, "Accept": "application/json"}, timeout=30.0
    )
    try:
        r = client.get(url)
        if r.status_code != 200:
            logger.info("Wikipedia pageviews %s → HTTP %s", title, r.status_code)
            return pd.DataFrame(columns=["timestamp", "mentions"])
        items = r.json().get("items") or []
        if not items:
            return pd.DataFrame(columns=["timestamp", "mentions"])
        rows = []
        for it in items:
            ts = str(it.get("timestamp", ""))[:8]
            if len(ts) != 8:
                continue
            d = date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))
            rows.append({"timestamp": d, "mentions": float(it.get("views") or 0)})
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia fetch failed for %s: %s", title, exc)
        return pd.DataFrame(columns=["timestamp", "mentions"])
    finally:
        if owns:
            client.close()


def fetch_wikipedia_for_assets(
    assets: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    sleep_s: float | None = None,
    fetch_fn: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    cfg = dict((config or {}).get("wikipedia") or {})
    sleep = float(sleep_s if sleep_s is not None else cfg.get("sleep_s", 0.25))
    overrides = dict(cfg.get("title_overrides") or {})
    project = str(cfg.get("project") or "en.wikipedia")
    fetcher = fetch_fn or fetch_wikipedia_pageviews
    end = date.today()
    start = end - timedelta(days=90)

    if assets is None or assets.empty:
        return _empty()

    rows: list[pd.DataFrame] = []
    for i, (_, asset) in enumerate(assets.iterrows()):
        aid = str(asset["asset_id"])
        title = overrides.get(aid)
        if not title:
            continue
        logger.info("Wikipedia [%d] %s → %s", i + 1, aid, title)
        series = fetcher(title, start=start, end=end, project=project)
        if series.empty:
            if sleep > 0:
                time.sleep(sleep)
            continue
        part = series.copy()
        part["asset_id"] = aid
        part["source"] = "wikipedia"
        part["unique_users"] = pd.NA
        part["engagement"] = pd.NA
        part["community_growth"] = pd.NA
        part["raw_payload_ref"] = f"wikipedia:{title}"
        rows.append(part)
        if sleep > 0:
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
