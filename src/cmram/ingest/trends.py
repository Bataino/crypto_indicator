"""Google Trends attention proxy via pytrends (unofficial, free).

V0.1: per-asset keyword interest over ``today 3-m`` (typically daily).
Rate-limit politely; never invent series on failure.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


def _patch_urllib3_retry() -> None:
    """pytrends still passes method_whitelist; urllib3≥2 renamed it."""
    try:
        import urllib3.util.retry as retry_mod
    except ImportError:
        return
    if getattr(retry_mod.Retry.__init__, "_cmram_patched", False):
        return
    _orig = retry_mod.Retry.__init__

    def _patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "method_whitelist" in kwargs and "allowed_methods" not in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        elif "method_whitelist" in kwargs:
            kwargs.pop("method_whitelist")
        return _orig(self, *args, **kwargs)

    _patched._cmram_patched = True  # type: ignore[attr-defined]
    retry_mod.Retry.__init__ = _patched  # type: ignore[method-assign]


def default_keyword(asset: dict[str, Any], *, template: str = "{name}") -> str:
    """Build a Trends keyword from asset fields + optional template."""
    symbol = str(asset.get("symbol") or "").strip()
    name = str(asset.get("name") or symbol or asset.get("asset_id") or "").strip()
    asset_id = str(asset.get("asset_id") or "").strip()
    # Short / ambiguous tickers → append crypto
    if "{symbol}" in template and len(symbol) <= 3 and "crypto" not in template.lower():
        return f"{symbol} crypto"
    try:
        return template.format(symbol=symbol, name=name, asset_id=asset_id).strip()
    except KeyError:
        return name or symbol or asset_id


def fetch_trends_series(
    keyword: str,
    *,
    timeframe: str = "today 3-m",
    hl: str = "en-US",
    tz: int = 0,
    trend_req_factory: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Return DataFrame columns: timestamp (date), mentions (interest 0–100).

    Empty frame on failure (caller decides coverage). Does not fabricate values.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return pd.DataFrame(columns=["timestamp", "mentions"])

    _patch_urllib3_retry()
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise ImportError(
            "pytrends is required for Google Trends ingest; pip install pytrends"
        ) from exc

    factory = trend_req_factory or (
        lambda: TrendReq(hl=hl, tz=tz, retries=2, backoff_factor=0.8)
    )
    pt = factory()
    try:
        pt.build_payload([keyword], timeframe=timeframe, geo="")
        raw = pt.interest_over_time()
    except Exception as exc:  # noqa: BLE001 — network / Google flakiness
        logger.warning("Trends failed for %r: %s", keyword, exc)
        return pd.DataFrame(columns=["timestamp", "mentions"])

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["timestamp", "mentions"])

    col = keyword if keyword in raw.columns else raw.columns[0]
    series = pd.to_numeric(raw[col], errors="coerce")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw.index).tz_localize(None).normalize(),
            "mentions": series.astype("float64"),
        }
    )
    # Drop partial current bucket when flagged
    if "isPartial" in raw.columns:
        partial = raw["isPartial"].astype(bool)
        out = out.loc[~partial.to_numpy()].copy()
    out = out.dropna(subset=["mentions"])
    out["timestamp"] = out["timestamp"].dt.date
    return out.reset_index(drop=True)


def fetch_trends_for_assets(
    assets: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    sleep_s: float | None = None,
    trend_req_factory: Callable[..., Any] | None = None,
    fetch_fn: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fetch Trends for each asset row (asset_id, symbol, name).

    Returns long social_daily-shaped frame with source='trends'.
    """
    if config and "trends" in config:
        cfg = dict(config["trends"] or {})
    else:
        cfg = dict(config or {})
    sleep = float(sleep_s if sleep_s is not None else cfg.get("sleep_s", 10.0))
    template = str(cfg.get("keyword_template") or "{name}")
    overrides = dict(cfg.get("keyword_overrides") or {})
    timeframe = str(cfg.get("timeframe") or "today 3-m")
    hl = str(cfg.get("hl") or "en-US")
    tz = int(cfg.get("tz") or 0)
    fetcher = fetch_fn or fetch_trends_series

    rows: list[pd.DataFrame] = []
    if assets is None or assets.empty:
        return _empty_social("trends")

    for i, (_, asset) in enumerate(assets.iterrows()):
        aid = str(asset["asset_id"])
        kw = overrides.get(aid) or default_keyword(asset.to_dict(), template=template)
        logger.info("Trends [%d/%d] %s → %r", i + 1, len(assets), aid, kw)
        series = fetcher(
            kw, timeframe=timeframe, hl=hl, tz=tz, trend_req_factory=trend_req_factory
        )
        if series.empty:
            continue
        part = series.copy()
        part["asset_id"] = aid
        part["source"] = "trends"
        part["unique_users"] = pd.NA
        part["engagement"] = pd.NA
        part["community_growth"] = pd.NA
        part["raw_payload_ref"] = f"trends:{kw}"
        rows.append(part)
        if i + 1 < len(assets) and sleep > 0:
            time.sleep(sleep)

    if not rows:
        return _empty_social("trends")
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


def _empty_social(source: str) -> pd.DataFrame:
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
