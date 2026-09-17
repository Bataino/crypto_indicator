"""CoinGecko HTTP client (httpx) with rate-limit sleep/backoff.

Survivorship note
-----------------
``/coins/markets`` and current listings only include coins CoinGecko still
tracks. Delisted / failed / rugged assets drop out of the live catalogue, so
a point-in-time historical universe reconstructed only from today's markets
pages is **survivorship-biased**. Prefer stored historical snapshots for
research once available; document this limit in any backtest that relies on
live discovery alone.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PUBLIC_BASE_URL = "https://api.coingecko.com/api/v3"
PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"

# Conservative defaults: free/public is ~5–15 calls/min and shared IPs trip 429s.
DEFAULT_MIN_INTERVAL_S = 90.0
DEMO_MIN_INTERVAL_S = 2.5
PRO_MIN_INTERVAL_S = 1.2


class CoinGeckoError(Exception):
    """Base error for CoinGecko client failures."""


class RateLimitError(CoinGeckoError):
    """Raised when rate limits are exhausted after retries."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class CoinGeckoClient:
    """Thin httpx wrapper around CoinGecko REST endpoints used by CMRAM ingest."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        min_interval_s: float | None = None,
        max_retries: int = 8,
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("COINGECKO_API_KEY")
        if self.api_key is not None:
            self.api_key = self.api_key.strip() or None

        plan = (os.environ.get("COINGECKO_API_PLAN") or "").strip().lower()
        if base_url:
            self.base_url = base_url.rstrip("/")
            self._plan = "custom"
        elif plan == "pro" or (self.api_key and plan != "demo" and os.environ.get("COINGECKO_PRO")):
            self.base_url = PRO_BASE_URL
            self._plan = "pro"
        elif self.api_key:
            # Demo keys use the public host + demo header.
            self.base_url = PUBLIC_BASE_URL
            self._plan = "demo"
        else:
            self.base_url = PUBLIC_BASE_URL
            self._plan = "public"

        if min_interval_s is not None:
            self.min_interval_s = float(min_interval_s)
        elif self._plan == "pro":
            self.min_interval_s = PRO_MIN_INTERVAL_S
        elif self._plan == "demo":
            self.min_interval_s = DEMO_MIN_INTERVAL_S
        else:
            self.min_interval_s = DEFAULT_MIN_INTERVAL_S

        self.max_retries = max_retries
        self._owns_client = client is None
        headers = {
            "Accept": "application/json",
            "User-Agent": "cmram-research/0.1 (+research; no trading)",
        }
        if self.api_key:
            if self._plan == "pro":
                headers["x-cg-pro-api-key"] = self.api_key
            else:
                # Demo (and unknown keyed public) — CoinGecko Demo header.
                headers["x-cg-demo-api-key"] = self.api_key

        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout_s,
        )
        self._last_request_at = 0.0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CoinGeckoClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """HTTP JSON request with inter-call sleep and 429/5xx backoff."""
        path = path if path.startswith("/") else f"/{path}"
        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._client.request(method, path, params=params)
            except httpx.HTTPError as exc:
                last_err = CoinGeckoError(f"CoinGecko transport error on {path}: {exc}")
                logger.warning("%s (attempt %s)", last_err, attempt + 1)
                time.sleep(min(60.0, 2.0 ** attempt))
                continue

            self._last_request_at = time.monotonic()

            if resp.status_code == 429:
                retry_after_hdr = resp.headers.get("retry-after")
                try:
                    retry_after = float(retry_after_hdr) if retry_after_hdr else None
                except ValueError:
                    retry_after = None
                # CoinGecko often returns Retry-After: 0 while still blocking;
                # never sleep 0 on 429 — use exponential floor instead.
                backoff = min(180.0, 30.0 * (2**attempt))
                if retry_after is not None and retry_after > 0:
                    sleep_s = max(retry_after, backoff * 0.5)
                else:
                    sleep_s = backoff
                msg = (
                    f"CoinGecko rate limit (429) on {path}; "
                    f"retry_after={retry_after_hdr!r} sleeping={sleep_s:.1f}s "
                    f"attempt={attempt + 1}/{self.max_retries + 1}"
                )
                logger.warning(msg)
                if attempt >= self.max_retries:
                    raise RateLimitError(msg, retry_after=retry_after)
                time.sleep(sleep_s)
                continue

            if resp.status_code in {500, 502, 503, 504}:
                msg = f"CoinGecko server error {resp.status_code} on {path}"
                logger.warning("%s (attempt %s)", msg, attempt + 1)
                last_err = CoinGeckoError(msg)
                if attempt >= self.max_retries:
                    raise last_err
                time.sleep(min(60.0, 2.0 ** attempt))
                continue

            if resp.status_code == 401 or resp.status_code == 403:
                raise CoinGeckoError(
                    f"CoinGecko auth error {resp.status_code} on {path}: {resp.text[:300]}"
                )

            if resp.status_code >= 400:
                raise CoinGeckoError(
                    f"CoinGecko HTTP {resp.status_code} on {path}: {resp.text[:500]}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise CoinGeckoError(f"CoinGecko invalid JSON on {path}: {exc}") from exc

        raise last_err or CoinGeckoError(f"CoinGecko request failed on {path}")

    def get_markets(
        self,
        *,
        vs_currency: str = "usd",
        page: int = 1,
        per_page: int = 250,
        order: str = "volume_desc",
    ) -> list[dict[str, Any]]:
        data = self.request(
            "GET",
            "/coins/markets",
            params={
                "vs_currency": vs_currency,
                "order": order,
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
            },
        )
        if not isinstance(data, list):
            raise CoinGeckoError(f"Unexpected markets payload type: {type(data)}")
        return data

    def get_market_chart(
        self,
        coin_id: str,
        *,
        vs_currency: str = "usd",
        days: int | str = 180,
    ) -> dict[str, Any]:
        """Daily-ish series of prices, market_caps, total_volumes.

        CoinGecko returns daily granularity when ``days`` > 90 (or ``max``).
        """
        data = self.request(
            "GET",
            f"/coins/{coin_id}/market_chart",
            params={"vs_currency": vs_currency, "days": str(days)},
        )
        if not isinstance(data, dict):
            raise CoinGeckoError(f"Unexpected market_chart payload type: {type(data)}")
        return data

    def get_ohlc(
        self,
        coin_id: str,
        *,
        vs_currency: str = "usd",
        days: int = 180,
    ) -> list[list[float]]:
        """OHLC candles: ``[timestamp_ms, open, high, low, close]``.

        For ``days`` in {180, 365} CoinGecko uses ~1-day candle bodies.
        """
        data = self.request(
            "GET",
            f"/coins/{coin_id}/ohlc",
            params={"vs_currency": vs_currency, "days": days},
        )
        if not isinstance(data, list):
            raise CoinGeckoError(f"Unexpected ohlc payload type: {type(data)}")
        return data
