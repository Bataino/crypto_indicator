"""Unit tests for CoinGecko market ingest (mocked httpx — no live API)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cmram.ingest.coingecko import CoinGeckoClient, CoinGeckoError, RateLimitError
from cmram.ingest.market import (
    build_daily_bars,
    discover_band_candidates,
    fetch_market_daily,
    ingest_coingecko,
    upsert_assets,
    upsert_market_daily,
)
from cmram.db.schema import init_db


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def _chart_payload(n_days: int = 100, start: date = date(2025, 1, 1)) -> dict:
    prices, mcaps, vols = [], [], []
    for i in range(n_days):
        d = date.fromordinal(start.toordinal() + i)
        ts = _ms(d.year, d.month, d.day)
        prices.append([ts, 1.0 + i * 0.01])
        mcaps.append([ts, 20_000_000.0 + i])
        vols.append([ts, 100_000.0 + i])
    return {"prices": prices, "market_caps": mcaps, "total_volumes": vols}


def _ohlc_payload(n_days: int = 100, start: date = date(2025, 1, 1)) -> list:
    out = []
    for i in range(n_days):
        d = date.fromordinal(start.toordinal() + i)
        ts = _ms(d.year, d.month, d.day)
        px = 1.0 + i * 0.01
        out.append([ts, px, px + 0.02, px - 0.01, px + 0.01])
    return out


def _markets_row(coin_id: str, mc: float, volume: float = 1e6) -> dict:
    return {
        "id": coin_id,
        "symbol": coin_id[:4],
        "name": coin_id.title(),
        "market_cap": mc,
        "total_volume": volume,
    }


class TestBuildDailyBars:
    def test_merges_ohlc_and_chart(self):
        chart = _chart_payload(5)
        ohlc = _ohlc_payload(5)
        bars = build_daily_bars("demo-coin", ohlc=ohlc, market_chart=chart)
        assert len(bars) == 5
        assert bars[0]["asset_id"] == "demo-coin"
        assert bars[0]["high"] >= bars[0]["low"]
        assert bars[0]["market_cap_usd"] is not None
        assert bars[0]["volume_usd"] is not None
        assert bars[0]["source"] == "coingecko"

    def test_chart_only_fallback_uses_close_for_ohlc(self):
        chart = _chart_payload(3)
        bars = build_daily_bars("x", ohlc=None, market_chart=chart)
        assert len(bars) == 3
        assert bars[1]["open"] == bars[1]["close"]
        assert bars[1]["high"] == bars[1]["close"]


class TestCoinGeckoClientMockTransport:
    def test_get_markets_parses_list(self):
        payload = [_markets_row("a", 10_000_000), _markets_row("b", 200_000_000)]

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/coins/markets" in str(request.url)
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        http = httpx.Client(
            base_url="https://api.coingecko.com/api/v3",
            transport=transport,
        )
        with CoinGeckoClient(client=http, min_interval_s=0, max_retries=0) as cg:
            rows = cg.get_markets(page=1, per_page=2)
        assert len(rows) == 2
        assert rows[0]["id"] == "a"

    def test_rate_limit_raises_after_retries(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"status": {}})

        transport = httpx.MockTransport(handler)
        http = httpx.Client(
            base_url="https://api.coingecko.com/api/v3",
            transport=transport,
        )
        with patch("cmram.ingest.coingecko.time.sleep", return_value=None):
            with CoinGeckoClient(client=http, min_interval_s=0, max_retries=1) as cg:
                with pytest.raises(RateLimitError):
                    cg.get_markets()

    def test_auth_error_is_clear(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        transport = httpx.MockTransport(handler)
        http = httpx.Client(
            base_url="https://api.coingecko.com/api/v3",
            transport=transport,
        )
        with CoinGeckoClient(client=http, min_interval_s=0, max_retries=0) as cg:
            with pytest.raises(CoinGeckoError, match="auth"):
                cg.get_market_chart("bitcoin")

    def test_demo_api_key_header(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, json=[])

        transport = httpx.MockTransport(handler)
        # Build client with key; inject transport via custom httpx client sharing headers
        cg = CoinGeckoClient(api_key="CG-test-demo", min_interval_s=0, max_retries=0)
        # Replace underlying client transport
        headers = dict(cg._client.headers)
        cg._client.close()
        cg._client = httpx.Client(
            base_url=cg.base_url,
            headers=headers,
            transport=transport,
        )
        cg._owns_client = True
        try:
            cg.get_markets()
        finally:
            cg.close()
        assert seen["headers"].get("x-cg-demo-api-key") == "CG-test-demo"


class TestDiscoverAndIngest:
    def test_discover_filters_mc_bands(self):
        page1 = [
            _markets_row("tiny", 1_000_000),
            _markets_row("band-a", 8_000_000, volume=5e6),
            _markets_row("band-b", 60_000_000, volume=4e6),
            _markets_row("huge", 5_000_000_000),
        ]
        client = MagicMock()
        client.get_markets.return_value = page1
        found = discover_band_candidates(client, limit=10, max_pages=1)
        ids = [r["id"] for r in found]
        assert ids == ["band-a", "band-b"]

    def test_ingest_writes_duckdb_and_raw(self, tmp_path: Path):
        chart = _chart_payload(100)
        ohlc = _ohlc_payload(100)
        markets = [
            _markets_row("alpha", 12_000_000, volume=9e6),
            _markets_row("beta", 40_000_000, volume=8e6),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/coins/markets" in url:
                return httpx.Response(200, json=markets)
            if "/market_chart" in url:
                return httpx.Response(200, json=chart)
            if "/ohlc" in url:
                return httpx.Response(200, json=ohlc)
            return httpx.Response(404, text=url)

        transport = httpx.MockTransport(handler)
        http = httpx.Client(
            base_url="https://api.coingecko.com/api/v3",
            transport=transport,
        )
        client = CoinGeckoClient(client=http, min_interval_s=0, max_retries=0)

        db_path = tmp_path / "t.duckdb"
        raw_dir = tmp_path / "raw"
        result = ingest_coingecko(
            client=client,
            limit=2,
            days=180,
            min_bars=90,
            duckdb_path=db_path,
            raw_dir=raw_dir,
        )
        assert result.assets_upserted == 2
        assert result.market_rows_upserted == 200
        assert (raw_dir / "coingecko" / "alpha" / "market_daily.parquet").is_file()

        conn = init_db(db_path)
        try:
            n_assets = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            n_md = conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0]
            assert n_assets == 2
            assert n_md == 200
            sample = conn.execute(
                "SELECT asset_id, close, market_cap_usd FROM market_daily "
                "ORDER BY timestamp LIMIT 1"
            ).fetchone()
            assert sample[0] in {"alpha", "beta"}
            assert sample[1] is not None
        finally:
            conn.close()

    def test_ingest_dry_run_skips_writes(self, tmp_path: Path):
        markets = [_markets_row("alpha", 12_000_000)]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=markets)

        transport = httpx.MockTransport(handler)
        http = httpx.Client(
            base_url="https://api.coingecko.com/api/v3",
            transport=transport,
        )
        client = CoinGeckoClient(client=http, min_interval_s=0, max_retries=0)
        db_path = tmp_path / "t.duckdb"
        result = ingest_coingecko(
            client=client,
            limit=1,
            dry_run=True,
            duckdb_path=db_path,
            raw_dir=tmp_path / "raw",
        )
        assert result.dry_run is True
        assert result.assets_upserted == 0
        assert result.market_rows_upserted == 0
        assert not db_path.exists()

    def test_fetch_market_daily(self):
        chart = _chart_payload(10)
        ohlc = _ohlc_payload(10)

        def handler(request: httpx.Request) -> httpx.Response:
            if "/ohlc" in str(request.url):
                return httpx.Response(200, json=ohlc)
            return httpx.Response(200, json=chart)

        transport = httpx.MockTransport(handler)
        http = httpx.Client(
            base_url="https://api.coingecko.com/api/v3",
            transport=transport,
        )
        client = CoinGeckoClient(client=http, min_interval_s=0, max_retries=0)
        bars = fetch_market_daily("alpha", client=client, days=180)
        assert len(bars) == 10

    def test_upsert_helpers(self, tmp_path: Path):
        conn = init_db(tmp_path / "u.duckdb")
        try:
            now = datetime.now(timezone.utc)
            n = upsert_assets(
                conn,
                [
                    {
                        "asset_id": "a",
                        "symbol": "A",
                        "name": "A",
                        "category": None,
                        "listing_date": None,
                        "source": "coingecko",
                        "is_active": True,
                        "first_seen": now,
                        "last_seen": now,
                    }
                ],
            )
            assert n == 1
            bars = build_daily_bars("a", ohlc=_ohlc_payload(2), market_chart=_chart_payload(2))
            m = upsert_market_daily(conn, bars)
            assert m == 2
        finally:
            conn.close()




    def test_ingest_stops_at_limit_after_skips(self, tmp_path: Path):
        """Oversampled pool: skip short history, stop once --limit assets land."""
        short = _chart_payload(10)
        long = _chart_payload(100)
        markets = [
            _markets_row("short-a", 12_000_000, volume=9e6),
            _markets_row("short-b", 15_000_000, volume=8e6),
            _markets_row("long-a", 20_000_000, volume=7e6),
            _markets_row("long-b", 25_000_000, volume=6e6),
            _markets_row("long-c", 30_000_000, volume=5e6),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/coins/markets" in url:
                return httpx.Response(200, json=markets)
            if "/market_chart" in url:
                if "short-" in url:
                    return httpx.Response(200, json=short)
                return httpx.Response(200, json=long)
            return httpx.Response(404, text=url)

        transport = httpx.MockTransport(handler)
        http = httpx.Client(base_url="https://api.coingecko.com/api/v3", transport=transport)
        client = CoinGeckoClient(client=http, min_interval_s=0, max_retries=0)
        result = ingest_coingecko(
            client=client,
            limit=2,
            days=180,
            min_bars=90,
            duckdb_path=tmp_path / "t.duckdb",
            raw_dir=tmp_path / "raw",
        )
        assert result.assets_upserted == 2
        assert set(result.candidates[:2])  # pool larger than limit
        assert len(result.candidates) >= 2
        assert result.market_rows_upserted == 200




    def test_ingest_skips_existing_and_counts_toward_limit(self, tmp_path: Path):
        """Assets already in DuckDB with enough bars are not re-fetched."""
        chart = _chart_payload(100)
        markets = [
            _markets_row("alpha", 12_000_000, volume=9e6),
            _markets_row("beta", 40_000_000, volume=8e6),
            _markets_row("gamma", 30_000_000, volume=7e6),
        ]
        calls = {"chart": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/coins/markets" in url:
                return httpx.Response(200, json=markets)
            if "/market_chart" in url:
                calls["chart"] += 1
                return httpx.Response(200, json=chart)
            return httpx.Response(404, text=url)

        transport = httpx.MockTransport(handler)
        http = httpx.Client(base_url="https://api.coingecko.com/api/v3", transport=transport)
        client = CoinGeckoClient(client=http, min_interval_s=0, max_retries=0)
        db_path = tmp_path / "t.duckdb"
        raw_dir = tmp_path / "raw"

        # Seed alpha as already present
        first = ingest_coingecko(
            client=client,
            asset_ids=["alpha"],
            limit=1,
            days=180,
            min_bars=90,
            duckdb_path=db_path,
            raw_dir=raw_dir,
            skip_existing=False,
            prefer_raw=False,
        )
        assert first.assets_upserted == 1
        charts_after_seed = calls["chart"]

        result = ingest_coingecko(
            client=client,
            limit=2,
            days=180,
            min_bars=90,
            duckdb_path=db_path,
            raw_dir=raw_dir,
            skip_existing=True,
            prefer_raw=False,
        )
        # limit=2 with 1 already → 1 new fetch (beta or gamma)
        assert result.assets_upserted == 1
        assert calls["chart"] == charts_after_seed + 1
        assert any(s.startswith("alpha:already_present:") for s in result.skipped_insufficient)


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("os").environ.get("CMRAM_LIVE_INGEST") != "1",
    reason="Set CMRAM_LIVE_INGEST=1 to hit live CoinGecko",
)
def test_live_smoke_limit_one(tmp_path: Path):
    """Optional live smoke — skipped by default."""
    result = ingest_coingecko(
        limit=1,
        days=180,
        min_bars=90,
        duckdb_path=tmp_path / "live.duckdb",
        raw_dir=tmp_path / "raw",
    )
    assert result.assets_upserted >= 1
    assert result.market_rows_upserted >= 90
