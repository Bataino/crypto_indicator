"""Mocked Santiment GraphQL tests — never hit live API / never log keys."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cmram.ingest.santiment import (
    SantimentCallBudgetError,
    SantimentClient,
    fetch_santiment_for_assets,
    fetch_social_volume_series,
    probe_coverage,
    resolve_slug,
)


class _FakeResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeGraphQLClient:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, json=None):
        self.calls.append({"url": url, "json": json})
        if not self.responses:
            return _FakeResp(500, {"errors": [{"message": "unexpected extra call"}]})
        return _FakeResp(200, self.responses.pop(0))

    def close(self):
        pass


def test_resolve_slug_override_and_asset_id():
    available = {"helium", "gala-v2", "sonic"}
    assert resolve_slug(
        {"asset_id": "gala", "symbol": "GALA", "name": "GALA"},
        available_slugs=available,
        overrides={"gala": "gala-v2"},
    ) == ("gala-v2", "override")
    assert resolve_slug(
        {"asset_id": "helium", "symbol": "HNT", "name": "Helium"},
        available_slugs=available,
    ) == ("helium", "asset_id")
    assert resolve_slug(
        {"asset_id": "missing-coin", "symbol": "ZZZ", "name": "Nope"},
        available_slugs=available,
    )[0] is None


def test_resolve_slug_unique_ticker():
    projects = [
        {"slug": "sonic", "name": "Sonic", "ticker": "S"},
        {"slug": "other", "name": "Other", "ticker": "OTR"},
    ]
    available = {"sonic", "other"}
    slug, reason = resolve_slug(
        {"asset_id": "sonic-3", "symbol": "S", "name": "Sonic"},
        available_slugs=available,
        projects=projects,
    )
    assert slug == "sonic"
    assert reason == "ticker"


def test_fetch_social_volume_series_mock():
    points = [
        {"datetime": "2026-01-01T00:00:00Z", "value": 3.0},
        {"datetime": "2026-01-02T00:00:00Z", "value": 5.0},
    ]

    def graphql_fn(**kwargs):
        return points

    df = fetch_social_volume_series(
        "helium",
        from_dt=date(2026, 1, 1),
        to_dt=date(2026, 1, 10),
        graphql_fn=graphql_fn,
    )
    assert len(df) == 2
    assert df["mentions"].tolist() == [3.0, 5.0]


def test_client_auth_header_and_budget(monkeypatch):
    monkeypatch.setenv("SANTIMENT_API_KEY", "test-san-key-not-real")
    fake = _FakeGraphQLClient(
        [
            {
                "data": {
                    "getMetric": {
                        "metadata": {
                            "isAccessible": True,
                            "isRestricted": True,
                            "restrictedFrom": "2025-09-01T00:00:00Z",
                            "restrictedTo": "2026-08-01T00:00:00Z",
                            "availableSlugs": ["helium"],
                        }
                    }
                }
            }
        ]
    )
    client = SantimentClient(api_key="test-san-key-not-real", call_budget=1, client=fake)
    meta = client.metric_metadata()
    assert meta["isAccessible"] is True
    assert len(fake.calls) == 1
    headers = fake.calls  # auth is on httpx client headers, not post body
    # Key must not appear in GraphQL body
    blob = str(fake.calls)
    assert "test-san-key-not-real" not in blob
    with pytest.raises(SantimentCallBudgetError):
        client.metric_metadata()
    client.close()


def test_probe_coverage_mocked(monkeypatch):
    monkeypatch.setenv("SANTIMENT_API_KEY", "test-san-key-not-real")
    assets = pd.DataFrame(
        [
            {"asset_id": "helium", "symbol": "HNT", "name": "Helium"},
            {"asset_id": "nope", "symbol": "NOPE", "name": "Nope Coin"},
        ]
    )
    fake = _FakeGraphQLClient(
        [
            {
                "data": {
                    "getMetric": {
                        "metadata": {
                            "isAccessible": True,
                            "isRestricted": True,
                            "restrictedFrom": "2025-09-01T00:00:00Z",
                            "restrictedTo": "2026-08-01T00:00:00Z",
                            "availableSlugs": ["helium", "gala-v2"],
                        }
                    }
                }
            },
            {
                "data": {
                    "allProjects": [
                        {"slug": "helium", "name": "Helium", "ticker": "HNT"},
                        {"slug": "gala-v2", "name": "Gala", "ticker": "GALA"},
                    ]
                }
            },
        ]
    )
    client = SantimentClient(api_key="test-san-key-not-real", call_budget=10, client=fake)
    cov = probe_coverage(assets, client=client, config={"santiment": {}})
    assert cov["n_covered"] == 1
    assert cov["covered"][0]["asset_id"] == "helium"
    assert cov["n_missing"] == 1
    assert cov["window_from"] == "2025-09-01"
    assert cov["window_to"] == "2026-08-01"
    client.close()


def test_fetch_santiment_for_assets_mocked(monkeypatch):
    monkeypatch.setenv("SANTIMENT_API_KEY", "test-san-key-not-real")
    assets = pd.DataFrame(
        [{"asset_id": "helium", "symbol": "HNT", "name": "Helium"}]
    )

    def fake_fetch(slug, **kwargs):
        return pd.DataFrame(
            {
                "timestamp": [date(2026, 1, 1), date(2026, 1, 2)],
                "mentions": [10.0, 12.0],
            }
        )

    coverage = {
        "n_covered": 1,
        "n_missing": 0,
        "covered": [
            {
                "asset_id": "helium",
                "slug": "helium",
                "map_reason": "asset_id",
            }
        ],
        "missing": [],
        "window_from": "2025-09-01",
        "window_to": "2026-08-01",
        "plan": {"free_plan_note": "test"},
    }
    fake = _FakeGraphQLClient([])  # no HTTP if coverage + fetch_fn provided
    client = SantimentClient(api_key="test-san-key-not-real", call_budget=5, client=fake)
    out, report = fetch_santiment_for_assets(
        assets,
        config={"santiment": {"sleep_s": 0, "call_budget": 5}},
        client=client,
        coverage=coverage,
        fetch_fn=fake_fetch,
    )
    assert len(out) == 2
    assert (out["source"] == "santiment").all()
    assert report["assets"] == 1
    assert "test-san-key-not-real" not in str(report)
    client.close()


def test_ingest_social_wires_santiment(monkeypatch):
    from unittest.mock import patch

    from cmram.ingest.social import ingest_social

    assets = pd.DataFrame(
        [{"asset_id": "helium", "symbol": "HNT", "name": "Helium"}]
    )
    mock_df = pd.DataFrame(
        {
            "timestamp": [date(2026, 1, 1)],
            "asset_id": ["helium"],
            "source": ["santiment"],
            "mentions": [7.0],
            "unique_users": [None],
            "engagement": [None],
            "community_growth": [None],
            "raw_payload_ref": ["santiment:social_volume_total:helium"],
        }
    )
    monkeypatch.setenv("SANTIMENT_API_KEY", "test-san-key-not-real")
    with patch(
        "cmram.ingest.social.fetch_santiment_for_assets",
        return_value=(mock_df, {"assets": 1, "rows": 1, "calls_used": 3}),
    ):
        social, report = ingest_social(
            assets,
            config={
                "sources": {
                    "trends": {"enabled": False},
                    "reddit": {"enabled": False},
                    "x": {"enabled": False},
                    "wikipedia": {"enabled": False},
                    "santiment": {"enabled": True},
                }
            },
            sources=["santiment"],
        )
    assert report["per_source"]["santiment"]["assets"] == 1
    assert (social["source"] == "santiment").all()
