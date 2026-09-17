"""Mocks for Narrative Acceleration (N) + multi-source social ingest."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.features.narrative import N_INPUT_COLS, combine_attention, narrative_inputs, score_N
from cmram.features.pipeline import compute_features_daily
from cmram.ingest.reddit_public import fetch_reddit_for_assets
from cmram.ingest.social import ingest_social, source_availability
from cmram.ingest.trends import default_keyword, fetch_trends_for_assets
from cmram.ingest.x_twitter import (
    XCostCapError,
    estimate_x_batch_cost,
    fetch_x_for_assets,
    fetch_x_recent_counts,
    fetch_x_recent_mentions,
    plan_x_batch,
    x_credentials_available,
)


def _synth_social(asset_ids: list[str], n_days: int = 60, *, start: date | None = None) -> pd.DataFrame:
    start = start or date(2024, 1, 1)
    rows = []
    for aid in asset_ids:
        base = 5.0 + hash(aid) % 7
        for i in range(n_days):
            # Rising attention in second half for "a"
            t = start + timedelta(days=i)
            if aid == "a":
                ment = base + (0.5 * i if i > 30 else 0.05 * i)
            else:
                ment = base + 0.02 * i
            rows.append(
                {
                    "timestamp": t,
                    "asset_id": aid,
                    "source": "trends",
                    "mentions": float(ment),
                    "unique_users": None,
                    "engagement": None,
                    "community_growth": None,
                    "raw_payload_ref": "mock",
                }
            )
    return pd.DataFrame(rows)


def _synth_market(asset_ids: list[str], n: int = 80) -> pd.DataFrame:
    start = date(2024, 1, 1)
    frames = []
    for j, aid in enumerate(asset_ids):
        dates = [start + timedelta(days=i) for i in range(n)]
        closes = 1.0 * np.cumprod(np.r_[1.0, np.full(n - 1, 1.01 + 0.001 * j)])
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "asset_id": aid,
                    "open": closes * 0.99,
                    "high": closes * 1.01,
                    "low": closes * 0.98,
                    "close": closes,
                    "volume_usd": np.full(n, 100_000.0 * (j + 1)),
                    "market_cap_usd": np.full(n, 20_000_000.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _membership(market: pd.DataFrame, from_idx: int = 50) -> pd.DataFrame:
    rows = []
    for aid, g in market.groupby("asset_id"):
        dates = list(pd.to_datetime(g["timestamp"]).dt.date)
        for ts in dates[from_idx:]:
            rows.append(
                {
                    "timestamp": ts,
                    "asset_id": aid,
                    "band": "A_5_50",
                    "market_cap_usd": 20_000_000.0,
                    "history_days": from_idx + 1,
                    "liquidity_pass": True,
                }
            )
    return pd.DataFrame(rows)


def test_score_n_null_when_no_social():
    out = score_N(None, index=pd.RangeIndex(3))
    assert out.isna().all()
    assert score_N(pd.DataFrame()).empty or score_N(pd.DataFrame()).isna().all()


def test_narrative_inputs_velocity_accel_lowbase():
    social = _synth_social(["a", "b", "c"], n_days=60)
    # legacy keeps signed velocity so rising>flat comparison is direct
    raw = narrative_inputs(social, narrative_mode="legacy")
    assert not raw.empty
    for col in N_INPUT_COLS:
        assert col in raw.columns
    assert raw["n_baseline"].notna().any()
    # Rising asset should tend to have higher velocity later
    late = raw[pd.to_datetime(raw["timestamp"]).dt.date >= date(2024, 2, 20)]
    a = late.loc[late["asset_id"] == "a", "n_velocity"].mean()
    b = late.loc[late["asset_id"] == "b", "n_velocity"].mean()
    assert a > b


def test_combine_attention_multi_source():
    social = _synth_social(["a"], n_days=5)
    extra = social.copy()
    extra["source"] = "reddit"
    extra["mentions"] = extra["mentions"] * 2
    combined = combine_attention(pd.concat([social, extra], ignore_index=True), method="mean")
    assert combined["sources_present"].str.contains("reddit").any()
    assert combined["sources_present"].str.contains("trends").any()


def test_model_c_uses_n_models_ab_ignore():
    market = _synth_market(["a", "b", "c"])
    mem = _membership(market)
    social = _synth_social(["a", "b", "c"], n_days=80)
    cfg = {"models": ["A", "B", "C"], "model_version": "test_n"}
    feat = compute_features_daily(market, mem, social_daily=social, config=cfg)
    assert set(feat["model"]) == {"A", "B", "C"}
    a = feat[feat["model"] == "A"]
    b = feat[feat["model"] == "B"]
    c = feat[feat["model"] == "C"]
    assert a["score_N"].isna().all()
    assert a["n_available"].eq(False).all()
    assert b["score_N"].isna().all()
    assert b["n_available"].eq(False).all()
    assert c["n_available"].any()
    assert c.loc[c["n_available"], "score_N"].notna().all()
    # Model C MREI includes N when available → differ from B on those rows
    c2 = c.loc[c["n_available"]].copy()
    if not c2.empty:
        merged = c2.merge(
            b[["timestamp", "asset_id", "band", "MREI"]],
            on=["timestamp", "asset_id", "band"],
            suffixes=("_c", "_b"),
        )
        assert not np.allclose(
            merged["MREI_c"].to_numpy(dtype=float),
            merged["MREI_b"].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_trends_fetch_uses_mock(monkeypatch):
    assets = pd.DataFrame(
        [{"asset_id": "gala", "symbol": "GALA", "name": "GALA"}]
    )

    def fake_fetch(keyword, **kwargs):
        days = [date(2024, 6, 1) + timedelta(days=i) for i in range(40)]
        return pd.DataFrame({"timestamp": days, "mentions": np.linspace(10, 40, 40)})

    out = fetch_trends_for_assets(
        assets, config={"trends": {"sleep_s": 0}}, fetch_fn=fake_fetch
    )
    assert len(out) == 40
    assert (out["source"] == "trends").all()
    assert out["asset_id"].eq("gala").all()


def test_reddit_fetch_uses_mock():
    assets = pd.DataFrame(
        [{"asset_id": "gala", "symbol": "GALA", "name": "GALA"}]
    )

    def fake(query, **kwargs):
        return pd.DataFrame(
            {
                "timestamp": [date(2024, 6, 1), date(2024, 6, 2)],
                "mentions": [3.0, 5.0],
            }
        )

    out = fetch_reddit_for_assets(
        assets, config={"reddit": {"sleep_s": 0}}, fetch_fn=fake
    )
    assert len(out) == 2
    assert (out["source"] == "reddit").all()


def test_x_skipped_without_credentials(monkeypatch):
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("TWITTER_TOKEN", raising=False)
    # Do not pull Abdul's box-secrets into this unit test
    monkeypatch.setattr("cmram.ingest.x_twitter._ensure_bearer_env", lambda: None)
    assert x_credentials_available() is False
    assets = pd.DataFrame(
        [{"asset_id": "gala", "symbol": "GALA", "name": "GALA"}]
    )
    out = fetch_x_for_assets(assets, config={"x": {"sleep_s": 0}})
    assert out.empty


def test_x_plan_limit_stops_batch(monkeypatch):
    from cmram.ingest.x_twitter import XPlanLimitError

    monkeypatch.setenv("X_BEARER_TOKEN", "test-token-not-real")
    monkeypatch.setattr("cmram.ingest.x_twitter._ensure_bearer_env", lambda: None)

    def boom(query, **kwargs):
        raise XPlanLimitError(
            "plan",
            status_code=403,
            error_class="x_api_forbidden_403",
            upgrade_hint="upgrade",
        )

    assets = pd.DataFrame(
        [
            {"asset_id": "a", "symbol": "AAA", "name": "A"},
            {"asset_id": "b", "symbol": "BBB", "name": "B"},
        ]
    )
    with pytest.raises(XPlanLimitError) as ei:
        fetch_x_for_assets(assets, config={"x": {"sleep_s": 0}}, fetch_fn=boom)
    assert ei.value.status_code == 403


def test_ingest_social_records_x_plan_blocker(monkeypatch):
    from unittest.mock import patch
    from cmram.ingest.x_twitter import XPlanLimitError

    assets = pd.DataFrame(
        [{"asset_id": "a", "symbol": "AAA", "name": "A"}]
    )
    err = XPlanLimitError(
        "blocked",
        status_code=402,
        error_class="x_api_payment_required_402",
        upgrade_hint="pay",
    )
    with patch("cmram.ingest.social.fetch_trends_for_assets", return_value=pd.DataFrame(columns=["timestamp","asset_id","source","mentions","unique_users","engagement","community_growth","raw_payload_ref"])):
        with patch("cmram.ingest.social.fetch_reddit_for_assets", return_value=pd.DataFrame(columns=["timestamp","asset_id","source","mentions","unique_users","engagement","community_growth","raw_payload_ref"])):
            with patch("cmram.ingest.social.fetch_x_for_assets", side_effect=err):
                social, report = ingest_social(
                    assets,
                    config={"sources": {"trends": {"enabled": True}, "reddit": {"enabled": True}, "x": {"enabled": True}, "wikipedia": {"enabled": False}}},
                    sources=["trends", "reddit", "x"],
                )
    assert report.get("x_plan_blocker", {}).get("error_class") == "x_api_payment_required_402"
    assert social.empty or "x" not in set(social.get("source", pd.Series(dtype=str)))


def test_ingest_social_combines_mocks():
    assets = pd.DataFrame(
        [
            {"asset_id": "a", "symbol": "AAA", "name": "Asset A"},
            {"asset_id": "b", "symbol": "BBB", "name": "Asset B"},
        ]
    )

    def trends_fn(assets_df, **kwargs):
        return _synth_social(["a"], n_days=10).assign(source="trends")

    # Patch module functions used inside ingest_social
    import cmram.ingest.social as social_mod

    social, report = ingest_social(
        assets,
        config={
            "sources": {
                "trends": {"enabled": True, "sleep_s": 0},
                "reddit": {"enabled": False},
                "x": {"enabled": False},
                "wikipedia": {"enabled": False},
            }
        },
        sources=["trends"],
    )
    # Without monkeypatch on fetch, live call may be empty — use direct unit on combine
    # Instead call with patched fetch_trends_for_assets
    from unittest.mock import patch

    mock_trends = _synth_social(["a", "b"], n_days=15)
    with patch("cmram.ingest.social.fetch_trends_for_assets", return_value=mock_trends):
        with patch(
            "cmram.ingest.social.fetch_reddit_for_assets",
            return_value=pd.DataFrame(columns=mock_trends.columns),
        ):
            with patch(
                "cmram.ingest.social.fetch_x_for_assets",
                return_value=pd.DataFrame(columns=mock_trends.columns),
            ):
                social, report = ingest_social(
                    assets,
                    config={
                        "sources": {
                            "trends": {"enabled": True},
                            "reddit": {"enabled": True},
                            "x": {"enabled": True},
                            "wikipedia": {"enabled": False},
                        }
                    },
                    sources=["trends", "reddit", "x"],
                )
    assert report["n_assets_with_any_source"] == 2
    assert "trends" in report["per_source"]
    flags = source_availability(
        {"sources": {"trends": {"enabled": True}, "x": {"enabled": True}}}
    )
    assert flags["trends_enabled"] is True
    assert "x_credentials" in flags


def test_default_keyword_short_symbol():
    kw = default_keyword({"symbol": "S", "name": "Sonic", "asset_id": "sonic-3"}, template="{symbol}")
    assert "crypto" in kw.lower() or kw == "S crypto"


def _x_env(monkeypatch, token: str = "test-token-not-real") -> None:
    monkeypatch.setenv("X_BEARER_TOKEN", token)
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("TWITTER_TOKEN", raising=False)
    monkeypatch.setattr("cmram.ingest.x_twitter._ensure_bearer_env", lambda: None)


def test_estimate_x_batch_cost_counts_formula():
    est = estimate_x_batch_cost(38, mode="counts", cost_per_request_usd=0.005)
    assert est["n_requests"] == 38
    assert est["requests_per_asset"] == 1
    assert est["estimated_cost_usd"] == pytest.approx(0.19)
    search = estimate_x_batch_cost(10, mode="search", max_pages=1, cost_per_request_usd=0.005)
    assert search["n_requests"] == 10
    assert search["estimated_cost_usd"] == pytest.approx(0.05)


def test_plan_x_batch_default_mode_counts():
    assets = pd.DataFrame(
        [{"asset_id": "a", "symbol": "AAA", "name": "A"}]
    )
    plan, work = plan_x_batch(assets, config={"x": {"sleep_s": 0}})
    assert plan.mode == "counts"
    assert plan.n_assets == 1
    assert plan.n_requests == 1
    assert plan.cost_cap_usd == pytest.approx(1.0)
    assert plan.over_cap is False
    assert len(work) == 1


class _FakeResp:
    def __init__(self, status_code: int, payload: dict, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else __import__("json").dumps(payload)
        self.headers: dict[str, str] = {}

    def json(self):
        return self._payload


class _RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if not self.responses:
            return _FakeResp(500, {"title": "unexpected extra call"})
        return self.responses.pop(0)

    def close(self):
        pass


def test_x_counts_endpoint_mocked(monkeypatch):
    _x_env(monkeypatch)
    payload = {
        "data": [
            {
                "start": "2026-09-10T00:00:00.000Z",
                "end": "2026-09-11T00:00:00.000Z",
                "tweet_count": 12,
            },
            {
                "start": "2026-09-11T00:00:00.000Z",
                "end": "2026-09-12T00:00:00.000Z",
                "tweet_count": 3,
            },
        ],
        "meta": {"total_tweet_count": 15},
    }
    client = _RecordingClient([_FakeResp(200, payload)])
    df = fetch_x_recent_counts("bitcoin crypto", client=client, lookback_days=7)
    assert len(client.calls) == 1
    assert "counts/recent" in client.calls[0]["url"]
    assert client.calls[0]["params"]["granularity"] == "day"
    assert "query" in client.calls[0]["params"]
    assert "max_results" not in client.calls[0]["params"]
    assert len(df) == 2
    assert df["mentions"].sum() == 15.0
    # never leak token into recorded params
    blob = str(client.calls)
    assert "test-token-not-real" not in blob
    assert "Bearer" not in blob


def test_x_counts_402_credits_depleted_stops(monkeypatch):
    from cmram.ingest.x_twitter import XPlanLimitError

    _x_env(monkeypatch)
    body = (
        '{"detail":"credits depleted","status":402,"title":"Payment Required",'
        '"type":"https://api.x.com/2/problems/credits-depleted"}'
    )
    client = _RecordingClient(
        [_FakeResp(402, {"detail": "credits depleted"}, text=body)]
    )
    with pytest.raises(XPlanLimitError) as ei:
        fetch_x_recent_counts("q", client=client)
    assert ei.value.status_code == 402
    assert ei.value.error_class == "x_api_credits_depleted_402"
    assert len(client.calls) == 1


def test_x_cost_cap_stops_before_fetch(monkeypatch):
    _x_env(monkeypatch)
    calls: list[str] = []

    def fake(query, **kwargs):
        calls.append(query)
        return pd.DataFrame({"timestamp": [date(2026, 9, 10)], "mentions": [1.0]})

    assets = pd.DataFrame(
        [{"asset_id": f"a{i}", "symbol": f"S{i}", "name": f"N{i}"} for i in range(10)]
    )
    # 10 × 1 × $0.005 = $0.05 > $0.01
    with pytest.raises(XCostCapError) as ei:
        fetch_x_for_assets(
            assets,
            config={
                "x": {
                    "sleep_s": 0,
                    "mode": "counts",
                    "estimated_cost_cap_usd": 0.01,
                    "cost_per_request_usd": 0.005,
                }
            },
            fetch_fn=fake,
        )
    assert calls == []
    assert ei.value.n_assets == 10
    assert ei.value.n_requests == 10
    assert ei.value.estimated_cost_usd == pytest.approx(0.05)
    assert ei.value.cost_cap_usd == pytest.approx(0.01)
    assert ei.value.mode == "counts"


def test_x_cost_cap_override_allows_fetch(monkeypatch):
    _x_env(monkeypatch)
    calls: list[str] = []

    def fake(query, **kwargs):
        calls.append(query)
        return pd.DataFrame({"timestamp": [date(2026, 9, 10)], "mentions": [1.0]})

    assets = pd.DataFrame(
        [{"asset_id": f"a{i}", "symbol": f"S{i}", "name": f"N{i}"} for i in range(10)]
    )
    out = fetch_x_for_assets(
        assets,
        config={
            "x": {
                "sleep_s": 0,
                "mode": "counts",
                "estimated_cost_cap_usd": 0.01,
                "cost_per_request_usd": 0.005,
            }
        },
        fetch_fn=fake,
        allow_over_cap=True,
    )
    assert len(calls) == 10
    assert len(out) == 10
    assert (out["raw_payload_ref"] == "x:recent_counts").all()


def test_default_mode_is_counts_not_search(monkeypatch):
    _x_env(monkeypatch)
    called = {"counts": 0, "search": 0}

    def counts(query, **kwargs):
        called["counts"] += 1
        return pd.DataFrame({"timestamp": [date(2026, 9, 10)], "mentions": [2.0]})

    def search(query, **kwargs):
        called["search"] += 1
        return pd.DataFrame({"timestamp": [date(2026, 9, 10)], "mentions": [9.0]})

    monkeypatch.setattr("cmram.ingest.x_twitter.fetch_x_recent_counts", counts)
    monkeypatch.setattr("cmram.ingest.x_twitter.fetch_x_recent_mentions", search)
    assets = pd.DataFrame([{"asset_id": "gala", "symbol": "GALA", "name": "GALA"}])
    out = fetch_x_for_assets(assets, config={"x": {"sleep_s": 0}})
    assert called["counts"] == 1
    assert called["search"] == 0
    assert out["raw_payload_ref"].eq("x:recent_counts").all()


def test_search_mode_caps_pages_at_one(monkeypatch):
    _x_env(monkeypatch)
    page1 = {
        "data": [{"created_at": "2026-09-16T12:00:00.000Z", "id": "1", "text": "hi"}],
        "meta": {"next_token": "PAGE2", "result_count": 1},
    }
    page2 = {
        "data": [{"created_at": "2026-09-16T13:00:00.000Z", "id": "2", "text": "nope"}],
        "meta": {"result_count": 1},
    }
    client = _RecordingClient([_FakeResp(200, page1), _FakeResp(200, page2)])
    df = fetch_x_recent_mentions("q", client=client, max_pages=1, max_results=10)
    assert len(client.calls) == 1
    assert "search/recent" in client.calls[0]["url"]
    assert client.calls[0]["params"]["max_results"] == 10
    assert len(df) == 1
    blob = str(client.calls)
    assert "test-token-not-real" not in blob


def test_search_mode_cost_uses_max_pages(monkeypatch):
    _x_env(monkeypatch)
    assets = pd.DataFrame(
        [{"asset_id": f"a{i}", "symbol": f"S{i}", "name": f"N{i}"} for i in range(5)]
    )
    plan, _ = plan_x_batch(
        assets,
        config={
            "x": {
                "mode": "search",
                "cost_per_request_usd": 0.005,
                "estimated_cost_cap_usd": 1.0,
                "search": {"max_pages": 1},
            }
        },
    )
    assert plan.mode == "search"
    assert plan.requests_per_asset == 1
    assert plan.n_requests == 5
    assert plan.estimated_cost_usd == pytest.approx(0.025)


def test_ingest_social_records_x_cost_cap(monkeypatch):
    from unittest.mock import patch

    _x_env(monkeypatch)
    assets = pd.DataFrame(
        [{"asset_id": f"a{i}", "symbol": f"S{i}", "name": f"N{i}"} for i in range(10)]
    )
    called = {"n": 0}

    def boom(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("fetch_x_for_assets must not run when over cost cap")

    with patch("cmram.ingest.social.fetch_trends_for_assets", return_value=pd.DataFrame()):
        with patch("cmram.ingest.social.fetch_reddit_for_assets", return_value=pd.DataFrame()):
            with patch("cmram.ingest.social.fetch_x_for_assets", side_effect=boom):
                social, report = ingest_social(
                    assets,
                    config={
                        "sources": {
                            "trends": {"enabled": False},
                            "reddit": {"enabled": False},
                            "wikipedia": {"enabled": False},
                            "x": {
                                "enabled": True,
                                "mode": "counts",
                                "sleep_s": 0,
                                "estimated_cost_cap_usd": 0.01,
                                "cost_per_request_usd": 0.005,
                            },
                        }
                    },
                    sources=["x"],
                )
    assert called["n"] == 0
    assert report["x_cost_cap_blocker"]["error_class"] == "x_cost_cap_exceeded"
    assert report["x_cost_cap_blocker"]["estimated_cost_usd"] == pytest.approx(0.05)
    assert social.empty


def test_config_social_yaml_counts_defaults():
    from cmram.config import load_social_config

    cfg = load_social_config()
    x = cfg["sources"]["x"]
    assert x["mode"] == "counts"
    assert x["estimated_cost_cap_usd"] == 1.00
    assert x["cost_per_request_usd"] == 0.005
    assert x["search"]["max_pages"] == 1


def test_quiet_rising_damps_crowded_baseline():
    """Quiet→rising: low-base riser beats already-crowded riser; legacy path still available."""
    from datetime import date, timedelta

    def _series(aid: str, base: float, rise: bool, n: int = 60):
        rows = []
        for i in range(n):
            if rise and i > 25:
                ment = base + 0.8 * (i - 25)
            else:
                ment = base + (0.0 if rise else 0.02 * i)
            rows.append(
                {
                    "timestamp": date(2024, 1, 1) + timedelta(days=i),
                    "asset_id": aid,
                    "source": "santiment",
                    "mentions": float(max(ment, 0.1)),
                }
            )
        return rows

    social = pd.DataFrame(
        _series("quiet", 0.5, True)
        + _series("crowd", 80.0, True)
        + _series("flat", 5.0, False)
    )
    qr = narrative_inputs(social, narrative_mode="quiet_rising")
    mid = qr[
        (pd.to_datetime(qr["timestamp"]).dt.date >= date(2024, 2, 1))
        & (pd.to_datetime(qr["timestamp"]).dt.date <= date(2024, 2, 15))
    ]
    q = mid.loc[mid["asset_id"] == "quiet", "n_vel_lowbase"].mean()
    c = mid.loc[mid["asset_id"] == "crowd", "n_vel_lowbase"].mean()
    f = mid.loc[mid["asset_id"] == "flat", "n_vel_lowbase"].mean()
    assert q > c
    assert q > f
    assert mid.loc[mid["asset_id"] == "crowd", "n_quiet_weight"].mean() < 0.05

    legacy = narrative_inputs(social, narrative_mode="legacy")
    assert (legacy["narrative_mode"] == "legacy").all()
    # legacy still assigns non-zero weight to crowded names
    late = legacy[pd.to_datetime(legacy["timestamp"]).dt.date >= date(2024, 2, 20)]
    assert late.loc[late["asset_id"] == "crowd", "n_vel_lowbase"].abs().mean() > 0


def test_model_e_n_only_gap_vs_social_saturation():
    market = _synth_market(["a", "b", "c"])
    mem = _membership(market)
    social = _synth_social(["a", "b", "c"], n_days=80)
    cfg = {
        "models": ["C", "E"],
        "model_version": "test_e",
        "narrative_mode": "quiet_rising",
    }
    feat = compute_features_daily(market, mem, social_daily=social, config=cfg)
    assert set(feat["model"]) == {"C", "E"}
    e = feat[feat["model"] == "E"]
    assert e["score_C"].isna().all()
    assert e["score_V"].isna().all()
    assert e["score_P"].isna().all()
    avail = e["n_available"].fillna(False).astype(bool)
    assert avail.any()
    # MREI is score_N; NSI is social saturation; Gap = N - saturation
    assert np.allclose(
        e.loc[avail, "MREI"].to_numpy(dtype=float),
        e.loc[avail, "score_N"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert np.allclose(
        e.loc[avail, "NSI"].to_numpy(dtype=float),
        e.loc[avail, "nsi_social"].to_numpy(dtype=float),
        equal_nan=True,
    )
    gap = e.loc[avail, "MREI"] - e.loc[avail, "NSI"]
    assert np.allclose(
        e.loc[avail, "Rotation_Gap"].to_numpy(dtype=float),
        gap.to_numpy(dtype=float),
        equal_nan=True,
    )
