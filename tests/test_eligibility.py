"""Unit tests for universe eligibility filter (no API)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from cmram.config import load_eligibility_config
from cmram.universe.eligibility import (
    evaluate_asset,
    evaluate_assets_frame,
    filter_market_by_eligibility,
    ineligible_asset_ids,
    load_eligibility_rules,
)
from cmram.universe.membership import build_universe_membership
from cmram.config import load_thresholds_config, load_universe_config


def _cfg(**overrides) -> dict:
    cfg = dict(load_eligibility_config())
    cfg.update(overrides)
    return cfg


def test_load_eligibility_config_has_denylist():
    cfg = load_eligibility_config()
    assert cfg.get("enabled") is True
    assert "nvidia-bstocks" in (cfg.get("deny_ids") or [])
    assert any("xstock" in str(k).lower() for k in (cfg.get("deny_keywords") or []))


def test_deny_id_exact():
    d = evaluate_asset("usd-coinvertible", eligibility_config=_cfg())
    assert d.eligible is False
    assert d.reason == "deny_id"
    assert "ineligible:deny_id" in (d.reason_excluded or "")


def test_allow_ids_override():
    d = evaluate_asset(
        "usd-coinvertible",
        eligibility_config=_cfg(allow_ids=["usd-coinvertible"]),
    )
    assert d.eligible is True


def test_keyword_tokenized_stock_in_name():
    d = evaluate_asset(
        "some-rando",
        name="Foo (Ondo Tokenized Stock)",
        eligibility_config=_cfg(deny_ids=[]),
    )
    assert d.eligible is False
    assert d.reason == "deny_keyword"


def test_id_substring_xstock():
    d = evaluate_asset(
        "circle-xstock",
        eligibility_config=_cfg(deny_ids=[]),
    )
    assert d.eligible is False
    assert d.reason in {"deny_id_substring", "deny_keyword"}


def test_category_stablecoins():
    d = evaluate_asset(
        "my-stable",
        category="stablecoins",
        eligibility_config=_cfg(deny_ids=[]),
    )
    assert d.eligible is False
    assert d.reason == "deny_category"
    assert d.detail == "stablecoins"


def test_category_list_pipe_separated():
    d = evaluate_asset(
        "x",
        category="layer-1|usd-stablecoin",
        eligibility_config=_cfg(deny_ids=[]),
    )
    assert d.eligible is False
    assert d.reason == "deny_category"


def test_normal_microcap_passes():
    d = evaluate_asset(
        "coti",
        symbol="COTI",
        name="COTI",
        category=None,
        eligibility_config=_cfg(),
    )
    assert d.eligible is True
    assert d.reason_excluded is None


def test_disabled_passes_everything():
    d = evaluate_asset(
        "nvidia-bstocks",
        eligibility_config=_cfg(enabled=False),
    )
    assert d.eligible is True


def test_evaluate_assets_frame_and_ineligible_map():
    assets = pd.DataFrame(
        [
            {"asset_id": "coti", "symbol": "COTI", "name": "coti", "category": None},
            {
                "asset_id": "eurite",
                "symbol": "EURI",
                "name": "Eurite",
                "category": None,
            },
            {
                "asset_id": "alphabet-class-a-ondo-tokenized-stock",
                "symbol": "GOOGLON",
                "name": "Alphabet Class A (Ondo Tokenized Stock)",
                "category": None,
            },
        ]
    )
    ev = evaluate_assets_frame(assets, _cfg())
    assert set(ev.loc[~ev["eligible"], "asset_id"]) == {
        "eurite",
        "alphabet-class-a-ondo-tokenized-stock",
    }
    bad = ineligible_asset_ids(assets, _cfg())
    assert "coti" not in bad
    assert "eurite" in bad


def test_filter_market_drops_ineligible():
    market = pd.DataFrame(
        {
            "timestamp": [date(2024, 1, 1)] * 3,
            "asset_id": ["coti", "nvidia-bstocks", "eurite"],
            "close": [1.0, 1.0, 1.0],
            "volume_usd": [1e5, 1e5, 1e5],
            "market_cap_usd": [2e7, 2e7, 2e7],
        }
    )
    assets = pd.DataFrame(
        {
            "asset_id": ["coti", "nvidia-bstocks", "eurite"],
            "symbol": ["COTI", "NVDAB", "EURI"],
            "name": ["coti", "NVIDIA (bStocks Tokenized Stock)", "Eurite"],
            "category": [None, None, None],
        }
    )
    filtered, excluded = filter_market_by_eligibility(market, assets, _cfg())
    assert set(filtered["asset_id"]) == {"coti"}
    assert set(excluded) == {"nvidia-bstocks", "eurite"}


def test_membership_excludes_ineligible_and_sets_attrs():
    n = 100
    start = date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]

    def bars(aid: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": dates,
                "asset_id": aid,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume_usd": 200_000.0,
                "market_cap_usd": 20_000_000.0,
            }
        )

    market = pd.concat(
        [bars("coti"), bars("usd-coinvertible"), bars("circle-xstock")],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        [
            {"asset_id": "coti", "symbol": "COTI", "name": "coti", "category": None},
            {
                "asset_id": "usd-coinvertible",
                "symbol": "USD-COINVERTIBLE",
                "name": "usd-coinvertible",
                "category": None,
            },
            {
                "asset_id": "circle-xstock",
                "symbol": "CRCLX",
                "name": "Circle xStock",
                "category": None,
            },
        ]
    )
    mem = build_universe_membership(
        market,
        load_universe_config(),
        load_thresholds_config(),
        assets=assets,
        eligibility_config=_cfg(),
    )
    assert set(mem["asset_id"]) == {"coti"}
    excluded = mem.attrs.get("eligibility_excluded") or {}
    assert "usd-coinvertible" in excluded
    assert "circle-xstock" in excluded


def test_load_eligibility_rules_normalizes():
    rules = load_eligibility_rules(
        {"enabled": True, "deny_ids": ["Foo"], "deny_keywords": ["Bar"]}
    )
    assert "foo" in rules["deny_ids"]
    assert rules["deny_keywords"] == ["bar"]
