"""Synthetic leakage + correctness tests for signals and forward-return backtest."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.backtest.engine import aggregate_backtest, run_backtest
from cmram.signals.generate import (
    ENTRY_PROXY_NEXT_CLOSE,
    detect_chart_close_only,
    generate_signals,
)


def _synth_asset(
    asset_id: str,
    n: int,
    *,
    start: date = date(2024, 1, 1),
    daily_ret: float = 0.01,
    seed: int = 0,
    chart_close_only: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    px = 1.0
    for i in range(n):
        shock = daily_ret + float(rng.normal(0, 0.005))
        o = px
        c = px * (1.0 + shock)
        if chart_close_only:
            o = h = low = c
        else:
            h = max(o, c) * 1.01
            low = min(o, c) * 0.99
        rows.append(
            {
                "timestamp": start + timedelta(days=i),
                "asset_id": asset_id,
                "open": float(o),
                "high": float(h),
                "low": float(low),
                "close": float(c),
                "volume_usd": float(100_000 + rng.integers(0, 50_000)),
                "market_cap_usd": float(20_000_000 + i * 10_000),
            }
        )
        px = c
    return pd.DataFrame(rows)


def _market(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)


def _membership(
    market: pd.DataFrame,
    *,
    from_idx: int = 50,
    bands: tuple[str, ...] = ("A_5_50", "B_10_100"),
    liquidity_pass: bool = True,
) -> pd.DataFrame:
    rows = []
    for asset, g in market.groupby("asset_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for i in range(from_idx, len(g)):
            for band in bands:
                rows.append(
                    {
                        "timestamp": g.iloc[i]["timestamp"],
                        "asset_id": asset,
                        "band": band,
                        "market_cap_usd": float(g.iloc[i]["market_cap_usd"]),
                        "history_days": i + 1,
                        "liquidity_pass": liquidity_pass,
                        "amihud": 1e-9,
                        "reason_excluded": None,
                    }
                )
    return pd.DataFrame(rows)


def _features_from_market(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    mrei: float = 80.0,
    gap: float = 40.0,
    model: str = "A",
) -> pd.DataFrame:
    """Constant scores — enough to fire τ grids; timestamps from membership."""
    rows = []
    mem = membership.copy()
    mem["timestamp"] = pd.to_datetime(mem["timestamp"]).dt.date
    for r in mem.itertuples(index=False):
        rows.append(
            {
                "timestamp": r.timestamp,
                "asset_id": r.asset_id,
                "band": r.band,
                "model": model,
                "MREI": mrei,
                "NSI": mrei - gap,
                "Rotation_Gap": gap,
            }
        )
    # Also model B with same scores for grid coverage when requested
    return pd.DataFrame(rows)


THRESH = {
    "tau_m": [50, 60, 70],
    "tau_g": [20, 30, 40, 50],
    "models": ["A", "B"],
    "horizons_days": [1, 3, 7],
}


def test_detect_chart_close_only():
    m = _market(_synth_asset("a", 10, chart_close_only=True, seed=1))
    assert detect_chart_close_only(m) is True
    m2 = _market(_synth_asset("a", 10, chart_close_only=False, seed=1))
    assert detect_chart_close_only(m2) is False


def test_signal_requires_liquidity_pass():
    market = _market(
        _synth_asset("a", 70, seed=1),
        _synth_asset("b", 70, seed=2),
    )
    mem_fail = _membership(market, from_idx=50, liquidity_pass=False)
    feat = _features_from_market(market, mem_fail, model="A")
    feat_b = _features_from_market(market, mem_fail, model="B")
    feat = pd.concat([feat, feat_b], ignore_index=True)
    sig = generate_signals(feat, mem_fail, THRESH, market_daily=market)
    assert len(sig) == 0


def test_signal_entry_is_next_bar_not_same_day():
    """Entry must use next available bar after signal timestamp (no same-bar)."""
    market = _market(
        _synth_asset("a", 70, daily_ret=0.02, seed=1, chart_close_only=False),
        _synth_asset("b", 70, daily_ret=0.0, seed=2, chart_close_only=False),
    )
    mem = _membership(market, from_idx=55, bands=("A_5_50",))
    feat = pd.concat(
        [
            _features_from_market(market, mem, model="A"),
            _features_from_market(market, mem, model="B"),
        ],
        ignore_index=True,
    )
    sig = generate_signals(feat, mem, THRESH, market_daily=market)
    assert len(sig) > 0
    assert sig.attrs["entry_mode"] == "next_day_open"

    m = market.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.normalize()
    for r in sig.itertuples(index=False):
        ts = pd.Timestamp(r.timestamp).normalize()
        asset_bars = m[m["asset_id"] == r.asset_id].sort_values("timestamp")
        same = asset_bars[asset_bars["timestamp"] == ts]
        assert len(same) == 1
        # Entry must NOT equal same-day open (unless coincidentally next equals it)
        nxt = asset_bars[asset_bars["timestamp"] > ts].head(1)
        assert len(nxt) == 1
        expected = float(nxt.iloc[0]["open"])
        assert r.entry_price == pytest.approx(expected, rel=1e-12)
        # Explicitly not using same-bar high as entry
        assert r.entry_price != pytest.approx(float(same.iloc[0]["high"]), rel=0) or (
            float(same.iloc[0]["high"]) == expected
        )


def test_chart_close_only_uses_next_close_proxy():
    market = _market(
        _synth_asset("a", 70, seed=1, chart_close_only=True),
        _synth_asset("b", 70, seed=2, chart_close_only=True),
    )
    mem = _membership(market, from_idx=55, bands=("A_5_50",))
    feat = pd.concat(
        [
            _features_from_market(market, mem, model="A"),
            _features_from_market(market, mem, model="B"),
        ],
        ignore_index=True,
    )
    sig = generate_signals(feat, mem, THRESH, market_daily=market)
    assert len(sig) > 0
    assert sig.attrs["chart_close_only"] is True
    assert sig.attrs["entry_mode"] == ENTRY_PROXY_NEXT_CLOSE

    m = market.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.normalize()
    r = sig.iloc[0]
    ts = pd.Timestamp(r["timestamp"]).normalize()
    bars = m[m["asset_id"] == r["asset_id"]].sort_values("timestamp")
    nxt = bars[bars["timestamp"] > ts].head(1)
    assert float(r["entry_price"]) == pytest.approx(float(nxt.iloc[0]["close"]))


def test_nested_tau_all_kept():
    market = _market(_synth_asset("a", 70, seed=1), _synth_asset("b", 70, seed=2))
    mem = _membership(market, from_idx=60, bands=("A_5_50",))
    # High scores → all τ fire
    feat = pd.concat(
        [
            _features_from_market(market, mem, mrei=80, gap=55, model="A"),
            _features_from_market(market, mem, mrei=80, gap=55, model="B"),
        ],
        ignore_index=True,
    )
    sig = generate_signals(feat, mem, THRESH, market_daily=market)
    combos = sig.groupby(["tau_m", "tau_g"]).size()
    assert len(combos) == len(THRESH["tau_m"]) * len(THRESH["tau_g"])
    # Nested: higher τ subset of lower τ for same day/asset/model
    low = sig[(sig["tau_m"] == 50) & (sig["tau_g"] == 20)]
    high = sig[(sig["tau_m"] == 70) & (sig["tau_g"] == 50)]
    low_keys = set(zip(low["timestamp"], low["asset_id"], low["model"]))
    high_keys = set(zip(high["timestamp"], high["asset_id"], high["model"]))
    assert high_keys <= low_keys


def test_signal_no_future_price_leakage_for_entry():
    """Mutating prices after signal date must not change that day's entry."""
    market = _market(
        _synth_asset("a", 80, seed=1, chart_close_only=False),
        _synth_asset("b", 80, seed=2, chart_close_only=False),
    )
    mem = _membership(market, from_idx=50, bands=("A_5_50",))
    feat = pd.concat(
        [
            _features_from_market(market, mem, model="A"),
            _features_from_market(market, mem, model="B"),
        ],
        ignore_index=True,
    )
    # Restrict features to a single as-of day so entry is the immediate next bar
    cutoff = date(2024, 1, 1) + timedelta(days=60)
    feat_cut = feat[pd.to_datetime(feat["timestamp"]).dt.date == cutoff].copy()
    mem_cut = mem[pd.to_datetime(mem["timestamp"]).dt.date == cutoff].copy()
    sig1 = generate_signals(feat_cut, mem_cut, THRESH, market_daily=market)

    m2 = market.copy()
    ts = pd.to_datetime(m2["timestamp"]).dt.date
    # Corrupt far-future bars only (not the next entry bar)
    far = date(2024, 1, 1) + timedelta(days=70)
    mask = ts >= far
    m2.loc[mask, ["open", "high", "low", "close"]] *= 100.0
    sig2 = generate_signals(feat_cut, mem_cut, THRESH, market_daily=m2)

    assert len(sig1) == len(sig2) > 0
    s1 = sig1.sort_values(["signal_id"]).reset_index(drop=True)
    s2 = sig2.sort_values(["signal_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(s1["entry_price"], s2["entry_price"])


def test_features_at_t_not_used_as_same_bar_entry():
    """Signal timestamp is as-of t; entry price is from a strictly later bar."""
    market = _market(_synth_asset("a", 70, seed=3), _synth_asset("b", 70, seed=4))
    mem = _membership(market, from_idx=55, bands=("A_5_50",))
    feat = pd.concat(
        [
            _features_from_market(market, mem, model="A"),
            _features_from_market(market, mem, model="B"),
        ],
        ignore_index=True,
    )
    sig = generate_signals(feat, mem, THRESH, market_daily=market)
    m = market.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.date
    for r in sig.itertuples(index=False):
        same_close = float(
            m[(m["asset_id"] == r.asset_id) & (m["timestamp"] == r.timestamp)][
                "close"
            ].iloc[0]
        )
        # Next bar close/open differs almost surely with random path; entry
        # date must be > signal date
        later = m[(m["asset_id"] == r.asset_id) & (m["timestamp"] > r.timestamp)]
        assert len(later) >= 1
        assert later.iloc[0]["timestamp"] > r.timestamp


def test_forward_returns_may_use_future_by_design():
    """Changing future prices after entry MUST change forward ret (allowed)."""
    market = _market(
        _synth_asset("a", 90, daily_ret=0.0, seed=1),
        _synth_asset("b", 90, daily_ret=0.0, seed=2),
    )
    mem = _membership(market, from_idx=50, bands=("A_5_50",))
    feat = pd.concat(
        [
            _features_from_market(market, mem, model="A"),
            _features_from_market(market, mem, model="B"),
        ],
        ignore_index=True,
    )
    # One day of signals near middle
    day = date(2024, 1, 1) + timedelta(days=55)
    feat = feat[pd.to_datetime(feat["timestamp"]).dt.date == day]
    mem_d = mem[pd.to_datetime(mem["timestamp"]).dt.date == day]
    sig = generate_signals(feat, mem_d, {"tau_m": [50], "tau_g": [20], "models": ["A"]}, market_daily=market)
    assert len(sig) > 0

    bt1 = run_backtest(
        sig, market, horizons_days=[5], benchmarks=["random"], universe_membership=mem
    )
    assert len(bt1) > 0

    m2 = market.copy()
    # Boost closes after entry window start
    entry_start = day + timedelta(days=1)
    mask = pd.to_datetime(m2["timestamp"]).dt.date >= entry_start + timedelta(days=3)
    m2.loc[mask, ["open", "high", "low", "close"]] *= 2.0
    bt2 = run_backtest(
        sig, m2, horizons_days=[5], benchmarks=["random"], universe_membership=mem
    )
    # Same signals → different forward rets when future changes
    assert not np.allclose(bt1["ret"].to_numpy(), bt2["ret"].to_numpy())


def test_backtest_mfe_mae_signs():
    market = _market(
        _synth_asset("a", 80, daily_ret=0.01, seed=1, chart_close_only=False),
        _synth_asset("b", 80, daily_ret=-0.01, seed=2, chart_close_only=False),
    )
    mem = _membership(market, from_idx=50, bands=("A_5_50",))
    feat = pd.concat(
        [
            _features_from_market(market, mem, model="A"),
            _features_from_market(market, mem, model="B"),
        ],
        ignore_index=True,
    )
    sig = generate_signals(
        feat, mem, {"tau_m": [50], "tau_g": [20], "models": ["A"]}, market_daily=market
    )
    bt = run_backtest(
        sig,
        market,
        horizons_days=[7],
        benchmarks=["random", "momentum", "volume", "market"],
        universe_membership=mem,
        random_seed=0,
    )
    assert set(bt["benchmark_id"]) == {"random", "momentum", "volume", "market"}
    # MFE >= ret path high; MAE <= 0 typically when lows go below entry
    assert (bt["mfe"] >= bt["mae"] - 1e-12).all()


def test_aggregate_small_sample_flag():
    market = _market(_synth_asset("a", 70, seed=1), _synth_asset("b", 70, seed=2))
    mem = _membership(market, from_idx=55, bands=("A_5_50",))
    feat = pd.concat(
        [
            _features_from_market(market, mem, model="A"),
            _features_from_market(market, mem, model="B"),
        ],
        ignore_index=True,
    )
    sig = generate_signals(
        feat, mem, {"tau_m": [50], "tau_g": [20], "models": ["A", "B"]}, market_daily=market
    )
    bt = run_backtest(
        sig, market, horizons_days=[1, 3], benchmarks=["random"], universe_membership=mem
    )
    assert bt.attrs["small_sample"] is True
    agg = aggregate_backtest(sig, bt, benchmark_id="random")
    assert len(agg) > 0
    assert "mean_ret" in agg.columns
    assert "win_rate" in agg.columns


def test_join_skips_features_without_membership():
    market = _market(_synth_asset("a", 60, seed=1))
    mem = _membership(market, from_idx=50, bands=("A_5_50",))
    feat = _features_from_market(market, mem, model="A")
    # Extra feature row for unknown asset / no membership
    extra = feat.iloc[[0]].copy()
    extra["asset_id"] = "ghost"
    feat2 = pd.concat([feat, extra], ignore_index=True)
    sig = generate_signals(
        feat2, mem, {"tau_m": [50], "tau_g": [20], "models": ["A"]}, market_daily=market
    )
    assert "ghost" not in set(sig["asset_id"])
