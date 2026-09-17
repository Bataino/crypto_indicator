"""Tests for Model E N-only validation helpers and group benchmarks."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.backtest.engine import run_backtest
from cmram.backtest.n_only_validation import (
    PREREG_TAU_G,
    PREREG_TAU_M,
    filter_prereg_e,
    leave_one_asset_out,
    per_asset_summary,
    summarize_paths,
)


def _bars(asset: str, n: int, *, start: date, drift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    px = 1.0
    for i in range(n):
        shock = drift + float(rng.normal(0, 0.01))
        c = px * (1.0 + shock)
        rows.append(
            {
                "timestamp": start + timedelta(days=i),
                "asset_id": asset,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume_usd": 100_000.0,
                "market_cap_usd": 20_000_000.0,
            }
        )
        px = c
    return pd.DataFrame(rows)


def test_filter_prereg_e():
    sig = pd.DataFrame(
        {
            "model": ["E", "E", "C", "E"],
            "tau_m": [50.0, 60.0, 50.0, 50.0],
            "tau_g": [20.0, 20.0, 20.0, 20.0],
            "band": ["A_5_50", "A_5_50", "A_5_50", "B_10_100"],
            "signal_id": ["1", "2", "3", "4"],
        }
    )
    out = filter_prereg_e(sig, bands=["A_5_50"])
    assert list(out["signal_id"]) == ["1"]
    assert float(out.iloc[0]["tau_m"]) == PREREG_TAU_M
    assert float(out.iloc[0]["tau_g"]) == PREREG_TAU_G


def test_group_median_mean_and_multi_seed_random():
    start = date(2024, 1, 1)
    n = 80
    # strong rises; weak falls — group median/mean between them
    market = pd.concat(
        [
            _bars("strong", n, start=start, drift=0.02, seed=1),
            _bars("mid", n, start=start, drift=0.005, seed=2),
            _bars("weak", n, start=start, drift=-0.01, seed=3),
        ],
        ignore_index=True,
    )
    # membership all days from day 40
    mem_rows = []
    for i in range(40, n):
        ts = start + timedelta(days=i)
        for a in ("strong", "mid", "weak"):
            mem_rows.append(
                {
                    "timestamp": ts,
                    "asset_id": a,
                    "band": "A_5_50",
                    "market_cap_usd": 20e6,
                    "history_days": i + 1,
                    "liquidity_pass": True,
                    "amihud": 1e-9,
                    "reason_excluded": None,
                }
            )
    membership = pd.DataFrame(mem_rows)
    # One signal on strong at day 50
    sig_ts = start + timedelta(days=50)
    signals = pd.DataFrame(
        [
            {
                "signal_id": "s1",
                "timestamp": sig_ts,
                "asset_id": "strong",
                "band": "A_5_50",
                "model": "E",
                "tau_m": 50.0,
                "tau_g": 20.0,
                "mrei": 90.0,
                "nsi": 10.0,
                "rotation_gap": 80.0,
            }
        ]
    )
    bt = run_backtest(
        signals,
        market,
        horizons_days=[7],
        benchmarks=["group_median", "group_mean", "random"],
        random_seeds=[42, 43, 44],
        universe_membership=membership,
    )
    assert set(bt["benchmark_id"]) == {"group_median", "group_mean", "random"}
    # strong should beat group median/mean at h=7
    for bench in ("group_median", "group_mean"):
        row = bt[(bt["benchmark_id"] == bench) & (bt["horizon_d"] == 7)].iloc[0]
        assert row["excess_ret"] > 0
    assert bt.attrs.get("random_seeds") == [42, 43, 44]


def test_per_asset_and_loo():
    signals = pd.DataFrame(
        {
            "signal_id": ["a", "b", "c", "d"],
            "band": ["A_5_50"] * 4,
            "model": ["E"] * 4,
            "tau_m": [50.0] * 4,
            "tau_g": [20.0] * 4,
            "asset_id": ["x", "x", "y", "z"],
        }
    )
    results = pd.DataFrame(
        {
            "signal_id": ["a", "b", "c", "d"],
            "horizon_d": [7, 7, 7, 7],
            "ret": [0.10, 0.08, -0.02, 0.01],
            "mfe": [0.1] * 4,
            "mae": [-0.01] * 4,
            "benchmark_id": ["group_median"] * 4,
            "excess_ret": [0.05, 0.04, -0.03, 0.00],
        }
    )
    pa = per_asset_summary(signals, results)
    assert list(pa["asset_id"])[0] == "x"
    assert int(pa.loc[pa["asset_id"] == "x", "n_signals"].iloc[0]) == 2

    grid = summarize_paths(signals, results, pool_bands=True)
    assert len(grid) == 1
    assert grid.iloc[0]["n"] == 4

    loo = leave_one_asset_out(signals, results)
    assert set(loo["dropped_asset"]) == {"x", "y", "z"}
    # Dropping x (best) should lower median excess vs full
    full = float(loo.iloc[0]["full_median_excess"])
    after_x = float(loo.loc[loo["dropped_asset"] == "x", "median_excess"].iloc[0])
    assert after_x <= full + 1e-12
