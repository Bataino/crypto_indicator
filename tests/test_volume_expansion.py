"""Tests for volume-expansion candidates and hot-vol × ret_3d summary."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.backtest.spike_indicators import compute_tech_indicators
from cmram.backtest.spike_shared_setups import build_setup_flags
from cmram.backtest.volume_expansion import (
    VOLUME_CANDIDATES,
    compute_ret_3d,
    candidates_as_shared_setups,
    summarize_hot_vol_ret3d,
)


def _synth_asset(
    asset_id: str,
    start: date,
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        vol = 1_000_000.0 if volumes is None else float(volumes[i])
        rows.append(
            {
                "timestamp": d,
                "asset_id": asset_id,
                "open": c,
                "high": c * 1.01,
                "low": c * 0.99,
                "close": c,
                "volume_usd": vol,
            }
        )
    return pd.DataFrame(rows)


def test_new_ema_distance_flags_exist():
    closes = [100.0 + i * 0.3 for i in range(80)]
    vols = [1e6] * 70 + [3e6] * 10  # late volume expansion
    mkt = _synth_asset("vea", date(2026, 1, 1), closes, vols)
    ind = compute_tech_indicators(mkt)
    flags = build_setup_flags(ind)
    for col in (
        "dist_ema20_le_10pct",
        "dist_ema20_le_5pct",
        "near_ema20_m5_p10",
        "rvol_gt_1_5",
    ):
        assert col in flags.columns
        assert flags[col].dtype == bool
    # Late bars should often show hot volume after the spike in volume
    assert flags["rvol_gt_1_5"].tail(5).any()


def test_near_ema_band_logic_on_synthetic_dist():
    # Build indicators with known dist by forcing flat then mild move
    closes = [100.0] * 60 + [100.0, 101.0, 102.0, 103.0, 104.0]
    mkt = _synth_asset("veb", date(2026, 1, 1), closes)
    ind = compute_tech_indicators(mkt)
    flags = build_setup_flags(ind)
    # On a nearly flat series, dist to EMA20 should be small → near band true
    late = flags.tail(3)
    assert late["near_ema20_m5_p10"].all() or late["near_ema20_m5_p10"].any()
    assert late["dist_ema20_le_10pct"].all() or late["dist_ema20_le_10pct"].any()


def test_compute_ret_3d():
    closes = [100.0, 101.0, 102.0, 110.0, 111.0]
    mkt = _synth_asset("vec", date(2026, 1, 1), closes)
    ret = compute_ret_3d(mkt)
    assert list(ret.columns) == ["timestamp", "asset_id", "ret_3d"]
    # day index 3: 110/100 - 1 = 0.10
    row = ret.iloc[3]
    assert row["ret_3d"] == pytest.approx(0.10)
    assert pd.isna(ret.iloc[0]["ret_3d"])
    assert pd.isna(ret.iloc[2]["ret_3d"])


def test_summarize_hot_vol_ret3d_shares():
    df = pd.DataFrame(
        {
            "rvol_30": [2.0, 2.0, 0.5, 3.0, 1.6],
            "ret_3d": [0.05, -0.02, 0.10, 0.01, np.nan],
        }
    )
    # hot with ret: rows 0,1,3 → 2 up, 1 down (row 4 hot but nan ret excluded)
    out = summarize_hot_vol_ret3d(df)
    assert out["n_hot_vol"] == 4
    assert out["n_hot_with_ret"] == 3
    assert out["n_ret_gt_0"] == 2
    assert out["n_ret_le_0"] == 1
    assert out["share_ret_gt_0"] == pytest.approx(2 / 3)
    assert out["share_ret_le_0"] == pytest.approx(1 / 3)


def test_candidates_include_baseline_and_abcd():
    setups = candidates_as_shared_setups()
    ids = [s.setup_id for s in setups]
    assert ids[0] == "baseline_rvol"
    assert "A_not_mooned_10" in ids
    assert "A_not_mooned_5" in ids
    assert "B_near_ema20" in ids
    assert "C_macd_rising" in ids
    assert "D_rsi_45_65" in ids
    assert len(VOLUME_CANDIDATES) == 6
    # Every candidate includes the volume core
    for s in setups:
        assert "rvol_gt_1_5" in s.flag_ids
