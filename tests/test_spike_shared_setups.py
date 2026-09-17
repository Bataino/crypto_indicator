"""Tests for shared boolean setup flags and selection logic."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.backtest.spike_indicators import compute_tech_indicators
from cmram.backtest.spike_shared_setups import (
    SETUP_FLAG_DEFS,
    build_setup_flags,
    compute_flag_support,
    compute_pair_support,
    plain_for_setup,
    select_shared_setups,
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


def test_build_setup_flags_basic_boolean_columns():
    # Uptrend then flat — enough history for EMA/MACD/RSI
    closes = [100.0 + i * 0.5 for i in range(80)]
    mkt = _synth_asset("aaa", date(2026, 1, 1), closes)
    ind = compute_tech_indicators(mkt)
    flags = build_setup_flags(ind, features=None)
    assert "timestamp" in flags.columns and "asset_id" in flags.columns
    for d in SETUP_FLAG_DEFS:
        assert d.flag_id in flags.columns
        assert flags[d.flag_id].dtype == bool
    # Late in an uptrend, price should often be above EMA20
    late = flags.tail(10)
    assert late["price_above_ema20"].any()
    assert late["ema20_gt_ema50"].any()


def test_rsi_cross_up_and_macd_rising_need_lags():
    # Construct a path that dips then recovers so RSI can cross 50
    rng = np.random.default_rng(0)
    base = [100.0] * 40
    dip = [100.0 - i for i in range(1, 16)]  # down
    rise = [85.0 + i * 2.0 for i in range(1, 30)]  # strong up
    closes = base + dip + rise
    mkt = _synth_asset("bbb", date(2026, 1, 1), closes)
    ind = compute_tech_indicators(mkt)
    flags = build_setup_flags(ind)
    # At least one RSI cross-up somewhere in the recovery
    assert flags["rsi_cross_up_50"].sum() >= 1
    # MACD hist pos rising should be True on some recovery days
    assert flags["macd_hist_pos_rising"].sum() >= 1


def test_reclaim_ema20_within_3_days():
    # Flat below a rising path then jump above — reclaim should fire
    closes = [50.0] * 60 + [50.0, 51.0, 52.0, 60.0, 61.0, 62.0]
    mkt = _synth_asset("ccc", date(2026, 1, 1), closes)
    ind = compute_tech_indicators(mkt)
    flags = build_setup_flags(ind)
    # After the jump, reclaim flag should appear within a few bars
    assert flags["reclaim_ema20_3d"].sum() >= 1


def test_feature_dependent_flags_when_scores_present():
    closes = [100.0 + i * 0.2 for i in range(60)]
    mkt = _synth_asset("ddd", date(2026, 1, 1), closes)
    ind = compute_tech_indicators(mkt)
    # Fake features Model C
    feat_rows = []
    for i, ts in enumerate(ind["timestamp"]):
        feat_rows.append(
            {
                "timestamp": ts,
                "asset_id": "ddd",
                "model": "C",
                "MREI": 50.0,
                "NSI": 40.0,
                "Rotation_Gap": 15.0 if i > 30 else -5.0,
                "score_C": 30.0 if i > 40 else 60.0,
                "score_V": 40.0 + (i % 5),
                "score_N": 30.0 + (2.0 if i > 45 else 0.0) + (i % 3),
                "score_P": 40.0 + (i % 4),
            }
        )
    features = pd.DataFrame(feat_rows)
    flags = build_setup_flags(ind, features, model="C")
    assert flags["gap_gt_0"].any()
    assert flags["gap_gt_10"].any()
    # quiet_rising_n / compression may or may not fire depending on path —
    # columns must exist and be bool
    assert flags["quiet_rising_n"].dtype == bool
    assert flags["compression_recovering"].dtype == bool


def test_compute_flag_support_and_lift():
    spike = pd.DataFrame(
        {
            "price_above_ema20": [True, True, True, False],
            "rvol_gt_1": [True, False, True, True],
        }
    )
    ctrl = pd.DataFrame(
        {
            "price_above_ema20": [True, False, False, False, False, False],
            "rvol_gt_1": [True, True, True, False, False, False],
        }
    )
    sup = compute_flag_support(spike, ctrl, flag_ids=["price_above_ema20", "rvol_gt_1"])
    assert len(sup) == 2
    row = sup.loc[sup["setup_id"] == "price_above_ema20"].iloc[0]
    assert row["spike_support"] == pytest.approx(0.75)
    assert row["ctrl_support"] == pytest.approx(1.0 / 6.0)
    assert row["lift"] == pytest.approx(0.75 / (1.0 / 6.0))


def test_compute_pair_support():
    spike = pd.DataFrame(
        {
            "a": [True, True, False, True],
            "b": [True, False, True, True],
            "c": [False, False, False, True],
        }
    )
    ctrl = pd.DataFrame(
        {
            "a": [True, False, False, False],
            "b": [True, True, False, False],
            "c": [False, False, False, False],
        }
    )
    pairs = compute_pair_support(spike, ctrl, top_singles=["a", "b", "c"])
    assert len(pairs) == 3  # C(3,2)
    ab = pairs.loc[pairs["setup_id"] == "a+b"].iloc[0]
    # spike: rows 0 and 3 → 2/4 = 0.5
    assert ab["spike_support"] == pytest.approx(0.5)


def test_select_shared_setups_lift_and_min_count():
    singles = pd.DataFrame(
        [
            {
                "setup_id": "good",
                "flag_ids": "good",
                "plain_english": "Good setup",
                "is_pair": False,
                "spike_count": 10,
                "spike_n": 20,
                "spike_support": 0.5,
                "ctrl_count": 10,
                "ctrl_n": 100,
                "ctrl_support": 0.1,
                "lift": 5.0,
            },
            {
                "setup_id": "weak",
                "flag_ids": "weak",
                "plain_english": "Weak",
                "is_pair": False,
                "spike_count": 8,
                "spike_n": 20,
                "spike_support": 0.4,
                "ctrl_count": 35,
                "ctrl_n": 100,
                "ctrl_support": 0.35,
                "lift": 0.4 / 0.35,  # < 1.2
            },
            {
                "setup_id": "rare",
                "flag_ids": "rare",
                "plain_english": "Rare",
                "is_pair": False,
                "spike_count": 2,
                "spike_n": 20,
                "spike_support": 0.1,
                "ctrl_count": 1,
                "ctrl_n": 100,
                "ctrl_support": 0.01,
                "lift": 10.0,  # high lift but too few hits
            },
            {
                "setup_id": "ok2",
                "flag_ids": "ok2",
                "plain_english": "Ok2",
                "is_pair": False,
                "spike_count": 6,
                "spike_n": 20,
                "spike_support": 0.3,
                "ctrl_count": 10,
                "ctrl_n": 100,
                "ctrl_support": 0.1,
                "lift": 3.0,
            },
            {
                "setup_id": "ok3",
                "flag_ids": "ok3",
                "plain_english": "Ok3",
                "is_pair": False,
                "spike_count": 7,
                "spike_n": 20,
                "spike_support": 0.35,
                "ctrl_count": 12,
                "ctrl_n": 100,
                "ctrl_support": 0.12,
                "lift": 0.35 / 0.12,
            },
        ]
    )
    selected = select_shared_setups(
        singles, None, lift_min=1.2, min_spike_count=5, top_n=6, min_n=3
    )
    ids = [s.setup_id for s in selected]
    assert "good" in ids
    assert "ok2" in ids
    assert "ok3" in ids
    assert "rare" not in ids  # below min count
    assert "weak" not in ids  # below lift
    assert ids[0] == "good"  # highest lift first


def test_plain_for_setup_combo():
    text = plain_for_setup(("price_above_ema20", "rvol_gt_1"))
    assert "Price above EMA20" in text
    assert "AND" in text
    assert "Relative volume" in text
