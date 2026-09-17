"""Tests for +50% spike detection helpers."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.backtest.spike_detect import (
    dedupe_non_overlapping,
    detect_spike_candidates,
    detect_spikes,
    forward_max_return,
    forward_max_return_by_asset,
    split_spikes_by_cut,
)


def _synth_asset(
    asset_id: str,
    start: date,
    closes: list[float],
) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        rows.append(
            {
                "timestamp": d,
                "asset_id": asset_id,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume_usd": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_forward_max_return_basic():
    close = pd.Series([100.0, 100.0, 100.0, 160.0, 100.0])
    # From i=0: future max of 100,100,160,100 = 160 → +60%
    out = forward_max_return(close, window=3)
    assert out.iloc[0] == pytest.approx(0.60)
    # From i=1: futures 100,160,100 → 160 → +60%
    assert out.iloc[1] == pytest.approx(0.60)
    # From i=2: futures 160,100 → 160 → +60%
    assert out.iloc[2] == pytest.approx(0.60)
    # From i=3: close=160, futures 100 → 100/160 - 1 = -37.5%
    assert out.iloc[3] == pytest.approx(100.0 / 160.0 - 1.0)
    # From i=4: no future → NaN
    assert np.isnan(out.iloc[4])


def test_forward_max_return_rejects_bad_window():
    with pytest.raises(ValueError):
        forward_max_return(pd.Series([1.0, 2.0]), window=0)


def test_detect_spike_exactly_50_percent():
    """+50% within 7d qualifies; +49.9% does not."""
    # 20 flat days then jump
    closes = [100.0] * 10 + [150.0] + [100.0] * 5
    mkt = _synth_asset("aaa", date(2026, 1, 1), closes)
    cand = detect_spike_candidates(mkt, threshold=0.50, window=7)
    # Day index 0..9 have close 100; from day 9, next bars include 150 → +50%
    hit = cand.loc[cand["is_spike_raw"]]
    assert len(hit) >= 1
    assert (hit["fwd_max_ret_7d"] >= 0.50 - 1e-9).all()

    # 149.9 is under threshold
    closes2 = [100.0] * 10 + [149.9] + [100.0] * 5
    mkt2 = _synth_asset("bbb", date(2026, 1, 1), closes2)
    cand2 = detect_spike_candidates(mkt2, threshold=0.50, window=7)
    assert cand2["is_spike_raw"].sum() == 0


def test_dedupe_keeps_first_in_cluster():
    dates = [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-20"),
    ]
    kept = dedupe_non_overlapping(dates, window=7)
    assert kept == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-20"),
    ]


def test_detect_spikes_dedupes_and_sets_signal_date():
    # Sustained path where many consecutive days see +50% forward
    closes = [100.0] * 5 + [100.0, 110.0, 120.0, 140.0, 160.0] + [160.0] * 5
    mkt = _synth_asset("ccc", date(2026, 3, 1), closes)
    spikes = detect_spikes(mkt, threshold=0.50, window=7)
    assert len(spikes) >= 1
    # Non-overlapping: should not keep every consecutive raw day
    raw = detect_spike_candidates(mkt, threshold=0.50, window=7)
    assert spikes["spike_date"].nunique() <= raw["is_spike_raw"].sum()
    # signal_date is prior bar
    row = spikes.iloc[0]
    assert pd.notna(row["signal_date"])
    assert pd.Timestamp(row["signal_date"]) < pd.Timestamp(row["spike_date"])


def test_detect_spikes_respects_eligible_filter():
    closes = [100.0] * 8 + [160.0] + [100.0] * 3
    mkt = pd.concat(
        [
            _synth_asset("in", date(2026, 1, 1), closes),
            _synth_asset("out", date(2026, 1, 1), closes),
        ],
        ignore_index=True,
    )
    # Only "in" is eligible on spike-capable days
    elig_dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(len(closes))]
    eligible = pd.DataFrame(
        {
            "timestamp": elig_dates,
            "asset_id": ["in"] * len(closes),
            "band": ["A_5_50"] * len(closes),
        }
    )
    spikes = detect_spikes(mkt, threshold=0.50, window=7, eligible=eligible)
    assert len(spikes) >= 1
    assert set(spikes["asset_id"]) == {"in"}


def test_forward_max_return_by_asset_two_names():
    mkt = pd.concat(
        [
            _synth_asset("a", date(2026, 1, 1), [10.0, 10.0, 20.0]),
            _synth_asset("b", date(2026, 1, 1), [10.0, 10.0, 11.0]),
        ],
        ignore_index=True,
    )
    fwd = forward_max_return_by_asset(mkt, window=2)
    a0 = fwd.loc[(fwd.asset_id == "a") & (fwd.timestamp == pd.Timestamp("2026-01-01"))]
    assert float(a0["fwd_max_ret_2d"].iloc[0]) == pytest.approx(1.0)


def test_split_spikes_by_cut():
    spikes = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"],
            "spike_date": [
                date(2026, 8, 1),
                date(2026, 8, 21),
                date(2026, 9, 1),
            ],
        }
    )
    disc, val = split_spikes_by_cut(spikes, date(2026, 8, 21))
    assert list(disc["asset_id"]) == ["a"]
    assert list(val["asset_id"]) == ["b", "c"]
