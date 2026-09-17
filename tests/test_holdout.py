"""Tests for time holdout date-split and cell selection logic."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.backtest.holdout import (
    PREREGISTERED_CELLS,
    build_holdout_cell_set,
    compute_time_split,
    filter_signals_to_cells,
    membership_calendar_dates,
    model_c_holdout_verdict,
    split_signals_by_cut,
    support_statement,
    summarize_tau_grid_h7,
    top_explore_cell,
)


def test_compute_time_split_70_30_exact_cut():
    """91 days → floor(91*0.7)=63 explore, cut = day index 63."""
    start = date(2026, 6, 18)
    dates = [start + timedelta(days=i) for i in range(91)]
    split = compute_time_split(dates, explore_frac=0.70)
    assert split["n_days"] == 91
    assert split["n_explore"] == 63
    assert split["n_holdout"] == 28
    assert split["explore_end"] == pd.Timestamp(dates[62]).normalize()
    assert split["cut_date"] == pd.Timestamp(dates[63]).normalize()
    assert split["holdout_start"] == split["cut_date"]
    assert split["holdout_end"] == pd.Timestamp(dates[-1]).normalize()
    # Known DB cut for current sample history shape
    assert split["cut_date"].date() == date(2026, 8, 20)
    assert split["explore_end"].date() == date(2026, 8, 19)


def test_compute_time_split_empty_and_single():
    empty = compute_time_split([], explore_frac=0.7)
    assert empty["n_days"] == 0
    assert empty["cut_date"] is None

    one = compute_time_split([date(2026, 1, 1)], explore_frac=0.7)
    assert one["n_explore"] == 1
    assert one["n_holdout"] == 0
    assert one["cut_date"] is None


def test_compute_time_split_keeps_both_sides():
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    split = compute_time_split(dates, explore_frac=0.70)
    assert split["n_explore"] == 7
    assert split["n_holdout"] == 3
    # Extreme frac still leaves ≥1 holdout when n>=2
    split2 = compute_time_split(dates, explore_frac=0.99)
    assert split2["n_explore"] == 9
    assert split2["n_holdout"] == 1


def test_compute_time_split_rejects_bad_frac():
    with pytest.raises(ValueError):
        compute_time_split([date(2024, 1, 1)], explore_frac=0.0)
    with pytest.raises(ValueError):
        compute_time_split([date(2024, 1, 1)], explore_frac=1.0)


def test_membership_calendar_dates_unique_sorted():
    mem = pd.DataFrame(
        {
            "timestamp": ["2026-06-20", "2026-06-18", "2026-06-18", "2026-06-19"],
            "asset_id": ["a", "a", "b", "a"],
            "band": ["A_5_50"] * 4,
        }
    )
    dates = membership_calendar_dates(mem)
    assert [d.date() for d in dates] == [
        date(2026, 6, 18),
        date(2026, 6, 19),
        date(2026, 6, 20),
    ]


def test_split_signals_by_cut():
    sig = pd.DataFrame(
        {
            "signal_id": ["s1", "s2", "s3"],
            "timestamp": [date(2026, 8, 19), date(2026, 8, 20), date(2026, 8, 21)],
            "model": ["A", "A", "B"],
        }
    )
    explore, holdout = split_signals_by_cut(sig, date(2026, 8, 20))
    assert list(explore["signal_id"]) == ["s1"]
    assert list(holdout["signal_id"]) == ["s2", "s3"]


def test_build_holdout_cell_set_adds_top_when_different():
    top = {"model": "C", "tau_m": 70.0, "tau_g": 40.0, "label": "explore_top_C_70_40"}
    cells = build_holdout_cell_set(top)
    keys = {(c["model"], c["tau_m"], c["tau_g"]) for c in cells}
    assert ("B", 50.0, 20.0) in keys
    assert ("C", 50.0, 20.0) in keys
    assert ("C", 70.0, 40.0) in keys
    assert len(cells) == 3


def test_build_holdout_cell_set_no_dup_when_top_is_prereg():
    top = {"model": "B", "tau_m": 50.0, "tau_g": 20.0, "label": "explore_top_B_50_20"}
    cells = build_holdout_cell_set(top)
    assert len(cells) == len(PREREGISTERED_CELLS)
    matched = [c for c in cells if c.get("also_explore_top")]
    assert len(matched) == 1
    assert matched[0]["model"] == "B"


def test_filter_signals_to_cells():
    sig = pd.DataFrame(
        {
            "signal_id": ["1", "2", "3"],
            "model": ["A", "B", "A"],
            "tau_m": [50.0, 50.0, 70.0],
            "tau_g": [20.0, 20.0, 40.0],
        }
    )
    cells = [{"model": "B", "tau_m": 50.0, "tau_g": 20.0}]
    out = filter_signals_to_cells(sig, cells)
    assert list(out["signal_id"]) == ["2"]


def test_top_explore_cell_and_support_statement():
    grid = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B"],
            "tau_m": [50.0, 50.0, 70.0, 60.0],
            "tau_g": [20.0, 20.0, 30.0, 50.0],
            "median_excess": [0.01, 0.05, -0.02, 0.99],
            "n": [25, 30, 22, 1],  # n=1 noise must not win when min_n=20
        }
    )
    top = top_explore_cell(grid, min_n=20)
    assert top["model"] == "B"
    assert top["tau_m"] == 50.0
    assert top_explore_cell(grid, min_n=100) is None

    holdout_pos = pd.DataFrame(
        [
            {
                "model": "B",
                "tau_m": 50.0,
                "tau_g": 20.0,
                "n": 10,
                "median_excess": 0.02,
            }
        ]
    )
    stmt = support_statement(top, holdout_pos)
    assert "positive" in stmt.lower()
    assert "does not validate alpha" in stmt.lower() or "NOT validate" in stmt

    holdout_neg = holdout_pos.copy()
    holdout_neg["median_excess"] = -0.03
    stmt2 = support_statement(top, holdout_neg)
    assert "fails to support" in stmt2.lower()

    holdout_tiny = holdout_pos.copy()
    holdout_tiny["n"] = 2
    stmt3 = support_statement(top, holdout_tiny)
    assert "fails to support" in stmt3.lower()


def test_summarize_tau_grid_h7_median_excess():
    signals = pd.DataFrame(
        {
            "signal_id": ["a", "b", "c"],
            "band": ["A_5_50", "A_5_50", "B_10_100"],
            "model": ["A", "A", "A"],
            "tau_m": [50.0, 50.0, 50.0],
            "tau_g": [20.0, 20.0, 20.0],
            "asset_id": ["x", "y", "z"],
        }
    )
    results = pd.DataFrame(
        {
            "signal_id": ["a", "b", "c"],
            "horizon_d": [7, 7, 7],
            "ret": [0.1, -0.05, 0.0],
            "mfe": [0.1, 0.0, 0.0],
            "mae": [-0.01, -0.05, 0.0],
            "benchmark_id": ["random", "random", "random"],
            "excess_ret": [0.08, -0.02, 0.01],
        }
    )
    grid = summarize_tau_grid_h7(signals, results, pool_bands=True)
    assert len(grid) == 1
    assert grid.iloc[0]["n"] == 3
    assert grid.iloc[0]["median_excess"] == pytest.approx(0.01)


def test_preregistered_cells_are_b_and_c():
    models = {c["model"] for c in PREREGISTERED_CELLS}
    assert models == {"B", "C"}
    assert all(c["tau_m"] == 50.0 and c["tau_g"] == 20.0 for c in PREREGISTERED_CELLS)


def test_top_explore_cell_model_filter():
    grid = pd.DataFrame(
        {
            "model": ["B", "C", "C", "A"],
            "tau_m": [50.0, 50.0, 70.0, 50.0],
            "tau_g": [20.0, 20.0, 20.0, 20.0],
            "median_excess": [0.99, 0.02, 0.05, 0.50],
            "n": [30, 25, 22, 40],
        }
    )
    # Unfiltered would pick B; Model C filter picks C 70/20
    top_all = top_explore_cell(grid, min_n=20)
    assert top_all["model"] == "B"
    top_c = top_explore_cell(grid, min_n=20, models=("C",))
    assert top_c["model"] == "C"
    assert top_c["tau_m"] == 70.0
    assert top_c["tau_g"] == 20.0


def test_model_c_holdout_verdict_positive_and_negative():
    pos = pd.DataFrame(
        [
            {
                "model": "C",
                "tau_m": 50.0,
                "tau_g": 20.0,
                "n": 12,
                "median_excess": 0.03,
            },
            {
                "model": "B",
                "tau_m": 50.0,
                "tau_g": 20.0,
                "n": 10,
                "median_excess": -0.02,
            },
        ]
    )
    stmt = model_c_holdout_verdict(pos)
    assert "weak directional" in stmt.lower() or "positive" in stmt.lower()
    assert "does not validate alpha" in stmt.lower() or "NOT validate" in stmt

    neg = pos.copy()
    neg.loc[neg["model"] == "C", "median_excess"] = -0.04
    stmt2 = model_c_holdout_verdict(neg)
    assert "does not support" in stmt2.lower() or "fails to support" in stmt2.lower()

