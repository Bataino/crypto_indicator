"""Tests for locked draft formula fair-test helpers and new soft flags."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from cmram.backtest.draft_formula_fair_test import (
    DRAFT_CANDIDATES,
    RVOL_CORE,
    LOCKED_DIST,
    _core_beats_baselines,
    _soft_helps_vs_core,
    candidates_as_shared_setups,
    run_draft_formula_fair_test,
    write_draft_formula_fair_test_report,
)
from cmram.backtest.spike_indicators import compute_tech_indicators
from cmram.backtest.spike_shared_setups import build_setup_flags


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


def test_soft_flags_rsi_le_70_and_gap_gt_m10_exist():
    closes = [100.0 + i * 0.2 for i in range(80)]
    mkt = _synth_asset("dft", date(2026, 1, 1), closes)
    ind = compute_tech_indicators(mkt)
    # Attach fake Rotation_Gap via features-style columns on indicators
    ind = ind.copy()
    ind["Rotation_Gap"] = [-15.0] * 40 + [5.0] * 40
    flags = build_setup_flags(ind)
    assert "rsi_le_70" in flags.columns
    assert "gap_gt_m10" in flags.columns
    assert flags["rsi_le_70"].dtype == bool
    assert flags["gap_gt_m10"].dtype == bool
    # Gap > -10 should be true on the later half
    assert flags["gap_gt_m10"].iloc[-1]
    assert not flags["gap_gt_m10"].iloc[0]


def test_draft_candidates_locked_shape():
    setups = candidates_as_shared_setups()
    ids = [s.setup_id for s in setups]
    assert ids[0] == "locked_core"
    assert "volume_alone" in ids
    assert "not_stretched_alone" in ids
    assert "locked_plus_rsi70" in ids
    assert "locked_plus_rsi45_65" in ids
    assert "locked_plus_gap_m10" in ids
    core = next(c for c in DRAFT_CANDIDATES if c.candidate_id == "locked_core")
    assert core.flag_ids == (RVOL_CORE, LOCKED_DIST)
    # Soft variants include locked core flags
    for c in DRAFT_CANDIDATES:
        if c.role.startswith("soft") or c.role == "core":
            assert RVOL_CORE in c.flag_ids
            assert LOCKED_DIST in c.flag_ids


def test_core_beats_baselines_and_soft_help_helpers():
    val = pd.DataFrame(
        [
            {
                "setup_id": "locked_core",
                "n_fire": 20,
                "n_hit": 4,
                "hit_rate": 0.20,
                "lift": 2.0,
                "beats_base": True,
                "n_fire_ok": True,
            },
            {
                "setup_id": "volume_alone",
                "n_fire": 50,
                "n_hit": 5,
                "hit_rate": 0.10,
                "lift": 1.5,
                "beats_base": True,
                "n_fire_ok": True,
            },
            {
                "setup_id": "not_stretched_alone",
                "n_fire": 40,
                "n_hit": 4,
                "hit_rate": 0.10,
                "lift": 1.2,
                "beats_base": True,
                "n_fire_ok": True,
            },
            {
                "setup_id": "locked_plus_rsi70",
                "n_fire": 15,
                "n_hit": 4,
                "hit_rate": 0.266,
                "lift": 2.5,
                "beats_base": True,
                "n_fire_ok": True,
            },
            {
                "setup_id": "locked_plus_gap_m10",
                "n_fire": 12,
                "n_hit": 1,
                "hit_rate": 0.083,
                "lift": 1.1,
                "beats_base": True,
                "n_fire_ok": True,
            },
        ]
    )
    out = _core_beats_baselines(val)
    assert out["beats_chance"] is True
    assert out["beats_volume_alone"] is True
    assert out["beats_not_stretched_alone"] is True
    assert _soft_helps_vs_core(val, "locked_plus_rsi70") is True
    assert _soft_helps_vs_core(val, "locked_plus_gap_m10") is False


def _make_tiny_universe(n_days: int = 40, n_assets: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic market + membership with at least one forward spike."""
    start = date(2026, 6, 1)
    frames = []
    mem_rows = []
    for a in range(n_assets):
        aid = f"a{a}"
        closes = []
        vols = []
        price = 10.0 + a
        for i in range(n_days):
            # Late spike on asset 0 so +50% label exists
            if a == 0 and i >= n_days - 10:
                price = price * 1.12
                vols.append(3_000_000.0)
            else:
                price = price * (1.0 + 0.002)
                vols.append(1_000_000.0)
            closes.append(price)
        frames.append(_synth_asset(aid, start, closes, vols))
        for i in range(n_days):
            mem_rows.append(
                {
                    "timestamp": start + timedelta(days=i),
                    "asset_id": aid,
                    "eligible": True,
                    "band": "A",
                }
            )
    market = pd.concat(frames, ignore_index=True)
    membership = pd.DataFrame(mem_rows)
    return market, membership


def test_run_draft_formula_fair_test_smoke(tmp_path):
    market, membership = _make_tiny_universe()
    result = run_draft_formula_fair_test(
        market,
        membership,
        features=None,
        explore_frac=0.70,
        alt_explore_frac=0.60,
        max_control=200,
        min_fires=1,
    )
    assert result.primary.cut_date is not None
    assert result.alternate is not None
    assert len(result.primary.validation_50) == len(DRAFT_CANDIDATES)
    assert "locked_beats_chance_later" in result.summary
    assert "soft_filters_help" in result.summary
    report = write_draft_formula_fair_test_report(
        result, tmp_path / "draft_formula_fair_test_abdul.md"
    )
    text = report.read_text(encoding="utf-8")
    assert "not alpha" in text.lower() or "No alpha" in text or "**not alpha**" in text
    assert "Locked draft formula" in text or "locked" in text.lower()
    assert "rvol_30 > 1.5" in text
    assert "Does the locked combo beat chance later?" in text
    assert "Do soft filters help?" in text
