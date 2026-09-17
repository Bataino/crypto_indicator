"""Unit tests for point-in-time universe membership (synthetic bars — no API)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.config import load_thresholds_config, load_universe_config
from cmram.db.schema import init_db
from cmram.universe.membership import (
    build_universe_membership,
    membership_summary,
    resolve_v_min_usd,
    write_universe_membership,
)


def _universe() -> dict:
    return load_universe_config()


def _thresholds(**overrides) -> dict:
    cfg = dict(load_thresholds_config())
    cfg.update(overrides)
    return cfg


def _synth_bars(
    asset_id: str,
    n_days: int,
    *,
    start: date | None = None,
    mc: float | np.ndarray | list[float] = 20_000_000.0,
    volume: float | np.ndarray | list[float] = 200_000.0,
    close0: float = 1.0,
    daily_ret: float = 0.01,
) -> pd.DataFrame:
    """Build contiguous daily OHLCV + MC bars for one asset."""
    start = start or date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    if np.isscalar(mc):
        mcs = np.full(n_days, float(mc))
    else:
        mcs = np.asarray(mc, dtype=float)
        assert len(mcs) == n_days
    if np.isscalar(volume):
        vols = np.full(n_days, float(volume))
    else:
        vols = np.asarray(volume, dtype=float)
        assert len(vols) == n_days

    closes = close0 * np.cumprod(np.r_[1.0, np.full(n_days - 1, 1.0 + daily_ret)])
    return pd.DataFrame(
        {
            "timestamp": dates,
            "asset_id": asset_id,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume_usd": vols,
            "market_cap_usd": mcs,
        }
    )


def test_resolve_v_min_uses_first_candidate():
    assert resolve_v_min_usd({"v_min_usd": [50_000, 100_000, 250_000]}) == 50_000.0
    assert (
        resolve_v_min_usd({"v_min_usd": [50_000], "v_min_usd_active": 100_000})
        == 100_000.0
    )


def test_band_a_membership_and_history_gate():
    bars = _synth_bars("a1", 100, mc=25_000_000.0, volume=200_000.0)
    mem = build_universe_membership(bars, _universe(), _thresholds())

    a = mem[mem["band"] == "A_5_50"]
    assert not a.empty
    assert a["history_days"].min() >= 90
    assert a["timestamp"].min() == date(2024, 1, 1) + timedelta(days=89)
    assert set(a["asset_id"]) == {"a1"}
    b = mem[mem["band"] == "B_10_100"]
    assert not b.empty
    assert set(b["asset_id"]) == {"a1"}


def test_bands_evaluated_separately_overlap_and_exclusive():
    only_a = _synth_bars("only_a", 100, mc=7_000_000.0)
    only_b = _synth_bars("only_b", 100, mc=75_000_000.0)
    both = _synth_bars("both", 100, mc=25_000_000.0)
    market = pd.concat([only_a, only_b, both], ignore_index=True)

    mem = build_universe_membership(market, _universe(), _thresholds())
    last = mem["timestamp"].max()
    day = mem[mem["timestamp"] == last]

    a_ids = set(day.loc[day["band"] == "A_5_50", "asset_id"])
    b_ids = set(day.loc[day["band"] == "B_10_100", "asset_id"])
    assert a_ids == {"only_a", "both"}
    assert b_ids == {"only_b", "both"}


def test_point_in_time_mc_no_lookahead():
    n = 100
    mcs = np.full(n, 1_000_000.0)
    mcs[-1] = 20_000_000.0
    bars = _synth_bars("pit", n, mc=mcs)
    mem = build_universe_membership(bars, _universe(), _thresholds())

    a = mem[mem["band"] == "A_5_50"]
    assert len(a) == 1
    assert a.iloc[0]["timestamp"] == date(2024, 1, 1) + timedelta(days=n - 1)
    assert a.iloc[0]["market_cap_usd"] == pytest.approx(20_000_000.0)


def test_exclude_missing_volume_or_mc():
    bars = _synth_bars("miss", 100, mc=20_000_000.0)
    bars.loc[bars.index[-1], "volume_usd"] = np.nan
    bars2 = _synth_bars("miss_mc", 100, mc=20_000_000.0)
    bars2.loc[bars2.index[-1], "market_cap_usd"] = np.nan
    market = pd.concat([bars, bars2], ignore_index=True)

    mem = build_universe_membership(market, _universe(), _thresholds())
    last = date(2024, 1, 1) + timedelta(days=99)
    day = mem[mem["timestamp"] == last]
    assert "miss" not in set(day["asset_id"])
    assert "miss_mc" not in set(day["asset_id"])


def test_liquidity_pass_volume_gate_draft():
    hi = _synth_bars("hi_vol", 100, mc=20_000_000.0, volume=500_000.0)
    lo = _synth_bars("lo_vol", 100, mc=20_000_000.0, volume=10_000.0)
    market = pd.concat([hi, lo], ignore_index=True)
    mem = build_universe_membership(market, _universe(), _thresholds())

    last = mem["timestamp"].max()
    day_a = mem[(mem["timestamp"] == last) & (mem["band"] == "A_5_50")]
    by_id = day_a.set_index("asset_id")
    assert bool(by_id.loc["hi_vol", "liquidity_pass"]) is True
    assert by_id.loc["hi_vol", "reason_excluded"] is None
    assert bool(by_id.loc["lo_vol", "liquidity_pass"]) is False
    assert "low_volume" in str(by_id.loc["lo_vol", "reason_excluded"])
    assert pd.notna(by_id.loc["hi_vol", "amihud"])


def test_amihud_percentile_gate_when_configured():
    n = 100
    calm = _synth_bars("calm", n, mc=20_000_000.0, volume=200_000.0, daily_ret=0.001)
    wild = _synth_bars("wild", n, mc=20_000_000.0, volume=200_000.0, daily_ret=0.20)
    market = pd.concat([calm, wild], ignore_index=True)

    thr = _thresholds()
    thr["amihud"] = {
        "lookback_days": 20,
        "illiq_max_percentile": 50,
        "calibration_status": "TBD",
    }
    mem = build_universe_membership(market, _universe(), thr)
    last = mem["timestamp"].max()
    day = mem[(mem["timestamp"] == last) & (mem["band"] == "A_5_50")].set_index(
        "asset_id"
    )

    assert day.loc["wild", "amihud"] > day.loc["calm", "amihud"]
    assert bool(day.loc["calm", "liquidity_pass"]) is True
    assert bool(day.loc["wild", "liquidity_pass"]) is False
    assert "high_amihud" in str(day.loc["wild", "reason_excluded"])


def test_history_below_min_excluded():
    bars = _synth_bars("short", 50, mc=20_000_000.0)
    mem = build_universe_membership(bars, _universe(), _thresholds())
    assert mem.empty


def test_write_universe_membership_roundtrip(tmp_path):
    bars = _synth_bars("w1", 100, mc=20_000_000.0, volume=200_000.0)
    mem = build_universe_membership(bars, _universe(), _thresholds())
    assert not mem.empty

    db = tmp_path / "t.duckdb"
    conn = init_db(db)
    try:
        n = write_universe_membership(conn, mem, replace=True)
        assert n == len(mem)
        got = conn.execute(
            "SELECT band, count(*) c FROM universe_membership GROUP BY 1 ORDER BY 1"
        ).fetchall()
        assert got
        summary = membership_summary(mem)
        assert summary["total_rows"] == len(mem)
        n2 = write_universe_membership(conn, mem, replace=True)
        assert n2 == n
        total = conn.execute("SELECT count(*) FROM universe_membership").fetchone()[0]
        assert total == n
    finally:
        conn.close()


def test_empty_market_returns_empty():
    mem = build_universe_membership(pd.DataFrame(), _universe(), _thresholds())
    assert list(mem.columns) == list(
        [
            "timestamp",
            "asset_id",
            "band",
            "market_cap_usd",
            "history_days",
            "liquidity_pass",
            "amihud",
            "reason_excluded",
        ]
    )
    assert mem.empty
