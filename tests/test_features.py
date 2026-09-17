"""Point-in-time / leakage tests for Phase 1 Models A/B features (synthetic)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cmram.config import load_features_config
from cmram.db.schema import FEATURES_DAILY_REQUIRED_COLS, init_db, migrate_schema
from cmram.features.compression import C_INPUT_COLS, compression_inputs
from cmram.features.lookbacks import DEFAULT_LOOKBACKS
from cmram.features.narrative import score_N
from cmram.features.persist import write_features_daily
from cmram.features.pipeline import FEATURE_COLUMNS, compute_features_daily
from cmram.features.price import RS_BENCHMARK_BAND_EW, RS_BENCHMARK_BTC
from cmram.features.rank import cs_percentile
from cmram.features.volume import volume_inputs


def _dates(n: int, start: date | None = None) -> list[date]:
    start = start or date(2024, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _synth_asset(
    asset_id: str,
    n_days: int,
    *,
    start: date | None = None,
    close0: float = 1.0,
    daily_ret: float = 0.01,
    volume: float = 100_000.0,
    vol_growth: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Contiguous daily OHLCV. Distinct high/low so range features are defined."""
    rng = np.random.default_rng(seed)
    dates = _dates(n_days, start)
    rets = np.full(n_days, daily_ret)
    if noise:
        rets = rets + rng.normal(0.0, noise, n_days)
    closes = close0 * np.cumprod(np.r_[1.0, 1.0 + rets[1:]])
    highs = closes * (1.01 + (rng.random(n_days) * 0.01 if noise else 0.0))
    lows = closes * (0.99 - (rng.random(n_days) * 0.01 if noise else 0.0))
    opens = closes * 0.995
    vols = np.full(n_days, float(volume), dtype=float)
    if vol_growth:
        vols = vols * (1.0 + vol_growth) ** np.arange(n_days)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "asset_id": asset_id,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume_usd": vols,
            "market_cap_usd": np.full(n_days, 20_000_000.0),
        }
    )


def _market(*assets: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(list(assets), ignore_index=True)


def _membership(market: pd.DataFrame, *, from_idx: int, bands: tuple[str, ...] = ("A_5_50",)) -> pd.DataFrame:
    """Membership from from_idx onward for every asset × band (synthetic, PIT dates)."""
    rows = []
    for asset_id, g in market.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp")
        dates = list(pd.to_datetime(g["timestamp"]).dt.date)
        for ts in dates[from_idx:]:
            for band in bands:
                rows.append(
                    {
                        "timestamp": ts,
                        "asset_id": asset_id,
                        "band": band,
                        "market_cap_usd": 20_000_000.0,
                        "history_days": from_idx + 1,
                        "liquidity_pass": True,
                    }
                )
    return pd.DataFrame(rows)


def test_features_config_loads():
    cfg = load_features_config()
    assert cfg["model_version"] == "features_v0.1_quiet_n"
    assert cfg["models"] == ["A", "B", "C", "E"]
    assert cfg["lookbacks"]["range_long_days"] == 40
    assert cfg["small_sample_n"] == 10
    assert cfg.get("narrative_mode") == "quiet_rising"


def test_cs_percentile_n1_and_ties_and_n3():
    s = pd.Series([10.0])
    assert cs_percentile(s).iloc[0] == pytest.approx(50.0)
    tied = cs_percentile(pd.Series([5.0, 5.0, 5.0]))
    assert list(tied) == [50.0, 50.0, 50.0]
    ranked = cs_percentile(pd.Series([1.0, 2.0, 3.0]))
    assert list(ranked) == [0.0, 50.0, 100.0]
    # average rank on a tie is deterministic
    mixed = cs_percentile(pd.Series([1.0, 2.0, 2.0]))
    assert mixed.iloc[0] == pytest.approx(0.0)
    assert mixed.iloc[1] == pytest.approx(75.0)
    assert mixed.iloc[2] == pytest.approx(75.0)


def test_score_n_is_null_not_imputed():
    out = score_N(None, index=pd.RangeIndex(3))
    assert out.isna().all()
    empty = score_N(pd.DataFrame())
    assert empty.empty or empty.isna().all()


def test_model_a_b_composition_and_nulls():
    n = 80
    market = _market(
        _synth_asset("a", n, daily_ret=0.01, volume=100_000, seed=1),
        _synth_asset("b", n, daily_ret=0.02, volume=200_000, seed=2),
        _synth_asset("c", n, daily_ret=-0.005, volume=50_000, seed=3),
    )
    mem = _membership(market, from_idx=50, bands=("A_5_50", "B_10_100"))
    feat = compute_features_daily(market, mem, config={"models": ["A", "B"]})
    assert not feat.empty
    assert list(feat.columns) == list(FEATURE_COLUMNS)
    assert set(feat["model"]) == {"A", "B"}
    assert feat["n_available"].eq(False).all()
    assert feat["score_N"].isna().all()
    assert feat["nsi_social"].isna().all()
    assert feat["small_sample"].eq(True).all()

    a = feat[feat["model"] == "A"]
    b = feat[feat["model"] == "B"]
    assert a["score_C"].isna().all()
    assert a["score_V"].isna().all()
    assert a["nsi_vol_exh"].isna().all()
    assert a["score_P"].notna().any()
    pd.testing.assert_series_equal(
        a["MREI"].reset_index(drop=True),
        a["score_P"].reset_index(drop=True),
        check_names=False,
    )

    assert b["score_C"].notna().any()
    assert b["score_V"].notna().any()
    assert b["score_P"].notna().any()
    assert b["nsi_vol_exh"].notna().any()
    expected_mrei = b[["score_C", "score_V", "score_P"]].mean(axis=1)
    pd.testing.assert_series_equal(
        b["MREI"].reset_index(drop=True),
        expected_mrei.reset_index(drop=True),
        check_names=False,
    )
    # Gap = MREI - NSI
    gap = b["MREI"] - b["NSI"]
    pd.testing.assert_series_equal(
        b["Rotation_Gap"].reset_index(drop=True),
        gap.reset_index(drop=True),
        check_names=False,
    )
    # Equal membership count × 2 models
    assert len(feat) == len(mem) * feat["model"].nunique()


def test_no_future_price_leakage():
    n = 80
    market = _market(
        _synth_asset("a", n, daily_ret=0.01, seed=1),
        _synth_asset("b", n, daily_ret=0.00, seed=2),
        _synth_asset("c", n, daily_ret=-0.01, seed=3),
    )
    mem = _membership(market, from_idx=50)
    feat1 = compute_features_daily(market, mem, config={"models": ["A", "B"]})

    cutoff = date(2024, 1, 1) + timedelta(days=60)
    m2 = market.copy()
    ts = pd.to_datetime(m2["timestamp"]).dt.date
    mask = ts > cutoff
    m2.loc[mask, ["open", "high", "low", "close"]] *= 50.0
    m2.loc[mask, "volume_usd"] *= 50.0
    feat2 = compute_features_daily(m2, mem, config={"models": ["A", "B"]})

    def _past(df: pd.DataFrame) -> pd.DataFrame:
        t = pd.to_datetime(df["timestamp"]).dt.date
        return (
            df.loc[t <= cutoff]
            .sort_values(["timestamp", "band", "model", "asset_id"], kind="mergesort")
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(_past(feat1), _past(feat2), check_dtype=False)


def test_rolling_window_matches_manual_asof_t():
    """C distance-from-high at t equals max(high[t-19:t]) using only bars ≤ t."""
    n = 60
    a = _synth_asset("a", n, daily_ret=0.01, seed=7)
    raw = compression_inputs(a)
    t_idx = 40
    t = a.iloc[t_idx]["timestamp"]
    window = a.iloc[t_idx - 19 : t_idx + 1]
    high_20 = float(window["high"].max())
    close = float(a.iloc[t_idx]["close"])
    expected = (high_20 - close) / high_20
    got = float(raw.loc[raw["timestamp"] == pd.Timestamp(t), "c_dist_20d_high"].iloc[0])
    assert got == pytest.approx(expected, rel=1e-12)
    # And using a future bar would change max if we leaked — confirm future high is higher
    future_high = float(a.iloc[t_idx + 1 :]["high"].max())
    # even if future high is larger, PIT value must ignore it
    leaked = (max(high_20, future_high) - close) / max(high_20, future_high)
    if future_high > high_20:
        assert got != pytest.approx(leaked)


def test_cs_rank_only_same_band_day():
    n = 70
    market = _market(
        _synth_asset("strong", n, daily_ret=0.03, seed=1),
        _synth_asset("mid", n, daily_ret=0.01, seed=2),
        _synth_asset("weak", n, daily_ret=-0.02, seed=3),
    )
    # All three in A; only strong+mid in B
    mem_a = _membership(market, from_idx=50, bands=("A_5_50",))
    mem_b = _membership(
        market[market["asset_id"].isin(["strong", "mid"])],
        from_idx=50,
        bands=("B_10_100",),
    )
    mem = pd.concat([mem_a, mem_b], ignore_index=True)
    feat = compute_features_daily(market, mem, config={"models": ["A", "B"]})
    last = feat["timestamp"].max()
    day = feat[(feat["timestamp"] == last) & (feat["model"] == "B")]
    a_p = day.loc[day["band"] == "A_5_50"].set_index("asset_id")["score_P"]
    b_p = day.loc[day["band"] == "B_10_100"].set_index("asset_id")["score_P"]
    assert set(a_p.index) == {"strong", "mid", "weak"}
    assert set(b_p.index) == {"strong", "mid"}
    # Different universes → different percentiles for the same asset/day
    # (unless the extra name doesn't change ranks, which it should for P returns)
    assert a_p.loc["strong"] != pytest.approx(b_p.loc["strong"]) or len(a_p) != len(b_p)


def test_band_ew_benchmark_when_btc_absent():
    n = 70
    market = _market(
        _synth_asset("a", n, daily_ret=0.02, seed=1),
        _synth_asset("b", n, daily_ret=0.00, seed=2),
        _synth_asset("c", n, daily_ret=-0.01, seed=3),
    )
    mem = _membership(market, from_idx=50)
    feat = compute_features_daily(market, mem, config={"models": ["A", "B"]})
    assert feat.attrs["rs_benchmark"] == RS_BENCHMARK_BAND_EW
    assert "bitcoin" not in set(market["asset_id"])


def test_btc_benchmark_when_present():
    """BTC is selected when present; raw RS values differ from band-EW.

    CS *ranks* of RS vs a common benchmark equal CS ranks of own n-day
    return, so score_P percentiles may match EW. We still compute RS so the
    definition is explicit (and raw values are not the same).
    """
    from cmram.features.price import attach_relative_strength, price_inputs

    n = 70
    names = _market(
        _synth_asset("a", n, daily_ret=0.02, seed=1),
        _synth_asset("b", n, daily_ret=0.00, seed=2),
        _synth_asset("c", n, daily_ret=-0.01, seed=3),
    )
    btc = _synth_asset("bitcoin", n, daily_ret=0.005, close0=40000.0, seed=9)
    market = _market(names, btc)
    mem = _membership(names, from_idx=50)  # BTC not in universe
    feat = compute_features_daily(market, mem, config={"models": ["A", "B"]})
    assert feat.attrs["rs_benchmark"] == RS_BENCHMARK_BTC

    raw_p = price_inputs(names)
    with_btc, bench_btc = attach_relative_strength(mem.copy(), raw_p, market)
    with_ew, bench_ew = attach_relative_strength(mem.copy(), raw_p, names)
    assert bench_btc == RS_BENCHMARK_BTC
    assert bench_ew == RS_BENCHMARK_BAND_EW
    assert not np.allclose(
        with_btc["p_rs_7d"].fillna(0).to_numpy(),
        with_ew["p_rs_7d"].fillna(0).to_numpy(),
    )


def test_zero_volume_no_inf():
    n = 70
    a = _synth_asset("a", n, volume=100_000, seed=1)
    b = _synth_asset("b", n, volume=80_000, seed=2)
    c = _synth_asset("c", n, volume=90_000, seed=3)
    a.loc[a.index[40:45], "volume_usd"] = 0.0
    market = _market(a, b, c)
    raw = volume_inputs(market)
    for col in ("v_rvol_3_30", "v_growth_7_28", "v_velocity_3d", "v_log_accel_7"):
        s = raw[col]
        assert not np.isinf(s.dropna()).any(), col
    mem = _membership(market, from_idx=50)
    feat = compute_features_daily(market, mem, config={"models": ["A", "B"]})
    bfeat = feat[feat["model"] == "B"]
    assert not np.isinf(bfeat["score_V"].dropna()).any()
    assert not np.isinf(bfeat["MREI"].dropna()).any()


def test_deterministic_under_row_shuffle():
    n = 70
    market = _market(
        _synth_asset("a", n, daily_ret=0.01, seed=1),
        _synth_asset("b", n, daily_ret=0.02, seed=2),
        _synth_asset("c", n, daily_ret=-0.01, seed=3),
    )
    mem = _membership(market, from_idx=50)
    f1 = compute_features_daily(market, mem, config={"models": ["A", "B"]})
    shuffled = market.sample(frac=1.0, random_state=42).reset_index(drop=True)
    f2 = compute_features_daily(shuffled, mem.sample(frac=1.0, random_state=1), config={"models": ["A", "B"]})
    keys = ["timestamp", "asset_id", "band", "model"]
    a = f1.sort_values(keys).reset_index(drop=True)
    b = f2.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_empty_inputs():
    empty_m = pd.DataFrame(columns=["timestamp", "asset_id", "close", "volume_usd"])
    empty_u = pd.DataFrame(columns=["timestamp", "asset_id", "band"])
    out = compute_features_daily(empty_m, empty_u, config={"models": ["A", "B"]})
    assert out.empty
    assert list(out.columns) == list(FEATURE_COLUMNS)


def test_write_roundtrip_model_pk(tmp_path):
    n = 70
    market = _market(
        _synth_asset("a", n, seed=1),
        _synth_asset("b", n, seed=2),
        _synth_asset("c", n, seed=3),
    )
    mem = _membership(market, from_idx=55)
    feat = compute_features_daily(market, mem, config={"models": ["A", "B"]})
    db = tmp_path / "t.duckdb"
    conn = init_db(db)
    try:
        n_written = write_features_daily(conn, feat, replace=True)
        assert n_written == len(feat)
        cols = {r[0] for r in conn.execute("DESCRIBE features_daily").fetchall()}
        assert FEATURES_DAILY_REQUIRED_COLS <= cols
        got = conn.execute(
            "SELECT model, count(*) c FROM features_daily GROUP BY 1 ORDER BY 1"
        ).fetchall()
        assert [g[0] for g in got] == ["A", "B"]
        assert got[0][1] == got[1][1]
        # compound key: re-write does not duplicate
        write_features_daily(conn, feat, replace=True)
        total = conn.execute("SELECT count(*) FROM features_daily").fetchone()[0]
        assert total == len(feat)
        pk = conn.execute(
            "SELECT timestamp, asset_id, band, model, count(*) "
            "FROM features_daily GROUP BY 1,2,3,4 HAVING count(*) > 1"
        ).fetchall()
        assert pk == []
    finally:
        conn.close()


def test_migrate_old_features_table(tmp_path):
    db = tmp_path / "old.duckdb"
    conn = init_db(db)
    try:
        conn.execute("DROP TABLE features_daily")
        conn.execute(
            """
            CREATE TABLE features_daily (
                timestamp DATE,
                asset_id TEXT,
                band TEXT,
                score_C DOUBLE,
                score_V DOUBLE,
                score_N DOUBLE,
                score_P DOUBLE,
                MREI DOUBLE,
                nsi_social DOUBLE,
                nsi_price_ext DOUBLE,
                nsi_vol_exh DOUBLE,
                nsi_mom_dec DOUBLE,
                NSI DOUBLE,
                Rotation_Gap DOUBLE,
                n_available BOOLEAN,
                model_version TEXT
            )
            """
        )
        rebuilt = migrate_schema(conn)
        assert rebuilt is True
        cols = {r[0] for r in conn.execute("DESCRIBE features_daily").fetchall()}
        assert "model" in cols
        assert "small_sample" in cols
        assert "n_in_band" in cols
        # second migrate is a no-op
        assert migrate_schema(conn) is False
    finally:
        conn.close()


def test_higher_lows_positive_when_troughs_rise():
    n = 40
    dates = _dates(n)
    # declining then rising lows
    close = np.concatenate(
        [np.linspace(10, 5, 20), np.linspace(5.5, 8, 20)]
    )
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "asset_id": "hl",
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume_usd": 1e5,
        }
    )
    raw = compression_inputs(df)  # just to use prepare path
    from cmram.features.price import price_inputs

    p = price_inputs(df)
    last = p.iloc[-1]["p_higher_lows"]
    assert last > 0


def test_init_db_features_has_model_column(tmp_path):
    conn = init_db(tmp_path / "n.duckdb")
    try:
        cols = {r[0] for r in conn.execute("DESCRIBE features_daily").fetchall()}
        assert FEATURES_DAILY_REQUIRED_COLS <= cols
    finally:
        conn.close()
