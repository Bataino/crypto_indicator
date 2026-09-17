"""Backtest engine: forward horizons, MFE/MAE, excess vs benchmarks.

Research only — no orders, no wallets, no claimed alpha.
With a tiny universe (n≪50), aggregates must be treated as SMALL SAMPLE / not conclusive.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cmram.signals.generate import (
    ENTRY_CONVENTION_NEXT_DAY_OPEN,
    build_entry_lookup,
    detect_chart_close_only,
)

DEFAULT_HORIZONS = [1, 3, 7, 14, 21, 30]
DEFAULT_BENCHMARKS = ["random", "momentum", "volume", "market"]
MOMENTUM_LOOKBACK = 7
RVOL_SHORT = 3
RVOL_LONG = 30

RESULT_COLUMNS = (
    "signal_id",
    "horizon_d",
    "ret",
    "mfe",
    "mae",
    "benchmark_id",
    "excess_ret",
)


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=list(RESULT_COLUMNS))


def _prepare_market(market_daily: pd.DataFrame) -> pd.DataFrame:
    m = market_daily.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.normalize()
    m["asset_id"] = m["asset_id"].astype(str)
    m = m.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    m = m.drop_duplicates(["asset_id", "timestamp"], keep="last")
    return m


def _asset_bar_index(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-asset frame indexed by date with open/high/low/close/volume."""
    out: dict[str, pd.DataFrame] = {}
    for asset, g in market.groupby("asset_id", sort=False):
        gg = g.set_index("timestamp").sort_index()
        gg.index = pd.to_datetime(gg.index).normalize()
        out[str(asset)] = gg
    return out


def _forward_path(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    horizon: int,
    *,
    chart_close_only: bool,
) -> tuple[float | None, float | None, float | None]:
    """Return (ret, mfe, mae) from entry bar through +horizon bars.

    Entry bar is the bar at ``entry_ts``. Horizon h uses the close of the
    bar ``h`` steps after the entry bar (trading-day offset on available
    bars). MFE/MAE use highs/lows on (entry, ..., entry+h] vs entry price
    (open, or close if chart-close-only proxy).
    """
    if bars is None or len(bars) == 0:
        return None, None, None
    if entry_ts not in bars.index:
        # next available on/after entry_ts
        later = bars.index[bars.index >= entry_ts]
        if len(later) == 0:
            return None, None, None
        entry_ts = later[0]

    loc = bars.index.get_loc(entry_ts)
    if isinstance(loc, slice):
        loc = loc.start
    elif isinstance(loc, np.ndarray):
        loc = int(np.flatnonzero(loc)[0])
    loc = int(loc)

    target = loc + int(horizon)
    if target >= len(bars):
        return None, None, None

    entry_row = bars.iloc[loc]
    if chart_close_only or (
        np.isclose(entry_row["open"], entry_row["high"])
        and np.isclose(entry_row["high"], entry_row["low"])
        and np.isclose(entry_row["low"], entry_row["close"])
    ):
        entry_px = float(entry_row["close"])
    else:
        entry_px = float(entry_row["open"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None, None, None

    # Window from entry through target inclusive for MFE/MAE
    window = bars.iloc[loc : target + 1]
    exit_px = float(bars.iloc[target]["close"])
    if not np.isfinite(exit_px):
        return None, None, None

    ret = exit_px / entry_px - 1.0
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    mfe = hi / entry_px - 1.0
    mae = lo / entry_px - 1.0
    return float(ret), float(mfe), float(mae)


def _ret7_by_asset_day(market: pd.DataFrame) -> pd.DataFrame:
    m = market.sort_values(["asset_id", "timestamp"], kind="mergesort").copy()
    m["ret_7"] = m.groupby("asset_id", sort=False)["close"].pct_change(MOMENTUM_LOOKBACK)
    return m[["timestamp", "asset_id", "ret_7", "volume_usd"]]


def _rvol_by_asset_day(market: pd.DataFrame) -> pd.DataFrame:
    m = market.sort_values(["asset_id", "timestamp"], kind="mergesort").copy()
    vol = m["volume_usd"].astype("float64").clip(lower=1.0)
    g = m.groupby("asset_id", sort=False)["volume_usd"].transform
    # Use clipped series via assign
    m = m.assign(_vol=vol)
    mean_s = m.groupby("asset_id", sort=False)["_vol"].transform(
        lambda s: s.rolling(RVOL_SHORT, min_periods=RVOL_SHORT).mean()
    )
    mean_l = m.groupby("asset_id", sort=False)["_vol"].transform(
        lambda s: s.rolling(RVOL_LONG, min_periods=RVOL_LONG).mean()
    )
    m["rvol"] = mean_s / mean_l.replace(0, np.nan)
    return m[["timestamp", "asset_id", "rvol", "volume_usd"]]


def _membership_by_day(
    membership: pd.DataFrame,
) -> dict[tuple[Any, str], list[str]]:
    """(date, band) → asset_ids present that day."""
    if membership is None or len(membership) == 0:
        return {}
    mem = membership.copy()
    mem["timestamp"] = pd.to_datetime(mem["timestamp"]).dt.normalize()
    out: dict[tuple[Any, str], list[str]] = {}
    for (ts, band), g in mem.groupby(["timestamp", "band"], sort=False):
        out[(pd.Timestamp(ts).normalize(), str(band))] = sorted(
            g["asset_id"].astype(str).unique().tolist()
        )
    return out


def _pick_random(
    candidates: list[str],
    *,
    exclude: str,
    rng: np.random.Generator,
) -> str | None:
    pool = [a for a in candidates if a != exclude]
    if not pool:
        pool = list(candidates)
    if not pool:
        return None
    return str(rng.choice(pool))


def _pick_top(
    day_frame: pd.DataFrame,
    candidates: list[str],
    score_col: str,
    *,
    exclude: str | None = None,
) -> str | None:
    """Highest score among candidates on that day (ties → first sorted asset)."""
    if day_frame is None or len(day_frame) == 0:
        return None
    sub = day_frame[day_frame["asset_id"].isin(candidates)].copy()
    if exclude is not None:
        sub = sub[sub["asset_id"] != exclude]
    if len(sub) == 0:
        sub = day_frame[day_frame["asset_id"].isin(candidates)].copy()
    if len(sub) == 0:
        return None
    sub = sub.dropna(subset=[score_col])
    if len(sub) == 0:
        return None
    sub = sub.sort_values(
        [score_col, "asset_id"], ascending=[False, True], kind="mergesort"
    )
    return str(sub.iloc[0]["asset_id"])


def _band_ew_forward(
    bars_by_asset: dict[str, pd.DataFrame],
    members: list[str],
    entry_ts: pd.Timestamp,
    horizon: int,
    *,
    chart_close_only: bool,
) -> float | None:
    return _band_stat_forward(
        bars_by_asset,
        members,
        entry_ts,
        horizon,
        chart_close_only=chart_close_only,
        stat="mean",
    )


def _band_stat_forward(
    bars_by_asset: dict[str, pd.DataFrame],
    members: list[str],
    entry_ts: pd.Timestamp,
    horizon: int,
    *,
    chart_close_only: bool,
    stat: str = "mean",
    exclude: str | None = None,
    fwd_ret_fn=None,
) -> float | None:
    """Equal-weight mean or median of same-band member forward returns.

    Used for ``market`` (EW mean fallback), ``group_mean``, and ``group_median``.
    By default includes all members with a completable path; pass ``exclude`` to
    drop the signal asset from the peer group.

    ``fwd_ret_fn(asset, entry_ts, horizon) -> float|None`` optional cache-aware
    lookup (avoids re-walking bars for every signal sharing a day/band).
    """
    rets = []
    for a in members:
        if exclude is not None and a == exclude:
            continue
        if fwd_ret_fn is not None:
            r = fwd_ret_fn(a, entry_ts, horizon)
        else:
            bars = bars_by_asset.get(a)
            if bars is None:
                continue
            r, _, _ = _forward_path(
                bars, entry_ts, horizon, chart_close_only=chart_close_only
            )
        if r is not None and np.isfinite(r):
            rets.append(float(r))
    if not rets:
        return None
    if stat == "median":
        return float(np.median(rets))
    if stat == "mean":
        return float(np.mean(rets))
    raise ValueError(f"unknown band forward stat: {stat}")


def run_backtest(
    signals: pd.DataFrame,
    market_daily: pd.DataFrame,
    *,
    horizons_days: list[int] | None = None,
    benchmarks: list[str] | None = None,
    random_seed: int = 42,
    random_seeds: list[int] | None = None,
    universe_membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-signal forward ret / MFE / MAE and excess vs benchmarks.

    Horizons default: 1,3,7,14,21,30. Does not place orders or touch wallets.

    For each signal × horizon × benchmark, one row with signal ``ret``/MFE/MAE
    and ``excess_ret = ret − benchmark_ret`` (NaN if bench unavailable).

    When ``random_seeds`` is provided and ``random`` is among benchmarks, the
    random benchmark is averaged across seeds: for each signal×horizon the
    mean of per-seed excesses is stored (deterministic; secondary robustness).

    Returns:
        DataFrame matching ``backtest_results`` columns. attrs include
        ``small_sample``, ``n_assets``, ``chart_close_only``.
    """
    horizons = list(horizons_days or DEFAULT_HORIZONS)
    benches = list(benchmarks or DEFAULT_BENCHMARKS)
    out = _empty_results()
    out.attrs["small_sample"] = True
    out.attrs["n_assets"] = 0
    out.attrs["chart_close_only"] = False
    out.attrs["horizons_days"] = horizons
    out.attrs["benchmarks"] = benches

    if signals is None or len(signals) == 0:
        return out
    if market_daily is None or len(market_daily) == 0:
        return out

    market = _prepare_market(market_daily)
    chart_only = detect_chart_close_only(market)
    out.attrs["chart_close_only"] = chart_only
    n_assets = int(market["asset_id"].nunique())
    out.attrs["n_assets"] = n_assets
    out.attrs["small_sample"] = n_assets < 50  # n=11 still tiny; Phase 1 not conclusive

    bars_by_asset = _asset_bar_index(market)
    ret7 = _ret7_by_asset_day(market)
    rvol = _rvol_by_asset_day(market)
    # Index helpers by date
    ret7_by_day: dict[pd.Timestamp, pd.DataFrame] = {
        pd.Timestamp(ts).normalize(): g
        for ts, g in ret7.groupby("timestamp", sort=False)
    }
    rvol_by_day: dict[pd.Timestamp, pd.DataFrame] = {
        pd.Timestamp(ts).normalize(): g
        for ts, g in rvol.groupby("timestamp", sort=False)
    }

    mem_map = _membership_by_day(universe_membership) if universe_membership is not None else {}
    # Fallback: derive membership from signals themselves if no univ table passed
    if not mem_map:
        sig_tmp = signals.copy()
        sig_tmp["timestamp"] = pd.to_datetime(sig_tmp["timestamp"]).dt.normalize()
        for (ts, band), g in sig_tmp.groupby(["timestamp", "band"], sort=False):
            mem_map[(pd.Timestamp(ts).normalize(), str(band))] = sorted(
                g["asset_id"].astype(str).unique().tolist()
            )

    # Entry date lookup from signal date (next bar)
    entry_lookup, _, _ = build_entry_lookup(
        market, entry_convention=ENTRY_CONVENTION_NEXT_DAY_OPEN
    )
    entry_map = {
        (str(r.asset_id), r.signal_date): pd.Timestamp(r.entry_date)
        for r in entry_lookup.itertuples(index=False)
    }

    # BTC presence
    has_btc = "bitcoin" in bars_by_asset

    seeds = (
        [int(s) for s in random_seeds]
        if random_seeds is not None and len(random_seeds) > 0
        else [int(random_seed)]
    )
    multi_random = "random" in benches and len(seeds) > 1
    rng = np.random.default_rng(int(seeds[0]))
    # Independent RNGs per seed for averaged random benchmark
    rngs = [np.random.default_rng(int(s)) for s in seeds]
    rows: list[dict[str, Any]] = []
    out.attrs["random_seeds"] = list(seeds)

    # Cache forward paths: (asset, entry_ts, h) -> (ret, mfe, mae)
    # and group stats: (members_tuple, entry_ts, h) -> (mean, median)
    fwd_cache: dict[tuple[str, pd.Timestamp, int], tuple[float | None, float | None, float | None]] = {}
    group_cache: dict[tuple[tuple[str, ...], pd.Timestamp, int], tuple[float | None, float | None]] = {}

    def _cached_fwd(asset_id: str, ets: pd.Timestamp, h: int):
        key = (asset_id, ets, int(h))
        if key not in fwd_cache:
            bars = bars_by_asset.get(asset_id)
            if bars is None:
                fwd_cache[key] = (None, None, None)
            else:
                fwd_cache[key] = _forward_path(
                    bars, ets, int(h), chart_close_only=chart_only
                )
        return fwd_cache[key]

    def _cached_fwd_ret(asset_id: str, ets: pd.Timestamp, h: int) -> float | None:
        r, _, _ = _cached_fwd(asset_id, ets, h)
        return r

    def _cached_group_stats(members_list: list[str], ets: pd.Timestamp, h: int):
        key = (tuple(members_list), ets, int(h))
        if key not in group_cache:
            rets = []
            for a in members_list:
                r = _cached_fwd_ret(a, ets, h)
                if r is not None and np.isfinite(r):
                    rets.append(float(r))
            if not rets:
                group_cache[key] = (None, None)
            else:
                group_cache[key] = (float(np.mean(rets)), float(np.median(rets)))
        return group_cache[key]

    for sig in signals.itertuples(index=False):
        signal_id = str(sig.signal_id)
        asset = str(sig.asset_id)
        band = str(sig.band)
        sig_date = pd.Timestamp(sig.timestamp).normalize().date()
        entry_ts = entry_map.get((asset, sig_date))
        if entry_ts is None:
            continue
        entry_ts = pd.Timestamp(entry_ts).normalize()
        members = mem_map.get(
            (pd.Timestamp(sig.timestamp).normalize(), band),
            [asset],
        )
        # Signal-day frame for ranking (use feature/signal as-of day, not entry)
        asof = pd.Timestamp(sig.timestamp).normalize()

        # Pre-pick benchmarks once per signal (same for all horizons)
        bench_assets: dict[str, str | None] = {}
        random_picks: list[str | None] = []
        if "random" in benches:
            if multi_random:
                random_picks = [
                    _pick_random(members, exclude=asset, rng=r) for r in rngs
                ]
                # Keep first pick as a representative for single-row path when needed
                bench_assets["random"] = random_picks[0]
            else:
                bench_assets["random"] = _pick_random(
                    members, exclude=asset, rng=rng
                )
        if "momentum" in benches:
            bench_assets["momentum"] = _pick_top(
                ret7_by_day.get(asof, pd.DataFrame()),
                members,
                "ret_7",
                exclude=None,
            )
        if "volume" in benches:
            day_r = rvol_by_day.get(asof, pd.DataFrame())
            # Prefer RVOL when computable for ≥1 member; else top volume
            use_rvol = False
            if len(day_r):
                sub = day_r[day_r["asset_id"].isin(members)]
                use_rvol = bool(sub["rvol"].notna().any())
            if use_rvol:
                bench_assets["volume"] = _pick_top(
                    day_r, members, "rvol", exclude=None
                )
            else:
                vol_day = ret7_by_day.get(asof, pd.DataFrame())
                bench_assets["volume"] = _pick_top(
                    vol_day, members, "volume_usd", exclude=None
                )
        # market handled specially (BTC or EW)

        if asset not in bars_by_asset:
            continue

        need_group = ("group_mean" in benches) or ("group_median" in benches) or (
            "market" in benches and not has_btc
        )

        for h in horizons:
            ret, mfe, mae = _cached_fwd(asset, entry_ts, int(h))
            if ret is None:
                continue

            g_mean = g_med = None
            if need_group:
                g_mean, g_med = _cached_group_stats(members, entry_ts, int(h))

            for bname in benches:
                bench_ret: float | None = None
                if bname == "market":
                    if has_btc:
                        bench_ret, _, _ = _cached_fwd("bitcoin", entry_ts, int(h))
                    else:
                        bench_ret = g_mean
                elif bname == "group_mean":
                    bench_ret = g_mean
                elif bname == "group_median":
                    bench_ret = g_med
                elif bname == "random" and multi_random:
                    seed_rets: list[float] = []
                    for ba in random_picks:
                        if ba is None:
                            continue
                        br = _cached_fwd_ret(ba, entry_ts, int(h))
                        if br is not None and np.isfinite(br):
                            seed_rets.append(float(br))
                    bench_ret = float(np.mean(seed_rets)) if seed_rets else None
                else:
                    ba = bench_assets.get(bname)
                    if ba is not None:
                        bench_ret = _cached_fwd_ret(ba, entry_ts, int(h))

                excess = (
                    float(ret - bench_ret)
                    if bench_ret is not None and np.isfinite(bench_ret)
                    else np.nan
                )
                rows.append(
                    {
                        "signal_id": signal_id,
                        "horizon_d": int(h),
                        "ret": float(ret),
                        "mfe": float(mfe) if mfe is not None else np.nan,
                        "mae": float(mae) if mae is not None else np.nan,
                        "benchmark_id": bname,
                        "excess_ret": excess,
                    }
                )

    if not rows:
        return out

    result = pd.DataFrame(rows)
    result = result[list(RESULT_COLUMNS)]
    result = result.sort_values(
        ["signal_id", "horizon_d", "benchmark_id"], kind="mergesort"
    ).reset_index(drop=True)
    result.attrs.update(out.attrs)
    return result


def aggregate_backtest(
    signals: pd.DataFrame,
    results: pd.DataFrame,
    *,
    benchmark_id: str = "random",
) -> pd.DataFrame:
    """Aggregate by band × model × τ_m × τ_g × horizon for one benchmark.

    Columns: n, mean_ret, median_ret, win_rate, best, worst, mean_mfe, mean_mae,
    mean_excess, n_assets_universe hint via attrs.
    """
    if signals is None or len(signals) == 0 or results is None or len(results) == 0:
        return pd.DataFrame()

    sig = signals[
        ["signal_id", "band", "model", "tau_m", "tau_g", "asset_id"]
    ].drop_duplicates("signal_id")
    sub = results[results["benchmark_id"] == benchmark_id].copy()
    if len(sub) == 0:
        # fall back: any first benchmark present
        if len(results) == 0:
            return pd.DataFrame()
        first = results["benchmark_id"].iloc[0]
        sub = results[results["benchmark_id"] == first].copy()
        benchmark_id = str(first)

    merged = sub.merge(sig, on="signal_id", how="inner")
    if len(merged) == 0:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        r = g["ret"]
        return pd.Series(
            {
                "n": int(len(g)),
                "mean_ret": float(r.mean()),
                "median_ret": float(r.median()),
                "win_rate": float((r > 0).mean()),
                "best": float(r.max()),
                "worst": float(r.min()),
                "mean_mfe": float(g["mfe"].mean()),
                "mean_mae": float(g["mae"].mean()),
                "mean_excess": float(g["excess_ret"].mean())
                if g["excess_ret"].notna().any()
                else np.nan,
                "median_excess": float(g["excess_ret"].median())
                if g["excess_ret"].notna().any()
                else np.nan,
                "n_unique_assets": int(g["asset_id"].nunique()),
            }
        )

    agg = (
        merged.groupby(["band", "model", "tau_m", "tau_g", "horizon_d"], sort=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    agg["benchmark_id"] = benchmark_id
    agg.attrs["small_sample"] = True
    agg.attrs["disclaimer"] = (
        "SMALL SAMPLE — universe still tiny (n=11 widened from 3 is not enough); NOT conclusive; "
        "do not claim alpha."
    )
    return agg
