"""Time holdout evaluation: explore (early) vs holdout (late) on membership dates.

Research only — no alpha claims. Full-sample τ grids are NOT validated.
Split history into earlier = exploration, later = holdout; report both.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cmram.backtest.engine import aggregate_backtest, run_backtest

LAGOS = ZoneInfo("Africa/Lagos")

DEFAULT_EXPLORE_FRAC = 0.70
HOLDOUT_HORIZON = 7
HOLDOUT_BENCHMARK = "random"
# Minimum completable paths to nominate an explore top-1 peek cell.
# Tiny-n cells (n=1) can dominate median_excess by noise — still shown in grid.
DEFAULT_MIN_N_TOP = 20

# Pre-registered cells evaluated on holdout regardless of explore peek.
# Documented intentionally — do not treat as validated parameters.
# Model C (N / Santiment narrative) focused; B kept as no-N comparison.
PREREGISTERED_CELLS: tuple[dict[str, Any], ...] = (
    {"model": "B", "tau_m": 50.0, "tau_g": 20.0, "label": "prereg_B_50_20"},
    {"model": "C", "tau_m": 50.0, "tau_g": 20.0, "label": "prereg_C_50_20"},
)
# Explore peek for holdout nomination is restricted to this model (N hypothesis).
EXPLORE_TOP_MODEL = "C"


def membership_calendar_dates(membership: pd.DataFrame) -> list[pd.Timestamp]:
    """Sorted unique calendar days present in universe_membership."""
    if membership is None or len(membership) == 0:
        return []
    ts = pd.to_datetime(membership["timestamp"]).dt.normalize().drop_duplicates()
    return sorted(pd.Timestamp(t).normalize() for t in ts.tolist())


def compute_time_split(
    dates: Sequence[pd.Timestamp | date | datetime | str],
    *,
    explore_frac: float = DEFAULT_EXPLORE_FRAC,
) -> dict[str, Any]:
    """Split sorted unique calendar dates into explore (early) / holdout (late).

    Uses floor(n * explore_frac) days for explore; remainder for holdout.
    ``cut_date`` is the first holdout calendar day (signals with timestamp >= cut
    belong to holdout; timestamp < cut belong to explore).

    Returns dict with:
      n_days, n_explore, n_holdout, explore_frac, explore_start, explore_end,
      holdout_start, holdout_end, cut_date, explore_dates, holdout_dates
    """
    if not (0.0 < float(explore_frac) < 1.0):
        raise ValueError(f"explore_frac must be in (0,1), got {explore_frac}")

    uniq = sorted(
        {pd.Timestamp(d).normalize() for d in dates}
    )
    n = len(uniq)
    if n == 0:
        return {
            "n_days": 0,
            "n_explore": 0,
            "n_holdout": 0,
            "explore_frac": float(explore_frac),
            "explore_start": None,
            "explore_end": None,
            "holdout_start": None,
            "holdout_end": None,
            "cut_date": None,
            "explore_dates": [],
            "holdout_dates": [],
        }
    if n == 1:
        # Degenerate: entire history is explore; no holdout day
        only = uniq[0]
        return {
            "n_days": 1,
            "n_explore": 1,
            "n_holdout": 0,
            "explore_frac": float(explore_frac),
            "explore_start": only,
            "explore_end": only,
            "holdout_start": None,
            "holdout_end": None,
            "cut_date": None,
            "explore_dates": [only],
            "holdout_dates": [],
        }

    n_explore = int(np.floor(n * float(explore_frac)))
    # Keep at least 1 day on each side when n >= 2
    n_explore = max(1, min(n_explore, n - 1))
    n_holdout = n - n_explore
    explore_dates = uniq[:n_explore]
    holdout_dates = uniq[n_explore:]
    cut = holdout_dates[0]
    return {
        "n_days": n,
        "n_explore": n_explore,
        "n_holdout": n_holdout,
        "explore_frac": float(explore_frac),
        "explore_start": explore_dates[0],
        "explore_end": explore_dates[-1],
        "holdout_start": holdout_dates[0],
        "holdout_end": holdout_dates[-1],
        "cut_date": cut,
        "explore_dates": explore_dates,
        "holdout_dates": holdout_dates,
    }


def split_signals_by_cut(
    signals: pd.DataFrame,
    cut_date: pd.Timestamp | date | datetime | str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition signals into explore (timestamp < cut) and holdout (>= cut)."""
    if signals is None or len(signals) == 0:
        empty = signals.iloc[0:0].copy() if signals is not None else pd.DataFrame()
        return empty, empty
    if cut_date is None:
        return signals.copy(), signals.iloc[0:0].copy()
    cut = pd.Timestamp(cut_date).normalize()
    ts = pd.to_datetime(signals["timestamp"]).dt.normalize()
    explore = signals.loc[ts < cut].copy()
    holdout = signals.loc[ts >= cut].copy()
    return explore, holdout


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _f(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _d(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return str(pd.Timestamp(x).date())


def summarize_tau_grid_h7(
    signals: pd.DataFrame,
    results: pd.DataFrame,
    *,
    horizon_d: int = HOLDOUT_HORIZON,
    benchmark_id: str = HOLDOUT_BENCHMARK,
    pool_bands: bool = True,
) -> pd.DataFrame:
    """Median / mean excess vs benchmark at a fixed horizon, by model × τ.

    When ``pool_bands`` is True, aggregates across bands for ranking clarity.
    Includes median_excess (primary peek metric) and mean_excess.
    """
    if signals is None or len(signals) == 0 or results is None or len(results) == 0:
        return pd.DataFrame()

    sig = signals[
        ["signal_id", "band", "model", "tau_m", "tau_g", "asset_id"]
    ].drop_duplicates("signal_id")
    sub = results[
        (results["benchmark_id"] == benchmark_id)
        & (results["horizon_d"] == int(horizon_d))
    ].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    merged = sub.merge(sig, on="signal_id", how="inner")
    if len(merged) == 0:
        return pd.DataFrame()

    group_keys = ["model", "tau_m", "tau_g"] if pool_bands else [
        "band",
        "model",
        "tau_m",
        "tau_g",
    ]

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": int(len(g)),
                "n_unique_assets": int(g["asset_id"].nunique()),
                "median_ret": float(g["ret"].median()),
                "mean_ret": float(g["ret"].mean()),
                "win_rate": float((g["ret"] > 0).mean()),
                "median_excess": float(g["excess_ret"].median())
                if g["excess_ret"].notna().any()
                else np.nan,
                "mean_excess": float(g["excess_ret"].mean())
                if g["excess_ret"].notna().any()
                else np.nan,
            }
        )

    out = (
        merged.groupby(group_keys, sort=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    out["horizon_d"] = int(horizon_d)
    out["benchmark_id"] = benchmark_id
    out = out.sort_values("median_excess", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    return out


def top_explore_cell(
    grid: pd.DataFrame,
    *,
    min_n: int = DEFAULT_MIN_N_TOP,
    models: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Return top (model, tau_m, tau_g) by median_excess among cells with n>=min_n.

    Full grid (including tiny-n cells) is still reported separately. If no cell
    meets ``min_n``, returns None (do not promote n=1 noise as explore top).
    When ``models`` is set, only those model letters are eligible for the peek
    (e.g. Model C for the N / Santiment holdout focus).
    """
    if grid is None or len(grid) == 0:
        return None
    ranked = grid.dropna(subset=["median_excess"]).copy()
    if models is not None:
        allowed = {str(m) for m in models}
        ranked = ranked[ranked["model"].astype(str).isin(allowed)]
    ranked = ranked[ranked["n"].astype(int) >= int(min_n)]
    ranked = ranked.sort_values(
        "median_excess", ascending=False, kind="mergesort"
    )
    if len(ranked) == 0:
        return None
    r = ranked.iloc[0]
    return {
        "model": str(r["model"]),
        "tau_m": float(r["tau_m"]),
        "tau_g": float(r["tau_g"]),
        "median_excess": float(r["median_excess"]),
        "n": int(r["n"]),
        "min_n": int(min_n),
        "label": f"explore_top_{r['model']}_{r['tau_m']:g}_{r['tau_g']:g}",
    }


def _cell_key(model: str, tau_m: float, tau_g: float) -> tuple[str, float, float]:
    return (str(model), float(tau_m), float(tau_g))


def build_holdout_cell_set(explore_top: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Pre-registered cells + explore top-1 if different."""
    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for c in PREREGISTERED_CELLS:
        key = _cell_key(c["model"], c["tau_m"], c["tau_g"])
        cells.append(dict(c))
        seen.add(key)
    if explore_top is not None:
        key = _cell_key(explore_top["model"], explore_top["tau_m"], explore_top["tau_g"])
        if key not in seen:
            cells.append(
                {
                    "model": explore_top["model"],
                    "tau_m": explore_top["tau_m"],
                    "tau_g": explore_top["tau_g"],
                    "label": explore_top["label"],
                }
            )
            seen.add(key)
        else:
            # Mark which prereg cell matched explore top
            for c in cells:
                if _cell_key(c["model"], c["tau_m"], c["tau_g"]) == key:
                    c["also_explore_top"] = True
    return cells


def filter_signals_to_cells(
    signals: pd.DataFrame, cells: Sequence[dict[str, Any]]
) -> pd.DataFrame:
    if signals is None or len(signals) == 0 or not cells:
        return signals.iloc[0:0].copy() if signals is not None else pd.DataFrame()
    mask = pd.Series(False, index=signals.index)
    for c in cells:
        mask = mask | (
            (signals["model"].astype(str) == str(c["model"]))
            & (signals["tau_m"].astype(float) == float(c["tau_m"]))
            & (signals["tau_g"].astype(float) == float(c["tau_g"]))
        )
    return signals.loc[mask].copy()


def evaluate_cells(
    signals: pd.DataFrame,
    results: pd.DataFrame,
    cells: Sequence[dict[str, Any]],
    *,
    horizon_d: int = HOLDOUT_HORIZON,
    benchmark_id: str = HOLDOUT_BENCHMARK,
) -> pd.DataFrame:
    """Per-cell metrics on a window (signals already window-filtered)."""
    grid = summarize_tau_grid_h7(
        signals,
        results,
        horizon_d=horizon_d,
        benchmark_id=benchmark_id,
        pool_bands=True,
    )
    rows = []
    for c in cells:
        match = grid[
            (grid["model"].astype(str) == str(c["model"]))
            & (grid["tau_m"].astype(float) == float(c["tau_m"]))
            & (grid["tau_g"].astype(float) == float(c["tau_g"]))
        ]
        row = {
            "label": c.get("label", f"{c['model']}_{c['tau_m']:g}_{c['tau_g']:g}"),
            "model": str(c["model"]),
            "tau_m": float(c["tau_m"]),
            "tau_g": float(c["tau_g"]),
            "also_explore_top": bool(c.get("also_explore_top", False)),
            "n": 0,
            "n_unique_assets": 0,
            "median_ret": np.nan,
            "mean_ret": np.nan,
            "win_rate": np.nan,
            "median_excess": np.nan,
            "mean_excess": np.nan,
            "horizon_d": int(horizon_d),
            "benchmark_id": benchmark_id,
        }
        if len(match):
            m = match.iloc[0]
            row.update(
                {
                    "n": int(m["n"]),
                    "n_unique_assets": int(m["n_unique_assets"]),
                    "median_ret": float(m["median_ret"]),
                    "mean_ret": float(m["mean_ret"]),
                    "win_rate": float(m["win_rate"]),
                    "median_excess": float(m["median_excess"])
                    if pd.notna(m["median_excess"])
                    else np.nan,
                    "mean_excess": float(m["mean_excess"])
                    if pd.notna(m["mean_excess"])
                    else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def support_statement(
    explore_top: dict[str, Any] | None,
    holdout_eval: pd.DataFrame,
    *,
    min_n: int = 5,
) -> str:
    """Plain-language: whether holdout supports or fails to support the explore peek.

    Conservative: positive median_excess on holdout for the explore-top cell
    (and/or overlapping prereg) with n>=min_n → weak support language only.
    Never claims alpha.
    """
    if explore_top is None:
        return (
            "No explore-top cell (empty explore grid). Holdout cannot confirm or "
            "refute a peek. **No support for any τ choice.**"
        )
    if holdout_eval is None or len(holdout_eval) == 0:
        return (
            "Holdout evaluation empty. **Fails to support** the explore peek "
            "(insufficient holdout data)."
        )

    key = _cell_key(explore_top["model"], explore_top["tau_m"], explore_top["tau_g"])
    match = holdout_eval[
        (holdout_eval["model"].astype(str) == key[0])
        & (holdout_eval["tau_m"].astype(float) == key[1])
        & (holdout_eval["tau_g"].astype(float) == key[2])
    ]
    if len(match) == 0:
        return (
            f"Explore peek cell {key[0]} τ_m={key[1]:g} τ_g={key[2]:g} was not "
            "evaluated on holdout. **Fails to support** the explore peek."
        )
    row = match.iloc[0]
    n = int(row["n"])
    med_ex = row["median_excess"]
    if n < min_n or pd.isna(med_ex):
        return (
            f"Explore peek ({key[0]} τ_m={key[1]:g} τ_g={key[2]:g}) has holdout "
            f"n={n} / median_excess={_pct(med_ex) if pd.notna(med_ex) else 'n/a'}. "
            "**Fails to support** the explore peek (too few holdout completable "
            "paths or missing excess)."
        )
    if float(med_ex) > 0:
        return (
            f"Explore peek ({key[0]} τ_m={key[1]:g} τ_g={key[2]:g}) shows "
            f"**positive** holdout median excess vs {HOLDOUT_BENCHMARK} at h={HOLDOUT_HORIZON} "
            f"(median_excess={_pct(float(med_ex))}, n={n}). "
            "This is **weak directional consistency only** — SMALL SAMPLE, nested τ "
            "peeked on explore, chart-close-only OHLC. **Does NOT validate alpha.** "
            "Treat as: holdout does not immediately reject the explore peek."
        )
    if float(med_ex) == 0:
        return (
            f"Explore peek ({key[0]} τ_m={key[1]:g} τ_g={key[2]:g}) has "
            f"**zero** holdout median excess (n={n}). **Fails to support** the "
            "explore peek as an improvement over random."
        )
    return (
        f"Explore peek ({key[0]} τ_m={key[1]:g} τ_g={key[2]:g}) shows "
        f"**negative** holdout median excess vs {HOLDOUT_BENCHMARK} at h={HOLDOUT_HORIZON} "
        f"(median_excess={_pct(float(med_ex))}, n={n}). "
        "**Fails to support** the explore peek — direction did not hold out of sample."
    )


def run_holdout_evaluation(
    signals: pd.DataFrame,
    market_daily: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    explore_frac: float = DEFAULT_EXPLORE_FRAC,
    horizons_days: list[int] | None = None,
    benchmarks: list[str] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Explore/holdout pipeline using existing backtest engine.

    Defaults to horizon h=7 and benchmark ``random`` only (the holdout peek
    metric) for speed. Pass fuller ``horizons_days`` / ``benchmarks`` if needed.

    Returns a dict with split meta, explore grid, holdout cell metrics, and
    a support statement. Does not write files.
    """
    # Holdout report centers on h=7 vs random; keep that lean by default.
    horizons = list(horizons_days) if horizons_days is not None else [HOLDOUT_HORIZON]
    if HOLDOUT_HORIZON not in horizons:
        horizons = sorted(set(horizons) | {HOLDOUT_HORIZON})
    benches = list(benchmarks) if benchmarks is not None else [HOLDOUT_BENCHMARK]
    if HOLDOUT_BENCHMARK not in benches:
        benches = list(benches) + [HOLDOUT_BENCHMARK]

    dates = membership_calendar_dates(membership)
    split = compute_time_split(dates, explore_frac=explore_frac)
    cut = split["cut_date"]

    sig_explore, sig_holdout = split_signals_by_cut(signals, cut)

    # Explore: full τ grid (all models/τ) at lean horizons/benchmarks
    bt_explore = run_backtest(
        sig_explore,
        market_daily,
        horizons_days=horizons,
        benchmarks=benches,
        random_seed=random_seed,
        universe_membership=membership,
    )

    explore_grid = summarize_tau_grid_h7(
        sig_explore, bt_explore, horizon_d=HOLDOUT_HORIZON, pool_bands=True
    )
    explore_grid_by_band = summarize_tau_grid_h7(
        sig_explore, bt_explore, horizon_d=HOLDOUT_HORIZON, pool_bands=False
    )
    explore_top = top_explore_cell(
        explore_grid, models=(EXPLORE_TOP_MODEL,)
    )
    cells = build_holdout_cell_set(explore_top)

    # Holdout: only the small pre-registered (+ optional top) cell set
    sig_h_cells = filter_signals_to_cells(sig_holdout, cells)
    bt_holdout = run_backtest(
        sig_h_cells,
        market_daily,
        horizons_days=horizons,
        benchmarks=benches,
        random_seed=random_seed + 1,
        universe_membership=membership,
    )
    holdout_eval = evaluate_cells(sig_h_cells, bt_holdout, cells)

    # Same cells on explore for side-by-side (filter explore BT; no re-run)
    sig_e_cells = filter_signals_to_cells(sig_explore, cells)
    if len(sig_e_cells) and len(bt_explore):
        ids_e = set(sig_e_cells["signal_id"].astype(str))
        bt_e_cells = bt_explore[bt_explore["signal_id"].astype(str).isin(ids_e)].copy()
    else:
        bt_e_cells = bt_explore.iloc[0:0].copy() if len(bt_explore) else bt_explore
    explore_eval = evaluate_cells(sig_e_cells, bt_e_cells, cells)

    agg_explore = aggregate_backtest(
        sig_explore, bt_explore, benchmark_id=HOLDOUT_BENCHMARK
    )
    agg_holdout = aggregate_backtest(
        sig_h_cells, bt_holdout, benchmark_id=HOLDOUT_BENCHMARK
    )

    statement = support_statement(explore_top, holdout_eval)
    model_c_verdict = model_c_holdout_verdict(holdout_eval)

    n_assets = int(
        getattr(bt_explore, "attrs", {}).get("n_assets")
        or (signals["asset_id"].nunique() if len(signals) else 0)
    )
    chart_only = bool(
        getattr(bt_explore, "attrs", {}).get("chart_close_only")
        or getattr(bt_holdout, "attrs", {}).get("chart_close_only")
    )

    return {
        "split": split,
        "n_signals_explore": int(len(sig_explore)),
        "n_signals_holdout": int(len(sig_holdout)),
        "n_assets": n_assets,
        "chart_close_only": chart_only,
        "explore_grid": explore_grid,
        "explore_grid_by_band": explore_grid_by_band,
        "explore_top": explore_top,
        "cells": cells,
        "explore_eval": explore_eval,
        "holdout_eval": holdout_eval,
        "aggregate_explore": agg_explore,
        "aggregate_holdout": agg_holdout,
        "support_statement": statement,
        "model_c_verdict": model_c_verdict,
        "horizons_days": horizons,
        "benchmarks": benches,
        "random_seed": random_seed,
    }


def model_c_holdout_verdict(holdout_eval: pd.DataFrame) -> str:
    """Plain-language verdict for pre-registered Model C vs B at τ_m=50 τ_g=20.

    Focused on whether holdout supports Model C / N. Never claims alpha.
    """
    if holdout_eval is None or len(holdout_eval) == 0:
        return (
            "**Verdict: inconclusive.** Holdout evaluation empty — cannot support "
            "or reject Model C / N."
        )

    def _row(model: str) -> pd.Series | None:
        m = holdout_eval[
            (holdout_eval["model"].astype(str) == model)
            & (holdout_eval["tau_m"].astype(float) == 50.0)
            & (holdout_eval["tau_g"].astype(float) == 20.0)
        ]
        return m.iloc[0] if len(m) else None

    row_c = _row("C")
    row_b = _row("B")
    if row_c is None:
        return (
            "**Verdict: fails to support Model C / N.** Pre-registered C τ_m=50 "
            "τ_g=20 was not evaluated on holdout."
        )

    n_c = int(row_c["n"])
    med_c = row_c["median_excess"]
    n_b = int(row_b["n"]) if row_b is not None else 0
    med_b = row_b["median_excess"] if row_b is not None else float("nan")

    parts = [
        f"Pre-registered **Model C** τ_m=50 τ_g=20 h=7 vs random: "
        f"holdout median_excess={_pct(med_c) if pd.notna(med_c) else 'n/a'} "
        f"(n={n_c})."
    ]
    if row_b is not None:
        parts.append(
            f"Same thresholds **Model B** (no N): "
            f"holdout median_excess={_pct(med_b) if pd.notna(med_b) else 'n/a'} "
            f"(n={n_b})."
        )

    if n_c < 5 or pd.isna(med_c):
        parts.append(
            "**Verdict: fails to support Model C / N** (too few holdout paths "
            "or missing excess). SMALL SAMPLE — not conclusive."
        )
    elif float(med_c) > 0:
        better = ""
        if row_b is not None and pd.notna(med_b) and float(med_c) > float(med_b):
            better = (
                " C's holdout median excess is higher than B's at the same τ — "
                "weak relative consistency only."
            )
        elif row_b is not None and pd.notna(med_b) and float(med_c) <= float(med_b):
            better = (
                " C did not beat B on holdout median excess at the same τ."
            )
        parts.append(
            "**Verdict: weak directional holdout consistency for Model C / N "
            "only** — positive median excess vs random."
            + better
            + " **Does NOT validate alpha or confirm Santiment N.** "
            "SMALL SAMPLE / nested peek / chart-close-only."
        )
    else:
        parts.append(
            "**Verdict: does not support Model C / N.** Holdout median excess "
            "vs random is not positive at the pre-registered cell. "
            "Direction failed to hold (or is zero). Not conclusive either way — "
            "do not claim alpha and do not reject the broader N idea from this "
            "alone."
        )
    return " ".join(parts)


def write_holdout_report(
    output_path: Path | str,
    evaluation: dict[str, Any],
) -> Path:
    """Write markdown holdout report with HARD warnings. No alpha claims."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split = evaluation.get("split") or {}
    now = datetime.now(tz=LAGOS).strftime("%Y-%m-%d %H:%M %Z")
    n_assets = int(evaluation.get("n_assets") or 0)
    chart_only = bool(evaluation.get("chart_close_only"))
    explore_top = evaluation.get("explore_top")
    explore_grid = evaluation.get("explore_grid")
    explore_eval = evaluation.get("explore_eval")
    holdout_eval = evaluation.get("holdout_eval")
    statement = str(evaluation.get("support_statement") or "")

    lines: list[str] = []
    lines.append("# CMRAM Phase 1 — Time Holdout Report (Model C / N focus)")
    lines.append("")
    lines.append(f"**Generated:** {now} (Africa/Lagos)")
    lines.append("**Status:** research exploration — **NOT a trading system**")
    lines.append("")
    lines.append("## ⚠️ HARD WARNINGS")
    lines.append("")
    lines.append(
        "> **DO NOT treat full-sample τ grids as validated.** "
        "This report splits membership history into an **explore** window "
        "(peek / summarize only) and a **holdout** window (pre-registered cells). "
        "Nested τ search on explore inflates multiple-testing risk. "
        f"**Universe n≈{n_assets} is still SMALL SAMPLE.** "
        "Chart-close-only OHLC (if noted) further weakens path metrics. "
        "**No alpha. No edge claim. Not conclusive. Do not trade on this.**"
    )
    lines.append("")
    lines.append(
        "> **Peeking disclaimer:** Ranking τ cells on the explore window is a "
        "*peek*, not a selection procedure that confers validity. Even if holdout "
        "median excess is positive, that is **weak directional consistency**, "
        "not statistical confirmation."
    )
    lines.append("")

    lines.append("## Intent")
    lines.append("")
    lines.append(
        "Earlier calendar days = exploration (summarize which τ look best; "
        "Model C top-1 peek for N / Santiment). "
        "Later calendar days = holdout (pre-registered **B** and **C** at "
        "τ_m=50 τ_g=20, plus explore top Model C cell if different). "
        "**Report both. No alpha claims.**"
    )
    lines.append("")

    lines.append("## Date split (from DB membership calendar)")
    lines.append("")
    lines.append(
        f"- Source: distinct `universe_membership.timestamp` calendar days "
        f"(n={split.get('n_days', 0)})"
    )
    lines.append(
        f"- Rule: first `{split.get('explore_frac', DEFAULT_EXPLORE_FRAC):.0%}` "
        f"of sorted unique days = explore "
        f"(`floor(n × frac)`), remainder = holdout"
    )
    lines.append(
        f"- **cut_date (first holdout day):** `{_d(split.get('cut_date'))}`"
    )
    lines.append(
        f"- Explore: `{_d(split.get('explore_start'))}` → `{_d(split.get('explore_end'))}` "
        f"({split.get('n_explore', 0)} days); signals with `timestamp < cut_date`"
    )
    lines.append(
        f"- Holdout: `{_d(split.get('holdout_start'))}` → `{_d(split.get('holdout_end'))}` "
        f"({split.get('n_holdout', 0)} days); signals with `timestamp >= cut_date`"
    )
    lines.append(
        f"- Signals: explore **{evaluation.get('n_signals_explore', 0)}**, "
        f"holdout **{evaluation.get('n_signals_holdout', 0)}**"
    )
    lines.append(f"- chart_close_only: `{chart_only}`")
    lines.append(
        f"- Peek metric: **median excess vs `{HOLDOUT_BENCHMARK}` at h={HOLDOUT_HORIZON}**"
    )
    lines.append("")

    lines.append("## Explore window — τ grid peek (h=7, excess vs random)")
    lines.append("")
    lines.append(
        "Bands pooled. Sorted by **median_excess** descending (all cells shown, "
        "including tiny-n). Top-1 nomination for holdout requires "
        "**n≥20** (`DEFAULT_MIN_N_TOP`). "
        "This is an **explore peek only** — do not auto-declare a winner."
    )
    lines.append("")
    if explore_grid is None or len(explore_grid) == 0:
        lines.append("_No explore aggregate rows._")
    else:
        lines.append(
            "| rank | model | τ_m | τ_g | n | n_assets | median ret | win% | "
            "**median excess** | mean excess |"
        )
        lines.append(
            "|------|-------|-----|-----|---|----------|------------|------|"
            "------------------|-------------|"
        )
        for i, r in enumerate(explore_grid.itertuples(index=False), start=1):
            lines.append(
                f"| {i} | {r.model} | {r.tau_m:g} | {r.tau_g:g} | {int(r.n)} | "
                f"{int(r.n_unique_assets)} | {_pct(r.median_ret)} | {_pct(r.win_rate)} | "
                f"**{_pct(r.median_excess)}** | {_pct(r.mean_excess)} |"
            )
    lines.append("")
    if explore_top:
        min_n = explore_top.get("min_n", DEFAULT_MIN_N_TOP)
        lines.append(
            f"**Explore top-1 (peek, n≥{min_n}):** Model {explore_top['model']} "
            f"τ_m={explore_top['tau_m']:g} τ_g={explore_top['tau_g']:g} "
            f"(median_excess={_pct(explore_top['median_excess'])}, n={explore_top['n']}). "
            "**Not a validated winner.**"
        )
    else:
        lines.append(
            f"**Explore top-1:** none (no cell with n≥{DEFAULT_MIN_N_TOP}, or empty grid)."
        )
    lines.append("")

    lines.append("## Pre-registered holdout set")
    lines.append("")
    lines.append(
        "Evaluated on holdout only for decision language. Includes:"
    )
    lines.append("- Model **B** τ_m=**50** τ_g=**20** (no-N comparison)")
    lines.append(
        "- Model **C** τ_m=**50** τ_g=**20** (N / Santiment narrative; draft prereg)"
    )
    lines.append(
        "- Explore top-1 **Model C** cell **if different** "
        "(peek — labeled as such)"
    )
    lines.append("")

    def _cell_table(df: pd.DataFrame | None, title: str) -> None:
        lines.append(f"### {title}")
        lines.append("")
        if df is None or len(df) == 0:
            lines.append("_No rows._")
            lines.append("")
            return
        lines.append(
            "| label | model | τ_m | τ_g | n | n_assets | median ret | win% | "
            "median excess | mean excess |"
        )
        lines.append(
            "|-------|-------|-----|-----|---|----------|------------|------|"
            "---------------|-------------|"
        )
        for r in df.itertuples(index=False):
            flag = " †" if getattr(r, "also_explore_top", False) else ""
            lines.append(
                f"| {r.label}{flag} | {r.model} | {r.tau_m:g} | {r.tau_g:g} | "
                f"{int(r.n)} | {int(r.n_unique_assets)} | {_pct(r.median_ret)} | "
                f"{_pct(r.win_rate)} | {_pct(r.median_excess)} | {_pct(r.mean_excess)} |"
            )
        lines.append("")
        lines.append("_† also matched explore top-1 peek_")
        lines.append("")

    _cell_table(explore_eval, "Same cells on EXPLORE (side-by-side; peeked)")
    _cell_table(holdout_eval, "Same cells on HOLDOUT (pre-registered eval)")

    lines.append("## Does holdout support the explore peek?")
    lines.append("")
    lines.append(statement)
    lines.append("")

    lines.append("## Does holdout support Model C / N?")
    lines.append("")
    verdict = str(evaluation.get("model_c_verdict") or "")
    lines.append(verdict if verdict else "_No Model C verdict computed._")
    lines.append("")

    lines.append("## Explicit non-conclusion")
    lines.append("")
    lines.append(
        "This holdout does **not** conclude that CMRAM works or fails as a strategy. "
        "It only checks whether an explore-window τ peek shows the same *direction* "
        "of median excess vs random on a later window. With a tiny eligible universe, "
        "short history, nested thresholds, and chart-close proxy entries, results remain "
        "**indistinguishable from noise** for any investment decision. "
        "**Do not trade on this. Do not claim alpha.**"
    )
    lines.append("")
    lines.append("---")
    lines.append("*CMRAM V0.1 Phase 1 research scaffold — time holdout*")

    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path
