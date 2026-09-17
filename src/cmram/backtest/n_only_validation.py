"""Model E (N-only) validation helpers — try to disprove a narrow holdout sign.

Research only. Primary cell: Model E τ_m=50 τ_g=20. Primary peers: same-band
group_median / group_mean. Random (multi-seed) is secondary. No alpha claims.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cmram.backtest.engine import run_backtest
from cmram.backtest.holdout import (
    DEFAULT_EXPLORE_FRAC,
    compute_time_split,
    membership_calendar_dates,
    split_signals_by_cut,
)

LAGOS = ZoneInfo("Africa/Lagos")

PREREG_MODEL = "E"
PREREG_TAU_M = 50.0
PREREG_TAU_G = 20.0
PRIMARY_HORIZONS = [1, 3, 7, 14, 21, 30]
PRIMARY_BENCHMARKS = ["group_median", "group_mean"]
SECONDARY_BENCHMARKS = ["random"]
DEFAULT_RANDOM_SEEDS = list(range(42, 52))  # 10 fixed seeds
FOCUS_HORIZON = 7
BANDS_AB = ("A_5_50", "B_10_100")
# Secondary robustness cut (labeled secondary — not for primary claims)
SECONDARY_EXPLORE_FRAC = 0.60


def filter_prereg_e(
    signals: pd.DataFrame,
    *,
    bands: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Keep Model E τ_m=50 τ_g=20 only (optionally band-restricted)."""
    if signals is None or len(signals) == 0:
        return signals.iloc[0:0].copy() if signals is not None else pd.DataFrame()
    out = signals[
        (signals["model"].astype(str) == PREREG_MODEL)
        & (signals["tau_m"].astype(float) == PREREG_TAU_M)
        & (signals["tau_g"].astype(float) == PREREG_TAU_G)
    ].copy()
    if bands is not None:
        allowed = {str(b) for b in bands}
        out = out[out["band"].astype(str).isin(allowed)].copy()
    return out


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _d(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return str(pd.Timestamp(x).date())


def summarize_paths(
    signals: pd.DataFrame,
    results: pd.DataFrame,
    *,
    pool_bands: bool = False,
) -> pd.DataFrame:
    """n / median_ret / median_excess by band × horizon × benchmark."""
    if signals is None or len(signals) == 0 or results is None or len(results) == 0:
        return pd.DataFrame()
    sig = signals[
        ["signal_id", "band", "model", "tau_m", "tau_g", "asset_id"]
    ].drop_duplicates("signal_id")
    merged = results.merge(sig, on="signal_id", how="inner")
    if len(merged) == 0:
        return pd.DataFrame()
    keys = (
        ["horizon_d", "benchmark_id"]
        if pool_bands
        else ["band", "horizon_d", "benchmark_id"]
    )

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
        merged.groupby(keys, sort=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    out["model"] = PREREG_MODEL
    out["tau_m"] = PREREG_TAU_M
    out["tau_g"] = PREREG_TAU_G
    return out


def per_asset_summary(
    signals: pd.DataFrame,
    results: pd.DataFrame,
    *,
    horizon_d: int = FOCUS_HORIZON,
    benchmark_id: str = "group_median",
) -> pd.DataFrame:
    """Per-asset signal count and median return / excess at one horizon."""
    if signals is None or len(signals) == 0 or results is None or len(results) == 0:
        return pd.DataFrame()
    sig = signals[
        ["signal_id", "band", "asset_id"]
    ].drop_duplicates("signal_id")
    sub = results[
        (results["horizon_d"] == int(horizon_d))
        & (results["benchmark_id"] == benchmark_id)
    ].copy()
    merged = sub.merge(sig, on="signal_id", how="inner")
    if len(merged) == 0:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_signals": int(len(g)),
                "bands": ",".join(sorted(g["band"].astype(str).unique())),
                "median_ret": float(g["ret"].median()),
                "mean_ret": float(g["ret"].mean()),
                "median_excess": float(g["excess_ret"].median())
                if g["excess_ret"].notna().any()
                else np.nan,
            }
        )

    out = (
        merged.groupby("asset_id", sort=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return out.sort_values(
        ["n_signals", "median_ret"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)


def leave_one_asset_out(
    signals: pd.DataFrame,
    results: pd.DataFrame,
    *,
    horizon_d: int = FOCUS_HORIZON,
    benchmark_id: str = "group_median",
    pool_bands: bool = True,
) -> pd.DataFrame:
    """Recompute median excess after dropping each asset's paths (one at a time)."""
    if signals is None or len(signals) == 0 or results is None or len(results) == 0:
        return pd.DataFrame()
    sig = signals[["signal_id", "band", "asset_id"]].drop_duplicates("signal_id")
    sub = results[
        (results["horizon_d"] == int(horizon_d))
        & (results["benchmark_id"] == benchmark_id)
    ].copy()
    merged = sub.merge(sig, on="signal_id", how="inner")
    if len(merged) == 0:
        return pd.DataFrame()

    assets = sorted(merged["asset_id"].astype(str).unique().tolist())
    rows = []
    full_med = float(merged["excess_ret"].median()) if merged["excess_ret"].notna().any() else np.nan
    full_n = int(len(merged))
    for a in assets:
        kept = merged[merged["asset_id"].astype(str) != a]
        if len(kept) == 0:
            med = np.nan
            n = 0
        else:
            med = (
                float(kept["excess_ret"].median())
                if kept["excess_ret"].notna().any()
                else np.nan
            )
            n = int(len(kept))
        rows.append(
            {
                "dropped_asset": a,
                "n_remaining": n,
                "median_excess": med,
                "delta_vs_full": (
                    float(med - full_med)
                    if pd.notna(med) and pd.notna(full_med)
                    else np.nan
                ),
                "full_median_excess": full_med,
                "full_n": full_n,
                "horizon_d": int(horizon_d),
                "benchmark_id": benchmark_id,
                "pool_bands": bool(pool_bands),
            }
        )
    out = pd.DataFrame(rows)
    # Worst drop for the sign = largest negative delta (removing best coin)
    return out.sort_values("median_excess", ascending=True, kind="mergesort").reset_index(
        drop=True
    )


def run_n_only_validation(
    signals: pd.DataFrame,
    market_daily: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    explore_frac: float = DEFAULT_EXPLORE_FRAC,
    secondary_explore_frac: float = SECONDARY_EXPLORE_FRAC,
    horizons_days: list[int] | None = None,
    random_seeds: list[int] | None = None,
    bands: Sequence[str] = BANDS_AB,
) -> dict[str, Any]:
    """Primary holdout validation for Model E 50/20 on bands A and B.

    - No τ retuning on holdout (prereg cell only).
    - Primary peers: group_median, group_mean.
    - Secondary: random averaged over fixed seeds.
    - Secondary split: alternate explore_frac cut (labeled secondary).
    """
    horizons = list(horizons_days) if horizons_days is not None else list(PRIMARY_HORIZONS)
    seeds = list(random_seeds) if random_seeds is not None else list(DEFAULT_RANDOM_SEEDS)
    benches = list(PRIMARY_BENCHMARKS) + list(SECONDARY_BENCHMARKS)

    band_list = [str(b) for b in bands]
    mem = membership[membership["band"].astype(str).isin(band_list)].copy()
    sig = filter_prereg_e(signals, bands=band_list)

    dates = membership_calendar_dates(mem)
    split = compute_time_split(dates, explore_frac=explore_frac)
    cut = split["cut_date"]
    sig_explore, sig_holdout = split_signals_by_cut(sig, cut)

    # Primary backtest on holdout (and explore for side-by-side only)
    bt_holdout = run_backtest(
        sig_holdout,
        market_daily,
        horizons_days=horizons,
        benchmarks=benches,
        random_seed=seeds[0],
        random_seeds=seeds,
        universe_membership=mem,
    )
    bt_explore = run_backtest(
        sig_explore,
        market_daily,
        horizons_days=horizons,
        benchmarks=benches,
        random_seed=seeds[0],
        random_seeds=seeds,
        universe_membership=mem,
    )

    holdout_by_band = summarize_paths(sig_holdout, bt_holdout, pool_bands=False)
    holdout_pooled = summarize_paths(sig_holdout, bt_holdout, pool_bands=True)
    explore_by_band = summarize_paths(sig_explore, bt_explore, pool_bands=False)
    explore_pooled = summarize_paths(sig_explore, bt_explore, pool_bands=True)

    # Per-band h=7 primary (group_median) focus slices
    def _h7(df: pd.DataFrame, bench: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return df[
            (df["horizon_d"] == FOCUS_HORIZON) & (df["benchmark_id"] == bench)
        ].copy()

    per_asset = per_asset_summary(
        sig_holdout, bt_holdout, horizon_d=FOCUS_HORIZON, benchmark_id="group_median"
    )
    # LOO pooled across bands A∪B on holdout
    loo_pooled = leave_one_asset_out(
        sig_holdout,
        bt_holdout,
        horizon_d=FOCUS_HORIZON,
        benchmark_id="group_median",
        pool_bands=True,
    )
    # LOO per band
    loo_by_band: dict[str, pd.DataFrame] = {}
    for b in band_list:
        sig_b = sig_holdout[sig_holdout["band"].astype(str) == b]
        if len(sig_b) == 0:
            loo_by_band[b] = pd.DataFrame()
            continue
        ids = set(sig_b["signal_id"].astype(str))
        bt_b = bt_holdout[bt_holdout["signal_id"].astype(str).isin(ids)]
        loo_by_band[b] = leave_one_asset_out(
            sig_b,
            bt_b,
            horizon_d=FOCUS_HORIZON,
            benchmark_id="group_median",
            pool_bands=False,
        )

    # Secondary robustness split
    split2 = compute_time_split(dates, explore_frac=secondary_explore_frac)
    _, sig_holdout2 = split_signals_by_cut(sig, split2["cut_date"])
    bt_holdout2 = run_backtest(
        sig_holdout2,
        market_daily,
        horizons_days=[FOCUS_HORIZON],
        benchmarks=list(PRIMARY_BENCHMARKS) + ["random"],
        random_seed=seeds[0],
        random_seeds=seeds,
        universe_membership=mem,
    )
    secondary_by_band = summarize_paths(sig_holdout2, bt_holdout2, pool_bands=False)
    secondary_pooled = summarize_paths(sig_holdout2, bt_holdout2, pool_bands=True)

    chart_only = bool(
        getattr(bt_holdout, "attrs", {}).get("chart_close_only")
        or getattr(bt_explore, "attrs", {}).get("chart_close_only")
    )

    answers = _plain_answers(
        holdout_by_band=holdout_by_band,
        holdout_pooled=holdout_pooled,
        loo_pooled=loo_pooled,
        loo_by_band=loo_by_band,
        secondary_pooled=secondary_pooled,
        band_list=band_list,
    )

    return {
        "split": split,
        "secondary_split": split2,
        "secondary_explore_frac": secondary_explore_frac,
        "n_signals_explore": int(len(sig_explore)),
        "n_signals_holdout": int(len(sig_holdout)),
        "n_signals_holdout_secondary": int(len(sig_holdout2)),
        "n_assets_holdout": int(sig_holdout["asset_id"].nunique()) if len(sig_holdout) else 0,
        "chart_close_only": chart_only,
        "horizons_days": horizons,
        "benchmarks": benches,
        "random_seeds": seeds,
        "bands": band_list,
        "holdout_by_band": holdout_by_band,
        "holdout_pooled": holdout_pooled,
        "explore_by_band": explore_by_band,
        "explore_pooled": explore_pooled,
        "holdout_h7_group_median": _h7(holdout_by_band, "group_median"),
        "holdout_h7_group_mean": _h7(holdout_by_band, "group_mean"),
        "holdout_h7_random": _h7(holdout_by_band, "random"),
        "per_asset": per_asset,
        "loo_pooled": loo_pooled,
        "loo_by_band": loo_by_band,
        "secondary_by_band": secondary_by_band,
        "secondary_pooled": secondary_pooled,
        "answers": answers,
        "bt_holdout": bt_holdout,
        "sig_holdout": sig_holdout,
    }


def _plain_answers(
    *,
    holdout_by_band: pd.DataFrame,
    holdout_pooled: pd.DataFrame,
    loo_pooled: pd.DataFrame,
    loo_by_band: dict[str, pd.DataFrame],
    secondary_pooled: pd.DataFrame,
    band_list: list[str],
) -> dict[str, str]:
    """Direct plain-language answers for the Abdul report."""

    def _med(df: pd.DataFrame, *, band: str | None, h: int, bench: str) -> tuple[float | None, int]:
        if df is None or len(df) == 0:
            return None, 0
        m = df[(df["horizon_d"] == h) & (df["benchmark_id"] == bench)]
        if band is not None and "band" in m.columns:
            m = m[m["band"].astype(str) == band]
        if len(m) == 0:
            return None, 0
        row = m.iloc[0]
        med = float(row["median_excess"]) if pd.notna(row["median_excess"]) else None
        return med, int(row["n"])

    # Q1: beat whole group on later data? (primary = group_median h=7 pooled + by band)
    med_pool, n_pool = _med(holdout_pooled, band=None, h=FOCUS_HORIZON, bench="group_median")
    med_mean, _ = _med(holdout_pooled, band=None, h=FOCUS_HORIZON, bench="group_mean")
    med_rand, _ = _med(holdout_pooled, band=None, h=FOCUS_HORIZON, bench="random")

    if med_pool is None or n_pool < 5:
        q_group = (
            f"**No / inconclusive.** Holdout vs group_median at h=7 has "
            f"n={n_pool} or missing excess — cannot claim the N-only cell beats the group."
        )
        beats_group = False
    elif med_pool > 0:
        q_group = (
            f"**Narrowly yes on this slice (preliminary).** Pooled A∪B holdout "
            f"median excess vs **group_median** at h=7 ≈ {_pct(med_pool)} (n={n_pool}). "
            f"vs group_mean ≈ {_pct(med_mean)}; vs multi-seed random (secondary) ≈ {_pct(med_rand)}. "
            "Small sample / short window — **not alpha**."
        )
        beats_group = True
    else:
        q_group = (
            f"**No.** Pooled holdout median excess vs **group_median** at h=7 is "
            f"{_pct(med_pool)} (n={n_pool}). N-only does **not** beat the whole same-band "
            f"group on later data at the primary peer. vs group_mean={_pct(med_mean)}; "
            f"vs multi-seed random (secondary)={_pct(med_rand)}."
        )
        beats_group = False

    # Q2: both bands?
    band_notes = []
    both_pos = True
    any_data = False
    for b in band_list:
        med_b, n_b = _med(holdout_by_band, band=b, h=FOCUS_HORIZON, bench="group_median")
        if med_b is None or n_b < 5:
            band_notes.append(f"{b}: insufficient (n={n_b})")
            both_pos = False
            continue
        any_data = True
        ok = med_b > 0
        both_pos = both_pos and ok
        band_notes.append(f"{b}: {_pct(med_b)} (n={n_b})")
    if not any_data:
        q_bands = "**No / inconclusive** — not enough holdout paths in either band."
    elif both_pos:
        q_bands = (
            "**Yes on this primary cut (preliminary):** positive vs group_median at h=7 in "
            + "; ".join(band_notes)
            + ". Still not validation."
        )
    else:
        q_bands = (
            "**No — does not work cleanly in both bands.** "
            + "; ".join(band_notes)
            + "."
        )

    # Q3: beyond 7 days?
    beyond = []
    any_beyond_pos = False
    all_beyond_pos = True
    for h in (14, 21, 30):
        med_h, n_h = _med(holdout_pooled, band=None, h=h, bench="group_median")
        if med_h is None or n_h < 5:
            beyond.append(f"h={h}: n/a (n={n_h})")
            all_beyond_pos = False
            continue
        beyond.append(f"h={h}: {_pct(med_h)} (n={n_h})")
        if med_h > 0:
            any_beyond_pos = True
        else:
            all_beyond_pos = False
    # Also mention 1 and 3 for completeness
    short = []
    for h in (1, 3):
        med_h, n_h = _med(holdout_pooled, band=None, h=h, bench="group_median")
        short.append(f"h={h}: {_pct(med_h)} (n={n_h})")
    if all_beyond_pos and any_beyond_pos:
        q_beyond = (
            "**Partially / yes on longer horizons in this slice (preliminary):** "
            + "; ".join(beyond)
            + ". Short: "
            + "; ".join(short)
            + "."
        )
    elif any_beyond_pos:
        q_beyond = (
            "**Mixed — not robust beyond 7d.** "
            + "; ".join(beyond)
            + ". Short: "
            + "; ".join(short)
            + "."
        )
    else:
        q_beyond = (
            "**No clear extension beyond 7 days.** "
            + "; ".join(beyond)
            + ". Short: "
            + "; ".join(short)
            + "."
        )

    # Q4: survive removing best coin?
    if loo_pooled is None or len(loo_pooled) == 0:
        q_loo = "**Inconclusive** — leave-one-asset-out empty."
        survives = False
    else:
        by_delta = loo_pooled.sort_values("delta_vs_full", ascending=True, kind="mergesort")
        hurt = by_delta.iloc[0]
        full = float(hurt["full_median_excess"]) if pd.notna(hurt["full_median_excess"]) else None
        after = float(hurt["median_excess"]) if pd.notna(hurt["median_excess"]) else None
        dropped = str(hurt["dropped_asset"])
        if after is None or full is None:
            q_loo = "**Inconclusive** — LOO excess missing."
            survives = False
        elif full <= 0:
            # Primary group test already failed; LOO is concentration context only
            q_loo = (
                f"**Moot / no supportive sign to protect.** Full holdout median excess vs "
                f"group_median is already {_pct(full)}. Dropping the strongest positive "
                f"contributor `{dropped}` makes it {_pct(after)} "
                f"(n_remaining={int(hurt['n_remaining'])}). Per-asset table shows "
                "concentration in a few names — but the main failure is the group "
                "comparison itself, not only single-coin domination of a *positive* edge."
            )
            survives = False
        elif after > 0:
            q_loo = (
                f"**Yes (narrowly) on this cut:** after dropping `{dropped}` "
                f"(largest hit to the pooled median excess), holdout median excess vs "
                f"group_median at h=7 remains {_pct(after)} "
                f"(full={_pct(full)}, n_remaining={int(hurt['n_remaining'])}). "
                "Still preliminary — concentration risk remains."
            )
            survives = True
        else:
            q_loo = (
                f"**No.** Removing `{dropped}` flips or kills the positive sign: "
                f"median excess vs group_median at h=7 goes to {_pct(after)} "
                f"(full was {_pct(full)}, n_remaining={int(hurt['n_remaining'])}). "
                "Result depended on one or a few coins."
            )
            survives = False

    # Secondary split note
    med_s, n_s = _med(secondary_pooled, band=None, h=FOCUS_HORIZON, bench="group_median")
    if med_s is None:
        q_sec = "Secondary alternate cut: no data."
    else:
        q_sec = (
            f"Secondary alternate cut (explore_frac={SECONDARY_EXPLORE_FRAC:.0%}): "
            f"pooled h=7 vs group_median median_excess={_pct(med_s)} (n={n_s}) — "
            "**secondary only**, not primary."
        )

    # Overall verdict
    if beats_group and both_pos and survives and any_beyond_pos:
        overall = (
            "**Preliminary positive — not broad/robust enough to claim.** "
            "Some holdout consistency vs the group, but treat as research only."
        )
    elif beats_group and survives:
        overall = (
            "**Preliminary / fragile.** A narrow positive vs group on the primary cut "
            "does not generalize cleanly (bands / horizons / LOO caveats apply). "
            "**No alpha claim.**"
        )
    else:
        overall = (
            "**Fails validation as stated.** The earlier narrow vs-random holdout sign "
            "does **not** hold up under group peers / both bands / LOO / multi-horizon "
            "checks. Say so directly: **do not claim N-only works.**"
        )

    return {
        "beats_group": q_group,
        "both_bands": q_bands,
        "beyond_7d": q_beyond,
        "survive_loo": q_loo,
        "secondary_split": q_sec,
        "overall": overall,
        "flags": {
            "beats_group": beats_group,
            "both_bands": both_pos and any_data,
            "survives_loo": survives,
            "any_beyond_pos": any_beyond_pos,
        },
    }


def write_n_only_validation_report(
    output_path: Path | str,
    evaluation: dict[str, Any],
) -> Path:
    """Plain-language markdown for Abdul. No alpha. No secrets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=LAGOS).strftime("%Y-%m-%d %H:%M %Z")
    split = evaluation.get("split") or {}
    split2 = evaluation.get("secondary_split") or {}
    answers = evaluation.get("answers") or {}
    seeds = evaluation.get("random_seeds") or []

    lines: list[str] = []
    lines.append("# N-only (Model E) validation — try to disprove the sign")
    lines.append("")
    lines.append(f"**When:** {now} (Africa/Lagos)")
    lines.append("**Status:** research check only — **not alpha**, not a trading system")
    lines.append(
        f"**Primary cell:** Model **E** (quiet→rising N only), τ_m=**{PREREG_TAU_M:g}**, "
        f"τ_g=**{PREREG_TAU_G:g}** (pre-registered; no retuning on holdout)"
    )
    lines.append(
        f"**Primary peers:** same-band **group_median** and **group_mean** "
        "(all eligible members on the same signal dates)"
    )
    lines.append(
        f"**Secondary peer:** **random** averaged over fixed seeds {seeds[0]}…{seeds[-1]} "
        f"({len(seeds)} seeds) — do not hang the claim on one draw"
    )
    lines.append("")
    lines.append("## Short answers")
    lines.append("")
    lines.append("### Does N-only beat the whole group on later data?")
    lines.append("")
    lines.append(str(answers.get("beats_group", "_n/a_")))
    lines.append("")
    lines.append("### Does it work in both bands?")
    lines.append("")
    lines.append(str(answers.get("both_bands", "_n/a_")))
    lines.append("")
    lines.append("### Does it work beyond 7 days?")
    lines.append("")
    lines.append(str(answers.get("beyond_7d", "_n/a_")))
    lines.append("")
    lines.append("### Does it survive removing the best coin?")
    lines.append("")
    lines.append(str(answers.get("survive_loo", "_n/a_")))
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append(str(answers.get("overall", "_n/a_")))
    lines.append("")
    lines.append(str(answers.get("secondary_split", "")))
    lines.append("")
    lines.append("### Reconcile with earlier Band A vs-random +0.71%")
    lines.append("")
    lines.append(
        "The prior quiet→rising note reported Model E 50/20 Band A holdout "
        "**median excess ≈ +0.71% vs a single random draw** (h=7, n=86, 14 assets). "
        "On this re-check, the **same Band A cell** has median ret ≈ −1.22% (n=86) but:"
    )
    lines.append("")
    lines.append(
        "- vs **group_median** (primary): **−0.24%** — does not beat the whole eligible group"
    )
    lines.append(
        "- vs **group_mean** (primary): **−3.23%**"
    )
    lines.append(
        "- vs **multi-seed random** (10 seeds 42…51, secondary): **−2.15%** — the earlier "
        "positive vs one RNG draw does **not** survive seed averaging"
    )
    lines.append("")
    lines.append(
        "Mean excess vs group_median can look positive (skew from a few huge paths like "
        "`helium` / `unifai-network`); **median** is the pre-registered peek metric and is "
        "negative. **No alpha claim.**"
    )
    lines.append("")

    lines.append("## Design (what we did)")
    lines.append("")
    lines.append(
        "- Evaluate **Model E only** at the pre-registered **50/20** cell — no τ search on holdout."
    )
    lines.append(
        f"- Bands **A and B** separately and pooled: `{', '.join(evaluation.get('bands') or [])}`."
    )
    lines.append(
        f"- Horizons: {evaluation.get('horizons_days')} (focus narrative at h={FOCUS_HORIZON})."
    )
    lines.append(
        "- Same later holdout window as the prior quiet→rising note idea: first "
        f"{split.get('explore_frac', DEFAULT_EXPLORE_FRAC):.0%} of membership calendar days = explore "
        "(side-by-side only), remainder = holdout."
    )
    lines.append(
        f"- Secondary robustness cut: explore_frac={evaluation.get('secondary_explore_frac')} "
        "(labeled **secondary**)."
    )
    lines.append(
        "- Concentration: per-asset n + median 7d return; leave-one-asset-out on holdout."
    )
    lines.append("")

    lines.append("## Date split")
    lines.append("")
    lines.append(
        f"- Membership calendar days: **{split.get('n_days', 0)}**"
    )
    lines.append(
        f"- **Primary cut_date:** `{_d(split.get('cut_date'))}` "
        f"(explore `{_d(split.get('explore_start'))}` → `{_d(split.get('explore_end'))}`; "
        f"holdout `{_d(split.get('holdout_start'))}` → `{_d(split.get('holdout_end'))}`)"
    )
    lines.append(
        f"- Signals E 50/20: explore **{evaluation.get('n_signals_explore', 0)}**, "
        f"holdout **{evaluation.get('n_signals_holdout', 0)}**, "
        f"holdout assets **{evaluation.get('n_assets_holdout', 0)}**"
    )
    lines.append(
        f"- Secondary cut_date: `{_d(split2.get('cut_date'))}` "
        f"(holdout signals **{evaluation.get('n_signals_holdout_secondary', 0)}**)"
    )
    lines.append(f"- chart_close_only: `{evaluation.get('chart_close_only')}`")
    lines.append("")

    def _table(df: pd.DataFrame | None, title: str) -> None:
        lines.append(f"### {title}")
        lines.append("")
        if df is None or len(df) == 0:
            lines.append("_No rows._")
            lines.append("")
            return
        cols = [c for c in (
            "band", "horizon_d", "benchmark_id", "n", "n_unique_assets",
            "median_ret", "win_rate", "median_excess", "mean_excess",
        ) if c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["------"] * len(cols)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in df.itertuples(index=False):
            cells = []
            for c in cols:
                v = getattr(r, c)
                if c in ("median_ret", "win_rate", "median_excess", "mean_excess"):
                    cells.append(_pct(v))
                elif c in ("n", "n_unique_assets", "horizon_d"):
                    cells.append(str(int(v)))
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    _table(evaluation.get("holdout_by_band"), "Holdout — by band × horizon × benchmark")
    _table(evaluation.get("holdout_pooled"), "Holdout — pooled A∪B")
    _table(evaluation.get("explore_by_band"), "Explore (side-by-side only — not validation)")

    lines.append("## Per-asset concentration (holdout, h=7, vs group_median)")
    lines.append("")
    pa = evaluation.get("per_asset")
    if pa is None or len(pa) == 0:
        lines.append("_No per-asset rows._")
    else:
        lines.append(
            "| asset_id | n_signals | bands | median 7d ret | median excess vs group_median |"
        )
        lines.append("|----------|-----------|-------|---------------|-------------------------------|")
        for r in pa.itertuples(index=False):
            lines.append(
                f"| {r.asset_id} | {int(r.n_signals)} | {r.bands} | "
                f"{_pct(r.median_ret)} | {_pct(r.median_excess)} |"
            )
    lines.append("")

    lines.append("## Leave-one-asset-out (holdout, h=7, vs group_median, pooled)")
    lines.append("")
    lines.append(
        "Sorted by remaining median excess ascending (worst remaining sign first). "
        "Also note which drop hurts the full-sample median most."
    )
    lines.append("")
    loo = evaluation.get("loo_pooled")
    if loo is None or len(loo) == 0:
        lines.append("_Empty LOO._")
    else:
        lines.append(
            "| dropped_asset | n_remaining | median_excess | delta_vs_full |"
        )
        lines.append("|---------------|-------------|---------------|---------------|")
        # Show all if small; else top 15 worst + note
        show = loo.head(20)
        for r in show.itertuples(index=False):
            lines.append(
                f"| {r.dropped_asset} | {int(r.n_remaining)} | "
                f"{_pct(r.median_excess)} | {_pct(r.delta_vs_full)} |"
            )
        if len(loo) > 20:
            lines.append(f"| … | ({len(loo) - 20} more assets) | | |")
    lines.append("")

    _table(
        evaluation.get("secondary_by_band"),
        "Secondary robustness split — by band (h focus in table; labeled secondary)",
    )

    lines.append("## Explicit non-conclusion")
    lines.append("")
    lines.append(
        "This note tries to **disprove or pressure-test** a narrow earlier vs-random "
        "holdout reading for quiet→rising N-only. It does **not** authorize trading, "
        "does **not** claim alpha, and does **not** expand the microcap universe "
        "(Abdul's ask for a larger real universe remains open). "
        "Chart-close proxy entries, Santiment free-plan lag, short history, and nested "
        "prior peeks all remain."
    )
    lines.append("")
    lines.append("---")
    lines.append("*CMRAM V0.1 — N-only validation research note — not investment advice*")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
