"""Narrative Acceleration Score (N) — attention velocity from social.

V0.1: combine social_daily mentions across sources → attention proxy →
velocity, acceleration, low-base multiplier → CS percentiles → score_N.

Does **not** invent social. Empty/missing social → null scores and
``n_available=false`` for those rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmram.features.lookbacks import DEFAULT_LOOKBACKS
from cmram.features.prep import group_rolling, safe_div
from cmram.features.rank import mean_available

N_INPUT_COLS = (
    "n_velocity",
    "n_accel",
    "n_vel_lowbase",
    "n_accel_lowbase",
)


def combine_attention(
    social: pd.DataFrame,
    *,
    method: str = "mean",
    attention_floor: float = 0.1,
) -> pd.DataFrame:
    """Collapse multi-source social_daily to one attention series per day/asset.

    Returns columns: timestamp, asset_id, attention, sources_present (str).
    """
    if social is None or len(social) == 0:
        return pd.DataFrame(columns=["timestamp", "asset_id", "attention", "sources_present"])

    df = social.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str)
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce")
    df = df.dropna(subset=["mentions"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "asset_id", "attention", "sources_present"])

    # Per (day, asset, source) first (last wins), then combine sources
    per_src = (
        df.groupby(["timestamp", "asset_id", "source"], sort=False)["mentions"]
        .mean()
        .reset_index()
    )
    src_list = (
        per_src.groupby(["timestamp", "asset_id"], sort=False)["source"]
        .agg(lambda s: ",".join(sorted(set(s.astype(str)))))
        .rename("sources_present")
    )
    if method == "sum":
        att = per_src.groupby(["timestamp", "asset_id"], sort=False)["mentions"].sum()
    elif method == "max":
        att = per_src.groupby(["timestamp", "asset_id"], sort=False)["mentions"].max()
    else:
        att = per_src.groupby(["timestamp", "asset_id"], sort=False)["mentions"].mean()

    out = att.rename("attention").reset_index()
    out = out.merge(src_list.reset_index(), on=["timestamp", "asset_id"], how="left")
    floor = float(attention_floor)
    out["attention"] = out["attention"].clip(lower=floor)
    return out


def narrative_inputs(
    social: pd.DataFrame,
    *,
    lookbacks: dict | None = None,
    combine: str = "mean",
    attention_floor: float = 0.1,
) -> pd.DataFrame:
    """Raw N inputs + baseline attention (for nsi_social), one row per day/asset."""
    lb = dict(DEFAULT_LOOKBACKS)
    if lookbacks:
        lb.update(lookbacks)
    short = int(lb.get("n_vel_short_days") or lb.get("narrative_short_days") or 7)
    long = int(lb.get("n_vel_long_days") or lb.get("narrative_long_days") or 28)
    accel_shift = int(lb.get("n_accel_days") or short)

    att = combine_attention(social, method=combine, attention_floor=attention_floor)
    if att.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "asset_id",
                "attention",
                "n_baseline",
                "n_velocity",
                "n_accel",
                "n_vel_lowbase",
                "n_accel_lowbase",
                "sources_present",
            ]
        )

    df = att.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    # Dense calendar per asset so rolling windows are day-based even if source is weekly
    frames = []
    for aid, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp")
        idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="D")
        daily = g.set_index("timestamp").reindex(idx)
        daily["attention"] = daily["attention"].ffill()
        daily["sources_present"] = daily["sources_present"].ffill()
        daily["asset_id"] = aid
        daily = daily.reset_index().rename(columns={"index": "timestamp"})
        frames.append(daily)
    dens = pd.concat(frames, ignore_index=True)
    dens["timestamp"] = pd.to_datetime(dens["timestamp"]).dt.tz_localize(None).dt.normalize()

    mean_s = group_rolling(dens, "attention", short, "mean", min_periods=max(2, short // 2))
    mean_l = group_rolling(dens, "attention", long, "mean", min_periods=max(3, long // 2))
    velocity = safe_div(mean_s, mean_l) - 1.0
    vel_prev = velocity.groupby(dens["asset_id"], sort=False).shift(accel_shift)
    accel = velocity - vel_prev
    baseline = mean_l
    # Low-base multiplier: boost rising attention from a quiet baseline
    mult = 1.0 / (1.0 + np.log1p(baseline.clip(lower=0)))
    dens["n_baseline"] = baseline
    dens["n_velocity"] = velocity
    dens["n_accel"] = accel
    dens["n_vel_lowbase"] = velocity * mult
    dens["n_accel_lowbase"] = accel * mult
    return dens[
        [
            "timestamp",
            "asset_id",
            "attention",
            "n_baseline",
            "n_velocity",
            "n_accel",
            "n_vel_lowbase",
            "n_accel_lowbase",
            "sources_present",
        ]
    ]


def score_N(
    social: pd.DataFrame | None,
    *,
    lookbacks: dict | None = None,
    index: pd.Index | None = None,
    combine: str = "mean",
    attention_floor: float = 0.1,
) -> pd.Series:
    """Narrative score 0–100.

    Without band context this ranks within each timestamp only (standalone).
    Pipeline uses narrative_inputs + band CS ranks instead. If social is empty
    and ``index`` is given, returns nulls aligned to index (no imputation).
    """
    if social is None or len(social) == 0:
        if index is None:
            return pd.Series(dtype="float64", name="score_N")
        return pd.Series(np.nan, index=index, dtype="float64", name="score_N")

    raw = narrative_inputs(
        social, lookbacks=lookbacks, combine=combine, attention_floor=attention_floor
    )
    if raw.empty:
        if index is None:
            return pd.Series(dtype="float64", name="score_N")
        return pd.Series(np.nan, index=index, dtype="float64", name="score_N")

    from cmram.features.rank import cs_percentile

    ranked = raw.copy()
    for col in N_INPUT_COLS:
        ranked[f"{col}_pct"] = ranked.groupby("timestamp", sort=False)[col].transform(
            cs_percentile
        )
    scores = mean_available(ranked, [f"{c}_pct" for c in N_INPUT_COLS])
    scores.name = "score_N"
    if index is not None:
        # Cannot align without keys; return nulls for standalone index use
        return pd.Series(np.nan, index=index, dtype="float64", name="score_N")
    return scores
