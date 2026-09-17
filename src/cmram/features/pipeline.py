"""Orchestrate daily feature computation into features_daily shape.

Point-in-time: every rolling window and the band equal-weight RS benchmark
use only bars and membership available at t. Cross-sectional percentiles
are within ``(timestamp, band)``. Models A/B unchanged; Model C adds N when
social_daily is present (``n_available=true`` only on rows with N).

This stage does **not** generate signals or claim predictive power.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cmram.features.compression import C_INPUT_COLS, compression_inputs
from cmram.features.gap import rotation_gap
from cmram.features.lookbacks import resolve_feature_config
from cmram.features.mrei import compute_mrei
from cmram.features.narrative import N_INPUT_COLS, narrative_inputs
from cmram.features.nsi import NSI_MOM_COLS, NSI_PRICE_COLS, NSI_VOL_COLS, nsi_inputs
from cmram.features.price import P_INPUT_COLS, attach_relative_strength, price_inputs
from cmram.features.rank import attach_sample_size, mean_available, rank_columns
from cmram.features.volume import V_INPUT_COLS, volume_inputs

FEATURE_COLUMNS = (
    "timestamp",
    "asset_id",
    "band",
    "model",
    "score_C",
    "score_V",
    "score_N",
    "score_P",
    "MREI",
    "nsi_social",
    "nsi_price_ext",
    "nsi_vol_exh",
    "nsi_mom_dec",
    "NSI",
    "Rotation_Gap",
    "n_available",
    "small_sample",
    "n_in_band",
    "model_version",
)

_RAW_RANK_COLS = (
    list(C_INPUT_COLS)
    + list(V_INPUT_COLS)
    + list(P_INPUT_COLS)
    + list(NSI_PRICE_COLS)
    + list(NSI_VOL_COLS)
    + list(NSI_MOM_COLS)
)


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FEATURE_COLUMNS))


def _prepare_membership(universe_membership: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "asset_id", "band"}
    missing = required - set(universe_membership.columns)
    if missing:
        raise ValueError(f"universe_membership missing columns: {sorted(missing)}")
    mem = universe_membership.copy()
    mem["timestamp"] = pd.to_datetime(mem["timestamp"]).dt.tz_localize(None).dt.normalize()
    mem["asset_id"] = mem["asset_id"].astype(str)
    mem["band"] = mem["band"].astype(str)
    mem = mem.drop_duplicates(["timestamp", "asset_id", "band"], keep="last")
    return mem.reset_index(drop=True)


def compute_features_daily(
    market_daily: pd.DataFrame,
    universe_membership: pd.DataFrame,
    social_daily: pd.DataFrame | None = None,
    *,
    model_version: str | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute C/V/P/(N), NSI, MREI, Rotation_Gap per (t, asset, band, model).

    Args:
        market_daily: OHLCV + MC bars (history may predate membership).
        universe_membership: eligible (timestamp, asset_id, band) rows.
        social_daily: optional multi-source attention; when present and
            scorable, Model C uses N and ``n_available`` is set per row.
        model_version: override; default from config.
        config: features.yaml mapping or partial override.

    Returns:
        DataFrame matching ``features_daily`` columns. One row set per
        requested model (A/B always; C when listed in config).
    """
    cfg = resolve_feature_config(config)
    lb = cfg["lookbacks"]
    version = model_version or cfg["model_version"]
    models = [m for m in cfg["models"] if m in ("A", "B", "C")]
    if not models:
        models = ["A", "B"]

    if market_daily is None or len(market_daily) == 0:
        return _empty_features()
    if universe_membership is None or len(universe_membership) == 0:
        return _empty_features()

    mem = _prepare_membership(universe_membership)
    raw_c = compression_inputs(market_daily, lookbacks=lb)
    raw_v = volume_inputs(
        market_daily,
        lookbacks=lb,
        volume_floor_usd=cfg["volume_floor_usd"],
        rvol_clip=cfg["rvol_clip"],
    )
    raw_p = price_inputs(market_daily, lookbacks=lb)
    raw_nsi = nsi_inputs(market_daily, raw_v, lookbacks=lb)

    base = mem.merge(raw_c, on=["timestamp", "asset_id"], how="left", sort=False)
    base = base.merge(raw_v, on=["timestamp", "asset_id"], how="left", sort=False)
    base, rs_benchmark = attach_relative_strength(
        base,
        raw_p,
        market_daily,
        lookbacks=lb,
        btc_asset_ids=tuple(cfg["btc_asset_ids"]),
    )
    nsi_join = raw_nsi.copy()
    nsi_join["timestamp"] = pd.to_datetime(nsi_join["timestamp"]).dt.tz_localize(None).dt.normalize()
    nsi_cols = list(NSI_PRICE_COLS) + list(NSI_VOL_COLS) + list(NSI_MOM_COLS)
    base = base.merge(
        nsi_join[["timestamp", "asset_id"] + nsi_cols],
        on=["timestamp", "asset_id"],
        how="left",
        sort=False,
    )
    base.attrs["rs_benchmark"] = rs_benchmark

    # Narrative inputs (honest nulls when social absent)
    has_social = social_daily is not None and len(social_daily) > 0
    if has_social:
        raw_n = narrative_inputs(
            social_daily,
            lookbacks=lb,
            combine=str(cfg.get("social_combine") or "mean"),
            attention_floor=float(cfg.get("attention_floor") or 0.1),
        )
        if len(raw_n):
            raw_n = raw_n.copy()
            raw_n["timestamp"] = (
                pd.to_datetime(raw_n["timestamp"]).dt.tz_localize(None).dt.normalize()
            )
            n_cols = list(N_INPUT_COLS) + ["n_baseline", "attention"]
            base = base.merge(
                raw_n[["timestamp", "asset_id"] + n_cols],
                on=["timestamp", "asset_id"],
                how="left",
                sort=False,
            )
        else:
            for col in list(N_INPUT_COLS) + ["n_baseline", "attention"]:
                base[col] = np.nan
    else:
        for col in list(N_INPUT_COLS) + ["n_baseline", "attention"]:
            base[col] = np.nan

    rank_cols = [c for c in list(_RAW_RANK_COLS) + list(N_INPUT_COLS) + ["n_baseline"] if c in base.columns]
    base = rank_columns(base, rank_cols)
    base = attach_sample_size(base, small_sample_n=int(cfg["small_sample_n"]))

    base["score_C"] = mean_available(base, [f"{c}_pct" for c in C_INPUT_COLS])
    base["score_V"] = mean_available(base, [f"{c}_pct" for c in V_INPUT_COLS])
    base["score_P"] = mean_available(base, [f"{c}_pct" for c in P_INPUT_COLS])
    n_pct_cols = [f"{c}_pct" for c in N_INPUT_COLS if f"{c}_pct" in base.columns]
    base["score_N"] = mean_available(base, n_pct_cols) if n_pct_cols else np.nan
    # n_available only where we actually have a scorable N
    base["n_available"] = base["score_N"].notna()
    # Social saturation: high baseline attention = more crowded
    if "n_baseline_pct" in base.columns:
        base["nsi_social"] = base["n_baseline_pct"].where(base["n_available"])
    else:
        base["nsi_social"] = np.nan
    base["nsi_price_ext"] = mean_available(base, [f"{c}_pct" for c in NSI_PRICE_COLS])
    base["nsi_vol_exh"] = mean_available(base, [f"{c}_pct" for c in NSI_VOL_COLS])
    base["nsi_mom_dec"] = mean_available(base, [f"{c}_pct" for c in NSI_MOM_COLS])
    base["model_version"] = version

    frames: list[pd.DataFrame] = []
    for model in models:
        part = base.copy()
        part["model"] = model
        if model == "A":
            # Price-only MREI; price + momentum NSI. C/V/volume-NSI unused.
            part["score_C"] = np.nan
            part["score_V"] = np.nan
            part["score_N"] = np.nan
            part["nsi_vol_exh"] = np.nan
            part["nsi_social"] = np.nan
            part["n_available"] = False
            part["MREI"] = compute_mrei(
                part["score_C"],
                part["score_V"],
                part["score_P"],
                None,
                n_available=False,
            )
            part["NSI"] = mean_available(
                part, ["nsi_price_ext", "nsi_mom_dec"]
            )
        elif model == "B":
            # Model B: C+V+P partial MREI; price/volume NSI. N unused.
            part["score_N"] = np.nan
            part["nsi_social"] = np.nan
            part["n_available"] = False
            part["MREI"] = compute_mrei(
                part["score_C"],
                part["score_V"],
                part["score_P"],
                None,
                n_available=False,
            )
            part["NSI"] = mean_available(
                part, ["nsi_price_ext", "nsi_vol_exh", "nsi_mom_dec"]
            )
        else:
            # Model C: C+V+P+N when n_available; else renormalize without N.
            part["MREI"] = compute_mrei(
                part["score_C"],
                part["score_V"],
                part["score_P"],
                part["score_N"],
                n_available=part["n_available"],
            )
            nsi_cols_c = ["nsi_price_ext", "nsi_vol_exh", "nsi_mom_dec"]
            # Include social saturation only where N present
            part["NSI"] = mean_available(
                part,
                nsi_cols_c
                + (["nsi_social"] if part["nsi_social"].notna().any() else []),
            )
            # For rows without N, NSI should not use null social (mean_available skips)
        part["Rotation_Gap"] = rotation_gap(part["MREI"], part["NSI"])
        frames.append(part)

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.date
    out["n_available"] = out["n_available"].fillna(False).astype(bool)
    out["small_sample"] = out["small_sample"].astype(bool)
    result = out[list(FEATURE_COLUMNS)].sort_values(
        ["timestamp", "band", "model", "asset_id"], kind="mergesort"
    ).reset_index(drop=True)
    result.attrs["rs_benchmark"] = rs_benchmark
    result.attrs["small_sample_n"] = int(cfg["small_sample_n"])
    result.attrs["model_version"] = version
    return result


def features_summary(features: pd.DataFrame) -> dict[str, Any]:
    """Counts / ranges / nulls by band × model (research report helper)."""
    if features is None or features.empty:
        return {"total_rows": 0, "groups": {}}
    score_cols = [
        "score_C",
        "score_V",
        "score_N",
        "score_P",
        "MREI",
        "nsi_social",
        "nsi_price_ext",
        "nsi_vol_exh",
        "nsi_mom_dec",
        "NSI",
        "Rotation_Gap",
    ]
    groups: dict[str, Any] = {}
    for (band, model), g in features.groupby(["band", "model"], sort=True):
        stats: dict[str, Any] = {
            "rows": int(len(g)),
            "assets": int(g["asset_id"].nunique()),
            "days": int(pd.to_datetime(g["timestamp"]).nunique()),
            "n_available_true": int(g["n_available"].fillna(False).astype(bool).sum()),
            "small_sample_true": int(g["small_sample"].fillna(False).astype(bool).sum()),
            "n_in_band_min": int(g["n_in_band"].min()) if g["n_in_band"].notna().any() else 0,
            "n_in_band_max": int(g["n_in_band"].max()) if g["n_in_band"].notna().any() else 0,
        }
        for col in score_cols:
            s = pd.to_numeric(g[col], errors="coerce")
            stats[col] = {
                "min": None if s.dropna().empty else float(s.min()),
                "max": None if s.dropna().empty else float(s.max()),
                "mean": None if s.dropna().empty else float(s.mean()),
                "nulls": int(s.isna().sum()),
            }
        groups[f"{band}|{model}"] = stats
    ts = pd.to_datetime(features["timestamp"])
    return {
        "total_rows": int(len(features)),
        "date_range": [str(ts.min().date()), str(ts.max().date())],
        "rs_benchmark": features.attrs.get("rs_benchmark"),
        "model_version": features.attrs.get("model_version"),
        "groups": groups,
    }
