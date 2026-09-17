"""Default lookbacks / knobs. YAML config overrides these.

Windows are inclusive of t and strictly backward-looking (pandas rolling).
"""

from __future__ import annotations

from typing import Any

DEFAULT_LOOKBACKS: dict[str, int] = {
    "high_days": 20,
    "range_short_days": 10,
    "range_long_days": 40,
    "vol_short_days": 7,
    "vol_long_days": 30,
    "sell_short_days": 5,
    "sell_long_days": 20,
    "stabilize_short_days": 5,
    "stabilize_long_days": 20,
    "rvol_short_days": 3,
    "rvol_long_days": 30,
    "vol_growth_short_days": 7,
    "vol_growth_long_days": 28,
    "velocity_days": 3,
    "logvol_accel_days": 7,
    "ret_short_days": 3,
    "ret_medium_days": 7,
    "rs_short_days": 7,
    "rs_long_days": 14,
    "higher_lows_short_days": 10,
    "higher_lows_long_days": 20,
    "breakout_days": 20,
    "sma_days": 20,
    "extension_days": 14,
    "mom_dec_days": 7,
    # Narrative N
    "n_vel_short_days": 7,
    "n_vel_long_days": 28,
    "n_accel_days": 7,
}

DEFAULT_SMALL_SAMPLE_N = 10
DEFAULT_VOLUME_FLOOR_USD = 1.0
DEFAULT_RVOL_CLIP = 10.0
DEFAULT_BTC_ASSET_IDS = ("bitcoin",)
DEFAULT_MODEL_VERSION = "features_v0.1"
DEFAULT_MODELS = ("A", "B", "C", "E")
DEFAULT_NARRATIVE_MODE = "quiet_rising"
DEFAULT_N_CROWD_EXP = 3.0
DEFAULT_N_CROWD_SCALE = 10.0
DEFAULT_N_CROWD_HARD_MAX = 80.0


def resolve_feature_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge caller/YAML config onto defaults. Does not invent social.

    If ``config`` is None, loads ``config/features.yaml`` when present.
    """
    yaml_cfg: dict[str, Any] = {}
    if config is None:
        try:
            from cmram.config import load_features_config

            yaml_cfg = load_features_config()
        except FileNotFoundError:
            yaml_cfg = {}
    else:
        yaml_cfg = dict(config)

    lookbacks = dict(DEFAULT_LOOKBACKS)
    lookbacks.update(yaml_cfg.get("lookbacks") or {})
    # Surface narrative mode knobs inside lookbacks for narrative_inputs
    _mode = str(yaml_cfg.get("narrative_mode") or DEFAULT_NARRATIVE_MODE).strip().lower()
    lookbacks.setdefault("narrative_mode", _mode)
    if yaml_cfg.get("n_crowd_exp") is not None:
        lookbacks["n_crowd_exp"] = float(yaml_cfg["n_crowd_exp"])
    else:
        lookbacks.setdefault("n_crowd_exp", DEFAULT_N_CROWD_EXP)
    if yaml_cfg.get("n_crowd_scale") is not None:
        lookbacks["n_crowd_scale"] = float(yaml_cfg["n_crowd_scale"])
    else:
        lookbacks.setdefault("n_crowd_scale", DEFAULT_N_CROWD_SCALE)
    if yaml_cfg.get("n_crowd_hard_max") is not None:
        lookbacks["n_crowd_hard_max"] = float(yaml_cfg["n_crowd_hard_max"])
    else:
        lookbacks.setdefault("n_crowd_hard_max", DEFAULT_N_CROWD_HARD_MAX)

    btc_ids = yaml_cfg.get("btc_asset_ids") or list(DEFAULT_BTC_ASSET_IDS)
    models = yaml_cfg.get("models") or list(DEFAULT_MODELS)
    return {
        "model_version": str(yaml_cfg.get("model_version") or DEFAULT_MODEL_VERSION),
        "models": [str(m) for m in models],
        "small_sample_n": int(yaml_cfg.get("small_sample_n") or DEFAULT_SMALL_SAMPLE_N),
        "volume_floor_usd": float(
            yaml_cfg.get("volume_floor_usd")
            if yaml_cfg.get("volume_floor_usd") is not None
            else DEFAULT_VOLUME_FLOOR_USD
        ),
        "rvol_clip": float(
            yaml_cfg.get("rvol_clip")
            if yaml_cfg.get("rvol_clip") is not None
            else DEFAULT_RVOL_CLIP
        ),
        "btc_asset_ids": tuple(str(x) for x in btc_ids),
        "lookbacks": lookbacks,
        "attention_floor": float(
            yaml_cfg.get("attention_floor")
            if yaml_cfg.get("attention_floor") is not None
            else 0.1
        ),
        "social_combine": str(yaml_cfg.get("social_combine") or "mean"),
        "narrative_mode": str(
            yaml_cfg.get("narrative_mode") or DEFAULT_NARRATIVE_MODE
        ).strip().lower(),
        "n_crowd_exp": float(
            yaml_cfg.get("n_crowd_exp")
            if yaml_cfg.get("n_crowd_exp") is not None
            else DEFAULT_N_CROWD_EXP
        ),
        "n_crowd_scale": float(
            yaml_cfg.get("n_crowd_scale")
            if yaml_cfg.get("n_crowd_scale") is not None
            else DEFAULT_N_CROWD_SCALE
        ),
        "n_crowd_hard_max": float(
            yaml_cfg.get("n_crowd_hard_max")
            if yaml_cfg.get("n_crowd_hard_max") is not None
            else DEFAULT_N_CROWD_HARD_MAX
        ),
    }
