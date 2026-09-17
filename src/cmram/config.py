"""Load YAML configs and resolve data paths for CMRAM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo root: .../cmram (parent of src/)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DUCKDB_NAME = "cmram.duckdb"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data)}")
    return data


def load_universe_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config/universe.yaml (bands A/B, min_history_days, etc.)."""
    return load_yaml(path or CONFIG_DIR / "universe.yaml")


def load_thresholds_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config/thresholds.yaml (τ grids, V_min — calibration TBD)."""
    return load_yaml(path or CONFIG_DIR / "thresholds.yaml")


def load_features_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config/features.yaml (lookbacks, model_version, small-sample flag)."""
    return load_yaml(path or CONFIG_DIR / "features.yaml")



def load_social_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config/social.yaml (Trends/Reddit/X/wiki/Santiment ingest knobs)."""
    p = Path(path) if path else CONFIG_DIR / "social.yaml"
    if not p.is_file():
        return {"enabled": False, "sources": {}}
    return load_yaml(p)

def load_eligibility_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config/eligibility.yaml (deny tokenized stocks / stables / FX)."""
    p = Path(path) if path else CONFIG_DIR / "eligibility.yaml"
    if not p.is_file():
        return {"enabled": False}
    return load_yaml(p)


def get_duckdb_path() -> Path:
    """DuckDB path from ``CMRAM_DUCKDB_PATH`` or default ``data/cmram.duckdb``."""
    override = os.environ.get("CMRAM_DUCKDB_PATH")
    if override:
        return Path(override).expanduser()
    return DATA_DIR / DEFAULT_DUCKDB_NAME


def get_raw_dir() -> Path:
    """Raw ingest dump directory (``CMRAM_RAW_DIR`` or ``data/raw``)."""
    override = os.environ.get("CMRAM_RAW_DIR")
    if override:
        return Path(override).expanduser()
    return DATA_DIR / "raw"


def get_features_dir() -> Path:
    """Feature output directory (``CMRAM_FEATURES_DIR`` or ``data/features``)."""
    override = os.environ.get("CMRAM_FEATURES_DIR")
    if override:
        return Path(override).expanduser()
    return DATA_DIR / "features"


def get_signals_dir() -> Path:
    """Signal output directory (``CMRAM_SIGNALS_DIR`` or ``data/signals``)."""
    override = os.environ.get("CMRAM_SIGNALS_DIR")
    if override:
        return Path(override).expanduser()
    return DATA_DIR / "signals"


def get_backtests_dir() -> Path:
    """Backtest output directory (``CMRAM_BACKTESTS_DIR`` or ``data/backtests``)."""
    override = os.environ.get("CMRAM_BACKTESTS_DIR")
    if override:
        return Path(override).expanduser()
    return DATA_DIR / "backtests"
