"""Scaffold smoke tests: imports, config load, DuckDB schema init."""

from __future__ import annotations

from pathlib import Path

import yaml

from cmram import __version__
from cmram.config import load_thresholds_config, load_universe_config
from cmram.db import TABLES, init_db, list_tables


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_importable():
    assert __version__ == "0.1.0"


def test_submodules_importable():
    import cmram.ingest  # noqa: F401
    import cmram.universe  # noqa: F401
    import cmram.features  # noqa: F401
    import cmram.signals  # noqa: F401
    import cmram.backtest  # noqa: F401
    import cmram.report  # noqa: F401
    import cmram.db  # noqa: F401


def test_universe_config_loads():
    cfg = load_universe_config()
    assert cfg["min_history_days"] == 90
    assert "A_5_50" in cfg["bands"]
    assert "B_10_100" in cfg["bands"]
    assert cfg["bands"]["A_5_50"]["market_cap_min_usd"] == 5_000_000
    assert cfg["bands"]["A_5_50"]["market_cap_max_usd"] == 50_000_000
    assert cfg["bands"]["B_10_100"]["market_cap_min_usd"] == 10_000_000
    assert cfg["bands"]["B_10_100"]["market_cap_max_usd"] == 100_000_000
    assert cfg["entry_convention"] == "next_day_open"
    assert cfg["social_enabled"] is False


def test_thresholds_config_loads():
    cfg = load_thresholds_config()
    assert cfg["calibration_status"] == "TBD"
    assert cfg["tau_m"] == [50, 60, 70]
    assert cfg["tau_g"] == [20, 30, 40, 50]
    assert cfg["v_min_usd"] == [50_000, 100_000, 250_000]
    assert cfg["horizons_days"] == [1, 3, 7, 14, 21, 30]


def test_yaml_files_exist_and_parse():
    for name in ("universe.yaml", "thresholds.yaml"):
        path = REPO_ROOT / "config" / name
        assert path.is_file(), path
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)


def test_schema_init_creates_tables(tmp_path):
    db_path = tmp_path / "cmram_test.duckdb"
    conn = init_db(db_path)
    try:
        names = list_tables(conn)
        for table in TABLES:
            assert table in names, f"missing table: {table}"
    finally:
        conn.close()


def test_schema_sql_file_exists():
    sql_path = REPO_ROOT / "src" / "cmram" / "db" / "schema.sql"
    assert sql_path.is_file()
    text = sql_path.read_text(encoding="utf-8")
    for table in TABLES:
        assert table in text


def test_specs_present():
    specs = REPO_ROOT / "specs"
    assert (specs / "CMRAM_V0.1_Research_Spec.md").is_file()
    assert (specs / "CMRAM_V0.1_Data_Engineering_Spec.md").is_file()


def test_features_daily_schema_supports_models_ab():
    """Compound key (timestamp, asset_id, band, model) for A/B rows."""
    sql_path = REPO_ROOT / "src" / "cmram" / "db" / "schema.sql"
    text = sql_path.read_text(encoding="utf-8")
    assert "PRIMARY KEY (timestamp, asset_id, band, model)" in text
    conn = init_db()
    try:
        cols = {r[0] for r in conn.execute("DESCRIBE features_daily").fetchall()}
        assert "model" in cols
        assert "small_sample" in cols
        assert "n_in_band" in cols
    finally:
        conn.close()
