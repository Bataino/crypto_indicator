"""DuckDB schema init for CMRAM (DE Spec V0.1 §6)."""

from __future__ import annotations

from pathlib import Path

import duckdb

SCHEMA_SQL_PATH = Path(__file__).with_name("schema.sql")

TABLES = (
    "assets",
    "market_daily",
    "universe_membership",
    "social_daily",
    "features_daily",
    "signals",
    "backtest_results",
    "runs",
    "ai_samples_daily",
)

# Columns required on features_daily after Phase 1 Models A/B (compound key).
FEATURES_DAILY_REQUIRED_COLS = frozenset(
    {
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
    }
)


def schema_sql() -> str:
    """Return the DDL from schema.sql."""
    return SCHEMA_SQL_PATH.read_text(encoding="utf-8")


def _table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = conn.execute(f"DESCRIBE {table}").fetchall()
    return {r[0] for r in rows}


def migrate_assets_eligibility(conn: duckdb.DuckDBPyConnection) -> bool:
    """Add universe_eligible / universe_exclude_reason on assets if missing."""
    names = set(list_tables(conn))
    if "assets" not in names:
        return False
    cols = _table_columns(conn, "assets")
    changed = False
    if "universe_eligible" not in cols:
        conn.execute("ALTER TABLE assets ADD COLUMN universe_eligible BOOLEAN")
        changed = True
    if "universe_exclude_reason" not in cols:
        conn.execute("ALTER TABLE assets ADD COLUMN universe_exclude_reason TEXT")
        changed = True
    return changed


def migrate_schema(conn: duckdb.DuckDBPyConnection) -> bool:
    """Upgrade existing DBs so features_daily can store Model A and B rows.

    Older scaffolds used PK-less ``(timestamp, asset_id, band)`` rows and could
    not hold both models. If required columns are missing, ``features_daily`` is
    dropped and recreated from ``schema.sql`` (CREATE IF NOT EXISTS). Other
    tables are left intact. Stale feature rows are not converted — rebuild via
    ``run_features``.

    Also ensures assets eligibility flag columns exist.

    Returns:
        True if a rebuild/migration ran, False if schema was already current.
    """
    changed = False
    names = set(list_tables(conn))
    if "features_daily" not in names:
        conn.execute(schema_sql())
        changed = True
    else:
        cols = _table_columns(conn, "features_daily")
        if not (FEATURES_DAILY_REQUIRED_COLS <= cols):
            conn.execute("DROP TABLE features_daily")
            conn.execute(schema_sql())
            changed = True
    if migrate_assets_eligibility(conn):
        changed = True
    return changed


def init_db(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Create empty CMRAM tables in a DuckDB database.

    Args:
        path: File path for persistent DB, or None for in-memory.

    Returns:
        Open DuckDB connection with all Phase 1 tables created (and migrated).
    """
    conn = duckdb.connect(str(path) if path is not None else ":memory:")
    conn.execute(schema_sql())
    migrate_schema(conn)
    return conn


def list_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return user table names present in the connection."""
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]
