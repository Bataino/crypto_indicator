"""Universe membership: MC bands, ≥90d history, liquidity PASS draft.

DE Spec V0.1 §7 — point-in-time only; bands evaluated separately.
Does not compute MREI / features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

MEMBERSHIP_COLUMNS = (
    "timestamp",
    "asset_id",
    "band",
    "market_cap_usd",
    "history_days",
    "liquidity_pass",
    "amihud",
    "reason_excluded",
)

_VOLUME_LOOKBACK = 20
_DEFAULT_AMIHUD_LOOKBACK = 20
_DEFAULT_V_MIN_USD = 50_000.0


def resolve_v_min_usd(thresholds_config: dict[str, Any]) -> float:
    """Draft V_min: optional active override, else first ``v_min_usd`` candidate."""
    if thresholds_config.get("v_min_usd_active") is not None:
        return float(thresholds_config["v_min_usd_active"])
    candidates = thresholds_config.get("v_min_usd") or []
    if not candidates:
        return float(_DEFAULT_V_MIN_USD)
    return float(candidates[0])


# Back-compat alias used internally / older drafts
_draft_v_min_usd = resolve_v_min_usd


def _amihud_lookback(thresholds_config: dict[str, Any]) -> int:
    amihud = thresholds_config.get("amihud") or {}
    return int(amihud.get("lookback_days") or _DEFAULT_AMIHUD_LOOKBACK)


def _illiq_max_percentile(thresholds_config: dict[str, Any]) -> float | None:
    amihud = thresholds_config.get("amihud") or {}
    pct = amihud.get("illiq_max_percentile")
    if pct is None:
        return None
    return float(pct)


def _prepare_market(market_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sort and add history_days + 20d volume median; return daily Amihud series."""
    required = {"timestamp", "asset_id", "close", "volume_usd", "market_cap_usd"}
    missing = required - set(market_daily.columns)
    if missing:
        raise ValueError(f"market_daily missing columns: {sorted(missing)}")

    df = market_daily.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str)
    df = df.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    df = df.drop_duplicates(["asset_id", "timestamp"], keep="last").reset_index(
        drop=True
    )

    g = df.groupby("asset_id", sort=False)
    df["history_days"] = g.cumcount() + 1

    prev_close = g["close"].shift(1)
    ret = (df["close"] / prev_close) - 1.0
    abs_ret = ret.abs()

    vol = df["volume_usd"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_illiq = abs_ret / vol
    daily_illiq = daily_illiq.where((vol > 0) & abs_ret.notna())

    df["volume_usd_20d_median"] = g["volume_usd"].transform(
        lambda s: s.rolling(_VOLUME_LOOKBACK, min_periods=_VOLUME_LOOKBACK).median()
    )
    return df, daily_illiq


def _attach_amihud(
    df: pd.DataFrame,
    daily_illiq: pd.Series,
    lookback: int,
) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_daily_illiq"] = daily_illiq.to_numpy()
    tmp["amihud"] = tmp.groupby("asset_id", sort=False)["_daily_illiq"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).mean()
    )
    return tmp.drop(columns=["_daily_illiq"])


def build_universe_membership(
    market_daily: pd.DataFrame,
    universe_config: dict[str, Any],
    thresholds_config: dict[str, Any],
    *,
    assets: pd.DataFrame | None = None,
    eligibility_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute point-in-time membership for bands A_5_50 and B_10_100.

    Rules (DE Spec §7):
      - MC in band on that day (point-in-time; that day's market_cap_usd only)
      - ≥ min_history_days bars available as of that day (inclusive count)
      - Exclude missing volume/MC
      - Universe eligibility (tokenized stocks / stables / FX deny list)
      - Liquidity PASS draft:
          * median 20d volume_usd ≥ first V_min candidate in thresholds.yaml
          * if amihud.illiq_max_percentile is set, also require amihud_20d ≤
            that cross-sectional percentile within (timestamp, band);
            if null, Amihud is recorded but does not gate PASS

    Returns:
        DataFrame matching ``universe_membership`` columns. One row per
        (timestamp, asset_id, band) that clears hard gates (MC band + history +
        non-null MC/volume + eligibility). ``liquidity_pass`` /
        ``reason_excluded`` capture the draft liquidity outcome.

    Eligibility exclusions are applied before band membership (ineligible
    assets do not appear). Callers may read ``.attrs["eligibility_excluded"]``
    for ``asset_id → reason`` of assets dropped by the filter.
    """
    empty = pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))
    empty.attrs["eligibility_excluded"] = {}
    if market_daily is None or len(market_daily) == 0:
        return empty

    from cmram.universe.eligibility import filter_market_by_eligibility

    market_daily, excluded = filter_market_by_eligibility(
        market_daily, assets, eligibility_config
    )

    bands = universe_config.get("bands") or {}
    if not bands:
        raise ValueError("universe.yaml missing bands")

    min_history = int(universe_config.get("min_history_days", 90))
    v_min = resolve_v_min_usd(thresholds_config)
    amihud_lb = _amihud_lookback(thresholds_config)
    illiq_pct = _illiq_max_percentile(thresholds_config)
    amihud_gate_active = illiq_pct is not None

    if market_daily is None or len(market_daily) == 0:
        out = empty.copy()
        out.attrs["eligibility_excluded"] = excluded
        return out

    prepared, daily_illiq = _prepare_market(market_daily)
    prepared = _attach_amihud(prepared, daily_illiq, amihud_lb)

    # Hard exclusions: missing MC or volume
    prepared = prepared[
        prepared["market_cap_usd"].notna() & prepared["volume_usd"].notna()
    ].copy()

    frames: list[pd.DataFrame] = []
    preferred = ["A_5_50", "B_10_100"]
    band_keys = [k for k in preferred if k in bands] + [
        k for k in bands if k not in preferred
    ]
    for band_id in band_keys:
        band_cfg = bands[band_id] or {}
        mc_min = float(band_cfg["market_cap_min_usd"])
        mc_max = float(band_cfg["market_cap_max_usd"])
        band_df = prepared[
            (prepared["market_cap_usd"] >= mc_min)
            & (prepared["market_cap_usd"] <= mc_max)
            & (prepared["history_days"] >= min_history)
        ].copy()
        if band_df.empty:
            continue
        band_df["band"] = band_id
        frames.append(band_df)

    if not frames:
        out = pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))
        out.attrs["eligibility_excluded"] = excluded
        return out

    out = pd.concat(frames, ignore_index=True)

    vol_median = out["volume_usd_20d_median"]
    vol_ok = vol_median.notna() & (vol_median.astype(float) >= v_min)

    if amihud_gate_active:
        # Cross-sectional Amihud ceiling within (timestamp, band)
        def _pct_thresh(s: pd.Series) -> float:
            valid = s.dropna()
            if valid.empty:
                return float("nan")
            return float(np.nanpercentile(valid.to_numpy(), illiq_pct))

        thresh = out.groupby(["timestamp", "band"], sort=False)["amihud"].transform(
            _pct_thresh
        )
        amihud_ok = out["amihud"].notna() & (out["amihud"] <= thresh)
    else:
        amihud_ok = pd.Series(True, index=out.index)

    liquidity_pass = vol_ok & amihud_ok
    reasons: list[str | None] = []
    for i in range(len(out)):
        r: list[str] = []
        if not bool(vol_ok.iloc[i]):
            if pd.isna(vol_median.iloc[i]):
                r.append("insufficient_volume_history")
            else:
                r.append("low_volume")
        if amihud_gate_active and not bool(amihud_ok.iloc[i]):
            if pd.isna(out["amihud"].iloc[i]):
                r.append("insufficient_amihud_history")
            else:
                r.append("high_amihud")
        reasons.append(";".join(r) if r else None)

    result = pd.DataFrame(
        {
            "timestamp": [ts.date() for ts in pd.to_datetime(out["timestamp"])],
            "asset_id": out["asset_id"].astype(str).tolist(),
            "band": out["band"].astype(str).tolist(),
            "market_cap_usd": out["market_cap_usd"].astype(float).tolist(),
            "history_days": out["history_days"].astype(int).tolist(),
            "liquidity_pass": [bool(x) for x in liquidity_pass.tolist()],
            "amihud": [
                float(x) if pd.notna(x) else float("nan") for x in out["amihud"].tolist()
            ],
            "reason_excluded": reasons,
        }
    )
    result = result.sort_values(
        ["timestamp", "band", "asset_id"], kind="mergesort"
    ).reset_index(drop=True)
    # Force object dtype so PASS rows stay Python None (not pandas StringDtype NA).
    cleaned: list[str | None] = []
    for x in result["reason_excluded"].tolist():
        if x is None or x is pd.NA:
            cleaned.append(None)
        elif isinstance(x, float) and np.isnan(x):
            cleaned.append(None)
        elif isinstance(x, str):
            cleaned.append(x)
        else:
            cleaned.append(None if pd.isna(x) else str(x))
    result["reason_excluded"] = pd.Series(cleaned, dtype=object)
    out_df = result[list(MEMBERSHIP_COLUMNS)]
    out_df.attrs["eligibility_excluded"] = excluded
    return out_df


def membership_summary(membership: pd.DataFrame) -> dict[str, Any]:
    """Aggregate row counts and liquidity_pass rates by band."""
    if membership is None or membership.empty:
        return {"total_rows": 0, "bands": {}}

    bands: dict[str, Any] = {}
    for band, g in membership.groupby("band", sort=False):
        n = len(g)
        n_pass = int(g["liquidity_pass"].sum())
        bands[str(band)] = {
            "rows": n,
            "liquidity_pass": n_pass,
            "liquidity_pass_pct": round(100.0 * n_pass / n, 2) if n else 0.0,
            "assets": int(g["asset_id"].nunique()),
            "days": int(g["timestamp"].nunique()),
        }
    return {"total_rows": int(len(membership)), "bands": bands}


def load_market_daily(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load all ``market_daily`` rows from an open DuckDB connection."""
    return conn.execute(
        """
        SELECT timestamp, asset_id, open, high, low, close,
               volume_usd, market_cap_usd, source, ingested_at
        FROM market_daily
        ORDER BY asset_id, timestamp
        """
    ).df()


def write_universe_membership(
    conn: duckdb.DuckDBPyConnection,
    membership: pd.DataFrame,
    *,
    replace: bool = True,
) -> int:
    """Write membership rows into ``universe_membership``.

    Args:
        conn: Open DuckDB connection.
        membership: Output of ``build_universe_membership``.
        replace: If True (default), delete existing rows then insert (full rebuild).

    Returns:
        Number of rows written.
    """
    if replace:
        conn.execute("DELETE FROM universe_membership")

    if membership is None or len(membership) == 0:
        return 0

    df = membership[list(MEMBERSHIP_COLUMNS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    conn.register("_universe_df", df)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO universe_membership
            SELECT
                timestamp, asset_id, band, market_cap_usd,
                history_days, liquidity_pass, amihud, reason_excluded
            FROM _universe_df
            """
        )
    finally:
        conn.unregister("_universe_df")
    return len(df)


def load_assets(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load ``assets`` metadata used by the eligibility filter."""
    try:
        return conn.execute(
            """
            SELECT asset_id, symbol, name, category
            FROM assets
            ORDER BY asset_id
            """
        ).df()
    except duckdb.Error:
        return pd.DataFrame(columns=["asset_id", "symbol", "name", "category"])


def flag_assets_eligibility(
    conn: duckdb.DuckDBPyConnection,
    assets: pd.DataFrame,
    eligibility_config: dict[str, Any] | None,
) -> dict[str, str]:
    """Set ``universe_eligible`` / ``universe_exclude_reason`` on assets rows.

    Returns map of ineligible asset_id → reason.
    """
    from cmram.universe.eligibility import evaluate_assets_frame

    if assets is None or len(assets) == 0:
        return {}
    ev = evaluate_assets_frame(assets, eligibility_config)
    excluded = {
        str(r.asset_id): str(r.reason_excluded)
        for r in ev.itertuples(index=False)
        if not bool(r.eligible)
    }
    # Ensure columns exist (migrate_schema should have added them)
    cols = {r[0] for r in conn.execute("DESCRIBE assets").fetchall()}
    if "universe_eligible" not in cols or "universe_exclude_reason" not in cols:
        from cmram.db.schema import migrate_assets_eligibility

        migrate_assets_eligibility(conn)

    conn.register("_elig_df", ev[["asset_id", "eligible", "reason_excluded"]])
    try:
        conn.execute(
            """
            UPDATE assets AS a
            SET
                universe_eligible = e.eligible,
                universe_exclude_reason = e.reason_excluded
            FROM _elig_df AS e
            WHERE a.asset_id = e.asset_id
            """
        )
    finally:
        conn.unregister("_elig_df")
    return excluded


def build_and_write_universe_membership(
    duckdb_path: str | Path,
    universe_config: dict[str, Any],
    thresholds_config: dict[str, Any],
    *,
    eligibility_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load market_daily from DuckDB, compute membership, write back.

    Also flags ``assets.universe_eligible`` / ``universe_exclude_reason``.
    """
    from cmram.db.schema import init_db
    from cmram.config import load_eligibility_config

    path = Path(duckdb_path)
    elig = (
        eligibility_config
        if eligibility_config is not None
        else load_eligibility_config()
    )
    conn = init_db(path)
    try:
        market = load_market_daily(conn)
        assets = load_assets(conn)
        excluded_flagged = flag_assets_eligibility(conn, assets, elig)
        membership = build_universe_membership(
            market,
            universe_config,
            thresholds_config,
            assets=assets,
            eligibility_config=elig,
        )
        # Prefer attrs from build; fall back to flagged map
        if not getattr(membership, "attrs", {}).get("eligibility_excluded"):
            membership.attrs["eligibility_excluded"] = excluded_flagged
        write_universe_membership(conn, membership, replace=True)
        return membership
    finally:
        conn.close()
