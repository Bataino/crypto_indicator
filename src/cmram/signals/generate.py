"""Emit signals: liquidity PASS AND MREI > τ_m AND Gap > τ_g.

Research only — keep all τ-tagged nested sets; no claimed alpha.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = (
    "signal_id",
    "timestamp",
    "asset_id",
    "band",
    "entry_price",
    "market_cap_usd",
    "volume_usd",
    "MREI",
    "NSI",
    "Rotation_Gap",
    "tau_m",
    "tau_g",
    "model",
    "liquidity_pass",
)

ENTRY_CONVENTION_NEXT_DAY_OPEN = "next_day_open"
ENTRY_PROXY_NEXT_CLOSE = "next_day_close_chart_proxy"


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SIGNAL_COLUMNS))


def _as_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize().dt.date


def _signal_id(
    timestamp,
    asset_id: str,
    band: str,
    model: str,
    tau_m: float,
    tau_g: float,
) -> str:
    """Deterministic UUID5 from research key fields (reproducible re-runs)."""
    key = f"{timestamp}|{asset_id}|{band}|{model}|{tau_m}|{tau_g}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cmram:signal:{key}"))


def detect_chart_close_only(market_daily: pd.DataFrame) -> bool:
    """True when O=H=L=C on every bar (CoinGecko chart-close ingest mode)."""
    if market_daily is None or len(market_daily) == 0:
        return False
    o = market_daily["open"].to_numpy(dtype=float)
    h = market_daily["high"].to_numpy(dtype=float)
    low = market_daily["low"].to_numpy(dtype=float)
    c = market_daily["close"].to_numpy(dtype=float)
    return bool(np.all(np.isclose(o, h) & np.isclose(h, low) & np.isclose(low, c)))


def build_entry_lookup(
    market_daily: pd.DataFrame,
    *,
    entry_convention: str = ENTRY_CONVENTION_NEXT_DAY_OPEN,
) -> tuple[pd.DataFrame, str, bool]:
    """Map each (asset_id, signal_date) → next-bar entry price.

    Convention: next available bar after signal ``timestamp``.
    If OHLC is chart-close-only (O=H=L=C), use that bar's close as the open
    proxy and report ``ENTRY_PROXY_NEXT_CLOSE``.

    Returns:
        (lookup_df with columns asset_id, signal_date, entry_date, entry_price,
         entry_is_proxy), entry_mode_label, chart_close_only flag
    """
    if market_daily is None or len(market_daily) == 0:
        empty = pd.DataFrame(
            columns=[
                "asset_id",
                "signal_date",
                "entry_date",
                "entry_price",
                "entry_is_proxy",
            ]
        )
        return empty, entry_convention, False

    if entry_convention != ENTRY_CONVENTION_NEXT_DAY_OPEN:
        raise ValueError(f"Unsupported entry_convention: {entry_convention}")

    m = market_daily.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.normalize()
    m["asset_id"] = m["asset_id"].astype(str)
    m = m.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    chart_only = detect_chart_close_only(m)
    # Per-bar proxy: if that bar itself is O=H=L=C, treat open as close proxy
    bar_proxy = (
        np.isclose(m["open"], m["high"])
        & np.isclose(m["high"], m["low"])
        & np.isclose(m["low"], m["close"])
    )
    m["_entry_price"] = np.where(bar_proxy, m["close"], m["open"]).astype(float)
    m["_entry_is_proxy"] = bar_proxy

    # Next bar within each asset
    m["entry_date"] = m.groupby("asset_id", sort=False)["timestamp"].shift(-1)
    m["entry_price"] = m.groupby("asset_id", sort=False)["_entry_price"].shift(-1)
    m["entry_is_proxy"] = m.groupby("asset_id", sort=False)["_entry_is_proxy"].shift(-1)

    lookup = m.loc[
        m["entry_date"].notna() & m["entry_price"].notna(),
        ["asset_id", "timestamp", "entry_date", "entry_price", "entry_is_proxy"],
    ].rename(columns={"timestamp": "signal_date"})
    lookup["signal_date"] = lookup["signal_date"].dt.date
    lookup["entry_date"] = pd.to_datetime(lookup["entry_date"]).dt.date
    lookup["entry_is_proxy"] = lookup["entry_is_proxy"].astype(bool)

    mode = ENTRY_PROXY_NEXT_CLOSE if chart_only else entry_convention
    return lookup, mode, chart_only


def generate_signals(
    features_daily: pd.DataFrame,
    universe_membership: pd.DataFrame,
    thresholds_config: dict[str, Any],
    *,
    market_daily: pd.DataFrame | None = None,
    entry_convention: str = ENTRY_CONVENTION_NEXT_DAY_OPEN,
) -> pd.DataFrame:
    """Generate signal rows for each day × band × model × (τ_m, τ_g).

    Rule: ``liquidity_pass AND MREI > τ_m AND Rotation_Gap > τ_g``.

    Requires a join to ``universe_membership`` so liquidity context is available.
    Rows without membership (no liquidity context) are skipped.

    Entry price: next-day open (next available bar). If OHLC is chart-close-only
    (O=H=L=C), use next day's close as proxy — documented via DataFrame.attrs.

    Returns:
        DataFrame matching ``signals`` columns. Nested τ combinations kept
        and tagged. attrs: ``entry_convention``, ``chart_close_only``,
        ``entry_mode``, ``n_skipped_no_entry``.
    """
    out = _empty_signals()
    out.attrs["entry_convention"] = entry_convention
    out.attrs["chart_close_only"] = False
    out.attrs["entry_mode"] = entry_convention
    out.attrs["n_skipped_no_entry"] = 0

    if features_daily is None or len(features_daily) == 0:
        return out
    if universe_membership is None or len(universe_membership) == 0:
        return out

    tau_m_grid = list(thresholds_config.get("tau_m") or [])
    tau_g_grid = list(thresholds_config.get("tau_g") or [])
    models = list(thresholds_config.get("models") or ["A", "B"])
    if not tau_m_grid or not tau_g_grid:
        return out

    feat = features_daily.copy()
    feat["timestamp"] = _as_date_series(feat["timestamp"])
    feat["asset_id"] = feat["asset_id"].astype(str)
    feat["band"] = feat["band"].astype(str)
    feat["model"] = feat["model"].astype(str)
    feat = feat[feat["model"].isin([str(m) for m in models])].copy()
    if len(feat) == 0:
        return out

    memb = universe_membership.copy()
    memb["timestamp"] = _as_date_series(memb["timestamp"])
    memb["asset_id"] = memb["asset_id"].astype(str)
    memb["band"] = memb["band"].astype(str)
    memb_cols = ["timestamp", "asset_id", "band", "liquidity_pass", "market_cap_usd"]
    memb = memb[memb_cols].drop_duplicates(
        ["timestamp", "asset_id", "band"], keep="last"
    )

    # Join: liquidity context required
    base = feat.merge(memb, on=["timestamp", "asset_id", "band"], how="inner")
    if len(base) == 0:
        return out

    # Volume on signal day from market (optional but schema requires volume_usd)
    if market_daily is not None and len(market_daily) > 0:
        md = market_daily.copy()
        md["timestamp"] = _as_date_series(md["timestamp"])
        md["asset_id"] = md["asset_id"].astype(str)
        vol = md[["timestamp", "asset_id", "volume_usd"]].drop_duplicates(
            ["timestamp", "asset_id"], keep="last"
        )
        base = base.merge(vol, on=["timestamp", "asset_id"], how="left")
    else:
        base["volume_usd"] = np.nan

    # If membership MC missing, leave NaN (research honesty)
    if "market_cap_usd" not in base.columns:
        base["market_cap_usd"] = np.nan

    lookup, entry_mode, chart_only = build_entry_lookup(
        market_daily if market_daily is not None else pd.DataFrame(),
        entry_convention=entry_convention,
    )
    out.attrs["chart_close_only"] = chart_only
    out.attrs["entry_mode"] = entry_mode

    if len(lookup) == 0:
        # Cannot form entries without market bars
        return out

    base = base.merge(
        lookup,
        left_on=["asset_id", "timestamp"],
        right_on=["asset_id", "signal_date"],
        how="left",
    )
    n_no_entry = int(base["entry_price"].isna().sum())
    base = base.loc[base["entry_price"].notna()].copy()
    out.attrs["n_skipped_no_entry"] = n_no_entry

    # Liquidity PASS only
    base = base.loc[base["liquidity_pass"].fillna(False).astype(bool)].copy()
    if len(base) == 0:
        return out

    rows: list[dict[str, Any]] = []
    for tau_m in tau_m_grid:
        tm = float(tau_m)
        for tau_g in tau_g_grid:
            tg = float(tau_g)
            mask = (base["MREI"] > tm) & (base["Rotation_Gap"] > tg)
            hit = base.loc[mask]
            for r in hit.itertuples(index=False):
                ts = r.timestamp
                rows.append(
                    {
                        "signal_id": _signal_id(
                            ts, r.asset_id, r.band, r.model, tm, tg
                        ),
                        "timestamp": ts,
                        "asset_id": r.asset_id,
                        "band": r.band,
                        "entry_price": float(r.entry_price),
                        "market_cap_usd": (
                            float(r.market_cap_usd)
                            if pd.notna(r.market_cap_usd)
                            else np.nan
                        ),
                        "volume_usd": (
                            float(r.volume_usd) if pd.notna(r.volume_usd) else np.nan
                        ),
                        "MREI": float(r.MREI) if pd.notna(r.MREI) else np.nan,
                        "NSI": float(r.NSI) if pd.notna(r.NSI) else np.nan,
                        "Rotation_Gap": (
                            float(r.Rotation_Gap)
                            if pd.notna(r.Rotation_Gap)
                            else np.nan
                        ),
                        "tau_m": tm,
                        "tau_g": tg,
                        "model": str(r.model),
                        "liquidity_pass": True,
                    }
                )

    if not rows:
        return out

    result = pd.DataFrame(rows)
    result = result[list(SIGNAL_COLUMNS)]
    result = result.sort_values(
        ["timestamp", "band", "model", "tau_m", "tau_g", "asset_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    result.attrs.update(out.attrs)
    # Stable hash of config grids for audit (not crypto-secure; research tag)
    grid_tag = hashlib.sha1(
        f"{tau_m_grid}|{tau_g_grid}|{models}|{entry_mode}".encode()
    ).hexdigest()[:12]
    result.attrs["grid_tag"] = grid_tag
    return result
