"""Market data ingest — CoinGecko primary (DE Spec §5.1).

Pulls real daily OHLCV + market cap into ``data/raw/`` and DuckDB tables
``assets`` + ``market_daily``. Research data collection only — no trading.

Survivorship / universe limits
------------------------------
Candidate discovery via live ``/coins/markets`` pages only sees coins still
listed on CoinGecko at scrape time. Assets that died, delisted, or left the
MC bands are missing → survivorship bias if used as a historical universe
without archived snapshots. Band membership here is a **recent snapshot**
filter (MC union of all bands in universe.yaml, currently spanning
Band C $1–100M through Band A/B), not point-in-time history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from cmram.config import (
    get_duckdb_path,
    get_raw_dir,
    load_universe_config,
)
from cmram.db.schema import init_db
from cmram.ingest.coingecko import CoinGeckoClient, CoinGeckoError, RateLimitError

logger = logging.getLogger(__name__)

SOURCE = "coingecko"
DEFAULT_HISTORY_DAYS = 180
MIN_BARS_DEFAULT = 90

# Fallback MC union spanning narrowest configured floor to widest ceiling.
# Prefer _band_union_bounds() which reads all bands from universe.yaml.
BAND_UNION_MC_MIN = 1_000_000
BAND_UNION_MC_MAX = 100_000_000


@dataclass
class IngestResult:
    """Summary of an ingest run."""

    assets_upserted: int = 0
    market_rows_upserted: int = 0
    skipped_insufficient: list[str] | None = None
    candidates: list[str] | None = None
    dry_run: bool = False
    raw_dir: str | None = None
    duckdb_path: str | None = None

    def __post_init__(self) -> None:
        if self.skipped_insufficient is None:
            self.skipped_insufficient = []
        if self.candidates is None:
            self.candidates = []


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ms_to_date(ts_ms: float | int) -> date:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).date()


def _band_union_bounds(universe: dict[str, Any] | None = None) -> tuple[float, float]:
    """MC filter spanning the union of *all* bands in universe.yaml.

    Uses min(market_cap_min_usd) and max(market_cap_max_usd) across every
    configured band (A/B/C/…), not a hard-coded A/B pair.
    """
    cfg = universe or load_universe_config()
    bands = cfg.get("bands") or {}
    mins: list[float] = []
    maxs: list[float] = []
    for _key, b in bands.items():
        if not isinstance(b, dict):
            continue
        if "market_cap_min_usd" in b:
            mins.append(float(b["market_cap_min_usd"]))
        if "market_cap_max_usd" in b:
            maxs.append(float(b["market_cap_max_usd"]))
    lo = min(mins) if mins else float(BAND_UNION_MC_MIN)
    hi = max(maxs) if maxs else float(BAND_UNION_MC_MAX)
    return lo, hi


def discover_band_candidates(
    client: CoinGeckoClient,
    *,
    limit: int | None = None,
    full: bool = False,
    max_pages: int | None = None,
    per_page: int = 250,
    mc_min: float | None = None,
    mc_max: float | None = None,
    universe: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Page ``/coins/markets`` (volume desc) and keep coins in MC band union.

    Default (not ``full``): stop once ``limit`` matches are found or a few
    pages are scanned — enough for ``--limit N`` smoke runs.
    ``full=True``: paginate until empty or ``max_pages`` (default 20).
    """
    lo, hi = _band_union_bounds(universe)
    if mc_min is not None:
        lo = mc_min
    if mc_max is not None:
        hi = mc_max

    if max_pages is None:
        max_pages = 20 if full else max(1, min(4, ((limit or 10) // 50) + 2))

    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        rows = client.get_markets(page=page, per_page=per_page, order="volume_desc")
        if not rows:
            break
        for row in rows:
            cid = row.get("id")
            mc = row.get("market_cap")
            if not cid or mc is None:
                continue
            try:
                mc_f = float(mc)
            except (TypeError, ValueError):
                continue
            if mc_f < lo or mc_f > hi:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            found.append(row)
            if limit is not None and len(found) >= limit:
                return found
        if not full and limit is not None and len(found) >= limit:
            break
        if len(rows) < per_page:
            break

    return found


def _series_to_daily_map(pairs: list[list[float]]) -> dict[date, float]:
    """Map CoinGecko ``[ts_ms, value]`` pairs to UTC date → last value that day."""
    out: dict[date, float] = {}
    for item in pairs or []:
        if not item or len(item) < 2:
            continue
        ts_ms, value = item[0], item[1]
        if value is None:
            continue
        d = _ms_to_date(ts_ms)
        out[d] = float(value)
    return out


def _ohlc_to_daily(ohlc: list[list[float]]) -> dict[date, dict[str, float]]:
    """Collapse OHLC candles to one bar per UTC date (last candle wins)."""
    out: dict[date, dict[str, float]] = {}
    for candle in ohlc or []:
        if not candle or len(candle) < 5:
            continue
        ts_ms, o, h, l, c = candle[:5]
        d = _ms_to_date(ts_ms)
        out[d] = {
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
        }
    return out


def build_daily_bars(
    asset_id: str,
    *,
    ohlc: list[list[float]] | None,
    market_chart: dict[str, Any],
    source: str = SOURCE,
    ingested_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Merge OHLC + market_chart into ``market_daily``-shaped bar dicts.

    If OHLC is missing/empty, fall back to chart close for O/H/L/C (documented
    limitation — not fabricated randomness; same observed price).
    """
    ingested_at = ingested_at or _utc_now()
    prices = _series_to_daily_map(market_chart.get("prices") or [])
    mcaps = _series_to_daily_map(market_chart.get("market_caps") or [])
    vols = _series_to_daily_map(market_chart.get("total_volumes") or [])
    ohlc_daily = _ohlc_to_daily(ohlc or [])

    dates = sorted(set(prices) | set(mcaps) | set(vols) | set(ohlc_daily))
    bars: list[dict[str, Any]] = []
    for d in dates:
        close = prices.get(d)
        ohlc_bar = ohlc_daily.get(d)
        if ohlc_bar is not None:
            open_, high, low, close = (
                ohlc_bar["open"],
                ohlc_bar["high"],
                ohlc_bar["low"],
                ohlc_bar["close"],
            )
        elif close is not None:
            open_ = high = low = close
        else:
            continue
        bars.append(
            {
                "timestamp": d,
                "asset_id": asset_id,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume_usd": vols.get(d),
                "market_cap_usd": mcaps.get(d),
                "source": source,
                "ingested_at": ingested_at,
            }
        )
    return bars


def fetch_market_daily(
    asset_id: str,
    *,
    start: Any = None,
    end: Any = None,
    client: CoinGeckoClient | None = None,
    days: int | str = DEFAULT_HISTORY_DAYS,
    include_ohlc: bool = True,
) -> list[dict[str, Any]]:
    """Fetch daily bars for one asset from CoinGecko.

    Returns:
        List of bar dicts with timestamp, OHLCV, market_cap_usd.

    Raises:
        CoinGeckoError / RateLimitError on API failure (no fake prices).
    """
    owns = client is None
    cg = client or CoinGeckoClient()
    try:
        chart = cg.get_market_chart(asset_id, days=days)
        ohlc: list[list[float]] | None = None
        if include_ohlc:
            # OHLC endpoint only accepts certain day values on free/demo.
            ohlc_days = days if isinstance(days, int) else 180
            if ohlc_days not in {1, 7, 14, 30, 90, 180, 365}:
                ohlc_days = 180
            try:
                ohlc = cg.get_ohlc(asset_id, days=ohlc_days)
            except CoinGeckoError as exc:
                logger.warning("OHLC unavailable for %s (%s); using chart close", asset_id, exc)
                ohlc = None
        bars = build_daily_bars(asset_id, ohlc=ohlc, market_chart=chart)
        if start is not None:
            start_d = pd.Timestamp(start).date()
            bars = [b for b in bars if b["timestamp"] >= start_d]
        if end is not None:
            end_d = pd.Timestamp(end).date()
            bars = [b for b in bars if b["timestamp"] <= end_d]
        return bars
    finally:
        if owns:
            cg.close()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


def _bars_to_dataframe(bars: list[dict[str, Any]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "asset_id",
                "open",
                "high",
                "low",
                "close",
                "volume_usd",
                "market_cap_usd",
                "source",
                "ingested_at",
            ]
        )
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    return df


def upsert_assets(
    conn: duckdb.DuckDBPyConnection,
    assets: list[dict[str, Any]],
) -> int:
    """Upsert ``assets``, preserving existing ``first_seen`` when present."""
    if not assets:
        return 0
    df = pd.DataFrame(assets)
    conn.register("_assets_df", df)
    conn.execute(
        """
        INSERT INTO assets AS a (
            asset_id, symbol, name, category, listing_date, source,
            is_active, first_seen, last_seen
        )
        SELECT
            asset_id, symbol, name, category, listing_date, source,
            is_active, first_seen, last_seen
        FROM _assets_df
        ON CONFLICT (asset_id) DO UPDATE SET
            symbol = excluded.symbol,
            name = excluded.name,
            category = coalesce(excluded.category, a.category),
            listing_date = coalesce(excluded.listing_date, a.listing_date),
            source = excluded.source,
            is_active = excluded.is_active,
            first_seen = a.first_seen,
            last_seen = excluded.last_seen
        """
    )
    conn.unregister("_assets_df")
    return len(df)


def upsert_market_daily(
    conn: duckdb.DuckDBPyConnection,
    bars: list[dict[str, Any]],
) -> int:
    """Insert or replace rows in ``market_daily``."""
    if not bars:
        return 0
    df = _bars_to_dataframe(bars)
    conn.register("_market_df", df)
    conn.execute(
        """
        INSERT OR REPLACE INTO market_daily
        SELECT
            timestamp, asset_id, open, high, low, close,
            volume_usd, market_cap_usd, source, ingested_at
        FROM _market_df
        """
    )
    conn.unregister("_market_df")
    return len(df)



def existing_bar_counts(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Return asset_id → bar count from ``market_daily`` (empty if table missing)."""
    try:
        rows = conn.execute(
            "SELECT asset_id, COUNT(*)::INTEGER AS n FROM market_daily GROUP BY asset_id"
        ).fetchall()
    except duckdb.Error:
        return {}
    return {str(aid): int(n) for aid, n in rows}


def load_bars_from_raw(
    cg_raw: Path,
    asset_id: str,
    *,
    include_ohlc: bool = False,
    ingested_at: datetime | None = None,
) -> list[dict[str, Any]] | None:
    """Build bars from on-disk ``market_chart.json`` if present (resume / offline)."""
    chart_path = cg_raw / asset_id / "market_chart.json"
    if not chart_path.is_file():
        return None
    try:
        chart = json.loads(chart_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Raw chart unreadable for %s: %s", asset_id, exc)
        return None
    if not isinstance(chart, dict):
        return None
    ohlc: list[list[float]] | None = None
    if include_ohlc:
        ohlc_path = cg_raw / asset_id / "ohlc.json"
        if ohlc_path.is_file():
            try:
                loaded = json.loads(ohlc_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    ohlc = loaded
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Raw OHLC unreadable for %s: %s", asset_id, exc)
    return build_daily_bars(
        asset_id, ohlc=ohlc, market_chart=chart, ingested_at=ingested_at
    )



def load_cached_market_candidates(cg_raw: Path) -> list[dict[str, Any]]:
    """Load the newest ``markets_candidates_*.json`` dump if present."""
    files = sorted(cg_raw.glob("markets_candidates_*.json"))
    if not files:
        return []
    path = files[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cached candidates unreadable (%s): %s", path.name, exc)
        return []
    if not isinstance(data, list):
        return []
    logger.info("Loaded %s cached candidates from %s", len(data), path.name)
    return data


def ingest_coingecko(
    *,
    api_key: str | None = None,
    asset_ids: list[str] | None = None,
    limit: int | None = 5,
    full: bool = False,
    dry_run: bool = False,
    days: int | str = DEFAULT_HISTORY_DAYS,
    include_ohlc: bool = False,
    min_bars: int | None = None,
    duckdb_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    client: CoinGeckoClient | None = None,
    universe: dict[str, Any] | None = None,
    skip_existing: bool = True,
    prefer_raw: bool = True,
    **kwargs: Any,
) -> IngestResult:
    """Pull daily OHLCV + market cap from CoinGecko into raw/DuckDB.

    Args:
        api_key: Optional; defaults to env ``COINGECKO_API_KEY``. Never hardcode.
        asset_ids: Optional CoinGecko ids; None = discover via markets + MC bands.
        limit: Target cap on assets with ≥``min_bars`` in DuckDB after this run
            (counts already-present assets when ``skip_existing``). Default 5.
        full: Paginate markets more aggressively for a wider candidate set.
        dry_run: Discover (and optionally resolve ids) but do not fetch charts
            or write DuckDB/parquet.
        days: History window for market_chart / OHLC (prefer ≥90 daily bars).
        include_ohlc: Also call ``/ohlc`` (extra rate-limit cost). Default False
            uses market_chart close for O/H/L/C.
        min_bars: Skip assets with fewer daily bars (default from universe.yaml).
        duckdb_path: DuckDB file path (default ``data/cmram.duckdb``).
        raw_dir: Raw dump directory (default ``data/raw``).
        client: Optional injected client (tests).
        universe: Optional preloaded universe config.
        skip_existing: If True, do not re-fetch assets that already have
            ≥``min_bars`` in ``market_daily`` (restarts resume).
        prefer_raw: If True, reuse on-disk ``market_chart.json`` before calling API.

    Returns:
        IngestResult with counts.

    Raises:
        CoinGeckoError / RateLimitError: API failures — no fake prices written.
    """
    _ = kwargs  # forward-compatible
    # Load API keys from env / box-secrets (never log values).
    try:
        from cmram.secrets_env import ensure_env_secrets

        ensure_env_secrets(("COINGECKO_API_KEY",))
    except Exception:  # noqa: BLE001 — ingest must still run if secrets helper fails
        logger.debug("ensure_env_secrets skipped", exc_info=True)

    cfg = universe or load_universe_config()
    if min_bars is None:
        min_bars = int(cfg.get("min_history_days") or MIN_BARS_DEFAULT)

    db_path = Path(duckdb_path) if duckdb_path is not None else get_duckdb_path()
    raw_root = Path(raw_dir) if raw_dir is not None else get_raw_dir()
    cg_raw = raw_root / "coingecko"
    result = IngestResult(dry_run=dry_run, raw_dir=str(cg_raw), duckdb_path=str(db_path))

    owns_client = client is None
    cg = client or CoinGeckoClient(api_key=api_key)
    try:
        meta_rows: list[dict[str, Any]] = []
        if asset_ids:
            # Skip /coins/markets when ids are explicit (saves rate-limit budget).
            # Symbol/name filled from id; later ingest can enrich.
            # Do not pre-slice by --limit when skip_existing: already-present
            # ids would waste the slice; the fetch loop enforces remaining slots.
            ids = list(asset_ids)
            meta_rows = [
                {
                    "id": cid,
                    "symbol": cid,
                    "name": cid,
                    "market_cap": None,
                    "total_volume": None,
                }
                for cid in ids
            ]
        else:
            # Oversample the band pool so --limit N can skip short-history coins
            # (top-volume micro-caps are often newly listed).
            target = limit if limit is not None else 5
            # Oversample modestly: many top-volume names in the MC bands are
            # newly listed and fail min_history_days. --full widens the pool.
            if full:
                pool_limit = None if limit is None else max(target * 10, target)
                pages = 20
            else:
                pool_limit = max(target * 3, target)
                pages = 3
            try:
                meta_rows = discover_band_candidates(
                    cg,
                    limit=pool_limit,
                    full=full,
                    max_pages=pages,
                    universe=cfg,
                )
            except RateLimitError as exc:
                cached = load_cached_market_candidates(cg_raw)
                if not cached:
                    raise
                logger.warning(
                    "Markets discovery rate-limited (%s); falling back to cached candidates",
                    exc,
                )
                meta_rows = cached
                if pool_limit is not None:
                    meta_rows = meta_rows[:pool_limit]
            if not meta_rows:
                cached = load_cached_market_candidates(cg_raw)
                if cached:
                    logger.warning(
                        "Live markets returned no band matches; using cached candidates"
                    )
                    meta_rows = cached
                    if pool_limit is not None:
                        meta_rows = meta_rows[:pool_limit]
            if not meta_rows:
                raise CoinGeckoError(
                    "No CoinGecko markets candidates found in MC band union "
                    f"(${BAND_UNION_MC_MIN:,.0f}–${BAND_UNION_MC_MAX:,.0f}). "
                    "Try --full, raise --limit, or pass --ids."
                )

        # Apply universe eligibility (tokenized stocks / stables / FX) before fetch.
        from cmram.config import load_eligibility_config
        from cmram.universe.eligibility import evaluate_asset

        elig_cfg = load_eligibility_config()
        kept: list[dict[str, Any]] = []
        elig_skipped: list[str] = []
        for row in meta_rows:
            aid = str(row.get("id") or "")
            decision = evaluate_asset(
                aid,
                symbol=row.get("symbol"),
                name=row.get("name"),
                category=row.get("category") or row.get("categories"),
                eligibility_config=elig_cfg,
            )
            if decision.eligible:
                kept.append(row)
            else:
                elig_skipped.append(aid)
                result.skipped_insufficient.append(
                    f"{aid}:eligibility:{decision.reason_excluded or 'ineligible'}"
                )
        if elig_skipped:
            logger.info(
                "Eligibility excluded %s candidates before fetch (sample: %s)",
                len(elig_skipped),
                ", ".join(elig_skipped[:12]),
            )
        meta_rows = kept

        result.candidates = [str(r["id"]) for r in meta_rows]
        logger.info(
            "CoinGecko candidates after eligibility (%s): %s",
            len(result.candidates),
            ", ".join(result.candidates[:40])
            + ("..." if len(result.candidates) > 40 else ""),
        )

        stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        _write_json(cg_raw / f"markets_candidates_{stamp}.json", meta_rows)

        if dry_run:
            logger.info("dry-run: skipping chart fetch and DuckDB writes")
            return result

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = init_db(db_path)
        all_bars: list[dict[str, Any]] = []
        asset_records: list[dict[str, Any]] = []
        now = _utc_now()
        already = existing_bar_counts(conn) if skip_existing else {}
        # Count only eligibility-passing assets toward --limit so denylisted
        # stables/tokenized stocks already in DuckDB do not consume slots.
        from cmram.config import load_eligibility_config as _load_elig
        from cmram.universe.eligibility import evaluate_asset as _eval_asset

        _elig = _load_elig()
        n_db_ready = 0
        for aid, n in already.items():
            if n < min_bars:
                continue
            if _eval_asset(aid, eligibility_config=_elig).eligible:
                n_db_ready += 1
        if skip_existing and limit is not None:
            remaining_slots = max(0, limit - n_db_ready)
        else:
            remaining_slots = limit  # None = unlimited new lands this run
        logger.info(
            "Resume state: %s assets already ≥%s bars; remaining slots=%s "
            "(skip_existing=%s prefer_raw=%s)",
            n_db_ready,
            min_bars,
            remaining_slots,
            skip_existing,
            prefer_raw,
        )

        try:
            if remaining_slots == 0:
                logger.info(
                    "Already at or above --limit %s with sufficient history; nothing to fetch",
                    limit,
                )
                return result

            newly_landed = 0
            for row in meta_rows:
                if remaining_slots is not None and newly_landed >= remaining_slots:
                    break
                asset_id = str(row["id"])
                if skip_existing and already.get(asset_id, 0) >= min_bars:
                    logger.info(
                        "Skipping %s: already have %s bars in DuckDB",
                        asset_id,
                        already[asset_id],
                    )
                    result.skipped_insufficient.append(
                        f"{asset_id}:already_present:{already[asset_id]}"
                    )
                    continue

                bars: list[dict[str, Any]] | None = None
                from_raw = False
                if prefer_raw:
                    bars = load_bars_from_raw(
                        cg_raw,
                        asset_id,
                        include_ohlc=include_ohlc,
                        ingested_at=now,
                    )
                    if bars is not None and len(bars) >= min_bars:
                        from_raw = True
                        logger.info(
                            "Using on-disk raw chart for %s (%s bars)",
                            asset_id,
                            len(bars),
                        )
                    elif bars is not None:
                        logger.info(
                            "Raw chart for %s too short (%s < %s); fetching API",
                            asset_id,
                            len(bars),
                            min_bars,
                        )
                        bars = None

                if bars is None:
                    try:
                        chart = cg.get_market_chart(asset_id, days=days)
                        ohlc: list[list[float]] | None = None
                        if include_ohlc:
                            ohlc_days = days if isinstance(days, int) else 180
                            if ohlc_days not in {1, 7, 14, 30, 90, 180, 365}:
                                ohlc_days = 180
                            try:
                                ohlc = cg.get_ohlc(asset_id, days=int(ohlc_days))
                            except CoinGeckoError as exc:
                                logger.warning(
                                    "OHLC failed for %s (%s); chart close fallback",
                                    asset_id,
                                    exc,
                                )
                                ohlc = None

                        _write_json(cg_raw / asset_id / "market_chart.json", chart)
                        if ohlc is not None:
                            _write_json(cg_raw / asset_id / "ohlc.json", ohlc)

                        bars = build_daily_bars(
                            asset_id, ohlc=ohlc, market_chart=chart, ingested_at=now
                        )
                    except RateLimitError as exc:
                        logger.error(
                            "Rate limited on %s after some progress; "
                            "stopping remaining fetches: %s",
                            asset_id,
                            exc,
                        )
                        result.skipped_insufficient.append(f"{asset_id}:rate_limit")
                        if result.market_rows_upserted == 0 and newly_landed == 0:
                            raise
                        break
                    except CoinGeckoError as exc:
                        logger.error("Skipping %s due to API error: %s", asset_id, exc)
                        result.skipped_insufficient.append(f"{asset_id}:api_error:{exc}")
                        continue

                if len(bars) < min_bars:
                    logger.warning(
                        "Skipping %s: only %s daily bars (need ≥%s)%s",
                        asset_id,
                        len(bars),
                        min_bars,
                        " [raw]" if from_raw else "",
                    )
                    result.skipped_insufficient.append(
                        f"{asset_id}:insufficient_bars:{len(bars)}"
                    )
                    continue

                parquet_path = cg_raw / asset_id / "market_daily.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                _bars_to_dataframe(bars).to_parquet(parquet_path, index=False)

                n = upsert_market_daily(conn, bars)
                result.market_rows_upserted += n
                all_bars.extend(bars)

                rec = {
                    "asset_id": asset_id,
                    "symbol": (row.get("symbol") or asset_id).upper(),
                    "name": row.get("name") or asset_id,
                    "category": None,
                    "listing_date": None,
                    "source": SOURCE,
                    "is_active": True,
                    "first_seen": now,
                    "last_seen": now,
                }
                asset_records.append(rec)
                result.assets_upserted += upsert_assets(conn, [rec])
                newly_landed += 1
                already[asset_id] = len(bars)
                if remaining_slots is not None and newly_landed >= remaining_slots:
                    logger.info(
                        "Reached --limit %s assets with sufficient history "
                        "(%s already in DB + %s new)",
                        limit,
                        n_db_ready,
                        newly_landed,
                    )
                    break

            if all_bars:
                combined = cg_raw / f"market_daily_{stamp}.parquet"
                _bars_to_dataframe(all_bars).to_parquet(combined, index=False)
        finally:
            conn.close()

        return result
    finally:
        if owns_client:
            cg.close()


# Re-export errors for callers / CLI
__all__ = [
    "BAND_UNION_MC_MAX",
    "BAND_UNION_MC_MIN",
    "IngestResult",
    "build_daily_bars",
    "discover_band_candidates",
    "load_cached_market_candidates",
    "existing_bar_counts",
    "fetch_market_daily",
    "ingest_coingecko",
    "load_bars_from_raw",
    "upsert_assets",
    "upsert_market_daily",
    "CoinGeckoError",
    "RateLimitError",
]
