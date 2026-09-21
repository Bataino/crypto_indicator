# Phase A — CoinGecko micro-cap data growth

**Zone:** Africa/Lagos (WAT / UTC+1)  
**Report time:** 2026-09-21 08:51:54 WAT  
**Goal:** Grow `market_daily` toward ≥100,000 coin-days for MC band **$1M–$100M** (Band C).

## Config / code changes

1. **`config/universe.yaml`** — added band `C_1_100`:
   - `market_cap_min_usd: 1_000_000`
   - `market_cap_max_usd: 100_000_000`
   - `label: "Band C"`
   - Kept existing `A_5_50` and `B_10_100`.
2. **`src/cmram/ingest/market.py`** — `_band_union_bounds` now unions **all** bands in config
   (min of `market_cap_min_usd` / max of `market_cap_max_usd` across every band), not
   hard-coded A/B only. Module docstring + fallback constants updated to $1M–$100M.
3. **`scripts/run_ingest.py`** — calls `ensure_env_secrets(("COINGECKO_API_KEY",))` before
   API work; prints present/length only (never secret values). `ingest_coingecko` also
   loads secrets internally.
4. Prefer-raw: short on-disk charts (`< min_bars`) now fall through to the API on the
   next run instead of being skipped.

Band union after change: **($1,000,000, $100,000,000)**.

## Start baseline

| Metric | Count |
|--------|------:|
| assets | 166 |
| market_daily rows | 29,188 |
| assets with ≥90 bars | 166 |

## Dry-run discovery

- Command: `scripts/run_ingest.py --full --dry-run --limit 2000`
- Band union used: **$1,000,000 – $100,000,000**
- Candidates after eligibility: **~2,009**
- Eligibility excluded: **~269** (tokenized stocks / stables / FX)
- Log: `logs/phase_a_dry_run.log`

## Ingest run

- Command: `scripts/run_ingest.py --full --limit 700 --days 180 -v`
- Log: `logs/phase_a_ingest.log`
- Resume: `skip_existing=True` (default) — 150 assets already ≥90 bars → 550 slots remaining
- Result: **Reached --limit 700** (150 already + 550 newly landed)
- This run upserts: `assets_upserted=550`, `market_rows_upserted=97,041`
- API key: present=True, length=27 (Demo). No hard rate-limit abort; client throttled
  (~2.5s demo interval) and completed. Apparent “429” regex hits in the verbose log were
  false positives inside etag/request-id hex on HTTP **200** responses.
- Short-history skips: ~122 candidates with &lt;90 daily bars (young listings / short raw).

## End counts

| Metric | Start | End | Δ |
|--------|------:|----:|--:|
| assets | 166 | **716** | +550 |
| market_daily rows | 29,188 | **126,229** | +97,041 |
| assets ≥90 bars | 166 | **716** | +550 |
| assets ≥180 bars | — | 655 | — |

### Success vs ≥100K target

**REACHED.** `market_daily` = **126,229** rows (≥100,000).

## Universe membership (post-ingest)

Command: `scripts/run_universe.py`

| Band | membership rows | assets | liquidity pass |
|------|----------------:|-------:|---------------:|
| A_5_50 | 42,535 | 575 | 98.8% |
| B_10_100 | 40,116 | 544 | 99.2% |
| **C_1_100** | **59,393** | **700** | 98.1% |
| Total membership rows | 142,044 | 700 unique | — |

Date range in membership: 2026-06-18 → 2026-09-21 (WAT calendar dates from UTC bars).

## Resume command (optional further growth)

```bash
cd /workspace/cmram
.venv/bin/python scripts/run_ingest.py --full --limit 900 --days 180 -v \
  2>&1 | tee -a logs/phase_a_ingest.log
.venv/bin/python scripts/run_universe.py
```

`skip_existing` (default) resumes without re-fetching assets that already have ≥90 bars.

## Explicit non-goals (honored)

- No LightGBM / feature training
- No Santiment / X ingest this phase
- No trading / wallets
- No GitHub push
