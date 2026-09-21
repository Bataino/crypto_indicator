# CMRAM — Crypto Micro-Cap Rotation Alpha Model

**Version:** V0.1 research scaffold  
**Status:** Phase 1 — ingest + universe + Models A/B/C features + social N (Trends/Reddit/X) + signals + forward-return backtest  
**Ingested assets:** **50** in DuckDB (≥90d sample). **Eligible membership (post filter):** **38** (excludes tokenized stocks / stables / FX).

## Purpose

We **create** the indicators ourselves (**MREI**, **NSI**, **Rotation Gap**) from market data **we pull**, then **test** whether they predict forward coin returns.

Hypothesis (to validate, not assume): assets with rising participation, accelerating attention, improving liquidity, and low narrative saturation have higher probability of positive forward returns.

## Non-goals

- No live trading, order placement, or wallets
- No ML until baseline Models A→D clear
- No assumed alpha / no fake backtest results

## Locked parameters (Phase 1)

| Param | Value |
|-------|--------|
| Band A | $5M–$50M MC |
| Band B | $10M–$100M MC |
| Min history | ≥ 90 daily bars |
| Entry convention | next-day open (proposed) |
| Social | Trends + Reddit + X counts + Santiment social_volume (FREE lag) |

Threshold grids (`τ_m`, `τ_g`, `V_min`) are **calibration TBD** — see `config/thresholds.yaml`.

## Install

```bash
cd /workspace/cmram
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Alternatively without install:

```bash
PYTHONPATH=src pytest
```

## Layout

```text
cmram/
  specs/           # Research + Data Engineering specs
  config/          # universe.yaml, thresholds.yaml, eligibility.yaml, features.yaml
  data/{raw,interim,features,signals,backtests}/
  src/cmram/       # ingest, universe, features, signals, backtest, report, db
  scripts/         # run_*.py CLIs
  tests/
```

## Market ingest (CoinGecko)

Research data collection only — **no trading**.

### What it does

1. Discovers candidates from `/coins/markets` (volume desc), keeping coins whose **current** market cap falls in the Band A∪B union (**$5M–$100M**).
2. For each asset, fetches:
   - `/coins/{id}/market_chart?vs_currency=usd&days=180` — daily prices, market caps, volumes
   - Optionally `/coins/{id}/ohlc?vs_currency=usd&days=180` via `--ohlc` (true OHLC; default uses chart close for O/H/L/C to halve API calls)
3. Writes raw JSON/Parquet under `data/raw/coingecko/` and upserts DuckDB tables `assets` + `market_daily` (default DB: `data/cmram.duckdb`).

### Survivorship note

Live markets pages only include coins CoinGecko still lists. Delisted/failed names are missing — **survivorship bias** if you treat today’s scrape as a historical universe. Point-in-time membership belongs in later `universe_membership` work with archived snapshots.

### Rate limits & API key

| Mode | Env | Notes |
|------|-----|--------|
| Public (no key) | — | Hard throttle (~90s/call); shared IPs often hit **429** — prefer Demo key |
| Demo | `COINGECKO_API_KEY` | Sends `x-cg-demo-api-key`; ~2.5s spacing |
| Pro | `COINGECKO_API_KEY` + `COINGECKO_PRO=1` (or `COINGECKO_API_PLAN=pro`) | `pro-api.coingecko.com` + `x-cg-pro-api-key` |

On 429 the client backs off (floor above `Retry-After: 0`) and retries. Mid-run rate limits **stop further fetches but keep already-upserted assets** (exit 0 if any progress; exit **2** only if nothing landed). Restarts **skip** assets already in DuckDB with ≥`min_history_days` bars and reuse on-disk `market_chart.json` when present (`--no-skip-existing` / `--no-prefer-raw` to disable). Live `/coins/markets` discovery falls back to the newest `data/raw/coingecko/markets_candidates_*.json` if rate-limited.

Optional paths: `CMRAM_DUCKDB_PATH`, `CMRAM_RAW_DIR`.

### How to run

```bash
# Small smoke run (recommended first; chart-only = fewer API calls)
python scripts/run_ingest.py --limit 3 -v

# True OHLC candles (extra CoinGecko calls)
python scripts/run_ingest.py --limit 3 --ohlc -v

# Dry-run discovery only
python scripts/run_ingest.py --limit 5 --dry-run

# Explicit CoinGecko ids (skips /coins/markets; best under public rate limits)
python scripts/run_ingest.py --ids saga-2,renzo,thena --limit 10

# Wider candidate scrape (still capped by --limit; resumes past already-ingested)
python scripts/run_ingest.py --full --limit 25 -v
```

### Current DuckDB landing (research sample)

Widen ingest landed **50** assets in `assets` / `market_daily` (many with ≥90 daily bars). That raw set included non-strategy names (tokenized stocks, stables/FX) that must not enter Band A/B research.

### Universe eligibility (data-quality filter)

Config: `config/eligibility.yaml`. Applied in `run_universe` before band membership:

- Prefer CoinGecko `assets.category` when present
- Else heuristics on `asset_id` / `symbol` / `name` + explicit `deny_ids`
- Excludes tokenized stocks (xStock / bStock / Ondo tokenized), stablecoins / fiat-backed / wrapped fiat, and listed FX-stable ids
- Flags `assets.universe_eligible` + `universe_exclude_reason`

**After filter:** **38** membership assets (12 excluded). **No alpha claimed.**

### Universe membership

Point-in-time bands A/B + history (≥90 bars) + eligibility + draft liquidity flags:

```bash
python scripts/run_universe.py
python scripts/run_universe.py --db data/cmram.duckdb
```

Writes `universe_membership` (liquidity PASS uses first `v_min_usd` candidate; Amihud stored, percentile gate off while `illiq_max_percentile: null`). Ineligible assets are omitted from membership and listed in the CLI report.

### Social / Narrative Acceleration (N)

Abdul: **N is the main hypothesis**; C/V/P support. V0.1 free sources (priority):

1. **Google Trends** (`pytrends`, `today 3-m` daily interest)
2. **Reddit** (public JSON; optional PRAW if `REDDIT_CLIENT_ID`/`SECRET`)
3. **X/Twitter** (bearer via `X_BEARER_TOKEN` / `TWITTER_BEARER_TOKEN`, or box-secrets card; never logged)
   - **Default = Counts: Recent** (`GET /2/tweets/counts/recent`, `granularity=day`, last 7 days, **one request per asset**). Mention totals only — we do **not** download tweet text.
   - **Search** (`/2/tweets/search/recent`) is an **expensive opt-in** (`--x-mode search`); default **1 page**, `max_results` 10.
   - Hard spend guard: `estimated_cost_cap_usd` (default **$1.00**). Estimate = assets × requests/asset × `cost_per_request_usd` (default **$0.005** for Counts: Recent). If the estimate is **above the cap**, the run **stops before any X HTTP**. Override only with `--x-allow-over-cap` or a higher `--x-cost-cap-usd`.
   - `cost_per_request_usd` is configurable. The **X developer console price wins** if it differs — update the YAML; do not assume $0.005 is frozen.
4. Optional: **Wikipedia pageviews** (mapped titles in `config/social.yaml`)
5. **Santiment** (`social_volume_total` via GraphQL, `SANTIMENT_API_KEY`)
   - Auth header: `Authorization: Apikey <key>` (env or box-secrets card; never logged).
   - **FREE plan limits** (typical): ~**1000 calls/mo**, ~**1y** history, ~**30d lag** on restricted metrics (`restrictedFrom` / `restrictedTo`). Leave call headroom.
   - Maps CoinGecko `asset_id` / symbol / name → Santiment slug (`config/social.yaml` overrides).
   - Attention proxy only — **not alpha**.

Later optional paid extras: LunarCrush TBD.

```bash
python scripts/run_social_ingest.py -v
python scripts/run_social_ingest.py --sources trends,wikipedia --limit 5

# Cheap X mention counts (default). Prints a cost estimate; stops if over cap.
python scripts/run_social_ingest.py --sources x --x-mode counts -v

# After adding $5–$10 X API credits (Counts: Recent ≈ $0.005/request; 38 assets ≈ $0.19):
python scripts/run_social_ingest.py --sources x --x-mode counts -v

# Optional tiny smoke first
python scripts/run_social_ingest.py --sources x --x-mode counts --x-max-assets 5 -v

# EXPENSIVE opt-in: actual posts, 1 page cap — do not use unless you need tweet text
python scripts/run_social_ingest.py --sources x --x-mode search --x-max-assets 3 -v

# Santiment free-plan: coverage probe first (1–2 GraphQL calls)
python scripts/run_social_ingest.py --sources santiment --santiment-probe-only -v

# Then ingest mapped assets within a safe session budget (leave monthly headroom)
python scripts/run_social_ingest.py --sources santiment --santiment-call-budget 150 -v
```

Writes `social_daily`, `data/raw/social/social_daily.parquet`, coverage at `data/raw/social/n_coverage.md`, `x_ingest_note.md`, and `santiment_coverage.md` (free-plan lag/history note). On X HTTP **402 credits depleted** / 403 plan limits the run **stops cleanly** (no retry loop). On cost-cap breach it also stops **before** calling X. **Not alpha.**

N score (research proxy): attention → 7d/28d velocity → acceleration → × low-base multiplier `1/(1+log1p(baseline))` → CS percentiles within `(timestamp, band)` → `score_N`. Model **C** = C+V+P+N when `n_available`; Models A/B unchanged (ignore N).

### Features (Models A/B/C)


Point-in-time C / V / P / (N) inputs → cross-sectional percentiles within `(timestamp, band)` → MREI / NSI / Rotation Gap. **No claimed alpha.** Social is not imputed: `n_available=true` only when `social_daily` yields a score.

```bash
python scripts/run_features.py
python scripts/run_features.py --db data/cmram.duckdb
python scripts/run_features.py --no-parquet
```

Writes versioned rows to DuckDB `features_daily` (compound key `timestamp, asset_id, band, model`) and `data/features/features_daily.parquet`. Lookbacks live in `config/features.yaml`.

| Model | MREI | NSI |
|-------|------|-----|
| A | P only | price extension + momentum deceleration |
| B | mean(C, V, P) | price + volume exhaustion + momentum |
| C | mean(C, V, P, N*) | price + volume + momentum (+ social saturation when N) |

Relative strength uses BTC when `bitcoin` is in `market_daily`; otherwise the **band equal-weight** n-day return of that day's members (PIT). With a modest n per band-day, cross-sectional percentiles remain coarse and `small_sample=true` may still apply.

### Signals (τ grid)

Rule: `liquidity_pass AND MREI > τ_m AND Rotation_Gap > τ_g` for each `(τ_m, τ_g)` in `config/thresholds.yaml`, Models A/B, both bands. Nested τ sets are **kept and tagged** (research clarity).

Entry: **next available bar** after signal date (`next_day_open`). If market OHLC is chart-close-only (`O=H=L=C`), entry uses that bar’s **close as open proxy** (documented in report attrs / CLI).

```bash
python scripts/run_signals.py
python scripts/run_signals.py --db data/cmram.duckdb
```

Writes DuckDB `signals` + `data/signals/signals.parquet`.

### Backtest (forward returns)

Per signal × horizon ∈ {1,3,7,14,21,30}: `ret`, MFE, MAE. Excess vs same-band **random**, **momentum** (top ret_7), **volume** (top RVOL else top volume), and **market** (BTC if present else equal-weight band).

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --db data/cmram.duckdb --seed 42
```

Writes `backtest_results`, `data/backtests/backtest_results.parquet`, `backtest_aggregate.parquet`, and `backtest_report.md`.

**⚠ SMALL SAMPLE:** with n≈38 eligible assets this is still exploratory — reports scream **NOT CONCLUSIVE** / do not claim alpha.

### Time holdout (explore vs holdout)

Full-sample τ grids are **not** validated. Split membership calendar days: first 70% = explore (peek grid), last 30% = holdout (pre-registered cells only).

```bash
python scripts/run_holdout.py
python scripts/run_holdout.py --db data/cmram.duckdb --explore-frac 0.70 --seed 42
```

Writes `data/backtests/holdout_report.md` with HARD warnings, explore τ table (median excess vs random at h=7), and holdout metrics for Model B/A τ_m=50 τ_g=20 (+ explore top-1 if different). **No alpha claims.**

### What’s still stubbed

| Stage | Module / script | Notes |
|-------|-----------------|-------|
| Dataset / feature reports | `report/dataset.py`, `report/feature_analysis.py` | optional narratives |
| Report CLI | `scripts/run_report.py` | wires remaining reports |

DuckDB schema: `src/cmram/db/schema.sql` (tables per DE Spec §6).

```bash
python scripts/run_report.py   # still stub
```

## Tests

```bash
pytest
# Optional live smoke (uses network / rate limits):
CMRAM_LIVE_INGEST=1 pytest -m integration
```

## Secrets

API keys via environment or box-secrets card (`COINGECKO_API_KEY`, `X_BEARER_TOKEN`, `SANTIMENT_API_KEY`, etc.). Presence + length only in logs — never print key values. Never commit `.env` or keys.

## Next step

Social N path: `run_social_ingest` → `run_features` (Model C) → signals/backtest. X needs a bearer token **and credits**; Santiment FREE has ~30d lag / ~1y window / ~1000 calls/mo. Reddit public often blocked from datacenter IPs. **Do not treat scores or backtest tables as alpha.**

## Related AI trainer

See [docs/RELATED_AI_TRAINER.md](docs/RELATED_AI_TRAINER.md) → https://github.com/Bataino/crypto_ai_trainer
