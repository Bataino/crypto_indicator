# CMRAM Data Engineering Specification

**Version:** V0.1  
**Status:** Research Phase — build only after Abdul signs off  
**Companion to:** `CMRAM_V0.1_Research_Spec.md`  
**Owner:** Quant Research (AbdulBateen)  
**Non-goals:** live trading, order placement, wallets, ML until Phase 1 clears

---

## 1. Purpose

Turn the CMRAM research concept into a reproducible dataset + feature + backtest pipeline that can answer:

> Does a high Rotation Gap (MREI − NSI), with liquidity control, historically identify micro-caps that outperform over 1–30d forward windows?

This doc locks **how data is sourced, stored, computed, and evaluated** — not trading.

---

## 2. Locked research parameters (inputs from research)

| Parameter | Value |
|-----------|--------|
| Universe Band A | Market cap `$5M–$50M` |
| Universe Band B | Market cap `$10M–$100M` |
| Min trading history | ≥ 90 daily bars |
| Signal | Liquidity PASS **and** MREI > τ_m **and** Gap > τ_g (τ tested, not assumed) |
| Forward horizons | 1 / 3 / 7 / 14 / 21 / 30 days |
| Models | A price → B +volume → C +social → D full Gap |
| Philosophy | No overfitting; drop features that don’t help; no ML yet |

Volume / Amihud liquidity thresholds: **TBD via calibration** (see §7).

---

## 3. Principles

1. **Point-in-time** — features and universe membership use only data available at signal timestamp (no future leakage).
2. **Survivorship-aware** — prefer APIs that include delisted/inactive assets; document bias if only survivors are available.
3. **Reproducible** — pinned deps, config-driven bands/thresholds, deterministic random seeds for benchmarks.
4. **Incremental** — market+liquidity first; social narrative as optional layer (Model C/D).
5. **Honest missingness** — if N (narrative) is unavailable, score MREI without N and flag `n_available=false`; do not impute fake social.

---

## 4. Python stack (proposed)

| Layer | Choice | Why |
|-------|--------|-----|
| Runtime | Python 3.11+ | Standard research stack |
| Tables / analytics | **DuckDB** (+ Parquet on disk) | Fast local analytics, no server; easy export |
| Optional ORM / SQL | SQLAlchemy only if needed | Keep thin |
| DataFrame | pandas | Features + reports |
| Arrays / stats | numpy, scipy | Returns, tests |
| HTTP | httpx | API clients |
| Config | pydantic-settings + YAML | Bands, τ, paths |
| Plots (reports) | matplotlib (optional plotly) | Backtest charts |
| Tests | pytest | Leakage + unit tests on scores |
| Notebooks | Jupyter only for exploration | Production path = scripts/CLI |

**Out of scope for V0.1:** Spark, cloud warehouses, Redis, live websockets, ML libs (sklearn/torch/etc.).

Suggested layout:

```text
cmram/
  specs/                    # research + this DE spec
  config/
    universe.yaml           # bands A/B, min history
    thresholds.yaml         # τ grids, liquidity drafts
  data/
    raw/                    # immutable API dumps (parquet/json)
    interim/                # cleaned, aligned
    features/               # C,V,N,P,NSI,MREI,Gap
    signals/                # signal events
    backtests/              # run outputs
  src/cmram/
    ingest/                 # market, liquidity, social clients
    universe/               # band filters, history gates
    features/               # C V N P NSI MREI Gap
    signals/                # threshold grids
    backtest/               # forward returns, MFE/MAE, benchmarks
    report/                 # dataset + backtest + feature reports
  tests/
  scripts/
    run_ingest.py
    run_features.py
    run_backtest.py
    run_report.py
  pyproject.toml / requirements.txt
```

---

## 5. Data sources (what we can actually get)

### 5.1 Market data — **Phase 1 primary**

| Source | Use | Notes |
|--------|-----|--------|
| **CoinGecko** (Demo/Pro API) | OHLCV, MC, volume, coin list, categories | Best free/dev starting point; rate limits; Pro if history depth needed |
| CoinMarketCap | Alternate OHLCV/MC | Needs API key; use as backup / cross-check |
| DefiLlama | Optional TVL / DEX volumes later | Not required for Model A/B |

**Required fields per bar (daily first):**  
`timestamp, asset_id, symbol, open, high, low, close, volume_usd, market_cap_usd`

Also store: `exchange_list` / pair availability when API provides it.

**Frequency:** daily for V0.1; hourly only after daily baseline exists.

### 5.2 Liquidity — **Phase 1 from market bars + proxies**

| Metric | Source | V0.1 approach |
|--------|--------|----------------|
| Dollar volume | Market API | Primary gate |
| Amihud ILLIQ | Derived | `mean(|ret| / dollar_volume)` over lookback |
| Spread / book depth | CEX APIs / DEX pools | **Deferred** unless free snapshot available; document as limitation |
| Slippage estimate | Rule of thumb from Amihud + volume | Research proxy only — not execution advice |

### 5.3 Social / narrative — **Phase 1 secondary (Model C/D)**

| Source | Feasibility | Plan |
|--------|-------------|------|
| Google Trends | Medium | Keyword/symbol trends where mappable |
| Reddit (public / API) | Medium | Subreddit + search mention counts if accessible |
| X/Twitter | Hard without paid API | Defer or use only if Abdul provides access |
| Telegram / Discord | Hard / private | Defer to post-validation |
| LunarCrush / similar | Paid | Optional later; not assumed |

**Rule:** Build full pipeline for C/V/P + NSI price/volume legs first. Wire N when a clean source exists; until then Models A/B (and partial D without N) only.

---

## 6. Logical database (DuckDB / Parquet)

IDs are stable strings (prefer CoinGecko `id`; keep `symbol` non-unique).

### 6.1 `assets`

| Column | Type | Notes |
|--------|------|-------|
| asset_id | TEXT PK | e.g. coingecko id |
| symbol | TEXT | |
| name | TEXT | |
| category | TEXT NULL | |
| listing_date | DATE NULL | |
| source | TEXT | `coingecko` etc. |
| is_active | BOOLEAN | as-of last ingest |
| first_seen | TIMESTAMP | |
| last_seen | TIMESTAMP | |

### 6.2 `market_daily`

| Column | Type | Notes |
|--------|------|-------|
| timestamp | DATE | UTC date of bar |
| asset_id | TEXT FK | |
| open, high, low, close | DOUBLE | |
| volume_usd | DOUBLE | |
| market_cap_usd | DOUBLE | |
| source | TEXT | |
| ingested_at | TIMESTAMP | |

**PK:** `(timestamp, asset_id)`

### 6.3 `universe_membership` (point-in-time)

| Column | Type | Notes |
|--------|------|-------|
| timestamp | DATE | |
| asset_id | TEXT | |
| band | TEXT | `A_5_50` or `B_10_100` |
| market_cap_usd | DOUBLE | as-of that day |
| history_days | INT | |
| liquidity_pass | BOOLEAN | |
| amihud | DOUBLE NULL | |
| reason_excluded | TEXT NULL | |

**PK:** `(timestamp, asset_id, band)`

### 6.4 `social_daily` (nullable / sparse)

| Column | Type | Notes |
|--------|------|-------|
| timestamp | DATE | |
| asset_id | TEXT | |
| source | TEXT | `trends`, `reddit`, … |
| mentions | DOUBLE NULL | |
| unique_users | DOUBLE NULL | |
| engagement | DOUBLE NULL | |
| community_growth | DOUBLE NULL | |
| raw_payload_ref | TEXT NULL | path to raw dump |

### 6.5 `features_daily`

| Column | Type | Notes |
|--------|------|-------|
| timestamp | DATE | |
| asset_id | TEXT | |
| band | TEXT | computed in-context of band ranks |
| score_C, score_V, score_N, score_P | DOUBLE | 0–100; N nullable |
| MREI | DOUBLE | |
| nsi_social, nsi_price_ext, nsi_vol_exh, nsi_mom_dec | DOUBLE | |
| NSI | DOUBLE | |
| Rotation_Gap | DOUBLE | MREI − NSI |
| n_available | BOOLEAN | |
| model_version | TEXT | e.g. `features_v0.1` |

### 6.6 `signals`

| Column | Type | Notes |
|--------|------|-------|
| signal_id | TEXT PK | uuid |
| timestamp | DATE | entry as-of |
| asset_id | TEXT | |
| band | TEXT | |
| entry_price | DOUBLE | close or next open — **lock one** (recommend next-day open for realism) |
| market_cap_usd | DOUBLE | |
| volume_usd | DOUBLE | |
| MREI | DOUBLE | |
| NSI | DOUBLE | |
| Rotation_Gap | DOUBLE | |
| tau_m | DOUBLE | |
| tau_g | DOUBLE | |
| model | TEXT | `A`/`B`/`C`/`D` |
| liquidity_pass | BOOLEAN | |

### 6.7 `backtest_results`

| Column | Type | Notes |
|--------|------|-------|
| signal_id | TEXT FK | |
| horizon_d | INT | 1,3,7,14,21,30 |
| ret | DOUBLE | |
| mfe | DOUBLE | |
| mae | DOUBLE | |
| benchmark_id | TEXT | random / momentum / volume / market |
| excess_ret | DOUBLE NULL | vs that benchmark |

### 6.8 `runs` (audit)

| Column | Type | Notes |
|--------|------|-------|
| run_id | TEXT PK | |
| started_at | TIMESTAMP | |
| config_hash | TEXT | |
| git_or_spec_version | TEXT | |
| notes | TEXT | |

---

## 7. Universe & liquidity pipeline

Daily job (conceptual):

1. Ingest / update `market_daily` for candidate set (broad scrape, then filter).
2. For each band A/B:
   - MC in band **on that day** (point-in-time).
   - ≥ 90 prior daily bars.
   - Exclude if volume or MC missing.
3. Liquidity PASS draft (calibrate, don’t freeze):
   - `volume_usd_20d_median ≥ V_min` (grid search candidates: e.g. $50k / $100k / $250k / $500k).
   - `amihud_20d ≤ ILLIQ_max` (set from cross-sectional percentiles, e.g. exclude worst 20%).
4. Write `universe_membership`.

**Bands always evaluated separately** — never pool A and B into one rank universe without labeling.

---

## 8. Feature engineering pipeline

Implements research formulas (see Quant Research formula draft + CMRAM §10–12).

### 8.1 Cross-sectional ranks

Within `(timestamp, band)`, percentile-rank continuous inputs to 0–100 before averaging into C/V/N/P/NSI.

### 8.2 Component stubs (implementation targets)

- **C:** DD from 20d high, range compression 10d/40d, vol compression 7d/30d, sell-pressure fade, stabilization flag.
- **V:** RVOL, volume growth, velocity, acceleration; damp if already extreme.
- **N:** attention velocity/acceleration from low base; `null` if no social row.
- **P:** higher lows, short returns, breakout vs 20d high, RS vs BTC (and optional micro basket).
- **NSI:** social saturation (if any) + price extension + volume exhaustion + momentum deceleration.
- **MREI:** equal-weight mean of available C,V,N,P (renormalize if N missing).
- **Gap:** MREI − NSI.

Config holds all lookback windows — no magic numbers buried in code only.

### 8.3 Model definitions

| Model | Features used |
|-------|----------------|
| A | P (+ price legs of NSI) |
| B | P + V (+ vol/price NSI) |
| C | P + V + N (+ full NSI when social exists) |
| D | Full MREI − NSI |

---

## 9. Signal generation

For each day, band, model, and `(τ_m, τ_g)` on a **pre-registered grid** (examples from research: Gap > 20/30/40/50; MREI grid TBD e.g. 50/60/70):

```text
liquidity_pass
AND MREI > τ_m
AND Rotation_Gap > τ_g
```

Emit one row per signal into `signals`.  
**Entry price convention (proposed):** next calendar day’s open (or next available bar) to reduce same-bar lookahead. Document clearly in reports.

Dedup rule: if multiple τ fire nested sets, either keep all for threshold study **or** analyze nested subsets explicitly — prefer **keep all tagged by τ** for research clarity.

---

## 10. Backtesting framework

### 10.1 Per-signal forward stats

For each horizon h ∈ {1,3,7,14,21,30}:

- `ret_h = price_{t+h} / entry − 1`
- MFE / MAE over `[t, t+h]` (high watermark / low watermark vs entry)
- Win flag: `ret_h > 0`

### 10.2 Aggregates (per band × model × τ × horizon)

- mean / median return, win rate, best / worst
- max drawdown of equal-weight signal portfolio (daily)
- signal frequency, profit factor, simple risk/reward
- distribution of MAE / MFE

### 10.3 Benchmarks (same universe/day)

1. **Random micro-cap** — same band, random assets, matched count / date
2. **Simple momentum** — top recent N-day return in band
3. **Volume breakout** — top RVOL in band
4. **Market** — BTC and/or equal-weight band index

Use same entry convention and horizons. Report **excess** vs each.

### 10.4 Statistical honesty (V0.1)

- Report sample sizes; no “significant” claim without n and a simple test (e.g. bootstrap CI on mean excess vs random).
- Prefer **holdout time split** (e.g. train thresholds on earlier window, evaluate later) before any τ “winner” is trusted.
- Multiple-testing note: many τ × horizons → treat grid as exploration until holdout confirms.

### 10.5 Leakage tests (required)

Automated tests must fail the build if:

- Feature at t uses price/volume after t
- Universe MC uses future cap
- Entry uses same-bar high for return without disclosure

---

## 11. Pipeline orchestration (batch)

```text
run_ingest → run_universe → run_features → run_signals → run_backtest → run_report
```

- Each stage reads/writes Parquet + DuckDB views.
- `config_hash` recorded in `runs`.
- Idempotent: re-running same config overwrites versioned output dirs, not raw.

No live scheduler required for V0.1; CLI + optional cron later.

---

## 12. Required agent outputs (maps to research §21)

| Deliverable | Contents |
|-------------|----------|
| **Dataset Report** | Sources, date range, #assets/bars per band, missingness, survivorship limits, social coverage |
| **Backtest Report** | Metrics by band × model × τ × horizon vs benchmarks; CIs; weaknesses |
| **Feature Analysis** | A→D contribution; components that add nothing → drop candidates |
| **Final Recommendation** | Evidence, confidence, limits, next steps — never binary “works/fails” alone |

---

## 13. Secrets & access

| Secret | Needed when |
|--------|-------------|
| `COINGECKO_API_KEY` | If Demo rate limits block history |
| `CMC_API_KEY` | If using CMC backup |
| Social API keys | Only when enabling Model C/D |

Store via env / secure secret request — never commit keys. No wallet keys ever.

---

## 14. V0.1 implementation order (suggested slices)

1. **Scaffold** repo layout + DuckDB + config (bands A/B locked).
2. **Ingest** CoinGecko daily OHLCV + MC for candidates.
3. **Universe** membership + draft liquidity gates.
4. **Features** C, V, P + price/volume NSI → MREI/Gap without N (Models A/B).
5. **Signals + backtest** grid + benchmarks + Dataset/Backtest reports.
6. **Social ingest** (Trends/Reddit as available) → N + Model C/D.
7. **Holdout** threshold confirmation + Feature Analysis + recommendation.

Do **not** start slice 2+ until Abdul approves this DE spec (or marks edits).

---

## 15. Open decisions (need Abdul)

1. **Entry convention:** next-day open (recommended) vs same-day close?
2. **CoinGecko:** Demo enough to start, or provide Pro key now?
3. **Liquidity draft:** start V_min grid at `$50k / $100k / $250k` median 20d USD volume?
4. **Social:** defer entirely until A/B backtest exists, or wire Google Trends in parallel?
5. **Data Eng Spec approval:** accept V0.1 as written, or edit first?

---

## 16. Explicit non-goals (again)

- No live trading bot, no order routing, no wallet connect.
- No ML / extra indicators until baseline A→D clears.
- Trader owns live tape/picks; Quant Research owns validation evidence only.
