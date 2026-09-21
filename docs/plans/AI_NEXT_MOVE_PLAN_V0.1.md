# AI Next-Move Plan V0.1 (Abdul)

**Status:** planning only — research, not live trading  
**Date:** 2026-09-21 (Africa/Lagos)  
**Model:** LightGBM (backup: sklearn logistic regression)  
**Goal:** Learn odds of the next coin move from ≥100K micro-cap data points ($1M–$100M), then fair-test on later dates.

---

## Plain goal

1. Build a big table of micro-cap coins ($1M–$100M).
2. For each coin-day, record “what happened next” (the label).
3. Train LightGBM on **earlier** dates only.
4. Test on **later** dates we did not peek at.
5. Report in plain words: better than chance or not. **No alpha claim. No orders.**

---

## What we have today

| Item | Now |
|------|-----|
| Coins | ~166 |
| Daily price rows | ~29K |
| Features rows | ~83K |
| Date span | ~2026-03-21 → 2026-09-17 |
| RAM free (box) | ~2–3 Gi typical |

Need: grow toward **≥100K training rows** in the $1M–$100M band.

Note: “100K+” means **coin-days (rows)**, not 100K coins alive at once. Live $1M–$100M names are usually thousands; history across many coins makes the rows.

---

## Phase A — Grow the data (biggest job)

1. **Universe band C:** market cap **$1M – $100M** (point-in-time), keep deny-list (stables / tokenized stocks).
2. **History:** prefer CoinGecko (existing pipeline); rate-limit hard; resume-friendly ingest.
3. **Stop when:** ≥100K `market_daily` rows for eligible C-band days (or clear wall: API limit / days of pull).
4. **Usage caps:**
   - No X/Twitter until credits exist.
   - No Santiment required for v1 AI (price/volume first).
   - Sleep between CoinGecko calls; nightly batches OK.
5. **Survivorship:** keep delisted if API allows; mark missing days honestly.

**Exit A:** row count report + coin count + date span in plain English.

---

## Phase B — Labels (what “next move” means)

Lock **one primary** label before training:

| ID | Meaning | Primary? |
|----|---------|----------|
| `fwd_ret_7d` | % price change over next 7 days | input feature target continuous |
| `up_7d` | close in 7d > today (binary) | **primary classify** |
| `spike_50_7d` | ≥ +50% within 7d (binary) | secondary (rarer) |
| `spike_100_7d` | ≥ +100% within 7d | report only |

Primary for v1: **`up_7d`** (simple “did it go up?”).  
Also report **`spike_50_7d`** so we stay tied to earlier spike work.

Point-in-time: labels use only future prices after the signal day; features use only past/present.

---

## Phase C — Features (inputs the model sees)

**v1 core (cheap, already partly built):**

- Volume expansion: `rvol_30` (and maybe 7/14)
- Stretch: distance to EMA20 / EMA50
- RSI (14)
- Recent returns (1d, 3d, 7d)
- Volatility (e.g. 14d realized)
- Market-cap bucket / log MC (point-in-time)

**Include as optional columns (weak so far, not required):**

- Gap / MREI / NSI / Santiment N — off by default for v1 train

**Baseline to beat (not AI):** locked draft rule  
`rvol_30 > 1.5` AND `dist_ema20 <= 5%`  
(fair-test already showed exploratory lift on small sample).

---

## Phase D — Train / fair test

1. **Time split:** earlier = train (+ light tune); later = final test only. Same spirit as cut ~70% explore (exact cut date printed once after data grows).
2. **Model:** LightGBM classifier → probability of `up_7d`.
3. **Caps:**
   - Max trees / depth small (fit in ~2–4 Gi RAM)
   - No deep nets, no GPT
   - One train run + one locked later score (avoid endless retuning)
4. **Metrics (plain):**
   - Accuracy / AUC on later period
   - Hit rate when model says “high chance” vs everyday chance
   - Compare vs locked draft formula and vs random
5. **Pass bar (research, not trading):** later period beat chance **and** beat the locked draft on the same later window, with enough fires. Still **not** “alpha.”

---

## Phase E — Report + handoff

- Short Abdul report (almost no jargon).
- Push code + report to `Bataino/crypto_indicator` when ready.
- Trader gets findings only if user asks — Quant Research does not place orders or connect wallets.

---

## Order of work (do in order)

1. Write/lock this plan (this doc) ✓  
2. Add band C config + widen ingest toward 100K rows  
3. Build label + feature matrix script  
4. Install LightGBM in venv; smoke train on current small data  
5. Full train after 100K rows  
6. Fair later test + plain report  

---

## Explicit non-goals (v1)

- Live signals / bots / wallets  
- Telegram / Discord scraping  
- Paid X until Abdul adds credits  
- Complex neural nets  
- Claiming the model “knows” the next move  

---

## Open knobs (defaults locked unless Abdul changes)

| Knob | Default |
|------|---------|
| MC band | $1M – $100M |
| Row target | ≥100K market days |
| Primary label | up in 7 days |
| Model | LightGBM |
| Social in v1 | off |
| Compare vs | locked vol + EMA stretch rule |
