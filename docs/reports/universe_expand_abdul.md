# Universe expand (Abdul) — CoinGecko Band A∪B

**When:** 17 Sep 2026, ~18:43 WAT (Africa/Lagos)  
**Status:** research data coverage only — **not alpha**, N not redesigned

## Goal

Widen the micro-cap universe from a tiny sample (~38 membership assets) using live CoinGecko `/coins/markets` pages in Band A∪B market-cap ($5M–$100M), apply eligibility (no tokenized stocks / stables), ingest ≥90d daily history (prefer 180d), resume with skip_existing, then rebuild `universe_membership`.

## Before → after

| metric | before | after |
|--------|--------|-------|
| assets in DuckDB | ~50 | **166** |
| assets with ≥90 daily bars | ~50 | **166** |
| eligibility-passing assets | ~38 | **150** |
| membership distinct assets | **~38** | **150** |
| membership rows | (small) | **20 858** |
| Band A (`A_5_50`) assets / rows | — | **125** / 9 490 |
| Band B (`B_10_100`) assets / rows | — | **149** / 11 368 |
| market_daily rows | ~8.5k | **29 188** |
| bar length (min / median / max) | — | **92 / 180 / 180** |
| membership calendar | — | **2026-06-18 → 2026-09-17** |

Clear jump: membership **~38 → 150** eligible assets (target 100–150 met).

## How we discovered & ingested

1. `scripts/run_ingest.py --full --limit 150 --days 180` (then a short top-up after tightening denies).
2. Paged CoinGecko markets (`volume_desc`, up to 20×250) and kept MC in **$5M–$100M** (Band A∪B union).
3. **Eligibility filter applied before chart fetch** (and again at membership): tokenized stocks / stables / FX denylist in `config/eligibility.yaml`.
4. `skip_existing` + `prefer_raw` resume; Demo key throttle (~2.5s min interval); market_chart only (no OHLC) to save quota.
5. `scripts/run_universe.py` rebuilt point-in-time membership + liquidity draft.

Live discovery pool (this run): **~1 161** eligibility-passing band candidates after excluding **~96** denylisted names from the scrape (plus many more denied via substring/keyword rules on later pages).

## Exclusions (eligibility)

**16** assets remain in DuckDB but are flagged ineligible and **do not** enter membership, including prior stables/tokenized names plus a few that slipped an earlier pass and were denied on rebuild:

- Stables / FX-like: `usd-coinvertible`, `eurite`, `f-x-protocol-fxusd`, `royal-euro`, `unitas`, `fidelity-digital-dollar`, `jupusd`, `straitsx-xusd`, `metronome-synth-usd`, `jpycoin`
- Tokenized stocks / wrappers: `nvidia-bstocks`, `spacex-bstocks-tokenized-stock`, `alphabet-class-a-ondo-tokenized-stock`, `sandisk-bstocks-tokenized-stock`, `circle-xstock`, `circle-internet-group-bstock`

Denylist was also extended for **robinhood-token**, **-rstock**, backpack-securities, prestocks, pre-ipo, tokenised-*, synth-usd / xusd so future scrapes skip them before fetch.

## Blockers / limits

- **Demo CoinGecko key + rate limits:** polite throttle; stopped cleanly at `--limit 150` eligible-with-history (not a hard API outage this run).
- **Short listing age:** many top-volume micro-caps fail `min_history_days=90` (skipped; pool oversampled via `--full`).
- **Survivorship bias:** live `/coins/markets` only sees coins CoinGecko still lists; delisted/rugged names are missing (documented in ingest module).
- **Point-in-time MC bands** for membership use stored daily MC; discovery itself is a **recent snapshot** filter for who to ingest.
- Ineligible rows already in DuckDB still count toward storage but not toward `--limit` eligible slots / membership.

## Code / config touched (this slice)

- `src/cmram/ingest/market.py` — load secrets safely; filter candidates by eligibility before fetch; count only eligible ≥90d assets toward `--limit`.
- `config/eligibility.yaml` — expanded deny patterns (robinhood / rstock / backpack / prestocks / a few stable ids).

## Tests

`pytest`: **115 passed, 1 skipped** (green after expand).

## Optional features / holdout

Re-ran **short** `config/features.yaml` on the wider membership (`n=150` assets; Band A/B n_in_band often 25–139, `small_sample` flags cleared vs the old ~38 set). Signals regenerated. Full backtest/holdout on the expanded signal grid was **deferred** after ~20+ min CPU (combinatorial cost); dig into indicators later. **No alpha claim.**

