# CMRAM Phase 1 — Interim Findings

**Date:** 17 September 2026 (WAT)  
**Status:** engineering research only — **not a trading system**

## Goal

CMRAM is testing whether simple, explainable signals can help us study rotation in smaller crypto assets. The main question is whether a rise in public attention can add useful information before or alongside price and volume changes.

We are not assuming that the idea works. We are building the measurement pipeline first, then checking it on data that was not used to choose the settings.

## What Phase 1 is

Phase 1 is the first end-to-end research pass:

- collect daily market data;
- define a point-in-time universe and two market-cap bands;
- calculate our own price, volume, compression, momentum, and narrative features;
- create transparent signals;
- measure forward returns; and
- run a time-based holdout rather than relying only on a full-sample backtest.

The current sample has 50 ingested assets, 38 assets passing the eligibility filter, and 91 calendar days from 18 June through 16 September 2026. This is still a small and short sample. The market data is chart-close-only, so the intended next-day-open entry is currently represented by the next bar's close. That weakens the return and path measurements.

## N is the main hypothesis

**N** means Narrative Acceleration: a research proxy for whether attention is rising and speeding up. In V0.1, it is built from attention levels, 7-day and 28-day changes, acceleration, and a low-base adjustment, then ranked within each band and day.

N is the main hypothesis. Compression, volume, and price strength (C/V/P) are supporting inputs. This priority matters: we want to learn whether attention adds information, not quietly treat it as an optional decoration on a price model.

## What we built

- A daily CoinGecko market-data ingest and DuckDB store.
- Eligibility and point-in-time membership for Bands A ($5M–$50M market cap) and B ($10M–$100M).
- Model A using price features; Model B using compression, volume, and price; and Model C adding N when N is available.
- MREI, NSI, and Rotation Gap features, followed by transparent threshold-based signals.
- Forward-return checks at 1, 3, 7, 14, 21, and 30 days, with simple same-band comparison baselines.
- An explore/holdout split: the first 70% of membership days for exploration and the last 30% for the pre-registered check.

The current run produced 7,008 signal rows and 144,260 signal-by-horizon result rows. These are engineering outputs, not evidence of a usable edge.

## A/B without N failed the holdout

Models A and B do not use N. On the 7-day holdout comparison against the same-band random baseline:

- pre-registered Model A (thresholds 50/20): median excess **-3.89%** across 208 signals;
- pre-registered Model B (thresholds 50/20): median excess **-8.57%** across 110 signals.

The setting that looked best during the exploration peek (Model B, thresholds 70/20) also went the wrong way in holdout: median excess **-23.26%** across only 19 signals.

So A/B did not repeat their exploration direction out of sample. This is a failed holdout for this Phase 1 check. It does not prove the broader idea is impossible; the sample is small, the history is short, and the entry data is a close proxy. It does mean we should not treat the exploratory results as validation.

## N V0.1 from Trends (+ wiki)

The first N run used Google Trends daily interest and mapped Wikipedia pageviews where an article title was available. Coverage was:

- 31 of 38 eligible assets had at least one N source;
- Trends covered 29 assets;
- Wikipedia covered 6 assets;
- Reddit covered 0 assets; and
- X covered 0 assets because the required access was not available.

N was available on 3,572 of 5,198 feature rows across the three models. Missing N was not filled in with a made-up value. This is useful pipeline coverage, but it is not proof that the proxy measures real participation or predicts returns.

I added disambiguating Trends queries for the seven assets that were missing N in `config/social.yaml`: `flock-2`, `g-token`, `lagrange`, `metal`, `sonic-3`, `sophon`, and `vethor-token`. The social-ingest CLI has an asset-count limit but no per-ID filter, so I did **not** re-run a broad ingest just to refresh these seven. The suggested queries are recorded in the config for the next targeted run.

## Model C early look is not conclusive

Model C is the first check of the main N hypothesis inside the full feature/signal pipeline. Its early output is mixed across bands, thresholds, and horizons, and N is only present for part of the sample. The existing backtest is not a clean, pre-registered Model C holdout, so it cannot tell us whether N improves A/B out of sample.

Treat this as a wiring and sanity check: N flows into Model C when available, missing values remain visible, and the resulting signals can be backtested. It is not an alpha result and does not justify selecting Model C for trading.

## Reddit and X are still needed

The V0.1 N input is incomplete. Reddit public endpoints are often blocked from this environment, and X requires a bearer token. The next social run should add those sources without printing or storing secrets in the repository. Their coverage should be checked separately from Trends and wiki so that a result is not attributed to a source that did not run.

## Do not trade

Do not trade this system, use these signals for orders, or describe any result here as alpha. The current evidence is exploratory, has small-sample and data-quality limitations, and has a failed A/B holdout.

## Next steps

1. Add a safe per-asset filter to the social-ingest CLI, then run the seven new Trends mappings only and verify coverage.
2. Run Reddit with the approved credentials or from a network where the public endpoint works; add X with its bearer credential. Never print API keys.
3. Recompute features, signals, and backtests after the social refresh, with a separately tracked N-source ablation.
4. Use a fresh, pre-registered Model C holdout. Do not tune thresholds on that period.
5. Replace chart-close proxies with true OHLC, expand the point-in-time universe and history, and account for delisted assets before making stronger claims.
6. Keep the output as research evidence until results are repeatable across time and data sources. No alpha claim is warranted in Phase 1.
