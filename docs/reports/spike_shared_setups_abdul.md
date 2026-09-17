# Shared setups before spikes — cross-coin flags

**When:** 2026-09-17 21:01 WAT (Africa/Lagos)
**Status:** research check only — **not alpha**, not a trading system, not a live bot
This replaces the old “average indicator → hand template” step with **shared yes/no setups** that many coins showed the day before a spike.

## Short answers

- Spikes used for discovery (default ≥ +50% in 7d): **82** total; **55** earlier / **27** later (eligible assets ≈ **150**).
- Cut date (first later day): **2026-08-21**.
- Shared setups kept (lift ≥ 1.2 vs normal days, min count 5): **6**.
- Did any shared setup beat chance later for **+50%** in 7d? **Yes (weak)**.
- Did any shared setup beat chance later for **+100%** in 7d? **Yes (weak)**.

### Overall

At least one shared setup edged above the later-period base rate on one of the thresholds. That is a weak, noisy reading — **no alpha claim**, do not trade this.

## What we did (plain)

1. Built a library of **yes/no setup flags** on each coin-day (price vs EMA, trend, MACD rising, RSI bands, Bollinger mid/upper, volume heat, EMA reclaim, and Gap / N / compression when scores exist).
2. For each earlier **spike**, looked at the day **before** (T−1) and counted how often each flag (and frequent 2-flag combos) was true — that is **spike support**.
3. Compared to **normal non-spike days** in the same earlier window (control). Kept setups where spikes showed the flag **meaningfully more often** (lift ≥ 1.2, at least 5 spike hits).
4. Locked the top shared setups from **earlier data only** (no peeking at later days).
5. On the **later** period only, asked: when a setup fires, how often does the coin hit **+50%** and **+100%** within 7 days, versus the everyday base rates?

## Date split

- Membership calendar days: **92**
- **Cut date:** `2026-08-21` (discover `2026-06-18` → `2026-08-20`; validate `2026-08-21` → `2026-09-17`)
- Explore fraction: **0.7**
- Control non-spike days (earlier): **5000**

## Spike counts

| set | n spikes | unique assets |
|------|------|------|
| all | 82 | 53 |
| discover | 55 | 35 |
| validate | 27 | 25 |

- Sample sizes are **small**. Treat every percentage with caution.

## Shared setups ranked (discover only)

Support = share of earlier spike T−1 days where the setup was true. Lift = spike support ÷ control support.

| rank | setup | spike support | control support | lift | spike hits |
|------|------|------|------|------|------|
| 1 | price_above_ema50+rvol_gt_1 | 30.91% | 6.78% | 4.56x | 17/55 |
| 2 | macd_hist_pos_rising+ema20_gt_ema50 | 9.09% | 2.68% | 3.39x | 5/55 |
| 3 | rvol_gt_1_5 | 25.45% | 8.00% | 3.18x | 14/55 |
| 4 | price_above_ema20+rvol_gt_1 | 32.73% | 10.86% | 3.01x | 18/55 |
| 5 | price_above_ema50 | 41.82% | 15.06% | 2.78x | 23/55 |
| 6 | ema20_gt_ema50 | 27.27% | 12.98% | 2.10x | 15/55 |

### In plain English

1. **price_above_ema50+rvol_gt_1** (combo): Price above EMA50 AND Relative volume > 1× 30d average — true on 30.91% of earlier pre-spike days vs 6.78% of normal days (lift 4.56x).
2. **macd_hist_pos_rising+ema20_gt_ema50** (combo): MACD histogram positive and rising vs prior day AND EMA20 above EMA50 (uptrend) — true on 9.09% of earlier pre-spike days vs 2.68% of normal days (lift 3.39x).
3. **rvol_gt_1_5** (single): Relative volume > 1.5× 30d average — true on 25.45% of earlier pre-spike days vs 8.00% of normal days (lift 3.18x).
4. **price_above_ema20+rvol_gt_1** (combo): Price above EMA20 AND Relative volume > 1× 30d average — true on 32.73% of earlier pre-spike days vs 10.86% of normal days (lift 3.01x).
5. **price_above_ema50** (single): Price above EMA50 — true on 41.82% of earlier pre-spike days vs 15.06% of normal days (lift 2.78x).
6. **ema20_gt_ema50** (single): EMA20 above EMA50 (uptrend) — true on 27.27% of earlier pre-spike days vs 12.98% of normal days (lift 2.10x).

### All single flags (discover support)

| flag | spike support | control support | lift | spike hits |
|------|------|------|------|------|
| rvol_gt_1_5 | 25.45% | 8.00% | 3.18x | 14/55 |
| price_above_ema50 | 41.82% | 15.06% | 2.78x | 23/55 |
| ema20_gt_ema50 | 27.27% | 12.98% | 2.10x | 15/55 |
| rsi_cross_up_50 | 9.09% | 4.50% | 2.02x | 5/55 |
| price_above_ema20 | 41.82% | 23.72% | 1.76x | 23/55 |
| rvol_gt_1 | 41.82% | 24.48% | 1.71x | 23/55 |
| reclaim_ema20_3d | 27.27% | 17.18% | 1.59x | 15/55 |
| macd_hist_pos_rising | 40.00% | 31.76% | 1.26x | 22/55 |
| bb_bandwidth_expanding | 54.55% | 44.62% | 1.22x | 30/55 |
| gap_gt_10 | 32.73% | 28.04% | 1.17x | 18/55 |
| compression_recovering | 27.27% | 24.62% | 1.11x | 15/55 |
| macd_hist_pos | 70.91% | 66.66% | 1.06x | 39/55 |
| gap_gt_0 | 50.91% | 50.14% | 1.02x | 28/55 |
| quiet_rising_n | 27.27% | 29.40% | 0.93x | 15/55 |
| rsi_45_65 | 32.73% | 35.76% | 0.92x | 18/55 |
| rsi_40_60 | 43.64% | 54.12% | 0.81x | 24/55 |
| bb_pctb_mid_upper | 25.45% | 35.54% | 0.72x | 14/55 |

## Later check: +50% and +100% within 7d

Base rate on later eligible days: **+50%** → 3.28%; **+100%** → 1.40%.

### +50% within 7d

| setup | fires | hits | hit rate | base | lift | beats base? |
|------|------|------|------|------|------|------|
| price_above_ema50+rvol_gt_1 | 1187 | 36 | 3.03% | 3.28% | 0.92x | no |
| macd_hist_pos_rising+ema20_gt_ema50 | 477 | 11 | 2.31% | 3.28% | 0.70x | no |
| rvol_gt_1_5 | 756 | 39 | 5.16% | 3.28% | 1.57x | yes |
| price_above_ema20+rvol_gt_1 | 1292 | 45 | 3.48% | 3.28% | 1.06x | no |
| price_above_ema50 | 2435 | 66 | 2.71% | 3.28% | 0.83x | no |
| ema20_gt_ema50 | 2095 | 42 | 2.00% | 3.28% | 0.61x | no |

### +100% within 7d

| setup | fires | hits | hit rate | base | lift | beats base? |
|------|------|------|------|------|------|------|
| price_above_ema50+rvol_gt_1 | 1187 | 19 | 1.60% | 1.40% | 1.14x | yes |
| macd_hist_pos_rising+ema20_gt_ema50 | 477 | 2 | 0.42% | 1.40% | 0.30x | no |
| rvol_gt_1_5 | 756 | 18 | 2.38% | 1.40% | 1.70x | yes |
| price_above_ema20+rvol_gt_1 | 1292 | 21 | 1.63% | 1.40% | 1.16x | yes |
| price_above_ema50 | 2435 | 21 | 0.86% | 1.40% | 0.61x | no |
| ema20_gt_ema50 | 2095 | 6 | 0.29% | 1.40% | 0.20x | no |

We mark **beats base** only when the setup fired at least 5 times on the later panel **and** lift ≥ 1.10. Small samples can look lucky.

## Honesty box

- Discover spikes: **55**. Later spikes: **27**. That is a thin sample.
- Setups were chosen from earlier data only; later numbers are a one-shot check, not a license to trade.
- Short history, crypto noise, overlapping themes — easy to overfit.
- **No alpha claim.** Do not treat this as a system.
- Old +50/+100 spike-backward reports are left unchanged; this is a separate shared-setups note.

