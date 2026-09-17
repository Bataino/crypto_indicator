# Spike backward look — what came before +50% moves

**When:** 2026-09-17 20:19 WAT (Africa/Lagos)
**Status:** research check only — **not alpha**, not a trading system, not a live bot
**Spike (locked):** forward max return ≥ **+50%** within **7 days** (daily closes). Also report **14 days**.
Overlapping spikes on the same coin are deduped (keep the first / non-overlapping window).

## Short answers

- How many spikes found? **82** (eligible assets ≈ **150**).
- Discover (earlier) / validate (later): **55** / **27**.
- Cut date (first validate day): **2026-08-21** (explore ≈ 70% of membership days).
- Did any rule beat chance on later data? **Yes (weak check only)** — R3_macd_hist_pos_bb_mid, R4_gap_and_near_ema20. Still **not** a claim of alpha.

### Overall

Some rules edged above the later-period base rate (lift about 1.1–1.2x). That is a weak signal in a noisy sample — exploratory only, **no alpha claim**, do not trade this.

## What we did

1. Looked at every eligible coin-day in the universe (~150 names).
2. Marked a **spike start day T** when the best close in the next 7 days was at least +50% above the close on T.
3. Removed overlapping spikes on the same coin (kept the earliest).
4. On the day **before** the spike (T−1), snapped common indicators (EMA, MACD, RSI, Bollinger, volume vs 30d, plus MREI/NSI/Gap/scores if present).
5. Used **earlier** spikes to see what looked different vs random non-spike days, and wrote 2–4 simple rules in plain English.
6. Tested those rules **only** on the **later** period: do days that match a rule hit +50% more often than the normal (base) rate?

## Date split

- Membership calendar days: **92**
- **Cut date:** `2026-08-21` (discover `2026-06-18` → `2026-08-20`; validate `2026-08-21` → `2026-09-17`)
- Explore fraction: **0.7**

## Spike counts

| set | n spikes | unique assets |
|------|------|------|
| all | 82 | 53 |
| discover | 55 | 35 |
| validate | 27 | 25 |

- Median forward max return at 7d among spikes: **62.77%**
- Median forward max return at 14d among spikes: **85.20%**

## What looked different before spikes (discover only)

Compared pre-spike (T−1) indicator medians vs a random control set of non-spike days in the earlier period (n_control=165).

| indicator | spike median | control median | difference |
|------|------|------|------|
| ema_20 | 0.04 | 0.04 | -0.00 |
| ema_50 | 0.03 | 0.04 | -0.01 |
| dist_ema_20 | -0.05 | -0.05 | -0.00 |
| dist_ema_50 | -0.10 | -0.12 | 0.02 |
| macd_line | -0.00 | -0.00 | 0.00 |
| macd_signal | -0.00 | -0.00 | 0.00 |
| macd_hist | 0.00 | 0.00 | 0.00 |
| rsi_14 | 45.18 | 41.44 | 3.75 |
| bb_pct_b | 0.38 | 0.31 | 0.07 |
| bb_bandwidth | 0.44 | 0.24 | 0.20 |
| rvol_30 | 0.93 | 0.76 | 0.17 |
| days_since_local_low | 8.00 | 5.00 | 3.00 |
| drawdown_60d | -0.38 | -0.36 | -0.02 |
| MREI | 54.58 | 47.52 | 7.06 |
| NSI | 50.94 | 47.18 | 3.77 |
| Rotation_Gap | 3.69 | 0.27 | 3.42 |
| score_C | 43.36 | 49.60 | -6.24 |
| score_V | 51.83 | 45.18 | 6.65 |
| score_N | 51.96 | 46.89 | 5.07 |
| score_P | 50.39 | 46.02 | 4.37 |

### Plain takeaways (discover)

- **MREI** tended to be **higher** before spikes (spike median 54.58 vs control 47.52).
- **score_V** tended to be **higher** before spikes (spike median 51.83 vs control 45.18).
- **score_C** tended to be **lower** before spikes (spike median 43.36 vs control 49.60).
- **score_N** tended to be **higher** before spikes (spike median 51.96 vs control 46.89).
- **score_P** tended to be **higher** before spikes (spike median 50.39 vs control 46.02).

## Pattern rules (written from discover only)

1. **R1_rsi_mid_above_ema20:** RSI between 31 and 64 AND price above EMA20 (dist_ema_20 > 0)
2. **R2_rvol_elevated_not_crashed:** Volume vs 30d average between 0.80x and 1.86x AND drawdown from 60d high milder than -40%
3. **R3_macd_hist_pos_bb_mid:** MACD histogram > 0 AND Bollinger %b between 0.40 and 1.05 (price in upper half of band, not a blow-off yet)
4. **R4_gap_and_near_ema20:** Rotation Gap >= 2.0 AND price not far below EMA20 (dist_ema_20 > -5%)

These rules were **not** tuned on the later period.

## Forward check (later period only)

Base rate on later eligible days (share that hit +50% within 7d): **3.28%**.

| rule | fires | hits | hit rate | base rate | lift | beats base? |
|------|------|------|------|------|------|------|
| R1_rsi_mid_above_ema20 | 1891 | 61 | 3.23% | 3.28% | 0.98x | no |
| R2_rvol_elevated_not_crashed | 1832 | 29 | 1.58% | 3.28% | 0.48x | no |
| R3_macd_hist_pos_bb_mid | 1974 | 77 | 3.90% | 3.28% | 1.19x | yes |
| R4_gap_and_near_ema20 | 1558 | 57 | 3.66% | 3.28% | 1.12x | yes |

Lift > 1 means the rule's hit rate is higher than the base rate. We only mark **beats base** when the rule fired at least 5 times **and** lift is at least 1.10 (10% above chance). Tiny lifts can be noise.

## Honesty box

- This is **backward pattern hunting** plus a one-shot later check.
- Small sample, short history, crypto noise — easy to overfit.
- **No alpha claim.** Do not trade this as a system.
- Spike events parquet is saved under `data/spikes/` for audit.

