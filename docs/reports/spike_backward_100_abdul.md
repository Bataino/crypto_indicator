# Spike backward look — what came before +100% moves

**When:** 2026-09-17 20:32 WAT (Africa/Lagos)
**Status:** research check only — **not alpha**, not a trading system, not a live bot
**Spike (locked):** forward max return ≥ **+100%** within **7 days** (daily closes). Also report **14 days**.
Overlapping spikes on the same coin are deduped (keep the first / non-overlapping window).

## Short answers

- How many spikes found? **28** (eligible assets ≈ **150**).
- Discover (earlier) / validate (later): **18** / **10**.
- Cut date (first validate day): **2026-08-21** (explore ≈ 70% of membership days).
- Did any rule beat chance on later data? **No** (or not with enough fires).

### Overall

No proposed rule clearly beat chance on the later period. Do **not** claim a predictive pattern.

## What we did

1. Looked at every eligible coin-day in the universe (~150 names).
2. Marked a **spike start day T** when the best close in the next 7 days was at least +100% above the close on T.
3. Removed overlapping spikes on the same coin (kept the earliest).
4. On the day **before** the spike (T−1), snapped common indicators (EMA, MACD, RSI, Bollinger, volume vs 30d, plus MREI/NSI/Gap/scores if present).
5. Used **earlier** spikes to see what looked different vs random non-spike days, and wrote 2–4 simple rules in plain English.
6. Tested those rules **only** on the **later** period: do days that match a rule hit +100% more often than the normal (base) rate?

## Date split

- Membership calendar days: **92**
- **Cut date:** `2026-08-21` (discover `2026-06-18` → `2026-08-20`; validate `2026-08-21` → `2026-09-17`)
- Explore fraction: **0.7**

## Spike counts

| set | n spikes | unique assets |
|------|------|------|
| all | 28 | 21 |
| discover | 18 | 13 |
| validate | 10 | 10 |

- Median forward max return at 7d among spikes: **124.14%**
- Median forward max return at 14d among spikes: **206.43%**

## What looked different before spikes (discover only)

Compared pre-spike (T−1) indicator medians vs a random control set of non-spike days in the earlier period (n_control=54).

| indicator | spike median | control median | difference |
|------|------|------|------|
| ema_20 | 0.04 | 0.02 | 0.02 |
| ema_50 | 0.04 | 0.02 | 0.02 |
| dist_ema_20 | -0.04 | -0.05 | 0.00 |
| dist_ema_50 | -0.15 | -0.12 | -0.03 |
| macd_line | -0.00 | -0.00 | 0.00 |
| macd_signal | -0.00 | -0.00 | 0.00 |
| macd_hist | 0.00 | 0.00 | -0.00 |
| rsi_14 | 45.25 | 40.56 | 4.69 |
| bb_pct_b | 0.43 | 0.32 | 0.10 |
| bb_bandwidth | 0.41 | 0.20 | 0.21 |
| rvol_30 | 0.67 | 0.81 | -0.14 |
| days_since_local_low | 6.50 | 8.00 | -1.50 |
| drawdown_60d | -0.39 | -0.35 | -0.04 |
| MREI | 45.96 | 49.02 | -3.06 |
| NSI | 48.09 | 49.76 | -1.67 |
| Rotation_Gap | 6.24 | 0.92 | 5.33 |
| score_C | 55.41 | 49.43 | 5.98 |
| score_V | 47.59 | 50.34 | -2.75 |
| score_N | 57.90 | 39.56 | 18.35 |
| score_P | 40.20 | 49.63 | -9.43 |

### Plain takeaways (discover)

- **score_N** tended to be **higher** before spikes (spike median 57.90 vs control 39.56).
- **score_P** tended to be **lower** before spikes (spike median 40.20 vs control 49.63).
- **score_C** tended to be **higher** before spikes (spike median 55.41 vs control 49.43).
- **Rotation_Gap** tended to be **higher** before spikes (spike median 6.24 vs control 0.92).
- **rsi_14** tended to be **higher** before spikes (spike median 45.25 vs control 40.56).

## Pattern rules (written from discover only)

1. **R1_rsi_mid_above_ema20:** RSI between 30 and 63 AND price above EMA20 (dist_ema_20 > 0)
2. **R2_rvol_elevated_not_crashed:** Volume vs 30d average between 0.80x and 1.35x AND drawdown from 60d high milder than -40%
3. **R3_macd_hist_pos_bb_mid:** MACD histogram > 0 AND Bollinger %b between 0.40 and 1.05 (price in upper half of band, not a blow-off yet)
4. **R4_gap_and_near_ema20:** Rotation Gap >= 3.6 AND price not far below EMA20 (dist_ema_20 > -5%)

These rules were **not** tuned on the later period.

## Forward check (later period only)

Base rate on later eligible days (share that hit +100% within 7d): **1.40%**.

| rule | fires | hits | hit rate | base rate | lift | beats base? |
|------|------|------|------|------|------|------|
| R1_rsi_mid_above_ema20 | 1792 | 20 | 1.12% | 1.40% | 0.80x | no |
| R2_rvol_elevated_not_crashed | 1427 | 7 | 0.49% | 1.40% | 0.35x | no |
| R3_macd_hist_pos_bb_mid | 1974 | 27 | 1.37% | 1.40% | 0.98x | no |
| R4_gap_and_near_ema20 | 1447 | 21 | 1.45% | 1.40% | 1.03x | no |

Lift > 1 means the rule's hit rate is higher than the base rate. We only mark **beats base** when the rule fired at least 5 times **and** lift is at least 1.10 (10% above chance). Tiny lifts can be noise.


## Compared with the +50% study (plain words)

Same method, same cut date (`2026-08-21`), same helpers — only the spike bar changed from **+50%** to **+100%** in 7 days.

| | +50% study | +100% study |
|------|------|------|
| Spikes found | 82 | **28** |
| Discover / validate spikes | 55 / 27 | **18 / 10** |
| Later-period base rate | ~3.3% of days | **~1.4%** of days |
| Any rule beat later chance? | Yes (weak) — R3, R4 | **No** |

Takeaways in plain words:

- Doubling the bar cuts the event count a lot (82 → 28). The later sample is small (10 validate spikes).
- At +50%, two rules edged a bit above chance on later days (about 1.1–1.2× lift). At +100%, **none** cleared the “beats base” bar (R4 was closest at ~1.03× — noise).
- Discover-side hints still show higher **Rotation Gap** and mid **RSI** before big moves, but with fewer events those hints are less trustworthy.
- Harder target + smaller sample → **weaker**, not stronger, evidence for simple pre-spike rules. Still **not alpha**.

## Honesty box

- This is **backward pattern hunting** plus a one-shot later check.
- Small sample, short history, crypto noise — easy to overfit.
- **No alpha claim.** Do not trade this as a system.
- Spike events parquet is saved under `data/spikes_100/` for audit.

