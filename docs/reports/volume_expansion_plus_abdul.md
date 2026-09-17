# Volume expansion (+ second condition) — Abdul check

**When:** 2026-09-17 21:11 WAT (Africa/Lagos)
**Status:** research check only — **not alpha**, not a trading system, not a live bot
Abdul's read: **volume expansion** matters; compression may be the wrong lead. Data often shows hot volume with price already ticking up — rarely hot volume with flat/down price.

## Short answers

- Spikes (≥ +50% in 7d): **82** total; **55** earlier / **27** later (eligible assets ≈ **150**).
- Cut date (first later day): **2026-08-21**.
- Core flag: **rvol_30 > 1.5** (same as shared-setup `rvol_gt_1_5`).
- **Which second condition helps?** **Best later add-on: A′: not already moon'd (≤5% above EMA20)** (+50% lift 2.68x vs volume-alone 1.57x); also above volume-alone: B: reclaiming / near EMA20 (−5%…+10%), D: RSI 45–65 (not overbought blowoff), A: not already moon'd (≤10% above EMA20). **C (MACD rising) did not improve** vs volume-alone. Small sample — **no alpha claim**.

- On spike T−1 days with hot volume (rvol>1.5): **100.00%** already had **ret_3d > 0** (price ticking up); **0.00%** had ret_3d ≤ 0 (n=20 hot-vol T−1 days with ret_3d). That supports Abdul's read that hot volume rarely sits with flat price.

### Overall

**Best later add-on: A′: not already moon'd (≤5% above EMA20)** (+50% lift 2.68x vs volume-alone 1.57x); also above volume-alone: B: reclaiming / near EMA20 (−5%…+10%), D: RSI 45–65 (not overbought blowoff), A: not already moon'd (≤10% above EMA20). **C (MACD rising) did not improve** vs volume-alone. Small sample — **no alpha claim**.

Volume-alone later lift: +50% → 1.57x; +100% → 1.70x (base rates 3.28% / 1.40%).

**Model note:** consider reframing Model C toward **volume expansion** in future design. **Do not rip out compression code yet** — keep it available until a volume-led redesign is specified and re-tested.

## What we did (plain)

1. Locked **volume expansion** as the core: `rvol_30 > 1.5`.
2. Pre-registered **one second condition at a time** (A ≤10% / A′ ≤5% above EMA20; B near EMA20 −5%…+10%; C MACD hist >0 rising; D RSI 45–65), plus volume-alone baseline — no fishing for extra combos.
3. On **earlier** spikes only, measured how often each candidate was true at T−1 vs normal days (descriptive support / lift).
4. On the **later** period only (same cut ≈ 2026-08-21), measured hit rates for **+50%** and **+100%** within 7d when each candidate fires, vs everyday base rates and vs volume-alone lift.
5. On spike T−1 days, counted hot-volume rows with **ret_3d ≤ 0** vs **ret_3d > 0** (Abdul's flat-vs-ticking-up question).

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

## Abdul check: hot volume vs ret_3d on spike T−1

Question: when volume is already hot the day before a spike, is price usually flat/down (`ret_3d ≤ 0`) or already ticking up (`ret_3d > 0`)?

| set | hot-vol T−1 (with ret_3d) | ret_3d > 0 | ret_3d ≤ 0 | median ret_3d (hot) |
|------|------|------|------|------|
| all spikes | 20 / hot=20 (of 82 T−1) | 100.00% (20) | 0.00% (0) | 54.83% |
| discover | 14 / hot=14 (of 55 T−1) | 100.00% (14) | 0.00% (0) | 61.20% |
| validate | 6 / hot=6 (of 27 T−1) | 100.00% (6) | 0.00% (0) | 21.39% |

For context, among **all** spike T−1 days with ret_3d (n=82): share ret_3d>0 = 50.00%, median ret_3d = 0.15%.

## Candidates (locked)

| id | label | flags |
|------|------|------|
| baseline_rvol | Baseline: volume alone | `rvol_gt_1_5` |
| A_not_mooned_10 | A: not already moon'd (≤10% above EMA20) | `rvol_gt_1_5` AND `dist_ema20_le_10pct` |
| A_not_mooned_5 | A′: not already moon'd (≤5% above EMA20) | `rvol_gt_1_5` AND `dist_ema20_le_5pct` |
| B_near_ema20 | B: reclaiming / near EMA20 (−5%…+10%) | `rvol_gt_1_5` AND `near_ema20_m5_p10` |
| C_macd_rising | C: MACD hist > 0 and rising | `rvol_gt_1_5` AND `macd_hist_pos_rising` |
| D_rsi_45_65 | D: RSI 45–65 (not overbought blowoff) | `rvol_gt_1_5` AND `rsi_45_65` |

## Discover support (earlier spikes only)

Support = share of earlier spike T−1 days where the candidate was true. Lift = spike support ÷ control support. Descriptive only — not validation.

| candidate | spike support | control support | lift | spike hits |
|------|------|------|------|------|
| baseline_rvol | 25.45% | 8.00% | 3.18x | 14/55 |
| A_not_mooned_10 | 3.64% | 5.46% | 0.67x | 2/55 |
| A_not_mooned_5 | 0.00% | 4.50% | 0.00x | 0/55 |
| B_near_ema20 | 3.64% | 3.92% | 0.93x | 2/55 |
| C_macd_rising | 25.45% | 5.04% | 5.05x | 14/55 |
| D_rsi_45_65 | 7.27% | 4.30% | 1.69x | 4/55 |

**Tension to note:** on earlier *spike T−1* days, hot volume often already sat with price well above EMA20 (see Abdul ret_3d check — median hot-vol ret_3d was large). So A/B (“not extended / near EMA”) rarely co-occurred with hot volume on those historical pre-spike days (low discover support). The **later** panel instead asks: when the combo fires on ordinary eligible days, does the hit rate improve? Those are different questions.

## Later check: +50% and +100% within 7d

Base rate on later eligible days: **+50%** → 3.28%; **+100%** → 1.40%.

Compare each second condition's **lift** to **baseline_rvol**. We mark **beats base** when fires ≥ 5 and lift ≥ 1.10 vs the everyday base rate (same rule as shared setups).

### +50% within 7d

| candidate | fires | hits | hit rate | base | lift | vs vol-alone | beats base? |
|------|------|------|------|------|------|------|------|
| baseline_rvol | 756 | 39 | 5.16% | 3.28% | 1.57x | — | yes |
| A_not_mooned_10 | 374 | 23 | 6.15% | 3.28% | 1.87x | +0.30x lift | yes |
| A_not_mooned_5 | 205 | 18 | 8.78% | 3.28% | 2.68x | +1.10x lift | yes |
| B_near_ema20 | 331 | 21 | 6.34% | 3.28% | 1.93x | +0.36x lift | yes |
| C_macd_rising | 429 | 22 | 5.13% | 3.28% | 1.56x | -0.01x lift | yes |
| D_rsi_45_65 | 400 | 25 | 6.25% | 3.28% | 1.91x | +0.33x lift | yes |

### +100% within 7d

| candidate | fires | hits | hit rate | base | lift | vs vol-alone | beats base? |
|------|------|------|------|------|------|------|------|
| baseline_rvol | 756 | 18 | 2.38% | 1.40% | 1.70x | — | yes |
| A_not_mooned_10 | 374 | 11 | 2.94% | 1.40% | 2.10x | +0.40x lift | yes |
| A_not_mooned_5 | 205 | 7 | 3.41% | 1.40% | 2.43x | +0.74x lift | yes |
| B_near_ema20 | 331 | 9 | 2.72% | 1.40% | 1.94x | +0.24x lift | yes |
| C_macd_rising | 429 | 10 | 2.33% | 1.40% | 1.66x | -0.04x lift | yes |
| D_rsi_45_65 | 400 | 10 | 2.50% | 1.40% | 1.78x | +0.08x lift | yes |

## Recommendation (research process, not a trade)

- Treat **volume expansion (`rvol_30 > 1.5`)** as the primary lead to stress-test next — consistent with shared-setups later lift and Abdul's read.
- Strongest later add-on in this cut: **A′: not already moon'd (≤5% above EMA20)**. Keep for a **repeat check** on a longer window / more spikes — do not promote yet.
- **C (MACD hist rising)** did not beat volume-alone later; do not treat it as a required second gate on this evidence.
- **Future Model C design:** consider centering on volume expansion (and optionally a light “not already extended” gate) rather than compression-first. **Leave compression code in place** for now.

## Honesty box

- Discover spikes: **55**. Later spikes: **27**. Thin sample.
- Candidates were locked before the later check; still one short history and overlapping crypto themes — easy to overfit.
- Hot-vol × ret_3d is **descriptive** on days that already preceded spikes (selected on the outcome). It does not prove causality.
- **No alpha claim.** Do not treat this as a system.
- Compression feature code is unchanged; this report only recommends a future redesign discussion.

