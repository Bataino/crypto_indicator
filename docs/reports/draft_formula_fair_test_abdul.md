# Locked draft formula — fair test (Abdul)

**When:** 2026-09-18 00:07 WAT (Africa/Lagos)
**Status:** research fair-test only — **not alpha**, not a trading system, not a live bot
Abdul chose **A then B**: first fair-test this locked draft (no Bollinger); X comes later.

## Locked formula (no Bollinger)

Signal day when **BOTH** are true:
1. `rvol_30 > 1.5` (volume expansion)
2. `dist_ema_20 <= 0.05` (price not more than 5% above EMA20)

Optional soft filters (reported WITH and WITHOUT):
- RSI ≤ 70 (primary soft); RSI 45–65 band as secondary
- Gap > −10 if available — **optional**; sentiment evidence is weak

## Short answers (primary later period)

- Primary cut date: **2026-08-21** (explore_frac=0.7).
- Spikes (≥ +50% in 7d): **82** total; eligible assets ≈ **150**.

- **Does the locked combo beat chance later?** **YES** (+50% lift 2.68x on 205 fires / 18 hits; +100% lift 2.43x on 205 fires / 7 hits; need ≥5 fires and lift ≥ 1.10× base to count as beating chance).
- **Better than volume alone later (+50%)?** **YES** (locked 2.68x vs volume 1.57x).
- **Better than not-stretched alone later (+50%)?** **YES** (locked 2.68x vs not-stretched 1.04x).
- **Do soft filters help?** **YES** — soft filter(s) raised later lift vs locked core: Locked + RSI 45–65 (secondary)
  - Locked + RSI ≤ 70: same fires/hits as locked core (no change)
  - Locked + RSI 45–65 (secondary): higher later lift (3.17x on +50% vs core 2.68x)
  - Locked + Gap > −10 (optional): did not raise later lift vs core (2.57x on +50%)
  - Locked + RSI ≤ 70 + Gap > −10 (optional): did not raise later lift vs core (2.57x on +50%)

Small sample. **No alpha claim.** Do not treat this as a trading system.

## What we did (plain)

1. Locked the draft **before** looking at later numbers: volume expansion AND not more than 5% above EMA20 (no Bollinger).
2. Also measured volume-alone and not-stretched-alone so we can see if the **combo** is doing real work.
3. Reported the same locked core **with** and **without** soft filters (RSI ≤ 70; RSI 45–65 secondary; Gap > −10 optional).
4. Primary fair check uses the same cut style as prior studies (~2026-08-21, explore_frac=0.7): hit rates for +50% and +100% within 7d on the **later** period only.
5. Secondary robustness: alternate cut with explore_frac=0.6 (cut 2026-08-12).

## Candidates (locked)

| id | role | flags |
|------|------|------|
| locked_core | core | `rvol_gt_1_5` AND `dist_ema20_le_5pct` |
| volume_alone | baseline | `rvol_gt_1_5` |
| not_stretched_alone | baseline | `dist_ema20_le_5pct` |
| locked_plus_rsi70 | soft | `rvol_gt_1_5` AND `dist_ema20_le_5pct` AND `rsi_le_70` |
| locked_plus_rsi45_65 | soft_secondary | `rvol_gt_1_5` AND `dist_ema20_le_5pct` AND `rsi_45_65` |
| locked_plus_gap_m10 | soft_optional | `rvol_gt_1_5` AND `dist_ema20_le_5pct` AND `gap_gt_m10` |
| locked_plus_rsi70_gap | soft_optional | `rvol_gt_1_5` AND `dist_ema20_le_5pct` AND `rsi_le_70` AND `gap_gt_m10` |

## Primary later period (fair test)

- Explore fraction: **0.7**
- **Cut date (first later day):** `2026-08-21` (earlier `2026-06-18` → `2026-08-20`; later `2026-08-21` → `2026-09-17`)
- Earlier spikes (descriptive only): **55**
- Control non-spike days (earlier): **5000**
- Later everyday base rates: +50% → 3.28%; +100% → 1.40%

### Discover support (earlier only — not the fair test)

How often each rule was true on earlier spike T−1 days vs normal days. Descriptive only.

| rule | spike support | control support | lift | spike hits |
|------|------|------|------|------|
| Locked draft (vol + not stretched ≤5%) | 0.00% | 4.50% | 0.00x | 0/55 |
| Volume alone | 25.45% | 8.00% | 3.18x | 14/55 |
| Not-stretched alone (≤5% above EMA20) | 65.45% | 90.10% | 0.73x | 36/55 |
| Locked + RSI ≤ 70 | 0.00% | 4.50% | 0.00x | 0/55 |
| Locked + RSI 45–65 (secondary) | 0.00% | 2.12% | 0.00x | 0/55 |
| Locked + Gap > −10 (optional) | 0.00% | 3.88% | 0.00x | 0/55 |
| Locked + RSI ≤ 70 + Gap > −10 (optional) | 0.00% | 3.88% | 0.00x | 0/55 |

### Later fair check: +50% and +100% within 7d

Primary question: when the rule fires on ordinary eligible days **after** the cut, how often do we get +50% / +100% within 7 days, vs everyday chance?

### +50% within 7d (later)

| rule | fires | hits | hit rate | base | lift | beats chance? |
|------|------|------|------|------|------|------|
| Locked draft (vol + not stretched ≤5%) | 205 | 18 | 8.78% | 3.28% | 2.68x | yes |
| Volume alone | 756 | 39 | 5.16% | 3.28% | 1.57x | yes |
| Not-stretched alone (≤5% above EMA20) | 2467 | 84 | 3.40% | 3.28% | 1.04x | no |
| Locked + RSI ≤ 70 | 205 | 18 | 8.78% | 3.28% | 2.68x | yes |
| Locked + RSI 45–65 (secondary) | 154 | 16 | 10.39% | 3.28% | 3.17x | yes |
| Locked + Gap > −10 (optional) | 166 | 14 | 8.43% | 3.28% | 2.57x | yes |
| Locked + RSI ≤ 70 + Gap > −10 (optional) | 166 | 14 | 8.43% | 3.28% | 2.57x | yes |

### +100% within 7d (later)

| rule | fires | hits | hit rate | base | lift | beats chance? |
|------|------|------|------|------|------|------|
| Locked draft (vol + not stretched ≤5%) | 205 | 7 | 3.41% | 1.40% | 2.43x | yes |
| Volume alone | 756 | 18 | 2.38% | 1.40% | 1.70x | yes |
| Not-stretched alone (≤5% above EMA20) | 2467 | 40 | 1.62% | 1.40% | 1.16x | yes |
| Locked + RSI ≤ 70 | 205 | 7 | 3.41% | 1.40% | 2.43x | yes |
| Locked + RSI 45–65 (secondary) | 154 | 5 | 3.25% | 1.40% | 2.32x | yes |
| Locked + Gap > −10 (optional) | 166 | 4 | 2.41% | 1.40% | 1.72x | yes |
| Locked + RSI ≤ 70 + Gap > −10 (optional) | 166 | 4 | 2.41% | 1.40% | 1.72x | yes |

## Secondary robustness (alternate cut)

- Explore fraction: **0.6**
- **Cut date (first later day):** `2026-08-12` (earlier `2026-06-18` → `2026-08-11`; later `2026-08-12` → `2026-09-17`)
- Earlier spikes (descriptive only): **40**
- Control non-spike days (earlier): **5000**
- Later everyday base rates: +50% → 3.48%; +100% → 1.19%

### Discover support (earlier only — not the fair test)

How often each rule was true on earlier spike T−1 days vs normal days. Descriptive only.

| rule | spike support | control support | lift | spike hits |
|------|------|------|------|------|
| Locked draft (vol + not stretched ≤5%) | 0.00% | 3.86% | 0.00x | 0/40 |
| Volume alone | 27.50% | 7.32% | 3.76x | 11/40 |
| Not-stretched alone (≤5% above EMA20) | 70.00% | 91.14% | 0.77x | 28/40 |
| Locked + RSI ≤ 70 | 0.00% | 3.86% | 0.00x | 0/40 |
| Locked + RSI 45–65 (secondary) | 0.00% | 1.84% | 0.00x | 0/40 |
| Locked + Gap > −10 (optional) | 0.00% | 3.46% | 0.00x | 0/40 |
| Locked + RSI ≤ 70 + Gap > −10 (optional) | 0.00% | 3.46% | 0.00x | 0/40 |

### Later fair check: +50% and +100% within 7d

Primary question: when the rule fires on ordinary eligible days **after** the cut, how often do we get +50% / +100% within 7 days, vs everyday chance?

### +50% within 7d (later)

| rule | fires | hits | hit rate | base | lift | beats chance? |
|------|------|------|------|------|------|------|
| Locked draft (vol + not stretched ≤5%) | 286 | 26 | 9.09% | 3.48% | 2.61x | yes |
| Volume alone | 888 | 50 | 5.63% | 3.48% | 1.62x | yes |
| Not-stretched alone (≤5% above EMA20) | 3574 | 124 | 3.47% | 3.48% | 1.00x | no |
| Locked + RSI ≤ 70 | 286 | 26 | 9.09% | 3.48% | 2.61x | yes |
| Locked + RSI 45–65 (secondary) | 188 | 18 | 9.57% | 3.48% | 2.75x | yes |
| Locked + Gap > −10 (optional) | 235 | 21 | 8.94% | 3.48% | 2.57x | yes |
| Locked + RSI ≤ 70 + Gap > −10 (optional) | 235 | 21 | 8.94% | 3.48% | 2.57x | yes |

### +100% within 7d (later)

| rule | fires | hits | hit rate | base | lift | beats chance? |
|------|------|------|------|------|------|------|
| Locked draft (vol + not stretched ≤5%) | 286 | 9 | 3.15% | 1.19% | 2.64x | yes |
| Volume alone | 888 | 20 | 2.25% | 1.19% | 1.89x | yes |
| Not-stretched alone (≤5% above EMA20) | 3574 | 44 | 1.23% | 1.19% | 1.03x | no |
| Locked + RSI ≤ 70 | 286 | 9 | 3.15% | 1.19% | 2.64x | yes |
| Locked + RSI 45–65 (secondary) | 188 | 5 | 2.66% | 1.19% | 2.23x | yes |
| Locked + Gap > −10 (optional) | 235 | 6 | 2.55% | 1.19% | 2.14x | yes |
| Locked + RSI ≤ 70 + Gap > −10 (optional) | 235 | 6 | 2.55% | 1.19% | 2.14x | yes |

### Alternate cut — quick yes/no

- Locked beats chance later? **YES**
- Soft filters help vs locked core? **YES**

## Recommendation (research process, not a trade)

- Locked draft **beats everyday chance** and **volume alone** on the primary later cut for +50% — keep it as the draft to stress-test next (still not promoted).
- Soft filters that helped later lift vs core: Locked + RSI 45–65 (secondary). Keep them optional for a repeat check — do not harden yet.
- **X later** (Abdul path B) — not run in this report.
- **No alpha claim.**

## Honesty box

- Total spikes in sample: **82**. Thin history.
- Candidates and soft filters were locked before the later look; still one short crypto window and overlapping themes — easy to overfit.
- Gap soft filter depends on Rotation Gap (sentiment/narrative); labeled **optional** because that evidence has been weak in prior checks.
- **No alpha claim.** Do not treat this as a system.

