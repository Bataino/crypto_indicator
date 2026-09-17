# N-only (Model E) validation — try to disprove the sign

**When:** 2026-09-17 19:38 WAT (Africa/Lagos)
**Status:** research check only — **not alpha**, not a trading system
**Primary cell:** Model **E** (quiet→rising N only), τ_m=**50**, τ_g=**20** (pre-registered; no retuning on holdout)
**Primary peers:** same-band **group_median** and **group_mean** (all eligible members on the same signal dates)
**Secondary peer:** **random** averaged over fixed seeds 42…51 (10 seeds) — do not hang the claim on one draw

## Short answers

### Does N-only beat the whole group on later data?

**No.** Pooled holdout median excess vs **group_median** at h=7 is -0.20% (n=209). N-only does **not** beat the whole same-band group on later data at the primary peer. vs group_mean=-3.20%; vs multi-seed random (secondary)=-2.16%.

### Does it work in both bands?

**No — does not work cleanly in both bands.** A_5_50: -0.24% (n=86); B_10_100: -0.18% (n=123).

### Does it work beyond 7 days?

**No clear extension beyond 7 days.** h=14: -0.10% (n=132); h=21: -3.69% (n=50); h=30: n/a (n=0). Short: h=1: -0.02% (n=280); h=3: 0.00% (n=255).

### Does it survive removing the best coin?

**Moot / no supportive sign to protect.** Full holdout median excess vs group_median is already -0.20%. Dropping the strongest positive contributor `pha` makes it -0.34% (n_remaining=180). Per-asset table shows concentration in a few names — but the main failure is the group comparison itself, not only single-coin domination of a *positive* edge.

### Overall

**Fails validation as stated.** The earlier narrow vs-random holdout sign does **not** hold up under group peers / both bands / LOO / multi-horizon checks. Say so directly: **do not claim N-only works.**

Secondary alternate cut (explore_frac=60%): pooled h=7 vs group_median median_excess=0.08% (n=634) — **secondary only**, not primary.

### Reconcile with earlier Band A vs-random +0.71%

The prior quiet→rising note reported Model E 50/20 Band A holdout **median excess ≈ +0.71% vs a single random draw** (h=7, n=86, 14 assets). On this re-check, the **same Band A cell** has median ret ≈ −1.22% (n=86) but:

- vs **group_median** (primary): **−0.24%** — does not beat the whole eligible group
- vs **group_mean** (primary): **−3.23%**
- vs **multi-seed random** (10 seeds 42…51, secondary): **−2.15%** — the earlier positive vs one RNG draw does **not** survive seed averaging

Mean excess vs group_median can look positive (skew from a few huge paths like `helium` / `unifai-network`); **median** is the pre-registered peek metric and is negative. **No alpha claim.**

## Design (what we did)

- Evaluate **Model E only** at the pre-registered **50/20** cell — no τ search on holdout.
- Bands **A and B** separately and pooled: `A_5_50, B_10_100`.
- Horizons: [1, 3, 7, 14, 21, 30] (focus narrative at h=7).
- Same later holdout window as the prior quiet→rising note idea: first 70% of membership calendar days = explore (side-by-side only), remainder = holdout.
- Secondary robustness cut: explore_frac=0.6 (labeled **secondary**).
- Concentration: per-asset n + median 7d return; leave-one-asset-out on holdout.

## Date split

- Membership calendar days: **92**
- **Primary cut_date:** `2026-08-21` (explore `2026-06-18` → `2026-08-20`; holdout `2026-08-21` → `2026-09-17`)
- Signals E 50/20: explore **3623**, holdout **292**, holdout assets **19**
- Secondary cut_date: `2026-08-12` (holdout signals **717**)
- chart_close_only: `True`

### Holdout — by band × horizon × benchmark

| band | horizon_d | benchmark_id | n | n_unique_assets | median_ret | win_rate | median_excess | mean_excess |
|------|------|------|------|------|------|------|------|------|
| A_5_50 | 1 | group_mean | 115 | 16 | 0.46% | 56.52% | -0.43% | 0.69% |
| A_5_50 | 1 | group_median | 115 | 16 | 0.46% | 56.52% | 0.06% | 1.56% |
| A_5_50 | 1 | random | 115 | 16 | 0.46% | 56.52% | -0.44% | 0.21% |
| A_5_50 | 3 | group_mean | 104 | 15 | 0.66% | 60.58% | -1.04% | -0.71% |
| A_5_50 | 3 | group_median | 104 | 15 | 0.66% | 60.58% | 0.84% | 1.18% |
| A_5_50 | 3 | random | 104 | 15 | 0.66% | 60.58% | -0.85% | -1.30% |
| A_5_50 | 7 | group_mean | 86 | 14 | -1.22% | 45.35% | -3.23% | 0.67% |
| A_5_50 | 7 | group_median | 86 | 14 | -1.22% | 45.35% | -0.24% | 4.35% |
| A_5_50 | 7 | random | 86 | 14 | -1.22% | 45.35% | -2.15% | -0.17% |
| A_5_50 | 14 | group_mean | 57 | 11 | 1.52% | 54.39% | -5.51% | 3.08% |
| A_5_50 | 14 | group_median | 57 | 11 | 1.52% | 54.39% | -0.34% | 8.89% |
| A_5_50 | 14 | random | 57 | 11 | 1.52% | 54.39% | -4.29% | 4.12% |
| A_5_50 | 21 | group_mean | 22 | 7 | -4.96% | 40.91% | -14.39% | -0.18% |
| A_5_50 | 21 | group_median | 22 | 7 | -4.96% | 40.91% | -5.94% | 8.66% |
| A_5_50 | 21 | random | 22 | 7 | -4.96% | 40.91% | -9.77% | -1.28% |
| B_10_100 | 1 | group_mean | 165 | 16 | -0.06% | 49.70% | -1.14% | 0.39% |
| B_10_100 | 1 | group_median | 165 | 16 | -0.06% | 49.70% | -0.18% | 1.13% |
| B_10_100 | 1 | random | 165 | 16 | -0.06% | 49.70% | -0.85% | -0.24% |
| B_10_100 | 3 | group_mean | 151 | 16 | -0.65% | 45.70% | -2.18% | -0.14% |
| B_10_100 | 3 | group_median | 151 | 16 | -0.65% | 45.70% | -0.42% | 1.33% |
| B_10_100 | 3 | random | 151 | 16 | -0.65% | 45.70% | -1.57% | -0.65% |
| B_10_100 | 7 | group_mean | 123 | 16 | -1.18% | 45.53% | -3.13% | 1.41% |
| B_10_100 | 7 | group_median | 123 | 16 | -1.18% | 45.53% | -0.18% | 4.26% |
| B_10_100 | 7 | random | 123 | 16 | -1.18% | 45.53% | -2.16% | 0.48% |
| B_10_100 | 14 | group_mean | 75 | 12 | 2.07% | 54.67% | -4.08% | 5.29% |
| B_10_100 | 14 | group_median | 75 | 12 | 2.07% | 54.67% | 0.04% | 9.99% |
| B_10_100 | 14 | random | 75 | 12 | 2.07% | 54.67% | -5.19% | 4.03% |
| B_10_100 | 21 | group_mean | 28 | 8 | -2.20% | 42.86% | -8.89% | -11.32% |
| B_10_100 | 21 | group_median | 28 | 8 | -2.20% | 42.86% | -1.82% | -4.54% |
| B_10_100 | 21 | random | 28 | 8 | -2.20% | 42.86% | -12.87% | -10.67% |

### Holdout — pooled A∪B

| horizon_d | benchmark_id | n | n_unique_assets | median_ret | win_rate | median_excess | mean_excess |
|------|------|------|------|------|------|------|------|
| 1 | group_mean | 280 | 19 | 0.26% | 52.50% | -0.75% | 0.51% |
| 1 | group_median | 280 | 19 | 0.26% | 52.50% | -0.02% | 1.31% |
| 1 | random | 280 | 19 | 0.26% | 52.50% | -0.64% | -0.06% |
| 3 | group_mean | 255 | 19 | 0.10% | 51.76% | -1.49% | -0.38% |
| 3 | group_median | 255 | 19 | 0.10% | 51.76% | 0.00% | 1.27% |
| 3 | random | 255 | 19 | 0.10% | 51.76% | -1.26% | -0.91% |
| 7 | group_mean | 209 | 18 | -1.18% | 45.45% | -3.20% | 1.11% |
| 7 | group_median | 209 | 18 | -1.18% | 45.45% | -0.20% | 4.30% |
| 7 | random | 209 | 18 | -1.18% | 45.45% | -2.16% | 0.21% |
| 14 | group_mean | 132 | 14 | 1.68% | 54.55% | -4.39% | 4.33% |
| 14 | group_median | 132 | 14 | 1.68% | 54.55% | -0.10% | 9.51% |
| 14 | random | 132 | 14 | 1.68% | 54.55% | -5.18% | 4.06% |
| 21 | group_mean | 50 | 10 | -3.44% | 42.00% | -11.56% | -6.42% |
| 21 | group_median | 50 | 10 | -3.44% | 42.00% | -3.69% | 1.27% |
| 21 | random | 50 | 10 | -3.44% | 42.00% | -12.73% | -6.53% |

### Explore (side-by-side only — not validation)

| band | horizon_d | benchmark_id | n | n_unique_assets | median_ret | win_rate | median_excess | mean_excess |
|------|------|------|------|------|------|------|------|------|
| A_5_50 | 1 | group_mean | 1678 | 82 | -0.29% | 45.83% | -0.50% | -0.03% |
| A_5_50 | 1 | group_median | 1678 | 82 | -0.29% | 45.83% | 0.07% | 0.57% |
| A_5_50 | 1 | random | 1678 | 82 | -0.29% | 45.83% | -0.40% | -0.01% |
| A_5_50 | 3 | group_mean | 1678 | 82 | -0.55% | 44.52% | -1.11% | 0.09% |
| A_5_50 | 3 | group_median | 1678 | 82 | -0.55% | 44.52% | 0.03% | 1.48% |
| A_5_50 | 3 | random | 1678 | 82 | -0.55% | 44.52% | -0.62% | 0.12% |
| A_5_50 | 7 | group_mean | 1678 | 82 | -0.38% | 48.51% | -2.05% | 0.89% |
| A_5_50 | 7 | group_median | 1678 | 82 | -0.38% | 48.51% | 0.44% | 3.94% |
| A_5_50 | 7 | random | 1678 | 82 | -0.38% | 48.51% | -0.88% | 0.71% |
| A_5_50 | 14 | group_mean | 1678 | 82 | -0.13% | 49.23% | -4.22% | 1.66% |
| A_5_50 | 14 | group_median | 1678 | 82 | -0.13% | 49.23% | 0.71% | 7.03% |
| A_5_50 | 14 | random | 1678 | 82 | -0.13% | 49.23% | -2.35% | 1.37% |
| A_5_50 | 21 | group_mean | 1678 | 82 | 1.13% | 52.15% | -5.49% | -0.31% |
| A_5_50 | 21 | group_median | 1678 | 82 | 1.13% | 52.15% | 0.35% | 6.15% |
| A_5_50 | 21 | random | 1678 | 82 | 1.13% | 52.15% | -3.23% | -0.33% |
| A_5_50 | 30 | group_mean | 1658 | 82 | 1.67% | 52.71% | -7.02% | -1.74% |
| A_5_50 | 30 | group_median | 1658 | 82 | 1.67% | 52.71% | 0.41% | 6.18% |
| A_5_50 | 30 | random | 1658 | 82 | 1.67% | 52.71% | -5.86% | -2.01% |
| B_10_100 | 1 | group_mean | 1945 | 94 | -0.38% | 45.04% | -0.37% | 0.19% |
| B_10_100 | 1 | group_median | 1945 | 94 | -0.38% | 45.04% | 0.05% | 0.60% |
| B_10_100 | 1 | random | 1945 | 94 | -0.38% | 45.04% | -0.19% | 0.12% |
| B_10_100 | 3 | group_mean | 1945 | 94 | -0.64% | 43.44% | -0.63% | 0.66% |
| B_10_100 | 3 | group_median | 1945 | 94 | -0.64% | 43.44% | 0.15% | 1.55% |
| B_10_100 | 3 | random | 1945 | 94 | -0.64% | 43.44% | -0.03% | 0.61% |
| B_10_100 | 7 | group_mean | 1945 | 94 | -1.16% | 44.63% | -1.06% | 1.20% |
| B_10_100 | 7 | group_median | 1945 | 94 | -1.16% | 44.63% | 0.55% | 3.10% |
| B_10_100 | 7 | random | 1945 | 94 | -1.16% | 44.63% | -0.21% | 1.60% |
| B_10_100 | 14 | group_mean | 1945 | 94 | -1.35% | 45.14% | -2.75% | 1.07% |
| B_10_100 | 14 | group_median | 1945 | 94 | -1.35% | 45.14% | 0.44% | 4.40% |
| B_10_100 | 14 | random | 1945 | 94 | -1.35% | 45.14% | -0.79% | 1.28% |
| B_10_100 | 21 | group_mean | 1945 | 94 | -0.68% | 48.64% | -3.05% | 0.18% |
| B_10_100 | 21 | group_median | 1945 | 94 | -0.68% | 48.64% | 0.63% | 4.18% |
| B_10_100 | 21 | random | 1945 | 94 | -0.68% | 48.64% | -1.31% | 0.63% |
| B_10_100 | 30 | group_mean | 1920 | 94 | -0.11% | 49.74% | -3.65% | -0.15% |
| B_10_100 | 30 | group_median | 1920 | 94 | -0.11% | 49.74% | 0.80% | 4.63% |
| B_10_100 | 30 | random | 1920 | 94 | -0.11% | 49.74% | -2.55% | -0.14% |

## Per-asset concentration (holdout, h=7, vs group_median)

| asset_id | n_signals | bands | median 7d ret | median excess vs group_median |
|----------|-----------|-------|---------------|-------------------------------|
| pha | 29 | A_5_50,B_10_100 | 0.80% | 1.96% |
| steem | 25 | A_5_50,B_10_100 | 5.43% | 1.24% |
| power-protocol | 18 | A_5_50,B_10_100 | 6.53% | 4.30% |
| alien-worlds | 18 | A_5_50,B_10_100 | -3.49% | -0.73% |
| synapse-2 | 18 | A_5_50,B_10_100 | -13.87% | -12.80% |
| zilliqa | 15 | A_5_50,B_10_100 | -2.21% | -0.01% |
| book-of-meme | 15 | B_10_100 | -8.99% | -8.17% |
| ordinals | 11 | B_10_100 | -0.89% | 0.17% |
| unifai-network | 10 | B_10_100 | 48.78% | 47.07% |
| thena | 10 | A_5_50 | -1.79% | -0.04% |
| cross-2 | 8 | A_5_50,B_10_100 | 15.68% | 20.40% |
| holoworld | 8 | A_5_50,B_10_100 | -6.05% | -3.14% |
| endurance | 7 | A_5_50,B_10_100 | -17.80% | -17.42% |
| renzo | 6 | A_5_50,B_10_100 | -2.28% | -1.39% |
| heima | 4 | A_5_50,B_10_100 | -8.41% | -12.66% |
| iostoken | 4 | A_5_50,B_10_100 | -8.85% | -3.16% |
| helium | 2 | A_5_50 | 156.70% | 156.95% |
| skyai | 1 | B_10_100 | 9.18% | 16.75% |

## Leave-one-asset-out (holdout, h=7, vs group_median, pooled)

Sorted by remaining median excess ascending (worst remaining sign first). Also note which drop hurts the full-sample median most.

| dropped_asset | n_remaining | median_excess | delta_vs_full |
|---------------|-------------|---------------|---------------|
| pha | 180 | -0.34% | -0.15% |
| steem | 184 | -0.32% | -0.13% |
| power-protocol | 191 | -0.30% | -0.11% |
| unifai-network | 199 | -0.30% | -0.11% |
| cross-2 | 201 | -0.28% | -0.08% |
| helium | 207 | -0.21% | -0.01% |
| ordinals | 198 | -0.20% | -0.01% |
| skyai | 208 | -0.20% | -0.01% |
| zilliqa | 194 | -0.20% | -0.01% |
| iostoken | 205 | -0.20% | 0.00% |
| renzo | 203 | -0.20% | 0.00% |
| thena | 199 | -0.20% | 0.00% |
| heima | 205 | -0.14% | 0.06% |
| holoworld | 201 | -0.14% | 0.06% |
| endurance | 202 | -0.10% | 0.09% |
| alien-worlds | 191 | -0.01% | 0.18% |
| book-of-meme | 194 | -0.00% | 0.19% |
| synapse-2 | 191 | 0.00% | 0.20% |

### Secondary robustness split — by band (h focus in table; labeled secondary)

| band | horizon_d | benchmark_id | n | n_unique_assets | median_ret | win_rate | median_excess | mean_excess |
|------|------|------|------|------|------|------|------|------|
| A_5_50 | 7 | group_mean | 282 | 61 | 7.42% | 75.18% | -1.47% | 1.21% |
| A_5_50 | 7 | group_median | 282 | 61 | 7.42% | 75.18% | 0.11% | 2.88% |
| A_5_50 | 7 | random | 282 | 61 | 7.42% | 75.18% | -0.37% | 1.93% |
| B_10_100 | 7 | group_mean | 352 | 65 | 7.62% | 71.88% | -0.80% | 1.35% |
| B_10_100 | 7 | group_median | 352 | 65 | 7.62% | 71.88% | 0.05% | 2.34% |
| B_10_100 | 7 | random | 352 | 65 | 7.62% | 71.88% | -0.19% | 1.78% |

## Explicit non-conclusion

This note tries to **disprove or pressure-test** a narrow earlier vs-random holdout reading for quiet→rising N-only. It does **not** authorize trading, does **not** claim alpha, and does **not** expand the microcap universe (Abdul's ask for a larger real universe remains open). Chart-close proxy entries, Santiment free-plan lag, short history, and nested prior peeks all remain.

---
*CMRAM V0.1 — N-only validation research note — not investment advice*
