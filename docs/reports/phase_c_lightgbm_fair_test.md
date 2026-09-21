# Phase C — LightGBM fair test (Abdul)

**When:** 2026-09-21 08:57:13 WAT (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Model:** `lightgbm` (one train + one locked later score)  
**Trees used (early stop):** 40

## Short answers

- **Beats chance on later?** **MIXED — ranking a bit better than a coin flip (AUC), but the high-prob band did not clear a meaningful lift bar vs everyday chance**
- **Beats locked draft rule on later (up in 7d)?** **NO — high-prob up-hit did not beat the locked rule on later**
- Later AUC: **0.5383** (0.5 = coin flip)
- Later accuracy: **52.21%** vs everyday majority baseline **51.82%** (gap **0.39%**)
- When model says “high chance” (prob ≥ 0.5036, method `top_quintile_explore_p80`):  
  up-in-7d hit **50.44%** vs later everyday **48.18%**  
  (lift **1.047x**, fires **1687**)
- Same high-prob band, secondary spike ≥50% in 7d: hit **2.90%** vs everyday **2.88%** (lift **1.008x**)

- **Research pass bar** (beat chance **and** beat locked draft on later): **FAIL**

**No alpha claim.** Do not trade on this. Small / short later window.

## What we did (plain)

1. Used the Phase B table of Band C coin-days with features + labels.
2. **Explore** = dates ≤ **2026-08-19** (36,856 rows).  
   **Later** = dates after that (16,658 rows). We did **not** peek at later while training or picking the threshold.
3. Carved a small validation slice from the **end of explore only** (train 30,359 / val 6,497) for early stopping.
4. Trained a **small** `lightgbm` to predict “did price go up in 7 days?” (`up_7d`).
5. Picked a “high chance” probability cutoff on **explore only** (top_quintile_explore_p80).
6. Scored **later once**. Compared to everyday later chance and to the locked draft rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.

## Features the model saw

`rvol_30`, `dist_ema_20`, `dist_ema_50`, `rsi_14`, `ret_1d`, `ret_3d`, `ret_7d`, `vol_14`, `log_mc`

## Later fair check (locked)

| Metric | Value |
|--------|------:|
| Later rows | 16,658 |
| Everyday up-in-7d rate | 48.18% |
| Everyday spike≥50% rate | 2.88% |
| AUC | 0.5383 |
| Accuracy (@ prob≥0.5) | 52.21% |
| Majority baseline accuracy | 51.82% |
| High-prob fires | 1,687 |
| High-prob up hit | 50.44% |
| High-prob up lift vs base | 1.047x |
| High-prob spike hit | 2.90% |
| High-prob spike lift vs base | 1.008x |

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | 924 |
| Up-in-7d hit | 52.60% |
| Up lift vs base | 1.092x |
| Spike≥50% hit | 3.57% |
| Spike lift vs base | 1.239x |

## Feature importance (train — descriptive only)

| feature | importance |
|---------|-----------:|
| `ret_7d` | 44.0000 |
| `vol_14` | 44.0000 |
| `rsi_14` | 42.0000 |
| `log_mc` | 41.0000 |
| `dist_ema_50` | 37.0000 |
| `dist_ema_20` | 28.0000 |
| `ret_3d` | 27.0000 |
| `rvol_30` | 9.0000 |
| `ret_1d` | 3.0000 |

## Notes

- Early stopping on explore-val collapsed (<10 trees); used pre-specified fallback: fixed 40 trees on full explore (train+val combined). Later window still untouched until final score.

## Files

| Artifact | Path |
|----------|------|
| Dataset parquet | `/workspace/cmram/data/ai/ai_samples_daily.parquet` |
| This report | `/workspace/cmram/docs/reports/phase_c_lightgbm_fair_test.md` |
| Repro script | `scripts/run_ai_train.py` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next move  
