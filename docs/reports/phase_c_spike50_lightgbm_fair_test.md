# Phase C spike50 — LightGBM fair test (Abdul)

**When:** 2026-09-21 09:00:12 WAT (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Primary label:** `spike_50_7d` (max close within 7d ≥ +50%)  
**Model:** `lightgbm` (one train + one locked later score; class imbalance handled via `scale_pos_weight`)  
**Trees used (early stop):** 43

## Short answers

- **Beats chance on later (spike)?** **YES — clearer than chance on later dates (spike AUC and high-prob spike lift)**
- **Beats locked draft rule on later (spike≥50% in 7d)?** **YES — high-prob spike hit beat the locked rule on later**
- Later spike AUC: **0.7045** (0.5 = coin flip)
- Later accuracy (@0.5): **70.02%** vs majority “no spike” baseline **97.12%** — **do not use accuracy alone** (base spike rate ~3%)
- When model says “high chance of spike” (prob ≥ 0.4796, method `top_quintile_explore_p80`):  
  spike≥50% hit **5.54%** vs later everyday **2.88%**  
  (lift **1.921x**, fires **5599**)
- Same high-prob band, secondary up-in-7d: hit **42.40%** vs everyday **48.18%** (lift **0.880x**)

- **Research pass bar** (beat chance **and** beat locked draft on later spike): **PASS**

**No alpha claim.** Do not trade on this. Small / short later window. Spike events are rare (~3%).

## What we did (plain)

1. Used the Phase B table of Band C coin-days with features + labels.
2. **Explore** = dates ≤ **2026-08-19** (36,856 rows).  
   **Later** = dates after that (16,658 rows). We did **not** peek at later while training or picking the threshold.
3. Carved a small validation slice from the **end of explore only** (train 30,359 / val 6,497) for early stopping.
4. Trained a **small** `lightgbm` to predict “did price spike ≥50% within 7 days?” (`spike_50_7d`), with `scale_pos_weight` for the ~3% positive class.
5. Picked a “high chance” probability cutoff on **explore only** (top_quintile_explore_p80).
6. Scored **later once**. Compared high-prob spike hit vs everyday later spike rate and vs the locked draft rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.

## Features the model saw

`rvol_30`, `dist_ema_20`, `dist_ema_50`, `rsi_14`, `ret_1d`, `ret_3d`, `ret_7d`, `vol_14`, `log_mc`

## Later fair check (locked) — primary = spike

| Metric | Value |
|--------|------:|
| Later rows | 16,658 |
| Everyday spike≥50% rate | 2.88% |
| Everyday up-in-7d rate | 48.18% |
| Spike AUC | 0.7045 |
| Accuracy (@ prob≥0.5) | 70.02% |
| Majority baseline accuracy | 97.12% |
| High-prob fires | 5,599 |
| High-prob spike hit | 5.54% |
| High-prob spike lift vs base | 1.921x |
| High-prob up hit | 42.40% |
| High-prob up lift vs base | 0.880x |

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | 924 |
| Spike≥50% hit | 3.57% |
| Spike lift vs base | 1.239x |
| Up-in-7d hit | 52.60% |
| Up lift vs base | 1.092x |

## Feature importance (train — descriptive only)

| feature | importance |
|---------|-----------:|
| `vol_14` | 69.0000 |
| `log_mc` | 68.0000 |
| `dist_ema_50` | 41.0000 |
| `dist_ema_20` | 27.0000 |
| `ret_7d` | 24.0000 |
| `ret_3d` | 19.0000 |
| `rvol_30` | 17.0000 |
| `ret_1d` | 17.0000 |
| `rsi_14` | 10.0000 |

## Notes

- Primary label spike_50_7d (~3% base): evaluate with AUC + high-prob spike lift vs base (accuracy alone is misleading vs ~97% majority).
- Explore p80 threshold fired ~34% of later rows (5,599 / 16,658) — a wide band, not a tiny high-conviction slice. Lift is still above base and above the locked rule, but treat selectivity as modest.
- High-prob band had *lower* up-in-7d hit than everyday later (0.88x lift) — spike-focused selection is not the same as “goes up” selection.

## Files

| Artifact | Path |
|----------|------|
| Dataset parquet | `/workspace/cmram/data/ai/ai_samples_daily.parquet` |
| Metrics JSON | `data/ai/phase_c_spike50_fair_test.json` |
| This report | `/workspace/cmram/docs/reports/phase_c_spike50_lightgbm_fair_test.md` |
| Repro script | `scripts/run_ai_train.py --target spike_50_7d` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next spike  
