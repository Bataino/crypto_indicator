# Phase C spike50 — tighten high-prob bands fair test (Abdul)

**When:** 2026-09-21 09:10:00 WAT (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Primary label:** `spike_50_7d` (max close within 7d ≥ +50%)  
**Model:** `lightgbm` (one train; band cuts locked on **explore only**; later scored once per band)  
**Trees used (early stop):** 43  
**Bands:** p80 (old) vs p90 (top ~10%) vs p95 (top ~5%) — quantiles of explore predicted probability

## Short answers

- **Later spike AUC:** **0.7045** (0.5 = coin flip); beats chance AUC? **YES**
- **Do tighter bands still beat chance + locked?** **YES — both p90 and p95 still beat chance + locked on later spike**
- **Locked draft rule on later** (same window): fires **924**, spike hit **3.57%** (lift **1.239x** vs everyday **2.88%**)

- **p80**: spike hit 5.54% (lift 1.921x vs base 2.88%); fires 5,599 (33.61% of later); beat chance+locked? **YES**
- **p90**: spike hit 7.53% (lift 2.612x vs base 2.88%); fires 2,551 (15.31% of later); beat chance+locked? **YES**
- **p95**: spike hit 9.43% (lift 3.273x vs base 2.88%); fires 1,230 (7.38% of later); beat chance+locked? **YES**

Old p80 band covered 33.61% of later (5,599 fires) — too wide for a ‘high-conviction’ read. p90 covers 15.31%; p95 covers 7.38%.

**No alpha claim.** Do not trade on this. Small / short later window. Spike events are rare (~3%).

## What we did (plain)

1. Same Phase B Band C coin-day table + same features as the prior spike50 fair test.
2. **Explore** = dates ≤ **2026-08-19** (36,856 rows).  
   **Later** = dates after that (16,658 rows). Thresholds chosen on explore only — **no peeking at later**.
3. Validation carved from the **end of explore only** (train 30,359 / val 6,497).
4. Trained one small `lightgbm` on `spike_50_7d` (`scale_pos_weight` for rare positives).
5. On **explore** predicted probs, locked three cuts: **p80** (old wide band), **p90**, **p95**.
6. Scored **later once** per band. Compared spike hit % and lift vs everyday later spike rate and vs locked rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.

## Features the model saw

`rvol_30`, `dist_ema_20`, `dist_ema_50`, `rsi_14`, `ret_1d`, `ret_3d`, `ret_7d`, `vol_14`, `log_mc`

## Band comparison on later (locked cuts)

| Band | Explore thr | Fires | % of later | Spike hit | Spike lift vs base | Up hit | Up lift | Beat chance (spike)? | Beat locked (spike)? |
|------|------------:|------:|-----------:|----------:|-------------------:|-------:|--------:|---------------------:|---------------------:|
| **p80** (explore_p80) | 0.4796 | 5,599 | 33.61% | 5.54% | 1.921x | 42.40% | 0.880x | YES | YES |
| **p90** (explore_p90) | 0.6230 | 2,551 | 15.31% | 7.53% | 2.612x | 39.44% | 0.818x | YES | YES |
| **p95** (explore_p95) | 0.7069 | 1,230 | 7.38% | 9.43% | 3.273x | 39.92% | 0.829x | YES | YES |

Everyday later spike rate: **2.88%**. Everyday later up-in-7d: **48.18%**.

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | 924 |
| Spike≥50% hit | 3.57% |
| Spike lift vs base | 1.239x |
| Up-in-7d hit | 52.60% |
| Up lift vs base | 1.092x |

## Shared later ranking (not band-specific)

| Metric | Value |
|--------|------:|
| Later rows | 16,658 |
| Spike AUC | 0.7045 |
| Accuracy (@ prob≥0.5) | 70.02% |
| Majority baseline accuracy | 97.12% |

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
- High-prob thresholds (p80/p90/p95) locked on explore probabilities only; later scored once per band — no peeking when choosing cuts.

## Files

| Artifact | Path |
|----------|------|
| Dataset parquet | `/workspace/cmram/data/ai/ai_samples_daily.parquet` |
| Metrics JSON | `/workspace/cmram/data/ai/phase_c_spike50_tighten_fair_test.json` |
| This report | `/workspace/cmram/docs/reports/phase_c_spike50_tighten_fair_test.md` |
| Repro script | `scripts/run_ai_train.py --target spike_50_7d --bands p80,p90,p95` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next spike  
- **No alpha**
