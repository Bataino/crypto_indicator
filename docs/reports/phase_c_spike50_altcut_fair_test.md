# Phase C spike50 — alternate time-cut fair test (Abdul)

**When:** 2026-09-21 09:12:16 WAT (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Primary label:** `spike_50_7d` (max close within 7d ≥ +50%)  
**Model:** `lightgbm` (one train; band cuts locked on **explore only**; later scored once per band)  
**Trees used (early stop):** 51  
**Alternate cut:** explore_frac=**0.6** → explore ≤ **2026-08-10**  
**Bands:** p80 / p90 / p95 — quantiles of explore predicted probability

## Short answers

- **Alt cut date (explore ≤):** **2026-08-10** (explore_frac=0.6)
- **Later spike AUC:** **0.6868** (0.5 = coin flip); beats chance AUC? **YES**
- **Do tighter bands still beat chance + locked?** **YES — both p90 and p95 still beat chance + locked on later spike**
- **Still beat locked draft on this alt cut?** **YES — all reported high-prob bands still beat the locked draft rule on later spike hit %**
- **Locked draft rule on later** (same window): fires **1,250**, spike hit **3.52%** (lift **1.043x** vs everyday **3.38%**)

- **p80**: spike hit 6.36% (lift 1.883x vs base 3.38%); fires 6,542 (29.06% of later); beat chance+locked? **YES**
- **p90**: spike hit 9.01% (lift 2.669x vs base 3.38%); fires 3,007 (13.36% of later); beat chance+locked? **YES**
- **p95**: spike hit 10.66% (lift 3.157x vs base 3.38%); fires 1,426 (6.33% of later); beat chance+locked? **YES**

**No alpha claim.** Do not trade on this. Robustness check only (alternate time cut). Spike events are rare (~3%).

## What we did (plain)

1. Same Phase B Band C coin-day table + same features as the spike50 tighten fair test.
2. **Alternate explore cut:** explore_frac=**0.6** → explore dates ≤ **2026-08-10** (31,004 rows).  
   **Later** = dates after that (22,510 rows). Thresholds chosen on this explore only — **no peeking at later**.
3. Validation carved from the **end of explore only** (train 25,216 / val 5,788).
4. Trained one small `lightgbm` on `spike_50_7d` (`scale_pos_weight` for rare positives).
5. On **explore** predicted probs, locked three cuts: **p80**, **p90**, **p95**.
6. Scored **later once** per band. Compared spike hit % and lift vs everyday later spike rate and vs locked rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.
7. Briefly compared to primary cut **2026-08-19** / explore_frac=0.7 from `phase_c_spike50_tighten_fair_test.md`.

## Features the model saw

`rvol_30`, `dist_ema_20`, `dist_ema_50`, `rsi_14`, `ret_1d`, `ret_3d`, `ret_7d`, `vol_14`, `log_mc`

## Band comparison on later (locked cuts — this alt explore)

| Band | Explore thr | Fires | % of later | Spike hit | Spike lift vs base | Up hit | Up lift | Beat chance (spike)? | Beat locked (spike)? |
|------|------------:|------:|-----------:|----------:|-------------------:|-------:|--------:|---------------------:|---------------------:|
| **p80** (explore_p80) | 0.4818 | 6,542 | 29.06% | 6.36% | 1.883x | 44.51% | 0.833x | YES | YES |
| **p90** (explore_p90) | 0.6547 | 3,007 | 13.36% | 9.01% | 2.669x | 43.30% | 0.810x | YES | YES |
| **p95** (explore_p95) | 0.7383 | 1,426 | 6.33% | 10.66% | 3.157x | 43.62% | 0.816x | YES | YES |

Everyday later spike rate: **3.38%**. Everyday later up-in-7d: **53.45%**.

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | 1,250 |
| Spike≥50% hit | 3.52% |
| Spike lift vs base | 1.043x |
| Up-in-7d hit | 56.32% |
| Up lift vs base | 1.054x |

## Brief vs primary cut (2026-08-19 / explore_frac=0.7)

Primary later spike AUC was **0.7045**; alt-cut later spike AUC is **0.6868**.  
Primary locked spike hit **3.57%** (lift 1.239x); alt locked spike hit **3.52%** (lift 1.043x).

| Band | Primary spike hit / lift | Alt-cut spike hit / lift | Lift delta (alt − primary) |
|------|-------------------------:|-------------------------:|---------------------------:|
| **p80** | 5.54% / 1.921x (5,599 fires) | 6.36% / 1.883x (6,542 fires) | -0.038x |
| **p90** | 7.53% / 2.612x (2,551 fires) | 9.01% / 2.669x (3,007 fires) | +0.057x |
| **p95** | 9.43% / 3.273x (1,230 fires) | 10.66% / 3.157x (1,426 fires) | -0.116x |

Same story directionally: tighter bands (p90/p95) still show higher later spike lift than the wide p80 band, and still beat the locked draft on spike hit — on this earlier cut as well. Numbers move with the window (different later sample); treat as a robustness check, not a second claim of edge.

## Shared later ranking (not band-specific)

| Metric | Value |
|--------|------:|
| Later rows | 22,510 |
| Spike AUC | 0.6868 |
| Accuracy (@ prob≥0.5) | 74.00% |
| Majority baseline accuracy | 96.62% |

## Feature importance (train — descriptive only)

| feature | importance |
|---------|-----------:|
| `log_mc` | 91.0000 |
| `vol_14` | 69.0000 |
| `dist_ema_50` | 49.0000 |
| `dist_ema_20` | 32.0000 |
| `ret_7d` | 27.0000 |
| `ret_3d` | 23.0000 |
| `rvol_30` | 18.0000 |
| `ret_1d` | 17.0000 |
| `rsi_14` | 16.0000 |

## Notes

- Primary label spike_50_7d (~3% base): evaluate with AUC + high-prob spike lift vs base (accuracy alone is misleading vs ~97% majority).
- High-prob thresholds (p80/p90/p95) locked on explore probabilities only; later scored once per band — no peeking when choosing cuts.
- Alternate time cut: explore_frac=0.6 → cut date **2026-08-10** (primary was explore_frac=0.7 / 2026-08-19). Thresholds still locked on this explore only.

## Files

| Artifact | Path |
|----------|------|
| Dataset parquet | `/workspace/cmram/data/ai/ai_samples_daily.parquet` |
| Metrics JSON | `/workspace/cmram/data/ai/phase_c_spike50_altcut_fair_test.json` |
| This report | `/workspace/cmram/docs/reports/phase_c_spike50_altcut_fair_test.md` |
| Primary-cut report | `docs/reports/phase_c_spike50_tighten_fair_test.md` |
| Repro script | `scripts/run_ai_train.py --target spike_50_7d --bands p80,p90,p95 --explore-frac 0.6` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next spike  
- **No alpha**
