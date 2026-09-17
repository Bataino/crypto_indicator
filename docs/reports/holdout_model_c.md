# CMRAM — Model C / N time holdout (plain language)

**Generated:** 2026-09-17 ~16:04 WAT (Africa/Lagos)  
**Status:** research only — **not a trading system**  
**Companion:** full tables in `holdout_report.md`

## What this is

We split membership history into an early **explore** window and a later **holdout** window (same 70/30 calendar rule as before). We then check whether Model C (which uses narrative score **N**, including Santiment social volume where available) looks better than random on the later window — and how it compares to Model B (no N) at the same thresholds.

**No alpha claims. Small sample. Chart-close-only entries.**

## Date cut (unchanged / recomputed same)

- Membership days: **91** (2026-06-18 → 2026-09-16)
- Rule: first 70% explore → **cut_date = 2026-08-20** (still valid)
- Explore: 2026-06-18 → 2026-08-19 (63 days); **4973** signals
- Holdout: 2026-08-20 → 2026-09-16 (28 days); **1983** signals
- Metric: median excess return vs same-band **random** at horizon **h=7**

## Cells reported

| Cell | Role |
|------|------|
| Model **B** τ_m=50 τ_g=20 h=7 | Pre-registered comparison (no N) |
| Model **C** τ_m=50 τ_g=20 h=7 | Pre-registered Model C / N |
| Explore top Model C | Peek only — here it **matched** C 50/20 (no extra cell) |

## Explore vs holdout (side-by-side)

### Explore (peeked — do not treat as validation)

| model | τ_m | τ_g | n | median ret | win% | median excess vs random |
|-------|-----|-----|---|------------|------|-------------------------|
| B | 50 | 20 | 343 | -1.67% | 44.3% | **+0.37%** |
| C | 50 | 20 | 361 | -1.47% | 44.0% | **+0.57%** † peek top Model C |

† Explore top Model C (n≥20) was C τ_m=50 τ_g=20 — same as the pre-registered C cell.

### Holdout (pre-registered eval)

| model | τ_m | τ_g | n | median ret | win% | median excess vs random |
|-------|-----|-----|---|------------|------|-------------------------|
| B | 50 | 20 | 110 | -3.67% | 30.0% | **-5.51%** |
| C | 50 | 20 | 81 | -3.65% | 34.6% | **-3.47%** |

## Verdict: does holdout support Model C / N?

**No — holdout does not support Model C / N at this pre-registered cell.**

- Model C’s holdout median excess vs random is **negative** (−3.47%, n=81).
- Model B is also negative (−5.51%, n=110). C is *less bad* than B on this one number, but that is **not** evidence of an N edge and is still worse than random.
- The explore peek for Model C was the same cell (50/20) and also **failed to hold** direction on holdout.

This does **not** prove N is useless forever. The sample is tiny, history is short, Santiment FREE data is lagged, and entries use chart-close proxies. It **does** mean: do not treat current Model C / Santiment N results as confirmed, and do not trade on them.

## Explicit non-conclusion

Do not claim alpha. Do not select Model C for live use from this holdout. Treat as one failed directional check for the Phase 1 N hypothesis under these thresholds and this window.

---
*CMRAM V0.1 Phase 1 — Model C holdout summary*
