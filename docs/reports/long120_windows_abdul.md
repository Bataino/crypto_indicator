# Long ~120-day windows — plain note for Abdul

**When:** 17 Sep 2026, ~17:05 WAT (Africa/Lagos)  
**Tag:** `features_long120_v0.1`  
**Why this ran:** Step 1 mid (30–60) showed **no** Gap edge vs random on holdout → Step 2 required.  
**Status:** research only — **not alpha**, not a trading system

## What we ran

1. Kept `config/features.yaml` (short) and `config/features_mid.yaml` intact.
2. Added `config/features_long120.yaml`: outer lookbacks ≈ **120d**, supporting shorts/mids ≈ **60d** (mapping in YAML header).
3. Features → signals → backtest → same 70/30 time holdout.
4. Focus: Models **B** and **C**, pre-registered τ_m=50 τ_g=20, median excess vs same-band **random** at **h=7**.

## Windows used (long120)

- Long style ≈ **120 days**; supporting inner ≈ **60 days**.
- Same membership calendar: **91** days; cut **2026-08-20**.
- Explore signals: 6155; holdout signals: 2146.
- Caveat: market history ≈180d; early membership rows often lack full 120d history (more nulls than short/mid).

## Coexistence / overwrite

Same as mid: DuckDB PK has no `model_version` → live tables overwritten. Artifacts: `*_long120_v0.1.*`, `features_daily_long120.parquet`. Short/mid copies remain as `*_short_v0.1.*` / `*_mid_v0.1.*`. Live DuckDB after this run = **long120**.

## Did Gap beat random on the later fair test?

### Holdout (pre-registered, h=7 vs random)

| model | τ_m | τ_g | n | median excess vs random |
|-------|-----|-----|---|-------------------------|
| B | 50 | 20 | 124 | **-10.30%** |
| C | 50 | 20 | 84 | **+4.17%** |

Explore peek (C 70/30) had **n=0** on holdout → peek unsupported.

**Headline (same bar as mid — clear Gap edge on fair later test?): no**

Breakdown:
- Model **B**: **no** (−10.30%)
- Model **C**: **yes** at prereg cell only (+4.17%, n=84) — weak / small-n
- Explore peek: **no** (holdout n=0)

This is **not** an alpha claim. Weak directional consistency for C only; B still worse than random; sample still tiny; chart-close proxy entries.

## Compare to mid / short (easy)

| version | B holdout med excess | C holdout med excess |
|---------|----------------------|----------------------|
| short `features_v0.1` | −5.51% (n=110) | −3.47% (n=81) |
| mid `features_mid_v0.1` | −10.63% (n=138) | −3.25% (n=114) |
| long120 `features_long120_v0.1` | −10.30% (n=124) | **+4.17%** (n=84) |

Only long120 Model C is positive on this one holdout metric. Do not treat as validated.

