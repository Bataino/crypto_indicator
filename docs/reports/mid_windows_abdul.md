# Mid windows (30–60) — plain note for Abdul

**When:** 17 Sep 2026, ~17:00 WAT (Africa/Lagos)  
**Tag:** `features_mid_v0.1`  
**Status:** research only — **not alpha**, not a trading system

## What we ran

1. Kept short `config/features.yaml` (`features_v0.1`) unchanged.
2. Added `config/features_mid.yaml` with C/V/P/NSI lookbacks mapped to ~30d (inner) and ~60d (outer). Mapping table is in that YAML’s header comments.
3. Recomputed features → signals → backtest → time holdout (same 70/30 cut as before).
4. Focus cells: Models **B** and **C**, pre-registered τ_m=50 τ_g=20, median excess vs same-band **random** at **h=7**.

## Windows used (mid)

- Inner ≈ **30 days**; outer ≈ **60 days** (see `config/features_mid.yaml`).
- Membership calendar unchanged: **91** days (2026-06-18 → 2026-09-16).
- Explore / holdout cut: **2026-08-20** (first 70% explore, last 30% holdout).
- Explore signals: 5763; holdout signals: 2101.

## Coexistence / overwrite

DuckDB `features_daily` primary key is `(timestamp, asset_id, band, model)` — **no** `model_version` in the key. Running mid **overwrites** the live DuckDB feature/signal/backtest tables. Short artifacts were copied aside as `*_short_v0.1.*`; mid as `*_mid_v0.1.*` / `features_daily_mid.parquet`.

## Did Gap beat random on the later fair test?

### Holdout (pre-registered, h=7 vs random)

| model | τ_m | τ_g | n | median excess vs random |
|-------|-----|-----|---|-------------------------|
| B | 50 | 20 | 138 | **-10.63%** |
| C | 50 | 20 | 114 | **-3.25%** |

Explore peek (C 70/30) also went negative on holdout (−3.75%, n=25).

**Answer: no** — Gap did **not** beat random on the later fair (holdout) test for mid 30–60 windows.

## Compare to short (easy)

Short `features_v0.1` holdout (same cells, prior run):

| model | τ_m | τ_g | n | median excess vs random |
|-------|-----|-----|---|-------------------------|
| B | 50 | 20 | 110 | **-5.51%** |
| C | 50 | 20 | 81 | **-3.47%** |

Mid is also negative vs random (B worse than short on this one number; C similar). Neither short nor mid shows a positive holdout median excess at the pre-registered cell. **No alpha claim.**

## Next

Because Step 1 shows **no** predictive edge on holdout → Step 2 (~120-day windows) was run; see `long120_windows_abdul.md`.

