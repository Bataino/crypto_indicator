# Quiet→rising N + Model E (N-only) — plain report for Abdul

**When:** 17 Sep 2026, ~19:21 WAT (Africa/Lagos)  
**Status:** research check only — **not alpha**, not a trading system  
**Universe:** ~150 eligible membership assets (Band A∪B)

## Short answer

**Did quiet→rising N alone beat random on the later holdout window?**  
**Yes — narrowly — at the pre-registered Model E cell (τ_m=50, τ_g=20, h=7).**  
Median excess vs same-band random ≈ **+0.71%** (n=86 paths, 14 assets).

That is a small directional edge on one cell, on a short later window. It is **not** proof N works, **not** permission to trade, and **not** an alpha claim. Model C (full C+V+P+N) at the same τ was still **worse than random** on holdout.

## What changed

### 1. Narrative score N redesigned (`quiet_rising`, default)

| piece | before (`legacy`) | now (`quiet_rising`) |
|-------|-------------------|----------------------|
| Idea | Soft low-base boost via `1/(1+log1p(baseline))` on raw vel/accel | **Was quiet, now rising**: only **positive** velocity/accel × quiet weight |
| Quiet check | Current long baseline | **Lagged** long baseline (shifted by short window) so a fresh rise does not immediately look “crowded” |
| Crowded talk | Still non-zero weight when baseline is high | Steep damp by absolute mention scale + **hard-zero** when lagged baseline > `n_crowd_hard_max` (80) |
| Config flag | — | `narrative_mode: quiet_rising` (default) or `legacy` in `config/features.yaml` |

Knobs (also under `lookbacks`): `n_crowd_scale=10`, `n_crowd_exp=3`, `n_crowd_hard_max=80`.

### 2. Model E (N-only)

Plain definition:

- **MREI** = `score_N` (only when `n_available`)
- **NSI** = `nsi_social` (cross-sectional baseline attention / social saturation)
- **Rotation_Gap** = MREI − NSI = N vs how crowded talk already is
- **C / V / P unused** (null on Model E rows)
- **Liquidity:** same membership `liquidity_pass` gate as other models (light volume gate only — no extra Amihud filter beyond membership)

Model C still = C+V+P+N when social is present.

### 3. Social coverage (Santiment)

- Probed first: **140/150** membership assets map to a Santiment slug (10 slug-missing).
- Appended gap series only (did **not** wipe existing): **35 → 140** assets with `source=santiment`.
- Session call use ≈ **107** (probe earlier + this fetch); stayed under a soft ~120 budget / monthly 1000 headroom.
- Also still have Trends (29) and Wikipedia (6). Features use existing `social_daily` combine.

### 4. Features + signals

- Recomputed features: model_version **`features_v0.1_quiet_n`**, models **A, B, C, E**.
- Rows: 20 858 per model; `n_available` true on Model C/E ≈ **14 430 / 20 858**.
- Signals regenerated over τ grids including E.

### 5. Lighter holdout (documented)

Full multi-band × all-model × full τ backtest is heavy. This check used:

- **One band:** `A_5_50` only  
- **Models:** `C` and `E` only  
- **Same cut idea:** first 70% membership days = explore, last 30% = holdout  
- **cut_date:** **2026-08-21** (explore 2026-06-18→08-20; holdout 08-21→09-17)  
- **Metric:** median excess return vs same-band **random** at horizon **h=7**  
- Pre-registered cells: B/C/E at τ_m=50 τ_g=20 (B empty here because models filter excluded B)

## Numbers (later / holdout window)

| cell | n | median ret | win% | **median excess vs random** |
|------|---|------------|------|-----------------------------|
| Model **C** τ_m=50 τ_g=20 | 162 | −3.11% | 35.2% | **−2.38%** |
| Model **E** τ_m=50 τ_g=20 | 86 | −1.22% | 45.3% | **+0.71%** |
| Explore peek E τ_m=60 τ_g=40 | 48 | −2.93% | 33.3% | **−1.63%** (peek failed) |

Explore (peek only — do not treat as validation):

| cell | n | median excess vs random |
|------|---|-------------------------|
| C 50/20 | 704 | +0.59% |
| E 50/20 | 1678 | +0.58% |
| E 60/40 (peek top) | 956 | +0.99% |

## Verdict (plain)

1. **Quiet→rising N-only (Model E) at the pre-registered 50/20 cell:** holdout median excess vs random was **positive** → **yes, it beat random on this later slice** (narrowly).  
2. **Model C (full stack with N) at 50/20:** holdout excess **negative** → still does **not** support C on this cut.  
3. **Explore peek (E 60/40):** failed on holdout (negative) → do not trust peek-selected τ.  
4. Sample is still small (E holdout n=86, 14 assets), history short, chart-close entries, Santiment free-plan lag. **No alpha claim.**

## Explicit non-conclusion

Do not trade on this. Do not treat Model E as validated. Treat as: *fixing N toward quiet→rising and testing N alone gave a weak positive vs-random reading on one pre-registered later cell; the peeked cell and Model C did not.*

## Code / config touched

- `src/cmram/features/narrative.py` — `quiet_rising` vs `legacy`
- `src/cmram/features/lookbacks.py` — mode + crowd knobs; models include E
- `src/cmram/features/pipeline.py` — Model E wiring; pass `narrative_mode`
- `config/features.yaml`, `config/thresholds.yaml` — E + quiet_rising knobs
- `src/cmram/backtest/holdout.py`, `scripts/run_holdout.py` — E prereg; `--bands` / `--models` lighter holdout
- Tests updated; **pytest: 117 passed, 1 skipped**

## Reproduce (no secrets printed)

```bash
# features + signals (uses DuckDB social_daily already on disk)
.venv/bin/python scripts/run_features.py
.venv/bin/python scripts/run_signals.py

# lighter holdout
.venv/bin/python scripts/run_holdout.py --bands A_5_50 --models C,E
```

Legacy N path: set `narrative_mode: legacy` in `config/features.yaml` and re-run features/signals.

---
*CMRAM V0.1 — quiet→rising N / Model E research note — not investment advice*
