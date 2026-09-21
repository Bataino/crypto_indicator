# Phase B — AI labeled dataset (next-move)

**Zone:** Africa/Lagos (WAT / UTC+1)  
**Report time:** 2026-09-21 08:53:59 WAT  
**Scope:** Research only — labels + features. **No LightGBM training** (Phase C next).  
**Social / Gap / X:** OFF for v1.

## What we built

For each Band C coin-day we record:

1. **Inputs the model may see later (features, past/present only):**
   - `rvol_30` — today's volume ÷ 30-day average volume
   - `dist_ema_20` / `dist_ema_50` — how far price is from EMA20 / EMA50
   - `rsi_14` — 14-day RSI
   - `ret_1d`, `ret_3d`, `ret_7d` — recent simple returns
   - `vol_14` — 14-day realized volatility (std of log returns)
   - `log_mc` — log of point-in-time market cap (USD)
2. **What happened next (labels, future only):**
   - **`up_7d`** (primary) — 1 if close 7 days later is higher than today
   - **`spike_50_7d`** (secondary) — 1 if max close within the next 7 days is ≥ +50%
   - **`fwd_ret_7d`** — continuous 7-day return

No future prices leak into features. Rows need a real bar at t+7 and non-null primary features.

## Universe

| Knob | Value |
|------|-------|
| Band | `C_1_100` (prefer liquidity_pass) |
| Explore fraction | 70% of unique calendar days |

## Counts (plain English)

| Metric | Value |
|--------|------:|
| Usable labeled rows | **53,514** |
| Distinct assets | 688 |
| Date span | 2026-06-18 → 2026-09-14 |
| Explore cut date (inclusive) | **2026-08-19** |
| Explore rows (≤ cut) | 36,856 |
| Later rows (> cut) | 16,658 |
| `up_7d` rate (everyday chance up) | **46.10%** |
| `spike_50_7d` rate | 2.81% |
| Rows with all primary features non-null | 53,514 |

Primary features required: `rvol_30`, `dist_ema_20`, `rsi_14`, `ret_1d`, `ret_7d`, `vol_14`, `log_mc`.

### Success vs ≥50K usable rows

**PASS** — 53,514 usable rows (≥50,000 target).

## Null rates (after filters)

| Column | Null rate |
|--------|----------:|
| `rvol_30` | 0.00% |
| `dist_ema_20` | 0.00% |
| `dist_ema_50` | 0.00% |
| `rsi_14` | 0.00% |
| `ret_1d` | 0.00% |
| `ret_3d` | 0.00% |
| `ret_7d` | 0.00% |
| `vol_14` | 0.00% |
| `log_mc` | 0.00% |
| `up_7d` | 0.00% |
| `spike_50_7d` | 0.00% |
| `fwd_ret_7d` | 0.00% |

(After label + primary-feature filters, feature null rates should be ~0 for primary columns; `dist_ema_50` may still have a few nulls if listed as secondary.)

## Files written

| Artifact | Path |
|----------|------|
| DuckDB table | `/workspace/cmram/data/cmram.duckdb` → table `ai_samples_daily` |
| Parquet | `/workspace/cmram/data/ai/ai_samples_daily.parquet` |
| This report | `/workspace/cmram/docs/reports/phase_b_ai_dataset.md` |

## Time split (for Phase C train)

- **Explore (train / light tune):** timestamps ≤ **2026-08-19**
- **Later (fair test only):** timestamps > **2026-08-19**
- Printed once at explore_frac=0.7.

Do **not** peek at later rows while tuning.

## Feature columns (v1)

`rvol_30`, `dist_ema_20`, `dist_ema_50`, `rsi_14`, `ret_1d`, `ret_3d`, `ret_7d`, `vol_14`, `log_mc`

## Next (Phase C — not done here)

1. Install `lightgbm` + `scikit-learn` if needed (`pip install 'cmram[ml]'` or deps in pyproject).
2. Train LightGBM classifier on explore rows → P(`up_7d`=1).
3. Score later window once; compare vs chance and vs locked draft  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.
4. Plain Abdul report — research only, no orders.

## Explicit non-goals (honored)

- No model training this phase
- No X / Santiment / social features
- No trading / wallets / live signals
- No GitHub push required for this step
