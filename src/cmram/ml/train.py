"""Phase C: LightGBM (or sklearn fallback) train + fair later test.

Research only — no trading, no wallets, no social/X.
Train and tune on explore dates only; score later once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cmram.ml.dataset import FEATURE_COLS, EXPLORE_FRAC_DEFAULT

logger = logging.getLogger(__name__)

LAGOS = ZoneInfo("Africa/Lagos")

# Locked Phase B cut (explore inclusive).
DEFAULT_CUT_DATE = "2026-08-19"
PRIMARY_LABEL = "up_7d"
SECONDARY_LABEL = "spike_50_7d"

# Locked draft rule (same as prior fair tests).
LOCKED_RVOL = 1.5
LOCKED_DIST_EMA20 = 0.05

# Small / RAM-friendly LightGBM defaults.
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 3,
    "min_child_samples": 80,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 120,
    "n_jobs": 1,
    "verbose": -1,
    "random_state": 42,
}

VAL_FRAC_OF_EXPLORE = 0.15  # last 15% of explore calendar days for early stop


@dataclass
class FairTestResult:
    model_name: str
    cut_date: str
    n_explore: int
    n_train: int
    n_val: int
    n_later: int
    feature_cols: list[str]
    # explore base / threshold
    explore_up_rate: float
    threshold: float
    threshold_method: str
    # later overall
    later_up_rate: float
    later_spike_rate: float
    later_auc: float | None
    later_accuracy: float
    later_accuracy_vs_chance: float  # accuracy - max(base, 1-base) majority baseline
    majority_baseline_acc: float
    # high-prob band on later
    n_high_later: int
    high_up_hit_rate: float | None
    high_up_lift_vs_base: float | None
    high_spike_hit_rate: float | None
    high_spike_lift_vs_base: float | None
    # locked rule on later (same window)
    locked_n_fire: int
    locked_up_hit_rate: float | None
    locked_up_lift: float | None
    locked_spike_hit_rate: float | None
    locked_spike_lift: float | None
    # plain yes/no
    beats_chance_auc: bool  # AUC > 0.5
    beats_chance_high_prob: bool  # high-prob up hit > later base (and enough fires)
    beats_locked_up: bool | None  # high-prob up hit > locked up hit (same later)
    beats_locked_spike: bool | None
    best_iteration: int | None = None
    notes: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ai_samples(
    parquet_path: Path | str | None = None,
    duckdb_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load ai_samples_daily from parquet (preferred) or DuckDB."""
    if parquet_path is not None and Path(parquet_path).is_file():
        df = pd.read_parquet(parquet_path)
        logger.info("loaded parquet %s rows=%s", parquet_path, len(df))
        return df
    if duckdb_path is not None and Path(duckdb_path).is_file():
        import duckdb

        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            df = con.execute("SELECT * FROM ai_samples_daily").df()
        finally:
            con.close()
        logger.info("loaded duckdb ai_samples_daily rows=%s", len(df))
        return df
    raise FileNotFoundError("Need parquet or DuckDB with ai_samples_daily")


def split_explore_later(
    samples: pd.DataFrame,
    *,
    cut_date: str = DEFAULT_CUT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cut = pd.Timestamp(cut_date)
    ts = pd.to_datetime(samples["timestamp"]).dt.tz_localize(None).dt.normalize()
    explore = samples.loc[ts <= cut].copy()
    later = samples.loc[ts > cut].copy()
    return explore, later, cut


def carve_explore_val(
    explore: pd.DataFrame,
    *,
    val_frac: float = VAL_FRAC_OF_EXPLORE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Carve validation from the *end* of explore only (never later)."""
    if not 0.0 < val_frac < 0.5:
        raise ValueError(f"val_frac must be in (0, 0.5), got {val_frac}")
    ts = pd.to_datetime(explore["timestamp"]).dt.tz_localize(None).dt.normalize()
    days = ts.drop_duplicates().sort_values().reset_index(drop=True)
    n_val_days = max(1, int(np.ceil(len(days) * val_frac)))
    n_val_days = min(n_val_days, len(days) - 1) if len(days) > 1 else 1
    val_start = pd.Timestamp(days.iloc[len(days) - n_val_days])
    train = explore.loc[ts < val_start].copy()
    val = explore.loc[ts >= val_start].copy()
    if len(train) == 0:
        # tiny explore: fall back to random 85/15 within explore
        rng = np.random.default_rng(42)
        mask = rng.random(len(explore)) < (1.0 - val_frac)
        train, val = explore.loc[mask].copy(), explore.loc[~mask].copy()
        val_start = pd.Timestamp(ts.min())
    return train, val, val_start


def _xy(df: pd.DataFrame, feature_cols: list[str], label: str = PRIMARY_LABEL):
    X = df[feature_cols].astype("float64")
    y = df[label].astype("float64").astype(int)
    return X, y


def _try_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except Exception as e:  # noqa: BLE001
        logger.warning("lightgbm unavailable: %s", e)
        return None


def train_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] | list[str] = FEATURE_COLS,
    label: str = PRIMARY_LABEL,
) -> tuple[Any, str, dict[str, float], int | None]:
    """Train small LightGBM; fall back to HistGradientBoosting or logistic."""
    cols = list(feature_cols)
    X_tr, y_tr = _xy(train_df, cols, label)
    X_va, y_va = _xy(val_df, cols, label)

    lgb = _try_lightgbm()
    if lgb is not None:
        try:
            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            # Prefer new eval_X/eval_y API when available.
            try:
                model.fit(
                    X_tr,
                    y_tr,
                    eval_X=X_va,
                    eval_y=y_va,
                    eval_metric="auc",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
            except TypeError:
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
            best_it = getattr(model, "best_iteration_", None)
            used_fallback = False
            # Pre-specified fallback: if early stop collapses (<10 trees),
            # retrain fixed small forest on full explore (train+val), never later.
            if best_it is None or int(best_it) < 10:
                logger.info(
                    "early stop best_iteration=%s < 10; "
                    "fallback fixed n_estimators=40 on full explore",
                    best_it,
                )
                X_full = pd.concat([X_tr, X_va], axis=0)
                y_full = pd.concat([y_tr, y_va], axis=0)
                params = dict(LGBM_PARAMS)
                params["n_estimators"] = 40
                model = lgb.LGBMClassifier(**params)
                model.fit(X_full, y_full)
                best_it = 40
                used_fallback = True
            else:
                best_it = int(best_it)
            imp = dict(
                zip(cols, (float(x) for x in model.feature_importances_), strict=True)
            )
            logger.info(
                "trained LightGBM trees=%s fallback=%s", best_it, used_fallback
            )
            return model, "lightgbm", imp, best_it
        except Exception as e:  # noqa: BLE001
            logger.warning("LightGBM train failed (%s); trying sklearn", e)

    from sklearn.ensemble import HistGradientBoostingClassifier

    try:
        # Internal validation_fraction is random within train only (never later).
        model = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=100,
            learning_rate=0.05,
            min_samples_leaf=80,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        imp = {c: 0.0 for c in cols}
        logger.info("trained HistGradientBoostingClassifier")
        return model, "hist_gradient_boosting", imp, getattr(model, "n_iter_", None)
    except Exception as e:  # noqa: BLE001
        logger.warning("HistGB failed (%s); logistic regression", e)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    C=0.5,
                    solver="lbfgs",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_tr, y_tr)
    coef = model.named_steps["clf"].coef_.ravel()
    imp = dict(zip(cols, (float(abs(x)) for x in coef), strict=True))
    logger.info("trained logistic regression")
    return model, "logistic_regression", imp, None


def predict_proba_positive(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    # class 1 column
    classes = getattr(model, "classes_", None)
    if classes is not None:
        idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
    else:
        idx = 1
    return np.asarray(proba[:, idx], dtype="float64")


def tune_threshold_explore(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    method: str = "top_quintile",
) -> tuple[float, str]:
    """Pick probability threshold on explore only (never later)."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype="float64")
    if method == "top_quintile":
        thr = float(np.quantile(proba, 0.80))
        return thr, "top_quintile_explore_p80"
    # F1 sweep on explore
    best_thr, best_f1 = 0.5, -1.0
    for q in np.linspace(0.50, 0.95, 19):
        thr = float(np.quantile(proba, q))
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, f"f1_explore_best_f1={best_f1:.4f}"


def locked_rule_mask(df: pd.DataFrame) -> pd.Series:
    return (df["rvol_30"] > LOCKED_RVOL) & (df["dist_ema_20"] <= LOCKED_DIST_EMA20)


def _safe_rate(hits: int, n: int) -> float | None:
    if n <= 0:
        return None
    return float(hits) / float(n)


def _lift(rate: float | None, base: float) -> float | None:
    if rate is None or base <= 0:
        return None
    return float(rate) / float(base)


def evaluate_fair_test(
    model: Any,
    model_name: str,
    explore: pd.DataFrame,
    later: pd.DataFrame,
    *,
    cut_date: pd.Timestamp,
    feature_cols: list[str],
    feature_importance: dict[str, float],
    n_train: int,
    n_val: int,
    threshold_method: str = "top_quintile",
    min_high_fires: int = 30,
    best_iteration: int | None = None,
) -> FairTestResult:
    from sklearn.metrics import accuracy_score, roc_auc_score

    notes: list[str] = []
    X_ex, y_ex = _xy(explore, feature_cols)
    proba_ex = predict_proba_positive(model, X_ex)
    thr, thr_name = tune_threshold_explore(y_ex.to_numpy(), proba_ex, method=threshold_method)

    X_la, y_la = _xy(later, feature_cols)
    proba_la = predict_proba_positive(model, X_la)
    y_la_np = y_la.to_numpy()
    spike_la = later[SECONDARY_LABEL].astype("float64").fillna(0).astype(int).to_numpy()

    later_up_rate = float(y_la_np.mean()) if len(y_la_np) else 0.0
    later_spike_rate = float(spike_la.mean()) if len(spike_la) else 0.0
    majority = max(later_up_rate, 1.0 - later_up_rate)
    pred_bin = (proba_la >= 0.5).astype(int)
    acc = float(accuracy_score(y_la_np, pred_bin)) if len(y_la_np) else 0.0

    try:
        auc = float(roc_auc_score(y_la_np, proba_la)) if len(np.unique(y_la_np)) > 1 else None
    except ValueError:
        auc = None
        notes.append("AUC undefined (single class in later).")

    high = proba_la >= thr
    n_high = int(high.sum())
    high_up = _safe_rate(int((y_la_np[high] == 1).sum()), n_high) if n_high else None
    high_spike = _safe_rate(int((spike_la[high] == 1).sum()), n_high) if n_high else None

    lock = locked_rule_mask(later).to_numpy()
    n_lock = int(lock.sum())
    lock_up = _safe_rate(int((y_la_np[lock] == 1).sum()), n_lock) if n_lock else None
    lock_spike = _safe_rate(int((spike_la[lock] == 1).sum()), n_lock) if n_lock else None

    beats_auc = bool(auc is not None and auc > 0.5)
    # Require a meaningful edge vs base (not noise): lift >= 1.05 and >=1pp gap.
    high_lift = _lift(high_up, later_up_rate)
    beats_high = bool(
        high_up is not None
        and n_high >= min_high_fires
        and high_lift is not None
        and high_lift >= 1.05
        and (high_up - later_up_rate) >= 0.01
    )
    beats_locked_up = None
    if high_up is not None and lock_up is not None and n_high >= min_high_fires and n_lock >= 5:
        beats_locked_up = bool(high_up > lock_up)
    beats_locked_spike = None
    if (
        high_spike is not None
        and lock_spike is not None
        and n_high >= min_high_fires
        and n_lock >= 5
    ):
        beats_locked_spike = bool(high_spike > lock_spike)

    explore_up = float(explore[PRIMARY_LABEL].astype(float).mean())

    return FairTestResult(
        model_name=model_name,
        cut_date=str(pd.Timestamp(cut_date).date()),
        n_explore=int(len(explore)),
        n_train=int(n_train),
        n_val=int(n_val),
        n_later=int(len(later)),
        feature_cols=list(feature_cols),
        explore_up_rate=explore_up,
        threshold=float(thr),
        threshold_method=thr_name,
        later_up_rate=later_up_rate,
        later_spike_rate=later_spike_rate,
        later_auc=auc,
        later_accuracy=acc,
        later_accuracy_vs_chance=acc - majority,
        majority_baseline_acc=majority,
        n_high_later=n_high,
        high_up_hit_rate=high_up,
        high_up_lift_vs_base=_lift(high_up, later_up_rate),
        high_spike_hit_rate=high_spike,
        high_spike_lift_vs_base=_lift(high_spike, later_spike_rate),
        locked_n_fire=n_lock,
        locked_up_hit_rate=lock_up,
        locked_up_lift=_lift(lock_up, later_up_rate),
        locked_spike_hit_rate=lock_spike,
        locked_spike_lift=_lift(lock_spike, later_spike_rate),
        beats_chance_auc=beats_auc,
        beats_chance_high_prob=beats_high,
        beats_locked_up=beats_locked_up,
        beats_locked_spike=beats_locked_spike,
        best_iteration=best_iteration,
        notes=notes,
        feature_importance=feature_importance,
    )


def run_phase_c(
    samples: pd.DataFrame,
    *,
    cut_date: str = DEFAULT_CUT_DATE,
    val_frac: float = VAL_FRAC_OF_EXPLORE,
    threshold_method: str = "top_quintile",
    feature_cols: tuple[str, ...] | list[str] = FEATURE_COLS,
) -> FairTestResult:
    cols = list(feature_cols)
    # drop any residual nulls in features/label
    need = cols + [PRIMARY_LABEL, SECONDARY_LABEL]
    df = samples.dropna(subset=[c for c in need if c in samples.columns]).copy()
    explore, later, cut = split_explore_later(df, cut_date=cut_date)
    train_df, val_df, val_start = carve_explore_val(explore, val_frac=val_frac)
    logger.info(
        "split explore=%s (train=%s val=%s val_start=%s) later=%s cut=%s",
        len(explore),
        len(train_df),
        len(val_df),
        val_start.date(),
        len(later),
        cut.date(),
    )
    model, name, imp, best_it = train_classifier(
        train_df, val_df, feature_cols=cols
    )
    if best_it is not None:
        best_it = int(best_it)
    result = evaluate_fair_test(
        model,
        name,
        explore,
        later,
        cut_date=cut,
        feature_cols=cols,
        feature_importance=imp,
        n_train=len(train_df),
        n_val=len(val_df),
        threshold_method=threshold_method,
        best_iteration=best_it,
    )
    if best_it is not None and best_it == 40:
        result.notes.append(
            "Early stopping on explore-val collapsed (<10 trees); "
            "used pre-specified fallback: fixed 40 trees on full explore "
            "(train+val combined). Later window still untouched until final score."
        )
    elif best_it is not None and best_it <= 2:
        result.notes.append(
            f"Early stopping picked best_iteration={best_it} "
            "(model barely left the first tree — treat later edge as weak)."
        )
    return result


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def write_phase_c_report(
    result: FairTestResult,
    path: Path | str,
    *,
    parquet_path: str | None = None,
    script_path: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M:%S WAT")

    # Conservative plain verdict (research — no alpha).
    auc_ok = result.beats_chance_auc and result.later_auc is not None and result.later_auc >= 0.52
    if result.beats_chance_high_prob and auc_ok:
        chance_verdict = "YES — clearer than chance on later dates (AUC and high-prob)"
    elif result.beats_chance_high_prob:
        chance_verdict = "MIXED — high-prob band beat base, but overall ranking weak"
    elif result.beats_chance_auc and result.later_auc is not None and result.later_auc < 0.52:
        chance_verdict = (
            "NO — AUC only barely above 0.5 (noise territory); "
            "high-prob band did not clearly beat everyday chance"
        )
    elif result.beats_chance_auc:
        chance_verdict = (
            "MIXED — ranking a bit better than a coin flip (AUC), "
            "but the high-prob band did not clear a meaningful lift bar vs everyday chance"
        )
    else:
        chance_verdict = "NO — did not clearly beat chance on later dates"

    locked_line = "n/a"
    if result.beats_locked_up is True:
        locked_line = "YES — high-prob up-hit beat the locked rule on later"
    elif result.beats_locked_up is False:
        locked_line = "NO — high-prob up-hit did not beat the locked rule on later"

    imp_sorted = sorted(
        result.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    )
    imp_lines = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in imp_sorted) or "| — | — |"

    notes_block = ""
    if result.notes:
        notes_block = "\n".join(f"- {n}" for n in result.notes)

    body = f"""# Phase C — LightGBM fair test (Abdul)

**When:** {now} (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Model:** `{result.model_name}` (one train + one locked later score)  
**Trees used (early stop):** {result.best_iteration if result.best_iteration is not None else "n/a"}

## Short answers

- **Beats chance on later?** **{chance_verdict}**
- **Beats locked draft rule on later (up in 7d)?** **{locked_line}**
- Later AUC: **{_num(result.later_auc, 4)}** (0.5 = coin flip)
- Later accuracy: **{_pct(result.later_accuracy)}** vs everyday majority baseline **{_pct(result.majority_baseline_acc)}** (gap **{_pct(result.later_accuracy_vs_chance)}**)
- When model says “high chance” (prob ≥ {_num(result.threshold, 4)}, method `{result.threshold_method}`):  
  up-in-7d hit **{_pct(result.high_up_hit_rate)}** vs later everyday **{_pct(result.later_up_rate)}**  
  (lift **{_num(result.high_up_lift_vs_base, 3)}x**, fires **{result.n_high_later}**)
- Same high-prob band, secondary spike ≥50% in 7d: hit **{_pct(result.high_spike_hit_rate)}** vs everyday **{_pct(result.later_spike_rate)}** (lift **{_num(result.high_spike_lift_vs_base, 3)}x**)

- **Research pass bar** (beat chance **and** beat locked draft on later): **{"PASS" if (result.beats_chance_high_prob and result.beats_chance_auc and result.beats_locked_up) else "FAIL"}**

**No alpha claim.** Do not trade on this. Small / short later window.

## What we did (plain)

1. Used the Phase B table of Band C coin-days with features + labels.
2. **Explore** = dates ≤ **{result.cut_date}** ({result.n_explore:,} rows).  
   **Later** = dates after that ({result.n_later:,} rows). We did **not** peek at later while training or picking the threshold.
3. Carved a small validation slice from the **end of explore only** (train {result.n_train:,} / val {result.n_val:,}) for early stopping.
4. Trained a **small** `{result.model_name}` to predict “did price go up in 7 days?” (`up_7d`).
5. Picked a “high chance” probability cutoff on **explore only** ({result.threshold_method}).
6. Scored **later once**. Compared to everyday later chance and to the locked draft rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.

## Features the model saw

{", ".join(f"`{c}`" for c in result.feature_cols)}

## Later fair check (locked)

| Metric | Value |
|--------|------:|
| Later rows | {result.n_later:,} |
| Everyday up-in-7d rate | {_pct(result.later_up_rate)} |
| Everyday spike≥50% rate | {_pct(result.later_spike_rate)} |
| AUC | {_num(result.later_auc, 4)} |
| Accuracy (@ prob≥0.5) | {_pct(result.later_accuracy)} |
| Majority baseline accuracy | {_pct(result.majority_baseline_acc)} |
| High-prob fires | {result.n_high_later:,} |
| High-prob up hit | {_pct(result.high_up_hit_rate)} |
| High-prob up lift vs base | {_num(result.high_up_lift_vs_base, 3)}x |
| High-prob spike hit | {_pct(result.high_spike_hit_rate)} |
| High-prob spike lift vs base | {_num(result.high_spike_lift_vs_base, 3)}x |

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | {result.locked_n_fire:,} |
| Up-in-7d hit | {_pct(result.locked_up_hit_rate)} |
| Up lift vs base | {_num(result.locked_up_lift, 3)}x |
| Spike≥50% hit | {_pct(result.locked_spike_hit_rate)} |
| Spike lift vs base | {_num(result.locked_spike_lift, 3)}x |

## Feature importance (train — descriptive only)

| feature | importance |
|---------|-----------:|
{imp_lines}

## Notes

{notes_block if notes_block else "- One train run; threshold locked on explore; later scored once."}

## Files

| Artifact | Path |
|----------|------|
| Dataset parquet | `{parquet_path or "data/ai/ai_samples_daily.parquet"}` |
| This report | `{path}` |
| Repro script | `{script_path or "scripts/run_ai_train.py"}` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next move  
"""
    path.write_text(body, encoding="utf-8")
    return path


def save_result_json(result: FairTestResult, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
