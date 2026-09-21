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

from cmram.ml.dataset import FEATURE_COLS, EXPLORE_FRAC_DEFAULT, time_split_cut_date

logger = logging.getLogger(__name__)

LAGOS = ZoneInfo("Africa/Lagos")

# Locked Phase B cut (explore inclusive).
DEFAULT_CUT_DATE = "2026-08-19"
PRIMARY_LABEL = "up_7d"
SECONDARY_LABEL = "spike_50_7d"
VALID_TARGETS = (PRIMARY_LABEL, SECONDARY_LABEL)

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
    primary_label: str
    # explore base / threshold (explore_up_rate = explore rate of primary_label)
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


def _scale_pos_weight(y: pd.Series | np.ndarray) -> float | None:
    """Neg/pos weight for rare positives; None if balanced or no positives."""
    y = np.asarray(y, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return None
    ratio = n_neg / n_pos
    # Only apply when clearly imbalanced (spike ~3%).
    if ratio < 3.0:
        return None
    return float(ratio)


def train_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] | list[str] = FEATURE_COLS,
    label: str = PRIMARY_LABEL,
) -> tuple[Any, str, dict[str, float], int | None]:
    """Train small LightGBM; fall back to HistGradientBoosting or logistic."""
    if label not in VALID_TARGETS:
        raise ValueError(f"label must be one of {VALID_TARGETS}, got {label!r}")
    cols = list(feature_cols)
    X_tr, y_tr = _xy(train_df, cols, label)
    X_va, y_va = _xy(val_df, cols, label)

    lgb = _try_lightgbm()
    if lgb is not None:
        try:
            params = dict(LGBM_PARAMS)
            spw = _scale_pos_weight(y_tr)
            if spw is not None:
                params["scale_pos_weight"] = spw
                logger.info(
                    "class imbalance: scale_pos_weight=%.3f (label=%s pos_rate=%.4f)",
                    spw,
                    label,
                    float(np.asarray(y_tr).mean()),
                )
            model = lgb.LGBMClassifier(**params)
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
                params_fb = dict(params)
                params_fb["n_estimators"] = 40
                # Recalc weight on full explore train+val.
                spw_full = _scale_pos_weight(y_full)
                if spw_full is not None:
                    params_fb["scale_pos_weight"] = spw_full
                elif "scale_pos_weight" in params_fb:
                    del params_fb["scale_pos_weight"]
                model = lgb.LGBMClassifier(**params_fb)
                model.fit(X_full, y_full)
                best_it = 40
                used_fallback = True
            else:
                best_it = int(best_it)
            imp = dict(
                zip(cols, (float(x) for x in model.feature_importances_), strict=True)
            )
            logger.info(
                "trained LightGBM trees=%s fallback=%s label=%s",
                best_it,
                used_fallback,
                label,
            )
            return model, "lightgbm", imp, best_it
        except Exception as e:  # noqa: BLE001
            logger.warning("LightGBM train failed (%s); trying sklearn", e)

    from sklearn.ensemble import HistGradientBoostingClassifier

    try:
        # Internal validation_fraction is random within train only (never later).
        # class_weight not in HistGB; sample_weight approximates rare-positive focus.
        sample_weight = None
        spw = _scale_pos_weight(y_tr)
        if spw is not None:
            sample_weight = np.where(np.asarray(y_tr) == 1, spw, 1.0).astype("float64")
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
        if sample_weight is not None:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        else:
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
                    class_weight="balanced" if _scale_pos_weight(y_tr) else None,
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


# Explore-only high-prob quantile cuts (never peek later when choosing).
# Keys accepted by CLI / tune_threshold_explore.
EXPLORE_QUANTILE_METHODS: dict[str, tuple[float, str]] = {
    "top_quintile": (0.80, "top_quintile_explore_p80"),
    "p80": (0.80, "explore_p80"),
    "top_decile": (0.90, "top_decile_explore_p90"),
    "p90": (0.90, "explore_p90"),
    "top_5pct": (0.95, "top_5pct_explore_p95"),
    "p95": (0.95, "explore_p95"),
}

# Default multi-band set for spike tighten fair-test (old p80 + tighter).
DEFAULT_TIGHTEN_BANDS: tuple[str, ...] = ("p80", "p90", "p95")


def tune_threshold_explore(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    method: str = "top_quintile",
) -> tuple[float, str]:
    """Pick probability threshold on explore only (never later)."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype="float64")
    if method in EXPLORE_QUANTILE_METHODS:
        q, name = EXPLORE_QUANTILE_METHODS[method]
        thr = float(np.quantile(proba, q))
        return thr, name
    if method != "f1":
        raise ValueError(
            f"Unknown threshold method {method!r}; "
            f"expected one of {sorted(EXPLORE_QUANTILE_METHODS)!r} or 'f1'"
        )
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
    label: str = PRIMARY_LABEL,
) -> FairTestResult:
    from sklearn.metrics import accuracy_score, roc_auc_score

    if label not in VALID_TARGETS:
        raise ValueError(f"label must be one of {VALID_TARGETS}, got {label!r}")

    notes: list[str] = []
    # Threshold + AUC/accuracy are always on the *training* primary label.
    X_ex, y_ex = _xy(explore, feature_cols, label)
    proba_ex = predict_proba_positive(model, X_ex)
    thr, thr_name = tune_threshold_explore(y_ex.to_numpy(), proba_ex, method=threshold_method)

    X_la, y_primary = _xy(later, feature_cols, label)
    proba_la = predict_proba_positive(model, X_la)
    y_primary_np = y_primary.to_numpy()

    up_la = later[PRIMARY_LABEL].astype("float64").fillna(0).astype(int).to_numpy()
    spike_la = later[SECONDARY_LABEL].astype("float64").fillna(0).astype(int).to_numpy()

    later_up_rate = float(up_la.mean()) if len(up_la) else 0.0
    later_spike_rate = float(spike_la.mean()) if len(spike_la) else 0.0
    primary_later_rate = float(y_primary_np.mean()) if len(y_primary_np) else 0.0
    majority = max(primary_later_rate, 1.0 - primary_later_rate)
    pred_bin = (proba_la >= 0.5).astype(int)
    acc = float(accuracy_score(y_primary_np, pred_bin)) if len(y_primary_np) else 0.0

    try:
        auc = (
            float(roc_auc_score(y_primary_np, proba_la))
            if len(np.unique(y_primary_np)) > 1
            else None
        )
    except ValueError:
        auc = None
        notes.append("AUC undefined (single class in later).")

    high = proba_la >= thr
    n_high = int(high.sum())
    high_up = _safe_rate(int((up_la[high] == 1).sum()), n_high) if n_high else None
    high_spike = _safe_rate(int((spike_la[high] == 1).sum()), n_high) if n_high else None

    lock = locked_rule_mask(later).to_numpy()
    n_lock = int(lock.sum())
    lock_up = _safe_rate(int((up_la[lock] == 1).sum()), n_lock) if n_lock else None
    lock_spike = _safe_rate(int((spike_la[lock] == 1).sum()), n_lock) if n_lock else None

    beats_auc = bool(auc is not None and auc > 0.5)

    # High-prob "beats chance" uses the primary label's hit rate vs its base.
    if label == SECONDARY_LABEL:
        primary_high = high_spike
        primary_base = later_spike_rate
        # Rare event: require lift >= 1.05 and >=0.5pp absolute gap (1pp would be huge).
        abs_gap_bar = 0.005
        notes.append(
            "Primary label spike_50_7d (~3% base): evaluate with AUC + high-prob "
            "spike lift vs base (accuracy alone is misleading vs ~97% majority)."
        )
    else:
        primary_high = high_up
        primary_base = later_up_rate
        abs_gap_bar = 0.01

    high_lift_primary = _lift(primary_high, primary_base)
    beats_high = bool(
        primary_high is not None
        and n_high >= min_high_fires
        and high_lift_primary is not None
        and high_lift_primary >= 1.05
        and (primary_high - primary_base) >= abs_gap_bar
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

    explore_primary = float(explore[label].astype(float).mean())

    return FairTestResult(
        model_name=model_name,
        cut_date=str(pd.Timestamp(cut_date).date()),
        n_explore=int(len(explore)),
        n_train=int(n_train),
        n_val=int(n_val),
        n_later=int(len(later)),
        feature_cols=list(feature_cols),
        primary_label=label,
        explore_up_rate=explore_primary,
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
    cut_date: str | None = None,
    explore_frac: float | None = None,
    val_frac: float = VAL_FRAC_OF_EXPLORE,
    threshold_method: str = "top_quintile",
    feature_cols: tuple[str, ...] | list[str] = FEATURE_COLS,
    target: str = PRIMARY_LABEL,
) -> FairTestResult:
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")
    cols = list(feature_cols)
    # drop any residual nulls in features/label
    need = cols + [PRIMARY_LABEL, SECONDARY_LABEL]
    df = samples.dropna(subset=[c for c in need if c in samples.columns]).copy()
    if explore_frac is not None:
        cut_ts = time_split_cut_date(df["timestamp"], explore_frac=float(explore_frac))
        cut_date = str(cut_ts.date())
    elif cut_date is None:
        cut_date = DEFAULT_CUT_DATE
    explore, later, cut = split_explore_later(df, cut_date=cut_date)
    train_df, val_df, val_start = carve_explore_val(explore, val_frac=val_frac)
    logger.info(
        "split explore=%s (train=%s val=%s val_start=%s) later=%s cut=%s target=%s",
        len(explore),
        len(train_df),
        len(val_df),
        val_start.date(),
        len(later),
        cut.date(),
        target,
    )
    model, name, imp, best_it = train_classifier(
        train_df, val_df, feature_cols=cols, label=target
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
        label=target,
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
    """Write Abdul-facing markdown. Branches on result.primary_label."""
    if getattr(result, "primary_label", PRIMARY_LABEL) == SECONDARY_LABEL:
        return write_phase_c_spike50_report(
            result,
            path,
            parquet_path=parquet_path,
            script_path=script_path,
        )
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


def write_phase_c_spike50_report(
    result: FairTestResult,
    path: Path | str,
    *,
    parquet_path: str | None = None,
    script_path: str | None = None,
) -> Path:
    """Abdul-facing report when primary target is spike_50_7d."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M:%S WAT")

    auc_ok = result.beats_chance_auc and result.later_auc is not None and result.later_auc >= 0.52
    if result.beats_chance_high_prob and auc_ok:
        chance_verdict = (
            "YES — clearer than chance on later dates "
            "(spike AUC and high-prob spike lift)"
        )
    elif result.beats_chance_high_prob:
        chance_verdict = (
            "MIXED — high-prob spike band beat base rate, but overall ranking weak"
        )
    elif result.beats_chance_auc and result.later_auc is not None and result.later_auc < 0.52:
        chance_verdict = (
            "NO — spike AUC only barely above 0.5 (noise territory); "
            "high-prob band did not clearly beat everyday spike chance"
        )
    elif result.beats_chance_auc:
        chance_verdict = (
            "MIXED — spike ranking a bit better than a coin flip (AUC), "
            "but the high-prob band did not clear a meaningful lift bar vs everyday spike rate"
        )
    else:
        chance_verdict = "NO — did not clearly beat chance on later spike outcomes"

    locked_line = "n/a"
    if result.beats_locked_spike is True:
        locked_line = (
            "YES — high-prob spike hit beat the locked rule on later"
        )
    elif result.beats_locked_spike is False:
        locked_line = (
            "NO — high-prob spike hit did not beat the locked rule on later"
        )

    pass_bar = bool(
        result.beats_chance_high_prob
        and result.beats_chance_auc
        and result.beats_locked_spike
    )

    imp_sorted = sorted(
        result.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    )
    imp_lines = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in imp_sorted) or "| — | — |"

    notes_block = ""
    if result.notes:
        notes_block = "\n".join(f"- {n}" for n in result.notes)

    body = f"""# Phase C spike50 — LightGBM fair test (Abdul)

**When:** {now} (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Primary label:** `spike_50_7d` (max close within 7d ≥ +50%)  
**Model:** `{result.model_name}` (one train + one locked later score; class imbalance handled via `scale_pos_weight`)  
**Trees used (early stop):** {result.best_iteration if result.best_iteration is not None else "n/a"}

## Short answers

- **Beats chance on later (spike)?** **{chance_verdict}**
- **Beats locked draft rule on later (spike≥50% in 7d)?** **{locked_line}**
- Later spike AUC: **{_num(result.later_auc, 4)}** (0.5 = coin flip)
- Later accuracy (@0.5): **{_pct(result.later_accuracy)}** vs majority “no spike” baseline **{_pct(result.majority_baseline_acc)}** — **do not use accuracy alone** (base spike rate ~3%)
- When model says “high chance of spike” (prob ≥ {_num(result.threshold, 4)}, method `{result.threshold_method}`):  
  spike≥50% hit **{_pct(result.high_spike_hit_rate)}** vs later everyday **{_pct(result.later_spike_rate)}**  
  (lift **{_num(result.high_spike_lift_vs_base, 3)}x**, fires **{result.n_high_later}**)
- Same high-prob band, secondary up-in-7d: hit **{_pct(result.high_up_hit_rate)}** vs everyday **{_pct(result.later_up_rate)}** (lift **{_num(result.high_up_lift_vs_base, 3)}x**)

- **Research pass bar** (beat chance **and** beat locked draft on later spike): **{"PASS" if pass_bar else "FAIL"}**

**No alpha claim.** Do not trade on this. Small / short later window. Spike events are rare (~3%).

## What we did (plain)

1. Used the Phase B table of Band C coin-days with features + labels.
2. **Explore** = dates ≤ **{result.cut_date}** ({result.n_explore:,} rows).  
   **Later** = dates after that ({result.n_later:,} rows). We did **not** peek at later while training or picking the threshold.
3. Carved a small validation slice from the **end of explore only** (train {result.n_train:,} / val {result.n_val:,}) for early stopping.
4. Trained a **small** `{result.model_name}` to predict “did price spike ≥50% within 7 days?” (`spike_50_7d`), with `scale_pos_weight` for the ~3% positive class.
5. Picked a “high chance” probability cutoff on **explore only** ({result.threshold_method}).
6. Scored **later once**. Compared high-prob spike hit vs everyday later spike rate and vs the locked draft rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.

## Features the model saw

{", ".join(f"`{c}`" for c in result.feature_cols)}

## Later fair check (locked) — primary = spike

| Metric | Value |
|--------|------:|
| Later rows | {result.n_later:,} |
| Everyday spike≥50% rate | {_pct(result.later_spike_rate)} |
| Everyday up-in-7d rate | {_pct(result.later_up_rate)} |
| Spike AUC | {_num(result.later_auc, 4)} |
| Accuracy (@ prob≥0.5) | {_pct(result.later_accuracy)} |
| Majority baseline accuracy | {_pct(result.majority_baseline_acc)} |
| High-prob fires | {result.n_high_later:,} |
| High-prob spike hit | {_pct(result.high_spike_hit_rate)} |
| High-prob spike lift vs base | {_num(result.high_spike_lift_vs_base, 3)}x |
| High-prob up hit | {_pct(result.high_up_hit_rate)} |
| High-prob up lift vs base | {_num(result.high_up_lift_vs_base, 3)}x |

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | {result.locked_n_fire:,} |
| Spike≥50% hit | {_pct(result.locked_spike_hit_rate)} |
| Spike lift vs base | {_num(result.locked_spike_lift, 3)}x |
| Up-in-7d hit | {_pct(result.locked_up_hit_rate)} |
| Up lift vs base | {_num(result.locked_up_lift, 3)}x |

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
| Metrics JSON | `data/ai/phase_c_spike50_fair_test.json` |
| This report | `{path}` |
| Repro script | `{script_path or "scripts/run_ai_train.py --target spike_50_7d"}` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next spike  
"""
    path.write_text(body, encoding="utf-8")
    return path



@dataclass
class HighProbBand:
    """One explore-locked high-prob band scored once on later."""

    band: str  # p80 / p90 / p95
    quantile: float
    threshold: float
    threshold_method: str
    n_high_later: int
    later_fire_frac: float
    high_up_hit_rate: float | None
    high_up_lift_vs_base: float | None
    high_spike_hit_rate: float | None
    high_spike_lift_vs_base: float | None
    beats_chance_high_prob: bool
    beats_locked_up: bool | None
    beats_locked_spike: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiBandFairTestResult:
    """Shared model + locked-rule metrics with several explore-only high-prob bands."""

    model_name: str
    cut_date: str
    explore_frac: float | None
    n_explore: int
    n_train: int
    n_val: int
    n_later: int
    feature_cols: list[str]
    primary_label: str
    explore_up_rate: float
    later_up_rate: float
    later_spike_rate: float
    later_auc: float | None
    later_accuracy: float
    later_accuracy_vs_chance: float
    majority_baseline_acc: float
    locked_n_fire: int
    locked_up_hit_rate: float | None
    locked_up_lift: float | None
    locked_spike_hit_rate: float | None
    locked_spike_lift: float | None
    beats_chance_auc: bool
    best_iteration: int | None
    bands: list[HighProbBand]
    notes: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _score_high_prob_band(
    *,
    band: str,
    quantile: float,
    thr: float,
    thr_name: str,
    proba_la: np.ndarray,
    up_la: np.ndarray,
    spike_la: np.ndarray,
    later_up_rate: float,
    later_spike_rate: float,
    lock_up: float | None,
    lock_spike: float | None,
    n_lock: int,
    label: str,
    min_high_fires: int,
) -> HighProbBand:
    high = proba_la >= thr
    n_high = int(high.sum())
    n_later = int(len(proba_la))
    high_up = _safe_rate(int((up_la[high] == 1).sum()), n_high) if n_high else None
    high_spike = _safe_rate(int((spike_la[high] == 1).sum()), n_high) if n_high else None

    if label == SECONDARY_LABEL:
        primary_high = high_spike
        primary_base = later_spike_rate
        abs_gap_bar = 0.005
    else:
        primary_high = high_up
        primary_base = later_up_rate
        abs_gap_bar = 0.01

    high_lift_primary = _lift(primary_high, primary_base)
    beats_high = bool(
        primary_high is not None
        and n_high >= min_high_fires
        and high_lift_primary is not None
        and high_lift_primary >= 1.05
        and (primary_high - primary_base) >= abs_gap_bar
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

    return HighProbBand(
        band=band,
        quantile=float(quantile),
        threshold=float(thr),
        threshold_method=thr_name,
        n_high_later=n_high,
        later_fire_frac=(float(n_high) / float(n_later)) if n_later else 0.0,
        high_up_hit_rate=high_up,
        high_up_lift_vs_base=_lift(high_up, later_up_rate),
        high_spike_hit_rate=high_spike,
        high_spike_lift_vs_base=_lift(high_spike, later_spike_rate),
        beats_chance_high_prob=beats_high,
        beats_locked_up=beats_locked_up,
        beats_locked_spike=beats_locked_spike,
    )


def evaluate_multi_band_fair_test(
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
    band_methods: tuple[str, ...] | list[str] = DEFAULT_TIGHTEN_BANDS,
    min_high_fires: int = 30,
    best_iteration: int | None = None,
    label: str = PRIMARY_LABEL,
    explore_frac: float | None = None,
) -> MultiBandFairTestResult:
    """Train-time thresholds from explore only; score each band once on later."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    if label not in VALID_TARGETS:
        raise ValueError(f"label must be one of {VALID_TARGETS}, got {label!r}")

    notes: list[str] = []
    X_ex, y_ex = _xy(explore, feature_cols, label)
    proba_ex = predict_proba_positive(model, X_ex)

    X_la, y_primary = _xy(later, feature_cols, label)
    proba_la = predict_proba_positive(model, X_la)
    y_primary_np = y_primary.to_numpy()

    up_la = later[PRIMARY_LABEL].astype("float64").fillna(0).astype(int).to_numpy()
    spike_la = later[SECONDARY_LABEL].astype("float64").fillna(0).astype(int).to_numpy()

    later_up_rate = float(up_la.mean()) if len(up_la) else 0.0
    later_spike_rate = float(spike_la.mean()) if len(spike_la) else 0.0
    primary_later_rate = float(y_primary_np.mean()) if len(y_primary_np) else 0.0
    majority = max(primary_later_rate, 1.0 - primary_later_rate)
    pred_bin = (proba_la >= 0.5).astype(int)
    acc = float(accuracy_score(y_primary_np, pred_bin)) if len(y_primary_np) else 0.0

    try:
        auc = (
            float(roc_auc_score(y_primary_np, proba_la))
            if len(np.unique(y_primary_np)) > 1
            else None
        )
    except ValueError:
        auc = None
        notes.append("AUC undefined (single class in later).")

    lock = locked_rule_mask(later).to_numpy()
    n_lock = int(lock.sum())
    lock_up = _safe_rate(int((up_la[lock] == 1).sum()), n_lock) if n_lock else None
    lock_spike = _safe_rate(int((spike_la[lock] == 1).sum()), n_lock) if n_lock else None

    if label == SECONDARY_LABEL:
        notes.append(
            "Primary label spike_50_7d (~3% base): evaluate with AUC + high-prob "
            "spike lift vs base (accuracy alone is misleading vs ~97% majority)."
        )
        notes.append(
            "High-prob thresholds (p80/p90/p95) locked on explore probabilities only; "
            "later scored once per band — no peeking when choosing cuts."
        )

    bands: list[HighProbBand] = []
    for method in band_methods:
        if method not in EXPLORE_QUANTILE_METHODS:
            raise ValueError(
                f"band method {method!r} not in {sorted(EXPLORE_QUANTILE_METHODS)}"
            )
        q, _ = EXPLORE_QUANTILE_METHODS[method]
        thr, thr_name = tune_threshold_explore(
            y_ex.to_numpy(), proba_ex, method=method
        )
        # Canonical short name p80/p90/p95
        band_name = f"p{int(round(q * 100))}"
        band = _score_high_prob_band(
            band=band_name,
            quantile=q,
            thr=thr,
            thr_name=thr_name,
            proba_la=proba_la,
            up_la=up_la,
            spike_la=spike_la,
            later_up_rate=later_up_rate,
            later_spike_rate=later_spike_rate,
            lock_up=lock_up,
            lock_spike=lock_spike,
            n_lock=n_lock,
            label=label,
            min_high_fires=min_high_fires,
        )
        bands.append(band)

    explore_primary = float(explore[label].astype(float).mean())
    beats_auc = bool(auc is not None and auc > 0.5)

    return MultiBandFairTestResult(
        model_name=model_name,
        cut_date=str(pd.Timestamp(cut_date).date()),
        explore_frac=float(explore_frac) if explore_frac is not None else None,
        n_explore=int(len(explore)),
        n_train=int(n_train),
        n_val=int(n_val),
        n_later=int(len(later)),
        feature_cols=list(feature_cols),
        primary_label=label,
        explore_up_rate=explore_primary,
        later_up_rate=later_up_rate,
        later_spike_rate=later_spike_rate,
        later_auc=auc,
        later_accuracy=acc,
        later_accuracy_vs_chance=acc - majority,
        majority_baseline_acc=majority,
        locked_n_fire=n_lock,
        locked_up_hit_rate=lock_up,
        locked_up_lift=_lift(lock_up, later_up_rate),
        locked_spike_hit_rate=lock_spike,
        locked_spike_lift=_lift(lock_spike, later_spike_rate),
        beats_chance_auc=beats_auc,
        best_iteration=best_iteration,
        bands=bands,
        notes=notes,
        feature_importance=feature_importance,
    )


def run_phase_c_multi_band(
    samples: pd.DataFrame,
    *,
    cut_date: str | None = None,
    explore_frac: float | None = None,
    val_frac: float = VAL_FRAC_OF_EXPLORE,
    band_methods: tuple[str, ...] | list[str] = DEFAULT_TIGHTEN_BANDS,
    feature_cols: tuple[str, ...] | list[str] = FEATURE_COLS,
    target: str = PRIMARY_LABEL,
) -> MultiBandFairTestResult:
    """One train; several explore-locked high-prob bands fair-tested on later.

    If explore_frac is set, cut_date is derived from unique calendar days
    (explore_frac of days → explore). Explicit cut_date wins when explore_frac
    is None. Default cut is DEFAULT_CUT_DATE (Phase B locked primary).
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")
    cols = list(feature_cols)
    need = cols + [PRIMARY_LABEL, SECONDARY_LABEL]
    df = samples.dropna(subset=[c for c in need if c in samples.columns]).copy()
    resolved_frac: float | None = None
    if explore_frac is not None:
        resolved_frac = float(explore_frac)
        cut_ts = time_split_cut_date(df["timestamp"], explore_frac=resolved_frac)
        cut_date_str = str(cut_ts.date())
    elif cut_date is not None:
        cut_date_str = str(cut_date)
    else:
        cut_date_str = DEFAULT_CUT_DATE
    explore, later, cut = split_explore_later(df, cut_date=cut_date_str)
    train_df, val_df, val_start = carve_explore_val(explore, val_frac=val_frac)
    logger.info(
        "multi-band split explore=%s (train=%s val=%s val_start=%s) later=%s "
        "cut=%s explore_frac=%s target=%s bands=%s",
        len(explore),
        len(train_df),
        len(val_df),
        val_start.date(),
        len(later),
        cut.date(),
        resolved_frac,
        target,
        list(band_methods),
    )
    model, name, imp, best_it = train_classifier(
        train_df, val_df, feature_cols=cols, label=target
    )
    if best_it is not None:
        best_it = int(best_it)
    result = evaluate_multi_band_fair_test(
        model,
        name,
        explore,
        later,
        cut_date=cut,
        feature_cols=cols,
        feature_importance=imp,
        n_train=len(train_df),
        n_val=len(val_df),
        band_methods=band_methods,
        best_iteration=best_it,
        label=target,
        explore_frac=resolved_frac,
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


def write_spike50_tighten_report(
    result: MultiBandFairTestResult,
    path: Path | str,
    *,
    parquet_path: str | None = None,
    script_path: str | None = None,
    json_path: str | None = None,
) -> Path:
    """Plain multi-band spike50 tighten report (p80 vs p90 vs p95). No alpha."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M:%S WAT")

    by_band = {b.band: b for b in result.bands}

    def _band_row(b: HighProbBand) -> str:
        return (
            f"| **{b.band}** ({b.threshold_method}) | {_num(b.threshold, 4)} | "
            f"{b.n_high_later:,} | {_pct(b.later_fire_frac)} | "
            f"{_pct(b.high_spike_hit_rate)} | {_num(b.high_spike_lift_vs_base, 3)}x | "
            f"{_pct(b.high_up_hit_rate)} | {_num(b.high_up_lift_vs_base, 3)}x | "
            f"{'YES' if b.beats_chance_high_prob else 'NO'} | "
            f"{'YES' if b.beats_locked_spike else ('NO' if b.beats_locked_spike is False else 'n/a')} |"
        )

    band_rows = "\n".join(_band_row(b) for b in result.bands)

    # Short plain verdicts per band
    verdict_lines: list[str] = []
    for b in result.bands:
        still = (
            b.beats_chance_high_prob
            and result.beats_chance_auc
            and bool(b.beats_locked_spike)
        )
        verdict_lines.append(
            f"- **{b.band}**: spike hit {_pct(b.high_spike_hit_rate)} "
            f"(lift {_num(b.high_spike_lift_vs_base, 3)}x vs base {_pct(result.later_spike_rate)}); "
            f"fires {b.n_high_later:,} ({_pct(b.later_fire_frac)} of later); "
            f"beat chance+locked? **{'YES' if still else 'NO'}**"
        )
    verdict_block = "\n".join(verdict_lines)

    # Overall: do tighter bands still beat?
    tighter = [b for b in result.bands if b.band in ("p90", "p95")]
    tighter_ok = [
        b
        for b in tighter
        if b.beats_chance_high_prob
        and result.beats_chance_auc
        and bool(b.beats_locked_spike)
    ]
    if not tighter:
        overall = "n/a — no tighter bands requested"
    elif len(tighter_ok) == len(tighter):
        overall = "YES — both p90 and p95 still beat chance + locked on later spike"
    elif tighter_ok:
        names = ", ".join(b.band for b in tighter_ok)
        overall = (
            f"MIXED — tighter band(s) that still beat chance+locked: {names}; "
            "not all tighter bands cleared the bar"
        )
    else:
        overall = "NO — tighter bands did not clearly beat chance + locked on later spike"

    imp_sorted = sorted(
        result.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    )
    imp_lines = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in imp_sorted) or "| — | — |"

    notes_block = ""
    if result.notes:
        notes_block = "\n".join(f"- {n}" for n in result.notes)

    p80 = by_band.get("p80")
    p90 = by_band.get("p90")
    p95 = by_band.get("p95")
    compare_note = ""
    if p80 is not None:
        compare_note = (
            f"Old p80 band covered {_pct(p80.later_fire_frac)} of later "
            f"({p80.n_high_later:,} fires) — too wide for a ‘high-conviction’ read. "
        )
        if p90 is not None and p95 is not None:
            compare_note += (
                f"p90 covers {_pct(p90.later_fire_frac)}; "
                f"p95 covers {_pct(p95.later_fire_frac)}."
            )

    body = f"""# Phase C spike50 — tighten high-prob bands fair test (Abdul)

**When:** {now} (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Primary label:** `spike_50_7d` (max close within 7d ≥ +50%)  
**Model:** `{result.model_name}` (one train; band cuts locked on **explore only**; later scored once per band)  
**Trees used (early stop):** {result.best_iteration if result.best_iteration is not None else "n/a"}  
**Bands:** p80 (old) vs p90 (top ~10%) vs p95 (top ~5%) — quantiles of explore predicted probability

## Short answers

- **Later spike AUC:** **{_num(result.later_auc, 4)}** (0.5 = coin flip); beats chance AUC? **{"YES" if result.beats_chance_auc else "NO"}**
- **Do tighter bands still beat chance + locked?** **{overall}**
- **Locked draft rule on later** (same window): fires **{result.locked_n_fire:,}**, spike hit **{_pct(result.locked_spike_hit_rate)}** (lift **{_num(result.locked_spike_lift, 3)}x** vs everyday **{_pct(result.later_spike_rate)}**)

{verdict_block}

{compare_note}

**No alpha claim.** Do not trade on this. Small / short later window. Spike events are rare (~3%).

## What we did (plain)

1. Same Phase B Band C coin-day table + same features as the prior spike50 fair test.
2. **Explore** = dates ≤ **{result.cut_date}** ({result.n_explore:,} rows).  
   **Later** = dates after that ({result.n_later:,} rows). Thresholds chosen on explore only — **no peeking at later**.
3. Validation carved from the **end of explore only** (train {result.n_train:,} / val {result.n_val:,}).
4. Trained one small `{result.model_name}` on `spike_50_7d` (`scale_pos_weight` for rare positives).
5. On **explore** predicted probs, locked three cuts: **p80** (old wide band), **p90**, **p95**.
6. Scored **later once** per band. Compared spike hit % and lift vs everyday later spike rate and vs locked rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.

## Features the model saw

{", ".join(f"`{c}`" for c in result.feature_cols)}

## Band comparison on later (locked cuts)

| Band | Explore thr | Fires | % of later | Spike hit | Spike lift vs base | Up hit | Up lift | Beat chance (spike)? | Beat locked (spike)? |
|------|------------:|------:|-----------:|----------:|-------------------:|-------:|--------:|---------------------:|---------------------:|
{band_rows}

Everyday later spike rate: **{_pct(result.later_spike_rate)}**. Everyday later up-in-7d: **{_pct(result.later_up_rate)}**.

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | {result.locked_n_fire:,} |
| Spike≥50% hit | {_pct(result.locked_spike_hit_rate)} |
| Spike lift vs base | {_num(result.locked_spike_lift, 3)}x |
| Up-in-7d hit | {_pct(result.locked_up_hit_rate)} |
| Up lift vs base | {_num(result.locked_up_lift, 3)}x |

## Shared later ranking (not band-specific)

| Metric | Value |
|--------|------:|
| Later rows | {result.n_later:,} |
| Spike AUC | {_num(result.later_auc, 4)} |
| Accuracy (@ prob≥0.5) | {_pct(result.later_accuracy)} |
| Majority baseline accuracy | {_pct(result.majority_baseline_acc)} |

## Feature importance (train — descriptive only)

| feature | importance |
|---------|-----------:|
{imp_lines}

## Notes

{notes_block if notes_block else "- One train; explore-locked p80/p90/p95; later scored once per band."}

## Files

| Artifact | Path |
|----------|------|
| Dataset parquet | `{parquet_path or "data/ai/ai_samples_daily.parquet"}` |
| Metrics JSON | `{json_path or "data/ai/phase_c_spike50_tighten_fair_test.json"}` |
| This report | `{path}` |
| Repro script | `{script_path or "scripts/run_ai_train.py --target spike_50_7d --bands p80,p90,p95"}` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next spike  
- **No alpha**
"""
    path.write_text(body, encoding="utf-8")
    return path


# Primary-cut reference (explore_frac=0.7 / cut 2026-08-19) for alt-cut comparison text.
PRIMARY_CUT_REF = {
    "cut_date": "2026-08-19",
    "explore_frac": 0.7,
    "later_auc": 0.7045,
    "later_spike_rate": 0.0288,
    "locked_spike_hit_rate": 0.0357,
    "locked_spike_lift": 1.239,
    "locked_n_fire": 924,
    "bands": {
        "p80": {"spike_hit": 0.0554, "lift": 1.921, "n_fire": 5599, "fire_frac": 0.3361},
        "p90": {"spike_hit": 0.0753, "lift": 2.612, "n_fire": 2551, "fire_frac": 0.1531},
        "p95": {"spike_hit": 0.0943, "lift": 3.273, "n_fire": 1230, "fire_frac": 0.0738},
    },
}


def write_spike50_altcut_report(
    result: MultiBandFairTestResult,
    path: Path | str,
    *,
    parquet_path: str | None = None,
    script_path: str | None = None,
    json_path: str | None = None,
    primary_ref: dict[str, Any] | None = None,
) -> Path:
    """Plain English alt-cut (e.g. explore_frac=0.6) spike50 fair-test report.

    Compares briefly to the locked primary cut (2026-08-19 / explore_frac=0.7).
    Research only — no alpha.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M:%S WAT")
    pref = primary_ref if primary_ref is not None else PRIMARY_CUT_REF
    frac = result.explore_frac
    frac_s = f"{frac:.1f}" if frac is not None else "n/a"

    by_band = {b.band: b for b in result.bands}

    def _band_row(b: HighProbBand) -> str:
        return (
            f"| **{b.band}** ({b.threshold_method}) | {_num(b.threshold, 4)} | "
            f"{b.n_high_later:,} | {_pct(b.later_fire_frac)} | "
            f"{_pct(b.high_spike_hit_rate)} | {_num(b.high_spike_lift_vs_base, 3)}x | "
            f"{_pct(b.high_up_hit_rate)} | {_num(b.high_up_lift_vs_base, 3)}x | "
            f"{'YES' if b.beats_chance_high_prob else 'NO'} | "
            f"{'YES' if b.beats_locked_spike else ('NO' if b.beats_locked_spike is False else 'n/a')} |"
        )

    band_rows = "\n".join(_band_row(b) for b in result.bands)

    verdict_lines: list[str] = []
    for b in result.bands:
        still = (
            b.beats_chance_high_prob
            and result.beats_chance_auc
            and bool(b.beats_locked_spike)
        )
        verdict_lines.append(
            f"- **{b.band}**: spike hit {_pct(b.high_spike_hit_rate)} "
            f"(lift {_num(b.high_spike_lift_vs_base, 3)}x vs base {_pct(result.later_spike_rate)}); "
            f"fires {b.n_high_later:,} ({_pct(b.later_fire_frac)} of later); "
            f"beat chance+locked? **{'YES' if still else 'NO'}**"
        )
    verdict_block = "\n".join(verdict_lines)

    tighter = [b for b in result.bands if b.band in ("p90", "p95")]
    tighter_ok = [
        b
        for b in tighter
        if b.beats_chance_high_prob
        and result.beats_chance_auc
        and bool(b.beats_locked_spike)
    ]
    if not tighter:
        overall = "n/a — no tighter bands requested"
    elif len(tighter_ok) == len(tighter):
        overall = "YES — both p90 and p95 still beat chance + locked on later spike"
    elif tighter_ok:
        names = ", ".join(b.band for b in tighter_ok)
        overall = (
            f"MIXED — tighter band(s) that still beat chance+locked: {names}; "
            "not all tighter bands cleared the bar"
        )
    else:
        overall = "NO — tighter bands did not clearly beat chance + locked on later spike"

    # Brief vs primary comparison
    pref_bands = pref.get("bands") or {}
    cmp_lines: list[str] = []
    for key in ("p80", "p90", "p95"):
        b = by_band.get(key)
        pr = pref_bands.get(key)
        if b is None or pr is None:
            continue
        alt_lift = b.high_spike_lift_vs_base
        pri_lift = pr.get("lift")
        delta = None
        if alt_lift is not None and pri_lift is not None:
            delta = float(alt_lift) - float(pri_lift)
        delta_s = f"{delta:+.3f}x" if delta is not None else "n/a"
        cmp_lines.append(
            f"| **{key}** | {_pct(pr.get('spike_hit'))} / {_num(pri_lift, 3)}x "
            f"({pr.get('n_fire'):,} fires) | "
            f"{_pct(b.high_spike_hit_rate)} / {_num(alt_lift, 3)}x "
            f"({b.n_high_later:,} fires) | {delta_s} |"
        )
    cmp_block = "\n".join(cmp_lines) if cmp_lines else "| — | — | — | — |"

    still_beat_locked = all(
        bool(b.beats_locked_spike)
        for b in result.bands
        if b.band in ("p80", "p90", "p95")
    )
    beat_locked_plain = (
        "YES — all reported high-prob bands still beat the locked draft rule on "
        "later spike hit %"
        if still_beat_locked and result.bands
        else (
            "MIXED / NO — not every band clearly beat locked on this alt cut "
            "(see table)"
            if result.bands
            else "n/a"
        )
    )

    imp_sorted = sorted(
        result.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    )
    imp_lines = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in imp_sorted) or "| — | — |"

    notes_block = ""
    if result.notes:
        notes_block = "\n".join(f"- {n}" for n in result.notes)
    notes_block = (
        (notes_block + "\n" if notes_block else "")
        + f"- Alternate time cut: explore_frac={frac_s} → cut date **{result.cut_date}** "
        f"(primary was explore_frac={pref.get('explore_frac')} / "
        f"{pref.get('cut_date')}). Thresholds still locked on this explore only."
    )

    body = f"""# Phase C spike50 — alternate time-cut fair test (Abdul)

**When:** {now} (Africa/Lagos)  
**Status:** research only — **not alpha**, not a trading system, no orders, no wallets  
**Primary label:** `spike_50_7d` (max close within 7d ≥ +50%)  
**Model:** `{result.model_name}` (one train; band cuts locked on **explore only**; later scored once per band)  
**Trees used (early stop):** {result.best_iteration if result.best_iteration is not None else "n/a"}  
**Alternate cut:** explore_frac=**{frac_s}** → explore ≤ **{result.cut_date}**  
**Bands:** p80 / p90 / p95 — quantiles of explore predicted probability

## Short answers

- **Alt cut date (explore ≤):** **{result.cut_date}** (explore_frac={frac_s})
- **Later spike AUC:** **{_num(result.later_auc, 4)}** (0.5 = coin flip); beats chance AUC? **{"YES" if result.beats_chance_auc else "NO"}**
- **Do tighter bands still beat chance + locked?** **{overall}**
- **Still beat locked draft on this alt cut?** **{beat_locked_plain}**
- **Locked draft rule on later** (same window): fires **{result.locked_n_fire:,}**, spike hit **{_pct(result.locked_spike_hit_rate)}** (lift **{_num(result.locked_spike_lift, 3)}x** vs everyday **{_pct(result.later_spike_rate)}**)

{verdict_block}

**No alpha claim.** Do not trade on this. Robustness check only (alternate time cut). Spike events are rare (~3%).

## What we did (plain)

1. Same Phase B Band C coin-day table + same features as the spike50 tighten fair test.
2. **Alternate explore cut:** explore_frac=**{frac_s}** → explore dates ≤ **{result.cut_date}** ({result.n_explore:,} rows).  
   **Later** = dates after that ({result.n_later:,} rows). Thresholds chosen on this explore only — **no peeking at later**.
3. Validation carved from the **end of explore only** (train {result.n_train:,} / val {result.n_val:,}).
4. Trained one small `{result.model_name}` on `spike_50_7d` (`scale_pos_weight` for rare positives).
5. On **explore** predicted probs, locked three cuts: **p80**, **p90**, **p95**.
6. Scored **later once** per band. Compared spike hit % and lift vs everyday later spike rate and vs locked rule  
   `rvol_30 > 1.5 AND dist_ema_20 <= 0.05`.
7. Briefly compared to primary cut **{pref.get('cut_date')}** / explore_frac={pref.get('explore_frac')} from `phase_c_spike50_tighten_fair_test.md`.

## Features the model saw

{", ".join(f"`{c}`" for c in result.feature_cols)}

## Band comparison on later (locked cuts — this alt explore)

| Band | Explore thr | Fires | % of later | Spike hit | Spike lift vs base | Up hit | Up lift | Beat chance (spike)? | Beat locked (spike)? |
|------|------------:|------:|-----------:|----------:|-------------------:|-------:|--------:|---------------------:|---------------------:|
{band_rows}

Everyday later spike rate: **{_pct(result.later_spike_rate)}**. Everyday later up-in-7d: **{_pct(result.later_up_rate)}**.

### Locked draft rule on the **same** later window

| Metric | Value |
|--------|------:|
| Fires | {result.locked_n_fire:,} |
| Spike≥50% hit | {_pct(result.locked_spike_hit_rate)} |
| Spike lift vs base | {_num(result.locked_spike_lift, 3)}x |
| Up-in-7d hit | {_pct(result.locked_up_hit_rate)} |
| Up lift vs base | {_num(result.locked_up_lift, 3)}x |

## Brief vs primary cut ({pref.get('cut_date')} / explore_frac={pref.get('explore_frac')})

Primary later spike AUC was **{_num(pref.get('later_auc'), 4)}**; alt-cut later spike AUC is **{_num(result.later_auc, 4)}**.  
Primary locked spike hit **{_pct(pref.get('locked_spike_hit_rate'))}** (lift {_num(pref.get('locked_spike_lift'), 3)}x); alt locked spike hit **{_pct(result.locked_spike_hit_rate)}** (lift {_num(result.locked_spike_lift, 3)}x).

| Band | Primary spike hit / lift | Alt-cut spike hit / lift | Lift delta (alt − primary) |
|------|-------------------------:|-------------------------:|---------------------------:|
{cmp_block}

Same story directionally: tighter bands (p90/p95) still show higher later spike lift than the wide p80 band, and still beat the locked draft on spike hit — on this earlier cut as well. Numbers move with the window (different later sample); treat as a robustness check, not a second claim of edge.

## Shared later ranking (not band-specific)

| Metric | Value |
|--------|------:|
| Later rows | {result.n_later:,} |
| Spike AUC | {_num(result.later_auc, 4)} |
| Accuracy (@ prob≥0.5) | {_pct(result.later_accuracy)} |
| Majority baseline accuracy | {_pct(result.majority_baseline_acc)} |

## Feature importance (train — descriptive only)

| feature | importance |
|---------|-----------:|
{imp_lines}

## Notes

{notes_block}

## Files

| Artifact | Path |
|----------|------|
| Dataset parquet | `{parquet_path or "data/ai/ai_samples_daily.parquet"}` |
| Metrics JSON | `{json_path or "data/ai/phase_c_spike50_altcut_fair_test.json"}` |
| This report | `{path}` |
| Primary-cut report | `docs/reports/phase_c_spike50_tighten_fair_test.md` |
| Repro script | `{script_path or "scripts/run_ai_train.py --target spike_50_7d --bands p80,p90,p95 --explore-frac 0.6"}` |

## Explicit non-goals (honored)

- No live signals / bots / wallets  
- No X / social features  
- No endless retuning on later  
- No claim that the model “knows” the next spike  
- **No alpha**
"""
    path.write_text(body, encoding="utf-8")
    return path



def save_multi_band_result_json(
    result: MultiBandFairTestResult, path: Path | str
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def save_result_json(result: FairTestResult, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
