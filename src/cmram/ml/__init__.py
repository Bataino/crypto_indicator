"""AI next-move dataset + Phase C train helpers. Research only — no live trading."""

from cmram.ml.dataset import (
    FEATURE_COLS,
    LABEL_COLS,
    PRIMARY_FEATURES,
    build_ai_samples,
    time_split_cut_date,
    write_ai_samples,
)

__all__ = [
    "FEATURE_COLS",
    "LABEL_COLS",
    "PRIMARY_FEATURES",
    "build_ai_samples",
    "time_split_cut_date",
    "write_ai_samples",
]
