"""Forward returns, MFE/MAE, benchmarks — research only, no orders."""

from cmram.backtest.engine import aggregate_backtest, run_backtest
from cmram.backtest.holdout import (
    compute_time_split,
    run_holdout_evaluation,
    write_holdout_report,
)
from cmram.backtest.persist import build_and_write_backtest

__all__ = [
    "run_backtest",
    "aggregate_backtest",
    "build_and_write_backtest",
    "compute_time_split",
    "run_holdout_evaluation",
    "write_holdout_report",
]
