"""Forward returns, MFE/MAE, benchmarks — research only, no orders."""

from cmram.backtest.engine import aggregate_backtest, run_backtest
from cmram.backtest.holdout import (
    compute_time_split,
    run_holdout_evaluation,
    write_holdout_report,
)
from cmram.backtest.persist import build_and_write_backtest
from cmram.backtest.n_only_validation import run_n_only_validation
from cmram.backtest.spike_detect import detect_spikes, forward_max_return
from cmram.backtest.spike_backward import run_spike_backward, write_spike_backward_report
from cmram.backtest.spike_shared_setups import run_spike_shared_setups, write_shared_setups_report
from cmram.backtest.volume_expansion import run_volume_expansion_study, write_volume_expansion_report
from cmram.backtest.draft_formula_fair_test import (
    run_draft_formula_fair_test,
    write_draft_formula_fair_test_report,
)

__all__ = [
    "run_backtest",
    "aggregate_backtest",
    "build_and_write_backtest",
    "compute_time_split",
    "run_holdout_evaluation",
    "write_holdout_report",
    "run_n_only_validation",
    "detect_spikes",
    "forward_max_return",
    "run_spike_backward",
    "write_spike_backward_report",
    "run_spike_shared_setups",
    "write_shared_setups_report",
    "run_volume_expansion_study",
    "write_volume_expansion_report",
    "run_draft_formula_fair_test",
    "write_draft_formula_fair_test_report",
]
