"""Signal generation over τ_m / τ_g grids."""

from cmram.signals.generate import generate_signals
from cmram.signals.persist import build_and_write_signals

__all__ = ["generate_signals", "build_and_write_signals"]
