"""Micro-Cap Rotation Expansion Index (MREI).

MREI = equal-weight mean of available C, V, N, P (each already 0–100).
If N is missing (``n_available=false``), renormalize over the remaining
legs — do not invent social.

Phase 1:
  Model A: only P is passed (C/V set null) → MREI = score_P
  Model B: C, V, P (N null) → MREI = mean(C, V, P)
"""

from __future__ import annotations

import pandas as pd

from cmram.features.rank import mean_available


def compute_mrei(
    score_C: pd.Series,
    score_V: pd.Series,
    score_P: pd.Series,
    score_N: pd.Series | None = None,
    *,
    n_available: pd.Series | bool | None = None,
) -> pd.Series:
    """MREI = equal-weight mean of available C, V, N, P (0–100)."""
    parts: dict[str, pd.Series] = {
        "C": score_C,
        "V": score_V,
        "P": score_P,
    }
    if score_N is not None:
        n = score_N
        if isinstance(n_available, bool) and not n_available:
            n = None
        elif isinstance(n_available, pd.Series):
            flag = n_available.reindex(score_N.index).fillna(False).astype(bool)
            n = score_N.where(flag)
        if n is not None:
            parts["N"] = n
    df = pd.DataFrame(parts)
    return mean_available(df, list(parts.keys()))
