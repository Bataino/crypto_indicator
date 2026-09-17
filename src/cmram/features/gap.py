"""Rotation Gap = MREI − NSI."""

from __future__ import annotations

import pandas as pd


def rotation_gap(mrei: pd.Series, nsi: pd.Series) -> pd.Series:
    """Primary research quantity: Rotation_Gap = MREI - NSI.

    Not a trade signal. No thresholding here.
    """
    return mrei.astype("float64") - nsi.astype("float64")
