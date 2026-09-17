"""Cross-sectional percentile ranks within (timestamp, band).

With tiny n (Phase 1 DuckDB has 3 assets) percentiles only take a few values
(0 / 50 / 100 plus ties). Ranks are deterministic: pandas average-rank,
stable input order does not change the numeric rank of a value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many names in the (timestamp, band) slice we flag small_sample.
DEFAULT_SMALL_SAMPLE_N = 10


def cs_percentile(values: pd.Series) -> pd.Series:
    """Map a 1-d slice to [0, 100] via average ranks.

    Formula: ``100 * (rank_avg - 1) / (n - 1)`` where ``n`` is the count of
    non-null values. ``n == 1`` → 50 (undefined percentile; midpoint).
    Nulls stay null and do not participate. Ties share the average rank so
    equal inputs always get equal scores.
    """
    values = pd.to_numeric(values, errors="coerce")
    valid = values.notna()
    n = int(valid.sum())
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    if n == 0:
        return out
    if n == 1:
        out.loc[valid] = 50.0
        return out
    ranks = values.rank(method="average", na_option="keep", ascending=True)
    return (ranks - 1.0) / (n - 1.0) * 100.0


def rank_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    group_keys: tuple[str, str] = ("timestamp", "band"),
) -> pd.DataFrame:
    """Add ``{col}_pct`` CS percentiles within ``group_keys``."""
    out = df
    for col in columns:
        if col not in out.columns:
            raise KeyError(col)
        out[f"{col}_pct"] = out.groupby(list(group_keys), sort=False)[col].transform(
            cs_percentile
        )
    return out


def mean_available(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Equal-weight mean of columns, skipping NaNs (renormalize). All-NaN → NaN."""
    if not columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[columns].mean(axis=1, skipna=True)


def attach_sample_size(
    df: pd.DataFrame,
    *,
    group_keys: tuple[str, str] = ("timestamp", "band"),
    small_sample_n: int = DEFAULT_SMALL_SAMPLE_N,
) -> pd.DataFrame:
    """``n_in_band`` = distinct assets in the (timestamp, band) slice."""
    out = df
    n = out.groupby(list(group_keys), sort=False)["asset_id"].transform("nunique")
    out["n_in_band"] = n.astype("int64")
    out["small_sample"] = out["n_in_band"] < int(small_sample_n)
    return out
