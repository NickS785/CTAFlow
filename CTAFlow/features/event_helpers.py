"""Helper utilities for constructing event study specifications."""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_on_percentile(
    df: pd.DataFrame,
    col: str,
    p: float = 0.98,
    min_separation: int = 0,
) -> pd.Series:
    """Flag events when ``col`` exceeds its right-tail percentile.

    Parameters
    ----------
    df:
        Source DataFrame containing the signal column.
    col:
        Column name to evaluate.
    p:
        Percentile threshold (default 0.98).
    min_separation:
        Minimum number of bars between successive events.  Values are counted
        using positional distance, making the helper agnostic to the actual
        timestamp spacing.
    """

    if col not in df.columns:
        return pd.Series(False, index=df.index)

    series = df[col].dropna()
    if series.empty:
        return pd.Series(False, index=df.index)

    threshold = series.quantile(p)
    candidates = df[col] >= threshold
    candidates = candidates.fillna(False)

    if min_separation <= 0:
        return candidates

    event_mask = np.zeros(len(candidates), dtype=bool)
    true_positions = np.flatnonzero(candidates.to_numpy())
    last_position = -np.inf
    for pos in true_positions:
        if pos - last_position > min_separation:
            event_mask[pos] = True
            last_position = pos

    return pd.Series(event_mask, index=df.index)


def baseline_hourly_mean(df: pd.DataFrame, ret_col: str = "ret") -> pd.Series:
    """Return the mean return for each hour-of-day as a Series."""

    if ret_col not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")

    returns = df[ret_col]
    hour_means = returns.groupby(idx.hour).transform("mean")
    return hour_means.reindex(df.index)


__all__ = ["event_on_percentile", "baseline_hourly_mean"]

