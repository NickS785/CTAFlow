"""Session filtering utilities for intraday data."""
from __future__ import annotations

from datetime import time
from typing import Union
from functools import lru_cache

import pandas as pd


__all__ = ["filter_session_ticks", "filter_session_bars"]

_TimeLike = Union[str, time]

# Cache for parsed time objects to avoid repeated parsing
@lru_cache(maxsize=128)
def _parse_time(value: _TimeLike) -> time:
    """Return a :class:`datetime.time` instance for ``value``."""

    if isinstance(value, time):
        return value

    # Convert string to time - cache result for repeated calls
    parsed = pd.to_datetime(value).time()
    if parsed.tzinfo is not None:
        # Drop timezone information if provided; session boundaries are clock times.
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _time_to_micros(value: time) -> int:
    """Convert ``value`` to microseconds from midnight."""

    return (
        ((value.hour * 60 + value.minute) * 60 + value.second) * 1_000_000
        + value.microsecond
    )


def _session_mask(times: pd.Series, start: time, end: time) -> pd.Series:
    """Boolean mask for rows with clock times inside the session window."""

    secs = (
        times.dt.hour.astype("int64") * 3_600_000_000
        + times.dt.minute.astype("int64") * 60_000_000
        + times.dt.second.astype("int64") * 1_000_000
        + times.dt.microsecond.astype("int64")
    )
    start_us = _time_to_micros(start)
    end_us = _time_to_micros(end)

    if start_us <= end_us:
        return (secs >= start_us) & (secs <= end_us)
    return (secs >= start_us) | (secs <= end_us)


def _ensure_series_tz(series: pd.Series, tz: str) -> pd.Series:
    """Ensure ``series`` is timezone-aware in ``tz``."""
    # Avoid extra Series wrapper - work directly with input
    if not isinstance(series.dtype, pd.DatetimeTZDtype) and series.dtype != 'datetime64[ns]':
        localized = pd.to_datetime(series)
    else:
        localized = series

    if localized.dt.tz is None:
        localized = localized.dt.tz_localize(tz)
    else:
        localized = localized.dt.tz_convert(tz)
    return localized


def filter_session_ticks(
    ticks: pd.DataFrame,
    tz: str,
    start: _TimeLike,
    end: _TimeLike,
) -> pd.DataFrame:
    """Return ticks whose timestamps fall inside ``[start, end]`` in ``tz``.

    Parameters
    ----------
    ticks:
        DataFrame with a ``ts`` column.
    tz:
        Olson timezone string representing the local session timezone.
    start, end:
        Session start/end times as ``"HH:MM"`` strings or :class:`datetime.time`.
    """

    if ticks.empty:
        return ticks.copy()

    session_start = _parse_time(start)
    session_end = _parse_time(end)

    # Convert timezone once and reuse
    timestamps = _ensure_series_tz(ticks["ts"], tz)

    # Create session mask
    mask = _session_mask(timestamps, session_start, session_end)
    if not mask.any():
        # Return empty DataFrame with correct structure
        return ticks.iloc[0:0].copy()

    # Single copy operation with filtered data
    filtered = ticks.loc[mask].copy()

    # Update timestamp column with timezone-aware version (reuse filtered timestamps)
    filtered["ts"] = pd.DatetimeIndex(timestamps.loc[mask].to_numpy())

    return filtered


def filter_session_bars(
    bars: pd.DataFrame,
    tz: str,
    start: _TimeLike,
    end: _TimeLike,
) -> pd.DataFrame:
    """Filter intraday bars by the local session window."""

    if bars.empty:
        return bars.copy()

    session_start = _parse_time(start)
    session_end = _parse_time(end)

    if isinstance(bars.index, pd.DatetimeIndex):
        timestamps = bars.index
    elif "ts" in bars.columns:
        timestamps = bars["ts"]
    else:
        raise KeyError("Bars must have a DatetimeIndex or a 'ts' column")

    localized = _ensure_series_tz(pd.Series(timestamps), tz)
    mask = _session_mask(localized, session_start, session_end)
    if not mask.any():
        empty = bars.iloc[0:0].copy()
        empty_index = pd.DatetimeIndex(localized.loc[mask].to_numpy())
        empty.index = empty_index
        empty["ts"] = empty_index
        return empty

    filtered = bars.loc[mask.values].copy()
    filtered_index = pd.DatetimeIndex(localized.loc[mask].to_numpy())
    filtered.index = filtered_index
    filtered["ts"] = filtered_index
    return filtered
