"""Session filtering utilities for intraday data.

This module centralizes common session definitions, default timezones, and
helpers for slicing intraday data into local trading windows.
"""
from __future__ import annotations

from datetime import time
from typing import Dict, Tuple, Union
from functools import lru_cache

import pandas as pd

__all__ = [
    "DEFAULT_SESSION_TZ",
    "TRADING_SESSIONS",
    "get_session_window",
    "filter_session_ticks",
    "filter_session_bars",
]

_TimeLike = Union[str, time]


# Default to CME (Chicago) hours unless otherwise specified
DEFAULT_SESSION_TZ = "America/Chicago"


def us_regular():
    """US regular trading hours (RTH) for equities and equity futures."""

    return time(8, 30), time(15, 0)

def asia():

    return (time(18, 30), time(3, 0))

def london():
    return (time(2,30), time(11, 0))


# Canonical session definitions for downstream features
TRADING_SESSIONS: Dict[str, Tuple[time, time]] = {
    "us_rth": us_regular(),
    "asia": asia(),
    "london": london(),
}


def get_session_window(name: str) -> Tuple[time, time]:
    """Return the (start, end) tuple for a named session.

    Parameters
    ----------
    name : str
        Key in :data:`TRADING_SESSIONS` (case-insensitive).
    """

    key = name.lower()
    if key not in TRADING_SESSIONS:
        raise KeyError(f"Unknown session '{name}'. Valid options: {sorted(TRADING_SESSIONS)}")
    return TRADING_SESSIONS[key]



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
        localized = localized.dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
    else:
        localized = localized.dt.tz_convert(tz)
    return localized


def filter_session_ticks(
    ticks: pd.DataFrame,
    tz: str = DEFAULT_SESSION_TZ,
    start: _TimeLike = us_regular()[0],
    end: _TimeLike = us_regular()[1],
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
    tz: str = DEFAULT_SESSION_TZ,
    start: _TimeLike = us_regular()[0],
    end: _TimeLike = us_regular()[1],
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
        empty_index = pd.DatetimeIndex(localized.values[mask.values])
        empty.index = empty_index
        empty["ts"] = empty_index
        return empty

    filtered = bars.loc[mask.values].copy()
    filtered_index = pd.DatetimeIndex(localized.values[mask.values])
    filtered.index = filtered_index
    filtered["ts"] = filtered_index
    return filtered
