"""Session-level return and volatility feature utilities."""
from __future__ import annotations

from datetime import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.session import DEFAULT_SESSION_TZ, us_regular


def _ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" not in df.columns:
            raise KeyError("DataFrame must have a DatetimeIndex or a 'ts' column")
        df = df.copy()
        df.index = pd.DatetimeIndex(df["ts"])
    if df.index.tz is None:
        df = df.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    else:
        df = df.tz_convert(tz)
    return df


def _group_by_session(
    intraday_df: pd.DataFrame,
    session_start: time,
    session_end: time,
    tz: str = DEFAULT_SESSION_TZ,
) -> pd.core.groupby.DataFrameGroupBy:
    localized = _ensure_datetime_index(intraday_df, tz)
    session_df = localized.between_time(session_start, session_end, inclusive="both")
    return session_df.groupby(session_df.index.normalize())


def session_returns(
    intraday_df: pd.DataFrame,
    price_col: str = "Close",
    session_start: time = us_regular()[0],
    session_end: time = us_regular()[1],
    tz: str = DEFAULT_SESSION_TZ,
) -> pd.Series:
    """Compute end-to-end percentage return for each trading session."""

    grouped = _group_by_session(intraday_df, session_start, session_end, tz)
    return grouped.apply(lambda df: df[price_col].iloc[-1] / df[price_col].iloc[0] - 1.0)


def session_realized_volatility(
    intraday_df: pd.DataFrame,
    price_col: str = "Close",
    session_start: time = us_regular()[0],
    session_end: time = us_regular()[1],
    tz: str = DEFAULT_SESSION_TZ,
) -> pd.Series:
    """Compute realised volatility for each session using intraday returns."""

    grouped = _group_by_session(intraday_df, session_start, session_end, tz)
    return grouped.apply(lambda df: np.sqrt(np.square(df[price_col].pct_change().dropna()).sum()))


def cumulative_session_returns(
    intraday_df: Optional[pd.DataFrame] = None,
    returns: Optional[pd.Series] = None,
    n_periods: Sequence[int] = (1, 5, 10),
    price_col: str = "Close",
    session_start: time = us_regular()[0],
    session_end: time = us_regular()[1],
    tz: str = DEFAULT_SESSION_TZ,
) -> pd.DataFrame:
    """Rolling cumulative session returns over ``n_periods`` sessions."""

    if returns is None:
        if intraday_df is None:
            raise ValueError("Provide intraday_df or precomputed session returns")
        returns = session_returns(intraday_df, price_col, session_start, session_end, tz)
    log_returns = np.log1p(returns)
    features = {}
    for n in n_periods:
        agg = log_returns.rolling(n).sum().shift(1)
        features[f"session_return_{n}"] = np.expm1(agg)
    return pd.DataFrame(features, index=returns.index)


def cumulative_session_volatility(
    intraday_df: Optional[pd.DataFrame] = None,
    volatility: Optional[pd.Series] = None,
    n_periods: Sequence[int] = (1, 5, 10),
    price_col: str = "Close",
    session_start: time = us_regular()[0],
    session_end: time = us_regular()[1],
    tz: str = DEFAULT_SESSION_TZ,
) -> pd.DataFrame:
    """Rolling average realised volatility over ``n_periods`` sessions."""

    if volatility is None:
        if intraday_df is None:
            raise ValueError("Provide intraday_df or precomputed session volatility")
        volatility = session_realized_volatility(intraday_df, price_col, session_start, session_end, tz)
    features = {}
    for n in n_periods:
        features[f"session_volatility_{n}"] = volatility.rolling(n).mean().shift(1)
    return pd.DataFrame(features, index=volatility.index)
