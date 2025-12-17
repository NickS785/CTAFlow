"""Datetime calendar features for trading and futures calendars."""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..config import MONTH_NUMBER_TO_CODE
from ..data.raw_formatting.contract_specs import ContractSpecs
from ..data.raw_formatting.dly_contract_manager import calculate_contract_expiry
from ..data.ticker_classifier import TickerCategory, TickerClassifier


def _normalize_index(dates: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    """Return a normalized DatetimeIndex from any datetime-like iterable."""
    idx = pd.DatetimeIndex(dates)
    return idx.normalize()


def _last_business_day(date: pd.Timestamp) -> pd.Timestamp:
    """Compute the last business day for the month of ``date``."""
    month_end = date.to_period("M").to_timestamp("M")
    return (month_end + pd.offsets.BMonthEnd(0)).normalize()


def _third_friday(date: pd.Timestamp) -> pd.Timestamp:
    """Return the third Friday for the month of ``date``."""
    first_day = date.replace(day=1)
    first_friday = first_day + pd.offsets.Week(weekday=4)
    return (first_friday + pd.DateOffset(weeks=2)).normalize()


def _contract_months_for_ticker(ticker: str, classifier: TickerClassifier) -> List[int]:
    """Resolve the active contract months for a ticker."""
    base = ticker.replace("_F", "").upper()
    try:
        specs = ContractSpecs.load_specs(base)
        return list(sorted(set(specs.months)))
    except KeyError:
        info = classifier.get_ticker_info(f"{base}_F")
        if info and info.category == TickerCategory.FINANCIAL:
            return [3, 6, 9, 12]
        return list(range(1, 13))


def _next_expiry_date(date: pd.Timestamp, months: List[int], ticker: str) -> pd.Timestamp:
    """Find the next expiry date on or after ``date`` for the ticker's cycle."""
    months_sorted = sorted(months)
    normalized = date.normalize()
    search_year = normalized.year

    while True:
        for month in months_sorted:
            candidate_year = search_year
            if month < normalized.month:
                candidate_year += 1
            month_code = MONTH_NUMBER_TO_CODE[month]
            expiry = calculate_contract_expiry(month_code, candidate_year, ticker=ticker)
            if expiry.normalize() >= normalized:
                return expiry.normalize()
        search_year += 1


def build_datetime_features(dates: Iterable[pd.Timestamp], ticker: str) -> pd.DataFrame:
    """Create trading calendar features for the supplied dates.

    Parameters
    ----------
    dates : Iterable[pd.Timestamp]
        Calendar dates to evaluate (will be normalized to midnight).
    ticker : str
        Base ticker symbol (e.g., ``"CL"`` or ``"ES"``). The ticker is
        used to infer contract months and expiry conventions.

    Returns
    -------
    pd.DataFrame
        Columns include:
        - ``is_last_trading_week``: True when date falls in the final
          trading week of the month.
        - ``is_opex_week``: True during the week containing equity/index
          options expiration (third Friday).
        - ``days_since_opex``: Non-negative days since the most recent
          monthly options expiration (NaN before it occurs in a month).
        - ``is_expiration_week``: True when within the expiry week for the
          contract cycle of ``ticker``.
        - ``weeks_until_expiration``: Whole weeks remaining until the next
          futures expiry for the ticker.
    """
    classifier = TickerClassifier()
    idx = _normalize_index(dates)

    last_bdays = idx.map(_last_business_day)
    last_week_starts = last_bdays - pd.to_timedelta(last_bdays.weekday, unit="D")
    is_last_trading_week = (idx >= last_week_starts) & (idx <= last_bdays)

    opex_days = idx.map(_third_friday)
    opex_week_starts = opex_days - pd.to_timedelta(opex_days.weekday, unit="D")
    is_opex_week = (idx >= opex_week_starts) & (idx <= (opex_week_starts + timedelta(days=6)))
    days_since_opex = np.where(idx >= opex_days, (idx - opex_days).days, np.nan)

    contract_months = _contract_months_for_ticker(ticker, classifier)
    next_expiries = idx.map(lambda d: _next_expiry_date(d, contract_months, ticker))
    expiry_week_starts = next_expiries - pd.to_timedelta(next_expiries.weekday, unit="D")
    is_expiration_week = (idx >= expiry_week_starts) & (idx <= next_expiries)
    weeks_until_expiration = ((next_expiries - idx) // np.timedelta64(7, "D")).astype(int)

    return pd.DataFrame({
        "is_last_trading_week": is_last_trading_week,
        "is_opex_week": is_opex_week,
        "days_since_opex": days_since_opex,
        "is_expiration_week": is_expiration_week,
        "weeks_until_expiration": weeks_until_expiration,
    }, index=idx)
