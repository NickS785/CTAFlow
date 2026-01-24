# CTAFlow/features/ext/yf_data.py
"""
Yahoo Finance Data Fetcher
==========================

Thin wrapper around yfinance for macro & market data.
Designed to return clean, indexed pandas Series/DataFrames.

Examples:
    - ^TNX (10Y Treasury Yield)
    - ^IRX (3M T-Bill)
    - DXY, SPX proxies
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from typing import Optional


class YahooFinanceFetcher:
    """Fetches time-series data from Yahoo Finance."""

    def __init__(self, auto_adjust: bool = True):
        self.auto_adjust = auto_adjust

    def fetch(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        field: str = "Close",
    ) -> pd.Series:
        """
        Fetch a single price/yield series.

        Parameters
        ----------
        ticker : str
            Yahoo ticker (e.g. '^TNX')
        start, end : str
            Date range
        interval : str
            '1d', '1wk', etc.
        field : str
            Column to extract (Close, Adj Close, etc.)

        Returns
        -------
        pd.Series
        """
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        s = df[field].rename(ticker)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    def fetch_df(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return full OHLCV DataFrame."""
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
