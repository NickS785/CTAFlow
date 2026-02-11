from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None  # type: ignore

try:
    import pandas_datareader.data as web
except Exception:  # pragma: no cover - optional dependency
    web = None  # type: ignore


class MacroClient:
    """
    Unified client for retrieving market-state context.

    Sources:
      - Yahoo Finance for index/ETF closes
      - FRED for rates/yield curves
    """

    FRED_SERIES: Dict[str, str] = {
        "YIELD_10Y": "DGS10",
        "YIELD_2Y": "DGS2",
    }

    MARKET_TICKERS: Dict[str, str] = {
        "SPX": "^GSPC",
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
    }

    SECTOR_TICKERS: Dict[str, str] = {
        "XLE": "XLE",
        "XLF": "XLF",
        "XLK": "XLK",
        "XLV": "XLV",
        "XLI": "XLI",
        "XLP": "XLP",
        "XLY": "XLY",
        "XLU": "XLU",
        "XLB": "XLB",
        "XLRE": "XLRE",
        "XLC": "XLC",
    }

    def __init__(self, fred_api_key: Optional[str] = None, auto_adjust: bool = True):
        self.fred_api_key = fred_api_key
        self.auto_adjust = bool(auto_adjust)

    def fetch_fred_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch FRED macro series. Returns empty frame if dependency/source is unavailable."""
        if web is None:
            return pd.DataFrame()

        try:
            end = end_date or datetime.utcnow()
            raw = web.DataReader(
                list(self.FRED_SERIES.values()),
                "fred",
                start_date,
                end,
                api_key=self.fred_api_key,
            )
            out = raw.rename(columns={v: k for k, v in self.FRED_SERIES.items()})
            out.index = pd.to_datetime(out.index)
            return out.sort_index()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _extract_close(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            level0 = set(data.columns.get_level_values(0))
            if "Close" in level0:
                close = data["Close"]
            elif "Adj Close" in level0:
                close = data["Adj Close"]
            else:
                raise KeyError("Expected 'Close' or 'Adj Close' in yfinance output.")
            if isinstance(close, pd.Series):
                close = close.to_frame()
            return close

        if "Close" in data.columns:
            return data[["Close"]]
        if "Adj Close" in data.columns:
            return data[["Adj Close"]].rename(columns={"Adj Close": "Close"})
        raise KeyError("Expected 'Close' or 'Adj Close' in yfinance output.")

    def _fetch_yahoo_close(
        self,
        ticker_map: Dict[str, str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()

        tickers = list(ticker_map.values())
        if not tickers:
            return pd.DataFrame()

        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=self.auto_adjust,
            progress=False,
        )
        close = self._extract_close(raw)
        if close.empty:
            return pd.DataFrame()

        if close.shape[1] == 1 and len(tickers) == 1:
            close.columns = [tickers[0]]

        alias_map = {ticker: alias for alias, ticker in ticker_map.items()}
        close = close.rename(columns=alias_map)
        close.index = pd.to_datetime(close.index)
        return close.sort_index()

    def fetch_market_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch market indices and sector ETFs as one aligned frame."""
        all_tickers = {**self.MARKET_TICKERS, **self.SECTOR_TICKERS}
        return self._fetch_yahoo_close(all_tickers, start_date=start_date, end_date=end_date)

    def get_macro_context(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Merge market and macro series into one forward-filled context frame."""
        market = self.fetch_market_data(start_date=start_date, end_date=end_date)
        fred = self.fetch_fred_data(start_date=start_date, end_date=end_date)

        if market.empty and fred.empty:
            return pd.DataFrame()
        if market.empty:
            return fred.ffill().dropna(how="all")
        if fred.empty:
            return market.ffill().dropna(how="all")

        full = market.join(fred, how="outer").sort_index()
        full = full.ffill()
        return full.dropna(how="all")

