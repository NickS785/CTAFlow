from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None  # type: ignore

try:
    from fredapi import Fred
except Exception:  # pragma: no cover - optional dependency
    Fred = None  # type: ignore


class MacroClient:
    """
    Unified client for retrieving market-state context.

    Sources:
      - Yahoo Finance for index/ETF closes
      - FRED for rates/yield curves and economic indicators
    """

    FRED_SERIES: Dict[str, str] = {
        "YIELD_10Y": "DGS10",
        "YIELD_2Y": "DGS2",
    }

    # Economic indicators from FRED.
    # "yoy_periods" = number of native-frequency periods for YoY pct_change
    #   12 for monthly series, 4 for quarterly.
    # "transform" controls how the raw level is converted:
    #   "yoy"   -> pct_change(yoy_periods) * 100  (annualised growth rate)
    #   "level" -> keep raw level as-is (already a rate / index)
    ECON_SERIES: Dict[str, Dict] = {
        "CPI":       {"series_id": "CPIAUCSL",  "yoy_periods": 12, "transform": "yoy"},
        "CORE_CPI":  {"series_id": "CPILFESL",  "yoy_periods": 12, "transform": "yoy"},
        "PCE":       {"series_id": "PCEPI",     "yoy_periods": 12, "transform": "yoy"},
        "CORE_PCE":  {"series_id": "PCEPILFE",  "yoy_periods": 12, "transform": "yoy"},
        "NGDP":      {"series_id": "GDP",       "yoy_periods": 4,  "transform": "yoy"},
        "RGDP":      {"series_id": "GDPC1",     "yoy_periods": 4,  "transform": "yoy"},
        "UNRATE":    {"series_id": "UNRATE",    "yoy_periods": 12, "transform": "level"},
        "FEDFUNDS":  {"series_id": "FEDFUNDS",  "yoy_periods": 12, "transform": "level"},
        "PAYEMS":    {"series_id": "PAYEMS",    "yoy_periods": 12, "transform": "yoy"},
        "UMCSENT":   {"series_id": "UMCSENT",   "yoy_periods": 12, "transform": "level"},
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

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        auto_adjust: bool = True,
        include_sectors: bool = False,
    ):
        self.fred_api_key = fred_api_key
        self.auto_adjust = bool(auto_adjust)
        self.include_sectors = bool(include_sectors)

    def _get_fred(self) -> Optional["Fred"]:
        """Return a Fred client or None if unavailable."""
        if Fred is None:
            logger.warning("fredapi not installed — pip install fredapi")
            return None
        if not self.fred_api_key:
            import os
            self.fred_api_key = os.getenv("FRED_API_KEY")
        if not self.fred_api_key:
            logger.warning("FRED_API_KEY not set — skipping FRED data")
            return None
        return Fred(api_key=self.fred_api_key)

    def fetch_fred_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch FRED macro series. Returns empty frame if dependency/source is unavailable."""
        fred = self._get_fred()
        if fred is None:
            return pd.DataFrame()

        try:
            end = end_date or datetime.utcnow()
            frames = {}
            for alias, series_id in self.FRED_SERIES.items():
                s = fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end,
                )
                if s is not None and not s.empty:
                    frames[alias] = s
            if not frames:
                return pd.DataFrame()
            out = pd.DataFrame(frames)
            out.index = pd.to_datetime(out.index)
            return out.sort_index()
        except Exception as e:
            logger.warning("FRED yield fetch failed: %s", e)
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
        """Fetch market indices (and optionally sector ETFs) as one aligned frame."""
        tickers = dict(self.MARKET_TICKERS)
        if self.include_sectors:
            tickers.update(self.SECTOR_TICKERS)
        return self._fetch_yahoo_close(tickers, start_date=start_date, end_date=end_date)

    def fetch_econ_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch FRED economic indicators and apply YoY / level transforms.

        YoY series (CPI, PCE, GDP, PAYEMS) are returned as annualised
        percentage growth rates (e.g. ``CPI_YOY = 3.2`` means 3.2 %).

        Level series (UNRATE, FEDFUNDS, UMCSENT) are kept as-is.

        The resulting frame is at *native* frequency (monthly / quarterly)
        — forward-filling to daily happens in ``get_macro_context()``.
        """
        fred = self._get_fred()
        if fred is None:
            return pd.DataFrame()

        end = end_date or datetime.utcnow()
        # Request extra history so YoY calculation doesn't clip early rows
        lookback_start = pd.Timestamp(start_date) - pd.DateOffset(months=15)

        raw_frames = {}
        for name, spec in self.ECON_SERIES.items():
            try:
                s = fred.get_series(
                    spec["series_id"],
                    observation_start=lookback_start,
                    observation_end=end,
                )
                if s is not None and not s.empty:
                    raw_frames[name] = s
            except Exception as e:
                logger.warning("FRED series %s (%s) failed: %s", name, spec["series_id"], e)

        if not raw_frames:
            logger.warning("No FRED econ series retrieved")
            return pd.DataFrame()

        raw = pd.DataFrame(raw_frames)
        out = pd.DataFrame(index=raw.index)
        for name, spec in self.ECON_SERIES.items():
            if name not in raw.columns:
                continue
            col = raw[name].dropna()
            if col.empty:
                continue

            if spec["transform"] == "yoy":
                yoy = col.pct_change(spec["yoy_periods"]) * 100
                out[f"{name}_YOY"] = yoy
            else:
                out[name] = col

        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        # Trim to requested range (YoY lookback rows fall before start_date)
        return out.loc[out.index >= pd.Timestamp(start_date)]

    def get_macro_context(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Merge market, rates, and economic series into one forward-filled frame.

        Economic indicators (monthly/quarterly) are forward-filled to the
        daily grid established by market data. Values only change on their
        FRED release date, so downstream ``diff()`` naturally captures
        release-day surprises.
        """
        market = self.fetch_market_data(start_date=start_date, end_date=end_date)
        fred = self.fetch_fred_data(start_date=start_date, end_date=end_date)
        econ = self.fetch_econ_data(start_date=start_date, end_date=end_date)

        frames = [f for f in (market, fred, econ) if not f.empty]
        if not frames:
            return pd.DataFrame()

        full = frames[0]
        for f in frames[1:]:
            full = full.join(f, how="outer")
        full = full.sort_index().ffill()
        return full.dropna(how="all")

