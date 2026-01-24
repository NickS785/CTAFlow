# CTAFlow/features/ext/fred.py
"""
FRED Data Fetcher
================

Fetches macroeconomic series from FRED.

Requires:
    pip install fredapi

Env var:
    FRED_API_KEY
"""

from __future__ import annotations

import os
import pandas as pd
from typing import Optional
from fredapi import Fred
from dotenv import load_dotenv
from CTAFlow.config import env
env

class FREDDataFetcher:
    """Fetch macro data from FRED."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY not set")

        self.fred = Fred(api_key=self.api_key)

    def fetch(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetch a FRED series.

        Examples:
            - CPIAUCSL (CPI)
            - CPIAUCSL YoY computed downstream
            - DFF (Fed Funds)
        """
        s = self.fred.get_series(
            series_id,
            observation_start=start,
            observation_end=end,
        )

        if s is None or s.empty:
            raise ValueError(f"No data returned for FRED series {series_id}")

        s = s.rename(series_id)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

