from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import pandas as pd


class MacroFeaturePrep:
    """
    Lean feature preparation for FRED macro context.

    Produces ~14 low-collinearity features from yields + economic indicators:
      - Composites replace raw yields (real rate, term spread, ff spread)
      - Single inflation proxy (CORE_CPI_YOY) instead of 4 redundant measures
      - Single growth proxy (RGDP_YOY) instead of nominal + real
      - UNRATE instead of PAYEMS (direct slack measure)
      - Release-day changes via diff(1) on forward-filled series

    Expected input: merged output of MacroClient.fetch_fred_data() +
    MacroClient.fetch_econ_data(), forward-filled to daily.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = float(fill_value)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build lean macro feature matrix from raw FRED context frame."""
        if df.empty:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)

        # Pick inflation proxy: prefer CORE_CPI, fall back to headline
        inflation_col = next(
            (c for c in ("CORE_CPI_YOY", "CPI_YOY") if c in df.columns), None
        )

        # 1) Real 10Y rate + change (replaces raw yield changes)
        if "YIELD_10Y" in df.columns and inflation_col:
            real_10y = df["YIELD_10Y"] - df[inflation_col]
            features["REAL_RATE_10Y"] = real_10y
            features["REAL_RATE_10Y_chg"] = real_10y.diff(1)

        # 2) Term spread + change (curve shape)
        if "YIELD_10Y" in df.columns and "YIELD_2Y" in df.columns:
            ts = df["YIELD_10Y"] - df["YIELD_2Y"]
            features["TERM_SPREAD"] = ts
            features["TERM_SPREAD_chg"] = ts.diff(1)

        # 3) Fed-funds spread to 10Y (monetary stance vs long end)
        if "FEDFUNDS" in df.columns and "YIELD_10Y" in df.columns:
            features["FF_SPREAD_10Y"] = df["YIELD_10Y"] - df["FEDFUNDS"]

        # 4) Real fed funds rate (monetary tightness)
        if "FEDFUNDS" in df.columns and inflation_col:
            features["REAL_FF_RATE"] = df["FEDFUNDS"] - df[inflation_col]

        # 5) Core inflation level + release-day surprise
        if inflation_col:
            features["CORE_INFLATION"] = df[inflation_col]
            features["CORE_INFLATION_chg"] = df[inflation_col].diff(1)

        # 6) Real GDP growth + release-day surprise
        if "RGDP_YOY" in df.columns:
            features["RGDP_YOY"] = df["RGDP_YOY"]
            features["RGDP_YOY_chg"] = df["RGDP_YOY"].diff(1)

        # 7) Unemployment level + release-day surprise
        if "UNRATE" in df.columns:
            features["UNRATE"] = df["UNRATE"]
            features["UNRATE_chg"] = df["UNRATE"].diff(1)

        # 8) Consumer sentiment + release-day surprise
        if "UMCSENT" in df.columns:
            features["UMCSENT"] = df["UMCSENT"]
            features["UMCSENT_chg"] = df["UMCSENT"].diff(1)

        return features.fillna(self.fill_value)

