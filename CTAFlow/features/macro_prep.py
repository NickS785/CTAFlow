from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import pandas as pd


class MacroFeaturePrep:
    """
    Feature preparation for macro + market-state context.

    - Rates -> absolute changes + term spread
    - Prices (indices + sectors) -> returns + relative strength vs SPX
    - VIX level + change
    - Economic indicators (YoY series + levels) -> release-day changes
    - Derived composites: real interest rates, fed-funds spreads
    """

    # YoY columns produced by MacroClient.fetch_econ_data()
    DEFAULT_YOY_COLS = (
        "CPI_YOY", "CORE_CPI_YOY", "PCE_YOY", "CORE_PCE_YOY",
        "NGDP_YOY", "RGDP_YOY", "PAYEMS_YOY",
    )
    # Level columns kept as-is by fetch_econ_data()
    DEFAULT_LEVEL_COLS = ("UNRATE", "FEDFUNDS", "UMCSENT")

    def __init__(
        self,
        rate_cols: Sequence[str] = ("YIELD_10Y", "YIELD_2Y"),
        vix_col: str = "VIX",
        spx_col: str = "SPX",
        rate_windows: Tuple[int, int] = (1, 30),
        return_windows: Tuple[int, int] = (1, 20),
        econ_yoy_cols: Sequence[str] = DEFAULT_YOY_COLS,
        econ_level_cols: Sequence[str] = DEFAULT_LEVEL_COLS,
        fill_value: float = 0.0,
    ):
        self.rate_cols = tuple(rate_cols)
        self.vix_col = vix_col
        self.spx_col = spx_col
        self.rate_windows = tuple(int(w) for w in rate_windows)
        self.return_windows = tuple(int(w) for w in return_windows)
        self.econ_yoy_cols = tuple(econ_yoy_cols)
        self.econ_level_cols = tuple(econ_level_cols)
        self.fill_value = float(fill_value)

    @staticmethod
    def _present(cols: Iterable[str], frame: pd.DataFrame) -> list:
        return [c for c in cols if c in frame.columns]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build macro feature matrix from raw macro context frame."""
        if df.empty:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)
        rate_cols = self._present(self.rate_cols, df)

        # 1) Rates: differences
        for col in rate_cols:
            for win in self.rate_windows:
                features[f"{col}_chg_{win}d"] = df[col].diff(win)

        # 2) Term spread (if both yields are available)
        if "YIELD_10Y" in df.columns and "YIELD_2Y" in df.columns:
            features["TERM_SPREAD"] = df["YIELD_10Y"] - df["YIELD_2Y"]

        # 3) Price-like columns: returns (indices + sectors, excluding VIX + econ)
        skip = set(rate_cols)
        skip.add(self.vix_col)
        skip.update(self.econ_yoy_cols)
        skip.update(self.econ_level_cols)
        price_cols = [c for c in df.columns if c not in skip]

        rel_win = max(self.return_windows) if self.return_windows else 20
        spx_ret = (
            df[self.spx_col].pct_change(rel_win)
            if self.spx_col in df.columns
            else None
        )

        for col in price_cols:
            for win in self.return_windows:
                features[f"{col}_ret_{win}d"] = df[col].pct_change(win)

            if spx_ret is not None and col != self.spx_col:
                features[f"{col}_rel_spx"] = df[col].pct_change(rel_win) - spx_ret

        # 4) VIX level and change
        if self.vix_col in df.columns:
            features[self.vix_col] = df[self.vix_col]
            features[f"{self.vix_col}_chg"] = df[self.vix_col].diff(1)

        # 5) Economic indicators ------------------------------------------
        # YoY series: keep level + release-day change (diff==0 between releases)
        yoy_cols = self._present(self.econ_yoy_cols, df)
        for col in yoy_cols:
            features[col] = df[col]
            features[f"{col}_chg"] = df[col].diff(1)

        # Level series: keep level + release-day change
        lvl_cols = self._present(self.econ_level_cols, df)
        for col in lvl_cols:
            features[col] = df[col]
            features[f"{col}_chg"] = df[col].diff(1)

        # 6) Derived composites -------------------------------------------
        # Real interest rates: nominal yield minus inflation expectations
        inflation_col = next(
            (c for c in ("CPI_YOY", "CORE_CPI_YOY") if c in df.columns), None
        )
        if inflation_col:
            if "YIELD_10Y" in df.columns:
                features["REAL_RATE_10Y"] = df["YIELD_10Y"] - df[inflation_col]
            if "YIELD_2Y" in df.columns:
                features["REAL_RATE_2Y"] = df["YIELD_2Y"] - df[inflation_col]

        # Real fed funds rate
        if "FEDFUNDS" in df.columns and inflation_col:
            features["REAL_FF_RATE"] = df["FEDFUNDS"] - df[inflation_col]

        # Fed-funds yield spreads
        if "FEDFUNDS" in df.columns:
            if "YIELD_10Y" in df.columns:
                features["FF_SPREAD_10Y"] = df["YIELD_10Y"] - df["FEDFUNDS"]
            if "YIELD_2Y" in df.columns:
                features["FF_SPREAD_2Y"] = df["YIELD_2Y"] - df["FEDFUNDS"]

        return features.fillna(self.fill_value)

