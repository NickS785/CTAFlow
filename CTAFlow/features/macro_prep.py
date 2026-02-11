from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import pandas as pd


class MacroFeaturePrep:
    """
    Feature preparation for macro + market-state context.

    - Rates -> absolute changes
    - Prices (indices + sectors) -> returns
    - Optional relative strength vs SPX
    """

    def __init__(
        self,
        rate_cols: Sequence[str] = ("YIELD_10Y", "YIELD_2Y"),
        vix_col: str = "VIX",
        spx_col: str = "SPX",
        rate_windows: Tuple[int, int] = (1, 30),
        return_windows: Tuple[int, int] = (1, 20),
        fill_value: float = 0.0,
    ):
        self.rate_cols = tuple(rate_cols)
        self.vix_col = vix_col
        self.spx_col = spx_col
        self.rate_windows = tuple(int(w) for w in rate_windows)
        self.return_windows = tuple(int(w) for w in return_windows)
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

        # 3) Price-like columns: returns (indices + sectors, excluding VIX)
        skip = set(rate_cols)
        skip.add(self.vix_col)
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

        return features.fillna(self.fill_value)

