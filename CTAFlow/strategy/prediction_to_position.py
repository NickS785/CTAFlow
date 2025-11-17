"""Utilities for consolidating overlapping screener predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictionToPosition:
    """Resolve overlapping screener predictions into discrete positions."""

    neutral_tolerance: float = 1e-5
    min_abs_correlation: float = 0.0
    correlation_field: str = "correlation"

    def aggregate(self, xy: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with one decision row per timestamp."""

        if xy is None:
            return pd.DataFrame()

        if xy.empty:
            return xy.copy()

        if "ts_decision" not in xy.columns:
            raise KeyError("PredictionToPosition requires a 'ts_decision' column")

        frame = xy.copy()
        returns_x = pd.to_numeric(frame["returns_x"], errors="coerce")
        weights = pd.to_numeric(frame.get(self.correlation_field, 1.0), errors="coerce").fillna(1.0)
        if self.min_abs_correlation > 0:
            weights = weights.where(weights.abs() >= self.min_abs_correlation, 0.0)
        frame["_ptp_score"] = returns_x * weights

        grouped = frame.groupby("ts_decision", sort=True)
        records = []
        for ts_value, subset in grouped:
            score = subset["_ptp_score"].sum(min_count=1)
            if not np.isfinite(score) or abs(score) <= self.neutral_tolerance:
                position = 0
            elif score > 0:
                position = 1
            else:
                position = -1

            base = subset.iloc[0].copy()
            base["returns_x"] = pd.to_numeric(subset["returns_x"], errors="coerce").mean()
            base["returns_y"] = pd.to_numeric(subset["returns_y"], errors="coerce").mean()
            base["prediction_position"] = position
            records.append(base)

        result = pd.DataFrame(records).drop(columns=["_ptp_score"], errors="ignore")
        return result.reset_index(drop=True)
