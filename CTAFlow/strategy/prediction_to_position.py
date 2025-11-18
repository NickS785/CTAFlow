"""Utilities for consolidating overlapping screener predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        weights_source = frame.get(self.correlation_field)
        if weights_source is None:
            weights = pd.Series(1.0, index=frame.index, dtype=float)
        else:
            weights = pd.to_numeric(weights_source, errors="coerce").fillna(1.0)
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

    def resolve(
        self,
        frame: pd.DataFrame,
        *,
        group_field: Optional[str] = None,
    ) -> pd.DataFrame:
        """Collapse colliding decisions using weighted scores per timestamp."""

        if frame is None:
            return pd.DataFrame()
        if frame.empty or "ts_decision" not in frame.columns:
            return frame.copy()

        working = frame.copy()
        working["_ptp_score"] = pd.to_numeric(working["returns_x"], errors="coerce").fillna(0.0)
        weights_source = working.get(self.correlation_field)
        if weights_source is None:
            weights = pd.Series(1.0, index=working.index, dtype=float)
        else:
            weights = pd.to_numeric(weights_source, errors="coerce").fillna(1.0)
        working["_ptp_score"] *= weights

        collapsed_rows = []
        grouped = working.groupby("ts_decision", sort=True)
        include_grouping = bool(group_field and group_field in working.columns)
        for _, subset in grouped:
            if include_grouping:
                values = subset[group_field].tolist()
                order = []
                seen_keys = set()
                for value in values:
                    key = "__nan__" if pd.isna(value) else value
                    if key not in seen_keys:
                        seen_keys.add(key)
                        order.append(value)

                members = [value for value in values if not pd.isna(value)]
                if members:
                    members = list(dict.fromkeys(members))

                for value in order:
                    if pd.isna(value):
                        per_subset = subset[subset[group_field].isna()]
                    else:
                        per_subset = subset[subset[group_field] == value]

                    if per_subset.empty:
                        continue

                    row = self._select_best_row(per_subset)
                    if members:
                        row["_group_members"] = members
                    collapsed_rows.append(row)
            else:
                row = self._select_best_row(subset)
                collapsed_rows.append(row)

        resolved = pd.DataFrame(collapsed_rows).drop(columns=["_ptp_score"], errors="ignore")
        return resolved.reset_index(drop=True)

    @staticmethod
    def _select_best_row(subset: pd.DataFrame) -> pd.Series:
        if len(subset) == 1:
            return subset.iloc[0].copy()

        scores = subset["_ptp_score"].abs()
        if scores.notna().any():
            idx = scores.idxmax()
            return subset.loc[idx].copy()

        magnitudes = pd.to_numeric(subset["returns_x"], errors="coerce").abs()
        if magnitudes.notna().any():
            idx = magnitudes.idxmax()
            return subset.loc[idx].copy()

        return subset.iloc[0].copy()
