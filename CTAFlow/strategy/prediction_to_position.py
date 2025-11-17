"""Utilities to merge overlapping screener predictions into single positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PredictionToPosition:
    """Collapse simultaneous pattern predictions into a single trade decision."""

    correlation_field: str = "correlation"

    def resolve(
        self,
        frame: pd.DataFrame,
        *,
        group_field: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a frame with collisions merged and metadata preserved."""

        if "ts_decision" not in frame.columns or frame.empty:
            result = frame.copy()
            result["source_count"] = 0 if result.empty else 1
            result["_group_members"] = [self._initial_group_members(row, group_field) for _, row in result.iterrows()]
            return result

        duplicated = frame["ts_decision"].duplicated(keep=False)
        if not duplicated.any():
            result = frame.copy()
            result["source_count"] = 1
            result["_group_members"] = [self._initial_group_members(row, group_field) for _, row in result.iterrows()]
            return result

        unique = frame.loc[~duplicated].copy()
        unique["source_count"] = 1
        unique["_group_members"] = [self._initial_group_members(row, group_field) for _, row in unique.iterrows()]
        blocks: List[pd.DataFrame] = [unique]

        for ts_value, subset in frame.loc[duplicated].groupby("ts_decision"):
            blocks.append(self._collapse_group(ts_value, subset, group_field))

        resolved = pd.concat(blocks, ignore_index=True, sort=False)
        resolved = resolved.sort_values("ts_decision").reset_index(drop=True)
        return resolved

    def _initial_group_members(self, row: pd.Series, group_field: Optional[str]) -> List[Any]:
        if group_field and group_field in row.index:
            value = row[group_field]
            return [] if pd.isna(value) else [value]
        return []

    def _collapse_group(
        self,
        ts_value: Any,
        subset: pd.DataFrame,
        group_field: Optional[str],
    ) -> pd.DataFrame:
        clean_subset = subset.copy()
        clean_subset["ts_decision"] = ts_value

        returns_x = pd.to_numeric(clean_subset["returns_x"], errors="coerce").fillna(0.0)
        weights = self._correlation_weights(clean_subset)
        weighted = returns_x * weights
        if not np.isfinite(weighted).any() or np.isclose(weighted.abs().sum(), 0.0):
            score = float(returns_x.sum())
        else:
            score = float(weighted.sum())

        returns_y = pd.to_numeric(clean_subset["returns_y"], errors="coerce").mean()
        side_hint = self._aggregate_hint(clean_subset.get("side_hint"))

        base = clean_subset.iloc[0].copy()
        base["ts_decision"] = ts_value
        base["returns_x"] = score
        base["returns_y"] = float(returns_y)
        base["side_hint"] = side_hint
        base["source_count"] = len(clean_subset)
        base["gate"] = "multi_pattern"
        base["pattern_type"] = "multi_pattern"
        base[self.correlation_field] = float(weights.replace([np.inf, -np.inf], np.nan).mean())
        base["source_gates"] = list(clean_subset.get("gate", []))

        if group_field and group_field in clean_subset.columns:
            base["_group_members"] = list(clean_subset[group_field].dropna())
        else:
            base["_group_members"] = []

        return base.to_frame().T

    def _correlation_weights(self, subset: pd.DataFrame) -> pd.Series:
        if self.correlation_field in subset.columns:
            series = pd.to_numeric(subset[self.correlation_field], errors="coerce").fillna(0.0)
            if not series.empty:
                return series
        return pd.Series(1.0, index=subset.index)

    @staticmethod
    def _aggregate_hint(series: Optional[pd.Series]) -> float:
        if series is None:
            return 0.0
        hints = pd.to_numeric(series, errors="coerce").replace(0, np.nan).dropna()
        if hints.empty:
            return 0.0
        mean_hint = hints.mean()
        if mean_hint == 0:
            return 0.0
        return float(np.sign(mean_hint))
