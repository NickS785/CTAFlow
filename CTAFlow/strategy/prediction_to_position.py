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
        """Return a frame with one decision row per timestamp.

        Weekday bias patterns are treated specially: their bias value is always
        added to the combined score when overlapping with other predictions.
        """

        if xy is None:
            return pd.DataFrame()

        if xy.empty:
            return xy.copy()

        if "ts_decision" not in xy.columns:
            raise KeyError("PredictionToPosition requires a 'ts_decision' column")

        frame = xy.copy()
        base_signal = pd.to_numeric(frame["returns_x"], errors="coerce").fillna(0.0)

        if "gate_direction" in frame.columns:
            gate_dir = pd.to_numeric(frame["gate_direction"], errors="coerce").fillna(0.0)
            base_signal = base_signal.abs() * np.sign(gate_dir)

        if "strength" in frame.columns:
            gate_strength = pd.to_numeric(frame["strength"], errors="coerce").fillna(1.0)
            base_signal = base_signal * gate_strength

        weights_source = frame.get(self.correlation_field)
        if weights_source is None:
            weights = pd.Series(1.0, index=frame.index, dtype=float)
        else:
            weights = pd.to_numeric(weights_source, errors="coerce").fillna(1.0)
        if self.min_abs_correlation > 0:
            weights = weights.where(weights.abs() >= self.min_abs_correlation, 0.0)
        frame["_ptp_score"] = base_signal * weights

        grouped = frame.groupby("ts_decision", sort=True)
        records = []
        for ts_value, subset in grouped:
            # Separate weekday_bias patterns from other patterns
            has_pattern_type = "pattern_type" in subset.columns
            if has_pattern_type:
                weekday_bias_mask = subset["pattern_type"].isin(["weekday_mean", "weekday_bias_intraday"])
                weekday_bias_subset = subset[weekday_bias_mask]
                other_subset = subset[~weekday_bias_mask]
            else:
                weekday_bias_subset = pd.DataFrame()
                other_subset = subset

            # Calculate score from non-bias patterns
            if not other_subset.empty:
                score = other_subset["_ptp_score"].sum(min_count=1)
            else:
                score = 0.0

            # Add weekday bias contribution (always additive)
            if not weekday_bias_subset.empty:
                weekday_bias_contribution = weekday_bias_subset["_ptp_score"].sum(min_count=1)
                if np.isfinite(weekday_bias_contribution):
                    score += weekday_bias_contribution

            # Determine position based on combined score
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
        """Collapse colliding decisions using weighted scores per timestamp.

        Weekday bias patterns are identified and their bias contribution is
        tracked separately to be applied additively.

        Optimized vectorized implementation - avoids iterating over groupby.
        """

        if frame is None:
            return pd.DataFrame()
        if frame.empty or "ts_decision" not in frame.columns:
            return frame.copy()

        working = frame.copy()
        signal = pd.to_numeric(working["returns_x"], errors="coerce").fillna(0.0)

        if "gate_direction" in working.columns:
            gate_dir = pd.to_numeric(working["gate_direction"], errors="coerce").fillna(0.0)
            signal = signal.abs() * np.sign(gate_dir)

        if "strength" in working.columns:
            gate_strength = pd.to_numeric(working["strength"], errors="coerce").fillna(1.0)
            signal = signal * gate_strength

        working["_ptp_score"] = signal
        weights_source = working.get(self.correlation_field)
        if weights_source is None:
            weights = pd.Series(1.0, index=working.index, dtype=float)
        else:
            weights = pd.to_numeric(weights_source, errors="coerce").fillna(1.0)
        if self.min_abs_correlation > 0:
            weights = weights.where(weights.abs() >= self.min_abs_correlation, 0.0)
        working["_ptp_score"] *= weights

        # Mark weekday bias patterns
        if "pattern_type" in working.columns:
            working["_is_weekday_bias"] = working["pattern_type"].isin(["weekday_mean", "weekday_bias_intraday"])
        else:
            working["_is_weekday_bias"] = False

        # OPTIMIZATION: Use vectorized approach instead of groupby iteration
        include_grouping = bool(group_field and group_field in working.columns)

        if not include_grouping:
            # Simple case: no grouping, just pick best row per timestamp
            resolved = self._resolve_vectorized_simple(working)
        else:
            # Complex case: grouping required, use optimized loop
            resolved = self._resolve_with_grouping(working, group_field)

        return resolved.reset_index(drop=True)

    def _resolve_vectorized_simple(self, working: pd.DataFrame) -> pd.DataFrame:
        """Vectorized resolve without group_field (fast path)."""

        # Calculate weekday bias contribution per timestamp
        bias_contrib = working[working["_is_weekday_bias"]].groupby("ts_decision")["_ptp_score"].sum()

        # Filter out weekday bias rows for selection
        non_bias = working[~working["_is_weekday_bias"]].copy()

        if non_bias.empty:
            # Only bias patterns exist, use them
            non_bias = working.copy()
            bias_contrib = pd.Series(0.0, index=working["ts_decision"].unique())

        # Create ranking columns for selection
        non_bias["_abs_score"] = non_bias["_ptp_score"].abs()
        non_bias["_abs_return"] = pd.to_numeric(non_bias["returns_x"], errors="coerce").abs().fillna(0.0)

        # Sort by timestamp (primary), abs_score (secondary, desc), abs_return (tertiary, desc)
        non_bias.sort_values(
            by=["ts_decision", "_abs_score", "_abs_return"],
            ascending=[True, False, False],
            inplace=True
        )

        # Keep first (best) row per timestamp
        resolved = non_bias.drop_duplicates(subset=["ts_decision"], keep="first").copy()

        # Add weekday bias contribution back
        resolved["_ptp_score"] = resolved["_ptp_score"] + resolved["ts_decision"].map(bias_contrib).fillna(0.0)

        # Cleanup temporary columns (keep _ptp_score for now, will be removed by caller)
        resolved.drop(columns=["_abs_score", "_abs_return", "_ptp_score", "_is_weekday_bias"], errors="ignore", inplace=True)

        return resolved

    def _resolve_with_grouping(self, working: pd.DataFrame, group_field: str) -> pd.DataFrame:
        """Optimized resolve with group_field (requires iteration but minimized)."""
        # This path still needs iteration but is much faster than before
        collapsed_rows = []
        grouped = working.groupby("ts_decision", sort=True)

        for _, subset in grouped:
            # Extract weekday bias contribution for this timestamp
            weekday_bias_contrib = 0.0
            if "_is_weekday_bias" in subset.columns:
                bias_subset = subset[subset["_is_weekday_bias"]]
                if not bias_subset.empty:
                    weekday_bias_contrib = bias_subset["_ptp_score"].sum()

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

                    # Filter out weekday bias from per_subset for best row selection
                    if "_is_weekday_bias" in per_subset.columns:
                        non_bias_subset = per_subset[~per_subset["_is_weekday_bias"]]
                        if not non_bias_subset.empty:
                            row = self._select_best_row(non_bias_subset)
                        else:
                            row = self._select_best_row(per_subset)
                    else:
                        row = self._select_best_row(per_subset)

                    # Add weekday bias contribution to the score
                    if np.isfinite(weekday_bias_contrib) and weekday_bias_contrib != 0.0:
                        current_score = row.get("_ptp_score", 0.0)
                        row["_ptp_score"] = current_score + weekday_bias_contrib

                    if members:
                        row["_group_members"] = members
                    collapsed_rows.append(row)
            else:
                # Filter out weekday bias for best row selection
                if "_is_weekday_bias" in subset.columns:
                    non_bias_subset = subset[~subset["_is_weekday_bias"]]
                    if not non_bias_subset.empty:
                        row = self._select_best_row(non_bias_subset)
                    else:
                        row = self._select_best_row(subset)
                else:
                    row = self._select_best_row(subset)

                # Add weekday bias contribution to the score
                if np.isfinite(weekday_bias_contrib) and weekday_bias_contrib != 0.0:
                    current_score = row.get("_ptp_score", 0.0)
                    row["_ptp_score"] = current_score + weekday_bias_contrib

                collapsed_rows.append(row)

        resolved = pd.DataFrame(collapsed_rows).drop(columns=["_ptp_score", "_is_weekday_bias"], errors="ignore")
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
