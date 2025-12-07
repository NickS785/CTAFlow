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

        # Identify weekday bias patterns (vectorized)
        if "pattern_type" in frame.columns:
            weekday_bias_mask = frame["pattern_type"].isin(["weekday_mean", "weekday_bias_intraday"])
        else:
            weekday_bias_mask = pd.Series(False, index=frame.index)

        # Separate bias and non-bias scores
        frame["_is_bias"] = weekday_bias_mask
        frame["_bias_score"] = frame["_ptp_score"].where(weekday_bias_mask, 0.0)
        frame["_other_score"] = frame["_ptp_score"].where(~weekday_bias_mask, 0.0)

        # Aggregate scores per timestamp (vectorized)
        grouped = frame.groupby("ts_decision", sort=False)
        agg_scores = pd.DataFrame({
            "other_score_sum": grouped["_other_score"].sum(),
            "bias_score_sum": grouped["_bias_score"].sum(),
        })

        # Calculate combined score and position
        agg_scores["combined_score"] = agg_scores["other_score_sum"] + agg_scores["bias_score_sum"]
        agg_scores["prediction_position"] = 0
        valid_scores = agg_scores["combined_score"].notna() & (agg_scores["combined_score"].abs() > self.neutral_tolerance)
        agg_scores.loc[valid_scores & (agg_scores["combined_score"] > 0), "prediction_position"] = 1
        agg_scores.loc[valid_scores & (agg_scores["combined_score"] < 0), "prediction_position"] = -1

        # Aggregate returns (vectorized)
        returns_x_num = pd.to_numeric(frame["returns_x"], errors="coerce")
        returns_y_num = pd.to_numeric(frame["returns_y"], errors="coerce")
        agg_returns = pd.DataFrame({
            "returns_x_mean": grouped[returns_x_num.name if hasattr(returns_x_num, 'name') else "returns_x"].mean(),
            "returns_y_mean": grouped[returns_y_num.name if hasattr(returns_y_num, 'name') else "returns_y"].mean(),
        })

        # Take first row per timestamp as base and update with aggregated values
        base_rows = frame.sort_values("ts_decision").drop_duplicates(subset=["ts_decision"], keep="first").set_index("ts_decision")

        # Update with aggregated values
        base_rows["returns_x"] = agg_returns["returns_x_mean"]
        base_rows["returns_y"] = agg_returns["returns_y_mean"]
        base_rows["prediction_position"] = agg_scores["prediction_position"]

        # Cleanup temporary columns
        result = base_rows.drop(columns=["_ptp_score", "_is_bias", "_bias_score", "_other_score"], errors="ignore")
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
        """Vectorized resolve with group_field using pre-sorted groupby operations."""
        # Calculate weekday bias contribution per timestamp (vectorized)
        bias_contrib = working[working["_is_weekday_bias"]].groupby("ts_decision")["_ptp_score"].sum()

        # Filter out weekday bias rows for selection
        non_bias = working[~working["_is_weekday_bias"]].copy()

        if non_bias.empty:
            # Only bias patterns exist, use them
            non_bias = working.copy()
            bias_contrib = pd.Series(0.0, index=working["ts_decision"].unique())

        # Create ranking columns for vectorized best-row selection
        non_bias["_abs_score"] = non_bias["_ptp_score"].abs()
        non_bias["_abs_return"] = pd.to_numeric(non_bias["returns_x"], errors="coerce").abs().fillna(0.0)

        # Sort by (ts_decision, group_field, _abs_score desc, _abs_return desc)
        # This ensures we can pick the first row per (ts_decision, group_field) pair as the best
        non_bias.sort_values(
            by=["ts_decision", group_field, "_abs_score", "_abs_return"],
            ascending=[True, True, False, False],
            inplace=True
        )

        # Keep first (best) row per (ts_decision, group_field) pair
        resolved = non_bias.drop_duplicates(subset=["ts_decision", group_field], keep="first").copy()

        # Add weekday bias contribution back
        resolved["_ptp_score"] = resolved["_ptp_score"] + resolved["ts_decision"].map(bias_contrib).fillna(0.0)

        # Build group_members column: for each timestamp, collect all unique group values
        # Group by ts_decision and aggregate all group_field values
        members_map = (
            resolved.groupby("ts_decision")[group_field]
            .apply(lambda x: list(x.dropna().unique()))
            .to_dict()
        )

        # Add _group_members to each row
        resolved["_group_members"] = resolved["ts_decision"].map(members_map)

        # Cleanup temporary columns
        resolved.drop(columns=["_abs_score", "_abs_return", "_ptp_score", "_is_weekday_bias"], errors="ignore", inplace=True)

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
