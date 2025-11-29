"""
Orderflow screener engine wrapper for HistoricalScreenerV2.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .base_engine import BaseScreenEngine
from .params import BaseScreenParams, OrderflowParams as EngineOrderflowParams
from .screener_types import SCREEN_ORDERFLOW
from . import orderflow_scan


class OrderflowScreenEngine(BaseScreenEngine):
    """Adapter that delegates to :func:`CTAFlow.screeners.orderflow_scan.orderflow_scan`."""

    screen_type = SCREEN_ORDERFLOW

    def prepare(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        regime_classifier=None,
    ) -> Dict[str, Any]:
        return {}

    def run(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(params, EngineOrderflowParams):
            raise TypeError("OrderflowScreenEngine requires OrderflowParams")

        native_params = orderflow_scan.OrderflowParams(
            session_start=params.session_start,
            session_end=params.session_end,
            tz=params.tz,
            bucket_size=params.bucket_size,
            vpin_window=params.vpin_window,
            threshold_z=params.threshold_z,
            min_days=params.min_days,
            cadence_target=params.cadence_target,
            grid_multipliers=tuple(params.grid_multipliers),
            month_filter=params.month_filter,
            season_filter=params.season_filter,
            name=params.name,
        )

        results = orderflow_scan.orderflow_scan({ticker: data}, native_params)
        return {
            "stats": results.get(ticker, {}),
            "patterns": [],
            "metadata": {"screen_type": self.screen_type, "ticker": ticker},
        }
