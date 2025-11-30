"""
Seasonality screener engine wrapper for HistoricalScreenerV2.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base_engine import BaseScreenEngine
from .params import BaseScreenParams, SeasonalityParams
from .screener_types import SCREEN_SEASONALITY
from .historical_screener import HistoricalScreener
from ..features.regime_classification import BaseRegimeClassifier


class SeasonalityScreenEngine(BaseScreenEngine):
    """Adapter around :class:`CTAFlow.screeners.historical_screener.HistoricalScreener` seasonality routines."""

    screen_type = SCREEN_SEASONALITY

    def prepare(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        regime_classifier: Optional[BaseRegimeClassifier] = None,
    ) -> Dict[str, Any]:
        return {"regime_classifier": regime_classifier}

    def run(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(params, SeasonalityParams):
            raise TypeError("SeasonalityScreenEngine requires SeasonalityParams")

        # Extract GPU settings from context (added by HistoricalScreenerV2)
        use_gpu = context.get('use_gpu', False)
        gpu_device_id = context.get('gpu_device_id', 0)

        screener = HistoricalScreener(
            {ticker: data},
            results_client=None,
            auto_write_results=False,
            verbose=False,
            use_gpu=use_gpu,
            gpu_device_id=gpu_device_id,
        )
        results = screener.st_seasonality_screen(
            target_times=params.target_times or [],
            period_length=params.period_length,
            dayofweek_screen=params.dayofweek_screen,
            months=params.months,
            season=params.season,
            session_start=params.seasonality_session_start,
            session_end=params.seasonality_session_end,
            tz=params.tz,
            use_regime_filtering=params.use_regime_filtering,
            regime_col=params.regime_col,
            target_regimes=params.target_regimes,
            regime_settings=params.regime_settings,
        )
        return {
            "stats": results.get(ticker, {}),
            "patterns": [],
            "metadata": {"screen_type": self.screen_type, "ticker": ticker},
        }
