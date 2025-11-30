"""
Momentum screener engine wrapper for HistoricalScreenerV2.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base_engine import BaseScreenEngine
from .params import BaseScreenParams, MomentumParams
from .screener_types import SCREEN_MOMENTUM
from .historical_screener import HistoricalScreener
from ..features.regime_classification import BaseRegimeClassifier


class MomentumScreenEngine(BaseScreenEngine):
    """Adapter around :class:`CTAFlow.screeners.historical_screener.HistoricalScreener` momentum routines."""

    screen_type = SCREEN_MOMENTUM

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
        if not isinstance(params, MomentumParams):
            raise TypeError("MomentumScreenEngine requires MomentumParams")

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
        results = screener.intraday_momentum_screen(
            session_starts=params.session_starts,
            session_ends=params.session_ends,
            sess_start_hrs=params.sess_start_hrs or 0,
            sess_start_minutes=params.sess_start_minutes or 0,
            sess_end_hrs=params.sess_end_hrs,
            sess_end_minutes=params.sess_end_minutes,
            st_momentum_days=params.st_momentum_days,
            test_vol=params.test_vol,
            months=params.months,
            season=params.season,
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
