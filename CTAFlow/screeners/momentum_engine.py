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

        screener = HistoricalScreener({ticker: data}, results_client=None, auto_write_results=False, verbose=False)
        session_kwargs: Dict[str, Any] = {}
        if params.session_starts:
            session_kwargs["session_starts"] = params.session_starts
        if params.session_ends:
            session_kwargs["session_ends"] = params.session_ends
        if params.sess_start_hrs is not None:
            session_kwargs["sess_start_hrs"] = params.sess_start_hrs
        if params.sess_start_minutes is not None:
            session_kwargs["sess_start_minutes"] = params.sess_start_minutes
        if params.sess_end_hrs is not None:
            session_kwargs["sess_end_hrs"] = params.sess_end_hrs
        if params.sess_end_minutes is not None:
            session_kwargs["sess_end_minutes"] = params.sess_end_minutes

        results = screener.intraday_momentum_screen(
            st_momentum_days=params.st_momentum_days,
            test_vol=params.test_vol,
            months=params.months,
            season=params.season,
            use_regime_filtering=params.use_regime_filtering,
            regime_col=params.regime_col,
            target_regimes=params.target_regimes,
            regime_settings=params.regime_settings,
            **session_kwargs,
        )
        return {
            "stats": results.get(ticker, {}),
            "patterns": [],
            "metadata": {"screen_type": self.screen_type, "ticker": ticker},
        }
