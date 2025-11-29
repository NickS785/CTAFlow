"""
Pluggable historical screener orchestrator.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Optional, Union, List
import logging

import pandas as pd

from .screener_types import (
    SCREEN_EVENT,
    SCREEN_MOMENTUM,
    SCREEN_ORDERFLOW,
    SCREEN_SEASONALITY,
)
from .params import BaseScreenParams
from .base_engine import BaseScreenEngine
from .momentum_engine import MomentumScreenEngine
from .seasonality_engine import SeasonalityScreenEngine
from .orderflow_engine import OrderflowScreenEngine
from .event_engine import EventScreenEngine
from ..features.regime_classification import BaseRegimeClassifier
from ..data import ResultsClient


class HistoricalScreenerV2:
    """
    Pluggable historical screener for multiple screen types.
    """

    def __init__(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        results_client: Optional[ResultsClient] = None,
        auto_write_results: bool = False,
        engines: Optional[Dict[str, BaseScreenEngine]] = None,
        regime_classifier: Optional[BaseRegimeClassifier] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.ticker_data = ticker_data
        self.results_client = results_client
        self.auto_write_results = auto_write_results
        self.regime_classifier = regime_classifier
        self.logger = logger or logging.getLogger(__name__)

        self.engines: Dict[str, BaseScreenEngine] = engines or {
            SCREEN_MOMENTUM: MomentumScreenEngine(),
            SCREEN_SEASONALITY: SeasonalityScreenEngine(),
            SCREEN_ORDERFLOW: OrderflowScreenEngine(),
            SCREEN_EVENT: EventScreenEngine(),
        }

    def run_screens(
        self,
        screen_params: Sequence[BaseScreenParams],
        output_format: str = "dict",
    ) -> Union[Dict[str, Dict[str, Any]], pd.DataFrame]:
        """Run one or more screens over the ticker universe."""

        results: Dict[str, Dict[str, Any]] = {}

        for params in screen_params:
            engine = self.engines.get(params.screen_type)
            if engine is None:
                raise ValueError(f"Unregistered screen_type: {params.screen_type}")

            screen_name = params.name or self._default_screen_name(params)
            self.logger.info("Running screen %s (%s)", screen_name, params.screen_type)
            results[screen_name] = {}

            for ticker, df in self.ticker_data.items():
                ctx = engine.prepare(
                    ticker=ticker,
                    data=df,
                    params=params,
                    regime_classifier=self.regime_classifier,
                )
                res = engine.run(
                    ticker=ticker,
                    data=df,
                    params=params,
                    context=ctx,
                )
                results[screen_name][ticker] = res

                if self.auto_write_results and self.results_client:
                    self._write_results(screen_name, ticker, params, res)

        if output_format == "dict":
            return results
        return self._flatten_results(results)

    def _default_screen_name(self, params: BaseScreenParams) -> str:
        if getattr(params, "season", None):
            return f"{params.season}_{params.screen_type}"
        if getattr(params, "months", None):
            month_str = "_".join(map(str, getattr(params, "months")))
            return f"months_{month_str}_{params.screen_type}"
        return f"all_{params.screen_type}"

    def _write_results(
        self,
        screen_name: str,
        ticker: str,
        params: BaseScreenParams,
        res: Dict[str, Any],
    ) -> None:
        if not self.results_client:
            return
        stats = res.get("stats")
        if stats is None:
            return
        if isinstance(stats, dict):
            stats_df = pd.DataFrame([stats])
        elif isinstance(stats, pd.DataFrame):
            stats_df = stats
        else:
            stats_df = pd.DataFrame(stats)

        metadata = res.get("metadata") or {}
        metadata.update({"screen_type": params.screen_type, "screen_name": screen_name})
        self.results_client.write_scan_results(params.screen_type, ticker, screen_name, stats_df, metadata=metadata)

    def _flatten_results(self, nested: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for screen_name, per_ticker in nested.items():
            for ticker, payload in per_ticker.items():
                stats = payload.get("stats") if isinstance(payload, dict) else payload
                if isinstance(stats, pd.DataFrame):
                    df = stats.copy()
                    df["screen_name"] = screen_name
                    df["ticker"] = ticker
                    rows.append(df)
                elif isinstance(stats, dict):
                    row = dict(stats)
                    row.update({"screen_name": screen_name, "ticker": ticker})
                    rows.append(pd.DataFrame([row]))
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)
