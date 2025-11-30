"""
Event screener engine delegating to :mod:`CTAFlow.screeners.event_screener`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base_engine import BaseScreenEngine
from .params import BaseScreenParams, EventParams
from .screener_types import SCREEN_EVENT
from .event_screener import run_event_screener
from ..features.regime_classification import BaseRegimeClassifier


class EventScreenEngine(BaseScreenEngine):
    """Lightweight adapter for running data release scans per ticker."""

    screen_type = SCREEN_EVENT

    def __init__(
        self,
        event_calendars: Optional[Dict[str, pd.DataFrame]] = None,
        default_tz: str = "America/Chicago",
        orderflow_provider=None,
    ) -> None:
        self.event_calendars = event_calendars or {}
        self.default_tz = default_tz
        self.orderflow_provider = orderflow_provider

    def prepare(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        regime_classifier: Optional[BaseRegimeClassifier] = None,
    ) -> Dict[str, Any]:
        events = self.event_calendars.get(ticker) or self.event_calendars.get("default", pd.DataFrame())
        context: Dict[str, Any] = {"events": events}
        if isinstance(params, EventParams) and params.use_orderflow and self.orderflow_provider:
            context["orderflow"] = self.orderflow_provider(ticker)
        return context

    def run(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(params, EventParams):
            raise TypeError("EventScreenEngine requires EventParams")

        events_df = context.get("events", pd.DataFrame())
        result = run_event_screener(
            bars=data,
            events=events_df,
            params=params,
            symbol=ticker,
            instrument_tz=params.tz or self.default_tz,
            orderflow=context.get("orderflow"),
        )
        return {
            "stats": result.summary,
            "patterns": result.patterns,
            "metadata": {"events": result.events},
        }
