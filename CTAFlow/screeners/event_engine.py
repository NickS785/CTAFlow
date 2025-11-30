"""
Event screener engine delegating to :mod:`CTAFlow.screeners.data_release_screener`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base_engine import BaseScreenEngine
from .params import BaseScreenParams, EventParams
from .screener_types import SCREEN_EVENT
from .data_release_screener import data_release_scan
from ..features.regime_classification import BaseRegimeClassifier


class EventScreenEngine(BaseScreenEngine):
    """Lightweight adapter for running data release scans per ticker."""

    screen_type = SCREEN_EVENT

    def __init__(self, event_calendars: Optional[Dict[str, pd.DataFrame]] = None, default_tz: str = "America/Chicago") -> None:
        self.event_calendars = event_calendars or {}
        self.default_tz = default_tz

    def prepare(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        regime_classifier: Optional[BaseRegimeClassifier] = None,
    ) -> Dict[str, Any]:
        events = self.event_calendars.get(ticker) or self.event_calendars.get("default", pd.DataFrame())
        return {"events": events}

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
        result = data_release_scan(
            bars=data,
            events=events_df,
            params=params,
            symbol=ticker,
            instrument_tz=params.tz or self.default_tz,
        )
        return {
            "stats": result.summary,
            "patterns": result.patterns,
            "metadata": {"events": result.events},
        }
