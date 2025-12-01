"""
Event screener engine delegating to :mod:`CTAFlow.screeners.event_screener`.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from datetime import date

import pandas as pd

from .base_engine import BaseScreenEngine
from .params import BaseScreenParams, EventParams
from .screener_types import SCREEN_EVENT
from .event_screener import run_event_screener
from ..features.regime_classification import BaseRegimeClassifier
from .event_presets import (
    EventDefinition,
    event_release_dt_for_date,
    get_events_for_ticker,
    is_matching_event_slot,
)


class EventScreenEngine(BaseScreenEngine):
    """Lightweight adapter for running data release scans per ticker."""

    screen_type = SCREEN_EVENT

    def __init__(
        self,
        event_calendars: Optional[Dict[str, pd.DataFrame]] = None,
        event_definitions: Optional[Dict[str, Iterable[EventDefinition]]] = None,
        default_tz: str = "America/Chicago",
        orderflow_provider=None,
    ) -> None:
        self.event_calendars = event_calendars or {}
        self.event_definitions = {
            key: list(defs) for key, defs in (event_definitions or {}).items()
        }
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
        if events.empty:
            events = self._generate_events_from_definitions(ticker, data, params)
            if not events.empty:
                self.event_calendars[ticker] = events
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
            use_gpu=context.get("use_gpu", False),
            gpu_device_id=context.get("gpu_device_id", 0),
        )
        return {
            "stats": result.summary,
            "patterns": result.patterns,
            "metadata": {"events": result.events},
        }

    def _generate_events_from_definitions(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
    ) -> pd.DataFrame:
        if not isinstance(params, EventParams):
            return pd.DataFrame()

        event_defs = list(self.event_definitions.get(ticker, []))
        if not event_defs:
            event_defs = get_events_for_ticker(ticker)

        if params.event_code:
            event_defs = [
                ed for ed in event_defs if ed.code == params.event_code
            ] or event_defs

        if not event_defs:
            return pd.DataFrame()

        sessions = self._extract_session_dates(data, params.tz or self.default_tz)
        rows: List[Dict[str, Any]] = []
        for event_def in event_defs:
            for session_date in sessions:
                if not is_matching_event_slot(session_date, event_def):
                    continue
                release_ts = event_release_dt_for_date(event_def, session_date, params.tz or self.default_tz)
                rows.append({
                    "event_code": event_def.code,
                    "release_ts": release_ts,
                })

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).drop_duplicates().sort_values("release_ts").reset_index(drop=True)
        return df

    @staticmethod
    def _extract_session_dates(data: pd.DataFrame, instrument_tz: str) -> List[date]:
        if "ts" in data.columns:
            ts = pd.to_datetime(data["ts"])
        else:
            ts = pd.to_datetime(data.index)
        ts = ts.dt.tz_localize(instrument_tz) if ts.dt.tz is None else ts.dt.tz_convert(instrument_tz)
        return sorted(ts.dt.date.unique())
