"""Sessionization utilities for aligning bar data to exchange sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Optional

import pandas as pd
import exchange_calendars as xcals

__all__ = ["Sessionizer", "SessionizerConfig"]


@dataclass
class SessionizerConfig:
    """Configuration used to resolve exchange calendars for symbols."""

    default_calendar: str = "XNYS"
    timezone: str = "America/Chicago"
    calendar_map: Mapping[str, str] = field(default_factory=dict)


class Sessionizer:
    """Attach exchange session identifiers to bar data."""

    def __init__(
        self,
        config: SessionizerConfig | None = None,
    ) -> None:
        self.config = config or SessionizerConfig()
        self._calendar_cache: MutableMapping[str, xcals.ExchangeCalendar] = {}

    def _resolve_calendar_name(
        self,
        *,
        symbol: Optional[str] = None,
        calendar: Optional[str] = None,
    ) -> str:
        if calendar:
            return calendar

        calendar_map = {k.upper(): v for k, v in (self.config.calendar_map or {}).items()}
        if symbol:
            symbol_norm = str(symbol).upper()
            candidates = [symbol_norm]
            if "_" in symbol_norm:
                candidates.append(symbol_norm.split("_", 1)[0])
            stripped = "".join(ch for ch in symbol_norm if not ch.isdigit())
            if stripped and stripped not in candidates:
                candidates.append(stripped)

            for candidate in candidates:
                if candidate in calendar_map:
                    return calendar_map[candidate]

        return self.config.default_calendar

    def _get_calendar(self, name: str) -> xcals.ExchangeCalendar:
        calendar = self._calendar_cache.get(name)
        if calendar is None:
            calendar = xcals.get_calendar(name)
            self._calendar_cache[name] = calendar
        return calendar

    def attach(
        self,
        df: pd.DataFrame,
        ts_col: str = "ts",
        *,
        symbol: Optional[str] = None,
        calendar: Optional[str] = None,
    ) -> pd.DataFrame:
        if ts_col not in df.columns:
            raise KeyError(f"DataFrame missing '{ts_col}' column required for sessionization")

        ts = pd.to_datetime(df[ts_col])
        if ts.dt.tz is None:
            raise ValueError("Sessionizer requires timezone-aware timestamps")

        ts = ts.dt.tz_convert(self.config.timezone)
        calendar_name = self._resolve_calendar_name(symbol=symbol, calendar=calendar)
        exchange_calendar = self._get_calendar(calendar_name)

        start = ts.min().date()
        end = ts.max().date()
        schedule = exchange_calendar.schedule.loc[str(start) : str(end)]
        if schedule.empty:
            out = df.copy()
            out[ts_col] = ts
            out["session_id"] = pd.Series(pd.NA, index=df.index, dtype="object")
            return out

        open_col = "open" if "open" in schedule.columns else "market_open"
        close_col = "close" if "close" in schedule.columns else "market_close"
        market_open = schedule[open_col].dt.tz_convert(self.config.timezone)
        market_close = schedule[close_col].dt.tz_convert(self.config.timezone)

        intervals = pd.DataFrame({"open": market_open, "close": market_close})
        intervals["session_id"] = intervals.index.strftime("%Y-%m-%d")

        left = pd.DataFrame({ts_col: ts})
        merged = pd.merge_asof(
            left.sort_values(ts_col),
            intervals.sort_values("open"),
            left_on=ts_col,
            right_on="open",
            direction="forward",
        )

        valid = merged[ts_col] < merged["close"]
        out = df.copy()
        out[ts_col] = ts
        out["session_id"] = pd.Series(pd.NA, index=df.index, dtype="object")
        out.loc[valid.values, "session_id"] = merged.loc[valid, "session_id"].values
        return out
