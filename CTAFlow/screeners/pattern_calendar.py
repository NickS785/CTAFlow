"""Active trading calendar utilities for screener patterns.

This module provides two primary helpers:

``PatternVault``
    Manages storage of screener pattern payloads in an HDF5 file. Patterns are
    written under ``screeners/{ticker}/{screen_name}`` and can be consolidated
    into an ``active`` key per ticker for quick lookups.

``ActivePatternCalendar``
    Builds a time-filtered view of upcoming pattern activations and external
    events, suitable for daily checks of a ticker universe.

Patterns are expected to be stored as DataFrames containing the pattern payload
and a few canonical columns:

``active``
    Integer flag (1/0) indicating whether the pattern should be surfaced.
``features_fp``
    Optional path to pre-computed forecasting features for the ticker.
``event_time``
    Timestamp at which the pattern is expected to trigger.
``expected_turnout``
    Optional numeric expectation or scoring for the pattern.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import json

import pandas as pd

from ..config import RESULTS_HDF_PATH
from ..data import IntradayFileManager, ResultsClient
from ..data.data_client import DataClient
from ..strategy.backtester import BacktestSummary


def _slugify(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_/]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_/ ")


@dataclass
class CalendarEntry:
    """Materialised calendar entry for an upcoming pattern or event."""

    ticker: str
    screen_name: Optional[str]
    pattern: Optional[str]
    event_time: pd.Timestamp
    expected_turnout: Optional[Any]
    features_fp: Optional[str] = None
    source_key: Optional[str] = None


class PatternVault:
    """Persistent store for screener patterns and their activation state."""

    def __init__(
        self,
        results_client: Optional[ResultsClient] = None,
        *,
        hdf_path: Optional[Path] = None,
        features_root: Optional[Path] = None,
        intraday_manager: Optional[IntradayFileManager] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.results_client = results_client or ResultsClient(results_path=hdf_path)
        self.store_path = Path(hdf_path or self.results_client.results_path or RESULTS_HDF_PATH)
        self.features_root = Path(features_root) if features_root else None
        self.intraday_manager = intraday_manager
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_ticker(ticker: str) -> str:
        return str(ticker).strip().upper()

    def _pattern_key(self, ticker: str, screen_name: str) -> str:
        ticker_norm = self._normalise_ticker(ticker)
        name_norm = _slugify(screen_name)
        return f"screeners/{ticker_norm}/{name_norm}"

    def _active_key(self, ticker: str) -> str:
        ticker_norm = self._normalise_ticker(ticker)
        return f"screeners/{ticker_norm}/active"

    # ------------------------------------------------------------------
    # Feature path helper
    # ------------------------------------------------------------------
    def build_features_fp(self, ticker: str) -> Optional[str]:
        """Return a deterministic features file path for ``ticker`` if available."""

        ticker_norm = self._normalise_ticker(ticker)

        if self.features_root:
            return str(Path(self.features_root) / f"{ticker_norm}_features.parquet")

        if self.intraday_manager is not None:
            base = Path(getattr(self.intraday_manager, "data_path", ""))
            if base:
                return str(base / "features" / f"{ticker_norm}.parquet")

        return None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _serialise_backtest_summary(value: Any) -> Any:
        """Return an HDF-friendly representation of ``BacktestSummary`` objects."""

        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, BacktestSummary):
            payload = asdict(value)
        elif is_dataclass(value):
            payload = asdict(value)
        elif isinstance(value, Mapping):
            payload = dict(value)
        else:
            return value

        try:
            return json.dumps(payload)
        except TypeError:
            return str(payload)

    def _encode_backtest_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if "backtest_summary" in df.columns:
            df = df.copy()
            df["backtest_summary"] = df["backtest_summary"].apply(self._serialise_backtest_summary).astype(
                "string"
            )
        return df

    def _ensure_dataframe(self, payload: Any) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, pd.Series):
            return payload.to_frame().T
        if isinstance(payload, Mapping):
            try:
                return pd.json_normalize(payload)
            except Exception:
                return pd.DataFrame([payload])
        return pd.DataFrame([[payload]], columns=["value"])

    def _write_df(self, key: str, df: pd.DataFrame, *, replace: bool = True) -> None:
        df_sanitized = DataClient._sanitize_for_hdf(df)
        min_itemsize = DataClient._min_itemsize_for_str_cols(df_sanitized)

        fmt: Mapping[str, Any] = dict(
            format="table",
            data_columns=True,
            complib=self.results_client.complib,
            complevel=self.results_client.complevel,
            min_itemsize=min_itemsize or None,
        )

        with pd.HDFStore(self.store_path, mode="a") as store:
            writer = store.put if replace else store.append
            writer(key, df_sanitized, **fmt)

    def _read_df(self, key: str) -> pd.DataFrame:
        with pd.HDFStore(self.store_path, mode="r") as store:
            if key not in store:
                raise KeyError(f"Key '{key}' not found in {self.store_path}")
            return store[key]

    def _list_pattern_keys(self, ticker: str) -> list[str]:
        prefix = f"/screeners/{self._normalise_ticker(ticker)}/"
        with pd.HDFStore(self.store_path, mode="r") as store:
            return [
                key.lstrip("/")
                for key in store.keys()
                if key.startswith(prefix) and not key.endswith("/active")
            ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_patterns(
        self,
        ticker: str,
        screen_name: str,
        payload: Any,
        *,
        features_fp: Optional[str] = None,
        active: bool = True,
        replace: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Persist ``payload`` under the canonical screener key for ``ticker``."""

        key = self._pattern_key(ticker, screen_name)
        df = self._ensure_dataframe(payload)

        ticker_norm = self._normalise_ticker(ticker)
        if "ticker" not in df.columns:
            df["ticker"] = ticker_norm
        if "screen_name" not in df.columns:
            df["screen_name"] = screen_name

        default_features = features_fp or self.build_features_fp(ticker_norm)
        if default_features is not None:
            if "features_fp" not in df.columns:
                df["features_fp"] = default_features
            else:
                df["features_fp"] = df["features_fp"].fillna(default_features)

        default_active = 1 if active else 0
        if "active" not in df.columns:
            df["active"] = default_active
        else:
            df["active"] = df["active"].fillna(default_active)
        df["active"] = df["active"].astype(int)

        df = self._encode_backtest_summary(df)

        self._write_df(key, df, replace=replace)

        if metadata:
            with pd.HDFStore(self.store_path, mode="a") as store:
                store.get_storer(key).attrs.metadata = dict(metadata)

        self.refresh_active_patterns(ticker)
        return key

    def refresh_active_patterns(self, ticker: str) -> str:
        """Rebuild the consolidated ``active`` key for ``ticker``."""

        ticker_norm = self._normalise_ticker(ticker)
        active_key = self._active_key(ticker_norm)
        frames: list[pd.DataFrame] = []

        try:
            keys = self._list_pattern_keys(ticker_norm)
        except FileNotFoundError:
            keys = []

        for key in keys:
            try:
                df = self._read_df(key)
            except KeyError:
                continue

            df_work = df.copy()
            df_work["source_key"] = key
            if "active" not in df_work:
                df_work["active"] = 1
            df_active = df_work[df_work["active"] == 1]
            if not df_active.empty:
                frames.append(df_active)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._write_df(active_key, combined, replace=True)
        return active_key

    def get_patterns(self, ticker: str, screen_name: str) -> pd.DataFrame:
        key = self._pattern_key(ticker, screen_name)
        return self._read_df(key)

    def get_active_patterns(self, ticker: str) -> pd.DataFrame:
        active_key = self._active_key(ticker)
        try:
            return self._read_df(active_key)
        except KeyError:
            self.refresh_active_patterns(ticker)
            try:
                return self._read_df(active_key)
            except KeyError:
                return pd.DataFrame()


class ActivePatternCalendar:
    """Construct a calendar of upcoming active patterns and events."""

    def __init__(
        self,
        pattern_vault: PatternVault,
        *,
        default_horizon_hours: int = 24,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.pattern_vault = pattern_vault
        self.default_horizon_hours = default_horizon_hours
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_event_column(df: pd.DataFrame) -> Optional[str]:
        for candidate in ("event_time", "next_event", "ts", "timestamp"):
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _resolve_expected_column(df: pd.DataFrame) -> Optional[str]:
        for candidate in ("expected_turnout", "strength", "expectation"):
            if candidate in df.columns:
                return candidate
        return None

    def _normalise_events(self, events: Optional[Iterable[Mapping[str, Any]]]) -> pd.DataFrame:
        if events is None:
            return pd.DataFrame()
        if isinstance(events, pd.DataFrame):
            return events.copy()
        return pd.json_normalize(list(events))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_calendar(
        self,
        tickers: Sequence[str],
        *,
        events: Optional[Iterable[Mapping[str, Any]]] = None,
        horizon_hours: Optional[int] = None,
        now: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Return a consolidated calendar of active patterns and provided events."""

        start_ts = pd.Timestamp(now or datetime.utcnow())
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        horizon = timedelta(hours=horizon_hours or self.default_horizon_hours)
        end_ts = start_ts + horizon

        rows: list[CalendarEntry] = []

        for ticker in tickers:
            df = self.pattern_vault.get_active_patterns(ticker)
            if df.empty:
                continue

            event_col = self._resolve_event_column(df)
            if event_col is None:
                continue

            events_df = df.copy()
            events_df[event_col] = pd.to_datetime(events_df[event_col], utc=True, errors="coerce")
            events_df = events_df.dropna(subset=[event_col])
            windowed = events_df[(events_df[event_col] >= start_ts) & (events_df[event_col] <= end_ts)]
            if windowed.empty:
                continue

            expected_col = self._resolve_expected_column(windowed)

            for _, row in windowed.iterrows():
                rows.append(
                    CalendarEntry(
                        ticker=self.pattern_vault._normalise_ticker(ticker),
                        screen_name=row.get("screen_name"),
                        pattern=row.get("pattern")
                        or row.get("pattern_type")
                        or row.get("description"),
                        event_time=row[event_col],
                        expected_turnout=row.get(expected_col) if expected_col else None,
                        features_fp=row.get("features_fp") or self.pattern_vault.build_features_fp(ticker),
                        source_key=row.get("source_key"),
                    )
                )

        # Combine with external events if provided
        extra_events = self._normalise_events(events)
        if not extra_events.empty:
            event_col = self._resolve_event_column(extra_events)
            if event_col:
                extra_events[event_col] = pd.to_datetime(extra_events[event_col], utc=True, errors="coerce")
                extra_events = extra_events.dropna(subset=[event_col])
                extra_events = extra_events[
                    (extra_events[event_col] >= start_ts) & (extra_events[event_col] <= end_ts)
                ]
                for _, row in extra_events.iterrows():
                    rows.append(
                        CalendarEntry(
                            ticker=row.get("ticker", ""),
                            screen_name=row.get("screen_name"),
                            pattern=row.get("pattern") or row.get("description"),
                            event_time=row[event_col],
                            expected_turnout=row.get("expected_turnout"),
                            features_fp=row.get("features_fp"),
                            source_key=row.get("source_key"),
                        )
                    )

        calendar_df = pd.DataFrame([entry.__dict__ for entry in rows])
        if not calendar_df.empty:
            calendar_df = calendar_df.sort_values("event_time").reset_index(drop=True)
        return calendar_df
