"""Utilities for tracking data update metadata and scheduling refresh jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from ..config import MARKET_DATA_PATH

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .data_client import DataClient


# ---------------------------------------------------------------------------
# Event keys
# ---------------------------------------------------------------------------
WEEKLY_MARKET_UPDATE_EVENT = "weekly_market_update"
INTRADAY_UPDATE_EVENT = "intraday_update"
DLY_UPDATE_EVENT = "dly_curve_update"
COT_REFRESH_EVENT = "cot_refresh"


def _default_metadata_path() -> Path:
    base = MARKET_DATA_PATH
    if base.suffix:
        base_dir = base.parent
    else:
        base_dir = base
    return base_dir / "market_update_metadata.json"


def _json_safe(value: Any) -> Any:
    """Convert ``value`` into a JSON-serializable representation."""

    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:  # pragma: no cover - fallback
            pass
    return repr(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UpdateMetadataStore:
    """Thread-safe metadata store persisted as JSON."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else _default_metadata_path()
        self._lock = RLock()
        self._cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        with self._lock:
            if self._cache is not None:
                return self._cache

            if not self.path.exists():
                data = {"version": 1, "events": {}}
                self._cache = data
                return data

            try:
                raw = self.path.read_text(encoding="utf-8")
                data = json.loads(raw)
            except Exception:
                data = {"version": 1, "events": {}}

            if "events" not in data or not isinstance(data["events"], dict):
                data["events"] = {}

            self._cache = data
            return data

    def _save(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            self._cache = data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_event(self, event_key: str) -> Dict[str, Any]:
        data = self._load()
        return data.setdefault("events", {}).setdefault(event_key, {})

    def get_last_success(self, event_key: str) -> Optional[datetime]:
        record = self.get_event(event_key)
        ts = record.get("last_success")
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    def record_attempt(
        self,
        event_key: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        data = self._load()
        record = data.setdefault("events", {}).setdefault(event_key, {})
        ts = (timestamp or _utcnow()).astimezone(timezone.utc).isoformat()
        record["last_attempt"] = ts
        if details:
            record["last_attempt_details"] = _json_safe(details)
        self._save(data)

    def record_success(
        self,
        event_key: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        data = self._load()
        record = data.setdefault("events", {}).setdefault(event_key, {})
        ts = (timestamp or _utcnow()).astimezone(timezone.utc).isoformat()
        record["last_success"] = ts
        if details is not None:
            record["last_success_details"] = _json_safe(details)
        self._save(data)

    def record_failure(
        self,
        event_key: str,
        error: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        data = self._load()
        record = data.setdefault("events", {}).setdefault(event_key, {})
        ts = (timestamp or _utcnow()).astimezone(timezone.utc).isoformat()
        record["last_failure"] = ts
        record["last_error"] = str(error)
        if details is not None:
            record["last_failure_details"] = _json_safe(details)
        self._save(data)


_STORE: Optional[UpdateMetadataStore] = None


def get_update_metadata_store(path: Optional[Path] = None) -> UpdateMetadataStore:
    global _STORE
    if _STORE is None:
        _STORE = UpdateMetadataStore(path)
    elif path is not None and Path(path) != _STORE.path:
        _STORE = UpdateMetadataStore(path)
    return _STORE


@dataclass
class UpdateScheduler:
    """Simple interval-based scheduler for update events."""

    event_key: str
    interval: timedelta
    store: UpdateMetadataStore

    def is_due(self, now: Optional[datetime] = None) -> bool:
        last_success = self.store.get_last_success(self.event_key)
        if last_success is None:
            return True
        now = now or _utcnow()
        return now - last_success >= self.interval

    def mark_attempt(self, details: Optional[Dict[str, Any]] = None) -> None:
        self.store.record_attempt(self.event_key, details=details)

    def mark_success(self, details: Optional[Dict[str, Any]] = None) -> None:
        self.store.record_success(self.event_key, details=details)

    def mark_failure(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.store.record_failure(self.event_key, error=error, details=details)


_WEEKLY_SCHEDULER: Optional[UpdateScheduler] = None


def get_weekly_update_scheduler(interval: timedelta = timedelta(days=7)) -> UpdateScheduler:
    global _WEEKLY_SCHEDULER
    store = get_update_metadata_store()
    if _WEEKLY_SCHEDULER is None or _WEEKLY_SCHEDULER.interval != interval:
        _WEEKLY_SCHEDULER = UpdateScheduler(
            event_key=WEEKLY_MARKET_UPDATE_EVENT,
            interval=interval,
            store=store,
        )
    return _WEEKLY_SCHEDULER


# ---------------------------------------------------------------------------
# DataFrame preparation helpers
# ---------------------------------------------------------------------------
def prepare_dataframe_for_append(
    client: "DataClient",
    key: str,
    frame: pd.DataFrame,
    *,
    allow_schema_expansion: bool = False,
    sort_columns: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """Prepare ``frame`` for appending to ``key`` using ``DataClient``.

    Returns a tuple of (prepared_frame, requires_replace). ``requires_replace``
    is True when schema expansion is necessary and the caller should perform a
    replace write with the returned DataFrame.
    """

    if frame is None or frame.empty:
        return pd.DataFrame(), False

    prepared = frame.copy()
    if not isinstance(prepared.index, pd.DatetimeIndex):
        prepared.index = pd.to_datetime(prepared.index)
    prepared = prepared.sort_index()
    prepared = prepared[~prepared.index.duplicated(keep="last")]

    if not client.market_key_exists(key):
        if sort_columns:
            prepared = prepared.reindex(columns=sorted(prepared.columns))
        return prepared, False

    original_columns = list(prepared.columns)

    tail = client.get_market_tail(key, nrows=1)
    existing_columns: Optional[pd.Index]
    last_index: Optional[pd.Timestamp]
    if not tail.empty:
        tail = tail.sort_index()
        existing_columns = tail.columns
        last_index = tail.index[-1]
    else:
        try:
            sample = client.read_market(key, start=0, stop=1)
        except KeyError:
            sample = pd.DataFrame()
        existing_columns = sample.columns if not sample.empty else None
        last_index = None

    if last_index is not None:
        prepared = prepared.loc[prepared.index > last_index]
    if prepared.empty:
        return pd.DataFrame(columns=existing_columns), False

    extra_columns = []
    if existing_columns is not None and len(existing_columns) > 0:
        missing_columns = [col for col in existing_columns if col not in prepared.columns]
        for col in missing_columns:
            prepared[col] = pd.NA
        extra_columns = [col for col in original_columns if col not in existing_columns]
        ordered_cols = list(existing_columns) + [col for col in prepared.columns if col not in existing_columns]
        prepared = prepared.reindex(columns=ordered_cols)

    if extra_columns and not allow_schema_expansion:
        prepared = prepared.drop(columns=extra_columns)
        extra_columns = []

    if extra_columns:
        try:
            existing_full = client.read_market(key)
        except KeyError:
            existing_full = pd.DataFrame()
        combined = pd.concat([existing_full, prepared], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        if sort_columns:
            combined = combined.reindex(columns=sorted(combined.columns))
        return combined, True

    if sort_columns:
        prepared = prepared.reindex(columns=sorted(prepared.columns))

    return prepared, False


# ---------------------------------------------------------------------------
# Result summarizers used for metadata recording
# ---------------------------------------------------------------------------
def summarize_append_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": result.get("mode"),
        "rows_written": result.get("rows_written", 0),
        "total_rows": result.get("total_rows", 0),
        "delta_rows": result.get("delta_rows", 0),
        "removed_rows": result.get("removed_rows", 0),
    }


def summarize_dly_update(result: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    updated = result.get("updated", {})
    total_updated = len(updated)
    total_rows = sum(v.get("rows_written", 0) or v.get("total_rows", 0) for v in updated.values())
    return {
        "ticker": ticker,
        "datasets_updated": total_updated,
        "total_rows": total_rows,
        "skipped": len(result.get("skipped", [])),
        "errors": list(result.get("errors", [])) if isinstance(result.get("errors"), (list, tuple, set)) else [],
    }


def summarize_cot_refresh(result: Dict[str, Any], report_type: str) -> Dict[str, Any]:
    ticker_results = result.get("ticker_results", {})
    success = ticker_results.get("success", []) if isinstance(ticker_results, dict) else []
    failed = ticker_results.get("failed", {}) if isinstance(ticker_results, dict) else {}
    return {
        "report_type": report_type,
        "raw_key": result.get("raw_key"),
        "total_rows": result.get("total_rows", 0),
        "tickers_success": len(success),
        "tickers_failed": len(failed),
    }


def summarize_update_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    updates = summary.get("updates", {})
    market_updates = sum(1 for details in updates.values() if details.get("market_result"))
    curve_updates = sum(1 for details in updates.values() if details.get("curve_result"))
    errors = summary.get("errors", {})
    return {
        "tickers_considered": len(updates),
        "market_updates": market_updates,
        "curve_updates": curve_updates,
        "skipped": len(summary.get("skipped", [])),
        "errors": len(errors),
    }

