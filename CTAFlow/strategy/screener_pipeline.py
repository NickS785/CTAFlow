"""Feature pipeline that converts screener patterns into sparse gates.

The :class:`ScreenerPipeline` class consumes the pattern payloads produced by
seasonality and orderflow screeners and materialises them as boolean "gate"
columns on a price/volume bar DataFrame. Each gate is accompanied by a set of
sidecar columns (strength metrics, metadata, bias direction, etc.) so that the
result can be consumed directly by downstream strategy research code.

The implementation is intentionally vectorised – even when hundreds of patterns
are supplied, the transformations operate via boolean masks instead of Python
loops so that millions of rows can be processed quickly.
"""

from __future__ import annotations

import numbers
import re
from datetime import time as time_cls
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["ScreenerPipeline"]


class ScreenerPipeline:
    """Normalise screener output and append sparse feature gates to bar data."""

    #: Pattern types handled by the seasonal extractor branch
    _SEASONAL_TYPES = {
        "weekday_mean",
        "weekday_returns",
        "time_predictive_nextday",
        "time_predictive_nextweek",
    }

    #: Pattern types handled by the orderflow extractor branch
    _ORDERFLOW_TYPES = {
        "orderflow_week_of_month",
        "orderflow_weekly",
        "orderflow_peak_pressure",
    }

    def __init__(self, tz: str = "America/Chicago", time_match: str = "auto", log: Any = None) -> None:
        """Configure the pipeline.

        Parameters
        ----------
        tz:
            Timezone used to localise naive timestamps. Existing aware timestamps
            are converted into this zone so that month and weekday logic remains
            consistent.
        time_match:
            Controls how intraday time comparisons are performed. Accepted
            values are ``"auto"`` (default), ``"hms"`` (second resolution),
            and ``"hmsf"`` (microsecond precision).
        log:
            Optional logger with a ``warning`` method for non-fatal issues.
        """

        if time_match not in {"auto", "hms", "hmsf"}:
            raise ValueError("time_match must be one of {'auto', 'hms', 'hmsf'}")

        self.tz = tz
        self.time_match = time_match
        self.log = log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_features(self, bars: pd.DataFrame, patterns: Any) -> pd.DataFrame:
        """Return a copy of ``bars`` with pattern gates appended.

        Parameters
        ----------
        bars:
            Bar data containing a ``ts`` timestamp column. The column is
            normalised to be timezone-aware and enriched with calendar columns
            used by the gate logic.
        patterns:
            Seasonality and orderflow pattern payloads. This can be the mapping
            returned by :class:`~CTAFlow.screeners.pattern_extractor.PatternExtractor`,
            a list of pattern dictionaries, or a single pattern dictionary.
        """

        df = self._validate_bars(bars)
        df = self._ensure_time_cols(df)

        gate_columns: List[str] = []
        for key, pattern in self._items_from_patterns(patterns):
            try:
                created = self._dispatch(df, pattern, key)
            except Exception as exc:  # pragma: no cover - defensive logging path
                if self.log is not None and hasattr(self.log, "warning"):
                    self.log.warning("[screener_pipeline] skipping '%s': %s", key, exc)
                continue

            gate_columns.extend(created)

        if gate_columns:
            any_active = df[gate_columns].any(axis=1)
            df["any_pattern_active"] = any_active.astype(np.int8)
        else:
            df["any_pattern_active"] = 0

        return df

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _validate_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(bars, pd.DataFrame):
            raise TypeError("bars must be a pandas DataFrame")
        if "ts" not in bars.columns:
            raise ValueError("bars must contain a 'ts' column")

        df = bars.copy()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        if df["ts"].isna().any():
            raise ValueError("bars['ts'] contains non-coercible timestamps")

        tz = self.tz
        ts = df["ts"]
        if ts.dt.tz is None:
            df["ts"] = ts.dt.tz_localize(tz)
        else:
            df["ts"] = ts.dt.tz_convert(tz)

        return df

    def _ensure_time_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ts = out["ts"].dt.tz_convert(self.tz)

        out["month"] = ts.dt.month.astype(np.int8)
        out["weekday"] = ts.dt.day_name()
        out["weekday_lower"] = out["weekday"].str.lower()
        out["weekday_idx"] = ts.dt.weekday.astype(np.int8)

        normalized = ts.dt.normalize()
        first_of_month = normalized - pd.to_timedelta(ts.dt.day - 1, unit="D")
        wom = 1 + ((ts.dt.day + first_of_month.dt.weekday - 1) // 7)
        out["wom"] = wom.astype(np.int8)

        out["clock_time"] = ts.dt.strftime("%H:%M:%S")
        out["clock_time_us"] = ts.dt.strftime("%H:%M:%S.%f")

        if "session_id" not in out.columns:
            out["session_id"] = ts.dt.strftime("%Y-%m-%d")

        return out

    def _items_from_patterns(self, patterns: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        if patterns is None:
            return []

        def _iter(obj: Any, key_hint: Optional[str] = None) -> Iterable[Tuple[str, Mapping[str, Any]]]:
            if isinstance(obj, Mapping):
                if "pattern_type" in obj:
                    key = self._select_pattern_key(obj, key_hint)
                    yield key, obj
                    return

                for child_key, value in obj.items():
                    if isinstance(value, Mapping):
                        yield from _iter(value, str(child_key))
                    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                        for idx, item in enumerate(value):
                            yield from _iter(item, str(child_key))
                return

            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                for idx, item in enumerate(obj):
                    yield from _iter(item, key_hint or f"pattern_{idx}")

        return list(_iter(patterns))

    @staticmethod
    def _select_pattern_key(pattern: Mapping[str, Any], key_hint: Optional[str]) -> str:
        for candidate in (pattern.get("key"), pattern.get("id"), key_hint, pattern.get("description")):
            if candidate:
                return str(candidate)
        return "pattern"

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------
    def _feature_base_name(self, key: Optional[str], fallback: str) -> str:
        raw = key or fallback
        slug = self._slugify(raw)
        return slug or self._slugify(fallback) or "pattern"

    @staticmethod
    def _slugify(value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^0-9a-z]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    @staticmethod
    def _normalize_weekday(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, numbers.Integral):
            names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            return names[int(value) % 7]
        return str(value).strip().lower() or None

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return isinstance(value, numbers.Number) and not isinstance(value, bool)

    def _broadcast_sidecar(self, df: pd.DataFrame, mask: pd.Series, value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            series = value.reindex(df.index)
            series.loc[~mask] = np.nan
            return series

        if isinstance(value, np.ndarray) and value.shape == (len(df),):
            series = pd.Series(value, index=df.index)
            series.loc[~mask] = np.nan
            return series

        dtype = float if self._is_numeric(value) else object
        series = pd.Series(np.nan, index=df.index, dtype=dtype)
        if mask.any():
            series.loc[mask] = value
        return series

    def _add_feature(
        self,
        df: pd.DataFrame,
        base_name: str,
        mask: pd.Series,
        sidecars: MutableMapping[str, Any],
    ) -> List[str]:
        if mask.empty:
            return []

        clean_mask = mask.fillna(False).astype(bool)
        gate_name = f"{base_name}_gate"
        df[gate_name] = clean_mask.astype(np.int8)

        for key, value in sidecars.items():
            col_name = f"{base_name}_{self._slugify(key)}"
            df[col_name] = self._broadcast_sidecar(df, clean_mask, value)

        return [gate_name]

    # ------------------------------------------------------------------
    # Pattern parsing helpers
    # ------------------------------------------------------------------
    def _resolve_months(self, key: Optional[str], pattern: Mapping[str, Any]) -> Optional[List[int]]:
        for candidate in (
            pattern.get("months"),
            pattern.get("metadata", {}).get("months"),
            pattern.get("pattern_payload", {}).get("months"),
        ):
            months = self._coerce_months(candidate)
            if months:
                return months

        params = pattern.get("screen_parameters") or {}
        for key_name in ("months", "seasonal_months", "seasonality_months"):
            months = self._coerce_months(params.get(key_name))
            if months:
                return months

        return self._parse_months_from_key(key)

    def _coerce_months(self, value: Any) -> Optional[List[int]]:
        if value in (None, "", [], ()):  # type: ignore[comparison-overlap]
            return None

        if isinstance(value, str):
            parts = re.split(r"[^0-9]+", value)
            items = [p for p in parts if p]
        elif isinstance(value, Sequence):
            items = list(value)
        else:
            items = [value]

        months: List[int] = []
        for item in items:
            try:
                month = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= month <= 12:
                months.append(month)

        if not months:
            return None
        return sorted(set(months))

    @staticmethod
    def _parse_months_from_key(key: Optional[str]) -> Optional[List[int]]:
        if not key:
            return None
        head = str(key).split("|")[0]
        match = re.match(r"months_([0-9_]+)_seasonality", head)
        if not match:
            return None
        parts = [p for p in match.group(1).split("_") if p]
        months: List[int] = []
        for part in parts:
            try:
                month = int(part)
            except ValueError:
                continue
            if 1 <= month <= 12:
                months.append(month)
        if not months:
            return None
        return months

    def _months_mask(self, df: pd.DataFrame, months: Optional[List[int]]) -> pd.Series:
        if not months:
            return pd.Series(True, index=df.index, dtype=bool)
        return df["month"].isin(months)

    # ------------------------------------------------------------------
    # Pattern dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, df: pd.DataFrame, pattern: Mapping[str, Any], key: Optional[str]) -> List[str]:
        pattern_type = str(pattern.get("pattern_type") or pattern.get("type") or "").strip()
        if not pattern_type:
            return []

        if pattern_type in self._SEASONAL_TYPES:
            return self._dispatch_seasonal(df, pattern, key, pattern_type)
        if pattern_type in self._ORDERFLOW_TYPES:
            return self._dispatch_orderflow(df, pattern, key, pattern_type)

        return []

    def _dispatch_seasonal(
        self,
        df: pd.DataFrame,
        pattern: Mapping[str, Any],
        key: Optional[str],
        pattern_type: str,
    ) -> List[str]:
        months = self._resolve_months(key, pattern)
        payload = pattern.get("pattern_payload", {})

        if pattern_type in {"weekday_mean", "weekday_returns"}:
            return self._extract_weekday_mean(df, payload, key, months)
        if pattern_type == "time_predictive_nextday":
            return self._extract_time_nextday(df, payload, key, months)
        if pattern_type == "time_predictive_nextweek":
            return self._extract_time_nextweek(df, payload, key, months)

        return []

    def _dispatch_orderflow(
        self,
        df: pd.DataFrame,
        pattern: Mapping[str, Any],
        key: Optional[str],
        pattern_type: str,
    ) -> List[str]:
        payload = pattern.get("pattern_payload", {})
        metadata = pattern.get("metadata", {})

        if pattern_type == "orderflow_week_of_month":
            return self._extract_oflow_wom(df, payload, metadata, key)
        if pattern_type == "orderflow_weekly":
            return self._extract_oflow_weekly(df, payload, metadata, key)
        if pattern_type == "orderflow_peak_pressure":
            return self._extract_oflow_peak(df, payload, metadata, key)

        return []

    # ------------------------------------------------------------------
    # Seasonal extractors
    # ------------------------------------------------------------------
    def _extract_weekday_mean(
        self,
        df: pd.DataFrame,
        payload: Mapping[str, Any],
        key: Optional[str],
        months: Optional[List[int]],
    ) -> List[str]:
        weekday = payload.get("day") or payload.get("weekday")
        weekday_norm = self._normalize_weekday(weekday)
        if weekday_norm is None:
            return []

        mask = df["weekday_lower"] == weekday_norm
        mask &= self._months_mask(df, months)

        base = self._feature_base_name(key, f"weekday_mean_{weekday_norm}")
        sidecars: Dict[str, Any] = {
            "weekday": weekday_norm,
            "mean": payload.get("mean"),
            "t": payload.get("t_stat"),
            "p": payload.get("p_value"),
            "strength": payload.get("strength"),
        }
        return self._add_feature(df, base, mask, sidecars)

    def _extract_time_nextday(
        self,
        df: pd.DataFrame,
        payload: Mapping[str, Any],
        key: Optional[str],
        months: Optional[List[int]],
    ) -> List[str]:
        hms, hmsf = self._time_to_strings(payload.get("time"))
        if hms is None:
            return []

        use_us = self._use_microseconds(df, payload.get("time"))
        column = "clock_time_us" if use_us else "clock_time"
        target = hmsf if use_us else hms

        mask = df[column] == target
        mask &= self._months_mask(df, months)

        base = self._feature_base_name(key, f"time_nextday_{target}")
        sidecars = {
            "time": target,
            "correlation": payload.get("correlation"),
            "p": payload.get("p_value"),
        }
        return self._add_feature(df, base, mask, sidecars)

    def _extract_time_nextweek(
        self,
        df: pd.DataFrame,
        payload: Mapping[str, Any],
        key: Optional[str],
        months: Optional[List[int]],
    ) -> List[str]:
        hms, hmsf = self._time_to_strings(payload.get("time"))
        if hms is None:
            return []

        use_us = self._use_microseconds(df, payload.get("time"))
        column = "clock_time_us" if use_us else "clock_time"
        target = hmsf if use_us else hms

        mask = df[column] == target
        mask &= self._months_mask(df, months)

        strongest_days = payload.get("strongest_days") or []
        if strongest_days:
            valid_days = {self._normalize_weekday(day) for day in strongest_days if day is not None}
            valid_days.discard(None)  # type: ignore[arg-type]
            if valid_days:
                mask &= df["weekday_lower"].isin(valid_days)

        base = self._feature_base_name(key, f"time_nextweek_{target}")
        sidecars = {
            "time": target,
            "correlation": payload.get("correlation"),
            "p": payload.get("p_value"),
            "strongest_days": list(strongest_days) if strongest_days else None,
        }
        return self._add_feature(df, base, mask, sidecars)

    # ------------------------------------------------------------------
    # Orderflow extractors
    # ------------------------------------------------------------------
    def _extract_oflow_wom(
        self,
        df: pd.DataFrame,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any],
        key: Optional[str],
    ) -> List[str]:
        weekday_norm = self._normalize_weekday(payload.get("weekday"))
        if weekday_norm is None:
            return []

        try:
            week_of_month = int(payload.get("week_of_month"))
        except (TypeError, ValueError):
            return []

        mask = (df["weekday_lower"] == weekday_norm) & (df["wom"].astype(int) == week_of_month)

        metric = payload.get("metric", "net_pressure")
        bias = self._format_bias(metadata.get("orderflow_bias") or payload.get("pressure_bias"))
        base = self._feature_base_name(key, f"oflow_wom_{weekday_norm}_w{week_of_month}_{metric}_{bias}")

        sidecars = {
            "weekday": weekday_norm,
            "week_of_month": week_of_month,
            "metric": metric,
            "t": payload.get("t_stat"),
            "q": payload.get("q_value"),
            "mean": payload.get("mean"),
            "n": payload.get("n"),
            "bias": bias,
        }
        return self._add_feature(df, base, mask, sidecars)

    def _extract_oflow_weekly(
        self,
        df: pd.DataFrame,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any],
        key: Optional[str],
    ) -> List[str]:
        weekday_norm = self._normalize_weekday(payload.get("weekday"))
        if weekday_norm is None:
            return []

        mask = df["weekday_lower"] == weekday_norm

        metric = payload.get("metric", "net_pressure")
        bias = self._format_bias(metadata.get("orderflow_bias") or payload.get("pressure_bias"))
        base = self._feature_base_name(key, f"oflow_weekly_{weekday_norm}_{metric}_{bias}")

        sidecars = {
            "weekday": weekday_norm,
            "metric": metric,
            "t": payload.get("t_stat"),
            "q": payload.get("q_value"),
            "mean": payload.get("mean"),
            "n": payload.get("n"),
            "bias": bias,
        }
        return self._add_feature(df, base, mask, sidecars)

    def _extract_oflow_peak(
        self,
        df: pd.DataFrame,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any],
        key: Optional[str],
    ) -> List[str]:
        weekday_norm = self._normalize_weekday(payload.get("weekday"))
        if weekday_norm is None:
            return []

        hms, hmsf = self._time_to_strings(payload.get("clock_time"))
        if hms is None:
            return []

        use_us = self._use_microseconds(df, payload.get("clock_time"))
        column = "clock_time_us" if use_us else "clock_time"
        target = hmsf if use_us else hms

        mask = (df["weekday_lower"] == weekday_norm) & (df[column] == target)

        metric = payload.get("metric", "net_pressure")
        bias = self._format_bias(metadata.get("orderflow_bias") or payload.get("pressure_bias"))
        base = self._feature_base_name(key, f"oflow_peak_{weekday_norm}_{target}_{metric}_{bias}")

        sidecars = {
            "weekday": weekday_norm,
            "time": target,
            "metric": metric,
            "bias": bias,
            "seasonality_mean": payload.get("seasonality_mean"),
            "seasonality_t": payload.get("seasonality_t_stat"),
            "seasonality_q": payload.get("seasonality_q_value"),
            "intraday_mean": payload.get("intraday_mean"),
            "intraday_n": payload.get("intraday_n"),
        }
        return self._add_feature(df, base, mask, sidecars)

    # ------------------------------------------------------------------
    # Time utilities
    # ------------------------------------------------------------------
    def _time_to_strings(self, value: Any) -> Tuple[Optional[str], Optional[str]]:
        if value is None:
            return None, None

        if isinstance(value, pd.Timestamp):  # type: ignore[name-defined]
            value = value.time()
        elif isinstance(value, np.datetime64):
            value = pd.Timestamp(value).time()

        if isinstance(value, time_cls):
            hms = value.strftime("%H:%M:%S")
            return hms, value.strftime("%H:%M:%S.%f")

        text = str(value).strip()
        if not text:
            return None, None

        if "." in text:
            head, frac = text.split(".", 1)
            frac = (frac + "000000")[:6]
            return head, f"{head}.{frac}"

        return text, f"{text}.000000"

    def _use_microseconds(self, df: pd.DataFrame, time_value: Any) -> bool:
        if self.time_match == "hmsf":
            return True
        if self.time_match == "hms":
            return False

        has_microseconds = df["ts"].dt.microsecond.ne(0).any()
        if not has_microseconds:
            return False

        if isinstance(time_value, time_cls):
            return bool(time_value.microsecond)

        try:
            text = str(time_value)
        except Exception:
            return False
        return "." in text and not text.endswith(".000000")

    @staticmethod
    def _format_bias(value: Any) -> str:
        if value is None:
            return "na"
        normalized = str(value).strip().lower()
        if normalized in {"buy", "positive", "long"}:
            return "buy"
        if normalized in {"sell", "negative", "short"}:
            return "sell"
        return "na"

