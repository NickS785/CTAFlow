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
from collections.abc import Iterable as IterableABC, Sequence as SequenceABC
from datetime import time as time_cls
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, NamedTuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from CTAFlow.screeners.pattern_extractor import PatternExtractor
    from .sessionizer import Sessionizer

__all__ = ["ScreenerPipeline", "extract_ticker_patterns", "HorizonMapper", "HorizonSpec"]


def extract_ticker_patterns(
    ticker: str,
    data: Mapping[str, pd.DataFrame],
    extractors: Sequence["PatternExtractor"],
    *,
    pattern_types: Optional[Iterable[str]] = None,
    screen_names: Optional[Iterable[str]] = None,
    pipeline: Optional["ScreenerPipeline"] = None,
) -> pd.DataFrame:
    """Extract screener pattern columns for ``ticker`` without bar data duplicates.

    The helper consolidates patterns from multiple :class:`PatternExtractor`
    instances, materialises the corresponding features once via
    :class:`ScreenerPipeline`, and returns only the resulting pattern columns.

    Parameters
    ----------
    ticker:
        The symbol whose patterns should be extracted.
    data:
        Mapping of ticker symbols to their bar data.
    extractors:
        Sequence of :class:`PatternExtractor` instances supplying pattern
        payloads.
    pattern_types:
        Optional iterable of pattern types to include. When provided the
        extractor output is filtered accordingly.
    screen_names:
        Optional iterable of screen identifiers to include.
    pipeline:
        Optional :class:`ScreenerPipeline` instance. When omitted a fresh
        instance is created with default settings.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the pattern columns generated for
        ``ticker``. When no patterns are available an empty DataFrame with the
        same index as the source bar data is returned.
    """

    if ticker not in data:
        raise KeyError(f"Ticker '{ticker}' is not present in the provided data mapping")

    bars = data[ticker]
    if pipeline is None:
        pipeline = ScreenerPipeline()

    baseline = pipeline.build_features(bars, [])
    baseline_columns = set(baseline.columns)

    combined_patterns: Dict[str, Mapping[str, Any]] = {}
    for extractor in extractors:
        filtered = extractor.filter_patterns(
            ticker,
            pattern_types=pattern_types,
            screen_names=screen_names,
        )
        for key, pattern in filtered.items():
            combined_patterns.setdefault(key, pattern)

    if not combined_patterns:
        return pd.DataFrame(index=bars.index.copy())

    enriched = pipeline.build_features(bars, combined_patterns)
    pattern_columns = [col for col in enriched.columns if col not in baseline_columns]

    if not pattern_columns:
        return pd.DataFrame(index=enriched.index.copy())

    return enriched.loc[:, pattern_columns].copy()


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

        if "ts" in bars.columns:
            df = bars.copy()
        elif isinstance(bars.index, pd.DatetimeIndex):
            df = bars.copy()
            df["ts"] = bars.index
        else:
            raise ValueError("bars must contain a 'ts' column or a DatetimeIndex")

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

    @classmethod
    def _items_from_patterns(cls, patterns: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        if patterns is None:
            return ()

        return tuple(cls._iter_patterns(patterns))

    @classmethod
    def _iter_patterns(
        cls, obj: Any, key_hint: Optional[str] = None
    ) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        if isinstance(obj, Mapping):
            if "pattern_type" in obj or "type" in obj:
                key = cls._select_pattern_key(obj, key_hint)
                yield key, obj
                return

            for child_key, value in obj.items():
                next_hint = str(child_key) if child_key is not None else key_hint
                yield from cls._iter_patterns(value, next_hint)
            return

        if isinstance(obj, SequenceABC) and not isinstance(obj, (str, bytes, bytearray)):
            if len(obj) == 2 and isinstance(obj[1], Mapping):
                first, second = obj
                hint = key_hint
                if isinstance(first, (str, numbers.Integral)):
                    hint = str(first)
                elif first is not None:
                    hint = str(first)
                yield from cls._iter_patterns(second, hint)
                return

        if isinstance(obj, IterableABC) and not isinstance(obj, (str, bytes, bytearray)):
            for idx, item in enumerate(obj):
                if isinstance(item, Mapping):
                    yield from cls._iter_patterns(item, key_hint)
                    continue

                if isinstance(item, SequenceABC) and not isinstance(item, (str, bytes, bytearray)):
                    if len(item) == 2 and isinstance(item[1], Mapping):
                        first, second = item
                        hint = key_hint
                        if isinstance(first, (str, numbers.Integral)):
                            hint = str(first)
                        elif first is not None:
                            hint = str(first)
                        yield from cls._iter_patterns(second, hint)
                        continue

                    yield from cls._iter_patterns(item, key_hint)
                    continue

                next_hint = key_hint or f"pattern_{idx}"
                yield from cls._iter_patterns(item, next_hint)

        return

    @staticmethod
    def _select_pattern_key(pattern: Mapping[str, Any], key_hint: Optional[str]) -> str:
        for candidate in (pattern.get("key"), pattern.get("id"), key_hint, pattern.get("description")):
            if candidate:
                return str(candidate)
        return "pattern"

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------
    @classmethod
    def _feature_base_name(cls, key: Optional[str], fallback: str) -> str:
        raw = key or fallback
        slug = cls._slugify(raw)
        return slug or cls._slugify(fallback) or "pattern"

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
            "intraday_mean": payload.get("intraday_mean"),
            "intraday_n": payload.get("intraday_n"),
        }
        return self._add_feature(df, base, mask, sidecars)

    # ------------------------------------------------------------------
    # Time utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _time_to_strings(value: Any) -> Tuple[Optional[str], Optional[str]]:
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

        try:
            parsed = pd.to_datetime(text).time()
        except Exception:
            if "." in text:
                head, frac = text.split(".", 1)
                frac = (frac + "000000")[:6]
                return head, f"{head}.{frac}"
            return text, f"{text}.000000"

        hms = parsed.strftime("%H:%M:%S")
        return hms, parsed.strftime("%H:%M:%S.%f")

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


class HorizonSpec(NamedTuple):
    """Description of the realised return horizon for a pattern."""

    name: str
    delta_minutes: Optional[int] = None


class HorizonError(Exception):
    """Base exception for horizon mapping failures."""


class HorizonInputError(HorizonError):
    """Raised when bar inputs fail validation or sanitation."""


class HorizonGateError(HorizonError):
    """Raised when gate realisation fails for a requested pattern."""


class HorizonMapper:
    """Derive (returns_x, returns_y) pairs for screener pattern gates.

    Parameters
    ----------
    tz:
        Target timezone used to normalise timestamp columns. Defaults to ``"UTC"``.
    allow_naive_ts:
        When ``False`` (default) timezone-naive timestamps result in a
        :class:`HorizonInputError`. Set to ``True`` to localise naive timestamps
        into ``tz``.
    time_match:
        Default policy for resolving time-bearing gate names. Accepted values are
        ``"auto"``, ``"second"``, and ``"microsecond"``.
    nan_policy:
        Strategy used by :meth:`build_xy` when cleaning realised returns. The
        default ``"drop"`` removes rows containing non-finite values. ``"zero"``
        replaces them with zero while ``"ffill"`` forward-fills from the most
        recent finite observation.
    return_clip:
        Tuple ``(lower, upper)`` used to clip extreme realised returns. Defaults
        to ``(-0.5, 0.5)``.
    asof_tolerance:
        Default tolerance for merge-asof alignment expressed as a pandas
        offset string (for example ``"1s"``). ``None`` disables tolerance
        checking.
    log:
        Optional logger exposing ``warning`` and ``info`` methods. When
        provided, structured diagnostics are emitted for missing or empty gates.
    """

    def __init__(
        self,
        *,
        tz: str = "UTC",
        allow_naive_ts: bool = False,
        time_match: str = "auto",
        nan_policy: str = "drop",
        return_clip: Tuple[float, float] = (-0.5, 0.5),
        asof_tolerance: Optional[str] = None,
        log: Any = None,
        sessionizer: Optional["Sessionizer"] = None,
    ) -> None:
        if time_match not in {"auto", "second", "microsecond"}:
            raise HorizonInputError(
                "time_match must be one of {'auto', 'second', 'microsecond'}"
            )

        if nan_policy not in {"drop", "zero", "ffill"}:
            raise HorizonInputError("nan_policy must be one of {'drop', 'zero', 'ffill'}")

        lower, upper = return_clip
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise HorizonInputError("return_clip must be a finite (lower, upper) tuple")

        self.tz = tz
        self.allow_naive_ts = allow_naive_ts
        self.time_match = time_match
        self.nan_policy = nan_policy
        self.return_clip = (float(lower), float(upper))
        self.asof_tolerance = asof_tolerance
        self.log = log
        self.sessionizer = sessionizer

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_warn(self, message: str, **context: Any) -> None:
        if self.log is not None and hasattr(self.log, "warning"):
            if context:
                self.log.warning("[HorizonMapper] %s | %s", message, context)
            else:
                self.log.warning("[HorizonMapper] %s", message)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _translate_time_match(policy: str) -> str:
        mapping = {"auto": "auto", "second": "hms", "microsecond": "hmsf"}
        return mapping.get(policy, "auto")

    def _resolve_time_match(self, df: pd.DataFrame, policy: str, time_value: Any) -> str:
        if policy == "auto":
            has_microseconds = df["ts"].dt.microsecond.ne(0).any()
            if has_microseconds:
                return "microsecond"
            text = str(time_value) if time_value is not None else ""
            if "." in text and not text.endswith(".000000"):
                return "microsecond"
            return "second"
        return policy

    # ------------------------------------------------------------------
    # Input sanitation helpers
    # ------------------------------------------------------------------
    def _require_columns(self, df: pd.DataFrame, required: Iterable[str]) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise HorizonInputError(
                "bars_with_features is missing required columns: "
                + ", ".join(sorted(missing))
            )

    def _sanitize_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ts = pd.to_datetime(out["ts"], errors="coerce", utc=False)
        if ts.isna().any():
            sample = out.loc[ts.isna()].head(5)
            raise HorizonInputError(
                "bars_with_features['ts'] contains non-coercible timestamps. "
                f"Sample: {sample.to_dict(orient='records')}"
            )

        if ts.dt.tz is None:
            if not self.allow_naive_ts:
                raise HorizonInputError(
                    "Received timezone-naive 'ts' values. Pass allow_naive_ts=True "
                    "or localise timestamps (e.g. df['ts'] = df['ts'].dt.tz_localize('UTC')).",
                )
            out["ts"] = ts.dt.tz_localize(self.tz)
        else:
            out["ts"] = ts.dt.tz_convert(self.tz)
        return out

    def _ensure_session_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        needs_session = "session_id" not in df.columns or df["session_id"].isna().any()
        if not needs_session:
            return df

        if self.sessionizer is None:
            raise HorizonInputError(
                "session_id missing; provide a Sessionizer to HorizonMapper to auto-sessionize"
            )

        symbol = df.attrs.get("symbol") or df.attrs.get("ticker")
        if symbol is None:
            for candidate in ("symbol", "ticker", "root"):
                if candidate in df.columns:
                    series = df[candidate].dropna()
                    if not series.empty:
                        symbol = series.iloc[0]
                        break

        calendar = df.attrs.get("calendar")
        sessionized = self.sessionizer.attach(
            df, ts_col="ts", symbol=symbol, calendar=calendar
        )
        return sessionized

    def _rebuild_clock_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df["ts"]).dt.tz_convert(self.tz)
        df = df.copy()
        df["clock_time"] = ts.dt.strftime("%H:%M:%S")
        df["clock_time_us"] = ts.dt.strftime("%H:%M:%S.%f")
        return df

    @staticmethod
    def _sanitize_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        out = df.copy()
        for column in columns:
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce", downcast=None)
        return out

    def _dedupe_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        if not out["ts"].is_monotonic_increasing:
            out = out.loc[~out["ts"].duplicated()].sort_values("ts").reset_index(drop=True)
            if not out["ts"].is_monotonic_increasing:
                raise HorizonInputError("ts must be strictly increasing after deduplication")
        return out

    # ------------------------------------------------------------------
    # Return helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _log_return(series: pd.Series, lag: int = 1) -> pd.Series:
        values = pd.Series(series, copy=True).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(values)
        return pd.Series(logged).diff(lag)

    def _clip_returns(self, series: pd.Series) -> pd.Series:
        lower, upper = self.return_clip
        return series.clip(lower=lower, upper=upper)

    def _clean_return(self, series: pd.Series, policy: str) -> pd.Series:
        cleaned = pd.Series(series, copy=True).astype(float)
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = self._clip_returns(cleaned)

        if policy == "drop":
            return cleaned
        if policy == "zero":
            return cleaned.fillna(0.0)
        if policy == "ffill":
            return cleaned.ffill()
        return cleaned

    @staticmethod
    def _format_time_suffix(time_value: Any, policy: str) -> Optional[str]:
        if time_value is None:
            return None

        try:
            parsed = pd.to_datetime(str(time_value)).time()
        except Exception:
            return None

        if policy == "microsecond":
            return parsed.strftime("%H%M%S.%f")
        return parsed.strftime("%H%M%S")

    # ------------------------------------------------------------------
    # Horizon selection
    # ------------------------------------------------------------------
    def pattern_horizon(self, pattern_type: str, default_intraday_minutes: int = 10) -> HorizonSpec:
        if pattern_type in {"weekday_mean", "orderflow_weekly", "orderflow_week_of_month"}:
            return HorizonSpec(name="same_day_oc")
        if pattern_type == "time_predictive_nextday":
            return HorizonSpec(name="next_day_cc")
        if pattern_type == "time_predictive_nextweek":
            return HorizonSpec(name="next_week_cc")
        if pattern_type == "orderflow_peak_pressure":
            return HorizonSpec(name="intraday_delta", delta_minutes=default_intraday_minutes)
        return HorizonSpec(name="same_day_oc")

    # ------------------------------------------------------------------
    # Target calculators
    # ------------------------------------------------------------------
    def _same_day_open_to_close(self, df: pd.DataFrame) -> pd.Series:
        opens = df.groupby("session_id")["open"].first().astype(float)
        closes = df.groupby("session_id")["close"].last().astype(float)
        ratio = closes / opens
        invalid = (opens <= 0) | (closes <= 0)
        if invalid.any():
            ratio = ratio.mask(invalid)
        returns = self._clean_return(np.log(ratio), policy="drop")
        return df["session_id"].map(returns)

    def _next_day_close_to_close(self, df: pd.DataFrame) -> pd.Series:
        closes = df.groupby("session_id")["close"].last().astype(float)
        future = closes.shift(-1)
        ratio = future / closes
        invalid = (closes <= 0) | (future <= 0)
        if invalid.any():
            ratio = ratio.mask(invalid)
        returns = self._clean_return(np.log(ratio), policy="drop")
        return df["session_id"].map(returns)

    def _next_week_close_to_close(self, df: pd.DataFrame, days: int = 5) -> pd.Series:
        closes = df.groupby("session_id")["close"].last().astype(float)
        future = closes.shift(-days)
        ratio = future / closes
        invalid = (closes <= 0) | (future <= 0)
        if invalid.any():
            ratio = ratio.mask(invalid)
        returns = self._clean_return(np.log(ratio), policy="drop")
        return df["session_id"].map(returns)

    def _intraday_delta_minutes(self, df: pd.DataFrame, minutes: int) -> pd.Series:
        if minutes <= 0:
            raise ValueError("minutes must be positive for intraday horizons")

        ts = pd.to_datetime(df["ts"]).dt.tz_convert(self.tz)
        close = pd.Series(df["close"].values, index=ts).astype(float)
        close = close.sort_index()
        delta = pd.Timedelta(minutes=minutes)
        future = close.reindex(close.index + delta)
        future.index = future.index - delta
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(future / close)
        cleaned = self._clean_return(pd.Series(raw, index=close.index), policy="drop")
        aligned = cleaned.reindex(ts)
        return pd.Series(aligned.values, index=df.index)

    def _forward_window_return_minutes(
        self, df: pd.DataFrame, minutes: int
    ) -> pd.Series:
        """Forward log-close return from ``t`` to ``t+Δ``.

        NEVER use iloc-based slices for time horizons; always reindex by ``ts + Δ``.
        """

        if minutes <= 0:
            raise ValueError("predictor minutes must be positive")

        ts = pd.to_datetime(df["ts"]).dt.tz_convert(self.tz)
        close = pd.Series(df["close"].values, index=ts).astype(float)
        close = close.sort_index()
        delta = pd.Timedelta(minutes=minutes)
        future = close.reindex(close.index + delta)
        future.index = future.index - delta
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(future / close)
        series = pd.Series(raw, index=close.index)
        cleaned = self._clean_return(series, policy="drop")
        aligned = cleaned.reindex(ts)
        result = pd.Series(aligned.values, index=df.index)
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        return result

    def _weekly_mean_scalar(self, df: pd.DataFrame, gate_mask: pd.Series) -> float:
        """Same-day open→close mean for sessions flagged by ``gate_mask``."""

        if gate_mask is None or not gate_mask.any():
            return np.nan

        session_ids = pd.Index(df.loc[gate_mask, "session_id"].unique())
        if session_ids.empty:
            return np.nan

        opens = df.groupby("session_id")["open"].first().astype(float)
        closes = df.groupby("session_id")["close"].last().astype(float)
        ratio = closes / opens
        invalid = (opens <= 0) | (closes <= 0)
        if invalid.any():
            ratio = ratio.mask(invalid)

        returns = self._clean_return(np.log(ratio), policy="drop")
        subset = returns.loc[returns.index.isin(session_ids)]
        if subset.empty:
            return np.nan

        mean_value = subset.mean()
        return float(mean_value) if np.isfinite(mean_value) else np.nan

    def _prev_week_return_same_session(self, df: pd.DataFrame) -> pd.Series:
        """Log-close change between a session and the session five days prior."""

        closes = df.groupby("session_id")["close"].last().astype(float)
        past = closes.shift(5)
        ratio = closes / past
        invalid = (closes <= 0) | (past <= 0)
        if invalid.any():
            ratio = ratio.mask(invalid)

        returns = self._clean_return(np.log(ratio), policy="drop")
        return df["session_id"].map(returns)

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    def _prepare_bars_frame(self, bars_with_features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(bars_with_features, pd.DataFrame):
            raise HorizonInputError("bars_with_features must be a pandas DataFrame")

        lower_map: Dict[str, str] = {}
        for column in bars_with_features.columns:
            lower = column.lower()
            if lower not in lower_map:
                lower_map[lower] = column

        required = ["ts", "close", "open"]
        missing = [col for col in required if col not in lower_map]
        if missing:
            raise HorizonInputError(
                "bars_with_features is missing required columns: "
                + ", ".join(sorted(missing))
            )

        rename_map = {lower_map[name]: name for name in required}
        df = bars_with_features.rename(columns=rename_map).copy()
        self._require_columns(df, required)
        df = self._sanitize_ts(df)
        df = self._ensure_session_ids(df)
        if "session_id" not in df.columns:
            raise HorizonInputError("session_id missing after sessionization; cannot proceed")
        if df["session_id"].isna().any():
            raise HorizonInputError("sessionizer produced NaN session identifiers")
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(self.tz)
        df = self._sanitize_numeric(df, ["open", "close"])
        df = self._dedupe_and_sort(df)
        df = self._rebuild_clock_columns(df)

        non_numeric = df[["open", "close"]].isna().any(axis=1)
        if non_numeric.any():
            sample = df.loc[non_numeric, ["ts", "open", "close"]].head(5)
            raise HorizonInputError(
                "Encountered non-numeric open/close values after coercion. "
                f"Sample: {sample.to_dict(orient='records')}"
            )

        return df

    # ------------------------------------------------------------------
    # Predictor extraction
    # ------------------------------------------------------------------
    def _predictor_from_payload(
        self,
        pattern: Mapping[str, Any],
        pattern_type: str,
        mode: str,
    ) -> Tuple[float, int]:
        payload = pattern.get("pattern_payload", {}) or {}
        metadata = pattern.get("metadata", {}) or {}
        bias_value = metadata.get("orderflow_bias") or payload.get("pressure_bias")
        bias_normalized = str(bias_value).lower() if bias_value is not None else ""
        side_hint = 1 if bias_normalized == "buy" else (-1 if bias_normalized == "sell" else 0)

        if mode == "mean":
            if pattern_type in {"weekday_mean", "orderflow_weekly", "orderflow_week_of_month"}:
                value = payload.get("mean")
            elif pattern_type == "orderflow_peak_pressure":
                value = payload.get("intraday_mean", payload.get("mean"))
            elif pattern_type in {"time_predictive_nextday", "time_predictive_nextweek"}:
                value = payload.get("mean")
                if value is None:
                    value = payload.get("correlation")
            else:
                value = payload.get("mean")
            return float(value) if value is not None else np.nan, side_hint

        if mode == "corr":
            value = payload.get("correlation")
            return float(value) if value is not None else np.nan, side_hint

        if mode == "strength":
            value = pattern.get("strength")
            if value is None:
                value = payload.get("t_stat")
            return float(value) if value is not None else np.nan, side_hint

        if mode == "bias":
            return float(side_hint), side_hint

        value = payload.get("mean")
        return float(value) if value is not None else np.nan, side_hint

    def _iter_pattern_returns(
        self,
        df: pd.DataFrame,
        patterns: Any,
        *,
        default_intraday_minutes: int,
        predictor_minutes: int,
        weekly_x_policy: str,
        time_match: str,
        tolerance: Optional[str],
    ) -> Iterable[Tuple[str, str, pd.Series, pd.Series, int, List[str]]]:
        if patterns is None:
            return

        if predictor_minutes <= 0:
            raise HorizonInputError("predictor_minutes must be positive")

        if weekly_x_policy not in {"mean", "prev_week"}:
            raise HorizonInputError("weekly_x_policy must be one of {'mean', 'prev_week'}")

        x_fw = self._forward_window_return_minutes(df, minutes=predictor_minutes)
        prev_week_series: Optional[pd.Series] = None
        horizon_cache: Dict[Tuple[str, Optional[int]], pd.Series] = {}
        time_bearing_types = {"time_predictive_nextday", "time_predictive_nextweek", "orderflow_peak_pressure"}
        weekly_types = {"weekday_mean", "orderflow_weekly", "orderflow_week_of_month"}

        for key, pattern in ScreenerPipeline._items_from_patterns(patterns):
            pattern_type = pattern.get("pattern_type")
            if not pattern_type:
                continue

            gate_col, candidates = self._infer_gate_column_name(
                df, key, pattern, time_match=time_match
            )
            if gate_col is None:
                expected = candidates[0] if candidates else None
                self._log_warn(
                    "Gate column missing",
                    key=key,
                    pattern_type=pattern_type,
                    expected=expected,
                    candidates=candidates,
                )
                continue

            spec = self.pattern_horizon(
                pattern_type, default_intraday_minutes=default_intraday_minutes
            )
            cache_key = (spec.name, spec.delta_minutes if spec.name == "intraday_delta" else None)
            if cache_key not in horizon_cache:
                if spec.name == "same_day_oc":
                    horizon_cache[cache_key] = self._same_day_open_to_close(df)
                elif spec.name == "next_day_cc":
                    horizon_cache[cache_key] = self._next_day_close_to_close(df)
                elif spec.name == "next_week_cc":
                    horizon_cache[cache_key] = self._next_week_close_to_close(df, days=5)
                elif spec.name == "intraday_delta":
                    horizon_cache[cache_key] = self._intraday_delta_minutes(
                        df, minutes=spec.delta_minutes or default_intraday_minutes
                    )
                else:
                    horizon_cache[cache_key] = self._same_day_open_to_close(df)

            returns_y = horizon_cache[cache_key]

            if pattern_type in time_bearing_types:
                returns_x_series = x_fw
            elif pattern_type in weekly_types:
                gate_mask = df[gate_col] == 1
                if weekly_x_policy == "prev_week":
                    if prev_week_series is None:
                        prev_week_series = self._prev_week_return_same_session(df)
                    returns_x_series = prev_week_series
                else:
                    payload_mean, _ = self._predictor_from_payload(pattern, pattern_type, "mean")
                    scalar = payload_mean
                    if not np.isfinite(scalar):
                        scalar = self._weekly_mean_scalar(df, gate_mask)
                    returns_x_series = pd.Series(scalar, index=df.index, dtype=float)
            else:
                payload_value, _ = self._predictor_from_payload(pattern, pattern_type, "mean")
                returns_x_series = pd.Series(payload_value, index=df.index, dtype=float)

            _, side_hint = self._predictor_from_payload(pattern, pattern_type, mode="bias")
            yield gate_col, pattern_type, returns_x_series, returns_y, side_hint, candidates

    @staticmethod
    def _signal_column_name(gate_col: str) -> str:
        if gate_col.endswith("_gate"):
            return f"{gate_col[:-5]}_signal"
        return f"{gate_col}_signal"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_predictor_columns(
        self,
        bars_with_features: pd.DataFrame,
        patterns: Any,
        *,
        default_intraday_minutes: int = 10,
        predictor_minutes: int = 5,
        weekly_x_policy: str = "mean",
    ) -> pd.DataFrame:
        """Return ``bars_with_features`` with per-pattern predictor columns appended."""

        if patterns is None:
            return bars_with_features.copy()

        df = self._prepare_bars_frame(bars_with_features)
        result = bars_with_features.copy()

        for gate_col, _, returns_x_series, _, _, _ in self._iter_pattern_returns(
            df,
            patterns,
            default_intraday_minutes=default_intraday_minutes,
            predictor_minutes=predictor_minutes,
            weekly_x_policy=weekly_x_policy,
            time_match=self.time_match,
            tolerance=self.asof_tolerance,
        ):
            signal_col = self._signal_column_name(gate_col)
            signal = pd.Series(np.nan, index=df.index, dtype=float)
            gate_mask = df[gate_col] == 1
            if gate_mask.any():
                aligned = returns_x_series.reindex(df.index)
                valid_mask = gate_mask & aligned.notna()
                if valid_mask.any():
                    signal.loc[valid_mask] = aligned.loc[valid_mask]
            result[signal_col] = signal

        return result

    def build_xy(
        self,
        bars_with_features: pd.DataFrame,
        patterns: Any,
        *,
        default_intraday_minutes: int = 10,
        predictor_minutes: int = 5,
        weekly_x_policy: str = "mean",
        allowed_months: Optional[Iterable[int]] = None,
        ensure_gates: bool = True,
        nan_policy: Optional[str] = None,
        time_match: Optional[str] = None,
        asof_tolerance: Optional[str] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        """Construct tidy decision rows for screener ``patterns``.

        Parameters
        ----------
        bars_with_features:
            Bar data containing at least ``ts``, ``open``, ``close``, and ``session_id`` columns.
        patterns:
            Pattern payloads produced by :class:`ScreenerPipeline` consumers.
        default_intraday_minutes:
            Horizon minutes used when a pattern requests ``intraday_delta`` without an explicit
            ``delta_minutes`` value.
        predictor_minutes:
            Backward-looking window length (in minutes) for time-bearing predictors.
        weekly_x_policy:
            Strategy for weekly and week-of-month patterns. ``"mean"`` uses the payload ``mean``
            when supplied, otherwise falls back to the historical same-day open→close mean for
            sessions with an active gate. ``"prev_week"`` uses the realised return between the
            session close and the close five sessions prior.
        allowed_months:
            Optional set of calendar months to retain when emitting decision rows.
        ensure_gates:
            When ``True`` the screener pipeline is re-run to materialise missing gates before
            constructing outputs.
        nan_policy:
            Overrides the instance-level NaN handling policy. See ``__init__`` for options.
        time_match:
            Override time gate resolution policy (``"auto"``, ``"second"``, or ``"microsecond"``).
        asof_tolerance:
            Override merge-asof tolerance expressed as a pandas offset string.
        debug:
            Emit detailed diagnostics via the configured logger when ``True``.
        """

        if patterns is None:
            return pd.DataFrame(
                columns=["ts_decision", "gate", "pattern_type", "returns_x", "returns_y", "side_hint"]
            )

        policy = nan_policy or self.nan_policy
        if policy not in {"drop", "zero", "ffill"}:
            raise HorizonInputError("nan_policy must be one of {'drop', 'zero', 'ffill'}")

        effective_time_match = time_match or self.time_match
        if effective_time_match not in {"auto", "second", "microsecond"}:
            raise HorizonInputError(
                "time_match must be one of {'auto', 'second', 'microsecond'}"
            )

        effective_tolerance = asof_tolerance or self.asof_tolerance

        base_df = self._prepare_bars_frame(bars_with_features)

        source_bars = base_df
        if ensure_gates:
            builder = ScreenerPipeline(
                tz=self.tz,
                time_match=self._translate_time_match(effective_time_match),
                log=self.log,
            )
            source_bars = builder.build_features(base_df, patterns)

        df = self._prepare_bars_frame(source_bars)
        if allowed_months is not None:
            allowed_set = {int(month) for month in allowed_months}
            month_mask = df["ts"].dt.month.isin(sorted(allowed_set))
        else:
            month_mask = pd.Series(True, index=df.index)
        if debug:
            sessions = df["session_id"].dropna()
            if not sessions.empty:
                self._log_warn(
                    "Session coverage",
                    first_session=str(sessions.iloc[0]),
                    last_session=str(sessions.iloc[-1]),
                    total_sessions=int(sessions.nunique()),
                )
        rows: List[pd.DataFrame] = []
        pre_row_count = len(df)

        for (
            gate_col,
            pattern_type,
            returns_x_series,
            returns_y,
            side_hint,
            candidates,
        ) in self._iter_pattern_returns(
            df,
            patterns,
            default_intraday_minutes=default_intraday_minutes,
            predictor_minutes=predictor_minutes,
            weekly_x_policy=weekly_x_policy,
            time_match=effective_time_match,
            tolerance=effective_tolerance,
        ):
            gate_series = df.get(gate_col)
            if gate_series is None:
                self._log_warn(
                    "Gate column disappeared between iteration and emission",
                    gate=gate_col,
                    candidates=candidates,
                )
                continue

            returns_x_clean = self._clean_return(returns_x_series, policy)
            returns_y_clean = self._clean_return(returns_y, policy)
            finite_x = pd.Series(np.isfinite(returns_x_clean), index=returns_x_clean.index)
            finite_y = pd.Series(np.isfinite(returns_y_clean), index=returns_y_clean.index)

            active_mask = (
                (gate_series == 1)
                & returns_y_clean.notna()
                & returns_x_clean.notna()
                & finite_x
                & finite_y
                & df["session_id"].notna()
                & month_mask
            )
            if not active_mask.any():
                self._log_warn(
                    "Gate column has no active, finite rows",
                    gate=gate_col,
                    candidates=candidates,
                )
                continue

            subset = df.loc[active_mask, ["ts"]].copy()
            subset["ts_decision"] = subset["ts"]
            subset["gate"] = gate_col
            subset["pattern_type"] = pattern_type
            subset["returns_x"] = returns_x_clean.loc[active_mask].values
            subset["returns_y"] = returns_y_clean.loc[active_mask].values
            subset["side_hint"] = side_hint
            rows.append(subset[["ts_decision", "gate", "pattern_type", "returns_x", "returns_y", "side_hint"]])

            if debug:
                dropped = int(((gate_series == 1) & ~active_mask).sum())
                self._log_warn(
                    "Appended decision rows",
                    gate=gate_col,
                    count=int(active_mask.sum()),
                    dropped_due_to_filters=dropped,
                )

        if not rows:
            if debug:
                self._log_warn(
                    "No rows produced after gate filtering",
                    pre_rows=pre_row_count,
                    policy=policy,
                    time_match=effective_time_match,
                )
            return pd.DataFrame(
                columns=["ts_decision", "gate", "pattern_type", "returns_x", "returns_y", "side_hint"]
            )

        result = pd.concat(rows, axis=0, ignore_index=True)
        if debug:
            self._log_warn(
                "HorizonMapper build complete",
                rows=len(result),
                pre_rows=pre_row_count,
                policy=policy,
                time_match=effective_time_match,
            )
        return result

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------
    def _infer_gate_column_name(
        self,
        df: pd.DataFrame,
        key: str,
        pattern: Mapping[str, Any],
        *,
        time_match: str,
    ) -> Tuple[Optional[str], List[str]]:
        slug_key = ScreenerPipeline._slugify(key)
        candidates: List[str] = []

        if slug_key:
            candidates.append(f"{slug_key}_gate")

        fallback = self._fallback_gate_candidates(
            df, pattern, slug_key=slug_key, time_match=time_match
        )
        candidates.extend([cand for cand in fallback if cand not in candidates])

        available = [cand for cand in candidates if cand in df.columns]
        if not available:
            return None, candidates

        if len(available) > 1:
            chosen = available[0]
            self._log_warn(
                "Multiple gate columns matched; selecting first",
                chosen=chosen,
                options=available,
            )
            return chosen, candidates

        return available[0], candidates

    def _fallback_gate_candidates(
        self,
        df: pd.DataFrame,
        pattern: Mapping[str, Any],
        *,
        slug_key: Optional[str],
        time_match: str,
    ) -> List[str]:
        pattern_type = pattern.get("pattern_type")
        payload = pattern.get("pattern_payload", {}) or {}
        metadata = pattern.get("metadata", {}) or {}

        candidates: List[str] = []

        if pattern_type == "weekday_mean":
            weekday = payload.get("day") or payload.get("weekday")
            weekday_norm = ScreenerPipeline._normalize_weekday(weekday)
            if weekday_norm:
                candidates.append(f"weekday_mean_{weekday_norm}_gate")

        elif pattern_type == "time_predictive_nextday":
            time_value = payload.get("time")
            effective = self._resolve_time_match(df, time_match, time_value)
            suffix_second = self._format_time_suffix(time_value, "second")
            suffix_micro = self._format_time_suffix(time_value, "microsecond")
            order = [suffix_micro, suffix_second] if effective == "microsecond" else [suffix_second, suffix_micro]
            for suffix in order:
                if suffix:
                    candidates.append(f"time_nextday_{suffix}_gate")

        elif pattern_type == "time_predictive_nextweek":
            time_value = payload.get("time")
            effective = self._resolve_time_match(df, time_match, time_value)
            suffix_second = self._format_time_suffix(time_value, "second")
            suffix_micro = self._format_time_suffix(time_value, "microsecond")
            order = [suffix_micro, suffix_second] if effective == "microsecond" else [suffix_second, suffix_micro]
            for suffix in order:
                if suffix:
                    candidates.append(f"time_nextweek_{suffix}_gate")

        elif pattern_type == "orderflow_weekly":
            weekday_norm = ScreenerPipeline._normalize_weekday(payload.get("weekday"))
            metric = payload.get("metric", "net_pressure")
            bias = ScreenerPipeline._format_bias(
                metadata.get("orderflow_bias") or payload.get("pressure_bias")
            )
            if weekday_norm:
                candidates.append(f"oflow_weekly_{weekday_norm}_{metric}_{bias}_gate")

        elif pattern_type == "orderflow_week_of_month":
            weekday_norm = ScreenerPipeline._normalize_weekday(payload.get("weekday"))
            metric = payload.get("metric", "net_pressure")
            bias = ScreenerPipeline._format_bias(
                metadata.get("orderflow_bias") or payload.get("pressure_bias")
            )
            try:
                wom = int(payload.get("week_of_month"))
            except (TypeError, ValueError):
                wom = None
            if weekday_norm and wom is not None:
                candidates.append(f"oflow_wom_{weekday_norm}_w{wom}_{metric}_{bias}_gate")

        elif pattern_type == "orderflow_peak_pressure":
            weekday_norm = ScreenerPipeline._normalize_weekday(payload.get("weekday"))
            metric = payload.get("metric", "net_pressure")
            bias = ScreenerPipeline._format_bias(
                metadata.get("orderflow_bias") or payload.get("pressure_bias")
            )
            time_value = payload.get("clock_time")
            effective = self._resolve_time_match(df, time_match, time_value)
            suffix_second = self._format_time_suffix(time_value, "second")
            suffix_micro = self._format_time_suffix(time_value, "microsecond")
            order = [suffix_micro, suffix_second] if effective == "microsecond" else [suffix_second, suffix_micro]
            for suffix in order:
                if weekday_norm and suffix:
                    candidates.append(f"oflow_peak_{weekday_norm}_{suffix}_{metric}_{bias}_gate")

        explicit = payload.get("gate") or metadata.get("gate")
        if isinstance(explicit, str) and explicit:
            normalized = explicit if explicit.endswith("_gate") else f"{explicit}_gate"
            candidates.append(normalized)

        return candidates

