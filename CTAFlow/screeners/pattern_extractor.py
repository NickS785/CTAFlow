"""Utilities for organising and analysing screener pattern output.

The :class:`PatternExtractor` class restructures raw screener payloads and
exposes convenience helpers for downstream analysis.  In addition to
retrieving pattern series, it now supports composition and deterministic
ranking::

    >>> combined = extractor_a + extractor_b  # merge two extractions
    >>> top_setups = combined.top_patterns("CL", n=3)
    >>> global_leaders = combined.top_patterns_global(n=10)

These helpers make it straightforward to surface the strongest statistical
relationships discovered across multiple screener runs.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass, field
from datetime import time
from types import SimpleNamespace
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import pandas as pd

try:  # Optional dependency during lightweight usage
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - SciPy not always available
    scipy_stats = None  # type: ignore[assignment]

try:  # Optional dependency during lightweight usage
    from .historical_screener import HistoricalScreener, ScreenParams
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    HistoricalScreener = Any  # type: ignore[assignment]
    ScreenParams = Any  # type: ignore[assignment]
from .orderflow_scan import OrderflowParams
from ..utils import aio
from ..utils.seasonal import aggregate_window, log_returns, monthly_returns, tod_mask
from ..utils.session import filter_session_bars


ScreenParamLike = Union[ScreenParams, OrderflowParams, SimpleNamespace]

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from ..data.data_client import ResultsClient


_LOG = logging.getLogger(__name__)


def validate_filtered_months(
    months: Optional[Iterable[Any]],
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "pattern_extractor",
) -> Optional[Set[int]]:
    """Normalise ``months`` into a canonical ``set[int]``.

    Non-integer entries are discarded with a warning. Values outside the range
    ``[1, 12]`` are ignored. When all entries are invalid ``None`` is returned.

    Parameters
    ----------
    months:
        Iterable of month designators. ``None`` and empty iterables yield
        ``None``. Strings of digits are coerced into integers.
    logger:
        Optional logger used for warning emission.
    context:
        Free-form label describing the validation caller for diagnostics.
    """

    if months in (None, "", (), [], {}):  # type: ignore[comparison-overlap]
        return None

    if isinstance(months, str):
        tokens = re.split(r"[^0-9]+", months)
        candidates: Iterable[Any] = [token for token in tokens if token]
    else:
        candidates = months

    valid: Set[int] = set()
    rejected: List[Any] = []

    for value in candidates:
        try:
            month = int(value)
        except (TypeError, ValueError):
            rejected.append(value)
            continue

        if 1 <= month <= 12:
            valid.add(month)
        else:
            rejected.append(value)

    logger = logger or _LOG

    if rejected and logger is not None and hasattr(logger, "warning"):
        logger.warning(
            "Discarding invalid filtered_months",
            extra={"context": context, "invalid": rejected},
        )

    if not valid:
        return None

    return valid


@dataclass
class PatternSummary:
    """Container describing a statistically significant pattern."""

    SESSION_PARAM_KEYS: ClassVar[Tuple[str, ...]] = (
        "session_start",
        "session_end",
        "session_tz",
        "window_anchor",
        "window_minutes",
        "period_length_min",
        "sess_start",
        "sess_end",
        "sess_start_hrs",
        "sess_start_minutes",
        "sess_end_hrs",
        "sess_end_minutes",
    )

    key: str
    symbol: str
    source_screen: str
    screen_params: Optional[ScreenParamLike]
    pattern_type: str
    description: str
    strength: Optional[float]
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation suitable for JSON/export."""

        params_dict: Optional[Dict[str, Any]]
        if self.screen_params is None:
            params_dict = None
        else:
            if hasattr(self.screen_params, "__dataclass_fields__"):
                field_names = self.screen_params.__dataclass_fields__.keys()  # type: ignore[attr-defined]
                params_dict = {
                    field_name: getattr(self.screen_params, field_name)
                    for field_name in field_names
                }
            else:
                params_dict = {
                    key: value
                    for key, value in vars(self.screen_params).items()
                    if not key.startswith("_")
                }

        return {
            "key": self.key,
            "symbol": self.symbol,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "strength": self.strength,
            "source_screen": self.source_screen,
            "screen_parameters": params_dict,
            "metadata": self.metadata,
            "pattern_payload": self.payload,
        }

    def get_session_params(self) -> Dict[str, Any]:
        """Return canonical session parameters merged from metadata/payload."""

        params: Dict[str, Any] = {}
        sources: Tuple[Mapping[str, Any], Mapping[str, Any]] = (
            self.metadata,
            self.payload,
        )
        for key in self.SESSION_PARAM_KEYS:
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                if key in source and source[key] is not None:
                    params.setdefault(key, source[key])
                    break
        return params

    @property
    def session_params(self) -> Dict[str, Any]:
        """Shortcut exposing :meth:`get_session_params`."""

        return self.get_session_params()


class PatternExtractor:
    """Restructure screener output and expose analysis helpers.

    Examples
    --------
    Combine extractions from multiple screener runs::

        >>> combined = extractor_a.concat(extractor_b)
        >>> list(combined.patterns)
        ['CL', 'GC', ...]

    Surface the strongest setups per ticker::

        >>> combined.rank_patterns(top=5)
        {'CL': [('scan|pattern', summary, 0.78), ...], ...}

    Notes
    -----
    Screen-level metadata is exposed via :attr:`metadata`. When present,
    ``metadata['filtered_months']`` holds the canonical set of months (1–12)
    during which gates may activate. The helper :meth:`get_filtered_months`
    returns the normalised ``set[int]`` representation and is automatically
    populated from historical results when not explicitly provided.
    """

    SUMMARY_COLUMNS: Tuple[str, ...] = (
        "pattern_type",
        "strength",
        "correlation",
        "time",
        "weekday",
        "week_of_month",
        "q_value",
        "p_value",
        "t_stat",
        "f_stat",
        "n",
        "source_screen",
        "scan_type",
        "scan_name",
        "description",
        "created_at",
        "period_length",
        "month_filter",
        "months_active",
        "months_mask_12",
        "months_names",
        "momentum_type",
        "bias",
        "session_key",
        "session_index",
        "session_start",
        "session_end",
        "session_tz",
        "window_anchor",
        "window_minutes",
        "target_times_hhmm",
        "period_length_min",
        "regime_filter",
        "momentum_params",
        "st_momentum_days",
        "best_weekday",
        "best_mean",
        "worst_weekday",
        "worst_mean",
        "strongest_days",
        "volatility_bias_ratio",
        "high_vol_days_count",
        "n_high_vol_days",
        "n_low_vol_days",
    )

    MOMENTUM_TYPE_LABELS: Mapping[str, str] = {
        "opening_momentum": "Opening momentum",
        "closing_momentum": "Closing momentum",
        "full_session": "Full session returns",
        "st_momentum": "Short-term momentum",
    }

    def __init__(
        self,
        screener: HistoricalScreener,
        results: Mapping[str, Mapping[str, Dict[str, Any]]],
        screen_params: Optional[Sequence[ScreenParamLike]] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._screener = screener
        self._results = results
        self._screen_params: Dict[str, ScreenParamLike] = (
            {params.name: params for params in screen_params}
            if screen_params is not None
            else {}
        )

        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._filtered_months: Optional[Set[int]] = None
        self._pattern_index: Dict[str, Dict[str, PatternSummary]] = {}
        self._build_index()
        self._initialise_month_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _copy_like(self) -> "PatternExtractor":
        """Create a shallow copy preserving loaded summaries without rebuilding."""

        clone = object.__new__(self.__class__)
        clone._screener = self._screener
        clone._results = dict(self._results)
        clone._screen_params = dict(self._screen_params)
        clone.metadata = dict(self.metadata)
        existing_months = getattr(self, "_filtered_months", None)
        clone._filtered_months = set(existing_months) if existing_months else None
        clone._pattern_index = {
            symbol: dict(entries)
            for symbol, entries in self._pattern_index.items()
        }
        return clone

    # ------------------------------------------------------------------
    # Month filter helpers
    # ------------------------------------------------------------------
    def _store_filtered_months(self, months: Optional[Iterable[int]]) -> None:
        if months is None:
            self._filtered_months = None
            self.metadata.pop("filtered_months", None)
            return

        canonical: Set[int] = set()
        for value in months:
            try:
                month = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= month <= 12:
                canonical.add(month)

        if canonical:
            self._filtered_months = set(canonical)
            self.metadata["filtered_months"] = sorted(self._filtered_months)
        else:
            self._filtered_months = None
            self.metadata.pop("filtered_months", None)

    def _initialise_month_metadata(self) -> None:
        metadata_months = None
        if "filtered_months" in self.metadata:
            metadata_months = validate_filtered_months(
                self.metadata.get("filtered_months"),
                logger=_LOG,
                context="metadata.filtered_months",
            )

        extracted = None
        if metadata_months is None:
            extracted = self._extract_filtered_months_from_results(self._results)
            metadata_months = extracted

        self._store_filtered_months(metadata_months)
        if metadata_months is not None:
            self.metadata.setdefault("month_merge_policy", "source")
        elif "month_merge_policy" in self.metadata:
            # Remove stale policy when no canonical months remain
            self.metadata.pop("month_merge_policy", None)

    def _extract_filtered_months_from_results(
        self, results: Mapping[str, Mapping[str, Dict[str, Any]]]
    ) -> Optional[Set[int]]:
        month_sets: List[Set[int]] = []

        for screen_name, ticker_results in results.items():
            if not isinstance(ticker_results, Mapping):
                continue
            for symbol, ticker_result in ticker_results.items():
                if not isinstance(ticker_result, Mapping):
                    continue

                raw_months = ticker_result.get("filtered_months")
                if isinstance(raw_months, str) and raw_months.lower() == "all":
                    continue

                parsed = validate_filtered_months(
                    raw_months,
                    logger=_LOG,
                    context=f"results.{screen_name}.{symbol}",
                )
                if parsed:
                    month_sets.append(parsed)

        if not month_sets:
            return None

        first = month_sets[0]
        if all(candidate == first for candidate in month_sets[1:]):
            return set(first)

        union: Set[int] = set().union(*month_sets)
        if union:
            _LOG.warning(
                "[PatternExtractor] Inconsistent filtered_months across ticker results; using union",
                extra={
                    "context": "results",
                    "values": [sorted(candidate) for candidate in month_sets],
                },
            )
        return union if union else None

    @staticmethod
    def _merge_month_sets(
        left: Optional[Set[int]],
        right: Optional[Set[int]],
        policy: str,
    ) -> Optional[Set[int]]:
        if policy == "left":
            return set(left) if left else None
        if policy == "right":
            return set(right) if right else None
        if policy == "union":
            combined: Set[int] = set()
            if left:
                combined.update(left)
            if right:
                combined.update(right)
            return combined or None

        # Default to intersection
        if left is None and right is None:
            return None
        if left is None:
            return set(right) if right else None
        if right is None:
            return set(left) if left else None

        intersection = set(left).intersection(right)
        return intersection or None

    def get_filtered_months(self) -> Optional[Set[int]]:
        """Return the canonical set of months allowed by the extractor."""

        current = getattr(self, "_filtered_months", None)
        if current is None:
            metadata_months = validate_filtered_months(
                self.metadata.get("filtered_months"), logger=_LOG, context="PatternExtractor"
            )
            if metadata_months:
                current = set(metadata_months)
                self._filtered_months = current

        return set(current) if current else None

    @property
    def patterns(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return patterns grouped by symbol and keyed by pattern identifier."""

        return {
            symbol: {key: summary.as_dict() for key, summary in entries.items()}
            for symbol, entries in self._pattern_index.items()
        }

    # ------------------------------------------------------------------
    # Concatenation helpers
    # ------------------------------------------------------------------
    def concat(
        self,
        other: "PatternExtractor",
        *,
        conflict: str = "prefer_strong",
        inplace: bool = False,
        month_merge_policy: str = "intersect",
    ) -> "PatternExtractor":
        """Merge the pattern inventory from ``other`` into this extractor.

        Parameters
        ----------
        other : PatternExtractor
            The extractor providing additional pattern summaries.
        conflict : {"prefer_strong", "prefer_left", "prefer_right", "keep_both"}, optional
            Strategy for handling collisions on ``(ticker, pattern_key)``.
            ``"prefer_strong"`` (default) keeps whichever summary has the higher
            :meth:`significance_score` with deterministic fallbacks.  The
            ``"keep_both"`` strategy preserves both entries by suffixing the new
            key with ``"#2"``, ``"#3"`` and so on.
        inplace : bool, optional
            If ``True`` the merge mutates ``self`` and returns it.  Otherwise a
            shallow copy is returned, leaving the originals untouched.
        month_merge_policy : {"intersect", "union", "left", "right"}, optional
            How ``filtered_months`` metadata is reconciled when both extractors
            define a month set. ``"intersect"`` (default) keeps only months that
            appear in both operands. ``"union"`` combines all months while
            ``"left"``/``"right"`` preserve the respective operand.

        Returns
        -------
        PatternExtractor
            Extractor containing the merged pattern index.
        """

        if not isinstance(other, PatternExtractor):
            raise TypeError("PatternExtractor.concat expects another PatternExtractor")

        strategy = conflict.lower()
        valid_strategies = {
            "prefer_strong",
            "prefer_left",
            "prefer_right",
            "keep_both",
        }
        if strategy not in valid_strategies:
            raise ValueError(f"Unsupported conflict strategy '{conflict}'")

        month_policy = month_merge_policy.lower()
        valid_month_policies = {"union", "intersect", "left", "right"}
        if month_policy not in valid_month_policies:
            raise ValueError(
                "month_merge_policy must be one of {'union', 'intersect', 'left', 'right'}"
            )

        target = self if inplace else self._copy_like()

        for symbol, summaries in other._pattern_index.items():
            target_entries = target._pattern_index.setdefault(symbol, {})
            for key, summary in summaries.items():
                if key not in target_entries:
                    target_entries[key] = summary
                    continue

                if strategy == "prefer_left":
                    continue
                if strategy == "prefer_right":
                    target_entries[key] = summary
                    continue
                if strategy == "keep_both":
                    suffix = 2
                    new_key = f"{key}#{suffix}"
                    while new_key in target_entries:
                        suffix += 1
                        new_key = f"{key}#{suffix}"
                    target_entries[new_key] = summary
                    continue

                chosen = self._choose_stronger_summary(target_entries[key], summary)
                target_entries[key] = chosen

        merged_results = dict(target._results)
        merged_results.update(other._results)
        target._results = merged_results

        merged_params = dict(target._screen_params)
        merged_params.update(other._screen_params)
        target._screen_params = merged_params

        left_months = target.get_filtered_months()
        right_months = other.get_filtered_months()
        merged_months = self._merge_month_sets(left_months, right_months, month_policy)
        target._store_filtered_months(merged_months)
        target.metadata["month_merge_policy"] = month_policy

        return target

    def __add__(self, other: "PatternExtractor") -> "PatternExtractor":
        """Return a new extractor containing patterns from both operands."""

        return self.concat(other, conflict="prefer_strong", inplace=False)

    def __iadd__(self, other: "PatternExtractor") -> "PatternExtractor":
        """Merge patterns from ``other`` into ``self`` in-place."""

        return self.concat(other, conflict="prefer_strong", inplace=True)

    @classmethod
    def concat_many(
        cls,
        extractors: Iterable["PatternExtractor"],
        *,
        conflict: str = "prefer_strong",
        month_merge_policy: str = "intersect",
    ) -> "PatternExtractor":
        """Fold an iterable of extractors into a single combined instance."""

        iterator = iter(extractors)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("concat_many requires at least one extractor") from exc

        result = first._copy_like()
        for extractor in iterator:
            result = result.concat(
                extractor,
                conflict=conflict,
                inplace=True,
                month_merge_policy=month_merge_policy,
            )
        return result

    def filter_patterns(
        self,
        symbol: str,
        *,
        pattern_types: Optional[Iterable[str]] = None,
        screen_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return pattern payloads filtered by type and/or originating screen."""

        entries = self._pattern_index.get(symbol)
        if not entries:
            return {}

        def _normalise(values: Optional[Iterable[str]]) -> Optional[Set[str]]:
            if values is None:
                return None
            if isinstance(values, str):
                values = [values]
            return {str(value) for value in values}

        type_filter = _normalise(pattern_types)
        screen_filter = _normalise(screen_names)

        filtered: Dict[str, Dict[str, Any]] = {}
        for key, summary in entries.items():
            if type_filter is not None and summary.pattern_type not in type_filter:
                continue
            if screen_filter is not None and summary.source_screen not in screen_filter:
                continue
            record = summary.as_dict()
            payload = record.get("pattern_payload") or {}
            for promoted in ("p_value", "t_stat", "f_stat", "n"):
                if promoted not in record and promoted in payload:
                    record[promoted] = payload[promoted]
            filtered[key] = record

        return filtered

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(coerced):
            return None
        return coerced

    @classmethod
    def _extract_payload_first(cls, summary: PatternSummary, keys: Sequence[str]) -> Optional[float]:
        payload = summary.payload or {}
        for key in keys:
            if key in payload:
                value = cls._coerce_optional_float(payload.get(key))
                if value is not None:
                    return value
        return None

    @classmethod
    def _extract_support(cls, summary: PatternSummary) -> float:
        value = cls._extract_payload_first(
            summary,
            ("support", "n", "n_obs", "seasonality_n"),
        )
        return float(value) if value is not None and value > 0 else 0.0

    @classmethod
    def _extract_t_stat(cls, summary: PatternSummary) -> float:
        value = cls._extract_payload_first(summary, ("t_stat", "seasonality_t_stat"))
        return float(value) if value is not None else 0.0

    @classmethod
    def _extract_correlation(cls, summary: PatternSummary) -> float:
        value = cls._extract_payload_first(summary, ("correlation", "seasonality_correlation"))
        return float(value) if value is not None else 0.0

    @classmethod
    def _strength_raw(cls, summary: PatternSummary) -> float:
        value = cls._coerce_optional_float(summary.strength)
        return float(value) if value is not None else 0.0

    @staticmethod
    def significance_score(summary: PatternSummary) -> float:
        """Return a deterministic significance score for ``summary``.

        The score combines multiple stability indicators with fixed weights::

            score = 0.30*n + 0.25*t + 0.20*c + 0.15*p + 0.10*h + 0.10*norm_strength

        where ``n`` encodes support, ``t`` the absolute *t*-statistic,
        ``c`` correlation, ``p`` tail probability, and ``h`` hit-rate style
        measures.  Missing values contribute ``0`` ensuring robustness.

        Returns
        -------
        float
            Weighted composite score in ``[0, 1]``.

        Notes
        -----
        TODO: expose scoring weights via ``PatternExtractor`` configuration.
        """

        support = PatternExtractor._extract_support(summary)
        n_component = math.log10(1.0 + support) / 4.0 if support > 0 else 0.0

        t_stat = abs(PatternExtractor._extract_t_stat(summary))
        t_component = min(t_stat, 10.0) / 10.0

        correlation = abs(PatternExtractor._extract_correlation(summary))
        c_component = min(max(correlation, 0.0), 1.0)

        hit_rate = PatternExtractor._extract_payload_first(summary, ("hit_rate",))
        h_component = min(max(hit_rate or 0.0, 0.0), 1.0)

        p_value = PatternExtractor._extract_payload_first(
            summary,
            ("p_value", "q_value", "seasonality_q_value"),
        )
        p_component = 1.0 - min(max(p_value or 0.0, 0.0), 1.0)

        raw_strength = PatternExtractor._strength_raw(summary)
        strength = max(raw_strength, 0.0)
        norm_strength = strength / (1.0 + strength)

        score = (
            0.30 * n_component
            + 0.25 * t_component
            + 0.20 * c_component
            + 0.15 * p_component
            + 0.10 * h_component
            + 0.10 * norm_strength
        )
        return float(score)

    def _choose_stronger_summary(
        self,
        left: PatternSummary,
        right: PatternSummary,
    ) -> PatternSummary:
        left_score = self.significance_score(left)
        right_score = self.significance_score(right)
        if right_score > left_score:
            return right
        if left_score > right_score:
            return left

        left_strength = PatternExtractor._strength_raw(left)
        right_strength = PatternExtractor._strength_raw(right)
        if right_strength > left_strength:
            return right
        if left_strength > right_strength:
            return left

        left_t = abs(PatternExtractor._extract_t_stat(left))
        right_t = abs(PatternExtractor._extract_t_stat(right))
        if right_t > left_t:
            return right
        if left_t > right_t:
            return left

        left_corr = abs(PatternExtractor._extract_correlation(left))
        right_corr = abs(PatternExtractor._extract_correlation(right))
        if right_corr > left_corr:
            return right
        if left_corr > right_corr:
            return left

        return left

    def rank_patterns(
        self,
        *,
        by: str = "score",
        ascending: bool = False,
        per_ticker: bool = True,
        top: Optional[int] = None,
        session_filters: Optional[Mapping[str, Any]] = None,
    ) -> Union[
        Dict[str, List[Tuple[str, PatternSummary, float]]],
        List[Tuple[str, str, PatternSummary, float]],
    ]:
        """Return ranked pattern summaries by ticker or globally.

        Parameters
        ----------
        by : {"score", "t_stat", "support", "strength"}, optional
            Metric used for sorting.  ``"score"`` (default) uses
            :meth:`significance_score`.
        ascending : bool, optional
            Sort direction.  Defaults to descending (strongest first).
        per_ticker : bool, optional
            When ``True`` the result is grouped by ticker.  Otherwise a global
            ranking is returned.
        top : int, optional
            Limit the number of entries per ticker (or overall when
            ``per_ticker=False``).
        session_filters : Mapping, optional
            When provided, restrict the ranking to summaries whose session
            metadata matches the supplied ``sess_*`` parameters (e.g.
            ``{"sess_start_hrs": 1}``). ``None`` or empty mappings disable the
            filter.

        Returns
        -------
        dict or list
            Structured ranking containing ``(pattern_key, summary, score)``
            tuples.
        """

        metric = by.lower()
        allowed = {"score", "t_stat", "support", "strength"}
        if metric not in allowed:
            raise ValueError(f"Unsupported ranking metric '{by}'")

        def _metric_value(summary: PatternSummary) -> float:
            if metric == "score":
                return self.significance_score(summary)
            if metric == "t_stat":
                return abs(PatternExtractor._extract_t_stat(summary))
            if metric == "support":
                return PatternExtractor._extract_support(summary)
            return PatternExtractor._strength_raw(summary)

        active_filters: Dict[str, Any] = {}
        if session_filters:
            active_filters = {
                key: value
                for key, value in session_filters.items()
                if value is not None
            }

        def _matches_session_filters(summary: PatternSummary) -> bool:
            if not active_filters:
                return True
            params = summary.get_session_params()
            for key, expected in active_filters.items():
                actual = params.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        return False
                else:
                    if actual != expected:
                        return False
            return True

        def _sort_key(
            ticker: str,
            pattern_key: str,
            metric_value: float,
            support_value: float,
            t_stat_value: float,
        ) -> Tuple[float, float, float, str, str]:
            primary = metric_value if ascending else -metric_value
            support_component = support_value if ascending else -support_value
            t_component = abs(t_stat_value) if ascending else -abs(t_stat_value)
            return (primary, support_component, t_component, ticker, pattern_key)

        ranked: Dict[str, List[Tuple[str, PatternSummary, float]]] = {}
        global_entries: List[Tuple[Tuple[float, float, float, str, str], str, str, PatternSummary, float]] = []

        for ticker, patterns in self._pattern_index.items():
            ticker_entries: List[
                Tuple[Tuple[float, float, float, str, str], str, PatternSummary, float]
            ] = []
            for pattern_key, summary in patterns.items():
                if not _matches_session_filters(summary):
                    continue
                score = self.significance_score(summary)
                metric_value = _metric_value(summary)
                support_value = PatternExtractor._extract_support(summary)
                t_stat_value = PatternExtractor._extract_t_stat(summary)
                sort_key = _sort_key(
                    ticker,
                    pattern_key,
                    metric_value,
                    support_value,
                    t_stat_value,
                )
                entry = (sort_key, pattern_key, summary, score)
                ticker_entries.append(entry)
                global_entries.append((sort_key, ticker, pattern_key, summary, score))

            ticker_entries.sort(key=lambda item: item[0])
            if top is not None:
                ticker_entries = ticker_entries[: top if top >= 0 else 0]
            ranked[ticker] = [(key, summary, score) for _, key, summary, score in ticker_entries]

        if per_ticker:
            return ranked

        global_entries.sort(key=lambda item: item[0])
        if top is not None:
            global_entries = global_entries[: top if top >= 0 else 0]
        return [
            (ticker, pattern_key, summary, score)
            for _, ticker, pattern_key, summary, score in global_entries
        ]

    def top_patterns(self, ticker: str, n: int = 10) -> List[Tuple[str, PatternSummary, float]]:
        """Return the ``n`` strongest patterns for ``ticker``."""

        ranked = self.rank_patterns(per_ticker=True, top=n)
        return ranked.get(ticker, [])

    def top_patterns_global(self, n: int = 50) -> List[Tuple[str, str, PatternSummary, float]]:
        """Return the top ``n`` patterns across all tickers."""

        return self.rank_patterns(per_ticker=False, top=n)

    def get_pattern_keys(self, symbol: str) -> List[str]:
        """Return all pattern identifiers for ``symbol``."""

        return list(self._pattern_index.get(symbol, {}).keys())

    def get_pattern_summary(self, symbol: str, pattern_key: str) -> PatternSummary:
        """Retrieve the :class:`PatternSummary` for the requested pattern."""

        try:
            return self._pattern_index[symbol][pattern_key]
        except KeyError as exc:  # pragma: no cover - user error surface
            raise KeyError(f"Unknown pattern '{pattern_key}' for symbol '{symbol}'") from exc

    def get_pattern_series(self, symbol: str, pattern_key: str) -> pd.Series:
        """Return the time-series associated with a pattern."""

        summary = self.get_pattern_summary(symbol, pattern_key)
        screen_type = summary.metadata.get("screen_type")

        if summary.pattern_type in {"weekday_mean", "weekday_returns"}:
            return self._extract_weekday_series(symbol, summary)

        if summary.pattern_type in {"time_predictive_nextday", "time_predictive_nextweek"}:
            return self._extract_time_of_day_series(symbol, summary)

        if summary.pattern_type == "weekend_hedging":
            return self._extract_weekend_hedging_series(symbol, summary)

        if summary.pattern_type in {
            "abnormal_month",
            "high_sharpe_month",
            "annual_seasonality",
            "lag_month_momentum",
            "prewindow_predictor",
        }:
            return self._extract_monthly_series(symbol, summary)

        if summary.metadata.get("screen_type") == "orderflow" or summary.pattern_type.startswith("orderflow_"):
            return self._extract_orderflow_series(symbol, summary)

        raise NotImplementedError(
            f"Pattern type '{summary.pattern_type}' from screen '{screen_type}' is not supported yet"
        )

    def get_pattern_dates(self, symbol: str, pattern_key: str) -> List[pd.Timestamp]:
        """Return the unique dates where the pattern occurs."""

        series = self.get_pattern_series(symbol, pattern_key)
        return list(pd.Index(series.index).unique())

    def rebase_price_series(
        self,
        symbol: str,
        pattern_key: str,
        price_col: str = "Close",
    ) -> pd.DataFrame:
        """Construct a rebased price series using pattern occurrences."""

        pattern_series = self.get_pattern_series(symbol, pattern_key)
        if pattern_series.empty:
            return pd.DataFrame(columns=["pattern_return", "close", "rebased"])

        price_series = self._get_price_series(symbol, price_col)
        daily_close = price_series.groupby(price_series.index.normalize()).last()
        aligned_close = daily_close.reindex(pattern_series.index, method="ffill")

        rebased = (pattern_series.fillna(0).add(1.0)).cumprod()

        return pd.DataFrame(
            {
                "pattern_return": pattern_series,
                "close": aligned_close,
                "rebased": rebased,
            }
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_key_component(value: str, *, uppercase: bool = False) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"[\s/]+", "_", text)
        text = re.sub(r"[^0-9A-Za-z_]+", "", text)
        return text.upper() if uppercase else text.lower()

    @classmethod
    def _build_results_key(cls, scan_type: str, ticker: str, scan_name: str) -> str:
        scan_type_part = cls._sanitize_key_component(scan_type)
        ticker_part = cls._sanitize_key_component(ticker, uppercase=True)
        scan_name_part = cls._sanitize_key_component(scan_name)
        if not all([scan_type_part, ticker_part, scan_name_part]):
            raise ValueError("scan_type, ticker, and scan_name must all be non-empty")
        return f"results/{scan_type_part}/{ticker_part}/{scan_name_part}"

    @staticmethod
    def _coerce_summary_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        string_columns = [
            "pattern_type",
            "time",
            "weekday",
            "source_screen",
            "scan_type",
            "scan_name",
            "description",
            "created_at",
        ]
        float_columns = ["strength", "correlation", "q_value", "p_value", "t_stat", "f_stat"]
        int_like_columns = ["week_of_month", "n"]

        for column in string_columns:
            if column in df.columns:
                df[column] = df[column].astype("string")

        for column in float_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in int_like_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    @staticmethod
    def _infer_scan_type_from_results(by_ticker: Mapping[str, Any]) -> str:
        orderflow_keys = {
            "df_weekly",
            "df_wom_weekday",
            "df_weekly_peak_pressure",
            "df_intraday_pressure",
            "df_buckets",
        }
        for result in by_ticker.values():
            if isinstance(result, Mapping) and orderflow_keys.intersection(result.keys()):
                return "orderflow"
        for result in by_ticker.values():
            if isinstance(result, Mapping) and "strongest_patterns" in result:
                return "seasonality"
        return "seasonality"

    def _summarize_patterns_for_ticker(
        self,
        *,
        symbol: str,
        scan_name: str,
        scan_type: str,
        entries: Mapping[str, PatternSummary],
        created_at: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        metadata = dict(metadata or {})
        for summary in entries.values():
            if summary.source_screen != scan_name:
                continue

            payload = summary.payload or {}
            time_value = payload.get("time") or payload.get("clock_time")
            time_str = None
            if time_value not in (None, ""):
                time_str = str(time_value)

            row = {
                "pattern_type": summary.pattern_type,
                "strength": summary.strength,
                "correlation": payload.get("correlation"),
                "time": time_str,
                "weekday": payload.get("day") or payload.get("weekday"),
                "week_of_month": payload.get("week_of_month"),
                "q_value": payload.get("q_value") or payload.get("seasonality_q_value"),
                "p_value": payload.get("p_value") or payload.get("seasonality_p_value"),
                "t_stat": payload.get("t_stat") or payload.get("seasonality_t_stat"),
                "f_stat": payload.get("f_stat") or payload.get("seasonality_f_stat"),
                "n": payload.get("n")
                or payload.get("seasonality_n")
                or payload.get("support"),
                "source_screen": summary.source_screen,
                "scan_type": scan_type,
                "scan_name": scan_name,
                "description": summary.description,
                "created_at": created_at,
                "period_length": metadata.get("period_length"),
                "month_filter": metadata.get("month_filter"),
            }
            rows.append(row)

        return rows

    def persist_to_results(
        self,
        results_client: "ResultsClient",
        *,
        created_at: Optional[str] = None,
    ) -> Dict[str, int]:
        """Persist pattern summaries to the results HDF store.

        Parameters
        ----------
        results_client : ResultsClient
            Client used to write summary DataFrames into ``RESULTS_HDF_PATH``.
        created_at : str, optional
            Optional ISO 8601 timestamp string to stamp each summary row.

        Returns
        -------
        Dict[str, int]
            Mapping of ``"results/{scan_type}/{ticker}/{scan_name}"`` keys to the
            number of rows written for that combination.

        Notes
        -----
        Each key is overwritten on every call (``replace=True``).  Downstream code
        can later query the stored summaries via::

            with pd.HDFStore(RESULTS_HDF_PATH, "r") as store:
                df = store.select("results/orderflow/RB/us_winter")
        """

        timestamp = created_at or pd.Timestamp.utcnow().isoformat()
        counts: Dict[str, int] = {}

        for scan_name, by_ticker in self._results.items():
            if not isinstance(by_ticker, Mapping):
                continue

            params = self._screen_params.get(scan_name)
            scan_type = getattr(params, "screen_type", None)
            if not scan_type:
                scan_type = self._infer_scan_type_from_results(by_ticker)

            scan_type = str(scan_type or "seasonality")

            for ticker in by_ticker.keys():
                entries = self._pattern_index.get(ticker, {})
                ticker_result = by_ticker.get(ticker, {})
                metadata = {}
                if isinstance(ticker_result, Mapping):
                    meta_candidate = ticker_result.get("metadata")
                    if isinstance(meta_candidate, Mapping):
                        metadata = dict(meta_candidate)
                rows = self._summarize_patterns_for_ticker(
                    symbol=ticker,
                    scan_name=scan_name,
                    scan_type=scan_type,
                    entries=entries,
                    created_at=timestamp,
                    metadata=metadata,
                )

                if not rows:
                    continue

                df = pd.DataFrame(rows, columns=self.SUMMARY_COLUMNS)
                df = self._coerce_summary_dtypes(df)
                key = self._build_results_key(scan_type, ticker, scan_name)
                results_client.write_results_df(key, df, replace=True)
                counts[key] = len(df)
                print(f"[PatternExtractor] wrote {len(df)} rows to {key}")

        return counts

    @classmethod
    def load_summaries_from_results(
        cls,
        results_client: "ResultsClient",
        *,
        scan_type: str,
        scan_name: str,
        tickers: Sequence[str],
        errors: str = "raise",
        **select_kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """Synchronously load summary rows for ``tickers`` from ``results_client``.

        This convenience method simply wraps
        :meth:`load_summaries_from_results_async` using
        :func:`CTAFlow.utils.aio.run`.
        """

        return aio.run(
            cls.load_summaries_from_results_async(
                results_client,
                scan_type=scan_type,
                scan_name=scan_name,
                tickers=tickers,
                errors=errors,
                **select_kwargs,
            )
        )

    @classmethod
    async def load_summaries_from_results_async(
        cls,
        results_client: "ResultsClient",
        *,
        scan_type: str,
        scan_name: str,
        tickers: Sequence[str],
        errors: str = "raise",
        **select_kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """Asynchronously load summary rows for ``tickers`` from ``results_client``.

        Parameters
        ----------
        results_client : ResultsClient
            Client used to read stored summary frames.
        scan_type : str
            Screener category (``"seasonality"`` or ``"orderflow"``).
        scan_name : str
            Specific scan identifier.
        tickers : Sequence[str]
            Iterable of ticker symbols to load.
        errors : {"raise", "ignore"}, optional
            Behaviour when a key is missing.  Defaults to raising.
        **select_kwargs
            Additional keyword arguments forwarded to :meth:`pandas.HDFStore.select`.

        Returns
        -------
        Dict[str, pandas.DataFrame]
            Mapping of normalised ticker symbols to DataFrames containing the
            persisted summary schema (:attr:`SUMMARY_COLUMNS`).
        """

        if not tickers:
            return {}

        alias_to_key = {
            str(ticker).upper(): results_client.make_key(scan_type, ticker, scan_name)
            for ticker in tickers
        }
        loaded = await results_client.load_many_results_async(
            alias_to_key,
            errors=errors,
            **select_kwargs,
        )

        summaries: Dict[str, pd.DataFrame] = {}
        for alias, df in loaded.items():
            if not isinstance(df, pd.DataFrame):
                continue
            summaries[alias] = df.reindex(columns=cls.SUMMARY_COLUMNS)

        return summaries

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        for screen_name, ticker_results in self._results.items():
            params = self._screen_params.get(screen_name)
            declared_type = getattr(params, "screen_type", None)

            for symbol, result in ticker_results.items():
                if not isinstance(result, Mapping) or "error" in result:
                    continue

                symbol_patterns = self._pattern_index.setdefault(symbol, {})
                screen_type = self._infer_screen_type(declared_type, result)

                if self._is_orderflow_result(result):
                    for summary in self._iter_orderflow_summaries(
                        symbol=symbol,
                        screen_name=screen_name,
                        ticker_result=result,
                        params=params,
                    ):
                        symbol_patterns[summary.key] = summary
                    continue

                for summary in self._iter_historical_summaries(
                    symbol=symbol,
                    screen_name=screen_name,
                    screen_type=screen_type,
                    ticker_result=result,
                    params=params,
                ):
                    symbol_patterns[summary.key] = summary

    def _infer_screen_type(
        self,
        declared_type: Optional[str],
        ticker_result: Mapping[str, Any],
    ) -> Optional[str]:
        """Infer the screen type when params were not provided.

        Some callers instantiate :class:`PatternExtractor` without passing the
        originating :class:`ScreenParams`.  When that happens we need to detect
        momentum outputs directly from the payload so the specialised parsing
        path runs instead of the generic ``strongest_patterns`` handler.
        """

        if declared_type:
            return declared_type

        if self._is_momentum_result(ticker_result):
            return "momentum"

        return None

    def _iter_historical_summaries(
        self,
        symbol: str,
        screen_name: str,
        screen_type: Optional[str],
        ticker_result: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[PatternSummary]:
        if screen_type == "momentum":
            yield from self._iter_momentum_summaries(
                symbol=symbol,
                screen_name=screen_name,
                ticker_result=ticker_result,
                params=params,
            )
            return

        strongest: Iterable[Dict[str, Any]] = ticker_result.get("strongest_patterns", [])  # type: ignore[assignment]
        for pattern in strongest:
            if not self._should_include_pattern(pattern):
                continue
            yield self._build_summary(
                symbol,
                screen_name,
                screen_type,
                params,
                pattern,
                origin="strongest_patterns",
            )

    def _iter_momentum_summaries(
        self,
        *,
        symbol: str,
        screen_name: str,
        ticker_result: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[PatternSummary]:
        momentum_sessions = [
            (session_key, session_value)
            for session_key, session_value in ticker_result.items()
            if isinstance(session_key, str)
            and session_key.startswith("session_")
            and isinstance(session_value, Mapping)
        ]

        if not momentum_sessions:
            return

        for session_key, session_data in momentum_sessions:
            if "error" in session_data:
                continue

            momentum_breakdown = session_data.get("momentum_by_dayofweek")
            if isinstance(momentum_breakdown, Mapping):
                for pattern in self._build_momentum_patterns(
                    ticker_result=ticker_result,
                    session_key=session_key,
                    session_data=session_data,
                    momentum_breakdown=momentum_breakdown,
                    params=params,
                ):
                    yield self._build_summary(
                        symbol,
                        screen_name,
                        "momentum",
                        params,
                        pattern,
                        origin=f"{session_key}.momentum_by_dayofweek",
                    )

                for pattern in self._build_momentum_summary_patterns(
                    ticker_result=ticker_result,
                    session_key=session_key,
                    session_data=session_data,
                    momentum_breakdown=momentum_breakdown,
                    params=params,
                ):
                    yield self._build_summary(
                        symbol,
                        screen_name,
                        "momentum",
                        params,
                        pattern,
                        origin=f"{session_key}.momentum_by_dayofweek.summary",
                    )

            for pattern in self._build_momentum_correlation_patterns(
                ticker_result=ticker_result,
                session_key=session_key,
                session_data=session_data,
                params=params,
            ):
                yield self._build_summary(
                    symbol,
                    screen_name,
                    "momentum",
                    params,
                    pattern,
                    origin=f"{session_key}.correlations",
                )

            for pattern in self._build_momentum_volatility_patterns(
                ticker_result=ticker_result,
                session_key=session_key,
                session_data=session_data,
                params=params,
            ):
                yield self._build_summary(
                    symbol,
                    screen_name,
                    "momentum",
                    params,
                    pattern,
                    origin=f"{session_key}.volatility",
                )

    def _build_momentum_patterns(
        self,
        *,
        ticker_result: Mapping[str, Any],
        session_key: str,
        session_data: Mapping[str, Any],
        momentum_breakdown: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[Dict[str, Any]]:
        context = self._momentum_pattern_context(
            ticker_result=ticker_result,
            session_key=session_key,
            session_data=session_data,
            params=params,
        )

        for momentum_type, label in self.MOMENTUM_TYPE_LABELS.items():
            series_key = f"{momentum_type}_by_dow"
            dow_stats = momentum_breakdown.get(series_key)
            if not isinstance(dow_stats, Mapping):
                continue

            for weekday_name, stats in dow_stats.items():
                if weekday_name == "anova" or not isinstance(stats, Mapping):
                    continue

                t_stat = stats.get("t_stat")
                if t_stat is None:
                    continue
                try:
                    t_value = float(t_stat)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(t_value) or abs(t_value) < 1.96:
                    continue

                p_vs_rest = self._coerce_optional_float(stats.get("p_value_vs_rest"))
                cohen_d = self._coerce_optional_float(stats.get("cohen_d_vs_rest"))
                significant_vs_rest = bool(stats.get("significant_vs_rest"))

                if not significant_vs_rest:
                    if p_vs_rest is None or p_vs_rest >= 0.02:
                        continue
                    if cohen_d is not None and abs(cohen_d) < 0.35:
                        continue

                pattern: Dict[str, Any] = {
                    "pattern_type": "momentum_weekday",
                    "momentum_type": momentum_type,
                    "weekday": weekday_name,
                    "description": f"{label} skew on {weekday_name}",
                    "t_stat": t_value,
                    "strength": abs(t_value),
                    "mean": stats.get("mean"),
                    "sharpe": stats.get("sharpe"),
                    "positive_pct": stats.get("positive_pct"),
                    "skew": stats.get("skew"),
                    "n": stats.get("n"),
                    "bias": self._infer_momentum_bias(stats.get("mean")),
                    "window_anchor": self._momentum_window_anchor(momentum_type),
                    "window_minutes": self._momentum_window_minutes(momentum_type, params),
                    "p_value_vs_rest": p_vs_rest,
                    "cohen_d_vs_rest": cohen_d,
                    "significant_vs_rest": significant_vs_rest,
                }

                months_active = stats.get("months_active")
                if months_active:
                    pattern["months_active"] = months_active
                months_mask = stats.get("months_mask_12")
                if months_mask:
                    pattern["months_mask_12"] = months_mask
                months_names = stats.get("months_names")
                if months_names:
                    pattern["months_names"] = months_names

                filtered_months = ticker_result.get("filtered_months")
                if filtered_months and "months_active" not in pattern:
                    pattern["months_active"] = filtered_months

                summary = momentum_breakdown.get("summary")
                if isinstance(summary, Mapping):
                    significant = summary.get("significant_patterns")
                    if isinstance(significant, list):
                        matched = next(
                            (
                                item
                                for item in significant
                                if isinstance(item, Mapping)
                                and item.get("momentum_type") == momentum_type
                            ),
                            None,
                        )
                        if isinstance(matched, Mapping):
                            if "p_value" in matched:
                                pattern.setdefault("p_value", matched.get("p_value"))
                            if "f_stat" in matched:
                                pattern.setdefault("f_stat", matched.get("f_stat"))

                self._apply_momentum_context(pattern, context)
                yield pattern

    def _momentum_pattern_context(
        self,
        *,
        ticker_result: Mapping[str, Any],
        session_key: str,
        session_data: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "session_key": session_key,
            "session_index": self._parse_session_index(session_key),
            "session_start": self._normalise_session_clock(
                session_data.get("session_start"), params, "session_starts", session_key
            ),
            "session_end": self._normalise_session_clock(
                session_data.get("session_end"), params, "session_ends", session_key
            ),
        }
        if params is not None and hasattr(params, "tz"):
            context["session_tz"] = getattr(params, "tz", None)

        if params is not None:
            for attr in ("sess_start_hrs", "sess_start_minutes", "sess_end_hrs", "sess_end_minutes"):
                context[attr] = getattr(params, attr, None)

        start_window = self._derive_window_minutes(params, prefix="sess_start")
        if start_window is not None:
            context.setdefault("period_length_min", start_window)
            context.setdefault("window_minutes", start_window)

        filtered_months = ticker_result.get("filtered_months")
        if filtered_months is not None:
            context["filtered_months"] = filtered_months

        regime_meta = ticker_result.get("regime_filter")
        if regime_meta is not None:
            context["regime_filter"] = regime_meta

        momentum_params = session_data.get("momentum_params") or ticker_result.get("momentum_params")
        if isinstance(momentum_params, Mapping):
            context["momentum_params"] = dict(momentum_params)
            st_days = momentum_params.get("st_momentum_days")
            if st_days is not None:
                context.setdefault("st_momentum_days", st_days)
            period_minutes = momentum_params.get("period_length_min")
            if period_minutes is not None:
                context.setdefault("period_length_min", period_minutes)

        return {key: value for key, value in context.items() if value is not None}

    def _normalise_session_clock(
        self,
        explicit_value: Any,
        params: Optional[ScreenParamLike],
        attr: str,
        session_key: str,
    ) -> Optional[str]:
        clock = self._safe_time_string(explicit_value)
        if clock is not None:
            return clock
        session_index = self._parse_session_index(session_key) or 0
        if params is None or not hasattr(params, attr):
            return None
        candidates = getattr(params, attr, None)
        if not candidates:
            return None
        try:
            chosen = candidates[session_index]
        except Exception:
            return None
        return self._safe_time_string(chosen)

    @staticmethod
    def _derive_window_minutes(
        params: Optional[ScreenParamLike], *, prefix: str
    ) -> Optional[int]:
        if params is None:
            return None
        hours = getattr(params, f"{prefix}_hrs", None)
        minutes = getattr(params, f"{prefix}_minutes", None)
        try:
            total = int(hours or 0) * 60 + int(minutes or 0)
        except (TypeError, ValueError):
            return None
        return total if total > 0 else None

    @staticmethod
    def _apply_momentum_context(pattern: Dict[str, Any], context: Mapping[str, Any]) -> None:
        if "window_minutes" not in pattern:
            momentum_type = pattern.get("momentum_type")
            if momentum_type == "opening_momentum" and "opening_window_minutes" in context:
                pattern["window_minutes"] = context.get("opening_window_minutes")
            elif momentum_type == "closing_momentum" and "closing_window_minutes" in context:
                pattern["window_minutes"] = context.get("closing_window_minutes")
            elif "period_length_min" in context:
                pattern["window_minutes"] = context.get("period_length_min")

        for key, value in context.items():
            pattern.setdefault(key, value)

    def _build_momentum_summary_patterns(
        self,
        *,
        ticker_result: Mapping[str, Any],
        session_key: str,
        session_data: Mapping[str, Any],
        momentum_breakdown: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[Dict[str, Any]]:
        context = self._momentum_pattern_context(
            ticker_result=ticker_result,
            session_key=session_key,
            session_data=session_data,
            params=params,
        )
        summary = momentum_breakdown.get("summary")
        flagged: Set[str] = set()

        if isinstance(summary, Mapping):
            candidates = summary.get("significant_patterns", [])
            if isinstance(candidates, list):
                for entry in candidates:
                    if not isinstance(entry, Mapping):
                        continue
                    momentum_type = entry.get("momentum_type")
                    if not momentum_type:
                        continue
                    p_value = self._coerce_optional_float(entry.get("p_value"))
                    if p_value is None or p_value >= 0.05:
                        continue
                    series_key = f"{momentum_type}_by_dow"
                    dow_stats = momentum_breakdown.get(series_key)
                    if not isinstance(dow_stats, Mapping):
                        continue
                    flagged.add(momentum_type)
                    pattern = self._build_momentum_summary_pattern(
                        momentum_type,
                        self.MOMENTUM_TYPE_LABELS.get(momentum_type, momentum_type),
                        dow_stats,
                        p_value,
                        self._coerce_optional_float(entry.get("f_stat")),
                        entry,
                    )
                    self._apply_momentum_context(pattern, context)
                    yield pattern

        for momentum_type, label in self.MOMENTUM_TYPE_LABELS.items():
            if momentum_type in flagged:
                continue
            series_key = f"{momentum_type}_by_dow"
            dow_stats = momentum_breakdown.get(series_key)
            if not isinstance(dow_stats, Mapping):
                continue
            anova = dow_stats.get("anova")
            if not isinstance(anova, Mapping) or not anova.get("significant"):
                continue
            p_value = self._coerce_optional_float(anova.get("p_value"))
            if p_value is None or p_value >= 0.05:
                continue
            pattern = self._build_momentum_summary_pattern(
                momentum_type,
                label,
                dow_stats,
                p_value,
                self._coerce_optional_float(anova.get("f_stat")),
                anova,
            )
            self._apply_momentum_context(pattern, context)
            yield pattern

    def _build_momentum_summary_pattern(
        self,
        momentum_type: str,
        label: str,
        dow_stats: Mapping[str, Any],
        p_value: Optional[float],
        f_stat: Optional[float],
        summary_info: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        best_day, best_stats = self._extract_weekday_extreme(dow_stats, largest=True)
        worst_day, worst_stats = self._extract_weekday_extreme(dow_stats, largest=False)
        strongest_day, strongest_stats = self._extract_strongest_weekday(dow_stats)

        description = f"{label} varies by weekday"
        if p_value is not None:
            description += f" (p={p_value:.4f})"
        if best_day:
            description += f"; strongest on {best_day}"

        strength = 0.0
        if strongest_stats is not None:
            strength = abs(self._coerce_optional_float(strongest_stats.get("t_stat")) or 0.0)
            if strength == 0.0:
                strength = abs(self._coerce_optional_float(strongest_stats.get("mean")) or 0.0)
        elif best_stats is not None:
            strength = abs(self._coerce_optional_float(best_stats.get("t_stat")) or 0.0)
            if strength == 0.0:
                strength = abs(self._coerce_optional_float(best_stats.get("mean")) or 0.0)

        pattern: Dict[str, Any] = {
            "pattern_type": "weekday_bias_intraday",
            "momentum_type": momentum_type,
            "weekday": best_day,
            "description": description,
            "strength": strength,
            "p_value": p_value,
            "f_stat": f_stat,
        }

        def _assign(name: str, stats: Optional[Mapping[str, Any]], key: str) -> None:
            if stats is None:
                return
            value = stats.get(key)
            if value is not None:
                pattern[name] = value

        if best_day:
            pattern["best_weekday"] = best_day
        _assign("best_mean", best_stats, "mean")

        if worst_day:
            pattern["worst_weekday"] = worst_day
        _assign("worst_mean", worst_stats, "mean")

        if strongest_day:
            pattern["strongest_days"] = [strongest_day]
            strong_t = self._coerce_optional_float((strongest_stats or {}).get("t_stat"))
            if strong_t is not None:
                pattern["t_stat"] = strong_t

        if "t_stat" not in pattern and best_stats is not None:
            best_t = self._coerce_optional_float(best_stats.get("t_stat"))
            if best_t is not None:
                pattern["t_stat"] = best_t

        best_n = (best_stats or {}).get("n")
        total_obs = summary_info.get("total_observations") if summary_info else None
        pattern["n"] = best_n or total_obs

        # Add correlation proxy based on t_stat sign for positioning logic
        # Positive t_stat means the pattern predicts positive returns (long bias)
        # Negative t_stat means the pattern predicts negative returns (short bias)
        t_stat_value = pattern.get("t_stat")
        if t_stat_value is not None:
            # Use t_stat sign as correlation proxy (normalized to [-1, 1] range)
            pattern["correlation"] = 1.0 if t_stat_value > 0 else -1.0
        elif "best_mean" in pattern:
            # Fallback to best_mean sign if t_stat not available
            best_mean_value = pattern["best_mean"]
            if best_mean_value is not None:
                pattern["correlation"] = 1.0 if best_mean_value > 0 else -1.0

        return pattern

    def _build_momentum_correlation_patterns(
        self,
        *,
        ticker_result: Mapping[str, Any],
        session_key: str,
        session_data: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[Dict[str, Any]]:
        correlations = session_data.get("correlations")
        if not isinstance(correlations, Mapping):
            return

        context = self._momentum_pattern_context(
            ticker_result=ticker_result,
            session_key=session_key,
            session_data=session_data,
            params=params,
        )

        specs = [
            ("open_close_corr", "open_close_pvalue", "momentum_oc", "Opening momentum predicts the close", "opening_momentum"),
            ("close_st_mom_corr", None, "momentum_cc", "Closing drive aligns with the short-term trend", "closing_momentum"),
            ("close_vs_rest_corr", "close_vs_rest_pvalue", "momentum_sc", "Closing thrust spills into the rest of the session", "closing_momentum"),
            ("open_st_mom_corr", None, "momentum_so", "Short-term trend predicts the open", "st_momentum"),
        ]

        for corr_key, p_key, pattern_type, description, momentum_type in specs:
            corr_value = self._coerce_optional_float(correlations.get(corr_key))
            if corr_value is None or not math.isfinite(corr_value):
                continue
            p_value = self._coerce_optional_float(correlations.get(p_key)) if p_key else None
            if p_value is None:
                p_value = self._pearson_p_value(corr_value, correlations.get("n_observations"))
            if p_value is None or p_value >= 0.05:
                continue
            pattern: Dict[str, Any] = {
                "pattern_type": pattern_type,
                "description": description,
                "correlation": corr_value,
                "strength": abs(corr_value),
                "p_value": p_value,
                "n": correlations.get("n_observations"),
                "momentum_type": momentum_type,
            }
            self._apply_momentum_context(pattern, context)
            yield pattern

    def _build_momentum_volatility_patterns(
        self,
        *,
        ticker_result: Mapping[str, Any],
        session_key: str,
        session_data: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[Dict[str, Any]]:
        # Volatility patterns (vol_persistence and volatility_weekday_bias) have been
        # removed from significant patterns as they are no longer needed.
        # This method now returns an empty iterator.
        return iter([])
        # Suppressed code below - keeping for reference only
        # volatility = session_data.get("volatility")
        # if not isinstance(volatility, Mapping):
        #     return
        #
        # context = self._momentum_pattern_context(
        #     ticker_result=ticker_result,
        #     session_key=session_key,
        #     session_data=session_data,
        #     params=params,
        # )
        #
        # if volatility.get("vol_correlation_significant"):
        #     pattern: Dict[str, Any] = {
        #         "pattern_type": "vol_persistence",
        #         "description": volatility.get(
        #             "vol_correlation_interpretation",
        #             "Opening volatility predicts closing volatility",
        #         ),
        #         "correlation": self._coerce_optional_float(volatility.get("opening_closing_vol_correlation")),
        #         "p_value": self._coerce_optional_float(volatility.get("vol_correlation_pvalue")),
        #         "n": (volatility.get("overall_stats") or {}).get("n_observations"),
        #         "n_high_vol_days": volatility.get("n_high_vol_days"),
        #         "n_low_vol_days": volatility.get("n_low_vol_days"),
        #     }
        #     self._apply_momentum_context(pattern, context)
        #     yield pattern
        #
        # weekday_breakdown = volatility.get("volatility_by_dayofweek")
        # if not isinstance(weekday_breakdown, Mapping):
        #     return
        #
        # counts: List[Tuple[str, Mapping[str, Any]]] = [
        #     (weekday, stats)
        #     for weekday, stats in weekday_breakdown.items()
        #     if isinstance(stats, Mapping)
        # ]
        # if not counts:
        #     return
        # total_high = sum(int(stats.get("high_vol_days_count") or 0) for _, stats in counts)
        # if total_high <= 0:
        #     return
        # avg_high = total_high / len(counts)
        # for weekday, stats in counts:
        #     high_count = int(stats.get("high_vol_days_count") or 0)
        #     if high_count < avg_high * 1.25 or high_count < 5:
        #         continue
        #     pattern = {
        #         "pattern_type": "volatility_weekday_bias",
        #         "weekday": weekday,
        #         "description": f"High volatility disproportionately occurs on {weekday}",
        #         "high_vol_days_count": high_count,
        #         "volatility_bias_ratio": high_count / avg_high if avg_high else None,
        #     }
        #     self._apply_momentum_context(pattern, context)
        #     yield pattern

    @classmethod
    def _iter_weekday_stats(
        cls,
        dow_stats: Mapping[str, Any],
    ) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        for weekday, stats in dow_stats.items():
            if not isinstance(stats, Mapping):
                continue
            if isinstance(weekday, str) and weekday.lower() == "anova":
                continue
            yield weekday, stats

    @classmethod
    def _extract_weekday_extreme(
        cls,
        dow_stats: Mapping[str, Any],
        *,
        largest: bool,
    ) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
        candidates = list(cls._iter_weekday_stats(dow_stats))
        if not candidates:
            return None, None

        chosen: Optional[Tuple[str, Mapping[str, Any]]] = None
        target = -math.inf if largest else math.inf
        for candidate in candidates:
            weekday, stats = candidate
            mean_value = cls._coerce_optional_float(stats.get("mean"))
            if mean_value is None:
                continue
            if largest and mean_value > target:
                target = mean_value
                chosen = candidate
            if not largest and mean_value < target:
                target = mean_value
                chosen = candidate

        if chosen is None:
            chosen = candidates[0]
        return chosen

    @classmethod
    def _extract_strongest_weekday(
        cls,
        dow_stats: Mapping[str, Any],
    ) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
        candidates = list(cls._iter_weekday_stats(dow_stats))
        if not candidates:
            return None, None

        best_score = -math.inf
        chosen: Optional[Tuple[str, Mapping[str, Any]]] = None
        for candidate in candidates:
            weekday, stats = candidate
            t_value = cls._coerce_optional_float(stats.get("t_stat"))
            if t_value is not None:
                score = abs(t_value)
            else:
                mean_value = cls._coerce_optional_float(stats.get("mean"))
                score = abs(mean_value) if mean_value is not None else 0.0
            if score > best_score:
                best_score = score
                chosen = candidate

        if chosen is None:
            return None, None
        return chosen

    @staticmethod
    def _pearson_p_value(correlation: float, n_observations: Any) -> Optional[float]:
        try:
            n = int(n_observations)
        except (TypeError, ValueError):
            return None
        df = n - 2
        if df <= 0 or not math.isfinite(correlation):
            return None
        denom = 1 - correlation ** 2
        if denom <= 0:
            return None
        try:
            t_stat = abs(correlation) * math.sqrt(df / denom)
        except (ValueError, ZeroDivisionError):
            return None

        if scipy_stats is not None:
            return float(2 * scipy_stats.t.sf(t_stat, df))

        # Normal approximation fallback when SciPy is unavailable
        approx = math.erfc(t_stat / math.sqrt(2))
        return float(min(1.0, approx))

    @staticmethod
    def _parse_session_index(session_key: str) -> Optional[int]:
        match = re.match(r"session_(\d+)", session_key)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _infer_momentum_bias(mean_value: Any) -> str:
        try:
            value = float(mean_value)
        except (TypeError, ValueError):
            return "neutral"
        if not math.isfinite(value) or value == 0:
            return "neutral"
        return "long" if value > 0 else "short"

    @staticmethod
    def _momentum_window_anchor(momentum_type: str) -> str:
        if momentum_type == "opening_momentum":
            return "start"
        if momentum_type == "closing_momentum":
            return "end"
        return "session"

    @staticmethod
    def _momentum_window_minutes(
        momentum_type: str,
        params: Optional[ScreenParamLike],
    ) -> Optional[int]:
        if params is None or not hasattr(params, "sess_start_hrs"):
            return None

        if momentum_type == "opening_momentum":
            hours = getattr(params, "sess_start_hrs", None)
            minutes = getattr(params, "sess_start_minutes", None)
            return PatternExtractor._combine_minutes(hours, minutes)

        if momentum_type == "closing_momentum":
            hours = getattr(params, "sess_end_hrs", None)
            minutes = getattr(params, "sess_end_minutes", None)
            if hours is None:
                hours = getattr(params, "sess_start_hrs", None)
            if minutes is None:
                minutes = getattr(params, "sess_start_minutes", None)
            return PatternExtractor._combine_minutes(hours, minutes)

        return None

    @staticmethod
    def _combine_minutes(hours: Any, minutes: Any) -> Optional[int]:
        try:
            total = int(hours or 0) * 60 + int(minutes or 0)
        except (TypeError, ValueError):
            return None
        return total if total > 0 else None

    @classmethod
    def _should_include_pattern(cls, pattern: Mapping[str, Any]) -> bool:
        pattern_type = str(pattern.get("pattern_type") or pattern.get("type") or "")
        if pattern_type != "weekend_hedging":
            return True

        p_value = cls._extract_weekend_p_value(pattern)
        if p_value is None:
            return False

        return p_value < 0.05

    @classmethod
    def _extract_weekend_p_value(cls, pattern: Mapping[str, Any]) -> Optional[float]:
        candidates: List[Any] = []
        for source_key in ("p_value", "seasonality_p_value"):
            if source_key in pattern:
                candidates.append(pattern.get(source_key))

        payload = pattern.get("pattern_payload")
        if isinstance(payload, Mapping):
            for source_key in ("p_value", "seasonality_p_value"):
                if source_key in payload:
                    candidates.append(payload.get(source_key))

        metadata = pattern.get("metadata")
        if isinstance(metadata, Mapping):
            for source_key in ("p_value", "seasonality_p_value"):
                if source_key in metadata:
                    candidates.append(metadata.get(source_key))

        for candidate in candidates:
            value = cls._coerce_optional_float(candidate)
            if value is not None:
                return value
        return None

    def _iter_orderflow_summaries(
        self,
        symbol: str,
        screen_name: str,
        ticker_result: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[PatternSummary]:
        weekly = ticker_result.get("df_weekly")
        if isinstance(weekly, pd.DataFrame) and not weekly.empty:
            mask_series = weekly.get("sig_fdr_5pct")
            if mask_series is not None:
                mask_series = mask_series.fillna(False).astype(bool)
            else:
                mask_series = pd.Series(True, index=weekly.index)
            for row in weekly.loc[mask_series].itertuples(index=False):
                payload = row._asdict()
                metric = str(payload.get("metric", ""))
                weekday = str(payload.get("weekday", ""))
                bias = self._orderflow_bias(metric, payload.get("mean"))
                metadata = {
                    "pattern_origin": "orderflow_weekly",
                    "screen_type": "orderflow",
                    "orderflow_bias": bias,
                }
                yield PatternSummary(
                    key="|".join(filter(None, [screen_name, "orderflow_weekly", metric, weekday])),
                    symbol=symbol,
                    source_screen=screen_name,
                    screen_params=params,
                    pattern_type="orderflow_weekly",
                    description=self._format_orderflow_weekly_description(metric, weekday, payload, bias),
                    strength=self._orderflow_strength(payload.get("t_stat"), payload.get("mean")),
                    payload=dict(payload),
                    metadata=metadata,
                )

        wom = ticker_result.get("df_wom_weekday")
        if isinstance(wom, pd.DataFrame) and not wom.empty:
            mask_series = wom.get("sig_fdr_5pct")
            if mask_series is not None:
                mask_series = mask_series.fillna(False).astype(bool)
            else:
                mask_series = pd.Series(True, index=wom.index)
            for row in wom.loc[mask_series].itertuples(index=False):
                payload = row._asdict()
                metric = str(payload.get("metric", ""))
                weekday = str(payload.get("weekday", ""))
                wom_value = payload.get("week_of_month")
                bias = self._orderflow_bias(metric, payload.get("mean"))
                parts = [screen_name, "orderflow_week_of_month", metric]
                if wom_value is not None and not pd.isna(wom_value):
                    parts.append(f"w{int(wom_value)}")
                parts.append(weekday)
                metadata = {
                    "pattern_origin": "orderflow_week_of_month",
                    "screen_type": "orderflow",
                    "orderflow_bias": bias,
                }
                yield PatternSummary(
                    key="|".join(filter(None, parts)),
                    symbol=symbol,
                    source_screen=screen_name,
                    screen_params=params,
                    pattern_type="orderflow_week_of_month",
                    description=self._format_orderflow_wom_description(metric, weekday, wom_value, payload, bias),
                    strength=self._orderflow_strength(payload.get("t_stat"), payload.get("mean")),
                    payload=dict(payload),
                    metadata=metadata,
                )

        peak = ticker_result.get("df_weekly_peak_pressure")
        if isinstance(peak, pd.DataFrame) and not peak.empty:
            for row in peak.itertuples(index=False):
                payload = row._asdict()
                metric = str(payload.get("metric", ""))
                weekday = str(payload.get("weekday", ""))
                clock_time = payload.get("clock_time")
                bias = str(payload.get("pressure_bias", "neutral"))
                parts = [screen_name, "orderflow_peak_pressure", metric, weekday]
                if clock_time is not None:
                    parts.append(str(clock_time))
                metadata = {
                    "pattern_origin": "orderflow_peak_pressure",
                    "screen_type": "orderflow",
                    "orderflow_bias": bias,
                }
                yield PatternSummary(
                    key="|".join(filter(None, parts)),
                    symbol=symbol,
                    source_screen=screen_name,
                    screen_params=params,
                    pattern_type="orderflow_peak_pressure",
                    description=self._format_orderflow_peak_description(metric, weekday, clock_time, payload, bias),
                    strength=self._orderflow_strength(payload.get("seasonality_t_stat"), payload.get("intraday_mean")),
                    payload=dict(payload),
                    metadata=metadata,
                )

    def _build_summary(
        self,
        symbol: str,
        screen_name: str,
        screen_type: Optional[str],
        params: Optional[ScreenParamLike],
        pattern: Mapping[str, Any],
        origin: str,
    ) -> PatternSummary:
        pattern_type = self._infer_pattern_type(pattern)
        strength = self._extract_strength(pattern)
        description = str(pattern.get("description", pattern_type))

        qualifiers: List[str] = [pattern_type]
        for key in ("day", "weekday", "time", "lag", "month", "momentum_type", "session_key"):
            if key in pattern and pattern[key] is not None:
                qualifiers.append(str(pattern[key]))
        key = "|".join([screen_name, *qualifiers])

        summary_key = "|".join([screen_name, *qualifiers])

        metadata: Dict[str, Any] = {}
        raw_metadata = pattern.get("metadata")
        if isinstance(raw_metadata, Mapping):
            metadata.update(raw_metadata)

        metadata["pattern_origin"] = origin
        if screen_type is not None and metadata.get("screen_type") is None:
            metadata["screen_type"] = screen_type

        payload_dict = dict(pattern)

        if screen_type == "seasonality" and isinstance(params, ScreenParams):
            metadata.setdefault("target_time", pattern.get("time"))
            metadata.setdefault("most_prevalent_day", pattern.get("most_prevalent_day"))

        months_sources = (
            metadata.get("months"),
            payload_dict.get("months"),
            payload_dict.get("months_active"),
            payload_dict.get("filtered_months"),
            self.metadata.get("filtered_months"),
        )

        for candidate in months_sources:
            canonical_months = self._coerce_month_list(candidate)
            if canonical_months:
                metadata["months"] = canonical_months
                break

        for meta_key in (
            "months_active",
            "months_mask_12",
            "months_names",
            "target_times_hhmm",
            "period_length_min",
            "momentum_params",
            "st_momentum_days",
            "momentum_type",
            "bias",
            "session_key",
            "session_index",
            "session_start",
            "session_end",
            "session_tz",
            "sess_start",
            "sess_end",
            "sess_start_hrs",
            "sess_start_minutes",
            "sess_end_hrs",
            "sess_end_minutes",
            "window_anchor",
            "window_minutes",
            "positive_pct",
            "skew",
        ):
            value = payload_dict.get(meta_key)
            if value is not None and meta_key not in metadata:
                metadata[meta_key] = value

        regime_meta = payload_dict.get("regime_filter") or metadata.get("regime_filter")
        if regime_meta is not None:
            metadata["regime_filter"] = regime_meta

        time_value = payload_dict.get("time")
        normalized_time = self._safe_time_string(time_value)
        if normalized_time is not None:
            metadata.setdefault("time", normalized_time)

        weekday_value = payload_dict.get("weekday") or payload_dict.get("day")
        if weekday_value is not None:
            metadata.setdefault("weekday", str(weekday_value))

        wom_value = payload_dict.get("week_of_month")
        if wom_value is not None and not pd.isna(wom_value):
            try:
                metadata.setdefault("week_of_month", int(wom_value))
            except (TypeError, ValueError):
                pass

        strongest_days = payload_dict.get("strongest_days")
        if strongest_days:
            metadata.setdefault("strongest_days", list(strongest_days))

        period_label = self._format_minutes_label(payload_dict.get("period_length_min"))
        if period_label is not None:
            metadata.setdefault("period_length", period_label)

        if payload_dict.get("period_length") and "period_length" not in metadata:
            metadata["period_length"] = payload_dict["period_length"]

        if "best_weekday" not in metadata and "weekday" in metadata:
            metadata["best_weekday"] = metadata["weekday"]

        return PatternSummary(
            key=summary_key,
            symbol=symbol,
            source_screen=screen_name,
            screen_params=params,
            pattern_type=pattern_type,
            description=description,
            strength=strength,
            payload=dict(pattern),
            metadata=metadata,
        )

    @staticmethod
    def _coerce_month_list(values: Iterable[Any] | Any) -> Optional[List[int]]:
        if values in (None, "", (), [], {}):  # type: ignore[comparison-overlap]
            return None

        if isinstance(values, str):
            tokens = re.split(r"[^0-9]+", values)
            candidates: Iterable[Any] = [token for token in tokens if token]
        elif isinstance(values, IterableABC):
            candidates = values
        else:
            candidates = [values]

        months: List[int] = []
        for candidate in candidates:
            try:
                month = int(candidate)
            except (TypeError, ValueError):
                continue
            if 1 <= month <= 12:
                months.append(month)

        if not months:
            return None

        return sorted(set(months))

    @classmethod
    def _safe_time_string(cls, value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            normalised = cls._normalize_time_value(value)
        except Exception:  # pragma: no cover - defensive guard
            return None

        text = normalised.isoformat()
        if not normalised.microsecond:
            return normalised.strftime("%H:%M:%S")
        return text

    @staticmethod
    def _format_minutes_label(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            minutes_float = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(minutes_float):  # type: ignore[arg-type]
            return None

        total_minutes = int(round(minutes_float))
        if total_minutes < 0:
            total_minutes = abs(total_minutes)

        hours, minutes = divmod(total_minutes, 60)
        return f"{hours}h{minutes}m"

    # ------------------------------------------------------------------
    # Series reconstruction helpers
    # ------------------------------------------------------------------
    def _extract_weekday_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        params = self._ensure_seasonality_params(summary, context="Weekday")

        session_data, price_col, is_synthetic = self._get_seasonality_session_data(symbol, params)
        returns = self._compute_intraday_returns(session_data, price_col, is_synthetic)
        if returns.empty:
            return pd.Series(dtype=float)

        daily_returns = returns.groupby(returns.index.normalize()).sum()
        target_day = summary.payload.get("day")
        if target_day is None:
            return daily_returns

        weekday_number = self._weekday_name_to_number(target_day)
        mask = daily_returns.index.dayofweek == weekday_number
        return daily_returns[mask]

    def _extract_time_of_day_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        params = self._ensure_seasonality_params(summary, context="Time-of-day")

        target_time_raw = summary.payload.get("time")
        if target_time_raw is None:
            raise ValueError("Pattern payload missing target time")

        target_time_obj = self._normalize_time_value(target_time_raw)

        session_data, price_col, is_synthetic = self._get_seasonality_session_data(symbol, params)
        if session_data.empty:
            return pd.Series(dtype=float)

        period_length = self._resolve_period_length(summary, params)

        returns = self._compute_time_of_day_returns(
            session_data,
            price_col,
            is_synthetic,
            target_time_obj,
            period_length,
        )
        return returns

    def _resolve_period_length(
        self,
        summary: PatternSummary,
        params: ScreenParamLike,
    ) -> Optional[pd.Timedelta]:
        """Determine the aggregation window for time-of-day series."""

        minutes = self._period_minutes_from_summary(summary)

        raw_period: Any
        if minutes is not None:
            raw_period = minutes
        else:
            raw_period = getattr(params, "period_length", None) if params is not None else None

        if raw_period is None:
            return None

        if isinstance(raw_period, pd.Timedelta):
            return raw_period

        minutes_value = self._coerce_minutes(raw_period)
        if minutes_value is None:
            return None

        return pd.Timedelta(minutes=minutes_value)

    @staticmethod
    def _period_minutes_from_summary(summary: PatternSummary) -> Optional[float]:
        """Extract ``period_length_min`` from summary metadata/payload."""

        containers = []
        if isinstance(summary.metadata, Mapping):
            containers.append(summary.metadata)
        if isinstance(summary.payload, Mapping):
            containers.append(summary.payload)

        for container in containers:
            value = container.get("period_length_min")
            minutes = PatternExtractor._coerce_minutes(value)
            if minutes is not None:
                return minutes
        return None

    @staticmethod
    def _coerce_minutes(value: Any) -> Optional[float]:
        if value in (None, "", 0, 0.0):
            return None

        if isinstance(value, pd.Timedelta):
            minutes = value.total_seconds() / 60.0
        else:
            try:
                minutes = float(value)
            except (TypeError, ValueError):
                return None

        if math.isnan(minutes) or minutes <= 0:
            return None

        return minutes

    def _extract_weekend_hedging_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        params = self._ensure_seasonality_params(summary, context="Weekend hedging")

        session_data, price_col, is_synthetic = self._get_seasonality_session_data(symbol, params)
        if session_data.empty:
            return pd.Series(dtype=float)

        session_start = self._screener._convert_times([params.seasonality_session_start or "00:00"])[0]
        session_end = self._screener._convert_times([params.seasonality_session_end or "23:59:59"])[0]

        full_returns = self._screener._calculate_full_session_returns(
            session_data,
            session_start,
            session_end,
            price_col,
            is_synthetic,
        )

        if full_returns.empty:
            return pd.Series(dtype=float)

        returns = full_returns.sort_index()
        returns.index = pd.to_datetime(returns.index)
        friday_mask = returns.index.dayofweek == 4
        if not friday_mask.any():
            return pd.Series(dtype=float)

        next_dates = returns.index.to_series().shift(-1)
        monday_mask = next_dates.dt.dayofweek == 0
        monday_returns = returns.shift(-1)
        valid = friday_mask & monday_mask
        if not valid.any():
            return pd.Series(dtype=float)

        series = monday_returns.loc[valid].copy()
        series.name = "weekend_hedging_return"
        return series

    def _extract_monthly_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        price_df = self._get_price_series(symbol, price_col="Close")
        if price_df.empty:
            return pd.Series(dtype=float)

        price_col = price_df.columns[0]
        use_log_returns = not self._screener.synthetic_tickers.get(symbol, False)
        monthly_ret = monthly_returns(
            price_df,
            price_col=price_col,
            use_log_returns=use_log_returns,
        )

        series = monthly_ret[price_col]
        if summary.pattern_type == "abnormal_month":
            month_name = summary.payload.get("month")
            if month_name:
                mask = series.index.month_name() == month_name
                series = series[mask]
        return series

    # ------------------------------------------------------------------
    # Parameter resolution helpers
    # ------------------------------------------------------------------
    def _ensure_seasonality_params(
        self,
        summary: PatternSummary,
        *,
        context: str,
    ) -> ScreenParams:
        params = summary.screen_params
        if self._is_seasonality_params(params):
            return params  # type: ignore[return-value]

        direct = self._screen_params.get(summary.source_screen)
        if self._is_seasonality_params(direct):
            summary.screen_params = direct
            return direct  # type: ignore[return-value]

        screen_root = summary.source_screen.split("|", 1)[0]
        root_match = self._screen_params.get(screen_root)
        if self._is_seasonality_params(root_match):
            summary.screen_params = root_match
            return root_match  # type: ignore[return-value]

        seasonality_params = [
            candidate
            for candidate in self._screen_params.values()
            if self._is_seasonality_params(candidate)
        ]

        target_time_raw = summary.payload.get("time")
        matches: List[ScreenParamLike] = []
        if target_time_raw is not None:
            try:
                target_time = self._normalize_time_value(target_time_raw)
            except Exception:  # pragma: no cover - defensive guard
                target_time = None

            if target_time is not None:
                for candidate in seasonality_params:
                    target_times = getattr(candidate, "target_times", None)
                    if not target_times:
                        continue
                    try:
                        normalized = [self._normalize_time_value(value) for value in target_times]
                    except Exception:  # pragma: no cover - defensive guard
                        continue
                    if target_time in normalized:
                        matches.append(candidate)

        if not matches and len(seasonality_params) == 1:
            matches = seasonality_params

        if matches:
            chosen = matches[0]
            summary.screen_params = chosen
            return chosen  # type: ignore[return-value]

        raise ValueError(
            f"{context} patterns require seasonality screen parameters; "
            f"no configuration available for screen '{summary.source_screen}'."
        )

    @staticmethod
    def _is_seasonality_params(params: Optional[ScreenParamLike]) -> bool:
        return params is not None and getattr(params, "screen_type", None) == "seasonality"

    @staticmethod
    def _normalize_time_value(value: Any) -> time:
        if isinstance(value, time):
            return value
        parsed = pd.to_datetime(value).time()
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return parsed

    # ------------------------------------------------------------------
    # Low-level utilities
    # ------------------------------------------------------------------
    def _get_price_series(self, symbol: str, price_col: str) -> pd.DataFrame:
        data = self._screener.data[symbol]
        if data.empty:
            return pd.DataFrame(columns=[price_col])
        if price_col not in data.columns:
            price_col = data.columns[0]
        return data[[price_col]].copy()

    def _get_seasonality_session_data(
        self,
        symbol: str,
        params: ScreenParamLike,
    ) -> Tuple[pd.DataFrame, str, bool]:
        data = self._screener.data[symbol]
        if data.empty:
            return pd.DataFrame(), "Close", False

        session_start = params.seasonality_session_start or "00:00"
        session_end = params.seasonality_session_end or "23:59:59"

        start_time = self._screener._convert_times([session_start])[0]
        end_time = self._screener._convert_times([session_end])[0]
        tz = params.tz or "America/Chicago"

        session_data = filter_session_bars(
            data,
            tz,
            start_time,
            end_time,
        )

        months = self._screener._parse_season_months(params.months, params.season)
        if months:
            session_data = self._screener._filter_by_months(session_data, months)

        if session_data.empty:
            return session_data, "Close", self._screener.synthetic_tickers.get(symbol, False)

        price_col = "Close" if "Close" in session_data.columns else session_data.columns[0]
        is_synthetic = self._screener.synthetic_tickers.get(symbol, False)
        return session_data, price_col, is_synthetic

    @staticmethod
    def _compute_intraday_returns(
        session_data: pd.DataFrame,
        price_col: str,
        is_synthetic: bool,
    ) -> pd.Series:
        if session_data.empty:
            return pd.Series(dtype=float)

        if is_synthetic:
            returns = session_data[price_col].diff()
        else:
            valid_prices = session_data[price_col] > 0
            prices = session_data.loc[valid_prices, price_col]
            returns = pd.Series(index=session_data.index, dtype=float)
            returns.loc[valid_prices] = log_returns(prices)
        return returns.dropna()

    @staticmethod
    def _compute_time_of_day_returns(
        session_data: pd.DataFrame,
        price_col: str,
        is_synthetic: bool,
        target_time: time,
        period_length: Optional[pd.Timedelta],
    ) -> pd.Series:
        returns = PatternExtractor._compute_intraday_returns(session_data, price_col, is_synthetic)
        if returns.empty:
            return pd.Series(dtype=float)

        mask = tod_mask(returns.index, target_time.strftime("%H:%M"))
        if period_length:
            window_bars = max(int(period_length.total_seconds() / 60 / 5), 1)
            daily_returns = aggregate_window(returns, mask, window_bars)
        else:
            daily_returns = returns[mask]

        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            daily_returns.index = pd.to_datetime(daily_returns.index)
        daily_returns = daily_returns.groupby(daily_returns.index.date).sum()
        daily_returns.index = pd.to_datetime(daily_returns.index)
        return daily_returns

    @staticmethod
    def _infer_pattern_type(pattern: Mapping[str, Any]) -> str:
        for key in ("type", "pattern_type", "momentum_type"):
            value = pattern.get(key)
            if value:
                return str(value)
        return pattern.get("description", "pattern").replace(" ", "_")

    @staticmethod
    def _extract_strength(pattern: Mapping[str, Any]) -> Optional[float]:
        for key in ("strength", "f_stat", "correlation", "sharpe", "t_stat"):
            value = pattern.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _weekday_name_to_number(name: str) -> int:
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        try:
            return weekdays[name.lower()]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown weekday '{name}'") from exc

    # ------------------------------------------------------------------
    # Orderflow / Momentum helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_momentum_result(ticker_result: Mapping[str, Any]) -> bool:
        if not isinstance(ticker_result, Mapping):
            return False

        if str(ticker_result.get("screen_type")) == "momentum":
            return True

        if "momentum_params" in ticker_result:
            return True

        for key, value in ticker_result.items():
            if not isinstance(key, str) or not key.startswith("session_"):
                continue
            if not isinstance(value, Mapping):
                continue
            if any(
                candidate in value and isinstance(value.get(candidate), Mapping)
                for candidate in ("momentum_by_dayofweek", "correlations", "volatility")
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Orderflow helpers
    # ------------------------------------------------------------------
    def _is_orderflow_result(self, ticker_result: Mapping[str, Any]) -> bool:
        keys = set(ticker_result.keys())
        if {"df_weekly", "df_intraday_pressure", "df_buckets"}.intersection(keys):
            metadata = ticker_result.get("metadata")
            return isinstance(metadata, Mapping)
        return False

    def _get_orderflow_result(self, screen_name: str, symbol: str) -> Optional[Mapping[str, Any]]:
        screen_results = self._results.get(screen_name)
        if not isinstance(screen_results, Mapping):
            return None
        result = screen_results.get(symbol)
        if not isinstance(result, Mapping):
            return None
        if not self._is_orderflow_result(result):
            return None
        return result

    def _extract_orderflow_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        result = self._get_orderflow_result(summary.source_screen, symbol)
        if result is None:
            return pd.Series(dtype=float)

        bucket_df = result.get("df_buckets")
        if not isinstance(bucket_df, pd.DataFrame) or bucket_df.empty:
            return pd.Series(dtype=float)

        metric = summary.payload.get("metric")
        if metric is None or metric not in bucket_df.columns:
            return pd.Series(dtype=float)

        df = bucket_df.copy()
        df["ts_end"] = pd.to_datetime(df["ts_end"])
        df = df.dropna(subset=["ts_end", metric])

        pattern_type = summary.pattern_type

        if pattern_type == "orderflow_weekly":
            weekday = summary.payload.get("weekday")
            if weekday is None or "weekday" not in df.columns:
                return pd.Series(dtype=float)
            mask = df["weekday"].astype(str) == str(weekday)
        elif pattern_type == "orderflow_week_of_month":
            mask = pd.Series(True, index=df.index)
            weekday = summary.payload.get("weekday")
            if weekday is not None and "weekday" in df.columns:
                mask &= df["weekday"].astype(str) == str(weekday)
            week_of_month = summary.payload.get("week_of_month")
            if week_of_month is not None and "week_of_month" in df.columns:
                try:
                    week_value = int(week_of_month)
                except (TypeError, ValueError):
                    week_value = None
                if week_value is not None:
                    mask &= df["week_of_month"].astype(int) == week_value
        elif pattern_type == "orderflow_peak_pressure":
            mask = pd.Series(True, index=df.index)
            weekday = summary.payload.get("weekday")
            if weekday is not None and "weekday" in df.columns:
                mask &= df["weekday"].astype(str) == str(weekday)
            clock_time = summary.payload.get("clock_time")
            if clock_time is not None and "clock_time" in df.columns:
                normalized = self._normalize_time_value(clock_time)
                mask &= df["clock_time"].apply(self._normalize_time_value) == normalized
        else:
            return pd.Series(dtype=float)

        if mask.empty or not mask.any():
            return pd.Series(dtype=float)

        series = df.loc[mask, ["ts_end", metric]].set_index("ts_end")[metric]
        series.name = metric
        return series.sort_index()

    @staticmethod
    def _normalize_time_value(value: Any) -> Optional[time]:
        if value is None:
            return None
        if isinstance(value, time):
            return value
        try:
            return pd.to_datetime(value).time()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _orderflow_bias(metric: str, mean_value: Any) -> str:
        try:
            mean = float(mean_value)
        except (TypeError, ValueError):
            return "neutral"
        if pd.isna(mean) or mean == 0:
            return "neutral"
        lower_metric = str(metric).lower()
        if lower_metric in {"buy_pressure", "quote_buy_pressure"}:
            return "buy" if mean > 0 else "sell"
        if lower_metric == "sell_pressure":
            return "sell" if mean > 0 else "buy"
        return "positive" if mean > 0 else "negative"

    @staticmethod
    def _orderflow_strength(primary: Any, fallback: Any) -> Optional[float]:
        for value in (primary, fallback):
            if value is None:
                continue
            try:
                return float(abs(value))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _format_orderflow_weekly_description(
        metric: str,
        weekday: str,
        payload: Dict[str, Any],
        bias: str,
    ) -> str:
        mean_value = payload.get("mean")
        t_stat = payload.get("t_stat")
        q_value = payload.get("q_value")
        parts = [f"{metric} bias {bias} on {weekday}"]
        if pd.notna(mean_value):
            parts.append(f"mean={float(mean_value):.4f}")
        if pd.notna(t_stat):
            parts.append(f"t={float(t_stat):.2f}")
        if pd.notna(q_value):
            parts.append(f"q={float(q_value):.3g}")
        if len(parts) == 1:
            return parts[0]
        return parts[0] + " (" + ", ".join(parts[1:]) + ")"

    @staticmethod
    def _format_orderflow_wom_description(
        metric: str,
        weekday: str,
        wom_value: Any,
        payload: Dict[str, Any],
        bias: str,
    ) -> str:
        prefix = f"{metric} week-of-month"
        if wom_value is not None and not pd.isna(wom_value):
            try:
                prefix += f" {int(wom_value)}"
            except (TypeError, ValueError):
                pass
        prefix += f" {weekday} bias {bias}"
        details: List[str] = []
        mean_value = payload.get("mean")
        t_stat = payload.get("t_stat")
        q_value = payload.get("q_value")
        if pd.notna(mean_value):
            details.append(f"mean={float(mean_value):.4f}")
        if pd.notna(t_stat):
            details.append(f"t={float(t_stat):.2f}")
        if pd.notna(q_value):
            details.append(f"q={float(q_value):.3g}")
        if details:
            prefix += " (" + ", ".join(details) + ")"
        return prefix

    @staticmethod
    def _format_orderflow_peak_description(
        metric: str,
        weekday: str,
        clock_time: Any,
        payload: Dict[str, Any],
        bias: str,
    ) -> str:
        time_display = str(clock_time) if clock_time is not None else "?"
        details: List[str] = []
        seasonality_mean = payload.get("seasonality_mean")
        intraday_mean = payload.get("intraday_mean")
        if pd.notna(seasonality_mean):
            details.append(f"seasonal_mean={float(seasonality_mean):.4f}")
        if pd.notna(intraday_mean):
            details.append(f"intraday_mean={float(intraday_mean):.4f}")
        detail_str = ""
        if details:
            detail_str = " (" + ", ".join(details) + ")"
        return f"{metric} peak {bias} on {weekday} at {time_display}{detail_str}"

__all__ = ["PatternExtractor", "PatternSummary", "validate_filtered_months"]
