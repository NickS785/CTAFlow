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
from dataclasses import dataclass, field
from datetime import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import pandas as pd

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

    if rejected and logger is not None and hasattr(logger, "warning"):
        logger.warning(
            "[PatternExtractor] Dropped invalid filtered_month entries",
            extra={"context": context, "invalid": rejected},
        )

    if not valid:
        return None

    return valid


@dataclass
class PatternSummary:
    """Container describing a statistically significant pattern."""

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
            "pattern_type": self.pattern_type,
            "description": self.description,
            "strength": self.strength,
            "source_screen": self.source_screen,
            "screen_parameters": params_dict,
            "metadata": self.metadata,
            "pattern_payload": self.payload,
        }


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
    )

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
        clone._filtered_months = set(self._filtered_months) if self._filtered_months else None
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

        if self._filtered_months is None:
            return None
        return set(self._filtered_months)

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
            filtered[key] = summary.as_dict()

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
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
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
                rows = self._summarize_patterns_for_ticker(
                    symbol=ticker,
                    scan_name=scan_name,
                    scan_type=scan_type,
                    entries=entries,
                    created_at=timestamp,
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
            screen_type = getattr(params, "screen_type", None)

            for symbol, result in ticker_results.items():
                if not isinstance(result, Mapping) or "error" in result:
                    continue

                symbol_patterns = self._pattern_index.setdefault(symbol, {})

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

    def _iter_historical_summaries(
        self,
        symbol: str,
        screen_name: str,
        screen_type: Optional[str],
        ticker_result: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[PatternSummary]:
        strongest: Iterable[Dict[str, Any]] = ticker_result.get("strongest_patterns", [])  # type: ignore[assignment]
        for pattern in strongest:
            yield self._build_summary(
                symbol,
                screen_name,
                screen_type,
                params,
                pattern,
                origin="strongest_patterns",
            )

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
        for key in ("day", "time", "lag", "month"):
            if key in pattern and pattern[key] is not None:
                qualifiers.append(str(pattern[key]))
        key = "|".join([screen_name, *qualifiers])

        metadata = {
            "pattern_origin": origin,
            "screen_type": screen_type,
        }
        if screen_type == "seasonality" and isinstance(params, ScreenParams):
            metadata.update(
                {
                    "target_time": pattern.get("time"),
                    "most_prevalent_day": pattern.get("most_prevalent_day"),
                }
            )

        return PatternSummary(
            key=key,
            symbol=symbol,
            source_screen=screen_name,
            screen_params=params,
            pattern_type=pattern_type,
            description=description,
            strength=strength,
            payload=dict(pattern),
            metadata=metadata,
        )

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

        period_length = params.period_length
        if isinstance(period_length, (int, float)):
            period_length = pd.Timedelta(minutes=period_length)

        returns = self._compute_time_of_day_returns(
            session_data,
            price_col,
            is_synthetic,
            target_time_obj,
            period_length,
        )
        return returns

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
