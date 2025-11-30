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
import time as time_module
from collections.abc import Iterable as IterableABC, Sequence as SequenceABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import time as time_cls
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    NamedTuple,
)

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from ..screeners.pattern_extractor import PatternExtractor
    from .sessionizer import Sessionizer

from .backtester import ScreenerBacktester
from .gpu_acceleration import (
    GPU_AVAILABLE,
    GPU_DEVICE_COUNT,
    to_backend_array,
    to_cpu,
)
from .prediction_to_position import PredictionToPosition

__all__ = [
    "ScreenerPipeline",
    "extract_ticker_patterns",
    "HorizonMapper",
    "HorizonSpec",
    "ScreenerBacktester",
    "PredictionToPosition",
]


def _is_numeric(value: Any) -> bool:
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


class WeekendPatternSpec(NamedTuple):
    gate_day: str
    target_day: str
    gate_time: Optional[str]


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
        "weekend_hedging",
    }

    #: Pattern types handled by the orderflow extractor branch
    _ORDERFLOW_TYPES = {
        "orderflow_week_of_month",
        "orderflow_weekly",
        "orderflow_peak_pressure",
    }

    #: Pattern types emitted by the momentum extractor
    _MOMENTUM_TYPES = {
        "momentum_weekday",
        "momentum_oc",
        "momentum_cc",
        "momentum_sc",
        "momentum_so",
        "weekday_bias_intraday",
    }

    def __init__(
        self,
        tz: str = "America/Chicago",
        time_match: str = "auto",
        log: Any = None,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
    ) -> None:
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
        use_gpu:
            Enable GPU acceleration for backtesting operations (default: True).
            Automatically falls back to CPU if GPU not available.
        gpu_device_id:
            GPU device ID to use for computations when multiple GPUs available (default: 0).
        """

        if time_match not in {"auto", "hms", "hmsf"}:
            raise ValueError("time_match must be one of {'auto', 'hms', 'hmsf'}")

        self.tz = tz
        self.time_match = time_match
        self.log = log
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.allow_naive_ts = False
        self.nan_policy = "drop"
        self.asof_tolerance = None
        self.return_clip = (-0.5, 0.5)
        self.sessionizer = None
        self.weekend_exit_policy = "last"
        self.data_clean_policy = "ffill"  # Policy for cleaning numeric columns: 'ffill', 'drop', 'zero'
        self._mapper_cache: Dict[Tuple[Any, ...], HorizonMapper] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_features(
        self,
        bars: pd.DataFrame,
        patterns: Any,
        *,
        allowed_months: Optional[Iterable[int]] = None,
        prepared_df: Optional[pd.DataFrame] = None,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
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
        allowed_months:
            Optional iterable of calendar months. When provided the emitted gate
            columns are zeroed outside of the specified months.
        prepared_df:
            Optional prevalidated and time-enriched DataFrame. Providing this
            avoids re-running the preparation steps when repeatedly building
            feature sets on the same bar data.
        max_workers:
            Optional thread pool size used to parallelise independent pattern
            extraction. When omitted and GPU is enabled, automatically uses
            GPU_DEVICE_COUNT workers. Otherwise, patterns are processed sequentially.
        """

        df = prepared_df if prepared_df is not None else self._prepare_bars(bars)

        items = tuple(self._items_from_patterns(patterns))
        gate_columns: List[str] = []

        def _build_single(key: str, pattern: Mapping[str, Any]) -> Tuple[List[str], Mapping[str, pd.Series]]:
            local_df = df.copy(deep=False)
            try:
                created_cols = self._dispatch(local_df, pattern, key)
            except Exception as exc:  # pragma: no cover - defensive logging path
                if self.log is not None and hasattr(self.log, "warning"):
                    self.log.warning("[screener_pipeline] skipping '%s': %s", key, exc)
                return [], {}

            if not created_cols:
                return [], {}

            return created_cols, {col: local_df[col] for col in created_cols}

        # Auto-determine worker count for GPU-enabled batching
        worker_count = max_workers
        if (
            worker_count is None
            and self.use_gpu
            and GPU_AVAILABLE
            and GPU_DEVICE_COUNT > 0
            and len(items) > 1
        ):
            # Align pattern extraction parallelism with available GPUs when requested
            worker_count = GPU_DEVICE_COUNT

        if worker_count and len(items) > 1:
            collected: List[Tuple[List[str], Mapping[str, pd.Series]]] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(_build_single, key, pattern): key for key, pattern in items
                }
                for future in as_completed(future_map):
                    collected.append(future.result())

            for created_cols, column_map in collected:
                if not created_cols:
                    continue
                gate_columns.extend(created_cols)
                for col in created_cols:
                    df[col] = column_map[col]
        else:
            for key, pattern in items:
                created = []
                try:
                    created = self._dispatch(df, pattern, key)
                except Exception as exc:  # pragma: no cover - defensive logging path
                    if self.log is not None and hasattr(self.log, "warning"):
                        self.log.warning("[screener_pipeline] skipping '%s': %s", key, exc)
                    continue

                gate_columns.extend(created)

        self._combine_gate_columns(df, gate_columns)

        if allowed_months is not None:
            allowed_set = {int(month) for month in allowed_months}
            month_mask = df["ts"].dt.month.isin(sorted(allowed_set))
            gate_cols = [col for col in df.columns if col.endswith("_gate")]
            if gate_cols:
                df.loc[~month_mask, gate_cols] = 0
                self._combine_gate_columns(df, gate_cols)
            else:
                df["any_pattern_active"] = 0

        return df

    def backtest_threshold(
        self,
        bars_with_features: pd.DataFrame,
        patterns: Any,
        *,
        threshold: float = 0.0,
        use_side_hint: bool = True,
        annualisation: int = 252,
        risk_free_rate: float = 0.0,
        include_metadata: Optional[Iterable[str]] = None,
        prediction_resolver: Optional[PredictionToPosition] = None,
        **build_xy_kwargs: Any,
    ) -> Dict[str, Any]:
        """Materialise decision rows and run a threshold backtest."""

        xy_kwargs = dict(build_xy_kwargs)
        combined_meta: List[str] = []

        def _extend_meta(source: Any) -> None:
            if source is None:
                return
            if isinstance(source, str):
                combined_meta.append(source)
                return
            combined_meta.extend(list(source))

        _extend_meta(xy_kwargs.pop("include_metadata", None))
        _extend_meta(include_metadata)
        combined_meta.append("correlation")
        xy_kwargs["include_metadata"] = list(dict.fromkeys(filter(None, combined_meta)))

        tester = ScreenerBacktester(
            annualisation=annualisation,
            risk_free_rate=risk_free_rate,
            use_gpu=self.use_gpu,
            gpu_device_id=self.gpu_device_id,
        )

        policy_map = {"auto": "auto", "hms": "second", "hmsf": "microsecond"}
        mapper_time_match = policy_map.get(self.time_match, "auto")

        if "time_match" in xy_kwargs:
            override = xy_kwargs["time_match"]
            if override in policy_map:
                xy_kwargs["time_match"] = policy_map[override]

        mapper = self._get_horizon_mapper(mapper_time_match)

        xy = mapper.build_xy(bars_with_features, patterns, **xy_kwargs)
        resolver = prediction_resolver if prediction_resolver is not None else PredictionToPosition()
        return tester.threshold(
            xy,
            threshold=threshold,
            use_side_hint=use_side_hint,
            prediction_resolver=resolver,
        )

    # ------------------------------------------------------------------
    # Horizon mapper helpers
    # ------------------------------------------------------------------
    def _mapper_cache_key(self, time_match: str) -> Tuple[Any, ...]:
        return (
            self.tz,
            bool(getattr(self, "allow_naive_ts", False)),
            time_match,
            getattr(self, "nan_policy", "drop"),
            getattr(self, "return_clip", (-0.5, 0.5)),
            getattr(self, "asof_tolerance", None),
            getattr(self, "weekend_exit_policy", "last"),
            id(getattr(self, "sessionizer", None)),
        )

    def _get_horizon_mapper(self, time_match: str) -> HorizonMapper:
        key = self._mapper_cache_key(time_match)
        cached = self._mapper_cache.get(key)
        if cached is not None:
            return cached

        mapper = HorizonMapper(
            tz=self.tz,
            allow_naive_ts=getattr(self, "allow_naive_ts", False),
            time_match=time_match,
            nan_policy=getattr(self, "nan_policy", "drop"),
            return_clip=getattr(self, "return_clip", (-0.5, 0.5)),
            asof_tolerance=getattr(self, "asof_tolerance", None),
            log=self.log,
            sessionizer=getattr(self, "sessionizer", None),
            weekend_exit_policy=getattr(self, "weekend_exit_policy", "last"),
        )
        self._mapper_cache[key] = mapper
        return mapper

    def build_and_backtest(
        self,
        bars: pd.DataFrame,
        patterns: Any,
        *,
        allowed_months: Optional[Iterable[int]] = None,
        threshold: float = 0.0,
        use_side_hint: bool = True,
        annualisation: int = 252,
        risk_free_rate: float = 0.0,
        include_metadata: Optional[Iterable[str]] = None,
        prediction_resolver: Optional[PredictionToPosition] = None,
        build_xy_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Convenience wrapper to build features then backtest.

        The bar validation and time enrichment steps are shared between feature
        construction and backtesting to minimise repeated DataFrame copies.
        """

        prepared = self._prepare_bars(bars)
        featured = self.build_features(
            prepared,
            patterns,
            allowed_months=allowed_months,
            prepared_df=prepared,
        )
        backtest_result = self.backtest_threshold(
            featured,
            patterns,
            threshold=threshold,
            use_side_hint=use_side_hint,
            annualisation=annualisation,
            risk_free_rate=risk_free_rate,
            include_metadata=include_metadata,
            prediction_resolver=prediction_resolver,
            **(dict(build_xy_kwargs) if build_xy_kwargs else {}),
        )
        return featured, backtest_result

    def build_feature_sets(
        self,
        bars: pd.DataFrame,
        patterns: Any,
        *,
        allowed_months: Optional[Iterable[int]] = None,
        max_workers: Optional[int] = None,
        prepared_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Build separate feature sets for each pattern key.

        This helper is intended for workflows that need to evaluate patterns
        independently (e.g., concurrent backtests) without recomputing the
        common timestamp preparation for every run. A preprepared DataFrame can
        be supplied to skip validation, and feature extraction can be
        parallelised via ``max_workers`` when many patterns are present.
        """

        items = list(self._items_from_patterns(patterns))
        if not items:
            return {}

        prepared = prepared_df if prepared_df is not None else self._prepare_bars(bars)
        featured = self.build_features(
            prepared,
            dict(items),
            allowed_months=allowed_months,
            prepared_df=prepared,
            max_workers=max_workers,
        )

        return {key: featured for key, _ in items}

    def concurrent_pattern_backtests(
        self,
        bars: pd.DataFrame,
        patterns: Any,
        *,
        threshold: float = 0.0,
        use_side_hint: bool = True,
        annualisation: int = 252,
        risk_free_rate: float = 0.0,
        include_metadata: Optional[Iterable[str]] = None,
        prediction_resolver: Optional[PredictionToPosition] = None,
        max_workers: Optional[int] = None,
        feature_workers: Optional[int] = None,
        build_xy_kwargs: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Run pattern backtests individually using a thread pool.

        Each pattern is evaluated in isolation so the resulting performance
        metrics can be compared directly. Feature sets for each pattern are
        materialised up front (optionally in parallel) and then reused during
        concurrent backtesting to avoid redundant gate construction. Results
        are ordered by ``total_return / max_drawdown`` and include a
        ``ranking_score`` field for convenience.

        Parameters
        ----------
        max_workers : Optional[int]
            Number of parallel workers for pattern backtesting. When GPU is enabled
            and max_workers is None or 1, patterns are processed sequentially to
            avoid thread/GPU contention. Set to a value >1 to force parallel execution.
        verbose : bool
            If True, prints progress updates including BacktestSummary for each
            completed pattern. Shows pattern name, completion count, and key metrics
            (return, Sharpe, drawdown, trades). Default: False.
        """

        items = list(self._items_from_patterns(patterns))
        if not items:
            return {}

        prepared = self._prepare_bars(bars)
        results: Dict[str, Dict[str, Any]] = {}
        build_xy_params = dict(build_xy_kwargs) if build_xy_kwargs else {}

        total_patterns = len(items)
        completed_count = 0
        start_time = time_module.time()

        if verbose:
            execution_mode = "GPU Sequential" if (self.use_gpu and GPU_AVAILABLE and (max_workers is None or max_workers == 1)) else f"ThreadPool ({max_workers or 'auto'} workers)"
            print(f"\n{'='*70}")
            print(f"Starting concurrent backtests for {total_patterns} patterns")
            print(f"Threshold: {threshold}, GPU: {self.use_gpu}, Mode: {execution_mode}")
            print(f"{'='*70}\n")

        worker_count = feature_workers
        if (
            worker_count is None
            and self.use_gpu
            and GPU_AVAILABLE
            and GPU_DEVICE_COUNT > 0
        ):
            # Align feature extraction parallelism with available GPUs when requested
            worker_count = GPU_DEVICE_COUNT

        feature_sets = self.build_feature_sets(
            prepared,
            patterns,
            prepared_df=prepared,
            max_workers=worker_count,
        )

        def _run_single(key: str, featured: pd.DataFrame, pattern: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
            outcome = self.backtest_threshold(
                featured,
                {key: pattern},
                threshold=threshold,
                use_side_hint=use_side_hint,
                annualisation=annualisation,
                risk_free_rate=risk_free_rate,
                include_metadata=include_metadata,
                prediction_resolver=prediction_resolver,
                **build_xy_params,
            )
            return key, outcome

        # When GPU is enabled, run sequentially to avoid thread/GPU contention
        # ThreadPoolExecutor adds overhead and can cause GPU serialization issues
        if self.use_gpu and GPU_AVAILABLE and (max_workers is None or max_workers == 1):
            # Sequential GPU execution with progress updates
            for key, pattern in items:
                if key not in feature_sets:
                    continue
                try:
                    pat_key, result = _run_single(key, feature_sets[key], pattern)
                    completed_count += 1

                    # Print progress if verbose
                    if verbose:
                        summary = result.get('summary')
                        if summary:
                            elapsed = time_module.time() - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0

                            print(f"[{completed_count}/{total_patterns}] {pat_key}")
                            print(f"  Return: {summary.total_return:>8.2%}  Sharpe: {summary.sharpe:>6.2f}  "
                                  f"MaxDD: {summary.max_drawdown:>8.2%}  Trades: {summary.trades:>4d}")
                            print(f"  Elapsed: {elapsed:.1f}s  Rate: {rate:.2f} patterns/sec\n")
                        else:
                            print(f"[{completed_count}/{total_patterns}] {pat_key} - No summary available\n")

                except Exception as exc:  # pragma: no cover - defensive path
                    completed_count += 1
                    if verbose:
                        print(f"[{completed_count}/{total_patterns}] {key} - FAILED: {exc}\n")
                    if self.log is not None and hasattr(self.log, "warning"):
                        self.log.warning("[screener_pipeline] backtest failed for '%s': %s", key, exc)
                    continue
                results[pat_key] = result
        else:
            # Use ThreadPoolExecutor for CPU or when explicitly requested
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_run_single, key, feature_sets[key], pattern): key
                    for key, pattern in items
                    if key in feature_sets
                }
                for future in as_completed(future_map):
                    key = future_map[future]
                    try:
                        pat_key, result = future.result()
                        completed_count += 1

                        # Print progress if verbose
                        if verbose:
                            summary = result.get('summary')
                            if summary:
                                elapsed = time_module.time() - start_time
                                rate = completed_count / elapsed if elapsed > 0 else 0

                                print(f"[{completed_count}/{total_patterns}] {pat_key}")
                                print(f"  Return: {summary.total_return:>8.2%}  Sharpe: {summary.sharpe:>6.2f}  "
                                      f"MaxDD: {summary.max_drawdown:>8.2%}  Trades: {summary.trades:>4d}")
                                print(f"  Elapsed: {elapsed:.1f}s  Rate: {rate:.2f} patterns/sec\n")
                            else:
                                print(f"[{completed_count}/{total_patterns}] {pat_key} - No summary available\n")

                    except Exception as exc:  # pragma: no cover - defensive path
                        completed_count += 1
                        if verbose:
                            print(f"[{completed_count}/{total_patterns}] {key} - FAILED: {exc}\n")
                        if self.log is not None and hasattr(self.log, "warning"):
                            self.log.warning("[screener_pipeline] backtest failed for '%s': %s", key, exc)
                        continue
                    results[pat_key] = result

        ranking = ScreenerBacktester.rank_results(results)
        ordered: Dict[str, Dict[str, Any]] = {}
        for key, score in ranking:
            payload = dict(results.get(key, {}))
            payload["ranking_score"] = score
            ordered[key] = payload

        if verbose:
            total_time = time_module.time() - start_time
            avg_rate = len(results) / total_time if total_time > 0 else 0
            print(f"{'='*70}")
            print(f"Completed {len(results)}/{total_patterns} backtests in {total_time:.1f}s")
            print(f"Average rate: {avg_rate:.2f} patterns/sec")

            if len(results) > 0:
                print(f"\nTop 3 patterns by ranking score:")
                for idx, (key, score) in enumerate(ranking[:3], 1):
                    summary = ordered[key].get('summary')
                    if summary:
                        print(f"  {idx}. {key}: score={score:.4f}, return={summary.total_return:.2%}, "
                              f"Sharpe={summary.sharpe:.2f}")
            print(f"{'='*70}\n")

        return ordered

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _prepare_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Validate and enrich bar data in a single pass.

        Combining validation and time column creation reduces the number of
        DataFrame copies required when repeatedly building feature sets during
        grid searches or concurrent backtests.
        """

        validated = self._validate_bars(bars)
        cleaned = self._clean_numeric_data(validated)
        return self._ensure_time_cols(cleaned, copy=False)

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

    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate numeric columns using the configured data_clean_policy.

        Handles non-numeric types and np.nan values in OHLCV and other numeric columns.
        Default policy is 'ffill' to forward fill missing values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with potentially dirty numeric data

        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned numeric columns
        """
        out = df.copy()

        # Identify numeric columns (OHLCV and common numeric columns)
        numeric_candidates = ['open', 'high', 'low', 'close', 'volume',
                             'Open', 'High', 'Low', 'Close', 'Volume',
                             'bid', 'ask', 'last', 'Bid', 'Ask', 'Last']

        numeric_cols = [col for col in numeric_candidates if col in out.columns]

        if not numeric_cols:
            return out

        policy = getattr(self, 'data_clean_policy', 'ffill')

        for col in numeric_cols:
            # Convert to numeric, coercing errors to NaN
            out[col] = pd.to_numeric(out[col], errors='coerce')

            # Replace inf values with NaN
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)

            # Apply cleaning policy
            if policy == 'ffill':
                out[col] = out[col].ffill()
            elif policy == 'drop':
                # Drop rows with NaN in this column
                out = out.dropna(subset=[col])
            elif policy == 'zero':
                out[col] = out[col].fillna(0.0)

        return out

    def _ensure_time_cols(self, df: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
        out = df.copy() if copy else df
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
        if hasattr(obj, "pattern_type") and hasattr(obj, "as_dict"):
            try:
                pattern_dict = dict(obj.as_dict())  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensive guard
                pattern_dict = {}
            if "pattern_payload" not in pattern_dict and hasattr(obj, "payload"):
                pattern_dict["pattern_payload"] = getattr(obj, "payload")
            if "metadata" not in pattern_dict and hasattr(obj, "metadata"):
                pattern_dict["metadata"] = getattr(obj, "metadata")
            key_value = getattr(obj, "key", None) or key_hint
            yield cls._select_pattern_key(pattern_dict, key_value), pattern_dict
            return

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

    @classmethod
    def _momentum_weekday_info(
        cls, payload: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> Tuple[Optional[str], str]:
        def _iter_candidates(candidate: Any) -> Iterable:
            if candidate is None:
                return ()
            if isinstance(candidate, (list, tuple, set)):
                return candidate
            return (candidate,)

        sources: List[Any] = [
            payload.get("weekday"),
            payload.get("day"),
            metadata.get("weekday"),
            metadata.get("best_weekday"),
            metadata.get("strongest_days"),
            payload.get("strongest_days"),
        ]

        for candidate in sources:
            for entry in _iter_candidates(candidate):
                normalized = cls._normalize_weekday(entry)
                if normalized:
                    return normalized, normalized

        return None, "all_days"

    @classmethod
    def _momentum_base_suffix(
        cls,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any],
        pattern_type: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        weekday_norm, weekday_token = cls._momentum_weekday_info(payload, metadata)
        momentum_type = (
            payload.get("momentum_type")
            or metadata.get("momentum_type")
            or "momentum"
        )
        pattern_token = cls._slugify(pattern_type) if pattern_type else None
        session_key = metadata.get("session_key") or payload.get("session_key")
        session_index = metadata.get("session_index") or payload.get("session_index")
        if pattern_token in {"momentum_cc", "momentum_sc", "momentum_oc"}:
            suffix = f"{pattern_token}_{cls._slugify(momentum_type)}_{weekday_token}"
        else:
            suffix = f"momentum_{momentum_type}_{weekday_token}"
        if session_key:
            suffix = f"{suffix}_{cls._slugify(str(session_key))}"
        elif session_index is not None:
            suffix = f"{suffix}_session{session_index}"
        return weekday_norm, suffix

    def _broadcast_sidecar(self, df: pd.DataFrame, mask: pd.Series, value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            series = value.reindex(df.index)
            series.loc[~mask] = np.nan
            return series

        if isinstance(value, np.ndarray) and value.shape == (len(df),):
            series = pd.Series(value, index=df.index)
            series.loc[~mask] = np.nan
            return series

        dtype = float if _is_numeric(value) else object
        series = pd.Series(np.nan, index=df.index, dtype=dtype)
        if mask.any():
            series.loc[mask] = value
        return series

    def _anchor_session_mask(
        self, df: pd.DataFrame, mask: pd.Series, *, window_anchor: str
    ) -> pd.Series:
        if not mask.any():
            return mask
        if "session_id" not in df.columns:
            return mask
        normalized = window_anchor.lower()
        if normalized not in {"start", "end"}:
            return mask
        valid = mask & df["session_id"].notna()
        if not valid.any():
            return mask
        anchored = pd.Series(False, index=df.index)
        subset = df.loc[valid, ["session_id", "ts"]]
        grouped = subset.groupby("session_id")
        if normalized == "start":
            indices = grouped["ts"].idxmin()
        else:
            indices = grouped["ts"].idxmax()
        anchored.loc[indices.values] = True
        return anchored

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

    @staticmethod
    def _coerce_minutes(value: Any) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            minutes = float(value)
            if not np.isfinite(minutes):
                return None
            minutes_int = int(round(minutes))
            return minutes_int if minutes_int > 0 else None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                numeric = float(text)
            except ValueError:
                matches = re.findall(r"(\d+)\s*([hHmM])", text)
                if not matches:
                    return None
                total = 0
                for amount, unit in matches:
                    qty = int(amount)
                    if unit.lower() == "h":
                        total += qty * 60
                    elif unit.lower() == "m":
                        total += qty
                return total if total > 0 else None
            else:
                if not np.isfinite(numeric):
                    return None
                minutes_int = int(round(numeric))
                return minutes_int if minutes_int > 0 else None

        return None

    @staticmethod
    def _combine_minutes(hours: Any, minutes: Any) -> Optional[int]:
        try:
            total = int(hours or 0) * 60 + int(minutes or 0)
        except (TypeError, ValueError):
            return None
        return total if total > 0 else None

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

        if pattern_type in self._MOMENTUM_TYPES:
            return self._dispatch_momentum(df, pattern, key, pattern_type)

        return []

    def _combine_gate_columns(self, df: pd.DataFrame, gate_columns: Sequence[str]) -> None:
        """Compute any-pattern-active flags with optional GPU acceleration."""

        if not gate_columns:
            df["any_pattern_active"] = 0
            return

        if self.use_gpu and GPU_AVAILABLE:
            gate_backend, xp = to_backend_array(
                df[gate_columns], use_gpu=True, device_id=self.gpu_device_id
            )
            any_active_backend = xp.any(gate_backend, axis=1)
            any_active = to_cpu(any_active_backend)
        else:
            any_active = df[gate_columns].any(axis=1)

        df["any_pattern_active"] = np.asarray(any_active, dtype=np.int8)

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
        if pattern_type == "weekend_hedging":
            return self._extract_weekend_hedging(df, pattern, payload, key, months)

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

    def _dispatch_momentum(
        self,
        df: pd.DataFrame,
        pattern: Mapping[str, Any],
        key: Optional[str],
        pattern_type: str,
    ) -> List[str]:
        payload = pattern.get("pattern_payload", {}) or {}
        metadata = pattern.get("metadata", {}) or {}

        weekday_norm, base_suffix = self._momentum_base_suffix(
            payload, metadata, pattern_type
        )

        if weekday_norm is None:
            mask = pd.Series(True, index=df.index, dtype=bool)
        else:
            mask = df["weekday_lower"] == weekday_norm

        months = self._resolve_months(key, pattern)
        mask &= self._months_mask(df, months)

        session_start = payload.get("session_start") or metadata.get("session_start")
        session_end = payload.get("session_end") or metadata.get("session_end")
        session_tz = metadata.get("session_tz") or payload.get("session_tz")
        window_anchor = str(payload.get("window_anchor") or metadata.get("window_anchor") or "session").lower()
        session_tz = metadata.get("session_tz")

        momentum_type = payload.get("momentum_type") or metadata.get("momentum_type") or "momentum"

        def _resolve_window_minutes() -> Optional[int]:
            base_candidates = (
                payload.get("window_minutes"),
                metadata.get("window_minutes"),
            )
            opening_candidates = (
                payload.get("opening_window_minutes"),
                metadata.get("opening_window_minutes"),
            )
            closing_candidates = (
                payload.get("closing_window_minutes"),
                metadata.get("closing_window_minutes"),
            )
            session_candidates = (
                metadata.get("sess_start_hrs"),
                metadata.get("sess_start_minutes"),
            )
            closing_session_candidates = (
                metadata.get("sess_end_hrs"),
                metadata.get("sess_end_minutes"),
            )

            candidate_pool: List[Any] = list(base_candidates)
            if momentum_type == "opening_momentum":
                candidate_pool.extend(opening_candidates)
                if any(session_candidates):
                    candidate_pool.append(
                        self._combine_minutes(
                            metadata.get("sess_start_hrs"), metadata.get("sess_start_minutes")
                        )
                    )
            elif momentum_type == "closing_momentum":
                candidate_pool.extend(closing_candidates)
                if any(closing_session_candidates):
                    candidate_pool.append(
                        self._combine_minutes(
                            metadata.get("sess_end_hrs"), metadata.get("sess_end_minutes")
                        )
                    )
            else:
                candidate_pool.extend(opening_candidates)
                candidate_pool.extend(closing_candidates)

            candidate_pool.extend(
                (
                    metadata.get("period_length_min"),
                    payload.get("period_length_min"),
                    metadata.get("period_length"),
                    payload.get("period_length"),
                )
            )

            for candidate in candidate_pool:
                minutes_val = self._coerce_minutes(candidate)
                if minutes_val is not None:
                    return minutes_val
            return None

        window_minutes_int = _resolve_window_minutes()

        start_hms, _ = self._session_clock_strings(session_start, session_tz)
        end_hms, _ = self._session_clock_strings(session_end, session_tz)

        lower_bound = start_hms
        upper_bound = end_hms

        if window_anchor == "start" and start_hms is not None:
            upper_bound = self._shift_time_str(start_hms, window_minutes_int, forward=True)
        elif window_anchor == "end" and end_hms is not None:
            lower_bound = self._shift_time_str(end_hms, window_minutes_int, forward=False)

        if lower_bound is not None and upper_bound is not None:
            if lower_bound <= upper_bound:
                mask &= (df["clock_time"] >= lower_bound) & (df["clock_time"] <= upper_bound)
            else:
                mask &= (df["clock_time"] >= lower_bound) | (df["clock_time"] <= upper_bound)
        else:
            if lower_bound is not None:
                mask &= df["clock_time"] >= lower_bound
            if upper_bound is not None:
                mask &= df["clock_time"] <= upper_bound

        mask = self._anchor_session_mask(df, mask, window_anchor=window_anchor)

        momentum_type = payload.get("momentum_type") or metadata.get("momentum_type") or "momentum"
        session_key = payload.get("session_key") or metadata.get("session_key")
        session_index = payload.get("session_index") or metadata.get("session_index")
        bias = payload.get("bias") or metadata.get("bias")
        if bias is None:
            mean_value = payload.get("mean")
            try:
                mean_float = float(mean_value)
            except (TypeError, ValueError):
                mean_float = 0.0
            if np.isfinite(mean_float):
                if mean_float > 0:
                    bias = "long"
                elif mean_float < 0:
                    bias = "short"
                else:
                    bias = "neutral"
            else:
                bias = "neutral"

        base = self._feature_base_name(key, base_suffix)

        period_minutes = None
        for candidate in (
            payload.get("period_length_min"),
            metadata.get("period_length_min"),
            metadata.get("period_length"),
            payload.get("period_length"),
        ):
            minutes_val = self._coerce_minutes(candidate)
            if minutes_val is not None:
                period_minutes = minutes_val
                break

        meta_params = metadata.get("momentum_params")
        if not isinstance(meta_params, Mapping):
            meta_params = {}
        st_momentum_days = (
            metadata.get("st_momentum_days")
            or payload.get("st_momentum_days")
            or meta_params.get("st_momentum_days")
        )

        sidecars: Dict[str, Any] = {
            "weekday": weekday_norm,
            "momentum_type": momentum_type,
            "bias": bias,
            "session_key": session_key,
            "session_index": session_index,
            "session_start": start_hms,
            "session_end": end_hms,
            "session_tz": session_tz or self.tz,
            "window_anchor": window_anchor,
            "window_minutes": window_minutes_int,
            "period_length_min": period_minutes,
            "st_momentum_days": st_momentum_days,
            "t_stat": payload.get("t_stat") or metadata.get("t_stat"),
            "mean": payload.get("mean"),
            "sharpe": payload.get("sharpe"),
            "positive_pct": payload.get("positive_pct"),
            "strength": pattern.get("strength") or payload.get("strength"),
            "correlation": payload.get("correlation") or metadata.get("correlation") or pattern.get("correlation"),
        }

        return self._add_feature(df, base, mask, sidecars)

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
        mask = self._anchor_session_mask(df, mask, window_anchor="end")

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

    def _extract_weekend_hedging(
        self,
        df: pd.DataFrame,
        pattern: Mapping[str, Any],
        payload: Mapping[str, Any],
        key: Optional[str],
        months: Optional[List[int]],
    ) -> List[str]:
        spec = self._weekend_pattern_spec(pattern, payload)
        if spec is None:
            return []

        base = self._feature_base_name(key, "weekend_hedging_friday")
        gate_mask, weekday_col = self._map_weekend_hedging_flags(
            df, spec, base_name=base, months=months
        )

        sidecars = {
            "gate_weekday": spec.gate_day,
            "target_weekday": spec.target_day,
            "gate_time": spec.gate_time or "session_close",
            "n": payload.get("n"),
            "corr": payload.get("corr_Fri_Mon"),
            "p": payload.get("p_value"),
            "mean_pos": payload.get("mean_Mon_given_Fri_pos"),
            "mean_neg": payload.get("mean_Mon_given_Fri_neg"),
            "bias": payload.get("bias"),
        }
        created = self._add_feature(df, base, gate_mask, sidecars)
        gate_col = created[0] if created else None
        self._register_pattern_features(
            pattern, gate_column=gate_col, weekday_column=weekday_col
        )
        if gate_col:
            self._validate_weekend_flags(
                df,
                gate_column=gate_col,
                weekday_column=weekday_col,
                spec=spec,
                months=months,
            )
        return created

    def _map_weekend_hedging_flags(
        self,
        df: pd.DataFrame,
        spec: WeekendPatternSpec,
        *,
        base_name: str,
        months: Optional[List[int]],
    ) -> Tuple[pd.Series, str]:
        ts_local = pd.to_datetime(df["ts"]).dt.tz_convert(self.tz)
        hhmm = ts_local.dt.strftime("%H:%M")
        month_mask = self._months_mask(df, months)

        weekday_series = df["weekday_lower"]
        if spec.gate_time:
            gate_mask = (
                (weekday_series == spec.gate_day)
                & (hhmm == spec.gate_time)
                & month_mask
            )
        else:
            candidates = (weekday_series == spec.gate_day) & month_mask
            gate_mask = self._anchor_session_mask(
                df, candidates.astype(bool), window_anchor="end"
            )

        weekday_mask = (weekday_series == spec.target_day) & month_mask

        weekday_col = f"{base_name}_weekday"
        df[weekday_col] = weekday_mask.astype(np.int8)
        return gate_mask, weekday_col

    def _register_pattern_features(
        self,
        pattern: Mapping[str, Any],
        *,
        gate_column: Optional[str],
        weekday_column: Optional[str],
    ) -> None:
        if not isinstance(pattern, MutableMapping):
            return
        features = pattern.setdefault("features", {})
        if not isinstance(features, MutableMapping):
            features = {}
            pattern["features"] = features
        if gate_column:
            features["pattern_gate_col"] = gate_column
        if weekday_column:
            features["pattern_weekday_col"] = weekday_column

    def _validate_weekend_flags(
        self,
        df: pd.DataFrame,
        *,
        gate_column: str,
        weekday_column: str,
        spec: WeekendPatternSpec,
        months: Optional[List[int]],
    ) -> None:
        if gate_column not in df.columns or weekday_column not in df.columns:
            raise KeyError("Weekend hedging gate/weekday columns missing from DataFrame")

        if "session_id" not in df.columns:
            raise ValueError("Weekend hedging validation requires a session_id column")

        gate_rows = df.loc[df[gate_column] == 1]
        if gate_rows.empty:
            raise ValueError("Weekend hedging gate produced no Friday rows at the gate time")

        if not (gate_rows["weekday_lower"] == spec.gate_day).all():
            raise ValueError("Weekend hedging gate emitted rows outside the configured gate weekday")

        duplicates = gate_rows["session_id"].value_counts()
        if (duplicates > 1).any():
            raise ValueError("Weekend hedging gate must emit exactly one row per Friday session")

        target_mask = (df["weekday_lower"] == spec.target_day) & self._months_mask(df, months)
        weekday_rows = df.loc[target_mask]
        if weekday_rows.empty:
            raise ValueError("Weekend hedging weekday column produced no Monday rows in the active months")

        weekday_values = weekday_rows[weekday_column]
        if weekday_values.isna().any() or not (weekday_values == 1).all():
            raise ValueError("Weekend hedging weekday column must flag every Monday row with 1s")

    def _weekend_pattern_spec(
        self, pattern: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> Optional[WeekendPatternSpec]:
        metadata = pattern.get("metadata") or {}
        best_weekday = (
            metadata.get("best_weekday")
            or payload.get("weekday")
            or metadata.get("weekday")
            or pattern.get("weekday")
        )
        weekday_pair = self._split_weekday_sequence(best_weekday)
        if weekday_pair is None:
            return None
        gate_day, target_day = weekday_pair

        gate_time = self._normalize_hhmm(
            metadata.get("gate_time_hhmm") or payload.get("gate_time_hhmm")
        )

        return WeekendPatternSpec(
            gate_day=gate_day,
            target_day=target_day,
            gate_time=gate_time,
        )

    def _split_weekday_sequence(self, value: Any) -> Optional[Tuple[str, str]]:
        if value is None:
            return None
        parts: List[str] = []
        if isinstance(value, (list, tuple)):
            parts = [str(item) for item in value if item is not None]
        else:
            text = str(value)
            if "->" in text:
                parts = [seg.strip() for seg in text.split("->") if seg.strip()]
            elif ">" in text:
                parts = [seg.strip() for seg in text.split(">") if seg.strip()]
            elif "to" in text.lower():
                parts = [seg.strip() for seg in re.split(r"to", text, flags=re.IGNORECASE) if seg.strip()]
            else:
                parts = [text]

        if len(parts) == 1:
            first = self._normalize_weekday(parts[0])
            return (first, first) if first else None
        if len(parts) >= 2:
            gate_day = self._normalize_weekday(parts[0])
            target_day = self._normalize_weekday(parts[1])
            if gate_day and target_day:
                return gate_day, target_day
        return None

    @staticmethod
    def _normalize_hhmm(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, time_cls):
            return value.strftime("%H:%M")
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = pd.to_datetime(text).time()
        except Exception:
            return None
        return parsed.strftime("%H:%M")

    def _normalize_hhmm_list(self, values: Any) -> List[str]:
        if values is None:
            return []
        if isinstance(values, (str, time_cls)):
            normalized = self._normalize_hhmm(values)
            return [normalized] if normalized else []
        result: List[str] = []
        for value in values:
            normalized = self._normalize_hhmm(value)
            if normalized:
                result.append(normalized)
        return sorted(dict.fromkeys(result))

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
        mask = self._anchor_session_mask(df, mask, window_anchor="end")

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
        mask = self._anchor_session_mask(df, mask, window_anchor="end")

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
        hms = parsed.strftime("%H:%M:%S")
        return hms, parsed.strftime("%H:%M:%S.%f")

    def _session_clock_strings(
        self, value: Any, source_tz: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        hms, hmsf = self._time_to_strings(value)
        if hms is None or not source_tz or source_tz == self.tz:
            return hms, hmsf
        try:
            base = pd.Timestamp(f"2000-01-01 {hms}", tz=source_tz)
            converted = base.tz_convert(self.tz)
        except Exception:
            return hms, hmsf
        return converted.strftime("%H:%M:%S"), converted.strftime("%H:%M:%S.%f")

    @staticmethod
    def _combine_minutes(hours: Any, minutes: Any) -> Optional[int]:
        try:
            total = int(hours or 0) * 60 + int(minutes or 0)
        except (TypeError, ValueError):
            return None
        return total if total > 0 else None

    def _convert_session_clock(self, clock: Optional[str], source_tz: Optional[str]) -> Optional[str]:
        if clock is None or not source_tz or source_tz == self.tz:
            return clock
        try:
            base = pd.Timestamp(f"2000-01-01 {clock}", tz=source_tz)
        except Exception:
            return clock
        try:
            converted = base.tz_convert(self.tz)
        except Exception:
            return clock
        return converted.time().strftime("%H:%M:%S")

    @staticmethod
    def _shift_time_str(base: Optional[str], minutes: Optional[int], *, forward: bool) -> Optional[str]:
        if base is None or minutes in (None, 0):
            return base
        try:
            timestamp = pd.Timestamp(f"2000-01-01 {base}")
        except Exception:
            return base
        delta = pd.Timedelta(minutes=int(minutes))
        shifted = timestamp + delta if forward else timestamp - delta
        return shifted.time().strftime("%H:%M:%S")

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
        weekend_exit_policy: str = "last",
    ) -> None:
        if time_match not in {"auto", "second", "microsecond"}:
            raise HorizonInputError(
                "time_match must be one of {'auto', 'second', 'microsecond'}"
            )

        if nan_policy not in {"drop", "zero", "ffill"}:
            raise HorizonInputError("nan_policy must be one of {'drop', 'zero', 'ffill'}")

        if weekend_exit_policy not in {"first", "last", "average"}:
            raise HorizonInputError(
                "weekend_exit_policy must be one of {'first', 'last', 'average'}"
            )

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
        self.weekend_exit_policy = weekend_exit_policy

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_warn(self, message: str, **context: Any) -> None:
        if self.log is not None and hasattr(self.log, "warning"):
            if context:
                self.log.warning("[HorizonMapper] %s | %s", message, context)
            else:
                self.log.warning("[HorizonMapper] %s", message)

    @staticmethod
    def _collect_metadata_fields(
        pattern: Mapping[str, Any], fields: SequenceABC[str]
    ) -> Dict[str, Any]:
        payload = pattern.get("pattern_payload", {}) or {}
        metadata = pattern.get("metadata", {}) or {}
        collected: Dict[str, Any] = {}
        for field in fields:
            if field in payload:
                collected[field] = payload.get(field)
            elif field in metadata:
                collected[field] = metadata.get(field)
            else:
                collected[field] = pattern.get(field)
        return collected

    @staticmethod
    def _pattern_feature_column(pattern: Mapping[str, Any], key: str) -> Optional[str]:
        if not isinstance(pattern, Mapping):
            return None
        features = pattern.get("features")
        if isinstance(features, Mapping):
            value = features.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _normalise_metadata_series(value: Any, index: pd.Index) -> pd.Series:
        length = len(index)
        if length == 0:
            return pd.Series(index=index, dtype=object)

        if isinstance(value, pd.Series):
            return value.reindex(index)

        if isinstance(value, pd.Index):
            value = value.tolist()

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            repeated = list(value) if isinstance(value, list) else tuple(value)
            return pd.Series([repeated] * length, index=index, dtype=object)

        if value is None:
            return pd.Series([np.nan] * length, index=index, dtype=float)

        if is_scalar(value):
            dtype = float if _is_numeric(value) else object
            return pd.Series([value] * length, index=index, dtype=dtype)

        return pd.Series([value] * length, index=index, dtype=object)

    @staticmethod
    def _coerce_minutes(value: Any) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            minutes = float(value)
            if not np.isfinite(minutes):
                return None
            minutes_int = int(round(minutes))
            return minutes_int if minutes_int > 0 else None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                numeric = float(text)
            except ValueError:
                matches = re.findall(r"(\d+)\s*([hHmM])", text)
                if not matches:
                    return None
                total = 0
                for amount, unit in matches:
                    unit_lower = unit.lower()
                    qty = int(amount)
                    if unit_lower == "h":
                        total += qty * 60
                    elif unit_lower == "m":
                        total += qty
                return total if total > 0 else None
            else:
                if not np.isfinite(numeric):
                    return None
                minutes_int = int(round(numeric))
                return minutes_int if minutes_int > 0 else None

        return None

    @staticmethod
    def _combine_minutes(hours: Any, minutes: Any) -> Optional[int]:
        try:
            total = int(hours or 0) * 60 + int(minutes or 0)
        except (TypeError, ValueError):
            return None
        return total if total > 0 else None

    @classmethod
    def _extract_period_minutes(cls, pattern: Mapping[str, Any]) -> Optional[int]:
        metadata = pattern.get("metadata") or {}
        payload = pattern.get("pattern_payload") or {}

        for source in (
            metadata.get("period_length_min"),
            payload.get("period_length_min"),
            metadata.get("period_length"),
            payload.get("period_length"),
        ):
            minutes = cls._coerce_minutes(source)
            if minutes is not None:
                return minutes

        return None

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
        if pattern_type == "weekend_hedging":
            return HorizonSpec(name="next_monday_oc")
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

    def _next_monday_open_to_close(self, df: pd.DataFrame) -> pd.Series:
        info = self._session_level_info(df)
        friday_mask = info["weekday"] == 4
        monday_mask = info["next_weekday"] == 0
        valid = friday_mask & monday_mask

        monday_returns = pd.Series(np.nan, index=info["session_id"], dtype=float)
        monday_returns.loc[info.loc[valid, "session_id"]] = info.loc[valid, "next_oc"].astype(float)
        return df["session_id"].map(monday_returns)

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

    def _session_level_info(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby("session_id")
        first_ts = grouped["ts"].first().dt.tz_convert(self.tz)
        oc_returns = self._same_day_open_to_close(df).groupby(df["session_id"]).first()
        ordered = first_ts.sort_values()
        oc_ordered = oc_returns.reindex(ordered.index)

        info = pd.DataFrame({
            "session_id": ordered.index,
            "weekday": ordered.dt.weekday,
            "oc_return": oc_ordered.astype(float),
        })
        info["next_weekday"] = info["weekday"].shift(-1)
        info["next_oc"] = info["oc_return"].shift(-1)
        info["next_session_id"] = info["session_id"].shift(-1)
        return info

    def _weekend_hedging_returns(
        self, df: pd.DataFrame, pattern: Mapping[str, Any]
    ) -> Tuple[pd.Series, pd.Series]:
        info = self._session_level_info(df)
        friday_mask = info["weekday"] == 4
        has_future = info["next_session_id"].notna()
        valid = friday_mask & has_future

        friday_returns = pd.Series(np.nan, index=info["session_id"], dtype=float)
        friday_returns.loc[info.loc[valid, "session_id"]] = info.loc[valid, "oc_return"].astype(float)

        default_monday = pd.Series(np.nan, index=info["session_id"], dtype=float)
        default_monday.loc[info.loc[valid, "session_id"]] = info.loc[valid, "next_oc"].astype(float)

        gate_col = self._pattern_feature_column(pattern, "pattern_gate_col")
        if gate_col and gate_col in df.columns:
            gate_rows = df[gate_col] == 1
            x_series = pd.Series(np.nan, index=df.index, dtype=float)
            if gate_rows.any():
                x_series.loc[gate_rows] = df.loc[gate_rows, "session_id"].map(friday_returns)
        else:
            x_series = df["session_id"].map(friday_returns)
        custom_y = self._weekend_returns_from_weekday_flags(
            df, pattern, info, valid, fallback_to_session_close=True
        )
        if custom_y is None:
            y_series = df["session_id"].map(default_monday)
        else:
            y_series = custom_y
        return x_series, y_series

    def _weekend_returns_from_weekday_flags(
        self,
        df: pd.DataFrame,
        pattern: Mapping[str, Any],
        info: pd.DataFrame,
        valid_mask: pd.Series,
        *,
        fallback_to_session_close: bool = False,
    ) -> Optional[pd.Series]:
        gate_col = self._pattern_feature_column(pattern, "pattern_gate_col")
        weekday_col = self._pattern_feature_column(pattern, "pattern_weekday_col")
        if not gate_col or not weekday_col:
            return None
        if gate_col not in df.columns or weekday_col not in df.columns:
            return None

        gate_series = df[gate_col].fillna(0).astype(int) == 1
        weekday_series = df[weekday_col].fillna(0).astype(int) == 1
        if not gate_series.any() or not weekday_series.any():
            return None

        weekday_rows = df.loc[weekday_series, ["session_id", "ts", "close"]].copy()
        if weekday_rows.empty:
            return None

        weekday_rows["close"] = weekday_rows["close"].astype(float)
        close_series = df["close"].astype(float)
        grouped = weekday_rows.groupby("session_id")
        policy = self.weekend_exit_policy
        exit_indices: Dict[Any, Any]
        grouped_indices: Dict[Any, List[Any]]

        if policy == "average":
            grouped_indices = {key: list(idxs) for key, idxs in grouped.groups.items()}
            exit_indices = {}
        elif policy == "first":
            grouped_indices = {}
            exit_indices = grouped["ts"].idxmin().to_dict()
        else:  # last
            grouped_indices = {}
            exit_indices = grouped["ts"].idxmax().to_dict()

        info_indexed = info.set_index("session_id")
        next_session_map = info_indexed["next_session_id"].to_dict()
        valid_sessions = set(info.loc[valid_mask, "session_id"])

        session_last_idx = (
            df.loc[df["session_id"].notna(), ["session_id", "ts"]]
            .groupby("session_id")["ts"]
            .idxmax()
        )
        session_last_prices = close_series.loc[session_last_idx.values].astype(float)
        session_last_prices.index = session_last_idx.index
        session_last_map = session_last_prices.to_dict()

        returns_y = pd.Series(np.nan, index=df.index, dtype=float)
        for idx in gate_series[gate_series].index:
            session_id = df.at[idx, "session_id"]
            if session_id not in valid_sessions:
                continue
            next_session = next_session_map.get(session_id)
            if not next_session:
                continue
            entry_price = close_series.loc[idx]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            if policy == "average":
                target_indices = grouped_indices.get(next_session)
                if not target_indices:
                    continue
                target_prices = close_series.loc[target_indices].astype(float)
                valid_prices = target_prices[target_prices > 0]
                if valid_prices.empty:
                    continue
                log_returns = np.log(valid_prices / float(entry_price))
                returns_y.loc[idx] = float(np.nanmean(log_returns.values))
            else:
                exit_idx = exit_indices.get(next_session)
                if exit_idx is not None and exit_idx in close_series.index:
                    exit_price = close_series.loc[exit_idx]
                else:
                    exit_price = session_last_map.get(next_session) if fallback_to_session_close else None
                if exit_price is None or pd.isna(exit_price) or exit_price <= 0:
                    continue
                returns_y.loc[idx] = float(np.log(float(exit_price) / float(entry_price)))

        return returns_y if returns_y.notna().any() else None

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

    def _opening_period_return_series(
        self, df: pd.DataFrame, minutes: int, session_cache: MutableMapping[str, Any]
    ) -> pd.Series:
        """Calculate returns for the opening period (first N minutes) of each session.

        Returns log returns from session open to close at t + N minutes from session start.
        """
        cache_key = f"opening_period_{minutes}"
        cached = session_cache.get(cache_key)
        if cached is not None:
            return cached

        if minutes <= 0:
            raise ValueError("opening period minutes must be positive")

        ts = pd.to_datetime(df["ts"]).dt.tz_convert(self.tz)
        close = pd.Series(df["close"].values, index=ts).astype(float)
        close = close.sort_index()

        # Get session open prices
        session_opens = df.groupby("session_id")["open"].first().astype(float)
        session_open_map = df["session_id"].map(session_opens)

        # Get session start times
        session_start_times = df.groupby("session_id")["ts"].first()
        session_start_map = pd.to_datetime(df["session_id"].map(session_start_times)).dt.tz_convert(self.tz)

        # Calculate target time (session_start + minutes)
        target_times = session_start_map + pd.Timedelta(minutes=minutes)

        # Get closing price at target time using reindex
        target_closes = close.reindex(target_times, method="nearest", tolerance=pd.Timedelta(minutes=5))
        target_closes.index = df.index

        # Calculate returns: log(close_at_target / session_open)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(target_closes / session_open_map)

        result = self._clean_return(raw, policy="drop")
        result.replace([np.inf, -np.inf], np.nan, inplace=True)

        session_cache[cache_key] = result
        return result

    def _closing_period_return_series(
        self, df: pd.DataFrame, minutes: int, session_cache: MutableMapping[str, Any]
    ) -> pd.Series:
        """Calculate returns for the closing period (last N minutes) of each session.

        Returns log returns from close at t - N minutes before session end to session close.
        """
        cache_key = f"closing_period_{minutes}"
        cached = session_cache.get(cache_key)
        if cached is not None:
            return cached

        if minutes <= 0:
            raise ValueError("closing period minutes must be positive")

        ts = pd.to_datetime(df["ts"]).dt.tz_convert(self.tz)
        close = pd.Series(df["close"].values, index=ts).astype(float)
        close = close.sort_index()

        # Get session close prices
        session_closes = df.groupby("session_id")["close"].last().astype(float)
        session_close_map = df["session_id"].map(session_closes)

        # Get session end times
        session_end_times = df.groupby("session_id")["ts"].last()
        session_end_map = pd.to_datetime(df["session_id"].map(session_end_times)).dt.tz_convert(self.tz)

        # Calculate target time (session_end - minutes)
        target_times = session_end_map - pd.Timedelta(minutes=minutes)

        # Get closing price at target time using reindex
        target_closes = close.reindex(target_times, method="nearest", tolerance=pd.Timedelta(minutes=5))
        target_closes.index = df.index

        # Calculate returns: log(session_close / close_at_target)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(session_close_map / target_closes)

        result = self._clean_return(raw, policy="drop")
        result.replace([np.inf, -np.inf], np.nan, inplace=True)

        session_cache[cache_key] = result
        return result

    def _same_day_return_series(
        self, df: pd.DataFrame, session_cache: MutableMapping[str, Any]
    ) -> pd.Series:
        cached = session_cache.get("same_day_series")
        if cached is None:
            cached = self._same_day_open_to_close(df)
            session_cache["same_day_series"] = cached
        return cached

    def _short_term_momentum_series(
        self,
        df: pd.DataFrame,
        window: int,
        *,
        trend_cache: MutableMapping[int, pd.Series],
        session_cache: MutableMapping[str, Any],
    ) -> pd.Series:
        if window <= 0:
            return pd.Series(np.nan, index=df.index, dtype=float)
        if window in trend_cache:
            return trend_cache[window]
        same_day = self._same_day_return_series(df, session_cache)
        session_returns = session_cache.get("session_returns")
        if session_returns is None:
            session_returns = same_day.groupby(df["session_id"]).first()
            session_cache["session_returns"] = session_returns
        trend = session_returns.shift(1).rolling(window=window, min_periods=window).sum()
        trend_series = df["session_id"].map(trend)
        trend_cache[window] = trend_series
        return trend_series

    def _extract_momentum_window_minutes(
        self, pattern: Mapping[str, Any], default_minutes: int
    ) -> int:
        metadata = pattern.get("metadata") or {}
        payload = pattern.get("pattern_payload") or {}
        momentum_type = (
            payload.get("momentum_type")
            or metadata.get("momentum_type")
            or "opening_momentum"
        )

        candidates: List[Any] = [
            payload.get("window_minutes"),
            metadata.get("window_minutes"),
        ]

        if momentum_type == "opening_momentum":
            candidates.extend(
                [
                    payload.get("opening_window_minutes"),
                    metadata.get("opening_window_minutes"),
                    self._combine_minutes(
                        metadata.get("sess_start_hrs"), metadata.get("sess_start_minutes")
                    ),
                ]
            )
        elif momentum_type == "closing_momentum":
            candidates.extend(
                [
                    payload.get("closing_window_minutes"),
                    metadata.get("closing_window_minutes"),
                    self._combine_minutes(
                        metadata.get("sess_end_hrs"), metadata.get("sess_end_minutes")
                    ),
                ]
            )
        else:
            candidates.extend(
                [
                    metadata.get("opening_window_minutes"),
                    metadata.get("closing_window_minutes"),
                ]
            )

        candidates.extend(
            [
                metadata.get("period_length_min"),
                payload.get("period_length_min"),
                metadata.get("period_length"),
                payload.get("period_length"),
            ]
        )

        for source in candidates:
            minutes = self._coerce_minutes(source)
            if minutes is not None:
                return minutes
        return max(1, int(default_minutes))

    def _extract_st_momentum_days(self, pattern: Mapping[str, Any]) -> Optional[int]:
        metadata = pattern.get("metadata") or {}
        payload = pattern.get("pattern_payload") or {}
        candidates = (
            metadata.get("st_momentum_days"),
            payload.get("st_momentum_days"),
            metadata.get("momentum_params", {}).get("st_momentum_days")
            if isinstance(metadata.get("momentum_params"), Mapping)
            else None,
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _momentum_xy_series(
        self,
        df: pd.DataFrame,
        pattern: Mapping[str, Any],
        *,
        default_intraday_minutes: int,
        forward_cache: MutableMapping[int, pd.Series],
        trend_cache: MutableMapping[int, pd.Series],
        session_cache: MutableMapping[str, Any],
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate X and Y returns series for momentum patterns.

        Pattern mappings (aligned with HistoricalScreener):
        - momentum_oc: opening_returns vs closing_returns
        - momentum_sc: rest_of_session_returns vs closing_returns (session spills into close)
        - momentum_so: opening_returns vs st_momentum
        - momentum_cc: closing_returns vs st_momentum
        """
        metadata = pattern.get("metadata") or {}
        payload = pattern.get("pattern_payload") or {}
        window_minutes = self._extract_momentum_window_minutes(
            pattern, default_intraday_minutes
        )
        momentum_type = (
            payload.get("momentum_type")
            or metadata.get("momentum_type")
            or "opening_momentum"
        )
        st_days = self._extract_st_momentum_days(pattern) or 3

        if window_minutes <= 0:
            window_minutes = default_intraday_minutes

        # Calculate period-specific returns
        opening_returns = self._opening_period_return_series(df, window_minutes, session_cache)
        closing_returns = self._closing_period_return_series(df, window_minutes, session_cache)
        full_session_returns = self._same_day_return_series(df, session_cache)
        st_momentum = self._short_term_momentum_series(
            df, st_days, trend_cache=trend_cache, session_cache=session_cache
        )

        # Map momentum_type to correct X and Y series
        if momentum_type == "opening_momentum":
            # momentum_oc: opening period predicts closing period
            returns_x = opening_returns
            returns_y = closing_returns
        elif momentum_type == "closing_momentum":
            # momentum_sc: rest of session spills into closing period
            rest_of_session = full_session_returns - closing_returns
            returns_x = rest_of_session
            returns_y = closing_returns
        elif momentum_type == "st_momentum":
            # momentum_so or momentum_cc: depends on pattern_type
            # For momentum_so: opening predicts st_momentum
            # For momentum_cc: closing predicts st_momentum
            pattern_type = pattern.get("pattern_type", "")
            if "momentum_so" in pattern_type:
                returns_x = opening_returns
            else:  # momentum_cc
                returns_x = closing_returns
            returns_y = st_momentum
        else:  # full_session or fallback
            returns_x = st_momentum
            returns_y = full_session_returns

        return returns_x, returns_y

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
    ) -> Iterable[Tuple[str, str, pd.Series, pd.Series, int, List[str], Mapping[str, Any]]]:
        if patterns is None:
            return

        if predictor_minutes <= 0:
            raise HorizonInputError("predictor_minutes must be positive")

        if weekly_x_policy not in {"mean", "prev_week"}:
            raise HorizonInputError("weekly_x_policy must be one of {'mean', 'prev_week'}")

        x_fw = self._forward_window_return_minutes(df, minutes=predictor_minutes)
        prev_week_series: Optional[pd.Series] = None
        horizon_cache: Dict[Tuple[str, Optional[int]], pd.Series] = {}
        momentum_forward_cache: Dict[int, pd.Series] = {}
        momentum_trend_cache: Dict[int, pd.Series] = {}
        session_return_cache: Dict[str, Any] = {}
        time_bearing_types = {"time_predictive_nextday", "time_predictive_nextweek", "orderflow_peak_pressure"}
        weekly_types = {"weekday_mean", "weekday_bias_intraday", "orderflow_weekly", "orderflow_week_of_month"}

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

            if pattern_type in ScreenerPipeline._MOMENTUM_TYPES:
                returns_x_series, returns_y = self._momentum_xy_series(
                    df,
                    pattern,
                    default_intraday_minutes=default_intraday_minutes,
                    forward_cache=momentum_forward_cache,
                    trend_cache=momentum_trend_cache,
                    session_cache=session_return_cache,
                )
            elif pattern_type == "weekend_hedging":
                returns_x_series, returns_y = self._weekend_hedging_returns(df, pattern)
            else:
                spec = self.pattern_horizon(
                    pattern_type, default_intraday_minutes=default_intraday_minutes
                )
                effective_delta = spec.delta_minutes
                if spec.name == "intraday_delta":
                    inferred_minutes = self._extract_period_minutes(pattern)
                    if inferred_minutes is not None:
                        effective_delta = inferred_minutes

                cache_key = (spec.name, effective_delta if spec.name == "intraday_delta" else None)
                if cache_key not in horizon_cache:
                    if spec.name == "same_day_oc":
                        horizon_cache[cache_key] = self._same_day_open_to_close(df)
                    elif spec.name == "next_day_cc":
                        horizon_cache[cache_key] = self._next_day_close_to_close(df)
                    elif spec.name == "next_week_cc":
                        horizon_cache[cache_key] = self._next_week_close_to_close(df, days=5)
                    elif spec.name == "intraday_delta":
                        horizon_cache[cache_key] = self._intraday_delta_minutes(
                            df,
                            minutes=(effective_delta or spec.delta_minutes or default_intraday_minutes),
                        )
                    elif spec.name == "next_monday_oc":
                        horizon_cache[cache_key] = self._next_monday_open_to_close(df)
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
            yield gate_col, pattern_type, returns_x_series, returns_y, side_hint, candidates, pattern

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
        ts_to_index: Dict[pd.Timestamp, List[Any]] = {}
        if "ts" in result.columns:
            ts_series = pd.to_datetime(result["ts"], errors="coerce")
            for idx, ts_value in zip(result.index, ts_series):
                if pd.isna(ts_value):
                    continue
                ts_to_index.setdefault(ts_value, []).append(idx)

        for gate_col, _, returns_x_series, _, _, _, _ in self._iter_pattern_returns(
            df,
            patterns,
            default_intraday_minutes=default_intraday_minutes,
            predictor_minutes=predictor_minutes,
            weekly_x_policy=weekly_x_policy,
            time_match=self.time_match,
            tolerance=self.asof_tolerance,
        ):
            signal_col = self._signal_column_name(gate_col)
            signal = pd.Series(np.nan, index=result.index, dtype=float)
            gate_mask = df[gate_col] == 1
            if gate_mask.any():
                aligned = returns_x_series.reindex(df.index)
                valid_mask = gate_mask & aligned.notna()
                if valid_mask.any():
                    ts_active = df.loc[valid_mask, "ts"].to_numpy()
                    values_active = aligned.loc[valid_mask].to_numpy()
                    for ts_value, value in zip(ts_active, values_active):
                        for label in ts_to_index.get(ts_value, []):
                            signal.loc[label] = value
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
        include_metadata: Optional[Iterable[str]] = None,
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
        include_metadata:
            Optional iterable of payload keys to copy onto the emitted decision rows.
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

        include_fields = list(dict.fromkeys(include_metadata or []))

        for (
            gate_col,
            pattern_type,
            returns_x_series,
            returns_y,
            side_hint,
            candidates,
            pattern,
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
            if include_fields:
                meta_values = self._collect_metadata_fields(pattern, include_fields)
                for field_name, field_value in meta_values.items():
                    subset[field_name] = self._normalise_metadata_series(
                        field_value, subset.index
                    )
            row_columns = ["ts_decision", "gate", "pattern_type", "returns_x", "returns_y", "side_hint", *include_fields]
            rows.append(subset[row_columns])

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
        subset = [
            col
            for col in [
                "ts_decision",
                "gate",
                "pattern_type",
                "returns_x",
                "returns_y",
                "side_hint",
            ]
            if col in result.columns
        ]
        if subset:
            result = result.drop_duplicates(subset=subset).reset_index(drop=True)
        else:
            result = result.drop_duplicates().reset_index(drop=True)
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

        explicit = self._pattern_feature_column(pattern, "pattern_gate_col")
        if explicit:
            if explicit in df.columns:
                return explicit, [explicit]
            candidates.append(explicit)

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
        elif pattern_type in ScreenerPipeline._MOMENTUM_TYPES:
            _, suffix = ScreenerPipeline._momentum_base_suffix(
                payload, metadata, pattern_type
            )
            candidates.append(f"{suffix}_gate")

        explicit = payload.get("gate") or metadata.get("gate")
        if isinstance(explicit, str) and explicit:
            normalized = explicit if explicit.endswith("_gate") else f"{explicit}_gate"
            candidates.append(normalized)

        return candidates

