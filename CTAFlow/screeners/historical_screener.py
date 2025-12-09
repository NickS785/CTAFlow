from ..utils.seasonal import last_year_predicts_this_year, intraday_lag_autocorr, abnormal_months, prewindow_feature, prewindow_predicts_month
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import calendar
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
from dataclasses import dataclass, field
from ..data import IntradayFileManager, DataClient, SyntheticSymbol, ResultsClient
from ..config import DLY_DATA_PATH, INTRADAY_ADB_PATH
from ..utils.session import filter_session_bars
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import logging
from ..features.regime_classification import (
    BaseRegimeClassifier,
    RegimeSpecificationLike,
    build_regime_classifier,
)
from ..strategy.gpu_acceleration import GPU_AVAILABLE, GPU_DEVICE_COUNT
from .calendar_effects import (
    CalendarEffectParams,
    run_calendar_edge_tests,
    run_calendar_lead_lag_tests,
)


@dataclass
class ScreenParams:
    """
    Parameters for configuring individual screening operations.

    Attributes
    ----------
    screen_type : str
        Type of screen: 'momentum' or 'seasonality'
    name : Optional[str]
        Custom name for this screen (used as dict key in composite outputs)
        If None, auto-generates from screen_type and season/months
    season : Optional[str]
        Season to analyze: 'winter', 'spring', 'summer', 'fall'
        Maps to months: winter=[12,1,2], spring=[3,4,5], summer=[6,7,8], fall=[9,10,11]
    months : Optional[List[int]]
        Specific months to analyze (1-12). Overridden by season if both specified.

    # Momentum screen parameters
    session_starts : Optional[List[Union[str, time]]]
        Session start times for momentum screen (e.g., ["02:30", "08:30"])
    session_ends : Optional[List[Union[str, time]]]
        Session end times for momentum screen (e.g., ["10:30", "13:30"])
    st_momentum_days : int
        Number of days for short-term momentum calculation (default: 3)
    sess_start_hrs : int
        Hours from session start to measure opening momentum (default: 1)
    sess_start_minutes : int
        Minutes from session start to measure opening momentum (default: 30)
    sess_end_hrs : Optional[int]
        Hours from session end for closing momentum (defaults to sess_start_hrs)
    sess_end_minutes : Optional[int]
        Minutes from session end for closing momentum (defaults to sess_start_minutes)
    test_vol : bool
        Whether to test volume patterns in momentum screen (default: True)

    # Seasonality screen parameters
    target_times : Optional[List[Union[str, time]]]
        Times to analyze for seasonality screen (e.g., ["09:30", "14:00"])
    period_length : Optional[Union[int, timedelta]]
        Length of period to aggregate for seasonality (default: None)
    dayofweek_screen : bool
        Whether to analyze day-of-week patterns (default: True)
    seasonality_session_start : Union[str, time]
        Local session start time applied before seasonality calculations (default "00:00").
    seasonality_session_end : Union[str, time]
        Local session end time applied before seasonality calculations (default "23:59:59").
    tz : str
        Olson timezone used to localize intraday data before applying session or
        time-of-day filters (default "America/Chicago").

    Examples
    --------
    # Momentum screen for winter months
    >>> winter_momentum = ScreenParams(
    ...     screen_type='momentum',
    ...     name='winter_momentum',
    ...     season='winter',
    ...     session_starts=["02:30", "08:30"],
    ...     session_ends=["10:30", "13:30"]
    ... )

    # Seasonality screen for spring
    >>> spring_seasonality = ScreenParams(
    ...     screen_type='seasonality',
    ...     name='spring_tod',
    ...     season='spring',
    ...     target_times=["09:30", "14:00"],
    ...     dayofweek_screen=True
    ... )

    # Custom months momentum screen
    >>> q1_momentum = ScreenParams(
    ...     screen_type='momentum',
    ...     months=[1, 2, 3],
    ...     session_starts=["09:30"],
    ...     session_ends=["16:00"]
    ... )
    """
    screen_type: str  # 'momentum' or 'seasonality'
    name: Optional[str] = None
    season: Optional[str] = None
    months: Optional[List[int]] = None

    # Regime filtering parameters
    use_regime_filtering: bool = False
    regime_col: Optional[str] = None
    target_regimes: Optional[List[int]] = None
    regime_settings: Optional[RegimeSpecificationLike] = None

    # Momentum screen parameters
    session_starts: Optional[List[Union[str, time]]] = None
    session_ends: Optional[List[Union[str, time]]] = None
    st_momentum_days: int = 3
    sess_start_hrs: int = 1
    sess_start_minutes: int = 30
    sess_end_hrs: Optional[int] = None
    sess_end_minutes: Optional[int] = None
    test_vol: bool = True

    # Seasonality screen parameters
    target_times: Optional[List[Union[str, time]]] = None
    period_length: Optional[Union[int, timedelta]] = None
    dayofweek_screen: bool = True
    seasonality_session_start: Union[str, time] = "00:00"
    seasonality_session_end: Union[str, time] = "23:59:59"
    tz: str = "America/Chicago"

    # Normalised derivatives (populated during validation)
    target_times_hhmm: Optional[List[str]] = field(default=None, init=False)

    def __post_init__(self):
        """Validate parameters and auto-generate name if not provided."""
        # Validate screen_type
        valid_types = ['momentum', 'seasonality']
        if self.screen_type.lower() not in valid_types:
            raise ValueError(f"screen_type must be one of {valid_types}, got '{self.screen_type}'")

        self.screen_type = self.screen_type.lower()

        # Validate screen-specific required parameters
        if self.screen_type == 'momentum':
            if self.session_starts is None or self.session_ends is None:
                raise ValueError("Momentum screens require session_starts and session_ends")
        elif self.screen_type == 'seasonality':
            if self.target_times is None:
                raise ValueError("Seasonality screens require target_times")

        # Validate regime filtering parameters
        if self.use_regime_filtering:
            if not self.regime_col:
                raise ValueError("regime_col must be provided when use_regime_filtering=True")
            if not self.target_regimes:
                raise ValueError("target_regimes must be provided when use_regime_filtering=True")

        if self.target_regimes is not None:
            try:
                regimes = sorted({int(value) for value in self.target_regimes})
            except (TypeError, ValueError) as exc:
                raise ValueError("target_regimes must contain integers") from exc
            self.target_regimes = regimes

        if self.target_times:
            normalised: List[str] = []
            for raw in self.target_times:
                if isinstance(raw, time):
                    clock = raw
                else:
                    try:
                        clock = pd.to_datetime(str(raw)).time()
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid target time value: {raw!r}") from exc
                normalised.append(clock.strftime("%H:%M"))
            self.target_times_hhmm = sorted(dict.fromkeys(normalised))

        # Auto-generate name if not provided
        if self.name is None:
            if self.season:
                self.name = f"{self.season}_{self.screen_type}"
            elif self.months:
                month_str = "_".join(map(str, self.months))
                self.name = f"months_{month_str}_{self.screen_type}"
            else:
                self.name = f"all_{self.screen_type}"

    @property
    def sess_end_mins(self) -> Optional[int]:
        """Backwards-compatible alias for ``sess_end_minutes``."""

        return self.sess_end_minutes

    @sess_end_mins.setter
    def sess_end_mins(self, value: Optional[int]) -> None:
        self.sess_end_minutes = value


class HistoricalScreener:
    """Screener created to find seasonal and momentum patterns in intraday and daily data"""

    def __init__(
        self,
        ticker_data: Dict[str, Union[pd.DataFrame, SyntheticSymbol]],
        file_mgr: IntradayFileManager = None,
        *,
        results_client: Optional[ResultsClient] = None,
        auto_write_results: bool = False,
        verbose: bool = True,
        use_gpu: bool = False,
        gpu_device_id: int = 0,
    ):
        """
        Initialize HistoricalScreener with ticker data.

        Parameters:
        -----------
        ticker_data : Dict[str, Union[pd.DataFrame, SyntheticSymbol]]
            Dictionary mapping ticker symbols to either DataFrames or SyntheticSymbol objects
        file_mgr : IntradayFileManager, optional
            File manager for loading additional data
        verbose : bool, optional
            Enable verbose logging (default: True)
        """
        self.data = ticker_data
        self.tickers = list(ticker_data.keys())
        self.mgr = file_mgr or IntradayFileManager(data_path=DLY_DATA_PATH, arctic_uri=INTRADAY_ADB_PATH)
        self.method = "arctic"
        self.verbose = verbose
        self.use_gpu = bool(use_gpu and GPU_AVAILABLE)
        self.gpu_device_id = int(gpu_device_id)
        self.results_client = results_client
        self.auto_write_results = bool(auto_write_results and results_client is not None)
        self._regime_cache: Dict[Tuple[str, str, str], Dict[str, pd.Series]] = {}

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.HistoricalScreener")
        if verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Track which tickers are synthetic spreads
        self.synthetic_tickers = {
            ticker: isinstance(data, SyntheticSymbol)
            for ticker, data in ticker_data.items()
        }

        if self.verbose:
            self.logger.info(f"Initialized HistoricalScreener with {len(self.tickers)} tickers")
            synth_count = sum(self.synthetic_tickers.values())
            if synth_count > 0:
                self.logger.info(f"  - {synth_count} synthetic spreads, {len(self.tickers) - synth_count} regular tickers")
            if self.results_client and self.auto_write_results:
                self.logger.info("  - Automatic result persistence enabled")
            if use_gpu and not self.use_gpu:
                self.logger.info("  - GPU scan requested but unavailable; defaulting to CPU")
            elif self.use_gpu:
                self.logger.info(
                    "  - Concurrent GPU scans enabled (device %s, %s GPUs detected)",
                    self.gpu_device_id,
                    GPU_DEVICE_COUNT,
                )


    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _resolve_workers(self, max_workers: Optional[int]) -> int:
        """Choose an executor pool size suited to the current hardware."""

        if max_workers is not None and max_workers > 0:
            return int(max_workers)

        if self.use_gpu and GPU_AVAILABLE and GPU_DEVICE_COUNT > 0:
            return GPU_DEVICE_COUNT

        return 1

    @staticmethod
    def _serialize_period_length(value: Optional[Union[int, float, str, timedelta]]) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, (int, float)):
            return str(value)

        try:
            delta = pd.to_timedelta(value)
        except Exception:
            return str(value)

        total_minutes = int(delta.total_seconds() // 60)
        if total_minutes == 0:
            return "0min"

        minutes_per_day = 24 * 60
        if total_minutes % minutes_per_day == 0:
            days = total_minutes // minutes_per_day
            return f"{days}d"

        if total_minutes % 60 == 0:
            hours = total_minutes // 60
            return f"{hours}h"

        return f"{total_minutes}min"


    # ------------------------------------------------------------------
    # Results storage helpers
    # ------------------------------------------------------------------
    def set_results_client(
        self,
        results_client: Optional[ResultsClient],
        *,
        auto_write: Optional[bool] = None,
    ) -> None:
        """Attach or update the ResultsClient used for persistence."""

        self.results_client = results_client
        if auto_write is not None:
            self.auto_write_results = bool(auto_write and results_client is not None)
        else:
            self.auto_write_results = bool(self.auto_write_results and results_client is not None)

    @staticmethod
    def _result_payload_to_frame(payload: Any) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, pd.Series):
            return payload.to_frame().T
        if isinstance(payload, dict):
            try:
                return pd.json_normalize(payload)
            except Exception:
                return pd.DataFrame([payload])
        return pd.DataFrame([[payload]], columns=["value"])

    def write_results_to_store(
        self,
        scan_type: str,
        screen_name: str,
        results: Dict[str, Any],
        *,
        replace: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Persist ticker-level screen outputs via the configured ResultsClient."""

        if not self.results_client:
            raise RuntimeError("ResultsClient is not configured for this HistoricalScreener")

        if not results:
            return {}

        writes: Dict[str, str] = {}
        base_meta = {
            "screen_name": screen_name,
            "scan_type": scan_type,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        }
        if metadata:
            base_meta.update(metadata)

        for ticker, payload in results.items():
            df = self._result_payload_to_frame(payload)
            if df.empty:
                continue

            df["ticker"] = ticker
            df["screen_name"] = screen_name
            df["scan_type"] = scan_type

            try:
                key = self.results_client.write_scan_results(
                    scan_type,
                    ticker,
                    screen_name,
                    df,
                    replace=replace,
                    metadata=base_meta,
                )
                writes[ticker] = key
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "Failed to persist results for %s/%s: %s",
                        ticker,
                        screen_name,
                        exc,
                    )
        return writes


    def intraday_momentum_screen(
        self,
        session_starts: List[Union[str, time]] = ["02:30", "08:30"],
        session_ends: List[Union[str, time]] = ["10:30", "13:30"],
        st_momentum_days: int = 3,
        period_length: Optional[Union[int, timedelta]] = None,
        sess_start_hrs: int = 1,
        sess_start_minutes: int = 30,
        sess_end_hrs: Optional[int] = None,
        sess_end_minutes: Optional[int] = None,
        test_vol: bool = True,
        months: Optional[List[int]] = None,
        season: Optional[str] = None,
        _selected_months: Optional[List[int]] = None,
        _precomputed_sessions: Optional[Dict[str, Dict[str, any]]] = None,
        use_regime_filtering: bool = False,
        regime_col: Optional[str] = None,
        target_regimes: Optional[List[int]] = None,
        regime_settings: Optional[RegimeSpecificationLike] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        tz: str = "America/Chicago",
    ) -> Dict[str, Dict[str, any]]:
        """
        Screen for intraday momentum patterns across multiple sessions.

        Parameters:
        -----------
        session_starts : List[Union[str, time]]
            List of session start times (e.g., ["02:30", "08:30"])
        session_ends : List[Union[str, time]]
            List of session end times (e.g., ["10:30", "13:30"])
        st_momentum_days : int
            Number of days for short-term momentum calculation
        sess_start_hrs : int
            Hours from session start to measure opening momentum
        sess_start_minutes : int
            Minutes from session start to measure opening momentum
        sess_end_hrs : Optional[int]
            Hours from session end for closing momentum (defaults to sess_start_hrs)
        sess_end_minutes : Optional[int]
            Minutes from session end for closing momentum (defaults to sess_start_minutes)
        test_vol : bool
            Whether to test volume patterns
        months : Optional[List[int]]
            Specific months to analyze (1-12). If None, analyzes all months
            Example: [1, 2, 12] for Jan, Feb, Dec
        season : Optional[str]
            Season to analyze: 'winter', 'spring', 'summer', 'fall'
            Overrides months parameter if specified
            - winter: Dec, Jan, Feb (12, 1, 2)
            - spring: Mar, Apr, May (3, 4, 5)
            - summer: Jun, Jul, Aug (6, 7, 8)
            - fall: Sep, Oct, Nov (9, 10, 11)
    use_regime_filtering : bool
        Apply a discrete regime filter to the underlying dataset before computing
        statistics.
    regime_col : Optional[str]
        Name of the regime column used when ``use_regime_filtering`` is True.
        target_regimes : Optional[List[int]]
            Regime states that must be present to include a row in the analysis.
        regime_settings : Optional[RegimeSpecificationLike]
            Optional classifier configuration used to materialise ``regime_col``
            when it is absent from the ticker data.
    regime_settings : Optional[RegimeSpecificationLike]
        Optional configuration describing how to construct ``regime_col`` when it does
        not already exist on the input DataFrame.
        tz : str
            Olson timezone used to localize ticker data before evaluating the session
            masks (default "America/Chicago").
        session_start : Optional[Union[str, time]]
            Local session start time for filtering intraday bars (default "00:00").
        session_end : Optional[Union[str, time]]
            Local session end time for filtering intraday bars (default "23:59:59").
        tz : str
            Olson timezone string used to interpret the session window.
        include_volume : bool
            Legacy flag retained for backwards compatibility. When True, raises a
            NotImplementedError explaining that volume analytics moved to OrderflowScan.
        _selected_months : Optional[List[int]]
            Internal override for month selection when using cached data.
        _precomputed_sessions : Optional[Dict[str, Dict[str, any]]]
            Pre-filtered session data keyed by ticker for performance.
        use_regime_filtering : bool
            Enable regime-based filtering before computing session statistics.
        regime_col : Optional[str]
            Name of the column containing discrete regime identifiers.
        target_regimes : Optional[List[int]]
            Regime states to include when ``use_regime_filtering`` is True.
        regime_settings : Optional[RegimeSpecificationLike]
            Optional classifier configuration used to derive ``regime_col`` when it is
            missing from the raw data.
        max_workers : Optional[int]
            Maximum number of parallel workers. If None, uses min(32, cpu_count + 4).
        show_progress : bool
            Show progress bar during processing (default: True)

        Returns:
        --------
        Dict[str, Dict[str, any]]
            Results dictionary with momentum analysis for each ticker and session

        Examples:
        ---------
        # Analyze only winter months
        results = screener.intraday_momentum_screen(
            session_starts=["02:30", "08:30"],
            session_ends=["10:30", "13:30"],
            season='winter'
        )

        # Analyze specific months (Q1)
        results = screener.intraday_momentum_screen(
            session_starts=["09:30"],
            session_ends=["16:00"],
            months=[1, 2, 3]
        )
        """
        session_starts, session_ends = self._convert_times(session_starts, session_ends)
        momentum_params = self._build_momentum_params(
            st_momentum_days,
            period_length,
        )
        regime_classifier = self._resolve_regime_classifier(regime_settings)
        regime_meta_template = self._build_regime_metadata(regime_col, target_regimes, regime_classifier)

        # Determine which months to analyze (allow override when precomputed)
        selected_months = (
            _selected_months
            if _selected_months is not None
            else self._parse_season_months(months, season)
        )

        # Default closing window to match opening window
        if sess_end_hrs is None:
            sess_end_hrs = sess_start_hrs
        if sess_end_minutes is None:
            sess_end_minutes = sess_start_minutes

        if self.verbose:
            self.logger.info(f"Starting intraday momentum screen for {len(self.tickers)} tickers")
            self.logger.info(f"  Sessions: {len(session_starts)} session(s)")
            self.logger.info(f"  Momentum days: {st_momentum_days}, Opening window: {sess_start_hrs}h {sess_start_minutes}m")
            if selected_months:
                self.logger.info(f"  Filtering to months: {selected_months}")

        results = {}
        precomputed_sessions = _precomputed_sessions or {}

        # Helper function for processing a single ticker
        def _process_single_ticker(t: str, skip_correlations: bool = False) -> Tuple[str, Dict[str, any]]:
            try:
                cache_entry = precomputed_sessions.get(t)
                regime_meta: Optional[Dict[str, Any]] = None

                if cache_entry:
                    is_synthetic = cache_entry.get('is_synthetic', self.synthetic_tickers.get(t, False))
                    ticker_data = cache_entry.get('data')
                    price_col = cache_entry.get('price_col')
                    filtered_months_label = cache_entry.get('filtered_months', selected_months if selected_months else 'all')
                    n_observations = cache_entry.get('n_observations', len(ticker_data) if ticker_data is not None else 0)
                    regime_meta = cache_entry.get('regime_filter')

                    if ticker_data is None or ticker_data.empty:
                        result = {
                            'error': cache_entry.get('error', 'No data available'),
                            'filtered_months': filtered_months_label,
                            'ticker': t,
                            'is_synthetic': is_synthetic,
                            'regime_filter': regime_meta,
                        }
                        if selected_months is not None:
                            result['selected_months'] = selected_months
                        return (t, result)
                else:
                    # Get data for this ticker
                    is_synthetic = self.synthetic_tickers.get(t, False)

                    if is_synthetic:
                        synthetic_obj = self.data[t]
                        ticker_data = synthetic_obj.price if hasattr(synthetic_obj, 'price') else synthetic_obj.data_engine.build_spread_series(return_ohlc=True)
                    else:
                        ticker_data = self.data[t]

                    if ticker_data.empty:
                        return (t, {'error': 'No data available', 'regime_filter': regime_meta_template})

                    ticker_data = self._localize_dataframe(ticker_data, tz)

                if (
                    ticker_data is not None
                    and use_regime_filtering
                    and regime_col
                    and regime_classifier is not None
                    and regime_col not in ticker_data.columns
                ):
                    ticker_data = self._ensure_regime_column(
                        ticker_data,
                        ticker=t,
                        regime_col=regime_col,
                        classifier=regime_classifier,
                    )

                # Filter by months/season if specified
                if selected_months is not None:
                    ticker_data = self._filter_by_months(ticker_data, selected_months)
                    if ticker_data.empty:
                        return (t, {
                            'error': 'No data available for selected months/season',
                            'selected_months': selected_months,
                            'regime_filter': regime_meta_template,
                        })

                if use_regime_filtering and regime_col and target_regimes:
                    try:
                        ticker_data = self._filter_by_regime(ticker_data, regime_col, target_regimes)
                    except KeyError:
                        return (
                            t,
                            {
                                'error': f"Regime column '{regime_col}' missing",
                                'selected_months': selected_months,
                                'regime_filter': regime_meta_template,
                            },
                        )

                    if ticker_data.empty:
                        return (
                            t,
                            {
                                'error': 'No data available for selected regimes',
                                'selected_months': selected_months,
                                'regime_filter': regime_meta_template,
                            },
                        )

                    regime_meta = regime_meta_template

                ticker_data = self._ensure_intraday_time_metadata(ticker_data)

                price_col = 'Close' if 'Close' in ticker_data.columns else ticker_data.columns[0]
                filtered_months_label = selected_months if selected_months else 'all'
                n_observations = len(ticker_data)

                if price_col is None and ticker_data is not None:
                    price_col = 'Close' if 'Close' in ticker_data.columns else ticker_data.columns[0]

                ticker_results = {
                    'ticker': t,
                    'is_synthetic': is_synthetic,
                    'n_observations': n_observations,
                    'filtered_months': filtered_months_label,
                    'regime_filter': regime_meta,
                    'momentum_params': momentum_params,
                }

                for i, (start_time, end_time) in enumerate(zip(session_starts, session_ends)):
                    session_key = f"session_{i}"

                    # Use pre-filtered session data when available
                    session_df = None
                    if cache_entry and 'sessions' in cache_entry:
                        session_df = cache_entry['sessions'].get((start_time, end_time))

                    if session_df is None and ticker_data is not None:
                        session_df = self._extract_session_data(ticker_data, start_time, end_time, price_col)

                    # Perform session momentum analysis with filtered data
                    momentum_stats = self._session_momentum_analysis(
                        ticker=t,
                        session_start=start_time,
                        session_end=end_time,
                        start_hrs=sess_start_hrs,
                        start_mins=sess_start_minutes,
                        end_hrs=sess_end_hrs,
                        end_minutes=sess_end_minutes,
                        momentum_days=st_momentum_days,
                        test_vol=test_vol,
                        data=ticker_data,
                        session_data=session_df,
                        price_col=price_col,
                        is_synthetic=is_synthetic,
                        skip_correlations=skip_correlations,
                    )
                    momentum_stats['momentum_params'] = dict(momentum_params)

                    ticker_results[session_key] = momentum_stats

                return (t, ticker_results)
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Error processing ticker {t}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                return (t, {'error': str(e), 'ticker': t})

        worker_count = self._resolve_workers(max_workers)

        # Determine if we should attempt cross-ticker batching
        # Only batch if: GPU enabled, enough tickers, and sequential processing
        # (batching with ThreadPoolExecutor adds complexity and memory pressure)
        attempt_cross_ticker_batching = (
            self.use_gpu and
            len(self.tickers) >= 2 and
            worker_count == 1
        )

        # Process tickers in parallel if worker_count > 1
        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(_process_single_ticker, ticker): ticker for ticker in self.tickers}

                iterator = as_completed(futures) if not show_progress else tqdm(
                    as_completed(futures),
                    total=len(self.tickers),
                    desc="Momentum Screen",
                    unit="ticker"
                )

                for future in iterator:
                    ticker, result = future.result()
                    results[ticker] = result
        else:
            # Sequential processing with optional progress bar
            iterator = tqdm(self.tickers, desc="Momentum Screen", unit="ticker") if show_progress else self.tickers

            # When attempting cross-ticker batching, collect return series first
            if attempt_cross_ticker_batching:
                # Pass 1: Extract return series for all tickers (skip correlations)
                ticker_return_series = {}
                for ticker in iterator:
                    ticker_sym, result = _process_single_ticker(ticker, skip_correlations=True)
                    results[ticker_sym] = result

                    # Collect return series for batching if available
                    if 'error' not in result:
                        for session_key in result:
                            if session_key.startswith('session_') and 'return_series' in result[session_key]:
                                # Store returns for this ticker-session combination
                                key = (ticker_sym, session_key)
                                ret_series = result[session_key]['return_series']
                                ticker_return_series[key] = {
                                    'opening_returns': ret_series['opening_returns'],
                                    'closing_returns': ret_series['closing_returns'],
                                    'st_momentum': ret_series['st_momentum'],
                                    'full_session_returns': ret_series['full_session_returns'],
                                }

                # Pass 2: Batch compute correlations for each session
                if ticker_return_series:
                    # Group by session
                    sessions = set(key[1] for key in ticker_return_series.keys())
                    for session_key in sessions:
                        # Extract data for this session across all tickers
                        session_ticker_data = {
                            ticker: ticker_return_series[(ticker, session_key)]
                            for ticker, sess in ticker_return_series.keys()
                            if sess == session_key
                        }

                        # Batch compute correlations
                        batched_corrs = self._batch_momentum_correlations_cross_ticker(
                            session_ticker_data
                        )

                        # Merge correlation stats back into results
                        if batched_corrs:
                            for ticker, corr_stats in batched_corrs.items():
                                if ticker in results and session_key in results[ticker]:
                                    results[ticker][session_key]['correlations'] = corr_stats
            else:
                # Standard sequential processing with per-ticker correlations
                for ticker in iterator:
                    ticker_sym, result = _process_single_ticker(ticker)
                    results[ticker_sym] = result

        if self.verbose:
            successful = sum(1 for r in results.values() if 'error' not in r)
            self.logger.info(f"Completed momentum screen: {successful}/{len(results)} successful")

        return results

    @staticmethod
    def _coerce_period_minutes(period_length: Optional[Union[int, float, timedelta]]) -> Optional[float]:
        if period_length is None:
            return None
        if isinstance(period_length, timedelta):
            return period_length.total_seconds() / 60.0
        try:
            minutes = float(period_length)
        except (TypeError, ValueError):
            return None
        return minutes

    def _build_momentum_params(
        self,
        st_momentum_days: int,
        period_length: Optional[Union[int, float, timedelta]],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {'st_momentum_days': int(st_momentum_days)}
        period_minutes = self._coerce_period_minutes(period_length)
        if period_minutes is not None:
            params['period_length_min'] = period_minutes
        return params

    def st_seasonality_screen(
        self,
        target_times: List[Union[str, time]],
        period_length: Optional[Union[int, timedelta]] = None,
        dayofweek_screen: bool = True,
        months: Optional[List[int]] = None,
        season: Optional[str] = None,
        session_start: Optional[Union[str, time]] = None,
        session_end: Optional[Union[str, time]] = None,
        tz: str = "America/Chicago",
        include_volume: bool = False,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        use_regime_filtering: bool = False,
        regime_col: Optional[str] = None,
        target_regimes: Optional[List[int]] = None,
        regime_settings: Optional[RegimeSpecificationLike] = None,
        include_calendar_effects: bool = True,
        calendar_horizons: Optional[Dict[str, int]] = None,
        calendar_min_obs: int = 50,
        include_return_series: bool = False,
    ) -> Dict[str, Dict[str, any]]:
        """
        Screen for seasonality patterns in intraday data.

        Tests for:
        - Day-of-week effects (returns and volatility)
        - Time-of-day predictability (next day/week correlations)
        - Lag autocorrelations at specific times
        - Volatility seasonality
        - Month-specific and seasonal patterns
        - Calendar effects (first/last days of month/quarter, lead-lag patterns)

        Parameters:
        -----------
        target_times : List[Union[str, time]]
            List of times to analyze (e.g., ["09:30", "14:00"])
        period_length : Optional[Union[int, timedelta]]
            Length of period to aggregate (e.g., timedelta(minutes=30))
            If None, uses single-bar returns at target times
        dayofweek_screen : bool
            Whether to analyze day-of-week patterns
        months : Optional[List[int]]
            Specific months to analyze (1-12). If None, analyzes all months
            Example: [1, 2, 12] for Jan, Feb, Dec
        season : Optional[str]
            Season to analyze: 'winter', 'spring', 'summer', 'fall'
            Overrides months parameter if specified
            - winter: Dec, Jan, Feb (12, 1, 2)
            - spring: Mar, Apr, May (3, 4, 5)
            - summer: Jun, Jul, Aug (6, 7, 8)
            - fall: Sep, Oct, Nov (9, 10, 11)
        include_calendar_effects : bool
            Whether to include calendar effects analysis (default: True)
            Tests for first/last 1/3/5 days of month/quarter and lead-lag patterns
        calendar_horizons : Optional[Dict[str, int]]
            Horizons for calendar effects (default: {'1d': 1, '3d': 3, '5d': 5})
        calendar_min_obs : int
            Minimum observations for calendar effects (default: 50)
        include_return_series : bool
            When True, attach the predictor/response return series used in the
            time-of-day predictability tests so downstream tools can reuse them
            without recomputing.

        Returns:
        --------
        Dict[str, Dict[str, any]]
            Results dictionary with seasonality analysis for each ticker. Each ticker
            entry includes metadata describing the session window and the number of
            sessions available after filtering. If include_calendar_effects=True,
            also includes 'calendar_edge_effects' and 'calendar_lead_lag' DataFrames.

        Examples:
        ---------
        # Test specific times for next-day predictability
        results = screener.st_seasonality_screen(
            target_times=["09:30", "14:00"],
            dayofweek_screen=True
        )

        # Test 30-minute windows
        results = screener.st_seasonality_screen(
            target_times=["09:30", "13:00"],
            period_length=timedelta(minutes=30),
            dayofweek_screen=True
        )

        # Analyze only winter months
        results = screener.st_seasonality_screen(
            target_times=["10:00"],
            season='winter',
            dayofweek_screen=True
        )

        # Analyze specific months (Q1)
        results = screener.st_seasonality_screen(
            target_times=["14:00"],
            months=[1, 2, 3],
            dayofweek_screen=True
        )
        """
        if include_volume:
            raise NotImplementedError("Volume analytics moved to OrderflowScan")

        session_start_value = session_start if session_start is not None else "00:00"
        session_end_value = session_end if session_end is not None else "23:59:59"

        session_start_time = self._convert_times([session_start_value])[0]
        session_end_time = self._convert_times([session_end_value])[0]

        # Convert times to time objects
        converted_times = self._convert_times(target_times)
        self._validate_target_times(converted_times, session_start_time, session_end_time)

        # Determine which months to analyze
        selected_months = self._parse_season_months(months, season)
        regime_classifier = self._resolve_regime_classifier(regime_settings)
        regime_meta_template = self._build_regime_metadata(regime_col, target_regimes, regime_classifier)

        if self.verbose:
            self.logger.info(f"Starting seasonality screen for {len(self.tickers)} tickers")
            self.logger.info(f"  Target times: {[str(t) for t in converted_times]}")
            self.logger.info(f"  Session window: {session_start_time} to {session_end_time} ({tz})")
            if selected_months:
                self.logger.info(f"  Filtering to months: {selected_months}")

        results = {}

        # Helper function for processing a single ticker
        def _process_single_ticker(ticker: str) -> Tuple[str, Dict[str, any]]:
            try:
                # Get data for this ticker
                is_synthetic = self.synthetic_tickers.get(ticker, False)

                if is_synthetic:
                    synthetic_obj = self.data[ticker]
                    data = synthetic_obj.price if hasattr(synthetic_obj, 'price') else synthetic_obj.data_engine.build_spread_series(return_ohlc=True)
                else:
                    data = self.data[ticker]

                if data.empty:
                    return (ticker, {'error': 'No data available', 'regime_filter': regime_meta_template})

                data = self._localize_dataframe(data, tz)

                # Filter to session window in local timezone
                session_data = filter_session_bars(
                    data,
                    tz,
                    session_start_time,
                    session_end_time,
                )

                if session_data.empty:
                    return (ticker, {
                        'error': 'No data available within session window',
                        'session_start': session_start_time.strftime("%H:%M:%S"),
                        'session_end': session_end_time.strftime("%H:%M:%S"),
                        'tz': tz,
                    })

                if (
                    use_regime_filtering
                    and regime_col
                    and regime_classifier is not None
                    and regime_col not in session_data.columns
                ):
                    session_data = self._ensure_regime_column(
                        session_data,
                        ticker=ticker,
                        regime_col=regime_col,
                        classifier=regime_classifier,
                    )

                if selected_months is not None:
                    session_data = self._filter_by_months(session_data, selected_months)
                    if session_data.empty:
                        return (ticker, {
                            'error': 'No data available for selected months/season',
                            'selected_months': selected_months,
                            'session_start': session_start_time.strftime("%H:%M:%S"),
                            'session_end': session_end_time.strftime("%H:%M:%S"),
                            'tz': tz,
                            'regime_filter': regime_meta_template,
                        })

                regime_meta: Optional[Dict[str, Any]] = None
                if use_regime_filtering and regime_col and target_regimes:
                    try:
                        session_data = self._filter_by_regime(session_data, regime_col, target_regimes)
                    except KeyError:
                        return (
                            ticker,
                            {
                                'error': f"Regime column '{regime_col}' missing",
                                'selected_months': selected_months,
                                'session_start': session_start_time.strftime("%H:%M:%S"),
                                'session_end': session_end_time.strftime("%H:%M:%S"),
                                'tz': tz,
                                'regime_filter': regime_meta_template,
                            },
                        )

                    if session_data.empty:
                        return (
                            ticker,
                            {
                                'error': 'No data available for selected regimes',
                                'selected_months': selected_months,
                                'session_start': session_start_time.strftime("%H:%M:%S"),
                                'session_end': session_end_time.strftime("%H:%M:%S"),
                                'tz': tz,
                                'regime_filter': regime_meta_template,
                            },
                        )

                    regime_meta = regime_meta_template

                session_data = self._ensure_intraday_time_metadata(session_data)

                # Determine price column
                price_col = 'Close' if 'Close' in session_data.columns else session_data.columns[0]

                ticker_results = {
                    'ticker': ticker,
                    'is_synthetic': is_synthetic,
                    'n_observations': len(session_data),
                    'filtered_months': selected_months if selected_months else 'all',
                    'regime_filter': regime_meta,
                }

                if isinstance(session_data.index, pd.DatetimeIndex):
                    index_dt = session_data.index
                else:
                    index_dt = pd.to_datetime(session_data.index)
                    session_data.index = index_dt
                if index_dt.tz is None:
                    session_dates = index_dt.normalize()
                else:
                    session_dates = index_dt.tz_convert(tz).normalize()
                months_meta = self._build_months_metadata(session_data.index, tz)

                ticker_results['metadata'] = {
                    'session_start': session_start_time.strftime("%H:%M:%S"),
                    'session_end': session_end_time.strftime("%H:%M:%S"),
                    'tz': tz,
                    'n_sessions': int(session_dates.nunique()),
                    'period_length': self._serialize_period_length(period_length),
                    'month_filter': ",".join(str(m) for m in sorted(selected_months)) if selected_months else None,
                }
                ticker_results['metadata'].update(months_meta)
                ticker_results['metadata']['regime_filter'] = regime_meta

                period_minutes: Optional[int]
                if period_length is None:
                    period_minutes = None
                elif isinstance(period_length, timedelta):
                    period_minutes = int(period_length.total_seconds() // 60)
                else:
                    try:
                        period_minutes = int(period_length)
                    except (TypeError, ValueError):
                        period_minutes = None

                normalized_times = [clock.strftime("%H:%M") for clock in converted_times]

                pattern_context = {
                    'target_times_hhmm': normalized_times,
                    'period_length_min': period_minutes if period_minutes is not None else 0,
                    'months_active': months_meta['months_active'],
                    'months_mask_12': months_meta['months_mask_12'],
                    'months_names': months_meta['months_names'],
                    'regime_filter': regime_meta,
                }
                ticker_results['pattern_context'] = pattern_context
                ticker_results['months_active'] = months_meta['months_active']
                ticker_results['months_mask_12'] = months_meta['months_mask_12']
                ticker_results['months_names'] = months_meta['months_names']

                # Day-of-week analysis
                if dayofweek_screen:
                    dow_results = self._analyze_dayofweek_patterns(session_data, price_col, is_synthetic)
                    ticker_results['dayofweek_returns'] = dow_results['returns']
                    ticker_results['dayofweek_volatility'] = dow_results['volatility']

                # Time-of-day predictability
                time_predictions = {}
                for target_time in converted_times:
                    pred_stats = self._test_time_predictability(
                        session_data,
                        price_col,
                        target_time,
                        is_synthetic,
                        period_length,
                        tz,
                        include_return_series=include_return_series,
                    )
                    enriched_stats = dict(pattern_context)
                    enriched_stats.update(pred_stats)
                    time_predictions[str(target_time)] = enriched_stats

                ticker_results['time_predictability'] = time_predictions

                weekend_pattern = self._compute_weekend_hedging_pattern(
                    session_data,
                    session_start_time,
                    session_end_time,
                    price_col,
                    is_synthetic,
                    tz,
                    pattern_context,
                )

                # Month-by-month analysis if filtering is applied
                if selected_months is not None:
                    month_analysis = self._analyze_by_month(session_data, price_col, is_synthetic, selected_months)
                    ticker_results['month_breakdown'] = month_analysis

                # Rank strongest patterns (must be done before calendar effects)
                strongest = self._rank_seasonal_strength(ticker_results)
                if weekend_pattern is not None:
                    strongest.append(weekend_pattern)

                # Calendar effects analysis (edge days, lead-lag)
                calendar_patterns = []
                if include_calendar_effects:
                    try:
                        cal_params = CalendarEffectParams(
                            price_col=price_col,
                            horizons=calendar_horizons,
                            min_obs=calendar_min_obs,
                            week_len=5,
                        )

                        # Resample to daily for calendar analysis
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data_with_index = data.set_index('date') if 'date' in data.columns else data
                        else:
                            data_with_index = data

                        # Resample to daily if higher frequency
                        freq = data_with_index.index.inferred_freq
                        if freq and freq not in ['D', 'B', 'C']:
                            data_daily = data_with_index.resample('D')[price_col].last().to_frame(name=price_col)
                            data_daily = data_daily.dropna()
                        else:
                            data_daily = data_with_index[[price_col]].copy() if price_col in data_with_index.columns else data_with_index.copy()

                        if not data_daily.empty:
                            # Run calendar edge and lead-lag tests
                            edge_df = run_calendar_edge_tests(data_daily, cal_params)
                            leadlag_df = run_calendar_lead_lag_tests(data_daily, cal_params)

                            ticker_results['calendar_edge_effects'] = edge_df
                            ticker_results['calendar_lead_lag'] = leadlag_df

                            # Extract significant calendar patterns for strongest_patterns
                            if not edge_df.empty:
                                sig_edges = edge_df[edge_df['sig_fdr_5pct']].copy()
                                for _, row in sig_edges.iterrows():
                                    calendar_patterns.append({
                                        'pattern_type': 'calendar',
                                        'calendar_pattern': row['pattern'],
                                        'event': row['event'],
                                        'horizon': row['horizon'],
                                        'mean_return': row['mean'],
                                        't_stat': row['t_stat'],
                                        'p_value': row['p_value'],
                                        'q_value': row['q_value'],
                                        'n_obs': row['n_obs'],
                                        'exploratory': row['exploratory']
                                    })

                            if not leadlag_df.empty:
                                sig_leadlag = leadlag_df[leadlag_df['sig_fdr_5pct']].copy()
                                for _, row in sig_leadlag.iterrows():
                                    calendar_patterns.append({
                                        'pattern_type': 'calendar',
                                        'calendar_pattern': row['pattern'],
                                        'predictor': row['predictor'],
                                        'response': row['response'],
                                        'beta': row['slope'],
                                        't_stat': row['t_stat'],
                                        'p_value': row['p_value'],
                                        'q_value': row['q_value'],
                                        'r2': row['r2'],
                                        'n_obs': row['n_obs'],
                                        'exploratory': row['exploratory']
                                    })
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(f"Calendar effects analysis failed for {ticker}: {str(e)}")
                        ticker_results['calendar_edge_effects'] = pd.DataFrame()
                        ticker_results['calendar_lead_lag'] = pd.DataFrame()

                # Combine all patterns
                strongest.extend(calendar_patterns)
                ticker_results['strongest_patterns'] = strongest

                return (ticker, ticker_results)
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Error processing ticker {ticker}: {str(e)}")
                return (ticker, {'error': str(e), 'ticker': ticker})

        worker_count = self._resolve_workers(max_workers)

        # Process tickers in parallel if worker_count > 1
        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(_process_single_ticker, ticker): ticker for ticker in self.tickers}

                iterator = as_completed(futures) if not show_progress else tqdm(
                    as_completed(futures),
                    total=len(self.tickers),
                    desc="Seasonality Screen",
                    unit="ticker"
                )

                for future in iterator:
                    ticker, result = future.result()
                    results[ticker] = result
        else:
            # Sequential processing with optional progress bar
            iterator = tqdm(self.tickers, desc="Seasonality Screen", unit="ticker") if show_progress else self.tickers
            for ticker in iterator:
                ticker_sym, result = _process_single_ticker(ticker)
                results[ticker_sym] = result

        if self.verbose:
            successful = sum(1 for r in results.values() if 'error' not in r)
            self.logger.info(f"Completed seasonality screen: {successful}/{len(results)} successful")

        return results

    def calendar_effects_screen(
        self,
        price_col: str = "close",
        horizons: Optional[Dict[str, int]] = None,
        min_obs: int = 50,
        week_len: int = 5,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Screen for calendar effects in daily price data.

        Tests for:
        - First/last 1, 3, 5 trading days of month/quarter effects
        - First/last week of month effects
        - Lead-lag: previous month/year weeks predicting current month

        Args:
            price_col: Column name for price data (default: 'close')
            horizons: Dict mapping horizon labels to trading days
                     (default: {'1d': 1, '3d': 3, '5d': 5})
            min_obs: Minimum observations for reliable statistics (default: 50)
                    Below this threshold, patterns flagged as exploratory
            week_len: Trading days per 'week of month' (default: 5)
            max_workers: Number of parallel workers (default: CPU count)
            show_progress: Show progress bar (default: True)

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]:
                Results dictionary with calendar analysis for each ticker.
                Each ticker entry contains:
                    - 'calendar_edges': DataFrame with edge effects
                    - 'calendar_lead_lag': DataFrame with predictive patterns
                    - 'metadata': Processing metadata

        Examples:
            # Run calendar effects screen with defaults
            results = screener.calendar_effects_screen()

            # Customize horizons and minimum observations
            results = screener.calendar_effects_screen(
                horizons={'1d': 1, '5d': 5, '10d': 10},
                min_obs=80
            )

            # Access results for a specific ticker
            es_results = results['ES']
            edge_effects = es_results['calendar_edges']
            lead_lag = es_results['calendar_lead_lag']

            # Filter to significant patterns
            significant = edge_effects[edge_effects['sig_fdr_5pct']]
        """
        params = CalendarEffectParams(
            price_col=price_col,
            horizons=horizons,
            min_obs=min_obs,
            week_len=week_len,
        )

        if self.verbose:
            self.logger.info(f"Starting calendar effects screen for {len(self.tickers)} tickers")
            self.logger.info(f"  Horizons: {params.horizons}")
            self.logger.info(f"  Minimum observations: {params.min_obs}")

        results = {}

        def _process_single_ticker(ticker: str) -> Tuple[str, Dict[str, any]]:
            try:
                # Get daily data for this ticker
                is_synthetic = self.synthetic_tickers.get(ticker, False)

                if is_synthetic:
                    synthetic_obj = self.data[ticker]
                    data = synthetic_obj.price if hasattr(synthetic_obj, 'price') else synthetic_obj.data_engine.build_spread_series(return_ohlc=True)
                else:
                    data = self.data[ticker]

                if data.empty:
                    return (ticker, {'error': 'No data available'})

                # Resample to daily if needed
                if not isinstance(data.index, pd.DatetimeIndex):
                    if 'date' in data.columns:
                        data = data.set_index('date')
                    else:
                        return (ticker, {'error': 'No DatetimeIndex or date column'})

                # Ensure we have daily data - resample if higher frequency
                if hasattr(data.index.inferred_freq, '__call__'):
                    freq = data.index.inferred_freq()
                else:
                    freq = data.index.inferred_freq

                if freq and freq not in ['D', 'B', 'C']:  # Not daily/business day
                    # Resample to daily close
                    data_daily = data.resample('D')[price_col].last().to_frame(name=price_col)
                    data_daily = data_daily.dropna()
                else:
                    data_daily = data[[price_col]].copy() if price_col in data.columns else data.copy()

                if data_daily.empty:
                    return (ticker, {'error': 'No daily data after resampling'})

                # Run calendar edge tests
                df_edges = run_calendar_edge_tests(data_daily, params)

                # Run lead-lag tests
                df_lead_lag = run_calendar_lead_lag_tests(data_daily, params)

                result = {
                    'calendar_edges': df_edges,
                    'calendar_lead_lag': df_lead_lag,
                    'metadata': {
                        'ticker': ticker,
                        'n_days': len(data_daily),
                        'date_range': f"{data_daily.index.min()} to {data_daily.index.max()}",
                        'price_col': price_col,
                        'horizons': params.horizons,
                    }
                }

                return (ticker, result)

            except Exception as e:
                return (ticker, {'error': str(e)})

        # Process tickers
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_single_ticker, ticker) for ticker in self.tickers]

                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(self.tickers), desc="Calendar Effects")

                for future in iterator:
                    ticker_sym, result = future.result()
                    results[ticker_sym] = result
        else:
            iterator = self.tickers
            if show_progress:
                iterator = tqdm(iterator, desc="Calendar Effects")

            for ticker in iterator:
                ticker_sym, result = _process_single_ticker(ticker)
                results[ticker_sym] = result

        if self.verbose:
            successful = sum(1 for r in results.values() if 'error' not in r)
            self.logger.info(f"Completed calendar effects screen: {successful}/{len(results)} successful")

            # Log summary statistics
            if successful > 0:
                total_edge_patterns = sum(
                    len(r['calendar_edges']) for r in results.values()
                    if 'calendar_edges' in r
                )
                total_lead_lag = sum(
                    len(r['calendar_lead_lag']) for r in results.values()
                    if 'calendar_lead_lag' in r
                )
                sig_edges = sum(
                    r['calendar_edges']['sig_fdr_5pct'].sum() for r in results.values()
                    if 'calendar_edges' in r and not r['calendar_edges'].empty
                )
                sig_lead_lag = sum(
                    r['calendar_lead_lag']['sig_fdr_5pct'].sum() for r in results.values()
                    if 'calendar_lead_lag' in r and not r['calendar_lead_lag'].empty
                )

                self.logger.info(f"  Total edge patterns tested: {total_edge_patterns}")
                self.logger.info(f"  Significant edge patterns (FDR 5%): {sig_edges}")
                self.logger.info(f"  Total lead-lag tests: {total_lead_lag}")
                self.logger.info(f"  Significant lead-lag (FDR 5%): {sig_lead_lag}")

        return results

    def _prepare_momentum_session_cache(
        self,
        session_pairs: Tuple[Tuple[time, time], ...],
        selected_months: Optional[List[int]],
        *,
        use_regime_filtering: bool = False,
        regime_col: Optional[str] = None,
        target_regimes: Optional[List[int]] = None,
        regime_settings: Optional[RegimeSpecificationLike] = None,
        tz: str = "America/Chicago",
    ) -> Dict[str, Dict[str, any]]:
        """Pre-filter ticker data for momentum screens to avoid redundant work."""
        cache: Dict[str, Dict[str, any]] = {}
        regime_classifier = self._resolve_regime_classifier(regime_settings)
        regime_meta_template = self._build_regime_metadata(regime_col, target_regimes, regime_classifier)

        for ticker in self.tickers:
            is_synthetic = self.synthetic_tickers.get(ticker, False)

            if is_synthetic:
                synthetic_obj = self.data[ticker]
                base_data = synthetic_obj.price if hasattr(synthetic_obj, 'price') else synthetic_obj.data_engine.build_spread_series(return_ohlc=True)
            else:
                base_data = self.data[ticker]

            if base_data.empty:
                cache[ticker] = {
                    'data': None,
                    'is_synthetic': is_synthetic,
                    'filtered_months': selected_months if selected_months else 'all',
                    'error': 'No data available'
                }
                continue

            filtered_data = base_data
            filtered_data = self._localize_dataframe(filtered_data, tz)
            if (
                filtered_data is not None
                and use_regime_filtering
                and regime_col
                and regime_classifier is not None
                and regime_col not in filtered_data.columns
            ):
                filtered_data = self._ensure_regime_column(
                    filtered_data,
                    ticker=ticker,
                    regime_col=regime_col,
                    classifier=regime_classifier,
                )
            if selected_months is not None:
                filtered_data = self._filter_by_months(filtered_data, selected_months)
                if filtered_data.empty:
                    cache[ticker] = {
                        'data': None,
                        'is_synthetic': is_synthetic,
                        'filtered_months': selected_months,
                        'regime_filter': regime_meta_template,
                        'error': 'No data available for selected months/season'
                    }
                    continue

            regime_meta: Optional[Dict[str, Any]] = None
            if use_regime_filtering and regime_col and target_regimes:
                try:
                    filtered_data = self._filter_by_regime(filtered_data, regime_col, target_regimes)
                except KeyError:
                    cache[ticker] = {
                        'data': None,
                        'is_synthetic': is_synthetic,
                        'filtered_months': selected_months if selected_months else 'all',
                        'regime_filter': regime_meta_template,
                        'error': f"Regime column '{regime_col}' missing from data"
                    }
                    continue

                if filtered_data.empty:
                    cache[ticker] = {
                        'data': None,
                        'is_synthetic': is_synthetic,
                        'filtered_months': selected_months if selected_months else 'all',
                        'regime_filter': regime_meta_template,
                        'error': 'No data available for selected regimes'
                    }
                    continue

                regime_meta = regime_meta_template

            price_col = 'Close' if 'Close' in filtered_data.columns else filtered_data.columns[0]

            session_map: Dict[Tuple[time, time], pd.DataFrame] = {}
            for session_start, session_end in session_pairs:
                session_map[(session_start, session_end)] = self._extract_session_data(
                    filtered_data,
                    session_start,
                    session_end,
                    price_col
                )

            cache[ticker] = {
                'data': filtered_data,
                'is_synthetic': is_synthetic,
                'price_col': price_col,
                'sessions': session_map,
                'n_observations': len(filtered_data),
                'filtered_months': selected_months if selected_months else 'all',
                'regime_filter': regime_meta,
            }

        return cache

    def run_screens(
        self,
        screen_params: List[ScreenParams],
        output_format: str = 'dict',
        use_concurrent: bool = True,
        max_workers: Optional[int] = None,
        batch_tickers: bool = False,
        align_method: str = 'intersection'
    ) -> Union[Dict[str, Dict], pd.DataFrame]:
        """
        Run multiple screens with composite outputs.

        This method allows you to run multiple screening configurations
        (e.g., different seasons, months, or screen types) and combine
        the results into a single output.

        Parameters
        ----------
        screen_params : List[ScreenParams]
            List of ScreenParams objects defining each screen to run
        output_format : str, default 'dict'
            Output format: 'dict' returns nested dictionary, 'dataframe' returns flat DataFrame
        use_concurrent : bool, default True
            Enable concurrent processing of screens for GPU acceleration.
            When True and GPU is available, screens are processed in parallel using
            ThreadPoolExecutor, maximizing GPU utilization. Set to False to fall back
            to sequential processing (useful for debugging).
        max_workers : Optional[int], default None
            Maximum number of concurrent workers. If None and GPU is available,
            defaults to GPU_DEVICE_COUNT. Otherwise defaults to 1 (sequential).
        batch_tickers : bool, default False
            Enable GPU-batched processing of all tickers simultaneously within each screen.
            When True, aligns ticker data to common dates and computes correlations/statistics
            for all tickers in a single GPU operation. Provides 5-10x speedup for screens with
            many tickers. Requires use_gpu=True. Set to False to process tickers sequentially.
        align_method : str, default 'intersection'
            Method for aligning ticker data when batch_tickers=True:
            - 'intersection': Use only dates common to ALL tickers (safest, no filled data)
            - 'union': Use all dates from ANY ticker (forward-fills gaps, more coverage)

        Notes
        -----
        Momentum screens share cached session slices when possible so running
        multiple configurations with overlapping hours avoids redundant
        filtering.

        When use_concurrent=True and GPU is available, screens are processed in
        parallel, providing significant speedup (typically 4-8x for 4+ screens).
        The momentum_cache is shared across all workers to avoid redundant computation.

        Returns
        -------
        Union[Dict[str, Dict], pd.DataFrame]
            Composite screening results:
            - If 'dict': {screen_name: {ticker: results}}
            - If 'dataframe': Flattened DataFrame with screen_name and ticker as index

        Examples
        --------
        # Run screens for multiple seasons with GPU batching
        >>> screens = [
        ...     ScreenParams(screen_type='momentum', season='winter',
        ...                  session_starts=["02:30"], session_ends=["10:30"]),
        ...     ScreenParams(screen_type='momentum', season='spring',
        ...                  session_starts=["02:30"], session_ends=["10:30"]),
        ...     ScreenParams(screen_type='seasonality', season='summer',
        ...                  target_times=["09:30", "14:00"])
        ... ]
        >>> results = screener.run_screens(screens, use_concurrent=True)
        >>> # Access results: results['winter_momentum']['CL_F']

        # Get results as DataFrame
        >>> df = screener.run_screens(screens, output_format='dataframe')
        >>> # Access: df.loc[('winter_momentum', 'CL_F')]

        # Disable concurrent processing for debugging
        >>> results = screener.run_screens(screens, use_concurrent=False)
        """
        composite_results = {}
        momentum_cache: Dict[
            Tuple[
                Tuple[time, ...],
                Tuple[time, ...],
                Optional[Tuple[int, ...]],
                Optional[Tuple[str, Tuple[int, ...]]],
            ],
            Dict[str, Dict[str, Any]],
        ] = {}

        # Determine if we should use concurrent processing
        should_use_concurrent = (
            use_concurrent
            and len(screen_params) > 1
            and GPU_AVAILABLE
        )

        if should_use_concurrent:
            # Concurrent processing with GPU acceleration
            if max_workers is None:
                max_workers = GPU_DEVICE_COUNT if GPU_DEVICE_COUNT > 0 else 4

            if self.logger:
                self.logger.info(
                    "Running %d screens concurrently with %d workers (GPU-accelerated)",
                    len(screen_params),
                    max_workers,
                )

            def process_single_screen(params: ScreenParams) -> Tuple[str, Dict[str, Dict[str, Any]]]:
                """Process a single screen and return (name, results)"""
                if params.screen_type == 'momentum':
                    session_starts, session_ends = self._convert_times(params.session_starts, params.session_ends)
                    selected_months = self._parse_season_months(params.months, params.season)
                    session_pairs = tuple(zip(session_starts, session_ends))
                    months_key = tuple(selected_months) if selected_months else None
                    regime_key = None
                    regime_signature = self._regime_signature(params.regime_settings)
                    if params.use_regime_filtering and params.regime_col and params.target_regimes:
                        regime_key = (params.regime_col, tuple(params.target_regimes), regime_signature)

                    cache_key = (tuple(session_starts), tuple(session_ends), months_key, regime_key)

                    # Check if cached (thread-safe dict lookup)
                    if cache_key not in momentum_cache:
                        # Compute and cache (potential race condition, but results are identical)
                        momentum_cache[cache_key] = self._prepare_momentum_session_cache(
                            session_pairs,
                            selected_months,
                            use_regime_filtering=params.use_regime_filtering,
                            regime_col=params.regime_col,
                            target_regimes=params.target_regimes,
                            regime_settings=params.regime_settings,
                            tz=params.tz,
                        )

                    screen_result = self._parse_params(
                        params,
                        session_starts=session_starts,
                        session_ends=session_ends,
                        _selected_months=selected_months,
                        _precomputed_sessions=momentum_cache[cache_key],
                        use_regime_filtering=params.use_regime_filtering,
                        regime_col=params.regime_col,
                        target_regimes=params.target_regimes,
                        regime_settings=params.regime_settings,
                    )
                else:
                    # Parse parameters and run appropriate screen
                    screen_result = self._parse_params(params, regime_settings=params.regime_settings)

                return params.name, screen_result

            # Process screens concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_screen, params): params
                    for params in screen_params
                }

                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        name, screen_result = future.result()
                        composite_results[name] = screen_result

                        # Auto-write results if enabled
                        if self.auto_write_results and self.results_client:
                            try:
                                self.write_results_to_store(
                                    params.screen_type,
                                    name,
                                    screen_result,
                                )
                            except Exception as exc:
                                if self.logger:
                                    self.logger.warning(
                                        "Automatic persistence failed for %s (%s): %s",
                                        name,
                                        params.screen_type,
                                        exc,
                                    )
                    except Exception as exc:
                        if self.logger:
                            self.logger.error(
                                "Screen '%s' failed: %s",
                                params.name,
                                exc,
                            )
                        # Continue processing other screens
                        continue
        else:
            # Sequential processing (original implementation)
            if self.logger and not use_concurrent:
                self.logger.info("Running %d screens sequentially (concurrent processing disabled)", len(screen_params))
            elif self.logger:
                self.logger.info("Running %d screens sequentially (GPU not available or single screen)", len(screen_params))

            for params in screen_params:
                if params.screen_type == 'momentum':
                    session_starts, session_ends = self._convert_times(params.session_starts, params.session_ends)
                    selected_months = self._parse_season_months(params.months, params.season)
                    session_pairs = tuple(zip(session_starts, session_ends))
                    months_key = tuple(selected_months) if selected_months else None
                    regime_key = None
                    regime_signature = self._regime_signature(params.regime_settings)
                    if params.use_regime_filtering and params.regime_col and params.target_regimes:
                        regime_key = (params.regime_col, tuple(params.target_regimes), regime_signature)

                    cache_key = (tuple(session_starts), tuple(session_ends), months_key, regime_key)

                    if cache_key not in momentum_cache:
                        momentum_cache[cache_key] = self._prepare_momentum_session_cache(
                            session_pairs,
                            selected_months,
                            use_regime_filtering=params.use_regime_filtering,
                            regime_col=params.regime_col,
                            target_regimes=params.target_regimes,
                            regime_settings=params.regime_settings,
                            tz=params.tz,
                        )

                    screen_result = self._parse_params(
                        params,
                        session_starts=session_starts,
                        session_ends=session_ends,
                        _selected_months=selected_months,
                        _precomputed_sessions=momentum_cache[cache_key],
                        use_regime_filtering=params.use_regime_filtering,
                        regime_col=params.regime_col,
                        target_regimes=params.target_regimes,
                        regime_settings=params.regime_settings,
                    )
                else:
                    # Parse parameters and run appropriate screen
                    screen_result = self._parse_params(params, regime_settings=params.regime_settings)

                if self.auto_write_results and self.results_client:
                    try:
                        self.write_results_to_store(
                            params.screen_type,
                            params.name,
                            screen_result,
                        )
                    except Exception as exc:
                        if self.logger:
                            self.logger.warning(
                                "Automatic persistence failed for %s (%s): %s",
                                params.name,
                                params.screen_type,
                                exc,
                            )

                # Store results with the screen name as key
                composite_results[params.name] = screen_result

        # Convert to requested format
        if output_format.lower() == 'dataframe':
            return self._composite_to_dataframe(composite_results)
        elif output_format.lower() == 'dict':
            return composite_results
        else:
            raise ValueError(f"output_format must be 'dict' or 'dataframe', got '{output_format}'")

    def _parse_params(self, params: ScreenParams, **kwargs) -> Dict[str, Dict]:
        """
        Parse ScreenParams and execute the appropriate screening method.

        Parameters
        ----------
        params : ScreenParams
            Screen configuration parameters
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying screen method.

        Returns
        -------
        Dict[str, Dict]
            Screening results from the appropriate method

        Raises
        ------
        ValueError
            If screen_type is invalid or required parameters are missing
        """
        if params.screen_type == 'momentum':
            momentum_kwargs = dict(kwargs)
            regime_settings_override = momentum_kwargs.pop('regime_settings', params.regime_settings)
            override_session_starts = momentum_kwargs.pop('session_starts', params.session_starts)
            override_session_ends = momentum_kwargs.pop('session_ends', params.session_ends)
            selected_months_override = momentum_kwargs.pop('_selected_months', None)
            precomputed_sessions = momentum_kwargs.pop('_precomputed_sessions', None)
            override_use_regime_filtering = momentum_kwargs.pop(
                'use_regime_filtering', params.use_regime_filtering
            )
            override_regime_col = momentum_kwargs.pop('regime_col', params.regime_col)
            override_target_regimes = momentum_kwargs.pop('target_regimes', params.target_regimes)

            # Run momentum screen with provided parameters
            return self.intraday_momentum_screen(
                session_starts=override_session_starts,
                session_ends=override_session_ends,
                st_momentum_days=params.st_momentum_days,
                period_length=params.period_length,
                sess_start_hrs=params.sess_start_hrs,
                sess_start_minutes=params.sess_start_minutes,
                sess_end_hrs=params.sess_end_hrs,
                sess_end_minutes=params.sess_end_minutes,
                test_vol=params.test_vol,
                months=params.months,
                season=params.season,
                _selected_months=selected_months_override,
                _precomputed_sessions=precomputed_sessions,
                use_regime_filtering=override_use_regime_filtering,
                regime_col=override_regime_col,
                target_regimes=override_target_regimes,
                regime_settings=regime_settings_override,
                tz=params.tz,
                **momentum_kwargs
            )

        elif params.screen_type == 'seasonality':
            regime_settings_override = kwargs.pop('regime_settings', params.regime_settings)
            # Run seasonality screen with provided parameters
            return self.st_seasonality_screen(
                target_times=params.target_times,
                period_length=params.period_length,
                dayofweek_screen=params.dayofweek_screen,
                months=params.months,
                season=params.season,
                session_start=params.seasonality_session_start,
                session_end=params.seasonality_session_end,
                tz=params.tz,
                use_regime_filtering=params.use_regime_filtering,
                regime_col=params.regime_col,
                target_regimes=params.target_regimes,
                regime_settings=regime_settings_override,
            )

        else:
            raise ValueError(f"Invalid screen_type: {params.screen_type}")

    def _composite_to_dataframe(
        self,
        composite_results: Dict[str, Dict[str, Dict]]
    ) -> pd.DataFrame:
        """
        Convert composite screening results to a flattened DataFrame.

        Parameters
        ----------
        composite_results : Dict[str, Dict[str, Dict]]
            Nested dict: {screen_name: {ticker: {metric: value}}}

        Returns
        -------
        pd.DataFrame
            Flattened DataFrame with MultiIndex (screen_name, ticker)

        Notes
        -----
        - Handles nested dictionaries by creating hierarchical column names
        - Preserves all metrics from screening results
        - Creates MultiIndex for easy filtering and grouping
        """
        flattened_rows = []

        for screen_name, ticker_results in composite_results.items():
            for ticker, metrics in ticker_results.items():
                # Flatten nested metrics
                flat_metrics = self._flatten_dict(metrics, parent_key='', sep='_')

                # Add identifying columns
                flat_metrics['screen_name'] = screen_name
                flat_metrics['ticker'] = ticker

                flattened_rows.append(flat_metrics)

        # Create DataFrame
        df = pd.DataFrame(flattened_rows)

        # Set MultiIndex if we have the required columns
        if 'screen_name' in df.columns and 'ticker' in df.columns:
            df = df.set_index(['screen_name', 'ticker'])

        return df

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '_'
    ) -> Dict:
        """
        Flatten a nested dictionary into a single-level dictionary.

        Parameters
        ----------
        d : Dict
            Nested dictionary to flatten
        parent_key : str, default ''
            Prefix for keys (used in recursion)
        sep : str, default '_'
            Separator for nested keys

        Returns
        -------
        Dict
            Flattened dictionary with concatenated keys

        Examples
        --------
        >>> nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        >>> _flatten_dict(nested)
        {'a': 1, 'b_c': 2, 'b_d_e': 3}
        """
        items = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                # Recursively flatten nested dicts
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists/tuples to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))

        return dict(items)

    # ------------------------------------------------------------------
    # Regime helpers
    # ------------------------------------------------------------------
    def _resolve_regime_classifier(
        self, settings: Optional[RegimeSpecificationLike]
    ) -> Optional[BaseRegimeClassifier]:
        if settings is None:
            return None
        return build_regime_classifier(settings)

    def _batch_correlations_if_possible(
        self,
        ticker_data: Dict[str, Dict[str, np.ndarray]],
        x_key: str,
        y_key: str,
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Try to compute correlations in batch if arrays are aligned.

        This method attempts to use GPU-batched correlation computation when:
        1. GPU is enabled (self.use_gpu=True)
        2. All ticker arrays have the same length (aligned data)
        3. At least 2 tickers to process

        Args:
            ticker_data: Dictionary mapping ticker -> {key: array}
                Example: {'CL': {'open': [...], 'close': [...]}, 'GC': {...}}
            x_key: Key for x arrays (e.g., 'open_returns')
            y_key: Key for y arrays (e.g., 'close_returns')

        Returns:
            Dictionary mapping ticker -> (correlation, p_value)
            Returns None if batching is not possible (falls back to sequential)

        Example:
            >>> ticker_arrays = {
            ...     'CL': {'open': open_returns_cl, 'close': close_returns_cl},
            ...     'GC': {'open': open_returns_gc, 'close': close_returns_gc},
            ... }
            >>> results = self._batch_correlations_if_possible(ticker_arrays, 'open', 'close')
            >>> if results:
            ...     for ticker, (corr, pval) in results.items():
            ...         print(f"{ticker}: r={corr:.3f}, p={pval:.3e}")
        """
        # Check if GPU batching is enabled
        if not self.use_gpu:
            return None

        # Need at least 2 tickers for batching to be worthwhile
        if len(ticker_data) < 2:
            return None

        # Check if all tickers have the required keys
        tickers = list(ticker_data.keys())
        if not all(x_key in ticker_data[t] and y_key in ticker_data[t] for t in tickers):
            return None

        # Check if all arrays have same length (required for batching)
        try:
            lengths = [len(ticker_data[t][x_key]) for t in tickers]
            if len(set(lengths)) != 1:
                # Arrays have different lengths, can't batch
                return None

            # Also check y arrays
            y_lengths = [len(ticker_data[t][y_key]) for t in tickers]
            if lengths != y_lengths:
                return None

        except (KeyError, TypeError):
            return None

        # Extract arrays for batching
        try:
            x_arrays = [ticker_data[t][x_key] for t in tickers]
            y_arrays = [ticker_data[t][y_key] for t in tickers]

            # Import and use GPU batch correlation
            from .gpu_stats import batch_pearson_correlation

            corrs, pvals = batch_pearson_correlation(x_arrays, y_arrays, use_gpu=True)

            # Build result dictionary
            return {
                ticker: (float(corrs[i]), float(pvals[i]))
                for i, ticker in enumerate(tickers)
            }

        except Exception as e:
            # If batching fails for any reason, fall back to sequential
            if self.logger:
                self.logger.debug(f"Batch correlation failed, falling back to sequential: {e}")
            return None

    def _regime_signature(self, settings: Optional[RegimeSpecificationLike]) -> Optional[str]:
        classifier = self._resolve_regime_classifier(settings)
        return classifier.cache_key() if classifier is not None else None

    def _build_regime_metadata(
        self,
        regime_col: Optional[str],
        target_regimes: Optional[Sequence[int]],
        classifier: Optional[BaseRegimeClassifier],
    ) -> Optional[Dict[str, Any]]:
        if not regime_col or not target_regimes:
            return None
        meta: Dict[str, Any] = {
            'column': regime_col,
            'targets': list(target_regimes),
        }
        if classifier is not None:
            meta['classifier'] = classifier.describe()
        return meta

    @staticmethod
    def _frame_fingerprint(df: pd.DataFrame) -> str:
        hashed = pd.util.hash_pandas_object(df.index, index=True)
        return str(int(hashed.sum()))

    def _ensure_regime_column(
        self,
        df: pd.DataFrame,
        *,
        ticker: str,
        regime_col: str,
        classifier: BaseRegimeClassifier,
    ) -> pd.DataFrame:
        if regime_col in df.columns:
            return df
        fingerprint = self._frame_fingerprint(df)
        cache_key = (ticker, regime_col, classifier.cache_key())
        bucket = self._regime_cache.setdefault(cache_key, {})
        series = bucket.get(fingerprint)
        if series is None:
            series = classifier.classify(df)
            bucket[fingerprint] = series
        result = df.copy()
        result[regime_col] = series.reindex(df.index)
        return result

    def _session_momentum_analysis(
        self,
        ticker: str,
        session_start: time,
        session_end: time,
        start_hrs: int = 1,
        start_mins: int = 30,
        end_hrs: Optional[int] = None,
        end_minutes: Optional[int] = None,
        momentum_days: int = 3,
        test_vol: bool = True,
        data: Optional[pd.DataFrame] = None,
        session_data: Optional[pd.DataFrame] = None,
        price_col: Optional[str] = None,
        is_synthetic: Optional[bool] = None,
        skip_correlations: bool = False,
    ) -> Dict[str, any]:
        """
        Analyze momentum patterns for a specific trading session.

        Calculates:
        - Opening momentum (first N minutes of session)
        - Closing momentum (last N minutes of session)
        - Full session returns
        - Correlation between open/close momentum
        - Short-term momentum (N-day rolling returns)
        - Volume patterns (if test_vol=True)

        Parameters:
        -----------
        ticker : str
            Ticker symbol to analyze
        session_start : time
            Session start time
        session_end : time
            Session end time
        start_hrs : int
            Hours for opening window
        start_mins : int
            Minutes for opening window
        end_hrs : Optional[int]
            Hours for closing window
        end_minutes : Optional[int]
            Minutes for closing window
        momentum_days : int
            Days for short-term momentum calculation
        test_vol : bool
            Whether to analyze volume patterns
        data : Optional[pd.DataFrame]
            Pre-filtered data to use (e.g., month-filtered). If None, loads from self.data
        session_data : Optional[pd.DataFrame]
            Pre-filtered session slice to avoid recomputing within the method.
        price_col : Optional[str]
            Column to use for price data; inferred if not supplied.
        is_synthetic : Optional[bool]
            Override for synthetic ticker detection when data already provided.

        Returns:
        --------
        Dict[str, any]
            Dictionary with momentum statistics
        """
        # Get data for this ticker if not provided
        if data is None:
            # Load data when not provided
            resolved_is_synthetic = self.synthetic_tickers.get(ticker, False) if is_synthetic is None else is_synthetic

            if resolved_is_synthetic:
                # Extract price data from SyntheticSymbol
                synthetic_obj = self.data[ticker]
                data = synthetic_obj.price if hasattr(synthetic_obj, 'price') else synthetic_obj.data_engine.build_spread_series(return_ohlc=True)
            else:
                data = self.data[ticker]

            is_synthetic = resolved_is_synthetic
        else:
            # If data is provided, determine synthetic status from the ticker when not supplied
            if is_synthetic is None:
                is_synthetic = self.synthetic_tickers.get(ticker, False)

        if data.empty:
            return {'error': 'No data available', 'ticker': ticker}

        # Determine price column based on data type
        if price_col is None:
            price_col = 'Close' if 'Close' in data.columns else data.columns[0]

        # Create time windows
        opening_window = timedelta(hours=start_hrs, minutes=start_mins)
        opening_window_minutes = start_hrs * 60 + start_mins

        # ``end_hrs``/``end_minutes`` default to the opening window.  Keep both the
        # ``timedelta`` representation (used for slicing) and the raw hour/minute
        # components so metadata serialization later on does not rely on undefined
        # names (the previous implementation referenced ``closing_hours`` and
        # ``closing_minutes`` without actually defining them).
        closing_hours = end_hrs if end_hrs is not None else start_hrs
        closing_minutes = end_minutes if end_minutes is not None else start_mins
        closing_window = timedelta(hours=closing_hours, minutes=closing_minutes)
        closing_window_minutes = closing_hours * 60 + closing_minutes

        # Extract session data using time of day filtering
        daily_sessions = session_data
        if daily_sessions is None:
            daily_sessions = self._extract_session_data(
                data,
                session_start,
                session_end,
                price_col
            )

        if daily_sessions.empty:
            return {'error': 'No session data found', 'ticker': ticker}

            # Calculate opening momentum (first N minutes of session)
        opening_returns = self._calculate_session_window_returns(
            daily_sessions,
            session_start,
            opening_window,
            price_col,
            is_synthetic,
            session_limit_start=session_start,  # Clamp lower bound
            session_limit_end=session_end  # Clamp upper bound
        )

        # Calculate closing momentum (last N minutes of session)
        closing_returns = self._calculate_session_window_returns(
            daily_sessions,
            session_end,
            closing_window,
            price_col,
            is_synthetic,
            from_end=True,
            session_limit_start=session_start,  # Clamp lower bound
            session_limit_end=session_end  # Clamp upper bound
        )

        # Calculate full session returns
        full_session_returns = self._calculate_full_session_returns(
            daily_sessions,
            session_start,
            session_end,
            price_col,
            is_synthetic
        )

        # Short-term momentum (N-day rolling returns)
        # IMPORTANT: Use shift(1) to avoid lookahead bias - st_momentum should only include PREVIOUS days
        # Without shift, today's full_session_returns would be included in st_momentum, creating spurious correlations
        st_momentum = full_session_returns.shift(1).rolling(window=momentum_days).sum()

        # Returns for the rest of the session (used by momentum_sc patterns)
        rest_of_session_returns = (full_session_returns - closing_returns).dropna()

        # Correlation analysis (skip if batching correlations across tickers)
        if skip_correlations:
            correlation_stats = None
        else:
            correlation_stats = self._calculate_momentum_correlations(
                opening_returns,
                closing_returns,
                st_momentum,
                full_session_returns,
            )

        # Day-of-week breakdown for momentum effects
        dow_momentum_stats = self._analyze_momentum_by_dayofweek(
            opening_returns,
            closing_returns,
            full_session_returns,
            st_momentum
        )

        # Volatility analysis (if requested)
        volatility_stats = None
        if test_vol:
            volatility_stats = self._analyze_volatility_patterns(
                opening_returns,
                closing_returns,
                full_session_returns
            )

        # Compile results with both statistics and raw data for plotting
        results = {
            'ticker': ticker,
            'is_synthetic': is_synthetic,
            'session_start': str(session_start),
            'session_end': str(session_end),
            'sess_start_hrs': int(start_hrs),
            'sess_start_minutes': int(start_mins),
            'sess_end_hrs': int(closing_hours),
            'sess_end_minutes': int(closing_minutes),
            'opening_window_minutes': opening_window_minutes,
            'closing_window_minutes': closing_window_minutes,
            'n_sessions': len(full_session_returns.dropna()),
            'opening_momentum': {
                'mean': float(opening_returns.mean()),
                'std': float(opening_returns.std()),
                'sharpe': float(opening_returns.mean() / opening_returns.std()) if opening_returns.std() > 0 else np.nan,
                'skew': float(opening_returns.skew()),
                'positive_pct': float((opening_returns > 0).sum() / len(opening_returns)) if len(opening_returns) > 0 else np.nan
            },
            'closing_momentum': {
                'mean': float(closing_returns.mean()),
                'std': float(closing_returns.std()),
                'sharpe': float(closing_returns.mean() / closing_returns.std()) if closing_returns.std() > 0 else np.nan,
                'skew': float(closing_returns.skew()),
                'positive_pct': float((closing_returns > 0).sum() / len(closing_returns)) if len(closing_returns) > 0 else np.nan
            },
            'full_session': {
                'mean': float(full_session_returns.mean()),
                'std': float(full_session_returns.std()),
                'sharpe': float(full_session_returns.mean() / full_session_returns.std()) if full_session_returns.std() > 0 else np.nan,
                'skew': float(full_session_returns.skew())
            },
            'st_momentum': {
                'mean': float(st_momentum.mean()),
                'std': float(st_momentum.std()),
                f'{momentum_days}d_autocorr': float(st_momentum.autocorr(lag=1))
            },
            # Store raw return series for plotting
            'return_series': {
                'opening_returns': opening_returns,
                'closing_returns': closing_returns,
                'full_session_returns': full_session_returns,
                'st_momentum': st_momentum,
                'rest_of_session_returns': rest_of_session_returns,
                'session_pre_close_returns': rest_of_session_returns,
            }
        }

        # Add correlation stats if calculated
        if correlation_stats is not None:
            results['correlations'] = correlation_stats

        # Add day-of-week momentum breakdown if available
        if dow_momentum_stats:
            results['momentum_by_dayofweek'] = dow_momentum_stats

        # Add volatility analysis if available
        if volatility_stats:
            results['volatility'] = volatility_stats

        return results

    def _extract_session_data(
            self,
            data: pd.DataFrame,
            session_start: time,
            session_end: time,
            price_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        [PERFORMANCE FIX: No more data.index.time]
        Filters bar data to include only times between session_start and session_end.
        Uses minutes-from-midnight for vectorized filtering.
        """
        if data.empty:
            return data

        # Pre-calculate minutes from midnight ONCE if not already done
        if 'minutes_from_midnight' not in data.columns:
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                data.index = pd.to_datetime(data.index, errors='coerce')

            # If tz-naive, localize to default timezone
            if data.index.tz is None:
                data = data.copy()
                data.index = data.index.tz_localize("America/Chicago", ambiguous='infer', nonexistent='shift_forward')

            data['minutes_from_midnight'] = data.index.hour * 60 + data.index.minute

        start_min = session_start.hour * 60 + session_start.minute
        end_min = session_end.hour * 60 + session_end.minute

        # Vectorized integer comparison
        if start_min <= end_min:
            mask = (data["minutes_from_midnight"] >= start_min) & (
                    data["minutes_from_midnight"] <= end_min
            )
        else:
            # Handle overnight sessions (e.g., 20:00 to 04:00)
            mask = (data["minutes_from_midnight"] >= start_min) | (
                    data["minutes_from_midnight"] <= end_min
            )

        # Drop the helper column if it was created temporarily
        if 'minutes_from_midnight' in data.columns and not data.columns.tolist().index(
                'minutes_from_midnight') < data.columns.size:
            data = data.drop(columns=['minutes_from_midnight'])

        return data[mask]

    def _calculate_session_window_returns(
        self,
        data: pd.DataFrame,
        anchor_time: time,
        window: timedelta,
        price_col: str,
        is_synthetic: bool,
        *,
        from_end: bool = False,
        session_limit_start: Optional[time] = None,
        session_limit_end: Optional[time] = None,
    ) -> pd.Series:
        """Compute per-session returns over a fixed intraday window.

        Parameters
        ----------
        data : pd.DataFrame
            Intraday bars already filtered to the session window.
        anchor_time : time
            Clock time used as the anchor for the window. When ``from_end`` is
            False the window starts at ``anchor_time``; when True the window ends
            at ``anchor_time``.
        window : timedelta
            Duration of the window to evaluate.
        price_col : str
            Column to use for price/spread values.
        is_synthetic : bool
            Whether the ticker is a synthetic spread (uses price differences
            instead of log returns).
        from_end : bool, optional
            If True, measure the window backwards from ``anchor_time`` (used for
            closing momentum). Defaults to False.
        session_limit_start, session_limit_end : Optional[time]
            Optional bounds that clamp the computed window to the known session
            start/end times. This prevents negative window ranges when the
            requested slice exceeds the session boundaries.

        Returns
        -------
        pd.Series
            Series of window returns indexed by session date.
        """

        if data.empty:
            return pd.Series(dtype=float)

        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)

        # Ensure timezone information is preserved for correct grouping
        timestamps = data.index
        tz = timestamps.tz

        def _to_minutes(value: time) -> int:
            return value.hour * 60 + value.minute

        anchor_minutes = _to_minutes(anchor_time)
        window_minutes = int(window.total_seconds() // 60)
        start_bound = anchor_minutes
        end_bound = anchor_minutes + window_minutes

        if from_end:
            start_bound = anchor_minutes - window_minutes
            end_bound = anchor_minutes

        if session_limit_start is not None:
            start_bound = max(start_bound, _to_minutes(session_limit_start))
        if session_limit_end is not None:
            end_bound = min(end_bound, _to_minutes(session_limit_end))

        minutes = data.index.hour * 60 + data.index.minute
        mask = (minutes >= start_bound) & (minutes <= end_bound)
        if not mask.any():
            return pd.Series(dtype=float)

        session_series = pd.Series(dtype=float)
        grouped = data.loc[mask].groupby(pd.Grouper(freq="1D"))

        for session_date, frame in grouped:
            if frame.empty:
                continue
            frame = frame.sort_index()
            start_price = frame[price_col].iloc[0]
            end_price = frame[price_col].iloc[-1]

            if is_synthetic:
                window_return = end_price - start_price
            else:
                if start_price <= 0 or end_price <= 0:
                    continue
                window_return = np.log(end_price / start_price)

            ts = session_date
            if tz is not None and ts.tzinfo is None:
                ts = ts.tz_localize(tz)

            session_series.loc[ts] = float(window_return)

        return session_series.sort_index()

    def _calculate_full_session_returns(
        self,
        session_data: pd.DataFrame,
        session_start: time,
        session_end: time,
        price_col: str,
        is_synthetic: bool
    ) -> pd.Series:
        """
        Calculate return from session start to session end for each day using vectorized operations.

        Parameters:
        -----------
        session_data : pd.DataFrame
            Data filtered to session hours
        session_start : time
            Session start time
        session_end : time
            Session end time
        price_col : str
            Column name for prices/spread values
        is_synthetic : bool
            Whether this is a synthetic spread

        Returns:
        --------
        pd.Series
            Daily session returns
        """
        if session_data.empty:
            return pd.Series(dtype=float)

        session_data = self._ensure_intraday_time_metadata(session_data)

        # Group by date and get first/last prices for the full session
        grouped = session_data.groupby('_session_date')[price_col]
        start_prices = grouped.first()
        end_prices = grouped.last()

        # Calculate returns vectorized
        if is_synthetic:
            # For spreads, use change in spread value
            returns = end_prices - start_prices
        else:
            # For prices, use log return
            # Filter out non-positive prices
            valid_mask = (start_prices > 0) & (end_prices > 0)
            returns = pd.Series(index=start_prices.index, dtype=float)
            returns[valid_mask] = np.log(end_prices[valid_mask] / start_prices[valid_mask])

        return returns.dropna()

    def _calculate_momentum_correlations(
        self,
        opening_returns: pd.Series,
        closing_returns: pd.Series,
        st_momentum: pd.Series,
        full_session_returns: pd.Series,
    ) -> Dict[str, float]:
        """Calculate correlations between different momentum measures using GPU batch operations."""
        # Align series
        combined = pd.DataFrame({
            'open': opening_returns,
            'close': closing_returns,
            'st_mom': st_momentum,
            'full': full_session_returns,
        }).dropna()

        if len(combined) < 10:
            return {
                'open_close_corr': np.nan,
                'open_close_pvalue': np.nan,
                'open_st_mom_corr': np.nan,
                'close_st_mom_corr': np.nan,
                'close_vs_rest_corr': np.nan,
                'close_vs_rest_pvalue': np.nan,
            }

        # Calculate rest of session first (needed for batch)
        rest_of_session = combined['full'] - combined['close']

        # Use GPU batch correlation for all correlations at once
        # Prepare batch: rows are different y series to correlate
        try:
            from ..utils.gpu_utils import gpu_batch_pearsonr

            # Batch correlations with 'close' as reference (base case)
            y_batch_close = np.vstack([
                combined['open'].values,  # open_close
                combined['st_mom'].values,  # close_st_mom
                rest_of_session.values if len(rest_of_session.unique()) > 1 else np.full(len(rest_of_session), np.nan)
            ])

            close_corrs, close_pvals = gpu_batch_pearsonr(
                combined['close'].values,
                y_batch_close,
                use_gpu=self.use_gpu,
                device_id=self.gpu_device_id,
                return_pvalues=True
            )

            # Separate batch result with 'open' as reference
            y_batch_open = np.vstack([combined['st_mom'].values])
            open_corrs, _ = gpu_batch_pearsonr(
                combined['open'].values,
                y_batch_open,
                use_gpu=self.use_gpu,
                device_id=self.gpu_device_id,
                return_pvalues=False
            )

            # Extract results from batched calculations
            open_close_corr = close_corrs[0]
            open_close_pval = close_pvals[0]
            close_st_corr = close_corrs[1]
            open_st_corr = open_corrs[0]

            if len(rest_of_session.unique()) > 1:
                close_rest_corr = close_corrs[2]
                close_rest_pval = close_pvals[2]
            else:
                close_rest_corr = np.nan
                close_rest_pval = np.nan

        except Exception as e:
            # Fallback to CPU scipy if GPU fails
            self.logger.debug(f"GPU batch correlation failed, using CPU fallback: {e}")

            open_close_corr, open_close_pval = stats.pearsonr(combined['open'], combined['close'])
            open_st_corr, _ = stats.pearsonr(combined['open'], combined['st_mom'])
            close_st_corr, _ = stats.pearsonr(combined['close'], combined['st_mom'])

            if len(rest_of_session.unique()) <= 1:
                close_rest_corr = np.nan
                close_rest_pval = np.nan
            else:
                try:
                    close_rest_corr, close_rest_pval = stats.pearsonr(combined['close'], rest_of_session)
                except ValueError:
                    close_rest_corr = np.nan
                    close_rest_pval = np.nan

        return {
            'open_close_corr': float(open_close_corr),
            'open_close_pvalue': float(open_close_pval),
            'open_st_mom_corr': float(open_st_corr),
            'close_st_mom_corr': float(close_st_corr),
            'close_vs_rest_corr': float(close_rest_corr) if not np.isnan(close_rest_corr) else np.nan,
            'close_vs_rest_pvalue': float(close_rest_pval) if not np.isnan(close_rest_pval) else np.nan,
            'n_observations': len(combined)
        }

    def _batch_momentum_correlations_cross_ticker(
        self,
        ticker_returns_data: Dict[str, Dict[str, pd.Series]],
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute momentum correlations for all tickers in batch using cross-ticker GPU batching.

        This method extracts aligned return series from all tickers and computes correlations
        simultaneously using GPU batch operations, providing significant speedup compared to
        sequential per-ticker correlation calculations.

        Parameters
        ----------
        ticker_returns_data : Dict[str, Dict[str, pd.Series]]
            Dictionary mapping ticker -> {'opening_returns': Series, 'closing_returns': Series,
            'st_momentum': Series, 'full_session_returns': Series}

        Returns
        -------
        Dict[str, Dict[str, float]] or None
            Dictionary mapping ticker -> correlation stats, or None if batching fails
            (e.g., arrays have different lengths).

        Notes
        -----
        - Requires all tickers to have aligned data (same length arrays)
        - Falls back to None if alignment check fails
        - Uses gpu_stats.batch_pearson_correlation for GPU acceleration
        - Computes 4 correlations per ticker simultaneously across all tickers
        """
        try:
            if not ticker_returns_data or len(ticker_returns_data) < 2:
                return None

            # Step 1: Check alignment - all arrays must have same length
            tickers = list(ticker_returns_data.keys())
            lengths = []
            for ticker in tickers:
                data = ticker_returns_data[ticker]
                # Align all series for this ticker
                combined = pd.DataFrame({
                    'open': data['opening_returns'],
                    'close': data['closing_returns'],
                    'st_mom': data['st_momentum'],
                    'full': data['full_session_returns'],
                }).dropna()

                if len(combined) < 10:
                    return None  # Insufficient data for this ticker
                lengths.append(len(combined))

            # Check if all tickers have same aligned length
            if len(set(lengths)) != 1:
                return None  # Can't batch - different lengths after alignment

            # Step 2: Extract aligned arrays for each ticker
            ticker_aligned_data = {}
            for ticker in tickers:
                data = ticker_returns_data[ticker]
                combined = pd.DataFrame({
                    'open': data['opening_returns'],
                    'close': data['closing_returns'],
                    'st_mom': data['st_momentum'],
                    'full': data['full_session_returns'],
                }).dropna()
                ticker_aligned_data[ticker] = combined

            n_observations = lengths[0]

            # Step 3: Prepare batch arrays
            open_arrays = [ticker_aligned_data[t]['open'].values for t in tickers]
            close_arrays = [ticker_aligned_data[t]['close'].values for t in tickers]
            st_mom_arrays = [ticker_aligned_data[t]['st_mom'].values for t in tickers]

            # Calculate rest_of_session for each ticker
            rest_arrays = []
            for t in tickers:
                combined = ticker_aligned_data[t]
                rest = combined['full'] - combined['close']
                rest_arrays.append(rest.values)

            # Step 4: Batch compute correlations using gpu_stats
            from .gpu_stats import batch_pearson_correlation

            # Correlation 1: open vs close
            open_close_corrs, open_close_pvals = batch_pearson_correlation(
                open_arrays, close_arrays, use_gpu=self.use_gpu
            )

            # Correlation 2: open vs st_momentum
            open_st_corrs, _ = batch_pearson_correlation(
                open_arrays, st_mom_arrays, use_gpu=self.use_gpu
            )

            # Correlation 3: close vs st_momentum
            close_st_corrs, _ = batch_pearson_correlation(
                close_arrays, st_mom_arrays, use_gpu=self.use_gpu
            )

            # Correlation 4: close vs rest_of_session
            close_rest_corrs, close_rest_pvals = batch_pearson_correlation(
                close_arrays, rest_arrays, use_gpu=self.use_gpu
            )

            # Step 5: Build result dictionary
            results = {}
            for i, ticker in enumerate(tickers):
                results[ticker] = {
                    'open_close_corr': float(open_close_corrs[i]),
                    'open_close_pvalue': float(open_close_pvals[i]),
                    'open_st_mom_corr': float(open_st_corrs[i]),
                    'close_st_mom_corr': float(close_st_corrs[i]),
                    'close_vs_rest_corr': float(close_rest_corrs[i]),
                    'close_vs_rest_pvalue': float(close_rest_pvals[i]),
                    'n_observations': n_observations
                }

            if self.logger:
                self.logger.debug(
                    f"Cross-ticker batching succeeded: {len(tickers)} tickers, "
                    f"{n_observations} observations"
                )

            return results

        except Exception as e:
            if self.logger:
                self.logger.debug(f"Cross-ticker batching failed: {e}")
            return None  # Fall back to sequential

    def _analyze_momentum_by_dayofweek(
            self,
            opening_returns: pd.Series,
            closing_returns: pd.Series,
            full_session_returns: pd.Series,
            st_momentum: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyzes momentum by day of week for multiple return types using GPU batch operations.

        Parameters
        ----------
        opening_returns : pd.Series
            Opening momentum returns indexed by date
        closing_returns : pd.Series
            Closing momentum returns indexed by date
        full_session_returns : pd.Series
            Full session returns indexed by date
        st_momentum : pd.Series
            Short-term momentum values indexed by date

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with keys: 'opening', 'closing', 'full_session', 'st_momentum'
            Each DataFrame contains columns: day_of_week, mean, std, t_stat, p_value, count, skew
        """
        from ..utils.gpu_utils import gpu_batch_ttest

        # Combine all series into a single DataFrame
        combined = pd.DataFrame({
            'opening': opening_returns,
            'closing': closing_returns,
            'full_session': full_session_returns,
            'st_momentum': st_momentum
        })

        # Drop rows where all values are NaN
        combined = combined.dropna(how='all')

        if combined.empty:
            return {
                'opening': pd.DataFrame(),
                'closing': pd.DataFrame(),
                'full_session': pd.DataFrame(),
                'st_momentum': pd.DataFrame()
            }

        # Ensure index is DatetimeIndex before accessing weekday
        if not isinstance(combined.index, pd.DatetimeIndex):
            # Convert index to datetime if it isn't already
            combined.index = pd.to_datetime(combined.index)

        # Now safely access weekday property
        combined['weekday'] = combined.index.weekday

        results = {}

        for col_name in ['opening', 'closing', 'full_session', 'st_momentum']:
            # Group by weekday and calculate basic stats
            grouped = combined.groupby('weekday')[col_name]
            stats_df = grouped.agg(['mean', 'std', 'count', 'skew']).reset_index()
            stats_df.rename(columns={'weekday': 'day_of_week'}, inplace=True)

            # Use GPU batch t-tests for all days simultaneously
            day_data = []
            valid_days = []
            for dow in range(7):
                mask = combined['weekday'] == dow
                day_values = combined.loc[mask, col_name].dropna().values
                if len(day_values) > 1:  # Need at least 2 samples for t-test
                    day_data.append(day_values)
                    valid_days.append(dow)

            if day_data:
                # Pad arrays to same length for batch processing
                max_len = max(len(arr) for arr in day_data)
                padded_data = []
                for arr in day_data:
                    if len(arr) < max_len:
                        padded = np.full(max_len, np.nan)
                        padded[:len(arr)] = arr
                        padded_data.append(padded)
                    else:
                        padded_data.append(arr)

                # Batch t-test on GPU
                batch_array = np.array(padded_data)
                try:
                    t_stats, p_values = gpu_batch_ttest(
                        batch_array,
                        popmean=0.0,
                        use_gpu=self.use_gpu,
                        device_id=self.gpu_device_id
                    )

                    # Map results back to stats_df
                    for i, dow in enumerate(valid_days):
                        row_idx = stats_df[stats_df['day_of_week'] == dow].index
                        if len(row_idx) > 0:
                            stats_df.loc[row_idx, 't_stat'] = t_stats[i]
                            stats_df.loc[row_idx, 'p_value'] = p_values[i]
                except Exception as e:
                    # CPU fallback
                    for i, dow in enumerate(valid_days):
                        row_idx = stats_df[stats_df['day_of_week'] == dow].index
                        if len(row_idx) > 0:
                            data = day_data[i]
                            t_stat = np.mean(data) / (np.std(data, ddof=1) / np.sqrt(len(data)))
                            p_value = stats.t.sf(np.abs(t_stat), len(data) - 1) * 2
                            stats_df.loc[row_idx, 't_stat'] = t_stat
                            stats_df.loc[row_idx, 'p_value'] = p_value
            else:
                # No valid days - compute manually
                stats_df['t_stat'] = stats_df['mean'] / (stats_df['std'] / np.sqrt(stats_df['count']))
                stats_df['p_value'] = stats.t.sf(np.abs(stats_df['t_stat']), stats_df['count'] - 1) * 2

            # Map weekday integer to name for display
            stats_df['day_of_week'] = stats_df['day_of_week'].map(lambda x: calendar.day_name[x])

            results[col_name] = stats_df

        return results
    def _analyze_volatility_patterns(
        self,
        opening_returns: pd.Series,
        closing_returns: pd.Series,
        full_session_returns: pd.Series
    ) -> Dict[str, any]:
        """
        Analyze volatility patterns and correlations between opening and closing volatility.

        Tests:
        - Correlation between opening volatility and closing volatility
        - Identifies days with unusually high or low volatility (z-score approach)
        - Analyzes volatility patterns by day of week

        Parameters:
        -----------
        opening_returns : pd.Series
            Opening momentum returns (indexed by date)
        closing_returns : pd.Series
            Closing momentum returns (indexed by date)
        full_session_returns : pd.Series
            Full session returns (indexed by date)

        Returns:
        --------
        Dict[str, any]
            Dictionary containing:
            - opening_closing_vol_correlation: Correlation coefficient between opening and closing volatility
            - vol_correlation_pvalue: P-value for significance
            - vol_correlation_interpretation: Human-readable interpretation
            - high_volatility_days: List of dates with z-score > 2
            - low_volatility_days: List of dates with z-score < -2
            - volatility_by_dayofweek: Mean volatility statistics for each weekday
            - overall_stats: Mean/std/min/max volatility statistics
        """
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Combine series into DataFrame
        df = pd.DataFrame({
            'opening': opening_returns,
            'closing': closing_returns,
            'full_session': full_session_returns
        }).dropna()

        if len(df) < 10:
            return {
                'error': 'Insufficient data for volatility analysis',
                'n': len(df)
            }

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Calculate realized volatility for opening and closing periods
        # Using rolling standard deviation (5-day window) of returns
        opening_vol = df['opening'].abs().rolling(window=5, min_periods=1).mean()
        closing_vol = df['closing'].abs().rolling(window=5, min_periods=1).mean()
        full_session_vol = df['full_session'].abs().rolling(window=5, min_periods=1).mean()

        # Align volatility series and drop NaN
        vol_df = pd.DataFrame({
            'opening_vol': opening_vol,
            'closing_vol': closing_vol,
            'full_session_vol': full_session_vol
        }).dropna()

        if len(vol_df) < 10:
            return {
                'error': 'Insufficient volatility data after calculation',
                'n': len(vol_df)
            }

        # Test correlation between opening and closing volatility using GPU batch operation
        from ..utils.gpu_utils import gpu_batch_pearsonr

        try:
            # Batch correlation (single correlation, but using GPU infrastructure)
            y_batch = vol_df['closing_vol'].values.reshape(1, -1)
            corrs, pvals = gpu_batch_pearsonr(
                vol_df['opening_vol'].values,
                y_batch,
                use_gpu=self.use_gpu,
                device_id=self.gpu_device_id,
                return_pvalues=True
            )
            opening_closing_corr = corrs[0]
            corr_pvalue = pvals[0]
        except Exception as e:
            # CPU fallback
            opening_closing_corr, corr_pvalue = stats.pearsonr(
                vol_df['opening_vol'],
                vol_df['closing_vol']
            )

        # Calculate z-scores for full session volatility to identify unusual days
        mean_vol = vol_df['full_session_vol'].mean()
        std_vol = vol_df['full_session_vol'].std()

        if std_vol > 0:
            vol_df['z_score'] = (vol_df['full_session_vol'] - mean_vol) / std_vol
        else:
            vol_df['z_score'] = 0.0

        # Identify high volatility days (z-score > 2)
        high_vol_mask = vol_df['z_score'] > 2.0
        high_vol_days = vol_df[high_vol_mask].index.strftime('%Y-%m-%d').tolist()
        high_vol_values = vol_df[high_vol_mask]['full_session_vol'].tolist()

        # Identify low volatility days (z-score < -2)
        low_vol_mask = vol_df['z_score'] < -2.0
        low_vol_days = vol_df[low_vol_mask].index.strftime('%Y-%m-%d').tolist()
        low_vol_values = vol_df[low_vol_mask]['full_session_vol'].tolist()

        # Analyze volatility by day of week
        vol_df['weekday'] = vol_df.index.dayofweek

        volatility_by_dow = {}
        for dow, day_data in vol_df.groupby('weekday'):
            if len(day_data) < 2:
                continue

            day_name = weekday_names[dow]

            volatility_by_dow[day_name] = {
                'n': len(day_data),
                'mean_opening_vol': float(day_data['opening_vol'].mean()),
                'mean_closing_vol': float(day_data['closing_vol'].mean()),
                'mean_full_session_vol': float(day_data['full_session_vol'].mean()),
                'std_full_session_vol': float(day_data['full_session_vol'].std()),
                'high_vol_days_count': int((day_data['z_score'] > 2.0).sum()),
                'low_vol_days_count': int((day_data['z_score'] < -2.0).sum())
            }

        # Overall statistics
        overall_stats = {
            'mean_opening_vol': float(vol_df['opening_vol'].mean()),
            'std_opening_vol': float(vol_df['opening_vol'].std()),
            'mean_closing_vol': float(vol_df['closing_vol'].mean()),
            'std_closing_vol': float(vol_df['closing_vol'].std()),
            'mean_full_session_vol': float(vol_df['full_session_vol'].mean()),
            'std_full_session_vol': float(vol_df['full_session_vol'].std()),
            'min_vol': float(vol_df['full_session_vol'].min()),
            'max_vol': float(vol_df['full_session_vol'].max()),
            'n_observations': len(vol_df)
        }

        # Interpretation
        if corr_pvalue < 0.05:
            if opening_closing_corr > 0.3:
                interpretation = f"Strong positive correlation (r={opening_closing_corr:.3f}) - opening volatility predicts closing volatility"
            elif opening_closing_corr > 0.1:
                interpretation = f"Moderate positive correlation (r={opening_closing_corr:.3f}) - some predictive relationship"
            elif opening_closing_corr < -0.3:
                interpretation = f"Strong negative correlation (r={opening_closing_corr:.3f}) - volatile opens lead to quiet closes"
            elif opening_closing_corr < -0.1:
                interpretation = f"Moderate negative correlation (r={opening_closing_corr:.3f}) - inverse relationship detected"
            else:
                interpretation = f"Weak but significant correlation (r={opening_closing_corr:.3f})"
        else:
            interpretation = f"No significant correlation (r={opening_closing_corr:.3f}, p={corr_pvalue:.3f})"

        # Compile results
        results = {
            'opening_closing_vol_correlation': float(opening_closing_corr),
            'vol_correlation_pvalue': float(corr_pvalue),
            'vol_correlation_significant': corr_pvalue < 0.05,
            'vol_correlation_interpretation': interpretation,
            'high_volatility_days': [
                {'date': date, 'volatility': float(vol), 'z_score': float(vol_df.loc[date, 'z_score'])}
                for date, vol in zip(high_vol_days, high_vol_values)
            ],
            'low_volatility_days': [
                {'date': date, 'volatility': float(vol), 'z_score': float(vol_df.loc[date, 'z_score'])}
                for date, vol in zip(low_vol_days, low_vol_values)
            ],
            'n_high_vol_days': len(high_vol_days),
            'n_low_vol_days': len(low_vol_days),
            'volatility_by_dayofweek': volatility_by_dow,
            'overall_stats': overall_stats
        }

        return results

    def _analyze_volume_patterns(
        self,
        session_data: pd.DataFrame,
        session_start: time,
        session_end: time
    ) -> Dict[str, any]:
        """
        Analyze volume patterns during sessions using vectorized operations.

        Identifies volume peaks (best times for liquidity) and troughs (worst times for liquidity)
        to help determine optimal trading times.

        Returns:
        --------
        Dict with volume statistics and optimal trading times based on liquidity patterns
        """
        if 'Volume' not in session_data.columns:
            return {}

        session_data_copy = self._ensure_intraday_time_metadata(session_data)

        # Group by date and calculate aggregates in one operation
        vol_grouped = session_data_copy.groupby('_session_date')['Volume']
        total_vols = vol_grouped.sum()
        avg_vols = vol_grouped.mean()

        if len(total_vols) == 0:
            return {}

        # Analyze volume by time of day to find peaks and troughs
        time_of_day_vol = session_data_copy.groupby('_seconds_from_midnight')['Volume'].agg(['mean', 'std', 'count'])
        time_of_day_vol = time_of_day_vol[time_of_day_vol['count'] >= 5]  # Minimum 5 observations

        if len(time_of_day_vol) == 0:
            return {
                'avg_daily_total_volume': float(total_vols.mean()),
                'std_daily_total_volume': float(total_vols.std()),
                'avg_bar_volume': float(avg_vols.mean()),
                'volume_trend': float(total_vols.pct_change().mean())
            }

        # Calculate z-scores for volume at each time
        time_of_day_vol['z_score'] = (
            (time_of_day_vol['mean'] - time_of_day_vol['mean'].mean()) /
            time_of_day_vol['mean'].std()
        )

        # Find top 3 volume peaks (high liquidity times)
        peaks = time_of_day_vol.nlargest(3, 'mean')
        peak_times = [
            {
                'time': self._seconds_to_time_label(time),
                'avg_volume': float(row['mean']),
                'z_score': float(row['z_score']),
                'n_observations': int(row['count'])
            }
            for time, row in peaks.iterrows()
        ]

        # Find top 3 volume troughs (low liquidity times - avoid trading)
        troughs = time_of_day_vol.nsmallest(3, 'mean')
        trough_times = [
            {
                'time': self._seconds_to_time_label(time),
                'avg_volume': float(row['mean']),
                'z_score': float(row['z_score']),
                'n_observations': int(row['count'])
            }
            for time, row in troughs.iterrows()
        ]

        # Identify significant peaks (z-score > 1.0)
        significant_peaks = time_of_day_vol[time_of_day_vol['z_score'] > 1.0]
        optimal_trading_times = [self._seconds_to_time_label(t) for t in significant_peaks.index]

        # Identify times to avoid (z-score < -1.0)
        avoid_times_data = time_of_day_vol[time_of_day_vol['z_score'] < -1.0]
        avoid_trading_times = [self._seconds_to_time_label(t) for t in avoid_times_data.index]

        # Calculate volume concentration (what % of volume occurs in top 20% of bars)
        top_20_pct_threshold = time_of_day_vol['mean'].quantile(0.8)
        top_20_pct_volume = time_of_day_vol[time_of_day_vol['mean'] >= top_20_pct_threshold]['mean'].sum()
        total_avg_volume = time_of_day_vol['mean'].sum()
        volume_concentration = float(top_20_pct_volume / total_avg_volume) if total_avg_volume > 0 else 0.0

        return {
            # Overall volume statistics
            'avg_daily_total_volume': float(total_vols.mean()),
            'std_daily_total_volume': float(total_vols.std()),
            'avg_bar_volume': float(avg_vols.mean()),
            'volume_trend': float(total_vols.pct_change().mean()),

            # Volume concentration metrics
            'volume_concentration': volume_concentration,
            'n_time_periods_analyzed': len(time_of_day_vol),

            # Peak volume times (best for trading - high liquidity)
            'volume_peaks': peak_times,
            'optimal_trading_times': optimal_trading_times,

            # Trough volume times (worst for trading - low liquidity)
            'volume_troughs': trough_times,
            'avoid_trading_times': avoid_trading_times,

            # Summary statistics
            'peak_to_trough_ratio': float(peaks['mean'].mean() / troughs['mean'].mean()) if troughs['mean'].mean() > 0 else np.nan,
            'volume_volatility_by_time': float(time_of_day_vol['mean'].std() / time_of_day_vol['mean'].mean()) if time_of_day_vol['mean'].mean() > 0 else np.nan
        }

    def _analyze_dayofweek_patterns(
        self,
        data: pd.DataFrame,
        price_col: str,
        is_synthetic: bool
    ) -> Dict[str, Dict]:
        """
        Analyze return and volatility patterns by day of week.

        Parameters:
        -----------
        data : pd.DataFrame
            Price/spread data
        price_col : str
            Column name for prices
        is_synthetic : bool
            Whether data is synthetic spread

        Returns:
        --------
        Dict with 'returns' and 'volatility' keys, each containing weekday stats
        """
        # Calculate returns
        if is_synthetic:
            returns = data[price_col].diff()
        else:
            valid_mask = data[price_col] > 0
            returns = pd.Series(index=data.index, dtype=float)
            prices = data[price_col][valid_mask]
            returns[valid_mask] = np.log(prices / prices.shift(1))

        returns = returns.dropna()

        # Ensure returns has DatetimeIndex before accessing dayofweek
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        # Add weekday labels
        df = pd.DataFrame({'return': returns, 'weekday': returns.index.dayofweek})
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        weekday_grouped = df.groupby('weekday')['return']

        # Returns analysis by weekday
        returns_stats = {}
        weekday_returns = []

        for dow, day_data in weekday_grouped:
            if len(day_data) < 2:
                continue

            weekday_returns.append(day_data.values)
            day_name = weekday_names[dow]

            returns_stats[day_name] = {
                'n': len(day_data),
                'mean': float(day_data.mean()),
                'std': float(day_data.std()),
                'sharpe': float(day_data.mean() / day_data.std()) if day_data.std() > 0 else np.nan,
                'skew': float(day_data.skew()),
                'positive_pct': float((day_data > 0).sum() / len(day_data)),
                't_stat': float(day_data.mean() / (day_data.std() / np.sqrt(len(day_data)))) if day_data.std() > 0 else np.nan
            }

        # ANOVA test across weekdays
        if len(weekday_returns) >= 3:
            from scipy.stats import f_oneway
            f_stat, p_value = f_oneway(*weekday_returns)
            returns_stats['anova'] = {
                'f_stat': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        # Volatility analysis by weekday
        volatility_stats = {}

        for dow, day_data in weekday_grouped:
            if len(day_data) < 2:
                continue

            day_name = weekday_names[dow]
            realized_vol = day_data.std()

            volatility_stats[day_name] = {
                'n': len(day_data),
                'realized_vol': float(realized_vol),
                'mean_abs_return': float(day_data.abs().mean()),
                'vol_of_vol': float(day_data.rolling(5).std().std()) if len(day_data) > 10 else np.nan
            }

        return {
            'returns': returns_stats,
            'volatility': volatility_stats
        }

    def _test_time_predictability(
        self,
        data: pd.DataFrame,
        price_col: str,
        target_time: time,
        is_synthetic: bool,
        period_length: Optional[timedelta] = None,
        tz: Optional[str] = None,
        include_return_series: bool = False,
    ) -> Dict[str, any]:
        """
        Test if returns at a specific time predict future returns.

        Parameters:
        -----------
        data : pd.DataFrame
            Price/spread data
        price_col : str
            Column name for prices
        target_time : time
            Time of day to analyze
        is_synthetic : bool
            Whether data is synthetic spread
        period_length : Optional[timedelta]
            Length of window to aggregate (if None, use single bar)
        tz : Optional[str]
            Timezone used for month derivation.

        Returns:
        --------
        Dict with correlation statistics for various lags and weekday breakdown for next-week correlation.
        When ``include_return_series`` is True, also includes the exact return
        series used for each correlation test to enable downstream reuse without
        recomputation.
        """
        from ..utils.seasonal import tod_mask, aggregate_window, log_returns

        # Calculate returns
        if is_synthetic:
            ret = data[price_col].diff()
        else:
            ret = log_returns(data[price_col])

        ret = ret.dropna()

        # Extract returns at target time
        mask = tod_mask(ret.index, target_time.strftime("%H:%M"))

        if period_length:
            # Aggregate window
            window_bars = int(period_length.total_seconds() / 60 / 5)  # Assuming 5-min bars
            daily_returns = aggregate_window(ret, mask, window_bars)
        else:
            # Single bar at target time
            daily_returns = ret[mask]

        # Ensure we have a DatetimeIndex before grouping by date
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            daily_returns.index = pd.to_datetime(daily_returns.index)

        # Group by date
        daily_returns = daily_returns.groupby(pd.Grouper(freq='1D')).sum()

        months_meta = self._build_months_metadata(daily_returns.index, tz)

        if len(daily_returns) < 20:
            return {
                'n': len(daily_returns),
                'next_day_corr': np.nan,
                'next_day_pvalue': np.nan,
                'next_week_corr': np.nan,
                'next_week_pvalue': np.nan,
                'error': 'Insufficient data',
                **months_meta,
            }

        # Batch correlations for next-day and next-week using GPU
        from ..utils.gpu_utils import gpu_batch_pearsonr

        x_1d = daily_returns[:-1].values
        y_1d = daily_returns[1:].values

        # Next-week correlation (5-7 business days)
        lag_week = 5
        x_1w = daily_returns[:-lag_week].values if len(daily_returns) > lag_week + 10 else np.array([])
        y_1w = daily_returns[lag_week:].values if len(daily_returns) > lag_week + 10 else np.array([])

        # Batch process both correlations together
        if len(x_1d) >= 20 and len(x_1w) >= 20:
            try:
                # Pad to same length for batch processing
                max_len = max(len(y_1d), len(y_1w))
                y_1d_padded = np.full(max_len, np.nan)
                y_1d_padded[:len(y_1d)] = y_1d
                y_1w_padded = np.full(max_len, np.nan)
                y_1w_padded[:len(y_1w)] = y_1w
                x_padded = np.full(max_len, np.nan)
                x_padded[:len(x_1d)] = x_1d  # Use x_1d as reference (longer than x_1w)

                y_batch = np.vstack([y_1d_padded, y_1w_padded])
                corrs, pvals = gpu_batch_pearsonr(
                    x_padded,
                    y_batch,
                    use_gpu=self.use_gpu,
                    device_id=self.gpu_device_id,
                    return_pvalues=True
                )
                r_1d, p_1d = corrs[0], pvals[0]
                r_1w, p_1w = corrs[1], pvals[1]
            except Exception as e:
                # CPU fallback
                r_1d, p_1d = stats.pearsonr(x_1d, y_1d)
                r_1w, p_1w = stats.pearsonr(x_1w, y_1w)
        elif len(x_1d) >= 20:
            r_1d, p_1d = stats.pearsonr(x_1d, y_1d)
            r_1w, p_1w = np.nan, np.nan
        else:
            r_1d, p_1d = np.nan, np.nan
            r_1w, p_1w = np.nan, np.nan

        # Analyze which days of the week show the pattern most strongly
        weekday_analysis = None
        if not np.isnan(r_1w) and p_1w < 0.05:
            weekday_analysis = self._analyze_weekday_prevalence(daily_returns, lag_week)

        result = {
            'n': len(daily_returns),
            'mean_return': float(daily_returns.mean()),
            'std_return': float(daily_returns.std()),
            'next_day_corr': float(r_1d) if not np.isnan(r_1d) else np.nan,
            'next_day_pvalue': float(p_1d) if not np.isnan(p_1d) else np.nan,
            'next_day_significant': p_1d < 0.05 if not np.isnan(p_1d) else False,
            'next_week_corr': float(r_1w) if not np.isnan(r_1w) else np.nan,
            'next_week_pvalue': float(p_1w) if not np.isnan(p_1w) else np.nan,
            'next_week_significant': p_1w < 0.05 if not np.isnan(p_1w) else False,
            **months_meta,
        }

        # Add weekday prevalence if available
        if weekday_analysis is not None:
            result['weekday_prevalence'] = weekday_analysis

        if include_return_series:
            def _serialize(series: pd.Series) -> Dict[str, float]:
                return {
                    str(idx): float(val)
                    for idx, val in series.dropna().items()
                }

            serialized: Dict[str, Any] = {
                'daily_returns': _serialize(daily_returns),
                'target_time': target_time.strftime("%H:%M"),
            }

            x_1d_series = pd.Series(x_1d, index=daily_returns.index[:-1]) if len(x_1d) else pd.Series(dtype=float)
            y_1d_series = pd.Series(y_1d, index=daily_returns.index[1:]) if len(y_1d) else pd.Series(dtype=float)
            serialized['next_day'] = {
                'predictor': _serialize(x_1d_series),
                'response': _serialize(y_1d_series),
            }

            if len(x_1w) and len(y_1w):
                x_1w_series = pd.Series(x_1w, index=daily_returns.index[:len(x_1w)])
                y_1w_series = pd.Series(y_1w, index=daily_returns.index[lag_week:lag_week + len(y_1w)])
                serialized['next_week'] = {
                    'predictor': _serialize(x_1w_series),
                    'response': _serialize(y_1w_series),
                }

            result['predictor_returns'] = serialized

        return result

    def _compute_weekend_hedging_pattern(
        self,
        session_data: pd.DataFrame,
        session_start: time,
        session_end: time,
        price_col: str,
        is_synthetic: bool,
        tz: str,
        pattern_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Compute Friday→Monday weekend hedging relationship."""

        full_session_returns = self._calculate_full_session_returns(
            session_data,
            session_start,
            session_end,
            price_col,
            is_synthetic,
        )

        if full_session_returns.empty:
            return None

        returns = full_session_returns.sort_index()
        returns.index = pd.to_datetime(returns.index)
        friday_mask = returns.index.dayofweek == 4

        if not friday_mask.any():
            return None

        next_dates = returns.index.to_series().shift(-1)
        next_weekday = next_dates.dt.dayofweek
        monday_values = returns.shift(-1)
        valid_mask = friday_mask & (next_weekday == 0)

        if valid_mask.sum() < 5:
            return None

        friday_returns = returns.loc[valid_mask]
        monday_returns = monday_values.loc[valid_mask]

        # GPU-accelerated correlation calculation
        from ..utils.gpu_utils import gpu_batch_pearsonr

        try:
            y_batch = monday_returns.values.reshape(1, -1)
            corrs, pvals = gpu_batch_pearsonr(
                friday_returns.values,
                y_batch,
                use_gpu=self.use_gpu,
                device_id=self.gpu_device_id,
                return_pvalues=True
            )
            corr, p_val = corrs[0], pvals[0]
        except (ValueError, Exception):
            # CPU fallback
            try:
                corr, p_val = stats.pearsonr(friday_returns.values, monday_returns.values)
            except ValueError:
                corr, p_val = np.nan, np.nan

        significance_level = 0.05
        if not np.isfinite(p_val) or p_val >= significance_level:
            return None

        pos_mask = friday_returns > 0
        neg_mask = friday_returns < 0
        mean_monday_pos = float(monday_returns[pos_mask].mean()) if pos_mask.any() else np.nan
        mean_monday_neg = float(monday_returns[neg_mask].mean()) if neg_mask.any() else np.nan

        overall_mean = float(monday_returns.mean()) if len(monday_returns) else 0.0
        if np.isnan(overall_mean) or overall_mean == 0:
            bias = 'neutral'
        else:
            bias = 'long' if overall_mean > 0 else 'short'

        months_meta = self._build_months_metadata(friday_returns.index, tz)

        pattern = {
            'type': 'weekend_hedging',
            'pattern_type': 'weekend_hedging',
            'weekday': 'Friday->Monday',
            'n': int(valid_mask.sum()),
            'corr_Fri_Mon': float(corr) if not np.isnan(corr) else np.nan,
            'p_value': float(p_val),
            'mean_Mon_given_Fri_pos': mean_monday_pos,
            'mean_Mon_given_Fri_neg': mean_monday_neg,
            'friday_mean_return': float(friday_returns.mean()),
            'monday_mean_return': float(monday_returns.mean()),
            'bias': bias,
            'description': 'Friday return vs Monday session return linkage',
            'strength': abs(corr) if not np.isnan(corr) else 0.0,
        }

        pattern.update(pattern_context)
        pattern.update(months_meta)

        return pattern

    def _analyze_weekday_prevalence(
        self,
        daily_returns: pd.Series,
        lag_week: int = 5
    ) -> Dict[str, any]:
        """
        Analyze which days of the week show the strongest predictive relationship.

        Parameters:
        -----------
        daily_returns : pd.Series
            Daily returns with date index
        lag_week : int
            Lag period for prediction (default: 5 days)

        Returns:
        --------
        Dict with weekday-specific correlations and statistics
        """
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Convert index to datetime if needed
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            daily_returns.index = pd.to_datetime(daily_returns.index)

        # Create DataFrame with current and lagged returns
        df = pd.DataFrame({
            'current': daily_returns,
            'lagged': daily_returns.shift(lag_week),
            'weekday': daily_returns.index.dayofweek
        }).dropna()

        if len(df) < 10:
            return {
                'error': 'Insufficient data for weekday analysis',
                'n': len(df)
            }

        # Calculate correlation for each weekday using GPU batch processing
        from ..utils.gpu_utils import gpu_batch_pearsonr

        weekday_correlations = {}
        weekday_counts = {}

        # Prepare batch data for all weekdays
        valid_days = []
        day_data_list = []
        for dow in range(5):  # Monday=0 to Friday=4
            day_data = df[df['weekday'] == dow]
            if len(day_data) >= 5:  # Need minimum 5 observations
                valid_days.append(dow)
                day_data_list.append(day_data)

        if valid_days:
            # Batch process all weekday correlations
            try:
                # Pad arrays to same length
                max_len = max(len(dd) for dd in day_data_list)
                x_batch = []
                y_batch = []

                for day_data in day_data_list:
                    lagged_padded = np.full(max_len, np.nan)
                    current_padded = np.full(max_len, np.nan)
                    lagged_padded[:len(day_data)] = day_data['lagged'].values
                    current_padded[:len(day_data)] = day_data['current'].values
                    x_batch.append(lagged_padded)
                    y_batch.append(current_padded)

                x_batch_array = np.array(x_batch)
                y_batch_array = np.array(y_batch)

                # Process in batch - correlate each x with corresponding y
                # Since gpu_batch_pearsonr expects (n_samples,) and (n_features, n_samples),
                # we need to process each weekday individually but can prepare them together
                corrs_list = []
                pvals_list = []
                for i in range(len(valid_days)):
                    y_single = y_batch_array[i].reshape(1, -1)
                    corrs, pvals = gpu_batch_pearsonr(
                        x_batch_array[i],
                        y_single,
                        use_gpu=self.use_gpu,
                        device_id=self.gpu_device_id,
                        return_pvalues=True
                    )
                    corrs_list.append(corrs[0])
                    pvals_list.append(pvals[0])

                # Map results back to weekday_correlations
                for i, dow in enumerate(valid_days):
                    day_data = day_data_list[i]
                    corr = corrs_list[i]
                    p_val = pvals_list[i]
                    months_meta = self._build_months_metadata(day_data.index, None)
                    weekday_correlations[weekday_names[dow]] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'n': len(day_data),
                        'significant': p_val < 0.05,
                        'abs_correlation': abs(corr),
                        **months_meta,
                    }
                    weekday_counts[weekday_names[dow]] = len(day_data)
            except Exception as e:
                # CPU fallback - process individually
                for i, dow in enumerate(valid_days):
                    day_data = day_data_list[i]
                    try:
                        corr, p_val = stats.pearsonr(day_data['lagged'], day_data['current'])
                        months_meta = self._build_months_metadata(day_data.index, None)
                        weekday_correlations[weekday_names[dow]] = {
                            'correlation': float(corr),
                            'p_value': float(p_val),
                            'n': len(day_data),
                            'significant': p_val < 0.05,
                            'abs_correlation': abs(corr),
                            **months_meta,
                        }
                        weekday_counts[weekday_names[dow]] = len(day_data)
                    except:
                        continue

        if not weekday_correlations:
            return {
                'error': 'No weekdays with sufficient data',
                'n': len(df)
            }

        # Identify strongest days (by absolute correlation)
        sorted_days = sorted(
            weekday_correlations.items(),
            key=lambda x: x[1]['abs_correlation'],
            reverse=True
        )

        # Get top days where pattern is most prevalent
        strongest_days = [day for day, stats in sorted_days[:3]]

        # Calculate mean correlation across significant days
        significant_correlations = [
            stats['correlation']
            for day, stats in weekday_correlations.items()
            if stats['significant']
        ]

        result = {
            'by_weekday': weekday_correlations,
            'strongest_days': strongest_days,
            'most_prevalent_day': sorted_days[0][0] if sorted_days else None,
            'most_prevalent_correlation': sorted_days[0][1]['correlation'] if sorted_days else np.nan,
            'n_significant_days': sum(1 for stats in weekday_correlations.values() if stats['significant']),
            'mean_correlation_significant': float(np.mean(significant_correlations)) if significant_correlations else np.nan,
            'total_observations': len(df),
            **self._build_months_metadata(df.index, None),
        }


        return result

    def _rank_seasonal_strength(
        self,
        ticker_results: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Identify and rank the strongest seasonal patterns.

        Parameters:
        -----------
        ticker_results : Dict
            Complete results for a ticker

        Returns:
        --------
        List of strongest patterns sorted by significance
        """
        patterns = []
        pattern_context = ticker_results.get('pattern_context', {}) or {}
        regime_meta = pattern_context.get('regime_filter')

        # Check day-of-week return patterns
        if 'dayofweek_returns' in ticker_results:
            dow_returns = ticker_results['dayofweek_returns']

            # Check ANOVA significance
            if 'anova' in dow_returns and dow_returns['anova']['significant']:
                patterns.append({
                    'type': 'weekday_returns',
                    'description': 'Significant day-of-week return pattern',
                    'f_stat': dow_returns['anova']['f_stat'],
                    'p_value': dow_returns['anova']['p_value'],
                    'strength': abs(dow_returns['anova']['f_stat']),
                    'months_active': ticker_results.get('months_active'),
                    'months_mask_12': ticker_results.get('months_mask_12'),
                    'months_names': ticker_results.get('months_names'),
                    'regime_filter': regime_meta,
                })

            # Check individual weekday t-stats
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                if day in dow_returns:
                    day_stats = dow_returns[day]
                    t_stat = day_stats.get('t_stat', 0)
                    if abs(t_stat) > 1.96:  # ~95% confidence
                        weekday_pattern = {
                            'type': 'weekday_mean',
                            'day': day,
                            'description': f'{day} has significant mean return',
                            'mean': day_stats['mean'],
                            't_stat': t_stat,
                            'p_value': 2 * (1 - stats.t.cdf(abs(t_stat), df=day_stats['n']-1)) if day_stats['n'] > 1 else 1.0,
                            'strength': abs(t_stat),
                            'months_active': day_stats.get('months_active'),
                            'months_mask_12': day_stats.get('months_mask_12'),
                            'months_names': day_stats.get('months_names'),
                            'regime_filter': regime_meta,
                        }
                        patterns.append(weekday_pattern)

        # Check time predictability
        if 'time_predictability' in ticker_results:
            for time_key, pred_stats in ticker_results['time_predictability'].items():
                time_label = str(pred_stats.get('time') or time_key)
                # Next-day prediction
                if pred_stats.get('next_day_significant', False):
                    payload = {
                        'correlation': pred_stats['next_day_corr'],
                        'mean': pred_stats.get('mean_return'),
                    }
                    if 'predictor_returns' in pred_stats:
                        payload['predictor_returns'] = pred_stats['predictor_returns']

                    pattern_entry = {
                        'type': 'time_predictive_nextday',
                        'time': time_label,
                        'description': f'{time_label} predicts next day return',
                        'correlation': pred_stats['next_day_corr'],
                        'p_value': pred_stats['next_day_pvalue'],
                        'strength': abs(pred_stats['next_day_corr']),
                        'months_active': pred_stats.get('months_active'),
                        'months_mask_12': pred_stats.get('months_mask_12'),
                        'months_names': pred_stats.get('months_names'),
                        'target_times_hhmm': pred_stats.get('target_times_hhmm'),
                        'period_length_min': pred_stats.get('period_length_min'),
                        'regime_filter': pred_stats.get('regime_filter', regime_meta),
                        'pattern_payload': payload,
                    }
                    patterns.append(pattern_entry)

                # Next-week prediction
                if pred_stats.get('next_week_significant', False):
                    payload = {
                        'correlation': pred_stats['next_week_corr'],
                        'mean': pred_stats.get('mean_return'),
                    }
                    if 'predictor_returns' in pred_stats:
                        payload['predictor_returns'] = pred_stats['predictor_returns']

                    pattern_entry = {
                        'type': 'time_predictive_nextweek',
                        'time': time_label,
                        'description': f'{time_label} predicts next week return',
                        'correlation': pred_stats['next_week_corr'],
                        'p_value': pred_stats['next_week_pvalue'],
                        'strength': abs(pred_stats['next_week_corr']),
                        'months_active': pred_stats.get('months_active'),
                        'months_mask_12': pred_stats.get('months_mask_12'),
                        'months_names': pred_stats.get('months_names'),
                        'target_times_hhmm': pred_stats.get('target_times_hhmm'),
                        'period_length_min': pred_stats.get('period_length_min'),
                        'regime_filter': pred_stats.get('regime_filter', regime_meta),
                    }

                    # Add weekday prevalence information if available
                    if 'weekday_prevalence' in pred_stats:
                        weekday_info = pred_stats['weekday_prevalence']
                        most_prevalent = weekday_info.get('most_prevalent_day')
                        if most_prevalent:
                            pattern_entry['most_prevalent_day'] = most_prevalent
                            strongest_days = weekday_info.get('strongest_days') or []
                            if not isinstance(strongest_days, (list, tuple)):
                                strongest_days = [strongest_days]
                            pattern_entry['strongest_days'] = list(strongest_days)
                            pattern_entry['description'] = (
                                f'{time_label} predicts next week return '
                                f'(strongest on {most_prevalent})'
                            )

                    patterns.append(pattern_entry)

        # Sort by strength (descending)
        patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)

        return patterns

    def _parse_season_months(
        self,
        months: Optional[List[int]],
        season: Optional[str]
    ) -> Optional[List[int]]:
        """
        Parse season string or month list into month numbers.

        Parameters:
        -----------
        months : Optional[List[int]]
            Explicit month numbers (1-12)
        season : Optional[str]
            Season name ('winter', 'spring', 'summer', 'fall')

        Returns:
        --------
        Optional[List[int]]
            List of month numbers, or None for all months
        """
        # Season overrides explicit months
        if season is not None:
            season_map = {
                'winter': [12, 1, 2],
                'spring': [3, 4, 5],
                'summer': [6, 7, 8],
                'fall': [9, 10, 11],
                'autumn': [9, 10, 11]  # Alias for fall
            }
            season_lower = season.lower()
            if season_lower not in season_map:
                raise ValueError(
                    f"Invalid season '{season}'. "
                    f"Must be one of: {list(season_map.keys())}"
                )
            return season_map[season_lower]

        if months is not None:
            # Validate month numbers
            if not all(1 <= m <= 12 for m in months):
                raise ValueError("Month numbers must be between 1 and 12")
            return sorted(months)

        return None  # All months

    def _filter_by_months(
        self,
        data: pd.DataFrame,
        months: List[int]
    ) -> pd.DataFrame:
        """Return ``data`` restricted to the supplied calendar ``months``."""

        if data.empty:
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            coerced = pd.to_datetime(data.index, errors='coerce')
            data = data.copy()
            data.index = coerced

        month_mask = data.index.month.isin(months)
        return data.loc[month_mask]

    def _filter_by_regime(
        self,
        data: pd.DataFrame,
        regime_col: str,
        target_regimes: List[int],
    ) -> pd.DataFrame:
        """Return rows where ``regime_col`` belongs to ``target_regimes``."""

        if data.empty:
            return data

        if regime_col not in data.columns:
            raise KeyError(regime_col)

        return data.loc[data[regime_col].isin(target_regimes)]

    @staticmethod
    def _time_to_seconds(value: time) -> int:
        return value.hour * 3600 + value.minute * 60 + value.second

    @staticmethod
    def _seconds_to_time_label(total_seconds: Union[int, float]) -> str:
        seconds_int = int(total_seconds)
        base_time = (datetime.min + timedelta(seconds=seconds_int)).time()
        return base_time.strftime("%H:%M:%S")

    def _ensure_intraday_time_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure cached seconds-from-midnight and session date columns exist."""

        if data is None or data.empty:
            return data

        result = data
        copied = False

        if not isinstance(result.index, pd.DatetimeIndex):
            coerced_index = pd.to_datetime(result.index, errors='coerce')
            if not coerced_index.equals(result.index):
                result = result.copy()
                result.index = coerced_index
                copied = True

        if '_seconds_from_midnight' not in result.columns:
            if not copied:
                result = result.copy()
                copied = True
            result['_seconds_from_midnight'] = (
                result.index.hour * 3600
                + result.index.minute * 60
                + result.index.second
            )

        if '_session_date' not in result.columns:
            if not copied:
                result = result.copy()
            result['_session_date'] = result.index.normalize()

        return result

    def _localize_dataframe(self, data: pd.DataFrame, tz: str) -> pd.DataFrame:
        """
        [FIXED PANDAS API ERROR]
        Ensures the index of the DataFrame is timezone-aware in the given tz.
        If the index is already localized, it converts it.
        If the index is naive, it localizes it.
        """
        if data.empty:
            return data

        idx = data.index

        if not isinstance(idx, pd.DatetimeIndex):
            # Attempt to convert a non-datetime index to datetime
            idx = pd.to_datetime(idx, errors="coerce")

        if idx.tz is None:
            # Naive index: localize it with DST handling
            try:
                data.index = idx.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
            except Exception as e:
                # Handle cases where localization fails (e.g., ambiguous times during DST)
                print(f"Warning: Failed to tz_localize index: {e}. Using naive index.")
                data.index = idx
                return data
        else:
            # Already localized: convert it
            data.index = idx.tz_convert(tz)

        return data

    @staticmethod
    def _months_mask_12(months: List[int]) -> str:
        mask = ['0'] * 12
        for month in months:
            if 1 <= month <= 12:
                mask[month - 1] = '1'
        return ''.join(mask)

    def _build_months_metadata(
        self,
        index: Union[pd.Index, pd.Series, Sequence],
        tz: Optional[str],
    ) -> Dict[str, Any]:
        if isinstance(index, pd.Series):
            base_index = index.index if isinstance(index.index, pd.DatetimeIndex) else pd.Index(index)
        else:
            base_index = pd.Index(index)

        if not isinstance(base_index, pd.DatetimeIndex):
            dt_index = pd.to_datetime(base_index, errors='coerce')
        else:
            dt_index = base_index

        if tz:
            try:
                if dt_index.tz is None:
                    dt_index = dt_index.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
                else:
                    dt_index = dt_index.tz_convert(tz)
            except (TypeError, ValueError):
                pass

        months = sorted({int(month) for month in dt_index.month if not pd.isna(month)})
        mask = self._months_mask_12(months) if months else '0' * 12
        names = [calendar.month_abbr[m] for m in months]

        return {
            'months_active': months,
            'months_mask_12': mask,
            'months_names': names,
        }

    def _analyze_by_month(
        self,
        data: pd.DataFrame,
        price_col: str,
        is_synthetic: bool,
        selected_months: List[int]
    ) -> Dict[int, Dict[str, any]]:
        """
        Analyze returns and volatility for each selected month.

        Parameters:
        -----------
        data : pd.DataFrame
            Price/spread data (already filtered by selected_months)
        price_col : str
            Column name for prices
        is_synthetic : bool
            Whether data is synthetic spread
        selected_months : List[int]
            Months to analyze

        Returns:
        --------
        Dict mapping month number to statistics
        """
        # Calculate returns
        if is_synthetic:
            returns = data[price_col].diff()
        else:
            valid_mask = data[price_col] > 0
            returns = pd.Series(index=data.index, dtype=float)
            prices = data[price_col][valid_mask]
            returns[valid_mask] = np.log(prices / prices.shift(1))

        returns = returns.dropna()

        # Add month labels
        df = pd.DataFrame({'return': returns, 'month': returns.index.month})

        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }

        month_stats = {}

        for month_num in selected_months:
            month_data = df[df['month'] == month_num]['return']

            if len(month_data) < 2:
                continue

            month_stats[month_num] = {
                'month_name': month_names[month_num],
                'n': len(month_data),
                'mean': float(month_data.mean()),
                'std': float(month_data.std()),
                'sharpe': float(month_data.mean() / month_data.std()) if month_data.std() > 0 else np.nan,
                'skew': float(month_data.skew()),
                'positive_pct': float((month_data > 0).sum() / len(month_data)),
                'realized_vol': float(month_data.std()),
                'mean_abs_return': float(month_data.abs().mean())
            }

        # Compare months if we have multiple
        if len(month_stats) > 1:
            month_returns = [df[df['month'] == m]['return'].values
                           for m in selected_months
                           if m in df['month'].values]

            if len(month_returns) >= 2:
                from scipy.stats import f_oneway
                f_stat, p_value = f_oneway(*month_returns)
                month_stats['anova'] = {
                    'f_stat': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'description': 'Test for differences across selected months'
                }

        return month_stats

    def cot_positioning_screener(self, data, start_date, end_date, check_for_update=True):

        return

    def monthly_seasonality_screen(
        self,
        use_log_returns: bool = True,
        test_lag_months: int = 2,
        test_annual_predictability: bool = True,
        test_prewindow: bool = True,
        prewindow_days: Tuple[int, int] = (90, 60)
    ) -> Dict[str, Dict[str, any]]:
        """
        Screen for monthly and longer-term seasonal patterns.

        Tests for:
        - Monthly return seasonality (mean, std, z-scores by month)
        - Last year same-month predictability
        - Previous N months predictive power
        - Pre-window (60-90 days before) predictive power
        - Abnormal months (statistical outliers)

        Parameters:
        -----------
        use_log_returns : bool
            Use log returns for regular tickers (True) or simple returns (False)
        test_lag_months : int
            Number of previous months to test for predictability (default: 2)
        test_annual_predictability : bool
            Test if last year's same month predicts this year's month
        test_prewindow : bool
            Test if pre-window (60-90 days before) predicts monthly returns
        prewindow_days : Tuple[int, int]
            Days before/after for pre-window (default: (90, 60) = 90 days before to 60 days before)

        Returns:
        --------
        Dict[str, Dict[str, any]]
            Results dictionary with monthly seasonality analysis for each ticker

        Examples:
        ---------
        # Full monthly seasonality analysis
        results = screener.monthly_seasonality_screen()

        # Test only lag month predictability
        results = screener.monthly_seasonality_screen(
            test_annual_predictability=False,
            test_prewindow=False,
            test_lag_months=3
        )
        """
        results = {}

        for ticker in self.tickers:
            # Get data for this ticker
            is_synthetic = self.synthetic_tickers.get(ticker, False)

            if is_synthetic:
                synthetic_obj = self.data[ticker]
                data = synthetic_obj.price if hasattr(synthetic_obj, 'price') else synthetic_obj.data_engine.build_spread_series(return_ohlc=True)
            else:
                data = self.data[ticker]

            if data.empty:
                results[ticker] = {'error': 'No data available'}
                continue

            # Determine price column
            price_col = 'Close' if 'Close' in data.columns else data.columns[0]
            prices = data[price_col]

            ticker_results = {
                'ticker': ticker,
                'is_synthetic': is_synthetic,
                'n_observations': len(data),
                'date_range': (str(data.index[0]), str(data.index[-1]))
            }

            # Calculate monthly statistics and abnormal months
            # First calculate monthly returns, then pass to abnormal_months
            from ..utils.seasonal import monthly_returns
            # Convert Series to DataFrame for monthly_returns function
            price_df = prices.to_frame(name=price_col)
            monthly_ret = monthly_returns(price_df, price_col=price_col, use_log_returns=use_log_returns and not is_synthetic)
            monthly_stats = abnormal_months(monthly_ret)
            ticker_results['monthly_statistics'] = self._format_monthly_stats(monthly_stats)

            # Test annual predictability (last year's month predicts this year)
            if test_annual_predictability:
                # last_year_predicts_this_year expects monthly returns Series, not prices
                annual_pred = last_year_predicts_this_year(monthly_ret)
                ticker_results['annual_predictability'] = self._format_annual_predictability(annual_pred)

            # Test lag month predictability (previous N months)
            if test_lag_months > 0:
                lag_pred = self._test_lag_month_predictability(
                    prices,
                    test_lag_months,
                    use_log_returns and not is_synthetic
                )
                ticker_results['lag_month_predictability'] = lag_pred

            # Test pre-window predictability
            if test_prewindow:
                # prewindow_predicts_month expects DataFrame with prices, not Series
                prewindow_pred = prewindow_predicts_month(
                    price_df,
                    price_col=price_col,
                    use_log_returns=use_log_returns and not is_synthetic
                )
                ticker_results['prewindow_predictability'] = {
                    'prewindow_days': prewindow_days,
                    'correlation': float(prewindow_pred['r']),  # Fixed: was 'correlation', should be 'r'
                    'p_value': float(prewindow_pred['p_value']),
                    'significant': prewindow_pred['p_value'] < 0.05,
                    'interpretation': self._interpret_correlation(
                        prewindow_pred['r'],  # Fixed: was 'correlation', should be 'r'
                        prewindow_pred['p_value']
                    )
                }

            # Rank strongest patterns
            strongest = self._rank_monthly_patterns(ticker_results)
            ticker_results['strongest_patterns'] = strongest

            results[ticker] = ticker_results

        return results

    def _format_monthly_stats(
        self,
        monthly_stats: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Format output from abnormal_months() utility.

        Parameters:
        -----------
        monthly_stats : Dict
            Raw output from abnormal_months()

        Returns:
        --------
        Dict with formatted monthly statistics
        """
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }

        formatted = {}

        for month_num, stats in monthly_stats.items():
            if month_num == 'abnormal':
                formatted['abnormal_months'] = stats
                continue

            month_name = month_names.get(month_num, f'Month_{month_num}')

            formatted[month_name] = {
                'month_number': month_num,
                'n_observations': stats.get('n', 0),
                'mean_return': float(stats.get('mean', np.nan)),
                'std_return': float(stats.get('std', np.nan)),
                'z_score': float(stats.get('z_score', np.nan)),
                'is_abnormal': stats.get('is_abnormal', False),
                'sharpe': float(stats['mean'] / stats['std']) if stats.get('std', 0) > 0 else np.nan
            }

        return formatted

    def _format_annual_predictability(
        self,
        annual_pred: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Format output from last_year_predicts_this_year() utility.

        Parameters:
        -----------
        annual_pred : pd.DataFrame
            Raw output from last_year_predicts_this_year() (DataFrame with columns: n, slope, r, p_value)

        Returns:
        --------
        Dict with formatted annual predictability results
        """
        # Calculate average correlation and p-value across all months
        avg_r = annual_pred['r'].mean()
        avg_p_value = annual_pred['p_value'].mean()
        n_significant = (annual_pred['p_value'] < 0.05).sum()

        return {
            'correlation': float(avg_r) if not np.isnan(avg_r) else np.nan,
            'p_value': float(avg_p_value) if not np.isnan(avg_p_value) else np.nan,
            'significant': n_significant > 0,  # At least one month is significant
            'n_significant_months': int(n_significant),
            'interpretation': self._interpret_correlation(
                avg_r if not np.isnan(avg_r) else 0,
                avg_p_value if not np.isnan(avg_p_value) else 1.0
            )
        }

    def _test_lag_month_predictability(
        self,
        prices: pd.Series,
        n_lags: int,
        use_log_returns: bool
    ) -> Dict[str, any]:
        """
        Test if previous N months predict current month returns.

        Parameters:
        -----------
        prices : pd.Series
            Price data
        n_lags : int
            Number of lag months to test
        use_log_returns : bool
            Whether to use log returns

        Returns:
        --------
        Dict with lag correlations and statistics
        """
        from ..utils.seasonal import monthly_returns

        # Calculate monthly returns
        # Convert Series to DataFrame for monthly_returns function
        price_col = prices.name if prices.name else 'price'
        price_df = prices.to_frame(name=price_col)
        monthly_ret = monthly_returns(price_df, price_col=price_col, use_log_returns=use_log_returns)

        lag_results = {}
        from ..utils.gpu_utils import gpu_batch_pearsonr

        # Prepare batch data for all lags
        valid_lags = []
        aligned_data_list = []

        for lag in range(1, n_lags + 1):
            # Shift returns by lag months
            lagged = monthly_ret.shift(lag)

            # Align and drop NaNs
            aligned = pd.DataFrame({
                'current': monthly_ret,
                f'lag_{lag}': lagged
            }).dropna()

            if len(aligned) < 10:
                lag_results[f'lag_{lag}_months'] = {
                    'n': len(aligned),
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'error': 'Insufficient data'
                }
                continue

            valid_lags.append(lag)
            aligned_data_list.append(aligned)

        # Batch process all lag correlations using GPU
        if valid_lags:
            try:
                # Pad arrays to same length for batch processing
                max_len = max(len(ad) for ad in aligned_data_list)
                y_batch = []

                for aligned in aligned_data_list:
                    lag_col = [c for c in aligned.columns if c.startswith('lag_')][0]
                    lagged_padded = np.full(max_len, np.nan)
                    lagged_padded[:len(aligned)] = aligned[lag_col].values
                    y_batch.append(lagged_padded)

                # Use first aligned dataset's 'current' as reference (padded)
                current_padded = np.full(max_len, np.nan)
                current_padded[:len(aligned_data_list[0])] = aligned_data_list[0]['current'].values

                y_batch_array = np.array(y_batch)
                corrs, pvals = gpu_batch_pearsonr(
                    current_padded,
                    y_batch_array,
                    use_gpu=self.use_gpu,
                    device_id=self.gpu_device_id,
                    return_pvalues=True
                )

                # Map results back to lag_results
                for i, lag in enumerate(valid_lags):
                    aligned = aligned_data_list[i]
                    corr = corrs[i]
                    p_val = pvals[i]
                    lag_results[f'lag_{lag}_months'] = {
                        'n': len(aligned),
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'interpretation': self._interpret_lag_correlation(lag, corr, p_val)
                    }
            except Exception as e:
                # CPU fallback - process individually
                for i, lag in enumerate(valid_lags):
                    aligned = aligned_data_list[i]
                    lag_col = [c for c in aligned.columns if c.startswith('lag_')][0]
                    corr, p_val = stats.pearsonr(aligned['current'], aligned[lag_col])
                    lag_results[f'lag_{lag}_months'] = {
                        'n': len(aligned),
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'interpretation': self._interpret_lag_correlation(lag, corr, p_val)
                    }

        # Summary
        significant_lags = [k for k, v in lag_results.items() if v.get('significant', False)]
        lag_results['summary'] = {
            'n_significant_lags': len(significant_lags),
            'significant_lags': significant_lags,
            'has_momentum': len(significant_lags) > 0
        }

        return lag_results

    def _interpret_lag_correlation(
        self,
        lag: int,
        correlation: float,
        p_value: float
    ) -> str:
        """
        Generate human-readable interpretation of lag correlation.

        Parameters:
        -----------
        lag : int
            Lag in months
        correlation : float
            Correlation coefficient
        p_value : float
            Statistical significance

        Returns:
        --------
        str
            Interpretation
        """
        if np.isnan(correlation) or np.isnan(p_value):
            return "Insufficient data"

        if p_value >= 0.05:
            return f"No significant relationship with {lag} month(s) ago"

        if correlation > 0.3:
            return f"Strong positive momentum - good {lag} month(s) ago predicts good performance"
        elif correlation > 0.1:
            return f"Moderate positive momentum - {lag} month(s) ago has predictive power"
        elif correlation < -0.3:
            return f"Strong mean reversion - good {lag} month(s) ago predicts poor performance"
        elif correlation < -0.1:
            return f"Moderate mean reversion - {lag} month(s) ago shows reversal tendency"
        else:
            return f"Weak relationship with {lag} month(s) ago"

    def _interpret_correlation(
        self,
        correlation: float,
        p_value: float
    ) -> str:
        """
        Generate human-readable interpretation of correlation.

        Parameters:
        -----------
        correlation : float
            Correlation coefficient
        p_value : float
            Statistical significance

        Returns:
        --------
        str
            Interpretation
        """
        if np.isnan(correlation) or np.isnan(p_value):
            return "Insufficient data"

        if p_value >= 0.05:
            return "No significant relationship"

        if abs(correlation) > 0.5:
            strength = "Very strong"
        elif abs(correlation) > 0.3:
            strength = "Strong"
        elif abs(correlation) > 0.1:
            strength = "Moderate"
        else:
            strength = "Weak"

        direction = "positive" if correlation > 0 else "negative"

        return f"{strength} {direction} relationship (statistically significant)"

    def _rank_monthly_patterns(
        self,
        ticker_results: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Identify and rank the strongest monthly patterns.

        Parameters:
        -----------
        ticker_results : Dict
            Complete results for a ticker

        Returns:
        --------
        List of strongest patterns sorted by significance
        """
        patterns = []

        # Check for abnormal months
        if 'monthly_statistics' in ticker_results:
            monthly_stats = ticker_results['monthly_statistics']

            if 'abnormal_months' in monthly_stats:
                for month in monthly_stats['abnormal_months']:
                    patterns.append({
                        'type': 'abnormal_month',
                        'month': month,
                        'description': f'{month} shows abnormal returns',
                        'strength': abs(monthly_stats[month].get('z_score', 0))
                    })

            # Check for high Sharpe months
            for month_name, stats in monthly_stats.items():
                if month_name == 'abnormal_months':
                    continue

                sharpe = stats.get('sharpe', 0)
                if not np.isnan(sharpe) and abs(sharpe) > 0.5:
                    patterns.append({
                        'type': 'high_sharpe_month',
                        'month': month_name,
                        'sharpe': sharpe,
                        'description': f'{month_name} has high risk-adjusted returns',
                        'strength': abs(sharpe)
                    })

        # Annual predictability
        if 'annual_predictability' in ticker_results:
            annual = ticker_results['annual_predictability']
            if annual.get('significant', False):
                patterns.append({
                    'type': 'annual_seasonality',
                    'description': 'Last year same-month predicts this year',
                    'correlation': annual['correlation'],
                    'p_value': annual['p_value'],
                    'strength': abs(annual['correlation'])
                })

        # Lag month predictability
        if 'lag_month_predictability' in ticker_results:
            lag_pred = ticker_results['lag_month_predictability']

            for lag_key, lag_stats in lag_pred.items():
                if lag_key == 'summary':
                    continue

                if lag_stats.get('significant', False):
                    patterns.append({
                        'type': 'lag_month_momentum',
                        'lag': lag_key,
                        'description': f'{lag_key.replace("_", " ")} shows predictive power',
                        'correlation': lag_stats['correlation'],
                        'p_value': lag_stats['p_value'],
                        'strength': abs(lag_stats['correlation'])
                    })

        # Pre-window predictability
        if 'prewindow_predictability' in ticker_results:
            prewindow = ticker_results['prewindow_predictability']
            if prewindow.get('significant', False):
                patterns.append({
                    'type': 'prewindow_predictor',
                    'description': f'Pre-window {prewindow["prewindow_days"]} days predicts monthly returns',
                    'correlation': prewindow['correlation'],
                    'p_value': prewindow['p_value'],
                    'strength': abs(prewindow['correlation'])
                })

        # Sort by strength (descending)
        patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)

        return patterns


    def _convert_times(self,session_starts:List, session_ends:Optional[List]=None) -> datetime:
        s_start = []
        if session_ends:
            s_end = []
        for i, session in enumerate(session_starts):
            s_start.append(
                session_starts[i] if isinstance(session_starts[i], time) else datetime.strptime(session_starts[i],
                                                                                                "%H:%M").time())
            if session_ends:
                s_end.append(session_ends[i] if isinstance(session_ends[i], time) else datetime.strptime(session_ends[i],
                                                                                                     "%H:%M").time())
        if session_ends:
            return s_start, s_end
        else:
            return s_start

    @staticmethod
    def _time_in_session(target: time, session_start: time, session_end: time) -> bool:
        start_us = (session_start.hour * 3600 + session_start.minute * 60 + session_start.second) * 1_000_000 + session_start.microsecond
        end_us = (session_end.hour * 3600 + session_end.minute * 60 + session_end.second) * 1_000_000 + session_end.microsecond
        target_us = (target.hour * 3600 + target.minute * 60 + target.second) * 1_000_000 + target.microsecond

        if start_us <= end_us:
            return start_us <= target_us <= end_us
        return target_us >= start_us or target_us <= end_us

    def _validate_target_times(
        self,
        target_times: List[time],
        session_start: time,
        session_end: time,
    ) -> None:
        for target_time in target_times:
            if not self._time_in_session(target_time, session_start, session_end):
                raise ValueError(
                    f"Target time {target_time.strftime('%H:%M:%S')} falls outside the session window "
                    f"{session_start.strftime('%H:%M:%S')} - {session_end.strftime('%H:%M:%S')}"
                )

    def plot_momentum_scatter(
        self,
        results: Dict[str, Dict[str, any]],
        ticker: str,
        session_key: str = 'session_0',
        x_var: str = 'opening_returns',
        y_var: str = 'closing_returns',
        show_regression: bool = True,
        title: Optional[str] = None
    ):
        """
        Create scatter plot of momentum returns for analysis.

        Parameters:
        -----------
        results : Dict[str, Dict[str, any]]
            Results from intraday_momentum_screen()
        ticker : str
            Ticker to plot
        session_key : str
            Session to analyze (e.g., 'session_0', 'session_1')
        x_var : str
            Variable for x-axis ('opening_returns', 'closing_returns', 'full_session_returns', 'st_momentum')
        y_var : str
            Variable for y-axis
        show_regression : bool
            Whether to show regression line
        title : Optional[str]
            Custom plot title

        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive scatter plot
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None

        # Extract data
        session_data = results[ticker][session_key]
        return_series = session_data['return_series']

        x_data = return_series[x_var].dropna()
        y_data = return_series[y_var].dropna()

        # Align series
        aligned = pd.DataFrame({x_var: x_data, y_var: y_data}).dropna()

        if len(aligned) < 2:
            print(f"Insufficient data for {ticker} {session_key}")
            return None

        # Create figure
        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=aligned[x_var],
            y=aligned[y_var],
            mode='markers',
            name='Returns',
            marker=dict(
                size=8,
                color=aligned.index.astype(str),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Date")
            ),
            text=[f"Date: {d}" for d in aligned.index],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{x_var}: %{{x:.4f}}<br>' +
                         f'{y_var}: %{{y:.4f}}<br>' +
                         '<extra></extra>'
        ))

        # Add regression line
        if show_regression and len(aligned) > 2:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(aligned[x_var], aligned[y_var])

            x_range = np.linspace(aligned[x_var].min(), aligned[x_var].max(), 100)
            y_fit = slope * x_range + intercept

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_fit,
                mode='lines',
                name=f'Regression (R²={r_value**2:.3f})',
                line=dict(color='red', dash='dash'),
                hovertemplate=f'R² = {r_value**2:.3f}<br>p-value = {p_value:.4f}<extra></extra>'
            ))

        # Layout
        if title is None:
            is_synthetic = session_data['is_synthetic']
            data_type = "Spread Change" if is_synthetic else "Log Returns"
            title = f"{ticker} - {session_key}<br>{x_var} vs {y_var} ({data_type})"

        fig.update_layout(
            title=title,
            xaxis_title=x_var.replace('_', ' ').title(),
            yaxis_title=y_var.replace('_', ' ').title(),
            hovermode='closest',
            template='plotly_white',
            height=600,
            width=800
        )

        return fig

    def plot_all_momentum_relationships(
        self,
        results: Dict[str, Dict[str, any]],
        ticker: str,
        session_key: str = 'session_0'
    ):
        """
        Create comprehensive momentum relationship plots.

        Parameters:
        -----------
        results : Dict[str, Dict[str, any]]
            Results from intraday_momentum_screen()
        ticker : str
            Ticker to analyze
        session_key : str
            Session to analyze

        Returns:
        --------
        Dict[str, plotly.graph_objects.Figure]
            Dictionary of figures with different relationship plots
        """
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None

        plots = {}

        # 1. Opening vs Closing Momentum
        plots['open_vs_close'] = self.plot_momentum_scatter(
            results, ticker, session_key,
            x_var='opening_returns',
            y_var='closing_returns',
            title=f"{ticker} - Opening vs Closing Momentum"
        )

        # 2. Opening vs Full Session
        plots['open_vs_full'] = self.plot_momentum_scatter(
            results, ticker, session_key,
            x_var='opening_returns',
            y_var='full_session_returns',
            title=f"{ticker} - Opening vs Full Session Returns"
        )

        # 3. Closing vs Full Session
        plots['close_vs_full'] = self.plot_momentum_scatter(
            results, ticker, session_key,
            x_var='closing_returns',
            y_var='full_session_returns',
            title=f"{ticker} - Closing vs Full Session Returns"
        )

        # 4. Short-term Momentum vs Opening
        plots['st_mom_vs_open'] = self.plot_momentum_scatter(
            results, ticker, session_key,
            x_var='st_momentum',
            y_var='opening_returns',
            title=f"{ticker} - Short-term Momentum vs Opening Returns"
        )

        # 5. Short-term Momentum vs Closing
        plots['st_mom_vs_close'] = self.plot_momentum_scatter(
            results, ticker, session_key,
            x_var='st_momentum',
            y_var='closing_returns',
            title=f"{ticker} - Short-term Momentum vs Closing Returns"
        )

        return plots

    @classmethod
    def from_scids(
        cls,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        volume_bucket: Optional[int] = None,
        resample_rule: str = "5min",
        write_db: bool = True,
        file_manager: Optional[IntradayFileManager] = None
    ):
        """
        Create HistoricalScreener from SCID files.

        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols (without _F suffix)
        start_date : Optional[str]
            Start date in YYYY-MM-DD format (default: None, loads all available)
        end_date : Optional[str]
            End date in YYYY-MM-DD format (default: None, loads to present)
        volume_bucket : Optional[int]
            Volume bucket size for bucketing (default: None, time-based only)
        resample_rule : str
            Resampling rule (default: '5min')
        write_db : bool
            Whether to write to database (default: True)
        file_manager : Optional[IntradayFileManager]
            Existing file manager to reuse (optional - singleton pattern handles Arctic reuse automatically)

        Returns:
        --------
        HistoricalScreener
            Initialized screener with loaded data

        Examples:
        ---------
        # Load all available data (Arctic instance automatically reused)
        screener = HistoricalScreener.from_scids(['CL', 'NG', 'HO'])

        # Load specific date range
        screener = HistoricalScreener.from_scids(
            ['CL', 'NG'],
            start_date='2021-01-01',
            end_date='2024-01-01',
            resample_rule='1T'
        )

        # Multiple screeners automatically share the same Arctic instance
        screener1 = HistoricalScreener.from_scids(['CL'])
        screener2 = HistoricalScreener.from_scids(['NG'])  # Reuses same Arctic connection

        Notes:
        ------
        The IntradayFileManager now uses a singleton pattern for Arctic instances,
        preventing LMDB "already opened" errors automatically. Manual file_manager
        parameter is optional.
        """
        # Reuse existing manager or create new one
        if file_manager is None:
            mgr = IntradayFileManager(DLY_DATA_PATH, arctic_uri=INTRADAY_ADB_PATH)
        else:
            mgr = file_manager

        # Parse dates if provided
        start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        data = {}
        for t in tickers:
            # Add _F suffix if not present
            symbol = t if t.endswith('_F') else f"{t}_F"

            try:
                df = mgr.load_front_month_series(
                    symbol=symbol,
                    start=start,
                    end=end,
                    resample_rule=resample_rule,
                    volume_bucket_size=volume_bucket,
                    detect_gaps=False
                )

                if not df.empty:
                    data[symbol] = df
                    print(f"Loaded {symbol}: {len(df):,} records from {df.index[0]} to {df.index[-1]}")
                    if gaps:
                        print(f"  - {len(gaps)} gaps detected")
                else:
                    print(f"Warning: No data loaded for {symbol}")

            except Exception as e:
                print(f"Error loading {symbol}: {e}")

        if not data:
            raise ValueError("No data loaded for any tickers")

        return cls(data, mgr)

    @classmethod
    def from_dclient(
        cls,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
        resample: Optional[str] = None,
        daily: bool = False,
        data_client: Optional[DataClient] = None
    ):
        """
        Create HistoricalScreener from market data using DataClient.

        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols (with or without _F suffix)
        start_date : Optional[str]
            Start date in YYYY-MM-DD format (default: None, loads all available)
        end_date : Optional[str]
            End date in YYYY-MM-DD format (default: None, loads to present)
        columns : Optional[List[str]]
            Specific columns to load (default: None, loads all columns)
        resample : Optional[str]
            Resample frequency (e.g., 'W', 'M' for weekly/monthly)
        daily : bool
            If True, loads daily data. If False, loads intraday data (default: False)
        data_client : Optional[DataClient]
            Existing DataClient instance (default: None, creates new)

        Returns:
        --------
        HistoricalScreener
            Initialized screener with loaded market data

        Examples:
        ---------
        # Load daily data for multiple tickers
        screener = HistoricalScreener.from_dclient(['CL_F', 'NG_F', 'HO_F'], daily=True)

        # Load intraday data (default)
        screener = HistoricalScreener.from_dclient(['CL_F', 'NG_F'])

        # Load daily data with specific date range
        screener = HistoricalScreener.from_dclient(
            ['CL', 'NG'],
            start_date='2020-01-01',
            end_date='2024-01-01',
            daily=True
        )

        # Load daily data and resample to weekly
        screener = HistoricalScreener.from_dclient(
            ['CL_F', 'NG_F'],
            start_date='2020-01-01',
            resample='W',
            daily=True
        )

        # Load only specific columns
        screener = HistoricalScreener.from_dclient(
            ['CL_F'],
            columns=['Open', 'High', 'Low', 'Close', 'Volume'],
            daily=True
        )

        # Reuse existing DataClient
        client = DataClient()
        screener = HistoricalScreener.from_dclient(['CL_F'], daily=True, data_client=client)

        Notes:
        ------
        - Uses DataClient.query_market_data() to load market data
        - Set daily=True for daily data, daily=False for intraday data
        - Automatically handles ticker symbols with or without _F suffix
        - Returns combined dataset for each ticker
        """
        # Create DataClient if not provided
        if data_client is None:
            client = DataClient()
        else:
            client = data_client

        # Ensure tickers have _F suffix
        normalized_tickers = [t if t.endswith('_F') else f"{t}_F" for t in tickers]

        data = {}
        for ticker in normalized_tickers:
            try:
                # Query market data for this ticker
                df = client.query_market_data(
                    tickers=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    columns=columns,
                    resample=resample,
                    daily=daily,
                    combine_datasets=True
                )

                # Handle both dict and DataFrame returns
                if isinstance(df, dict):
                    # If dict returned, extract the ticker data
                    if ticker in df and not df[ticker].empty:
                        data[ticker] = df[ticker]
                        data_type = "daily" if daily else "intraday"
                        print(f"Loaded {ticker} ({data_type}): {len(df[ticker]):,} records from {df[ticker].index[0]} to {df[ticker].index[-1]}")
                    else:
                        print(f"Warning: No data loaded for {ticker}")
                elif isinstance(df, pd.DataFrame):
                    # If DataFrame returned directly
                    if not df.empty:
                        data[ticker] = df
                        data_type = "daily" if daily else "intraday"
                        print(f"Loaded {ticker} ({data_type}): {len(df):,} records from {df.index[0]} to {df.index[-1]}")
                    else:
                        print(f"Warning: No data loaded for {ticker}")
                else:
                    print(f"Warning: Unexpected data type for {ticker}: {type(df)}")

            except Exception as e:
                print(f"Error loading {ticker}: {e}")

        if not data:
            raise ValueError("No data loaded for any tickers")

        return cls(data, None)

    @classmethod
    def from_parquet(
        cls,
        symbols: List[str],
        parquet_path: Optional[str] = None,
        timeframe: Optional[str] = None,
        volume_bucket_size: Optional[int] = None,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        max_concurrent: int = 5
    ):
        """
        Create HistoricalScreener from Parquet files using AsyncParquetWriter infrastructure.

        Parameters
        ----------
        symbols : List[str]
            List of symbols to load (e.g., ['CL_F', 'NG_F', 'ZC_F'])
        parquet_path : Optional[str]
            Base path for Parquet storage (default: INTRADAY_DATA_PATH from config)
        timeframe : Optional[str]
            Time-based resampling identifier (e.g., '1T', '5T', '15T')
            If None, loads raw tick data
        volume_bucket_size : Optional[int]
            Volume bucket size identifier (e.g., 500, 1000)
            Cannot be used with timeframe
        start : Optional[Union[str, datetime]]
            Start date/datetime filter
        end : Optional[Union[str, datetime]]
            End date/datetime filter
        max_concurrent : int
            Maximum concurrent async reads (default: 5)

        Returns
        -------
        HistoricalScreener
            Initialized screener with loaded parquet data

        Examples
        --------
        # Load 1-minute data for multiple symbols
        >>> screener = HistoricalScreener.from_parquet(
        ...     symbols=['CL_F', 'NG_F'],
        ...     timeframe='1T',
        ...     start='2023-01-01',
        ...     end='2024-01-01'
        ... )

        # Load volume bucket data
        >>> screener = HistoricalScreener.from_parquet(
        ...     symbols=['CL_F'],
        ...     volume_bucket_size=500,
        ...     start=datetime(2023, 1, 1)
        ... )

        # Load from custom path
        >>> screener = HistoricalScreener.from_parquet(
        ...     symbols=['ZC_F', 'ZS_F'],
        ...     parquet_path='F:/Data/intraday/',
        ...     timeframe='5T'
        ... )

        # Load with high concurrency
        >>> screener = HistoricalScreener.from_parquet(
        ...     symbols=['CL_F', 'NG_F', 'HO_F', 'RB_F'],
        ...     timeframe='1T',
        ...     max_concurrent=10
        ... )

        Notes
        -----
        - Uses AsyncParquetWriter for efficient async I/O
        - Automatically handles timezone-aware datetime filtering
        - Parquet files expected at: {parquet_path}/{symbol}/{symbol}_{timeframe}.parquet
        - Returns empty dict for symbols with no data (with warning)
        """
        from ..data.raw_formatting.intraday_manager import AsyncParquetWriter
        import asyncio

        # Initialize AsyncParquetWriter
        writer = AsyncParquetWriter(
            parquet_base_path=parquet_path if parquet_path else None,
            enable_logging=True
        )

        # Convert string dates to datetime if needed
        if start and isinstance(start, str):
            start = pd.to_datetime(start)
        if end and isinstance(end, str):
            end = pd.to_datetime(end)

        # Create async loading coroutine
        async def load_all_symbols():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def load_symbol(symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
                async with semaphore:
                    try:
                        df = await writer.read_dataframe(
                            symbol=symbol,
                            timeframe=timeframe,
                            volume_bucket_size=volume_bucket_size,
                            start=start,
                            end=end
                        )

                        if df.empty:
                            print(f"Warning: No data loaded for {symbol}")
                            return symbol, None

                        # Print summary
                        freq_str = f"{timeframe}" if timeframe else f"vol{volume_bucket_size}" if volume_bucket_size else "raw"
                        print(f"Loaded {symbol} ({freq_str}): {len(df):,} records from {df.index[0]} to {df.index[-1]}")
                        return symbol, df

                    except Exception as e:
                        print(f"Error loading {symbol}: {e}")
                        return symbol, None

            # Load all symbols concurrently
            tasks = [load_symbol(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            return results

        # Run the async loading
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(load_all_symbols())

        # Build data dictionary
        data = {}
        for symbol, df in results:
            if df is not None:
                data[symbol] = df

        if not data:
            raise ValueError(f"No data loaded for any symbols: {symbols}")

        # Create IntradayFileManager instance for compatibility
        mgr = IntradayFileManager(data_path=DLY_DATA_PATH, arctic_uri=INTRADAY_ADB_PATH)

        return cls(data, mgr)


def main():
    return