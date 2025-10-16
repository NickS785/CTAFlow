from ..utils.seasonal import last_year_predicts_this_year, intraday_lag_autocorr, abnormal_months, prewindow_feature, prewindow_predicts_month
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
from dataclasses import dataclass, field
from ..data import IntradayFileManager, DataClient, SyntheticSymbol
from ..config import DLY_DATA_PATH, INTRADAY_ADB_PATH
from ..utils.session import filter_session_bars
from scipy import stats


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
    sess_end_mins : Optional[int]
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
        Olson timezone for the seasonality session filter (default "America/Chicago").

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

    # Momentum screen parameters
    session_starts: Optional[List[Union[str, time]]] = None
    session_ends: Optional[List[Union[str, time]]] = None
    st_momentum_days: int = 3
    sess_start_hrs: int = 1
    sess_start_minutes: int = 30
    sess_end_hrs: Optional[int] = None
    sess_end_mins: Optional[int] = None
    test_vol: bool = True

    # Seasonality screen parameters
    target_times: Optional[List[Union[str, time]]] = None
    period_length: Optional[Union[int, timedelta]] = None
    dayofweek_screen: bool = True
    seasonality_session_start: Union[str, time] = "00:00"
    seasonality_session_end: Union[str, time] = "23:59:59"
    tz: str = "America/Chicago"

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

        # Auto-generate name if not provided
        if self.name is None:
            if self.season:
                self.name = f"{self.season}_{self.screen_type}"
            elif self.months:
                month_str = "_".join(map(str, self.months))
                self.name = f"months_{month_str}_{self.screen_type}"
            else:
                self.name = f"all_{self.screen_type}"


class HistoricalScreener:
    """Screener created to find seasonal and momentum patterns in intraday and daily data"""

    def __init__(self, ticker_data: Dict[str, Union[pd.DataFrame, SyntheticSymbol]], file_mgr:IntradayFileManager=None):
        """
        Initialize HistoricalScreener with ticker data.

        Parameters:
        -----------
        ticker_data : Dict[str, Union[pd.DataFrame, SyntheticSymbol]]
            Dictionary mapping ticker symbols to either DataFrames or SyntheticSymbol objects
        file_mgr : IntradayFileManager, optional
            File manager for loading additional data
        """
        self.data = ticker_data
        self.tickers = list(ticker_data.keys())
        self.mgr = file_mgr or IntradayFileManager(data_path=DLY_DATA_PATH, arctic_uri=INTRADAY_ADB_PATH)
        self.method = "arctic"

        # Track which tickers are synthetic spreads
        self.synthetic_tickers = {
            ticker: isinstance(data, SyntheticSymbol)
            for ticker, data in ticker_data.items()
        }


    def intraday_momentum_screen(
        self,
        session_starts: List[Union[str, time]] = ["02:30", "08:30"],
        session_ends: List[Union[str, time]] = ["10:30", "13:30"],
        st_momentum_days: int = 3,
        sess_start_hrs: int = 1,
        sess_start_minutes: int = 30,
        sess_end_hrs: Optional[int] = None,
        sess_end_mins: Optional[int] = None,
        test_vol: bool = True,
        months: Optional[List[int]] = None,
        season: Optional[str] = None,
        _selected_months: Optional[List[int]] = None,
        _precomputed_sessions: Optional[Dict[str, Dict[str, any]]] = None
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
        sess_end_mins : Optional[int]
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

        # Determine which months to analyze (allow override when precomputed)
        selected_months = (
            _selected_months
            if _selected_months is not None
            else self._parse_season_months(months, season)
        )

        # Default closing window to match opening window
        if sess_end_hrs is None:
            sess_end_hrs = sess_start_hrs
        if sess_end_mins is None:
            sess_end_mins = sess_start_minutes

        results = {}

        precomputed_sessions = _precomputed_sessions or {}

        for t in self.tickers:
            cache_entry = precomputed_sessions.get(t)

            if cache_entry:
                is_synthetic = cache_entry.get('is_synthetic', self.synthetic_tickers.get(t, False))
                ticker_data = cache_entry.get('data')
                price_col = cache_entry.get('price_col')
                filtered_months_label = cache_entry.get('filtered_months', selected_months if selected_months else 'all')
                n_observations = cache_entry.get('n_observations', len(ticker_data) if ticker_data is not None else 0)

                if ticker_data is None or ticker_data.empty:
                    results[t] = {
                        'error': cache_entry.get('error', 'No data available'),
                        'filtered_months': filtered_months_label,
                        'ticker': t,
                        'is_synthetic': is_synthetic
                    }
                    if selected_months is not None:
                        results[t]['selected_months'] = selected_months
                    continue
            else:
                # Get data for this ticker
                is_synthetic = self.synthetic_tickers.get(t, False)

                if is_synthetic:
                    synthetic_obj = self.data[t]
                    ticker_data = synthetic_obj.price if hasattr(synthetic_obj, 'price') else synthetic_obj.data_engine.build_spread_series(return_ohlc=True)
                else:
                    ticker_data = self.data[t]

                if ticker_data.empty:
                    results[t] = {'error': 'No data available'}
                    continue

                # Filter by months/season if specified
                if selected_months is not None:
                    ticker_data = self._filter_by_months(ticker_data, selected_months)
                    if ticker_data.empty:
                        results[t] = {
                            'error': 'No data available for selected months/season',
                            'selected_months': selected_months
                        }
                        continue

                price_col = 'Close' if 'Close' in ticker_data.columns else ticker_data.columns[0]
                filtered_months_label = selected_months if selected_months else 'all'
                n_observations = len(ticker_data)

            if price_col is None and ticker_data is not None:
                price_col = 'Close' if 'Close' in ticker_data.columns else ticker_data.columns[0]

            ticker_results = {
                'ticker': t,
                'is_synthetic': is_synthetic,
                'n_observations': n_observations,
                'filtered_months': filtered_months_label
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
                    end_mins=sess_end_mins,
                    momentum_days=st_momentum_days,
                    test_vol=test_vol,
                    data=ticker_data,
                    session_data=session_df,
                    price_col=price_col,
                    is_synthetic=is_synthetic
                )

                ticker_results[session_key] = momentum_stats

            results[t] = ticker_results

        return results

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
    ) -> Dict[str, Dict[str, any]]:
        """
        Screen for seasonality patterns in intraday data.

        Tests for:
        - Day-of-week effects (returns and volatility)
        - Time-of-day predictability (next day/week correlations)
        - Lag autocorrelations at specific times
        - Volatility seasonality
        - Month-specific and seasonal patterns

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

        Returns:
        --------
        Dict[str, Dict[str, any]]
            Results dictionary with seasonality analysis for each ticker. Each ticker
            entry includes metadata describing the session window and the number of
            sessions available after filtering.

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

            # Filter to session window in local timezone
            session_data = filter_session_bars(
                data,
                tz,
                session_start_time,
                session_end_time,
            )

            if session_data.empty:
                results[ticker] = {
                    'error': 'No data available within session window',
                    'session_start': session_start_time.strftime("%H:%M:%S"),
                    'session_end': session_end_time.strftime("%H:%M:%S"),
                    'tz': tz,
                }
                continue

            if selected_months is not None:
                session_data = self._filter_by_months(session_data, selected_months)
                if session_data.empty:
                    results[ticker] = {
                        'error': 'No data available for selected months/season',
                        'selected_months': selected_months,
                        'session_start': session_start_time.strftime("%H:%M:%S"),
                        'session_end': session_end_time.strftime("%H:%M:%S"),
                        'tz': tz,
                    }
                    continue

            # Determine price column
            price_col = 'Close' if 'Close' in session_data.columns else session_data.columns[0]

            ticker_results = {
                'ticker': ticker,
                'is_synthetic': is_synthetic,
                'n_observations': len(session_data),
                'filtered_months': selected_months if selected_months else 'all'
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
            ticker_results['metadata'] = {
                'session_start': session_start_time.strftime("%H:%M:%S"),
                'session_end': session_end_time.strftime("%H:%M:%S"),
                'tz': tz,
                'n_sessions': int(session_dates.nunique()),
            }

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
                    period_length
                )
                time_predictions[str(target_time)] = pred_stats

            ticker_results['time_predictability'] = time_predictions

            # Month-by-month analysis if filtering is applied
            if selected_months is not None:
                month_analysis = self._analyze_by_month(session_data, price_col, is_synthetic, selected_months)
                ticker_results['month_breakdown'] = month_analysis

            # Rank strongest patterns
            strongest = self._rank_seasonal_strength(ticker_results)
            ticker_results['strongest_patterns'] = strongest

            results[ticker] = ticker_results

        return results

    def _prepare_momentum_session_cache(
        self,
        session_pairs: Tuple[Tuple[time, time], ...],
        selected_months: Optional[List[int]]
    ) -> Dict[str, Dict[str, any]]:
        """Pre-filter ticker data for momentum screens to avoid redundant work."""
        cache: Dict[str, Dict[str, any]] = {}

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
            if selected_months is not None:
                filtered_data = self._filter_by_months(base_data, selected_months)
                if filtered_data.empty:
                    cache[ticker] = {
                        'data': None,
                        'is_synthetic': is_synthetic,
                        'filtered_months': selected_months,
                        'error': 'No data available for selected months/season'
                    }
                    continue

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
                'filtered_months': selected_months if selected_months else 'all'
            }

        return cache

    def run_screens(
        self,
        screen_params: List[ScreenParams],
        output_format: str = 'dict'
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

        Notes
        -----
        Momentum screens share cached session slices when possible so running
        multiple configurations with overlapping hours avoids redundant
        filtering.

        Returns
        -------
        Union[Dict[str, Dict], pd.DataFrame]
            Composite screening results:
            - If 'dict': {screen_name: {ticker: results}}
            - If 'dataframe': Flattened DataFrame with screen_name and ticker as index

        Examples
        --------
        # Run screens for multiple seasons
        >>> screens = [
        ...     ScreenParams(screen_type='momentum', season='winter',
        ...                  session_starts=["02:30"], session_ends=["10:30"]),
        ...     ScreenParams(screen_type='momentum', season='spring',
        ...                  session_starts=["02:30"], session_ends=["10:30"]),
        ...     ScreenParams(screen_type='seasonality', season='summer',
        ...                  target_times=["09:30", "14:00"])
        ... ]
        >>> results = screener.run_screens(screens)
        >>> # Access results: results['winter_momentum']['CL_F']

        # Get results as DataFrame
        >>> df = screener.run_screens(screens, output_format='dataframe')
        >>> # Access: df.loc[('winter_momentum', 'CL_F')]

        # Custom screen names
        >>> screens = [
        ...     ScreenParams(screen_type='momentum', name='asia_session',
        ...                  season='winter', session_starts=["02:30"],
        ...                  session_ends=["08:30"]),
        ...     ScreenParams(screen_type='momentum', name='us_session',
        ...                  season='winter', session_starts=["09:30"],
        ...                  session_ends=["16:00"])
        ... ]
        >>> results = screener.run_screens(screens)
        """
        composite_results = {}
        momentum_cache: Dict[Tuple[Tuple[time, time], Tuple[time, time], Optional[Tuple[int, ...]]], Dict[str, Dict[str, any]]] = {}

        for params in screen_params:
            if params.screen_type == 'momentum':
                session_starts, session_ends = self._convert_times(params.session_starts, params.session_ends)
                selected_months = self._parse_season_months(params.months, params.season)
                session_pairs = tuple(zip(session_starts, session_ends))
                months_key = tuple(selected_months) if selected_months else None
                cache_key = (tuple(session_starts), tuple(session_ends), months_key)

                if cache_key not in momentum_cache:
                    momentum_cache[cache_key] = self._prepare_momentum_session_cache(session_pairs, selected_months)

                screen_result = self._parse_params(
                    params,
                    session_starts=session_starts,
                    session_ends=session_ends,
                    _selected_months=selected_months,
                    _precomputed_sessions=momentum_cache[cache_key]
                )
            else:
                # Parse parameters and run appropriate screen
                screen_result = self._parse_params(params)

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
            override_session_starts = momentum_kwargs.pop('session_starts', params.session_starts)
            override_session_ends = momentum_kwargs.pop('session_ends', params.session_ends)
            selected_months_override = momentum_kwargs.pop('_selected_months', None)
            precomputed_sessions = momentum_kwargs.pop('_precomputed_sessions', None)

            # Run momentum screen with provided parameters
            return self.intraday_momentum_screen(
                session_starts=override_session_starts,
                session_ends=override_session_ends,
                st_momentum_days=params.st_momentum_days,
                sess_start_hrs=params.sess_start_hrs,
                sess_start_minutes=params.sess_start_minutes,
                sess_end_hrs=params.sess_end_hrs,
                sess_end_mins=params.sess_end_mins,
                test_vol=params.test_vol,
                months=params.months,
                season=params.season,
                _selected_months=selected_months_override,
                _precomputed_sessions=precomputed_sessions,
                **momentum_kwargs
            )

        elif params.screen_type == 'seasonality':
            # Run seasonality screen with provided parameters
            return self.st_seasonality_screen(
                target_times=params.target_times,
                period_length=params.period_length,
                dayofweek_screen=params.dayofweek_screen,
                months=params.months,
                season=params.season,
                session_start=params.seasonality_session_start,
                session_end=params.seasonality_session_end,
                tz=params.tz
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

    def _session_momentum_analysis(
        self,
        ticker: str,
        session_start: time,
        session_end: time,
        start_hrs: int = 1,
        start_mins: int = 30,
        end_hrs: Optional[int] = None,
        end_mins: Optional[int] = None,
        momentum_days: int = 3,
        test_vol: bool = True,
        data: Optional[pd.DataFrame] = None,
        session_data: Optional[pd.DataFrame] = None,
        price_col: Optional[str] = None,
        is_synthetic: Optional[bool] = None
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
        end_mins : Optional[int]
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
        closing_window = timedelta(hours=end_hrs or start_hrs, minutes=end_mins or start_mins)

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
            is_synthetic
        )

        # Calculate closing momentum (last N minutes of session)
        closing_returns = self._calculate_session_window_returns(
            daily_sessions,
            session_end,
            closing_window,
            price_col,
            is_synthetic,
            from_end=True
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

        # Correlation analysis
        correlation_stats = self._calculate_momentum_correlations(
            opening_returns,
            closing_returns,
            st_momentum
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
            'correlations': correlation_stats,
            # Store raw return series for plotting
            'return_series': {
                'opening_returns': opening_returns,
                'closing_returns': closing_returns,
                'full_session_returns': full_session_returns,
                'st_momentum': st_momentum
            }
        }

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
        price_col: str
    ) -> pd.DataFrame:
        """Extract data within session hours for each day."""
        # Filter by time of day
        mask = (data.index.time >= session_start) & (data.index.time <= session_end)
        return data[mask]

    def _calculate_session_window_returns(
        self,
        session_data: pd.DataFrame,
        anchor_time: time,
        window: timedelta,
        price_col: str,
        is_synthetic: bool,
        from_end: bool = False
    ) -> pd.Series:
        """
        Calculate returns for a specific window within each session using vectorized operations.

        Parameters:
        -----------
        session_data : pd.DataFrame
            Data filtered to session hours
        anchor_time : time
            Time to anchor the window (start or end)
        window : timedelta
            Length of the window
        price_col : str
            Column name for prices/spread values
        is_synthetic : bool
            Whether this is a synthetic spread
        from_end : bool
            If True, window goes backward from anchor_time

        Returns:
        --------
        pd.Series
            Daily returns for the window
        """
        if session_data.empty:
            return pd.Series(dtype=float)

        # Create date column for grouping
        session_data = session_data.copy()
        session_data['date'] = session_data.index.date

        # Create time-based masks for window boundaries
        if from_end:
            # Window goes backward from anchor_time
            window_end_time = anchor_time
            window_start_seconds = (pd.Timestamp.combine(pd.Timestamp.today().date(), anchor_time) - window).time()
            window_mask = (session_data.index.time <= window_end_time) & (session_data.index.time >= window_start_seconds)
        else:
            # Window goes forward from anchor_time
            window_start_time = anchor_time
            window_end_seconds = (pd.Timestamp.combine(pd.Timestamp.today().date(), anchor_time) + window).time()
            window_mask = (session_data.index.time >= window_start_time) & (session_data.index.time <= window_end_seconds)

        # Filter to window
        window_data = session_data[window_mask].copy()

        if window_data.empty:
            return pd.Series(dtype=float)

        # Group by date and get first/last prices
        grouped = window_data.groupby('date')[price_col]
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

        # Create date column for grouping
        session_data = session_data.copy()
        session_data['date'] = session_data.index.date

        # Group by date and get first/last prices for the full session
        grouped = session_data.groupby('date')[price_col]
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
        st_momentum: pd.Series
    ) -> Dict[str, float]:
        """Calculate correlations between different momentum measures."""
        # Align series
        combined = pd.DataFrame({
            'open': opening_returns,
            'close': closing_returns,
            'st_mom': st_momentum
        }).dropna()

        if len(combined) < 10:
            return {
                'open_close_corr': np.nan,
                'open_close_pvalue': np.nan,
                'open_st_mom_corr': np.nan,
                'close_st_mom_corr': np.nan
            }

        # Correlation between opening and closing momentum
        open_close_corr, open_close_pval = stats.pearsonr(combined['open'], combined['close'])

        # Correlation between opening momentum and short-term momentum
        open_st_corr, _ = stats.pearsonr(combined['open'], combined['st_mom'])

        # Correlation between closing momentum and short-term momentum
        close_st_corr, _ = stats.pearsonr(combined['close'], combined['st_mom'])

        return {
            'open_close_corr': float(open_close_corr),
            'open_close_pvalue': float(open_close_pval),
            'open_st_mom_corr': float(open_st_corr),
            'close_st_mom_corr': float(close_st_corr),
            'n_observations': len(combined)
        }

    def _analyze_momentum_by_dayofweek(
        self,
        opening_returns: pd.Series,
        closing_returns: pd.Series,
        full_session_returns: pd.Series,
        st_momentum: pd.Series
    ) -> Dict[str, any]:
        """
        Analyze if momentum effects are stronger on specific days of the week.

        Tests whether opening momentum, closing momentum, full session returns,
        and short-term momentum show different characteristics on different weekdays.

        Parameters:
        -----------
        opening_returns : pd.Series
            Opening momentum returns (indexed by date)
        closing_returns : pd.Series
            Closing momentum returns (indexed by date)
        full_session_returns : pd.Series
            Full session returns (indexed by date)
        st_momentum : pd.Series
            Short-term momentum (indexed by date)

        Returns:
        --------
        Dict[str, any]
            Dictionary containing:
            - opening_momentum_by_dow: Stats for opening momentum by weekday
            - closing_momentum_by_dow: Stats for closing momentum by weekday
            - full_session_by_dow: Stats for full session by weekday
            - st_momentum_by_dow: Stats for short-term momentum by weekday
            - anova_tests: ANOVA F-tests for each momentum type across weekdays
        """
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Combine all series into single DataFrame
        df = pd.DataFrame({
            'opening': opening_returns,
            'closing': closing_returns,
            'full_session': full_session_returns,
            'st_momentum': st_momentum
        }).dropna()

        if len(df) < 10:
            return {
                'error': 'Insufficient data for day-of-week analysis',
                'n': len(df)
            }

        # Ensure index is DatetimeIndex and add weekday column
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df['weekday'] = df.index.dayofweek

        # Analyze each momentum type by weekday
        momentum_types = {
            'opening_momentum': 'opening',
            'closing_momentum': 'closing',
            'full_session': 'full_session',
            'st_momentum': 'st_momentum'
        }

        results = {}

        for momentum_name, col_name in momentum_types.items():
            weekday_stats = {}
            weekday_data_for_anova = []

            for dow in range(5):  # Monday=0 to Friday=4
                day_data = df[df['weekday'] == dow][col_name]

                if len(day_data) < 2:
                    continue

                weekday_data_for_anova.append(day_data.values)
                day_name = weekday_names[dow]

                # Calculate statistics for this weekday
                mean_val = day_data.mean()
                std_val = day_data.std()
                sharpe = mean_val / std_val if std_val > 0 else np.nan

                # T-statistic against zero (testing if mean is significantly different from 0)
                t_stat = mean_val / (std_val / np.sqrt(len(day_data))) if std_val > 0 else np.nan

                weekday_stats[day_name] = {
                    'n': len(day_data),
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'sharpe': float(sharpe),
                    'skew': float(day_data.skew()),
                    'positive_pct': float((day_data > 0).sum() / len(day_data)),
                    't_stat': float(t_stat) if not np.isnan(t_stat) else np.nan
                }

            # ANOVA test across weekdays for this momentum type
            if len(weekday_data_for_anova) >= 3:
                from scipy.stats import f_oneway
                f_stat, p_value = f_oneway(*weekday_data_for_anova)
                weekday_stats['anova'] = {
                    'f_stat': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': (
                        f"Significant day-of-week effect (p={p_value:.4f})" if p_value < 0.05
                        else f"No significant day-of-week effect (p={p_value:.4f})"
                    )
                }

            results[f'{momentum_name}_by_dow'] = weekday_stats

        # Overall summary
        results['summary'] = {
            'total_observations': len(df),
            'n_weekdays_analyzed': 5,
            'significant_patterns': []
        }

        # Identify which momentum types have significant day-of-week effects
        for momentum_name in momentum_types.keys():
            key = f'{momentum_name}_by_dow'
            if key in results and 'anova' in results[key]:
                if results[key]['anova']['significant']:
                    results['summary']['significant_patterns'].append({
                        'momentum_type': momentum_name,
                        'f_stat': results[key]['anova']['f_stat'],
                        'p_value': results[key]['anova']['p_value']
                    })

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

        # Test correlation between opening and closing volatility
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
        for dow in range(5):  # Monday=0 to Friday=4
            day_data = vol_df[vol_df['weekday'] == dow]

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
                {'date': date, 'volatility': float(vol), 'z_score': float(vol_df.loc[pd.to_datetime(date), 'z_score'])}
                for date, vol in zip(high_vol_days, high_vol_values)
            ],
            'low_volatility_days': [
                {'date': date, 'volatility': float(vol), 'z_score': float(vol_df.loc[pd.to_datetime(date), 'z_score'])}
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

        # Vectorized approach - create date column and group by
        session_data_copy = session_data.copy()
        session_data_copy['date'] = session_data_copy.index.date
        session_data_copy['time'] = session_data_copy.index.time

        # Group by date and calculate aggregates in one operation
        vol_grouped = session_data_copy.groupby('date')['Volume']
        total_vols = vol_grouped.sum()
        avg_vols = vol_grouped.mean()

        if len(total_vols) == 0:
            return {}

        # Analyze volume by time of day to find peaks and troughs
        time_of_day_vol = session_data_copy.groupby('time')['Volume'].agg(['mean', 'std', 'count'])
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
                'time': str(time),
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
                'time': str(time),
                'avg_volume': float(row['mean']),
                'z_score': float(row['z_score']),
                'n_observations': int(row['count'])
            }
            for time, row in troughs.iterrows()
        ]

        # Identify significant peaks (z-score > 1.0)
        significant_peaks = time_of_day_vol[time_of_day_vol['z_score'] > 1.0]
        optimal_trading_times = [str(t) for t in significant_peaks.index]

        # Identify times to avoid (z-score < -1.0)
        avoid_times_data = time_of_day_vol[time_of_day_vol['z_score'] < -1.0]
        avoid_trading_times = [str(t) for t in avoid_times_data.index]

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

        # Add weekday labels
        df = pd.DataFrame({'return': returns, 'weekday': returns.index.dayofweek})
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Returns analysis by weekday
        returns_stats = {}
        weekday_returns = []

        for dow in range(5):  # Monday=0 to Friday=4
            day_data = df[df['weekday'] == dow]['return']
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

        for dow in range(5):
            day_data = df[df['weekday'] == dow]['return']
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
        period_length: Optional[timedelta] = None
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

        Returns:
        --------
        Dict with correlation statistics for various lags and weekday breakdown for next-week correlation
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
        daily_returns = daily_returns.groupby(daily_returns.index.date).sum()

        if len(daily_returns) < 20:
            return {
                'n': len(daily_returns),
                'next_day_corr': np.nan,
                'next_day_pvalue': np.nan,
                'next_week_corr': np.nan,
                'next_week_pvalue': np.nan,
                'error': 'Insufficient data'
            }

        # Next-day correlation
        x_1d = daily_returns[:-1].values
        y_1d = daily_returns[1:].values
        if len(x_1d) >= 20:
            r_1d, p_1d = stats.pearsonr(x_1d, y_1d)
        else:
            r_1d, p_1d = np.nan, np.nan

        # Next-week correlation (5-7 business days)
        lag_week = 5
        if len(daily_returns) > lag_week + 10:
            x_1w = daily_returns[:-lag_week].values
            y_1w = daily_returns[lag_week:].values
            r_1w, p_1w = stats.pearsonr(x_1w, y_1w)
        else:
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
            'next_week_significant': p_1w < 0.05 if not np.isnan(p_1w) else False
        }

        # Add weekday prevalence if available
        if weekday_analysis is not None:
            result['weekday_prevalence'] = weekday_analysis

        return result

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

        # Calculate correlation for each weekday
        weekday_correlations = {}
        weekday_counts = {}

        for dow in range(5):  # Monday=0 to Friday=4
            day_data = df[df['weekday'] == dow]

            if len(day_data) < 5:  # Need minimum 5 observations
                continue

            # Calculate correlation for this weekday
            if len(day_data) >= 5:
                try:
                    corr, p_val = stats.pearsonr(day_data['lagged'], day_data['current'])
                    weekday_correlations[weekday_names[dow]] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'n': len(day_data),
                        'significant': p_val < 0.05,
                        'abs_correlation': abs(corr)
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
            'total_observations': len(df)
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
                    'strength': abs(dow_returns['anova']['f_stat'])
                })

            # Check individual weekday t-stats
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                if day in dow_returns:
                    day_stats = dow_returns[day]
                    t_stat = day_stats.get('t_stat', 0)
                    if abs(t_stat) > 1.96:  # ~95% confidence
                        patterns.append({
                            'type': 'weekday_mean',
                            'day': day,
                            'description': f'{day} has significant mean return',
                            'mean': day_stats['mean'],
                            't_stat': t_stat,
                            'p_value': 2 * (1 - stats.t.cdf(abs(t_stat), df=day_stats['n']-1)) if day_stats['n'] > 1 else 1.0,
                            'strength': abs(t_stat)
                        })

        # Check time predictability
        if 'time_predictability' in ticker_results:
            for time_str, pred_stats in ticker_results['time_predictability'].items():
                # Next-day prediction
                if pred_stats.get('next_day_significant', False):
                    patterns.append({
                        'type': 'time_predictive_nextday',
                        'time': time_str,
                        'description': f'{time_str} predicts next day return',
                        'correlation': pred_stats['next_day_corr'],
                        'p_value': pred_stats['next_day_pvalue'],
                        'strength': abs(pred_stats['next_day_corr'])
                    })

                # Next-week prediction
                if pred_stats.get('next_week_significant', False):
                    pattern_entry = {
                        'type': 'time_predictive_nextweek',
                        'time': time_str,
                        'description': f'{time_str} predicts next week return',
                        'correlation': pred_stats['next_week_corr'],
                        'p_value': pred_stats['next_week_pvalue'],
                        'strength': abs(pred_stats['next_week_corr'])
                    }

                    # Add weekday prevalence information if available
                    if 'weekday_prevalence' in pred_stats:
                        weekday_info = pred_stats['weekday_prevalence']
                        if 'most_prevalent_day' in weekday_info and weekday_info['most_prevalent_day']:
                            pattern_entry['most_prevalent_day'] = weekday_info['most_prevalent_day']
                            pattern_entry['strongest_days'] = weekday_info.get('strongest_days', [])
                            pattern_entry['description'] = (
                                f'{time_str} predicts next week return '
                                f'(strongest on {weekday_info["most_prevalent_day"]})'
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
        """
        Filter DataFrame to only include data from specified months.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data with DatetimeIndex
        months : List[int]
            Month numbers to include (1-12)

        Returns:
        --------
        pd.DataFrame
            Filtered data
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data must have a DatetimeIndex")

        month_mask = data.index.month.isin(months)
        return data[month_mask]

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

            # Calculate correlation
            corr, p_val = stats.pearsonr(aligned['current'], aligned[f'lag_{lag}'])

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
                df, gaps = mgr.load_front_month_series(
                    symbol=symbol,
                    start=start,
                    end=end,
                    resample_rule=resample_rule,
                    volume_bucket_size=volume_bucket
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



def main():
    return