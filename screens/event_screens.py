"""
Event-driven screening for futures markets using scheduled economic data releases.

This module provides screening functions for analyzing price reactions to scheduled
events like EIA reports, USDA releases, Fed announcements, and other calendar-based
market-moving events.

Examples:
    # Screen natural gas around EIA storage reports
    results, screener = natural_gas_storage_screen(start_date="2020-01-01")

    # Screen crude oil around inventory reports
    results, screener = crude_oil_inventory_screen(start_date="2020-01-01")

    # Screen grains around USDA WASDE reports
    results, screener = usda_wasde_screen(start_date="2020-01-01")
"""

from CTAFlow.screeners.event_screener import run_event_screener
from CTAFlow.screeners.params import EventParams
from CTAFlow.strategy import ScreenerPipeline
from CTAFlow.data import DataClient, read_exported_df
from CTAFlow.config import INTRADAY_DATA_PATH, DLY_DATA_PATH
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple


# Initialize data client
cli = DataClient()


def create_eia_storage_calendar(start_date: str = "2020-01-01",
                                  end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Create event calendar for EIA Natural Gas Storage reports.

    EIA releases weekly natural gas storage data every Thursday at 10:30 AM ET
    during the heating season (October-March) and at 10:30 AM ET year-round.

    Args:
        start_date: Start date for calendar (YYYY-MM-DD)
        end_date: End date for calendar (default: today)

    Returns:
        DataFrame with columns:
            - release_ts: Timestamp of scheduled release (Thursday 10:30 ET)
            - event_code: 'EIA_NG_STORAGE'
            - value: Actual storage change (bcf) - to be filled
            - consensus: Bloomberg consensus forecast (bcf) - to be filled

    Notes:
        This is a template calendar. Actual values and consensus forecasts
        should be populated from:
        - EIA API: https://www.eia.gov/opendata/
        - Bloomberg Terminal: EIA_NG_STORAGE <GO>
        - Trading Economics API
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Generate all Thursdays between start and end
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    thursdays = date_range[date_range.dayofweek == 3]  # Thursday = 3

    # Create release timestamps at 10:30 AM ET
    release_times = pd.to_datetime([
        f"{date.strftime('%Y-%m-%d')} 10:30:00"
        for date in thursdays
    ])

    # Localize to Eastern Time
    release_times = release_times.tz_localize('America/New_York')

    # Create calendar DataFrame
    calendar = pd.DataFrame({
        'release_ts': release_times,
        'event_code': 'EIA_NG_STORAGE',
        'value': np.nan,       # Populate with actual storage change
        'consensus': np.nan,   # Populate with Bloomberg consensus
    })

    print(f"Created EIA storage calendar: {len(calendar)} events from {start_date} to {end_date}")
    print("NOTE: 'value' and 'consensus' columns need to be populated from data source")

    return calendar


def create_eia_petroleum_calendar(start_date: str = "2020-01-01",
                                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Create event calendar for EIA Petroleum Status Reports.

    EIA releases weekly petroleum inventory data every Wednesday at 10:30 AM ET.

    Args:
        start_date: Start date for calendar
        end_date: End date for calendar (default: today)

    Returns:
        DataFrame with EIA petroleum inventory release schedule
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Generate all Wednesdays
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    wednesdays = date_range[date_range.dayofweek == 2]  # Wednesday = 2

    release_times = pd.to_datetime([
        f"{date.strftime('%Y-%m-%d')} 10:30:00"
        for date in wednesdays
    ])
    release_times = release_times.tz_localize('America/New_York')

    calendar = pd.DataFrame({
        'release_ts': release_times,
        'event_code': 'EIA_PETROLEUM',
        'crude_stocks': np.nan,        # Crude oil stocks change
        'gasoline_stocks': np.nan,     # Gasoline stocks change
        'distillate_stocks': np.nan,   # Distillate stocks change
        'crude_consensus': np.nan,     # Crude consensus
        'gasoline_consensus': np.nan,  # Gasoline consensus
    })

    print(f"Created EIA petroleum calendar: {len(calendar)} events")

    return calendar


def create_usda_wasde_calendar(start_date: str = "2020-01-01",
                                end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Create event calendar for USDA WASDE (World Agricultural Supply and Demand Estimates).

    USDA releases WASDE reports monthly, typically around the 10th of each month
    at 12:00 PM ET.

    Args:
        start_date: Start date for calendar
        end_date: End date for calendar (default: today)

    Returns:
        DataFrame with USDA WASDE release schedule
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # WASDE is typically released on the 10th of each month (or next business day)
    # Approximate schedule - actual dates should be fetched from USDA calendar
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start

    # Set to 10th of each month at 12:00 PM ET
    release_dates = [
        pd.Timestamp(f"{date.year}-{date.month:02d}-10 12:00:00")
        for date in date_range
    ]

    release_times = pd.DatetimeIndex(release_dates).tz_localize('America/New_York')

    calendar = pd.DataFrame({
        'release_ts': release_times,
        'event_code': 'USDA_WASDE',
        'corn_yield': np.nan,      # Corn yield estimate
        'soy_yield': np.nan,       # Soybean yield estimate
        'wheat_yield': np.nan,     # Wheat yield estimate
        'corn_consensus': np.nan,  # Corn consensus
        'soy_consensus': np.nan,   # Soy consensus
    })

    print(f"Created USDA WASDE calendar: {len(calendar)} events")
    print("NOTE: Actual release dates may vary - verify against USDA calendar")

    return calendar


def natural_gas_storage_screen(
    start_date: str = "2020-01-01",
    event_calendar: Optional[pd.DataFrame] = None,
    event_window_pre_minutes: int = 30,
    event_window_post_minutes: int = 60,
    include_t1_close: bool = True,
    surprise_mode: str = 'diff',
    min_events: int = 10,
) -> Tuple[Dict[str, any], pd.DataFrame]:
    """
    Screen natural gas (NG) futures around EIA weekly storage reports.

    Analyzes:
    - Pre-event momentum (30 minutes before)
    - Post-event reaction (60 minutes after)
    - T+0 close (end of day)
    - T+1 close (next trading day)
    - Surprise impact (actual vs consensus)

    Args:
        start_date: Start date for historical data
        event_calendar: Pre-built event calendar with actual/consensus data
                       If None, creates template calendar (needs manual population)
        event_window_pre_minutes: Minutes before release to measure pre-event price
        event_window_post_minutes: Minutes after release to measure post-event price
        include_t1_close: Include next-day close returns
        surprise_mode: How to compute surprise ('diff', 'pct', or 'z')
        min_events: Minimum events required for statistical tests

    Returns:
        Tuple of (results_dict, event_calendar)
        results_dict contains:
            - 'event_results': EventScreenerResult with events, summary, patterns
            - 'bars': Raw OHLCV data used
            - 'metadata': Screen configuration and statistics

    Example:
        >>> # Create calendar with actual data
        >>> calendar = create_eia_storage_calendar("2020-01-01")
        >>> # Populate calendar.value and calendar.consensus from data source
        >>>
        >>> # Run screen
        >>> results, calendar = natural_gas_storage_screen(
        ...     start_date="2020-01-01",
        ...     event_calendar=calendar,
        ...     surprise_mode='diff'
        ... )
        >>>
        >>> # Analyze results
        >>> event_result = results['event_results']
        >>> print(event_result.summary)  # Statistical summary
        >>> print(event_result.patterns)  # Tradeable patterns
    """
    print("="*70)
    print("NATURAL GAS EIA STORAGE EVENT SCREEN")
    print("="*70)

    # Load NG futures data (5-minute bars)
    print(f"\n[1] Loading NG futures data from {start_date}...")
    bars = read_exported_df(INTRADAY_DATA_PATH / "CSV/NG_5min.csv")
    bars = bars.loc[start_date:]
    print(f"    Loaded {len(bars)} bars from {bars.index.min()} to {bars.index.max()}")

    # Create or validate event calendar
    if event_calendar is None:
        print("\n[2] Creating template event calendar...")
        event_calendar = create_eia_storage_calendar(start_date)
        print("    WARNING: Calendar created without actual/consensus values")
        print("    Populate 'value' and 'consensus' columns before running with surprise analysis")
    else:
        print(f"\n[2] Using provided event calendar: {len(event_calendar)} events")

    # Configure event parameters
    # Only use surprise analysis if calendar has actual/consensus data
    has_fundamentals = (
        event_calendar is not None
        and 'value' in event_calendar.columns
        and 'consensus' in event_calendar.columns
        and event_calendar['value'].notna().any()
        and event_calendar['consensus'].notna().any()
    )

    params = EventParams(
        event_code='EIA_NG_STORAGE',
        event_window_pre_minutes=event_window_pre_minutes,
        event_window_post_minutes=event_window_post_minutes,
        include_t1_close=include_t1_close,
        value_col='value' if has_fundamentals else None,
        consensus_col='consensus' if has_fundamentals else None,
        surprise_mode=surprise_mode if has_fundamentals else None,
        min_events=min_events,
        corr_threshold=0.3,  # Minimum correlation for pattern detection
    )

    print(f"\n[3] Running event screener...")
    print(f"    Pre-event window: {event_window_pre_minutes} minutes")
    print(f"    Post-event window: {event_window_post_minutes} minutes")
    if has_fundamentals:
        print(f"    Surprise analysis: ENABLED (mode={surprise_mode})")
    else:
        print(f"    Surprise analysis: DISABLED (pure price behavior analysis)")
    print(f"    Minimum events: {min_events}")

    # Run event screener
    event_result = run_event_screener(
        bars=bars,
        events=event_calendar,
        params=params,
        symbol='NG',
        instrument_tz='America/Chicago',  # CME timezone
    )

    # Compile results
    results = {
        'event_results': event_result,
        'bars': bars,
        'metadata': {
            'symbol': 'NG',
            'start_date': start_date,
            'n_bars': len(bars),
            'n_events': len(event_result.events),
            'params': params,
        }
    }

    print(f"\n[4] Event screening complete")
    print(f"    Events analyzed: {len(event_result.events)}")
    print(f"    Patterns detected: {len(event_result.patterns)}")

    # Display summary statistics
    if not event_result.summary.empty:
        print(f"\n[5] Summary Statistics:")
        summary = event_result.summary.iloc[0]
        print(f"    Mean event return: {summary['mean_r_event']:.4f}")
        print(f"    Mean T0 return: {summary['mean_r_T0']:.4f}")
        if 'rho_event_T0' in summary:
            print(f"    Event-T0 correlation: {summary['rho_event_T0']:.3f}")
        if 'rho_surprise_event' in summary:
            print(f"    Surprise-event correlation: {summary['rho_surprise_event']:.3f}")

    # Display detected patterns
    if event_result.patterns:
        print(f"\n[6] Detected Patterns:")
        for pattern in event_result.patterns:
            print(f"    * {pattern['pattern_type']:30s} strength={pattern['strength']:.3f} "
                  f"direction={pattern['gate_direction']:+d}")

    print("\n" + "="*70)

    return results, event_calendar


def crude_oil_inventory_screen(
    start_date: str = "2020-01-01",
    tickers: List[str] = ["CL", "RB", "HO"],
    event_calendar: Optional[pd.DataFrame] = None,
    event_window_pre_minutes: int = 30,
    event_window_post_minutes: int = 60,
) -> Tuple[Dict[str, any], pd.DataFrame]:
    """
    Screen crude oil and products around EIA petroleum inventory reports.

    Analyzes multiple related instruments:
    - CL: Crude oil futures
    - RB: RBOB gasoline futures
    - HO: Heating oil futures

    Args:
        start_date: Start date for historical data
        tickers: List of tickers to screen (default: CL, RB, HO)
        event_calendar: Event calendar with petroleum inventory data
        event_window_pre_minutes: Pre-event window
        event_window_post_minutes: Post-event window

    Returns:
        Tuple of (results_dict, event_calendar)
        results_dict contains results for each ticker
    """
    print("="*70)
    print("CRUDE OIL & PRODUCTS EIA INVENTORY EVENT SCREEN")
    print("="*70)

    # Create event calendar if not provided
    if event_calendar is None:
        print("\n[1] Creating EIA petroleum inventory calendar...")
        event_calendar = create_eia_petroleum_calendar(start_date)
    else:
        print(f"\n[1] Using provided event calendar: {len(event_calendar)} events")

    results = {}

    # Process each ticker
    for ticker in tickers:
        print(f"\n[2] Processing {ticker}...")

        # Load data
        bars = read_exported_df(INTRADAY_DATA_PATH / f"CSV/{ticker}_5min.csv")
        bars = bars.loc[start_date:]
        print(f"    Loaded {len(bars)} bars")

        # Determine which inventory component to use based on ticker
        value_col = 'crude_stocks' if ticker == 'CL' else \
                   'gasoline_stocks' if ticker == 'RB' else \
                   'distillate_stocks' if ticker == 'HO' else 'crude_stocks'

        consensus_col = 'crude_consensus' if ticker == 'CL' else \
                       'gasoline_consensus' if ticker == 'RB' else \
                       'crude_consensus'  # Default to crude

        # Check if calendar has fundamental data
        has_fundamentals = (
            event_calendar is not None
            and value_col in event_calendar.columns
            and consensus_col in event_calendar.columns
            and event_calendar[value_col].notna().any()
            and event_calendar[consensus_col].notna().any()
        )

        # Configure parameters
        params = EventParams(
            event_code='EIA_PETROLEUM',
            event_window_pre_minutes=event_window_pre_minutes,
            event_window_post_minutes=event_window_post_minutes,
            value_col=value_col if has_fundamentals else None,
            consensus_col=consensus_col if has_fundamentals else None,
            surprise_mode='diff' if has_fundamentals else None,
            min_events=10,
        )

        # Run event screener
        event_result = run_event_screener(
            bars=bars,
            events=event_calendar,
            params=params,
            symbol=ticker,
            instrument_tz='America/Chicago',
        )

        results[ticker] = {
            'event_results': event_result,
            'bars': bars,
            'patterns': event_result.patterns,
        }

        print(f"    Events: {len(event_result.events)}, Patterns: {len(event_result.patterns)}")

    print("\n" + "="*70)

    return results, event_calendar


def usda_wasde_screen(
    start_date: str = "2020-01-01",
    tickers: List[str] = ["ZC", "ZS", "ZW"],
    event_calendar: Optional[pd.DataFrame] = None,
    event_window_pre_minutes: int = 60,
    event_window_post_minutes: int = 120,
) -> Tuple[Dict[str, any], pd.DataFrame]:
    """
    Screen grain futures around USDA WASDE reports.

    WASDE (World Agricultural Supply and Demand Estimates) is released monthly
    and impacts corn, soybean, and wheat markets.

    Args:
        start_date: Start date for historical data
        tickers: List of grain tickers (default: ZC, ZS, ZW)
        event_calendar: USDA WASDE calendar
        event_window_pre_minutes: Pre-event window (longer for WASDE)
        event_window_post_minutes: Post-event window (longer for WASDE)

    Returns:
        Tuple of (results_dict, event_calendar)
    """
    print("="*70)
    print("GRAIN FUTURES USDA WASDE EVENT SCREEN")
    print("="*70)

    if event_calendar is None:
        print("\n[1] Creating USDA WASDE calendar...")
        event_calendar = create_usda_wasde_calendar(start_date)
    else:
        print(f"\n[1] Using provided event calendar: {len(event_calendar)} events")

    results = {}

    for ticker in tickers:
        print(f"\n[2] Processing {ticker}...")

        bars = read_exported_df(INTRADAY_DATA_PATH / f"CSV/{ticker}_5min.csv")
        bars = bars.loc[start_date:]

        # Map ticker to WASDE field
        value_col = 'corn_yield' if ticker == 'ZC' else \
                   'soy_yield' if ticker == 'ZS' else \
                   'wheat_yield' if ticker == 'ZW' else 'corn_yield'

        consensus_col = 'corn_consensus' if ticker == 'ZC' else \
                       'soy_consensus' if ticker == 'ZS' else 'corn_consensus'

        # Check if calendar has fundamental data
        has_fundamentals = (
            event_calendar is not None
            and value_col in event_calendar.columns
            and consensus_col in event_calendar.columns
            and event_calendar[value_col].notna().any()
            and event_calendar[consensus_col].notna().any()
        )

        params = EventParams(
            event_code='USDA_WASDE',
            event_window_pre_minutes=event_window_pre_minutes,
            event_window_post_minutes=event_window_post_minutes,
            value_col=value_col if has_fundamentals else None,
            consensus_col=consensus_col if has_fundamentals else None,
            surprise_mode='pct' if has_fundamentals else None,  # Percentage surprise for yield data
            min_events=5,  # Lower threshold (monthly reports)
        )

        event_result = run_event_screener(
            bars=bars,
            events=event_calendar,
            params=params,
            symbol=ticker,
            instrument_tz='America/Chicago',
        )

        results[ticker] = {
            'event_results': event_result,
            'bars': bars,
            'patterns': event_result.patterns,
        }

        print(f"    Events: {len(event_result.events)}, Patterns: {len(event_result.patterns)}")

    print("\n" + "="*70)

    return results, event_calendar


def backtest_event_patterns(
    results: Dict[str, any],
    ticker: str,
    threshold: float = 0.01,
    use_gpu: bool = True,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Backtest event patterns using ScreenerPipeline.

    Args:
        results: Results dict from event screen (e.g., natural_gas_storage_screen)
        ticker: Ticker to backtest
        threshold: Threshold for backtesting (default: 0.01)
        use_gpu: Enable GPU acceleration
        verbose: Enable verbose logging

    Returns:
        Dict with backtest results for each pattern
    """
    print(f"\nBacktesting event patterns for {ticker}...")

    # Extract patterns and data
    patterns = results['event_results'].patterns
    bars = results['bars']

    if not patterns:
        print("  No patterns to backtest")
        return {}

    print(f"  Found {len(patterns)} patterns to backtest")

    # Initialize pipeline
    pipeline = ScreenerPipeline(use_gpu=use_gpu)

    # Run backtests
    backtest_results = pipeline.concurrent_pattern_backtests(
        bars=bars,
        patterns=patterns,
        threshold=threshold,
        verbose=verbose,
    )

    print(f"  Backtested {len(backtest_results)} patterns")

    return backtest_results


if __name__ == "__main__":
    """
    Example usage: Screen natural gas around EIA storage reports.

    NOTE: This example uses template calendars. For production use:
    1. Fetch actual event data from EIA API, Bloomberg, or Trading Economics
    2. Populate 'value' and 'consensus' columns in event calendars
    3. Adjust event windows based on instrument volatility
    4. Validate timezone handling for your data source
    """

    # Example 1: Natural gas storage screen with template calendar
    print("\n" + "="*70)
    print("EXAMPLE 1: Natural Gas EIA Storage Screen")
    print("="*70)

    results_ng, calendar_ng = natural_gas_storage_screen(
        start_date="2020-01-01",
        event_calendar=None,  # Will create template
        event_window_pre_minutes=30,
        event_window_post_minutes=60,
        surprise_mode='diff',
    )

    # To use with actual data:
    # 1. Fetch EIA storage data
    # calendar_ng['value'] = fetch_eia_storage_actuals()
    # calendar_ng['consensus'] = fetch_bloomberg_consensus()
    #
    # 2. Re-run screen with populated calendar
    # results_ng, _ = natural_gas_storage_screen(
    #     start_date="2020-01-01",
    #     event_calendar=calendar_ng,
    # )
    #
    # 3. Backtest patterns
    # backtest_results = backtest_event_patterns(results_ng, 'NG', threshold=0.01)

    # Example 2: Crude oil inventory screen
    print("\n" + "="*70)
    print("EXAMPLE 2: Crude Oil & Products Inventory Screen")
    print("="*70)

    results_oil, calendar_oil = crude_oil_inventory_screen(
        start_date="2020-01-01",
        tickers=["CL", "RB"],
        event_window_pre_minutes=30,
        event_window_post_minutes=60, 
    )

    # Example 3: USDA WASDE grains screen
    print("\n" + "="*70)
    print("EXAMPLE 3: Grain Futures USDA WASDE Screen")
    print("="*70)

    results_grains, calendar_wasde = usda_wasde_screen(
        start_date="2020-01-01",
        tickers=["ZC", "ZS"],
        event_window_pre_minutes=60,
        event_window_post_minutes=120,
    )

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Populate event calendars with actual data from EIA/USDA/Bloomberg")
    print("2. Re-run screens with complete event data")
    print("3. Backtest detected patterns using backtest_event_patterns()")
    print("4. Integrate significant patterns into trading strategies")
