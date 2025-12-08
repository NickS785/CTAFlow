"""Utilities for backtesting calendar patterns individually.

This module provides helpers to backtest calendar patterns without month filtering,
treating each calendar pattern as a separate backtest run.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import pandas as pd
from .screener_pipeline import ScreenerPipeline
from .backtester import ScreenerBacktester


def backtest_calendar_patterns_individually(
    bars: pd.DataFrame,
    calendar_patterns: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float = 0.0,
    use_side_hint: bool = True,
    group_field: Optional[str] = None,
    pipeline: Optional[ScreenerPipeline] = None,
    backtester: Optional[ScreenerBacktester] = None,
    tz: str = 'UTC',
) -> Dict[str, Dict[str, Any]]:
    """Backtest each calendar pattern individually with month filtering applied.

    This function creates separate backtest runs for each calendar pattern,
    ensuring that each pattern is backtested independently and month filtering
    is applied per pattern (if months are specified in the pattern payload).

    Args:
        bars: DataFrame with bar data (must have ts, session_date, open, close columns)
        calendar_patterns: Dict mapping pattern names to pattern payloads
                          Each payload should have:
                          - pattern_type: 'calendar'
                          - event: calendar event name
                          - horizon: trading horizon
                          - calendar_pattern: pattern identifier
                          - months: (optional) list of months to filter [1-12]
        threshold: Signal threshold for backtesting
        use_side_hint: Whether to use correlation hints
        group_field: Optional field for grouping (e.g., 'ticker')
        pipeline: Optional ScreenerPipeline instance (will create if None)
        backtester: Optional ScreenerBacktester instance (will create if None)
        tz: Timezone for pipeline (default: 'UTC')

    Returns:
        Dictionary mapping pattern names to backtest results

    Example:
        >>> calendar_patterns = {
        ...     'first_day_jan': {
        ...         'pattern_type': 'calendar',
        ...         'event': 'first_day_of_month',
        ...         'horizon': '1d',
        ...         'calendar_pattern': 'edge_day',
        ...         'months': [1],  # January only
        ...     },
        ...     'last_day_dec': {
        ...         'pattern_type': 'calendar',
        ...         'event': 'last_day_of_month',
        ...         'horizon': '1d',
        ...         'calendar_pattern': 'edge_day',
        ...         'months': [12],  # December only
        ...     },
        ... }
        >>> results = backtest_calendar_patterns_individually(bars, calendar_patterns)
        >>> for pattern_name, result in results.items():
        ...     print(f"{pattern_name}: {result['summary'].total_return:.4f}")
    """
    if pipeline is None:
        pipeline = ScreenerPipeline(tz=tz)

    if backtester is None:
        backtester = ScreenerBacktester(use_gpu=pipeline.use_gpu)

    # Build XY data for each pattern separately
    xy_map = {}

    for pattern_name, pattern_payload in calendar_patterns.items():
        # Ensure pattern has required calendar fields
        if pattern_payload.get('pattern_type') != 'calendar':
            continue

        # Create single-pattern dict for pipeline
        single_pattern = {pattern_name: pattern_payload}

        try:
            # Build features for this pattern (gates will include month filtering)
            featured = pipeline.build_features(bars, single_pattern)

            # Build XY data for this pattern
            mapper = pipeline._get_horizon_mapper('auto')
            xy = mapper.build_xy(featured, single_pattern)

            # Store XY data if not empty
            if not xy.empty:
                xy_map[pattern_name] = xy
        except Exception as e:
            # Skip patterns that fail to process
            print(f"Warning: Skipping pattern {pattern_name}: {e}")
            continue

    # Run batch backtest on all patterns
    if not xy_map:
        return {}

    results = backtester.batch_patterns(
        xy_map,
        threshold=threshold,
        use_side_hint=use_side_hint,
        group_field=group_field,
    )

    return results


def create_calendar_pattern_payload(
    event: str,
    horizon: str = '1d',
    calendar_pattern: str = 'edge_day',
    **kwargs
) -> Dict[str, Any]:
    """Create a calendar pattern payload dictionary.

    Args:
        event: Calendar event name (e.g., 'first_day_of_month', 'last_day_of_quarter')
        horizon: Trading horizon (e.g., '1d', '2d', '1w')
        calendar_pattern: Pattern type (e.g., 'edge_day', 'lead_lag')
        **kwargs: Additional pattern parameters

    Returns:
        Pattern payload dictionary ready for backtesting

    Example:
        >>> payload = create_calendar_pattern_payload(
        ...     event='first_day_of_month',
        ...     horizon='1d',
        ...     mean=0.0015,
        ...     p_value=0.03,
        ... )
    """
    payload = {
        'pattern_type': 'calendar',
        'event': event,
        'horizon': horizon,
        'calendar_pattern': calendar_pattern,
    }
    payload.update(kwargs)
    return payload


def calendar_patterns_from_dataframe(
    patterns_df: pd.DataFrame,
    event_col: str = 'event',
    horizon_col: str = 'horizon',
    pattern_col: str = 'pattern',
    name_template: str = 'cal_{idx}',
) -> Dict[str, Dict[str, Any]]:
    """Convert a DataFrame of calendar patterns to pattern payload dict.

    Args:
        patterns_df: DataFrame with calendar pattern information
        event_col: Column name for event (default: 'event')
        horizon_col: Column name for horizon (default: 'horizon')
        pattern_col: Column name for calendar_pattern (default: 'pattern')
        name_template: Template for generating pattern names (default: 'cal_{idx}')
                      Use {idx} for row index, {event} for event name, {horizon} for horizon

    Returns:
        Dictionary mapping pattern names to payloads

    Example:
        >>> # patterns_df has columns: event, horizon, pattern, mean, p_value
        >>> payloads = calendar_patterns_from_dataframe(patterns_df)
        >>> results = backtest_calendar_patterns_individually(bars, payloads)
    """
    pattern_payloads = {}

    for idx, row in patterns_df.iterrows():
        # Generate pattern name from template
        name = name_template.format(
            idx=idx,
            event=row.get(event_col, ''),
            horizon=row.get(horizon_col, ''),
        )

        # Create payload with all columns
        payload = {
            'pattern_type': 'calendar',
            'event': row[event_col],
            'horizon': row[horizon_col],
            'calendar_pattern': row[pattern_col],
        }

        # Add any additional columns from the row
        for col in patterns_df.columns:
            if col not in [event_col, horizon_col, pattern_col]:
                payload[col] = row[col]

        pattern_payloads[name] = payload

    return pattern_payloads
