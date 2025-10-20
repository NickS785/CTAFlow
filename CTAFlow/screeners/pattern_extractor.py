"""
Pattern extraction module for screener results.

This module processes results from HistoricalScreener and OrderflowScanner to identify
significant recurring patterns (weekly, monthly, seasonal, etc.) and generate time series
for each pattern.

Supported screener types:
- Orderflow scans: events, buckets, wom_weekday, weekly_peak, weekly, intraday_pressure
- Historical momentum screens: session returns, momentum correlations
- Historical seasonality screens: time-of-day patterns, day-of-week patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Literal, Any
from datetime import datetime, date, time, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np


__all__ = [
    "PatternExtractor",
    "PatternResult",
    "extract_patterns_from_orderflow",
    "extract_patterns_from_historical",
]


@dataclass
class PatternResult:
    """
    Container for a single extracted pattern with its time series.

    Attributes
    ----------
    pattern_type : str
        Type of pattern (e.g., 'weekly', 'wom_weekday', 'seasonal', 'event')
    symbol : str
        Trading symbol this pattern applies to
    metric : str
        Metric name (e.g., 'buy_pressure', 'momentum', 'returns')
    description : str
        Human-readable description of the pattern
    trigger_conditions : Dict[str, Any]
        Conditions that trigger this pattern (e.g., {'weekday': 'Monday', 'week_of_month': 1})
    dates : pd.DatetimeIndex
        Dates when this pattern occurred
    values : np.ndarray
        Values corresponding to each date
    statistics : Dict[str, float]
        Statistical measures (mean, std, t_stat, p_value, etc.)
    significance : str
        Significance level ('high', 'medium', 'low', 'exploratory')
    frequency : str
        Pattern frequency ('weekly', 'monthly', 'daily', 'intraday')
    metadata : Dict[str, Any]
        Additional metadata from the screener
    """
    pattern_type: str
    symbol: str
    metric: str
    description: str
    trigger_conditions: Dict[str, Any]
    dates: pd.DatetimeIndex
    values: np.ndarray
    statistics: Dict[str, float]
    significance: str
    frequency: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        """Convert pattern to pandas Series with datetime index."""
        return pd.Series(self.values, index=self.dates, name=f"{self.symbol}_{self.pattern_type}")

    def __repr__(self) -> str:
        return (
            f"PatternResult({self.pattern_type}, {self.symbol}, {self.metric}, "
            f"freq={self.frequency}, sig={self.significance}, n={len(self.dates)})"
        )


class PatternExtractor:
    """
    Extracts significant patterns from screener results.

    This class processes results from both OrderflowScanner and HistoricalScreener
    to identify significant recurring patterns and generate time series for each pattern.

    Examples
    --------
    # Extract patterns from orderflow scan results
    >>> from CTAFlow.screeners import OrderflowScanner, PatternExtractor
    >>> scanner = OrderflowScanner(params)
    >>> results = scanner.scan(tick_data)
    >>> extractor = PatternExtractor()
    >>> patterns = extractor.extract_orderflow_patterns(results)
    >>> # Get all Monday patterns
    >>> monday_patterns = extractor.filter_patterns(weekday='Monday')

    # Extract from historical screener results
    >>> from CTAFlow.screeners import HistoricalScreener, PatternExtractor
    >>> screener = HistoricalScreener.from_parquet(['CL_F', 'NG_F'], timeframe='1T')
    >>> results = screener.run_screens([momentum_params, seasonality_params])
    >>> extractor = PatternExtractor()
    >>> patterns = extractor.extract_historical_patterns(results)

    # Filter and analyze patterns
    >>> high_sig = extractor.filter_patterns(significance='high')
    >>> weekly_patterns = extractor.filter_patterns(frequency='weekly')
    >>> cl_patterns = extractor.filter_patterns(symbol='CL_F')
    """

    def __init__(self, min_observations: int = 10, significance_threshold: float = 0.05):
        """
        Initialize PatternExtractor.

        Parameters
        ----------
        min_observations : int
            Minimum number of observations to consider a pattern valid (default: 10)
        significance_threshold : float
            P-value threshold for statistical significance (default: 0.05)
        """
        self.min_observations = min_observations
        self.significance_threshold = significance_threshold
        self.patterns: List[PatternResult] = []

    def extract_orderflow_patterns(
        self,
        orderflow_results: Dict[str, Dict[str, Any]],
        include_exploratory: bool = False
    ) -> List[PatternResult]:
        """
        Extract patterns from OrderflowScanner results.

        Parameters
        ----------
        orderflow_results : Dict[str, Dict[str, Any]]
            Results from OrderflowScanner.scan() or OrderflowScanner.run_scans()
            Expected structure: {symbol: {category: dataframe/data, ...}}
            Categories: 'df_events', 'df_weekly', 'df_wom_weekday', 'df_weekly_peak_pressure', etc.
        include_exploratory : bool
            Include exploratory (low sample size) patterns (default: False)

        Returns
        -------
        List[PatternResult]
            List of extracted patterns

        Notes
        -----
        Processes the following orderflow result types:
        - df_weekly: Day-of-week seasonality patterns
        - df_wom_weekday: Week-of-month + weekday patterns
        - df_weekly_peak_pressure: Peak pressure times by weekday
        - df_events: Event detection (runs of significant pressure)
        """
        patterns = []

        for symbol, symbol_data in orderflow_results.items():
            if isinstance(symbol_data, dict) and 'error' in symbol_data:
                print(f"Skipping {symbol}: {symbol_data['error']}")
                continue

            # Extract weekly patterns (day-of-week)
            if 'df_weekly' in symbol_data:
                df_weekly = symbol_data['df_weekly']
                if isinstance(df_weekly, pd.DataFrame) and not df_weekly.empty:
                    patterns.extend(self._extract_weekly_patterns(df_weekly, symbol, include_exploratory))

            # Extract week-of-month + weekday patterns
            if 'df_wom_weekday' in symbol_data:
                df_wom = symbol_data['df_wom_weekday']
                if isinstance(df_wom, pd.DataFrame) and not df_wom.empty:
                    patterns.extend(self._extract_wom_weekday_patterns(df_wom, symbol, include_exploratory))

            # Extract peak pressure patterns
            if 'df_weekly_peak_pressure' in symbol_data:
                df_peak = symbol_data['df_weekly_peak_pressure']
                if isinstance(df_peak, pd.DataFrame) and not df_peak.empty:
                    patterns.extend(self._extract_peak_pressure_patterns(df_peak, symbol, include_exploratory))

            # Extract event patterns
            if 'df_events' in symbol_data:
                df_events = symbol_data['df_events']
                if isinstance(df_events, pd.DataFrame) and not df_events.empty:
                    patterns.extend(self._extract_event_patterns(df_events, symbol))

        self.patterns.extend(patterns)
        return patterns

    def extract_historical_patterns(
        self,
        historical_results: Dict[str, Dict[str, Dict]],
        min_correlation: float = 0.3,
        include_exploratory: bool = False
    ) -> List[PatternResult]:
        """
        Extract patterns from HistoricalScreener results.

        Parameters
        ----------
        historical_results : Dict[str, Dict[str, Dict]]
            Results from HistoricalScreener.run_screens()
            Expected structure: {screen_name: {symbol: screen_results}}
        min_correlation : float
            Minimum correlation coefficient to consider a pattern significant (default: 0.3)
        include_exploratory : bool
            Include exploratory (low sample size) patterns (default: False)

        Returns
        -------
        List[PatternResult]
            List of extracted patterns

        Notes
        -----
        Processes the following historical screener result types:
        - Momentum screens: session momentum, opening/closing returns
        - Seasonality screens: time-of-day patterns, day-of-week patterns
        """
        patterns = []

        for screen_name, screen_data in historical_results.items():
            for symbol, symbol_results in screen_data.items():
                if isinstance(symbol_results, dict) and 'error' in symbol_results:
                    print(f"Skipping {symbol} in {screen_name}: {symbol_results['error']}")
                    continue

                # Determine screen type from screen_name or results
                if 'momentum' in screen_name.lower() or 'session' in str(symbol_results).lower():
                    patterns.extend(
                        self._extract_momentum_patterns(
                            symbol_results, symbol, screen_name, min_correlation, include_exploratory
                        )
                    )
                elif 'seasonality' in screen_name.lower() or 'tod' in screen_name.lower():
                    patterns.extend(
                        self._extract_seasonality_patterns(
                            symbol_results, symbol, screen_name, include_exploratory
                        )
                    )

        self.patterns.extend(patterns)
        return patterns

    def filter_patterns(
        self,
        pattern_type: Optional[str] = None,
        symbol: Optional[str] = None,
        metric: Optional[str] = None,
        significance: Optional[Literal['high', 'medium', 'low', 'exploratory']] = None,
        frequency: Optional[str] = None,
        weekday: Optional[str] = None,
        week_of_month: Optional[int] = None,
        min_observations: Optional[int] = None
    ) -> List[PatternResult]:
        """
        Filter patterns by various criteria.

        Parameters
        ----------
        pattern_type : Optional[str]
            Filter by pattern type (e.g., 'weekly', 'wom_weekday', 'event')
        symbol : Optional[str]
            Filter by symbol
        metric : Optional[str]
            Filter by metric name
        significance : Optional[str]
            Filter by significance level
        frequency : Optional[str]
            Filter by frequency
        weekday : Optional[str]
            Filter by weekday in trigger conditions
        week_of_month : Optional[int]
            Filter by week of month in trigger conditions
        min_observations : Optional[int]
            Minimum number of observations

        Returns
        -------
        List[PatternResult]
            Filtered patterns
        """
        filtered = self.patterns

        if pattern_type:
            filtered = [p for p in filtered if p.pattern_type == pattern_type]
        if symbol:
            filtered = [p for p in filtered if p.symbol == symbol]
        if metric:
            filtered = [p for p in filtered if p.metric == metric]
        if significance:
            filtered = [p for p in filtered if p.significance == significance]
        if frequency:
            filtered = [p for p in filtered if p.frequency == frequency]
        if weekday:
            filtered = [p for p in filtered if p.trigger_conditions.get('weekday') == weekday]
        if week_of_month:
            filtered = [p for p in filtered if p.trigger_conditions.get('week_of_month') == week_of_month]
        if min_observations:
            filtered = [p for p in filtered if len(p.dates) >= min_observations]

        return filtered

    def get_pattern_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all extracted patterns.

        Returns
        -------
        pd.DataFrame
            Summary with columns: symbol, pattern_type, metric, frequency, significance,
            n_observations, mean, std, t_stat, p_value, description
        """
        if not self.patterns:
            return pd.DataFrame()

        summary_data = []
        for pattern in self.patterns:
            summary_data.append({
                'symbol': pattern.symbol,
                'pattern_type': pattern.pattern_type,
                'metric': pattern.metric,
                'frequency': pattern.frequency,
                'significance': pattern.significance,
                'n_observations': len(pattern.dates),
                'mean': pattern.statistics.get('mean', np.nan),
                'std': pattern.statistics.get('std', np.nan),
                't_stat': pattern.statistics.get('t_stat', np.nan),
                'p_value': pattern.statistics.get('p_value', np.nan),
                'description': pattern.description,
                **pattern.trigger_conditions
            })

        return pd.DataFrame(summary_data)

    def _extract_weekly_patterns(
        self, df: pd.DataFrame, symbol: str, include_exploratory: bool
    ) -> List[PatternResult]:
        """Extract day-of-week patterns."""
        patterns = []

        for _, row in df.iterrows():
            # Skip exploratory if not included
            if row.get('exploratory', False) and not include_exploratory:
                continue

            # Skip non-significant patterns
            if 'sig_fdr_5pct' in row and not row['sig_fdr_5pct']:
                continue

            # Determine significance
            p_value = row.get('p_value', 1.0)
            sig = self._classify_significance(p_value, row.get('exploratory', False))

            # Build pattern
            pattern = PatternResult(
                pattern_type='weekly',
                symbol=symbol,
                metric=row.get('metric', 'unknown'),
                description=f"{row.get('weekday', 'N/A')} {row.get('metric', 'unknown')} pattern",
                trigger_conditions={'weekday': row.get('weekday', None)},
                dates=pd.DatetimeIndex([]),  # No specific dates in weekly summary
                values=np.array([row.get('mean', 0.0)]),
                statistics={
                    'mean': row.get('mean', 0.0),
                    't_stat': row.get('t_stat', 0.0),
                    'p_value': p_value,
                    'n': row.get('n', 0)
                },
                significance=sig,
                frequency='weekly',
                metadata={'exploratory': row.get('exploratory', False)}
            )
            patterns.append(pattern)

        return patterns

    def _extract_wom_weekday_patterns(
        self, df: pd.DataFrame, symbol: str, include_exploratory: bool
    ) -> List[PatternResult]:
        """Extract week-of-month + weekday patterns."""
        patterns = []

        for _, row in df.iterrows():
            # Skip exploratory if not included
            if row.get('exploratory', False) and not include_exploratory:
                continue

            # Skip non-significant patterns
            if 'sig_fdr_5pct' in row and not row['sig_fdr_5pct']:
                continue

            # Determine significance
            p_value = row.get('p_value', 1.0)
            sig = self._classify_significance(p_value, row.get('exploratory', False))

            # Build pattern
            pattern = PatternResult(
                pattern_type='wom_weekday',
                symbol=symbol,
                metric=row.get('metric', 'unknown'),
                description=(
                    f"Week {row.get('week_of_month', 'N/A')} {row.get('weekday', 'N/A')} "
                    f"{row.get('metric', 'unknown')} pattern"
                ),
                trigger_conditions={
                    'week_of_month': row.get('week_of_month', None),
                    'weekday': row.get('weekday', None)
                },
                dates=pd.DatetimeIndex([]),
                values=np.array([row.get('mean', 0.0)]),
                statistics={
                    'mean': row.get('mean', 0.0),
                    't_stat': row.get('t_stat', 0.0),
                    'p_value': p_value,
                    'n': row.get('n', 0)
                },
                significance=sig,
                frequency='monthly',
                metadata={'exploratory': row.get('exploratory', False)}
            )
            patterns.append(pattern)

        return patterns

    def _extract_peak_pressure_patterns(
        self, df: pd.DataFrame, symbol: str, include_exploratory: bool
    ) -> List[PatternResult]:
        """Extract peak pressure time patterns."""
        patterns = []

        for _, row in df.iterrows():
            # Skip exploratory if not included
            if row.get('exploratory', False) and not include_exploratory:
                continue

            # Skip non-significant patterns
            if 'seasonality_sig_fdr_5pct' in row and not row['seasonality_sig_fdr_5pct']:
                continue

            # Determine significance
            p_value = row.get('seasonality_q_value', 1.0)
            sig = self._classify_significance(p_value, row.get('exploratory', False))

            # Build pattern
            pattern = PatternResult(
                pattern_type='peak_pressure',
                symbol=symbol,
                metric=row.get('metric', 'unknown'),
                description=(
                    f"{row.get('weekday', 'N/A')} peak {row.get('pressure_bias', 'N/A')} pressure "
                    f"at {row.get('clock_time', 'N/A')}"
                ),
                trigger_conditions={
                    'weekday': row.get('weekday', None),
                    'clock_time': str(row.get('clock_time', None)),
                    'pressure_bias': row.get('pressure_bias', None)
                },
                dates=pd.DatetimeIndex([]),
                values=np.array([row.get('seasonality_mean', 0.0)]),
                statistics={
                    'seasonality_mean': row.get('seasonality_mean', 0.0),
                    'seasonality_t_stat': row.get('seasonality_t_stat', 0.0),
                    'seasonality_q_value': p_value,
                    'seasonality_n': row.get('seasonality_n', 0),
                    'intraday_mean': row.get('intraday_mean', 0.0),
                    'intraday_median': row.get('intraday_median', 0.0)
                },
                significance=sig,
                frequency='weekly',
                metadata={'exploratory': row.get('exploratory', False)}
            )
            patterns.append(pattern)

        return patterns

    def _extract_event_patterns(
        self, df: pd.DataFrame, symbol: str
    ) -> List[PatternResult]:
        """Extract significant event patterns."""
        patterns = []

        for _, row in df.iterrows():
            # Events are already filtered for significance by the scanner
            ts_start = pd.to_datetime(row.get('ts_start'))
            ts_end = pd.to_datetime(row.get('ts_end'))

            pattern = PatternResult(
                pattern_type='event',
                symbol=symbol,
                metric=row.get('metric', 'unknown'),
                description=(
                    f"{row.get('direction', 'N/A')} {row.get('metric', 'unknown')} event "
                    f"at {ts_start.strftime('%Y-%m-%d %H:%M')}"
                ),
                trigger_conditions={
                    'direction': row.get('direction', None),
                    'time_range': (ts_start, ts_end)
                },
                dates=pd.DatetimeIndex([ts_start]),
                values=np.array([row.get('max_abs_z', 0.0)]),
                statistics={
                    'max_abs_z': row.get('max_abs_z', 0.0),
                    'run_len': row.get('run_len', 0)
                },
                significance='high',  # Events are pre-filtered
                frequency='intraday',
                metadata={
                    'ts_start': ts_start,
                    'ts_end': ts_end,
                    'run_len': row.get('run_len', 0)
                }
            )
            patterns.append(pattern)

        return patterns

    def _extract_momentum_patterns(
        self,
        results: Dict,
        symbol: str,
        screen_name: str,
        min_correlation: float,
        include_exploratory: bool
    ) -> List[PatternResult]:
        """Extract momentum patterns from historical screener."""
        patterns = []

        # Look for momentum-related keys in results
        for key, value in results.items():
            if isinstance(value, dict):
                # Check for correlation data
                if 'correlation' in key.lower() or 'corr' in key.lower():
                    corr_value = value.get('correlation', value.get('corr', 0.0))
                    p_value = value.get('p_value', 1.0)

                    if abs(corr_value) >= min_correlation:
                        sig = self._classify_significance(p_value, False)
                        pattern = PatternResult(
                            pattern_type='momentum',
                            symbol=symbol,
                            metric='momentum_correlation',
                            description=f"{screen_name} {key} correlation pattern",
                            trigger_conditions={'session': key},
                            dates=pd.DatetimeIndex([]),
                            values=np.array([corr_value]),
                            statistics={
                                'correlation': corr_value,
                                'p_value': p_value,
                                'n': value.get('n', 0)
                            },
                            significance=sig,
                            frequency='daily',
                            metadata={'screen_name': screen_name}
                        )
                        patterns.append(pattern)

        return patterns

    def _extract_seasonality_patterns(
        self,
        results: Dict,
        symbol: str,
        screen_name: str,
        include_exploratory: bool
    ) -> List[PatternResult]:
        """Extract seasonality patterns from historical screener."""
        patterns = []

        # Look for seasonality-related keys
        for key, value in results.items():
            if isinstance(value, dict):
                # Check for time-of-day or day-of-week patterns
                if 'mean' in value and 't_stat' in value:
                    p_value = value.get('p_value', 1.0)

                    if p_value < self.significance_threshold or include_exploratory:
                        sig = self._classify_significance(p_value, value.get('exploratory', False))
                        pattern = PatternResult(
                            pattern_type='seasonality',
                            symbol=symbol,
                            metric=key,
                            description=f"{screen_name} {key} seasonal pattern",
                            trigger_conditions={'time_period': key},
                            dates=pd.DatetimeIndex([]),
                            values=np.array([value.get('mean', 0.0)]),
                            statistics={
                                'mean': value.get('mean', 0.0),
                                't_stat': value.get('t_stat', 0.0),
                                'p_value': p_value,
                                'n': value.get('n', 0)
                            },
                            significance=sig,
                            frequency='intraday',
                            metadata={'screen_name': screen_name}
                        )
                        patterns.append(pattern)

        return patterns

    def _classify_significance(self, p_value: float, exploratory: bool) -> str:
        """Classify significance level based on p-value."""
        if exploratory:
            return 'exploratory'
        elif p_value < 0.001:
            return 'high'
        elif p_value < 0.05:
            return 'medium'
        else:
            return 'low'


# Convenience functions
def extract_patterns_from_orderflow(
    orderflow_results: Dict[str, Dict[str, Any]],
    min_observations: int = 10,
    include_exploratory: bool = False
) -> List[PatternResult]:
    """
    Extract patterns from orderflow scan results (convenience function).

    Parameters
    ----------
    orderflow_results : Dict[str, Dict[str, Any]]
        Results from OrderflowScanner
    min_observations : int
        Minimum observations for valid pattern
    include_exploratory : bool
        Include exploratory patterns

    Returns
    -------
    List[PatternResult]
        Extracted patterns
    """
    extractor = PatternExtractor(min_observations=min_observations)
    return extractor.extract_orderflow_patterns(orderflow_results, include_exploratory)


def extract_patterns_from_historical(
    historical_results: Dict[str, Dict[str, Dict]],
    min_observations: int = 10,
    min_correlation: float = 0.3,
    include_exploratory: bool = False
) -> List[PatternResult]:
    """
    Extract patterns from historical screener results (convenience function).

    Parameters
    ----------
    historical_results : Dict[str, Dict[str, Dict]]
        Results from HistoricalScreener
    min_observations : int
        Minimum observations for valid pattern
    min_correlation : float
        Minimum correlation for momentum patterns
    include_exploratory : bool
        Include exploratory patterns

    Returns
    -------
    List[PatternResult]
        Extracted patterns
    """
    extractor = PatternExtractor(min_observations=min_observations)
    return extractor.extract_historical_patterns(historical_results, min_correlation, include_exploratory)
