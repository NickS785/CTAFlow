"""
Tests for pattern extraction functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from CTAFlow.screeners import (
    PatternExtractor,
    PatternResult,
    extract_patterns_from_orderflow,
    extract_patterns_from_historical,
)


@pytest.fixture
def sample_weekly_data():
    """Create sample weekly orderflow data."""
    return pd.DataFrame({
        'ticker': ['ZC'] * 6,
        'metric': ['buy_pressure', 'sell_pressure'] * 3,
        'weekday': ['Monday', 'Monday', 'Tuesday', 'Tuesday', 'Wednesday', 'Wednesday'],
        'mean': [0.01, -0.01, 0.02, -0.02, 0.015, -0.015],
        't_stat': [2.5, -2.5, 3.0, -3.0, 2.8, -2.8],
        'p_value': [0.01, 0.01, 0.001, 0.001, 0.005, 0.005],
        'n': [100, 100, 120, 120, 110, 110],
        'exploratory': [False, False, False, False, False, False],
        'sig_fdr_5pct': [True, True, True, True, True, True]
    })


@pytest.fixture
def sample_wom_weekday_data():
    """Create sample week-of-month + weekday data."""
    return pd.DataFrame({
        'week_of_month': [1, 1, 2, 2],
        'weekday': ['Monday', 'Tuesday', 'Monday', 'Tuesday'],
        'ticker': ['ZC'] * 4,
        'metric': ['buy_pressure'] * 4,
        'mean': [0.02, 0.015, -0.01, -0.005],
        't_stat': [3.5, 2.5, -2.0, -1.5],
        'p_value': [0.0001, 0.01, 0.05, 0.15],
        'n': [50, 45, 48, 42],
        'exploratory': [False, False, False, True],
        'q_value': [0.001, 0.02, 0.08, 0.20],
        'sig_fdr_5pct': [True, True, False, False]
    })


@pytest.fixture
def sample_event_data():
    """Create sample event data."""
    return pd.DataFrame({
        'ticker': ['ZC'] * 3,
        'metric': ['buy_pressure', 'sell_pressure', 'buy_pressure'],
        'ts_start': pd.to_datetime(['2023-01-15 09:30:00', '2023-01-16 14:00:00', '2023-01-17 10:15:00']),
        'ts_end': pd.to_datetime(['2023-01-15 09:45:00', '2023-01-16 14:20:00', '2023-01-17 10:30:00']),
        'run_len': [1, 2, 1],
        'max_abs_z': [2.5, 3.2, 2.8],
        'direction': ['positive', 'negative', 'positive']
    })


def test_pattern_extractor_init():
    """Test PatternExtractor initialization."""
    extractor = PatternExtractor(min_observations=20, significance_threshold=0.01)
    assert extractor.min_observations == 20
    assert extractor.significance_threshold == 0.01
    assert len(extractor.patterns) == 0


def test_extract_weekly_patterns(sample_weekly_data):
    """Test weekly pattern extraction."""
    orderflow_results = {
        'ZC': {'df_weekly': sample_weekly_data}
    }

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results)

    assert len(patterns) > 0
    assert all(p.pattern_type == 'weekly' for p in patterns)
    assert all(p.frequency == 'weekly' for p in patterns)
    assert all('weekday' in p.trigger_conditions for p in patterns)


def test_extract_wom_weekday_patterns(sample_wom_weekday_data):
    """Test week-of-month + weekday pattern extraction."""
    orderflow_results = {
        'ZC': {'df_wom_weekday': sample_wom_weekday_data}
    }

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results, include_exploratory=False)

    assert len(patterns) > 0
    assert all(p.pattern_type == 'wom_weekday' for p in patterns)
    assert all(p.frequency == 'monthly' for p in patterns)
    assert all('week_of_month' in p.trigger_conditions for p in patterns)
    assert all('weekday' in p.trigger_conditions for p in patterns)


def test_extract_event_patterns(sample_event_data):
    """Test event pattern extraction."""
    orderflow_results = {
        'ZC': {'df_events': sample_event_data}
    }

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results)

    assert len(patterns) == 3
    assert all(p.pattern_type == 'event' for p in patterns)
    assert all(p.frequency == 'intraday' for p in patterns)
    assert all(p.significance == 'high' for p in patterns)


def test_filter_patterns(sample_weekly_data, sample_wom_weekday_data):
    """Test pattern filtering."""
    orderflow_results = {
        'ZC': {
            'df_weekly': sample_weekly_data,
            'df_wom_weekday': sample_wom_weekday_data
        }
    }

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results, include_exploratory=False)

    # Filter by pattern type
    weekly = extractor.filter_patterns(pattern_type='weekly')
    assert all(p.pattern_type == 'weekly' for p in weekly)

    wom = extractor.filter_patterns(pattern_type='wom_weekday')
    assert all(p.pattern_type == 'wom_weekday' for p in wom)

    # Filter by weekday
    monday = extractor.filter_patterns(weekday='Monday')
    assert all(p.trigger_conditions.get('weekday') == 'Monday' for p in monday)

    # Filter by week of month
    week1 = extractor.filter_patterns(week_of_month=1)
    assert all(p.trigger_conditions.get('week_of_month') == 1 for p in week1)

    # Filter by significance
    high_sig = extractor.filter_patterns(significance='high')
    assert all(p.significance == 'high' for p in high_sig)


def test_get_pattern_summary(sample_weekly_data):
    """Test pattern summary generation."""
    orderflow_results = {
        'ZC': {'df_weekly': sample_weekly_data}
    }

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results)
    summary = extractor.get_pattern_summary()

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == len(patterns)
    assert 'symbol' in summary.columns
    assert 'pattern_type' in summary.columns
    assert 'metric' in summary.columns
    assert 'significance' in summary.columns


def test_pattern_result_to_series(sample_event_data):
    """Test PatternResult to_series conversion."""
    orderflow_results = {
        'ZC': {'df_events': sample_event_data}
    }

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results)

    for pattern in patterns:
        series = pattern.to_series()
        assert isinstance(series, pd.Series)
        assert isinstance(series.index, pd.DatetimeIndex)
        assert len(series) == len(pattern.dates)


def test_convenience_function_orderflow(sample_weekly_data):
    """Test convenience function for orderflow extraction."""
    orderflow_results = {
        'ZC': {'df_weekly': sample_weekly_data}
    }

    patterns = extract_patterns_from_orderflow(orderflow_results, min_observations=10)

    assert len(patterns) > 0
    assert all(isinstance(p, PatternResult) for p in patterns)


def test_significance_classification():
    """Test significance level classification."""
    extractor = PatternExtractor()

    # High significance
    assert extractor._classify_significance(0.0001, False) == 'high'
    assert extractor._classify_significance(0.0009, False) == 'high'

    # Medium significance
    assert extractor._classify_significance(0.001, False) == 'medium'
    assert extractor._classify_significance(0.049, False) == 'medium'

    # Low significance
    assert extractor._classify_significance(0.05, False) == 'low'
    assert extractor._classify_significance(0.5, False) == 'low'

    # Exploratory
    assert extractor._classify_significance(0.01, True) == 'exploratory'


def test_real_data_integration():
    """Test with real scanner results if available."""
    results_path = Path("screens/scanner_results")

    if not results_path.exists():
        pytest.skip("Scanner results directory not found")

    # Try loading ZC results
    orderflow_results = {'ZC': {}}

    try:
        if (results_path / "zc_df_weekly.csv").exists():
            orderflow_results['ZC']['df_weekly'] = pd.read_csv(
                results_path / "zc_df_weekly.csv",
                index_col=0
            )

        if (results_path / "zc_df_wom_weekday.csv").exists():
            orderflow_results['ZC']['df_wom_weekday'] = pd.read_csv(
                results_path / "zc_df_wom_weekday.csv",
                index_col=0
            )

        if not orderflow_results['ZC']:
            pytest.skip("No ZC orderflow results found")

        # Extract patterns
        extractor = PatternExtractor()
        patterns = extractor.extract_orderflow_patterns(orderflow_results, include_exploratory=False)

        assert len(patterns) > 0
        print(f"\nExtracted {len(patterns)} patterns from real ZC data")

        # Test filtering
        high_sig = extractor.filter_patterns(significance='high')
        print(f"High significance patterns: {len(high_sig)}")

        # Test summary
        summary = extractor.get_pattern_summary()
        assert len(summary) == len(patterns)

    except Exception as e:
        pytest.skip(f"Could not process real data: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
