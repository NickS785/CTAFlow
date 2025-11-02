"""
Standalone tests for pattern extraction (without pytest).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from CTAFlow.screeners import (
    PatternExtractor,
    PatternResult,
    extract_patterns_from_orderflow,
)


def test_pattern_extractor_init():
    """Test PatternExtractor initialization."""
    print("\n[TEST] PatternExtractor initialization...")
    extractor = PatternExtractor(min_observations=20, significance_threshold=0.01)
    assert extractor.min_observations == 20
    assert extractor.significance_threshold == 0.01
    assert len(extractor.patterns) == 0
    print("  PASSED")


def test_extract_weekly_patterns():
    """Test weekly pattern extraction."""
    print("\n[TEST] Weekly pattern extraction...")

    sample_data = pd.DataFrame({
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

    orderflow_results = {'ZC': {'df_weekly': sample_data}}

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results)

    assert len(patterns) > 0, "No patterns extracted"
    assert all(p.pattern_type == 'weekly' for p in patterns), "Wrong pattern type"
    assert all(p.frequency == 'weekly' for p in patterns), "Wrong frequency"
    print(f"  PASSED - Extracted {len(patterns)} weekly patterns")


def test_filter_patterns():
    """Test pattern filtering."""
    print("\n[TEST] Pattern filtering...")

    sample_data = pd.DataFrame({
        'ticker': ['ZC'] * 4,
        'metric': ['buy_pressure'] * 4,
        'weekday': ['Monday', 'Tuesday', 'Monday', 'Tuesday'],
        'mean': [0.01, 0.02, 0.015, 0.025],
        't_stat': [2.5, 3.0, 2.8, 3.5],
        'p_value': [0.01, 0.001, 0.005, 0.0001],
        'n': [100, 120, 110, 130],
        'exploratory': [False, False, False, False],
        'sig_fdr_5pct': [True, True, True, True]
    })

    orderflow_results = {'ZC': {'df_weekly': sample_data}}

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results)

    # Filter by weekday
    monday = extractor.filter_patterns(weekday='Monday')
    assert len(monday) == 2, f"Expected 2 Monday patterns, got {len(monday)}"
    assert all(p.trigger_conditions.get('weekday') == 'Monday' for p in monday)

    # Filter by significance
    high_sig = extractor.filter_patterns(significance='high')
    assert len(high_sig) >= 1, "Expected at least 1 high significance pattern"

    print(f"  PASSED - Filters working correctly")


def test_get_pattern_summary():
    """Test pattern summary generation."""
    print("\n[TEST] Pattern summary generation...")

    sample_data = pd.DataFrame({
        'ticker': ['ZC'] * 2,
        'metric': ['buy_pressure', 'sell_pressure'],
        'weekday': ['Monday', 'Monday'],
        'mean': [0.01, -0.01],
        't_stat': [2.5, -2.5],
        'p_value': [0.01, 0.01],
        'n': [100, 100],
        'exploratory': [False, False],
        'sig_fdr_5pct': [True, True]
    })

    orderflow_results = {'ZC': {'df_weekly': sample_data}}

    extractor = PatternExtractor()
    patterns = extractor.extract_orderflow_patterns(orderflow_results)
    summary = extractor.get_pattern_summary()

    assert isinstance(summary, pd.DataFrame), "Summary is not a DataFrame"
    assert len(summary) == len(patterns), "Summary length doesn't match patterns"
    assert 'symbol' in summary.columns
    assert 'pattern_type' in summary.columns
    assert 'metric' in summary.columns

    print(f"  PASSED - Summary DataFrame generated with {len(summary)} rows")


def test_real_data_integration():
    """Test with real scanner results if available."""
    print("\n[TEST] Real data integration...")

    results_path = Path("screens/scanner_results")

    if not results_path.exists():
        print("  SKIPPED - Scanner results directory not found")
        return

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
            print("  SKIPPED - No ZC orderflow results found")
            return

        # Extract patterns
        extractor = PatternExtractor()
        patterns = extractor.extract_orderflow_patterns(orderflow_results, include_exploratory=False)

        assert len(patterns) > 0, "No patterns extracted from real data"
        print(f"  PASSED - Extracted {len(patterns)} patterns from real ZC data")

        # Test filtering
        high_sig = extractor.filter_patterns(significance='high')
        print(f"    - High significance patterns: {len(high_sig)}")

        # Show some patterns
        print(f"    - Sample patterns:")
        for p in patterns[:3]:
            print(f"      * {p.description} (sig={p.significance}, p={p.statistics.get('p_value', 0):.4f})")

        # Test summary
        summary = extractor.get_pattern_summary()
        assert len(summary) == len(patterns), "Summary length mismatch"

    except Exception as e:
        print(f"  FAILED - {e}")
        raise


def test_convenience_function():
    """Test convenience function."""
    print("\n[TEST] Convenience function...")

    sample_data = pd.DataFrame({
        'ticker': ['ZC'] * 2,
        'metric': ['buy_pressure', 'sell_pressure'],
        'weekday': ['Monday', 'Monday'],
        'mean': [0.01, -0.01],
        't_stat': [2.5, -2.5],
        'p_value': [0.01, 0.01],
        'n': [100, 100],
        'exploratory': [False, False],
        'sig_fdr_5pct': [True, True]
    })

    orderflow_results = {'ZC': {'df_weekly': sample_data}}
    patterns = extract_patterns_from_orderflow(orderflow_results, min_observations=10)

    assert len(patterns) > 0, "No patterns from convenience function"
    assert all(isinstance(p, PatternResult) for p in patterns)

    print(f"  PASSED - Convenience function working")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PATTERN EXTRACTION TESTS")
    print("="*80)

    tests = [
        test_pattern_extractor_init,
        test_extract_weekly_patterns,
        test_filter_patterns,
        test_get_pattern_summary,
        test_convenience_function,
        test_real_data_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED - {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
