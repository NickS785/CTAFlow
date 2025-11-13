from __future__ import annotations

from datetime import time
from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if 'CTAFlow' not in sys.modules:
    package = types.ModuleType('CTAFlow')
    package.__path__ = [str(ROOT / 'CTAFlow')]
    sys.modules['CTAFlow'] = package

if 'CTAFlow.screeners' not in sys.modules:
    screeners_pkg = types.ModuleType('CTAFlow.screeners')
    screeners_pkg.__path__ = [str(ROOT / 'CTAFlow' / 'screeners')]
    sys.modules['CTAFlow.screeners'] = screeners_pkg

if 'CTAFlow.utils' not in sys.modules:
    utils_spec = importlib.util.spec_from_file_location(
        'CTAFlow.utils',
        ROOT / 'CTAFlow' / 'utils' / '__init__.py',
        submodule_search_locations=[str(ROOT / 'CTAFlow' / 'utils')],
    )
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules['CTAFlow.utils'] = utils_module
    assert utils_spec.loader is not None
    utils_spec.loader.exec_module(utils_module)

if 'CTAFlow.data' not in sys.modules:
    data_stub = types.ModuleType('CTAFlow.data')
    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

    data_stub.IntradayFileManager = _Stub
    data_stub.DataClient = _Stub
    data_stub.SyntheticSymbol = _Stub
    data_stub.ResultsClient = _Stub
    sys.modules['CTAFlow.data'] = data_stub

if 'CTAFlow.config' not in sys.modules:
    config_stub = types.ModuleType('CTAFlow.config')
    config_stub.DLY_DATA_PATH = ''
    config_stub.INTRADAY_ADB_PATH = ''
    sys.modules['CTAFlow.config'] = config_stub

MODULE_PATH = ROOT / 'CTAFlow' / 'screeners' / 'historical_screener.py'
SPEC = importlib.util.spec_from_file_location(
    'CTAFlow.screeners.historical_screener', MODULE_PATH
)
historical_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None  # for mypy/static checkers
SPEC.loader.exec_module(historical_module)
HistoricalScreener = historical_module.HistoricalScreener


class MinimalSeasonalityScreener(HistoricalScreener):
    """Slimmed screener that returns deterministic diagnostics for testing."""

    def _analyze_dayofweek_patterns(self, *args, **kwargs):  # pragma: no cover - overridden
        return {'returns': {}, 'volatility': {}}

    def _test_time_predictability(  # pragma: no cover - deterministic override
        self,
        data,
        price_col,
        target_time: time,
        is_synthetic,
        period_length=None,
        tz: str | None = None,
    ):
        return {
            'n': 60,
            'mean_return': 0.0,
            'std_return': 1.0,
            'next_day_corr': 0.25,
            'next_day_pvalue': 0.01,
            'next_day_significant': True,
            'next_week_corr': 0.15,
            'next_week_pvalue': 0.02,
            'next_week_significant': True,
            'weekday_prevalence': {
                'most_prevalent_day': 'Monday',
                'strongest_days': ['Monday'],
            },
            # Return a multi-entry list to ensure the production code narrows it
            'target_times_hhmm': ['03:00', '07:30'],
            'period_length_min': 60,
            'months_active': [9],
            'months_mask_12': '000000000010',
            'months_names': ['Sep'],
        }

    def _compute_weekend_hedging_pattern(self, *args, **kwargs):  # pragma: no cover - not needed
        return None

    def _analyze_by_month(self, *args, **kwargs):  # pragma: no cover - skipped
        return {}


def test_time_pattern_metadata_keeps_single_target_time():
    idx = pd.date_range('2023-09-01 02:30', periods=80, freq='5min', tz='UTC')
    prices = np.linspace(50.0, 51.0, len(idx))
    df = pd.DataFrame({'Close': prices}, index=idx)

    screener = MinimalSeasonalityScreener({'CS': df}, verbose=False)

    results = screener.st_seasonality_screen(
        target_times=['03:00', '07:30'],
        period_length=60,
        dayofweek_screen=False,
        session_start='02:30',
        session_end='09:30',
        tz='UTC',
        show_progress=False,
        max_workers=1,
    )

    ticker_results = results['CS']

    # pattern_context should not inherit bulk target time lists
    assert 'target_times_hhmm' not in ticker_results['pattern_context']

    time_patterns = ticker_results['time_predictability']
    assert set(time_patterns.keys()) == {'03:00:00', '07:30:00'}

    for stats in time_patterns.values():
        assert stats['target_times_hhmm'] == [stats['time']]

    pattern_times = {}
    for pattern in ticker_results['strongest_patterns']:
        if pattern['type'].startswith('time_predictive'):
            pattern_times.setdefault(pattern['time'], []).append(pattern)
            # Each pattern key should reference the active time only
            assert pattern['target_times_hhmm'] == [pattern['time']]

            if pattern['type'] == 'time_predictive_nextweek':
                # Regression guard: no NameError, description includes the time label
                assert pattern['time'] in pattern['description']
                assert 'strongest on Monday' in pattern['description']

    # We expect entries for each configured gate
    assert set(pattern_times.keys()) == {'03:00', '07:30'}
