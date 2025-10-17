"""Utility functions for model/math operations."""

from .seasonal import deseasonalize_monthly, zscore_normalize, SeasonalAnalysis, IntradayPairSpec, abnormal_months, prewindow_feature, prewindow_predicts_month, last_year_predicts_this_year
from .pca_analysis import PCAAnalyzer
from .tenor_interpolation import TenorInterpolator, create_tenor_grid
from .vol_weighted_returns import vol_weighted_returns, log_returns
from .unit_conversions import (
    gallons_to_barrels,
    barrels_to_gallons,
    bushels_to_kilograms,
    bushels_to_metric_tons,
)
from .session import filter_session_bars, filter_session_ticks
from .volume_bucket import auto_bucket_size, ticks_to_volume_buckets
from .computation_utils import (
    classify_regime_percentile,
    calculate_cot_indices_batch,
    calculate_obv_vectorized,
    calculate_rolling_sum_vectorized,
    calculate_realized_variance_vectorized,
    calculate_realized_semivariance_vectorized,
    calculate_cumulative_delta_vectorized,
    cache_volatility_calculation,
    batch_percentile_calculation,
)

__all__ = [
    'TenorInterpolator',
    'create_tenor_grid',
    'PCAAnalyzer',
    "deseasonalize_monthly",
    "zscore_normalize",
    "SeasonalAnalysis",
    "gallons_to_barrels",
    "barrels_to_gallons",
    "bushels_to_kilograms",
    "bushels_to_metric_tons",
    "abnormal_months",
    "last_year_predicts_this_year",
    "log_returns",
    "prewindow_feature",
    "prewindow_predicts_month",
    "filter_session_bars",
    "filter_session_ticks",
    "auto_bucket_size",
    "ticks_to_volume_buckets",
    "vol_weighted_returns",
    # Centralized computation utilities
    "classify_regime_percentile",
    "calculate_cot_indices_batch",
    "calculate_obv_vectorized",
    "calculate_rolling_sum_vectorized",
    "calculate_realized_variance_vectorized",
    "calculate_realized_semivariance_vectorized",
    "calculate_cumulative_delta_vectorized",
    "cache_volatility_calculation",
    "batch_percentile_calculation",
]
