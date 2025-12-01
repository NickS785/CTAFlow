"""
GPU-accelerated computation utilities for screening operations.

This module provides GPU-accelerated versions of statistical computations
commonly used in screening operations, using CuPy when available.

Usage:
    from CTAFlow.utils.gpu_utils import gpu_batch_correlation, gpu_batch_ttest

    # Batch correlation calculation
    correlations, p_values = gpu_batch_correlation(returns_x, returns_y_batch, use_gpu=True)

    # Batch t-test
    t_stats, p_values = gpu_batch_ttest(sample_batch, popmean=0, use_gpu=True)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union, Dict, Any
from contextlib import nullcontext

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Import GPU acceleration infrastructure
try:
    from ..strategy.gpu_acceleration import (
        GPU_AVAILABLE,
        GPU_DEVICE_COUNT,
        to_gpu,
        to_cpu,
        to_backend_array,
        get_array_module,
        _cupy_cummax_1d,
    )
    if GPU_AVAILABLE:
        import cupy as cp
    else:
        cp = np  # type: ignore
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE_COUNT = 0
    cp = np  # type: ignore

    def to_gpu(arr, device_id=0):
        return arr

    def to_cpu(arr):
        return np.asarray(arr)

    def to_backend_array(arr, use_gpu=False, device_id=0, stream=None):
        return np.asarray(arr), np

    def get_array_module(arr):
        return np


__all__ = [
    'gpu_batch_correlation',
    'gpu_batch_ttest',
    'gpu_batch_mean_std',
    'gpu_batch_quantile',
    'gpu_rolling_window',
    'gpu_batch_zscore',
    'gpu_batch_pearsonr',
    'gpu_batch_spearmanr',
]


def gpu_batch_pearsonr(
    x: Union[np.ndarray, 'pd.Series'],
    y_batch: Union[np.ndarray, 'pd.DataFrame'],
    *,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
    return_pvalues: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Calculate Pearson correlation coefficients in batch.

    Computes correlation between a single array x and multiple arrays in y_batch
    simultaneously using GPU acceleration when available.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Reference array
    y_batch : array-like, shape (n_features, n_samples) or (n_samples, n_features)
        Batch of arrays to correlate with x. If 2D, correlates each row/column with x.
    use_gpu : bool, default=True
        Use GPU acceleration if available
    device_id : int, default=0
        GPU device ID
    stream : object, optional
        CUDA stream for async operations
    return_pvalues : bool, default=True
        If True, returns (correlations, p_values). If False, returns only correlations.

    Returns
    -------
    correlations : ndarray
        Correlation coefficients, shape (n_features,)
    p_values : ndarray, optional
        Two-tailed p-values, shape (n_features,). Only returned if return_pvalues=True.

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y_batch = np.random.randn(50, 100)  # 50 series, 100 samples each
    >>> corrs, pvals = gpu_batch_pearsonr(x, y_batch)
    >>> print(f"Correlations shape: {corrs.shape}")  # (50,)
    """
    # Convert inputs to numpy
    x_arr = np.asarray(x).ravel()

    if hasattr(y_batch, 'values'):  # DataFrame
        y_arr = np.asarray(y_batch.values)
    else:
        y_arr = np.asarray(y_batch)

    # Ensure y_batch is 2D with shape (n_features, n_samples)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    elif y_arr.ndim == 2:
        # If shape is (n_samples, n_features), transpose
        if y_arr.shape[0] == len(x_arr) and y_arr.shape[1] != len(x_arr):
            y_arr = y_arr.T
    else:
        raise ValueError(f"y_batch must be 1D or 2D, got {y_arr.ndim}D")

    n_features, n_samples = y_arr.shape

    if n_samples != len(x_arr):
        raise ValueError(f"Sample size mismatch: x has {len(x_arr)} samples, y_batch has {n_samples}")

    if n_samples < 3:
        warnings.warn("Sample size too small for reliable correlation (n<3)")
        if return_pvalues:
            return np.full(n_features, np.nan), np.full(n_features, np.nan)
        return np.full(n_features, np.nan)

    # CPU fallback or GPU processing
    if not use_gpu or not GPU_AVAILABLE:
        return _cpu_batch_pearsonr(x_arr, y_arr, return_pvalues=return_pvalues)

    # GPU processing
    try:
        with cp.cuda.Device(device_id):
            stream_cm = stream if stream is not None else nullcontext()

            with stream_cm:
                # Transfer to GPU
                x_gpu, xp = to_backend_array(x_arr, use_gpu=True, device_id=device_id, stream=stream)
                y_gpu, _ = to_backend_array(y_arr, use_gpu=True, device_id=device_id, stream=stream)

                # Compute correlations on GPU
                correlations_gpu = _gpu_pearsonr_kernel(x_gpu, y_gpu, xp)

                # Compute p-values if requested
                if return_pvalues:
                    pvalues_gpu = _gpu_pearsonr_pvalues(correlations_gpu, n_samples, xp)

                    # Transfer back to CPU
                    correlations = to_cpu(correlations_gpu)
                    pvalues = to_cpu(pvalues_gpu)

                    return correlations, pvalues
                else:
                    correlations = to_cpu(correlations_gpu)
                    return correlations

    except Exception as e:
        warnings.warn(f"GPU correlation failed ({e}), falling back to CPU")
        return _cpu_batch_pearsonr(x_arr, y_arr, return_pvalues=return_pvalues)


def _gpu_pearsonr_kernel(x_gpu, y_batch_gpu, xp):
    """GPU kernel for Pearson correlation calculation."""
    # Center the data
    x_centered = x_gpu - xp.mean(x_gpu)
    y_centered = y_batch_gpu - xp.mean(y_batch_gpu, axis=1, keepdims=True)

    # Compute standard deviations
    x_std = xp.sqrt(xp.sum(x_centered ** 2))
    y_std = xp.sqrt(xp.sum(y_centered ** 2, axis=1))

    # Compute correlation
    numerator = xp.sum(x_centered * y_centered, axis=1)
    denominator = x_std * y_std

    # Avoid division by zero
    correlations = xp.where(denominator != 0, numerator / denominator, 0.0)

    # Clip to valid range [-1, 1]
    correlations = xp.clip(correlations, -1.0, 1.0)

    return correlations


def _gpu_pearsonr_pvalues(correlations, n_samples, xp):
    """Compute p-values for Pearson correlations on GPU."""
    # Use Fisher transformation for p-value calculation
    # t = r * sqrt((n-2) / (1 - r^2))
    # df = n - 2

    r_squared = correlations ** 2
    df = n_samples - 2

    # Avoid division by zero
    denominator = 1.0 - r_squared
    denominator = xp.maximum(denominator, 1e-10)

    t_stat = correlations * xp.sqrt(df / denominator)

    # Approximate p-value using t-distribution
    # For large samples, use normal approximation
    # For exact, would need scipy.stats on GPU (not available)
    # Using conservative two-tailed test approximation

    if n_samples > 30:
        # Normal approximation for large samples
        pvalues = 2.0 * (1.0 - _gpu_norm_cdf(xp.abs(t_stat), xp))
    else:
        # Conservative estimate for small samples
        # Transfer to CPU for exact calculation
        t_stat_cpu = to_cpu(t_stat)
        from scipy import stats
        pvalues_cpu = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat_cpu), df))
        pvalues = to_backend_array(pvalues_cpu, use_gpu=True)[0]

    return pvalues


def _gpu_norm_cdf(x, xp):
    """Approximate normal CDF on GPU using error function."""
    # CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
    # CuPy has erf function
    if hasattr(xp, 'erf'):
        return 0.5 * (1.0 + xp.erf(x / xp.sqrt(2.0)))
    else:
        # Fallback approximation
        return 0.5 * (1.0 + xp.tanh(x * 0.7978845608))


def _cpu_batch_pearsonr(x_arr, y_arr, return_pvalues=True):
    """CPU fallback for batch Pearson correlation."""
    from scipy import stats

    n_features = y_arr.shape[0]
    correlations = np.zeros(n_features)
    pvalues = np.zeros(n_features) if return_pvalues else None

    for i in range(n_features):
        try:
            r, p = stats.pearsonr(x_arr, y_arr[i])
            correlations[i] = r
            if return_pvalues:
                pvalues[i] = p
        except Exception:
            correlations[i] = np.nan
            if return_pvalues:
                pvalues[i] = np.nan

    if return_pvalues:
        return correlations, pvalues
    return correlations


def gpu_batch_correlation(
    x: Union[np.ndarray, 'pd.Series'],
    y_batch: Union[np.ndarray, 'pd.DataFrame'],
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for gpu_batch_pearsonr for backward compatibility."""
    return gpu_batch_pearsonr(x, y_batch, **kwargs)


def gpu_batch_ttest(
    sample_batch: Union[np.ndarray, 'pd.DataFrame'],
    *,
    popmean: float = 0.0,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform one-sample t-tests in batch on GPU.

    Parameters
    ----------
    sample_batch : array-like, shape (n_tests, n_samples)
        Batch of samples to test. Each row is tested against popmean.
    popmean : float, default=0.0
        Expected population mean
    use_gpu : bool, default=True
        Use GPU acceleration if available
    device_id : int, default=0
        GPU device ID
    stream : object, optional
        CUDA stream for async operations

    Returns
    -------
    t_statistics : ndarray, shape (n_tests,)
        T-statistics for each test
    p_values : ndarray, shape (n_tests,)
        Two-tailed p-values for each test
    """
    # Convert to numpy
    if hasattr(sample_batch, 'values'):
        batch_arr = np.asarray(sample_batch.values)
    else:
        batch_arr = np.asarray(sample_batch)

    if batch_arr.ndim == 1:
        batch_arr = batch_arr.reshape(1, -1)

    n_tests, n_samples = batch_arr.shape

    if n_samples < 2:
        warnings.warn("Sample size too small for t-test (n<2)")
        return np.full(n_tests, np.nan), np.full(n_tests, np.nan)

    # CPU fallback or GPU processing
    if not use_gpu or not GPU_AVAILABLE:
        return _cpu_batch_ttest(batch_arr, popmean=popmean)

    try:
        with cp.cuda.Device(device_id):
            stream_cm = stream if stream is not None else nullcontext()

            with stream_cm:
                # Transfer to GPU
                batch_gpu, xp = to_backend_array(batch_arr, use_gpu=True, device_id=device_id, stream=stream)

                # Compute statistics on GPU
                means = xp.mean(batch_gpu, axis=1)
                stds = xp.std(batch_gpu, axis=1, ddof=1)

                # T-statistic: (mean - popmean) / (std / sqrt(n))
                t_stats = (means - popmean) / (stds / xp.sqrt(n_samples))

                # Compute p-values
                df = n_samples - 1

                if n_samples > 30:
                    # Normal approximation for large samples
                    p_vals = 2.0 * (1.0 - _gpu_norm_cdf(xp.abs(t_stats), xp))
                else:
                    # Transfer to CPU for exact t-distribution
                    t_stats_cpu = to_cpu(t_stats)
                    from scipy import stats
                    p_vals_cpu = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats_cpu), df))
                    p_vals = to_backend_array(p_vals_cpu, use_gpu=True)[0]

                # Transfer results back to CPU
                t_statistics = to_cpu(t_stats)
                p_values = to_cpu(p_vals)

                return t_statistics, p_values

    except Exception as e:
        warnings.warn(f"GPU t-test failed ({e}), falling back to CPU")
        return _cpu_batch_ttest(batch_arr, popmean=popmean)


def _cpu_batch_ttest(batch_arr, popmean=0.0):
    """CPU fallback for batch t-test."""
    from scipy import stats

    n_tests = batch_arr.shape[0]
    t_statistics = np.zeros(n_tests)
    p_values = np.zeros(n_tests)

    for i in range(n_tests):
        try:
            t, p = stats.ttest_1samp(batch_arr[i], popmean=popmean)
            t_statistics[i] = t
            p_values[i] = p
        except Exception:
            t_statistics[i] = np.nan
            p_values[i] = np.nan

    return t_statistics, p_values


def gpu_batch_mean_std(
    data_batch: Union[np.ndarray, 'pd.DataFrame'],
    *,
    axis: int = 1,
    ddof: int = 1,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
    nan_policy: str = "omit",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation in batch on GPU.

    Parameters
    ----------
    data_batch : array-like, shape (n_series, n_samples)
        Batch of data series
    axis : int, default=1
        Axis along which to compute statistics (1 for row-wise)
    ddof : int, default=1
        Delta degrees of freedom for std calculation
    use_gpu : bool, default=True
        Use GPU acceleration if available
    device_id : int, default=0
        GPU device ID
    stream : object, optional
        CUDA stream

    Parameters
    ----------
    nan_policy : {'omit', 'propagate'}
        How to handle NaN values. When 'omit', NaNs are ignored similar to pandas
        aggregations. When 'propagate', any NaN in a series results in NaN outputs.

    Returns
    -------
    means : ndarray
        Mean values
    stds : ndarray
        Standard deviations
    """
    if hasattr(data_batch, 'values'):
        batch_arr = np.asarray(data_batch.values)
    else:
        batch_arr = np.asarray(data_batch)

    nan_safe_mean = np.nanmean if nan_policy == "omit" else np.mean
    nan_safe_std = np.nanstd if nan_policy == "omit" else np.std

    if not use_gpu or not GPU_AVAILABLE:
        means = nan_safe_mean(batch_arr, axis=axis)
        stds = nan_safe_std(batch_arr, axis=axis, ddof=ddof)
        return means, stds

    try:
        with cp.cuda.Device(device_id):
            stream_cm = stream if stream is not None else nullcontext()

            with stream_cm:
                batch_gpu, xp = to_backend_array(batch_arr, use_gpu=True, device_id=device_id, stream=stream)

                mean_func = xp.nanmean if nan_policy == "omit" else xp.mean
                std_func = xp.nanstd if nan_policy == "omit" else xp.std

                means_gpu = mean_func(batch_gpu, axis=axis)
                stds_gpu = std_func(batch_gpu, axis=axis, ddof=ddof)

                means = to_cpu(means_gpu)
                stds = to_cpu(stds_gpu)

                return means, stds

    except Exception as e:
        warnings.warn(f"GPU mean/std failed ({e}), falling back to CPU")
        means = nan_safe_mean(batch_arr, axis=axis)
        stds = nan_safe_std(batch_arr, axis=axis, ddof=ddof)
        return means, stds


def gpu_batch_quantile(
    data_batch: Union[np.ndarray, 'pd.DataFrame'],
    quantiles: Union[float, np.ndarray] = 0.5,
    *,
    axis: int = 1,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
) -> np.ndarray:
    """Compute quantiles in batch on GPU.

    Parameters
    ----------
    data_batch : array-like, shape (n_series, n_samples)
        Batch of data series
    quantiles : float or array-like
        Quantile(s) to compute (0.0 to 1.0)
    axis : int, default=1
        Axis along which to compute quantiles
    use_gpu : bool, default=True
        Use GPU acceleration if available
    device_id : int, default=0
        GPU device ID
    stream : object, optional
        CUDA stream

    Returns
    -------
    quantile_values : ndarray
        Computed quantiles
    """
    if hasattr(data_batch, 'values'):
        batch_arr = np.asarray(data_batch.values)
    else:
        batch_arr = np.asarray(data_batch)

    if not use_gpu or not GPU_AVAILABLE:
        return np.quantile(batch_arr, quantiles, axis=axis)

    try:
        with cp.cuda.Device(device_id):
            stream_cm = stream if stream is not None else nullcontext()

            with stream_cm:
                batch_gpu, xp = to_backend_array(batch_arr, use_gpu=True, device_id=device_id, stream=stream)

                if hasattr(xp, 'quantile'):
                    quantile_gpu = xp.quantile(batch_gpu, quantiles, axis=axis)
                else:
                    # Fallback: use percentile
                    percentiles = np.asarray(quantiles) * 100
                    quantile_gpu = xp.percentile(batch_gpu, percentiles, axis=axis)

                return to_cpu(quantile_gpu)

    except Exception as e:
        warnings.warn(f"GPU quantile failed ({e}), falling back to CPU")
        return np.quantile(batch_arr, quantiles, axis=axis)


def gpu_batch_zscore(
    data_batch: Union[np.ndarray, 'pd.DataFrame'],
    *,
    axis: int = 1,
    ddof: int = 1,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
) -> np.ndarray:
    """Compute z-scores in batch on GPU.

    Parameters
    ----------
    data_batch : array-like, shape (n_series, n_samples)
        Batch of data series
    axis : int, default=1
        Axis along which to compute z-scores
    ddof : int, default=1
        Delta degrees of freedom for std calculation
    use_gpu : bool, default=True
        Use GPU acceleration if available
    device_id : int, default=0
        GPU device ID
    stream : object, optional
        CUDA stream

    Returns
    -------
    zscores : ndarray
        Z-score transformed data
    """
    if hasattr(data_batch, 'values'):
        batch_arr = np.asarray(data_batch.values)
    else:
        batch_arr = np.asarray(data_batch)

    if not use_gpu or not GPU_AVAILABLE:
        means = np.mean(batch_arr, axis=axis, keepdims=True)
        stds = np.std(batch_arr, axis=axis, ddof=ddof, keepdims=True)
        return (batch_arr - means) / np.maximum(stds, 1e-10)

    try:
        with cp.cuda.Device(device_id):
            stream_cm = stream if stream is not None else nullcontext()

            with stream_cm:
                batch_gpu, xp = to_backend_array(batch_arr, use_gpu=True, device_id=device_id, stream=stream)

                means_gpu = xp.mean(batch_gpu, axis=axis, keepdims=True)
                stds_gpu = xp.std(batch_gpu, axis=axis, ddof=ddof, keepdims=True)

                # Avoid division by zero
                stds_gpu = xp.maximum(stds_gpu, 1e-10)

                zscores_gpu = (batch_gpu - means_gpu) / stds_gpu

                return to_cpu(zscores_gpu)

    except Exception as e:
        warnings.warn(f"GPU z-score failed ({e}), falling back to CPU")
        means = np.mean(batch_arr, axis=axis, keepdims=True)
        stds = np.std(batch_arr, axis=axis, ddof=ddof, keepdims=True)
        return (batch_arr - means) / np.maximum(stds, 1e-10)


def gpu_rolling_window(
    data: Union[np.ndarray, 'pd.Series'],
    window: int,
    operation: str = 'mean',
    *,
    min_periods: Optional[int] = None,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
) -> np.ndarray:
    """Compute rolling window statistics on GPU.

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        Time series data
    window : int
        Rolling window size
    operation : str, default='mean'
        Operation to perform: 'mean', 'std', 'min', 'max', 'sum'
    min_periods : int, optional
        Minimum number of observations required
    use_gpu : bool, default=True
        Use GPU acceleration if available
    device_id : int, default=0
        GPU device ID
    stream : object, optional
        CUDA stream

    Returns
    -------
    result : ndarray
        Rolling window results
    """
    if hasattr(data, 'values'):
        data_arr = np.asarray(data.values).ravel()
    else:
        data_arr = np.asarray(data).ravel()

    if min_periods is None:
        min_periods = window

    # CPU path using pandas for performance
    if not use_gpu or not GPU_AVAILABLE or pd is None:
        if pd is None:
            # Simple numpy fallback
            from numpy.lib.stride_tricks import sliding_window_view

            if len(data_arr) < window:
                return np.full_like(data_arr, np.nan, dtype=float)

            windows = sliding_window_view(data_arr, window)
            pad = np.full(window - 1, np.nan)
            if operation == 'mean':
                return np.concatenate([pad, np.nanmean(windows, axis=1)])
            elif operation == 'std':
                return np.concatenate([pad, np.nanstd(windows, axis=1)])
            elif operation == 'min':
                return np.concatenate([pad, np.nanmin(windows, axis=1)])
            elif operation == 'max':
                return np.concatenate([pad, np.nanmax(windows, axis=1)])
            elif operation == 'sum':
                return np.concatenate([pad, np.nansum(windows, axis=1)])
            raise ValueError(f"Unknown operation: {operation}")

        s = pd.Series(data_arr)
        if operation == 'mean':
            return s.rolling(window, min_periods=min_periods).mean().values
        elif operation == 'std':
            return s.rolling(window, min_periods=min_periods).std().values
        elif operation == 'min':
            return s.rolling(window, min_periods=min_periods).min().values
        elif operation == 'max':
            return s.rolling(window, min_periods=min_periods).max().values
        elif operation == 'sum':
            return s.rolling(window, min_periods=min_periods).sum().values
        raise ValueError(f"Unknown operation: {operation}")

    # GPU implementation
    try:
        with cp.cuda.Device(device_id):
            stream_cm = stream if stream is not None else nullcontext()
            with stream_cm:
                data_gpu, xp = to_backend_array(data_arr, use_gpu=True, device_id=device_id, stream=stream)

                if data_gpu.size < window:
                    return np.full_like(data_arr, np.nan, dtype=float)

                windows = xp.lib.stride_tricks.sliding_window_view(data_gpu, window)
                counts = xp.sum(~xp.isnan(windows), axis=1)

                if operation == 'mean':
                    core = xp.nanmean(windows, axis=1)
                elif operation == 'std':
                    core = xp.nanstd(windows, axis=1)
                elif operation == 'min':
                    core = xp.nanmin(windows, axis=1)
                elif operation == 'max':
                    core = xp.nanmax(windows, axis=1)
                elif operation == 'sum':
                    core = xp.nansum(windows, axis=1)
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                pad = xp.full(window - 1, xp.nan)
                result = xp.concatenate([pad, core])

                if min_periods > window:
                    min_periods = window
                invalid = counts < min_periods
                if invalid.any():
                    result = result.copy()
                    result[window - 1:][invalid] = xp.nan

                return to_cpu(result)
    except Exception as e:
        warnings.warn(f"GPU rolling operations failed ({e}), falling back to CPU")
        return gpu_rolling_window(
            data_arr,
            window,
            operation=operation,
            min_periods=min_periods,
            use_gpu=False,
        )


def gpu_batch_spearmanr(
    x: Union[np.ndarray, 'pd.Series'],
    y_batch: Union[np.ndarray, 'pd.DataFrame'],
    *,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
    return_pvalues: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Calculate Spearman rank correlation coefficients in batch.

    Note: Currently uses CPU fallback as GPU rank computation is complex.
    Future versions may implement GPU-accelerated ranking.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Reference array
    y_batch : array-like, shape (n_features, n_samples)
        Batch of arrays to correlate with x
    use_gpu : bool, default=True
        Use GPU acceleration if available (currently uses CPU)
    device_id : int, default=0
        GPU device ID
    stream : object, optional
        CUDA stream
    return_pvalues : bool, default=True
        If True, returns (correlations, p_values)

    Returns
    -------
    correlations : ndarray
        Spearman correlation coefficients
    p_values : ndarray, optional
        Two-tailed p-values
    """
    # Spearman requires ranking, which is complex on GPU
    # Use CPU implementation
    from scipy import stats

    x_arr = np.asarray(x).ravel()

    if hasattr(y_batch, 'values'):
        y_arr = np.asarray(y_batch.values)
    else:
        y_arr = np.asarray(y_batch)

    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    elif y_arr.shape[0] == len(x_arr) and y_arr.shape[1] != len(x_arr):
        y_arr = y_arr.T

    n_features = y_arr.shape[0]
    correlations = np.zeros(n_features)
    pvalues = np.zeros(n_features) if return_pvalues else None

    for i in range(n_features):
        try:
            r, p = stats.spearmanr(x_arr, y_arr[i])
            correlations[i] = r
            if return_pvalues:
                pvalues[i] = p
        except Exception:
            correlations[i] = np.nan
            if return_pvalues:
                pvalues[i] = np.nan

    if return_pvalues:
        return correlations, pvalues
    return correlations
