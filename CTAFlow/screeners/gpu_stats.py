"""GPU-accelerated statistical operations for screening.

This module provides GPU-batched versions of common statistical operations
used in historical screening, enabling simultaneous computation across all tickers.
"""

from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np  # type: ignore
    GPU_AVAILABLE = False

__all__ = [
    'GPU_AVAILABLE',
    'batch_pearson_correlation',
    'batch_t_test',
    'batch_mean_std',
]


def batch_pearson_correlation(
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pearson correlations for multiple array pairs simultaneously.

    This function batches correlation calculations across all tickers, providing
    significant speedup compared to scipy.stats.pearsonr in a loop.

    Args:
        x_arrays: List of x arrays (one per ticker), all must have same length
        y_arrays: List of y arrays (one per ticker), all must have same length
        use_gpu: Use GPU acceleration if available (default: True)

    Returns:
        correlations: Array of correlation coefficients (length: n_tickers)
        p_values: Array of p-values (length: n_tickers)

    Example:
        >>> # Instead of:
        >>> for ticker in tickers:
        ...     corr, p = scipy.stats.pearsonr(x[ticker], y[ticker])
        >>>
        >>> # Do:
        >>> x_list = [x[ticker] for ticker in tickers]
        >>> y_list = [y[ticker] for ticker in tickers]
        >>> corrs, p_vals = batch_pearson_correlation(x_list, y_list)

    Notes:
        - All arrays must have the same length (aligned data required)
        - GPU provides 10-100x speedup depending on number of tickers
        - Gracefully falls back to CPU if GPU unavailable
    """
    if not x_arrays or not y_arrays:
        return np.array([]), np.array([])

    if len(x_arrays) != len(y_arrays):
        raise ValueError(f"Mismatched lengths: {len(x_arrays)} x arrays, {len(y_arrays)} y arrays")

    # Check all arrays have same length
    n_samples = len(x_arrays[0])
    if not all(len(x) == n_samples for x in x_arrays):
        raise ValueError("All x arrays must have same length")
    if not all(len(y) == n_samples for y in y_arrays):
        raise ValueError("All y arrays must have same length")

    xp = cp if (use_gpu and GPU_AVAILABLE) else np

    # Stack arrays: Shape (n_tickers, n_samples)
    x_stack = xp.array(x_arrays, dtype=xp.float64)
    y_stack = xp.array(y_arrays, dtype=xp.float64)

    # Compute means
    x_mean = x_stack.mean(axis=1, keepdims=True)
    y_mean = y_stack.mean(axis=1, keepdims=True)

    # Center the data
    x_centered = x_stack - x_mean
    y_centered = y_stack - y_mean

    # Compute correlation coefficients
    numerator = (x_centered * y_centered).sum(axis=1)
    x_ss = (x_centered ** 2).sum(axis=1)
    y_ss = (y_centered ** 2).sum(axis=1)
    denominator = xp.sqrt(x_ss * y_ss)

    # Handle division by zero
    correlations = xp.where(denominator != 0, numerator / denominator, 0.0)

    # Compute p-values using t-statistic
    # t = r * sqrt((n-2) / (1 - r^2))
    r_squared = correlations ** 2
    # Avoid division by zero when |r| = 1
    denominator_t = xp.where(r_squared < 1.0, 1.0 - r_squared, 1e-10)
    t_stat = correlations * xp.sqrt((n_samples - 2) / denominator_t)

    # Transfer t_stat to CPU for scipy if needed
    if use_gpu and GPU_AVAILABLE:
        t_stat_cpu = cp.asnumpy(t_stat)
    else:
        t_stat_cpu = t_stat

    # Compute p-values using scipy.stats
    from scipy import stats
    p_values = 2 * stats.t.sf(np.abs(t_stat_cpu), n_samples - 2)

    # Transfer results back to CPU if on GPU
    if use_gpu and GPU_AVAILABLE:
        correlations = cp.asnumpy(correlations)

    return correlations, p_values


def batch_t_test(
    arrays: List[np.ndarray],
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute t-statistics and p-values for multiple arrays simultaneously.

    Tests whether each array's mean is significantly different from zero.

    Args:
        arrays: List of arrays (one per ticker), all must have same length
        use_gpu: Use GPU acceleration if available (default: True)

    Returns:
        means: Array of means (length: n_tickers)
        t_stats: Array of t-statistics (length: n_tickers)
        p_values: Array of p-values (length: n_tickers)
        counts: Array of sample counts (length: n_tickers)

    Example:
        >>> # Instead of:
        >>> for ticker in tickers:
        ...     mean = data[ticker].mean()
        ...     t_stat = mean / (data[ticker].std() / sqrt(len(data[ticker])))
        >>>
        >>> # Do:
        >>> arrays = [data[ticker] for ticker in tickers]
        >>> means, t_stats, p_vals, counts = batch_t_test(arrays)
    """
    if not arrays:
        return np.array([]), np.array([]), np.array([]), np.array([])

    xp = cp if (use_gpu and GPU_AVAILABLE) else np

    # Stack arrays
    n_samples = len(arrays[0])
    arrays_stack = xp.array(arrays, dtype=xp.float64)

    # Compute statistics
    means = arrays_stack.mean(axis=1)
    stds = arrays_stack.std(axis=1, ddof=1)
    counts = xp.full(len(arrays), n_samples, dtype=xp.int64)

    # Compute t-statistics
    # t = mean / (std / sqrt(n))
    se = stds / xp.sqrt(n_samples)
    t_stats = xp.where(se != 0, means / se, 0.0)

    # Transfer to CPU for scipy
    if use_gpu and GPU_AVAILABLE:
        t_stats_cpu = cp.asnumpy(t_stats)
    else:
        t_stats_cpu = t_stats

    # Compute p-values
    from scipy import stats
    p_values = 2 * stats.t.sf(np.abs(t_stats_cpu), n_samples - 1)

    # Transfer results to CPU
    if use_gpu and GPU_AVAILABLE:
        means = cp.asnumpy(means)
        t_stats = cp.asnumpy(t_stats)
        counts = cp.asnumpy(counts)

    return means, t_stats, p_values, counts


def batch_mean_std(
    arrays: List[np.ndarray],
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute means, standard deviations, and counts for multiple arrays.

    Args:
        arrays: List of arrays (one per ticker), all must have same length
        use_gpu: Use GPU acceleration if available (default: True)

    Returns:
        means: Array of means (length: n_tickers)
        stds: Array of standard deviations (length: n_tickers)
        counts: Array of sample counts (length: n_tickers)

    Example:
        >>> arrays = [data[ticker] for ticker in tickers]
        >>> means, stds, counts = batch_mean_std(arrays)
    """
    if not arrays:
        return np.array([]), np.array([]), np.array([])

    xp = cp if (use_gpu and GPU_AVAILABLE) else np

    # Stack arrays
    n_samples = len(arrays[0])
    arrays_stack = xp.array(arrays, dtype=xp.float64)

    # Compute statistics
    means = arrays_stack.mean(axis=1)
    stds = arrays_stack.std(axis=1, ddof=1)
    counts = xp.full(len(arrays), n_samples, dtype=xp.int64)

    # Transfer to CPU
    if use_gpu and GPU_AVAILABLE:
        means = cp.asnumpy(means)
        stds = cp.asnumpy(stds)
        counts = cp.asnumpy(counts)

    return means, stds, counts


def batch_autocorrelation(
    arrays: List[np.ndarray],
    lag: int = 1,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelations for multiple arrays simultaneously.

    Args:
        arrays: List of arrays (one per ticker), all must have same length
        lag: Lag for autocorrelation (default: 1)
        use_gpu: Use GPU acceleration if available (default: True)

    Returns:
        correlations: Array of autocorrelation coefficients (length: n_tickers)
        p_values: Array of p-values (length: n_tickers)

    Example:
        >>> arrays = [returns[ticker] for ticker in tickers]
        >>> autocorrs, p_vals = batch_autocorrelation(arrays, lag=1)
    """
    if not arrays:
        return np.array([]), np.array([])

    # Create lagged versions
    x_arrays = [arr[:-lag] if lag > 0 else arr for arr in arrays]
    y_arrays = [arr[lag:] if lag > 0 else arr for arr in arrays]

    # Use batch pearson correlation
    return batch_pearson_correlation(x_arrays, y_arrays, use_gpu=use_gpu)


def validate_array_lengths(arrays: List[np.ndarray], name: str = "arrays") -> None:
    """Validate that all arrays have the same length.

    Args:
        arrays: List of arrays to validate
        name: Name for error messages

    Raises:
        ValueError: If arrays have different lengths
    """
    if not arrays:
        return

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All {name} must have same length. Got lengths: {lengths}. "
            f"Hint: Use align_ticker_data() to align tickers to common date range."
        )
