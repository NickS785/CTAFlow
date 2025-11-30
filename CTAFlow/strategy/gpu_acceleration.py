"""GPU acceleration utilities for backtesting operations.

This module provides GPU-accelerated versions of compute-intensive operations
using CuPy when available. Falls back gracefully to NumPy when GPU is not available.

Usage:
    from CTAFlow.strategy.gpu_acceleration import GPU_AVAILABLE, gpu_backtest_returns

    if GPU_AVAILABLE:
        print("GPU acceleration enabled")

    positions, pnl = gpu_backtest_returns(returns_x, returns_y, threshold=0.0)
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Dict, Optional, Tuple, Union

import logging

import numpy as np

try:  # Pandas is optional in some lightweight environments
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - defensive import for optional dependency
    pd = None

# Try to import CuPy for GPU acceleration
_logger = logging.getLogger(__name__)


def _notify_gpu_fallback(reason: str) -> None:
    """Log a single-line warning when GPU acceleration is unavailable."""

    _logger.warning("GPU acceleration unavailable (%s); falling back to CPU", reason)


try:
    import cupy as cp
    # Verify GPU is actually available
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        GPU_AVAILABLE = True
        GPU_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        # CUDA runtime not available
        cp = np  # type: ignore
        GPU_AVAILABLE = False
        GPU_DEVICE_COUNT = 0
        _notify_gpu_fallback(str(exc))
except ImportError:
    # CuPy not installed
    cp = np  # type: ignore
    GPU_AVAILABLE = False
    GPU_DEVICE_COUNT = 0
    _notify_gpu_fallback("CuPy not installed")

__all__ = [
    'GPU_AVAILABLE',
    'GPU_DEVICE_COUNT',
    'get_array_module',
    'to_gpu',
    'to_cpu',
    'to_backend_array',
    'gpu_backtest_returns',
    'gpu_backtest_threshold',
    'gpu_batch_threshold_sweep',
    'gpu_cumulative_pnl',
    'get_gpu_info',
]


_stream_local = threading.local()


def _resolve_stream(stream: Optional[object]) -> object:
    """Return a context-manager-compatible CUDA stream, defaulting to thread-local.

    Using a unique stream per thread prevents contention on the default stream when
    multiple CPU workers submit GPU work concurrently. Falls back to a no-op context
    manager when GPU acceleration is unavailable.
    """

    if not GPU_AVAILABLE or cp is np:
        return nullcontext()

    if stream is not None:
        return stream

    cached = getattr(_stream_local, "stream", None)
    if cached is None:
        cached = cp.cuda.Stream(non_blocking=True)
        _stream_local.stream = cached
    return cached


def _cupy_cummax_1d(arr, xp):
    """Compute cumulative maximum for 1D CuPy arrays.

    CuPy doesn't support maximum.accumulate, so we implement it manually.

    Args:
        arr: 1D array
        xp: array module (cp for CuPy)

    Returns:
        Cumulative maximum array of same shape
    """
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got {arr.ndim}D")

    result = xp.empty_like(arr)
    result[0] = arr[0]

    for i in range(1, len(arr)):
        result[i] = xp.maximum(result[i-1], arr[i])

    return result


def _cupy_cummax(arr, xp):
    """Compute cumulative maximum along axis 1 for CuPy arrays.

    CuPy doesn't support maximum.accumulate, so we implement it manually.
    This function is optimized for 2D arrays with axis=1.

    Args:
        arr: 2D array (n_thresholds, n_samples)
        xp: array module (cp for CuPy)

    Returns:
        Cumulative maximum array of same shape
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")

    # Use vectorized approach: for each position, take max of all previous values
    # More efficient than looping
    result = xp.empty_like(arr)
    result[:, 0] = arr[:, 0]

    for i in range(1, arr.shape[1]):
        result[:, i] = xp.maximum(result[:, i-1], arr[:, i])

    return result


def get_gpu_info() -> dict:
    """Return information about GPU availability and capabilities."""
    if not GPU_AVAILABLE:
        return {
            'available': False,
            'device_count': 0,
            'backend': 'numpy',
            'message': 'CuPy not installed or CUDA not available'
        }

    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        return {
            'available': True,
            'device_count': GPU_DEVICE_COUNT,
            'backend': 'cupy',
            'device_name': props['name'].decode('utf-8'),
            'compute_capability': f"{props['major']}.{props['minor']}",
            'total_memory_gb': props['totalGlobalMem'] / (1024**3),
            'multiprocessor_count': props['multiProcessorCount'],
        }
    except Exception as e:
        return {
            'available': False,
            'device_count': 0,
            'backend': 'numpy',
            'error': str(e)
        }


def get_array_module(values: Union[np.ndarray, 'cp.ndarray']):
    """Return the array module (NumPy or CuPy) backing ``values``."""

    if GPU_AVAILABLE and hasattr(cp, "get_array_module"):
        return cp.get_array_module(values)
    return np


def to_gpu(arr: np.ndarray, device_id: int = 0) -> Union[np.ndarray, 'cp.ndarray']:
    """Transfer NumPy array to GPU.

    Args:
        arr: NumPy array to transfer
        device_id: GPU device ID (default: 0)

    Returns:
        CuPy array if GPU available, otherwise returns input array unchanged
    """
    if not GPU_AVAILABLE:
        return arr

    with cp.cuda.Device(device_id):
        return cp.asarray(arr)


def to_cpu(arr: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
    """Transfer array from GPU to CPU.

    Args:
        arr: Array to transfer (CuPy or NumPy)

    Returns:
        NumPy array
    """
    if not GPU_AVAILABLE:
        return arr

    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def to_backend_array(
    values: Union[np.ndarray, 'cp.ndarray', "pd.Series", "pd.DataFrame", list, tuple],
    *,
    use_gpu: bool,
    device_id: int,
    stream: Optional[object] = None,
) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np, 'cp']]:
    """Return an array backed by either NumPy or CuPy along with the module used.

    Accepts pandas Series/DataFrames in addition to generic array likes so callers
    can pass DataFrame columns directly when accelerating.
    """

    base = values
    if pd is not None and isinstance(values, (pd.Series, pd.DataFrame)):
        base = values.to_numpy(copy=False)
    elif not isinstance(values, (np.ndarray,)):
        base = np.asarray(values)

    if not use_gpu or not GPU_AVAILABLE:
        return base, np

    with cp.cuda.Device(device_id):
        stream_cm = _resolve_stream(stream)
        with stream_cm:
            return cp.asarray(base), cp


def gpu_backtest_returns(
    returns_x: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    returns_y: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    threshold: float = 0.0,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
    return_backend: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate positions and PnL using GPU acceleration.

    Accepts NumPy arrays or pandas objects. When ``use_gpu`` is True and a GPU is
    available, computations are performed with CuPy on the selected device.
    """

    rx, xp = to_backend_array(
        returns_x, use_gpu=use_gpu, device_id=device_id, stream=stream
    )
    ry, _ = to_backend_array(returns_y, use_gpu=use_gpu, device_id=device_id, stream=stream)

    positions_backend = xp.where(
        rx >= threshold, 1.0,
        xp.where(rx <= -threshold, -1.0, 0.0)
    )
    pnl_backend = positions_backend * ry

    if return_backend:
        return positions_backend, pnl_backend, xp

    if xp is np:
        return positions_backend, pnl_backend

    return cp.asnumpy(positions_backend), cp.asnumpy(pnl_backend)


def gpu_backtest_threshold(
    returns_x: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    returns_y: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    correlation: Optional[Union[np.ndarray, "pd.Series", "pd.DataFrame"]] = None,
    threshold: float = 0.0,
    use_side_hint: bool = True,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
    return_backend: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate positions and PnL with correlation-based side hint.

    Supports pandas inputs so downstream callers can pass DataFrame columns
    directly. Uses CuPy for all calculations when a GPU is available.
    """

    rx, xp = to_backend_array(
        returns_x, use_gpu=use_gpu, device_id=device_id, stream=stream
    )
    ry, _ = to_backend_array(returns_y, use_gpu=use_gpu, device_id=device_id, stream=stream)

    if correlation is not None and use_side_hint:
        corr, _ = to_backend_array(
            correlation, use_gpu=use_gpu, device_id=device_id, stream=stream
        )
        adjusted_x = rx * xp.sign(corr)
    else:
        adjusted_x = rx

    positions_backend = xp.where(
        adjusted_x >= threshold, 1.0,
        xp.where(adjusted_x <= -threshold, -1.0, 0.0)
    )
    pnl_backend = positions_backend * ry

    if return_backend:
        return positions_backend, pnl_backend, xp

    if xp is np:
        return positions_backend, pnl_backend

    return cp.asnumpy(positions_backend), cp.asnumpy(pnl_backend)


def gpu_batch_threshold_sweep(
    returns_x: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    returns_y: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    thresholds: Union[np.ndarray, list, tuple],
    correlation: Optional[Union[np.ndarray, "pd.Series", "pd.DataFrame"]] = None,
    use_side_hint: bool = True,
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Batch backtest multiple threshold values with a single GPU transfer.

    Performs threshold backtesting for multiple threshold values simultaneously,
    keeping all data on the GPU and minimizing CPU↔GPU transfers. This is
    significantly faster than calling gpu_backtest_threshold() in a loop.

    Args:
        returns_x: Predictor returns
        returns_y: Target returns
        thresholds: Array of threshold values to test (e.g., [0.0, 0.5, 1.0, 1.5, 2.0])
        correlation: Optional correlation values for side hints
        use_side_hint: Whether to use correlation sign for position direction
        use_gpu: Whether to use GPU acceleration
        device_id: GPU device ID
        stream: Optional CUDA stream for async operations

    Returns:
        Dictionary mapping each threshold to a dict containing:
            - 'positions': Position array for this threshold
            - 'pnl': PnL array for this threshold
            - 'cumulative': Cumulative PnL array
            - 'drawdown': Drawdown array
            - 'max_drawdown': Maximum drawdown value

    Example:
        >>> thresholds = [0.0, 0.5, 1.0, 1.5]
        >>> results = gpu_batch_threshold_sweep(returns_x, returns_y, thresholds)
        >>> for threshold, metrics in results.items():
        ...     print(f"Threshold {threshold}: Max DD = {metrics['max_drawdown']:.4f}")
    """

    # Convert thresholds to array
    thresholds_array = np.asarray(thresholds, dtype=float).ravel()
    if len(thresholds_array) == 0:
        return {}

    # CPU fallback for non-GPU execution
    if not use_gpu or not GPU_AVAILABLE:
        results = {}
        for threshold in thresholds_array:
            pos, pnl = gpu_backtest_threshold(
                returns_x=returns_x,
                returns_y=returns_y,
                correlation=correlation,
                threshold=float(threshold),
                use_side_hint=use_side_hint,
                use_gpu=False,
                device_id=device_id,
                stream=stream,
            )
            cumulative = np.cumsum(pnl)
            rolling_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - rolling_max
            max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

            results[float(threshold)] = {
                'positions': pos,
                'pnl': pnl,
                'cumulative': cumulative,
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
            }
        return results

    # GPU batch processing
    with cp.cuda.Device(device_id):
        stream_cm = _resolve_stream(stream)

        with stream_cm:
            # Transfer data to GPU once
            rx, xp = to_backend_array(
                returns_x, use_gpu=True, device_id=device_id, stream=stream_cm
            )
            ry, _ = to_backend_array(
                returns_y, use_gpu=True, device_id=device_id, stream=stream_cm
            )

            # Apply correlation adjustment if needed
            if correlation is not None and use_side_hint:
                corr, _ = to_backend_array(
                    correlation, use_gpu=True, device_id=device_id, stream=stream_cm
                )
                adjusted_x = rx * xp.sign(corr)
            else:
                adjusted_x = rx

            # Transfer thresholds to GPU and reshape for broadcasting
            # Shape: (n_thresholds, 1) for broadcasting against (n_samples,)
            thresholds_gpu = xp.asarray(thresholds_array).reshape(-1, 1)

            # Broadcast comparison for all thresholds simultaneously
            # Result shape: (n_thresholds, n_samples)
            adjusted_x_broadcast = adjusted_x.reshape(1, -1)  # (1, n_samples)

            # Vectorized position calculation for all thresholds
            positions_all = xp.where(
                adjusted_x_broadcast >= thresholds_gpu, 1.0,
                xp.where(adjusted_x_broadcast <= -thresholds_gpu, -1.0, 0.0)
            )

            # Calculate PnL for all thresholds
            pnl_all = positions_all * ry.reshape(1, -1)

            # Calculate cumulative PnL and drawdown for all thresholds
            cumulative_all = xp.cumsum(pnl_all, axis=1)
            # CuPy doesn't support maximum.accumulate, use manual implementation
            rolling_max_all = xp.maximum.accumulate(cumulative_all, axis=1) if not use_gpu else _cupy_cummax(cumulative_all, xp)
            drawdown_all = cumulative_all - rolling_max_all

            # Vectorized drawdown minimums to reduce transfers
            drawdown_mins = drawdown_all.min(axis=1)

            # Transfer batched arrays back to CPU only once
            positions_cpu_all = xp.asnumpy(positions_all)
            pnl_cpu_all = xp.asnumpy(pnl_all)
            cumulative_cpu_all = xp.asnumpy(cumulative_all)
            drawdown_cpu_all = xp.asnumpy(drawdown_all)
            drawdown_mins_cpu = xp.asnumpy(drawdown_mins)

            # Package results from shared CPU buffers
            results = {}
            for i, threshold in enumerate(thresholds_array):
                results[float(threshold)] = {
                    'positions': positions_cpu_all[i, :],
                    'pnl': pnl_cpu_all[i, :],
                    'cumulative': cumulative_cpu_all[i, :],
                    'drawdown': drawdown_cpu_all[i, :],
                    'max_drawdown': float(drawdown_mins_cpu[i]) if len(drawdown_cpu_all[i, :]) > 0 else 0.0,
                }

    return results


def gpu_cumulative_pnl(
    pnl: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
    return_backend: bool = False,
) -> np.ndarray:
    """Calculate cumulative PnL using GPU acceleration."""

    pnl_backend, xp = to_backend_array(
        pnl, use_gpu=use_gpu, device_id=device_id, stream=stream
    )
    cumulative_backend = xp.cumsum(pnl_backend)

    if return_backend:
        return cumulative_backend, xp

    if xp is np:
        return cumulative_backend

    return cp.asnumpy(cumulative_backend)


# Module-level diagnostics
def _print_gpu_status():
    """Print GPU status when module is imported (for debugging)."""
    info = get_gpu_info()
    if info['available']:
        print(f"GPU Acceleration: ENABLED ({info['device_name']})")
    else:
        print(f"GPU Acceleration: DISABLED ({info.get('message', 'Unknown')})")


# Uncomment for debug output on import
# _print_gpu_status()
