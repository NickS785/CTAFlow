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

from typing import Optional, Tuple, Union

import numpy as np

try:  # Pandas is optional in some lightweight environments
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - defensive import for optional dependency
    pd = None

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    # Verify GPU is actually available
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        GPU_AVAILABLE = True
        GPU_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        # CUDA runtime not available
        cp = np  # type: ignore
        GPU_AVAILABLE = False
        GPU_DEVICE_COUNT = 0
except ImportError:
    # CuPy not installed
    cp = np  # type: ignoreFGix
    GPU_AVAILABLE = False
    GPU_DEVICE_COUNT = 0

__all__ = [
    'GPU_AVAILABLE',
    'GPU_DEVICE_COUNT',
    'to_gpu',
    'to_cpu',
    'gpu_backtest_returns',
    'gpu_backtest_threshold',
    'gpu_cumulative_pnl',
    'get_gpu_info',
]


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


def _to_backend_array(
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
        if stream is not None:
            with stream:
                return cp.asarray(base), cp
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

    rx, xp = _to_backend_array(
        returns_x, use_gpu=use_gpu, device_id=device_id, stream=stream
    )
    ry, _ = _to_backend_array(returns_y, use_gpu=use_gpu, device_id=device_id, stream=stream)

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

    rx, xp = _to_backend_array(
        returns_x, use_gpu=use_gpu, device_id=device_id, stream=stream
    )
    ry, _ = _to_backend_array(returns_y, use_gpu=use_gpu, device_id=device_id, stream=stream)

    if correlation is not None and use_side_hint:
        corr, _ = _to_backend_array(
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


def gpu_cumulative_pnl(
    pnl: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    use_gpu: bool = True,
    device_id: int = 0,
    stream: Optional[object] = None,
    return_backend: bool = False,
) -> np.ndarray:
    """Calculate cumulative PnL using GPU acceleration."""

    pnl_backend, xp = _to_backend_array(
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
