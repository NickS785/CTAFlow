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

import numpy as np
from typing import Tuple, Optional, Union

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


def gpu_backtest_returns(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    threshold: float = 0.0,
    use_gpu: bool = True,
    device_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate positions and PnL using GPU acceleration.

    Computes trading positions based on threshold and calculates PnL.

    Args:
        returns_x: Predictor returns (features)
        returns_y: Target returns (outcomes)
        threshold: Entry threshold (long if >= threshold, short if <= -threshold)
        use_gpu: Whether to use GPU acceleration
        device_id: GPU device ID

    Returns:
        Tuple of (positions, pnl) as NumPy arrays
    """
    if not use_gpu or not GPU_AVAILABLE:
        # CPU fallback
        positions = np.where(
            returns_x >= threshold, 1.0,
            np.where(returns_x <= -threshold, -1.0, 0.0)
        )
        pnl = positions * returns_y
        return positions, pnl

    # GPU computation
    with cp.cuda.Device(device_id):
        rx_gpu = cp.asarray(returns_x)
        ry_gpu = cp.asarray(returns_y)

        positions_gpu = cp.where(
            rx_gpu >= threshold, 1.0,
            cp.where(rx_gpu <= -threshold, -1.0, 0.0)
        )
        pnl_gpu = positions_gpu * ry_gpu

        # Transfer back to CPU
        positions = cp.asnumpy(positions_gpu)
        pnl = cp.asnumpy(pnl_gpu)

    return positions, pnl


def gpu_backtest_threshold(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    correlation: Optional[np.ndarray] = None,
    threshold: float = 0.0,
    use_side_hint: bool = True,
    use_gpu: bool = True,
    device_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate positions and PnL with correlation-based side hint.

    Args:
        returns_x: Predictor returns
        returns_y: Target returns
        correlation: Correlation values for each signal
        threshold: Entry threshold
        use_side_hint: Whether to use correlation sign for position direction
        use_gpu: Whether to use GPU acceleration
        device_id: GPU device ID

    Returns:
        Tuple of (positions, pnl) as NumPy arrays
    """
    if not use_gpu or not GPU_AVAILABLE:
        # CPU fallback
        if correlation is not None and use_side_hint:
            # Apply correlation sign to returns_x
            adjusted_x = returns_x * np.sign(correlation)
        else:
            adjusted_x = returns_x

        positions = np.where(
            adjusted_x >= threshold, 1.0,
            np.where(adjusted_x <= -threshold, -1.0, 0.0)
        )
        pnl = positions * returns_y
        return positions, pnl

    # GPU computation
    with cp.cuda.Device(device_id):
        rx_gpu = cp.asarray(returns_x)
        ry_gpu = cp.asarray(returns_y)

        if correlation is not None and use_side_hint:
            corr_gpu = cp.asarray(correlation)
            adjusted_x_gpu = rx_gpu * cp.sign(corr_gpu)
        else:
            adjusted_x_gpu = rx_gpu

        positions_gpu = cp.where(
            adjusted_x_gpu >= threshold, 1.0,
            cp.where(adjusted_x_gpu <= -threshold, -1.0, 0.0)
        )
        pnl_gpu = positions_gpu * ry_gpu

        positions = cp.asnumpy(positions_gpu)
        pnl = cp.asnumpy(pnl_gpu)

    return positions, pnl


def gpu_cumulative_pnl(
    pnl: np.ndarray,
    use_gpu: bool = True,
    device_id: int = 0,
) -> np.ndarray:
    """Calculate cumulative PnL using GPU acceleration.

    Args:
        pnl: Per-period PnL array
        use_gpu: Whether to use GPU acceleration
        device_id: GPU device ID

    Returns:
        Cumulative PnL as NumPy array
    """
    if not use_gpu or not GPU_AVAILABLE:
        return np.cumsum(pnl)

    with cp.cuda.Device(device_id):
        pnl_gpu = cp.asarray(pnl)
        cumulative_gpu = cp.cumsum(pnl_gpu)
        return cp.asnumpy(cumulative_gpu)


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
