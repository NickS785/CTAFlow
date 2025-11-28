"""Test script for GPU-accelerated backtester functionality."""

import pathlib
import sys

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path for direct test execution
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CTAFlow.strategy.backtester import ScreenerBacktester
from CTAFlow.strategy.gpu_acceleration import (
    GPU_AVAILABLE,
    get_gpu_info,
    gpu_backtest_threshold,
)

def test_gpu_backtester():
    """Test GPU backtester with synthetic data."""

    # Print GPU info
    print("=" * 60)
    print("GPU Acceleration Status")
    print("=" * 60)
    info = get_gpu_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print()

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 10000

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='5min')
    returns_x = np.random.randn(n_samples) * 0.01
    returns_y = returns_x * 0.5 + np.random.randn(n_samples) * 0.005
    correlation = np.full(n_samples, 0.5)

    xy = pd.DataFrame({
        'ts_decision': dates,
        'returns_x': returns_x,
        'returns_y': returns_y,
        'correlation': correlation,
    })

    print("=" * 60)
    print("Running Backtest Comparison")
    print("=" * 60)
    print(f"Sample size: {n_samples:,}")
    print()

    # Test with GPU enabled (will use CPU if GPU not available)
    bt_gpu = ScreenerBacktester(use_gpu=True)
    result_gpu = bt_gpu.threshold(xy, threshold=0.005, use_side_hint=True)

    print("GPU-Enabled Backtester (using", "GPU" if bt_gpu.use_gpu else "CPU", ")")
    print(f"  Total Return: {result_gpu['summary'].total_return:.6f}")
    print(f"  Mean Return: {result_gpu['summary'].mean_return:.6f}")
    print(f"  Hit Rate: {result_gpu['summary'].hit_rate:.3f}")
    print(f"  Sharpe Ratio: {result_gpu['summary'].sharpe:.3f}")
    print(f"  Max Drawdown: {result_gpu['summary'].max_drawdown:.6f}")
    print(f"  Number of Trades: {result_gpu['summary'].trades}")
    print()

    # Test with GPU disabled (force CPU)
    bt_cpu = ScreenerBacktester(use_gpu=False)
    result_cpu = bt_cpu.threshold(xy, threshold=0.005, use_side_hint=True)

    print("CPU-Only Backtester")
    print(f"  Total Return: {result_cpu['summary'].total_return:.6f}")
    print(f"  Mean Return: {result_cpu['summary'].mean_return:.6f}")
    print(f"  Hit Rate: {result_cpu['summary'].hit_rate:.3f}")
    print(f"  Sharpe Ratio: {result_cpu['summary'].sharpe:.3f}")
    print(f"  Max Drawdown: {result_cpu['summary'].max_drawdown:.6f}")
    print(f"  Number of Trades: {result_cpu['summary'].trades}")
    print()

    # Verify results match
    print("=" * 60)
    print("Verification")
    print("=" * 60)

    total_return_match = np.isclose(
        result_gpu['summary'].total_return,
        result_cpu['summary'].total_return,
        rtol=1e-10
    )

    pnl_match = np.allclose(
        result_gpu['pnl'].values,
        result_cpu['pnl'].values,
        rtol=1e-10,
        equal_nan=True
    )

    print(f"Total returns match: {total_return_match}")
    print(f"PnL series match: {pnl_match}")

    if total_return_match and pnl_match:
        print("\n[PASS] GPU and CPU results are identical!")
        print("[PASS] GPU acceleration module is working correctly!")
    else:
        print("\n[FAIL] Warning: Results differ between GPU and CPU!")
        print("  This may indicate an issue with GPU acceleration.")

    print()
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)

    return result_gpu, result_cpu


def test_gpu_backtest_threshold_accepts_pandas_inputs():
    """Ensure GPU helpers accept pandas objects and stay consistent with CPU."""

    np.random.seed(0)
    n_samples = 128
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    returns_x = pd.Series(np.random.randn(n_samples) * 0.01, index=dates)
    returns_y = pd.Series(np.random.randn(n_samples) * 0.01, index=dates)
    correlation = pd.Series(np.random.uniform(-1, 1, size=n_samples), index=dates)

    positions_cpu, pnl_cpu = gpu_backtest_threshold(
        returns_x.values,
        returns_y.values,
        correlation=correlation.values,
        threshold=0.0,
        use_side_hint=True,
        use_gpu=False,
    )

    positions_gpu, pnl_gpu = gpu_backtest_threshold(
        returns_x,
        returns_y,
        correlation=correlation,
        threshold=0.0,
        use_side_hint=True,
        use_gpu=GPU_AVAILABLE,
    )

    assert np.allclose(positions_cpu, positions_gpu, rtol=1e-10, equal_nan=True)
    assert np.allclose(pnl_cpu, pnl_gpu, rtol=1e-10, equal_nan=True)

if __name__ == "__main__":
    test_gpu_backtester()
