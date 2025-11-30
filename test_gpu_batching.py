"""Test GPU batching functionality for threshold sweeps.

This test suite verifies that:
1. Batch threshold sweep produces correct results
2. Batch results match sequential threshold() calls
3. GPU batching works on both CPU and GPU backends
4. Performance improvement is measurable
"""

import time
import numpy as np
import pandas as pd
from CTAFlow.strategy.gpu_acceleration import (
    GPU_AVAILABLE,
    gpu_batch_threshold_sweep,
    gpu_backtest_threshold,
)
from CTAFlow.strategy.backtester import ScreenerBacktester


def create_test_data(n_samples=1000, seed=42):
    """Create synthetic test data for backtesting."""
    np.random.seed(seed)

    returns_x = np.random.randn(n_samples) * 0.02  # 2% volatility
    returns_y = np.random.randn(n_samples) * 0.02
    correlation = np.random.uniform(-1, 1, n_samples)

    # Add some realistic structure
    # Returns_y has some dependency on returns_x
    returns_y = 0.3 * returns_x + 0.7 * returns_y

    return returns_x, returns_y, correlation


def test_gpu_batch_basic():
    """Test that batch threshold sweep produces valid output."""
    print("\n" + "=" * 60)
    print("TEST: Basic Batch Functionality")
    print("=" * 60)

    returns_x, returns_y, correlation = create_test_data(n_samples=500)
    thresholds = [0.0, 0.5, 1.0, 1.5]

    results = gpu_batch_threshold_sweep(
        returns_x=returns_x,
        returns_y=returns_y,
        thresholds=thresholds,
        correlation=correlation,
        use_side_hint=True,
        use_gpu=GPU_AVAILABLE,  # Will fallback to CPU if no GPU
    )

    print(f"Tested {len(thresholds)} thresholds")
    print(f"Sample size: {len(returns_x)}")
    print(f"GPU Available: {GPU_AVAILABLE}")

    # Verify all thresholds have results
    assert len(results) == len(thresholds), "Missing threshold results"

    # Verify structure of each result
    for threshold, metrics in results.items():
        assert 'positions' in metrics, f"Missing positions for threshold {threshold}"
        assert 'pnl' in metrics, f"Missing pnl for threshold {threshold}"
        assert 'cumulative' in metrics, f"Missing cumulative for threshold {threshold}"
        assert 'drawdown' in metrics, f"Missing drawdown for threshold {threshold}"
        assert 'max_drawdown' in metrics, f"Missing max_drawdown for threshold {threshold}"

        # Verify array shapes
        assert len(metrics['positions']) == len(returns_x)
        assert len(metrics['pnl']) == len(returns_x)
        assert len(metrics['cumulative']) == len(returns_x)

        print(f"  Threshold {threshold:>4.1f}: "
              f"Trades={int((metrics['positions'] != 0).sum()):>4d}, "
              f"Total PnL={metrics['pnl'].sum():>8.4f}, "
              f"Max DD={metrics['max_drawdown']:>8.4f}")

    print("[PASS] Basic batch functionality test")
    return results


def test_batch_vs_sequential():
    """Verify batch results match sequential threshold() calls."""
    print("\n" + "=" * 60)
    print("TEST: Batch vs Sequential Correctness")
    print("=" * 60)

    returns_x, returns_y, correlation = create_test_data(n_samples=300)
    thresholds = [0.0, 0.5, 1.0]

    # Get batch results
    batch_results = gpu_batch_threshold_sweep(
        returns_x=returns_x,
        returns_y=returns_y,
        thresholds=thresholds,
        correlation=correlation,
        use_side_hint=True,
        use_gpu=GPU_AVAILABLE,
    )

    # Get sequential results
    sequential_results = {}
    for threshold in thresholds:
        pos, pnl = gpu_backtest_threshold(
            returns_x=returns_x,
            returns_y=returns_y,
            correlation=correlation,
            threshold=threshold,
            use_side_hint=True,
            use_gpu=GPU_AVAILABLE,
        )
        cumulative = np.cumsum(pnl)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - rolling_max
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        sequential_results[threshold] = {
            'positions': pos,
            'pnl': pnl,
            'cumulative': cumulative,
            'max_drawdown': max_dd,
        }

    # Compare results
    all_match = True
    for threshold in thresholds:
        batch = batch_results[threshold]
        sequential = sequential_results[threshold]

        # Compare arrays (allowing for floating point tolerance)
        pos_match = np.allclose(batch['positions'], sequential['positions'], atol=1e-10)
        pnl_match = np.allclose(batch['pnl'], sequential['pnl'], atol=1e-10)
        cum_match = np.allclose(batch['cumulative'], sequential['cumulative'], atol=1e-10)
        dd_match = abs(batch['max_drawdown'] - sequential['max_drawdown']) < 1e-10

        match_status = "MATCH" if (pos_match and pnl_match and cum_match and dd_match) else "MISMATCH"
        print(f"  Threshold {threshold}: {match_status}")

        if not (pos_match and pnl_match and cum_match and dd_match):
            print(f"    Positions: {'OK' if pos_match else 'FAIL'}")
            print(f"    PnL: {'OK' if pnl_match else 'FAIL'}")
            print(f"    Cumulative: {'OK' if cum_match else 'FAIL'}")
            print(f"    Max DD: {'OK' if dd_match else 'FAIL'}")
            all_match = False

    if all_match:
        print("[PASS] Batch and sequential results match perfectly")
    else:
        print("[FAIL] Some batch results don't match sequential")
        raise AssertionError("Batch and sequential results don't match")


def test_backtester_batch_method():
    """Test the ScreenerBacktester.batch_threshold_sweep() method."""
    print("\n" + "=" * 60)
    print("TEST: ScreenerBacktester Batch Method")
    print("=" * 60)

    # Create test DataFrame similar to ScreenerPipeline.build_xy output
    n_samples = 400
    returns_x, returns_y, correlation = create_test_data(n_samples=n_samples)

    xy = pd.DataFrame({
        'returns_x': returns_x,
        'returns_y': returns_y,
        'correlation': correlation,
        'ts_decision': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
    })

    tester = ScreenerBacktester(use_gpu=GPU_AVAILABLE, annualisation=252)

    thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = tester.batch_threshold_sweep(xy, thresholds, use_side_hint=True)

    print(f"Tested {len(thresholds)} thresholds")
    print(f"Results structure verified")

    # Verify all thresholds have complete backtest results
    for threshold, result in results.items():
        assert 'summary' in result, f"Missing summary for threshold {threshold}"
        assert 'pnl' in result
        assert 'positions' in result
        assert 'cumulative' in result

        summary = result['summary']
        print(f"  Threshold {threshold:>4.1f}: "
              f"Trades={summary.trades:>4d}, "
              f"Return={summary.total_return:>8.4f}, "
              f"Sharpe={summary.sharpe:>6.2f}, "
              f"Max DD={summary.max_drawdown:>8.4f}")

    print("[PASS] ScreenerBacktester batch method test")
    return results


def test_performance_comparison():
    """Compare performance of batch vs sequential approaches."""
    print("\n" + "=" * 60)
    print("TEST: Performance Comparison")
    print("=" * 60)

    n_samples = 2000
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    n_thresholds = len(thresholds)

    returns_x, returns_y, correlation = create_test_data(n_samples=n_samples)

    xy = pd.DataFrame({
        'returns_x': returns_x,
        'returns_y': returns_y,
        'correlation': correlation,
        'ts_decision': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
    })

    tester = ScreenerBacktester(use_gpu=GPU_AVAILABLE, annualisation=252)

    # Batch approach
    start_batch = time.perf_counter()
    batch_results = tester.batch_threshold_sweep(xy, thresholds, use_side_hint=True)
    end_batch = time.perf_counter()
    batch_time = end_batch - start_batch

    # Sequential approach
    start_seq = time.perf_counter()
    sequential_results = {}
    for threshold in thresholds:
        sequential_results[threshold] = tester.threshold(
            xy, threshold=threshold, use_side_hint=True
        )
    end_seq = time.perf_counter()
    sequential_time = end_seq - start_seq

    speedup = sequential_time / batch_time if batch_time > 0 else float('inf')

    print(f"Sample size: {n_samples}")
    print(f"Number of thresholds: {n_thresholds}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"\nSequential time: {sequential_time*1000:.2f} ms")
    print(f"Batch time:      {batch_time*1000:.2f} ms")
    print(f"Speedup:         {speedup:.2f}x")

    if speedup > 1.0:
        print(f"[PASS] Batch approach is {speedup:.2f}x faster")
    else:
        print(f"[WARN] Batch approach is slower (might be expected for small datasets on CPU)")

    return {
        'sequential_time': sequential_time,
        'batch_time': batch_time,
        'speedup': speedup,
    }


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)

    returns_x, returns_y, correlation = create_test_data(n_samples=100)

    # Test 1: Empty thresholds
    results = gpu_batch_threshold_sweep(
        returns_x=returns_x,
        returns_y=returns_y,
        thresholds=[],
        use_gpu=GPU_AVAILABLE,
    )
    assert len(results) == 0, "Empty thresholds should return empty dict"
    print("  [OK] Empty thresholds handled correctly")

    # Test 2: Single threshold
    results = gpu_batch_threshold_sweep(
        returns_x=returns_x,
        returns_y=returns_y,
        thresholds=[1.0],
        use_gpu=GPU_AVAILABLE,
    )
    assert len(results) == 1, "Single threshold should return single result"
    assert 1.0 in results, "Result should be keyed by threshold value"
    print("  [OK] Single threshold handled correctly")

    # Test 3: Without correlation
    results = gpu_batch_threshold_sweep(
        returns_x=returns_x,
        returns_y=returns_y,
        thresholds=[0.0, 1.0],
        correlation=None,
        use_side_hint=False,
        use_gpu=GPU_AVAILABLE,
    )
    assert len(results) == 2, "Should work without correlation"
    print("  [OK] No correlation handled correctly")

    # Test 4: Very large threshold (should have no trades)
    results = gpu_batch_threshold_sweep(
        returns_x=returns_x,
        returns_y=returns_y,
        thresholds=[100.0],
        use_gpu=GPU_AVAILABLE,
    )
    assert (results[100.0]['positions'] == 0).all(), "Very large threshold should produce no positions"
    print("  [OK] Large threshold handled correctly")

    print("[PASS] All edge cases handled correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("GPU BATCHING TEST SUITE")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")

    try:
        # Run tests
        test_gpu_batch_basic()
        test_batch_vs_sequential()
        test_backtester_batch_method()
        test_edge_cases()
        performance_metrics = test_performance_comparison()

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("All tests PASSED")
        print(f"\nPerformance Summary:")
        print(f"  Batch speedup: {performance_metrics['speedup']:.2f}x")
        print(f"  Backend: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (NumPy)'}")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
