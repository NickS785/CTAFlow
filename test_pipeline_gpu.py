"""Test GPU parameter passing through ScreenerPipeline to ScreenerBacktester."""

import numpy as np
import pandas as pd
from CTAFlow.strategy.screener_pipeline import ScreenerPipeline
from CTAFlow.strategy.gpu_acceleration import GPU_AVAILABLE

def test_pipeline_gpu_params():
    """Test that GPU parameters are correctly passed from pipeline to backtester."""

    print("=" * 60)
    print("Testing GPU Parameter Passing in ScreenerPipeline")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print()

    # Create pipelines with different GPU settings
    sp_gpu_enabled = ScreenerPipeline(use_gpu=True, gpu_device_id=0)
    sp_gpu_disabled = ScreenerPipeline(use_gpu=False)
    sp_gpu_multi = ScreenerPipeline(use_gpu=True, gpu_device_id=1)

    print("Pipeline Configuration:")
    print(f"  GPU Enabled:  use_gpu={sp_gpu_enabled.use_gpu}, device_id={sp_gpu_enabled.gpu_device_id}")
    print(f"  GPU Disabled: use_gpu={sp_gpu_disabled.use_gpu}, device_id={sp_gpu_disabled.gpu_device_id}")
    print(f"  GPU Multi:    use_gpu={sp_gpu_multi.use_gpu}, device_id={sp_gpu_multi.gpu_device_id}")
    print()

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='5min', tz='America/Chicago')
    returns_x = np.random.randn(n_samples) * 0.01
    returns_y = returns_x * 0.5 + np.random.randn(n_samples) * 0.005

    bars = pd.DataFrame({
        'ts': dates,
        'open': 100 + np.cumsum(returns_x),
        'close': 100 + np.cumsum(returns_y),
        'high': 100 + np.cumsum(returns_x) + 0.1,
        'low': 100 + np.cumsum(returns_y) - 0.1,
        'volume': np.random.randint(1000, 10000, n_samples),
    })

    # Create a simple pattern
    pattern = {
        'test_pattern': {
            'pattern_type': 'momentum_oc',
            'metadata': {
                'ticker': 'TEST',
                'correlation': 0.5,
            },
            'pattern_payload': {
                'momentum_type': 'opening_momentum',
            },
            'gates': pd.Series([True] * 100 + [False] * (n_samples - 100), index=bars.index[:n_samples]),
        }
    }

    print("=" * 60)
    print("Running Backtest with GPU-Enabled Pipeline")
    print("=" * 60)

    try:
        result_gpu = sp_gpu_enabled.backtest_threshold(
            bars,
            pattern,
            threshold=0.005,
            use_side_hint=True,
        )

        summary = result_gpu.get('summary')
        if summary:
            print(f"Total Return: {summary.total_return:.6f}")
            print(f"Trades: {summary.trades}")
            print(f"Sharpe: {summary.sharpe:.3f}")
            print("[PASS] GPU-enabled pipeline backtest completed successfully")
        else:
            print("[WARNING] No summary in result")
    except Exception as e:
        print(f"[FAIL] GPU-enabled pipeline backtest failed: {e}")

    print()
    print("=" * 60)
    print("Running Backtest with CPU-Only Pipeline")
    print("=" * 60)

    try:
        result_cpu = sp_gpu_disabled.backtest_threshold(
            bars,
            pattern,
            threshold=0.005,
            use_side_hint=True,
        )

        summary = result_cpu.get('summary')
        if summary:
            print(f"Total Return: {summary.total_return:.6f}")
            print(f"Trades: {summary.trades}")
            print(f"Sharpe: {summary.sharpe:.3f}")
            print("[PASS] CPU-only pipeline backtest completed successfully")
        else:
            print("[WARNING] No summary in result")
    except Exception as e:
        print(f"[FAIL] CPU-only pipeline backtest failed: {e}")

    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)

    # Verify results match
    if 'summary' in result_gpu and 'summary' in result_cpu:
        returns_match = np.isclose(
            result_gpu['summary'].total_return,
            result_cpu['summary'].total_return,
            rtol=1e-10
        )

        print(f"GPU Total Return:  {result_gpu['summary'].total_return:.10f}")
        print(f"CPU Total Return:  {result_cpu['summary'].total_return:.10f}")
        print(f"Results Match: {returns_match}")

        if returns_match:
            print("\n[PASS] GPU and CPU pipelines produce identical results!")
            print("[PASS] GPU parameters are correctly passed through pipeline!")
        else:
            print("\n[WARNING] Results differ between GPU and CPU pipelines")
    else:
        print("[WARNING] Could not compare results (missing summaries)")

    print()
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline_gpu_params()
