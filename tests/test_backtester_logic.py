import numpy as np
import pandas as pd

from CTAFlow.strategy.backtester import ScreenerBacktester
from CTAFlow.strategy.prediction_to_position import PredictionToPosition


def test_cpu_threshold_respects_correlation_sign():
    xy = pd.DataFrame(
        {
            "returns_x": [0.5, 0.5, -0.4, 0.2],
            "returns_y": [0.1, -0.2, 0.05, -0.1],
            "correlation": [-1.0, 1.0, -0.5, 0.3],
            "ts_decision": pd.date_range("2024-01-01", periods=4, freq="h"),
        }
    )

    tester = ScreenerBacktester(use_gpu=False)
    result = tester.threshold(xy, threshold=0.0, use_side_hint=True)

    expected = np.sign(xy["returns_x"].to_numpy() * np.sign(xy["correlation"].to_numpy()))
    np.testing.assert_array_equal(result["positions"].to_numpy(), expected)


def test_prediction_to_position_respects_gate_metadata():
    ts = pd.to_datetime(["2024-01-01 09:00", "2024-01-01 09:00", "2024-01-01 10:00"])
    df = pd.DataFrame(
        {
            "ts_decision": ts,
            "returns_x": [0.5, 0.1, 0.2],
            "returns_y": [0.2, 0.05, -0.1],
            "gate_direction": [-1, 1, 1],
            "strength": [2.0, 0.1, 1.0],
            "correlation": [1.0, 1.0, 1.0],
        }
    )

    resolver = PredictionToPosition()
    aggregated = resolver.aggregate(df)

    assert len(aggregated) == 2
    assert aggregated.loc[aggregated["ts_decision"] == ts[0], "prediction_position"].iloc[0] == -1
    assert "_ptp_score" not in aggregated.columns


def test_batch_patterns_matches_individual_results():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    xy_a = pd.DataFrame(
        {
            "returns_x": [0.3, -0.6, 0.0, 0.8, -0.2],
            "returns_y": [0.05, -0.1, 0.02, 0.1, -0.03],
            "correlation": [1, -1, 1, 1, -1],
            "ts_decision": idx,
        }
    )
    xy_b = pd.DataFrame(
        {
            "returns_x": [-0.4, 0.1, 0.2, -0.5, 0.7],
            "returns_y": [-0.02, 0.03, -0.01, 0.04, 0.05],
            "correlation": [-1, 1, -1, 1, 1],
            "ts_decision": idx,
        }
    )

    tester = ScreenerBacktester(use_gpu=False)
    single_a = tester.threshold(xy_a, threshold=0.1, use_side_hint=True)
    single_b = tester.threshold(xy_b, threshold=0.1, use_side_hint=True)

    batched = tester.batch_patterns({"a": xy_a, "b": xy_b}, threshold=0.1, use_side_hint=True)

    np.testing.assert_allclose(batched["a"]["positions"].to_numpy(), single_a["positions"].to_numpy())
    np.testing.assert_allclose(batched["b"]["positions"].to_numpy(), single_b["positions"].to_numpy())

    np.testing.assert_allclose(batched["a"]["pnl"].to_numpy(), single_a["pnl"].to_numpy())
    np.testing.assert_allclose(batched["b"]["pnl"].to_numpy(), single_b["pnl"].to_numpy())
