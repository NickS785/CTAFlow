import pandas as pd
import pytest

from CTAFlow.strategy.prediction_to_position import PredictionToPosition


def test_prediction_to_position_collapses_duplicate_timestamps():
    resolver = PredictionToPosition(neutral_tolerance=1e-6)
    df = pd.DataFrame(
        {
            "ts_decision": pd.to_datetime([
                "2024-01-01 08:30", "2024-01-01 08:30", "2024-01-02 08:30",
            ]),
            "returns_x": [0.02, -0.01, -0.03],
            "returns_y": [0.01, 0.01, -0.02],
            "correlation": [0.8, -0.6, 0.5],
            "gate": ["a_gate", "b_gate", "c_gate"],
            "pattern_type": ["momentum_weekday"] * 3,
        }
    )

    result = resolver.aggregate(df)
    assert len(result) == 2
    assert list(result["prediction_position"]) == [1, -1]
    assert result.loc[0, "returns_x"] == pytest.approx(0.005)
    assert result.loc[0, "ts_decision"] == pd.Timestamp("2024-01-01 08:30")


def test_prediction_to_position_handles_empty_frame():
    resolver = PredictionToPosition()
    df = pd.DataFrame(columns=["ts_decision", "returns_x", "returns_y"])
    resolved = resolver.aggregate(df)
    assert resolved.empty
