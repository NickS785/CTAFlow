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


def test_resolve_collapses_with_group_members():
    resolver = PredictionToPosition()
    ts = pd.Timestamp("2024-01-01 08:30", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_decision": [ts, ts, ts + pd.Timedelta(hours=1)],
            "returns_x": [0.1, -0.4, 0.2],
            "returns_y": [0.05, 0.1, -0.2],
            "correlation": [0.4, 0.9, 0.7],
            "momentum_type": ["open", "close", "open"],
        }
    )

    resolved = resolver.resolve(df, group_field="momentum_type")

    assert len(resolved) == 3
    ts_rows = resolved[resolved["ts_decision"] == ts]
    assert sorted(ts_rows["momentum_type"].tolist()) == ["close", "open"]
    assert ts_rows.iloc[0]["_group_members"] == ["open", "close"]
    assert ts_rows.iloc[1]["_group_members"] == ["open", "close"]
    close_row = ts_rows[ts_rows["momentum_type"] == "close"].iloc[0]
    assert close_row["returns_x"] == pytest.approx(-0.4)


def test_resolve_respects_grouping_field_per_ticker():
    resolver = PredictionToPosition()
    ts = pd.Timestamp("2024-01-05 09:00")
    df = pd.DataFrame(
        {
            "ts_decision": [ts, ts, ts],
            "returns_x": [0.05, -0.02, 0.01],
            "returns_y": [0.02, 0.01, 0.03],
            "correlation": [0.9, 0.7, 0.5],
            "ticker": ["ZC_F", "ZW_F", "ZC_F"],
        }
    )

    resolved = resolver.resolve(df, group_field="ticker")

    assert len(resolved) == 2
    assert set(resolved["ticker"]) == {"ZC_F", "ZW_F"}
    corn_row = resolved[resolved["ticker"] == "ZC_F"].iloc[0]
    assert corn_row["returns_x"] == pytest.approx(0.05)
    wheat_row = resolved[resolved["ticker"] == "ZW_F"].iloc[0]
    assert wheat_row["returns_x"] == pytest.approx(-0.02)
    assert corn_row["_group_members"] == ["ZC_F", "ZW_F"]
    assert wheat_row["_group_members"] == ["ZC_F", "ZW_F"]
