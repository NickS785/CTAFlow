import numpy as np
import pandas as pd
from datetime import time

from CTAFlow.screeners.historical_screener import HistoricalScreener
from CTAFlow.strategy.screener_pipeline import HorizonMapper, ScreenerPipeline


def test_intraday_response_window_changes_returns():
    ts = pd.date_range(
        "2024-01-01 09:30",
        periods=3,
        freq="30min",
        tz="America/Chicago",
    )
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": [100.0, 110.0, 90.0],
            "close": [100.0, 110.0, 80.0],
            "session_id": [1, 1, 1],
        }
    )

    patterns = {
        "short": {
            "pattern_type": "time_predictive_intraday",
            "time": "09:30 -> 10:00",
            "pattern_payload": {
                "time": "09:30 -> 10:00",
                "correlation": 0.5,
                "period_length_min": 30,
            },
        },
        "long": {
            "pattern_type": "time_predictive_intraday",
            "time": "09:30 -> 10:30",
            "pattern_payload": {
                "time": "09:30 -> 10:30",
                "correlation": 0.5,
                "period_length_min": 60,
            },
        },
    }

    pipeline = ScreenerPipeline(use_gpu=False, tz="America/Chicago", time_match="hms")
    features = pipeline.build_features(df, patterns)
    horizon_mapper = HorizonMapper(tz="America/Chicago", time_match="second")
    xy = horizon_mapper.build_xy(
        features,
        patterns,
        default_intraday_minutes=30,
        predictor_minutes=30,
    )

    short_return = float(xy.loc[xy["gate"].str.contains("short"), "returns_y"].iloc[0])
    long_return = float(xy.loc[xy["gate"].str.contains("long"), "returns_y"].iloc[0])

    assert short_return != long_return
    assert not np.isnan(short_return)
    assert not np.isnan(long_return)


def test_min_spacing_filters_overlapping_target_times():
    clocks = [time(2, 30), time(2, 40), time(3, 0)]

    filtered = HistoricalScreener._filter_times_by_spacing(clocks, min_spacing_minutes=30)

    assert [c.strftime("%H:%M") for c in filtered] == ["02:30", "03:00"]
