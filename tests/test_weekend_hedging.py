import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from CTAFlow.strategy.screener_pipeline import HorizonMapper, ScreenerPipeline


def _weekend_pattern() -> dict:
    return {
        "pattern_type": "weekend_hedging",
        "pattern_payload": {
            "weekday": "Friday->Monday",
            "gate_time_hhmm": "13:30",
            "bias": "buy",
            "p_value": 0.01,
        },
        "metadata": {
            "best_weekday": "Friday->Monday",
            "gate_time_hhmm": "13:30",
        },
    }


def _weekend_bars(two_weeks: bool = False, holiday_second_week: bool = False) -> pd.DataFrame:
    tz = "America/Chicago"
    base_friday = pd.Timestamp("2024-01-05 13:30", tz=tz)
    target_times = ["08:30", "09:00", "10:30", "13:30"]
    rows: list[dict[str, object]] = []

    weeks = 2 if two_weeks else 1
    for week in range(weeks):
        friday_ts = base_friday + pd.Timedelta(weeks=week)
        friday_price = 100.0 + week
        rows.append(
            {
                "ts": friday_ts,
                "open": friday_price - 1.0,
                "close": friday_price,
                "session_id": friday_ts.strftime("%Y-%m-%d"),
            }
        )

        monday_date = (friday_ts + pd.Timedelta(days=3)).date()
        if holiday_second_week and week == weeks - 1 and two_weeks:
            tuesday_date = (friday_ts + pd.Timedelta(days=4)).date()
            for idx, clock in enumerate(target_times):
                tuesday_ts = pd.Timestamp(f"{tuesday_date} {clock}", tz=tz)
                tuesday_price = friday_price + 3.0 + idx
                rows.append(
                    {
                        "ts": tuesday_ts,
                        "open": tuesday_price,
                        "close": tuesday_price,
                        "session_id": tuesday_ts.strftime("%Y-%m-%d"),
                    }
                )
        else:
            for idx, clock in enumerate(target_times):
                monday_ts = pd.Timestamp(f"{monday_date} {clock}", tz=tz)
                monday_price = friday_price + 2.0 + idx
                rows.append(
                    {
                        "ts": monday_ts,
                        "open": monday_price,
                        "close": monday_price,
                        "session_id": monday_ts.strftime("%Y-%m-%d"),
                    }
                )

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


def test_weekend_mapper_flags_numeric():
    bars = _weekend_bars()
    pattern = _weekend_pattern()
    pipeline = ScreenerPipeline(tz="America/Chicago")
    enriched = pipeline.build_features(bars.copy(), {"usa_winter": pattern})

    features = pattern.get("features")
    assert isinstance(features, dict)
    gate_col = features["pattern_gate_col"]
    weekday_col = features["pattern_weekday_col"]

    assert enriched[gate_col].sum() == 1
    assert enriched.loc[enriched[gate_col] == 1, "clock_time"].iloc[0].startswith("13:30")
    assert enriched[weekday_col].dtype == np.int8

    monday_rows = enriched.loc[enriched["weekday_lower"] == "monday", weekday_col]
    assert not monday_rows.empty
    assert (monday_rows == 1).all()
    assert monday_rows.sum() == 4


def test_weekend_mapper_single_gate_counts_multiple_weeks():
    bars = _weekend_bars(two_weeks=True)
    pattern = _weekend_pattern()
    pipeline = ScreenerPipeline(tz="America/Chicago")
    enriched = pipeline.build_features(bars.copy(), {"usa_winter": pattern})

    features = pattern["features"]
    gate_col = features["pattern_gate_col"]
    weekday_col = features["pattern_weekday_col"]

    assert enriched[gate_col].sum() == 2
    session_counts = enriched.loc[enriched[gate_col] == 1, "session_id"].value_counts()
    assert (session_counts == 1).all()

    monday_rows = enriched.loc[enriched[weekday_col] == 1]
    assert set(monday_rows["weekday_lower"]) == {"monday"}
    monday_counts = monday_rows["session_id"].value_counts()
    assert (monday_counts == 4).all()


def test_weekend_backtester_links_friday_to_monday():
    bars = _weekend_bars(two_weeks=True)
    pattern = _weekend_pattern()
    pipeline = ScreenerPipeline(tz="America/Chicago")
    enriched = pipeline.build_features(bars.copy(), {"usa_winter": pattern})

    mapper = HorizonMapper(tz="America/Chicago", weekend_exit_policy="last")
    xy = mapper.build_xy(enriched, {"usa_winter": pattern}, ensure_gates=False)

    assert len(xy) == 2
    expected_returns = []
    for week in range(2):
        entry = 100.0 + week
        exit_price = entry + 5.0
        expected_returns.append(np.log(exit_price / entry))
    assert np.allclose(sorted(xy["returns_y"].values), sorted(expected_returns))


def test_weekend_backtester_rolls_through_holiday():
    bars = _weekend_bars(two_weeks=True, holiday_second_week=True)
    pattern = _weekend_pattern()
    pipeline = ScreenerPipeline(tz="America/Chicago")
    enriched = pipeline.build_features(bars.copy(), {"usa_winter": pattern})

    mapper = HorizonMapper(tz="America/Chicago", weekend_exit_policy="last")
    xy = mapper.build_xy(enriched, {"usa_winter": pattern}, ensure_gates=False)

    assert len(xy) == 2
    expected_returns = [np.log(105.0 / 100.0), np.log(107.0 / 101.0)]
    assert np.allclose(sorted(xy["returns_y"].values), sorted(expected_returns))
