import re
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CTAFlow" / "strategy" / "screener_pipeline.py"

spec = importlib.util.spec_from_file_location("screener_pipeline", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
ScreenerPipeline = module.ScreenerPipeline
HorizonMapper = module.HorizonMapper


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _build_patterns():
    return {
        "months_9_10_seasonality|weekday_mean|Tuesday": {
            "pattern_type": "weekday_mean",
            "pattern_payload": {
                "day": "Tuesday",
                "mean": 0.15,
                "t_stat": 2.8,
                "p_value": 0.02,
            },
            "metadata": {"screen_type": "seasonality"},
        },
        "months_9_10_seasonality|time_predictive_nextday|13:30:00": {
            "pattern_type": "time_predictive_nextday",
            "pattern_payload": {
                "time": "13:30:00",
                "correlation": -0.45,
                "p_value": 0.01,
            },
            "metadata": {"screen_type": "seasonality"},
        },
        "months_9_10_seasonality|time_predictive_nextweek|13:30:00": {
            "pattern_type": "time_predictive_nextweek",
            "pattern_payload": {
                "time": "13:30:00",
                "correlation": 0.55,
                "p_value": 0.02,
                "strongest_days": ["Wednesday"],
            },
            "metadata": {"screen_type": "seasonality"},
        },
        "orderflow_scan|orderflow_weekly|friday|buy_pressure": {
            "pattern_type": "orderflow_weekly",
            "pattern_payload": {
                "weekday": "Friday",
                "metric": "buy_pressure",
                "mean": 0.17,
                "t_stat": 3.1,
                "q_value": 0.03,
                "n": 30,
            },
            "metadata": {"orderflow_bias": "buy"},
        },
        "orderflow_scan|orderflow_week_of_month|friday|w1": {
            "pattern_type": "orderflow_week_of_month",
            "pattern_payload": {
                "weekday": "Friday",
                "week_of_month": 1,
                "metric": "net_pressure",
                "mean": 0.21,
                "t_stat": 2.4,
                "q_value": 0.05,
                "n": 18,
                "pressure_bias": "sell",
            },
            "metadata": {},
        },
        "orderflow_scan|orderflow_peak_pressure|friday|0930": {
            "pattern_type": "orderflow_peak_pressure",
            "pattern_payload": {
                "weekday": "Friday",
                "clock_time": "09:30:00.123000",
                "metric": "net_pressure",
                "pressure_bias": "sell",
                "seasonality_mean": 0.22,
                "seasonality_t_stat": 2.5,
                "seasonality_q_value": 0.04,
                "intraday_mean": 0.31,
                "intraday_n": 5,
            },
            "metadata": {},
        },
        "other|pattern": {
            "pattern_type": "mystery",
            "pattern_payload": {},
        },
    }


def test_screener_pipeline_generates_sparse_gates():
    tz = "America/Chicago"
    timestamps = [
        "2023-09-01 07:00",
        "2023-09-01 09:30",
        "2023-09-01 13:30",
        "2023-09-04 07:00",
        "2023-09-05 07:00",
        "2023-09-05 13:30",
        "2023-09-06 07:00",
        "2023-09-06 13:30",
        "2023-09-08 07:00",
        "2023-09-08 09:30",
        "2023-10-03 07:00",
        "2023-10-03 13:30",
    ]

    bars = pd.DataFrame({"ts": pd.to_datetime(timestamps)})

    pipeline = ScreenerPipeline(tz=tz)
    features = pipeline.build_features(bars, _build_patterns())

    assert features["ts"].dt.tz is not None
    assert features["ts"].dt.tz.zone == tz
    for column in ["month", "weekday", "wom", "clock_time", "clock_time_us", "session_id"]:
        assert column in features.columns

    weekday_key = "months_9_10_seasonality|weekday_mean|Tuesday"
    weekday_base = _slug(weekday_key)
    weekday_gate = f"{weekday_base}_gate"
    weekday_mean_col = f"{weekday_base}_mean"

    assert features.loc[features["weekday"] == "Tuesday", weekday_gate].eq(1).all()
    assert features.loc[features["weekday"] != "Tuesday", weekday_gate].eq(0).all()
    assert set(features.loc[features[weekday_gate] == 1, weekday_mean_col]) == {0.15}
    assert f"{weekday_base}_t" not in features.columns

    nextday_key = "months_9_10_seasonality|time_predictive_nextday|13:30:00"
    nextday_gate = f"{_slug(nextday_key)}_gate"
    assert features.loc[features["clock_time"] == "13:30:00", nextday_gate].eq(1).all()

    nextweek_key = "months_9_10_seasonality|time_predictive_nextweek|13:30:00"
    nextweek_gate = f"{_slug(nextweek_key)}_gate"
    mask_wed = (features["weekday"] == "Wednesday") & (features["clock_time"] == "13:30:00")
    assert features.loc[mask_wed, nextweek_gate].eq(1).all()
    assert features.loc[~mask_wed & (features["clock_time"] == "13:30:00"), nextweek_gate].eq(0).all()

    weekly_key = "orderflow_scan|orderflow_weekly|friday|buy_pressure"
    weekly_base = _slug(weekly_key)
    weekly_gate = f"{weekly_base}_gate"
    weekly_bias_col = f"{weekly_base}_bias"
    assert features.loc[features["weekday"] == "Friday", weekly_gate].eq(1).all()
    assert set(features.loc[features[weekly_gate] == 1, weekly_bias_col]) == {"buy"}
    assert f"{weekly_base}_t" not in features.columns
    assert f"{weekly_base}_q" not in features.columns

    wom_key = "orderflow_scan|orderflow_week_of_month|friday|w1"
    wom_base = _slug(wom_key)
    wom_gate = f"{wom_base}_gate"
    wom_mask = (features["weekday"] == "Friday") & (features["wom"] == 1)
    assert features.loc[wom_mask, wom_gate].eq(1).all()
    assert features.loc[~wom_mask, wom_gate].eq(0).all()
    assert f"{wom_base}_t" not in features.columns
    assert f"{wom_base}_q" not in features.columns

    peak_key = "orderflow_scan|orderflow_peak_pressure|friday|0930"
    peak_base = _slug(peak_key)
    peak_gate = f"{peak_base}_gate"
    peak_time_col = f"{peak_base}_time"
    peak_bias_col = f"{peak_base}_bias"

    friday_0930 = (features["weekday"] == "Friday") & (features["clock_time"] == "09:30:00")
    assert features.loc[friday_0930, peak_gate].eq(1).all()
    assert features.loc[friday_0930, peak_time_col].dropna().unique().tolist() == ["09:30:00"]
    assert set(features.loc[friday_0930, peak_bias_col]) == {"sell"}
    assert f"{peak_base}_seasonality_t" not in features.columns
    assert f"{peak_base}_seasonality_q" not in features.columns

    assert features.loc[features["weekday"] == "Monday", "any_pattern_active"].eq(0).all()

    gate_columns = [col for col in features.columns if col.endswith("_gate") and col != "any_pattern_active"]
    assert peak_gate in gate_columns
    assert "other_pattern_gate" not in gate_columns


def test_month_mask_disables_winter_gate_in_summer():
    tz = "America/Chicago"
    timestamps = [
        "2024-01-02 13:30",
        "2024-01-09 13:30",
        "2024-06-04 13:30",
        "2024-06-11 13:30",
    ]

    bars = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
    pipeline = ScreenerPipeline(tz=tz)
    patterns = {
        "winter_weekday": {
            "pattern_type": "weekday_mean",
            "pattern_payload": {"day": "Tuesday", "mean": 0.1},
            "metadata": {"screen_type": "seasonality"},
        }
    }

    features = pipeline.build_features(bars, patterns, allowed_months={12, 1, 2})
    gate_col = f"{_slug('winter_weekday')}_gate"

    january_mask = features["ts"].dt.month == 1
    june_mask = features["ts"].dt.month == 6

    assert features.loc[january_mask, gate_col].eq(1).all()
    assert features.loc[june_mask, gate_col].eq(0).all()
    assert features.loc[january_mask, "_month_allowed"].eq(True).all()
    assert features.loc[june_mask, "_month_allowed"].eq(False).all()


def test_no_months_all_months_active():
    tz = "America/Chicago"
    timestamps = [
        "2024-02-06 13:30",
        "2024-05-07 13:30",
        "2024-08-06 13:30",
    ]

    bars = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
    pipeline = ScreenerPipeline(tz=tz)
    patterns = {
        "weekday_all": {
            "pattern_type": "weekday_mean",
            "pattern_payload": {"day": "Tuesday", "mean": 0.1},
            "metadata": {"screen_type": "seasonality"},
        }
    }

    features = pipeline.build_features(bars, patterns)
    gate_col = f"{_slug('weekday_all')}_gate"

    assert features[gate_col].eq(1).all()
    assert features["_month_allowed"].eq(True).all()


def test_horizon_mapper_accepts_mixed_case_price_columns():
    mapper = HorizonMapper(tz="America/Chicago")
    df = pd.DataFrame(
        {
            "TS": pd.date_range("2023-09-01", periods=2, freq="D", tz="America/Chicago"),
            "Open": [100.0, 101.0],
            "Close": [102.0, 101.5],
            "session_id": ["2023-09-01", "2023-09-02"],
            "example_pattern_gate": [1, 0],
        }
    )
    patterns = {
        "Example Pattern": {
            "pattern_type": "weekday_mean",
            "pattern_payload": {"mean": 0.2},
        }
    }

    result = mapper.build_xy(df, patterns)

    assert not result.empty


def test_horizon_mapper_adds_predictor_columns():
    tz = "America/Chicago"
    ts = pd.date_range("2023-09-05 09:00", periods=3, freq="h")
    bars = pd.DataFrame(
        {
            "ts": ts,
            "open": [99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 121.0],
        }
    )

    patterns = {
        "time_predictive_nextday|10:00:00": {
            "pattern_type": "time_predictive_nextday",
            "pattern_payload": {"time": "10:00:00"},
        }
    }

    pipeline = ScreenerPipeline(tz=tz)
    features = pipeline.build_features(bars, patterns)

    mapper = HorizonMapper(tz=tz)
    enriched = mapper.add_predictor_columns(
        features, patterns, predictor_minutes=60
    )

    gate_col = "time_predictive_nextday_10_00_00_gate"
    signal_col = "time_predictive_nextday_10_00_00_signal"

    assert signal_col in enriched.columns
    assert signal_col not in features.columns

    active_mask = enriched[gate_col] == 1
    assert active_mask.sum() == 1

    expected = np.log(110.0 / 100.0)
    assert np.isclose(enriched.loc[active_mask, signal_col].iloc[0], expected)
    assert enriched.loc[~active_mask, signal_col].isna().all()

