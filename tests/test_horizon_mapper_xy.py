from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from datetime import date as date_cls, time as time_cls

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = ROOT / "CTAFlow" / "strategy"
PACKAGE_NAME = "CTAFlow.strategy"

if PACKAGE_NAME not in sys.modules:
    strategy_pkg = types.ModuleType(PACKAGE_NAME)
    strategy_pkg.__path__ = [str(STRATEGY_DIR)]
    sys.modules[PACKAGE_NAME] = strategy_pkg

SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.screener_pipeline",
    STRATEGY_DIR / "screener_pipeline.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[f"{PACKAGE_NAME}.screener_pipeline"] = module
SPEC.loader.exec_module(module)
HorizonMapper = module.HorizonMapper

SESSIONIZER_PATH = ROOT / "CTAFlow" / "strategy" / "sessionizer.py"
SESSIONIZER_NAME = "CTAFlow.strategy.sessionizer"
sessionizer_spec = importlib.util.spec_from_file_location(SESSIONIZER_NAME, SESSIONIZER_PATH)
sessionizer_module = importlib.util.module_from_spec(sessionizer_spec)
assert sessionizer_spec.loader is not None
sys.modules[SESSIONIZER_NAME] = sessionizer_module
sessionizer_spec.loader.exec_module(sessionizer_module)
Sessionizer = sessionizer_module.Sessionizer


def _make_sample_bars() -> pd.DataFrame:
    tz = "America/Chicago"
    session_dates = pd.bdate_range("2024-01-01", periods=8, tz=tz)

    rows: list[dict[str, object]] = []
    for idx, session_start in enumerate(session_dates):
        open_price = 100.0 + idx * 10.0
        closes = [open_price + 1.0, open_price + 2.0, open_price + 3.0]
        times = [
            session_start + pd.Timedelta(hours=6, minutes=55),
            session_start + pd.Timedelta(hours=7),
            session_start + pd.Timedelta(hours=7, minutes=5),
        ]
        for ts, close_price in zip(times, closes, strict=True):
            rows.append(
                {
                    "ts": ts,
                    "open": open_price,
                    "close": close_price,
                    "session_id": idx,
                }
            )

    df = pd.DataFrame(rows)
    decision_mask = df["ts"].dt.time == time_cls(7, 0)
    weekday_lower = df["ts"].dt.tz_convert(tz).dt.day_name().str.lower()
    week_of_month = ((df["ts"].dt.day - 1) // 7) + 1

    df["time_nextday_070000_gate"] = decision_mask.astype(np.int8)
    df["time_nextweek_070000_gate"] = decision_mask.astype(np.int8)
    df["weekday_mean_tuesday_gate"] = (
        decision_mask & (weekday_lower == "tuesday")
    ).astype(np.int8)
    df["oflow_weekly_tuesday_net_pressure_buy_gate"] = (
        decision_mask & (weekday_lower == "tuesday")
    ).astype(np.int8)
    df["oflow_weekly_monday_net_pressure_buy_gate"] = (
        decision_mask & (weekday_lower == "monday")
    ).astype(np.int8)
    df["oflow_wom_tuesday_w1_net_pressure_buy_gate"] = (
        decision_mask & (weekday_lower == "tuesday") & (week_of_month == 1)
    ).astype(np.int8)

    return df


def _session_close(df: pd.DataFrame, session_id: int) -> float:
    session_rows = df[df["session_id"] == session_id]
    return float(session_rows["close"].iloc[-1])


def _make_dst_transition_bars() -> pd.DataFrame:
    tz = "America/Chicago"
    session_dates = pd.bdate_range("2024-03-08", periods=5, tz=tz)

    rows: list[dict[str, object]] = []
    for idx, session_start in enumerate(session_dates):
        open_price = 150.0 + idx
        closes = [open_price + 0.5, open_price + 1.0, open_price + 2.0]
        times = [
            session_start + pd.Timedelta(hours=6, minutes=55),
            session_start + pd.Timedelta(hours=7),
            session_start + pd.Timedelta(hours=7, minutes=5),
        ]
        for ts, close_price in zip(times, closes, strict=True):
            rows.append(
                {
                    "ts": ts,
                    "open": open_price,
                    "close": close_price,
                }
            )

    df = pd.DataFrame(rows)
    df["time_nextday_070000_gate"] = (df["ts"].dt.time == time_cls(7, 0)).astype(np.int8)
    return df


def test_time_predictive_nextday_returns_x_and_y():
    df = _make_sample_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = [
        {
            "pattern_type": "time_predictive_nextday",
            "pattern_payload": {"time": "07:00"},
            "key": "time_nextday_070000",
        }
    ]

    result = mapper.build_xy(df, patterns, predictor_minutes=5, ensure_gates=False)

    # All sessions except the last have a next-day close
    assert len(result) == len(df["session_id"].unique()) - 1

    # returns_x uses the 5-minute window starting at the decision bar
    first_session = df[df["session_id"] == 0]
    decision_close = float(first_session.loc[first_session["ts"].dt.time == time_cls(7, 0), "close"].iloc[0])
    future_close = float(first_session.loc[first_session["ts"].dt.time == time_cls(7, 5), "close"].iloc[0])
    expected_x = np.log(future_close / decision_close)
    assert result.loc[0, "returns_x"] == pytest.approx(expected_x)

    # returns_y uses next-day close→close
    expected_y = np.log(_session_close(df, 1) / _session_close(df, 0))
    assert result.loc[0, "returns_y"] == pytest.approx(expected_y)


def test_time_predictive_nextweek_returns():
    df = _make_sample_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = [
        {
            "pattern_type": "time_predictive_nextweek",
            "pattern_payload": {"time": "07:00"},
            "key": "time_nextweek_070000",
        }
    ]

    result = mapper.build_xy(df, patterns, predictor_minutes=5, ensure_gates=False)

    assert len(result) == 3

    first_session = df[df["session_id"] == 0]
    decision_close = float(first_session.loc[first_session["ts"].dt.time == time_cls(7, 0), "close"].iloc[0])
    future_close = float(first_session.loc[first_session["ts"].dt.time == time_cls(7, 5), "close"].iloc[0])
    expected_x = np.log(future_close / decision_close)
    assert result.loc[0, "returns_x"] == pytest.approx(expected_x)

    expected_y = np.log(_session_close(df, 5) / _session_close(df, 0))
    assert result.loc[0, "returns_y"] == pytest.approx(expected_y)


def test_orderflow_peak_pressure_forward_returns():
    df = _make_sample_bars()
    peak_mask = (
        (df["ts"].dt.day_name().str.lower() == "tuesday")
        & (df["ts"].dt.time == time_cls(7, 0))
    )
    df["peak_tuesday_gate"] = peak_mask.astype(np.int8)

    patterns = [
        {
            "pattern_type": "orderflow_peak_pressure",
            "pattern_payload": {
                "weekday": "tuesday",
                "clock_time": "07:00",
                "metric": "net_pressure",
                "pressure_bias": "buy",
            },
            "metadata": {"orderflow_bias": "buy"},
            "key": "peak_tuesday",
        }
    ]

    mapper = HorizonMapper(tz="America/Chicago")
    result = mapper.build_xy(
        df,
        patterns,
        predictor_minutes=5,
        default_intraday_minutes=5,
        ensure_gates=False,
    )

    assert len(result) == 2

    decision_ts = result.loc[0, "ts_decision"]
    decision_close = float(df.loc[df["ts"] == decision_ts, "close"].iloc[0])
    future_close = float(
        df.loc[df["ts"] == decision_ts + pd.Timedelta(minutes=5), "close"].iloc[0]
    )
    expected = np.log(future_close / decision_close)

    assert result.loc[0, "returns_x"] == pytest.approx(expected)
    assert result.loc[0, "returns_y"] == pytest.approx(expected)


def test_weekly_mean_policy_uses_payload_value():
    df = _make_sample_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = [
        {
            "pattern_type": "weekday_mean",
            "pattern_payload": {"day": "tuesday", "mean": 0.0123},
            "key": "weekday_mean_tuesday",
        }
    ]

    result = mapper.build_xy(df, patterns, ensure_gates=False)

    assert not result.empty
    assert result["returns_x"].tolist() == pytest.approx([0.0123] * len(result))


def test_weekly_mean_policy_falls_back_to_historical_mean():
    df = _make_sample_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = [
        {
            "pattern_type": "orderflow_week_of_month",
            "pattern_payload": {
                "weekday": "tuesday",
                "week_of_month": 1,
                "metric": "net_pressure",
                "pressure_bias": "buy",
            },
            "key": "oflow_wom_tuesday_w1_net_pressure_buy",
        }
    ]

    result = mapper.build_xy(df, patterns, ensure_gates=False)

    session_one = df[df["session_id"] == 1]
    session_open = float(session_one["open"].iloc[0])
    expected_mean = np.log(_session_close(df, 1) / session_open)
    assert result["returns_x"].iloc[0] == pytest.approx(expected_mean)


def test_weekly_prev_week_policy_uses_realised_return():
    df = _make_sample_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = [
        {
            "pattern_type": "orderflow_weekly",
            "pattern_payload": {
                "weekday": "tuesday",
                "metric": "net_pressure",
                "pressure_bias": "buy",
            },
            "key": "oflow_weekly_tuesday_net_pressure_buy",
        }
    ]

    result = mapper.build_xy(df, patterns, weekly_x_policy="prev_week", ensure_gates=False)

    # First Tuesday lacks a prior-week observation and should be excluded
    assert len(result) == 1

    expected = np.log(_session_close(df, 6) / _session_close(df, 1))
    assert result.loc[0, "returns_x"] == pytest.approx(expected)

    # Ensure the surviving row corresponds to the second Tuesday (session 6)
    surviving_session = df[df["ts"] == result.loc[0, "ts_decision"]]["session_id"].iloc[0]
    assert surviving_session == 6


def test_sessionizer_handles_dst_transition_without_time_shift():
    df = _make_dst_transition_bars()
    sessionizer = Sessionizer()
    mapper = HorizonMapper(tz="America/Chicago", sessionizer=sessionizer)
    patterns = [
        {
            "pattern_type": "time_predictive_nextday",
            "pattern_payload": {"time": "07:00"},
            "key": "time_nextday_070000",
        }
    ]

    result = mapper.build_xy(df, patterns, predictor_minutes=5, ensure_gates=False)

    expected_rows = df["time_nextday_070000_gate"].sum() - 1
    assert len(result) == expected_rows

    first_decision = result.iloc[0]["ts_decision"]
    decision_close = float(df.loc[df["ts"] == first_decision, "close"].iloc[0])
    future_close = float(
        df.loc[df["ts"] == first_decision + pd.Timedelta(minutes=5), "close"].iloc[0]
    )
    assert result.iloc[0]["returns_x"] == pytest.approx(np.log(future_close / decision_close))

    dst_date = date_cls(2024, 3, 11)
    assert dst_date in set(result["ts_decision"].dt.date)


def test_forward_predictor_trims_tail_without_future_data():
    df = _make_sample_bars()
    session_to_trim = df["session_id"].max() - 1
    mask = (df["session_id"] == session_to_trim) & (df["ts"].dt.time == time_cls(7, 5))
    df = df.loc[~mask].copy()

    mapper = HorizonMapper(tz="America/Chicago")
    patterns = [
        {
            "pattern_type": "time_predictive_nextday",
            "pattern_payload": {"time": "07:00"},
            "key": "time_nextday_070000",
        }
    ]

    result = mapper.build_xy(df, patterns, predictor_minutes=5, ensure_gates=False)

    expected_rows = len(df["session_id"].unique()) - 2
    assert len(result) == expected_rows
    trimmed_date = df.loc[df["session_id"] == session_to_trim, "ts"].dt.date.iloc[0]
    assert trimmed_date not in set(result["ts_decision"].dt.date)


def _make_month_mask_bars() -> pd.DataFrame:
    tz = "America/Chicago"
    sessions = [
        pd.Timestamp("2024-01-31 07:00", tz=tz),
        pd.Timestamp("2024-02-01 07:00", tz=tz),
        pd.Timestamp("2024-02-02 07:00", tz=tz),
        pd.Timestamp("2024-02-05 07:00", tz=tz),
    ]

    rows: list[dict[str, object]] = []
    for idx, decision_ts in enumerate(sessions):
        open_price = 100.0 + idx
        close_price = open_price + 1.0
        rows.append(
            {
                "ts": decision_ts - pd.Timedelta(minutes=5),
                "open": open_price,
                "close": open_price + 0.5,
                "session_id": idx,
                "time_nextday_070000_gate": 0,
            }
        )
        rows.append(
            {
                "ts": decision_ts,
                "open": open_price,
                "close": close_price,
                "session_id": idx,
                "time_nextday_070000_gate": 1,
            }
        )
        rows.append(
            {
                "ts": decision_ts + pd.Timedelta(minutes=5),
                "open": open_price,
                "close": close_price + 0.5,
                "session_id": idx,
                "time_nextday_070000_gate": 0,
            }
        )

    return pd.DataFrame(rows)


def test_build_xy_with_allowed_months_filters_decisions():
    df = _make_month_mask_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = [
        {
            "pattern_type": "time_predictive_nextday",
            "pattern_payload": {"time": "07:00"},
            "key": "time_nextday_070000",
        }
    ]

    result = mapper.build_xy(
        df,
        patterns,
        predictor_minutes=5,
        ensure_gates=False,
        allowed_months={2},
    )

    assert not result.empty
    assert set(result["ts_decision"].dt.month) == {2}
