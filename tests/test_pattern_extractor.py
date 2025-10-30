import pandas as pd
import numpy as np
import pytest
from datetime import time
from dataclasses import dataclass
from typing import Any, List, Optional

from CTAFlow.CTAFlow.screeners.orderflow_scan import OrderflowParams
from CTAFlow.CTAFlow.screeners.pattern_extractor import PatternExtractor
import CTAFlow.CTAFlow.screeners.pattern_extractor as pattern_module


@dataclass
class FakeScreenParams:
    screen_type: str
    name: str
    months: Optional[List[int]] = None
    target_times: Optional[List[str]] = None
    period_length: Optional[int] = None
    seasonality_session_start: str = "00:00"
    seasonality_session_end: str = "23:59:59"
    tz: str = "America/Chicago"
    season: Optional[str] = None
    session_starts: Optional[List[str]] = None
    session_ends: Optional[List[str]] = None


if getattr(pattern_module, "ScreenParams", None) is Any:  # pragma: no cover - optional dependency fallback
    pattern_module.ScreenParams = FakeScreenParams


class DummyScreener:
    def __init__(self) -> None:
        self.data = {"ZS": pd.DataFrame()}
        self.synthetic_tickers = {}


class SeasonalityDummyScreener:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = {"GC": data}
        self.synthetic_tickers = {}

    def _convert_times(self, session_starts, session_ends=None):
        def _convert(value):
            if isinstance(value, time):
                return value
            parsed = pd.to_datetime(value).time()
            if parsed.tzinfo is not None:
                parsed = parsed.replace(tzinfo=None)
            return parsed

        start_times = [_convert(value) for value in session_starts]
        if session_ends is None:
            return start_times
        end_times = [_convert(value) for value in session_ends]
        return start_times, end_times

    def _parse_season_months(self, months, season):
        return months

    def _filter_by_months(self, data, months):
        if not months:
            return data
        return data[data.index.month.isin(months)]


def _orderflow_result_fixture(ticker: str = "ZS") -> dict:
    tz = "America/Chicago"
    ts_end = pd.date_range("2023-09-01 08:30", periods=6, freq="1h", tz=tz)
    df_buckets = pd.DataFrame(
        {
            "ticker": [ticker] * len(ts_end),
            "bucket": range(len(ts_end)),
            "ts_start": ts_end - pd.Timedelta(minutes=30),
            "ts_end": ts_end,
            "buy_pressure": np.linspace(-0.2, 0.25, len(ts_end)),
            "sell_pressure": np.linspace(0.1, -0.15, len(ts_end)),
        }
    )
    df_buckets["weekday"] = df_buckets["ts_end"].dt.day_name()
    df_buckets["week_of_month"] = 1
    df_buckets["clock_time"] = df_buckets["ts_end"].dt.time

    df_weekly = pd.DataFrame(
        {
            "weekday": ["Friday"],
            "ticker": [ticker],
            "metric": ["buy_pressure"],
            "mean": [0.12],
            "t_stat": [3.1],
            "p_value": [0.01],
            "q_value": [0.02],
            "sig_fdr_5pct": [True],
            "n": [len(ts_end)],
            "exploratory": [False],
        }
    )

    df_wom_weekday = pd.DataFrame(
        {
            "week_of_month": [1],
            "weekday": ["Friday"],
            "ticker": [ticker],
            "metric": ["buy_pressure"],
            "mean": [0.15],
            "t_stat": [2.8],
            "p_value": [0.02],
            "q_value": [0.03],
            "sig_fdr_5pct": [True],
            "n": [len(ts_end)],
            "exploratory": [False],
        }
    )

    df_weekly_peak_pressure = pd.DataFrame(
        {
            "ticker": [ticker],
            "metric": ["buy_pressure"],
            "weekday": ["Friday"],
            "clock_time": [ts_end[0].time().replace(microsecond=123456)],
            "pressure_bias": ["buy"],
            "seasonality_mean": [0.12],
            "seasonality_t_stat": [3.1],
            "seasonality_q_value": [0.02],
            "seasonality_n": [len(ts_end)],
            "seasonality_sig_fdr_5pct": [True],
            "intraday_mean": [0.2],
            "intraday_median": [0.18],
            "intraday_n": [3],
            "exploratory": [False],
        }
    )

    metadata = {
        "session_start": "08:30",
        "session_end": "15:30",
        "tz": tz,
        "bucket_size": 1000,
        "n_sessions": 40,
        "n_buckets": len(df_buckets),
    }

    return {
        "df_buckets": df_buckets,
        "df_weekly": df_weekly,
        "df_wom_weekday": df_wom_weekday,
        "df_weekly_peak_pressure": df_weekly_peak_pressure,
        "metadata": metadata,
    }


def test_pattern_extractor_handles_orderflow_results():
    results = {"orderflow_scan": {"ZS": _orderflow_result_fixture()}}
    params = [OrderflowParams(session_start="08:30", session_end="15:30", name="orderflow_scan")]
    extractor = PatternExtractor(DummyScreener(), results, params)

    keys = extractor.get_pattern_keys("ZS")
    assert any(key.startswith("orderflow_scan|orderflow_weekly") for key in keys)
    assert any(key.startswith("orderflow_scan|orderflow_week_of_month") for key in keys)
    assert any(key.startswith("orderflow_scan|orderflow_peak_pressure") for key in keys)

    weekly_key = next(k for k in keys if k.startswith("orderflow_scan|orderflow_weekly"))
    weekly_summary = extractor.get_pattern_summary("ZS", weekly_key)
    assert weekly_summary.metadata["screen_type"] == "orderflow"
    weekly_series = extractor.get_pattern_series("ZS", weekly_key)
    assert not weekly_series.empty
    assert weekly_series.name == "buy_pressure"
    assert (weekly_series.index.tz is not None)


def test_pattern_extractor_resolves_missing_seasonality_params():
    index = pd.date_range(
        "2023-01-03 08:00",
        "2023-01-06 16:00",
        freq="5min",
        tz="America/Chicago",
    )
    prices = pd.Series(100 + 0.001 * np.arange(len(index)), index=index)
    data = pd.DataFrame({"Close": prices})
    screener = SeasonalityDummyScreener(data)

    results = {
        "months_12_1_2_3_seasonality": {
            "GC": {
                "strongest_patterns": [
                    {
                        "type": "weekday_returns",
                        "day": "Tuesday",
                        "description": "Tuesday edge",
                        "strength": 2.0,
                    },
                    {
                        "type": "time_predictive_nextday",
                        "time": "10:30:00",
                        "description": "10:30 predicts next day",
                        "strength": 0.5,
                    },
                ],
                "metadata": {
                    "session_start": "08:00",
                    "session_end": "16:00",
                    "tz": "America/Chicago",
                },
            }
        }
    }

    params = [
        FakeScreenParams(
            screen_type="seasonality",
            name="custom_name",
            months=[12, 1, 2, 3],
            target_times=["10:30"],
            period_length=30,
            seasonality_session_start="08:00",
            seasonality_session_end="16:00",
            tz="America/Chicago",
        )
    ]

    extractor = PatternExtractor(screener, results, params)

    keys = extractor.get_pattern_keys("GC")
    weekday_key = next(k for k in keys if "weekday_returns" in k)
    tod_key = next(k for k in keys if "time_predictive_nextday" in k)

    weekday_series = extractor.get_pattern_series("GC", weekday_key)
    assert not weekday_series.empty
    assert all(weekday_series.index.dayofweek == 1)

    tod_series = extractor.get_pattern_series("GC", tod_key)
    assert not tod_series.empty
    assert tod_series.index.tz is None

    assert extractor.get_pattern_summary("GC", weekday_key).screen_params is params[0]
    assert extractor.get_pattern_summary("GC", tod_key).screen_params is params[0]


def test_persist_to_results_writes_hdf_keys(tmp_path):
    from pathlib import Path

    class StubResultsClient:
        def __init__(self, results_path: Path, *, complib: str = "blosc", complevel: int = 9) -> None:
            self.results_path = Path(results_path)
            self.complib = complib
            self.complevel = complevel

        def write_results_df(self, key: str, df: pd.DataFrame, *, replace: bool = True) -> None:
            fmt = dict(format="table", data_columns=True, complib=self.complib, complevel=self.complevel)
            with pd.HDFStore(self.results_path, mode="a") as store:
                writer = store.put if replace else store.append
                writer(key, df, **fmt)

    seasonality_name = "months_9_10_11_seasonality"
    orderflow_name = "us_winter"

    results = {
        seasonality_name: {
            "NG": {
                "strongest_patterns": [
                    {
                        "type": "weekday_returns",
                        "day": "Monday",
                        "description": "Monday edge",
                        "strength": 1.2,
                        "correlation": 0.65,
                        "q_value": 0.015,
                        "t_stat": 2.8,
                        "n": 45,
                    },
                    {
                        "type": "time_predictive_nextday",
                        "time": "10:30:00.500000",
                        "description": "10:30 predicts next day",
                        "strength": 0.75,
                        "correlation": 0.42,
                        "q_value": 0.02,
                        "t_stat": 2.1,
                        "n": 38,
                    },
                ],
                "metadata": {},
            }
        },
        orderflow_name: {
            "RB": _orderflow_result_fixture("RB"),
        },
    }

    params = [
        FakeScreenParams(
            screen_type="seasonality",
            name=seasonality_name,
            months=[9, 10, 11],
            target_times=["10:30"],
            period_length=30,
        ),
        OrderflowParams(session_start="08:30", session_end="15:30", name=orderflow_name),
    ]

    extractor = PatternExtractor(DummyScreener(), results, params)
    results_path = tmp_path / "results.h5"
    client = StubResultsClient(results_path)

    counts = extractor.persist_to_results(client, created_at="2023-10-01T00:00:00Z")

    expected_keys = {
        "results/seasonality/NG/months_9_10_11_seasonality": 2,
        "results/orderflow/RB/us_winter": 3,
    }

    assert counts == expected_keys

    with pd.HDFStore(results_path, "r") as store:
        seasonal_df = store.select("results/seasonality/NG/months_9_10_11_seasonality")
        orderflow_df = store.select("results/orderflow/RB/us_winter")

    assert list(seasonal_df.columns) == list(PatternExtractor.SUMMARY_COLUMNS)
    assert list(orderflow_df.columns) == list(PatternExtractor.SUMMARY_COLUMNS)

    assert set(seasonal_df["pattern_type"]) == {"weekday_returns", "time_predictive_nextday"}
    assert (seasonal_df["created_at"] == "2023-10-01T00:00:00Z").all()
    assert pytest.approx(orderflow_df.loc[orderflow_df["pattern_type"] == "orderflow_peak_pressure", "t_stat"].iloc[0]) == 3.1
    assert "08:30:00.123456" in orderflow_df.loc[
        orderflow_df["pattern_type"] == "orderflow_peak_pressure", "time"
    ].iloc[0]
