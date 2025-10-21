import pandas as pd
import numpy as np

from CTAFlow.CTAFlow.screeners.orderflow_scan import OrderflowParams
from CTAFlow.CTAFlow.screeners.pattern_extractor import PatternExtractor


class DummyScreener:
    def __init__(self) -> None:
        self.data = {"ZS": pd.DataFrame()}
        self.synthetic_tickers = {}


def _orderflow_result_fixture() -> dict:
    tz = "America/Chicago"
    ts_end = pd.date_range("2023-09-01 08:30", periods=6, freq="1h", tz=tz)
    df_buckets = pd.DataFrame(
        {
            "ticker": ["ZS"] * len(ts_end),
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
            "ticker": ["ZS"],
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
            "ticker": ["ZS"],
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
            "ticker": ["ZS"],
            "metric": ["buy_pressure"],
            "weekday": ["Friday"],
            "clock_time": [ts_end[0].time()],
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

    event_start = ts_end[1]
    event_end = ts_end[3]
    df_events = pd.DataFrame(
        {
            "ticker": ["ZS"],
            "metric": ["buy_pressure"],
            "ts_start": [event_start],
            "ts_end": [event_end],
            "run_len": [3],
            "max_abs_z": [3.6],
            "direction": ["positive"],
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
        "df_events": df_events,
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
    assert any(key.startswith("orderflow_scan|orderflow_event_run") for key in keys)

    weekly_key = next(k for k in keys if k.startswith("orderflow_scan|orderflow_weekly"))
    weekly_summary = extractor.get_pattern_summary("ZS", weekly_key)
    assert weekly_summary.metadata["screen_type"] == "orderflow"
    weekly_series = extractor.get_pattern_series("ZS", weekly_key)
    assert not weekly_series.empty
    assert weekly_series.name == "buy_pressure"
    assert (weekly_series.index.tz is not None)

    event_key = next(k for k in keys if k.startswith("orderflow_scan|orderflow_event_run"))
    event_series = extractor.get_pattern_series("ZS", event_key)
    assert not event_series.empty
    assert event_series.index.min() >= pd.Timestamp("2023-09-01 09:30", tz="America/Chicago")
