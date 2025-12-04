from __future__ import annotations

import asyncio

import sys
import types
from datetime import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "sierrapy" not in sys.modules:
    parser_mod = types.ModuleType("sierrapy.parser")
    parser_mod.ScidReader = object
    parser_mod.AsyncScidReader = object
    parser_mod.bucket_by_volume = lambda *args, **kwargs: None
    parser_mod.resample_ohlcv = lambda *args, **kwargs: None
    sierrapy_mod = types.ModuleType("sierrapy")
    sierrapy_mod.parser = parser_mod
    sys.modules["sierrapy"] = sierrapy_mod
    sys.modules["sierrapy.parser"] = parser_mod

if "numba" not in sys.modules:
    numba_mod = types.ModuleType("numba")

    def _identity_decorator(*_args, **_kwargs):
        def _wrap(func):
            return func

        return _wrap

    numba_mod.jit = _identity_decorator
    numba_mod.njit = _identity_decorator
    numba_mod.guvectorize = _identity_decorator
    numba_mod.vectorize = _identity_decorator
    numba_mod.prange = range  # type: ignore[assignment]

    sys.modules["numba"] = numba_mod

import numpy as np
import pandas as pd
import pytest

from CTAFlow.data.data_client import ResultsClient
from CTAFlow.screeners.orderflow_scan import OrderflowParams
from CTAFlow.screeners.pattern_extractor import (
    PatternExtractor,
    PatternSummary,
    validate_filtered_months,
)
import CTAFlow.screeners.pattern_extractor as pattern_module


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
    st_momentum_days: int = 3
    sess_start_hrs: int = 1
    sess_start_minutes: int = 30
    sess_end_hrs: Optional[int] = None
    sess_end_minutes: Optional[int] = None


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


def _make_summary(
    *,
    key: str,
    symbol: str,
    strength: Optional[float] = None,
    payload: Optional[dict] = None,
    pattern_type: str = "demo",
    source_screen: str = "test",
    description: str = "demo pattern",
    metadata: Optional[dict] = None,
) -> PatternSummary:
    return PatternSummary(
        key=key,
        symbol=symbol,
        source_screen=source_screen,
        screen_params=None,
        pattern_type=pattern_type,
        description=description,
        strength=strength,
        payload=payload or {},
        metadata=dict(metadata or {}),
    )


def _make_extractor(
    patterns: dict,
    *,
    results: Optional[dict] = None,
    screen_params: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> PatternExtractor:
    extractor = object.__new__(PatternExtractor)
    extractor._screener = DummyScreener()
    extractor._results = dict(results or {})
    extractor._screen_params = dict(screen_params or {})
    extractor._pattern_index = {
        symbol: dict(entries) for symbol, entries in patterns.items()
    }
    extractor.metadata = dict(metadata or {})
    extractor._filtered_months = None
    return extractor


def test_pattern_summary_session_params_merges_sources():
    summary = _make_summary(
        key="scan|sess",
        symbol="CL",
        payload={"sess_end": "12:00"},
        metadata={"sess_start_hrs": 1, "sess_start_minutes": 15},
    )

    params = summary.get_session_params()

    assert params["sess_start_hrs"] == 1
    assert params["sess_start_minutes"] == 15
    assert params["sess_end"] == "12:00"
    assert summary.session_params == params


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
            "seasonality_p_value": [0.02],
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


def test_seasonality_pattern_keys_preserved_with_target_time():
    results = {
        "seasonality_scan": {
            "CL": {
                "strongest_patterns": [
                    {
                        "type": "time_predictive_nextday",
                        "time": "07:00",
                        "description": "07:00 predicts next day return",
                        "strength": 0.5,
                        "target_times_hhmm": ["07:00"],
                        "period_length_min": 20,
                    },
                    {
                        "type": "weekend_hedging",
                        "pattern_type": "weekend_hedging",
                        "weekday": "Friday->Monday",
                        "strength": 0.3,
                        "p_value": 0.01,
                    },
                ]
            }
        }
    }

    extractor = PatternExtractor(DummyScreener(), results, [])
    patterns = extractor.patterns["CL"]

    assert "seasonality_scan|time_predictive_nextday|07:00" in patterns
    assert any(
        key.startswith("seasonality_scan|weekend_hedging") for key in patterns
    )

    tod_summary = patterns["seasonality_scan|time_predictive_nextday|07:00"]
    assert tod_summary["metadata"].get("target_times_hhmm") == ["07:00"]


def test_weekend_pattern_with_high_p_value_is_ignored():
    results = {
        "seasonality_scan": {
            "CL": {
                "strongest_patterns": [
                    {
                        "type": "weekend_hedging",
                        "pattern_type": "weekend_hedging",
                        "weekday": "Friday->Monday",
                        "strength": 0.25,
                        "p_value": 0.42,
                    }
                ]
            }
        }
    }

    extractor = PatternExtractor(DummyScreener(), results, [])
    patterns = extractor.patterns.get("CL", {})

    assert "seasonality_scan|weekend_hedging" not in patterns


def test_pattern_metadata_includes_time_months_and_period_labels():
    results = {
        "usa_all": {
            "CL": {
                "filtered_months": [1, 2, 3],
                "strongest_patterns": [
                    {
                        "type": "time_predictive_nextweek",
                        "time": "08:30:00",
                        "description": "08:30 predicts next week",
                        "strength": 0.06,
                        "period_length_min": 120,
                        "months_active": [1, 2, 3],
                        "strongest_days": ["Monday", "Wednesday"],
                    },
                    {
                        "type": "time_predictive_nextday",
                        "time": "09:00:00.500000",
                        "description": "09:00 predicts next day",
                        "strength": 0.04,
                        "period_length_min": 60,
                        "months_active": [1, 2, 3],
                    },
                    {
                        "type": "weekday_returns",
                        "day": "Tuesday",
                        "description": "Tuesday drift",
                        "strength": 0.02,
                        "period_length_min": 120,
                    },
                    {
                        "type": "weekend_hedging",
                        "weekday": "Friday->Monday",
                        "description": "Weekend linkage",
                        "strength": 0.07,
                        "period_length_min": 120,
                        "months_active": [1, 2, 3],
                        "p_value": 0.01,
                    },
                ],
            }
        }
    }

    extractor = PatternExtractor(DummyScreener(), results, [])
    patterns = extractor.patterns["CL"]

    assert len(patterns) == len(results["usa_all"]["CL"]["strongest_patterns"])

    nextweek_key = "usa_all|time_predictive_nextweek|08:30:00"
    assert nextweek_key in patterns
    nextweek = patterns[nextweek_key]
    assert nextweek["key"] == nextweek_key
    assert nextweek["symbol"] == "CL"
    assert nextweek["metadata"]["time"] == "08:30:00"
    assert nextweek["metadata"]["months"] == [1, 2, 3]
    assert nextweek["metadata"]["period_length"] == "2h0m"
    assert nextweek["metadata"]["period_length_min"] == 120
    assert nextweek["metadata"]["strongest_days"] == ["Monday", "Wednesday"]

    micro_key = "usa_all|time_predictive_nextday|09:00:00.500000"
    micro_pattern = patterns[micro_key]
    assert micro_pattern["metadata"]["time"] == "09:00:00.500000"

    weekday_key = "usa_all|weekday_returns|Tuesday"
    weekday_pattern = patterns[weekday_key]
    assert weekday_pattern["metadata"]["weekday"].lower() == "tuesday"

    weekend_key = "usa_all|weekend_hedging|Friday->Monday"
    weekend_pattern = patterns[weekend_key]
    assert weekend_pattern["metadata"]["months"] == [1, 2, 3]


def test_period_length_resolution_prefers_metadata_over_params():
    params = FakeScreenParams(
        screen_type="seasonality",
        name="usa_all",
        target_times=["08:30"],
        period_length=180,
        seasonality_session_start="08:00",
        seasonality_session_end="16:00",
        tz="America/Chicago",
    )

    summary = _make_summary(
        key="usa_all|time_predictive_nextday|08:30:00",
        symbol="CL",
        payload={"time": "08:30:00"},
    )
    summary.metadata["period_length_min"] = 45

    extractor = _make_extractor({}, screen_params={params.name: params})

    resolved = extractor._resolve_period_length(summary, params)
    assert resolved == pd.Timedelta(minutes=45)


def test_pattern_extractor_emits_momentum_weekday_patterns():
    ScreenParams = pattern_module.ScreenParams
    params = ScreenParams(
        screen_type="momentum",
        name="usa_spring_momentum",
        months=[3, 4, 5],
        session_starts=["02:30", "08:30"],
        session_ends=["10:30", "15:00"],
        sess_start_hrs=1,
        sess_start_minutes=30,
        sess_end_hrs=1,
        sess_end_minutes=0,
        tz="America/Chicago",
        period_length=90,
    )

    results = {
        params.name: {
            "CS": {
                "filtered_months": [3, 4, 5],
                "ticker": "CS",
                "momentum_params": {
                    "st_momentum_days": params.st_momentum_days,
                    "period_length_min": params.period_length,
                },
                "session_0": {
                    "session_start": "02:30:00",
                    "session_end": "10:30:00",
                    "momentum_params": {
                        "st_momentum_days": params.st_momentum_days,
                        "period_length_min": params.period_length,
                    },
                    "momentum_by_dayofweek": {
                        "opening_momentum_by_dow": {
                            "Monday": {
                                "n": 30,
                                "mean": 0.012,
                                "std": 0.02,
                                "sharpe": 0.6,
                                "positive_pct": 0.6,
                                "t_stat": 2.5,
                                "skew": 0.1,
                                "months_active": [3, 4, 5],
                                "months_mask_12": "001110000000",
                                "months_names": ["Mar", "Apr", "May"],
                                "p_value_vs_rest": 0.005,
                                "cohen_d_vs_rest": 0.6,
                                "significant_vs_rest": True,
                            },
                            "anova": {
                                "f_stat": 4.1,
                                "p_value": 0.01,
                                "significant": True,
                            },
                        },
                        "summary": {
                            "total_observations": 60,
                            "n_weekdays_analyzed": 5,
                            "significant_patterns": [
                                {
                                    "momentum_type": "opening_momentum",
                                    "f_stat": 4.1,
                                    "p_value": 0.01,
                                }
                            ],
                        },
                    },
                },
            }
        }
    }

    screener = DummyScreener()
    screener.data["CS"] = pd.DataFrame()

    extractor = PatternExtractor(screener, results, [params])
    patterns = extractor.filter_patterns("CS")

    assert patterns, "Expected momentum patterns to be generated"
    key, summary = next(iter(patterns.items()))
    assert "momentum_weekday" in key
    assert summary["pattern_type"] == "momentum_weekday"
    metadata = summary["metadata"]
    assert metadata["weekday"].lower() == "monday"
    assert metadata["momentum_type"] == "opening_momentum"
    assert metadata["session_key"] == "session_0"
    assert metadata["window_anchor"] == "start"
    assert metadata["window_minutes"] == params.sess_start_hrs * 60 + params.sess_start_minutes
    assert metadata["session_start"] == "02:30:00"
    assert metadata["session_end"] == "10:30:00"
    assert metadata["bias"] == "long"
    assert metadata["months"] == [3, 4, 5]
    assert metadata["momentum_params"]["st_momentum_days"] == params.st_momentum_days
    assert metadata["period_length_min"] == pytest.approx(params.period_length)
    assert metadata["sess_start_minutes"] == params.sess_start_minutes
    assert metadata["sess_end_minutes"] == params.sess_end_minutes
    assert summary["strength"] == pytest.approx(2.5, rel=1e-6)


def test_pattern_extractor_extracts_momentum_correlation_and_volatility_patterns():
    ScreenParams = pattern_module.ScreenParams
    params = ScreenParams(
        screen_type="momentum",
        name="usa_momentum_full",
        months=[1, 2, 3],
        session_starts=["08:30"],
        session_ends=["15:00"],
        st_momentum_days=5,
        sess_start_hrs=1,
        sess_start_minutes=0,
        tz="America/Chicago",
        period_length=60,
    )

    session_payload = {
        "session_start": "08:30:00",
        "session_end": "15:00:00",
        "momentum_params": {
            "st_momentum_days": params.st_momentum_days,
            "period_length_min": params.period_length,
        },
                "momentum_by_dayofweek": {
                    "full_session_by_dow": {
                        "Monday": {
                            "mean": -0.001,
                            "t_stat": -2.1,
                            "n": 100,
                            "p_value_vs_rest": 0.004,
                            "cohen_d_vs_rest": -0.4,
                            "significant_vs_rest": True,
                        },
                        "Friday": {
                            "mean": 0.0025,
                            "t_stat": 2.8,
                            "n": 110,
                            "p_value_vs_rest": 0.009,
                            "cohen_d_vs_rest": 0.5,
                            "significant_vs_rest": True,
                        },
                "anova": {
                    "significant": True,
                    "p_value": 0.01,
                    "f_stat": 4.5,
                },
            },
            "summary": {
                "total_observations": 210,
                "significant_patterns": [
                    {
                        "momentum_type": "full_session",
                        "p_value": 0.01,
                        "f_stat": 4.5,
                    }
                ],
            },
        },
        "correlations": {
            "open_close_corr": 0.32,
            "open_close_pvalue": 0.012,
            "close_st_mom_corr": 0.41,
            "close_vs_rest_corr": -0.28,
            "close_vs_rest_pvalue": 0.018,
            "n_observations": 240,
        },
        "volatility": {
            "vol_correlation_significant": True,
            "opening_closing_vol_correlation": 0.58,
            "vol_correlation_pvalue": 1e-08,
            "vol_correlation_interpretation": "Strong positive correlation",
            "n_high_vol_days": 18,
            "n_low_vol_days": 3,
            "overall_stats": {"n_observations": 500},
            "volatility_by_dayofweek": {
                "Monday": {"high_vol_days_count": 10},
                "Tuesday": {"high_vol_days_count": 2},
                "Wednesday": {"high_vol_days_count": 2},
                "Thursday": {"high_vol_days_count": 2},
                "Friday": {"high_vol_days_count": 2},
            },
        },
    }

    results = {
        params.name: {
            "CS": {
                "filtered_months": params.months,
                "ticker": "CS",
                "momentum_params": {
                    "st_momentum_days": params.st_momentum_days,
                    "period_length_min": params.period_length,
                },
                "session_0": session_payload,
            }
        }
    }

    screener = DummyScreener()
    screener.data["CS"] = pd.DataFrame()

    extractor = PatternExtractor(screener, results, [params])
    patterns = extractor.filter_patterns("CS")

    oc_pattern = next(
        summary for summary in patterns.values() if summary["pattern_type"] == "momentum_oc"
    )
    assert oc_pattern["p_value"] < 0.05
    assert oc_pattern["metadata"]["momentum_params"]["st_momentum_days"] == params.st_momentum_days

    cc_pattern = next(
        summary for summary in patterns.values() if summary["pattern_type"] == "momentum_cc"
    )
    assert cc_pattern["p_value"] < 0.05

    sc_pattern = next(
        summary for summary in patterns.values() if summary["pattern_type"] == "momentum_sc"
    )
    assert sc_pattern["p_value"] < 0.05

    ti_pattern = next(
        summary
        for summary in patterns.values()
        if summary["pattern_type"] == "weekday_bias_intraday"
    )
    assert ti_pattern["metadata"]["best_weekday"] == "Friday"
    assert ti_pattern["metadata"]["momentum_params"]["period_length_min"] == pytest.approx(
        params.period_length
    )

    # vol_persistence and volatility_weekday_bias patterns have been removed
    # These patterns are no longer extracted by PatternExtractor
    # vol_pattern = next(
    #     summary for summary in patterns.values() if summary["pattern_type"] == "vol_persistence"
    # )
    # assert vol_pattern["p_value"] < 0.05
    # bias_pattern = next(
    #     summary
    #     for summary in patterns.values()
    #     if summary["pattern_type"] == "volatility_weekday_bias"
    # )
    # assert bias_pattern["metadata"]["weekday"] == "Monday"


def test_pattern_extractor_detects_momentum_without_params():
    results = {
        "usa_intraday_momentum": {
            "CS": {
                "session_0": {
                    "session_start": "08:30:00",
                    "session_end": "15:00:00",
                    "momentum_params": {"st_momentum_days": 3, "period_length_min": 90},
                    "momentum_by_dayofweek": {
                        "full_session_by_dow": {
                            "Monday": {
                                "n": 120,
                                "mean": 0.0015,
                                "t_stat": 2.2,
                                "positive_pct": 0.58,
                                "p_value_vs_rest": 0.01,
                                "cohen_d_vs_rest": 0.4,
                                "significant_vs_rest": True,
                            },
                        }
                    },
                }
            }
        }
    }

    extractor = PatternExtractor(DummyScreener(), results, [])
    patterns = extractor.filter_patterns("CS")

    assert patterns, "Momentum payloads should emit summaries even without params"
    summary = next(
        details for details in patterns.values() if details["pattern_type"] == "momentum_weekday"
    )
    assert summary["metadata"]["momentum_params"]["st_momentum_days"] == 3
    assert summary["metadata"]["session_key"] == "session_0"


def test_pe_promotes_months_from_results_to_metadata():
    results = {
        "seasonality": {
            "CL": {"filtered_months": [1, 2], "strongest_patterns": []},
            "NG": {"filtered_months": [2, 1], "strongest_patterns": []},
        }
    }
    extractor = PatternExtractor(DummyScreener(), results, [])

    assert extractor.metadata.get("filtered_months") == [1, 2]
    assert extractor.get_filtered_months() == {1, 2}


def test_pe_resolves_conflicts_with_union_and_warns(caplog):
    caplog.set_level("WARNING")
    results = {
        "seasonality": {
            "CL": {"filtered_months": [1, 2], "strongest_patterns": []},
            "NG": {"filtered_months": [2, 3], "strongest_patterns": []},
        }
    }

    extractor = PatternExtractor(DummyScreener(), results, [])

    assert extractor.metadata.get("filtered_months") == [1, 2, 3]
    assert {1, 2, 3} == extractor.get_filtered_months()
    assert any("Inconsistent filtered_months" in record.message for record in caplog.records)


def test_pe_concat_merges_months_per_policy():
    base_results = {"screen": {"CL": {"strongest_patterns": []}}}
    left = PatternExtractor(
        DummyScreener(),
        base_results,
        [],
        metadata={"filtered_months": [1, 2]},
    )
    right = PatternExtractor(
        DummyScreener(),
        base_results,
        [],
        metadata={"filtered_months": [2, 3]},
    )

    union = left.concat(right, month_merge_policy="union")
    assert union.metadata.get("filtered_months") == [1, 2, 3]
    assert union.metadata.get("month_merge_policy") == "union"

    intersection = left.concat(right, month_merge_policy="intersect")
    assert intersection.metadata.get("filtered_months") == [2]
    assert intersection.metadata.get("month_merge_policy") == "intersect"


def test_invalid_month_values_are_dropped_with_warning(caplog):
    caplog.set_level("WARNING")
    months = validate_filtered_months(["1", 0, 13, "feb", 2])
    assert months == {1, 2}
    assert any("Discarding invalid filtered_months" in record.message for record in caplog.records)


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
                        "period_length_min": 30,
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


def test_time_of_day_series_prefers_result_period_length(monkeypatch):
    index = pd.date_range(
        "2023-01-03 08:00",
        "2023-01-06 16:00",
        freq="5min",
        tz="America/Chicago",
    )
    prices = pd.Series(100 + 0.001 * np.arange(len(index)), index=index)
    data = pd.DataFrame({"Close": prices})
    screener = SeasonalityDummyScreener(data)

    screen_name = "result_period_length"
    results = {
        screen_name: {
            "GC": {
                "strongest_patterns": [
                    {
                        "type": "time_predictive_nextday",
                        "time": "10:30:00",
                        "description": "10:30 predicts next day",
                        "strength": 0.5,
                        "period_length_min": 45,
                    }
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
            name=screen_name,
            months=[1, 2, 3],
            target_times=["10:30"],
            period_length=15,
            seasonality_session_start="08:00",
            seasonality_session_end="16:00",
            tz="America/Chicago",
        )
    ]

    extractor = PatternExtractor(screener, results, params)

    captured: dict = {}

    def fake_time_of_day(
        session_data, price_col, is_synthetic, target_time_obj, period_length
    ):
        captured["period_length"] = period_length
        return pd.Series(dtype=float)

    monkeypatch.setattr(
        pattern_module.PatternExtractor,
        "_compute_time_of_day_returns",
        staticmethod(fake_time_of_day),
    )

    tod_key = next(
        key
        for key in extractor.get_pattern_keys("GC")
        if "time_predictive_nextday" in key
    )

    extractor.get_pattern_series("GC", tod_key)

    assert captured.get("period_length") == pd.Timedelta(minutes=45)


def test_concat_disjoint_tickers():
    left_summary = _make_summary(
        key="scan|left",
        symbol="CL",
        strength=1.0,
        payload={"t_stat": 2.0, "support": 15},
    )
    right_summary = _make_summary(
        key="scan|right",
        symbol="NG",
        strength=0.5,
        payload={"t_stat": 1.5, "support": 8},
    )
    left = _make_extractor({"CL": {left_summary.key: left_summary}})
    right = _make_extractor({"NG": {right_summary.key: right_summary}})

    merged = left.concat(right)

    assert set(merged._pattern_index) == {"CL", "NG"}
    assert "NG" not in left._pattern_index
    assert merged.get_pattern_summary("NG", right_summary.key) is right_summary


def test_concat_overlap_prefer_strong():
    weaker = _make_summary(
        key="scan|pattern",
        symbol="CL",
        strength=0.6,
        payload={"t_stat": 2.1, "support": 15, "correlation": 0.2},
    )
    stronger = _make_summary(
        key="scan|pattern",
        symbol="CL",
        strength=0.5,
        payload={"t_stat": 4.4, "support": 40, "correlation": 0.25},
    )
    left = _make_extractor({"CL": {weaker.key: weaker}})
    right = _make_extractor({"CL": {stronger.key: stronger}})

    merged = left.concat(right, conflict="prefer_strong")

    assert merged.get_pattern_summary("CL", weaker.key) is stronger


def test_concat_overlap_prefer_left_right():
    left_summary = _make_summary(
        key="scan|pattern",
        symbol="CL",
        strength=0.4,
        payload={"support": 5},
    )
    right_summary = _make_summary(
        key="scan|pattern",
        symbol="CL",
        strength=1.2,
        payload={"support": 10},
    )
    left = _make_extractor({"CL": {left_summary.key: left_summary}})
    right = _make_extractor({"CL": {right_summary.key: right_summary}})

    prefer_left = left.concat(right, conflict="prefer_left")
    prefer_right = left.concat(right, conflict="prefer_right")

    assert prefer_left.get_pattern_summary("CL", left_summary.key) is left_summary
    assert prefer_right.get_pattern_summary("CL", left_summary.key) is right_summary


def test_concat_keep_both_suffixing():
    base = _make_summary(key="scan|pattern", symbol="CL", strength=0.3)
    existing_suffix = _make_summary(key="scan|pattern#2", symbol="CL", strength=0.2)
    newcomer = _make_summary(
        key="scan|pattern",
        symbol="CL",
        strength=0.9,
        payload={"support": 20},
    )
    left = _make_extractor({"CL": {base.key: base, existing_suffix.key: existing_suffix}})
    right = _make_extractor({"CL": {newcomer.key: newcomer}})

    merged = left.concat(right, conflict="keep_both")

    keys = set(merged.get_pattern_keys("CL"))
    assert keys == {"scan|pattern", "scan|pattern#2", "scan|pattern#3"}
    assert merged.get_pattern_summary("CL", "scan|pattern#3") is newcomer


def test_concat_many_chain_associativity():
    s1 = _make_summary(key="scan|p1", symbol="CL", strength=0.5)
    s2 = _make_summary(key="scan|p2", symbol="NG", strength=0.4)
    s3 = _make_summary(key="scan|p3", symbol="CL", strength=0.7)

    a = _make_extractor({"CL": {s1.key: s1}}, results={"r1": {"CL": 1}})
    b = _make_extractor({"NG": {s2.key: s2}})
    c = _make_extractor({"CL": {s3.key: s3}}, results={"r2": {"CL": 2}})

    chained = PatternExtractor.concat_many([a, b, c])
    sequential = a.concat(b).concat(c)

    assert chained.patterns == sequential.patterns


def test_concat_preserves_other_state():
    left_summary = _make_summary(key="scan|left", symbol="CL", strength=0.5)
    right_summary = _make_summary(key="scan|right", symbol="NG", strength=0.9)

    left = _make_extractor(
        {"CL": {left_summary.key: left_summary}},
        results={"left": {"CL": 1}, "shared": {"CL": "left"}},
        screen_params={"left": "params", "shared": "left_param"},
    )
    right = _make_extractor(
        {"NG": {right_summary.key: right_summary}},
        results={"shared": {"CL": "right"}, "new": {"NG": 2}},
        screen_params={"shared": "right_param", "new": "param"},
    )

    merged = left.concat(right, conflict="prefer_right")

    assert merged._results["shared"] == {"CL": "right"}
    assert merged._results["left"] == {"CL": 1}
    assert merged._results["new"] == {"NG": 2}
    assert left._results.get("new") is None

    assert merged._screen_params["shared"] == "right_param"
    assert merged._screen_params["left"] == "params"


def test_rank_handles_missing_fields():
    summary = _make_summary(key="scan|missing", symbol="CL")
    extractor = _make_extractor({"CL": {summary.key: summary}})

    score = PatternExtractor.significance_score(summary)
    assert isinstance(score, float)

    ranked = extractor.rank_patterns()
    assert ranked["CL"] == [(summary.key, summary, score)]


def test_rank_tie_breakers_deterministic():
    low_support = _make_summary(
        key="scan|a",
        symbol="CL",
        payload={"support": 10, "t_stat": 1.0},
    )
    high_support_first = _make_summary(
        key="scan|b",
        symbol="CL",
        payload={"support": 30, "t_stat": 1.5},
    )
    high_support_second = _make_summary(
        key="scan|c",
        symbol="CL",
        payload={"support": 30, "t_stat": 1.5},
    )
    extractor = _make_extractor(
        {
            "CL": {
                low_support.key: low_support,
                high_support_second.key: high_support_second,
                high_support_first.key: high_support_first,
            }
        }
    )

    ranked = extractor.rank_patterns(by="strength")
    ordered_keys = [entry[0] for entry in ranked["CL"]]
    assert ordered_keys == ["scan|b", "scan|c", "scan|a"]


def test_rank_per_ticker_top_n():
    s1 = _make_summary(key="scan|a", symbol="CL", payload={"support": 20, "t_stat": 3.0})
    s2 = _make_summary(key="scan|b", symbol="CL", payload={"support": 10, "t_stat": 2.0})
    s3 = _make_summary(key="scan|c", symbol="NG", payload={"support": 5, "t_stat": 1.0})

    extractor = _make_extractor({"CL": {s1.key: s1, s2.key: s2}, "NG": {s3.key: s3}})

    ranked = extractor.rank_patterns(top=1)

    assert len(ranked["CL"]) == 1
    assert ranked["NG"][0][0] == s3.key


def test_rank_global_sorted_desc():
    s1 = _make_summary(key="scan|a", symbol="CL", payload={"support": 50, "t_stat": 4.0})
    s2 = _make_summary(key="scan|b", symbol="NG", payload={"support": 10, "t_stat": 1.0})

    extractor = _make_extractor({"CL": {s1.key: s1}, "NG": {s2.key: s2}})

    ranked = extractor.rank_patterns(per_ticker=False)
    scores = [entry[3] for entry in ranked]

    assert scores == sorted(scores, reverse=True)
    assert ranked[0][0] == "CL"


def test_rank_patterns_session_filters():
    s1 = _make_summary(
        key="scan|sess1",
        symbol="CL",
        metadata={"sess_start_hrs": 1, "sess_start_minutes": 30},
    )
    s2 = _make_summary(
        key="scan|sess2",
        symbol="CL",
        metadata={"sess_start_hrs": 2, "sess_start_minutes": 0},
    )

    extractor = _make_extractor({"CL": {s1.key: s1, s2.key: s2}})

    ranked = extractor.rank_patterns(session_filters={"sess_start_hrs": 1})

    assert len(ranked["CL"]) == 1
    assert ranked["CL"][0][0] == s1.key


def test_significance_weight_sensitivity():
    weak = _make_summary(key="scan|weak", symbol="CL", payload={"support": 20, "t_stat": 1.0})
    strong = _make_summary(key="scan|strong", symbol="CL", payload={"support": 20, "t_stat": 4.0})

    assert PatternExtractor.significance_score(strong) > PatternExtractor.significance_score(weak)


def test_rank_with_negative_strength_or_corr():
    summary = _make_summary(
        key="scan|neg",
        symbol="CL",
        strength=-2.0,
        payload={"correlation": -0.9, "support": 12, "t_stat": 2.5},
    )
    extractor = _make_extractor({"CL": {summary.key: summary}})

    score = PatternExtractor.significance_score(summary)
    assert 0.0 <= score <= 1.0

    ranked = extractor.rank_patterns()
    assert ranked["CL"][0][0] == summary.key

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
                        "p_value": 0.01,
                        "t_stat": 2.8,
                        "f_stat": 5.2,
                        "n": 45,
                    },
                    {
                        "type": "time_predictive_nextday",
                        "time": "10:30:00.500000",
                        "description": "10:30 predicts next day",
                        "strength": 0.75,
                        "correlation": 0.42,
                        "q_value": 0.02,
                        "p_value": 0.03,
                        "t_stat": 2.1,
                        "f_stat": 4.8,
                        "n": 38,
                    },
                ],
                "metadata": {
                    "session_start": "08:00",
                    "session_end": "16:00",
                    "tz": "America/Chicago",
                    "period_length": "30min",
                    "month_filter": "9,10,11",
                },
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
    assert seasonal_df["period_length"].dropna().unique().tolist() == ["30min"]
    assert seasonal_df["month_filter"].dropna().unique().tolist() == ["9,10,11"]
    assert orderflow_df["period_length"].isna().all()
    assert orderflow_df["month_filter"].isna().all()

    weekday_row = seasonal_df.loc[seasonal_df["pattern_type"] == "weekday_returns"].iloc[0]
    assert pytest.approx(weekday_row["t_stat"]) == 2.8
    assert pytest.approx(weekday_row["p_value"]) == 0.01
    assert pytest.approx(weekday_row["f_stat"]) == 5.2
    assert pytest.approx(weekday_row["q_value"]) == 0.015

    orderflow_peak = orderflow_df.loc[orderflow_df["pattern_type"] == "orderflow_peak_pressure"].iloc[0]
    assert pytest.approx(orderflow_peak["t_stat"]) == 3.1
    assert pytest.approx(orderflow_peak["p_value"]) == 0.02
    assert pd.isna(orderflow_peak["f_stat"])
    assert "08:30:00.123456" in orderflow_peak["time"]


def test_load_summaries_from_results_async(tmp_path):
    scan_type = "seasonality"
    scan_name = "months_9_10_11_seasonality"
    results_path = tmp_path / "results_async.h5"
    client = ResultsClient(results_path)

    base_rows = [
        {
            "pattern_type": "weekday_returns",
            "strength": 1.0,
            "correlation": 0.5,
            "time": "",
            "weekday": "Monday",
            "week_of_month": 1,
            "q_value": 0.01,
            "p_value": 0.02,
            "t_stat": 2.5,
            "f_stat": 4.0,
            "n": 30,
            "source_screen": scan_name,
            "scan_type": scan_type,
            "scan_name": scan_name,
            "description": "Test summary",
            "created_at": "2023-10-01T00:00:00Z",
        }
    ]
    ng_df = pd.DataFrame(base_rows, columns=PatternExtractor.SUMMARY_COLUMNS)
    rb_df = ng_df.copy()
    rb_df["pattern_type"] = "time_predictive_nextday"

    client.write_results_df(client.make_key(scan_type, "NG", scan_name), ng_df, replace=True)
    client.write_results_df(client.make_key(scan_type, "RB", scan_name), rb_df, replace=True)

    loaded = asyncio.run(
        PatternExtractor.load_summaries_from_results_async(
            client,
            scan_type=scan_type,
            scan_name=scan_name,
            tickers=["NG", "rb"],
        )
    )

    assert set(loaded.keys()) == {"NG", "RB"}
    assert list(loaded["NG"].columns) == list(PatternExtractor.SUMMARY_COLUMNS)
    assert list(loaded["RB"].columns) == list(PatternExtractor.SUMMARY_COLUMNS)
    assert loaded["NG"].loc[0, "pattern_type"] == "weekday_returns"
    assert loaded["RB"].loc[0, "pattern_type"] == "time_predictive_nextday"

    with pytest.raises(KeyError):
        asyncio.run(
            PatternExtractor.load_summaries_from_results_async(
                client,
                scan_type=scan_type,
                scan_name=scan_name,
                tickers=["CL"],
            )
        )

    loaded_ignore = asyncio.run(
        PatternExtractor.load_summaries_from_results_async(
            client,
            scan_type=scan_type,
            scan_name=scan_name,
            tickers=["NG", "CL"],
            errors="ignore",
        )
    )
    assert set(loaded_ignore.keys()) == {"NG"}

    sync_loaded = PatternExtractor.load_summaries_from_results(
        client,
        scan_type=scan_type,
        scan_name=scan_name,
        tickers=["NG", "rb"],
    )
    assert set(sync_loaded.keys()) == {"NG", "RB"}
