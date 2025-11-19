from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = ROOT / "CTAFlow" / "strategy"
PACKAGE_NAME = "CTAFlow.strategy"

if PACKAGE_NAME not in sys.modules:
    strategy_pkg = types.ModuleType(PACKAGE_NAME)
    strategy_pkg.__path__ = [str(STRATEGY_DIR)]
    sys.modules[PACKAGE_NAME] = strategy_pkg


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, STRATEGY_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backtester_module = _load_module(f"{PACKAGE_NAME}.backtester", "backtester.py")
pipeline_module = _load_module(f"{PACKAGE_NAME}.screener_pipeline", "screener_pipeline.py")

ScreenerPipeline = pipeline_module.ScreenerPipeline
HorizonMapper = pipeline_module.HorizonMapper
BacktestSummary = backtester_module.BacktestSummary


def _make_simple_bars() -> pd.DataFrame:
    tz = "America/Chicago"
    ts = pd.to_datetime(["2024-01-01 07:00", "2024-01-02 07:00"]).tz_localize(tz)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": [100.0, 105.0],
            "close": [101.0, 103.0],
            "session_id": ["2024-01-01", "2024-01-02"],
            "weekday_mean_monday_gate": [1, 0],
        }
    )


def _weekday_pattern() -> dict[str, object]:
    return {
        "pattern_type": "weekday_mean",
        "pattern_payload": {
            "day": "Monday",
            "mean": 0.2,
        },
        "metadata": {
            "months_active": [1, 2],
            "months_mask_12": "110000000000",
            "target_times_hhmm": ["07:00"],
        },
    }


def _multirow_bars() -> pd.DataFrame:
    tz = "America/Chicago"
    ts = pd.to_datetime(
        [
            "2024-01-01 07:00",
            "2024-01-01 12:00",
            "2024-01-02 07:00",
            "2024-01-02 12:00",
        ]
    ).tz_localize(tz)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": [100.0, 101.0, 102.0, 103.0],
            "close": [101.0, 101.5, 103.0, 103.5],
            "session_id": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
        }
    )


def _orderflow_weekly_pattern() -> dict[str, object]:
    return {
        "pattern_type": "orderflow_weekly",
        "pattern_payload": {
            "weekday": "monday",
            "metric": "net_pressure",
            "pressure_bias": "buy",
            "mean": 0.05,
            "n": 25,
        },
        "metadata": {"orderflow_bias": "buy"},
    }


def _orderflow_wom_pattern() -> dict[str, object]:
    return {
        "pattern_type": "orderflow_week_of_month",
        "pattern_payload": {
            "weekday": "monday",
            "week_of_month": 1,
            "metric": "net_pressure",
            "pressure_bias": "buy",
            "mean": 0.07,
            "n": 15,
        },
        "metadata": {"orderflow_bias": "buy"},
    }


def test_build_xy_preserves_list_metadata():
    bars = _make_simple_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = {"weekday_mean_monday": _weekday_pattern()}

    result = mapper.build_xy(
        bars,
        patterns,
        ensure_gates=False,
        include_metadata=["months_active", "months_mask_12", "target_times_hhmm"],
    )

    assert not result.empty
    assert list(result.columns)[:3] == ["ts_decision", "gate", "pattern_type"]
    assert result.loc[0, "gate"] == "weekday_mean_monday_gate"
    assert result.loc[0, "months_active"] == [1, 2]
    assert result.loc[0, "months_mask_12"] == "110000000000"
    assert result.loc[0, "target_times_hhmm"] == ["07:00"]


def test_backtest_threshold_runs_on_pipeline_output():
    bars = _make_simple_bars()
    mapper = HorizonMapper(tz="America/Chicago")
    patterns = {"weekday_mean_monday": _weekday_pattern()}

    xy = mapper.build_xy(bars, patterns, ensure_gates=False)
    assert not xy.empty

    pipeline = ScreenerPipeline(tz="America/Chicago")
    outcome = pipeline.backtest_threshold(
        bars,
        patterns,
        threshold=0.05,
        ensure_gates=False,
    )

    summary = outcome["summary"]
    assert isinstance(summary, BacktestSummary)
    assert summary.trades == 1

    expected_return = xy.loc[0, "returns_y"]
    assert outcome["pnl"].iloc[0] == pytest.approx(expected_return)
    assert outcome["positions"].iloc[0] == pytest.approx(1.0)


def test_backtester_group_breakdown():
    tz = "UTC"
    xy = pd.DataFrame(
        {
            "ts_decision": pd.date_range("2024-01-01", periods=2, tz=tz),
            "gate": ["g1", "g2"],
            "pattern_type": ["momentum_weekday", "momentum_weekday"],
            "returns_x": [0.2, -0.3],
            "returns_y": [0.15, 0.1],
            "side_hint": [1, 0],
            "momentum_type": ["opening", "closing"],
        }
    )

    backtester = backtester_module.ScreenerBacktester()
    result = backtester.threshold(xy, group_field="momentum_type")

    breakdown = result.get("group_breakdown")
    assert breakdown is not None
    assert set(breakdown.keys()) == {"opening", "closing"}
    assert breakdown["opening"]["trades"] == 1
    assert breakdown["closing"]["total_return"] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "pattern_key, pattern_factory, gate_column",
    [
        ("weekday_mean_monday", _weekday_pattern, "weekday_mean_monday_gate"),
        (
            "oflow_weekly_monday_net_pressure_buy",
            _orderflow_weekly_pattern,
            "oflow_weekly_monday_net_pressure_buy_gate",
        ),
        (
            "oflow_wom_monday_w1_net_pressure_buy",
            _orderflow_wom_pattern,
            "oflow_wom_monday_w1_net_pressure_buy_gate",
        ),
    ],
)
def test_session_level_gates_anchor_to_close(pattern_key, pattern_factory, gate_column):
    bars = _multirow_bars()
    pipeline = ScreenerPipeline(tz="America/Chicago")
    patterns = {pattern_key: pattern_factory()}

    featured = pipeline.build_features(bars, patterns)
    assert gate_column in featured.columns

    gate_series = featured[gate_column]
    assert gate_series.sum() == 1

    monday_rows = featured["session_id"] == "2024-01-01"
    last_idx = featured.loc[monday_rows, "ts"].idxmax()
    assert gate_series.loc[last_idx] == 1
    assert gate_series.loc[monday_rows].sum() == 1


def test_backtester_collapses_duplicate_decisions():
    ts = pd.Timestamp("2024-01-01 14:00", tz="UTC")
    xy = pd.DataFrame(
        {
            "ts_decision": [ts, ts],
            "gate": ["g1", "g2"],
            "pattern_type": ["momentum_weekday", "momentum_weekday"],
            "returns_x": [0.2, -0.4],
            "returns_y": [0.1, 0.1],
            "correlation": [0.5, 0.9],
            "side_hint": [1, -1],
        }
    )

    backtester = backtester_module.ScreenerBacktester()
    result = backtester.threshold(xy)

    pnl = result["pnl"]
    positions = result["positions"]

    assert len(pnl) == 1
    assert len(positions) == 1
    assert positions.iloc[0] == -1
    assert pnl.iloc[0] == pytest.approx(-0.1)
    assert result["summary"].trades == 1


def test_backtester_counts_unique_trade_days_per_group():
    tz = "UTC"
    xy = pd.DataFrame(
        {
            "ts_decision": [
                pd.Timestamp("2024-01-01 09:00", tz=tz),
                pd.Timestamp("2024-01-01 15:00", tz=tz),
                pd.Timestamp("2024-01-02 09:30", tz=tz),
                pd.Timestamp("2024-01-01 10:00", tz=tz),
                pd.Timestamp("2024-01-01 14:00", tz=tz),
                pd.Timestamp("2024-01-02 11:00", tz=tz),
            ],
            "returns_x": [0.2, 0.05, -0.25, 0.3, -0.15, -0.2],
            "returns_y": [0.1, 0.06, -0.08, 0.09, -0.05, -0.03],
            "ticker": ["ZC_F", "ZC_F", "ZC_F", "ZW_F", "ZW_F", "ZW_F"],
        }
    )

    backtester = backtester_module.ScreenerBacktester()
    result = backtester.threshold(xy, group_field="ticker")

    summary = result["summary"]
    assert summary.trades == 4  # two tickers traded across two unique days

    breakdown = result["group_breakdown"]
    assert breakdown["ZC_F"]["trades"] == 2
    assert breakdown["ZW_F"]["trades"] == 2
