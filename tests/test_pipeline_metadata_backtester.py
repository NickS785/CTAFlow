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
