import sys
import types
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

if "sierrapy" not in sys.modules:
    sierrapy = types.ModuleType("sierrapy")
    parser = types.ModuleType("sierrapy.parser")

    class _DummyScidReader:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def load_front_month_series(self, *args, **kwargs):
            raise NotImplementedError

    parser.ScidReader = _DummyScidReader
    parser.AsyncScidReader = _DummyScidReader
    parser.bucket_by_volume = lambda *args, **kwargs: None
    parser.resample_ohlcv = lambda *args, **kwargs: None
    sierrapy.parser = parser
    sys.modules["sierrapy"] = sierrapy
    sys.modules["sierrapy.parser"] = parser

import tests.conftest  # noqa: F401,E402

if "CTAFlow.features" not in sys.modules:
    sys.modules["CTAFlow.features"] = types.ModuleType("CTAFlow.features")

_event_path = ROOT / "CTAFlow" / "features" / "event_study.py"
spec_event = importlib.util.spec_from_file_location("CTAFlow.features.event_study", _event_path)
event_module = importlib.util.module_from_spec(spec_event)
sys.modules["CTAFlow.features.event_study"] = event_module
spec_event.loader.exec_module(event_module)

_helpers_path = ROOT / "CTAFlow" / "features" / "event_helpers.py"
spec_helpers = importlib.util.spec_from_file_location("CTAFlow.features.event_helpers", _helpers_path)
helpers_module = importlib.util.module_from_spec(spec_helpers)
sys.modules["CTAFlow.features.event_helpers"] = helpers_module
spec_helpers.loader.exec_module(helpers_module)

from CTAFlow.features.event_helpers import baseline_hourly_mean, event_on_percentile
from CTAFlow.features.event_study import (
    EventSpec,
    aar_car,
    aar_hac_tstats,
    bootstrap_cis,
    build_abret,
    event_matrix,
    run_event_study,
)


def _build_df():
    idx = pd.date_range("2024-01-01", periods=200, freq="min")
    df = pd.DataFrame({"ret": np.zeros(len(idx)), "signal": -np.ones(len(idx))}, index=idx)
    event_positions = [50, 100, 150]
    for pos in event_positions:
        df.iloc[pos, df.columns.get_loc("ret")] = 0.01
        df.iloc[pos + 1, df.columns.get_loc("ret")] = 0.005
        df.iloc[pos, df.columns.get_loc("signal")] = 5
    return df, event_positions


def test_event_matrix_and_aar():
    df, event_positions = _build_df()

    events = pd.Series(False, index=df.index)
    events.iloc[event_positions] = True

    abret = build_abret(df, "ret", baseline_hourly_mean(df))
    M = event_matrix(df, events, abret, pre=1, post=2)

    assert M.shape == (4, len(event_positions))
    stats = aar_car(M)
    assert np.isclose(stats["aar"].loc[0], 0.01, atol=1e-3)
    assert np.isclose(stats["car"].loc[2], 0.015, atol=2e-3)

    cis = bootstrap_cis(M, horizons=[0], n_boot=100, seed=1)
    assert 0 in cis
    assert cis[0][0] <= cis[0][1]

    tstats = aar_hac_tstats(M, maxlags=2)
    assert 0 in tstats.index


def test_run_event_study_and_helpers():
    df, _ = _build_df()

    spec = EventSpec(
        name="spike",
        event_fn=lambda frame: event_on_percentile(frame, "signal", p=0.999, min_separation=10),
        pre=1,
        post=2,
        baseline_fn=baseline_hourly_mean,
    )

    result = run_event_study(df, spec)
    assert result["n_events"] == 3
    assert not result["matrix"].empty
    assert np.isclose(result["aar"].loc[0], 0.01, atol=1e-3)


def test_no_events_returns_empty():
    df, _ = _build_df()

    spec = EventSpec(name="none", event_fn=lambda frame: pd.Series(False, index=frame.index), pre=1, post=1)
    result = run_event_study(df, spec)
    assert result["n_events"] == 0
    assert result["matrix"].empty
    assert result["cis"] == {}
