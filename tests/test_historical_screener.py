import sys
import types
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def stub_optional_dependencies():
    """Provide lightweight stand-ins for optional extensions used in tests."""
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

    # numba is optional in the CI environment; provide no-op decorators when absent.
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

    if "CTAFlow.config" not in sys.modules:
        config_mod = types.ModuleType("CTAFlow.config")
        config_mod.RAW_MARKET_DATA_PATH = Path(".")
        config_mod.FUTURES_MAP = {"COT": {"codes": {}}, "tickers": {}}
        config_mod.TICKER_TO_CODE = {}
        config_mod.CODE_TO_TICKER = {}
        config_mod.COT_DATA_PATH = Path(".")
        config_mod.DLY_DATA_PATH = Path(".")
        config_mod.MARKET_DATA_PATH = Path(".")
        config_mod.DAILY_ADB_PATH = "lmdb://"
        config_mod.CURVE_ADB_PATH = "lmdb://"
        config_mod.COT_ADB_PATH = "lmdb://"
        config_mod.INTRADAY_ADB_PATH = "lmdb://"
        config_mod.INTRADAY_DATA_PATH = Path(".")
        config_mod.RESULTS_HDF_PATH = Path("./results.h5")
        config_mod.ENABLE_WEEKLY_UPDATES = False
        config_mod.MONTH_CODE_MAP = {}

        def _get_cot_code(identifier: str) -> str:
            return identifier

        config_mod.get_cot_code = _get_cot_code
        sys.modules["CTAFlow.config"] = config_mod

    try:
        import CTAFlow  # type: ignore[import-not-found]
    except Exception:
        cta_module = types.ModuleType("CTAFlow")
        cta_module.__path__ = [str(Path("CTAFlow"))]
        sys.modules["CTAFlow"] = cta_module

    for subpkg in ("screeners", "utils", "data"):
        module_name = f"CTAFlow.{subpkg}"
        if module_name in sys.modules:
            continue
        try:
            __import__(module_name)
        except Exception:
            pkg = types.ModuleType(module_name)
            pkg.__path__ = [str(Path("CTAFlow") / subpkg)]
            sys.modules[module_name] = pkg

    data_module = sys.modules.get("CTAFlow.data")
    if data_module is not None:
        if not hasattr(data_module, "DataClient") or not hasattr(data_module, "ResultsClient"):
            from CTAFlow.data.data_client import DataClient, ResultsClient  # type: ignore

            data_module.DataClient = DataClient  # type: ignore[attr-defined]
            data_module.ResultsClient = ResultsClient  # type: ignore[attr-defined]

        if not hasattr(data_module, "SyntheticSymbol"):
            from CTAFlow.data.raw_formatting.synthetic import SyntheticSymbol  # type: ignore

            data_module.SyntheticSymbol = SyntheticSymbol  # type: ignore[attr-defined]

        if not hasattr(data_module, "IntradayFileManager"):
            from CTAFlow.data.raw_formatting.intraday_manager import IntradayFileManager  # type: ignore

            data_module.IntradayFileManager = IntradayFileManager  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def example_dataframe() -> pd.DataFrame:
    path = Path("docs/example.csv")
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df.set_index("Datetime")


@pytest.fixture()
def historical_screener(example_dataframe):
    from CTAFlow.screeners.historical_screener import HistoricalScreener

    return HistoricalScreener({"HO": example_dataframe}, file_mgr=None, verbose=False)


def test_run_screens_returns_patterns_structure(historical_screener):
    from CTAFlow.screeners.historical_screener import ScreenParams

    params = ScreenParams(
        screen_type="seasonality",
        name="example_screen",
        target_times=["00:30", "00:45"],
        season="fall",
        seasonality_session_end="23:59",
    )

    results = historical_screener.run_screens([params])

    assert "example_screen" in results
    screen_payload = results["example_screen"]
    assert "HO" in screen_payload

    ticker_payload = screen_payload["HO"]
    assert "error" not in ticker_payload
    assert isinstance(ticker_payload.get("strongest_patterns"), list)
    assert "time_predictability" in ticker_payload

    # Each time bucket should expose context needed by PatternExtractor
    for time_payload in ticker_payload["time_predictability"].values():
        assert "target_times_hhmm" in time_payload
        assert "period_length_min" in time_payload


def test_rank_seasonal_strength_includes_weekday_context(historical_screener):
    months_meta = {
        "months_active": [9, 10, 11],
        "months_mask_12": "000000001110",
        "months_names": ["Sep", "Oct", "Nov"],
    }

    ticker_results = {
        "pattern_context": {
            **months_meta,
            "period_length_min": 120,
            "target_times_hhmm": ["09:30"],
            "regime_filter": None,
        },
        "months_active": months_meta["months_active"],
        "months_mask_12": months_meta["months_mask_12"],
        "months_names": months_meta["months_names"],
        "time_predictability": {
            "09:30": {
                "time": "09:30",
                "next_week_significant": True,
                "next_week_corr": 0.42,
                "next_week_pvalue": 0.01,
                "next_day_significant": False,
                "months_active": months_meta["months_active"],
                "months_mask_12": months_meta["months_mask_12"],
                "months_names": months_meta["months_names"],
                "target_times_hhmm": ["09:30"],
                "period_length_min": 120,
                "regime_filter": None,
                "weekday_prevalence": {
                    "most_prevalent_day": "Friday",
                    "strongest_days": ["Friday", "Monday"],
                },
            }
        },
    }

    patterns = historical_screener._rank_seasonal_strength(ticker_results)
    time_patterns = [p for p in patterns if p.get("type") == "time_predictive_nextweek"]

    assert time_patterns, "expected at least one time predictive pattern"
    pattern = time_patterns[0]

    assert "09:30" in pattern["description"]
    assert pattern.get("most_prevalent_day") == "Friday"
    assert pattern.get("strongest_days") == ["Friday", "Monday"]


def test_weekend_hedging_requires_significant_p_value(historical_screener):
    screener = historical_screener
    tz = "UTC"

    friday_returns = np.array([0.01, 0.02, 0.03, -0.02, 0.01, -0.015])
    monday_returns = np.array([0.02, -0.01, 0.015, 0.005, -0.02, 0.01])

    base_friday = pd.Timestamp("2024-01-05 09:00", tz=tz)
    rows: list[dict[str, object]] = []

    for idx, (fri_ret, mon_ret) in enumerate(zip(friday_returns, monday_returns, strict=True)):
        friday_start = base_friday + pd.Timedelta(weeks=idx)
        friday_end = friday_start + pd.Timedelta(hours=5)
        monday_start = friday_start + pd.Timedelta(days=3)
        monday_end = monday_start + pd.Timedelta(hours=5)

        friday_open = 100.0 + idx
        friday_close = friday_open * float(np.exp(fri_ret))
        monday_open = 110.0 + idx
        monday_close = monday_open * float(np.exp(mon_ret))

        rows.append({"ts": friday_start, "Close": friday_open})
        rows.append({"ts": friday_end, "Close": friday_close})
        rows.append({"ts": monday_start, "Close": monday_open})
        rows.append({"ts": monday_end, "Close": monday_close})

    session_data = pd.DataFrame(rows).set_index("ts")

    pattern = screener._compute_weekend_hedging_pattern(
        session_data,
        session_start=time(9, 0),
        session_end=time(14, 0),
        price_col="Close",
        is_synthetic=False,
        tz=tz,
        pattern_context={},
    )

    assert pattern is None


def test_parse_params_resolves_duplicate_regime_kwargs(historical_screener, monkeypatch):
    from CTAFlow.screeners.historical_screener import ScreenParams

    params = ScreenParams(
        screen_type="momentum",
        name="momentum_test",
        session_starts=["02:30"],
        session_ends=["10:30"],
        use_regime_filtering=True,
        regime_col="regime_state",
        target_regimes=[1, 2],
    )

    captured_kwargs: dict[str, object] = {}

    def _fake_momentum_screen(**kwargs):
        captured_kwargs.update(kwargs)
        return {"HO": {"pattern": "mock"}}

    monkeypatch.setattr(historical_screener, "intraday_momentum_screen", _fake_momentum_screen)

    result = historical_screener._parse_params(
        params,
        session_starts=params.session_starts,
        session_ends=params.session_ends,
        _selected_months=[3, 4, 5],
        _precomputed_sessions={"HO": {}},
        use_regime_filtering=True,
        regime_col="regime_state",
        target_regimes=[1, 2],
    )

    assert "HO" in result
    assert captured_kwargs["use_regime_filtering"] is True
    assert captured_kwargs["regime_col"] == "regime_state"
    assert captured_kwargs["target_regimes"] == [1, 2]


def test_intraday_momentum_screen_includes_analysis_params(historical_screener):
    screener = historical_screener
    results = screener.intraday_momentum_screen(
        session_starts=["00:00"],
        session_ends=["23:59"],
        st_momentum_days=4,
        period_length=30,
        max_workers=1,
        show_progress=False,
    )

    ticker_payload = results["HO"]
    assert ticker_payload["momentum_params"]["st_momentum_days"] == 4
    assert ticker_payload["momentum_params"]["opening_window_minutes"] == pytest.approx(30.0)
    assert ticker_payload["momentum_params"]["closing_window_minutes"] == pytest.approx(30.0)
    session_payload = ticker_payload.get("session_0")
    assert session_payload is not None
    assert session_payload["momentum_params"]["st_momentum_days"] == 4
    assert session_payload["momentum_params"]["period_length_min"] == pytest.approx(30.0)
    assert session_payload["momentum_params"]["opening_window_minutes"] == pytest.approx(30.0)
    assert session_payload["momentum_params"]["closing_window_minutes"] == pytest.approx(30.0)
    assert "sess_start_window_minutes" not in session_payload["momentum_params"]
    assert session_payload["opening_window_minutes"] == pytest.approx(30.0)
    assert session_payload["closing_window_minutes"] == pytest.approx(30.0)
