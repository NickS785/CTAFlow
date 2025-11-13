import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "CTAFlow"
SCREENERS_DIR = PACKAGE_ROOT / "screeners"
def _load_historical_screener():
    # Stub third-party modules used only during optional imports
    if "sierrapy" not in sys.modules:
        sierrapy = types.ModuleType("sierrapy")
        parser_mod = types.ModuleType("sierrapy.parser")
        parser_mod.ScidReader = type("ScidReader", (), {})
        sierrapy.parser = parser_mod
        sys.modules["sierrapy"] = sierrapy
        sys.modules["sierrapy.parser"] = parser_mod

    # Provide minimal CTAFlow package scaffolding for relative imports
    if "CTAFlow" not in sys.modules:
        pkg = types.ModuleType("CTAFlow")
        pkg.__path__ = [str(PACKAGE_ROOT)]
        sys.modules["CTAFlow"] = pkg

    if "CTAFlow.screeners" not in sys.modules:
        screeners_pkg = types.ModuleType("CTAFlow.screeners")
        screeners_pkg.__path__ = [str(SCREENERS_DIR)]
        sys.modules["CTAFlow.screeners"] = screeners_pkg

    if "CTAFlow.utils" not in sys.modules:
        utils_pkg = types.ModuleType("CTAFlow.utils")
        utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
        sys.modules["CTAFlow.utils"] = utils_pkg

    # Stub config/data dependencies required only for full runtime usage
    if "CTAFlow.config" not in sys.modules:
        config_mod = types.ModuleType("CTAFlow.config")
        config_mod.DLY_DATA_PATH = ""
        config_mod.INTRADAY_ADB_PATH = ""
        sys.modules["CTAFlow.config"] = config_mod

    if "CTAFlow.data" not in sys.modules:
        data_mod = types.ModuleType("CTAFlow.data")
        data_mod.IntradayFileManager = type("IntradayFileManager", (), {})
        data_mod.DataClient = type("DataClient", (), {})
        data_mod.SyntheticSymbol = type("SyntheticSymbol", (), {})
        data_mod.ResultsClient = type("ResultsClient", (), {})
        sys.modules["CTAFlow.data"] = data_mod

    spec = importlib.util.spec_from_file_location(
        "CTAFlow.screeners.historical_screener",
        SCREENERS_DIR / "historical_screener.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["CTAFlow.screeners.historical_screener"] = module
    spec.loader.exec_module(module)
    return module.HistoricalScreener


HistoricalScreener = _load_historical_screener()


def test_rank_seasonal_strength_preserves_single_target_time_and_description():
    screener = HistoricalScreener.__new__(HistoricalScreener)

    ticker_results = {
        "pattern_context": {"regime_filter": None},
        "time_predictability": {
            "09:30:00": {
                "time": "09:30",
                "next_day_significant": True,
                "next_day_corr": 0.25,
                "next_day_pvalue": 0.01,
                "next_week_significant": True,
                "next_week_corr": 0.2,
                "next_week_pvalue": 0.02,
                "months_active": [1, 2],
                "months_mask_12": "110000000000",
                "months_names": ["Jan", "Feb"],
                "target_times_hhmm": ["09:30", "10:30"],
                "weekday_prevalence": {
                    "most_prevalent_day": "Monday",
                    "strongest_days": ["Monday"],
                },
            }
        },
    }

    patterns = HistoricalScreener._rank_seasonal_strength(screener, ticker_results)

    assert patterns, "Expected at least one ranked pattern"

    next_day = next(entry for entry in patterns if entry["type"] == "time_predictive_nextday")
    next_week = next(entry for entry in patterns if entry["type"] == "time_predictive_nextweek")

    assert next_day["target_times_hhmm"] == ["09:30"]
    assert next_week["target_times_hhmm"] == ["09:30"]
    assert "09:30" in next_week["description"]
    assert "strongest on Monday" in next_week["description"]
