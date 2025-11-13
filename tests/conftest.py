import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

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

if "CTAFlow.config" not in sys.modules:
    config_mod = types.ModuleType("CTAFlow.config")
    app_root = ROOT / "CTAFlow"
    config_mod.APP_ROOT = app_root
    config_mod.RAW_MARKET_DATA_PATH = ROOT
    config_mod.MARKET_DATA_PATH = ROOT
    config_mod.COT_DATA_PATH = ROOT
    config_mod.DLY_DATA_PATH = ROOT
    config_mod.INTRADAY_DATA_PATH = ROOT
    config_mod.RESULTS_HDF_PATH = ROOT / "results.h5"
    config_mod.DAILY_ADB_PATH = "lmdb://"
    config_mod.CURVE_ADB_PATH = "lmdb://"
    config_mod.COT_ADB_PATH = "lmdb://"
    config_mod.INTRADAY_ADB_PATH = "lmdb://"
    config_mod.MODEL_DATA_PATH = ROOT
    config_mod.ENABLE_WEEKLY_UPDATES = False
    config_mod.MONTH_CODE_MAP = {}
    config_mod.FUTURES_MAP = {"COT": {"codes": {}}, "tickers": {}}
    config_mod.TICKER_TO_CODE = {}
    config_mod.CODE_TO_TICKER = {}

    def _get_cot_code(identifier: str) -> str:
        return identifier

    config_mod.get_cot_code = _get_cot_code
    sys.modules["CTAFlow.config"] = config_mod

package_root = ROOT / "CTAFlow"
if "CTAFlow" in sys.modules:
    del sys.modules["CTAFlow"]

spec = importlib.util.spec_from_file_location(
    "CTAFlow",
    package_root / "__init__.py",
    submodule_search_locations=[str(package_root)],
)
cta_module = importlib.util.module_from_spec(spec)
if spec and spec.loader:
    spec.loader.exec_module(cta_module)
sys.modules["CTAFlow"] = cta_module


@pytest.fixture(scope="session")
def example_dataframe() -> pd.DataFrame:
    path = ROOT / "docs" / "example.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df.set_index("Datetime")
