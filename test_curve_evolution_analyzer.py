import numpy as np
import pandas as pd
import importlib.util
import types
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal stubs to satisfy module imports without heavy dependencies
# ---------------------------------------------------------------------------
stub = types.ModuleType("CTAFlow.data.contract_handling.curve_manager")


class FuturesCurve:
    def __init__(self, ref_date=None, curve_month_labels=None, prices=None):
        self.ref_date = ref_date
        self.curve_month_labels = curve_month_labels or []
        self.prices = prices
        self.seq_prices = prices
        self.seq_labels = curve_month_labels or []


class SpreadData:  # pragma: no cover - minimal placeholder
    pass


class SpreadFeature:  # pragma: no cover - minimal placeholder
    pass


def _is_empty(data):  # minimal helper used by analyzer
    try:
        return len(data) == 0
    except Exception:
        return data is None


stub.FuturesCurve = FuturesCurve
stub.SpreadData = SpreadData
stub.SpreadFeature = SpreadFeature
stub._is_empty = _is_empty
stub.MONTH_CODE_MAP = {c: i + 1 for i, c in enumerate('FGHJKMNQUVXZ')}

cta_pkg = types.ModuleType("CTAFlow")
cta_pkg.__path__ = [str(Path(__file__).resolve().parent)]
features_pkg = types.ModuleType("CTAFlow.features")
features_pkg.__path__ = []
data_pkg = types.ModuleType("CTAFlow.data")
data_pkg.__path__ = []
contract_pkg = types.ModuleType("CTAFlow.data.contract_handling")
contract_pkg.__path__ = []

sys.modules["CTAFlow"] = cta_pkg
sys.modules["CTAFlow.features"] = features_pkg
sys.modules["CTAFlow.data"] = data_pkg
sys.modules["CTAFlow.data.contract_handling"] = contract_pkg
sys.modules["CTAFlow.data.contract_handling.curve_manager"] = stub

# Minimal utils package so curve_analysis can import seasonal utilities
utils_pkg = types.ModuleType("CTAFlow.utils")
utils_pkg.__path__ = []
sys.modules["CTAFlow.utils"] = utils_pkg

seasonal_path = Path(__file__).resolve().parent / "CTAFlow" / "utils" / "seasonal.py"
seasonal_spec = importlib.util.spec_from_file_location(
    "CTAFlow.utils.seasonal", seasonal_path
)
seasonal_module = importlib.util.module_from_spec(seasonal_spec)
sys.modules["CTAFlow.utils.seasonal"] = seasonal_module
seasonal_spec.loader.exec_module(seasonal_module)

# Stub numba to avoid optional dependency requirements
numba_stub = types.ModuleType("numba")

def jit(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper


def prange(*args, **kwargs):
    return range(*args)


numba_stub.jit = jit
numba_stub.prange = prange
sys.modules["numba"] = numba_stub

# Stub plotly modules used in curve_analysis
plotly_stub = types.ModuleType("plotly")
graph_objects_stub = types.ModuleType("plotly.graph_objects")
subplots_stub = types.ModuleType("plotly.subplots")
express_stub = types.ModuleType("plotly.express")

plotly_stub.graph_objects = graph_objects_stub
plotly_stub.subplots = subplots_stub
plotly_stub.express = express_stub

def make_subplots(*args, **kwargs):
    return None

subplots_stub.make_subplots = make_subplots

class Figure:  # minimal placeholder
    def __init__(self, *args, **kwargs):
        pass


graph_objects_stub.Figure = Figure

sys.modules["plotly"] = plotly_stub
sys.modules["plotly.graph_objects"] = graph_objects_stub
sys.modules["plotly.subplots"] = subplots_stub
sys.modules["plotly.express"] = express_stub

# Load curve_analysis module directly
MODULE_PATH = Path(__file__).resolve().parent / "CTAFlow" / "features" / "curve_analysis.py"
spec = importlib.util.spec_from_file_location("CTAFlow.features.curve_analysis", MODULE_PATH)
curve_analysis = importlib.util.module_from_spec(spec)
sys.modules["CTAFlow.features.curve_analysis"] = curve_analysis
spec.loader.exec_module(curve_analysis)  # type: ignore

CurveEvolutionAnalyzer = curve_analysis.CurveEvolutionAnalyzer
FuturesCurve = stub.FuturesCurve


def test_deseasonalized_log_prices():
    """CurveEvolutionAnalyzer should remove simple monthly seasonality."""

    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    curves = []
    for date in dates:
        price = 10 + date.month  # Introduce seasonal pattern
        curve = FuturesCurve(ref_date=date,
                             curve_month_labels=['F'],
                             prices=np.array([price]))
        curves.append(curve)

    series = pd.Series(curves, index=dates)
    analyzer = CurveEvolutionAnalyzer(series)

    log_prices_deseason = analyzer.get_log_prices_matrix(deseasonalize=True)
    df_deseason = pd.DataFrame(log_prices_deseason[:, 0], index=dates, columns=['price'])
    monthly_means = df_deseason.groupby(df_deseason.index.month).mean()['price']
    assert np.allclose(monthly_means.values, 0.0, atol=1e-8)

    raw_log_prices = analyzer.get_log_prices_matrix(deseasonalize=False)
    df_raw = pd.DataFrame(raw_log_prices[:, 0], index=dates, columns=['price'])
    monthly_means_raw = df_raw.groupby(df_raw.index.month).mean()['price']
    assert not np.allclose(monthly_means_raw.values, 0.0, atol=1e-2)

