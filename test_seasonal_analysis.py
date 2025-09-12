import types
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# Use existing CTAFlow stub with path from previous tests
utils_pkg = types.ModuleType("CTAFlow.utils")
utils_pkg.__path__ = []
sys.modules["CTAFlow.utils"] = utils_pkg

MODULE_PATH = Path(__file__).resolve().parent / "CTAFlow" / "utils" / "seasonal.py"
spec = importlib.util.spec_from_file_location("CTAFlow.utils.seasonal", MODULE_PATH)
seasonal = importlib.util.module_from_spec(spec)
sys.modules["CTAFlow.utils.seasonal"] = seasonal
spec.loader.exec_module(seasonal)

SeasonalAnalysis = seasonal.SeasonalAnalysis
deseasonalize_monthly = seasonal.deseasonalize_monthly


def test_deseasonalize_monthly():
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    data = np.array([m for m in ((np.arange(24) % 12) + 1)], dtype=float).reshape(-1,1)
    result = deseasonalize_monthly(data, dates)
    df = pd.DataFrame(result[:,0], index=dates)
    monthly_means = df.groupby(df.index.month).mean()[0]
    assert np.allclose(monthly_means.values, 0.0, atol=1e-8)


def test_seasonal_analysis_workflow():
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    values = np.array([(i % 12) + 1 for i in range(24)], dtype=float)
    df = pd.DataFrame({'price': values}, index=dates)
    sa = SeasonalAnalysis(df)
    sa.fit_seasonal_model()
    rmse = sa.test_seasonal_model()
    assert rmse < 1e-8
    scores, comps = sa.deseasonalized_pca(n_components=1)
    assert scores.shape[1] == 1 and comps.shape == (1,1)
    filtered = sa.kalman_filter()
    assert filtered.shape == df.shape
