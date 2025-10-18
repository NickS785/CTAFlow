import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight stub for optional dependencies that are not available in
# the execution environment (e.g. python-dotenv used by config.py).
if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")

    def _noop_load_dotenv(*args, **kwargs):
        return None

    dotenv.load_dotenv = _noop_load_dotenv
    sys.modules["dotenv"] = dotenv

if "sierrapy" not in sys.modules:
    sierrapy = types.ModuleType("sierrapy")
    parser = types.ModuleType("sierrapy.parser")

    class _DummyScidReader:  # pragma: no cover - placeholder for optional dependency
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def load_front_month_series(self, *args, **kwargs):
            raise NotImplementedError("ScidReader is not available in the test environment")

    parser.ScidReader = _DummyScidReader
    parser.AsyncScidReader = _DummyScidReader
    parser.bucket_by_volume = lambda *args, **kwargs: None
    parser.resample_ohlcv = lambda *args, **kwargs: None
    sierrapy.parser = parser
    sys.modules["sierrapy"] = sierrapy
    sys.modules["sierrapy.parser"] = parser

if "numba" not in sys.modules:
    numba = types.ModuleType("numba")

    def _jit(func=None, **kwargs):  # pragma: no cover - simple passthrough
        if func is None:
            def decorator(f):
                return f

            return decorator
        return func

    numba.jit = _jit
    numba.prange = range
    sys.modules["numba"] = numba

if "statsmodels" not in sys.modules:
    statsmodels = types.ModuleType("statsmodels")
    api = types.ModuleType("statsmodels.api")

    class _OLSResult:  # pragma: no cover - lightweight analytical stub
        def __init__(self, params: np.ndarray, tvalues: np.ndarray, rsq: float):
            self.params = params
            self.tvalues = tvalues
            self.pvalues = np.full_like(params, np.nan, dtype=float)
            self.rsquared = rsq

    class OLS:  # pragma: no cover
        def __init__(self, y, X, hasconst: bool = True):
            self.y = np.asarray(y, dtype=float)
            self.X = np.asarray(X, dtype=float)

        def fit(self, cov_type: str | None = None, cov_kwds: dict | None = None):
            beta, *_ = np.linalg.lstsq(self.X, self.y, rcond=None)
            residuals = self.y - self.X @ beta
            dof = max(len(self.y) - self.X.shape[1], 1)
            sigma2 = float(np.sum(residuals ** 2) / dof)
            xtx_inv = np.linalg.pinv(self.X.T @ self.X)
            se = np.sqrt(np.diag(xtx_inv) * sigma2)
            with np.errstate(divide="ignore", invalid="ignore"):
                tvalues = beta / se
            rsq = 1.0 - (np.sum(residuals ** 2) / np.sum((self.y - self.y.mean()) ** 2)) if self.y.var() else 0.0
            return _OLSResult(beta, tvalues, rsq)

    api.OLS = OLS
    statsmodels.api = api
    sys.modules["statsmodels"] = statsmodels
    sys.modules["statsmodels.api"] = api
