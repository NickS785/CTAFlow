import sys
import types
from pathlib import Path

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

try:
    import CTAFlow as _cta_module  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional when deps missing
    _cta_module = None  # type: ignore[assignment]
else:
    sys.modules.setdefault("CTAFlow.CTAFlow", _cta_module)

try:
    import CTAFlow.screeners as _cta_screeners  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional when deps missing
    _cta_screeners = None  # type: ignore[assignment]
else:
    sys.modules.setdefault("CTAFlow.CTAFlow.screeners", _cta_screeners)
