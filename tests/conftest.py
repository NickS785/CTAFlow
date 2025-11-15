import importlib.util
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

def _load_repo_package(module_name: str, package_root: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


CTA_PACKAGE = ROOT / "CTAFlow"
_cta_module = None
if CTA_PACKAGE.exists():
    try:  # Prefer existing installation when it already points at the repo
        import CTAFlow as _imported_cta  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional when deps missing
        _imported_cta = None  # type: ignore[assignment]

    if _imported_cta is not None:
        module_path = Path(getattr(_imported_cta, "__file__", "")).resolve().parent
        if module_path == CTA_PACKAGE.resolve():
            _cta_module = _imported_cta  # type: ignore[assignment]

    if _cta_module is None:
        try:
            _cta_module = _load_repo_package("CTAFlow", CTA_PACKAGE)
        except Exception:  # pragma: no cover - fallback when optional deps missing
            _cta_module = None  # type: ignore[assignment]

if _cta_module is not None:
    sys.modules.setdefault("CTAFlow.CTAFlow", _cta_module)

try:
    import CTAFlow.screeners as _cta_screeners  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional when deps missing
    _cta_screeners = None  # type: ignore[assignment]
else:
    sys.modules.setdefault("CTAFlow.CTAFlow.screeners", _cta_screeners)
