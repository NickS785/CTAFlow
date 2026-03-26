try:
    from .macro_client import MacroClient
except Exception:  # pragma: no cover - optional dependencies
    MacroClient = None  # type: ignore

try:
    from .ibkr_client import IBKRConfig, IBKRContract, IBKRTickDataSource
except Exception:  # pragma: no cover - optional dependencies
    IBKRConfig = None  # type: ignore
    IBKRContract = None  # type: ignore
    IBKRTickDataSource = None  # type: ignore

try:
    from .gcs_client import GCSConfig, GCSTickerSpec, GCSTickDataSource
except Exception:  # pragma: no cover - optional dependencies
    GCSConfig = None  # type: ignore
    GCSTickerSpec = None  # type: ignore
    GCSTickDataSource = None  # type: ignore

try:
    from .s3_client import S3TickerSpec, S3TickDataSource
except Exception:  # pragma: no cover - optional dependencies
    S3TickerSpec = None  # type: ignore
    S3TickDataSource = None  # type: ignore

__all__ = [
    "MacroClient",
    "IBKRConfig", "IBKRContract", "IBKRTickDataSource",
    "GCSConfig", "GCSTickerSpec", "GCSTickDataSource",
    "S3TickerSpec", "S3TickDataSource",
]
