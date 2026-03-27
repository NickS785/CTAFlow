try:
    from .macro_client import MacroClient
except Exception:  # pragma: no cover - optional dependencies
    MacroClient = None  # type: ignore

__all__ = [
    "MacroClient",
]
