"""
CTAFlow Strategy Module

Trading strategy implementation using the CTA forecasting framework.

This module contains:
- CTAStrategy: Trading strategy based on CTA positioning predictions
- Position sizing and risk management
- Strategy backtesting and performance evaluation
"""

from .screener_pipeline import ScreenerPipeline, HorizonMapper
from .sessionizer import Sessionizer, SessionizerConfig

__all__ = [
    'ScreenerPipeline',
    "HorizonMapper",
    "Sessionizer",
    "SessionizerConfig",
]

try:
    from .strategy import RegimeStrategy  # pragma: no cover - optional legacy export
except ModuleNotFoundError:  # pragma: no cover - historical module missing in lightweight envs
    RegimeStrategy = None  # type: ignore[assignment]
else:
    __all__.append('RegimeStrategy')

