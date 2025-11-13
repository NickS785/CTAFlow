"""
CTAFlow Strategy Module

Trading strategy implementation using the CTA forecasting framework.

This module contains:
- CTAStrategy: Trading strategy based on CTA positioning predictions
- Position sizing and risk management
- Strategy backtesting and performance evaluation
"""

from .screener_pipeline import ScreenerPipeline, HorizonMapper, ScreenerBacktester
from .sessionizer import Sessionizer, SessionizerConfig
from .backtester import BacktestSummary

__all__ = [
    'ScreenerPipeline',
    "HorizonMapper",
    "ScreenerBacktester",
    "Sessionizer",
    "SessionizerConfig",
    "BacktestSummary",
]

try:
    from .strategy import RegimeStrategy  # pragma: no cover - optional legacy export
except ModuleNotFoundError:  # pragma: no cover - historical module missing in lightweight envs
    RegimeStrategy = None  # type: ignore[assignment]
else:
    __all__.append('RegimeStrategy')

