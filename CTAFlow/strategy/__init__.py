"""
CTAFlow Strategy Module

Trading strategy implementation using the CTA forecasting framework.

This module contains:
- CTAStrategy: Trading strategy based on CTA positioning predictions
- Position sizing and risk management
- Strategy backtesting and performance evaluation
"""

from .strategy import RegimeStrategy

__all__ = [
    'RegimeStrategy',
]