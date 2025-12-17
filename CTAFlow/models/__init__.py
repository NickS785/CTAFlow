"""
CTAFlow Forecaster Module

Machine learning models and forecasting framework for CTA positioning prediction.

This module contains:
- CTAForecast: Main forecasting orchestrator class
- Model implementations: Linear, Ridge, Lasso, LightGBM, XGBoost, Random Forest
- Time series cross-validation and model evaluation
- Feature engineering and target variable creation
- Grid search hyperparameter optimization
"""

from .base_models import (
    CTAForecast,
    CTALinear,
    CTALight,
    CTAXGBoost,
    CTARForest,
)
from .intraday_momentum import IntradayMomentum
from .volatility import RVForecast

__all__ = [
    # Main forecasting class
    'CTAForecast',
    
    # Individual model classes
    'CTALinear',
    'CTALight',
    'CTAXGBoost',
    'CTARForest',
    'IntradayMomentum',
    'RVForecast',
]