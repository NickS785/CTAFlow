"""
CTAFlow Features Module

Feature engineering, signal processing, and technical analysis components for the CTA positioning prediction system.

This module contains classes and functions for:
- Technical analysis and signal generation
- COT (Commitment of Traders) data processing  
- Feature engineering and transformation
- Advanced futures curve analysis with Lévy areas and path signatures
- Spread analysis and term structure features
- Intraday microstructure features
- Regime detection and seasonality analysis
"""

# Feature engineering classes
from .feature_engineering import IntradayFeatures
from .curve.advanced_features import SpreadData, FuturesCurve, CurveShapeAnalyzer, CurveEvolutionAnalyzer
from .seasonal_anamoly import intraday_autocorr_between_times, intraday_lag_autocorr, last_year_predicts_this_year, abnormal_months, prewindow_feature, prewindow_predicts_month
from .signals_processing import COTAnalyzer, TechnicalAnalysis
from .cyclical.seasonality import *
from .regime_classification import (
    BaseRegimeClassifier,
    CrowdingRegimeClassifier,
    RegimeSpecification,
    RegimeSpecificationLike,
    TrendRegimeClassifier,
    VolatilityRegimeClassifier,
    build_regime_classifier,
)
from .volume import *
SpreadFeatures = SpreadData

__all__ = [
    # Feature engineering classes
    'IntradayFeatures',
    'SpreadData',
    'FuturesCurve',
    'CurveShapeAnalyzer',
    'CurveEvolutionAnalyzer',
    # Signal processing classes
    'COTAnalyzer',
    'TechnicalAnalysis',
    # Volatility deseasonalization
    'DiurnalAdjuster',
    'RollingDiurnalAdjuster',
    'fft_spectrum',
    'compute_realized_volatility',
    'deseasonalize_volatility',
    'deseasonalize_volume',
    'estimate_diurnal_pattern',
    # Regime classification helpers
    'BaseRegimeClassifier',
    'TrendRegimeClassifier',
    'VolatilityRegimeClassifier',
    'CrowdingRegimeClassifier',
    'RegimeSpecification',
    'RegimeSpecificationLike',
    'build_regime_classifier',
    # Volume
    "MarketProfileExtractor",
]
