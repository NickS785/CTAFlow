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
from .curve_analysis import SpreadData, FuturesCurve, CurveShapeAnalyzer, CurveEvolutionAnalyzer
from .seasonal_anamoly import intraday_autocorr_between_times, intraday_lag_autocorr, last_year_predicts_this_year, abnormal_months, prewindow_feature, prewindow_predicts_month
from .signals_processing import COTAnalyzer, TechnicalAnalysis
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
]
