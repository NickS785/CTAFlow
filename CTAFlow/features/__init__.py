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
from .curve_analysis import (
    CurveShapeAnalyzer, CurveEvolution,
    SpreadAnalyzer
)
from .. import FuturesCurve, SpreadData
from ..data.contract_handling.futures_curve_manager import SpreadFeature, SeqData, Contract

# Signal processing and technical analysis classes
from .signals_processing import COTProcessor, TechnicalAnalysis

# Backward compatibility aliases
SpreadFeatures = SpreadData
SpreadAnalysis = SpreadAnalyzer

__all__ = [
    # Feature engineering classes
    'IntradayFeatures',
    'CurveShapeAnalyzer',
    'CurveEvolution',
    'SpreadAnalyzer',
    
    # Signal processing classes
    'COTProcessor',
    'TechnicalAnalysis',
    'FuturesCurve',
    'SpreadData',
    'SeqData',
    'Contract',
    'SpreadFeature',
    # Aliases for backward compatibility
    'SpreadFeatures',
    'SpreadAnalysis',
]