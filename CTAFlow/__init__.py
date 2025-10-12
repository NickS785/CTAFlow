
"""
CTAFlow - CTA (Commodity Trading Advisor) Positioning Prediction System

A comprehensive framework for predicting institutional positioning in commodity futures markets
using technical indicators, COT (Commitment of Traders) data, and machine learning models.

Core Components:
- data: Data processing, retrieval, and analysis modules
- forecaster: Machine learning models for CTA positioning prediction  
- strategy: Trading strategy implementation using forecasting framework
- config: Configuration and data paths

Key Features:
- Technical indicator calculation with selective computation
- COT data processing and feature engineering
- Multiple ML models: Linear, LightGBM, XGBoost, Random Forest
- Weekly resampling aligned with COT reporting schedule
- Grid search hyperparameter optimization
- Time series cross-validation

Example Usage:
    >>> from CTAFlow import CTAForecast
    >>> forecaster = CTAForecast('CL_F')
    >>> forecaster.prepare_data(selected_indicators=['moving_averages', 'rsi'])
    >>> result = forecaster.train_model(model_type='lightgbm', target_type='return')
    >>> print(result['test_metrics'])
"""

# Version information
__version__ = "1.0.0"
__author__ = "CTA Research Team"
__email__ = "research@example.com"

# Import main classes for easy access
try:
    # Core forecasting classes
    from .forecaster.forecast import CTAForecast, CTALinear, CTALight, CTAXGBoost, CTARForest

    # Primary data processing classes
    from .data.data_processor import DataProcessor
    from .data.data_client import DataClient

    # Feature processing classes
    from .features.feature_engineering import IntradayFeatures
    from .features.curve_analysis import CurveShapeAnalyzer, CurveEvolutionAnalyzer
    from .features.signals_processing import COTAnalyzer, TechnicalAnalysis

    # Data container classes
    from .data.raw_formatting.spread_manager import FuturesCurveManager, FuturesCurve, SpreadData
    from .screeners import ScreenParams, HistoricalScreener

    # Configuration
    from .config import *
except ImportError as e:
    # Handle cases where dependencies might not be available
    print(f"Warning: Some CTAFlow components could not be imported: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")

# Define public API
__all__ = [
    # Core forecasting classes
    'CTAForecast',
    'CTALinear',
    'CTALight',
    'CTAXGBoost',
    'CTARForest',

    # Primary data processing classes
    'DataProcessor',
    'DataClient',

    # Feature processing classes
    'COTAnalyzer',
    'TechnicalAnalysis',
    'IntradayFeatures',

    # Analysis classes (Phase 2 consolidated)
    'CurveShapeAnalyzer',
    'CurveEvolutionAnalyzer',

    # Data container classes
    'FuturesCurveManager',
    'FuturesCurve',
    'SpreadData',

    # Screener classes
    'ScreenParams',
    'HistoricalScreener',

    # Utility functions and constants from config
    'MARKET_DATA_PATH',
    'COT_DATA_PATH',
    'MODEL_DATA_PATH',
]

# Package metadata
DESCRIPTION = "CTA positioning prediction system using COT data and technical analysis"
LONG_DESCRIPTION = __doc__

# Supported model types
SUPPORTED_MODELS = ['linear', 'ridge', 'lasso', 'elastic_net', 'lightgbm', 'xgboost', 'randomforest']

# Supported technical indicator groups  
SUPPORTED_INDICATORS = ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence', 'vol_normalized']

# Supported COT feature groups
SUPPORTED_COT_FEATURES = ['positioning', 'flows', 'extremes', 'market_structure', 'interactions', 'spreads']

def get_version():
    """Return the current version of CTAFlow."""
    return __version__

def get_supported_models():
    """Return list of supported model types."""
    return SUPPORTED_MODELS.copy()

def get_supported_indicators():
    """Return list of supported technical indicator groups."""
    return SUPPORTED_INDICATORS.copy()

def get_supported_cot_features():
    """Return list of supported COT feature groups.""" 
    return SUPPORTED_COT_FEATURES.copy()