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
    from .forecaster.forecast import CTAForecast, CTALinear, CTALight, CTAXGBoost, CTARForest
    from .data.data_client import DataClient
    from .data.signals_processing import COTProcessor, TechnicalAnalysis
    from .data.futures_curve_manager import FuturesCurveManager
    from .strategy.strategy import RegimeStrategy
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
    
    # Data processing classes
    'DataClient',
    'COTProcessor',
    'TechnicalAnalysis',
    'FuturesCurveManager',
    
    # Strategy classes
    'RegimeStrategy',
    
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