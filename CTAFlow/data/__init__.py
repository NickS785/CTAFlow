"""
CTAFlow Data Module

Data processing, retrieval, and analysis components for the CTA positioning prediction system.

This module contains classes and functions for:
- Market data retrieval from HDF5 stores
- COT (Commitment of Traders) data processing
- Technical analysis and signal generation
- Futures curve management
- Data preprocessing and feature engineering
"""

# Core data processing classes
from .data_client import DataClient
from .simple_processor import SimpleDataProcessor
from .raw_formatting import *
from .retrieval import fetch_market_cot_data, fetch_data_sync, convert_cot_date_to_datetime
from .ticker_classifier import TickerClassifier
from .classifications_reference import (
    COMMODITY_TICKERS,
    FINANCIAL_TICKERS,
    ALL_TICKERS,
    get_ticker_classification_info,
    get_all_classifications
)
from .raw_formatting.synthetic import CrossProductEngine, IntradaySpreadEngine , CrossSpreadLeg, IntradayLeg
# Legacy data processing utilities (deprecated - use SimpleDataProcessor instead)
try:
    from .data_processor import DataProcessor, process_all_futures_curves
    import warnings
    warnings.warn(
        "DataProcessor is deprecated. Use SimpleDataProcessor instead for better performance and simplicity.",
        DeprecationWarning,
        stacklevel=2
    )
except ImportError:
    # Handle case where DataProcessor might not be available
    DataProcessor = None
    process_all_futures_curves = None

__all__ = [
    # Primary data processing classes
    'SimpleDataProcessor',
    'DataClient',

    # Core data classes
    'FuturesCurveManager',
    'TickerClassifier',
    'SpreadData',
    'SpreadReturns',
    'FuturesCurve',
    'RollDateManager',
    'create_enhanced_curve_manager_with_roll_tracking',
    'IntradaySpreadEngine',
    'CrossSpreadLeg',

    # Legacy classes (deprecated)
    'DataProcessor',
    'process_all_futures_curves',

    # Data retrieval functions
    'fetch_market_cot_data',
    'fetch_data_sync',
    'convert_cot_date_to_datetime',
    "DLYFolderUpdater",
    "DLYContractManager"

    # Classification mappings
    'COMMODITY_TICKERS',
    'FINANCIAL_TICKERS',
    'ALL_TICKERS',
    'get_ticker_classification_info',
    'get_all_classifications',
]