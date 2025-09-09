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
from .contract_handling.futures_curve_manager import FuturesCurveManager, SpreadData
from .retrieval import fetch_market_cot_data, fetch_data_sync, convert_cot_date_to_datetime
from .ticker_classifier import TickerClassifier
from .classifications_reference import (
    COMMODITY_TICKERS, 
    FINANCIAL_TICKERS, 
    ALL_TICKERS,
    get_ticker_classification_info,
    get_all_classifications
)

# Data processing utilities
try:
    from .data_processor import DataProcessor
except ImportError:
    # Handle case where DataProcessor might not be available
    DataProcessor = None

__all__ = [
    # Core data classes
    'DataClient',
    'FuturesCurveManager',
    'TickerClassifier',
    'DataProcessor',
    'SpreadData'
    
    # Data retrieval functions
    'fetch_market_cot_data',
    'fetch_data_sync',
    'convert_cot_date_to_datetime',
    
    # Classification mappings
    'COMMODITY_TICKERS',
    'FINANCIAL_TICKERS', 
    'ALL_TICKERS',
    'get_ticker_classification_info',
    'get_all_classifications',
]