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

from pathlib import Path
from typing import Optional, Union

import pandas as pd

# Core data processing classes
from .data_client import DataClient, ResultsClient
from .volume_bucketed_loader import VolumeBucketedDataLoader
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

# Data processing utilities
try:
    from .data_processor import DataProcessor
except ImportError:
    # Handle case where DataProcessor might not be available
    DataProcessor = None

_DEFAULT_EXAMPLE_CSV = Path(__file__).resolve().parents[2] / 'docs' / 'example.csv'


def _load_example_csv(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the bundled example CSV and return a time-indexed DataFrame."""

    csv_path = Path(path) if path is not None else _DEFAULT_EXAMPLE_CSV
    if not csv_path.exists():  # pragma: no cover - defensive
        raise FileNotFoundError(f"Example CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df


def read_synthetic_csv(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Return crack-spread OHLC data from the shared CSV sample."""

    df = _load_example_csv(path)
    crack_cols = [col for col in df.columns if col.startswith('crack_')]
    if not crack_cols:
        return df.copy()
    return df[crack_cols].copy()


def read_exported_df(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Return the exported-format OHLC data with crack spread columns suffixed with ``.1``."""

    df = _load_example_csv(path)
    crack_cols = [col for col in df.columns if col.startswith('crack_')]
    if not crack_cols:
        return df.copy()

    renamed = df[crack_cols].rename(columns=lambda col: f"{col}.1")
    base = df.drop(columns=crack_cols)
    return pd.concat([base, renamed], axis=1)


__all__ = [
    # Primary data processing classes
    'DataProcessor',
    'DataClient',
    'ResultsClient',
    'VolumeBucketedDataLoader',

    # Core data classes
    'FuturesCurveManager',
    'TickerClassifier',
    'SpreadData',
    'SpreadReturns',
    'FuturesCurve',
    'IntradaySpreadEngine',
    'CrossSpreadLeg',
    "SyntheticSymbol",
    "ContractSpecs",
    "Contract",
    "ContractInfo",

    # Data retrieval functions
    'fetch_market_cot_data',
    'fetch_data_sync',
    'convert_cot_date_to_datetime',
    "DLYFolderUpdater",
    "DLYContractManager",

    # Example data helpers
    'read_synthetic_csv',
    'read_exported_df',

    # Classification mappings
    'COMMODITY_TICKERS',
    'FINANCIAL_TICKERS',
    'ALL_TICKERS',
    'get_ticker_classification_info',
    'get_all_classifications',
]