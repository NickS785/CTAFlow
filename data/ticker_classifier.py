"""
Ticker Classification System for COT Report Type Management

This module provides automatic classification of futures tickers into:
- Commodities (use disaggregated reports)
- Financials (use Traders in Financial Futures [TFF] reports)

It discovers available CSV files and creates ticker objects with appropriate
COT report configuration.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum
import glob

from config import RAW_MARKET_DATA_PATH, FUTURES_MAP, TICKER_TO_CODE


class ReportType(Enum):
    """COT report types"""
    DISAGGREGATED = "disaggregated"
    TFF = "traders_in_financial_futures_fut"


class TickerCategory(Enum):
    """Ticker categories for COT classification"""
    COMMODITY = "commodity"
    FINANCIAL = "financial"


@dataclass
class TickerInfo:
    """Information about a futures ticker"""
    ticker_symbol: str
    commodity_name: str
    cot_code: str
    category: TickerCategory
    report_type: ReportType
    csv_file_path: Optional[Path] = None
    storage_path: str = None
    
    def __post_init__(self):
        """Set storage path based on category"""
        if self.storage_path is None:
            if self.category == TickerCategory.FINANCIAL:
                self.storage_path = f"cot/tff/{self.ticker_symbol}"
            else:
                self.storage_path = f"cot/{self.ticker_symbol}"


class TickerClassifier:
    """Classifies tickers into commodities vs financials and manages their COT configurations"""
    
    # Financial futures that use TFF reports (based on financial instrument knowledge)
    FINANCIAL_CATEGORIES = {
        # Treasury Securities
        'TREASURY_BONDS', 'TREASURY_NOTES_10Y', 'TREASURY_NOTES_5Y', 'TREASURY_NOTES_2Y',
        # Interest Rate Products  
        'EURODOLLAR',
        # Equity Indices
        'SP_500', 'NASDAQ_100', 'DOW_JONES', 'RUSSELL_2000', 'VIX',
        # Currency Pairs
        'EUR_USD', 'GBP_USD', 'JPY_USD', 'CAD_USD', 'AUD_USD', 'CHF_USD'
    }
    
    def __init__(self):
        self.available_csv_files = self._discover_csv_files()
        self.ticker_objects = self._create_ticker_objects()
        
    def _discover_csv_files(self) -> Dict[str, Path]:
        """Discover available CSV files matching *_25.csv pattern"""
        csv_files = {}
        pattern = str(RAW_MARKET_DATA_PATH / "*_25.csv")
        
        for file_path in glob.glob(pattern):
            file_path = Path(file_path)
            # Extract ticker prefix (e.g., 'es' from 'es_25.csv')
            ticker_prefix = file_path.stem.split('_')[0].upper()
            
            # Map to full ticker symbol (e.g., 'ES' -> 'ES_F')
            ticker_symbol = f"{ticker_prefix}_F"
            csv_files[ticker_symbol] = file_path
            
        return csv_files
    
    def _create_ticker_objects(self) -> Dict[str, TickerInfo]:
        """Create TickerInfo objects for ALL tickers in futures mappings (regardless of CSV availability)"""
        ticker_objects = {}
        
        # Process ALL tickers in the mappings
        for ticker_symbol in TICKER_TO_CODE.keys():
            # Get commodity name and COT code
            commodity_name = FUTURES_MAP['tickers'][ticker_symbol]
            cot_code = TICKER_TO_CODE[ticker_symbol]
            
            # Classify as commodity or financial based on knowledge
            category = self._classify_ticker(commodity_name)
            report_type = ReportType.TFF if category == TickerCategory.FINANCIAL else ReportType.DISAGGREGATED
            
            # Check if CSV file exists for this ticker
            csv_path = self.available_csv_files.get(ticker_symbol)
            
            # Create ticker object
            ticker_info = TickerInfo(
                ticker_symbol=ticker_symbol,
                commodity_name=commodity_name,
                cot_code=cot_code,
                category=category,
                report_type=report_type,
                csv_file_path=csv_path  # Will be None if no CSV file
            )
            
            ticker_objects[ticker_symbol] = ticker_info
            
        return ticker_objects
    
    def _classify_ticker(self, commodity_name: str) -> TickerCategory:
        """Classify ticker as commodity or financial based on commodity name"""
        if commodity_name in self.FINANCIAL_CATEGORIES:
            return TickerCategory.FINANCIAL
        else:
            return TickerCategory.COMMODITY
    
    def get_commodities(self) -> List[TickerInfo]:
        """Get all commodity tickers"""
        return [ticker for ticker in self.ticker_objects.values() 
                if ticker.category == TickerCategory.COMMODITY]
    
    def get_financials(self) -> List[TickerInfo]:
        """Get all financial tickers"""
        return [ticker for ticker in self.ticker_objects.values() 
                if ticker.category == TickerCategory.FINANCIAL]
    
    def get_ticker_info(self, ticker_symbol: str) -> Optional[TickerInfo]:
        """Get TickerInfo for specific ticker"""
        return self.ticker_objects.get(ticker_symbol)
    
    def get_report_type(self, ticker_symbol: str) -> Optional[ReportType]:
        """Get appropriate COT report type for ticker"""
        ticker_info = self.get_ticker_info(ticker_symbol)
        return ticker_info.report_type if ticker_info else None
    
    def get_storage_path(self, ticker_symbol: str) -> Optional[str]:
        """Get HDF5 storage path for ticker"""
        ticker_info = self.get_ticker_info(ticker_symbol)
        return ticker_info.storage_path if ticker_info else None
    
    def get_all_tickers(self) -> List[TickerInfo]:
        """Get all available tickers"""
        return list(self.ticker_objects.values())
    
    def print_classification_summary(self):
        """Print summary of ticker classifications"""
        commodities = self.get_commodities()
        financials = self.get_financials()
        
        # Count tickers with CSV files
        commodities_with_csv = [t for t in commodities if t.csv_file_path is not None]
        financials_with_csv = [t for t in financials if t.csv_file_path is not None]
        
        print(f"\n=== Complete Ticker Classification Summary ===")
        print(f"Total Tickers in Mappings: {len(self.ticker_objects)}")
        print(f"Total CSV Files Available: {len(self.available_csv_files)}")
        print(f"Commodities (Disaggregated): {len(commodities)} ({len(commodities_with_csv)} with CSV)")
        print(f"Financials (TFF): {len(financials)} ({len(financials_with_csv)} with CSV)")
        
        print(f"\n--- Commodities ({len(commodities)} total) ---")
        for ticker in sorted(commodities, key=lambda x: x.ticker_symbol):
            csv_status = "CSV+" if ticker.csv_file_path else "CSV-"
            print(f"  {ticker.ticker_symbol:8} -> {ticker.commodity_name:25} [{ticker.cot_code}] [{csv_status}]")
        
        print(f"\n--- Financials ({len(financials)} total) ---")
        for ticker in sorted(financials, key=lambda x: x.ticker_symbol):
            csv_status = "CSV+" if ticker.csv_file_path else "CSV-"
            print(f"  {ticker.ticker_symbol:8} -> {ticker.commodity_name:25} [{ticker.cot_code}] [{csv_status}]")
        
        print(f"\n--- Processing Status ---")
        total_with_csv = len(commodities_with_csv) + len(financials_with_csv)
        total_without_csv = len(self.ticker_objects) - total_with_csv
        print(f"  Tickers ready for full processing (with CSV): {total_with_csv}")
        print(f"  Tickers for COT-only processing (no CSV): {total_without_csv}")
        print(f"  Classification coverage: 100% ({len(self.ticker_objects)} tickers)")


def get_ticker_classifier() -> TickerClassifier:
    """Get singleton ticker classifier instance"""
    if not hasattr(get_ticker_classifier, '_instance'):
        get_ticker_classifier._instance = TickerClassifier()
    return get_ticker_classifier._instance


def is_financial_ticker(ticker_symbol: str) -> bool:
    """Check if ticker is a financial future (uses TFF reports)"""
    classifier = get_ticker_classifier()
    ticker_info = classifier.get_ticker_info(ticker_symbol)
    return ticker_info.category == TickerCategory.FINANCIAL if ticker_info else False


def get_cot_report_type(ticker_symbol: str) -> str:
    """Get the appropriate COT report type for a ticker"""
    classifier = get_ticker_classifier()
    report_type = classifier.get_report_type(ticker_symbol)
    return report_type.value if report_type else "disaggregated"


def get_cot_storage_path(ticker_symbol: str) -> str:
    """Get the HDF5 storage path for a ticker's COT data"""
    classifier = get_ticker_classifier()
    storage_path = classifier.get_storage_path(ticker_symbol)
    return storage_path or f"cot/{ticker_symbol}"


if __name__ == "__main__":
    # Demo usage
    classifier = TickerClassifier()
    classifier.print_classification_summary()
    
    # Example queries
    print(f"\n=== Example Queries ===")
    print(f"ES_F report type: {get_cot_report_type('ES_F')}")
    print(f"ZC_F report type: {get_cot_report_type('ZC_F')}")
    print(f"ES_F storage path: {get_cot_storage_path('ES_F')}")
    print(f"ZC_F storage path: {get_cot_storage_path('ZC_F')}")
    print(f"ES_F is financial: {is_financial_ticker('ES_F')}")
    print(f"ZC_F is financial: {is_financial_ticker('ZC_F')}")