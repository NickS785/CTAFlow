from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import toml
import os


intraday_db_path = "lmdb:\\F:\\data\\intraday"

# Weekly Update Configuration
# Set to False to disable automated weekly updates
ENABLE_WEEKLY_UPDATES = os.getenv("CTAFLOW_ENABLE_WEEKLY_UPDATES", "false").lower() in ("true", "1", "yes")

# ArcticDB URIs (primary data storage)
INTRADAY_ADB_PATH = "lmdb://F:/Data/intraday"  # ArcticDB URI for intraday data
DAILY_ADB_PATH = "lmdb://F:/Data/daily"  # ArcticDB URI for daily market data
CURVE_ADB_PATH = "lmdb://F:/Data/curves"  # ArcticDB URI for forward curve data
COT_ADB_PATH = "lmdb://F:/Data/cot"  # ArcticDB URI for COT data

# Legacy HDF5 paths (deprecated - use Arctic instead)
MARKET_DATA_PATH = Path("F:\\", 'Data', 'market_data.hd5')
COT_DATA_PATH = Path("F:\\", 'Data', 'cot_data.hd5')

# File system paths
DATA_DIR = Path("F:\\Data ")
RAW_MARKET_DATA_PATH = Path("F:\\", 'charts')  # CSV market data files
DLY_DATA_PATH = Path("F:\\", 'SierraChart', 'Data')  # DLY futures data files
MODEL_DATA_PATH = Path("F:\\", "ML", "models")
EXAMPLE_DATA_PATH = Path(__file__).parent / "example_data"
APP_ROOT = Path(__file__).parent
with open(APP_ROOT / 'data' / 'futures_mappings.toml') as map:
    FUTURES_MAP = toml.load(map)

# Create bidirectional mappings
REV_TICKER_CODES = {v: k for k, v in FUTURES_MAP['COT']['codes'].items()}  # code -> ticker name
TICKER_TO_CODE = {ticker: FUTURES_MAP['COT']['codes'][commodity] 
                  for ticker, commodity in FUTURES_MAP['tickers'].items()}  # ticker -> code
CODE_TO_TICKER = {code: ticker for ticker, code in TICKER_TO_CODE.items()}  # code -> ticker




def get_cot_code(ticker_or_commodity: str) -> str:
    """
    Get COT code from ticker symbol or commodity name.
    
    Parameters:
    -----------
    ticker_or_commodity : str
        Either a ticker symbol (e.g., 'ZC_F') or commodity name (e.g., 'CORN')
    
    Returns:
    --------
    str
        COT code (e.g., '002602')
    
    Raises:
    -------
    KeyError
        If ticker/commodity not found
    """
    ticker_upper = ticker_or_commodity.upper()
    
    # Try ticker first
    if ticker_upper in TICKER_TO_CODE:
        return TICKER_TO_CODE[ticker_upper]
    
    # Try commodity name
    if ticker_upper in FUTURES_MAP['COT']['codes']:
        return FUTURES_MAP['COT']['codes'][ticker_upper]
    
    raise KeyError(f"No COT code found for '{ticker_or_commodity}'. "
                   f"Available tickers: {list(TICKER_TO_CODE.keys())}, "
                   f"Available commodities: {list(FUTURES_MAP['COT']['codes'].keys())}")

def get_ticker_from_code(cot_code: str) -> str:
    """
    Get ticker symbol from COT code.
    
    Parameters:
    -----------
    cot_code : str
        COT code (e.g., '002602')
    
    Returns:
    --------
    str
        Ticker symbol (e.g., 'ZC_F')
    
    Raises:
    -------
    KeyError
        If COT code not found
    """
    if cot_code in CODE_TO_TICKER:
        return CODE_TO_TICKER[cot_code]
    
    raise KeyError(f"No ticker found for COT code '{cot_code}'. "
                   f"Available codes: {list(CODE_TO_TICKER.keys())}")

def get_commodity_from_code(cot_code: str) -> str:
    """
    Get commodity name from COT code.
    
    Parameters:
    -----------
    cot_code : str
        COT code (e.g., '002602')
    
    Returns:
    --------
    str
        Commodity name (e.g., 'CORN')
    
    Raises:
    -------
    KeyError
        If COT code not found
    """
    if cot_code in REV_TICKER_CODES:
        return REV_TICKER_CODES[cot_code]
    
    raise KeyError(f"No commodity found for COT code '{cot_code}'. "
                   f"Available codes: {list(REV_TICKER_CODES.keys())}")

def get_all_mappings() -> dict:
    """
    Get all available mappings between tickers, commodities, and COT codes.
    
    Returns:
    --------
    dict
        Dictionary with all mapping relationships
    """
    return {
        'ticker_to_code': TICKER_TO_CODE,
        'code_to_ticker': CODE_TO_TICKER,
        'commodity_to_code': FUTURES_MAP['COT']['codes'],
        'code_to_commodity': REV_TICKER_CODES,
        'ticker_to_commodity': FUTURES_MAP['tickers']
    }

def search_mappings(search_term: str) -> dict:
    """
    Search for mappings containing the search term.
    
    Parameters:
    -----------
    search_term : str
        Term to search for (case-insensitive)
    
    Returns:
    --------
    dict
        Dictionary with matching mappings
    """
    search_lower = search_term.lower()
    results = {
        'matching_tickers': [],
        'matching_commodities': [],
        'matching_codes': []
    }
    
    # Search tickers
    for ticker in TICKER_TO_CODE.keys():
        if search_lower in ticker.lower():
            results['matching_tickers'].append({
                'ticker': ticker,
                'commodity': FUTURES_MAP['tickers'][ticker],
                'cot_code': TICKER_TO_CODE[ticker]
            })
    
    # Search commodities
    for commodity in FUTURES_MAP['COT']['codes'].keys():
        if search_lower in commodity.lower():
            code = FUTURES_MAP['COT']['codes'][commodity]
            ticker = CODE_TO_TICKER.get(code, 'N/A')
            results['matching_commodities'].append({
                'commodity': commodity,
                'cot_code': code,
                'ticker': ticker
            })
    
    # Search codes
    for code in REV_TICKER_CODES.keys():
        if search_lower in code:
            commodity = REV_TICKER_CODES[code]
            ticker = CODE_TO_TICKER.get(code, 'N/A')
            results['matching_codes'].append({
                'cot_code': code,
                'commodity': commodity,
                'ticker': ticker
            })
    
    return results


