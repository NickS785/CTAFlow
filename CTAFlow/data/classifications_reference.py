"""
Complete Ticker Classifications Reference

This module provides easy access to all ticker classifications and demonstrates
the automatic COT report type system.
"""

from .ticker_classifier import get_ticker_classifier, is_financial_ticker, get_cot_report_type, get_cot_storage_path
from ..config import TICKER_TO_CODE


# Complete classification mappings for ALL 48 tickers in futures_mappings.toml
COMMODITY_TICKERS = {
    # Energy
    'CL_F': {'name': 'Crude Oil', 'category': 'Energy', 'cot_code': '067651', 'csv': True},
    'HO_F': {'name': 'Heating Oil', 'category': 'Energy', 'cot_code': '022651', 'csv': True}, 
    'NG_F': {'name': 'Natural Gas', 'category': 'Energy', 'cot_code': '023651', 'csv': True},
    'RB_F': {'name': 'RBOB Gasoline', 'category': 'Energy', 'cot_code': '111659', 'csv': True},
    'BZ_F': {'name': 'Brent Crude', 'category': 'Energy', 'cot_code': '132741', 'csv': False},
    
    # Metals - Precious
    'GC_F': {'name': 'Gold', 'category': 'Metals', 'cot_code': '088691', 'csv': True},
    'SI_F': {'name': 'Silver', 'category': 'Metals', 'cot_code': '084691', 'csv': False},
    'PA_F': {'name': 'Palladium', 'category': 'Metals', 'cot_code': '075651', 'csv': False},
    'PL_F': {'name': 'Platinum', 'category': 'Metals', 'cot_code': '076651', 'csv': False},
    
    # Metals - Base
    'HG_F': {'name': 'Copper', 'category': 'Metals', 'cot_code': '085692', 'csv': False},
    'ALI_F': {'name': 'Aluminum', 'category': 'Metals', 'cot_code': '191693', 'csv': False},
    
    # Livestock
    'LE_F': {'name': 'Live Cattle', 'category': 'Livestock', 'cot_code': '057642', 'csv': True},
    'GE_F': {'name': 'Feeder Cattle', 'category': 'Livestock', 'cot_code': '061641', 'csv': False},
    'HE_F': {'name': 'Lean Hogs', 'category': 'Livestock', 'cot_code': '054642', 'csv': False},
    
    # Agriculture - Grains
    'ZC_F': {'name': 'Corn', 'category': 'Agriculture', 'cot_code': '002602', 'csv': True},
    'ZS_F': {'name': 'Soybeans', 'category': 'Agriculture', 'cot_code': '005602', 'csv': True},
    'ZW_F': {'name': 'Wheat', 'category': 'Agriculture', 'cot_code': '001602', 'csv': False},
    'KE_F': {'name': 'Hard Red Winter Wheat', 'category': 'Agriculture', 'cot_code': '001612', 'csv': False},
    'MWE_F': {'name': 'Hard Red Spring Wheat', 'category': 'Agriculture', 'cot_code': '001626', 'csv': False},
    'ZO_F': {'name': 'Oats', 'category': 'Agriculture', 'cot_code': '004601', 'csv': False},
    'ZR_F': {'name': 'Rice', 'category': 'Agriculture', 'cot_code': '039601', 'csv': False},
    'ZL_F': {'name': 'Soybean Oil', 'category': 'Agriculture', 'cot_code': '007601', 'csv': False},
    'ZM_F': {'name': 'Soybean Meal', 'category': 'Agriculture', 'cot_code': '026601', 'csv': False},
    
    # Agriculture - Softs
    'KC_F': {'name': 'Coffee', 'category': 'Softs', 'cot_code': '083731', 'csv': False},
    'SB_F': {'name': 'Sugar', 'category': 'Softs', 'cot_code': '080732', 'csv': False},
    'CC_F': {'name': 'Cocoa', 'category': 'Softs', 'cot_code': '073732', 'csv': False},
    'CT_F': {'name': 'Cotton', 'category': 'Softs', 'cot_code': '033661', 'csv': False},
    'OJ_F': {'name': 'Orange Juice', 'category': 'Softs', 'cot_code': '040701', 'csv': False},
    
    # Agriculture - Dairy
    'DC_F': {'name': 'Milk Class III', 'category': 'Dairy', 'cot_code': '052641', 'csv': False},
    'DA_F': {'name': 'Butter', 'category': 'Dairy', 'cot_code': '050642', 'csv': False},
    'DY_F': {'name': 'Cheese', 'category': 'Dairy', 'cot_code': '063642', 'csv': False},
    'NF_F': {'name': 'Nonfat Dry Milk', 'category': 'Dairy', 'cot_code': '052642', 'csv': False},
}

FINANCIAL_TICKERS = {
    # Equity Indices
    'ES_F': {'name': 'S&P 500 E-mini', 'category': 'Equity Index', 'cot_code': '138741', 'csv': True},
    'NQ_F': {'name': 'NASDAQ 100 E-mini', 'category': 'Equity Index', 'cot_code': '209742', 'csv': False},
    'YM_F': {'name': 'Dow Jones E-mini', 'category': 'Equity Index', 'cot_code': '124603', 'csv': False},
    'RTY_F': {'name': 'Russell 2000 E-mini', 'category': 'Equity Index', 'cot_code': '239742', 'csv': False},
    'VX_F': {'name': 'VIX', 'category': 'Equity Index', 'cot_code': '1170E1', 'csv': False},
    
    # Treasury Securities
    'ZB_F': {'name': '30Y Treasury Bonds', 'category': 'Treasury', 'cot_code': '020601', 'csv': True},
    'ZN_F': {'name': '10Y Treasury Notes', 'category': 'Treasury', 'cot_code': '043602', 'csv': True},
    'ZT_F': {'name': '2Y Treasury Notes', 'category': 'Treasury', 'cot_code': '040602', 'csv': True},
    'ZF_F': {'name': '5Y Treasury Notes', 'category': 'Treasury', 'cot_code': '042602', 'csv': False},

    # Currency Pairs
    '6E_F': {'name': 'EUR/USD', 'category': 'Currency', 'cot_code': '099741', 'csv': False},
    '6B_F': {'name': 'GBP/USD', 'category': 'Currency', 'cot_code': '096742', 'csv': False},
    '6J_F': {'name': 'JPY/USD', 'category': 'Currency', 'cot_code': '097741', 'csv': False},
    '6C_F': {'name': 'CAD/USD', 'category': 'Currency', 'cot_code': '090741', 'csv': False},
    '6A_F': {'name': 'AUD/USD', 'category': 'Currency', 'cot_code': '232741', 'csv': False},
    '6S_F': {'name': 'CHF/USD', 'category': 'Currency', 'cot_code': '092741', 'csv': False},
}

ALL_TICKERS = {**COMMODITY_TICKERS, **FINANCIAL_TICKERS}


def get_ticker_classification_info(ticker: str) -> dict:
    """
    Get complete classification information for a ticker.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol (e.g., 'ES_F', 'ZC_F')
        
    Returns:
    --------
    dict
        Complete classification info
    """
    if ticker not in ALL_TICKERS:
        raise ValueError(f"Ticker {ticker} not found in available classifications")
    
    base_info = ALL_TICKERS[ticker]
    
    return {
        'ticker': ticker,
        'name': base_info['name'],
        'category': base_info['category'],
        'cot_code': base_info['cot_code'],
        'is_financial': is_financial_ticker(ticker),
        'report_type': get_cot_report_type(ticker),
        'storage_path': get_cot_storage_path(ticker),
        'classification': 'Financial' if is_financial_ticker(ticker) else 'Commodity'
    }


def get_all_classifications() -> dict:
    """Get classifications for all available tickers."""
    return {ticker: get_ticker_classification_info(ticker) for ticker in ALL_TICKERS.keys()}


def get_commodities_by_category() -> dict:
    """Group symbol tickers by category."""
    categories = {}
    for ticker, info in COMMODITY_TICKERS.items():
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(ticker)
    return categories


def get_financials_by_category() -> dict:
    """Group financial tickers by category."""
    categories = {}
    for ticker, info in FINANCIAL_TICKERS.items():
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(ticker)
    return categories


def print_classification_summary():
    """Print a complete summary of all classifications."""
    print("=" * 80)
    print("COMPLETE TICKER CLASSIFICATION REFERENCE")
    print("=" * 80)
    
    print(f"\nTotal Available Tickers: {len(ALL_TICKERS)}")
    print(f"  • Commodities: {len(COMMODITY_TICKERS)}")
    print(f"  • Financials: {len(FINANCIAL_TICKERS)}")
    
    print(f"\n" + "-" * 80)
    print("COMMODITIES (Use Disaggregated Reports)")
    print("-" * 80)
    
    commodities_by_cat = get_commodities_by_category()
    for category, tickers in commodities_by_cat.items():
        print(f"\n{category} ({len(tickers)} tickers):")
        for ticker in sorted(tickers):
            info = get_ticker_classification_info(ticker)
            print(f"  {ticker:<6} → {info['name']:<20} [{info['cot_code']}] → {info['storage_path']}")
    
    print(f"\n" + "-" * 80)
    print("FINANCIALS (Use TFF Reports)")
    print("-" * 80)
    
    financials_by_cat = get_financials_by_category()
    for category, tickers in financials_by_cat.items():
        print(f"\n{category} ({len(tickers)} tickers):")
        for ticker in sorted(tickers):
            info = get_ticker_classification_info(ticker)
            print(f"  {ticker:<6} → {info['name']:<20} [{info['cot_code']}] → {info['storage_path']}")
    
    print(f"\n" + "=" * 80)
    print("RAW DATA STORAGE PATHS")
    print("=" * 80)
    print("• Commodities → Raw data stored in: cot/raw")
    print("• Financials  → Raw data stored in: cot/tff")
    
    print(f"\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("""
# Get classification info
from .classifications_reference import get_ticker_classification_info

info = get_ticker_classification_info('ES_F')
print(f"{info['ticker']}: {info['classification']}, {info['report_type']}")

# Use with DataClient
from .data_client import DataClient
client = DataClient()

# Automatic report type selection
es_data = client.download_cot_for_ticker('ES_F')  # Uses TFF
zc_data = client.download_cot_for_ticker('ZC_F')  # Uses disaggregated

# Automatic storage path detection  
es_metrics = client.query_cot_metrics('ES_F')  # Finds cot/tff/ES_F/metrics
zc_metrics = client.query_cot_metrics('ZC_F')  # Finds cot/ZC_F/metrics
""")


if __name__ == "__main__":
    print_classification_summary()
    
    # Demonstrate individual ticker lookup
    print(f"\n" + "=" * 80)
    print("INDIVIDUAL TICKER EXAMPLES")
    print("=" * 80)
    
    examples = ['ES_F', 'ZC_F', 'GC_F', 'ZB_F']
    for ticker in examples:
        info = get_ticker_classification_info(ticker)
        print(f"{ticker}: {info['name']} → {info['classification']} → {info['report_type']}")