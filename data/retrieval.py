import pandas as pd
import numpy as np
from config import MARKET_DATA_PATH, COT_DATA_PATH, APP_ROOT
import toml
import asyncio
from pathlib import Path

# Load futures mappings from TOML file
mapping = toml.load(APP_ROOT / 'data/futures_mappings.toml')

async def fetch_market_data(ticker_symbol: str) -> pd.DataFrame:
    """Async helper to load market data from HDF5 store."""
    def _load_market_data():
        with pd.HDFStore(MARKET_DATA_PATH, mode='r') as store:
            if ticker_symbol in store.keys():
                return store[ticker_symbol]
            else:
                available_keys = list(store.keys())
                raise KeyError(f"Ticker {ticker_symbol} not found. Available keys: {available_keys}")
    
    # Run the synchronous HDF5 operation in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_market_data)

async def fetch_cot_data(cot_symbol: str) -> pd.DataFrame:
    """Async helper to load COT data from HDF5 store."""
    def _load_cot_data():
        with pd.HDFStore(COT_DATA_PATH, mode='r') as store:
            if cot_symbol in store.keys():
                cot_data= store[cot_symbol]
            else:
                available_keys = list(store.keys())
                raise KeyError(f"COT symbol {cot_symbol} not found. Available keys: {available_keys}")

        cot_data = cot_data[cot_data['type'] == 'F_ALL']
        cot_filtered = cot_data.drop(columns=['type', 'date']).set_index(pd.to_datetime(cot_data['date']))


        return cot_filtered
    
    # Run the synchronous HDF5 operation in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_cot_data)

async def fetch_market_cot_data(ticker_symbol: str, resample=True, resample_period='1w'):
    """Fetch both market data and COT data for a given ticker symbol.
    
    Args:
        ticker_symbol: The futures ticker symbol (e.g., 'CL_F')
        resample: Whether to resample the data
        resample_period: Resampling period if resample=True
        
    Returns:
        pd.DataFrame: concatenated market and cot_data
    """
    try:
        # Get the COT commodity name from ticker mapping
        if ticker_symbol not in mapping['tickers']:
            raise ValueError(f"Ticker {ticker_symbol} not found in futures mappings")
        
        commodity_name = mapping['tickers'][ticker_symbol]

        commodity_name, ticker_symbol = f'/{commodity_name}', f'/{ticker_symbol}'
        # Fetch market data and COT data concurrently
        market_data_task = fetch_market_data(ticker_symbol)
        cot_data_task = fetch_cot_data(commodity_name)
        
        market_data, cot_data = await asyncio.gather(market_data_task, cot_data_task)
        
        # Apply resampling if requested
        if resample:
            market_data = market_data.resample(resample_period).agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()


        return pd.concat([market_data, cot_data], axis=1).ffill()
        
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()




