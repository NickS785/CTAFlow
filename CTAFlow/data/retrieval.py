import datetime

import pandas as pd
import numpy as np
from config import MARKET_DATA_PATH, COT_DATA_PATH, APP_ROOT
import toml
import asyncio
from pathlib import Path
from data.data_client import DataClient as DClient


# Load futures mappings from TOML file
mapping = toml.load(APP_ROOT / 'data/futures_mappings.toml')
dclient = DClient()

def convert_cot_date_to_datetime(date_series: pd.Series) -> pd.DatetimeIndex:
    """
    Convert COT date format (YYMMDD) to proper datetime index.
    
    The COT date format uses 6 digits: YYMMDD
    - YY: 2-digit year (00-99, where 00-29 = 2000-2029, 30-99 = 1930-1999)
    - MM: 2-digit month (01-12)
    - DD: 2-digit day (01-31)
    
    Examples:
    - 101228 = 2010-12-28
    - 240115 = 2024-01-15
    - 991231 = 1999-12-31
    - 000101 = 2000-01-01
    
    Parameters:
    -----------
    date_series : pd.Series
        Series containing COT dates in YYMMDD format
        
    Returns:
    --------
    pd.DatetimeIndex
        Properly formatted datetime index
    """
    def parse_cot_date(date_val):
        """Parse individual COT date value."""
        if pd.isna(date_val):
            return pd.NaT
        
        # Convert to string and ensure 6 digits with zero padding
        date_str = str(int(date_val)).zfill(6)
        
        if len(date_str) != 6:
            return pd.NaT
        
        try:
            # Extract year, month, day
            yy = int(date_str[:2])
            mm = int(date_str[2:4])
            dd = int(date_str[4:6])
            
            # Convert 2-digit year to 4-digit year
            # COT data convention: 00-29 = 2000-2029, 30-99 = 1930-1999
            if yy <= 29:
                yyyy = 2000 + yy
            else:
                yyyy = 1900 + yy
            
            # Create datetime
            return pd.Timestamp(year=yyyy, month=mm, day=dd)
            
        except (ValueError, TypeError):
            return pd.NaT
    
    # Apply conversion to each value in the series
    converted_dates = date_series.apply(parse_cot_date)
    
    return pd.DatetimeIndex(converted_dates)

async def fetch_market_data(ticker_symbol: str, daily=True, **query_params) -> pd.DataFrame:
    """Async helper to load market data from HDF5 store."""
    def _load_market_data():
        if 'daily' not in query_params:
            query_params['daily'] = daily


        data = dclient.query_market_data(ticker_symbol, **query_params)

        return data

    
    # Run the synchronous HDF5 operation in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_market_data)

async def fetch_cot_data(ticker_symbol, **kwargs) -> pd.DataFrame:
    """Async helper to load COT data from HDF5 store."""
    def _load_cot_data():
        cot_filtered = dclient.query_by_ticker(ticker_symbol, **kwargs)

        # Convert COT date format to proper datetime index
        if 'As_of_Date_In_Form_YYMMDD' in cot_filtered.columns:
            # Fill missing values before conversion
            date_series = cot_filtered['As_of_Date_In_Form_YYMMDD'].bfill()
            datetime_index = convert_cot_date_to_datetime(date_series)
            cot_filtered.set_index(datetime_index, inplace=True)
            cot_filtered.index.name = 'datetime'
            cot_filtered.dropna(axis=1, how="all", inplace=True)
        else:
            print(f"Warning: 'As_of_Date_In_Form_YYMMDD' column not found in COT data")
            
        return cot_filtered


    # Run the synchronous HDF5 operation in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_cot_data)

async def fetch_market_cot_data(ticker_symbol: str,daily=True, resample=False, resample_period='1W', **kwargs):
    """Fetch both market data and COT data for a given ticker symbol.
    
    Args:
        ticker_symbol: The futures ticker symbol (e.g., 'CL_F')
        daily: Whether to use daily data or not
        resample: Whether to resample the data
        resample_period: Resampling period if resample=True
        
    Returns:
        pd.DataFrame: concatenated market and cot_data
    """
    try:
        # Get the COT commodity name from ticker mapping
        if ticker_symbol not in mapping['tickers']:
            raise ValueError(f"Ticker {ticker_symbol} not found in futures mappings")

        # Fetch market data and COT data concurrently
        market_data_task = fetch_market_data(ticker_symbol, daily, **kwargs)
        cot_data_task = fetch_cot_data(ticker_symbol)
        
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

def fetch_data_sync(ticker_symbol,daily=True, resample=False, **kwargs):

    results = asyncio.run(fetch_market_cot_data(ticker_symbol,daily=daily, resample=resample, **kwargs))

    return results
