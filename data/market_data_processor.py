#!/usr/bin/env python3
"""
Market Data Processor: Format and store CSV market data in HDF5 format.

This module processes CSV files from RAW_MARKET_DATA_PATH and stores them
in MARKET_DATA_PATH with proper data types and datetime indexing.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
import warnings
import concurrent.futures
import threading
from queue import Queue
import time

from config import RAW_MARKET_DATA_PATH, MARKET_DATA_PATH, TICKER_TO_CODE
from data.data_client import DataClient


class MarketDataProcessor:
    """
    Process and format market data CSV files for HDF5 storage.
    """
    
    def __init__(self, raw_data_path: Optional[Path] = None, hdf5_path: Optional[Path] = None):
        """
        Initialize the market data processor.
        
        Parameters:
        -----------
        raw_data_path : Path, optional
            Path to directory containing CSV files. Defaults to RAW_MARKET_DATA_PATH.
        hdf5_path : Path, optional
            Path to HDF5 file for storage. Defaults to MARKET_DATA_PATH.
        """
        self.raw_data_path = Path(raw_data_path) if raw_data_path else RAW_MARKET_DATA_PATH
        self.hdf5_path = Path(hdf5_path) if hdf5_path else MARKET_DATA_PATH
        self.client = DataClient()
        
        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self._progress_queue = Queue()
        self._results_lock = threading.Lock()
        
        # Expected CSV columns and their target dtypes
        self.expected_columns = {
            'Date': 'datetime64[ns]',
            'Time': 'str',
            'Open': 'float64',
            'High': 'float64', 
            'Low': 'float64',
            'Last': 'float64',
            'Volume': 'int64',
            'NumberOfTrades': 'int64',
            'BidVolume': 'int64',
            'AskVolume': 'int64'
        }
        
        # Alternative column names (case variations, common aliases)
        self.column_aliases = {
            'date': 'Date',
            'time': 'Time',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'last': 'Last',
            'close': 'Last',  # Sometimes 'Last' is called 'Close'
            'volume': 'Volume',
            'vol': 'Volume',
            'numberoftrades': 'NumberOfTrades',
            'trades': 'NumberOfTrades',
            'trade_count': 'NumberOfTrades',
            'bidvolume': 'BidVolume',
            'bid_vol': 'BidVolume',
            'askvolume': 'AskVolume',
            'ask_vol': 'AskVolume'
        }
    
    def discover_csv_files(self, year: str = '25') -> List[Tuple[str, Path]]:
        """
        Discover CSV files matching the pattern {XX}_{year}.csv.
        
        Parameters:
        -----------
        year : str, default '25'
            Year suffix to look for (e.g., '25' for 2025)
            
        Returns:
        --------
        List[Tuple[str, Path]]
            List of (ticker_prefix, file_path) tuples
        """
        pattern = f"*_{year}.csv"
        files = []
        
        if not self.raw_data_path.exists():
            warnings.warn(f"Raw data path does not exist: {self.raw_data_path}")
            return files
        
        for csv_file in self.raw_data_path.glob(pattern):
            # Extract ticker prefix (first 2 characters before '_') - case insensitive
            match = re.match(r'^([A-Za-z]{2})_' + year + r'\.csv$', csv_file.name, re.IGNORECASE)
            if match:
                ticker_prefix = match.group(1).upper()  # Convert to uppercase
                files.append((ticker_prefix, csv_file))
            else:
                warnings.warn(f"CSV file doesn't match expected pattern: {csv_file.name}")
        
        return sorted(files)
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to standard format.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with potentially non-standard column names
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized column names
        """
        df = df.copy()
        
        # Clean column names - remove leading/trailing whitespace and normalize
        df.columns = df.columns.str.strip().str.replace(' ', '')
        
        # Create mapping for column name normalization
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in self.column_aliases:
                column_mapping[col] = self.column_aliases[col_lower]
            elif col in self.expected_columns:
                column_mapping[col] = col
        
        # Apply the mapping
        df = df.rename(columns=column_mapping)
        
        return df
    
    def format_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proper datetime index from Date and Time columns.
        OPTIMIZED VERSION - combines date/time efficiently without string conversion.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with Date and Time columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with proper datetime index
        """
        df = df.copy()
        
        if 'Date' not in df.columns or 'Time' not in df.columns:
            raise ValueError("DataFrame must contain 'Date' and 'Time' columns")
        
        try:
            # OPTIMIZATION: Combine date and time in a single step using string concatenation
            # This avoids the expensive conversion to datetime objects and back to strings
            
            # Clean Time column once (vectorized operations)
            time_series = df['Time']
            if time_series.dtype == 'object':
                # Fast string cleaning - only if needed
                time_series = time_series.astype(str).str.strip()
                # Only clean hidden characters if they exist (check first value)
                if '\u00a0' in str(time_series.iloc[0]) or '\t' in str(time_series.iloc[0]):
                    time_series = time_series.str.replace('\u00a0', '').str.replace('\t', '')
            
            # Create datetime strings directly - much faster than converting datetime objects
            datetime_strings = df['Date'].astype(str) + ' ' + time_series.astype(str)
            
            # Single datetime parsing with most common format first
            try:
                # Try the most common format first (based on your data)
                df.index = pd.to_datetime(datetime_strings, format='%Y/%m/%d %H:%M:%S')
            except ValueError:
                try:
                    # Second most common format
                    df.index = pd.to_datetime(datetime_strings, format='%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Handle format variations
                        df.index = pd.to_datetime(datetime_strings, format='%m/%d/%Y %H:%M:%S')
                    except ValueError:
                        # Only use automatic parsing as last resort (slow)
                        df.index = pd.to_datetime(datetime_strings)
            
            df.index.name = 'datetime'
            
            # Remove the original Date and Time columns
            df = df.drop(['Date', 'Time'], axis=1)
            
        except Exception as e:
            # Enhanced error reporting
            print(f"Datetime optimization failed, trying fallback method...")
            print(f"DataFrame info during error:")
            print(f"  Shape: {df.shape}")
            print(f"  Date sample: {df['Date'].head(2).tolist()}")
            print(f"  Time sample: {df['Time'].head(2).tolist()}")
            print(f"  Combined sample: {datetime_strings.head(2).tolist() if 'datetime_strings' in locals() else 'N/A'}")
            raise ValueError(f"Error creating datetime index: {e}")
        
        # Sort by datetime index and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def validate_and_convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and convert data types for market data columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with market data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with proper data types
        """
        df = df.copy()
        
        # Define expected columns (excluding Date/Time which are handled separately)
        numeric_columns = ['Open', 'High', 'Low', 'Last', 'Volume', 'NumberOfTrades', 'BidVolume', 'AskVolume']
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    if col in ['Volume', 'NumberOfTrades', 'BidVolume', 'AskVolume']:
                        # Convert to integer, handling any NaN values
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    else:
                        # Convert to float for OHLC data
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                except Exception as e:
                    warnings.warn(f"Error converting column {col} to numeric: {e}")
                    # Keep original data if conversion fails
        
        # Validate OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Last']):
            # Check for obvious data errors
            invalid_mask = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Last']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Last'])
            )
            
            if invalid_mask.any():
                warnings.warn(f"Found {invalid_mask.sum()} rows with invalid OHLC relationships")
        
        return df
    
    def process_csv_file(self, file_path: Path, ticker_prefix: str) -> pd.DataFrame:
        """
        Process a single CSV file into properly formatted market data.
        
        Parameters:
        -----------
        file_path : Path
            Path to the CSV file
        ticker_prefix : str
            Ticker prefix (e.g., 'ZC' for corn)
            
        Returns:
        --------
        pd.DataFrame
            Processed market data with datetime index
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Remove any unnamed index columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            print(f"Processing {file_path.name}: {len(df)} rows, columns: {list(df.columns)}")
            
            # Normalize column names
            df = self.normalize_column_names(df)
            
            # Check if we have required columns
            required_cols = ['Date', 'Time', 'Open', 'High', 'Low', 'Last']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Format datetime index
            df = self.format_datetime_index(df)
            
            # Validate and convert data types
            df = self.validate_and_convert_dtypes(df)
            
            # Add metadata columns
            df['ticker_prefix'] = ticker_prefix
            
            # Try to resolve full ticker name
            try:
                # Look for matching ticker in our mappings
                possible_tickers = [ticker for ticker in TICKER_TO_CODE.keys() 
                                  if ticker.startswith(ticker_prefix)]
                if possible_tickers:
                    df['ticker_symbol'] = possible_tickers[0]  # Take first match
                else:
                    df['ticker_symbol'] = f"{ticker_prefix}_F"  # Default format
            except Exception:
                df['ticker_symbol'] = f"{ticker_prefix}_F"  # Default format
            
            print(f"Successfully processed {file_path.name}: {len(df)} rows, "
                  f"date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            raise
    
    def store_market_data(self, df: pd.DataFrame, ticker_symbol: str, replace: bool = True) -> None:
        """
        Store processed market data in HDF5 file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed market data with datetime index
        ticker_symbol : str
            Ticker symbol to use as HDF5 key
        replace : bool, default True
            Whether to replace existing data or append
        """
        try:
            key = f"market/{ticker_symbol}"
            self.client.write_market(df, key, replace=replace)
            print(f"Stored {len(df)} rows for {ticker_symbol} in {key}")
        except Exception as e:
            print(f"Error storing data for {ticker_symbol}: {e}")
            raise
    
    def _process_single_file_worker(self, file_info: Tuple[str, Path], results: Dict[str, int], replace: bool = True) -> None:
        """
        Worker function for processing a single file in a thread.
        
        Parameters:
        -----------
        file_info : Tuple[str, Path]
            Tuple of (ticker_prefix, file_path)
        results : Dict[str, int]
            Shared results dictionary (thread-safe access required)
        replace : bool
            Whether to replace existing data
        """
        ticker_prefix, file_path = file_info
        start_time = time.time()
        
        try:
            # Process the CSV file
            with self._progress_lock:
                print(f"[THREAD] Starting {ticker_prefix}: {file_path.name}")
            
            df = self.process_csv_file(file_path, ticker_prefix)
            
            # Get ticker symbol for storage
            ticker_symbol = df['ticker_symbol'].iloc[0] if 'ticker_symbol' in df.columns else f"{ticker_prefix}_F"
            
            # Thread-safe storage (HDF5 operations need to be serialized)
            with self._results_lock:
                self.store_market_data(df, ticker_symbol, replace=replace)
            
            # Update results
            with self._results_lock:
                results[ticker_symbol] = len(df)
            
            elapsed = time.time() - start_time
            with self._progress_lock:
                print(f"[THREAD] Completed {ticker_prefix}: {len(df):,} rows in {elapsed:.1f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_key = f"{ticker_prefix}_ERROR"
            with self._results_lock:
                results[error_key] = 0
            
            with self._progress_lock:
                print(f"[THREAD] Failed {ticker_prefix} after {elapsed:.1f}s: {e}")
    
    def process_all_csv_files_threaded(
        self, 
        year: str = '25', 
        replace: bool = True,
        max_workers: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process all CSV files using multi-threading for improved performance.
        
        Parameters:
        -----------
        year : str, default '25'
            Year suffix to look for
        replace : bool, default True
            Whether to replace existing data or append
        max_workers : int, optional
            Maximum number of worker threads. Defaults to CPU count.
            
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping ticker symbols to number of rows processed
        """
        results = {}
        
        # Discover CSV files
        csv_files = self.discover_csv_files(year)
        
        if not csv_files:
            print(f"No CSV files found matching pattern *_{year}.csv in {self.raw_data_path}")
            return results
        
        # Determine optimal number of workers
        if max_workers is None:
            import os
            max_workers = min(len(csv_files), os.cpu_count() or 4)
        
        print(f"Processing {len(csv_files)} CSV files using {max_workers} threads:")
        for ticker_prefix, file_path in csv_files:
            print(f"  {ticker_prefix} -> {file_path.name}")
        
        start_time = time.time()
        
        # Process files using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file_worker, file_info, results, replace): file_info
                for file_info in csv_files
            }
            
            # Wait for completion and handle any exceptions
            for future in concurrent.futures.as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    ticker_prefix, _ = file_info
                    print(f"Thread execution failed for {ticker_prefix}: {e}")
        
        total_time = time.time() - start_time
        successful = {k: v for k, v in results.items() if not k.endswith('_ERROR')}
        failed = {k: v for k, v in results.items() if k.endswith('_ERROR')}
        
        print(f"\nMulti-threaded processing completed in {total_time:.1f} seconds")
        print(f"Successfully processed: {len(successful)} files")
        print(f"Failed to process: {len(failed)} files")
        print(f"Total rows processed: {sum(successful.values()):,}")
        if successful:
            print(f"Average processing speed: {sum(successful.values()) / total_time:.0f} rows/second")
        
        return results
    
    def process_batch_with_progress(
        self,
        file_batch: List[Tuple[str, Path]],
        replace: bool = True,
        max_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Process a batch of files with real-time progress reporting.
        
        Parameters:
        -----------
        file_batch : List[Tuple[str, Path]]
            List of (ticker_prefix, file_path) tuples to process
        replace : bool, default True
            Whether to replace existing data
        max_workers : int, optional
            Maximum number of worker threads
        show_progress : bool, default True
            Whether to show progress updates
            
        Returns:
        --------
        Dict[str, int]
            Processing results
        """
        results = {}
        completed_count = 0
        total_files = len(file_batch)
        
        if max_workers is None:
            import os
            max_workers = min(total_files, os.cpu_count() or 4)
        
        if show_progress:
            print(f"Processing {total_files} files with {max_workers} threads...")
        
        start_time = time.time()
        
        def progress_callback(future):
            nonlocal completed_count
            completed_count += 1
            if show_progress:
                elapsed = time.time() - start_time
                progress = (completed_count / total_files) * 100
                eta = (elapsed / completed_count) * (total_files - completed_count) if completed_count > 0 else 0
                print(f"Progress: {completed_count}/{total_files} ({progress:.1f}%) - ETA: {eta:.1f}s")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for file_info in file_batch:
                future = executor.submit(self._process_single_file_worker, file_info, results, replace)
                future.add_done_callback(progress_callback)
                futures.append(future)
            
            # Wait for all to complete
            concurrent.futures.wait(futures)
        
        total_time = time.time() - start_time
        if show_progress:
            successful = {k: v for k, v in results.items() if not k.endswith('_ERROR')}
            print(f"Batch completed in {total_time:.1f}s - {sum(successful.values()):,} total rows")
        
        return results
    
    def process_all_csv_files(self, year: str = '25', replace: bool = True, use_threading: bool = True, max_workers: Optional[int] = None) -> Dict[str, int]:
        """
        Process all CSV files matching the pattern and store in HDF5.
        Now uses multi-threading by default for improved performance.
        
        Parameters:
        -----------
        year : str, default '25'
            Year suffix to look for
        replace : bool, default True
            Whether to replace existing data or append
        use_threading : bool, default True
            Whether to use multi-threading for processing
        max_workers : int, optional
            Maximum number of worker threads (only used if use_threading=True)
            
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping ticker symbols to number of rows processed
        """
        if use_threading:
            return self.process_all_csv_files_threaded(year=year, replace=replace, max_workers=max_workers)
        else:
            # Legacy single-threaded processing (kept for compatibility)
            return self._process_all_csv_files_sequential(year=year, replace=replace)
    
    def _process_all_csv_files_sequential(self, year: str = '25', replace: bool = True) -> Dict[str, int]:
        """
        Sequential (single-threaded) processing - kept for compatibility and debugging.
        
        Parameters:
        -----------
        year : str, default '25'
            Year suffix to look for
        replace : bool, default True
            Whether to replace existing data or append
            
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping ticker symbols to number of rows processed
        """
        results = {}
        
        # Discover CSV files
        csv_files = self.discover_csv_files(year)
        
        if not csv_files:
            print(f"No CSV files found matching pattern *_{year}.csv in {self.raw_data_path}")
            return results
        
        print(f"Found {len(csv_files)} CSV files to process (sequential mode):")
        for ticker_prefix, file_path in csv_files:
            print(f"  {ticker_prefix} -> {file_path.name}")
        
        start_time = time.time()
        
        # Process each file sequentially
        for ticker_prefix, file_path in csv_files:
            try:
                # Process the CSV file
                df = self.process_csv_file(file_path, ticker_prefix)
                
                # Get ticker symbol for storage
                ticker_symbol = df['ticker_symbol'].iloc[0] if 'ticker_symbol' in df.columns else f"{ticker_prefix}_F"
                
                # Store in HDF5
                self.store_market_data(df, ticker_symbol, replace=replace)
                
                results[ticker_symbol] = len(df)
                
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")
                results[f"{ticker_prefix}_ERROR"] = 0
        
        total_time = time.time() - start_time
        successful = {k: v for k, v in results.items() if not k.endswith('_ERROR')}
        print(f"Sequential processing completed in {total_time:.1f} seconds")
        print(f"Total rows processed: {sum(successful.values()):,}")
        
        return results
    
    def get_processing_summary(self, results: Dict[str, int]) -> None:
        """
        Print a summary of processing results.
        
        Parameters:
        -----------
        results : Dict[str, int]
            Results from process_all_csv_files
        """
        print("\n" + "="*60)
        print("MARKET DATA PROCESSING SUMMARY")
        print("="*60)
        
        successful = {k: v for k, v in results.items() if not k.endswith('_ERROR')}
        failed = {k: v for k, v in results.items() if k.endswith('_ERROR')}
        
        print(f"Successfully processed: {len(successful)} files")
        print(f"Failed to process: {len(failed)} files")
        print(f"Total rows stored: {sum(successful.values()):,}")
        
        if successful:
            print(f"\nSuccessful files:")
            for ticker, rows in successful.items():
                print(f"  {ticker}: {rows:,} rows")
        
        if failed:
            print(f"\nFailed files:")
            for ticker_error in failed.keys():
                ticker = ticker_error.replace('_ERROR', '')
                print(f"  {ticker}: Processing failed")
        
        print("="*60)


# Convenience functions for direct use
def process_market_data(year: str = '25', replace: bool = True, use_threading: bool = True, max_workers: Optional[int] = None) -> Dict[str, int]:
    """
    Process all market data CSV files for a given year using multi-threading.
    
    Parameters:
    -----------
    year : str, default '25'
        Year suffix to look for (e.g., '25' for 2025)
    replace : bool, default True
        Whether to replace existing data or append
    use_threading : bool, default True
        Whether to use multi-threading for faster processing
    max_workers : int, optional
        Maximum number of worker threads. Defaults to CPU count.
        
    Returns:
    --------
    Dict[str, int]
        Dictionary mapping ticker symbols to number of rows processed
    """
    processor = MarketDataProcessor()
    results = processor.process_all_csv_files(year=year, replace=replace, use_threading=use_threading, max_workers=max_workers)
    processor.get_processing_summary(results)
    return results

def process_market_data_fast(year: str = '25', replace: bool = True, max_workers: Optional[int] = None) -> Dict[str, int]:
    """
    Fast multi-threaded processing of all market data CSV files.
    Optimized for maximum performance.
    
    Parameters:
    -----------
    year : str, default '25'
        Year suffix to look for (e.g., '25' for 2025)
    replace : bool, default True
        Whether to replace existing data or append
    max_workers : int, optional
        Maximum number of worker threads. Defaults to CPU count.
        
    Returns:
    --------
    Dict[str, int]
        Dictionary mapping ticker symbols to number of rows processed
    """
    processor = MarketDataProcessor()
    
    # Use maximum available cores for fastest processing
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    print(f"🚀 FAST MODE: Using {max_workers} threads for maximum performance")
    results = processor.process_all_csv_files_threaded(year=year, replace=replace, max_workers=max_workers)
    processor.get_processing_summary(results)
    return results


def process_single_file(ticker_prefix: str, year: str = '25', replace: bool = True) -> Optional[int]:
    """
    Process a single market data CSV file.
    
    Parameters:
    -----------
    ticker_prefix : str
        Two-character ticker prefix (e.g., 'ZC')
    year : str, default '25'
        Year suffix
    replace : bool, default True
        Whether to replace existing data or append
        
    Returns:
    --------
    Optional[int]
        Number of rows processed, or None if failed
    """
    processor = MarketDataProcessor()
    
    # Find the specific file
    csv_files = processor.discover_csv_files(year)
    target_file = None
    
    for prefix, file_path in csv_files:
        if prefix == ticker_prefix.upper():
            target_file = file_path
            break
    
    if not target_file:
        print(f"No CSV file found for ticker prefix '{ticker_prefix}' with year '{year}'")
        return None
    
    try:
        # Process the file
        df = processor.process_csv_file(target_file, ticker_prefix)
        ticker_symbol = df['ticker_symbol'].iloc[0] if 'ticker_symbol' in df.columns else f"{ticker_prefix}_F"
        
        # Store in HDF5
        processor.store_market_data(df, ticker_symbol.upper(), replace=replace)
        
        print(f"Successfully processed {ticker_symbol}: {len(df)} rows")
        return len(df)
        
    except Exception as e:
        print(f"Failed to process {ticker_prefix}: {e}")
        return None


def query_market_data(*args, **kwargs):
    """
    Convenience function for querying market data.
    
    This is a direct wrapper around DataClient.query_market_data() for easy access.
    
    Parameters:
    -----------
    Same as DataClient.query_market_data()
    
    Returns:
    --------
    Same as DataClient.query_market_data()
    
    Examples:
    ---------
    >>> # Query single ticker
    >>> df = query_market_data('ZC_F', start_date='2024-01-01')
    
    >>> # Query multiple tickers with resampling
    >>> data = query_market_data(['ZC_F', 'CL_F'], resample='D')
    
    >>> # Advanced filtering
    >>> df = query_market_data('ZC_F', where='Volume > 1000', columns=['Open', 'High', 'Low', 'Last'])
    """
    from data.data_client import DataClient
    client = DataClient()
    return client.query_market_data(*args, **kwargs)

def create_daily_market_data(replace: bool = True, progress: bool = True):
    """
    Convenience function to create daily resampled data for all market symbols.
    
    This function loads all existing market data, resamples it to daily frequency
    using proper OHLC aggregation rules, and stores it under market/daily/{symbol} keys.
    
    Parameters:
    -----------
    replace : bool, default True
        Whether to replace existing daily data or skip if it already exists
    progress : bool, default True
        Whether to show progress messages during processing
        
    Returns:
    --------
    Dict[str, Any]
        Processing results including processed symbols, failures, and statistics
        
    Examples:
    ---------
    >>> # Create daily data for all market symbols
    >>> results = create_daily_market_data()
    >>> print(f"Processed {len(results['processed'])} symbols")
    
    >>> # Update only missing daily data
    >>> results = create_daily_market_data(replace=False)
    
    >>> # Silent processing
    >>> results = create_daily_market_data(progress=False)
    """
    from data.data_client import DataClient
    client = DataClient()
    return client.create_daily_resampled_data(replace=replace, progress=progress)


if __name__ == "__main__":
    # Example usage
    print("Market Data Processor")
    print("Usage examples:")
    print("  process_market_data('25')  # Process all 2025 data")
    print("  process_single_file('ZC', '25')  # Process corn data")
    print("  query_market_data('ZC_F', start_date='2024-01-01')  # Query market data")