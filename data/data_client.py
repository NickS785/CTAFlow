
"""
DataClient: Two-store HDF5 manager with integrated COT downloader (cot_reports).

This version adds robust dtype sanitization before HDF5 writes to avoid
PyTables TypeError for mixed-object columns (e.g., numeric columns that
contain strings like ".", "-" or "—").
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Any

import pandas as pd

# Optional dependency used by `download_cot`
# pip install cot_reports
try:
    import cot_reports as cot
except Exception:  # pragma: no cover - allows import of module without cot_reports installed
    cot = None

from config import *


class DataClient:
    """
    Manage two HDF5 stores (market + COT).

    Market store
    ------------
    - Arbitrary schema (caller-defined)
    - Enforces sorted, duplicate-free DatetimeIndex on write

    COT store
    ---------
    - Raw table at `cot_raw_key`
    - Filter by contract codes and store combined result under a *single* key
    - Optional per-code slices method (for convenience)

    All writes sanitize dtypes so numeric-like object columns are coerced to numeric,
    preventing PyTables serialization errors.
    """

    def __init__(
        self,
        *,
        market_path: Union[str, os.PathLike]  = None,
        cot_path: Union[str, os.PathLike] = None,
        cot_raw_key: str = "cot/raw",
        complevel: int = 9,
        complib: str = "blosc",
        create_dirs: bool = True,
    ) -> None:
        market_path = market_path or MARKET_DATA_PATH
        cot_path = cot_path or COT_DATA_PATH
        self.market_path = Path(market_path)
        self.cot_path = Path(cot_path)
        self.cot_raw_key = cot_raw_key
        self.complevel = complevel
        self.complib = complib

        if create_dirs:
            for p in {self.market_path.parent, self.cot_path.parent}:
                if p and not p.exists():
                    p.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_parent(fp: Path) -> None:
        if fp.parent and not fp.parent.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _min_itemsize_for_str_cols(df: pd.DataFrame, base: int = 16) -> Dict[str, int]:
        sizes: Dict[str, int] = {}
        for c in df.columns:
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object":
                max_len = int(df[c].astype(str).str.len().max() or base)
                sizes[c] = max(base, max_len)
        return sizes

    @staticmethod
    def _normalize_code_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.upper()


    @staticmethod
    def _find_code_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
        """Heuristically find a CFTC contract/market code column."""
        if explicit:
            if explicit not in df.columns:
                raise KeyError(f"code_col '{explicit}' not found in columns.")
            return explicit
        prefs = [
            "CFTC_Contract_Market_Code",
            "CFTC_Market_Code",
            "Contract_Market_Code",
            "Market_Code",
            "CFTC_SubGroup_Code",
            "Commodity_Code",
        ]
        lower_map = {c.lower(): c for c in df.columns}
        for want in prefs:
            if want.lower() in lower_map:
                return lower_map[want.lower()]
        for c in df.columns:
            if "market" in c.lower() and "code" in c.lower():
                return c
        raise KeyError(
            "Could not auto-detect a contract code column. "
            "Pass `code_col` (e.g., 'CFTC_Contract_Market_Code')."
        )


    # --- NEW: dtype sanitization to avoid PyTables mixed-object errors --------
    @staticmethod
    def _clean_numeric_strings(s: pd.Series, preserve_text: bool = False) -> pd.Series:
        # remove commas/thin spaces, normalize unicode dashes, strip
        s2 = (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("\u2009", "", regex=False)  # thin space
            .str.replace("\u202f", "", regex=False)  # narrow no-break space
            .str.replace("\u2014", "", regex=False)  # em dash
            .str.replace("\u2212", "-", regex=False)  # minus sign → hyphen-minus
            .str.strip()
        )
        
        if preserve_text:
            # For text columns, only replace obvious null indicators
            s2 = s2.replace({"NA": None, "N/A": None, "nan": None, "": None})
        else:
            # For numeric columns, also replace dots and dashes that might indicate missing data
            s2 = s2.replace({"": None, ".": None, "-": None, "—": None, "NA": None, "N/A": None, "nan": None})
        
        return s2

    @classmethod
    def _is_text_column(cls, col_name: str) -> bool:
        """
        Identify columns that should be preserved as text (Market/Exchange names, etc.)
        """
        text_indicators = [
            "market", "exchange", "name", "description", "symbol", "ticker",
            "commodity", "contract", "group", "subgroup", "category", "type",
            "region", "location", "currency", "unit"
        ]
        col_lower = col_name.lower()
        return any(indicator in col_lower for indicator in text_indicators)
    
    @classmethod
    def _sanitize_for_hdf(cls, df: pd.DataFrame, prefer_numeric: bool = True) -> pd.DataFrame:
        """
        For object-dtype columns, coerce to numeric when feasible; otherwise cast to string.
        This prevents PyTables TypeError for mixed-integer object columns.
        Preserves text columns like Market/Exchange names.
        """
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_object_dtype(out[c]):
                # Check if this is a text column that should be preserved
                is_text_col = cls._is_text_column(c)
                
                if is_text_col:
                    # For text columns, preserve content and don't convert to numeric
                    cleaned = cls._clean_numeric_strings(out[c], preserve_text=True)
                    out[c] = cleaned.astype(str)
                else:
                    # For other columns, apply existing logic
                    cleaned = cls._clean_numeric_strings(out[c], preserve_text=False)
                    if prefer_numeric:
                        num = pd.to_numeric(cleaned, errors="coerce")
                        # If ≥80% become numbers (or column name looks numeric-like), adopt numeric
                        numeric_like_name = any(
                            kw in c.lower()
                            for kw in [
                                "open", "long", "short", "spreading", "traders", "change", "pct", "percent",
                                "futonly", "futures", "oi", "interest", "positions", "_all", "_old"
                            ]
                        )
                        if numeric_like_name or (num.notna().mean() >= 0.8):
                            out[c] = num
                        else:
                            out[c] = cleaned.astype(str)
                    else:
                        out[c] = cleaned.astype(str)
        return out

    # ---------------------------------------------------------------------
    # Market I/O
    # ---------------------------------------------------------------------
    def write_market(
        self,
        df: pd.DataFrame,
        key: str,
        *,
        replace: bool = True,
        data_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self._ensure_parent(self.market_path)
        df2 = self._sanitize_for_hdf(df)  # sanitize before writing

        min_itemsize = self._min_itemsize_for_str_cols(df2)
        fmt: Dict[str, Any] = dict(
            format="table",
            data_columns=data_columns or True,
            complib=self.complib,
            complevel=self.complevel,
            min_itemsize=min_itemsize or None,
        )
        with pd.HDFStore(self.market_path, "a") as store:
            (store.put if replace else store.append)(key, df2, **fmt)

    def read_market(
        self,
        key: str,
        *,
        where: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        with pd.HDFStore(self.market_path, "r") as store:
            if key not in store:
                raise KeyError(f"Key '{key}' not found in {self.market_path}")
            return store.select(key, where=where, columns=columns, start=start, stop=stop)
    
    def list_market_data(self) -> List[str]:
        """
        List all available market data keys in the HDF5 store.
        
        Returns:
        --------
        List[str]
            List of available market data keys
        """
        if not self.market_path.exists():
            return []
        
        try:
            with pd.HDFStore(self.market_path, "r") as store:
                all_keys = list(store.keys())
                # Filter for market data keys (they start with /market/)
                market_keys = [key for key in all_keys if key.startswith('/market/')]
                # Remove the leading slash for consistency with our usage
                return [key[1:] if key.startswith('/') else key for key in market_keys]
        except Exception:
            return []
    
    def get_market_summary(self, key: str) -> Dict[str, Any]:
        """
        Get summary information about a specific market dataset.
        
        Parameters:
        -----------
        key : str
            Market data key (e.g., 'market/ZC_F')
            
        Returns:
        --------
        Dict[str, Any]
            Summary information about the dataset
        """
        try:
            df = self.read_market(key)
            
            summary = {
                'key': key,
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'start': df.index.min(),
                    'end': df.index.max()
                },
                'data_types': df.dtypes.to_dict()
            }
            
            # Add OHLC statistics if available
            ohlc_cols = ['Open', 'High', 'Low', 'Last']
            if all(col in df.columns for col in ohlc_cols):
                summary['price_stats'] = {
                    'min_price': float(df[ohlc_cols].min().min()),
                    'max_price': float(df[ohlc_cols].max().max()),
                    'avg_close': float(df['Last'].mean()) if 'Last' in df.columns else None
                }
            
            # Add volume statistics if available
            if 'Volume' in df.columns:
                summary['volume_stats'] = {
                    'total_volume': int(df['Volume'].sum()),
                    'avg_volume': float(df['Volume'].mean()),
                    'max_volume': int(df['Volume'].max())
                }
            
            return summary
            
        except Exception as e:
            return {
                'key': key,
                'error': str(e),
                'rows': 0,
                'columns': [],
                'date_range': None
            }
    
    def query_market_data(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,

        resample: Optional[str] = None,
        where: Optional[str] = None,
        combine_datasets: bool = False,
        daily: bool = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Advanced market data querying function with flexible filtering and aggregation.
        
        Parameters:
        -----------
        tickers : str, sequence of str, or None
            Ticker symbol(s) to query. If None, returns all available data.
            Can use ticker symbols (e.g., 'ZC_F', 'CL_F') or commodity names (e.g., 'CORN', 'CRUDE_OIL').
            
        start_date : str, optional
            Start date filter in 'YYYY-MM-DD' format
            
        end_date : str, optional  
            End date filter in 'YYYY-MM-DD' format
            
        columns : sequence of str, optional
            Specific columns to return. Returns all if not specified.
            
        resample : str, optional
            Pandas resampling frequency (e.g., 'D' for daily, 'W' for weekly, 'H' for hourly).
            Automatically aggregates OHLC data appropriately.
            
        where : str, optional
            Custom pandas HDFStore where clause for advanced filtering.
            
        combine_datasets : bool, default False
            If True and multiple tickers requested, combines all data into single DataFrame.
            If False, returns dict with ticker as keys.
            
        Returns:
        --------
        pd.DataFrame or Dict[str, pd.DataFrame]
            Market data matching the query criteria.
            Returns DataFrame if single ticker or combine_datasets=True.
            Returns Dict if multiple tickers and combine_datasets=False.
            
        Examples:
        ---------
        >>> client = DataClient()
        
        # Query single ticker
        >>> df = client.query_market_data('ZC_F', start_date='2024-01-01', end_date='2024-12-31')
        
        # Query multiple tickers with resampling  
        >>> data = client.query_market_data(['ZC_F', 'CL_F'], resample='D', columns=['Open', 'High', 'Low', 'Last'])
        
        # Combine multiple tickers into single DataFrame
        >>> df = client.query_market_data(['ZC_F', 'CL_F'], combine_datasets=True)
        
        # Query by commodity name
        >>> df = client.query_market_data('CORN', start_date='2024-06-01')
        
        # Advanced filtering
        >>> df = client.query_market_data('ZC_F', where='Volume > 1000', columns=['Open', 'High', 'Low', 'Last', 'Volume'])
        """
        try:
            from config import get_ticker_symbol, COMMODITY_TO_TICKER
        except ImportError:
            # Fallback if config functions are not available
            def get_ticker_symbol(ticker):
                return ticker
            COMMODITY_TO_TICKER = {}
        
        # Handle ticker input
        if tickers is None:
            # Get all available market data
            available_keys = self.list_market_data()
            market_keys = [key for key in available_keys if key.startswith('market/')]
        else:
            if isinstance(tickers, str):
                tickers = [tickers]
            
            market_keys = []
            for ticker in tickers:
                # Try to resolve ticker symbol
                try:
                    resolved_ticker = get_ticker_symbol(ticker.upper())
                    market_key = f"market/{resolved_ticker}"
                    market_keys.append(market_key)
                except (KeyError, ValueError):
                    # Try as direct market key
                    if ticker.startswith('market/'):
                        market_keys.append(ticker)
                    else:
                        market_keys.append(f"market/{ticker}")
        
        # Build where clause for date filtering
        date_conditions = []
        if start_date:
            date_conditions.append(f"index >= '{start_date}'")
        if end_date:
            date_conditions.append(f"index <= '{end_date}'")
        
        combined_where = None
        if date_conditions and where:
            combined_where = f"({' & '.join(date_conditions)}) & ({where})"
        elif date_conditions:
            combined_where = ' & '.join(date_conditions)
        elif where:
            combined_where = where
        
        # Query data
        results = {}
        available_keys = self.list_market_data()
        
        for market_key in market_keys:
            if market_key not in available_keys:
                print(f"Warning: {market_key} not found in available data")
                continue
            if daily:
                split_key = market_key.split('/')
                market_key = "".join([split_key[0], '/daily/', split_key[-1]])
                
            try:
                # Read data with filtering
                df = self.read_market(
                    market_key,
                    where=combined_where,
                    columns=columns
                )
                
                # Apply resampling if requested
                if resample and len(df) > 0:
                    df = self._resample_ohlc_data(df, resample)
                
                # Extract ticker name for results key
                ticker_name = market_key.replace('market/', '')
                results[ticker_name] = df
                
            except Exception as e:
                print(f"Error querying {market_key}: {e}")
                continue
        
        # Return results
        if not results:
            return pd.DataFrame() if len(market_keys) == 1 or combine_datasets else {}
        
        if len(results) == 1 and not combine_datasets:
            return list(results.values())[0]
        elif combine_datasets:
            return self._combine_market_datasets(results)
        else:
            return results
    
    def _resample_ohlc_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample OHLC data with appropriate aggregation rules.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Market data with OHLC columns
        freq : str
            Pandas frequency string
            
        Returns:
        --------
        pd.DataFrame
            Resampled data
        """
        if len(df) == 0:
            return df
        
        # Define aggregation rules for different column types
        agg_rules = {}
        
        # OHLC aggregation
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open']:
                agg_rules[col] = 'first'
            elif col_lower in ['high']:
                agg_rules[col] = 'max'
            elif col_lower in ['low']:
                agg_rules[col] = 'min'
            elif col_lower in ['last', 'close']:
                agg_rules[col] = 'last'
            elif col_lower in ['volume', 'bidvolume', 'askvolume']:
                agg_rules[col] = 'sum'
            elif col_lower in ['numberoftrades']:
                agg_rules[col] = 'sum'
            else:
                # Default to mean for other numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    agg_rules[col] = 'mean'
                else:
                    agg_rules[col] = 'first'
        
        return df.resample(freq).agg(agg_rules).dropna()
    
    def _combine_market_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple market datasets into a single DataFrame.
        
        Parameters:
        -----------
        datasets : Dict[str, pd.DataFrame]
            Dictionary of ticker -> DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with MultiIndex columns (ticker, column)
        """
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return list(datasets.values())[0]
        
        # Create MultiIndex columns
        combined_data = {}
        for ticker, df in datasets.items():
            for col in df.columns:
                combined_data[(ticker, col)] = df[col]
        
        result = pd.DataFrame(combined_data)
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=['Ticker', 'Field'])
        
        return result
    
    def create_daily_resampled_data(self,keys=None, replace: bool = True, progress: bool = True) -> Dict[str, Any]:
        """
        Load all market data keys, resample to daily frequency, and store as market/daily/{symbol}.
        
        This function processes all existing market data and creates daily OHLC aggregations
        with proper volume summation, storing them with the 'market/daily/' prefix for
        easy identification and retrieval.
        
        Parameters:
        -----------
        replace : bool, default True
            Whether to replace existing daily data or skip if it already exists
        progress : bool, default True
            Whether to show progress messages during processing
            
        Returns:
        --------
        Dict[str, Any]
            Processing results with the following structure:
            {
                'processed': List[str] - Successfully processed symbols
                'skipped': List[str] - Symbols skipped (already exist and replace=False)
                'failed': Dict[str, str] - Failed symbols with error messages
                'summary': Dict[str, Any] - Overall statistics
            }
            
        Examples:
        ---------
        >>> client = DataClient()
        
        # Create daily data for all market symbols
        >>> results = client.create_daily_resampled_data()
        >>> print(f"Processed {len(results['processed'])} symbols")
        
        # Update only missing daily data
        >>> results = client.create_daily_resampled_data(replace=False)
        """
        import time
        
        results = {
            'processed': [],
            'skipped': [],
            'failed': {},
            'summary': {}
        }
        
        # Get all market data keys
        all_keys = self.list_market_data() if not keys else keys
        market_keys = [key for key in all_keys if key.startswith('market/') and not key.startswith('market/daily/')]
        
        if not market_keys:
            if progress:
                print("No market data keys found to process")
            results['summary'] = {
                'total_keys': 0,
                'processed_count': 0,
                'skipped_count': 0,
                'failed_count': 0,
                'processing_time': 0
            }
            return results
        
        if progress:
            print(f"DAILY RESAMPLING: Processing {len(market_keys)} market data keys")
            print("=" * 60)
        
        start_time = time.time()
        
        for i, market_key in enumerate(market_keys):
            # Extract symbol from market key (e.g., 'market/ZC_F' -> 'ZC_F')
            symbol = market_key.replace('market/', '')
            daily_key = f"market/daily/{symbol}"
            
            if progress:
                print(f"[{i+1}/{len(market_keys)}] Processing {symbol}...")
            
            try:
                # Check if daily data already exists
                if not replace and daily_key in all_keys:
                    if progress:
                        print(f"  Skipping {symbol} - daily data already exists")
                    results['skipped'].append(symbol)
                    continue
                
                # Load the original data
                df = self.read_market(market_key)
                
                if len(df) == 0:
                    if progress:
                        print(f"  Skipping {symbol} - no data found")
                    results['skipped'].append(symbol)
                    continue
                
                original_rows = len(df)
                date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
                
                # Resample to daily using the existing method
                daily_df = self._resample_ohlc_data(df, '1D')
                
                if len(daily_df) == 0:
                    if progress:
                        print(f"  Warning: {symbol} - resampling resulted in empty dataset")
                    results['failed'][symbol] = "Resampling resulted in empty dataset"
                    continue
                
                # Store the daily data
                self.write_market(daily_df, daily_key, replace=True)
                
                daily_rows = len(daily_df)
                compression_ratio = original_rows / daily_rows if daily_rows > 0 else 0
                
                if progress:
                    print(f"  Success: {symbol}")
                    print(f"    Original: {original_rows:,} rows ({date_range})")
                    print(f"    Daily: {daily_rows:,} rows (compression: {compression_ratio:.1f}x)")
                    print(f"    Stored as: {daily_key}")
                
                results['processed'].append(symbol)
                
            except Exception as e:
                if progress:
                    print(f"  Failed: {symbol} - {str(e)}")
                results['failed'][symbol] = str(e)
                continue
        
        processing_time = time.time() - start_time
        
        # Create summary
        results['summary'] = {
            'total_keys': len(market_keys),
            'processed_count': len(results['processed']),
            'skipped_count': len(results['skipped']),
            'failed_count': len(results['failed']),
            'processing_time': processing_time
        }
        
        if progress:
            print("\n" + "=" * 60)
            print("DAILY RESAMPLING COMPLETE")
            print("=" * 60)
            print(f"Total symbols: {results['summary']['total_keys']}")
            print(f"Successfully processed: {results['summary']['processed_count']}")
            print(f"Skipped (already exist): {results['summary']['skipped_count']}")
            print(f"Failed: {results['summary']['failed_count']}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            if results['processed']:
                print(f"\nProcessed symbols: {', '.join(results['processed'])}")
            
            if results['failed']:
                print(f"\nFailed symbols:")
                for symbol, error in results['failed'].items():
                    print(f"  {symbol}: {error}")
            
            # Show available daily data keys
            updated_keys = self.list_market_data()
            daily_keys = [key for key in updated_keys if key.startswith('market/daily/')]
            print(f"\nTotal daily datasets now available: {len(daily_keys)}")
        
        return results
    
    def list_daily_market_data(self) -> List[str]:
        """
        List all available daily market data keys.
        
        Returns:
        --------
        List[str]
            List of daily market data keys (e.g., ['market/daily/ZC_F', 'market/daily/CL_F'])
        """
        all_keys = self.list_market_data()
        return [key for key in all_keys if key.startswith('market/daily/')]
    
    def get_daily_market_summary(self) -> Dict[str, Any]:
        """
        Get summary of all daily market data.
        
        Returns:
        --------
        Dict[str, Any]
            Summary including count, date ranges, and available symbols
        """
        daily_keys = self.list_daily_market_data()
        
        if not daily_keys:
            return {
                'count': 0,
                'symbols': [],
                'date_ranges': {},
                'total_rows': 0
            }
        
        summary = {
            'count': len(daily_keys),
            'symbols': [key.replace('market/daily/', '') for key in daily_keys],
            'date_ranges': {},
            'total_rows': 0
        }
        
        for key in daily_keys:
            try:
                symbol = key.replace('market/daily/', '')
                df = self.read_market(key, start=0, stop=1)  # Get first row
                if len(df) > 0:
                    # Get row count efficiently
                    with pd.HDFStore(self.market_path, "r") as store:
                        storer = store.get_storer(key)
                        nrows = storer.nrows if storer else 0
                        
                        if nrows > 0:
                            # Get date range efficiently
                            first_row = store.select(key, start=0, stop=1)
                            last_row = store.select(key, start=nrows-1, stop=nrows) if nrows > 1 else first_row
                            
                            summary['date_ranges'][symbol] = {
                                'start': first_row.index[0].strftime('%Y-%m-%d') if len(first_row) > 0 else None,
                                'end': last_row.index[0].strftime('%Y-%m-%d') if len(last_row) > 0 else None,
                                'rows': nrows
                            }
                            summary['total_rows'] += nrows
            except Exception:
                continue
        
        return summary

    # ---------------------------------------------------------------------
    # COT I/O
    # ---------------------------------------------------------------------
    def write_cot_raw(
        self,
        df: pd.DataFrame,
        *,
        code_col: Optional[str] = None,
        replace: bool = True,
    ) -> None:
        self._ensure_parent(self.cot_path)
        df2 = df.copy()
        if code_col:
            if code_col not in df2.columns:
                raise KeyError(f"code_col '{code_col}' not found in columns.")
            df2[code_col] = self._normalize_code_series(df2[code_col])
        df2 = self._sanitize_for_hdf(df2)  # sanitize before writing

        data_columns: Union[bool, Sequence[str]] = True
        if code_col and code_col in df2.columns:
            data_columns = [code_col]

        min_itemsize = self._min_itemsize_for_str_cols(df2)
        fmt: Dict[str, Any] = dict(
            format="table",
            data_columns=data_columns,
            complib=self.complib,
            complevel=self.complevel,
            min_itemsize=min_itemsize or None,
        )
        with pd.HDFStore(self.cot_path, "a") as store:
            (store.put if replace else store.append)(self.cot_raw_key, df2, **fmt)

    def read_cot_raw(
        self,
        *,
        where: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        with pd.HDFStore(self.cot_path, "r") as store:
            if self.cot_raw_key not in store:
                raise KeyError(f"Key '{self.cot_raw_key}' not found in {self.cot_path}")
            return store.select(self.cot_raw_key, where=where, columns=columns, start=start, stop=stop)
    
    def query_cot_by_codes(
        self,
        codes: Union[str, Sequence[str]],
        *,
        code_col: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query COT data for specific contract codes.
        
        Parameters:
        -----------
        codes : str or sequence of str
            Contract code(s) to filter by (e.g., '002602' for corn, ['002602', '001602'] for corn and wheat)
        code_col : str, optional
            Column name containing contract codes. Auto-detected if not provided.
        columns : sequence of str, optional
            Specific columns to return. Returns all if not specified.
        start_date : str, optional
            Start date filter in 'YYYY-MM-DD' format
        end_date : str, optional
            End date filter in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.DataFrame
            Filtered COT data
        """
        # Normalize codes input
        if isinstance(codes, str):
            codes = [codes]
        codes_norm = [str(c).strip().upper() for c in codes]
        
        with pd.HDFStore(self.cot_path, "r") as store:
            if self.cot_raw_key not in store:
                raise KeyError(f"Key '{self.cot_raw_key}' not found in {self.cot_path}")
            
            # Auto-detect code column if not provided
            if code_col is None:
                sample = store.select(self.cot_raw_key, start=0, stop=5)
                code_col = self._find_code_col(sample)
            
            # Build where clause for codes
            if len(codes_norm) == 1:
                code_filter = f"{code_col} == {repr(codes_norm[0])}"
            else:
                code_list = ", ".join(repr(c) for c in codes_norm)
                code_filter = f"{code_col} in [{code_list}]"
            
            # Add date filters if provided
            where_clauses = [code_filter]
            if start_date:
                where_clauses.append(f"index >= {repr(start_date)}")
            if end_date:
                where_clauses.append(f"index <= {repr(end_date)}")
            
            where_clause = " & ".join(where_clauses)
            
            try:
                # Try using HDF5 query first (faster)
                return store.select(self.cot_raw_key, where=where_clause, columns=columns)
            except Exception:
                # Fall back to loading all data and filtering in memory
                df = store.select(self.cot_raw_key, columns=columns)
                
                # Filter by codes
                code_series = self._normalize_code_series(df[code_col])
                mask = code_series.isin(codes_norm)
                df_filtered = df[mask]
                
                # Apply date filters if provided
                if start_date or end_date:
                    if hasattr(df_filtered.index, 'date'):
                        idx = df_filtered.index
                    else:
                        # Try to find date column
                        date_cols = [c for c in df_filtered.columns if 'date' in c.lower()]
                        if date_cols:
                            idx = pd.to_datetime(df_filtered[date_cols[0]], errors='coerce')
                        else:
                            idx = df_filtered.index
                    
                    if start_date:
                        df_filtered = df_filtered[idx >= start_date]
                    if end_date:
                        df_filtered = df_filtered[idx <= end_date]
                
                return df_filtered
    
    def list_cot_codes(
        self,
        *,
        code_col: Optional[str] = None,
        include_names: bool = True,
    ) -> pd.DataFrame:
        """
        List all available contract codes in the COT data.
        
        Parameters:
        -----------
        code_col : str, optional
            Column name containing contract codes. Auto-detected if not provided.
        include_names : bool, default True
            Whether to include Market_and_Exchange_Names for context
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with unique codes and optionally market names
        """
        with pd.HDFStore(self.cot_path, "r") as store:
            if self.cot_raw_key not in store:
                raise KeyError(f"Key '{self.cot_raw_key}' not found in {self.cot_path}")
            
            # Auto-detect code column if not provided
            if code_col is None:
                sample = store.select(self.cot_raw_key, start=0, stop=5)
                code_col = self._find_code_col(sample)
            
            # Get unique codes with market names if requested
            if include_names:
                columns = [code_col, 'Market_and_Exchange_Names']
                try:
                    df = store.select(self.cot_raw_key, columns=columns)
                    
                    # Get unique combinations of code and market name
                    unique_df = df[[code_col, 'Market_and_Exchange_Names']].drop_duplicates()
                    unique_df = unique_df.sort_values(code_col).reset_index(drop=True)
                    
                    # Normalize codes for consistency
                    unique_df[code_col] = self._normalize_code_series(unique_df[code_col])
                    
                    return unique_df
                    
                except KeyError:
                    # Market names column not found, fall back to codes only
                    include_names = False
            
            if not include_names:
                df = store.select(self.cot_raw_key, columns=[code_col])
                unique_codes = self._normalize_code_series(df[code_col]).unique()
                
                result_df = pd.DataFrame({code_col: sorted(unique_codes)})
                return result_df
    
    def get_cot_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the COT data store.
        
        Returns:
        --------
        dict
            Summary with total rows, date range, available codes count, etc.
        """
        with pd.HDFStore(self.cot_path, "r") as store:
            if self.cot_raw_key not in store:
                raise KeyError(f"Key '{self.cot_raw_key}' not found in {self.cot_path}")
            
            # Get basic info without loading all data
            info = store.get_storer(self.cot_raw_key)
            total_rows = info.nrows
            
            # Sample data to get structure info
            sample = store.select(self.cot_raw_key, start=0, stop=100)
            
            # Try to get date range
            date_range = "Unknown"
            try:
                date_cols = [c for c in sample.columns if 'date' in c.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    dates = pd.to_datetime(sample[date_col], errors='coerce').dropna()
                    if len(dates) > 0:
                        date_range = f"{dates.min().date()} to {dates.max().date()}"
            except Exception:
                pass
            
            # Get unique codes count
            codes_count = 0
            try:
                code_col = self._find_code_col(sample)
                unique_codes = self._normalize_code_series(sample[code_col]).nunique()
                codes_count = unique_codes
            except Exception:
                pass
            
            return {
                "total_rows": total_rows,
                "total_columns": len(sample.columns),
                "date_range_sample": date_range,
                "unique_codes_sample": codes_count,
                "columns": list(sample.columns),
                "data_types": sample.dtypes.value_counts().to_dict()
            }
    
    def query_cot_advanced(
        self,
        *,
        codes: Optional[Union[str, Sequence[str]]] = None,
        market_names: Optional[Union[str, Sequence[str]]] = None,
        columns: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_open_interest: Optional[float] = None,
        trader_categories: Optional[Sequence[str]] = None,
        custom_where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Advanced query for COT data with multiple filtering options.
        
        Parameters:
        -----------
        codes : str or sequence of str, optional
            Contract code(s) to filter by
        market_names : str or sequence of str, optional
            Market name patterns to search for (case-insensitive, partial matching)
        columns : sequence of str, optional
            Specific columns to return
        start_date : str, optional
            Start date filter in 'YYYY-MM-DD' format
        end_date : str, optional  
            End date filter in 'YYYY-MM-DD' format
        min_open_interest : float, optional
            Minimum open interest threshold
        trader_categories : sequence of str, optional
            Focus on specific trader categories ('NonComm', 'Comm', 'NonRept', etc.)
        custom_where : str, optional
            Custom HDF5 where clause for advanced filtering
        limit : int, optional
            Maximum number of rows to return
            
        Returns:
        --------
        pd.DataFrame
            Filtered COT data
        """
        with pd.HDFStore(self.cot_path, "r") as store:
            if self.cot_raw_key not in store:
                raise KeyError(f"Key '{self.cot_raw_key}' not found in {self.cot_path}")
            
            # Build filtering logic
            where_clauses = []
            
            # Handle codes filter
            if codes:
                if isinstance(codes, str):
                    codes = [codes]
                codes_norm = [str(c).strip().upper() for c in codes]
                
                sample = store.select(self.cot_raw_key, start=0, stop=5)
                code_col = self._find_code_col(sample)
                
                if len(codes_norm) == 1:
                    where_clauses.append(f"{code_col} == {repr(codes_norm[0])}")
                else:
                    code_list = ", ".join(repr(c) for c in codes_norm)
                    where_clauses.append(f"{code_col} in [{code_list}]")
            
            # Handle date filters
            if start_date:
                where_clauses.append(f"index >= {repr(start_date)}")
            if end_date:
                where_clauses.append(f"index <= {repr(end_date)}")
            
            # Add custom where clause
            if custom_where:
                where_clauses.append(f"({custom_where})")
            
            # Build final where clause
            where_clause = " & ".join(where_clauses) if where_clauses else None
            
            try:
                # Try HDF5 query first
                df = store.select(
                    self.cot_raw_key, 
                    where=where_clause, 
                    columns=columns,
                    stop=limit
                )
            except Exception:
                # Fall back to memory filtering
                df = store.select(self.cot_raw_key, columns=columns, stop=limit)
                
                # Apply filters in memory
                if codes:
                    code_col = self._find_code_col(df)
                    code_series = self._normalize_code_series(df[code_col])
                    df = df[code_series.isin(codes_norm)]
                
                # Date filters
                if start_date or end_date:
                    date_cols = [c for c in df.columns if 'date' in c.lower()]
                    if date_cols:
                        idx = pd.to_datetime(df[date_cols[0]], errors='coerce')
                        if start_date:
                            df = df[idx >= start_date]
                        if end_date:
                            df = df[idx <= end_date]
            
            # Apply post-query filters that require full data access
            if market_names:
                if isinstance(market_names, str):
                    market_names = [market_names]
                
                if 'Market_and_Exchange_Names' in df.columns:
                    mask = pd.Series(False, index=df.index)
                    for name_pattern in market_names:
                        pattern_mask = df['Market_and_Exchange_Names'].str.contains(
                            name_pattern, case=False, na=False, regex=False
                        )
                        mask |= pattern_mask
                    df = df[mask]
            
            if min_open_interest:
                oi_cols = [c for c in df.columns if 'open_interest' in c.lower() and 'all' in c.lower()]
                if oi_cols:
                    oi_col = oi_cols[0]
                    df = df[pd.to_numeric(df[oi_col], errors='coerce') >= min_open_interest]
            
            if trader_categories:
                # Filter columns to focus on specific trader categories
                category_cols = []
                for category in trader_categories:
                    matching_cols = [c for c in df.columns if category.lower() in c.lower()]
                    category_cols.extend(matching_cols)
                
                if category_cols:
                    # Keep essential columns + category-specific columns
                    essential_cols = [c for c in df.columns if any(
                        essential in c.lower() 
                        for essential in ['market', 'date', 'code', 'open_interest_all']
                    )]
                    keep_cols = list(set(essential_cols + category_cols))
                    available_cols = [c for c in keep_cols if c in df.columns]
                    if available_cols:
                        df = df[available_cols]
            
            return df
    
    def query_by_ticker(
        self,
        tickers: Union[str, Sequence[str]],
        **kwargs
    ) -> pd.DataFrame:
        """
        Query COT data using ticker symbols (e.g., 'ZC_F', 'CL_F').
        
        Parameters:
        -----------
        tickers : str or sequence of str
            Ticker symbol(s) to query
        **kwargs
            Additional arguments passed to query_cot_by_codes
            
        Returns:
        --------
        pd.DataFrame
            Filtered COT data
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Convert tickers to COT codes
        codes = []
        for ticker in tickers:
            try:
                code = get_cot_code(ticker.upper())
                codes.append(code)
            except KeyError:
                raise KeyError(f"No COT code mapping found for ticker '{ticker}'. "
                              f"Available tickers: {list(TICKER_TO_CODE.keys())}")
        
        return self.query_cot_by_codes(codes, **kwargs)
    
    def query_by_commodity(
        self,
        commodities: Union[str, Sequence[str]], 
        **kwargs
    ) -> pd.DataFrame:
        """
        Query COT data using commodity names (e.g., 'CORN', 'CRUDE_OIL').
        
        Parameters:
        -----------
        commodities : str or sequence of str
            Commodity name(s) to query
        **kwargs
            Additional arguments passed to query_cot_by_codes
            
        Returns:
        --------
        pd.DataFrame
            Filtered COT data
        """
        if isinstance(commodities, str):
            commodities = [commodities]
        
        # Convert commodities to COT codes
        codes = []
        for commodity in commodities:
            try:
                code = get_cot_code(commodity.upper())
                codes.append(code)
            except KeyError:
                raise KeyError(f"No COT code mapping found for commodity '{commodity}'. "
                              f"Available commodities: {list(FUTURES_MAP['COT']['codes'].keys())}")
        
        return self.query_cot_by_codes(codes, **kwargs)
    
    def list_available_instruments(self) -> pd.DataFrame:
        """
        List all available instruments with their mappings.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with ticker, commodity, and COT code mappings
        """
        instruments = []
        for ticker, commodity in FUTURES_MAP['tickers'].items():
            code = FUTURES_MAP['COT']['codes'][commodity]
            instruments.append({
                'ticker': ticker,
                'commodity': commodity,
                'cot_code': code,
                'category': self._get_instrument_category(commodity)
            })
        
        return pd.DataFrame(instruments).sort_values(['category', 'commodity'])
    
    @staticmethod
    def _get_instrument_category(commodity: str) -> str:
        """Get instrument category for grouping."""
        commodity_lower = commodity.lower()
        
        if any(grain in commodity_lower for grain in ['corn', 'wheat', 'soybean', 'oats', 'rice']):
            return 'Agricultural - Grains'
        elif any(livestock in commodity_lower for livestock in ['cattle', 'hogs', 'milk', 'butter', 'cheese']):
            return 'Agricultural - Livestock/Dairy'
        elif any(soft in commodity_lower for soft in ['coffee', 'sugar', 'cocoa', 'cotton', 'orange']):
            return 'Agricultural - Softs'
        elif any(energy in commodity_lower for energy in ['crude', 'oil', 'gasoline', 'heating', 'gas', 'brent']):
            return 'Energy'
        elif any(metal in commodity_lower for metal in ['gold', 'silver', 'platinum', 'palladium', 'copper', 'aluminum']):
            return 'Metals'
        elif any(financial in commodity_lower for financial in ['treasury', 'eurodollar', 'bond', 'notes']):
            return 'Financial'
        elif any(currency in commodity_lower for currency in ['eur', 'gbp', 'jpy', 'cad', 'aud', 'chf', 'usd']):
            return 'Currency'
        elif any(equity in commodity_lower for equity in ['sp_500', 'nasdaq', 'dow', 'russell', 'vix']):
            return 'Equity Indices'
        else:
            return 'Other'

    # ---------------------------------------------------------------------
    # COT downloading via cot_reports
    # ---------------------------------------------------------------------
    def download_cot(
        self,
        *,
        report_type: str = "disaggregated_fut",
        years: Optional[Union[int, Sequence[int]]] = None,
        code_col: Optional[str] = None,
        write_raw: bool = True,
        replace: bool = True,
    ) -> pd.DataFrame:
        if cot is None:
            raise ImportError(
                "cot_reports is not installed. Run `pip install cot_reports` to use download_cot()."
            )

        if years is None:
            df = cot.cot_all(cot_report_type=report_type)
        elif isinstance(years, int):
            df = cot.cot_year(year=years, cot_report_type=report_type)
        else:
            frames = [cot.cot_year(year=y, cot_report_type=report_type) for y in years]
            df = pd.concat(frames, ignore_index=True)

        # Normalize code column if present/known
        try:
            code_col_resolved = self._find_code_col(df, code_col)
            df[code_col_resolved] = self._normalize_code_series(df[code_col_resolved])
        except KeyError:
            code_col_resolved = None

        # Sanitize dtypes before writing
        df = self._sanitize_for_hdf(df)

        if write_raw:
            self._ensure_parent(self.cot_path)
            data_columns: Union[bool, Sequence[str]] = True
            if code_col_resolved:
                data_columns = [code_col_resolved]
            min_itemsize = self._min_itemsize_for_str_cols(df)
            fmt: Dict[str, Any] = dict(
                format="table",
                data_columns=data_columns,
                complib=self.complib,
                complevel=self.complevel,
                min_itemsize=min_itemsize or None,
            )
            with pd.HDFStore(self.cot_path, "a") as store:
                (store.put if replace else store.append)(self.cot_raw_key, df, **fmt)

        return df

    # ---------------------------------------------------------------------
    # COT: store combined codeset under a caller-provided key
    # ---------------------------------------------------------------------
    def store_cot_codeset(
        self,
        dest_key: str,
        codes: Sequence[Union[str, int]],
        *,
        code_col: Optional[str] = None,
        replace: bool = True,
        keep_cols: Optional[Sequence[str]] = None,
    ) -> str:
        if not codes:
            raise ValueError("`codes` must be a non-empty sequence.")

        codes_norm = [str(c).strip().upper() for c in codes]

        with pd.HDFStore(self.cot_path, mode="a") as store:
            if self.cot_raw_key not in store:
                raise KeyError(f"Key '{self.cot_raw_key}' not found in {self.cot_path}")

            # Sample to detect code column if needed
            sample = store.select(self.cot_raw_key, start=0, stop=5)
            code_col_resolved = self._find_code_col(sample, code_col)

            # Determine if we can push filters to disk
            can_query = False
            try:
                _ = store.select(self.cot_raw_key, where=f"{code_col_resolved} == '___PROBE___'", start=0, stop=0)
                can_query = True
            except Exception:
                can_query = False

            dfs = []
            df_all = None
            series_norm = None
            for code in codes_norm:
                if can_query:
                    df_code = store.select(self.cot_raw_key, where=f"{code_col_resolved} == {repr(code)}")
                else:
                    if df_all is None:
                        df_all = store.select(self.cot_raw_key)
                        series_norm = self._normalize_code_series(df_all[code_col_resolved])
                    df_code = df_all.loc[series_norm == code]

                if not df_code.empty:
                    dfs.append(df_code)

            if not dfs:
                raise ValueError(f"No rows found in COT for codes: {codes_norm}")

            df_out = pd.concat(dfs).sort_index()
            df_out = df_out[~df_out.index.duplicated(keep="last")]

            if keep_cols:
                missing = [c for c in keep_cols if c not in df_out.columns]
                if missing:
                    raise KeyError(f"Missing columns requested in keep_cols: {missing}")
                df_out = df_out.loc[:, list(keep_cols)]

            # Sanitize before write
            df_out = self._sanitize_for_hdf(df_out)

            min_itemsize = self._min_itemsize_for_str_cols(df_out)
            fmt: Dict[str, Any] = dict(
                format="table",
                data_columns=True,
                complib=self.complib,
                complevel=self.complevel,
                min_itemsize=min_itemsize or None,
            )
            (store.put if replace else store.append)(dest_key, df_out, **fmt)

            # Attach metadata
            st = store.get_storer(dest_key)
            st.attrs.codes = codes_norm
            st.attrs.code_col = code_col_resolved

        return dest_key

    def read_cot_codeset(
        self,
        key: str,
        *,
        where: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> pd.DataFrame:
        with pd.HDFStore(self.cot_path, "r") as store:
            if key not in store:
                raise KeyError(f"Key '{key}' not found in {self.cot_path}")
            return store.select(key, where=where, columns=columns, start=start, stop=stop)

    def get_cot_codeset_meta(self, key: str) -> Dict[str, Any]:
        with pd.HDFStore(self.cot_path, "r") as store:
            if key not in store:
                raise KeyError(f"Key '{key}' not found in {self.cot_path}")
            st = store.get_storer(key)
            return {
                "codes": getattr(st.attrs, "codes", None),
                "code_col": getattr(st.attrs, "code_col", None),
            }

    # ---------------------------------------------------------------------
    # (Optional) Per-code writer (kept for convenience)
    # ---------------------------------------------------------------------
    def filter_and_store_cot_by_codes(
        self,
        codes: Sequence[Union[str, int]],
        *,
        code_col: str = "CFTC_Contract_Market_Code",
        dest_prefix: str = "cot/by_code",
        replace: bool = True,
        keep_cols: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        codes_norm = [str(c).strip().upper() for c in codes]
        written: Dict[str, str] = {}

        with pd.HDFStore(self.cot_path, mode="a") as store:
            if self.cot_raw_key not in store:
                raise KeyError(f"Key '{self.cot_raw_key}' not found in {self.cot_path}")

            can_query = False
            try:
                _ = store.select(self.cot_raw_key, where=f"{code_col} == '___PROBE___'", start=0, stop=0)
                can_query = True
            except Exception:
                can_query = False

            for code in codes_norm:
                if can_query:
                    df_code = store.select(self.cot_raw_key, where=f"{code_col} == {repr(code)}")
                else:
                    df_all = store.select(self.cot_raw_key)
                    s = self._normalize_code_series(df_all[code_col])
                    df_code = df_all.loc[s == code]

                if df_code.empty:
                    continue

                if keep_cols:
                    missing = [c for c in keep_cols if c not in df_code.columns]
                    if missing:
                        raise KeyError(f"Missing columns requested in keep_cols: {missing}")
                    df_code = df_code.loc[:, list(keep_cols)]

                df_code = df_code[~df_code.index.duplicated(keep="last")].sort_index()
                df_code = self._sanitize_for_hdf(df_code)

                dest_key = f"{dest_prefix}/{code}"
                min_itemsize = self._min_itemsize_for_str_cols(df_code)
                fmt: Dict[str, Any] = dict(
                    format="table",
                    data_columns=True,
                    complib=self.complib,
                    complevel=self.complevel,
                    min_itemsize=min_itemsize or None,
                )
                (store.put if replace else store.append)(dest_key, df_code, **fmt)
                written[code] = dest_key

        return written
