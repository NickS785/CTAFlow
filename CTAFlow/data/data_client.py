
"""
DataClient: Two-store HDF5 manager with integrated COT downloader (cot_reports).

This version adds robust dtype sanitization before HDF5 writes to avoid
PyTables TypeError for mixed-object columns (e.g., numeric columns that
contain strings like ".", "-" or "—").
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

# Optional dependency used by `download_cot`
# pip install cot_reports
from CTAFlow.features.signals_processing import COTAnalyzer
from .ticker_classifier import (
    get_cot_report_type,
    get_cot_storage_path,
    get_ticker_classifier,
    is_financial_ticker,
)


try:
    import cot_reports as cot
except Exception:  # pragma: no cover - allows import of module without cot_reports installed
    cot = None

from ..config import (
    COT_DATA_PATH,
    CODE_TO_TICKER,
    DLY_DATA_PATH,
    FUTURES_MAP,
    MARKET_DATA_PATH,
    TICKER_TO_CODE,
    get_cot_code,
)
from .update_management import (
    COT_REFRESH_EVENT,
    WEEKLY_MARKET_UPDATE_EVENT,
    get_update_metadata_store,
    get_weekly_update_scheduler,
    summarize_cot_refresh,
    summarize_update_summary,
)


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

    Directories are *not* created automatically when the client is instantiated. Pass
    ``create_dirs=True`` to ``__init__`` if you want the parent directories for the
    market/COT stores created up front. Otherwise, write paths call ``_ensure_parent``
    immediately before persisting data so directories are created only when writes
    actually occur.
    """

    _weekly_update_lock = threading.Lock()
    _weekly_update_checked = False

    def __init__(
        self,
        *,
        market_path: Union[str, os.PathLike]  = None,
        cot_path: Union[str, os.PathLike] = None,
        cot_raw_key: str = "cot/raw",
        complevel: int = 9,
        complib: str = "blosc",
        create_dirs: bool = False,
    ) -> None:
        market_path = market_path or MARKET_DATA_PATH
        cot_path = cot_path or COT_DATA_PATH
        self._cot_processor: Optional[COTAnalyzer] = None
        self.market_path = Path(market_path)
        self.cot_path = Path(cot_path)
        self.cot_raw_key = cot_raw_key
        self.complevel = complevel
        self.complib = complib
        self.market_data_path = self.market_path

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

    def _get_cot_processor(self) -> COTAnalyzer:
        if self._cot_processor is None:
            self._cot_processor = COTAnalyzer()
        return self._cot_processor

    def _ensure_weekly_updates(self) -> None:
        """Run the weekly market data update if it is due."""

        cls = type(self)
        with cls._weekly_update_lock:
            if cls._weekly_update_checked:
                return
            cls._weekly_update_checked = True

        scheduler = get_weekly_update_scheduler()
        if not scheduler.is_due():
            return

        attempt_details = {"trigger": "query_market_data"}
        scheduler.mark_attempt(attempt_details)

        try:
            from .data_processor import SimpleDataProcessor
        except Exception as exc:  # pragma: no cover - defensive
            scheduler.mark_failure(f"import_error: {exc}", details=attempt_details)
            print(f"[WeeklyUpdate] Unable to import SimpleDataProcessor: {exc}")
            return

        try:
            processor = SimpleDataProcessor(self)
            summary = processor.update_all_tickers(
                dly_folder=str(DLY_DATA_PATH),
                cot_progress=False,
                record_metadata=True,
                metadata_event=WEEKLY_MARKET_UPDATE_EVENT,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            scheduler.mark_failure(str(exc), details=attempt_details)
            print(f"[WeeklyUpdate] update_all_tickers failed: {exc}")
            return

        scheduler.mark_success(summarize_update_summary(summary))

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
    
    def _get_raw_key_for_report_type(self, report_type: str) -> str:
        """
        Get the appropriate raw storage key based on COT report type.
        
        Parameters:
        -----------
        report_type : str
            The COT report type
            
        Returns:
        --------
        str
            The storage key to use for raw data
        """
        if report_type == "traders_in_financial_futures_fut":
            return "cot/tff"
        else:
            return self.cot_raw_key  # Default: "cot/raw"
    
    def _get_raw_key_for_codes(self, codes: Union[str, Sequence[str]]) -> str:
        """
        Determine the appropriate raw key based on the COT codes being queried.
        
        Parameters:
        -----------
        codes : str or sequence of str
            COT codes to check
            
        Returns:
        --------
        str
            The raw key that should contain these codes
        """
        if isinstance(codes, str):
            codes = [codes]
        
        # Check if any of the codes correspond to financial tickers
        for code in codes:
            try:
                ticker = CODE_TO_TICKER.get(code.strip().upper())
                if ticker and is_financial_ticker(ticker):
                    return "cot/tff"
            except:
                continue
        
        # Default to regular COT raw if no financial tickers found
        return self.cot_raw_key


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

    @staticmethod
    def _find_report_date_column(df: pd.DataFrame) -> str:
        """Identify the column containing the COT report date."""

        lower_map = {c.lower(): c for c in df.columns}
        preferred = [
            "report_date_as_yyyy-mm-dd",
            "report_date_as_mm_dd_yyyy",
            "report_date_as_mm_dd_yy",
            "report_date",
        ]
        for key in preferred:
            if key in lower_map:
                return lower_map[key]

        for col in df.columns:
            col_lower = col.lower()
            if "report" in col_lower and "date" in col_lower:
                return col

        for col in df.columns:
            if "date" in col.lower():
                return col

        raise KeyError("Could not detect a report date column in DataFrame.")


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
            "symbol", "contract", "group", "subgroup", "category", "type",
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

    def append_market_continuous(
        self,
        df: pd.DataFrame,
        key: str,
        *,
        data_columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Append ``df`` to ``key`` while preserving existing schema ordering."""

        if df is None or df.empty:
            return {
                "mode": "noop",
                "rows_written": 0,
                "total_rows": self._get_market_rowcount(key),
                "delta_rows": 0,
                "schema_changed": False,
                "extra_columns": [],
                "missing_columns": [],
                "removed_rows": 0,
            }

        df_sorted = df.sort_index()
        df_sorted = df_sorted[~df_sorted.index.duplicated(keep="last")]

        if not self.market_key_exists(key):
            self.write_market(df_sorted, key, replace=True, data_columns=data_columns)
            total_rows = self._get_market_rowcount(key)
            return {
                "mode": "replace",
                "rows_written": len(df_sorted),
                "total_rows": total_rows,
                "delta_rows": len(df_sorted),
                "schema_changed": False,
                "extra_columns": [],
                "missing_columns": [],
                "removed_rows": 0,
            }

        tail = self.get_market_tail(key, nrows=1)
        last_ts = tail.index[-1] if not tail.empty else None

        append_df: Optional[pd.DataFrame] = None
        existing_columns: Optional[List[str]] = None
        missing_columns: List[str] = []
        extra_columns: List[str] = []
        removed_rows = 0
        prior_rows = 0

        with pd.HDFStore(self.market_path, "a") as store:
            try:
                storer = store.get_storer(key)
            except (KeyError, ValueError):
                storer = None

            if storer is None or storer.nrows == 0:
                append_df = df_sorted
                existing_columns = list(df_sorted.columns)
            else:
                prior_rows = storer.nrows
                existing_columns = self._extract_market_columns(store, key, storer)
                if existing_columns is None:
                    append_df = df_sorted
                    existing_columns = list(df_sorted.columns)
                else:
                    extra_columns = [c for c in df_sorted.columns if c not in existing_columns]
                    if extra_columns:
                        return {
                            "mode": "schema_mismatch",
                            "rows_written": 0,
                            "total_rows": prior_rows,
                            "delta_rows": 0,
                            "schema_changed": True,
                            "extra_columns": extra_columns,
                            "missing_columns": [],
                            "removed_rows": 0,
                        }

                    missing_columns = [c for c in existing_columns if c not in df_sorted.columns]
                    append_df = df_sorted
                    if last_ts is not None:
                        overlap_mask = append_df.index <= last_ts
                        if overlap_mask.any():
                            overlap_start = append_df.index[overlap_mask].min()
                            try:
                                store.remove(key, where=f'index >= "{overlap_start.isoformat()}"')
                            except Exception:
                                overlap_start = None
                            else:
                                storer = store.get_storer(key)
                                current_rows = storer.nrows if storer is not None else 0
                                removed_rows = prior_rows - current_rows
                            if overlap_start is not None:
                                append_df = append_df.loc[append_df.index >= overlap_start]
                            else:
                                append_df = append_df.loc[append_df.index > last_ts]
                        else:
                            append_df = append_df.loc[append_df.index > last_ts]

        if append_df is None or append_df.empty:
            total_rows = self._get_market_rowcount(key)
            return {
                "mode": "noop",
                "rows_written": 0,
                "total_rows": total_rows,
                "delta_rows": 0,
                "schema_changed": False,
                "extra_columns": extra_columns,
                "missing_columns": missing_columns,
                "removed_rows": removed_rows,
            }

        if existing_columns is not None:
            append_df = append_df.reindex(columns=existing_columns)

        self.write_market(append_df, key, replace=False, data_columns=data_columns)

        total_rows = self._get_market_rowcount(key)
        delta_rows = len(append_df) - removed_rows
        mode = "append_with_revisions" if removed_rows else "append"

        return {
            "mode": mode,
            "rows_written": len(append_df),
            "total_rows": total_rows,
            "delta_rows": delta_rows,
            "schema_changed": False,
            "extra_columns": extra_columns,
            "missing_columns": missing_columns,
            "removed_rows": removed_rows,
        }

    def _get_market_rowcount(self, key: str) -> int:
        if not self.market_path.exists():
            return 0
        try:
            with pd.HDFStore(self.market_path, "r") as store:
                storer = store.get_storer(key)
                if storer is None:
                    return 0
                return storer.nrows
        except (KeyError, OSError, FileNotFoundError):
            return 0

    def get_market_rowcount(self, key: str) -> int:
        """Public wrapper returning the rowcount for ``key``."""

        return self._get_market_rowcount(key)

    @staticmethod
    def _extract_market_columns(
        store: pd.HDFStore, key: str, storer: Any
    ) -> Optional[List[str]]:
        try:
            sample = store.select(key, stop=0)
            if hasattr(sample, "columns") and len(sample.columns) > 0:
                return list(sample.columns)
        except Exception:
            pass

        try:
            colnames = list(getattr(storer.table, "colnames", []))
            return [c for c in colnames if c not in {"index", "level_0"}]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Market metadata helpers
    # ------------------------------------------------------------------
    def market_key_exists(self, key: str) -> bool:
        """Return True if a market dataset exists for the provided key."""
        if not self.market_path.exists():
            return False

        try:
            with pd.HDFStore(self.market_path, "r") as store:
                # Pandas stores keys with a leading slash
                normalized_key = key if key.startswith("/") else f"/{key}"
                return normalized_key in store.keys()
        except (OSError, FileNotFoundError):
            return False

    def get_market_tail(self, key: str, nrows: int = 1) -> pd.DataFrame:
        """
        Return the last ``nrows`` observations for a market dataset.

        Parameters
        ----------
        key : str
            Market key (e.g., ``"market/CL_F"``).
        nrows : int, default 1
            Number of rows to retrieve from the end of the dataset.
        """

        if nrows <= 0:
            raise ValueError("nrows must be positive")

        if not self.market_path.exists():
            return pd.DataFrame()

        try:
            with pd.HDFStore(self.market_path, "r") as store:
                if key not in store and f"/{key}" not in store.keys():
                    return pd.DataFrame()

                storer = store.get_storer(key)
                if storer is None or storer.nrows == 0:
                    return pd.DataFrame()

                start = max(storer.nrows - nrows, 0)
                return store.select(key, start=start, stop=storer.nrows)
        except (KeyError, OSError, FileNotFoundError):
            return pd.DataFrame()

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
                # Filter for market data keys and remove leading slash for consistency
                market_keys = []
                for key in all_keys:
                    if key.startswith('/market/'):
                        # Remove leading slash to match our usage pattern
                        clean_key = key[1:]  # Remove the leading '/'
                        market_keys.append(clean_key)
                return market_keys
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
            Can use ticker symbols (e.g., 'ZC_F', 'CL_F') or symbol names (e.g., 'CORN', 'CRUDE_OIL').
            
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
            
        daily : bool, default False
            If True, queries daily resampled data from market/daily/* keys instead of raw data.
            This is much faster for analysis requiring daily frequency data.
            
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
        
        # Query by symbol name
        >>> df = client.query_market_data('CORN', start_date='2024-06-01')
        
        # Advanced filtering
        >>> df = client.query_market_data('ZC_F', where='Volume > 1000', columns=['Open', 'High', 'Low', 'Last', 'Volume'])
        
        # Query daily data (much faster for daily analysis)
        >>> daily_df = client.query_market_data('ZC_F', daily=True, start_date='2024-01-01')
        
        # Query multiple tickers with daily data
        >>> daily_data = client.query_market_data(['ZC_F', 'CL_F'], daily=True, columns=['Open', 'High', 'Low', 'Last'])
        """

        self._ensure_weekly_updates()

        # Handle ticker input
        if isinstance(tickers, str):
            tickers = [tickers]

        market_keys: Optional[List[str]]
        if tickers is None:
            market_keys = None  # Determine after opening the store
        else:
            market_keys = []
            for ticker in tickers:
                # Ensure ticker has _F suffix (all market symbols end with _F)
                ticker_upper = ticker.upper()
                if not ticker_upper.endswith('_F'):
                    ticker_upper = f"{ticker_upper}_F"

                # Build market key
                if ticker.startswith('market/'):
                    market_keys.append(ticker)
                else:
                    if daily:
                        market_keys.append(f"market/daily/{ticker_upper}")
                    else:
                        market_keys.append(f"market/{ticker_upper}")
        
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
        results: Dict[str, pd.DataFrame] = {}
        requested_count = len(market_keys) if market_keys is not None else 0

        if not self.market_path.exists():
            return pd.DataFrame() if requested_count == 1 or combine_datasets else {}

        try:
            with pd.HDFStore(self.market_path, "r") as store:
                if market_keys is None:
                    available_keys = [
                        key.lstrip("/")
                        for key in store.keys()
                        if key.startswith("/market/")
                    ]
                    if daily:
                        market_keys = [key for key in available_keys if key.startswith("market/daily/")]
                    else:
                        market_keys = [
                            key
                            for key in available_keys
                            if key.startswith("market/") and not key.startswith("market/daily/")
                        ]
                requested_count = len(market_keys)

                for market_key in market_keys:
                    if market_key not in store:
                        print(f"Warning: {market_key} not found in available data")
                        continue

                    try:
                        df = store.select(
                            market_key,
                            where=combined_where,
                            columns=columns,
                        )

                        # Apply resampling if requested
                        if resample and len(df) > 0:
                            df = self._resample_ohlc_data(df, resample)

                        # Extract ticker name for results key
                        if daily:
                            ticker_name = market_key.replace('market/daily/', '')
                        else:
                            ticker_name = market_key.replace('market/', '')
                        results[ticker_name] = df

                    except Exception as e:
                        print(f"Error querying {market_key}: {e}")
                        continue
        except Exception as e:
            print(f"Error opening market data store: {e}")
            return pd.DataFrame() if requested_count == 1 or combine_datasets else {}

        # Return results
        if not results:
            return pd.DataFrame() if requested_count == 1 or combine_datasets else {}
        
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

    async def write_all_metrics(self, selected_cot_features=None, progress: bool = True):
        """
        Asynchronously write COT metrics for all available tickers.
        
        Parameters:
        -----------
        selected_cot_features : list, optional
            List of COT feature groups to calculate. If None, uses default set.
        progress : bool, default True
            Whether to show progress messages during processing
            
        Returns:
        --------
        dict
            Results with success/failure status for each ticker
        """
        import time
        
        if selected_cot_features is None:
            selected_cot_features = ['positioning', 'flows', 'extremes', 'market_structure']
        
        if progress:
            print(f"COT METRICS PROCESSING: Starting async processing for {len(TICKER_TO_CODE)} tickers")
            print("=" * 60)
        
        start_time = time.time()
        
        # Create tasks for all tickers
        tasks = []
        ticker_list = list(TICKER_TO_CODE.keys())
        
        for ticker in ticker_list:
            task = self.write_ticker_cot_metrics(ticker, selected_cot_features)
            tasks.append((ticker, task))
        
        # Execute all tasks concurrently
        results = {
            'success': [],
            'failed': {},
            'summary': {}
        }

        for i, (ticker, task) in enumerate(tasks):
            if progress:
                print(f"[{i+1}/{len(tasks)}] Processing {ticker}...")
            
            try:
                await task
                results['success'].append(ticker)
                if progress:
                    print(f"  ✓ Success: {ticker}")
            except Exception as e:
                results['failed'][ticker] = str(e)
                if progress:
                    print(f"  ✗ Failed: {ticker} - {str(e)}")
        
        processing_time = time.time() - start_time
        
        results['summary'] = {
            'total_tickers': len(ticker_list),
            'successful': len(results['success']),
            'failed': len(results['failed']),
            'processing_time': processing_time
        }
        
        if progress:
            print("\n" + "=" * 60)
            print("COT METRICS PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Total tickers: {results['summary']['total_tickers']}")
            print(f"Successfully processed: {results['summary']['successful']}")
            print(f"Failed: {results['summary']['failed']}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            if results['success']:
                print(f"\nSuccessful tickers: {', '.join(results['success'])}")
            
            if results['failed']:
                print(f"\nFailed tickers:")
                for ticker, error in results['failed'].items():
                    print(f"  {ticker}: {error}")
        
        return results

    def query_cot_metrics(self, ticker_symbol: str, 
                          columns: Optional[Sequence[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          where: Optional[str] = None) -> pd.DataFrame:
        """
        Query pre-calculated COT metrics for a specific ticker with filtering options.
        
        Automatically determines the correct storage path based on whether ticker
        is a financial future (stored in cot/tff/) or symbol (stored in cot/).
        
        Parameters:
        -----------
        ticker_symbol : str
            The ticker symbol (e.g., 'ZC_F', 'CL_F')
        columns : sequence of str, optional
            Specific columns to retrieve
        start_date : str, optional
            Start date filter (YYYY-MM-DD format)
        end_date : str, optional
            End date filter (YYYY-MM-DD format)
        where : str, optional
            Additional where clause for filtering
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated COT metrics
        """
        try:
            # Get appropriate storage path
            storage_path = get_cot_storage_path(ticker_symbol)
            metrics_key = f'{storage_path}/metrics'
            
            with pd.HDFStore(self.cot_path, "r") as store:
                if metrics_key not in store:
                    raise KeyError(f"COT metrics not found for {ticker_symbol} at {metrics_key}. Run write_all_metrics() first.")
                
                # Build where clause for date filtering
                where_clauses = []
                if start_date:
                    where_clauses.append(f"index >= '{start_date}'")
                if end_date:
                    where_clauses.append(f"index <= '{end_date}'")
                if where:
                    where_clauses.append(where)
                
                combined_where = ' & '.join(where_clauses) if where_clauses else None
                
                return store.select(metrics_key, columns=columns, where=combined_where)
        except Exception as e:
            raise KeyError(f"Error querying COT metrics for {ticker_symbol}: {str(e)}")
    
    def query_curve_data(
        self,
        symbol: str,
        curve_types: Optional[Union[str, Sequence[str]]] = None,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        where: Optional[str] = None,
        combine_datasets: bool = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Query futures curve data for a given symbol.
        
        Parameters:
        -----------
        symbol : str
            Base symbol (e.g., 'CL', 'ZC') - '_F' suffix will be added automatically
        curve_types : str, sequence of str, or None
            Curve data types to query. Available types:
            - 'curve': Full curve data (price)
            - 'volume_curve': Volume curve data
            - 'oi_curve': Open interest curve data
            - 'front': Front month contract data
            - 'dte': Days to expiry data
            - 'seq_curve': Sequential curve data (price)
            - 'seq_volume': Sequential volume data
            - 'seq_oi': Sequential open interest data
            - 'seq_labels': Sequential curve labels
            - 'seq_dte': Sequential days to expiry
            - 'seq_spreads': Sequential spreads data
            - 'spot': Spot price data
            - 'roll_dates': Roll dates metadata
            If None, queries all available curve types.
        start_date : str, optional
            Start date filter in 'YYYY-MM-DD' format
        end_date : str, optional
            End date filter in 'YYYY-MM-DD' format
        columns : sequence of str, optional
            Specific columns to return
        where : str, optional
            Custom pandas HDFStore where clause
        combine_datasets : bool, default False
            If True, combine all curve types into single DataFrame with MultiIndex columns
            
        Returns:
        --------
        pd.DataFrame or Dict[str, pd.DataFrame]
            Curve data matching the query criteria
            
        Examples:
        ---------
        >>> client = DataClient()
        
        # Query all curve data for crude oil
        >>> curve_data = client.query_curve_data('CL')
        
        # Query specific curve types
        >>> front_data = client.query_curve_data('ZC', curve_types=['front', 'dte'])
        
        # Query with date filtering
        >>> recent_curves = client.query_curve_data('CL', start_date='2024-01-01')
        
        # Combine all curve types into single DataFrame
        >>> combined = client.query_curve_data('ZC', combine_datasets=True)
        """
        # Ensure symbol has _F suffix
        if not symbol.endswith('_F'):
            symbol_key = f"{symbol}_F"
        else:
            symbol_key = symbol
        
        # Define available curve types and their market keys
        available_curve_types = {
            "curve": f"{symbol_key}/curve",
            "volume_curve": f"{symbol_key}/curve_volume",
            "oi_curve": f"{symbol_key}/curve_oi",
            "front": f"{symbol_key}/front",
            "dte": f"{symbol_key}/dte",
            "seq_curve": f"{symbol_key}/seq_curve",
            "seq_volume": f"{symbol_key}/seq_volume",
            "seq_oi": f"{symbol_key}/seq_oi",
            "seq_labels": f"{symbol_key}/seq_labels",
            "seq_dte": f"{symbol_key}/seq_dte",
            "seq_spreads": f"{symbol_key}/seq_spreads",
            "spot": f"{symbol_key}/spot",
            "roll_dates": f"{symbol_key}/roll_dates"
        }
        
        # Determine which curve types to query
        if curve_types is None:
            curve_types_to_query = list(available_curve_types.keys())
        elif isinstance(curve_types, str):
            curve_types_to_query = [curve_types]
        else:
            curve_types_to_query = list(curve_types)
        
        # Validate curve types
        invalid_types = [ct for ct in curve_types_to_query if ct not in available_curve_types]
        if invalid_types:
            raise ValueError(f"Invalid curve types: {invalid_types}. Available types: {list(available_curve_types.keys())}")
        
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
        
        # Query each curve type while minimizing HDF open/close cycles
        results: Dict[str, pd.DataFrame] = {}

        if not self.market_path.exists():
            return pd.DataFrame() if len(curve_types_to_query) == 1 or combine_datasets else {}

        try:
            with pd.HDFStore(self.market_path, "r") as store:
                for curve_type in curve_types_to_query:
                    market_key = f"market/{available_curve_types[curve_type]}"

                    if market_key not in store:
                        print(f"Warning: {market_key} not found in available data")
                        continue

                    try:
                        df = store.select(
                            market_key,
                            where=combined_where,
                            columns=columns
                        )
                        results[curve_type] = df
                    except KeyError:
                        print(f"Warning: {market_key} not found in available data")
                        continue
                    except Exception as e:
                        print(f"Error querying {market_key}: {e}")
                        continue
        except (OSError, FileNotFoundError) as exc:
            print(f"Error opening market data store: {exc}")
            return pd.DataFrame() if len(curve_types_to_query) == 1 or combine_datasets else {}

        # Return results
        if not results:
            return pd.DataFrame() if len(curve_types_to_query) == 1 or combine_datasets else {}
        
        if len(results) == 1 and not combine_datasets:
            return list(results.values())[0]
        elif combine_datasets:
            return self._combine_curve_datasets(results)
        else:
            return results
    
    def _combine_curve_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple curve datasets into a single DataFrame.
        
        Parameters:
        -----------
        datasets : Dict[str, pd.DataFrame]
            Dictionary of curve_type -> DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with MultiIndex columns (curve_type, column)
        """
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return list(datasets.values())[0]
        
        # Create MultiIndex columns
        combined_data = {}
        for curve_type, df in datasets.items():
            for col in df.columns:
                combined_data[(curve_type, col)] = df[col]
        
        result = pd.DataFrame(combined_data)
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=['CurveType', 'Field'])
        
        return result
    
    def list_available_curve_data(self) -> Dict[str, List[str]]:
        """
        List all available curve data by symbol.
        
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary mapping symbol to list of available curve types
        """
        available_keys = self.list_market_data()
        curve_data = {}
        
        # Look for curve-related keys
        curve_patterns = {
            'curve': '/curve',
            'volume_curve': '/curve_volume',
            'oi_curve': '/curve_oi',
            'front': '/front',
            'dte': '/dte',
            'seq_curve': '/seq_curve',
            'seq_volume': '/seq_volume',
            'seq_oi': '/seq_oi',
            'seq_labels': '/seq_labels',
            'seq_dte': '/seq_dte',
            'seq_spreads': '/seq_spreads'
        }
        
        for key in available_keys:
            if key.startswith('market/'):
                # Extract potential symbol (everything before the curve type)
                key_suffix = key.replace('market/', '')
                
                for curve_type, pattern in curve_patterns.items():
                    if key_suffix.endswith(pattern):
                        # Extract symbol by removing the pattern
                        symbol = key_suffix.replace(pattern, '')
                        if symbol not in curve_data:
                            curve_data[symbol] = []
                        curve_data[symbol].append(curve_type)
                        break
        
        return curve_data
    
    def get_curve_data_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get summary information about curve data for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get curve summary for (e.g., 'CL_F' or 'CL')
            
        Returns:
        --------
        Dict[str, Any]
            Summary information including available curve types, date ranges, etc.
        """
        # Ensure symbol has _F suffix
        if not symbol.endswith('_F'):
            symbol_key = f"{symbol}_F"
        else:
            symbol_key = symbol
        
        curve_types = {
            "curve": f"{symbol_key}/curve",
            "volume_curve": f"{symbol_key}/curve_volume",
            "oi_curve": f"{symbol_key}/curve_oi",
            "front": f"{symbol_key}/front",
            "dte": f"{symbol_key}/dte",
            "seq_curve": f"{symbol_key}/seq_curve",
            "seq_volume": f"{symbol_key}/seq_volume",
            "seq_oi": f"{symbol_key}/seq_oi",
            "seq_labels": f"{symbol_key}/seq_labels",
            "seq_dte": f"{symbol_key}/seq_dte",
            "seq_spreads": f"{symbol_key}/seq_spreads"
        }
        
        available_keys = self.list_market_data()
        summary = {
            'symbol': symbol_key,
            'available_curve_types': [],
            'curve_info': {}
        }
        
        for curve_type, curve_path in curve_types.items():
            market_key = f"market/{curve_path}"
            
            if market_key in available_keys:
                summary['available_curve_types'].append(curve_type)
                
                try:
                    # Get basic info about this curve type
                    sample = self.read_market(market_key, start=0, stop=1)
                    
                    with pd.HDFStore(self.market_path, "r") as store:
                        if market_key in store:
                            storer = store.get_storer(market_key)
                            nrows = storer.nrows if storer else 0
                            
                            # Get date range efficiently
                            if nrows > 0:
                                first_row = store.select(market_key, start=0, stop=1)
                                last_row = store.select(market_key, start=nrows-1, stop=nrows) if nrows > 1 else first_row
                                
                                summary['curve_info'][curve_type] = {
                                    'rows': nrows,
                                    'columns': list(sample.columns) if len(sample) > 0 else [],
                                    'date_range': {
                                        'start': first_row.index[0] if len(first_row) > 0 else None,
                                        'end': last_row.index[0] if len(last_row) > 0 else None
                                    }
                                }
                                
                except Exception as e:
                    summary['curve_info'][curve_type] = {'error': str(e)}
        
        return summary
    
    def display_curve_data_summary(self, symbol: str) -> None:
        """
        Display curve data summary in a readable format.
        
        Parameters:
        -----------
        symbol : str
            Symbol to display summary for
        """
        try:
            summary = self.get_curve_data_summary(symbol)
            
            print(f"=" * 80)
            print(f"CURVE DATA SUMMARY: {summary['symbol']}")
            print(f"=" * 80)
            
            available_types = summary['available_curve_types']
            if not available_types:
                print("No curve data found for this symbol.")
                return
            
            print(f"Available curve types: {len(available_types)}")
            print(f"Types: {', '.join(available_types)}")
            print()
            
            # Display details for each curve type
            for curve_type in available_types:
                info = summary['curve_info'].get(curve_type, {})
                
                if 'error' in info:
                    print(f"{curve_type.upper()}: Error - {info['error']}")
                    continue
                
                print(f"{curve_type.upper()}:")
                print(f"  Rows: {info.get('rows', 0):,}")
                print(f"  Columns: {len(info.get('columns', []))}")
                
                if 'date_range' in info:
                    date_range = info['date_range']
                    if date_range['start'] and date_range['end']:
                        start_str = date_range['start'].strftime('%Y-%m-%d') if hasattr(date_range['start'], 'strftime') else str(date_range['start'])
                        end_str = date_range['end'].strftime('%Y-%m-%d') if hasattr(date_range['end'], 'strftime') else str(date_range['end'])
                        print(f"  Date Range: {start_str} to {end_str}")
                
                # Show first few columns
                columns = info.get('columns', [])
                if columns:
                    cols_to_show = columns[:5]  # Show first 5 columns
                    cols_str = ', '.join(cols_to_show)
                    if len(columns) > 5:
                        cols_str += f" ... (+{len(columns) - 5} more)"
                    print(f"  Columns: {cols_str}")
                print()
            
            print(f"=" * 80)
            
        except Exception as e:
            print(f"Error displaying curve summary for {symbol}: {e}")
    
    def read_roll_dates(self, symbol: str) -> pd.DataFrame:
        """
        Read roll dates DataFrame for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get roll dates for (e.g., 'CL_F' or 'CL')
            
        Returns:
        --------
        pd.DataFrame
            Roll dates DataFrame with comprehensive roll metadata
            
        Raises:
        -------
        KeyError
            If roll_dates data not found for the symbol
        """
        # Ensure symbol has _F suffix
        if not symbol.endswith('_F'):
            symbol_key = f"{symbol}_F"
        else:
            symbol_key = symbol
        
        roll_dates_key = f"market/{symbol_key}/roll_dates"
        
        try:
            return self.read_market(roll_dates_key)
        except KeyError:
            raise KeyError(f"Roll dates not found for symbol '{symbol_key}'. "
                         f"Run enhanced curve manager with roll tracking enabled.")
    
    def query_roll_dates(
        self,
        symbol: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        roll_pattern: Optional[str] = None,
        min_confidence: Optional[float] = None,
        columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """
        Query roll dates with filtering options.
        
        Parameters:
        -----------
        symbol : str
            Symbol to query roll dates for
        start_date : str, optional
            Start date filter in 'YYYY-MM-DD' format
        end_date : str, optional  
            End date filter in 'YYYY-MM-DD' format
        roll_pattern : str, optional
            Filter by specific contract transition pattern (e.g., 'H->J', 'X->Z')
        min_confidence : float, optional
            Minimum confidence threshold for roll events
        columns : sequence of str, optional
            Specific columns to return
            
        Returns:
        --------
        pd.DataFrame
            Filtered roll dates DataFrame
            
        Examples:
        ---------
        >>> client = DataClient()
        
        # Get all roll dates for crude oil
        >>> rolls = client.query_roll_dates('CL')
        
        # Filter by date range
        >>> recent_rolls = client.query_roll_dates('CL', start_date='2023-01-01')
        
        # Filter by roll pattern and confidence
        >>> h_to_j_rolls = client.query_roll_dates('CL', roll_pattern='H->J', min_confidence=0.8)
        """
        # Get the base roll dates
        roll_df = self.read_roll_dates(symbol)
        
        # Build where clauses for filtering
        where_clauses = []
        
        if start_date:
            where_clauses.append(f"index >= '{start_date}'")
        if end_date:
            where_clauses.append(f"index <= '{end_date}'")
        if min_confidence is not None:
            where_clauses.append(f"confidence >= {min_confidence}")
        if roll_pattern:
            # Parse roll pattern like 'H->J'
            if '->' in roll_pattern:
                from_contract, to_contract = roll_pattern.split('->')
                where_clauses.append(f"from_contract_expiration_code == '{from_contract.strip()}'")
                where_clauses.append(f"to_contract_expiration_code == '{to_contract.strip()}'")
        
        # Apply filters
        if where_clauses:
            query_str = ' & '.join(where_clauses)
            filtered_df = roll_df.query(query_str)
        else:
            filtered_df = roll_df
        
        # Select specific columns if requested
        if columns:
            available_cols = [col for col in columns if col in filtered_df.columns]
            if available_cols:
                filtered_df = filtered_df[available_cols]
        
        return filtered_df
    
    def get_roll_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get summary statistics for roll events of a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get roll summary for
            
        Returns:
        --------
        Dict[str, Any]
            Summary statistics including roll counts, patterns, timing metrics
        """
        try:
            roll_df = self.read_roll_dates(symbol)
            
            if len(roll_df) == 0:
                return {'symbol': symbol, 'total_rolls': 0, 'message': 'No roll data available'}
            
            # Calculate summary statistics
            summary = {
                'symbol': symbol,
                'total_rolls': len(roll_df),
                'date_range': {
                    'start': roll_df.index.min(),
                    'end': roll_df.index.max()
                },
                'days_to_expiry_stats': {
                    'mean': roll_df['days_to_expiry'].mean(),
                    'std': roll_df['days_to_expiry'].std(),
                    'min': roll_df['days_to_expiry'].min(),
                    'max': roll_df['days_to_expiry'].max()
                },
                'confidence_stats': {
                    'mean': roll_df['confidence'].mean(),
                    'min': roll_df['confidence'].min(),
                    'low_confidence_count': len(roll_df[roll_df['confidence'] < 0.7])
                },
                'interval_stats': {
                    'mean_days': roll_df['interval_days'].mean(),
                    'std_days': roll_df['interval_days'].std()
                },
                'roll_patterns': roll_df.groupby(['from_contract_expiration_code', 'to_contract_expiration_code']).size().to_dict(),
                'reasons': roll_df['reason'].value_counts().to_dict()
            }
            
            return summary
            
        except KeyError:
            return {'symbol': symbol, 'error': 'Roll dates not available'}
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}
    
    def list_symbols_with_roll_dates(self) -> List[str]:
        """
        List all symbols that have roll_dates data available.
        
        Returns:
        --------
        List[str]
            List of symbols with available roll_dates data
        """
        available_keys = self.list_market_data()
        symbols_with_rolls = []
        
        for key in available_keys:
            if key.endswith('/roll_dates'):
                # Extract symbol from 'market/SYMBOL_F/roll_dates'
                symbol = key.replace('market/', '').replace('/roll_dates', '')
                symbols_with_rolls.append(symbol)
        
        return sorted(symbols_with_rolls)
    
    def list_available_cot_metrics(self) -> List[str]:
        """
        List all tickers that have COT metrics available.
        
        Returns:
        --------
        List[str]
            List of ticker symbols with available COT metrics
        """
        try:
            with pd.HDFStore(self.cot_path, "r") as store:
                all_keys = list(store.keys())
                
                # Find metrics keys and extract ticker symbols
                metrics_keys = [key for key in all_keys if key.endswith('/metrics')]
                tickers = []
                
                for key in metrics_keys:
                    # Extract ticker from 'cot/TICKER/metrics' format
                    if key.startswith('/cot/'):
                        ticker = key.replace('/cot/', '').replace('/metrics', '')
                        tickers.append(ticker)
                
                return sorted(tickers)
        except Exception:
            return []
    
    def get_cot_metrics_columns(self, ticker_symbol: str) -> List[str]:
        """
        Get the available column names for COT metrics of a specific ticker.
        
        Parameters:
        -----------
        ticker_symbol : str
            The ticker symbol (e.g., 'ZC_F', 'CL_F')
            
        Returns:
        --------
        List[str]
            List of available column names
        """
        try:
            # Get appropriate storage path
            storage_path = get_cot_storage_path(ticker_symbol)
            metrics_key = f'{storage_path}/metrics'
            
            with pd.HDFStore(self.cot_path, "r") as store:
                if metrics_key not in store:
                    raise KeyError(f"COT metrics not found for {ticker_symbol} at {metrics_key}")
                
                # Get a small sample to check columns
                sample = store.select(metrics_key, start=0, stop=1)
                return list(sample.columns)
        except Exception as e:
            raise KeyError(f"Error getting columns for {ticker_symbol}: {str(e)}")
    
    def get_cot_metrics_info(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about COT metrics for a ticker.
        
        Parameters:
        -----------
        ticker_symbol : str
            The ticker symbol (e.g., 'ZC_F', 'CL_F')
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with metrics information including columns, date range, row count, etc.
        """
        try:
            # Get appropriate storage path
            storage_path = get_cot_storage_path(ticker_symbol)
            metrics_key = f'{storage_path}/metrics'
            
            with pd.HDFStore(self.cot_path, "r") as store:
                if metrics_key not in store:
                    raise KeyError(f"COT metrics not found for {ticker_symbol} at {metrics_key}")
                
                # Get basic info
                storer = store.get_storer(metrics_key)
                total_rows = storer.nrows
                
                # Sample data to get more details
                sample = store.select(metrics_key, start=0, stop=5)
                
                # Get date range
                date_range = {}
                try:
                    if len(sample) > 0 and hasattr(sample.index, 'min'):
                        # Get actual date range
                        first_row = store.select(metrics_key, start=0, stop=1)
                        last_row = store.select(metrics_key, start=total_rows-1, stop=total_rows)
                        
                        if len(first_row) > 0 and len(last_row) > 0:
                            date_range = {
                                'start': first_row.index.min(),
                                'end': last_row.index.max()
                            }
                except:
                    date_range = {'start': 'Unknown', 'end': 'Unknown'}
                
                # Column information
                columns = list(sample.columns) if len(sample) > 0 else []
                
                # Group columns by category based on common naming patterns
                column_categories = {
                    'positioning': [col for col in columns if any(term in col.lower() for term in ['position', 'net', 'long', 'short'])],
                    'flows': [col for col in columns if any(term in col.lower() for term in ['flow', 'change', 'delta'])],
                    'extremes': [col for col in columns if any(term in col.lower() for term in ['extreme', 'percentile', 'zscore', 'outlier'])],
                    'market_structure': [col for col in columns if any(term in col.lower() for term in ['concentration', 'ratio', 'structure', 'dominance'])],
                    'other': []
                }
                
                # Assign uncategorized columns to 'other'
                categorized = set()
                for cat_cols in column_categories.values():
                    categorized.update(cat_cols)
                column_categories['other'] = [col for col in columns if col not in categorized]
                
                # Remove empty categories
                column_categories = {k: v for k, v in column_categories.items() if v}
                
                return {
                    'ticker': ticker_symbol,
                    'storage_path': storage_path,
                    'metrics_key': metrics_key,
                    'total_rows': total_rows,
                    'total_columns': len(columns),
                    'date_range': date_range,
                    'columns': columns,
                    'column_categories': column_categories,
                    'data_types': sample.dtypes.to_dict() if len(sample) > 0 else {},
                    'sample_data': sample.head(3) if len(sample) > 0 else None
                }
                
        except Exception as e:
            raise KeyError(f"Error getting metrics info for {ticker_symbol}: {str(e)}")
    
    def query_multiple_cot_metrics(self, 
                                   ticker_symbols: Sequence[str],
                                   columns: Optional[Sequence[str]] = None,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   where: Optional[str] = None,
                                   combine_datasets: bool = False) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Query COT metrics for multiple tickers.
        
        Parameters:
        -----------
        ticker_symbols : sequence of str
            List of ticker symbols to query
        columns : sequence of str, optional
            Specific columns to retrieve from all tickers
        start_date : str, optional
            Start date filter (YYYY-MM-DD format)
        end_date : str, optional
            End date filter (YYYY-MM-DD format)
        where : str, optional
            Additional where clause for filtering
        combine_datasets : bool, default False
            If True, combine all ticker data into single DataFrame with MultiIndex columns
            
        Returns:
        --------
        Dict[str, pd.DataFrame] or pd.DataFrame
            Dictionary mapping ticker to DataFrame, or combined DataFrame if combine_datasets=True
        """
        results = {}
        
        for ticker in ticker_symbols:
            try:
                df = self.query_cot_metrics(
                    ticker, 
                    columns=columns, 
                    start_date=start_date, 
                    end_date=end_date, 
                    where=where
                )
                results[ticker] = df
            except Exception as e:
                print(f"Warning: Could not query {ticker}: {e}")
                continue
        
        if not results:
            return {} if not combine_datasets else pd.DataFrame()
        
        if combine_datasets and results:
            # Create MultiIndex columns DataFrame
            combined_data = {}
            for ticker, df in results.items():
                for col in df.columns:
                    combined_data[(ticker, col)] = df[col]
            
            combined_df = pd.DataFrame(combined_data)
            combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns, names=['Ticker', 'Metric'])
            return combined_df
        
        return results
    
    def display_cot_metrics_info(self, ticker_symbol: str) -> None:
        """
        Display detailed information about COT metrics for a ticker in a readable format.
        
        Parameters:
        -----------
        ticker_symbol : str
            The ticker symbol (e.g., 'ZC_F', 'CL_F')
        """
        try:
            info = self.get_cot_metrics_info(ticker_symbol)
            
            print(f"=" * 80)
            print(f"COT METRICS INFORMATION: {info['ticker']}")
            print(f"=" * 80)
            
            print(f"Storage Path: {info['storage_path']}")
            print(f"Metrics Key: {info['metrics_key']}")
            print(f"Total Rows: {info['total_rows']:,}")
            print(f"Total Columns: {info['total_columns']}")
            
            # Date range
            date_range = info['date_range']
            if 'start' in date_range and 'end' in date_range:
                print(f"Date Range: {date_range['start']} to {date_range['end']}")
            
            # Column categories
            print(f"\n" + "-" * 80)
            print("COLUMN CATEGORIES")
            print("-" * 80)
            
            for category, columns in info['column_categories'].items():
                print(f"\n{category.upper().replace('_', ' ')} ({len(columns)} columns):")
                for col in sorted(columns):
                    data_type = info['data_types'].get(col, 'unknown')
                    print(f"  • {col:<50} ({data_type})")
            
            # Sample data
            if info['sample_data'] is not None and len(info['sample_data']) > 0:
                print(f"\n" + "-" * 80)
                print("SAMPLE DATA (First 3 rows)")
                print("-" * 80)
                print(info['sample_data'].to_string())
            
            print(f"\n" + "=" * 80)
            
        except Exception as e:
            print(f"Error displaying metrics info for {ticker_symbol}: {e}")
    
    def display_all_cot_metrics_summary(self) -> None:
        """
        Display a summary of all available COT metrics.
        """
        try:
            available_tickers = self.list_available_cot_metrics()
            
            print(f"=" * 80)
            print(f"COT METRICS SUMMARY")
            print(f"=" * 80)
            print(f"Available Tickers: {len(available_tickers)}")
            
            if not available_tickers:
                print("No COT metrics found. Run write_all_metrics() first.")
                return
            
            print(f"\n{'Ticker':<8} {'Rows':<10} {'Columns':<10} {'Date Range':<30}")
            print("-" * 70)
            
            for ticker in sorted(available_tickers):
                try:
                    info = self.get_cot_metrics_info(ticker)
                    date_range_str = f"{info['date_range'].get('start', 'Unknown')} to {info['date_range'].get('end', 'Unknown')}"
                    if len(date_range_str) > 28:
                        date_range_str = date_range_str[:25] + "..."
                    
                    print(f"{ticker:<8} {info['total_rows']:<10,} {info['total_columns']:<10} {date_range_str:<30}")
                except Exception as e:
                    print(f"{ticker:<8} {'Error':<10} {'Error':<10} {str(e)[:30]:<30}")
            
            print(f"\n" + "=" * 80)
            print("Use get_cot_metrics_info(ticker) for detailed column information")
            print("Use display_cot_metrics_info(ticker) for formatted display")
            
        except Exception as e:
            print(f"Error displaying metrics summary: {e}")

    async def write_ticker_cot_metrics(self, ticker_symbol: str, selected_cot_features=None):
        """
        Write COT metrics for a ticker using appropriate report type and storage path.
        
        Automatically determines if ticker is a financial future (uses TFF) or symbol (uses disaggregated),
        and stores data in the appropriate HDF5 path (cot/tff/ for financials, cot/ for commodities).
        """
        if selected_cot_features is None:
            selected_cot_features = ['positioning', 'flows', 'extremes', 'market_structure']
            
        # Get appropriate report type and storage path
        report_type = get_cot_report_type(ticker_symbol)
        storage_path = get_cot_storage_path(ticker_symbol)
        
        print(f"Processing {ticker_symbol}: report_type={report_type}, storage_path={storage_path}")

        df = self.query_cot_by_codes(TICKER_TO_CODE[ticker_symbol])
        if len(df) > 0:
            cot_processor = self._get_cot_processor()
            df = cot_processor.load_and_clean_data(df)
            cot_metrics = cot_processor.calculate_enhanced_cot_features(
                df, selected_cot_features=selected_cot_features
            )
            
            self._ensure_parent(self.cot_path)
            with pd.HDFStore(self.cot_path, "a") as store:
                # Store raw COT data
                store.put(storage_path, df, format="table")
                print(f"Successfully saved {storage_path} with {len(df)} rows")

                # Store calculated metrics
                metrics_path = f"{storage_path}/metrics"
                store.put(metrics_path, cot_metrics, format="table")
                print(f'Successfully saved {metrics_path} with {len(cot_metrics)} rows')
                return
        else:
            print(f"Failed to find COT data for {ticker_symbol}")
            raise KeyError(f"No COT data found for ticker {ticker_symbol}")

    
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
        # Note: This method appends to the default raw key since report type is not specified
        # For new usage, prefer using download_cot with write_raw=True for proper classification
        with pd.HDFStore(self.cot_path, "a") as store:
            (store.put if replace else store.append)(self.cot_raw_key, df2, **fmt)

    def read_cot_raw(
        self,
        *,
        where: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        report_type: str = "disaggregated_fut",
    ) -> pd.DataFrame:
        raw_key = self._get_raw_key_for_report_type(report_type)
        
        with pd.HDFStore(self.cot_path, "r") as store:
            if raw_key not in store:
                raise KeyError(f"Key '{raw_key}' not found in {self.cot_path}. Available keys: {list(store.keys())}")
            return store.select(raw_key, where=where, columns=columns, start=start, stop=stop)
    
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
        
        # Determine appropriate raw key based on codes
        raw_key = self._get_raw_key_for_codes(codes_norm)
        
        with pd.HDFStore(self.cot_path, "r") as store:
            if raw_key not in store:
                raise KeyError(f"Key '{raw_key}' not found in {self.cot_path}. Available keys: {list(store.keys())}")
            
            # Auto-detect code column if not provided
            if code_col is None:
                sample = store.select(raw_key, start=0, stop=5)
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
                return store.select(raw_key, where=where_clause, columns=columns)
            except Exception:
                # Fall back to loading all data and filtering in memory
                df = store.select(raw_key, columns=columns)
                
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
            # Try both raw keys and combine results
            available_keys = []
            if self.cot_raw_key in store:
                available_keys.append(self.cot_raw_key)
            if "cot/tff/raw" in store:
                available_keys.append("cot/tff/raw")
                
            if not available_keys:
                raise KeyError(f"No COT raw data found. Available keys: {list(store.keys())}")
            
            all_results = []
            
            for raw_key in available_keys:
                # Auto-detect code column if not provided
                if code_col is None:
                    sample = store.select(raw_key, start=0, stop=5)
                    code_col = self._find_code_col(sample)
                
                # Get unique codes with market names if requested
                if include_names:
                    columns = [code_col, 'Market_and_Exchange_Names']
                    try:
                        df = store.select(raw_key, columns=columns)
                        
                        # Get unique combinations of code and market name
                        unique_df = df[[code_col, 'Market_and_Exchange_Names']].drop_duplicates()
                        all_results.append(unique_df)
                        
                    except KeyError:
                        # Market names column not found, fall back to codes only
                        include_names = False
                        break
                        
                if not include_names:
                    df = store.select(raw_key, columns=[code_col])
                    unique_codes = self._normalize_code_series(df[code_col]).unique()
                    codes_df = pd.DataFrame({code_col: unique_codes})
                    all_results.append(codes_df)
            
            # Combine and return results
            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                if include_names:
                    result_df = combined_df[[code_col, 'Market_and_Exchange_Names']].drop_duplicates()
                else:
                    unique_codes = self._normalize_code_series(combined_df[code_col]).unique()
                    result_df = pd.DataFrame({code_col: sorted(unique_codes)})
                    
                # Normalize codes and sort
                result_df[code_col] = self._normalize_code_series(result_df[code_col])
                result_df = result_df.sort_values(code_col).reset_index(drop=True)
                return result_df
            else:
                return pd.DataFrame({code_col: []})
    
    def get_cot_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the COT data store.
        
        Returns:
        --------
        dict
            Summary with total rows, date range, available codes count, etc.
        """
        with pd.HDFStore(self.cot_path, "r") as store:
            # Check available raw keys
            available_keys = []
            if self.cot_raw_key in store:
                available_keys.append(self.cot_raw_key)
            if "cot/tff/raw" in store:
                available_keys.append("cot/tff/raw")
                
            if not available_keys:
                raise KeyError(f"No COT raw data found. Available keys: {list(store.keys())}")
            
            # Get combined info from all raw keys
            total_rows = 0
            all_samples = []
            
            for raw_key in available_keys:
                info = store.get_storer(raw_key)
                total_rows += info.nrows
                
                # Sample data to get structure info
                sample = store.select(raw_key, start=0, stop=50)
                all_samples.append(sample)
            
            # Use first sample for structure analysis
            sample = all_samples[0] if all_samples else pd.DataFrame()
            
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
        key=None,
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
        if key is None:
            # If codes are provided, determine appropriate raw key
            if codes:
                path_key = self._get_raw_key_for_codes(codes)
            else:
                path_key = self.cot_raw_key
        else:
            path_key = key

        with pd.HDFStore(self.cot_path, "r") as store:
            if path_key not in store:
                raise KeyError(f"Key '{path_key}' not found in {self.cot_path}. Available keys: {list(store.keys())}")
            
            # Build filtering logic
            where_clauses = []
            
            # Handle codes filter
            if codes:
                if isinstance(codes, str):
                    codes = [codes]
                codes_norm = [str(c).strip().upper() for c in codes]
                
                sample = store.select(path_key, start=0, stop=5)
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
                    path_key, 
                    where=where_clause, 
                    columns=columns,
                    stop=limit
                )
            except Exception:
                # Fall back to memory filtering
                df = store.select(path_key, columns=columns, stop=limit)
                
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
        Query COT data using symbol names (e.g., 'CORN', 'CRUDE_OIL').
        
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
                raise KeyError(f"No COT code mapping found for symbol '{commodity}'. "
                              f"Available commodities: {list(FUTURES_MAP['COT']['codes'].keys())}")
        
        return self.query_cot_by_codes(codes, **kwargs)
    
    def list_available_instruments(self) -> pd.DataFrame:
        """
        List all available instruments with their mappings.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with ticker, symbol, and COT code mappings
        """
        instruments = []
        for ticker, commodity in FUTURES_MAP['tickers'].items():
            code = FUTURES_MAP['COT']['codes'][commodity]
            instruments.append({
                'ticker': ticker,
                'symbol': commodity,
                'cot_code': code,
                'category': self._get_instrument_category(commodity)
            })
        
        return pd.DataFrame(instruments).sort_values(['category', 'symbol'])
    
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
        write_raw: bool = False,
        replace: bool = False,
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
            
            # Use appropriate raw key based on report type
            raw_key = self._get_raw_key_for_report_type(report_type)
            
            with pd.HDFStore(self.cot_path, "a") as store:
                (store.put if replace else store.append)(raw_key, df, **fmt)

        return df

    def download_cot_for_ticker(
        self,
        ticker_symbol: str,
        *,
        years: Optional[Union[int, Sequence[int]]] = None,
        write_raw: bool = False,
        replace: bool = False,
    ) -> pd.DataFrame:
        """
        Download COT data for a specific ticker using the appropriate report type.
        
        Automatically determines if the ticker is a financial future (uses TFF) or 
        symbol (uses disaggregated), and downloads the appropriate COT report.
        
        Parameters:
        -----------
        ticker_symbol : str
            The ticker symbol (e.g., 'ES_F', 'ZC_F')
        years : int or sequence of int, optional
            Year(s) to download. If None, downloads all available years.
        write_raw : bool, default False
            Whether to write raw data to HDF5
        replace : bool, default False  
            Whether to replace existing data
            
        Returns:
        --------
        pd.DataFrame
            Downloaded COT data
            
        Examples:
        ---------
        # Download TFF data for S&P 500 futures
        es_data = client.download_cot_for_ticker('ES_F', years=2024)
        
        # Download disaggregated data for corn futures  
        zc_data = client.download_cot_for_ticker('ZC_F', years=[2023, 2024])
        """
        # Get appropriate report type for this ticker
        report_type = get_cot_report_type(ticker_symbol)
        
        print(f"Downloading COT data for {ticker_symbol} using report type: {report_type}")
        
        # Download using the appropriate report type
        return self.download_cot(
            report_type=report_type,
            years=years,
            write_raw=write_raw,
            replace=replace
        )

    def download_all_available_cot_data(
        self,
        *,
        years: Optional[Union[int, Sequence[int]]] = None,
        write_raw: bool = True,
        replace: bool = False,
        progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download COT data for all tickers that have CSV files available.
        
        Automatically downloads disaggregated reports for commodities and 
        TFF reports for financial futures.
        
        Parameters:
        -----------
        years : int or sequence of int, optional
            Year(s) to download. If None, downloads all available years.
        write_raw : bool, default True
            Whether to write raw data to HDF5
        replace : bool, default False
            Whether to replace existing data
        progress : bool, default True
            Whether to show progress information
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping ticker symbols to their COT data
        """
        classifier = get_ticker_classifier()
        available_tickers = [ticker.ticker_symbol for ticker in classifier.get_all_tickers()]
        
        results = {}
        commodity_tickers = []
        financial_tickers = []
        
        # Group by report type
        for ticker in available_tickers:
            if is_financial_ticker(ticker):
                financial_tickers.append(ticker)
            else:
                commodity_tickers.append(ticker)
        
        if progress:
            print(f"Downloading COT data for {len(available_tickers)} tickers:")
            print(f"  - {len(commodity_tickers)} commodities (disaggregated)")
            print(f"  - {len(financial_tickers)} financials (TFF)")
        
        # Download disaggregated data for commodities
        if commodity_tickers:
            if progress:
                print(f"\nDownloading disaggregated data...")
            
            disagg_data = self.download_cot(
                report_type="disaggregated_fut",
                years=years,
                write_raw=write_raw,
                replace=replace
            )
            
            # Filter for each symbol ticker
            for ticker in commodity_tickers:
                try:
                    cot_code = TICKER_TO_CODE[ticker]
                    ticker_data = disagg_data[disagg_data['CFTC_Contract_Market_Code'] == cot_code].copy()
                    results[ticker] = ticker_data
                    if progress:
                        print(f"  {ticker}: {len(ticker_data)} rows")
                except Exception as e:
                    if progress:
                        print(f"  {ticker}: Error - {e}")
        
        # Download TFF data for financials
        if financial_tickers:
            if progress:
                print(f"\nDownloading TFF data...")
            
            tff_data = self.download_cot(
                report_type="traders_in_financial_futures_fut",
                years=years,
                write_raw=write_raw,
                replace=replace
            )
            
            # Filter for each financial ticker
            for ticker in financial_tickers:
                try:
                    cot_code = TICKER_TO_CODE[ticker]
                    ticker_data = tff_data[tff_data['CFTC_Contract_Market_Code'] == cot_code].copy()
                    results[ticker] = ticker_data
                    if progress:
                        print(f"  {ticker}: {len(ticker_data)} rows")
                except Exception as e:
                    if progress:
                        print(f"  {ticker}: Error - {e}")

        if progress:
            print(f"\nDownload complete. Successfully downloaded data for {len(results)} tickers.")

        return results

    def refresh_latest_cot_year(
        self,
        *,
        report_type: str = "disaggregated_fut",
        write_ticker_keys: bool = True,
        progress: bool = True,
    ) -> Dict[str, Any]:

        """
        Replace the most recent year of raw COT data and optionally write ticker-specific keys.

        Parameters:
        -----------
        report_type : str
            COT report type to refresh
        write_ticker_keys : bool, default True
            If True, also filter and write individual ticker data to cot/{symbol}_F keys
        progress : bool, default True
            Whether to show progress messages

        Returns:
        --------
        Dict[str, Any]
            Results including combined data and ticker-specific results
        """

        metadata_store = get_update_metadata_store()
        attempt_details = {"report_type": report_type}
        metadata_store.record_attempt(COT_REFRESH_EVENT, details=attempt_details)

        try:
            self._ensure_parent(self.cot_path)
            raw_key = self._get_raw_key_for_report_type(report_type)

            existing = pd.DataFrame()
            if self.cot_path.exists():
                with pd.HDFStore(self.cot_path, "r") as store:
                    if raw_key in store:
                        existing = store.select(raw_key)

            date_col: Optional[str] = None
            target_year = int(pd.Timestamp.today().year)

            if not existing.empty:
                try:
                    date_col = self._find_report_date_column(existing)
                    existing_dates = pd.to_datetime(existing[date_col], errors="coerce")
                    valid_dates = existing_dates.dropna()
                    if not valid_dates.empty:
                        target_year = int(valid_dates.max().year)
                except Exception:
                    date_col = None

            if progress:
                print(f"Downloading latest COT data for year {target_year} using {report_type}...")

            new_data = self.download_cot(
                report_type=report_type,
                years=target_year,
                write_raw=False,
                replace=False,
            )

            if new_data.empty:
                raise ValueError(
                    f"No COT data returned for year {target_year} using report type '{report_type}'."
                )

            if date_col is None or date_col not in new_data.columns:
                date_col = self._find_report_date_column(new_data)

            column_order = list(existing.columns) if not existing.empty else []
            for col in new_data.columns:
                if col not in column_order:
                    column_order.append(col)

            historical = existing
            if not existing.empty and date_col in existing.columns:
                existing_dates = pd.to_datetime(existing[date_col], errors="coerce")
                year_series = existing_dates.dt.year
                keep_mask = (year_series != target_year) | year_series.isna()
                historical = existing.loc[keep_mask]

            historical = historical.reindex(columns=column_order)
            new_aligned = new_data.reindex(columns=column_order)
            combined = pd.concat([historical, new_aligned], ignore_index=True, sort=False)

            try:
                code_col = self._find_code_col(combined)
            except KeyError:
                code_col = None

            if date_col in combined.columns:
                combined["_sort_date"] = pd.to_datetime(combined[date_col], errors="coerce")
                sort_cols = ["_sort_date"]
                if code_col and code_col in combined.columns:
                    sort_cols.insert(0, code_col)
                combined = combined.sort_values(by=sort_cols, kind="mergesort")
                dedup_subset = [code_col, "_sort_date"] if code_col and code_col in combined.columns else ["_sort_date"]
                combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
                if "_sort_date" in combined.columns:
                    combined = combined.drop(columns="_sort_date")

            combined = self._sanitize_for_hdf(combined)

            data_columns: Union[bool, Sequence[str]] = True
            if code_col and code_col in combined.columns:
                data_columns = [code_col]

            min_itemsize = self._min_itemsize_for_str_cols(combined)
            fmt: Dict[str, Any] = dict(
                format="table",
                data_columns=data_columns,
                complib=self.complib,
                complevel=self.complevel,
                min_itemsize=min_itemsize or None,
            )

            # Write combined data to main raw key
            with pd.HDFStore(self.cot_path, "a") as store:
                store.put(raw_key, combined, **fmt)

            if progress:
                print(f"Updated {raw_key} with {len(combined)} rows")

            results = {
                'combined_data': combined,
                'raw_key': raw_key,
                'ticker_results': {},
                'total_rows': len(combined)
            }

            # Write individual ticker keys if requested
            if write_ticker_keys and code_col:
                results['ticker_results'] = self._write_ticker_cot_keys(
                    combined, code_col, progress=progress
                )

        except Exception as exc:
            metadata_store.record_failure(COT_REFRESH_EVENT, str(exc), details=attempt_details)
            raise

        metadata_store.record_success(
            COT_REFRESH_EVENT,
            summarize_cot_refresh(results, report_type),
        )

        return results

    def _write_ticker_cot_keys(
        self,
        combined_data: pd.DataFrame,
        code_col: str,
        progress: bool = True
    ) -> Dict[str, Any]:
        """
        Filter combined COT data by ticker codes and write to individual cot/{symbol}_F keys.

        Parameters:
        -----------
        combined_data : pd.DataFrame
            Combined COT data with all tickers
        code_col : str
            Column name containing COT codes
        progress : bool
            Whether to show progress messages

        Returns:
        --------
        Dict[str, Any]
            Results for each ticker
        """
        ticker_results = {
            'success': [],
            'failed': {},
            'total_tickers': 0
        }

        if progress:
            print("\nFiltering and writing ticker-specific COT data...")

        # Get available ticker mappings
        available_tickers = []
        try:
            from ..config import TICKER_TO_CODE, CODE_TO_TICKER
            available_tickers = list(TICKER_TO_CODE.keys())
            ticker_results['total_tickers'] = len(available_tickers)
        except ImportError:
            if progress:
                print("Warning: TICKER_TO_CODE not available, skipping ticker-specific writes")
            return ticker_results

        # Process each ticker
        self._ensure_parent(self.cot_path)
        with pd.HDFStore(self.cot_path, "a") as store:
            for i, ticker in enumerate(available_tickers):
                if progress and (i + 1) % 10 == 0:
                    print(f"  Processing ticker {i+1}/{len(available_tickers)}: {ticker}")

                try:
                    cot_code = TICKER_TO_CODE[ticker]

                    # Filter data for this ticker
                    ticker_mask = combined_data[code_col].astype(str).str.strip().str.upper() == str(cot_code).strip().upper()
                    ticker_data = combined_data[ticker_mask].copy()

                    if len(ticker_data) == 0:
                        ticker_results['failed'][ticker] = "No data found for COT code"
                        continue

                    # Clean and process with COTAnalyzer
                    cot_processor = self._get_cot_processor()
                    ticker_data = cot_processor.load_and_clean_data(ticker_data)

                    # Set appropriate index if needed
                    if not isinstance(ticker_data.index, pd.DatetimeIndex):
                        date_cols = [
                            "Report_Date_as_YYYY-MM-DD",
                            "Report_Date_as_MM_DD_YYYY",
                            "Date",
                            "date",
                        ]
                        for col in date_cols:
                            if col in ticker_data.columns:
                                ticker_data.index = pd.to_datetime(ticker_data[col], errors='coerce')
                                ticker_data = ticker_data[ticker_data.index.notna()]
                                break

                    # Write to ticker-specific key
                    ticker_key = f"cot/{ticker}"
                    store.put(ticker_key, ticker_data, format="table",
                             data_columns=True, min_itemsize=self._min_itemsize_for_str_cols(ticker_data))

                    ticker_results['success'].append(ticker)

                    if progress and len(ticker_results['success']) <= 5:
                        print(f"    ✓ {ticker}: {len(ticker_data)} rows -> {ticker_key}")

                except Exception as e:
                    ticker_results['failed'][ticker] = str(e)
                    if progress and len(ticker_results['failed']) <= 3:
                        print(f"    ✗ {ticker}: {str(e)}")

        if progress:
            print(f"\nTicker filtering complete:")
            print(f"  Success: {len(ticker_results['success'])}")
            print(f"  Failed: {len(ticker_results['failed'])}")
            if ticker_results['success']:
                print(f"  Sample successful tickers: {', '.join(ticker_results['success'][:5])}")

        return ticker_results

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
        
        # Determine appropriate raw key based on codes
        raw_key = self._get_raw_key_for_codes(codes_norm)

        self._ensure_parent(self.cot_path)
        with pd.HDFStore(self.cot_path, mode="a") as store:
            if raw_key not in store:
                raise KeyError(f"Key '{raw_key}' not found in {self.cot_path}")

            # Sample to detect code column if needed
            sample = store.select(raw_key, start=0, stop=5)
            code_col_resolved = self._find_code_col(sample, code_col)

            # Determine if we can push filters to disk
            can_query = False
            try:
                _ = store.select(raw_key, where=f"{code_col_resolved} == '___PROBE___'", start=0, stop=0)
                can_query = True
            except Exception:
                can_query = False

            dfs = []
            df_all = None
            series_norm = None
            for code in codes_norm:
                if can_query:
                    df_code = store.select(raw_key, where=f"{code_col_resolved} == {repr(code)}")
                else:
                    if df_all is None:
                        df_all = store.select(raw_key)
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

        # Determine appropriate raw key based on codes
        raw_key = self._get_raw_key_for_codes(codes_norm)

        self._ensure_parent(self.cot_path)
        with pd.HDFStore(self.cot_path, mode="a") as store:
            if raw_key not in store:
                raise KeyError(f"Key '{raw_key}' not found in {self.cot_path}")

            can_query = False
            try:
                _ = store.select(raw_key, where=f"{code_col} == '___PROBE___'", start=0, stop=0)
                can_query = True
            except Exception:
                can_query = False

            for code in codes_norm:
                if can_query:
                    df_code = store.select(raw_key, where=f"{code_col} == {repr(code)}")
                else:
                    df_all = store.select(raw_key)
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

