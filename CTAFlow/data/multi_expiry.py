import pandas as pd
import numpy as np
import mmap
import os
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Generator, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: Try to import TickerClassifier and expiry rules
try:
    from CTAFlow.data.ticker_classifier import get_ticker_classifier, TickerCategory
    from CTAFlow.data import read_exported_df
    from CTAFlow.data.contract_expiry_rules import calculate_expiry, get_roll_buffer_days

    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False

# Optional: Try to import SierraPy for SCID gap patching
try:
    from sierrapy.parser.async_scid_reader import ScidReader
    HAS_SIERRAPY = True
except ImportError:
    HAS_SIERRAPY = False

# Enable Copy-on-Write for Pandas 2.0+ (Crucial for zero-copy slicing)
pd.options.mode.copy_on_write = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemEfficient")

MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}


class CsvDeliveryYearIndexer:
    """
    Scans a CSV file and creates byte offsets for each delivery year.
    Uses the same logic as spread_loader.py to determine delivery years.

    For a contract month, if trading month > contract month, it belongs to next year's delivery.
    Example: Feb contract (month=2) trading in March (month=3) delivers in next year's Feb.
    """

    def build_index(self, filepath: str, contract_month_int: int) -> Tuple[bytes, Dict[int, Tuple[int, int]]]:
        """
        Build index mapping delivery_year -> (byte_offset, byte_length)

        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        contract_month_int : int
            Contract month (1=Jan, 2=Feb, ..., 12=Dec)

        Returns:
        --------
        Tuple[bytes, Dict[int, Tuple[int, int]]]
            (header_bytes, {delivery_year: (offset, length)})
        """
        index_map = {}

        if not os.path.exists(filepath):
            return b"", {}

        with open(filepath, "r+b") as f:
            # Memory map the file for zero-copy scanning
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # 1. Read Header
                header_line = mm.readline()
                header_str = header_line.decode('utf-8').lower()
                headers = header_str.split(',')

                # Find Date Column
                try:
                    date_col_idx = next(i for i, h in enumerate(headers) if 'date' in h or 'time' in h)
                except StopIteration:
                    logger.error(f"No date column in {filepath}")
                    return header_line, {}

                chunk_start = len(header_line)
                current_delivery_year = -1

                # 2. Linear Scan (Optimized)
                while True:
                    line = mm.readline()
                    if not line: break

                    try:
                        # Parse date string manually (faster than pandas here)
                        line_str = line.decode('utf-8')
                        parts = line_str.split(',')
                        date_str = parts[date_col_idx].strip()

                        # Heuristic: YYYY-MM-DD or MM/DD/YYYY or M/D/YYYY
                        trading_year = None
                        trading_month = None

                        if '/' in date_str:  # MM/DD/YYYY or M/D/YYYY
                            date_parts = date_str.split('/')
                            if len(date_parts) >= 3:
                                trading_month = int(date_parts[0])
                                year_str = date_parts[2]
                                # Handle 2-digit or 4-digit years
                                if len(year_str) == 2:
                                    year_int = int(year_str)
                                    trading_year = 2000 + year_int if year_int < 50 else 1900 + year_int
                                else:
                                    trading_year = int(year_str)
                        elif '-' in date_str:  # YYYY-MM-DD or YYYY-M-D
                            date_parts = date_str.split('-')
                            if len(date_parts) >= 3:
                                year_str = date_parts[0]
                                # Handle 2-digit or 4-digit years
                                if len(year_str) == 2:
                                    year_int = int(year_str)
                                    trading_year = 2000 + year_int if year_int < 50 else 1900 + year_int
                                else:
                                    trading_year = int(year_str)
                                trading_month = int(date_parts[1])

                        # Skip invalid dates
                        if trading_year is None or trading_month is None:
                            continue
                        if trading_year < 1900 or trading_year > 2100:
                            continue

                        # Calculate delivery year using spread_loader logic
                        # If trading month > contract month, it belongs to next year's delivery
                        delivery_year = trading_year
                        if trading_month > contract_month_int:
                            delivery_year += 1

                        if delivery_year != current_delivery_year:
                            # Close previous delivery year block
                            if current_delivery_year != -1:
                                prev_len = (mm.tell() - len(line)) - chunk_start
                                index_map[current_delivery_year] = (chunk_start, prev_len)

                            # Start new block
                            current_delivery_year = delivery_year
                            chunk_start = mm.tell() - len(line)

                    except (ValueError, IndexError):
                        continue

                # Close final block
                if current_delivery_year != -1:
                    index_map[current_delivery_year] = (chunk_start, mm.tell() - chunk_start)

                return header_line, index_map


class LazyContractSlice:
    """
    Holds NO data. Only the 'recipe' to fetch it.
    """

    def __init__(self, ticker: str, month_code: str, delivery_year: int,
                 filepath: str, byte_offset: int, byte_length: int, header_bytes: bytes):
        self.ticker = ticker
        self.month_code = month_code
        self.month_int = MONTH_MAP[month_code]
        self.delivery_year = delivery_year
        self.filepath = filepath
        self.byte_offset = byte_offset
        self.byte_length = byte_length
        self.header_bytes = header_bytes
        self.expiry_date = self._estimate_expiry()

        # Get roll buffer days (matches spread_loader.py)
        try:
            if HAS_CLASSIFIER:
                self.roll_buffer_days = get_roll_buffer_days(f"{self.ticker}_F")
            else:
                self.roll_buffer_days = 0  # Default: roll at expiry
        except:
            self.roll_buffer_days = 0

    def _estimate_expiry(self) -> pd.Timestamp:
        """
        Determines expiry date using contract_expiry_rules module.
        Uses the same logic as spread_loader.py for consistency.
        """
        try:
            if HAS_CLASSIFIER:
                ticker_symbol = f"{self.ticker}_F"
                return calculate_expiry(ticker_symbol, self.delivery_year, self.month_int)
            else:
                # Fallback for when contract_expiry_rules is not available
                return self._expiry_fallback()
        except Exception as e:
            logger.warning(f"Error calculating expiry for {self.ticker}{self.month_code}{self.delivery_year}: {e}")
            return self._expiry_fallback()

    def _expiry_fallback(self) -> pd.Timestamp:
        """Simple fallback expiry calculation (3rd Friday of delivery month)"""
        try:
            first_day = pd.Timestamp(year=self.delivery_year, month=self.month_int, day=1)
            days_to_fri = (4 - first_day.dayofweek + 7) % 7
            return first_day + pd.Timedelta(days=days_to_fri + 14)
        except ValueError:
            return pd.Timestamp.max

    def load(self) -> pd.DataFrame:
        """Reads ONLY the specific bytes for this contract year."""
        with open(self.filepath, "rb") as f:
            f.seek(self.byte_offset)
            raw_bytes = f.read(self.byte_length)

        # Combine header + body
        with BytesIO(self.header_bytes + raw_bytes) as bio:
            df = read_exported_df(bio)

        # Optimize types
        df.columns = [c.lower().strip() for c in df.columns]


        # Map 'last' to 'close' for consistency
        if 'last' in df.columns and 'close' not in df.columns:
            df['close'] = df['last']

        # Keep minimal columns & use float32 (columns are already lowercased)
        keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[keep_cols].astype('float32')

        return df.sort_index()

    def __repr__(self):
        return f"<LazyContract {self.ticker}{self.month_code}{self.delivery_year}>"


class SpreadEngine:
    def __init__(self, data_dir: str, ticker: str):
        self.data_dir = data_dir
        self.ticker = ticker
        self.lazy_pool: List[LazyContractSlice] = []
        self.active_memory: Dict[str, pd.DataFrame] = {}  # Cache: "2024Z" -> DF
        self.contract_map: Optional[pd.DataFrame] = None  # Maps M1,M2,M3 -> contract codes over time

        self._index_files()

    def _index_single_file(self, code: str, m_int: int) -> List[LazyContractSlice]:
        """
        Index a single CSV file. Used by _index_files for parallel processing.

        Uses the same delivery year logic as spread_loader.py:
        - If trading month > contract month, data belongs to next year's delivery

        Returns:
        --------
        List[LazyContractSlice]
            List of contract slices for this month code
        """
        fname = f"{self.ticker}_{code}.csv"
        fpath = os.path.join(self.data_dir, fname)

        # Run Indexer with contract month to determine correct delivery years
        indexer = CsvDeliveryYearIndexer()
        header, delivery_year_map = indexer.build_index(fpath, m_int)

        contracts = []
        for delivery_year, (offset, length) in delivery_year_map.items():
            # Create LazyContractSlice with correct delivery year
            lc = LazyContractSlice(self.ticker, code, delivery_year, fpath, offset, length, header)
            contracts.append(lc)

        return contracts

    def _index_files(self, max_workers: int = 8):
        """
        Scans all CSVs and builds the Lazy Pool using multithreading.

        Parameters:
        -----------
        max_workers : int
            Number of threads to use for parallel indexing (default: 8)
        """
        logger.info(f"Indexing files for {self.ticker} using {max_workers} threads...")

        # Prepare tasks for parallel execution
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all indexing tasks
            for code, m_int in MONTH_MAP.items():
                future = executor.submit(self._index_single_file, code, m_int)
                tasks.append(future)

            # Collect results as they complete
            for future in as_completed(tasks):
                try:
                    contracts = future.result()
                    self.lazy_pool.extend(contracts)
                except Exception as e:
                    logger.error(f"Error indexing file: {e}")

        self.lazy_pool.sort(key=lambda x: x.expiry_date)
        logger.info(f"Indexed {len(self.lazy_pool)} contract slices.")

    def _get_ffill_limit(self, freq: str) -> int:
        """Calculate ffill limit (4 hours) based on frequency."""
        freq_map = {'1min': 240, '5min': 48, '15min': 16, '30min': 8, '1h': 4, '1H': 4}
        return freq_map.get(freq, 16)

    def build_contract_map(
        self,
        start_date: str,
        end_date: str,
        max_months: int = 12,
        freq: str = 'D',
    ) -> pd.DataFrame:
        """
        Build a DataFrame mapping M1, M2, M3... to contract codes over time.

        This creates a lookup table showing which contract (e.g., CLH25, CLJ25)
        corresponds to each position (M1, M2, M3...) at any given date.
        Useful for gap patching and understanding the curve structure.

        Parameters:
        -----------
        start_date : str
            Start date for the map
        end_date : str
            End date for the map
        max_months : int, default 12
            Maximum number of contract months to track
        freq : str, default 'D'
            Frequency for the index ('D' for daily, 'W' for weekly)

        Returns:
        --------
        pd.DataFrame
            Index: DatetimeIndex at specified frequency
            Columns: M1, M2, M3, ..., M{max_months}
            Values: Contract codes (e.g., "CLH25", "RBJ25")

        Example:
        --------
        >>> engine = SpreadEngine("F:\\monthly_contracts\\CL\\", "CL")
        >>> contract_map = engine.build_contract_map("2024-01-01", "2025-01-01")
        >>> print(contract_map)
                        M1      M2      M3      M4      M5      M6
        2024-01-01   CLG24   CLH24   CLJ24   CLK24   CLM24   CLN24
        2024-01-02   CLG24   CLH24   CLJ24   CLK24   CLM24   CLN24
        ...
        2024-01-22   CLH24   CLJ24   CLK24   CLM24   CLN24   CLQ24  # After roll
        ...

        >>> # Get contract code for M3 on 2024-06-15
        >>> contract_map.loc['2024-06-15', 'M3']
        'CLU24'
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Create date range at specified frequency
        dates = pd.date_range(start, end, freq=freq)

        # Initialize result DataFrame
        columns = [f'M{i}' for i in range(1, max_months + 1)]
        contract_map = pd.DataFrame(index=dates, columns=columns, dtype=str)

        logger.info(f"Building contract map: {len(dates)} dates, {max_months} months")

        for date in dates:
            # Find active contracts for this date
            active_contracts = []
            for c in self.lazy_pool:
                buffer = pd.Timedelta(days=c.roll_buffer_days)
                if c.expiry_date > (date + buffer):
                    active_contracts.append(c)

            # Map contracts to M1, M2, M3...
            for m_idx in range(min(max_months, len(active_contracts))):
                contract = active_contracts[m_idx]
                contract_code = f"{self.ticker}{contract.month_code}{str(contract.delivery_year)[-2:]}"
                contract_map.loc[date, f'M{m_idx + 1}'] = contract_code

        # Store in instance for later use
        self.contract_map = contract_map

        # Log summary
        logger.info(f"Contract map built: {contract_map.shape}")
        logger.info(f"Sample (first row): {contract_map.iloc[0].to_dict()}")
        logger.info(f"Sample (last row): {contract_map.iloc[-1].to_dict()}")

        return contract_map

    def get_contract_at(self, date: pd.Timestamp, position: str = 'M1') -> Optional[str]:
        """
        Get the contract code for a given position at a specific date.

        Parameters:
        -----------
        date : pd.Timestamp
            The date to look up
        position : str, default 'M1'
            The position (M1, M2, M3, etc.)

        Returns:
        --------
        str or None
            Contract code (e.g., "CLH25") or None if not found
        """
        if self.contract_map is None:
            logger.warning("Contract map not built. Call build_contract_map() first.")
            return None

        # Find nearest date in map
        if date in self.contract_map.index:
            return self.contract_map.loc[date, position]

        # Find closest date
        idx = self.contract_map.index.get_indexer([date], method='nearest')[0]
        if idx >= 0 and idx < len(self.contract_map):
            return self.contract_map.iloc[idx][position]

        return None

    def get_scid_paths_for_gaps(
        self,
        df: pd.DataFrame,
        scid_directory: str,
        exchange: str = "NYMEX",
    ) -> pd.DataFrame:
        """
        Get SCID file paths needed to patch gaps in a DataFrame.

        Returns a summary DataFrame showing which SCID files are needed
        to fill gaps for each contract column.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with M1, M2, M3... columns containing gaps
        scid_directory : str
            Base directory for SCID files
        exchange : str, default "NYMEX"
            Exchange suffix for SCID files

        Returns:
        --------
        pd.DataFrame
            Summary of gaps and required SCID files

        Example:
        --------
        >>> gap_summary = engine.get_scid_paths_for_gaps(df, "F:\\SierraChart\\Data")
        >>> print(gap_summary)
           Column  GapCount  ContractCodes              SCIDPath                    Exists
        0  M1      1523      [CLH25, CLJ25]  F:\\...\\CLH25-NYMEX.scid  True
        1  M2      2891      [CLJ25, CLK25]  F:\\...\\CLJ25-NYMEX.scid  True
        ...
        """
        if df.empty:
            return pd.DataFrame()

        # Build contract map if not exists
        if self.contract_map is None:
            self.build_contract_map(
                str(df.index.min().date()),
                str(df.index.max().date()),
                max_months=12
            )

        # Get M columns
        m_columns = [col for col in df.columns if col.startswith('M') and col[1:].isdigit()]
        m_columns = sorted(m_columns, key=lambda x: int(x[1:]))

        results = []
        for col in m_columns:
            gap_count = df[col].isna().sum()
            if gap_count == 0:
                continue

            # Get unique contracts that this column maps to
            if col in self.contract_map.columns:
                contract_codes = self.contract_map[col].dropna().unique().tolist()
            else:
                contract_codes = []

            # Build SCID paths for each contract
            scid_paths = []
            for code in contract_codes:
                scid_path = Path(scid_directory) / f"{code}-{exchange}.scid"
                scid_paths.append({
                    'Column': col,
                    'GapCount': gap_count,
                    'ContractCode': code,
                    'SCIDPath': str(scid_path),
                    'Exists': scid_path.exists()
                })

            results.extend(scid_paths)

        return pd.DataFrame(results)

    def export_contract_map_csv(self, filepath: str) -> None:
        """Export the contract map to CSV for reference."""
        if self.contract_map is None:
            logger.warning("Contract map not built. Call build_contract_map() first.")
            return

        self.contract_map.to_csv(filepath)
        logger.info(f"Contract map exported to {filepath}")

    def _build_scid_filepath(self, scid_directory: str, contract: 'LazyContractSlice',
                             exchange: str) -> Path:
        """Build SCID filepath from contract info."""
        # Convert month int to month code
        month_codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                      7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
        month_code = month_codes.get(contract.month_int, 'F')
        year_2digit = str(contract.delivery_year)[-2:]

        # Format: {ticker}{month_code}{2-digit year}-{exchange}.scid
        filename = f"{self.ticker}{month_code}{year_2digit}-{exchange}.scid"
        return Path(scid_directory) / filename

    def _load_scid_contract(self, scid_path: Path, start_date: pd.Timestamp,
                           end_date: pd.Timestamp, resample_rule: Optional[str] = None,
                           target_tz: str = 'America/Chicago') -> Optional[pd.Series]:
        """Load a single contract's SCID file and return Close prices.

        Parameters:
        -----------
        scid_path : Path
            Path to SCID file
        start_date : pd.Timestamp
            Start date for filtering
        end_date : pd.Timestamp
            End date for filtering
        resample_rule : str, optional
            Resample rule (e.g., '5min', '15min')
        target_tz : str, default 'America/Chicago'
            Target timezone to convert UTC data to. Common options:
            - 'America/Chicago' (CST/CDT for CME, CBOT, NYMEX)
            - 'America/New_York' (EST/EDT for ICE)
            - None to keep UTC

        Returns:
        --------
        pd.Series or None
            Close prices with timezone-converted index
        """
        if not HAS_SIERRAPY:
            return None

        if not scid_path.exists():
            logger.debug(f"SCID file not found: {scid_path}")
            return None

        try:
            from sierrapy.parser.scid_parse import FastScidReader

            with FastScidReader(str(scid_path)).open() as reader:
                df = reader.to_pandas(
                    columns=["Close"],
                    resample_rule=resample_rule,
                )

            if df.empty:
                return None

            # Convert timezone from UTC to target timezone
            if df.index.tz is not None and target_tz is not None:
                # Convert UTC to target timezone, then make naive
                df.index = df.index.tz_convert(target_tz).tz_localize(None)
            elif df.index.tz is not None:
                # Just strip timezone if target_tz is None
                df.index = df.index.tz_localize(None)

            # Filter to date range (after timezone conversion)
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            return df['Close'] if 'Close' in df.columns else None

        except Exception as e:
            logger.debug(f"Failed to load SCID {scid_path}: {e}")
            return None

    def patch_gaps_from_scid(
        self,
        df: pd.DataFrame,
        scid_directory: str,
        exchange: str = "NYMEX",
        service: str = "sierra",
        resample_rule: Optional[str] = None,
        target_tz: str = 'America/Chicago',
        contract_map: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Patch NaN gaps in DataFrame using SCID files from Sierra Chart.

        Loads individual contract SCID files for M1, M2, M3, etc. and fills
        gaps in each column using the corresponding contract's data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with M1, M2, M3... columns containing gaps (NaN values)
        scid_directory : str
            Directory containing SCID files
            Format: {ticker}{month_code}{2-digit year}-{exchange}.scid
            Example: CLH25-NYMEX.scid
        exchange : str, default "NYMEX"
            Exchange suffix for SCID files (NYMEX, CME, CBOT, etc.)
        service : str, default "sierra"
            Service name for ScidReader file conventions
        resample_rule : str, optional
            Resample rule (e.g., '5min', '15min'). If None, uses original frequency.
        target_tz : str, default 'America/Chicago'
            Target timezone to convert SCID UTC data to. Should match your DataFrame timezone.
            Common options: 'America/Chicago' (CME, CBOT, NYMEX), 'America/New_York' (ICE)
        contract_map : pd.DataFrame, optional
            Contract map DataFrame with columns ['M1', 'M2', 'M3', ...] containing
            contract codes (e.g., 'H2020', 'J2020') for each timestamp.
            If provided, uses this to determine which SCID file to use for each period.
            If None, falls back to lazy_pool-based contract selection (less accurate).

        Returns:
        --------
        pd.DataFrame
            DataFrame with gaps filled from SCID data where available

        Example:
        --------
        >>> engine = SpreadEngine("F:\\monthly_contracts\\CL\\", "CL")
        >>> df = pd.concat([chunk for chunk in engine.get_sequential_prices(...)])
        >>> contract_map = engine.build_contract_map(df.index.min(), df.index.max())
        >>> df_patched = engine.patch_gaps_from_scid(
        ...     df,
        ...     scid_directory="F:\\SierraChart\\Data",
        ...     exchange="NYMEX",
        ...     target_tz='America/Chicago',
        ...     contract_map=contract_map  # Use contract map for accurate mapping
        ... )
        """
        if not HAS_SIERRAPY:
            logger.warning("sierrapy not available, cannot patch gaps from SCID files")
            return df

        if df.empty:
            return df

        logger.info(f"Patching gaps from SCID files in {scid_directory}")

        # Get date range from DataFrame
        start_date = df.index.min()
        end_date = df.index.max()

        # Create patched DataFrame
        patched_df = df.copy()
        total_gaps_before = df.isna().sum().sum()

        # Get M columns that exist in the DataFrame
        m_columns = [col for col in df.columns if col.startswith('M') and col[1:].isdigit()]
        m_columns = sorted(m_columns, key=lambda x: int(x[1:]))

        logger.info(f"Found {len(m_columns)} contract columns to patch: {m_columns}")

        # Cache for loaded SCID data
        scid_cache: Dict[str, pd.Series] = {}

        if contract_map is not None:
            # USE CONTRACT MAP for accurate contract-to-M mapping
            logger.info("Using contract map for accurate SCID file selection")

            # Handle duplicate timestamps in DataFrame index
            if patched_df.index.duplicated().any():
                logger.warning(f"Found {patched_df.index.duplicated().sum()} duplicate timestamps in DataFrame, keeping last")
                patched_df = patched_df[~patched_df.index.duplicated(keep='last')]

            # Align contract map to DataFrame index (forward fill from daily to intraday)
            # Use large limit to cover full days of intraday data (288 bars for 5-min data)
            contract_map_aligned = contract_map.reindex(patched_df.index, method='ffill')

            # For each M column, find gaps and patch using contract map
            for col in m_columns:
                if col not in contract_map_aligned.columns:
                    logger.warning(f"{col} not in contract map, skipping")
                    continue

                # Find gaps in this M column
                gap_mask = patched_df[col].isna()
                if not gap_mask.any():
                    continue

                # Get contract codes for gap timestamps
                gap_timestamps = patched_df.index[gap_mask]
                gap_contracts = contract_map_aligned.loc[gap_timestamps, col]

                # Group by contract code to minimize SCID loads
                for raw_contract_code in gap_contracts.dropna().unique():
                    # Normalize contract code for cache key and filename
                    contract_code = str(raw_contract_code).strip().upper()
                    ticker_upper = self.ticker.upper()

                    # Load SCID if not cached
                    if contract_code not in scid_cache:
                        # Parse contract code - handle multiple formats
                        # Format 1: 'CLJ20' (ticker + month + 2-digit year)
                        if contract_code.startswith(ticker_upper):
                            filename = f"{contract_code}-{exchange}.scid"

                        # Format 2: 'J2020' (month + 4-digit year) - legacy
                        elif len(contract_code) >= 5 and contract_code[1:].isdigit():
                            month_code = contract_code[0]
                            year = int(contract_code[1:])
                            year_2digit = str(year)[-2:]
                            filename = f"{ticker_upper}{month_code}{year_2digit}-{exchange}.scid"

                        # Format 3: 'J20' (month + 2-digit year) - need to add ticker
                        elif len(contract_code) <= 3:
                            filename = f"{ticker_upper}{contract_code}-{exchange}.scid"

                        else:
                            logger.warning(f"Unrecognized contract code format: '{contract_code}' (ticker={self.ticker})")
                            scid_cache[contract_code] = None
                            continue

                        scid_path = Path(scid_directory) / filename

                        if scid_path.exists():
                            scid_data = self._load_scid_contract(scid_path, start_date, end_date, resample_rule, target_tz)

                            if scid_data is not None:
                                # Remove duplicate timestamps if present
                                if scid_data.index.duplicated().any():
                                    logger.debug(f"Removing {scid_data.index.duplicated().sum()} duplicates from SCID data")
                                    scid_data = scid_data[~scid_data.index.duplicated(keep='last')]

                                # Ensure monotonic index for reindexing
                                if not scid_data.index.is_monotonic_increasing:
                                    logger.debug(f"Sorting SCID data index")
                                    scid_data = scid_data.sort_index()

                            scid_cache[contract_code] = scid_data
                        else:
                            logger.debug(f"SCID file not found: {scid_path}")
                            scid_cache[contract_code] = None

                    scid_data = scid_cache.get(contract_code)
                    if scid_data is None:
                        continue

                    # Find gaps for this specific contract (use raw code to match contract_map_aligned)
                    contract_gap_mask = gap_mask & (contract_map_aligned[col] == raw_contract_code)
                    if not contract_gap_mask.any():
                        continue

                    # Get gap timestamps
                    contract_gap_timestamps = patched_df.index[contract_gap_mask]

                    # Reindex SCID data to match gap timestamps using merge_asof for robustness
                    # Create a Series of gap timestamps to fill
                    gap_df = pd.DataFrame({'ts': contract_gap_timestamps})
                    scid_df = pd.DataFrame({'ts': scid_data.index, 'value': scid_data.values})

                    # Use merge_asof to find nearest SCID value for each gap timestamp
                    merged = pd.merge_asof(
                        gap_df.sort_values('ts'),
                        scid_df.sort_values('ts'),
                        on='ts',
                        direction='nearest',
                        tolerance=pd.Timedelta('5min')
                    )

                    # Fill gaps using the matched values
                    if merged['value'].notna().any():
                        # Create a mapping from timestamp to value
                        fill_values = merged.set_index('ts')['value']
                        patched_df.loc[fill_values.index, col] = fill_values.values

        else:
            # FALLBACK: Use lazy_pool-based contract selection (less accurate)
            logger.warning("No contract map provided, using lazy_pool fallback (may be inaccurate)")

            weekly_dates = pd.date_range(start_date, end_date, freq='W')

            for period_start in weekly_dates:
                period_end = period_start + pd.Timedelta(days=7)
                period_mid = period_start + pd.Timedelta(days=3)

                # Find active contracts for this period
                active_contracts = []
                for c in self.lazy_pool:
                    buffer = pd.Timedelta(days=c.roll_buffer_days) if hasattr(c, 'roll_buffer_days') else pd.Timedelta(days=0)
                    if c.expiry_date > (period_mid + buffer):
                        active_contracts.append(c)

                if not active_contracts:
                    continue

                # Map M1, M2, M3... to contracts
                for m_idx, col in enumerate(m_columns):
                    if m_idx >= len(active_contracts):
                        break

                    contract = active_contracts[m_idx]
                    contract_key = f"{contract.month_code}{contract.delivery_year}"

                    # Load SCID if not cached
                    if contract_key not in scid_cache:
                        scid_path = self._build_scid_filepath(scid_directory, contract, exchange)
                        scid_data = self._load_scid_contract(scid_path, start_date, end_date, resample_rule, target_tz)
                        scid_cache[contract_key] = scid_data

                    scid_data = scid_cache[contract_key]
                    if scid_data is None:
                        continue

                    # Get rows in this period that have gaps
                    period_mask = (patched_df.index >= period_start) & (patched_df.index < period_end)
                    gap_mask = period_mask & patched_df[col].isna()

                    if not gap_mask.any():
                        continue

                    # Reindex SCID data to match gap timestamps
                    gap_timestamps = patched_df.index[gap_mask]
                    scid_reindexed = scid_data.reindex(gap_timestamps, method='nearest', tolerance='5min')

                    # Fill gaps
                    patched_df.loc[gap_mask, col] = scid_reindexed.values

        # Log per-column improvement
        for col in m_columns:
            before = df[col].isna().sum()
            after = patched_df[col].isna().sum()
            if before > 0:
                filled = before - after
                logger.info(f"{col}: {before} -> {after} gaps (filled {filled}, {filled/before*100:.1f}%)")

        # Log overall improvement
        total_gaps_after = patched_df.isna().sum().sum()
        if total_gaps_before > 0:
            reduction = (1 - total_gaps_after / total_gaps_before) * 100
            logger.info(f"Total gaps: {total_gaps_before} -> {total_gaps_after} ({reduction:.1f}% reduction)")

        return patched_df

    def load_continuous_from_scid(
        self,
        scid_directory: str,
        start_date: str,
        end_date: str,
        exchange: str = "NYMEX",
        service: str = "sierra",
        resample_rule: str = "5min",
        max_months: int = 12,
    ) -> pd.DataFrame:
        """
        Load continuous contract data directly from SCID files.

        This is an alternative to CSV-based loading that uses Sierra Chart's
        SCID files directly. Useful when SCID data is more complete.

        Parameters:
        -----------
        scid_directory : str
            Directory containing SCID files
        start_date, end_date : str
            Date range to load
        exchange : str, default "NYMEX"
            Exchange suffix (NYMEX, CME, CBOT, etc.)
        service : str, default "sierra"
            Service name for ScidReader
        resample_rule : str, default "5min"
            Resample frequency
        max_months : int, default 12
            Maximum number of contract months to load

        Returns:
        --------
        pd.DataFrame
            DataFrame with M1, M2, M3... columns from SCID data
        """
        if not HAS_SIERRAPY:
            raise ImportError("sierrapy required for SCID loading. Install with: pip install sierrapy")

        logger.info(f"Loading continuous data from SCID: {scid_directory}")

        scid_reader = ScidReader(scid_directory, default_service=service)

        # Load front-month continuous series
        df = scid_reader.load_front_month_continuous(
            self.ticker,
            service=service,
            start=start_date,
            end=end_date,
            columns=["Open", "High", "Low", "Close", "TotalVolume"],
            include_metadata=True,
            resample_rule=resample_rule,
            allow_tail=True,
        )

        if df.empty:
            logger.warning("No SCID data loaded")
            return pd.DataFrame()

        # Rename Close to M1 for compatibility
        result = pd.DataFrame(index=df.index)
        result['M1'] = df['Close']

        # For M2, M3, etc., we'd need to load each contract separately
        # and align them. For now, M1 is the primary use case.
        logger.info(f"Loaded {len(result)} rows from SCID (M1 only)")

        return result

    def _match_m1_to_contract(self, m1_price: float, timestamp: pd.Timestamp,
                               tolerance: float = 0.02) -> Optional[int]:
        """
        Match constant M1 price to a contract in the lazy pool.

        Returns the index of the matching contract in lazy_pool (sorted by expiry).
        """
        # Load contracts that are active at this timestamp
        for idx, contract in enumerate(self.lazy_pool):
            # Skip expired contracts
            if contract.expiry_date < timestamp:
                continue

            # Load contract data if not cached
            cid = str(contract)
            if cid not in self.active_memory:
                try:
                    self.active_memory[cid] = contract.load()
                except Exception:
                    continue

            contract_df = self.active_memory[cid]

            # Find price at or near this timestamp
            if timestamp in contract_df.index:
                contract_price = contract_df.loc[timestamp, 'close']
            else:
                # Try to find nearest timestamp within 1 hour
                mask = (contract_df.index >= timestamp - pd.Timedelta(hours=1)) & \
                       (contract_df.index <= timestamp + pd.Timedelta(hours=1))
                nearby = contract_df[mask]
                if nearby.empty:
                    continue
                contract_price = nearby['close'].iloc[-1]

            # Check if prices match within tolerance
            if pd.notna(contract_price) and pd.notna(m1_price):
                price_diff = abs(m1_price - contract_price) / m1_price
                if price_diff < tolerance:
                    return idx

        return None

    def _identify_contract_structure(self, m1_series: pd.Series, sample_freq: str = 'W') -> pd.DataFrame:
        """
        Identify which lazy_pool contract matches M1 at different points in time.

        By matching M1 prices to contracts in lazy_pool, we can determine:
        - Which contract is currently the front month
        - When rolls occur (when the matching contract index changes)
        - How to map M2, M3, M4 relative to the matched M1

        Parameters:
        -----------
        m1_series : pd.Series
            Constant front month price series
        sample_freq : str, default 'W'
            How often to sample for matching (weekly is usually sufficient)

        Returns:
        --------
        pd.DataFrame with columns:
            - date: sample date
            - m1_contract_idx: index of matching contract in lazy_pool
            - m1_contract: string representation of matching contract
        """
        # Sample M1 at regular intervals
        m1_sampled = m1_series.resample(sample_freq).last().dropna()

        results = []
        for timestamp, m1_price in m1_sampled.items():
            match_idx = self._match_m1_to_contract(m1_price, timestamp)
            if match_idx is not None:
                results.append({
                    'date': timestamp,
                    'm1_contract_idx': match_idx,
                    'm1_contract': str(self.lazy_pool[match_idx])
                })

        if not results:
            logger.warning("Could not match M1 to any contracts")
            return pd.DataFrame(columns=['date', 'm1_contract_idx', 'm1_contract'])

        return pd.DataFrame(results)

    def _detect_rolls_hybrid(self, m1_series: pd.Series) -> list:
        """
        Hybrid roll detection: expiry rules as anchor + price matching to refine.

        Strategy:
        1. Get expected roll dates from expiry rules (~12/year for monthly)
        2. For each expected roll, use price matching to find exact date within ±5 days
        3. This prevents false positives while still using actual market data

        Returns list of roll dates.
        """
        start_date = m1_series.index.min()
        end_date = m1_series.index.max()

        # 1. Get expected rolls from expiry rules (the anchor)
        expected_rolls = self._get_expected_roll_dates(start_date, end_date)
        logger.info(f"  Expected {len(expected_rolls)} rolls from expiry rules")

        if not expected_rolls:
            return []

        # 2. For each expected roll, try to refine using price matching
        refined_rolls = []
        window_days = 5

        for expected_date in expected_rolls:
            window_start = expected_date - pd.Timedelta(days=window_days)
            window_end = expected_date + pd.Timedelta(days=window_days)

            # Get M1 prices in this window
            m1_window = m1_series[(m1_series.index >= window_start) &
                                   (m1_series.index <= window_end)]

            if m1_window.empty:
                # No data in window, use expected date
                refined_rolls.append(expected_date)
                logger.info(f"  Roll {expected_date.date()}: using expected (no data)")
                continue

            # Sample daily and try to find contract switch
            daily_samples = m1_window.resample('D').last().dropna()
            prev_match = None
            roll_found = False

            for date, price in daily_samples.items():
                current_match = self._match_m1_to_contract(price, date, tolerance=0.015)

                if prev_match is not None and current_match is not None:
                    if current_match != prev_match:
                        # Found the actual roll date
                        refined_rolls.append(date)
                        logger.info(f"  Roll {expected_date.date()}: refined to {date.date()} "
                                   f"(contract {prev_match} → {current_match})")
                        roll_found = True
                        break

                if current_match is not None:
                    prev_match = current_match

            if not roll_found:
                # Couldn't refine, use expected date
                refined_rolls.append(expected_date)
                logger.info(f"  Roll {expected_date.date()}: using expected (no switch detected)")

        logger.info(f"  Final: {len(refined_rolls)} rolls")
        return refined_rolls

    def _get_expected_roll_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list:
        """
        Get expected roll dates based on contract expiry rules (fallback method).
        """
        if not HAS_CLASSIFIER:
            logger.warning("contract_expiry_rules not available")
            return []

        expected_rolls = []
        ticker_symbol = f"{self.ticker}_F"

        try:
            roll_buffer = get_roll_buffer_days(ticker_symbol)
        except:
            roll_buffer = 0

        current = start_date.replace(day=1)
        while current <= end_date:
            year = current.year
            month = current.month

            try:
                expiry = calculate_expiry(ticker_symbol, year, month)
                roll_date = expiry - pd.Timedelta(days=roll_buffer)

                if start_date <= roll_date <= end_date:
                    expected_rolls.append(roll_date)
            except Exception:
                pass

            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        return sorted(expected_rolls)

    def get_sequential_prices(self, start_date: str, end_date: str, freq: str = '1h',
                             max_months: int = None, return_expiries: bool = False,
                             front_month_filepath: str = None) -> Generator:
        """
        Yields continuous contract prices between expiration dates.
        Much faster than day-by-day processing - processes entire periods between rolls.

        Contract composition only changes at expiry dates, so we process data in chunks
        between expirations and yield each chunk as a single DataFrame.

        Parameters:
        -----------
        start_date : str
            Start date for the series
        end_date : str
            End date for the series
        freq : str
            Resampling frequency (default '1h')
        max_months : int, optional
            Maximum number of months to include (default: all available)
        return_expiries : bool, default False
            If True, yield (df_period, expiries_dict) tuples instead of just df_period.
            expiries_dict maps column names (M1, M2, ...) to expiry timestamps.
        front_month_filepath : str, optional
            Path to constant front month CSV file. If provided, this will be used as M1
            (the base reference) and all other contracts will be reindexed to match.
            This ensures M1 always has complete data, reducing gaps in curve features.

        Yields:
        -------
        pd.DataFrame or Tuple[pd.DataFrame, Dict[str, pd.Timestamp]]
            If return_expiries=False: DataFrame with columns M1, M2, M3, ..., MN (contract prices)
            If return_expiries=True: Tuple of (DataFrame, expiries_dict)
            One DataFrame per expiry period
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Load constant front month if provided
        constant_m1 = None
        m1_roll_dates = []
        if front_month_filepath:
            logger.info(f"Loading constant front month from {front_month_filepath}")
            constant_m1_df = read_exported_df(front_month_filepath)

            # Extract close column
            if 'Close' in constant_m1_df.columns:
                constant_m1 = constant_m1_df['Close']
            elif 'close' in constant_m1_df.columns:
                constant_m1 = constant_m1_df['close']
            elif 'Last' in constant_m1_df.columns:
                constant_m1 = constant_m1_df['Last']
            elif 'last' in constant_m1_df.columns:
                constant_m1 = constant_m1_df['last']
            else:
                raise ValueError(f"No 'Close' or 'Last' column found in {front_month_filepath}")

            # Filter to date range
            constant_m1 = constant_m1[(constant_m1.index >= start) & (constant_m1.index <= end)]

            # Hybrid roll detection: expiry rules + price matching refinement
            logger.info("Detecting rolls (hybrid: expiry rules + price matching)...")
            m1_roll_dates = self._detect_rolls_hybrid(constant_m1)
            logger.info(f"Found {len(m1_roll_dates)} roll dates")

            # Resample to target frequency if needed
            if freq != '5min':
                logger.info(f"Resampling constant M1 from 5min to {freq}")
                constant_m1 = constant_m1.resample(freq).last().ffill(limit=self._get_ffill_limit(freq))

            logger.info(f"Constant M1 loaded: {len(constant_m1)} bars from {constant_m1.index[0]} to {constant_m1.index[-1]}")

        # 1. Get period boundaries
        # If using constant M1, use detected roll dates
        # Otherwise, use expiry dates from contracts
        if constant_m1 is not None and len(m1_roll_dates) > 0:
            # Use M1 roll dates as period boundaries
            logger.info(f"Using {len(m1_roll_dates)} M1 roll dates as period boundaries")
            boundaries = sorted([start] + m1_roll_dates + [end])
            use_contract_matching = True
        elif constant_m1 is not None:
            # Have constant M1 but no detected rolls - use expiry rules
            logger.info("No rolls detected, using expiry-based boundaries")
            expiry_roll_dates = set()
            for c in self.lazy_pool:
                buffer = pd.Timedelta(days=c.roll_buffer_days)
                roll_date = c.expiry_date + buffer
                if start <= roll_date <= end:
                    expiry_roll_dates.add(roll_date)
            boundaries = sorted([start] + list(expiry_roll_dates) + [end])
            use_contract_matching = True
        else:
            # Original behavior: no constant M1, use contract expiry dates
            expiry_roll_dates = set()
            for c in self.lazy_pool:
                buffer = pd.Timedelta(days=c.roll_buffer_days)
                roll_date = c.expiry_date + buffer
                if start <= roll_date <= end:
                    expiry_roll_dates.add(roll_date)
            boundaries = sorted([start] + list(expiry_roll_dates) + [end])
            use_contract_matching = False

        logger.info(f"Processing {len(boundaries)-1} periods between expirations")

        # 2. Process each period between boundaries
        for i in range(len(boundaries) - 1):
            period_start = boundaries[i]
            period_end = boundaries[i + 1]

            logger.info(f"Period {i+1}/{len(boundaries)-1}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")

            # Determine active contracts for this period
            if use_contract_matching and constant_m1 is not None:
                # Use contract matching to determine M2, M3, M4
                # Find which contract M1 matches at the midpoint of this period
                period_mid = period_start + (period_end - period_start) / 2

                # Get M1 price at midpoint
                m1_at_mid = constant_m1[(constant_m1.index >= period_mid - pd.Timedelta(hours=4)) &
                                        (constant_m1.index <= period_mid + pd.Timedelta(hours=4))]
                if not m1_at_mid.empty:
                    m1_price = m1_at_mid.iloc[-1]
                    m1_contract_idx = self._match_m1_to_contract(m1_price, period_mid)
                else:
                    m1_contract_idx = None

                if m1_contract_idx is not None:
                    # M2 is the next contract after M1, M3 is the one after that, etc.
                    # Get all contracts after the matched M1
                    start_idx = m1_contract_idx + 1  # Start from next contract after M1
                    available_contracts = self.lazy_pool[start_idx:]

                    if max_months and len(available_contracts) > max_months - 1:
                        active_contracts = available_contracts[:max_months - 1]
                    else:
                        active_contracts = available_contracts

                    logger.info(f"  M1 matches contract {m1_contract_idx} ({self.lazy_pool[m1_contract_idx]})")
                    logger.info(f"  Using contracts {start_idx} onwards for M2, M3, M4...")
                else:
                    # Couldn't match - fall back to midpoint check
                    logger.warning(f"  Could not match M1 price at {period_mid.date()}, using fallback")
                    available_contracts = []
                    for c in self.lazy_pool:
                        buffer = pd.Timedelta(days=c.roll_buffer_days)
                        if c.expiry_date > (period_mid + buffer):
                            available_contracts.append(c)
                    if max_months:
                        active_contracts = available_contracts[:max_months]
                    else:
                        active_contracts = available_contracts
            else:
                # Original behavior: midpoint check without shifting
                period_mid = period_start + (period_end - period_start) / 2

                active_contracts = []
                for c in self.lazy_pool:
                    buffer = pd.Timedelta(days=c.roll_buffer_days)
                    if c.expiry_date > (period_mid + buffer):
                        active_contracts.append(c)

                if max_months:
                    active_contracts = active_contracts[:max_months]

            logger.info(f"  Active contracts: {len(active_contracts)}")

            if len(active_contracts) < 1:
                logger.warning(f"  No active contracts for period {i+1}, skipping")
                continue

            # Load contracts for this period
            loaded_contracts = {}
            try:
                for contract in active_contracts:
                    cid = str(contract)
                    if cid not in self.active_memory:
                        logger.debug(f"  Loading {cid}")
                        self.active_memory[cid] = contract.load()
                    loaded_contracts[contract] = self.active_memory[cid]
            except Exception as e:
                logger.error(f"  Error loading contracts for period {i+1}: {e}")
                continue

            # Slice and resample data for the entire period
            period_data = {}

            # Determine the base index to use for reindexing
            base_index = None
            if constant_m1 is not None:
                # Use constant M1's index for this period as the base
                base_index = constant_m1[(constant_m1.index >= period_start) & (constant_m1.index < period_end)].index
                logger.info(f"  Using constant M1 as base index: {len(base_index)} timestamps")

            try:
                # If we have a constant M1, start with it
                if constant_m1 is not None:
                    m1_period = constant_m1[(constant_m1.index >= period_start) & (constant_m1.index < period_end)]
                    if not m1_period.empty:
                        period_data["M1"] = m1_period
                        logger.info(f"  Added constant M1: {len(m1_period)} values")
                    # Process M2, M3, M4, ... from lazy pool (skip first contract since M1 is constant)
                    contract_start_idx = 1
                else:
                    # Original behavior: use all contracts from lazy pool starting with M1
                    contract_start_idx = 0

                for idx, contract in enumerate(active_contracts[contract_start_idx:], start=contract_start_idx):
                    full_df = loaded_contracts[contract]

                    # Slice for entire period
                    # Add 4-hour overlap to prevent gaps at period boundaries
                    overlap = pd.Timedelta(hours=4)
                    slice_df = full_df[(full_df.index >= period_start - overlap) & (full_df.index < period_end)]

                    if not slice_df.empty and 'close' in slice_df.columns:
                        # Resample with limited forward fill (4 hours max)
                        ffill_limit = self._get_ffill_limit(freq)
                        resampled = slice_df['close'].resample(freq).last().ffill(limit=ffill_limit)

                        # If we have a base_index (from constant M1), reindex to match
                        if base_index is not None and len(base_index) > 0:
                            # Reindex to M1's timestamps, forward fill gaps up to limit
                            resampled = resampled.reindex(base_index).ffill(limit=ffill_limit)

                        period_data[f"M{idx + 1}"] = resampled
            except Exception as e:
                logger.error(f"  Error slicing/resampling for period {i+1}: {e}")
                continue

            # Create DataFrame for this period
            if period_data:
                df_before = pd.DataFrame(period_data)

                # Adjust dropna threshold based on whether we have constant M1
                if constant_m1 is not None:
                    # With constant M1, we can be less strict since M1 is always present
                    # Only drop rows where M1 is missing (shouldn't happen) or all other contracts are missing
                    df_period = df_before.dropna(subset=['M1'])
                    logger.info(f"  DataFrame: {df_before.shape} -> {df_period.shape} (with constant M1, dropped only M1-NaN rows)")
                else:
                    # Original behavior: Use thresh=3 to require at least 3 contracts with data
                    df_period = df_before.dropna(thresh=3)
                    logger.info(f"  DataFrame: {df_before.shape} -> {df_period.shape} (after dropna thresh=3)")
                if not df_period.empty:
                    logger.info(f"  Yielding {df_period.shape[0]} rows")

                    if return_expiries:
                        # Build expiry mapping for this period
                        expiries_dict = {}
                        for idx, contract in enumerate(active_contracts):
                            col_name = f"M{idx + 1}"
                            if col_name in df_period.columns:
                                expiries_dict[col_name] = contract.expiry_date
                        yield df_period, expiries_dict
                    else:
                        yield df_period
                else:
                    logger.warning(f"  DataFrame empty after dropna, not yielding")
            else:
                logger.warning(f"  No period_data collected for period {i+1}")

            # Garbage collection - remove contracts not needed for next period
            if i < len(boundaries) - 2:
                next_period_mid = boundaries[i + 1] + (boundaries[i + 2] - boundaries[i + 1]) / 2
                next_needed = set()

                for c in self.lazy_pool:
                    buffer = pd.Timedelta(days=c.roll_buffer_days)
                    if c.expiry_date > (next_period_mid + buffer):
                        next_needed.add(str(c))

                # Remove unneeded contracts
                for cid in list(self.active_memory.keys()):
                    if cid not in next_needed:
                        del self.active_memory[cid]


# --- Usage ---
if __name__ == "__main__":
    engine = SpreadEngine("F:\\monthly_contracts\\CL\\", "CL")

    # Get sequential contract prices (M1, M2, M3, ..., MN)
    res = []
    print("Streaming Contract Prices:")
    for daily_df in engine.get_sequential_prices("2011-01-01", "2025-12-28"):
        res.append(daily_df)

    df = pd.concat(res)