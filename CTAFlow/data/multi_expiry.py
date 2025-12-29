import pandas as pd
import numpy as np
import mmap
import os
import logging
from io import BytesIO
from typing import List, Dict, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: Try to import TickerClassifier and expiry rules
try:
    from CTAFlow.data.ticker_classifier import get_ticker_classifier, TickerCategory
    from CTAFlow.data import read_exported_df
    from CTAFlow.data.contract_expiry_rules import calculate_expiry, get_roll_buffer_days

    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False

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

    def get_sequential_prices(self, start_date: str, end_date: str, freq: str = '1h',
                             max_months: int = None) -> Generator[pd.DataFrame, None, None]:
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

        Yields:
        -------
        pd.DataFrame
            DataFrame with columns M1, M2, M3, ..., MN (contract prices)
            One DataFrame per expiry period
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # 1. Get all expiry dates with roll buffers in our date range
        roll_dates = set()
        for c in self.lazy_pool:
            buffer = pd.Timedelta(days=c.roll_buffer_days)
            roll_date = c.expiry_date + buffer
            if start <= roll_date <= end:
                roll_dates.add(roll_date)

        # Create period boundaries: start, roll_dates, end
        boundaries = sorted([start] + list(roll_dates) + [end])

        logger.info(f"Processing {len(boundaries)-1} periods between expirations")

        # 2. Process each period between boundaries
        for i in range(len(boundaries) - 1):
            period_start = boundaries[i]
            period_end = boundaries[i + 1]

            logger.info(f"Period {i+1}/{len(boundaries)-1}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")

            # Determine active contracts for this period (midpoint check)
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

            try:
                for idx, contract in enumerate(active_contracts):
                    full_df = loaded_contracts[contract]

                    # Slice for entire period
                    slice_df = full_df[(full_df.index >= period_start) & (full_df.index < period_end)]

                    if not slice_df.empty and 'close' in slice_df.columns:
                        # Resample for entire period at once
                        resampled = slice_df['close'].resample(freq).last().ffill()
                        period_data[f"M{idx + 1}"] = resampled
            except Exception as e:
                logger.error(f"  Error slicing/resampling for period {i+1}: {e}")
                continue

            # Create DataFrame for this period
            if period_data:
                df_before = pd.DataFrame(period_data)
                # Use thresh=3 to require at least 3 contracts with data (M1, M2, M3 minimum)
                # This prevents losing all data when far-dated contracts have sparse timestamps
                df_period = df_before.dropna(thresh=3)
                logger.info(f"  DataFrame: {df_before.shape} -> {df_period.shape} (after dropna thresh=3)")
                if not df_period.empty:
                    logger.info(f"  Yielding {df_period.shape[0]} rows")
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