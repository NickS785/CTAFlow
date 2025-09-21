#!/usr/bin/env python3
"""
Intraday file management for SCID contract data.

This module provides the IntradayFileManager class for discovering, parsing, and
managing intraday SCID files. It integrates with the curve data to identify
front month (M0) contracts and extract relevant time series data.
"""

from typing import Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import re
import logging
import asyncio
from dataclasses import dataclass
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq

from ...config import DLY_DATA_PATH
from ...data.sierra.parse_file import read_scid, AsyncScidReader, read_multiple_scid_async
from ...data.sierra.fast_parse import FastScidReader
from ...data.data_client import DataClient
from .curve_manager import SpreadData


@dataclass
class ContractPeriod:
    """Information about when a contract was the front month (M0)."""
    contract_code: str
    symbol: str
    start_date: datetime
    end_date: datetime
    file_path: Optional[Path] = None


@dataclass
class IntradayData:
    """Container for intraday SCID data with metadata."""
    symbol: str
    contract_code: str
    data: pd.DataFrame
    start_date: datetime
    end_date: datetime
    total_records: int
    file_path: Path


class IntradayFileManager:
    """
    Manager for intraday SCID contract files.

    This class handles:
    - Discovery of SCID files in DLY_DATA_PATH
    - Parsing contract codes from filenames
    - Integration with curve data to identify M0 periods
    - Extraction of intraday data for front month contracts
    """

    def __init__(self,
                 data_path: Optional[Path] = None,
                 data_client: Optional[DataClient] = None,
                 enable_logging: bool = True):
        """
        Initialize the intraday file manager.

        Args:
            data_path: Path to directory containing SCID files (default: DLY_DATA_PATH)
            data_client: DataClient instance for curve data queries
            enable_logging: Enable logging for debugging
        """
        self.data_path = Path(data_path) if data_path else Path(DLY_DATA_PATH)
        self.data_client = data_client or DataClient()
        self.enable_logging = enable_logging

        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

        # Cache for discovered files and contract mappings
        self._scid_files: Dict[str, List[Path]] = {}
        self._contract_periods: Dict[str, List[ContractPeriod]] = {}
        self._file_cache: Dict[str, IntradayData] = {}
        self._curve_data_cache: Dict[str, SpreadData] = {}

        # Discover SCID files on initialization
        self._discover_scid_files()

    def _discover_scid_files(self) -> None:
        """Discover and catalog all SCID files in the data directory."""
        if not self.data_path.exists():
            if self.logger:
                self.logger.warning(f"Data path does not exist: {self.data_path}")
            return

        scid_pattern = re.compile(r'^([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{2})-([A-Z]+)\.scid$')

        for scid_file in self.data_path.glob("*.scid"):
            match = scid_pattern.match(scid_file.name)
            if match:
                base_symbol = match.group(1)
                month_code = match.group(2)
                year_code = match.group(3)
                exchange = match.group(4)

                # Construct full symbol (e.g., CL_F, ES_F)
                full_symbol = f"{base_symbol}_F"

                # Store file mapping
                if full_symbol not in self._scid_files:
                    self._scid_files[full_symbol] = []

                self._scid_files[full_symbol].append(scid_file)

                if self.logger:
                    self.logger.debug(f"Discovered SCID file: {scid_file.name} -> {full_symbol}")

        if self.logger:
            total_files = sum(len(files) for files in self._scid_files.values())
            self.logger.info(f"Discovered {total_files} SCID files for {len(self._scid_files)} symbols")

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available SCID files."""
        return list(self._scid_files.keys())

    def get_scid_files_for_symbol(self, symbol: str) -> List[Path]:
        """Get all SCID files for a given symbol."""
        return self._scid_files.get(symbol, [])

    def _parse_contract_from_filename(self, file_path: Path) -> Optional[Tuple[str, str, str, str]]:
        """
        Parse contract information from SCID filename.

        Returns:
            Tuple of (base_symbol, month_code, year_code, exchange) or None
        """
        scid_pattern = re.compile(r'^([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{2})-([A-Z]+)\.scid$')
        match = scid_pattern.match(file_path.name)

        if match:
            return match.group(1), match.group(2), match.group(3), match.group(4)
        return None

    def _convert_month_year_to_date(self, month_code: str, year_code: str) -> datetime:
        """Convert month code and year code to datetime."""
        month_map = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }

        month = month_map.get(month_code)
        if month is None:
            raise ValueError(f"Invalid month code: {month_code}")

        # Convert 2-digit year to 4-digit (assuming 20xx for now)
        year = 2000 + int(year_code)

        return datetime(year, month, 1)

    def _load_curve_data_bulk(self, symbols: List[str],
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, SpreadData]:
        """
        Load curve data for multiple symbols in a single DataClient operation.

        Args:
            symbols: List of symbols to load
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary mapping symbols to SpreadData objects
        """
        curve_data_map = {}

        # Use DataClient to load all symbols at once
        for symbol in symbols:
            try:
                # Check if we need to query the data client
                if symbol not in self._curve_data_cache:
                    spread_data = SpreadData(symbol)

                    # Single call to load curve data
                    spread_data.load_from_client(
                        self.data_client,
                        start_date=start_date,
                        end_date=end_date
                    )

                    self._curve_data_cache[symbol] = spread_data

                    if self.logger:
                        self.logger.info(f"Loaded curve data for {symbol}")
                else:
                    spread_data = self._curve_data_cache[symbol]

                curve_data_map[symbol] = spread_data

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error loading curve data for {symbol}: {e}")
                continue

        return curve_data_map

    def identify_m0_periods(self, symbol: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[ContractPeriod]:
        """
        Identify periods when each contract was the front month (M0).

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            List of ContractPeriod objects with M0 information
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self._contract_periods:
            return self._contract_periods[cache_key]

        # Load curve data efficiently
        curve_data_map = self._load_curve_data_bulk([symbol], start_date, end_date)

        if symbol not in curve_data_map:
            if self.logger:
                self.logger.warning(f"No curve data available for {symbol}")
            return []

        spread_data = curve_data_map[symbol]

        if spread_data.seq_data is None or spread_data.seq_data.seq_labels is None:
            if self.logger:
                self.logger.warning(f"No sequential data available for {symbol}")
            return []

        # Extract M0 contract information from seq_labels
        contract_periods = []
        seq_labels = spread_data.seq_data.seq_labels
        dates = spread_data.index

        if len(dates) == 0 or seq_labels.shape[0] == 0:
            return []

        # Track when each contract becomes/stops being M0
        current_m0 = None
        m0_start_date = None

        for i, date in enumerate(dates):
            if i < seq_labels.shape[0] and seq_labels.shape[1] > 0:
                # Get M0 contract label for this date
                m0_contract = seq_labels[i, 0]  # First column is M0

                if m0_contract != current_m0:
                    # M0 contract changed
                    if current_m0 is not None and m0_start_date is not None:
                        # Close previous M0 period
                        contract_periods.append(ContractPeriod(
                            contract_code=current_m0,
                            symbol=symbol,
                            start_date=m0_start_date,
                            end_date=date - timedelta(days=1)
                        ))

                    # Start new M0 period
                    current_m0 = m0_contract
                    m0_start_date = date

        # Close final M0 period
        if current_m0 is not None and m0_start_date is not None:
            contract_periods.append(ContractPeriod(
                contract_code=current_m0,
                symbol=symbol,
                start_date=m0_start_date,
                end_date=dates[-1]
            ))

        # Match contract periods with SCID files
        scid_files = self.get_scid_files_for_symbol(symbol)
        for period in contract_periods:
            for scid_file in scid_files:
                parsed = self._parse_contract_from_filename(scid_file)
                if parsed:
                    base_symbol, month_code, year_code, exchange = parsed
                    contract_date = self._convert_month_year_to_date(month_code, year_code)

                    # Match by contract expiry month/year
                    if (contract_date.year == period.start_date.year and
                        contract_date.month == period.start_date.month):
                        period.file_path = scid_file
                        break

        self._contract_periods[cache_key] = contract_periods

        if self.logger:
            self.logger.info(f"Identified {len(contract_periods)} M0 periods for {symbol}")

        return contract_periods

    def identify_m0_periods_bulk(self, symbols: List[str],
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, List[ContractPeriod]]:
        """
        Identify M0 periods for multiple symbols efficiently.

        Args:
            symbols: List of symbols to analyze
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary mapping symbols to their M0 periods
        """
        # Load all curve data in a single operation
        curve_data_map = self._load_curve_data_bulk(symbols, start_date, end_date)

        all_periods = {}

        for symbol in symbols:
            if symbol not in curve_data_map:
                all_periods[symbol] = []
                continue

            # Use the already loaded curve data
            periods = self.identify_m0_periods(symbol, start_date, end_date)
            all_periods[symbol] = periods

        return all_periods

    def load_m0_intraday_data(self, symbol: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             resample_rule: Optional[str] = None,
                             volume_bucket_size: Optional[int] = None) -> List[IntradayData]:
        """
        Load intraday data for all M0 periods of a symbol.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date filter
            end_date: End date filter
            resample_rule: Optional time-based resampling rule (e.g., '1T', '5T', '1H')
            volume_bucket_size: Optional volume bucket size for aggregation

        Returns:
            List of IntradayData objects for each M0 period
        """
        m0_periods = self.identify_m0_periods(symbol, start_date, end_date)
        intraday_data_list = []

        for period in m0_periods:
            if period.file_path is None:
                if self.logger:
                    self.logger.warning(f"No SCID file found for {period.contract_code}")
                continue

            # Apply date filtering
            period_start = period.start_date
            period_end = period.end_date

            if start_date and period_end < start_date:
                continue
            if end_date and period_start > end_date:
                continue

            # Adjust period boundaries based on filters
            if start_date and period_start < start_date:
                period_start = start_date
            if end_date and period_end > end_date:
                period_end = end_date

            try:
                # Load SCID data
                cache_key = f"{symbol}_{period.contract_code}_{period_start}_{period_end}_{resample_rule}_{volume_bucket_size}"

                if cache_key in self._file_cache:
                    intraday_data = self._file_cache[cache_key]
                else:
                    if self.logger:
                        processing_info = f"Loading SCID data: {period.file_path.name} from {period_start} to {period_end}"
                        if volume_bucket_size:
                            processing_info += f" (volume buckets: {volume_bucket_size})"
                        elif resample_rule:
                            processing_info += f" (resample: {resample_rule})"
                        self.logger.info(processing_info)

                    # Read SCID file with date filtering
                    df = read_scid(
                        str(period.file_path),
                        start=pd.Timestamp(period_start, tz='UTC'),
                        end=pd.Timestamp(period_end, tz='UTC')
                    )

                    if len(df) == 0:
                        if self.logger:
                            self.logger.warning(f"No data in date range for {period.file_path.name}")
                        continue

                    # Apply volume bucket aggregation (takes priority over time resampling)
                    if volume_bucket_size:
                        df = self._aggregate_volume_buckets(df, volume_bucket_size)
                    # Optional time-based resampling
                    elif resample_rule:
                        df = self._resample_ohlcv(df, rule=resample_rule)

                    intraday_data = IntradayData(
                        symbol=symbol,
                        contract_code=period.contract_code,
                        data=df,
                        start_date=period_start,
                        end_date=period_end,
                        total_records=len(df),
                        file_path=period.file_path
                    )

                    self._file_cache[cache_key] = intraday_data

                intraday_data_list.append(intraday_data)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error loading SCID file {period.file_path}: {e}")
                continue

        if self.logger:
            total_records = sum(data.total_records for data in intraday_data_list)
            self.logger.info(f"Loaded {len(intraday_data_list)} M0 periods with "
                           f"{total_records:,} total records for {symbol}")

        return intraday_data_list

    async def load_m0_intraday_data_async(self, symbol: str,
                                         start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None,
                                         resample_rule: Optional[str] = None,
                                         volume_bucket_size: Optional[int] = None,
                                         batch_size: int = 5) -> List[IntradayData]:
        """
        Asynchronously load intraday data for all M0 periods of a symbol.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date filter
            end_date: End date filter
            resample_rule: Optional time-based resampling rule
            volume_bucket_size: Optional volume bucket size for aggregation
            batch_size: Number of SCID files to process concurrently

        Returns:
            List of IntradayData objects for each M0 period
        """
        m0_periods = self.identify_m0_periods(symbol, start_date, end_date)
        intraday_data_list = []

        # Filter periods with valid file paths
        valid_periods = [p for p in m0_periods if p.file_path is not None]

        if not valid_periods:
            if self.logger:
                self.logger.warning(f"No valid SCID files found for {symbol}")
            return []

        # Prepare file reading tasks
        file_read_tasks = []
        for period in valid_periods:
            # Apply date filtering
            period_start = period.start_date
            period_end = period.end_date

            if start_date and period_end < start_date:
                continue
            if end_date and period_start > end_date:
                continue

            # Adjust period boundaries based on filters
            if start_date and period_start < start_date:
                period_start = start_date
            if end_date and period_end > end_date:
                period_end = end_date

            # Create async reading task
            file_read_tasks.append({
                'period': period,
                'file_path': str(period.file_path),
                'start_date': period_start,
                'end_date': period_end,
                'start_ts': pd.Timestamp(period_start, tz='UTC'),
                'end_ts': pd.Timestamp(period_end, tz='UTC')
            })

        if not file_read_tasks:
            return []

        # Read all SCID files asynchronously
        file_paths = [task['file_path'] for task in file_read_tasks]

        if self.logger:
            self.logger.info(f"Reading {len(file_paths)} SCID files asynchronously for {symbol}")

        # Use AsyncScidReader for concurrent file reading
        async with AsyncScidReader(max_workers=min(batch_size * 2, 16)) as async_reader:
            # Read files in batches
            file_results = await async_reader.read_multiple_scid(
                file_paths,
                batch_size=batch_size,
                start=file_read_tasks[0]['start_ts'],  # Use overall range
                end=file_read_tasks[-1]['end_ts']
            )

        # Process results and create IntradayData objects
        for task, (file_path, df) in zip(file_read_tasks, file_results):
            if len(df) == 0:
                if self.logger:
                    self.logger.warning(f"No data in date range for {Path(file_path).name}")
                continue

            period = task['period']

            # Apply post-processing
            if volume_bucket_size:
                df = self._aggregate_volume_buckets(df, volume_bucket_size)
            elif resample_rule:
                df = self._resample_ohlcv(df, rule=resample_rule)

            if len(df) == 0:
                continue

            intraday_data = IntradayData(
                symbol=symbol,
                contract_code=period.contract_code,
                data=df,
                start_date=task['start_date'],
                end_date=task['end_date'],
                total_records=len(df),
                file_path=period.file_path
            )

            intraday_data_list.append(intraday_data)

        if self.logger:
            total_records = sum(data.total_records for data in intraday_data_list)
            self.logger.info(f"Loaded {len(intraday_data_list)} M0 periods with "
                           f"{total_records:,} total records for {symbol} (async)")

        return intraday_data_list

    def _aggregate_volume_buckets(self, df: pd.DataFrame, volume_bucket_size: int) -> pd.DataFrame:
        """
        Aggregate tick data into volume buckets.

        Args:
            df: Input DataFrame with tick data
            volume_bucket_size: Target volume per bucket

        Returns:
            DataFrame aggregated by volume buckets
        """
        if len(df) == 0 or 'TotalVolume' not in df.columns:
            return df

        aggregated_data = []
        current_bucket = {
            'start_time': None,
            'end_time': None,
            'open': None,
            'high': float('-inf'),
            'low': float('inf'),
            'close': None,
            'volume': 0,
            'num_trades': 0,
            'bid_volume': 0,
            'ask_volume': 0,
            'tick_count': 0
        }

        for timestamp, row in df.iterrows():
            # Initialize bucket if first tick
            if current_bucket['start_time'] is None:
                current_bucket['start_time'] = timestamp
                current_bucket['open'] = row['Close']
                current_bucket['high'] = row['High']
                current_bucket['low'] = row['Low']

            # Update bucket with current tick
            current_bucket['end_time'] = timestamp
            current_bucket['close'] = row['Close']
            current_bucket['high'] = max(current_bucket['high'], row['High'])
            current_bucket['low'] = min(current_bucket['low'], row['Low'])
            current_bucket['volume'] += row.get('TotalVolume', 0)
            current_bucket['num_trades'] += row.get('NumTrades', 0)
            current_bucket['bid_volume'] += row.get('BidVolume', 0)
            current_bucket['ask_volume'] += row.get('AskVolume', 0)
            current_bucket['tick_count'] += 1

            # Check if bucket is complete
            if current_bucket['volume'] >= volume_bucket_size:
                # Create aggregated record
                bucket_record = {
                    'Open': current_bucket['open'],
                    'High': current_bucket['high'],
                    'Low': current_bucket['low'],
                    'Close': current_bucket['close'],
                    'TotalVolume': current_bucket['volume'],
                    'NumTrades': current_bucket['num_trades'],
                    'BidVolume': current_bucket['bid_volume'],
                    'AskVolume': current_bucket['ask_volume'],
                    'TickCount': current_bucket['tick_count'],
                    'BucketDuration': (current_bucket['end_time'] - current_bucket['start_time']).total_seconds()
                }

                # Use end_time as the timestamp for the bucket
                aggregated_data.append((current_bucket['end_time'], bucket_record))

                # Reset bucket
                current_bucket = {
                    'start_time': None,
                    'end_time': None,
                    'open': None,
                    'high': float('-inf'),
                    'low': float('inf'),
                    'close': None,
                    'volume': 0,
                    'num_trades': 0,
                    'bid_volume': 0,
                    'ask_volume': 0,
                    'tick_count': 0
                }

        # Handle remaining partial bucket
        if current_bucket['start_time'] is not None and current_bucket['volume'] > 0:
            bucket_record = {
                'Open': current_bucket['open'],
                'High': current_bucket['high'],
                'Low': current_bucket['low'],
                'Close': current_bucket['close'],
                'TotalVolume': current_bucket['volume'],
                'NumTrades': current_bucket['num_trades'],
                'BidVolume': current_bucket['bid_volume'],
                'AskVolume': current_bucket['ask_volume'],
                'TickCount': current_bucket['tick_count'],
                'BucketDuration': (current_bucket['end_time'] - current_bucket['start_time']).total_seconds()
            }
            aggregated_data.append((current_bucket['end_time'], bucket_record))

        if not aggregated_data:
            return pd.DataFrame()

        # Convert to DataFrame
        timestamps, records = zip(*aggregated_data)
        result_df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps, name='DateTime'))

        return result_df

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str = "1T") -> pd.DataFrame:
        """
        Resample a tick/second stream into OHLCV using pandas (UTC index required).

        Args:
            df: Input DataFrame with tick data
            rule: Resampling rule (e.g., "1T" for 1 minute)

        Returns:
            Resampled OHLCV DataFrame
        """
        if df.index.tz is None:
            df = df.tz_localize("UTC")

        resampled = df.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'TotalVolume': 'sum',
            'NumTrades': 'sum',
            'BidVolume': 'sum',
            'AskVolume': 'sum'
        }).dropna()

        return resampled

    def get_continuous_m0_data(self, symbol: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              resample_rule: Optional[str] = None,
                              volume_bucket_size: Optional[int] = None,
                              remove_gaps: bool = True,
                              add_roll_markers: bool = True) -> pd.DataFrame:
        """
        Get continuous M0 intraday data by concatenating all M0 periods.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date filter
            end_date: End date filter
            resample_rule: Optional time-based resampling rule (e.g., '1T', '5T', '1H')
            volume_bucket_size: Optional volume bucket size for volume-based aggregation
            remove_gaps: Remove time gaps between contracts
            add_roll_markers: Add markers to identify contract roll dates

        Returns:
            Continuous DataFrame with M0 intraday data
        """
        intraday_data_list = self.load_m0_intraday_data(symbol, start_date, end_date)

        if not intraday_data_list:
            return pd.DataFrame()

        # Process each M0 period
        dataframes = []
        roll_dates = []

        for i, intraday_data in enumerate(intraday_data_list):
            df = intraday_data.data.copy()

            # Apply volume bucket aggregation if requested
            if volume_bucket_size is not None:
                if self.logger:
                    self.logger.info(f"Aggregating {intraday_data.contract_code} into "
                                   f"{volume_bucket_size} volume buckets")
                df = self._aggregate_volume_buckets(df, volume_bucket_size)

            # Apply time-based resampling if requested (and not using volume buckets)
            elif resample_rule is not None:
                if self.logger:
                    self.logger.info(f"Resampling {intraday_data.contract_code} to {resample_rule}")
                df = self._resample_ohlcv(df, rule=resample_rule)

            if len(df) == 0:
                continue

            # Add contract metadata
            df['contract_code'] = intraday_data.contract_code
            df['contract_start'] = intraday_data.start_date
            df['contract_end'] = intraday_data.end_date

            # Add roll marker (True for last day of contract)
            if add_roll_markers:
                last_day = df.index[-1].date()
                df['is_roll_day'] = df.index.date == last_day
                if i < len(intraday_data_list) - 1:  # Not the last contract
                    roll_dates.append(df.index[-1])

            dataframes.append(df)

        if not dataframes:
            return pd.DataFrame()

        # Concatenate all M0 periods
        continuous_df = pd.concat(dataframes, axis=0).sort_index()

        # Remove duplicate timestamps at roll boundaries
        continuous_df = continuous_df[~continuous_df.index.duplicated(keep='first')]

        # Optionally remove gaps between contracts
        if remove_gaps and len(roll_dates) > 0:
            # Calculate time gaps and optionally fill or mark them
            for roll_date in roll_dates:
                next_idx = continuous_df.index.get_indexer([roll_date], method='bfill')[0] + 1
                if next_idx < len(continuous_df):
                    gap_duration = (continuous_df.index[next_idx] - roll_date).total_seconds()
                    if gap_duration > 86400:  # Gap > 1 day
                        if self.logger:
                            self.logger.info(f"Gap detected at roll date {roll_date}: "
                                           f"{gap_duration/3600:.1f} hours")

        # Add summary statistics
        if add_roll_markers and len(roll_dates) > 0:
            continuous_df.attrs['roll_dates'] = roll_dates
            continuous_df.attrs['num_contracts'] = len(intraday_data_list)

        if self.logger:
            total_records = len(continuous_df)
            date_range = f"{continuous_df.index[0]} to {continuous_df.index[-1]}"

            if volume_bucket_size:
                avg_volume_per_bucket = continuous_df['TotalVolume'].mean()
                self.logger.info(f"Created continuous M0 dataset for {symbol}: "
                               f"{total_records:,} volume buckets ({volume_bucket_size} target volume) "
                               f"from {date_range}, avg volume/bucket: {avg_volume_per_bucket:.0f}")
            else:
                self.logger.info(f"Created continuous M0 dataset for {symbol}: "
                               f"{total_records:,} records from {date_range}")

        return continuous_df

    async def get_continuous_m0_data_async(self, symbol: str,
                                          start_date: Optional[datetime] = None,
                                          end_date: Optional[datetime] = None,
                                          resample_rule: Optional[str] = None,
                                          volume_bucket_size: Optional[int] = None,
                                          remove_gaps: bool = True,
                                          add_roll_markers: bool = True,
                                          batch_size: int = 5) -> pd.DataFrame:
        """
        Asynchronously get continuous M0 intraday data by concatenating all M0 periods.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date filter
            end_date: End date filter
            resample_rule: Optional time-based resampling rule (e.g., '1T', '5T', '1H')
            volume_bucket_size: Optional volume bucket size for volume-based aggregation
            remove_gaps: Remove time gaps between contracts
            add_roll_markers: Add markers to identify contract roll dates
            batch_size: Number of SCID files to process concurrently

        Returns:
            Continuous DataFrame with M0 intraday data
        """
        intraday_data_list = await self.load_m0_intraday_data_async(
            symbol, start_date, end_date, resample_rule, volume_bucket_size, batch_size
        )

        if not intraday_data_list:
            return pd.DataFrame()

        # Process each M0 period (already processed async)
        dataframes = []
        roll_dates = []

        for i, intraday_data in enumerate(intraday_data_list):
            df = intraday_data.data.copy()

            if len(df) == 0:
                continue

            # Add contract metadata
            df['contract_code'] = intraday_data.contract_code
            df['contract_start'] = intraday_data.start_date
            df['contract_end'] = intraday_data.end_date

            # Add roll marker (True for last day of contract)
            if add_roll_markers:
                last_day = df.index[-1].date()
                df['is_roll_day'] = df.index.date == last_day
                if i < len(intraday_data_list) - 1:  # Not the last contract
                    roll_dates.append(df.index[-1])

            dataframes.append(df)

        if not dataframes:
            return pd.DataFrame()

        # Concatenate all M0 periods
        continuous_df = pd.concat(dataframes, axis=0).sort_index()

        # Remove duplicate timestamps at roll boundaries
        continuous_df = continuous_df[~continuous_df.index.duplicated(keep='first')]

        # Optionally remove gaps between contracts
        if remove_gaps and len(roll_dates) > 0:
            # Calculate time gaps and optionally fill or mark them
            for roll_date in roll_dates:
                next_idx = continuous_df.index.get_indexer([roll_date], method='bfill')[0] + 1
                if next_idx < len(continuous_df):
                    gap_duration = (continuous_df.index[next_idx] - roll_date).total_seconds()
                    if gap_duration > 86400:  # Gap > 1 day
                        if self.logger:
                            self.logger.info(f"Gap detected at roll date {roll_date}: "
                                           f"{gap_duration/3600:.1f} hours")

        # Add summary statistics
        if add_roll_markers and len(roll_dates) > 0:
            continuous_df.attrs['roll_dates'] = roll_dates
            continuous_df.attrs['num_contracts'] = len(intraday_data_list)

        if self.logger:
            total_records = len(continuous_df)
            date_range = f"{continuous_df.index[0]} to {continuous_df.index[-1]}"

            if volume_bucket_size:
                avg_volume_per_bucket = continuous_df['TotalVolume'].mean()
                self.logger.info(f"Created continuous M0 dataset for {symbol}: "
                               f"{total_records:,} volume buckets ({volume_bucket_size} target volume) "
                               f"from {date_range}, avg volume/bucket: {avg_volume_per_bucket:.0f} (async)")
            else:
                self.logger.info(f"Created continuous M0 dataset for {symbol}: "
                               f"{total_records:,} records from {date_range} (async)")

        return continuous_df

    async def export_continuous_m0_to_parquet(self,
                                            symbol: str,
                                            output_path: Union[str, Path],
                                            start_date: Optional[datetime] = None,
                                            end_date: Optional[datetime] = None,
                                            resample_rule: Optional[str] = None,
                                            volume_bucket_size: Optional[int] = None,
                                            batch_size: int = 8,
                                            max_memory_mb: int = 2048,
                                            single_file: bool = True,
                                            compression: str = "snappy") -> Dict[str, any]:
        """
        Export continuous M0 data to Parquet file(s) with memory-efficient processing.

        Args:
            symbol: Symbol to process (e.g., 'CL_F')
            output_path: Output directory or file path
            start_date: Start date filter
            end_date: End date filter
            resample_rule: Optional time-based resampling rule
            volume_bucket_size: Optional volume bucket aggregation
            batch_size: Number of files to process concurrently
            max_memory_mb: Maximum memory usage before writing to disk
            single_file: If True, attempt to write all data to one file; if False, write per contract
            compression: Parquet compression algorithm

        Returns:
            Dictionary with export statistics and file paths
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.logger:
            self.logger.info(f"Starting Parquet export for {symbol} continuous M0 data")
            self.logger.info(f"Output path: {output_path}, single_file: {single_file}")

        # Get M0 periods
        m0_periods = self.identify_m0_periods(symbol, start_date, end_date)
        if not m0_periods:
            raise ValueError(f"No M0 periods found for {symbol}")

        export_stats = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'single_file': single_file,
            'compression': compression,
            'file_paths': [],
            'total_records': 0,
            'total_size_mb': 0,
            'processing_time_seconds': 0,
            'memory_peak_mb': 0,
            'contract_periods': len(m0_periods)
        }

        start_time = time.time()

        if single_file:
            # Attempt to write all data to a single Parquet file
            return await self._export_single_parquet_file(
                symbol, m0_periods, output_path, resample_rule,
                volume_bucket_size, batch_size, max_memory_mb,
                compression, export_stats, start_time
            )
        else:
            # Write individual Parquet files per contract period
            return await self._export_multiple_parquet_files(
                symbol, m0_periods, output_path, resample_rule,
                volume_bucket_size, batch_size, compression,
                export_stats, start_time
            )

    async def _export_single_parquet_file(self,
                                        symbol: str,
                                        m0_periods: List[ContractPeriod],
                                        output_path: Path,
                                        resample_rule: Optional[str],
                                        volume_bucket_size: Optional[int],
                                        batch_size: int,
                                        max_memory_mb: int,
                                        compression: str,
                                        export_stats: Dict,
                                        start_time: float) -> Dict[str, any]:
        """Export all M0 data to a single Parquet file with chunked processing."""
        import psutil
        import time

        process = psutil.Process()

        # Determine output file path
        output_file = output_path / f"{symbol}_continuous_m0.parquet"

        # Initialize Parquet writer for streaming
        parquet_writer = None
        schema = None

        try:
            total_records_written = 0
            current_memory_mb = 0

            for i, period in enumerate(m0_periods):
                if self.logger:
                    self.logger.info(f"Processing period {i+1}/{len(m0_periods)}: "
                                   f"{period.contract_code} ({period.start_date} to {period.end_date})")

                # Check memory usage
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                export_stats['memory_peak_mb'] = max(export_stats['memory_peak_mb'], current_memory_mb)

                if current_memory_mb > max_memory_mb:
                    if self.logger:
                        self.logger.warning(f"Memory usage ({current_memory_mb:.1f}MB) exceeds limit "
                                          f"({max_memory_mb}MB). Consider using single_file=False")

                # Load data for this period using FastScidReader
                if not period.file_path or not period.file_path.exists():
                    if self.logger:
                        self.logger.warning(f"SCID file not found for {period.contract_code}, skipping")
                    continue

                try:
                    with FastScidReader(str(period.file_path)) as reader:
                        # Use time filtering if period has specific dates
                        start_ms = int(period.start_date.timestamp() * 1000) if period.start_date else None
                        end_ms = int(period.end_date.timestamp() * 1000) if period.end_date else None

                        # Get pandas DataFrame
                        df = reader.to_pandas(start_ms=start_ms, end_ms=end_ms, tz='UTC')

                        if len(df) == 0:
                            continue

                        # Apply processing
                        if volume_bucket_size:
                            df = self._aggregate_volume_buckets(df, volume_bucket_size)
                        elif resample_rule:
                            df = self._resample_ohlcv(df, rule=resample_rule)

                        # Add metadata columns
                        df['contract_code'] = period.contract_code
                        df['symbol'] = symbol

                        # Convert to PyArrow Table
                        table = pa.Table.from_pandas(df, preserve_index=True)

                        # Initialize writer with schema from first table
                        if parquet_writer is None:
                            schema = table.schema
                            parquet_writer = pq.ParquetWriter(
                                output_file,
                                schema=schema,
                                compression=compression,
                                write_statistics=True
                            )

                        # Write table to Parquet file
                        parquet_writer.write_table(table)
                        total_records_written += len(df)

                        if self.logger:
                            self.logger.info(f"  Wrote {len(df):,} records for {period.contract_code}")

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error processing {period.contract_code}: {e}")
                    continue

            # Close writer
            if parquet_writer:
                parquet_writer.close()

            # Update export statistics
            if output_file.exists():
                export_stats['file_paths'] = [str(output_file)]
                export_stats['total_records'] = total_records_written
                export_stats['total_size_mb'] = output_file.stat().st_size / (1024 * 1024)
                export_stats['processing_time_seconds'] = time.time() - start_time

                if self.logger:
                    self.logger.info(f"Successfully exported {total_records_written:,} records "
                                   f"to {output_file.name} ({export_stats['total_size_mb']:.1f}MB)")

            return export_stats

        except Exception as e:
            # Clean up on error
            if parquet_writer:
                parquet_writer.close()
            if output_file.exists():
                output_file.unlink()
            raise RuntimeError(f"Failed to export single Parquet file: {e}")

    async def _export_multiple_parquet_files(self,
                                           symbol: str,
                                           m0_periods: List[ContractPeriod],
                                           output_path: Path,
                                           resample_rule: Optional[str],
                                           volume_bucket_size: Optional[int],
                                           batch_size: int,
                                           compression: str,
                                           export_stats: Dict,
                                           start_time: float) -> Dict[str, any]:
        """Export M0 data to individual Parquet files per contract period."""
        import time

        exported_files = []
        total_records = 0
        total_size_mb = 0

        # Process in batches for efficiency
        for i in range(0, len(m0_periods), batch_size):
            batch = m0_periods[i:i + batch_size]

            if self.logger:
                self.logger.info(f"Processing batch {i//batch_size + 1}: "
                               f"contracts {i+1}-{min(i+batch_size, len(m0_periods))}")

            # Process batch concurrently
            tasks = []
            for period in batch:
                task = self._export_single_contract_parquet(
                    symbol, period, output_path, resample_rule,
                    volume_bucket_size, compression
                )
                tasks.append(task)

            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    if self.logger:
                        self.logger.error(f"Batch export error: {result}")
                    continue

                if result and 'file_path' in result:
                    exported_files.append(result['file_path'])
                    total_records += result.get('records', 0)
                    total_size_mb += result.get('size_mb', 0)

        # Update export statistics
        export_stats['file_paths'] = exported_files
        export_stats['total_records'] = total_records
        export_stats['total_size_mb'] = total_size_mb
        export_stats['processing_time_seconds'] = time.time() - start_time

        if self.logger:
            self.logger.info(f"Exported {len(exported_files)} Parquet files with "
                           f"{total_records:,} total records ({total_size_mb:.1f}MB)")

        return export_stats

    async def _export_single_contract_parquet(self,
                                            symbol: str,
                                            period: ContractPeriod,
                                            output_path: Path,
                                            resample_rule: Optional[str],
                                            volume_bucket_size: Optional[int],
                                            compression: str) -> Dict[str, any]:
        """Export a single contract period to Parquet file."""
        if not period.file_path or not period.file_path.exists():
            return {'error': f"SCID file not found for {period.contract_code}"}

        # Generate output filename
        date_str = period.start_date.strftime('%Y%m%d')
        output_file = output_path / f"{symbol}_{period.contract_code}_{date_str}.parquet"

        try:
            with FastScidReader(str(period.file_path)) as reader:
                # Use time filtering
                start_ms = int(period.start_date.timestamp() * 1000) if period.start_date else None
                end_ms = int(period.end_date.timestamp() * 1000) if period.end_date else None

                # Get DataFrame
                df = reader.to_pandas(start_ms=start_ms, end_ms=end_ms, tz='UTC')

                if len(df) == 0:
                    return {'error': f"No data found for {period.contract_code}"}

                # Apply processing
                if volume_bucket_size:
                    df = self._aggregate_volume_buckets(df, volume_bucket_size)
                elif resample_rule:
                    df = self._resample_ohlcv(df, rule=resample_rule)

                # Add metadata
                df['contract_code'] = period.contract_code
                df['symbol'] = symbol
                df['front_month_start'] = period.start_date
                df['front_month_end'] = period.end_date

                # Export to Parquet
                df.to_parquet(
                    output_file,
                    compression=compression,
                    index=True,
                    engine='pyarrow'
                )

                file_size_mb = output_file.stat().st_size / (1024 * 1024)

                return {
                    'file_path': str(output_file),
                    'contract_code': period.contract_code,
                    'records': len(df),
                    'size_mb': file_size_mb,
                    'date_range': (period.start_date, period.end_date)
                }

        except Exception as e:
            return {'error': f"Failed to export {period.contract_code}: {e}"}

    def load_continuous_m0_from_parquet(self,
                                      parquet_path: Union[str, Path],
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None,
                                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load continuous M0 data from Parquet file(s) with optional filtering.

        Args:
            parquet_path: Path to Parquet file or directory containing Parquet files
            start_date: Optional start date filter
            end_date: Optional end date filter
            columns: Optional list of columns to load

        Returns:
            DataFrame with continuous M0 data
        """
        parquet_path = Path(parquet_path)

        if parquet_path.is_file():
            # Single Parquet file
            df = pd.read_parquet(parquet_path, columns=columns, engine='pyarrow')
        elif parquet_path.is_dir():
            # Directory with multiple Parquet files
            parquet_files = list(parquet_path.glob("*.parquet"))
            if not parquet_files:
                raise ValueError(f"No Parquet files found in {parquet_path}")

            dfs = []
            for file_path in sorted(parquet_files):
                df_chunk = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
                dfs.append(df_chunk)

            df = pd.concat(dfs, axis=0).sort_index()
        else:
            raise ValueError(f"Parquet path not found: {parquet_path}")

        # Apply date filtering if requested
        if start_date or end_date:
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date, tz='UTC')]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date, tz='UTC')]

        # Remove duplicates that might occur at file boundaries
        df = df[~df.index.duplicated(keep='first')]

        if self.logger:
            self.logger.info(f"Loaded {len(df):,} records from Parquet: "
                           f"{df.index[0]} to {df.index[-1]}")

        return df

    def get_parquet_file_info(self, parquet_path: Union[str, Path]) -> Dict[str, any]:
        """
        Get metadata information about Parquet file(s).

        Args:
            parquet_path: Path to Parquet file or directory

        Returns:
            Dictionary with file information
        """
        parquet_path = Path(parquet_path)
        info = {
            'path': str(parquet_path),
            'type': 'unknown',
            'files': [],
            'total_size_mb': 0,
            'total_rows': 0,
            'schema': None,
            'date_range': None
        }

        try:
            if parquet_path.is_file():
                # Single file
                info['type'] = 'single_file'
                file_info = pq.read_metadata(parquet_path)
                info['total_rows'] = file_info.num_rows
                info['total_size_mb'] = parquet_path.stat().st_size / (1024 * 1024)
                info['schema'] = file_info.schema.to_arrow_schema()
                info['files'] = [str(parquet_path)]

                # Try to get date range from index
                df_sample = pd.read_parquet(parquet_path, columns=[], engine='pyarrow')
                if len(df_sample) > 0:
                    info['date_range'] = (df_sample.index.min(), df_sample.index.max())

            elif parquet_path.is_dir():
                # Directory with multiple files
                info['type'] = 'directory'
                parquet_files = list(parquet_path.glob("*.parquet"))

                if parquet_files:
                    total_rows = 0
                    total_size = 0
                    min_date = None
                    max_date = None

                    for file_path in parquet_files:
                        file_info = pq.read_metadata(file_path)
                        total_rows += file_info.num_rows
                        total_size += file_path.stat().st_size
                        info['files'].append(str(file_path))

                        # Get schema from first file
                        if info['schema'] is None:
                            info['schema'] = file_info.schema.to_arrow_schema()

                        # Update date range
                        df_sample = pd.read_parquet(file_path, columns=[], engine='pyarrow')
                        if len(df_sample) > 0:
                            file_min = df_sample.index.min()
                            file_max = df_sample.index.max()
                            min_date = min(min_date, file_min) if min_date else file_min
                            max_date = max(max_date, file_max) if max_date else file_max

                    info['total_rows'] = total_rows
                    info['total_size_mb'] = total_size / (1024 * 1024)
                    info['date_range'] = (min_date, max_date) if min_date else None

        except Exception as e:
            info['error'] = str(e)

        return info

    def get_contract_statistics(self, symbol: str) -> Dict[str, any]:
        """Get statistics about available contracts for a symbol."""
        m0_periods = self.identify_m0_periods(symbol)
        scid_files = self.get_scid_files_for_symbol(symbol)

        stats = {
            'symbol': symbol,
            'total_scid_files': len(scid_files),
            'total_m0_periods': len(m0_periods),
            'files_with_m0_mapping': sum(1 for p in m0_periods if p.file_path is not None),
            'date_range': None,
            'contracts': []
        }

        if m0_periods:
            stats['date_range'] = {
                'start': min(p.start_date for p in m0_periods),
                'end': max(p.end_date for p in m0_periods)
            }

            for period in m0_periods:
                contract_info = {
                    'contract_code': period.contract_code,
                    'start_date': period.start_date,
                    'end_date': period.end_date,
                    'duration_days': (period.end_date - period.start_date).days,
                    'has_scid_file': period.file_path is not None,
                    'scid_file': period.file_path.name if period.file_path else None
                }
                stats['contracts'].append(contract_info)

        return stats

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._file_cache.clear()
        self._contract_periods.clear()
        self._curve_data_cache.clear()

        if self.logger:
            self.logger.info("Cleared internal caches")

    def validate_scid_files(self, symbol: str) -> Dict[str, any]:
        """
        Validate SCID files for a symbol by checking readability and basic statistics.

        Returns:
            Dictionary with validation results
        """
        scid_files = self.get_scid_files_for_symbol(symbol)
        validation_results = {
            'symbol': symbol,
            'total_files': len(scid_files),
            'valid_files': 0,
            'invalid_files': 0,
            'file_details': []
        }

        for scid_file in scid_files:
            file_result = {
                'filename': scid_file.name,
                'file_path': str(scid_file),
                'valid': False,
                'error': None,
                'records': 0,
                'date_range': None,
                'file_size_mb': scid_file.stat().st_size / (1024 * 1024)
            }

            try:
                with FastScidReader(str(scid_file)) as reader:
                    file_result['records'] = len(reader)
                    file_result['variant'] = reader.schema.variant
                    file_result['scale'] = getattr(reader, 'scale', 1.0)  # FastScidReader doesn't have scale

                    if len(reader) > 0:
                        # Get first and last timestamps using FastScidReader's efficient API
                        times_ms = reader.times_epoch_ms()
                        first_ts = pd.to_datetime(times_ms[0], unit='ms', utc=True).to_pydatetime()
                        last_ts = pd.to_datetime(times_ms[-1], unit='ms', utc=True).to_pydatetime()

                        file_result['date_range'] = {
                            'start': first_ts,
                            'end': last_ts
                        }

                    file_result['valid'] = True
                    validation_results['valid_files'] += 1

            except Exception as e:
                file_result['error'] = str(e)
                validation_results['invalid_files'] += 1

            validation_results['file_details'].append(file_result)

        return validation_results

    def analyze_volume_patterns(self, symbol: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               bucket_sizes: List[int] = None) -> Dict[str, any]:
        """
        Analyze volume patterns for different bucket sizes.

        Args:
            symbol: Trading symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            bucket_sizes: List of volume bucket sizes to analyze

        Returns:
            Dictionary with volume pattern analysis
        """
        if bucket_sizes is None:
            bucket_sizes = [100, 500, 1000, 2000, 5000]

        analysis = {
            'symbol': symbol,
            'date_range': {'start': start_date, 'end': end_date},
            'bucket_analysis': {},
            'recommendations': {}
        }

        # Load raw tick data first
        raw_data = self.get_continuous_m0_data(symbol, start_date, end_date)

        if len(raw_data) == 0:
            return analysis

        total_volume = raw_data['TotalVolume'].sum()
        total_ticks = len(raw_data)
        avg_volume_per_tick = total_volume / total_ticks if total_ticks > 0 else 0

        analysis['raw_statistics'] = {
            'total_ticks': total_ticks,
            'total_volume': total_volume,
            'avg_volume_per_tick': avg_volume_per_tick,
            'volume_std': raw_data['TotalVolume'].std(),
            'max_volume_tick': raw_data['TotalVolume'].max(),
            'min_volume_tick': raw_data['TotalVolume'].min()
        }

        # Analyze each bucket size
        for bucket_size in bucket_sizes:
            try:
                bucketed_data = self._aggregate_volume_buckets(raw_data, bucket_size)

                if len(bucketed_data) == 0:
                    continue

                bucket_stats = {
                    'bucket_size': bucket_size,
                    'total_buckets': len(bucketed_data),
                    'avg_volume_per_bucket': bucketed_data['TotalVolume'].mean(),
                    'volume_std': bucketed_data['TotalVolume'].std(),
                    'avg_ticks_per_bucket': bucketed_data['TickCount'].mean(),
                    'avg_duration_seconds': bucketed_data['BucketDuration'].mean(),
                    'min_duration': bucketed_data['BucketDuration'].min(),
                    'max_duration': bucketed_data['BucketDuration'].max(),
                    'coverage_ratio': len(bucketed_data) * bucket_size / total_volume
                }

                # Calculate efficiency metrics
                bucket_stats['compression_ratio'] = total_ticks / len(bucketed_data) if len(bucketed_data) > 0 else 0
                bucket_stats['time_efficiency'] = bucket_stats['avg_duration_seconds'] / 60  # minutes per bucket

                analysis['bucket_analysis'][bucket_size] = bucket_stats

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error analyzing bucket size {bucket_size}: {e}")

        # Generate recommendations
        if analysis['bucket_analysis']:
            # Find optimal bucket size based on balance of compression and time efficiency
            best_bucket = None
            best_score = 0

            for bucket_size, stats in analysis['bucket_analysis'].items():
                # Score based on compression ratio and reasonable time per bucket (5-60 minutes)
                compression_score = min(stats['compression_ratio'] / 100, 1.0)  # Normalize to max 100:1
                time_score = 1.0 if 5 <= stats['time_efficiency'] <= 60 else 0.5

                combined_score = compression_score * 0.7 + time_score * 0.3

                if combined_score > best_score:
                    best_score = combined_score
                    best_bucket = bucket_size

            analysis['recommendations'] = {
                'optimal_bucket_size': best_bucket,
                'optimal_score': best_score,
                'small_bucket_sizes': [size for size in bucket_sizes if size < avg_volume_per_tick * 5],
                'large_bucket_sizes': [size for size in bucket_sizes if size > avg_volume_per_tick * 50],
                'suggested_range': {
                    'min': int(avg_volume_per_tick * 2),
                    'max': int(avg_volume_per_tick * 20),
                    'optimal': int(avg_volume_per_tick * 10)
                }
            }

        return analysis

    def get_volume_bucket_summary(self, symbol: str,
                                 volume_bucket_size: int,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, any]:
        """
        Get summary statistics for a specific volume bucket configuration.
        """
        bucketed_data = self.get_continuous_m0_data(
            symbol, start_date, end_date, volume_bucket_size=volume_bucket_size
        )

        if len(bucketed_data) == 0:
            return {'error': 'No data available'}

        summary = {
            'symbol': symbol,
            'bucket_size': volume_bucket_size,
            'total_buckets': len(bucketed_data),
            'date_range': {
                'start': bucketed_data.index[0],
                'end': bucketed_data.index[-1],
                'duration_days': (bucketed_data.index[-1] - bucketed_data.index[0]).days
            },
            'volume_statistics': {
                'total_volume': bucketed_data['TotalVolume'].sum(),
                'avg_volume_per_bucket': bucketed_data['TotalVolume'].mean(),
                'volume_std': bucketed_data['TotalVolume'].std(),
                'min_volume': bucketed_data['TotalVolume'].min(),
                'max_volume': bucketed_data['TotalVolume'].max()
            },
            'timing_statistics': {
                'avg_duration_minutes': bucketed_data['BucketDuration'].mean() / 60,
                'min_duration_seconds': bucketed_data['BucketDuration'].min(),
                'max_duration_seconds': bucketed_data['BucketDuration'].max(),
                'std_duration_minutes': bucketed_data['BucketDuration'].std() / 60
            },
            'tick_statistics': {
                'avg_ticks_per_bucket': bucketed_data['TickCount'].mean(),
                'min_ticks_per_bucket': bucketed_data['TickCount'].min(),
                'max_ticks_per_bucket': bucketed_data['TickCount'].max(),
                'total_ticks': bucketed_data['TickCount'].sum()
            },
            'price_statistics': {
                'avg_bucket_range': (bucketed_data['High'] - bucketed_data['Low']).mean(),
                'max_bucket_range': (bucketed_data['High'] - bucketed_data['Low']).max(),
                'avg_price_move': abs(bucketed_data['Close'] - bucketed_data['Open']).mean()
            }
        }

        # Add contract roll information if available
        if 'contract_code' in bucketed_data.columns:
            unique_contracts = bucketed_data['contract_code'].nunique()
            summary['contract_information'] = {
                'unique_contracts': unique_contracts,
                'buckets_per_contract': len(bucketed_data) / unique_contracts if unique_contracts > 0 else 0
            }

        return summary