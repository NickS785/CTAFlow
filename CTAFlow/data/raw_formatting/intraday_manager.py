#!/usr/bin/env python3
"""
Intraday file management for SCID contract data using sierrapy.

This module provides the IntradayFileManager class for discovering, parsing, and
managing intraday SCID files using the sierrapy.parser ScidReader which includes
built-in front month loading and volume bucketing capabilities.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union, Set
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

from sierrapy.parser import ScidReader

from ...config import DLY_DATA_PATH, MARKET_DATA_PATH
from ..data_client import DataClient
from ..update_management import (
    INTRADAY_UPDATE_EVENT,
    get_update_metadata_store,
    prepare_dataframe_for_append,
    summarize_append_result,
)
import h5py


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
    Manager for intraday SCID contract files using sierrapy.

    This class handles:
    - Discovery of SCID files in DLY_DATA_PATH
    - Parsing contract codes from filenames
    - Integration with curve data to identify M0 periods
    - Extraction of intraday data for front month contracts using ScidReader
    - Automatic front month loading via ScidReader.load_front_month()
    - Volume bucketing via ScidReader.bucket_volume()
    """

    def __init__(self,
                 data_path: Optional[Path] = None,
                 market_data_path: Optional[Path] = None,
                 enable_logging: bool = True):
        """
        Initialize the intraday file manager.

        Args:
            data_path: Path to directory containing SCID files (default: DLY_DATA_PATH)
            market_data_path: Path to HDF5 market data file (default: MARKET_DATA_PATH)
            enable_logging: Enable logging for debugging
        """
        self.data_path = Path(data_path) if data_path else Path(DLY_DATA_PATH)
        self.market_data_path = Path(market_data_path) if market_data_path else MARKET_DATA_PATH
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

        # Initialize DataClient for HDF5 operations
        self.data_client = DataClient(market_path=self.market_data_path)

        # Discover SCID files on initialization
        self._discover_scid_files()

    def _collect_scid_files(self, folder: Optional[Path] = None) -> Dict[str, List[Path]]:
        """Return mapping of symbols to SCID paths using a filename parser."""
        base = Path(folder) if folder else self.data_path
        mapping: Dict[str, List[Path]] = {}
        if not base.exists():
            return mapping

        pattern = re.compile(r'^([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{2})-([A-Z]+)\.scid$', re.IGNORECASE)

        for entry in base.iterdir():
            if not entry.is_file() or entry.suffix.lower() != '.scid':
                continue
            match = pattern.match(entry.name)
            if not match:
                continue
            base_symbol = match.group(1).upper()
            full_symbol = f"{base_symbol}_F"
            mapping.setdefault(full_symbol, []).append(entry)

        for files in mapping.values():
            files.sort()
        return mapping

    def _discover_scid_files(self) -> None:
        """Discover and cache SCID files under ``self.data_path``."""
        mapping = self._collect_scid_files(self.data_path)
        if not mapping:
            if self.logger:
                self.logger.warning(f"No SCID files discovered in {self.data_path}")
            return

        self._scid_files = mapping

        if self.logger:
            total_files = sum(len(files) for files in mapping.values())
            self.logger.info(
                f"Discovered {total_files} SCID files for {len(mapping)} symbols"
            )

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

    def _load_front_month_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load front month data directly from HDF5 market data.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')

        Returns:
            DataFrame with front month contract codes indexed by date, or None if not found
        """
        try:
            with h5py.File(self.market_data_path, 'r') as f:
                front_path = f'market/{symbol}/front/table'
                if front_path not in f:
                    if self.logger:
                        self.logger.warning(f"No front month data found for {symbol} in HDF5")
                    return None

                front_data = f[front_path][:]

                # Convert to DataFrame with timezone-naive timestamps
                timestamps = pd.to_datetime(front_data['index'], utc=True).tz_localize(None)
                contracts = [code.decode('utf-8') for code in front_data['front']]

                df = pd.DataFrame({
                    'front_contract': contracts
                }, index=timestamps)

                if self.logger:
                    self.logger.info(f"Loaded {len(df)} front month records for {symbol}")

                return df

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading front month data for {symbol}: {e}")
            return None

    def _load_expiry_data(self, symbol: str) -> Optional[Dict[str, datetime]]:
        """
        Load contract expiry dates from HDF5 market data.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')

        Returns:
            Dictionary mapping contract codes to expiry dates, or None if not found
        """
        try:
            with h5py.File(self.market_data_path, 'r') as f:
                expiry_path = f'market/{symbol}/expiry/table'
                if expiry_path not in f:
                    if self.logger:
                        self.logger.warning(f"No expiry data found for {symbol} in HDF5")
                    return None

                expiry_data = f[expiry_path][:]

                # Convert to dictionary with timezone-naive dates
                expiry_dict = {}
                for row in expiry_data:
                    contract_code = row['index'].decode('utf-8')
                    expiry_timestamp = pd.to_datetime(row['expiry_date'], utc=True).tz_localize(None)
                    expiry_dict[contract_code] = expiry_timestamp.to_pydatetime()

                if self.logger:
                    self.logger.info(f"Loaded expiry data for {len(expiry_dict)} contracts for {symbol}")

                return expiry_dict

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading expiry data for {symbol}: {e}")
            return None

    def identify_m0_periods(self, symbol: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[ContractPeriod]:
        """
        Identify periods when each contract was the front month (M0) using simplified approach.

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

        # Try to load front month data from HDF5 first
        front_df = self._load_front_month_data(symbol)

        if front_df is not None:
            # Use HDF5 front month data
            contract_periods = self._extract_periods_from_front_data(
                front_df, symbol, start_date, end_date
            )
        else:
            # Fallback: Calculate periods from available SCID files and expiry dates
            contract_periods = self._calculate_periods_from_expiry(
                symbol, start_date, end_date
            )

        # Match contract periods with SCID files
        scid_files = self.get_scid_files_for_symbol(symbol)
        for period in contract_periods:
            period.file_path = self._find_matching_scid_file(period.contract_code, scid_files)

        self._contract_periods[cache_key] = contract_periods

        if self.logger:
            self.logger.info(f"Identified {len(contract_periods)} M0 periods for {symbol}")

        return contract_periods

    def _extract_periods_from_front_data(self, front_df: pd.DataFrame, symbol: str,
                                       start_date: Optional[datetime] = None,
                                       end_date: Optional[datetime] = None) -> List[ContractPeriod]:
        """Extract contract periods from front month DataFrame."""
        # Apply date filtering
        if start_date:
            start_ts = pd.Timestamp(start_date)
            front_df = front_df[front_df.index >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date)
            front_df = front_df[front_df.index <= end_ts]

        if len(front_df) == 0:
            return []

        contract_periods = []
        current_contract = None
        period_start = None

        for date, row in front_df.iterrows():
            contract_code = row['front_contract']

            if contract_code != current_contract:
                # Contract changed
                if current_contract is not None and period_start is not None:
                    # Close previous period
                    contract_periods.append(ContractPeriod(
                        contract_code=current_contract,
                        symbol=symbol,
                        start_date=period_start.to_pydatetime(),
                        end_date=(date - timedelta(days=1)).to_pydatetime()
                    ))

                # Start new period
                current_contract = contract_code
                period_start = date

        # Close final period
        if current_contract is not None and period_start is not None:
            contract_periods.append(ContractPeriod(
                contract_code=current_contract,
                symbol=symbol,
                start_date=period_start.to_pydatetime(),
                end_date=front_df.index[-1].to_pydatetime()
            ))

        return contract_periods

    def _calculate_periods_from_expiry(self, symbol: str,
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> List[ContractPeriod]:
        """
        Fallback method: Calculate M0 periods from expiry dates with 30 DTE rolling.
        """
        expiry_dict = self._load_expiry_data(symbol)
        scid_files = self.get_scid_files_for_symbol(symbol)

        if not expiry_dict or not scid_files:
            if self.logger:
                self.logger.warning(f"No expiry data or SCID files available for {symbol}")
            return []

        # Parse available contracts from SCID files
        available_contracts = []
        for scid_file in scid_files:
            parsed = self._parse_contract_from_filename(scid_file)
            if parsed:
                base_symbol, month_code, year_code, exchange = parsed
                contract_code = f"{month_code}{year_code}"
                if contract_code in expiry_dict:
                    available_contracts.append({
                        'code': contract_code,
                        'expiry': expiry_dict[contract_code],
                        'file_path': scid_file
                    })

        if not available_contracts:
            return []

        # Sort by expiry date
        available_contracts.sort(key=lambda x: x['expiry'])

        contract_periods = []

        for i, contract in enumerate(available_contracts):
            expiry_date = contract['expiry']

            # Calculate when this contract becomes front month (30 DTE before expiry)
            roll_date = expiry_date - timedelta(days=30)

            # Determine period start (either roll date or when previous contract expires)
            if i == 0:
                # First contract starts from earliest available date
                period_start = start_date or (expiry_date - timedelta(days=90))
            else:
                # Start when previous contract rolled
                prev_expiry = available_contracts[i-1]['expiry']
                period_start = prev_expiry - timedelta(days=30)

            # Period ends at roll date (30 DTE)
            period_end = roll_date

            # Apply date filtering
            if start_date and period_end < start_date:
                continue
            if end_date and period_start > end_date:
                continue

            # Adjust boundaries
            if start_date and period_start < start_date:
                period_start = start_date
            if end_date and period_end > end_date:
                period_end = end_date

            contract_periods.append(ContractPeriod(
                contract_code=contract['code'],
                symbol=symbol,
                start_date=period_start,
                end_date=period_end,
                file_path=contract['file_path']
            ))

        return contract_periods

    def _find_matching_scid_file(self, contract_code: str, scid_files: List[Path]) -> Optional[Path]:
        """Find SCID file matching the contract code."""
        for scid_file in scid_files:
            parsed = self._parse_contract_from_filename(scid_file)
            if parsed:
                base_symbol, month_code, year_code, exchange = parsed
                file_contract_code = f"{month_code}{year_code}"
                if file_contract_code == contract_code:
                    return scid_file
        return None

    def load_front_month_continuous(self,
                                   symbol: str,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   volume_bucket_size: Optional[int] = None,
                                   resample_rule: Optional[str] = None) -> pd.DataFrame:
        """
        Load continuous front month data using ScidReader.load_front_month().

        This method leverages sierrapy's built-in front month loading capability
        which automatically stitches together front month contracts.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date filter
            end_date: End date filter
            volume_bucket_size: Optional volume bucket size for bucketing
            resample_rule: Optional time-based resampling rule (e.g., '1T', '5T', '1H')

        Returns:
            Continuous DataFrame with front month intraday data
        """
        scid_files = self.get_scid_files_for_symbol(symbol)

        if not scid_files:
            if self.logger:
                self.logger.warning(f"No SCID files found for {symbol}")
            return pd.DataFrame()

        if self.logger:
            self.logger.info(f"Loading continuous front month data for {symbol} using ScidReader")

        # Use ScidReader to load front month data across all files
        try:
            # Convert file paths to strings
            file_paths = [str(f) for f in scid_files]

            # Use ScidReader's load_front_month method
            reader = ScidReader(file_paths)

            # Load front month data
            df = reader.load_front_month(
                start_date=start_date,
                end_date=end_date
            )

            if len(df) == 0:
                if self.logger:
                    self.logger.warning(f"No front month data loaded for {symbol}")
                return pd.DataFrame()

            # Apply volume bucketing if requested
            if volume_bucket_size:
                if self.logger:
                    self.logger.info(f"Bucketing data by volume: {volume_bucket_size} contracts per bucket")
                df = reader.bucket_volume(df, bucket_size=volume_bucket_size)

            # Apply time-based resampling if requested (and not using volume buckets)
            elif resample_rule:
                if self.logger:
                    self.logger.info(f"Resampling data to {resample_rule}")
                df = self._resample_ohlcv(df, rule=resample_rule)

            if self.logger:
                total_records = len(df)
                date_range = f"{df.index[0]} to {df.index[-1]}"
                self.logger.info(f"Loaded {total_records:,} records from {date_range}")

            return df

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading front month data for {symbol}: {e}")
            return pd.DataFrame()

    def load_m0_intraday_data(self, symbol: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             resample_rule: Optional[str] = None,
                             volume_bucket_size: Optional[int] = None) -> List[IntradayData]:
        """
        Load intraday data for all M0 periods of a symbol using ScidReader.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date filter
            end_date: End date filter
            resample_rule: Optional time-based resampling rule (e.g., '1T', '5T', '1H')
            volume_bucket_size: Optional volume bucket size for aggregation

        Returns:
            List of IntradayData objects for each M0 period
        """
        # Convert to timezone-naive datetime if needed
        if start_date and hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if end_date and hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)

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
                if self.logger:
                    processing_info = f"Loading SCID data: {period.file_path.name} from {period_start} to {period_end}"
                    if volume_bucket_size:
                        processing_info += f" (volume buckets: {volume_bucket_size})"
                    elif resample_rule:
                        processing_info += f" (resample: {resample_rule})"
                    self.logger.info(processing_info)

                # Read SCID file using ScidReader
                reader = ScidReader(str(period.file_path))

                # Load data with date filtering
                df = reader.read(
                    start_date=period_start,
                    end_date=period_end
                )

                if len(df) == 0:
                    if self.logger:
                        self.logger.warning(f"No data in date range for {period.file_path.name}")
                    continue

                # Apply volume bucket aggregation (takes priority over time resampling)
                if volume_bucket_size:
                    df = reader.bucket_volume(df, bucket_size=volume_bucket_size)
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

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str = "1T") -> pd.DataFrame:
        """
        Resample a tick/second stream into OHLCV using pandas.

        Args:
            df: Input DataFrame with tick data
            rule: Resampling rule (e.g., "1T" for 1 minute)

        Returns:
            Resampled OHLCV DataFrame (timezone-naive)
        """
        # Ensure timezone-naive index for consistent processing
        if df.index.tz is not None:
            df = df.tz_localize(None)

        resampled = df.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
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
        intraday_data_list = self.load_m0_intraday_data(
            symbol, start_date, end_date, resample_rule, volume_bucket_size
        )

        if not intraday_data_list:
            return pd.DataFrame()

        # Process each M0 period
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
                avg_volume_per_bucket = continuous_df.get('Volume', continuous_df.get('TotalVolume', pd.Series())).mean()
                self.logger.info(f"Created continuous M0 dataset for {symbol}: "
                               f"{total_records:,} volume buckets ({volume_bucket_size} target volume) "
                               f"from {date_range}, avg volume/bucket: {avg_volume_per_bucket:.0f}")
            else:
                self.logger.info(f"Created continuous M0 dataset for {symbol}: "
                               f"{total_records:,} records from {date_range}")

        return continuous_df

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
                reader = ScidReader(str(scid_file))
                df = reader.read()

                file_result['records'] = len(df)

                if len(df) > 0:
                    file_result['date_range'] = {
                        'start': df.index[0],
                        'end': df.index[-1]
                    }

                file_result['valid'] = True
                validation_results['valid_files'] += 1

            except Exception as e:
                file_result['error'] = str(e)
                validation_results['invalid_files'] += 1

            validation_results['file_details'].append(file_result)

        return validation_results

    def write_intraday_to_hdf5(self,
                              symbol: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              timeframe: Optional[str] = None,
                              volume_bucket_size: Optional[int] = None,
                              replace: bool = False) -> Dict[str, any]:
        """
        Write intraday data to HDF5 in DataClient format: market/{ticker}/{timeframe or bucket_size}

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start_date: Start date filter
            end_date: End date filter
            timeframe: Optional time-based resampling rule (e.g., '1T', '5T', '1H')
                      If provided, will write to market/{symbol}/{timeframe}
            volume_bucket_size: Optional volume bucket size for aggregation
                               If provided, will write to market/{symbol}/vol_{bucket_size}
            replace: If True, replace existing data; if False, append

        Returns:
            Dictionary with write statistics

        Examples:
            # Write 1-minute bars
            manager.write_intraday_to_hdf5('CL_F', timeframe='1T')
            # -> market/CL_F/1T

            # Write 500-contract volume buckets
            manager.write_intraday_to_hdf5('CL_F', volume_bucket_size=500)
            # -> market/CL_F/vol_500

            # Write raw tick data (no timeframe/bucket)
            manager.write_intraday_to_hdf5('CL_F')
            # -> market/CL_F
        """
        if self.logger:
            self.logger.info(f"Writing intraday data for {symbol} to HDF5")

        # Load continuous M0 data
        df = self.get_continuous_m0_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            resample_rule=timeframe,
            volume_bucket_size=volume_bucket_size,
            remove_gaps=True,
            add_roll_markers=True
        )

        metadata_store = get_update_metadata_store()
        key = f"market/{symbol}"
        if volume_bucket_size:
            key += f"/vol_{volume_bucket_size}"
        elif timeframe:
            key += f"/{timeframe}"

        event_details = {
            "symbol": symbol,
            "key": key,
            "timeframe": timeframe,
            "volume_bucket_size": volume_bucket_size,
        }

        if df.empty:
            if self.logger:
                self.logger.warning(f"No data loaded for {symbol}")
            metadata_store.record_success(
                INTRADAY_UPDATE_EVENT,
                {**event_details, "mode": "noop", "rows_written": 0},
            )
            return {
                'symbol': symbol,
                'key': key,
                'records_written': 0,
                'success': False,
                'error': 'No data loaded'
            }

        metadata_store.record_attempt(INTRADAY_UPDATE_EVENT, details=event_details)

        try:
            if replace:
                prepared = df.sort_index()
                prepared = prepared[~prepared.index.duplicated(keep="last")]
                self.data_client.write_market(prepared, key, replace=True)
                stats = {
                    'symbol': symbol,
                    'key': key,
                    'records_written': len(prepared),
                    'mode': 'replace',
                    'date_range': {
                        'start': prepared.index[0],
                        'end': prepared.index[-1]
                    },
                    'timeframe': timeframe,
                    'volume_bucket_size': volume_bucket_size,
                    'success': True
                }
                metadata_store.record_success(INTRADAY_UPDATE_EVENT, stats)
                return stats

            prepared, require_replace = prepare_dataframe_for_append(
                self.data_client,
                key,
                df,
                allow_schema_expansion=False,
            )

            if prepared.empty and self.data_client.market_key_exists(key):
                stats = {
                    'symbol': symbol,
                    'key': key,
                    'records_written': 0,
                    'mode': 'noop',
                    'timeframe': timeframe,
                    'volume_bucket_size': volume_bucket_size,
                    'success': True
                }
                metadata_store.record_success(INTRADAY_UPDATE_EVENT, stats)
                return stats

            if require_replace:
                previous_rows = self.data_client.get_market_rowcount(key)
                self.data_client.write_market(prepared, key, replace=True)
                total_rows = len(prepared)
                stats = {
                    'symbol': symbol,
                    'key': key,
                    'records_written': total_rows,
                    'mode': 'schema_replace',
                    'delta_rows': total_rows - previous_rows,
                    'date_range': {
                        'start': prepared.index[0],
                        'end': prepared.index[-1]
                    },
                    'timeframe': timeframe,
                    'volume_bucket_size': volume_bucket_size,
                    'success': True
                }
                metadata_store.record_success(INTRADAY_UPDATE_EVENT, stats)
                if self.logger:
                    self.logger.info(
                        f"Rewrote {key} with schema expansion; rows={total_rows:,}"
                    )
                return stats

            append_result = self.data_client.append_market_continuous(prepared, key)

            stats = {
                'symbol': symbol,
                'key': key,
                'records_written': append_result.get('rows_written', 0),
                'mode': append_result.get('mode'),
                'date_range': {
                    'start': prepared.index[0],
                    'end': prepared.index[-1]
                } if not prepared.empty else None,
                'timeframe': timeframe,
                'volume_bucket_size': volume_bucket_size,
                'success': append_result.get('mode') != 'schema_mismatch'
            }

            if append_result.get('mode') == 'schema_mismatch':
                raise ValueError(f"Schema mismatch when appending to {key}: {append_result}")

            metadata_store.record_success(
                INTRADAY_UPDATE_EVENT,
                {**event_details, **summarize_append_result(append_result)},
            )

            if self.logger:
                rows = append_result.get('rows_written', 0)
                if rows:
                    self.logger.info(
                        f"Appended {rows:,} records to {key} ({prepared.index[0]} to {prepared.index[-1]})"
                    )
                else:
                    self.logger.info(f"No new intraday rows appended to {key}")

            return stats

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error writing data to HDF5: {e}")

            metadata_store.record_failure(INTRADAY_UPDATE_EVENT, str(e), details=event_details)

            return {
                'symbol': symbol,
                'key': key,
                'records_written': 0,
                'success': False,
                'error': str(e)
            }

    def batch_write_intraday_to_hdf5(self,
                                    symbols: List[str],
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None,
                                    timeframe: Optional[str] = None,
                                    volume_bucket_size: Optional[int] = None,
                                    replace: bool = False) -> Dict[str, Dict[str, any]]:
        """
        Write intraday data for multiple symbols to HDF5.

        Args:
            symbols: List of trading symbols
            start_date: Start date filter
            end_date: End date filter
            timeframe: Optional time-based resampling rule
            volume_bucket_size: Optional volume bucket size
            replace: If True, replace existing data; if False, append

        Returns:
            Dictionary mapping symbols to their write statistics
        """
        results = {}

        for symbol in symbols:
            if self.logger:
                self.logger.info(f"Processing {symbol}...")

            result = self.write_intraday_to_hdf5(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                volume_bucket_size=volume_bucket_size,
                replace=replace
            )

            results[symbol] = result

        # Summary statistics
        total_records = sum(r.get('records_written', 0) for r in results.values())
        successful = sum(1 for r in results.values() if r.get('success', False))

        if self.logger:
            self.logger.info(
                f"Batch write complete: {successful}/{len(symbols)} symbols successful, "
                f"{total_records:,} total records written"
            )

        return results

    def write_multiple_timeframes(self,
                                 symbol: str,
                                 timeframes: List[str],
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 replace: bool = False) -> Dict[str, Dict[str, any]]:
        """
        Write the same symbol data at multiple timeframes.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframe strings (e.g., ['1T', '5T', '15T', '1H'])
            start_date: Start date filter
            end_date: End date filter
            replace: If True, replace existing data; if False, append

        Returns:
            Dictionary mapping timeframes to their write statistics

        Example:
            results = manager.write_multiple_timeframes(
                'CL_F',
                timeframes=['1T', '5T', '15T', '1H', '1D']
            )
            # Writes to:
            # market/CL_F/1T
            # market/CL_F/5T
            # market/CL_F/15T
            # market/CL_F/1H
            # market/CL_F/1D
        """
        results = {}

        for timeframe in timeframes:
            if self.logger:
                self.logger.info(f"Writing {symbol} at {timeframe} timeframe...")

            result = self.write_intraday_to_hdf5(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                replace=replace
            )

            results[timeframe] = result

        return results
