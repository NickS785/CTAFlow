#!/usr/bin/env python3
"""
Optimized intraday file management for SCID contract data using sierrapy.

This module provides a streamlined IntradayFileManager for efficient loading
of continuous front month data with built-in gap detection.
"""

from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re
import logging
import asyncio
from dataclasses import dataclass

from sierrapy.parser import ScidReader, AsyncScidReader, bucket_by_volume, resample_ohlcv

from ...config import DLY_DATA_PATH, MARKET_DATA_PATH, INTRADAY_ADB_PATH
from ..data_client import DataClient
from ..update_management import (
    INTRADAY_UPDATE_EVENT,
    get_update_metadata_store,
    prepare_dataframe_for_append,
    summarize_append_result,
)

# ArcticDB integration
try:
    from arcticdb import Arctic
    HAS_ARCTICDB = True
except ImportError:
    Arctic = None
    HAS_ARCTICDB = False


# Arctic instance cache (singleton pattern to prevent LMDB reopen)
_ARCTIC_INSTANCES = {}


def get_arctic_instance(uri: str) -> Optional[Arctic]:
    """
    Get or create Arctic instance with singleton pattern.

    This prevents LMDB "already opened" errors by reusing instances
    within the same process.

    Args:
        uri: Arctic URI (e.g., 'lmdb://path/to/db')

    Returns:
        Arctic instance or None if not available
    """
    if not HAS_ARCTICDB:
        return None

    # Check cache first
    if uri in _ARCTIC_INSTANCES:
        return _ARCTIC_INSTANCES[uri]

    # Create new instance and cache it
    try:
        instance = Arctic(uri)
        _ARCTIC_INSTANCES[uri] = instance
        return instance
    except Exception as e:
        logging.warning(f"Could not create Arctic instance for {uri}: {e}")
        return None


def clear_arctic_cache(uri: Optional[str] = None) -> None:
    """
    Clear cached Arctic instances.

    Args:
        uri: Specific URI to clear, or None to clear all instances

    Examples:
        # Clear all cached instances
        clear_arctic_cache()

        # Clear specific instance
        clear_arctic_cache('lmdb://F:/Data/intraday/')
    """
    if uri is not None:
        if uri in _ARCTIC_INSTANCES:
            del _ARCTIC_INSTANCES[uri]
    else:
        _ARCTIC_INSTANCES.clear()


def get_cached_arctic_uris() -> List[str]:
    """
    Get list of currently cached Arctic URIs.

    Returns:
        List of URIs with active cached instances
    """
    return list(_ARCTIC_INSTANCES.keys())


@dataclass
class GapInfo:
    """Information about a gap in the continuous data."""
    start_timestamp: datetime
    end_timestamp: datetime
    duration_hours: float
    gap_type: str  # 'weekend', 'holiday', 'data_missing', 'contract_roll'
    contracts_affected: List[str]


class IntradayFileManager:
    """
    Optimized manager for intraday SCID contract files using sierrapy.

    Uses ScidReader.load_front_month_series for efficient front month loading
    and ScidReader.load_scid_files for batch processing with gap detection.
    """

    # SCID filename pattern
    SCID_PATTERN = re.compile(r'^([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{2})-([A-Z]+)\.scid$', re.IGNORECASE)

    # Month code mapping
    MONTH_MAP = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }

    def __init__(self,
                 data_path: Optional[Path] = None,
                 market_data_path: Optional[Path] = None,
                 arctic_uri: Optional[str] = None,
                 enable_logging: bool = True):
        """
        Initialize the intraday file manager.

        Args:
            data_path: Path to directory containing SCID files (default: DLY_DATA_PATH)
            market_data_path: Path to HDF5 market data file (default: MARKET_DATA_PATH)
            arctic_uri: ArcticDB URI for intraday data (default: INTRADAY_ADB_PATH)
            enable_logging: Enable logging
        """
        self.data_path = Path(data_path) if data_path else Path(DLY_DATA_PATH)
        self.market_data_path = Path(market_data_path) if market_data_path else MARKET_DATA_PATH
        self.arctic_uri = arctic_uri or INTRADAY_ADB_PATH

        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

        # Cache for discovered files
        self._scid_files: Dict[str, List[Path]] = {}

        # Initialize DataClient for HDF5
        self.data_client = DataClient(market_path=self.market_data_path)

        # Initialize ArcticDB connection using singleton pattern
        self.arctic = get_arctic_instance(self.arctic_uri)
        if self.arctic and self.logger:
            # Check if this is a cached instance
            is_cached = self.arctic_uri in _ARCTIC_INSTANCES and _ARCTIC_INSTANCES[self.arctic_uri] is self.arctic
            if is_cached:
                self.logger.info(f"Reusing existing ArcticDB connection at {self.arctic_uri}")
            else:
                self.logger.info(f"Created new ArcticDB connection at {self.arctic_uri}")

        # Discover SCID files
        self._discover_scid_files()

    def _discover_scid_files(self) -> None:
        """Discover and cache SCID files."""
        if not self.data_path.exists():
            if self.logger:
                self.logger.warning(f"Data path does not exist: {self.data_path}")
            return

        mapping: Dict[str, List[Path]] = {}

        for entry in self.data_path.iterdir():
            if not entry.is_file() or entry.suffix.lower() != '.scid':
                continue

            match = self.SCID_PATTERN.match(entry.name)
            if not match:
                continue

            base_symbol = match.group(1).upper()
            full_symbol = f"{base_symbol}_F"
            mapping.setdefault(full_symbol, []).append(entry)

        # Sort files by filename
        for files in mapping.values():
            files.sort()

        self._scid_files = mapping

        if self.logger:
            total_files = sum(len(files) for files in mapping.values())
            self.logger.info(f"Discovered {total_files} SCID files for {len(mapping)} symbols")

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available SCID files."""
        return list(self._scid_files.keys())

    def get_scid_files_for_symbol(self, symbol: str) -> List[Path]:
        """Get all SCID files for a given symbol."""
        return self._scid_files.get(symbol, [])

    def load_front_month_series(self,
                                symbol: str,
                                start: Optional[datetime] = None,
                                end: Optional[datetime] = None,
                                resample_rule: Optional[str] = None,
                                volume_bucket_size: Optional[int] = None,
                                detect_gaps: bool = True) -> Tuple[pd.DataFrame, Optional[List[GapInfo]]]:
        """
        Load continuous front month data using ScidReader.load_front_month_series.

        This is the PRIMARY method for loading front month data efficiently.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            start: Start date filter
            end: End date filter
            resample_rule: Time-based resampling (e.g., '1T', '5T', '1H')
            volume_bucket_size: Volume-based bucketing
            detect_gaps: If True, analyze and return gap information

        Returns:
            Tuple of (DataFrame, List[GapInfo]) where GapInfo is None if detect_gaps=False
        """
        scid_files = self.get_scid_files_for_symbol(symbol)

        if not scid_files:
            if self.logger:
                self.logger.warning(f"No SCID files found for {symbol}")
            return pd.DataFrame(), None

        if self.logger:
            self.logger.info(f"Loading front month series for {symbol} from {len(scid_files)} files")

        try:
            # Initialize ScidReader with directory
            reader = ScidReader(self.data_path)

            # Convert file paths to strings for load_scid_files
            file_paths = [str(f) for f in scid_files]

            # Load front month series - ScidReader handles stitching automatically
            df = reader.load_front_month_series(
                ticker=symbol.replace('_F', ''),
                start=start,
                end=end,
                resample_rule=resample_rule
            )

            if df.empty:
                if self.logger:
                    self.logger.warning(f"No data loaded for {symbol}")
                return pd.DataFrame(), None

            # Apply volume bucketing if requested (after front month loading)
            if volume_bucket_size:
                if self.logger:
                    self.logger.info(f"Bucketing by volume: {volume_bucket_size} contracts per bucket")
                df = reader.bucket_volume(df, bucket_size=volume_bucket_size)

            # Detect gaps if requested
            gaps = None
            if detect_gaps:
                gaps = self._detect_gaps(df, symbol)
                if self.logger and gaps:
                    self.logger.info(f"Detected {len(gaps)} gaps in {symbol} data")

            if self.logger:
                date_range = f"{df.index[0]} to {df.index[-1]}"
                self.logger.info(f"Loaded {len(df):,} records from {date_range}")

            return df, gaps

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading front month series for {symbol}: {e}")
            return pd.DataFrame(), None

    def load_scid_files_batch(self,
                              symbol: str,
                              start: Optional[datetime] = None,
                              end: Optional[datetime] = None,
                              resample_rule: Optional[str] = None) -> pd.DataFrame:
        """
        Load multiple SCID files using ScidReader.load_scid_files (batch mode).

        Use this when you need all contract data, not just front month.

        Args:
            symbol: Trading symbol
            start: Start date filter
            end: End date filter
            resample_rule: Time-based resampling

        Returns:
            DataFrame with multi-contract data
        """
        scid_files = self.get_scid_files_for_symbol(symbol)

        if not scid_files:
            return pd.DataFrame()

        if self.logger:
            self.logger.info(f"Loading {len(scid_files)} SCID files for {symbol} in batch mode")

        try:
            file_paths = [str(f) for f in scid_files]
            reader = ScidReader(self.data_path)

            # Load all files - returns Dict[str, pd.DataFrame]
            file_dict = reader.load_scid_files(
                file_paths=file_paths,
                start_ms=int(start.timestamp() * 1000) if start else None,
                end_ms=int(end.timestamp() * 1000) if end else None,
                resample_rule=resample_rule
            )

            # Concatenate all DataFrames
            if file_dict:
                df = pd.concat(file_dict.values(), ignore_index=False)
                if self.logger:
                    self.logger.info(f"Loaded {len(df):,} records from {len(file_dict)} files")
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in batch load for {symbol}: {e}")
            return pd.DataFrame()

    def _detect_gaps(self, df: pd.DataFrame, symbol: str) -> List[GapInfo]:
        """
        Detect gaps in continuous data with classification.

        Args:
            df: DataFrame with datetime index
            symbol: Trading symbol

        Returns:
            List of GapInfo objects describing detected gaps
        """
        if df.empty or len(df) < 2:
            return []

        gaps = []

        # Calculate time differences between consecutive records
        time_diffs = df.index.to_series().diff()

        # Determine expected frequency (mode of time differences)
        freq_counts = time_diffs.value_counts()
        if len(freq_counts) == 0:
            return []

        expected_freq = freq_counts.index[0]
        threshold = expected_freq * 3  # 3x expected frequency

        # Find gaps exceeding threshold
        gap_mask = time_diffs > threshold

        for idx in df.index[gap_mask]:
            gap_start_idx = df.index.get_loc(idx) - 1
            if gap_start_idx < 0:
                continue

            gap_start = df.index[gap_start_idx]
            gap_end = idx
            duration_hours = (gap_end - gap_start).total_seconds() / 3600

            # Classify gap type
            gap_type = self._classify_gap(gap_start, gap_end, duration_hours)

            # Extract affected contracts if available
            contracts_affected = []
            if 'contract_code' in df.columns:
                contracts_at_gap = df.loc[[gap_start, gap_end], 'contract_code'].unique().tolist()
                contracts_affected = contracts_at_gap

            gap_info = GapInfo(
                start_timestamp=gap_start,
                end_timestamp=gap_end,
                duration_hours=duration_hours,
                gap_type=gap_type,
                contracts_affected=contracts_affected
            )
            gaps.append(gap_info)

        return gaps

    def _classify_gap(self, start: datetime, end: datetime, duration_hours: float) -> str:
        """Classify gap type based on timing and duration."""
        # Weekend gap (Friday close to Monday open)
        if start.weekday() == 4 and end.weekday() == 0 and duration_hours < 72:
            return 'weekend'

        # Holiday gap (24-72 hours on weekday)
        if duration_hours >= 24 and duration_hours <= 72:
            return 'holiday'

        # Contract roll (look for change between Friday and Monday)
        if start.weekday() >= 3 and end.weekday() <= 1 and duration_hours > 48:
            return 'contract_roll'

        # Data missing
        return 'data_missing'

    def get_gap_summary(self, gaps: List[GapInfo]) -> Dict[str, any]:
        """Generate summary statistics for detected gaps."""
        if not gaps:
            return {'total_gaps': 0}

        gap_types = {}
        for gap in gaps:
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1

        total_gap_hours = sum(gap.duration_hours for gap in gaps)

        return {
            'total_gaps': len(gaps),
            'by_type': gap_types,
            'total_gap_hours': total_gap_hours,
            'avg_gap_hours': total_gap_hours / len(gaps),
            'longest_gap': max(gaps, key=lambda g: g.duration_hours),
            'data_missing_count': gap_types.get('data_missing', 0)
        }

    def write_to_hdf5(self,
                     symbol: str,
                     start: Optional[datetime] = None,
                     end: Optional[datetime] = None,
                     timeframe: Optional[str] = None,
                     volume_bucket_size: Optional[int] = None,
                     replace: bool = False) -> Dict[str, any]:
        """
        Write front month data to HDF5 in DataClient format.

        Args:
            symbol: Trading symbol
            start: Start date filter
            end: End date filter
            timeframe: Time-based resampling rule (e.g., '1T', '5T', '1H')
            volume_bucket_size: Volume bucket size
            replace: If True, replace existing data; if False, append

        Returns:
            Dictionary with write statistics

        Examples:
            # Write 1-minute bars
            manager.write_to_hdf5('CL_F', timeframe='1T')

            # Write 500-contract volume buckets
            manager.write_to_hdf5('CL_F', volume_bucket_size=500)
        """
        if self.logger:
            self.logger.info(f"Writing intraday data for {symbol} to HDF5")

        # Load data using optimized front month series
        df, gaps = self.load_front_month_series(
            symbol=symbol,
            start=start,
            end=end,
            resample_rule=timeframe,
            volume_bucket_size=volume_bucket_size,
            detect_gaps=True
        )

        # Log gaps if detected
        if gaps:
            gap_summary = self.get_gap_summary(gaps)
            if self.logger:
                self.logger.info(
                    f"Gap summary for {symbol}: {gap_summary['total_gaps']} gaps, "
                    f"{gap_summary['data_missing_count']} data missing"
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
                    'gaps_detected': len(gaps) if gaps else 0,
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
                    'gaps_detected': len(gaps) if gaps else 0,
                    'success': True
                }
                metadata_store.record_success(INTRADAY_UPDATE_EVENT, stats)
                if self.logger:
                    self.logger.info(f"Rewrote {key} with schema expansion; rows={total_rows:,}")
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
                'gaps_detected': len(gaps) if gaps else 0,
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
                    self.logger.info(f"No new rows appended to {key}")

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

    def batch_write_to_hdf5(self,
                           symbols: List[str],
                           start: Optional[datetime] = None,
                           end: Optional[datetime] = None,
                           timeframe: Optional[str] = None,
                           volume_bucket_size: Optional[int] = None,
                           replace: bool = False) -> Dict[str, Dict[str, any]]:
        """
        Write intraday data for multiple symbols to HDF5.

        Args:
            symbols: List of trading symbols
            start: Start date filter
            end: End date filter
            timeframe: Time-based resampling rule
            volume_bucket_size: Volume bucket size
            replace: If True, replace existing data

        Returns:
            Dictionary mapping symbols to write statistics
        """
        results = {}

        for symbol in symbols:
            if self.logger:
                self.logger.info(f"Processing {symbol}...")

            result = self.write_to_hdf5(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe,
                volume_bucket_size=volume_bucket_size,
                replace=replace
            )

            results[symbol] = result

        # Summary
        total_records = sum(r.get('records_written', 0) for r in results.values())
        successful = sum(1 for r in results.values() if r.get('success', False))

        if self.logger:
            self.logger.info(
                f"Batch write complete: {successful}/{len(symbols)} successful, "
                f"{total_records:,} total records"
            )

        return results

    def write_to_arctic(self,
                       symbol: str,
                       library: str = "intraday_market",
                       start: Optional[datetime] = None,
                       end: Optional[datetime] = None,
                       timeframe: Optional[str] = None,
                       volume_bucket_size: Optional[int] = None) -> Dict[str, any]:
        """
        Write front month data to ArcticDB.

        Args:
            symbol: Trading symbol (e.g., 'CL_F')
            library: ArcticDB library name (default: 'intraday_market')
            start: Start date filter
            end: End date filter
            timeframe: Time-based resampling rule (e.g., '1T', '5T', '1H')
            volume_bucket_size: Volume bucket size

        Returns:
            Dictionary with write statistics

        Examples:
            # Write 1-minute bars
            manager.write_to_arctic('CL_F', timeframe='1T')

            # Write 500-contract volume buckets
            manager.write_to_arctic('CL_F', volume_bucket_size=500)

            # Write to specific library
            manager.write_to_arctic('CL_F', library='intraday_ticks')
        """
        if not self.arctic:
            return {
                'success': False,
                'error': 'ArcticDB not available',
                'symbol': symbol
            }

        if self.logger:
            self.logger.info(f"Writing intraday data for {symbol} to ArcticDB library '{library}'")

        # Load data using optimized front month series
        df, gaps = self.load_front_month_series(
            symbol=symbol,
            start=start,
            end=end,
            resample_rule=timeframe,
            volume_bucket_size=volume_bucket_size,
            detect_gaps=True
        )

        if df.empty:
            if self.logger:
                self.logger.warning(f"No data loaded for {symbol}")
            return {
                'success': False,
                'error': 'No data loaded',
                'symbol': symbol,
                'records_written': 0
            }

        # Construct symbol name
        arctic_symbol = f"{symbol}"
        if timeframe:
            arctic_symbol += f"_{timeframe}"
        elif volume_bucket_size:
            arctic_symbol += f"_vol{volume_bucket_size}"

        # Prepare metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'volume_bucket_size': volume_bucket_size,
            'start_date': str(df.index[0]),
            'end_date': str(df.index[-1]),
            'total_records': len(df),
            'gaps_detected': len(gaps) if gaps else 0,
            'data_columns': list(df.columns),
            'source': 'IntradayFileManager'
        }

        # Add gap summary if gaps detected
        if gaps:
            gap_summary = self.get_gap_summary(gaps)
            metadata['gap_summary'] = gap_summary
            if self.logger:
                self.logger.info(
                    f"Gap summary for {symbol}: {gap_summary['total_gaps']} gaps, "
                    f"{gap_summary['data_missing_count']} data missing"
                )

        try:
            lib = self.arctic[library]
            lib.write(arctic_symbol, df, metadata=metadata)

            stats = {
                'success': True,
                'symbol': symbol,
                'arctic_symbol': arctic_symbol,
                'library': library,
                'records_written': len(df),
                'date_range': {
                    'start': df.index[0],
                    'end': df.index[-1]
                },
                'timeframe': timeframe,
                'volume_bucket_size': volume_bucket_size,
                'gaps_detected': len(gaps) if gaps else 0
            }

            if self.logger:
                self.logger.info(
                    f"Wrote {len(df):,} records to ArcticDB: {library}/{arctic_symbol} "
                    f"({df.index[0]} to {df.index[-1]})"
                )

            return stats

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error writing data to ArcticDB: {e}")

            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'arctic_symbol': arctic_symbol,
                'library': library,
                'records_written': 0
            }

    async def _load_and_write_arctic_async(self,
                                           symbol: str,
                                           library: str,
                                           start: Optional[datetime],
                                           end: Optional[datetime],
                                           timeframe: Optional[str],
                                           volume_bucket_size: Optional[int]) -> Dict[str, any]:
        """
        Async helper to load front month data and write to ArcticDB.

        Uses synchronous load_front_month_series() wrapped in executor to avoid blocking.

        Args:
            symbol: Trading symbol
            library: ArcticDB library name
            start: Start date filter
            end: End date filter
            timeframe: Time-based resampling rule
            volume_bucket_size: Volume bucket size

        Returns:
            Dictionary with write statistics
        """
        if self.logger:
            self.logger.info(f"[Async] Loading {symbol}...")

        scid_files = self.get_scid_files_for_symbol(symbol)

        if not scid_files:
            if self.logger:
                self.logger.warning(f"No SCID files found for {symbol}")
            return {
                'success': False,
                'error': 'No SCID files found',
                'symbol': symbol,
                'records_written': 0
            }

        try:
            # Convert paths to strings
            file_paths = [str(f) for f in scid_files]

            # Convert datetime to milliseconds for AsyncScidReader
            start_ms = int(start.timestamp() * 1000) if start else None
            end_ms = int(end.timestamp() * 1000) if end else None

            # Create AsyncScidReader instance with data directory
            async_reader = AsyncScidReader(self.data_path)

            # Load SCID files - returns Dict[str, pd.DataFrame]
            result_dict = await async_reader.load_scid_files(
                file_paths,
                start_ms=start_ms,
                end_ms=end_ms,
                resample_rule=timeframe,
                volume_per_bar=volume_bucket_size
            )

            # Combine all dataframes from the result dict
            if not result_dict:
                df = pd.DataFrame()
            else:
                # Concatenate all dataframes
                df = pd.concat(result_dict.values(), ignore_index=False)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='last')]

            if df.empty:
                if self.logger:
                    self.logger.warning(f"No data loaded for {symbol}")
                return {
                    'success': False,
                    'error': 'No data loaded',
                    'symbol': symbol,
                    'records_written': 0
                }

            # Resampling and volume bucketing already handled by AsyncScidReader.load_scid_files
            # via resample_rule and volume_per_bar parameters

            # Detect gaps
            gaps = self._detect_gaps(df, symbol)
            if self.logger and gaps:
                self.logger.info(f"Detected {len(gaps)} gaps in {symbol} data")

            # Construct symbol name
            arctic_symbol = f"{symbol}"
            if timeframe:
                arctic_symbol += f"_{timeframe}"
            elif volume_bucket_size:
                arctic_symbol += f"_vol{volume_bucket_size}"

            # Prepare metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'volume_bucket_size': volume_bucket_size,
                'start_date': str(df.index[0]),
                'end_date': str(df.index[-1]),
                'total_records': len(df),
                'gaps_detected': len(gaps) if gaps else 0,
                'data_columns': list(df.columns),
                'source': 'IntradayFileManager_async'
            }

            # Add gap summary if gaps detected
            if gaps:
                gap_summary = self.get_gap_summary(gaps)
                metadata['gap_summary'] = gap_summary

            # Write to ArcticDB (synchronous but quick)
            lib = self.arctic[library]
            lib.write(arctic_symbol, df, metadata=metadata)

            stats = {
                'success': True,
                'symbol': symbol,
                'arctic_symbol': arctic_symbol,
                'library': library,
                'records_written': len(df),
                'date_range': {
                    'start': df.index[0],
                    'end': df.index[-1]
                },
                'timeframe': timeframe,
                'volume_bucket_size': volume_bucket_size,
                'gaps_detected': len(gaps) if gaps else 0
            }

            if self.logger:
                self.logger.info(
                    f"[Async] Wrote {len(df):,} records to ArcticDB: {library}/{arctic_symbol}"
                )

            return stats

        except Exception as e:
            if self.logger:
                self.logger.error(f"[Async] Error processing {symbol}: {e}")

            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'records_written': 0
            }

    def batch_write_to_arctic(self,
                             symbols: Optional[List[str]] = None,
                             library: str = "intraday_market",
                             start: Optional[datetime] = None,
                             end: Optional[datetime] = None,
                             timeframe: Optional[str] = None,
                             volume_bucket_size: Optional[int] = None,
                             max_concurrent: int = 5) -> Dict[str, Dict[str, any]]:
        """
        Write intraday data for multiple symbols (or all available) to ArcticDB using async I/O.

        Uses AsyncScidReader.load_scid_files() for concurrent file reading with
        pre-collected file paths (no redundant path discovery).

        Args:
            symbols: List of trading symbols. If None, processes ALL available symbols from .scid files.
            library: ArcticDB library name
            start: Start date filter
            end: End date filter
            timeframe: Time-based resampling rule
            volume_bucket_size: Volume bucket size
            max_concurrent: Maximum number of concurrent tasks (default: 5)

        Returns:
            Dictionary mapping symbols to write statistics

        Examples:
            # Process ALL symbols from .scid files
            results = manager.batch_write_to_arctic(
                timeframe='5T',
                max_concurrent=5
            )

            # Process specific symbols concurrently
            results = manager.batch_write_to_arctic(
                symbols=['CL_F', 'NG_F', 'ZC_F', 'ZS_F', 'GC_F'],
                timeframe='1T',
                max_concurrent=5
            )

            # Process all symbols with volume bucketing
            results = manager.batch_write_to_arctic(
                volume_bucket_size=500,
                max_concurrent=10
            )
        """
        # Get symbols to process
        if symbols is None:
            symbols = self.get_available_symbols()
            if self.logger:
                self.logger.info(f"Processing ALL available symbols: {len(symbols)} symbols found")
        else:
            # Validate provided symbols
            available = set(self.get_available_symbols())
            invalid_symbols = [s for s in symbols if s not in available]
            if invalid_symbols and self.logger:
                self.logger.warning(f"Symbols not found in SCID files: {invalid_symbols}")
            symbols = [s for s in symbols if s in available]

        if not symbols:
            if self.logger:
                self.logger.warning("No valid symbols to process")
            return {}

        if not self.arctic:
            return {
                symbol: {
                    'success': False,
                    'error': 'ArcticDB not available',
                    'symbol': symbol
                }
                for symbol in symbols
            }

        if self.logger:
            self.logger.info(
                f"Starting async batch write for {len(symbols)} symbols "
                f"(max_concurrent={max_concurrent})"
            )

        async def run_batch():
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def bounded_task(symbol):
                async with semaphore:
                    return await self._load_and_write_arctic_async(
                        symbol=symbol,
                        library=library,
                        start=start,
                        end=end,
                        timeframe=timeframe,
                        volume_bucket_size=volume_bucket_size
                    )

            # Create tasks for all symbols
            tasks = [bounded_task(symbol) for symbol in symbols]

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert results to dictionary
            result_dict = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    result_dict[symbol] = {
                        'success': False,
                        'error': str(result),
                        'symbol': symbol,
                        'records_written': 0
                    }
                else:
                    result_dict[symbol] = result

            return result_dict

        # Run the async batch process
        results = asyncio.run(run_batch())

        # Summary
        total_records = sum(r.get('records_written', 0) for r in results.values())
        successful = sum(1 for r in results.values() if r.get('success', False))

        if self.logger:
            self.logger.info(
                f"Batch write to ArcticDB complete: {successful}/{len(symbols)} successful, "
                f"{total_records:,} total records"
            )

        return results

    def update_existing_arctic_keys(self,
                                    library: str = "intraday_market",
                                    symbols: Optional[List[str]] = None,
                                    start: Optional[datetime] = None,
                                    max_concurrent: int = 5) -> Dict[str, Dict[str, any]]:
        """
        Update existing symbols in ArcticDB with new data since last update.

        This method reads metadata from existing ArcticDB symbols to determine
        the last data point, then loads and appends only new data since that point.

        Args:
            library: ArcticDB library name (default: 'intraday_market')
            symbols: List of symbols to update (if None, updates all symbols in library)
            start: Override start date (if None, uses last data point from metadata)
            max_concurrent: Maximum number of concurrent tasks (default: 5)

        Returns:
            Dictionary mapping symbols to update statistics

        Examples:
            # Update all symbols in library
            results = manager.update_existing_arctic_keys(library='intraday_market')

            # Update specific symbols
            results = manager.update_existing_arctic_keys(
                library='intraday_market',
                symbols=['CL_F_1T', 'NG_F_1T'],
                max_concurrent=3
            )
        """
        if not self.arctic:
            return {
                'error': 'ArcticDB not available',
                'success': False
            }

        try:
            lib = self.arctic[library]
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cannot access library '{library}': {e}")
            return {
                'error': f'Cannot access library: {e}',
                'success': False
            }

        # Get list of symbols to update
        if symbols is None:
            try:
                symbols = lib.list_symbols()
                if self.logger:
                    self.logger.info(f"Found {len(symbols)} symbols in library '{library}'")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Cannot list symbols in library '{library}': {e}")
                return {
                    'error': f'Cannot list symbols: {e}',
                    'success': False
                }

        if not symbols:
            if self.logger:
                self.logger.warning(f"No symbols to update in library '{library}'")
            return {
                'message': 'No symbols to update',
                'success': True,
                'results': {}
            }

        if self.logger:
            self.logger.info(f"Updating {len(symbols)} symbols in library '{library}'")

        # Parse symbols to extract base symbol, timeframe, and volume_bucket_size
        update_tasks = []
        for arctic_symbol in symbols:
            # Parse symbol name: CL_F_1T or CL_F_vol500
            parts = arctic_symbol.split('_')

            # Extract base symbol (e.g., CL_F)
            if len(parts) >= 2:
                base_symbol = f"{parts[0]}_{parts[1]}"
            else:
                if self.logger:
                    self.logger.warning(f"Cannot parse symbol: {arctic_symbol}")
                continue

            # Extract timeframe or volume bucket
            timeframe = None
            volume_bucket_size = None

            if len(parts) > 2:
                suffix = '_'.join(parts[2:])
                if suffix.startswith('vol'):
                    try:
                        volume_bucket_size = int(suffix[3:])
                    except ValueError:
                        if self.logger:
                            self.logger.warning(f"Cannot parse volume bucket from: {suffix}")
                else:
                    timeframe = suffix

            # Get last update time from metadata
            update_start = start
            if update_start is None:
                try:
                    item = lib.read(arctic_symbol)
                    metadata = item.metadata
                    if metadata and 'end_date' in metadata:
                        # Parse end_date and add 1 second to avoid overlap
                        last_date = pd.to_datetime(metadata['end_date'])
                        update_start = last_date + timedelta(seconds=1)
                        if self.logger:
                            self.logger.info(
                                f"{arctic_symbol}: Last data at {last_date}, "
                                f"updating from {update_start}"
                            )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Cannot read metadata for {arctic_symbol}: {e}, "
                            f"will load all available data"
                        )

            update_tasks.append({
                'arctic_symbol': arctic_symbol,
                'base_symbol': base_symbol,
                'timeframe': timeframe,
                'volume_bucket_size': volume_bucket_size,
                'start': update_start
            })

        # Run async updates
        if self.logger:
            self.logger.info(f"Starting async updates for {len(update_tasks)} symbols")

        async def run_updates():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def update_task(task):
                async with semaphore:
                    return await self._update_arctic_symbol_async(
                        library=library,
                        arctic_symbol=task['arctic_symbol'],
                        base_symbol=task['base_symbol'],
                        timeframe=task['timeframe'],
                        volume_bucket_size=task['volume_bucket_size'],
                        start=task['start']
                    )

            tasks = [update_task(task) for task in update_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            result_dict = {}
            for task, result in zip(update_tasks, results):
                arctic_symbol = task['arctic_symbol']
                if isinstance(result, Exception):
                    result_dict[arctic_symbol] = {
                        'success': False,
                        'error': str(result),
                        'symbol': arctic_symbol,
                        'records_appended': 0
                    }
                else:
                    result_dict[arctic_symbol] = result

            return result_dict

        results = asyncio.run(run_updates())

        # Summary
        total_records = sum(r.get('records_appended', 0) for r in results.values())
        successful = sum(1 for r in results.values() if r.get('success', False))

        if self.logger:
            self.logger.info(
                f"Update complete: {successful}/{len(results)} successful, "
                f"{total_records:,} total records appended"
            )

        return {
            'success': True,
            'library': library,
            'symbols_updated': successful,
            'total_symbols': len(results),
            'total_records_appended': total_records,
            'results': results
        }

    async def _update_arctic_symbol_async(self,
                                         library: str,
                                         arctic_symbol: str,
                                         base_symbol: str,
                                         timeframe: Optional[str],
                                         volume_bucket_size: Optional[int],
                                         start: Optional[datetime]) -> Dict[str, any]:
        """
        Async helper to update a single ArcticDB symbol with new data.

        Args:
            library: ArcticDB library name
            arctic_symbol: Full ArcticDB symbol name (e.g., CL_F_1T)
            base_symbol: Base symbol (e.g., CL_F)
            timeframe: Time-based resampling rule
            volume_bucket_size: Volume bucket size
            start: Start date for new data

        Returns:
            Dictionary with update statistics
        """
        if self.logger:
            self.logger.info(f"[Async] Updating {arctic_symbol}...")

        scid_files = self.get_scid_files_for_symbol(base_symbol)

        if not scid_files:
            return {
                'success': False,
                'error': 'No SCID files found',
                'symbol': arctic_symbol,
                'records_appended': 0
            }

        try:
            # Convert paths to strings
            file_paths = [str(f) for f in scid_files]

            # Load new data asynchronously
            df = await AsyncScidReader.load_scid_files(
                file_paths,
                start=start,
                end=None  # Load to present
            )

            if df.empty:
                if self.logger:
                    self.logger.info(f"No new data for {arctic_symbol}")
                return {
                    'success': True,
                    'symbol': arctic_symbol,
                    'records_appended': 0,
                    'message': 'No new data available'
                }

            # Apply resampling if requested
            if timeframe:
                df = resample_ohlcv(df, rule=timeframe)

            # Apply volume bucketing if requested
            if volume_bucket_size:
                df = bucket_by_volume(df, bucket_size=volume_bucket_size)

            if df.empty:
                return {
                    'success': True,
                    'symbol': arctic_symbol,
                    'records_appended': 0,
                    'message': 'No new data after resampling/bucketing'
                }

            # Read existing data to append
            lib = self.arctic[library]
            existing_item = lib.read(arctic_symbol)
            existing_df = existing_item.data
            existing_metadata = existing_item.metadata or {}

            # Combine existing and new data
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

            # Update metadata
            updated_metadata = existing_metadata.copy()
            updated_metadata.update({
                'last_update': str(datetime.now()),
                'end_date': str(combined_df.index[-1]),
                'total_records': len(combined_df),
                'records_appended': len(df)
            })

            # Write updated data (overwrites with new version)
            lib.write(arctic_symbol, combined_df, metadata=updated_metadata)

            stats = {
                'success': True,
                'symbol': arctic_symbol,
                'records_appended': len(df),
                'total_records': len(combined_df),
                'date_range': {
                    'start': combined_df.index[0],
                    'end': combined_df.index[-1]
                },
                'new_data_range': {
                    'start': df.index[0],
                    'end': df.index[-1]
                } if len(df) > 0 else None
            }

            if self.logger:
                self.logger.info(
                    f"[Async] Updated {arctic_symbol}: appended {len(df):,} records "
                    f"(total: {len(combined_df):,})"
                )

            return stats

        except Exception as e:
            if self.logger:
                self.logger.error(f"[Async] Error updating {arctic_symbol}: {e}")

            return {
                'success': False,
                'error': str(e),
                'symbol': arctic_symbol,
                'records_appended': 0
            }

    def write_multiple_timeframes(self,
                                 symbol: str,
                                 timeframes: List[str],
                                 start: Optional[datetime] = None,
                                 end: Optional[datetime] = None,
                                 replace: bool = False) -> Dict[str, Dict[str, any]]:
        """
        Write the same symbol at multiple timeframes.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes (e.g., ['1T', '5T', '15T', '1H'])
            start: Start date filter
            end: End date filter
            replace: If True, replace existing data

        Returns:
            Dictionary mapping timeframes to write statistics
        """
        results = {}

        for timeframe in timeframes:
            if self.logger:
                self.logger.info(f"Writing {symbol} at {timeframe}...")

            result = self.write_to_hdf5(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe,
                replace=replace
            )

            results[timeframe] = result

        return results

    def validate_scid_files(self, symbol: str) -> Dict[str, any]:
        """
        Validate SCID files by checking readability.

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
