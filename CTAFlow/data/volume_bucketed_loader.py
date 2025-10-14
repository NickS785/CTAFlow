"""
Volume-bucketed data loader for screeners with dual backend support.

This module provides a streamlined interface for loading volume-bucketed intraday data,
replacing the complex tick data caching system with a simpler, more efficient approach.

Supports two backends:
- ScidReader (lightweight, read-only): Fast loading for screening workflows
- IntradayFileManager (full-featured): Write capabilities, gap detection, storage Th

Key Benefits:
- Consistent statistical properties (each bucket has similar volume)
- Pre-aggregated data reduces computation
- Natural alignment with VPIN calculation
- Simpler caching based on (symbol, bucket_size) instead of (scid_folder, window)
- Choice of lightweight or full-featured backend
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

from sierrapy.parser import ScidReader
from .raw_formatting.intraday_manager import IntradayFileManager, GapInfo
from .data_client import DataClient
from ..config import DLY_DATA_PATH, MARKET_DATA_PATH


class VolumeBucketedDataLoader:
    """
    Loader for volume-bucketed intraday data with intelligent caching and dual backend support.

    This class provides efficient volume-bucketed data loading for screeners with two backend options:

    1. **ScidReader** (lightweight): Fast, read-only loading via sierrapy
    2. **IntradayFileManager** (full-featured): Write capabilities, gap detection, HDF5/ArcticDB

    Examples
    --------
    >>> # Lightweight read-only (fastest for screening)
    >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500, use_file_manager=False)
    >>> data = loader.load_symbols(['CL_F', 'NG_F'], start_date="2020-01-01")
    >>>
    >>> # Full-featured with write capabilities
    >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500, use_file_manager=True)
    >>> data = loader.load_symbols(['CL_F', 'NG_F'])
    >>> loader.write_to_hdf5(['CL_F', 'NG_F'])  # Store for fast loading
    >>>
    >>> # Use with HistoricalScreener
    >>> from CTAFlow.screeners import HistoricalScreener
    >>> screener = HistoricalScreener(data, volume_bucket_loader=loader)
    """

    def __init__(
        self,
        volume_bucket_size: Union[int, Dict[str, int]] = 500,
        data_path: Optional[Path] = None,
        market_data_path: Optional[Path] = None,
        use_file_manager: bool = False,
        enable_logging: bool = True
    ):
        """
        Initialize the volume-bucketed data loader.

        Parameters
        ----------
        volume_bucket_size : Union[int, Dict[str, int]]
            Target volume per bucket. Can be:
            - int: Same bucket size for all symbols (default: 500 contracts)
            - Dict[str, int]: Per-symbol bucket sizes for volume-aware bucketing
              Example: {'CL_F': 500, 'RB_F': 100, 'NG_F': 300}
        data_path : Optional[Path]
            Path to SCID files (default: DLY_DATA_PATH)
        market_data_path : Optional[Path]
            Path to HDF5 market data (default: MARKET_DATA_PATH)
        use_file_manager : bool
            If True, uses IntradayFileManager (write capable, heavier)
            If False, uses ScidReader (read-only, lightweight)
            Default: False (lightweight)
        enable_logging : bool
            Enable logging (default: True)
        """
        self.volume_bucket_size = volume_bucket_size
        self.data_path = Path(data_path) if data_path else Path(DLY_DATA_PATH)
        self.market_data_path = market_data_path or MARKET_DATA_PATH
        self.use_file_manager = use_file_manager

        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

        # Initialize backend
        if use_file_manager:
            # Full-featured backend with write capabilities
            self.file_manager = IntradayFileManager(
                data_path=self.data_path,
                market_data_path=self.market_data_path,
                enable_logging=enable_logging
            )
            self.scid_reader = None
            if self.logger:
                self.logger.info("Using IntradayFileManager backend (full-featured)")
        else:
            # Lightweight read-only backend
            self.scid_reader = ScidReader(self.data_path)
            self.file_manager = None
            if self.logger:
                self.logger.info("Using ScidReader backend (lightweight)")

        # Initialize DataClient for HDF5 access (both backends can read from HDF5)
        self.data_client = DataClient(market_path=self.market_data_path)

        # Cache structure: {(symbol, bucket_size): DataFrame}
        self._volume_bucket_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

        # Gap tracking (only available with IntradayFileManager)
        self._gap_info: Dict[str, List[GapInfo]] = {}

    def get_bucket_size(self, symbol: str) -> int:
        """
        Get the bucket size for a specific symbol.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'CL_F', 'RB_F')

        Returns
        -------
        int
            Bucket size for the symbol

        Examples
        --------
        >>> # Uniform bucket size
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500)
        >>> loader.get_bucket_size('CL_F')  # Returns 500
        >>>
        >>> # Per-symbol bucket sizes
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size={'CL_F': 500, 'RB_F': 100})
        >>> loader.get_bucket_size('CL_F')  # Returns 500
        >>> loader.get_bucket_size('RB_F')  # Returns 100
        >>> loader.get_bucket_size('NG_F')  # Returns 500 (default fallback)
        """
        if isinstance(self.volume_bucket_size, dict):
            return self.volume_bucket_size.get(symbol, 500)  # default fallback
        return self.volume_bucket_size

    def load_symbols(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load volume-bucketed data for multiple symbols from SCID files.

        Parameters
        ----------
        symbols : List[str]
            List of trading symbols (e.g., ['CL_F', 'NG_F'])
        start : Optional[datetime]
            Start date filter
        end : Optional[datetime]
            End date filter
        use_cache : bool
            Use cached data if available (default: True)

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping symbols to volume-bucketed DataFrames

        Examples
        --------
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500)
        >>> data = loader.load_symbols(['CL_F', 'NG_F'], start=datetime(2020, 1, 1))
        >>> print(f"Loaded {len(data['CL_F'])} buckets for CL_F")
        """
        result = {}

        for symbol in symbols:
            bucket_size = self.get_bucket_size(symbol)
            cache_key = (symbol, bucket_size)

            # Check cache
            if use_cache and cache_key in self._volume_bucket_cache:
                if self.logger:
                    self.logger.info(f"Using cached volume buckets for {symbol} (bucket_size={bucket_size})")
                result[symbol] = self._volume_bucket_cache[cache_key]
                continue

            # Load using appropriate backend
            if self.use_file_manager:
                df = self._load_with_file_manager(symbol, bucket_size, start, end)
            else:
                df = self._load_with_scid_reader(symbol, bucket_size, start, end)

            if df.empty:
                if self.logger:
                    self.logger.warning(f"No data loaded for {symbol}")
                continue

            # Cache and return
            self._volume_bucket_cache[cache_key] = df
            result[symbol] = df

            if self.logger:
                self.logger.info(
                    f"Loaded {len(df):,} volume buckets for {symbol} "
                    f"({df.index[0]} to {df.index[-1]})"
                )

        return result

    def _load_with_file_manager(
        self,
        symbol: str,
        bucket_size: int,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data using IntradayFileManager backend (full-featured)."""
        if self.logger:
            self.logger.info(
                f"Loading {symbol} with IntradayFileManager "
                f"({bucket_size}-contract buckets)"
            )

        df, gaps = self.file_manager.load_front_month_series(
            symbol=symbol,
            start=start,
            end=end,
            volume_bucket_size=bucket_size,
            detect_gaps=True
        )

        # Store gap information
        if gaps:
            self._gap_info[symbol] = gaps
            if self.logger:
                gap_summary = self.file_manager.get_gap_summary(gaps)
                self.logger.info(
                    f"{symbol}: {gap_summary['total_gaps']} gaps detected "
                    f"({gap_summary['data_missing_count']} data missing)"
                )

        return df

    def _load_with_scid_reader(
        self,
        symbol: str,
        bucket_size: int,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data using ScidReader backend (lightweight)."""
        if self.logger:
            self.logger.info(
                f"Loading {symbol} with ScidReader "
                f"({bucket_size}-contract buckets)"
            )

        try:
            # Extract base ticker (remove _F suffix)
            base_ticker = symbol.replace('_F', '')

            # Load front month series with volume bucketing
            df = self.scid_reader.load_front_month_series(
                ticker=base_ticker,
                start=start,
                end=end,
                volume_per_bar=bucket_size  # ScidReader parameter name
            )

            return df

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading {symbol} with ScidReader: {e}")
            return pd.DataFrame()

    def load_from_hdf5(
        self,
        symbols: List[str],
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load volume-bucketed data from HDF5 storage.

        This is faster than loading from SCID files if data has been pre-processed
        and stored in HDF5 format. Works with both backends.

        Parameters
        ----------
        symbols : List[str]
            List of trading symbols
        use_cache : bool
            Use cached data if available (default: True)

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping symbols to volume-bucketed DataFrames

        Examples
        --------
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500, use_file_manager=True)
        >>> # First, ensure data is written to HDF5
        >>> loader.write_to_hdf5(['CL_F', 'NG_F'])
        >>> # Then load from HDF5 (much faster)
        >>> data = loader.load_from_hdf5(['CL_F', 'NG_F'])
        """
        result = {}

        for symbol in symbols:
            bucket_size = self.get_bucket_size(symbol)
            cache_key = (symbol, bucket_size)

            # Check cache
            if use_cache and cache_key in self._volume_bucket_cache:
                result[symbol] = self._volume_bucket_cache[cache_key]
                continue

            # Construct HDF5 key
            key = f"market/{symbol}/vol_{bucket_size}"

            try:
                df = self.data_client.read_market(key)

                if df.empty:
                    if self.logger:
                        self.logger.warning(f"No data found in HDF5 for {symbol}")
                    continue

                # Cache and return
                self._volume_bucket_cache[cache_key] = df
                result[symbol] = df

                if self.logger:
                    self.logger.info(
                        f"Loaded {len(df):,} volume buckets from HDF5 for {symbol}"
                    )

            except KeyError:
                if self.logger:
                    self.logger.warning(
                        f"Volume bucket data not found in HDF5 for {symbol}. "
                        f"Use write_to_hdf5() to store it first (requires use_file_manager=True)."
                    )

        return result

    def write_to_hdf5(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        replace: bool = False
    ) -> Dict[str, Dict]:
        """
        Write volume-bucketed data to HDF5 storage for fast loading.

        Requires IntradayFileManager backend (use_file_manager=True).

        Parameters
        ----------
        symbols : List[str]
            List of trading symbols
        start : Optional[datetime]
            Start date filter
        end : Optional[datetime]
            End date filter
        replace : bool
            Replace existing data (default: False, appends instead)

        Returns
        -------
        Dict[str, Dict]
            Write statistics for each symbol

        Raises
        ------
        RuntimeError
            If using ScidReader backend (use_file_manager=False)

        Examples
        --------
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500, use_file_manager=True)
        >>> # Write volume buckets to HDF5 for fast loading
        >>> results = loader.write_to_hdf5(['CL_F', 'NG_F'], replace=True)
        >>> print(f"Wrote {results['CL_F']['records_written']} buckets")
        """
        if not self.use_file_manager:
            raise RuntimeError(
                "write_to_hdf5() requires IntradayFileManager backend. "
                "Initialize with use_file_manager=True"
            )

        results = {}

        for symbol in symbols:
            bucket_size = self.get_bucket_size(symbol)

            if self.logger:
                self.logger.info(f"Writing volume buckets for {symbol} to HDF5 (bucket_size={bucket_size})")

            result = self.file_manager.write_to_hdf5(
                symbol=symbol,
                start=start,
                end=end,
                volume_bucket_size=bucket_size,
                replace=replace
            )

            results[symbol] = result

        return results

    def get_gap_info(self, symbol: str) -> Optional[List[GapInfo]]:
        """
        Get gap information for a symbol.

        Only available when using IntradayFileManager backend.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        Optional[List[GapInfo]]
            List of detected gaps, or None if not available

        Examples
        --------
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500, use_file_manager=True)
        >>> data = loader.load_symbols(['CL_F'])
        >>> gaps = loader.get_gap_info('CL_F')
        >>> if gaps:
        ...     for gap in gaps:
        ...         print(f"Gap: {gap.start_timestamp} to {gap.end_timestamp}")
        """
        if not self.use_file_manager:
            if self.logger:
                self.logger.warning(
                    "Gap detection requires IntradayFileManager backend (use_file_manager=True)"
                )
            return None

        return self._gap_info.get(symbol)

    def get_gap_summary(self, symbol: str) -> Optional[Dict]:
        """
        Get gap summary statistics for a symbol.

        Only available when using IntradayFileManager backend.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        Optional[Dict]
            Gap summary statistics, or None if not available
        """
        if not self.use_file_manager:
            return None

        gaps = self._gap_info.get(symbol)
        if gaps:
            return self.file_manager.get_gap_summary(gaps)
        return None

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached volume bucket data.

        Parameters
        ----------
        symbol : Optional[str]
            Specific symbol to clear, or None to clear all

        Examples
        --------
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500)
        >>> # Clear all cached data
        >>> loader.clear_cache()
        >>> # Clear specific symbol
        >>> loader.clear_cache('CL_F')
        """
        if symbol is None:
            cleared = len(self._volume_bucket_cache)
            self._volume_bucket_cache.clear()
            self._gap_info.clear()
            if self.logger:
                self.logger.info(f"Cleared {cleared} cached symbol(s)")
        else:
            # Clear all cache entries for this symbol (all bucket sizes)
            keys_to_remove = [k for k in self._volume_bucket_cache.keys() if k[0] == symbol]
            for key in keys_to_remove:
                del self._volume_bucket_cache[key]
            if symbol in self._gap_info:
                del self._gap_info[symbol]
            if self.logger:
                self.logger.info(f"Cleared cache for {symbol}")

    def get_cached_symbols(self) -> List[Tuple[str, int]]:
        """
        Get list of (symbol, bucket_size) pairs currently in cache.

        Returns
        -------
        List[Tuple[str, int]]
            List of (symbol, bucket_size) tuples

        Examples
        --------
        >>> loader = VolumeBucketedDataLoader(volume_bucket_size=500)
        >>> loader.load_symbols(['CL_F', 'NG_F'])
        >>> cached = loader.get_cached_symbols()
        >>> print(f"Cached: {cached}")
        [('CL_F', 500), ('NG_F', 500)]
        """
        return list(self._volume_bucket_cache.keys())

    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols available in SCID files.

        Returns
        -------
        List[str]
            List of available trading symbols
        """
        if self.use_file_manager:
            return self.file_manager.get_available_symbols()
        else:
            # ScidReader doesn't have this method, scan directory
            if not self.data_path.exists():
                return []

            symbols = set()
            for file in self.data_path.glob('*.scid'):
                # Extract base symbol from filename (e.g., CLF25-CME.scid -> CL)
                name = file.stem
                if len(name) >= 2:
                    base_symbol = name[:2]
                    symbols.add(f"{base_symbol}_F")

            return sorted(list(symbols))

    @property
    def bucket_size(self) -> Union[int, Dict[str, int]]:
        """Get the current volume bucket size(s)."""
        return self.volume_bucket_size

    @property
    def backend_type(self) -> str:
        """Get the current backend type."""
        return "IntradayFileManager" if self.use_file_manager else "ScidReader"

    def __repr__(self) -> str:
        """String representation."""
        cached_count = len(self._volume_bucket_cache)

        # Format bucket size display
        if isinstance(self.volume_bucket_size, dict):
            bucket_display = f"{len(self.volume_bucket_size)} symbol-specific sizes"
        else:
            bucket_display = str(self.volume_bucket_size)

        return (
            f"VolumeBucketedDataLoader(bucket_size={bucket_display}, "
            f"backend='{self.backend_type}', cached_symbols={cached_count}, "
            f"data_path='{self.data_path}')"
        )


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("EXAMPLE 1: Lightweight ScidReader Backend (Read-Only)")
    print("=" * 70)

    # Create lightweight loader
    loader_light = VolumeBucketedDataLoader(
        volume_bucket_size=500,
        use_file_manager=False  # Lightweight
    )

    print(f"\n{loader_light}")
    print(f"Available symbols: {loader_light.get_available_symbols()[:5]}...")

    # Load data
    symbols = ['CL_F', 'NG_F']
    data = loader_light.load_symbols(symbols, start=datetime(2023, 1, 1))

    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"  Buckets: {len(df):,}")
        if not df.empty:
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Columns: {list(df.columns)}")

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Full-Featured IntradayFileManager Backend")
    print("=" * 70)

    # Create full-featured loader
    loader_full = VolumeBucketedDataLoader(
        volume_bucket_size=500,
        use_file_manager=True  # Full-featured
    )

    print(f"\n{loader_full}")

    # Load data with gap detection
    data_full = loader_full.load_symbols(['CL_F'], start=datetime(2023, 1, 1))

    if 'CL_F' in data_full:
        print(f"\nCL_F:")
        print(f"  Buckets: {len(data_full['CL_F']):,}")

        # Check gaps
        gap_summary = loader_full.get_gap_summary('CL_F')
        if gap_summary:
            print(f"  Gaps detected: {gap_summary['total_gaps']}")
            print(f"  Data missing: {gap_summary['data_missing_count']}")

        # Write to HDF5
        print("\nWriting to HDF5...")
        results = loader_full.write_to_hdf5(['CL_F'], replace=True)
        print(f"  Wrote {results['CL_F']['records_written']:,} buckets")

        # Load from HDF5 (much faster)
        print("\nLoading from HDF5...")
        loader_full.clear_cache()  # Clear cache to test HDF5 loading
        data_hdf5 = loader_full.load_from_hdf5(['CL_F'])
        print(f"  Loaded {len(data_hdf5['CL_F']):,} buckets from HDF5")

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Per-Symbol Volume Bucket Sizes (Volume-Aware)")
    print("=" * 70)

    # Create loader with different bucket sizes for different symbols
    # CL_F (Crude Oil): High volume → larger buckets
    # RB_F (Gasoline): Lower volume → smaller buckets
    # NG_F (Natural Gas): Medium-high volume → medium buckets
    loader_per_symbol = VolumeBucketedDataLoader(
        volume_bucket_size={
            'CL_F': 500,  # High-volume contract
            'RB_F': 100,  # Lower-volume contract
            'NG_F': 300   # Medium-high volume contract
        },
        use_file_manager=False
    )

    print(f"\n{loader_per_symbol}")
    print("\nPer-Symbol Bucket Sizes:")
    for symbol in ['CL_F', 'RB_F', 'NG_F', 'HO_F']:
        bucket_size = loader_per_symbol.get_bucket_size(symbol)
        print(f"  {symbol}: {bucket_size} contracts/bucket")

    # Load with volume-aware bucketing
    print("\nLoading with volume-aware bucket sizes...")
    data_per_symbol = loader_per_symbol.load_symbols(
        ['CL_F', 'RB_F', 'NG_F'],
        start=datetime(2023, 1, 1)
    )

    for symbol, df in data_per_symbol.items():
        bucket_size = loader_per_symbol.get_bucket_size(symbol)
        print(f"\n{symbol}:")
        print(f"  Bucket size: {bucket_size} contracts")
        print(f"  Total buckets: {len(df):,}")
        if not df.empty:
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
