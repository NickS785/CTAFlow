from .volume import MarketProfileExtractor, VPINExtractor
from .volume.profile import NumberBarsExtractor
from .base_extractor import ScidBaseExtractor
from ..config import DLY_DATA_PATH
from ..data.raw_formatting.contract_specs import CONTRACT_SPECS_RAW, ContractSpecs
from ..utils.volume_bucket import auto_bucket_size
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from datetime import datetime, date, time, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import logging

logger = logging.getLogger(__name__)


def get_default_tick_size(ticker: str) -> Optional[float]:
    """
    Get the default minimum tick size for a ticker from CONTRACT_SPECS_RAW.

    Args:
        ticker: Contract symbol (e.g., 'CL', 'RB', 'HO')

    Returns:
        Tick size in the contract's price units, or None if not found
    """
    sym = ticker.upper()
    if sym in CONTRACT_SPECS_RAW:
        return CONTRACT_SPECS_RAW[sym].get('min_tick')
    return None


def _extract_single_date_worker(args: tuple) -> Dict:
    """
    Worker function for multiprocessing. Creates fresh extractors per process.

    Args:
        args: Tuple of (config_dict, date_str) to avoid pickling issues

    Returns:
        Dict with 'date', 'vpin', 'profile', 'number_bars', 'number_bars_meta' keys
    """
    config_dict, date_str = args
    dt = pd.Timestamp(date_str)

    # Recreate config from dict
    config = FeatureExtractorConfig(**config_dict)

    # Create fresh extractors (can't pickle the manager objects)
    from .base_extractor import ScidBaseExtractor
    from .volume import VPINExtractor, MarketProfileExtractor
    from .volume.profile import NumberBarsExtractor

    base = ScidBaseExtractor(config.data_dir, config.ticker, config.tz)
    vpin_ext = VPINExtractor(
        config.data_dir, config.ticker, config.tz,
        config.vpin_bucket_size, config.vpin_window
    )
    profile_ext = MarketProfileExtractor(config.data_dir, config.ticker, config.tz) if config.include_profile else None
    nb_ext = NumberBarsExtractor(
        config.data_dir, config.ticker, config.tz,
        tick_size=config.profile_tick_size or 0.01,
        interval=config.num_bars_interval,
        vwap_window=config.num_bars_vwap_window,
        num_levels=config.num_bars_levels,
        centering_method=config.num_bars_centering
    ) if config.include_number_bars else None

    # Build time windows
    vpin_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.vpin_start_time}")
    vpin_end = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.vpin_end_time}")

    starts = [vpin_start]
    ends = [vpin_end]

    profile_start = None
    if config.include_profile:
        profile_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.profile_start_time}")
        starts.append(profile_start)

    if config.include_number_bars:
        nb_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.num_bars_start_time}")
        nb_end = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.num_bars_end_time}")
        starts.append(nb_start)
        ends.append(nb_end)

    if config.include_pre_summary and config.pre_summary_start_time:
        pre_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.pre_summary_start_time}")
        starts.append(pre_start)

    start_dt = min(starts)
    end_dt = max(ends)

    results = {
        'date': dt,
        'vpin': pd.DataFrame(),
        'profile': pd.DataFrame(),
        'number_bars': np.array([]),
        'number_bars_meta': pd.DataFrame()
    }

    try:
        df_raw = base.get_stitched_data(
            start_time=start_dt,
            end_time=end_dt,
            columns=["Close", "TotalVolume", "BidVolume", "AskVolume"]
        )
    except Exception as e:
        logger.warning(f"Failed to fetch data for {dt.date()}: {e}")
        return results

    if df_raw.empty:
        return results

    # Profile: from profile_start to vpin_start (prior session context)
    poc, val, vah = np.nan, np.nan, np.nan
    if config.include_profile and profile_ext is not None and profile_start is not None:
        profile_data = df_raw.between_time(profile_start.time(), vpin_start.time(), inclusive='left')
        if not profile_data.empty:
            try:
                profile = profile_ext.calculate_volume_profile(
                    profile_data,
                    tick_size=config.profile_tick_size
                )
                results['profile'] = profile

                if not profile.empty and 'TotalVolume' in profile.columns:
                    total_vol = profile['TotalVolume'].sum()
                    poc = profile['TotalVolume'].idxmax()

                    sorted_profile = profile.sort_values('TotalVolume', ascending=False)
                    cumvol = sorted_profile['TotalVolume'].cumsum()
                    va_threshold = total_vol * config.value_area_pct
                    va_levels = sorted_profile[cumvol <= va_threshold].index
                    val = va_levels.min() if len(va_levels) > 0 else poc
                    vah = va_levels.max() if len(va_levels) > 0 else poc

            except Exception as e:
                logger.warning(f"Profile calculation failed for {dt.date()}: {e}")

    # Number Bars extraction - use calculate_number_bars with pre-fetched data
    if config.include_number_bars and nb_ext is not None:
        nb_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.num_bars_start_time}")
        nb_end = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.num_bars_end_time}")
        try:
            # Filter data to number bars window (with VWAP lookback for centering)
            lookback = pd.Timedelta(config.num_bars_vwap_window) * 2
            nb_data = df_raw.between_time(
                (nb_start - lookback).time(),
                nb_end.time(),
                inclusive='both'
            )
            if not nb_data.empty:
                tensor, meta = nb_ext.calculate_number_bars(
                    df=nb_data,
                    start_time=nb_start,
                    interval=config.num_bars_interval,
                    tick_size=config.profile_tick_size,
                    vwap_window=config.num_bars_vwap_window,
                    num_levels=config.num_bars_levels
                )
                results['number_bars'] = tensor
                results['number_bars_meta'] = meta
        except Exception as e:
            logger.warning(f"Number bars extraction failed for {dt.date()}: {e}")

    # VPIN: from vpin_start to vpin_end
    vpin_data = df_raw.between_time(vpin_start.time(), vpin_end.time(), inclusive='both')

    # Calculate Initial Balance (first N minutes from profile/RTH start)
    ib_high, ib_low = np.nan, np.nan
    if config.include_ib and profile_start is not None:
        ib_end = profile_start + pd.Timedelta(minutes=config.ib_minutes)
        ib_data = df_raw.between_time(profile_start.time(), ib_end.time(), inclusive='both')
        if not ib_data.empty and 'Close' in ib_data.columns:
            ib_high = ib_data['Close'].max()
            ib_low = ib_data['Close'].min()

    # Pre-summary calculation
    pre_summary = {}
    if config.include_pre_summary and config.pre_summary_start_time:
        pre_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {config.pre_summary_start_time}")
        pre_data = df_raw.between_time(pre_start.time(), vpin_start.time(), inclusive='left')
        if not pre_data.empty:
            total_vol = pre_data['TotalVolume'].sum()
            vwap = (pre_data['Close'] * pre_data['TotalVolume']).sum() / total_vol if total_vol > 0 else np.nan
            delta = np.nan
            if 'AskVolume' in pre_data.columns and 'BidVolume' in pre_data.columns:
                delta = float(pre_data['AskVolume'].sum()) - float(pre_data['BidVolume'].sum())
            pre_summary = {
                'pre_vwap': vwap,
                'pre_volume': total_vol,
                'pre_delta': delta,
                'pre_high': pre_data['Close'].max(),
                'pre_low': pre_data['Close'].min()
            }

    if not vpin_data.empty:
        try:
            vpin_df = vpin_ext.calculate_vpin(
                vpin_data,
                bucket_volume=config.vpin_bucket_size or vpin_ext.bucket_volume,
                window=config.vpin_window,
                include_sequence_features=config.include_sequence_features
            )
            if not vpin_df.empty:
                vpin_df['poc'] = poc
                vpin_df['val'] = val
                vpin_df['vah'] = vah
                if config.include_ib:
                    vpin_df['ib_high'] = ib_high
                    vpin_df['ib_low'] = ib_low
                for key, value in pre_summary.items():
                    vpin_df[key] = value
            results['vpin'] = vpin_df
        except Exception as e:
            logger.warning(f"VPIN calculation failed for {dt.date()}: {e}")

    return results


@dataclass
class FeatureExtractorConfig:
    ticker: str
    data_dir: str = DLY_DATA_PATH
    tz: str = "America/Chicago"

    # VPIN Configuration
    vpin_bucket_size: Optional[int] = None  # None = use auto_bucket_size
    vpin_window: int = 20
    vpin_start_time: str = "08:30"
    vpin_end_time: str = "09:30"
    auto_bucket: bool = True  # Use auto_bucket_size when vpin_bucket_size is None

    # Profile Configuration
    include_profile: bool = True  # Toggle profile extraction
    profile_tick_size: Optional[float] = None  # None = use default from contract_specs
    profile_start_time: str = "02:00"
    profile_end_time: str = "09:30"

    # Number Bars Configuration
    include_number_bars: bool = False  # Toggle number bars extraction
    num_bars_start_time: str = "08:30"
    num_bars_end_time: str = "09:30"
    num_bars_interval: str = "15min"  # Time interval for number bars
    num_bars_levels: int = 100  # Price levels above/below center
    num_bars_centering: str = "rolling_vwap"  # 'rolling_vwap', 'bar_vwap', 'bar_mid'
    num_bars_vwap_window: str = "1h"  # VWAP lookback for centering

    # Neural network training options
    include_sequence_features: bool = True  # Golden Trio for LSTM
    profile_n_bins: int = 96  # Fixed histogram size for CNN-1D

    # Profile and IB options
    value_area_pct: float = 0.7  # Value area percentage (default 70%)
    include_ib: bool = True  # Include Initial Balance (first hour H/L)
    ib_minutes: int = 60  # Initial Balance period in minutes

    # Pre-period summary for VPIN
    include_pre_summary: bool = False  # Include summary stats before VPIN window
    pre_summary_start_time: Optional[str] = None  # Start of pre-period (e.g., "02:00")

    def __post_init__(self):
        """Resolve default tick size from contract specs if not provided."""
        if self.profile_tick_size is None:
            default_tick = get_default_tick_size(self.ticker)
            if default_tick is not None:
                self.profile_tick_size = default_tick
                logger.debug(f"Using default tick size {default_tick} for {self.ticker}")


class MultiFeatureExtraction(ScidBaseExtractor):
    """
    Extracts VPIN, Volume Profile, and Number Bars features for multiple dates efficiently.

    Uses a single get_stitched_data call per date to fetch raw tick data,
    then computes all requested features from that shared data.

    Features:
    - VPIN: Volume-synchronized probability of informed trading
    - Profile: Volume profile with POC, VAL, VAH
    - Number Bars: Sierra Chart-style footprint charts as normalized tensors
    """

    def __init__(self, config: FeatureExtractorConfig, dates: List[Union[str, pd.Timestamp, date]]):
        super().__init__(config.data_dir, config.ticker, config.tz)
        self.config = config
        self.vpin_extractor = VPINExtractor(
            config.data_dir, config.ticker, config.tz,
            config.vpin_bucket_size, config.vpin_window
        )
        self.profile_extractor = MarketProfileExtractor(
            config.data_dir, config.ticker, config.tz
        ) if config.include_profile else None

        self.number_bars_extractor = NumberBarsExtractor(
            config.data_dir, config.ticker, config.tz,
            tick_size=config.profile_tick_size or 0.01,
            interval=config.num_bars_interval,
            vwap_window=config.num_bars_vwap_window,
            num_levels=config.num_bars_levels,
            centering_method=config.num_bars_centering
        ) if config.include_number_bars else None

        self.dates = [pd.Timestamp(d) for d in dates]
        self._bucket_size_cache: Optional[int] = None  # Cache for auto_bucket_size

    def _build_time_range(self, dt: pd.Timestamp, start_time: str, end_time: str) -> tuple:
        """Build datetime range from date and time strings."""
        start_dt = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {start_time}")
        end_dt = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {end_time}")
        return start_dt, end_dt

    def _get_combined_window(self, dt: pd.Timestamp) -> tuple:
        """Get the earliest start and latest end across all extraction windows."""
        starts = []
        ends = []

        # VPIN window
        vpin_start, vpin_end = self._build_time_range(
            dt, self.config.vpin_start_time, self.config.vpin_end_time
        )
        starts.append(vpin_start)
        ends.append(vpin_end)

        # Profile window
        if self.config.include_profile:
            profile_start, profile_end = self._build_time_range(
                dt, self.config.profile_start_time, self.config.profile_end_time
            )
            starts.append(profile_start)
            ends.append(profile_end)

        # Number bars window
        if self.config.include_number_bars:
            nb_start, nb_end = self._build_time_range(
                dt, self.config.num_bars_start_time, self.config.num_bars_end_time
            )
            starts.append(nb_start)
            ends.append(nb_end)

        # Pre-summary window
        if self.config.include_pre_summary and self.config.pre_summary_start_time:
            pre_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {self.config.pre_summary_start_time}")
            starts.append(pre_start)

        return min(starts), max(ends)

    def _calculate_bucket_size(self, df_raw: pd.DataFrame) -> int:
        """
        Calculate optimal bucket size using auto_bucket_size.

        Args:
            df_raw: Raw tick data with AskVolume and BidVolume columns

        Returns:
            Optimal bucket size for VPIN calculation
        """
        if self._bucket_size_cache is not None:
            return self._bucket_size_cache

        if df_raw.empty:
            return 50  # Default fallback

        # Prepare data for auto_bucket_size
        ticks = df_raw.reset_index().rename(columns={df_raw.index.name or 'index': 'ts'})
        if 'ts' not in ticks.columns:
            ticks['ts'] = df_raw.index

        try:
            bucket_size = auto_bucket_size(ticks, cadence_target=50)
            self._bucket_size_cache = bucket_size
            logger.info(f"Auto bucket size determined: {bucket_size}")
            return bucket_size
        except Exception as e:
            logger.warning(f"auto_bucket_size failed: {e}, using default 50")
            return 50

    def _calculate_pre_summary(self, df_raw: pd.DataFrame, pre_start: pd.Timestamp,
                               pre_end: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate summary statistics for the pre-VPIN period.

        Returns dict with vwap, total_volume, total_delta, high, low
        """
        pre_data = df_raw.between_time(pre_start.time(), pre_end.time(), inclusive='left')
        if pre_data.empty:
            return {'pre_vwap': np.nan, 'pre_volume': np.nan, 'pre_delta': np.nan,
                    'pre_high': np.nan, 'pre_low': np.nan}

        total_vol = pre_data['TotalVolume'].sum()
        vwap = (pre_data['Close'] * pre_data['TotalVolume']).sum() / total_vol if total_vol > 0 else np.nan

        delta = np.nan
        if 'AskVolume' in pre_data.columns and 'BidVolume' in pre_data.columns:
            delta = float(pre_data['AskVolume'].sum()) - float(pre_data['BidVolume'].sum())

        return {
            'pre_vwap': vwap,
            'pre_volume': total_vol,
            'pre_delta': delta,
            'pre_high': pre_data['Close'].max(),
            'pre_low': pre_data['Close'].min()
        }

    def extract_date(self, dt: pd.Timestamp) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """
        Extract all features for a single date using one stitched data call.

        Returns dict with 'vpin', 'profile', 'number_bars', 'number_bars_meta' entries.
        """
        # Get combined time window
        start_dt, end_dt = self._get_combined_window(dt)

        # Initialize results
        results = {
            'date': dt,
            'vpin': pd.DataFrame(),
            'profile': pd.DataFrame(),
            'number_bars': np.array([]),
            'number_bars_meta': pd.DataFrame()
        }

        # Single fetch for all data needed
        try:
            df_raw = self.get_stitched_data(
                start_time=start_dt,
                end_time=end_dt,
                columns=["Close", "TotalVolume", "BidVolume", "AskVolume"]
            )
        except Exception as e:
            logger.warning(f"Failed to fetch data for {dt.date()}: {e}")
            return results

        if df_raw.empty:
            return results

        # Build time ranges
        vpin_start, vpin_end = self._build_time_range(
            dt, self.config.vpin_start_time, self.config.vpin_end_time
        )

        # Profile-related calculations
        poc, val, vah = np.nan, np.nan, np.nan
        profile_start = None

        if self.config.include_profile and self.profile_extractor is not None:
            profile_start, _ = self._build_time_range(
                dt, self.config.profile_start_time, self.config.profile_end_time
            )
            profile_data = df_raw.between_time(
                profile_start.time(), vpin_start.time(), inclusive='left'
            )
            if not profile_data.empty:
                try:
                    profile = self.profile_extractor.calculate_volume_profile(
                        profile_data,
                        tick_size=self.config.profile_tick_size
                    )
                    results['profile'] = profile

                    if not profile.empty and 'TotalVolume' in profile.columns:
                        total_vol = profile['TotalVolume'].sum()
                        poc = profile['TotalVolume'].idxmax()

                        sorted_profile = profile.sort_values('TotalVolume', ascending=False)
                        cumvol = sorted_profile['TotalVolume'].cumsum()
                        va_threshold = total_vol * self.config.value_area_pct
                        va_levels = sorted_profile[cumvol <= va_threshold].index
                        val = va_levels.min() if len(va_levels) > 0 else poc
                        vah = va_levels.max() if len(va_levels) > 0 else poc

                except Exception as e:
                    logger.warning(f"Profile calculation failed for {dt.date()}: {e}")

        # Number Bars extraction - use calculate_number_bars with pre-fetched data
        if self.config.include_number_bars and self.number_bars_extractor is not None:
            nb_start, nb_end = self._build_time_range(
                dt, self.config.num_bars_start_time, self.config.num_bars_end_time
            )
            try:
                # Filter data to number bars window (with VWAP lookback for centering)
                nb_data = df_raw.between_time(
                    (nb_start - pd.Timedelta(self.config.num_bars_vwap_window) * 2).time(),
                    nb_end.time(),
                    inclusive='both'
                )
                if not nb_data.empty:
                    tensor, meta = self.number_bars_extractor.calculate_number_bars(
                        df=nb_data,
                        start_time=nb_start,
                        interval=self.config.num_bars_interval,
                        tick_size=self.config.profile_tick_size,
                        vwap_window=self.config.num_bars_vwap_window,
                        num_levels=self.config.num_bars_levels
                    )
                    results['number_bars'] = tensor
                    results['number_bars_meta'] = meta
            except Exception as e:
                logger.warning(f"Number bars extraction failed for {dt.date()}: {e}")

        # Determine bucket size for VPIN
        bucket_size = self.config.vpin_bucket_size
        if bucket_size is None and self.config.auto_bucket:
            bucket_size = self._calculate_bucket_size(df_raw)
        elif bucket_size is None:
            bucket_size = self.vpin_extractor.bucket_volume

        # VPIN extraction
        vpin_data = df_raw.between_time(vpin_start.time(), vpin_end.time(), inclusive='both')

        # Initial Balance calculation
        ib_high, ib_low = np.nan, np.nan
        if self.config.include_ib and profile_start is not None:
            ib_end = profile_start + pd.Timedelta(minutes=self.config.ib_minutes)
            ib_data = df_raw.between_time(profile_start.time(), ib_end.time(), inclusive='both')
            if not ib_data.empty and 'Close' in ib_data.columns:
                ib_high = ib_data['Close'].max()
                ib_low = ib_data['Close'].min()

        # Pre-summary calculation
        pre_summary = {}
        if self.config.include_pre_summary and self.config.pre_summary_start_time:
            pre_start = pd.Timestamp(f"{dt.strftime('%Y-%m-%d')} {self.config.pre_summary_start_time}")
            pre_summary = self._calculate_pre_summary(df_raw, pre_start, vpin_start)

        if not vpin_data.empty:
            try:
                vpin_df = self.vpin_extractor.calculate_vpin(
                    vpin_data,
                    bucket_volume=bucket_size,
                    window=self.config.vpin_window,
                    include_sequence_features=self.config.include_sequence_features
                )
                if not vpin_df.empty:
                    vpin_df['poc'] = poc
                    vpin_df['val'] = val
                    vpin_df['vah'] = vah
                    if self.config.include_ib:
                        vpin_df['ib_high'] = ib_high
                        vpin_df['ib_low'] = ib_low
                    # Add pre-summary to VPIN
                    for key, value in pre_summary.items():
                        vpin_df[key] = value
                results['vpin'] = vpin_df
            except Exception as e:
                logger.warning(f"VPIN calculation failed for {dt.date()}: {e}")

        return results

    def _get_config_dict(self) -> dict:
        """Convert config to dict for pickling in multiprocessing."""
        return {
            'ticker': self.config.ticker,
            'data_dir': self.config.data_dir,
            'tz': self.config.tz,
            # VPIN
            'vpin_bucket_size': self.config.vpin_bucket_size,
            'vpin_window': self.config.vpin_window,
            'vpin_start_time': self.config.vpin_start_time,
            'vpin_end_time': self.config.vpin_end_time,
            'auto_bucket': self.config.auto_bucket,
            # Profile
            'include_profile': self.config.include_profile,
            'profile_tick_size': self.config.profile_tick_size,
            'profile_start_time': self.config.profile_start_time,
            'profile_end_time': self.config.profile_end_time,
            # Number Bars
            'include_number_bars': self.config.include_number_bars,
            'num_bars_start_time': self.config.num_bars_start_time,
            'num_bars_end_time': self.config.num_bars_end_time,
            'num_bars_interval': self.config.num_bars_interval,
            'num_bars_levels': self.config.num_bars_levels,
            'num_bars_centering': self.config.num_bars_centering,
            'num_bars_vwap_window': self.config.num_bars_vwap_window,
            # Neural network options
            'include_sequence_features': self.config.include_sequence_features,
            'profile_n_bins': self.config.profile_n_bins,
            # Profile/IB
            'value_area_pct': self.config.value_area_pct,
            'include_ib': self.config.include_ib,
            'ib_minutes': self.config.ib_minutes,
            # Pre-summary
            'include_pre_summary': self.config.include_pre_summary,
            'pre_summary_start_time': self.config.pre_summary_start_time,
        }

    def extract_all(
            self,
            verbose: bool = False,
            n_jobs: int = 1,
            use_threads: bool = False,
    ) -> List[Dict[str, pd.DataFrame]]:
        """
        Extract features for all dates.

        Parameters
        ----------
        verbose : bool, default False
            Print progress info
        n_jobs : int, default 1
            Number of parallel workers. Set to -1 to use all available CPUs.
            Set to 1 for sequential processing (no parallelism).
        use_threads : bool, default False
            If True, use ThreadPoolExecutor (better for I/O-bound, no pickling).
            If False, use ProcessPoolExecutor (better for CPU-bound, bypasses GIL).

        Returns
        -------
        List[Dict]
            List of dicts, each containing 'date', 'vpin', and 'profile'.
        """
        if n_jobs == 1:
            # Sequential processing
            results = []
            for i, dt in enumerate(self.dates):
                if verbose:
                    print(f"Extracting {i+1}/{len(self.dates)}: {dt.date()}")
                results.append(self.extract_date(dt))
            return results

        # Parallel processing
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        n_jobs = min(n_jobs, len(self.dates))

        if use_threads:
            # ThreadPoolExecutor - can use instance methods directly
            executor_class = ThreadPoolExecutor
            work_items = self.dates

            def worker(dt):
                return self.extract_date(dt)

            if verbose:
                print(f"Processing {len(self.dates)} dates with {n_jobs} threads...")

            with executor_class(max_workers=n_jobs) as executor:
                futures = {executor.submit(worker, dt): dt for dt in work_items}
                results = []
                for i, future in enumerate(as_completed(futures)):
                    if verbose:
                        dt = futures[future]
                        print(f"Completed {i+1}/{len(self.dates)}: {dt.date()}")
                    results.append(future.result())

        else:
            # ProcessPoolExecutor - need to use module-level function
            config_dict = self._get_config_dict()
            work_items = [(config_dict, dt.isoformat()) for dt in self.dates]

            if verbose:
                print(f"Processing {len(self.dates)} dates with {n_jobs} processes...")

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(_extract_single_date_worker, item): item[1] for item in work_items}
                results = []
                for i, future in enumerate(as_completed(futures)):
                    if verbose:
                        date_str = futures[future]
                        print(f"Completed {i+1}/{len(self.dates)}: {date_str[:10]}")
                    results.append(future.result())

        # Sort by date to maintain order
        results.sort(key=lambda x: x['date'])
        return results

    def extract_vpin_summary(
            self,
            verbose: bool = False,
            n_jobs: int = 1,
            use_threads: bool = False,
    ) -> pd.DataFrame:
        """
        Extract VPIN summary (last value per date) for all dates.

        Parameters
        ----------
        verbose : bool, default False
            Print progress info
        n_jobs : int, default 1
            Number of parallel workers (-1 for all CPUs)
        use_threads : bool, default False
            Use threads instead of processes

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date with VPIN metrics.
        """
        all_results = self.extract_all(verbose=verbose, n_jobs=n_jobs, use_threads=use_threads)

        summaries = []
        for res in all_results:
            if res['vpin'].empty:
                continue
            # Get last VPIN value for the day
            last_row = res['vpin'].iloc[-1].to_dict()
            last_row['date'] = res['date']
            summaries.append(last_row)

        if not summaries:
            return pd.DataFrame()

        df = pd.DataFrame(summaries).set_index('date')
        return df

    def extract_profile_summary(
            self,
            verbose: bool = False,
            n_jobs: int = 1,
            use_threads: bool = False,
    ) -> pd.DataFrame:
        """
        Extract profile summary metrics (POC, value area, delta) for all dates.

        Parameters
        ----------
        verbose : bool, default False
            Print progress info
        n_jobs : int, default 1
            Number of parallel workers (-1 for all CPUs)
        use_threads : bool, default False
            Use threads instead of processes

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date with profile metrics.
        """
        all_results = self.extract_all(verbose=verbose, n_jobs=n_jobs, use_threads=use_threads)

        summaries = []
        for res in all_results:
            profile = res['profile']
            if profile.empty:
                continue

            total_vol = profile['TotalVolume'].sum()
            poc_price = profile['TotalVolume'].idxmax()  # Point of Control

            # Value Area (70% of volume around POC)
            sorted_profile = profile.sort_values('TotalVolume', ascending=False)
            cumvol = sorted_profile['TotalVolume'].cumsum()
            va_threshold = total_vol * 0.7
            va_levels = sorted_profile[cumvol <= va_threshold].index
            val = va_levels.min() if len(va_levels) > 0 else poc_price
            vah = va_levels.max() if len(va_levels) > 0 else poc_price

            summary = {
                'date': res['date'],
                'poc': poc_price,
                'val': val,
                'vah': vah,
                'total_volume': total_vol,
            }

            if 'Delta' in profile.columns:
                summary['total_delta'] = profile['Delta'].sum()
                summary['poc_delta'] = profile.loc[poc_price, 'Delta']

            summaries.append(summary)

        if not summaries:
            return pd.DataFrame()

        return pd.DataFrame(summaries).set_index('date')

    def export_to_npz(
            self,
            results: List[Dict[str, pd.DataFrame]],
            output_dir: str,
            prefix: str = None,
            fit_size: float = 0.8,
            encoder_config: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Export extraction results to NPZ (encoded profiles) and Parquet (VPIN).

        Parameters
        ----------
        results : List[Dict]
            Output from extract_all()
        output_dir : str
            Directory to save files
        prefix : str, optional
            Filename prefix (default: ticker name)
        fit_size : float, default 0.8
            Fraction of data to use for fitting the encoder (first N days)
        encoder_config : Dict, optional
            Override VolumeProfileEncoderConfig parameters

        Returns
        -------
        Dict[str, str]
            Paths to saved files {'profiles': path, 'vpin': path}
        """
        from pathlib import Path
        from .volume.profile_encoder import (
            VolumeProfileEncoder, VolumeProfileEncoderConfig, save_profiles_npz
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix or self.config.ticker

        # Filter non-empty profiles
        valid_results = [r for r in results if not r['profile'].empty]

        if not valid_results:
            logger.warning("No valid results to export")
            return {}

        # Extract profile DataFrames
        profiles = [r['profile'] for r in valid_results]
        dates = [r['date'] for r in valid_results]

        # Fit encoder on training portion
        n_train = int(len(profiles) * fit_size)
        train_profiles = profiles[:n_train]

        # Use profile_n_bins from config, allow encoder_config to override
        enc_cfg = {'num_bins': self.config.profile_n_bins}
        if encoder_config:
            enc_cfg.update(encoder_config)
        cfg = VolumeProfileEncoderConfig(**enc_cfg)
        encoder = VolumeProfileEncoder(cfg)
        encoder.fit(train_profiles)

        # Transform all profiles -> (N, C, B)
        encoded = encoder.transform_many(profiles)

        # Save profiles NPZ
        profile_path = output_dir / f"{prefix}_profiles.npz"
        save_profiles_npz(profile_path, dates, encoded)

        # VPIN -> Parquet (variable-length sequences with datetime index)
        vpin_frames = []
        for r in valid_results:
            if not r['vpin'].empty:
                vpin = r['vpin'].copy()
                vpin['date'] = r['date'].date()
                vpin_frames.append(vpin)

        paths = {'profiles': str(profile_path)}

        if vpin_frames:
            vpin_df = pd.concat(vpin_frames, axis=0)
            vpin_path = output_dir / f"{prefix}_vpin.parquet"
            vpin_df.to_parquet(vpin_path)
            paths['vpin'] = str(vpin_path)

        logger.info(f"Exported {len(valid_results)} days (fit on {n_train}) to {output_dir}")
        return paths