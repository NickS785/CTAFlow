import pandas as pd
import numpy as np
from datetime import datetime, date, time
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
import logging
from dataclasses import dataclass
# Import from your provided files
from sierrapy.parser.scid_parse import FastScidReader, ScidTickerFileManager, ScidContractInfo
from ...data.contract_expiry_rules import calculate_expiry, get_roll_buffer_days
from ..base_extractor import ScidBaseExtractor
logger = logging.getLogger(__name__)

@dataclass
class MarketProfileConfig:
    data_dir: str
    ticker: str
    tz: str = "America/Chicago"
    tick_size: float = 0.01


@dataclass
class NumberBarsConfig:
    """Configuration for NumberBarsExtractor."""
    data_dir: str
    ticker: str
    tz: str = "America/Chicago"
    tick_size: float = 0.25  # Price increment (e.g., 0.25 for ES, 0.01 for NQ)
    interval: str = "15min"  # Default time interval for bars
    vwap_window: str = "1h"  # Rolling VWAP lookback for centering
    num_levels: int = 100  # Price levels above/below center (total = 2*num_levels+1)
    centering_method: str = "rolling_vwap"  # 'rolling_vwap', 'bar_vwap', 'bar_mid'
    volume_bucket_size: Optional[int] = None  # If set, use volume-bucketed bars instead of time




class MarketProfileExtractor(ScidBaseExtractor):
    def __init__(self, data_dir: str, ticker: Optional[str] = None, tz: str = "America/Chicago", tick_size=None):
        """
        Initialize MarketProfileExtractor.

        Args:
            data_dir: Path to SCID data directory
            ticker: Default ticker symbol (can be overridden per-call)
            tz: Timezone for time interpretation (default: America/Chicago)
        """
        super().__init__(data_dir, ticker, tz)
        self.tick_size = tick_size if tick_size else None
        return

    def calculate_volume_profile(self, df: pd.DataFrame, tick_size: Optional[float] = None) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        data = df.copy()

        # Convert volume columns to float64 to prevent uint32 overflow on subtraction
        for col in ['TotalVolume', 'BidVolume', 'AskVolume']:
            if col in data.columns:
                data[col] = data[col].astype(np.float64)

        # Binning logic
        if tick_size is not None and tick_size > 0:
            data['PriceBin'] = (np.round(data['Close'] / tick_size) * tick_size).astype(float)
            data['PriceBin'] = data['PriceBin'].round(decimals=6)
        else:
            data['PriceBin'] = data['Close']

        agg_dict = {'TotalVolume': 'sum'}
        for col in ['BidVolume', 'AskVolume', 'NumTrades']:
            if col in data.columns:
                agg_dict[col] = 'sum'

        profile = data.groupby('PriceBin').agg(agg_dict)
        profile.index.name = 'Price'
        profile.sort_index(ascending=True, inplace=True)

        if 'BidVolume' in profile.columns and 'AskVolume' in profile.columns:
            # Ensure float64 after aggregation to prevent overflow
            profile['Delta'] = profile['AskVolume'].astype(np.float64) - profile['BidVolume'].astype(np.float64)

        return profile

    def get_profile(self, start_time: str, end_time: str, tick_size: float = None) -> pd.DataFrame:
        """
        Main entry point for getting a profile over a period.
        """
        # Fetch underlying data
        df_raw = self.get_stitched_data(start_time, end_time)

        # Transform to profile
        return self.calculate_volume_profile(df_raw, tick_size=tick_size)

    def __getitem__(self, item: slice):
        """
        Slice syntax: extractor['2023-01-01':'2023-01-02':0.01]
        """
        if not isinstance(item, slice):
            raise TypeError("Expected slice object")

        start_str = self._to_time_string(item.start)
        end_str = self._to_time_string(item.stop)
        tick_size = float(item.step) if isinstance(item.step, (int, float)) else None

        return self.get_profile(start_str, end_str, tick_size=tick_size)



class NumberBarsExtractor(MarketProfileExtractor):
    """
    Generates Sierra Chart-style 'Number Bars' (Footprint charts) as normalized tensors.

    Structure:
    - Resamples tick data into bars (time-based or volume-bucketed).
    - For each bar, groups volume by price level.
    - Centers the price grid around a reference price (Rolling VWAP by default).
    - Produces a fixed-size 3D array: [Time, Price_Levels, Features].

    Features per price level:
    - TotalVolume: Total contracts/shares traded
    - BidVolume: Volume at bid (selling pressure)
    - AskVolume: Volume at ask (buying pressure)
    - Delta: AskVolume - BidVolume (order flow imbalance)
    - NumTrades: Number of individual trades

    Usage:
        # From config
        config = NumberBarsConfig(data_dir='/path', ticker='ES', tick_size=0.25)
        extractor = NumberBarsExtractor.from_config(config)

        # Direct initialization
        extractor = NumberBarsExtractor('/path', 'ES', tick_size=0.25, interval='15min')

        # Slice syntax - time-based bars
        tensor, meta = extractor['2024-01-01 08:30':'2024-01-01 15:00']  # default interval
        tensor, meta = extractor['2024-01-01 08:30':'2024-01-01 15:00':'30min']  # 30min bars

        # Slice syntax - volume-bucketed bars
        tensor, meta = extractor['2024-01-01 08:30':'2024-01-01 15:00':5000]  # 5000 volume bars
    """

    def __init__(self,
                 data_dir: str,
                 ticker: Optional[str] = None,
                 tz: str = "America/Chicago",
                 tick_size: float = 0.25,
                 interval: str = "15min",
                 vwap_window: str = "1h",
                 num_levels: int = 100,
                 centering_method: str = "rolling_vwap"):
        """
        Initialize NumberBarsExtractor.

        Args:
            data_dir: Path to SCID data directory
            ticker: Default ticker symbol
            tz: Timezone for time interpretation
            tick_size: Minimum price increment (e.g., 0.25 for ES, 0.5 for CL)
            interval: Default time interval for bars ('15min', '30min', '1h', etc.)
            vwap_window: Lookback period for rolling VWAP centering
            num_levels: Price levels above/below center (total grid = 2*num_levels+1)
            centering_method: 'rolling_vwap', 'bar_vwap', or 'bar_mid'
        """
        super().__init__(data_dir, ticker, tz, tick_size)
        self.interval = interval
        self.vwap_window = vwap_window
        self.num_levels = num_levels
        self.centering_method = centering_method

    @classmethod
    def from_config(cls, config: NumberBarsConfig) -> 'NumberBarsExtractor':
        """Create extractor from NumberBarsConfig dataclass."""
        return cls(
            data_dir=config.data_dir,
            ticker=config.ticker,
            tz=config.tz,
            tick_size=config.tick_size,
            interval=config.interval,
            vwap_window=config.vwap_window,
            num_levels=config.num_levels,
            centering_method=config.centering_method
        )

    def _calculate_rolling_vwap(self, df: pd.DataFrame, window_time: str) -> pd.Series:
        """
        Calculates a rolling VWAP based on a time window.
        VWAP = Sum(Price * Volume) / Sum(Volume)
        """
        df_1min = df.resample('1min').agg({
            'Close': 'mean',
            'TotalVolume': 'sum'
        }).dropna()

        pv = df_1min['Close'] * df_1min['TotalVolume']
        vol = df_1min['TotalVolume']

        roll_pv = pv.rolling(window_time).sum()
        roll_vol = vol.rolling(window_time).sum()

        return (roll_pv / roll_vol).ffill()

    def _create_volume_buckets(self, df: pd.DataFrame, bucket_size: int) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
        """
        Create volume-bucketed bars from tick data.

        Args:
            df: DataFrame with tick data (must have TotalVolume column)
            bucket_size: Volume threshold per bucket

        Returns:
            List of (bucket_end_timestamp, bucket_df) tuples
        """
        if df.empty:
            return []

        df = df.copy()
        df['CumVolume'] = df['TotalVolume'].cumsum()
        df['BucketId'] = (df['CumVolume'] // bucket_size).astype(int)

        buckets = []
        for bucket_id, group in df.groupby('BucketId'):
            if not group.empty:
                bucket_end = group.index[-1]
                buckets.append((bucket_end, group))

        return buckets

    def _get_center_price(self,
                          group: pd.DataFrame,
                          vwap_series: Optional[pd.Series],
                          timestamp: pd.Timestamp,
                          tick_size: float) -> float:
        """
        Calculate center price based on centering method.

        Args:
            group: Bar data
            vwap_series: Pre-calculated rolling VWAP series
            timestamp: Bar timestamp
            tick_size: Price increment

        Returns:
            Center price snapped to tick
        """
        center_price = None

        if self.centering_method == "rolling_vwap" and vwap_series is not None:
            try:
                center_price = vwap_series.asof(timestamp)
            except Exception:
                pass

        elif self.centering_method == "bar_vwap":
            if 'TotalVolume' in group.columns and group['TotalVolume'].sum() > 0:
                pv = (group['Close'] * group['TotalVolume']).sum()
                v = group['TotalVolume'].sum()
                center_price = pv / v

        elif self.centering_method == "bar_mid":
            center_price = (group['Close'].max() + group['Close'].min()) / 2

        # Fallback to mean if center_price is still None or NaN
        if center_price is None or pd.isna(center_price):
            center_price = group['Close'].mean()

        # Snap to nearest tick
        return round(center_price / tick_size) * tick_size

    def _build_bar_grid(self,
                        group: pd.DataFrame,
                        center_price: float,
                        tick_size: float,
                        num_levels: int) -> np.ndarray:
        """
        Build a single bar's price-level grid.

        Args:
            group: Tick data for this bar
            center_price: Center price for the grid
            tick_size: Price increment
            num_levels: Levels above/below center

        Returns:
            2D array [grid_height, num_features]
        """
        grid_height = (num_levels * 2) + 1
        num_features = 5
        bar_grid = np.zeros((grid_height, num_features))

        if group.empty:
            return bar_grid

        group = group.copy()
        group['PriceBin'] = (np.round(group['Close'] / tick_size) * tick_size).round(6)

        # Convert volume columns to float64 to prevent uint32 overflow on subtraction
        for col in ['TotalVolume', 'BidVolume', 'AskVolume']:
            if col in group.columns:
                group[col] = group[col].astype(np.float64)

        profile = group.groupby('PriceBin').agg({
            'TotalVolume': 'sum',
            'BidVolume': 'sum',
            'AskVolume': 'sum',
            'NumTrades': 'sum'
        })
        profile['Delta'] = profile['AskVolume'] - profile['BidVolume']

        offsets = ((profile.index - center_price) / tick_size).astype(int)
        valid_mask = (offsets >= -num_levels) & (offsets <= num_levels)

        if valid_mask.any():
            valid_offsets = offsets[valid_mask]
            valid_data = profile.loc[valid_mask]
            indices = valid_offsets + num_levels

            bar_grid[indices, 0] = valid_data['TotalVolume'].values
            bar_grid[indices, 1] = valid_data['BidVolume'].values
            bar_grid[indices, 2] = valid_data['AskVolume'].values
            bar_grid[indices, 3] = valid_data['Delta'].values
            bar_grid[indices, 4] = valid_data['NumTrades'].values

        return bar_grid

    def calculate_number_bars(
            self,
            df: pd.DataFrame,
            start_time: Optional[Union[str, pd.Timestamp]] = None,
            interval: Optional[Union[str, int]] = None,
            tick_size: Optional[float] = None,
            vwap_window: Optional[str] = None,
            num_levels: Optional[int] = None,
            centering_method: Optional[str] = None,
            normalize: bool = True,
            fixed_center: Optional[float] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate Number Bars tensor from pre-fetched tick data.

        This method processes raw tick data without fetching - use this when you
        already have the data loaded (e.g., in pipelines like MultiFeatureExtraction).

        Args:
            df: DataFrame with tick data. Must have columns:
                - Close: Price
                - TotalVolume: Total volume
                - BidVolume: Volume at bid
                - AskVolume: Volume at ask
                - NumTrades: Number of trades (optional, will use 1 if missing)
                Index must be DatetimeIndex.
            start_time: Start of extraction period (filters df). If None, uses df start.
            interval: Bar interval - str for time-based ('15min', '30min'),
                     int for volume-bucketed (e.g., 5000 contracts per bar).
                     Defaults to self.interval.
            tick_size: Price increment (defaults to self.tick_size)
            vwap_window: VWAP lookback (defaults to self.vwap_window)
            num_levels: Price levels (defaults to self.num_levels)
            centering_method: Override centering method ('rolling_vwap', 'bar_vwap', 'bar_mid')
            normalize: If True (default), output normalized features:
                      - Channel 0: Volume shape (level_vol / bar_total_vol)
                      - Channel 1: Imbalance % ((ask - bid) / level_total_vol)
                      - Channel 2: Bar return (log return, constant across levels)
                      - Channel 3: Price offset ((price - center) / center, for spatial awareness)
                      Magnitude (log1p of total bar volume) stored in metadata.
                      If False, output raw volumes: [TotalVol, BidVol, AskVol, Delta, Trades]
            fixed_center: If provided, use this price as the center for ALL bars instead of
                         calculating per-bar centers. This ensures alignment with volume profile
                         (e.g., use profile VWAP as fixed center). Will be snapped to tick_size.

        Returns:
            tensor: 3D array [N_Bars, 2*num_levels+1, num_features]
                   If normalize=True: 4 features [VolumeShape, Imbalance%, BarReturn, PriceOffset]
                   If normalize=False: 5 features [TotalVol, BidVol, AskVol, Delta, Trades]
            metadata: DataFrame with [Time, CenterPrice, BarVolume, LogMagnitude, BarReturn] per bar
        """
        if df.empty:
            return np.array([]), pd.DataFrame()

        # Use instance defaults if not specified
        interval = interval if interval is not None else self.interval
        tick_size = tick_size if tick_size is not None else self.tick_size
        vwap_window = vwap_window if vwap_window is not None else self.vwap_window
        num_levels = num_levels if num_levels is not None else self.num_levels
        centering_method = centering_method if centering_method is not None else self.centering_method

        # Ensure NumTrades column exists
        raw_df = df.copy()
        if 'NumTrades' not in raw_df.columns:
            raw_df['NumTrades'] = 1

        # Determine start time for filtering
        if start_time is not None:
            start_ts = pd.Timestamp(start_time)
        else:
            start_ts = raw_df.index[0]

        # Normalize timezone - align start_ts with DataFrame index timezone
        if start_ts.tz is None and raw_df.index.tz is not None:
            # start_ts is naive, df is tz-aware: localize start_ts to df's timezone
            start_ts = start_ts.tz_localize(raw_df.index.tz)
        elif start_ts.tz is not None and raw_df.index.tz is None:
            # start_ts is tz-aware, df is naive: remove timezone from start_ts
            start_ts = start_ts.tz_localize(None)
        elif start_ts.tz is not None and raw_df.index.tz is not None and start_ts.tz != raw_df.index.tz:
            # Both have timezones but different: convert start_ts to df's timezone
            start_ts = start_ts.tz_convert(raw_df.index.tz)

        # Handle fixed center alignment (for profile coordination)
        fixed_center_snapped = None
        if fixed_center is not None:
            # Snap fixed center to nearest tick
            fixed_center_snapped = round(fixed_center / tick_size) * tick_size

        # Calculate rolling VWAP for centering (uses all data for lookback)
        vwap_series = None
        if centering_method == "rolling_vwap" and fixed_center is None:
            # Only calculate VWAP if not using fixed center
            vwap_series = self._calculate_rolling_vwap(raw_df, vwap_window)

        # Determine bucketing method
        is_volume_bucketed = isinstance(interval, int)

        tensor_list = []
        meta_list = []
        grid_height = (num_levels * 2) + 1
        raw_num_features = 5  # Raw grid always has 5 features
        out_num_features = 4 if normalize else 5  # 4 features when normalized (added PriceOffset)

        def _normalize_bar_grid(grid: np.ndarray, bar_volume: float, bar_return: float,
                               center_price: float, tick_size: float, num_levels: int) -> np.ndarray:
            """
            Normalize raw bar grid to [VolumeShape, Imbalance%, BarReturn, PriceOffset].

            Args:
                grid: Raw grid [levels, 5] with [TotalVol, BidVol, AskVol, Delta, Trades]
                bar_volume: Total volume in the bar
                bar_return: Log return for the bar (constant across levels)
                center_price: Center price of the grid
                tick_size: Price tick size
                num_levels: Number of levels above/below center

            Returns:
                Normalized grid [levels, 4] with [VolumeShape, Imbalance%, BarReturn, PriceOffset]
            """
            normalized = np.zeros((grid.shape[0], 4), dtype=np.float32)

            # Channel 0: Volume shape (level_vol / bar_total_vol)
            if bar_volume > 0:
                normalized[:, 0] = grid[:, 0] / bar_volume
            else:
                normalized[:, 0] = 0.0

            # Channel 1: Imbalance % at each level ((ask - bid) / level_total_vol)
            level_totals = grid[:, 0]  # TotalVolume per level
            level_deltas = grid[:, 3]  # Delta (AskVol - BidVol) per level
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized[:, 1] = np.where(
                    level_totals > 0,
                    level_deltas / level_totals,
                    0.0
                )

            # Channel 2: Bar return (constant across all levels)
            normalized[:, 2] = bar_return

            # Channel 3: Price offset from center (normalized by center price)
            # This gives the network explicit price information for each bin
            # Example: if center=100, tick=0.25, level 0 is at price 75 (25 ticks below)
            # offset_ticks goes from -num_levels to +num_levels
            offset_ticks = np.arange(-num_levels, num_levels + 1, dtype=np.float32)
            # Normalize: (price - center) / center = (offset_ticks * tick_size) / center
            normalized[:, 3] = (offset_ticks * tick_size) / center_price

            return normalized

        def _calculate_bar_return(group: pd.DataFrame) -> float:
            """Calculate log return for a bar from first to last price."""
            if group.empty or 'Close' not in group.columns:
                return 0.0
            first_price = group['Close'].iloc[0]
            last_price = group['Close'].iloc[-1]
            if first_price > 0 and last_price > 0:
                return float(np.log(last_price / first_price))
            return 0.0

        if is_volume_bucketed:
            # Volume-bucketed bars - filter to requested period
            raw_df_filtered = raw_df[raw_df.index >= start_ts]
            buckets = self._create_volume_buckets(raw_df_filtered, interval)

            for bucket_end, group in buckets:
                # Use fixed center if provided, otherwise calculate per-bar center
                if fixed_center_snapped is not None:
                    center_price = fixed_center_snapped
                else:
                    center_price = self._get_center_price(group, vwap_series, bucket_end, tick_size)
                bar_grid = self._build_bar_grid(group, center_price, tick_size, num_levels)
                bar_volume = float(group['TotalVolume'].sum())
                bar_return = _calculate_bar_return(group)

                if normalize:
                    bar_grid = _normalize_bar_grid(bar_grid, bar_volume, bar_return,
                                                   center_price, tick_size, num_levels)

                tensor_list.append(bar_grid)
                meta_list.append({
                    'Time': bucket_end,
                    'CenterPrice': center_price,
                    'BarVolume': bar_volume,
                    'LogMagnitude': np.log1p(bar_volume),
                    'BarReturn': bar_return
                })
        else:
            # Time-based bars
            grouper = raw_df.groupby(pd.Grouper(freq=interval))

            for timestamp, group in grouper:
                if timestamp < start_ts:
                    continue

                if group.empty:
                    tensor_list.append(np.zeros((grid_height, out_num_features)))
                    meta_list.append({
                        'Time': timestamp,
                        'CenterPrice': fixed_center_snapped if fixed_center_snapped is not None else np.nan,
                        'BarVolume': 0,
                        'LogMagnitude': 0.0,
                        'BarReturn': 0.0
                    })
                    continue

                # Use fixed center if provided, otherwise calculate per-bar center
                if fixed_center_snapped is not None:
                    center_price = fixed_center_snapped
                else:
                    center_price = self._get_center_price(group, vwap_series, timestamp, tick_size)
                bar_grid = self._build_bar_grid(group, center_price, tick_size, num_levels)
                bar_volume = float(group['TotalVolume'].sum())
                bar_return = _calculate_bar_return(group)

                if normalize:
                    bar_grid = _normalize_bar_grid(bar_grid, bar_volume, bar_return,
                                                   center_price, tick_size, num_levels)

                tensor_list.append(bar_grid)
                meta_list.append({
                    'Time': timestamp,
                    'CenterPrice': center_price,
                    'BarVolume': bar_volume,
                    'LogMagnitude': np.log1p(bar_volume),
                    'BarReturn': bar_return
                })

        if not tensor_list:
            return np.array([]), pd.DataFrame()

        final_tensor = np.stack(tensor_list, axis=0).astype(np.float32)
        metadata_df = pd.DataFrame(meta_list)

        return final_tensor, metadata_df

    def get_number_bars(self,
                        start_time: Union[str, pd.Timestamp],
                        end_time: Union[str, pd.Timestamp],
                        interval: Optional[Union[str, int]] = None,
                        tick_size: Optional[float] = None,
                        vwap_window: Optional[str] = None,
                        num_levels: Optional[int] = None,
                        normalize: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract Number Bars tensor by fetching data and processing.

        This is a convenience method that fetches data via get_stitched_data
        and then calls calculate_number_bars. For pipelines where data is
        already loaded, use calculate_number_bars directly.

        Args:
            start_time: Start of extraction period
            end_time: End of extraction period
            interval: Bar interval - str for time-based ('15min', '30min'),
                     int for volume-bucketed (e.g., 5000 contracts per bar).
                     Defaults to self.interval.
            tick_size: Price increment (defaults to self.tick_size)
            vwap_window: VWAP lookback (defaults to self.vwap_window)
            num_levels: Price levels (defaults to self.num_levels)
            normalize: If True (default), output normalized [VolumeShape, Imbalance%]

        Returns:
            tensor: 3D array [N_Bars, 2*num_levels+1, num_features]
            metadata: DataFrame with [Time, CenterPrice, BarVolume, LogMagnitude] per bar
        """
        # Use instance defaults if not specified
        vwap_window = vwap_window if vwap_window is not None else self.vwap_window

        # Fetch data with lookback for VWAP
        lookback_delta = pd.Timedelta(vwap_window) * 2
        start_ts = pd.Timestamp(start_time)
        data_start = start_ts - lookback_delta

        raw_df = self.get_stitched_data(
            start_time=data_start,
            end_time=end_time,
            columns=["Close", "TotalVolume", "BidVolume", "AskVolume", "NumTrades"]
        )

        if raw_df.empty:
            return np.array([]), pd.DataFrame()

        # Delegate to calculate_number_bars
        return self.calculate_number_bars(
            df=raw_df,
            start_time=start_ts,
            normalize=normalize,
            interval=interval,
            tick_size=tick_size,
            vwap_window=vwap_window,
            num_levels=num_levels
        )

    def __getitem__(self, item: slice) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Slice syntax for extracting Number Bars.

        Usage:
            # Default interval (from config)
            tensor, meta = extractor['2024-01-01 08:30':'2024-01-01 15:00']

            # Time-based bars
            tensor, meta = extractor['2024-01-01 08:30':'2024-01-01 15:00':'30min']

            # Volume-bucketed bars
            tensor, meta = extractor['2024-01-01 08:30':'2024-01-01 15:00':5000]

        Args:
            item: slice(start_time, end_time, interval)
                 - interval can be str ('30min') or int (volume bucket size)

        Returns:
            tensor: 3D array [N_Bars, Price_Levels, Features]
            metadata: DataFrame with bar information
        """
        if not isinstance(item, slice):
            raise TypeError("Expected slice object, e.g., extractor['2024-01-01':'2024-01-02']")

        start_str = self._to_time_string(item.start)
        end_str = self._to_time_string(item.stop)

        # Determine interval type
        interval = None
        if item.step is not None:
            if isinstance(item.step, str):
                interval = item.step
            elif isinstance(item.step, int):
                interval = item.step
            elif isinstance(item.step, float):
                # If float, assume it's meant to be an int volume bucket
                interval = int(item.step)

        return self.get_number_bars(start_str, end_str, interval=interval)
