import pandas as pd
import numpy as np
from ..base_extractor import ScidBaseExtractor
from dataclasses import dataclass

@dataclass
class VPINConfig:
    data_dir: str
    ticker: str
    tz : str = ("America/Chicago")
    window_size: int = 60
    bucket_volume: int = 150


class VPINExtractor(ScidBaseExtractor):
    """
    Extracts VPIN (Volume-Synchronized Probability of Informed Trading) metrics.
    """
    def __init__(self, data_dir, ticker, tz="America/Chicago", bucket_volume=150, window=60):
        super().__init__(data_dir, ticker, tz=tz)
        self.bucket_volume = bucket_volume
        self.window = window
        return

    def _calculate_max_run(self, series: pd.Series, window: int) -> pd.Series:
        n = len(series)
        max_runs = np.zeros(n)
        for i in range(n):
            start_idx = max(0, i - window + 1)
            window_data = series.iloc[start_idx:i + 1].values
            if len(window_data) == 0: continue

            # Calculate consecutive runs of 1s
            runs = []
            current_run = 0
            for val in window_data:
                if val == 1:
                    current_run += 1
                else:
                    if current_run > 0: runs.append(current_run)
                    current_run = 0
            if current_run > 0: runs.append(current_run)

            max_runs[i] = max(runs) if runs else 0
        return pd.Series(max_runs, index=series.index)

    def calculate_vpin(self,
                       df: pd.DataFrame,
                       bucket_volume: float,
                       window: int = 50,
                       min_bucket_vol: float = 0.0,
                       include_sequence_features: bool = False) -> pd.DataFrame:
        """
        Applies VPIN logic to a raw DataFrame (Close, BidVolume, AskVolume).

        Parameters
        ----------
        df : pd.DataFrame
            Raw tick data with Close, BidVolume, AskVolume columns
        bucket_volume : float
            Volume per bucket
        window : int, default 50
            Rolling window for VPIN calculation
        min_bucket_vol : float, default 0.0
            Minimum volume threshold per bucket
        include_sequence_features : bool, default False
            If True, include the "Golden Trio" features for neural network training:
            - bucket_return: Log return within bucket (direction signal)
            - log_duration: Log of bucket duration in seconds (urgency/speed signal)
            These features + VPIN form the optimal input for LSTM-based models.

        Returns
        -------
        pd.DataFrame
            VPIN metrics indexed by bucket end timestamp
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Ensure we have signed volume
        if 'BidVolume' not in df.columns or 'AskVolume' not in df.columns:
            raise ValueError("Data must contain 'BidVolume' and 'AskVolume'")

        # Clean - explicitly convert to float64 to avoid uint32 overflow on subtraction
        df = df.copy()
        df['buy_vol'] = pd.to_numeric(df['AskVolume'], errors='coerce').astype(np.float64).fillna(0.0).clip(lower=0.0)
        df['sell_vol'] = pd.to_numeric(df['BidVolume'], errors='coerce').astype(np.float64).fillna(0.0).clip(lower=0.0)
        df['close'] = pd.to_numeric(df['Close'], errors='coerce').astype(np.float64)

        # Reset index to treat Timestamp as a column for grouping
        df = df.reset_index().rename(columns={df.index.name: 'ts'})
        if 'ts' not in df.columns:  # fallback if index didn't have a name
            df.rename(columns={'index': 'ts'}, inplace=True)

        # Preserve timezone information for later restoration
        original_tz = df['ts'].dt.tz if hasattr(df['ts'].dtype, 'tz') else None

        # 1. Volume-clock bucketing with proper boundary splitting
        vol = (df['buy_vol'] + df['sell_vol']).to_numpy(dtype=np.float64)
        if vol.sum() == 0: return pd.DataFrame()

        buy = df['buy_vol'].to_numpy(dtype=np.float64)
        sell = df['sell_vol'].to_numpy(dtype=np.float64)
        ts = df['ts'].values
        close = df['close'].to_numpy(dtype=np.float64)

        # Split ticks across bucket boundaries
        buckets = []
        cum_vol = 0.0
        bucket_id = 0
        bucket_buy, bucket_sell = 0.0, 0.0
        bucket_ts_start, bucket_ts_end = None, None
        bucket_close_first, bucket_close_last = np.nan, np.nan

        for i in range(len(vol)):
            tick_vol = vol[i]
            tick_buy = buy[i]
            tick_sell = sell[i]

            if tick_vol <= 0:
                continue

            # Ratio for splitting buy/sell proportionally
            buy_ratio = tick_buy / tick_vol if tick_vol > 0 else 0.5
            sell_ratio = tick_sell / tick_vol if tick_vol > 0 else 0.5

            remaining_vol = tick_vol
            while remaining_vol > 1e-9:
                # How much volume until next bucket boundary?
                next_boundary = (bucket_id + 1) * bucket_volume
                space_in_bucket = next_boundary - cum_vol

                # Volume to assign to current bucket
                assign_vol = min(remaining_vol, space_in_bucket)
                assign_buy = assign_vol * buy_ratio
                assign_sell = assign_vol * sell_ratio

                # Update bucket accumulators
                bucket_buy += assign_buy
                bucket_sell += assign_sell
                if bucket_ts_start is None:
                    bucket_ts_start = ts[i]
                    bucket_close_first = close[i]
                bucket_ts_end = ts[i]
                bucket_close_last = close[i]

                cum_vol += assign_vol
                remaining_vol -= assign_vol

                # Check if bucket is full
                if cum_vol >= next_boundary - 1e-9:
                    buckets.append({
                        'bucket': bucket_id,
                        'ts_start': bucket_ts_start,
                        'ts_end': bucket_ts_end,
                        'buy': bucket_buy,
                        'sell': bucket_sell,
                        'vol': bucket_buy + bucket_sell,
                        'close_first': bucket_close_first,
                        'close_last': bucket_close_last,
                    })
                    # Reset for next bucket
                    bucket_id += 1
                    bucket_buy, bucket_sell = 0.0, 0.0
                    bucket_ts_start, bucket_ts_end = None, None
                    bucket_close_first, bucket_close_last = np.nan, np.nan

        # Don't include incomplete final bucket (standard VPIN practice)

        if not buckets:
            return pd.DataFrame()

        gb = pd.DataFrame(buckets)

        # Round buy/sell to integers (split logic fractionalizes them)
        gb['buy'] = gb['buy'].round().astype(np.int64)
        gb['sell'] = gb['sell'].round().astype(np.int64)
        gb['vol'] = gb['buy'] + gb['sell']

        if min_bucket_vol > 0:
            gb = gb[gb['vol'] >= min_bucket_vol].reset_index(drop=True)

        # 3. VPIN Calculation - ensure float64 to prevent uint32 overflow
        gb['buy'] = gb['buy'].astype(np.float64)
        gb['sell'] = gb['sell'].astype(np.float64)
        gb['imbalance'] = (gb['buy'] - gb['sell']).abs()
        gb['imb_frac'] = gb['imbalance'] / gb['vol'].replace(0, np.nan)

        # Rolling VPIN sum(|buy-sell|) / sum(vol)
        gb['vpin'] = (
                gb['imbalance'].rolling(window, min_periods=1).sum() /
                gb['vol'].rolling(window, min_periods=1).sum()
        )

        # 4. Sequence Features ("Golden Trio" for Neural Nets)
        if include_sequence_features:
            # Bucket Return: Log return within bucket (direction signal)
            # Raw log returns - standardize downstream if needed
            gb['bucket_return'] = np.log(
                gb['close_last'] / gb['close_first'].replace(0, np.nan)
            ).fillna(0)

            # Duration: Time elapsed in bucket (urgency/speed signal)
            # Log transform to compress range; add small epsilon to avoid log(0)
            duration_sec = (gb['ts_end'] - gb['ts_start']).dt.total_seconds()
            gb['log_duration'] = np.log(duration_sec + 0.001)

            # Signed imbalance for direction
            gb['signed_imbalance'] = (gb['buy'] - gb['sell']) / gb['vol'].replace(0, np.nan)

        # 5. Diagnostics
        gb['buy_dom'] = (gb['buy'] > gb['sell']).astype(int)
        gb['sell_dom'] = (gb['sell'] > gb['buy']).astype(int)
        gb['max_buy_run'] = self._calculate_max_run(gb['buy_dom'], window)
        gb['max_sell_run'] = self._calculate_max_run(gb['sell_dom'], window)
        gb['vol_ratio'] = gb['vol'] / bucket_volume

        # Keep close_last as 'close', drop other intermediate columns
        gb = gb.rename(columns={'close_last': 'close'})
        gb = gb.drop(columns=['close_first', 'ts_start'], errors='ignore')

        # Restore timezone to ts_end before setting as index
        if original_tz is not None:
            # Convert ts_end to datetime with timezone
            gb['ts_end'] = pd.to_datetime(gb['ts_end']).dt.tz_localize('UTC').dt.tz_convert(original_tz)

        return gb.set_index('ts_end')

    def get_vpin(self,
                 start_time: str,
                 end_time: str,
                 bucket_volume= None,
                 window: int = 50) -> pd.DataFrame:
        """
        End-to-end VPIN extraction.

        Args:
            start_time: Start of data fetch
            end_time: End of data fetch
            bucket_volume: Volume per bucket (V)
            window: Rolling window size (n)
        """
        # 1. Fetch Stitched Data
        df_raw = self.get_stitched_data(
            start_time,
            end_time,
            columns=["Close", "BidVolume", "AskVolume", "TotalVolume"]
        )
        if bucket_volume is None:
            bucket_volume = self.bucket_volume

        # 2. Calculate VPIN
        return self.calculate_vpin(df_raw, bucket_volume=bucket_volume, window=window)

    def __getitem__(self, item: slice, window : int = None):
        """
        Slice syntax: extractor['2023-01-01':'2023-01-02':100, 20]
        """
        if not isinstance(item, slice):
            raise TypeError("Expected slice object")

        start_str = self._to_time_string(item.start)
        end_str = self._to_time_string(item.stop)
        bucket_volume = float(item.step) if isinstance(item.step, (int, float)) else self.bucket_volume
        window = self.window if window is None else window

        return self.get_vpin(start_str, end_str, bucket_volume=bucket_volume, window=window)


import numpy as np
import pandas as pd
import torch

# CuPy availability flag for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SequenceRasterizer:
    """
    Converts sequential VPIN data into spatial grid representations.

    This rasterizer maps VPIN buckets onto a 2D grid where:
    - X-axis: Time bars (e.g., 4 x 15-minute intervals)
    - Y-axis: Price bins centered around VWAP

    Output channels: [Density, LogVolume, Imbalance, Returns]

    Can process individual DataFrames or batch process parquet files
    to create date-keyed npz files for training.

    Example:
        >>> rasterizer = SequenceRasterizer(bins=64, span_pct=0.01)
        >>> # Single date
        >>> tensor = rasterizer.rasterize(vpin_df, num_bars=4)
        >>> # Batch process parquet to npz
        >>> rasterizer.parquet_to_npz('vpin.parquet', 'rasterized.npz')
        >>> # Load from npz
        >>> data_dict = rasterizer.load_npz('rasterized.npz')
    """

    def __init__(self, bins=64, span_pct=0.01, vol_scale=10.0, price_scale=100.0):
        """
        Parameters
        ----------
        bins : int
            Number of vertical price levels (default: 64)
        span_pct : float
            Vertical range +/- from VWAP as percentage (0.01 = 1%)
        vol_scale : float
            Divisor for log volume normalization (default: 10.0)
        price_scale : float
            Multiplier for price normalization (default: 100.0)
        """
        self.bins = bins
        self.span_pct = span_pct
        self.vol_scale = vol_scale
        self.price_scale = price_scale

        # Define grid edges (Scaled space)
        # If span is 1% and scale is 100, edges are -1.0 to 1.0
        limit = span_pct * price_scale
        self.edges = np.linspace(-limit, limit, bins + 1, dtype=np.float32)

    def rasterize(self, df, num_bars=4, interval_mins=None, session_start=None):
        """
        Converts VPIN DataFrame -> (Num_Bars, Channels, Bins)
        Channels: [Density, Volume, Imbalance, Returns]

        Parameters
        ----------
        df : pd.DataFrame
            VPIN data with 'ts_end', 'close', 'profile_vwap', 'vol', 'imb_frac', 'bucket_return'.
            If ts_end is the index (common from calculate_vpin), it will be extracted.
        num_bars : int
            Number of time bars to split data into
        interval_mins : float, optional
            Minutes per bar. If None, auto-computed to evenly split data's time range.
        session_start : str or time, optional
            Session start time (e.g., "08:30"). If provided with interval_mins, builds
            fixed time edges from session_start instead of data bounds.

        Returns
        -------
        torch.Tensor
            Shape (num_bars, 4, bins) - channels are [Density, LogVolume, Imbalance, Returns]
        """
        # Handle ts_end as index or column
        df = df.copy()
        if 'ts_end' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['ts_end'] = df.index
            else:
                raise ValueError("DataFrame must have 'ts_end' column or DatetimeIndex")

        # 1. Get time bounds from data - normalize timezone handling
        ts_end = pd.to_datetime(df['ts_end'])

        # Normalize to tz-naive UTC for consistent comparisons
        if ts_end.dt.tz is not None:
            ts_end = ts_end.dt.tz_convert('UTC').dt.tz_localize(None)

        min_ts = ts_end.min()
        max_ts = ts_end.max()

        # Build time edges - use session_start if provided, otherwise data bounds
        if session_start is not None and interval_mins is not None:
            # Parse session_start if string
            if isinstance(session_start, str):
                h, m = map(int, session_start.split(':'))
                session_start_time = pd.Timestamp(min_ts.date()).replace(hour=h, minute=m)
            else:
                session_start_time = pd.Timestamp(min_ts.date()).replace(
                    hour=session_start.hour, minute=session_start.minute
                )
            time_edges = [session_start_time + pd.Timedelta(minutes=i * interval_mins) for i in range(num_bars + 1)]
        else:
            # Auto-compute interval if not provided
            if interval_mins is None:
                total_mins = (max_ts - min_ts).total_seconds() / 60
                interval_mins = total_mins / num_bars if num_bars > 0 else 1.0

            # Build time edges from data's start time
            time_edges = [min_ts + pd.Timedelta(minutes=i * interval_mins) for i in range(num_bars + 1)]

        # 2. Setup Spatial Coordinates
        # 0.0 = VWAP, Scaled by 100
        p_norm = ((df['close'] - df['profile_vwap']) / df['profile_vwap']) * self.price_scale

        # 3. Vectorized Binning - use normalized ts_end for comparison
        t_idx = np.searchsorted(time_edges, ts_end) - 1
        b_idx = np.searchsorted(self.edges, p_norm) - 1

        # Filter valid
        valid = (t_idx >= 0) & (t_idx < num_bars) & (b_idx >= 0) & (b_idx < self.bins)

        # 4. Construct Grid (T, Bins, Channels)
        # We will use 4 Channels: [Count, LogVol, Imbalance, LastReturn]
        grid = np.zeros((num_bars, self.bins, 4), dtype=np.float32)

        if not np.any(valid):
            return torch.tensor(grid).permute(0, 2, 1)  # Return empty (T, C, B)

        t_v, b_v = t_idx[valid], b_idx[valid]
        df_v = df.iloc[valid]

        # Ch 0: Density (Count)
        np.add.at(grid[..., 0], (t_v, b_v), 1.0)

        # Ch 1: Volume (Sum then Log)
        np.add.at(grid[..., 1], (t_v, b_v), df_v['vol'].values)

        # Ch 2: Imbalance (Mean)
        np.add.at(grid[..., 2], (t_v, b_v), df_v['imb_frac'].values)

        # Ch 3: Return (Last) - Using argsort to get time-ordered overwrite
        # Sort by time to ensure 'last' really means last
        sort_idx = np.argsort(ts_end.iloc[valid].values)
        t_s, b_s = t_v[sort_idx], b_v[sort_idx]
        vals_s = df_v['bucket_return'].values[sort_idx] * self.price_scale

        # Advanced indexing assignment overwrites, leaving the last value effective
        grid[t_s, b_s, 3] = vals_s

        # Post-process Means & Logs
        counts = np.maximum(grid[..., 0], 1.0)
        grid[..., 2] /= counts  # Average Imbalance
        grid[..., 1] = np.log1p(grid[..., 1]) / self.vol_scale  # Log Volume

        # Output Shape: (Num_Bars, Bins, Channels) -> Permute to (Num_Bars, Channels, Bins) for CNN
        return torch.tensor(grid, dtype=torch.float32).permute(0, 2, 1)

    def parquet_to_npz(
        self,
        parquet_path: str,
        output_path: str,
        num_bars: int = 4,
        interval_mins: float = None,
        date_col: str = "date",
        ts_col: str = "ts_end",
        verbose: bool = True
    ) -> dict:
        """
        Convert sequential VPIN parquet file to date-keyed npz file.

        Processes all dates in the parquet file and saves rasterized tensors
        as a numpy archive with dates as keys.

        Parameters
        ----------
        parquet_path : str
            Path to input VPIN parquet file. Must contain columns:
            - date or ts_end: For grouping by date
            - close: Price for spatial binning
            - profile_vwap: Reference price for normalization
            - vol: Volume per bucket
            - imb_frac: Imbalance fraction
            - bucket_return: Return per bucket
        output_path : str
            Path for output .npz file
        num_bars : int
            Number of time bars per day (default: 4)
        interval_mins : float, optional
            Minutes per bar. If None, auto-computed per date.
        date_col : str
            Column name for date grouping (default: "date")
        ts_col : str
            Column name for timestamp (default: "ts_end")
        verbose : bool
            Print progress information (default: True)

        Returns
        -------
        dict
            Dictionary with 'dates' (list) and 'shape' (tuple) info
        """
        if verbose:
            print(f"Loading parquet: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        # Ensure ts_col exists as column (may be index from calculate_vpin)
        if ts_col not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if df.columns[0] != ts_col:
                    df = df.rename(columns={df.columns[0]: ts_col})

        # Ensure date column exists
        if date_col not in df.columns:
            if ts_col in df.columns:
                ts_series = pd.to_datetime(df[ts_col])
                # Normalize timezone before extracting date
                if ts_series.dt.tz is not None:
                    ts_series = ts_series.dt.tz_convert('UTC').dt.tz_localize(None)
                df[date_col] = ts_series.dt.date.astype(str)
            else:
                raise ValueError(f"Cannot find date column. Expected '{date_col}' or '{ts_col}'")

        # Ensure ts_end is datetime for rasterization (normalize timezone)
        if ts_col in df.columns:
            ts_series = pd.to_datetime(df[ts_col])
            if ts_series.dt.tz is not None:
                ts_series = ts_series.dt.tz_convert('UTC').dt.tz_localize(None)
            df[ts_col] = ts_series

        # Get unique dates
        dates = sorted(df[date_col].unique())
        if verbose:
            print(f"Found {len(dates)} unique dates")

        # Process each date
        rasterized_data = {}
        skipped = 0

        for i, date in enumerate(dates):
            date_df = df[df[date_col] == date].copy()

            # Check required columns
            required = ['close', 'profile_vwap', 'vol', 'imb_frac', 'bucket_return', ts_col]
            missing = [c for c in required if c not in date_df.columns]
            if missing:
                if verbose and i == 0:
                    print(f"Warning: Missing columns {missing}, skipping dates without required data")
                skipped += 1
                continue

            # Add date column if rasterize expects it
            if 'date' not in date_df.columns:
                date_df['date'] = date

            try:
                tensor = self.rasterize(
                    date_df,
                    num_bars=num_bars,
                    interval_mins=interval_mins
                )
                # Store as numpy array (T, C, Bins)
                rasterized_data[str(date)] = tensor.numpy()
            except Exception as e:
                if verbose:
                    print(f"  Error processing {date}: {e}")
                skipped += 1
                continue

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(dates)} dates...")

        if verbose:
            print(f"Processed {len(rasterized_data)} dates, skipped {skipped}")

        # Save to npz
        np.savez_compressed(output_path, **rasterized_data)

        if verbose:
            sample_shape = next(iter(rasterized_data.values())).shape if rasterized_data else None
            print(f"Saved to: {output_path}")
            print(f"  Shape per date: {sample_shape}")

        return {
            'dates': list(rasterized_data.keys()),
            'shape': sample_shape,
            'num_dates': len(rasterized_data),
            'skipped': skipped
        }

    @staticmethod
    def load_npz(npz_path: str, as_tensor: bool = False) -> dict:
        """
        Load date-keyed rasterized data from npz file.

        Parameters
        ----------
        npz_path : str
            Path to .npz file created by parquet_to_npz
        as_tensor : bool
            If True, convert arrays to torch tensors (default: False)

        Returns
        -------
        dict
            Dictionary mapping date strings to arrays/tensors of shape (T, C, Bins)
        """
        data = np.load(npz_path, allow_pickle=True)

        result = {}
        for key in data.files:
            arr = data[key]
            if as_tensor:
                result[key] = torch.tensor(arr, dtype=torch.float32)
            else:
                result[key] = arr.astype(np.float32)

        return result

    @staticmethod
    def load_npz_with_dates(npz_path: str) -> tuple:
        """
        Load npz file and return arrays with aligned dates.

        Parameters
        ----------
        npz_path : str
            Path to .npz file

        Returns
        -------
        tuple
            (arrays_dict, sorted_dates_list)
        """
        data = SequenceRasterizer.load_npz(npz_path, as_tensor=False)
        sorted_dates = sorted(data.keys())
        return data, sorted_dates


class CupySequenceRasterizer:
    """
    GPU-accelerated version of SequenceRasterizer using CuPy.

    This class provides significant speedups for batch rasterization by
    leveraging GPU parallelism for binning and aggregation operations.

    Falls back to CPU (numpy) if CuPy is not available.

    Example:
        >>> rasterizer = CupySequenceRasterizer(bins=64, span_pct=0.01)
        >>> # Single date (returns torch tensor on CPU)
        >>> tensor = rasterizer.rasterize(vpin_df, num_bars=4)
        >>> # Batch process parquet to npz (GPU accelerated)
        >>> rasterizer.parquet_to_npz('vpin.parquet', 'rasterized.npz')
    """

    def __init__(self, bins=64, span_pct=0.01, vol_scale=10.0, price_scale=100.0):
        """
        Parameters
        ----------
        bins : int
            Number of vertical price levels (default: 64)
        span_pct : float
            Vertical range +/- from VWAP as percentage (0.01 = 1%)
        vol_scale : float
            Divisor for log volume normalization (default: 10.0)
        price_scale : float
            Multiplier for price normalization (default: 100.0)
        """
        self.bins = bins
        self.span_pct = span_pct
        self.vol_scale = vol_scale
        self.price_scale = price_scale
        self.use_gpu = CUPY_AVAILABLE

        # Define grid edges
        limit = span_pct * price_scale
        self.edges_np = np.linspace(-limit, limit, bins + 1, dtype=np.float32)

        if self.use_gpu:
            self.edges_gpu = cp.asarray(self.edges_np)

    def _rasterize_gpu(self, df, time_edges_ns, num_bars):
        """GPU-accelerated rasterization using CuPy."""
        n = len(df)

        # Transfer data to GPU
        p_norm = ((df['close'].values - df['profile_vwap'].values) /
                  df['profile_vwap'].values) * self.price_scale
        ts_end_ns = df['ts_end'].values.astype('datetime64[ns]').astype(np.int64)
        vol = df['vol'].values.astype(np.float32)
        imb = df['imb_frac'].values.astype(np.float32)
        ret = df['bucket_return'].values.astype(np.float32) * self.price_scale

        # Move to GPU
        p_norm_gpu = cp.asarray(p_norm, dtype=cp.float32)
        ts_gpu = cp.asarray(ts_end_ns, dtype=cp.int64)
        vol_gpu = cp.asarray(vol, dtype=cp.float32)
        imb_gpu = cp.asarray(imb, dtype=cp.float32)
        ret_gpu = cp.asarray(ret, dtype=cp.float32)
        time_edges_gpu = cp.asarray(time_edges_ns, dtype=cp.int64)

        # Binning on GPU
        t_idx = cp.searchsorted(time_edges_gpu, ts_gpu) - 1
        b_idx = cp.searchsorted(self.edges_gpu, p_norm_gpu) - 1

        # Valid mask
        valid = (t_idx >= 0) & (t_idx < num_bars) & (b_idx >= 0) & (b_idx < self.bins)

        # Initialize grid
        grid = cp.zeros((num_bars, self.bins, 4), dtype=cp.float32)

        if not cp.any(valid):
            result = cp.asnumpy(grid)
            return torch.tensor(result, dtype=torch.float32).permute(0, 2, 1)

        # Extract valid indices
        t_v = t_idx[valid]
        b_v = b_idx[valid]
        vol_v = vol_gpu[valid]
        imb_v = imb_gpu[valid]
        ret_v = ret_gpu[valid]

        # Compute linear indices for scatter operations
        linear_idx = t_v * self.bins + b_v

        # Channel 0: Density (count) - use bincount
        counts = cp.bincount(linear_idx.astype(cp.int32), minlength=num_bars * self.bins)
        grid[..., 0] = counts.reshape(num_bars, self.bins).astype(cp.float32)

        # Channel 1: Volume sum - use bincount with weights
        vol_sum = cp.bincount(linear_idx.astype(cp.int32), weights=vol_v, minlength=num_bars * self.bins)
        grid[..., 1] = vol_sum.reshape(num_bars, self.bins).astype(cp.float32)

        # Channel 2: Imbalance sum (will divide by count later)
        imb_sum = cp.bincount(linear_idx.astype(cp.int32), weights=imb_v, minlength=num_bars * self.bins)
        grid[..., 2] = imb_sum.reshape(num_bars, self.bins).astype(cp.float32)

        # Channel 3: Last return - need to handle "last" semantics
        # Sort by timestamp to get correct last value
        valid_indices = cp.where(valid)[0]
        ts_valid = ts_gpu[valid]
        sort_order = cp.argsort(ts_valid)

        t_sorted = t_v[sort_order]
        b_sorted = b_v[sort_order]
        ret_sorted = ret_v[sort_order]

        # Create flat indices and use advanced indexing (last write wins)
        t_np = cp.asnumpy(t_sorted).astype(np.int64)
        b_np = cp.asnumpy(b_sorted).astype(np.int64)
        ret_np = cp.asnumpy(ret_sorted)

        # Do the last-value assignment on CPU (more reliable for overwrites)
        grid_ch3 = np.zeros((num_bars, self.bins), dtype=np.float32)
        grid_ch3[t_np, b_np] = ret_np
        grid[..., 3] = cp.asarray(grid_ch3)

        # Post-process: means and logs
        counts_safe = cp.maximum(grid[..., 0], 1.0)
        grid[..., 2] /= counts_safe  # Average imbalance
        grid[..., 1] = cp.log1p(grid[..., 1]) / self.vol_scale  # Log volume

        # Transfer back to CPU
        result = cp.asnumpy(grid)
        return torch.tensor(result, dtype=torch.float32).permute(0, 2, 1)

    def _rasterize_cpu(self, df, time_edges, num_bars):
        """CPU fallback rasterization (same as SequenceRasterizer)."""
        p_norm = ((df['close'] - df['profile_vwap']) / df['profile_vwap']) * self.price_scale

        t_idx = np.searchsorted(time_edges, df['ts_end']) - 1
        b_idx = np.searchsorted(self.edges_np, p_norm) - 1

        valid = (t_idx >= 0) & (t_idx < num_bars) & (b_idx >= 0) & (b_idx < self.bins)

        grid = np.zeros((num_bars, self.bins, 4), dtype=np.float32)

        if not np.any(valid):
            return torch.tensor(grid).permute(0, 2, 1)

        t_v, b_v = t_idx[valid], b_idx[valid]
        df_v = df.iloc[valid]

        np.add.at(grid[..., 0], (t_v, b_v), 1.0)
        np.add.at(grid[..., 1], (t_v, b_v), df_v['vol'].values)
        np.add.at(grid[..., 2], (t_v, b_v), df_v['imb_frac'].values)

        sort_idx = np.argsort(df_v['ts_end'].values)
        t_s, b_s = t_v[sort_idx], b_v[sort_idx]
        vals_s = df_v['bucket_return'].values[sort_idx] * self.price_scale
        grid[t_s, b_s, 3] = vals_s

        counts = np.maximum(grid[..., 0], 1.0)
        grid[..., 2] /= counts
        grid[..., 1] = np.log1p(grid[..., 1]) / self.vol_scale

        return torch.tensor(grid, dtype=torch.float32).permute(0, 2, 1)

    def rasterize(self, df, num_bars=4, interval_mins=None, session_start=None):
        """
        Converts VPIN DataFrame -> (Num_Bars, Channels, Bins)
        Channels: [Density, Volume, Imbalance, Returns]

        Uses GPU if CuPy is available, otherwise falls back to CPU.

        Parameters
        ----------
        df : pd.DataFrame
            VPIN data with 'ts_end', 'close', 'profile_vwap', 'vol', 'imb_frac', 'bucket_return'.
            If ts_end is the index (common from calculate_vpin), it will be extracted.
        num_bars : int
            Number of time bars to split data into
        interval_mins : float, optional
            Minutes per bar. If None, auto-computed to evenly split data's time range.
        session_start : str or time, optional
            Session start time (e.g., "08:30"). If provided with interval_mins, builds
            fixed time edges from session_start instead of data bounds.

        Returns
        -------
        torch.Tensor
            Shape (num_bars, 4, bins) - channels are [Density, LogVolume, Imbalance, Returns]
        """
        # Handle ts_end as index or column
        df = df.copy()
        if 'ts_end' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['ts_end'] = df.index
            else:
                raise ValueError("DataFrame must have 'ts_end' column or DatetimeIndex")

        # Get time bounds from data - normalize timezone handling
        ts_end = pd.to_datetime(df['ts_end'])

        # Normalize to tz-naive UTC for consistent comparisons
        if ts_end.dt.tz is not None:
            ts_end = ts_end.dt.tz_convert('UTC').dt.tz_localize(None)

        # Store normalized timestamps back for GPU/CPU processing
        df['ts_end'] = ts_end

        min_ts = ts_end.min()
        max_ts = ts_end.max()

        # Build time edges - use session_start if provided, otherwise data bounds
        if session_start is not None and interval_mins is not None:
            # Parse session_start if string
            if isinstance(session_start, str):
                h, m = map(int, session_start.split(':'))
                session_start_time = pd.Timestamp(min_ts.date()).replace(hour=h, minute=m)
            else:
                session_start_time = pd.Timestamp(min_ts.date()).replace(
                    hour=session_start.hour, minute=session_start.minute
                )
            time_edges = [session_start_time + pd.Timedelta(minutes=i * interval_mins) for i in range(num_bars + 1)]
        else:
            # Auto-compute interval if not provided
            if interval_mins is None:
                total_mins = (max_ts - min_ts).total_seconds() / 60
                interval_mins = total_mins / num_bars if num_bars > 0 else 1.0

            # Build time edges from data's start time
            time_edges = [min_ts + pd.Timedelta(minutes=i * interval_mins) for i in range(num_bars + 1)]

        if self.use_gpu:
            time_edges_ns = np.array([t.value for t in time_edges], dtype=np.int64)
            return self._rasterize_gpu(df, time_edges_ns, num_bars)
        else:
            return self._rasterize_cpu(df, time_edges, num_bars)

    def rasterize_batch(self, df_list, num_bars=4, interval_mins=None):
        """
        Batch rasterize multiple DataFrames.

        Parameters
        ----------
        df_list : list of pd.DataFrame
            List of VPIN DataFrames, one per date
        num_bars : int
            Number of time bars to split data into
        interval_mins : float, optional
            Minutes per bar. If None, each DataFrame's time range is evenly split.

        Returns
        -------
        np.ndarray
            Stacked array of shape (N, num_bars, channels, bins)
        """
        results = []
        for df in df_list:
            tensor = self.rasterize(df, num_bars=num_bars, interval_mins=interval_mins)
            results.append(tensor.numpy())

        return np.stack(results, axis=0)

    def parquet_to_npz(
        self,
        parquet_path: str,
        output_path: str,
        num_bars: int = 4,
        interval_mins: float = None,
        date_col: str = "date",
        ts_col: str = "ts_end",
        verbose: bool = True,
        batch_size: int = 100
    ) -> dict:
        """
        Convert sequential VPIN parquet file to date-keyed npz file.

        GPU-accelerated batch processing for improved performance.

        Parameters
        ----------
        parquet_path : str
            Path to input VPIN parquet file
        output_path : str
            Path for output .npz file
        num_bars : int
            Number of time bars per day
        interval_mins : float, optional
            Minutes per bar. If None, auto-computed per date.
        date_col : str
            Column name for date grouping
        ts_col : str
            Column name for timestamp
        verbose : bool
            Print progress information
        batch_size : int
            Number of dates to process before syncing GPU (default: 100)

        Returns
        -------
        dict
            Dictionary with processing info
        """
        if verbose:
            backend = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
            print(f"Loading parquet: {parquet_path}")
            print(f"Backend: {backend}")

        df = pd.read_parquet(parquet_path)

        # Ensure ts_col exists as column (may be index from calculate_vpin)
        if ts_col not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if df.columns[0] != ts_col:
                    df = df.rename(columns={df.columns[0]: ts_col})

        if date_col not in df.columns:
            if ts_col in df.columns:
                ts_series = pd.to_datetime(df[ts_col])
                # Normalize timezone before extracting date
                if ts_series.dt.tz is not None:
                    ts_series = ts_series.dt.tz_convert('UTC').dt.tz_localize(None)
                df[date_col] = ts_series.dt.date.astype(str)
            else:
                raise ValueError(f"Cannot find date column '{date_col}' or '{ts_col}'")

        # Ensure ts_end is datetime for rasterization (normalize timezone)
        if ts_col in df.columns:
            ts_series = pd.to_datetime(df[ts_col])
            if ts_series.dt.tz is not None:
                ts_series = ts_series.dt.tz_convert('UTC').dt.tz_localize(None)
            df[ts_col] = ts_series

        dates = sorted(df[date_col].unique())
        if verbose:
            print(f"Found {len(dates)} unique dates")

        rasterized_data = {}
        skipped = 0

        for i, date in enumerate(dates):
            date_df = df[df[date_col] == date].copy()

            required = ['close', 'profile_vwap', 'vol', 'imb_frac', 'bucket_return', ts_col]
            missing = [c for c in required if c not in date_df.columns]
            if missing:
                if verbose and i == 0:
                    print(f"Warning: Missing columns {missing}")
                skipped += 1
                continue

            if 'date' not in date_df.columns:
                date_df['date'] = date

            try:
                tensor = self.rasterize(
                    date_df,
                    num_bars=num_bars,
                    interval_mins=interval_mins
                )
                rasterized_data[str(date)] = tensor.numpy()
            except Exception as e:
                if verbose:
                    print(f"  Error processing {date}: {e}")
                skipped += 1
                continue

            # Sync GPU periodically to prevent memory buildup
            if self.use_gpu and (i + 1) % batch_size == 0:
                cp.get_default_memory_pool().free_all_blocks()

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(dates)} dates...")

        if verbose:
            print(f"Processed {len(rasterized_data)} dates, skipped {skipped}")

        np.savez_compressed(output_path, **rasterized_data)

        if verbose:
            sample_shape = next(iter(rasterized_data.values())).shape if rasterized_data else None
            print(f"Saved to: {output_path}")
            print(f"  Shape per date: {sample_shape}")

        return {
            'dates': list(rasterized_data.keys()),
            'shape': sample_shape,
            'num_dates': len(rasterized_data),
            'skipped': skipped,
            'backend': 'cupy' if self.use_gpu else 'numpy'
        }

    @staticmethod
    def load_npz(npz_path: str, as_tensor: bool = False) -> dict:
        """Load date-keyed rasterized data from npz file."""
        return SequenceRasterizer.load_npz(npz_path, as_tensor=as_tensor)

    @staticmethod
    def load_npz_with_dates(npz_path: str) -> tuple:
        """Load npz file and return arrays with aligned dates."""
        return SequenceRasterizer.load_npz_with_dates(npz_path)


def get_rasterizer(use_gpu: bool = True, n_bins: int = None, **kwargs):
    """
    Factory function to get the appropriate rasterizer.

    Parameters
    ----------
    use_gpu : bool
        If True and CuPy is available, returns CupySequenceRasterizer.
        Otherwise returns SequenceRasterizer.
    n_bins : int, optional
        Number of vertical price levels. Alias for 'bins' kwarg.
        If provided, overrides 'bins' in kwargs.
    **kwargs
        Arguments passed to rasterizer constructor:
        - bins: int (default 64) - number of vertical price levels
        - span_pct: float (default 0.01) - vertical range +/- from VWAP
        - vol_scale: float (default 10.0) - divisor for log volume
        - price_scale: float (default 100.0) - multiplier for price normalization

    Returns
    -------
    SequenceRasterizer or CupySequenceRasterizer
    """
    # Handle n_bins alias
    if n_bins is not None:
        kwargs['bins'] = n_bins

    if use_gpu and CUPY_AVAILABLE:
        return CupySequenceRasterizer(**kwargs)
    return SequenceRasterizer(**kwargs)


class VPINSequenceProcessor:
    def __init__(self,
                 interval_minutes=15,
                 session_start="09:30",
                 session_end="16:15",
                 max_len=64,
                 price_scale=100.0):
        """
        Args:
            interval_minutes (int): Length of each time bar in minutes (e.g., 15, 30, 60).
            session_start (str): HH:MM string for session start (e.g., "09:30").
            session_end (str): HH:MM string for session end (e.g., "16:15").
            max_len (int): Max number of VPIN buckets allowed per interval (truncates if over, pads if under).
            price_scale (float): Multiplier for normalized close.
        """
        self.interval_minutes = interval_minutes
        self.session_start = session_start
        self.session_end = session_end
        self.max_len = max_len
        self.price_scale = price_scale

        # Pre-calculate session boundaries in minutes from midnight
        h_start, m_start = map(int, session_start.split(':'))
        h_end, m_end = map(int, session_end.split(':'))

        self.start_minutes = h_start * 60 + m_start
        self.end_minutes = h_end * 60 + m_end

        # Calculate expected number of intervals per day
        total_duration = self.end_minutes - self.start_minutes
        self.num_intervals = int(np.ceil(total_duration / interval_minutes))

        print(f"Processor Configured: {self.num_intervals} intervals of {interval_minutes}m each per day.")

    def process(self, csv_path):
        """
        Returns:
            tensor: (Num_Days, Num_Intervals, Max_Len, 2)
            dates: List[str]
        """
        # 1. Load Data
        df = pd.read_csv(csv_path)
        df['ts_end'] = pd.to_datetime(df['ts_end'])
        df['date'] = df['ts_end'].dt.date.astype(str)

        # 2. Feature Engineering
        # Normalized Close & Imbalance
        df['norm_close'] = ((df['close'] - df['profile_vwap']) / df['profile_vwap']) * self.price_scale
        df['imb_frac'] = df['imb_frac'].clip(-1.0, 1.0)

        # 3. Calculate Interval IDs
        # Convert timestamp to minutes from midnight
        minutes_from_midnight = df['ts_end'].dt.hour * 60 + df['ts_end'].dt.minute

        # Shift time so session start is 0
        minutes_from_start = minutes_from_midnight - self.start_minutes

        # Calculate Interval ID
        df['interval_id'] = np.floor(minutes_from_start / self.interval_minutes).astype(int)

        # Filter: Keep only buckets within the defined session
        mask_valid = (minutes_from_start >= 0) & (minutes_from_midnight < self.end_minutes)
        df = df[mask_valid].copy()

        # 4. Grouping
        # We need to ensure every day has exactly `self.num_intervals`
        daily_tensors = []
        unique_dates = sorted(df['date'].unique())

        feature_cols = ['norm_close', 'imb_frac']

        # Pre-group by date and interval for speed
        # dict key: (date, interval_id) -> value: numpy array of features
        grouped_data = {
            name: group[feature_cols].values
            for name, group in df.groupby(['date', 'interval_id'])
        }

        for date in unique_dates:
            day_intervals = []

            for i in range(self.num_intervals):
                key = (date, i)

                if key in grouped_data:
                    # Get raw buckets
                    seq = torch.tensor(grouped_data[key], dtype=torch.float32)

                    # Truncate if too long
                    if seq.size(0) > self.max_len:
                        seq = seq[:self.max_len]

                    # Pad if too short
                    pad_size = self.max_len - seq.size(0)
                    if pad_size > 0:
                        padding = torch.zeros((pad_size, 2), dtype=torch.float32)
                        seq = torch.cat([seq, padding], dim=0)

                else:
                    # Missing Interval (No buckets traded?) -> Zero Padding
                    seq = torch.zeros((self.max_len, 2), dtype=torch.float32)

                day_intervals.append(seq)

            # Stack intervals for this day: (Num_Intervals, Max_Len, 2)
            daily_tensor = torch.stack(day_intervals)
            daily_tensors.append(daily_tensor)

        # 5. Final Stack
        if not daily_tensors:
            print("Warning: No valid data found within session limits.")
            return torch.empty(0), []

        # Output: (Num_Days, Num_Intervals, Max_Len, 2)
        output_tensor = torch.stack(daily_tensors)

        print(f"Processing Complete. Output Shape: {output_tensor.shape}")
        return output_tensor, unique_dates