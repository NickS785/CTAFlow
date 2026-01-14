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


import torch
from datetime import datetime, timedelta


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