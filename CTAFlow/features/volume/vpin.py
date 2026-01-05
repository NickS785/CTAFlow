import pandas as pd
import numpy as np
from ..base_extractor import ScidBaseExtractor


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
                       min_bucket_vol: float = 0.0) -> pd.DataFrame:
        """
        Applies VPIN logic to a raw DataFrame (Close, BidVolume, AskVolume).
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Ensure we have signed volume
        if 'BidVolume' not in df.columns or 'AskVolume' not in df.columns:
            raise ValueError("Data must contain 'BidVolume' and 'AskVolume'")

        # Clean
        df = df.copy()
        df['buy_vol'] = pd.to_numeric(df['AskVolume'], errors='coerce').fillna(0.0).clip(lower=0.0)
        df['sell_vol'] = pd.to_numeric(df['BidVolume'], errors='coerce').fillna(0.0).clip(lower=0.0)

        # Reset index to treat Timestamp as a column for grouping
        df = df.reset_index().rename(columns={df.index.name: 'ts'})
        if 'ts' not in df.columns:  # fallback if index didn't have a name
            df.rename(columns={'index': 'ts'}, inplace=True)

        # 1. Cumulative Volume Bucketing
        vol = (df['buy_vol'] + df['sell_vol']).to_numpy()
        if vol.sum() == 0: return pd.DataFrame()

        cum = np.cumsum(vol)
        bucket_ids = np.floor((cum - 1e-12) / bucket_volume).astype(int)

        # 2. Aggregate per Bucket
        gb = pd.DataFrame({
            'bucket': bucket_ids,
            'ts': df['ts'],
            'buy': df['buy_vol'],
            'sell': df['sell_vol'],
            'vol': vol
        }).groupby('bucket', as_index=False).agg(
            ts_end=('ts', 'max'),
            buy=('buy', 'sum'),
            sell=('sell', 'sum'),
            vol=('vol', 'sum'),
            n_ticks=('ts', 'count')
        )

        if min_bucket_vol > 0:
            gb = gb[gb['vol'] >= min_bucket_vol].reset_index(drop=True)

        # 3. VPIN Calculation
        gb['imbalance'] = (gb['buy'] - gb['sell']).abs()
        gb['imb_frac'] = gb['imbalance'] / gb['vol'].replace(0, np.nan)

        # Rolling VPIN sum(|buy-sell|) / sum(vol)
        gb['vpin'] = (
                gb['imbalance'].rolling(window, min_periods=1).sum() /
                gb['vol'].rolling(window, min_periods=1).sum()
        )

        # 4. Diagnostics
        gb['buy_dom'] = (gb['buy'] > gb['sell']).astype(int)
        gb['sell_dom'] = (gb['sell'] > gb['buy']).astype(int)
        gb['max_buy_run'] = self._calculate_max_run(gb['buy_dom'], window)
        gb['max_sell_run'] = self._calculate_max_run(gb['sell_dom'], window)
        gb['vol_ratio'] = gb['vol'] / bucket_volume

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
