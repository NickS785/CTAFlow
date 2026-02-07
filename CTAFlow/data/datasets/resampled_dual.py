"""
Dataset for dual-resolution training: raw 5-min + resampled bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


def resample_ohlcv(
    df: pd.DataFrame,
    target_freq: str = "30min",
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Resample OHLCV data to a coarser frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DatetimeIndex and OHLCV columns
    target_freq : str
        Target frequency (e.g., "15min", "30min", "60min", "1H")
    feature_cols : list, optional
        Additional feature columns to resample (uses mean)

    Returns
    -------
    pd.DataFrame
        Resampled dataframe
    """
    agg_dict = {}

    # Standard OHLCV aggregation
    if "Open" in df.columns:
        agg_dict["Open"] = "first"
    if "High" in df.columns:
        agg_dict["High"] = "max"
    if "Low" in df.columns:
        agg_dict["Low"] = "min"
    if "Close" in df.columns:
        agg_dict["Close"] = "last"
    if "Volume" in df.columns:
        agg_dict["Volume"] = "sum"

    # Additional features: use last value (most recent)
    if feature_cols:
        for col in feature_cols:
            if col in df.columns and col not in agg_dict:
                agg_dict[col] = "last"

    # Resample
    resampled = df.resample(target_freq).agg(agg_dict)
    resampled = resampled.dropna(how="all")

    return resampled


class DualResolutionDataset(Dataset):
    """
    Dataset providing both raw 5-min and resampled data for dual-scale models.

    Returns:
        x_short: [short_lookback, n_features] - Raw 5-min bars
        x_long: [long_lookback, n_features] - Resampled bars
        y: int - Class label
        raw_return: float - Raw return for PnL

    Parameters
    ----------
    df : pd.DataFrame
        Raw 5-min bar data with features
    feature_cols : list
        Feature column names
    target_col : str
        Target return column
    class_col : str
        Target class column
    short_lookback : int
        Number of 5-min bars for short branch (default: 128)
    long_lookback : int
        Number of resampled bars for long branch (default: 128)
    resample_freq : str
        Resampling frequency for long branch (default: "30min")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        class_col: str,
        short_lookback: int = 128,
        long_lookback: int = 128,
        resample_freq: str = "30min",
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.class_col = class_col
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
        self.resample_freq = resample_freq

        # Parse resample frequency to get multiplier
        freq_map = {"15min": 3, "30min": 6, "60min": 12, "1H": 12, "2H": 24}
        self.resample_mult = freq_map.get(resample_freq, 6)

        # Required raw bars for long lookback
        self.raw_bars_for_long = long_lookback * self.resample_mult

        # Total required raw lookback
        self.total_raw_lookback = max(short_lookback, self.raw_bars_for_long)

        # Build valid sample indices
        n = len(df)
        ok = np.ones(n, dtype=bool)

        # Must have valid class and return
        ok &= df[class_col].notna().to_numpy()
        ok &= df[target_col].notna().to_numpy()

        # Must have enough history for both branches
        ok[:self.total_raw_lookback - 1] = False

        self.sample_pos = np.flatnonzero(ok)
        if len(self.sample_pos) == 0:
            raise ValueError(
                f"No eligible samples! lookback={self.total_raw_lookback}, "
                f"valid_targets={df[class_col].notna().sum()}"
            )

        # Cache raw feature array
        X_raw = df[feature_cols].ffill().to_numpy(dtype=np.float32)
        self.X = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.Y_class = df[class_col].to_numpy(dtype=np.int64)
        self.Y_return = df[target_col].to_numpy(dtype=np.float32)

        # Pre-compute resampled data
        self._build_resampled_cache()

        print(f"DualResolutionDataset: {len(self.sample_pos):,} samples")
        print(f"  Short: {short_lookback} bars @ 5min")
        print(f"  Long:  {long_lookback} bars @ {resample_freq}")

    def _build_resampled_cache(self):
        """Pre-compute resampled feature array for efficiency."""
        # Resample the dataframe
        df_resampled = resample_ohlcv(
            self.df,
            target_freq=self.resample_freq,
            feature_cols=self.feature_cols,
        )

        # Keep only feature columns that exist after resampling
        valid_cols = [c for c in self.feature_cols if c in df_resampled.columns]
        X_resampled = df_resampled[valid_cols].ffill().to_numpy(dtype=np.float32)
        self.X_long = np.nan_to_num(X_resampled, nan=0.0, posinf=0.0, neginf=0.0)

        # Build index mapping: raw timestamp -> resampled index
        self.resampled_index = df_resampled.index
        self.raw_to_resampled = {}
        for i, ts in enumerate(df_resampled.index):
            self.raw_to_resampled[ts] = i

    def _get_resampled_idx(self, raw_ts: pd.Timestamp) -> int:
        """Find the resampled bar index that contains this raw timestamp."""
        # Floor to resample frequency
        floored = raw_ts.floor(self.resample_freq)
        return self.raw_to_resampled.get(floored, -1)

    def __len__(self) -> int:
        return len(self.sample_pos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
        i = self.sample_pos[idx]

        # Short branch: last short_lookback raw 5-min bars
        j0_short = i - self.short_lookback + 1
        x_short = torch.from_numpy(self.X[j0_short:i + 1].copy())

        # Long branch: find corresponding resampled bars
        raw_ts = self.df.index[i]
        resampled_idx = self._get_resampled_idx(raw_ts)

        if resampled_idx >= self.long_lookback - 1:
            j0_long = resampled_idx - self.long_lookback + 1
            x_long = torch.from_numpy(self.X_long[j0_long:resampled_idx + 1].copy())
        else:
            # Pad with zeros if not enough history
            available = resampled_idx + 1
            pad_len = self.long_lookback - available
            x_avail = self.X_long[:resampled_idx + 1]
            x_padded = np.zeros((self.long_lookback, x_avail.shape[1]), dtype=np.float32)
            x_padded[pad_len:] = x_avail
            x_long = torch.from_numpy(x_padded)

        y_class = int(self.Y_class[i])
        y_return = float(self.Y_return[i])

        return x_short, x_long, y_class, y_return


def dual_collate_fn(batch):
    """Collate function for DualResolutionDataset."""
    x_shorts, x_longs, ys, returns = zip(*batch)
    x_short = torch.stack(x_shorts, dim=0)
    x_long = torch.stack(x_longs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    r = torch.tensor(returns, dtype=torch.float32)
    return x_short, x_long, y, r
