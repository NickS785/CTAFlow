"""Intraday sliding-window dataset for HybridMixtureNetwork.

Provides ``IntradayHybridDataset`` for creating (x_seq, ae_input, y_ret, y_std)
samples from intraday bar DataFrames, with configurable stride for
non-overlapping forecast windows in test/val splits.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IntradayHybridDataset(Dataset):
    """Sliding-window dataset for intraday HybridMixtureNetwork.

    Each sample:
      x_seq    : (seq_len, n_features) -- intraday bar features
      ae_input : (ae_window_bars, f_ae) -- regime features (daily, repeated per bar)
      y_ret    : scalar target return
      y_std    : scalar target vol
      y_class  : (optional) int class label

    Parameters
    ----------
    df : pd.DataFrame
        Intraday bar DataFrame with DatetimeIndex.
    feature_cols : list[str]
        Technical feature column names for x_seq.
    regime_cols : list[str]
        Regime feature column names for ae_input.
    seq_len : int
        Lookback window in bars for the main sequence.
    ae_window_days : int
        Regime encoder lookback in trading days.
    bars_per_day : int
        Number of bars per trading day (ae_window_bars = ae_window_days * bars_per_day).
    n_classes : int
        Number of classification classes (0 = regression only).
    stride : int
        Step size between consecutive valid sample indices.
        Training: stride=1 (maximum overlap, all bars are sample starts).
        Test/val: stride=target_horizon_bars (non-overlapping forecast windows,
        avoids evaluating on bars whose horizons overlap).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        regime_cols: List[str],
        seq_len: int,
        ae_window_days: int,
        bars_per_day: int,
        n_classes: int = 0,
        stride: int = 1,
    ):
        self.seq_len = seq_len
        self.ae_window = ae_window_days * bars_per_day
        self.n_classes = n_classes
        self.stride = max(1, stride)
        warmup = max(self.seq_len, self.ae_window)

        self.X = df[feature_cols].values.astype(np.float32)
        self.R = df[regime_cols].values.astype(np.float32)
        self.y_ret = df["target"].values.astype(np.float32)
        self.y_std = df["target_std"].values.astype(np.float32)
        self.has_classes = "target_class" in df.columns and n_classes > 0
        if self.has_classes:
            self.y_class = df["target_class"].values.astype(np.int64)
        self.dates = df.index

        # Valid indices with stride
        valid = np.arange(warmup, len(df), self.stride)
        ok = np.isfinite(self.y_ret[valid]) & np.isfinite(self.y_std[valid])
        if self.has_classes:
            ok &= np.isfinite(self.y_class[valid].astype(np.float64))
        self.indices = valid[ok]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t = self.indices[idx]
        x_seq = torch.from_numpy(self.X[t - self.seq_len : t])
        ae_input = torch.from_numpy(self.R[t - self.ae_window : t])
        y_ret = torch.tensor(self.y_ret[t])
        y_std = torch.tensor(self.y_std[t])
        if self.has_classes:
            return (
                x_seq,
                ae_input,
                y_ret,
                y_std,
                torch.tensor(self.y_class[t], dtype=torch.long),
            )
        return x_seq, ae_input, y_ret, y_std

    def get_date(self, idx: int) -> pd.Timestamp:
        return self.dates[self.indices[idx]]
