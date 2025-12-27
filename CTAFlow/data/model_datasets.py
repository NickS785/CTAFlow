import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Dict


class MomentumWindowDataset(Dataset):
    def __init__(self, X_df: pd.DataFrame, y: pd.Series, lookback: int = 20, normalize: bool = True,
                 fit_stats: Optional[Dict[str, np.ndarray]] = None):
        """
        Dataset for windowed time series data.

        Parameters:
        -----------
        X_df : pd.DataFrame
            Feature data
        y : pd.Series
            Target data
        lookback : int
            Number of time steps to look back
        normalize : bool
            Whether to normalize features using z-score normalization
        fit_stats : Optional[Dict[str, np.ndarray]]
            Pre-computed mean and std for normalization (for val/test sets)
            Should contain 'mean' and 'std' keys
        """
        X_df = X_df.sort_index()
        y = y.sort_index()

        # strict alignment
        common = X_df.index.intersection(y.index)
        X_df = X_df.loc[common]
        y = y.loc[common]

        # basic cleanup (choose your own policy)
        X_df = X_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        # Clip extreme values to prevent gradient issues
        X_df = X_df.clip(lower=-1e6, upper=1e6)

        self.X = X_df.to_numpy(dtype=np.float32)
        self.y = y.to_numpy(dtype=np.float32)
        self.lookback = lookback
        self.n_features = self.X.shape[1]

        # Normalize features
        self.normalize = normalize
        if normalize:
            if fit_stats is not None:
                # Use provided statistics (for validation/test sets)
                self.mean = fit_stats['mean']
                self.std = fit_stats['std']
            else:
                # Compute statistics from this dataset (for training set)
                self.mean = np.nanmean(self.X, axis=0, keepdims=True).astype(np.float32)
                self.std = np.nanstd(self.X, axis=0, keepdims=True).astype(np.float32)
                # Avoid division by zero
                self.std = np.where(self.std < 1e-8, 1.0, self.std)

            # Apply normalization
            self.X = (self.X - self.mean) / self.std
            # Handle any remaining NaNs
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return max(0, len(self.y) - self.lookback + 1)

    def __getitem__(self, idx):
        # window ends at idx + lookback - 1
        sl = slice(idx, idx + self.lookback)
        x_win = self.X[sl]                      # (L, C)
        x_win = torch.from_numpy(x_win).T       # (C, L)
        y_t = torch.tensor(self.y[idx + self.lookback - 1])
        return x_win, y_t

    def get_normalization_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Return normalization statistics for use in validation/test sets."""
        if self.normalize:
            return {'mean': self.mean, 'std': self.std}
        return None

def make_window_dataset(X_df, y, lookback=20, batch_size=32):
    ds = MomentumWindowDataset(X_df, y, lookback=lookback)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return ds, dl