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




class FinancialDataset(Dataset):
    def __init__(self, cl_features_path, vpin_parquet_path, max_len=200,
                 vpin_cols=None, target_data_path=None, target_col=None):
        """
        Dataset for combined summary features and VPIN sequences.

        Parameters
        ----------
        cl_features_path : str
            Path to CSV with summary features (must have 'Datetime' column)
        vpin_parquet_path : str
            Path to VPIN parquet (must have 'date' column for matching)
        max_len : int
            Maximum VPIN sequence length per day (truncates oldest if exceeded)
        vpin_cols : list, optional
            VPIN columns to include in sequence. Default: ['vpin', 'bucket_return', 'log_duration']
        target_data_path : str, optional
            Path to separate target data file (CSV/parquet with 'Datetime' and target_col)
        target_col : str, optional
            Column name for target. Default: None (returns 0)
        """
        # 1. Load Summary Features
        self.df_cl = pd.read_csv(cl_features_path)
        self.df_cl['Datetime'] = pd.to_datetime(self.df_cl['Datetime'])
        self.df_cl['date'] = self.df_cl['Datetime'].dt.date

        # Drop non-feature columns
        exclude_cols = ['Datetime', 'Target', 'date', target_col] if target_col else ['Datetime', 'Target', 'date']
        self.feature_cols = [c for c in self.df_cl.columns if c not in exclude_cols]
        self.features = self.df_cl[self.feature_cols].values.astype(np.float32)

        # Target - from separate file or from cl_features
        self.target_col = target_col
        if target_data_path and target_col:
            # Load from separate file
            if target_data_path.endswith('.parquet'):
                df_target = pd.read_parquet(target_data_path)
            else:
                df_target = pd.read_csv(target_data_path)
            df_target['Datetime'] = pd.to_datetime(df_target['Datetime'])
            df_target['date'] = df_target['Datetime'].dt.date
            # Merge on date
            target_map = dict(zip(df_target['date'], df_target[target_col]))
            self.targets = np.array([target_map.get(d, np.nan) for d in self.df_cl['date']], dtype=np.float32)
        elif target_col and target_col in self.df_cl.columns:
            self.targets = self.df_cl[target_col].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(self.df_cl), dtype=np.float32)

        # 2. Load VPIN Data
        self.df_vpin = pd.read_parquet(vpin_parquet_path)

        # VPIN columns to use
        self.vpin_cols = vpin_cols or ['vpin', 'bucket_return', 'log_duration']
        available_cols = [c for c in self.vpin_cols if c in self.df_vpin.columns]
        if not available_cols:
            raise ValueError(f"None of {self.vpin_cols} found in VPIN data. Available: {list(self.df_vpin.columns)}")
        self.vpin_cols = available_cols

        # Group VPIN by date for fast lookup
        self.df_vpin['date'] = pd.to_datetime(self.df_vpin['date']).dt.date
        self.vpin_by_date = {
            date: group[self.vpin_cols].values.astype(np.float32)
            for date, group in self.df_vpin.groupby('date')
        }

        self.max_len = max_len
        self.n_vpin_features = len(self.vpin_cols)

    def __len__(self):
        return len(self.df_cl)

    def __getitem__(self, idx):
        # A. Get Summary Data
        summary_vec = torch.tensor(self.features[idx])

        # B. Get VPIN Sequence by date
        target_date = self.df_cl.iloc[idx]['date']
        vpin_data = self.vpin_by_date.get(target_date)

        if vpin_data is None or len(vpin_data) == 0:
            # No VPIN data for this date - return zeros
            vpin_data = np.zeros((1, self.n_vpin_features), dtype=np.float32)

        # C. Truncate (keep latest if too long)
        if len(vpin_data) > self.max_len:
            vpin_data = vpin_data[-self.max_len:]

        # D. Convert to Tensor - shape: (Seq_Len, n_features)
        vpin_tensor = torch.tensor(vpin_data)
        length = len(vpin_data)

        # E. Target
        target = torch.tensor(self.targets[idx])

        return summary_vec, vpin_tensor, target, length