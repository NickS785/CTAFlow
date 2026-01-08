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




class DualDataset(Dataset):
    def __init__(self, summary_data, sequential_data, target_data=None, max_len=200,
                 sequential_cols=None, target_col=None, date_col='Datetime', sequential_date_col='date'):
        """
        Dataset for combined summary features and sequential data.

        Parameters
        ----------
        summary_data : pd.DataFrame
            DataFrame with summary features (must have date_col for matching)
        sequential_data : pd.DataFrame
            DataFrame with sequential data (must have sequential_date_col for matching)
        target_data : pd.Series or pd.DataFrame, optional
            Target data. If DataFrame, must specify target_col. If None, returns zeros.
        max_len : int
            Maximum sequence length per day (truncates oldest if exceeded)
        sequential_cols : list, optional
            Columns to include from sequential data. If None, uses all numeric columns.
        target_col : str, optional
            Column name for target if target_data is DataFrame. Default: None
        date_col : str, default 'Datetime'
            Column name for date in summary_data
        sequential_date_col : str, default 'date'
            Column name for date in sequential_data
        """
        # 1. Process Summary Features
        self.df_summary = summary_data.copy()

        # Ensure date column exists and is datetime
        if date_col not in self.df_summary.columns:
            # If index is datetime, use it
            if isinstance(self.df_summary.index, pd.DatetimeIndex):
                self.df_summary[date_col] = self.df_summary.index
            else:
                raise ValueError(f"Date column '{date_col}' not found in summary_data")

        self.df_summary[date_col] = pd.to_datetime(self.df_summary[date_col])
        self.df_summary['date'] = self.df_summary[date_col].dt.date

        # Identify feature columns (exclude date/target columns)
        exclude_cols = [date_col, 'date', 'Target', target_col] if target_col else [date_col, 'date', 'Target']
        exclude_cols = [c for c in exclude_cols if c is not None]
        self.feature_cols = [c for c in self.df_summary.columns if c not in exclude_cols]
        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)

        # 2. Process Target Data
        self.target_col = target_col
        if target_data is not None:
            if isinstance(target_data, pd.Series):
                # Direct series
                self.targets = target_data.values.astype(np.float32)
            elif isinstance(target_data, pd.DataFrame):
                # DataFrame - extract target_col
                if target_col is None:
                    raise ValueError("target_col must be specified when target_data is a DataFrame")
                if target_col not in target_data.columns:
                    raise ValueError(f"target_col '{target_col}' not found in target_data")

                # Align by date
                target_data = target_data.copy()
                if 'Datetime' in target_data.columns:
                    target_data['Datetime'] = pd.to_datetime(target_data['Datetime'])
                    target_data['date'] = target_data['Datetime'].dt.date
                elif isinstance(target_data.index, pd.DatetimeIndex):
                    target_data['date'] = target_data.index.date

                target_map = dict(zip(target_data['date'], target_data[target_col]))
                self.targets = np.array([target_map.get(d, np.nan) for d in self.df_summary['date']], dtype=np.float32)
            else:
                # Assume it's array-like
                self.targets = np.asarray(target_data, dtype=np.float32)
        elif target_col and target_col in self.df_summary.columns:
            # Target in summary data
            self.targets = self.df_summary[target_col].values.astype(np.float32)
        else:
            # No target provided
            self.targets = np.zeros(len(self.df_summary), dtype=np.float32)

        # 3. Process Sequential Data
        self.df_sequential = sequential_data.copy()

        # Ensure date column exists
        if sequential_date_col not in self.df_sequential.columns:
            if isinstance(self.df_sequential.index, pd.DatetimeIndex):
                self.df_sequential[sequential_date_col] = self.df_sequential.index
            else:
                raise ValueError(f"Date column '{sequential_date_col}' not found in sequential_data")

        # Sequential columns to use
        if sequential_cols is None:
            # Use all numeric columns except date
            self.sequential_cols = [c for c in self.df_sequential.select_dtypes(include=[np.number]).columns
                                   if c != sequential_date_col]
        else:
            available_cols = [c for c in sequential_cols if c in self.df_sequential.columns]
            if not available_cols:
                raise ValueError(f"None of {sequential_cols} found in sequential_data. Available: {list(self.df_sequential.columns)}")
            self.sequential_cols = available_cols

        # Group sequential data by date for fast lookup
        self.df_sequential['date'] = pd.to_datetime(self.df_sequential[sequential_date_col]).dt.date
        self.sequential_by_date = {
            date: group[self.sequential_cols].values.astype(np.float32)
            for date, group in self.df_sequential.groupby('date')
        }

        self.max_len = max_len
        self.n_sequential_features = len(self.sequential_cols)

        # CRITICAL: Align all data sources to common dates
        self._align_to_common_dates()

    def _align_to_common_dates(self):
        """Ensure summary, sequential, and target all have exactly the same dates.

        This prevents index misalignment in the DataLoader by finding the intersection
        of all dates and filtering each data source to only include common dates.
        """
        # Get unique dates from each source
        summary_dates = set(self.df_summary['date'].unique())
        sequential_dates = set(self.df_sequential['date'].unique())

        # For targets, we assume they're already aligned with summary at this point
        # (handled during initialization)
        target_dates = summary_dates.copy()

        # Find intersection of all dates
        common_dates = summary_dates & sequential_dates & target_dates

        if len(common_dates) == 0:
            raise ValueError(
                f"No common dates found between data sources.\n"
                f"Summary: {len(summary_dates)} dates, "
                f"Sequential: {len(sequential_dates)} dates, "
                f"Target: {len(target_dates)} dates"
            )

        # Convert to sorted list for consistent ordering
        common_dates = sorted(common_dates)

        # To keep targets aligned during filtering and sorting, add them to df_summary temporarily
        if len(self.targets) != len(self.df_summary):
            raise ValueError(
                f"Target length ({len(self.targets)}) doesn't match summary length "
                f"({len(self.df_summary)}) before alignment. Cannot proceed."
            )

        # Add targets as a temporary column
        self.df_summary['__target__'] = self.targets

        # Filter summary data to common dates and sort
        summary_mask = self.df_summary['date'].isin(common_dates)
        self.df_summary = self.df_summary[summary_mask].sort_values('date').reset_index(drop=True)

        # Extract targets back from sorted/filtered summary
        self.targets = self.df_summary['__target__'].values.astype(np.float32)

        # Remove temporary target column
        self.df_summary = self.df_summary.drop(columns=['__target__'])

        # Filter sequential data to common dates
        sequential_mask = self.df_sequential['date'].isin(common_dates)
        self.df_sequential = self.df_sequential[sequential_mask].sort_values('date')

        # Rebuild sequential_by_date with filtered data
        self.sequential_by_date = {
            date: group[self.sequential_cols].values.astype(np.float32)
            for date, group in self.df_sequential.groupby('date')
        }

        # Rebuild features array to match filtered summary
        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)

        # Verify alignment
        n_summary = len(self.df_summary)
        n_targets = len(self.targets)
        n_sequential_dates = len(self.sequential_by_date)

        if n_summary != n_targets:
            raise ValueError(
                f"After alignment, summary ({n_summary}) and target ({n_targets}) "
                f"lengths don't match!"
            )

        if n_summary != n_sequential_dates:
            raise ValueError(
                f"After alignment, summary has {n_summary} dates but sequential data "
                f"has {n_sequential_dates} dates!"
            )

        # Verify that every date in df_summary has corresponding sequential data
        summary_dates_set = set(self.df_summary['date'].unique())
        sequential_dates_set = set(self.sequential_by_date.keys())
        if summary_dates_set != sequential_dates_set:
            missing_in_seq = summary_dates_set - sequential_dates_set
            missing_in_summary = sequential_dates_set - summary_dates_set
            raise ValueError(
                f"Date mismatch after alignment!\n"
                f"Dates in summary but not in sequential: {missing_in_seq}\n"
                f"Dates in sequential but not in summary: {missing_in_summary}"
            )

        print(f"✓ DualDataset aligned to {len(common_dates)} common dates")
        print(f"  Summary: {n_summary} rows")
        print(f"  Sequential: {len(self.df_sequential)} total rows across {n_sequential_dates} dates")
        print(f"  Targets: {n_targets} values")
        print(f"  Date range: {min(common_dates)} to {max(common_dates)}")

    @classmethod
    def from_files(cls, summary_path, sequential_path, target_path=None, **kwargs):
        """Load DualDataset from file paths (backward compatibility).

        Parameters
        ----------
        summary_path : str
            Path to CSV/Parquet with summary features
        sequential_path : str
            Path to CSV/Parquet with sequential data
        target_path : str, optional
            Path to separate target data file
        **kwargs
            Additional arguments passed to DualDataset.__init__

        Returns
        -------
        DualDataset
            Initialized dataset with loaded data
        """
        # Load summary data
        if summary_path.endswith('.parquet'):
            summary_data = pd.read_parquet(summary_path)
        else:
            summary_data = pd.read_csv(summary_path)

        # Load sequential data
        if sequential_path.endswith('.parquet'):
            sequential_data = pd.read_parquet(sequential_path)
        else:
            sequential_data = pd.read_csv(sequential_path)

        # Load target data if provided
        target_data = None
        if target_path is not None:
            if target_path.endswith('.parquet'):
                target_data = pd.read_parquet(target_path)
            else:
                target_data = pd.read_csv(target_path)

        return cls(summary_data=summary_data, sequential_data=sequential_data,
                   target_data=target_data, **kwargs)

    def __len__(self):
        return len(self.df_summary)

    def __getitem__(self, idx):
        # A. Get Summary Data
        summary_vec = torch.tensor(self.features[idx])

        # B. Get Sequential Data by date
        target_date = self.df_summary.iloc[idx]['date']
        sequential_data = self.sequential_by_date.get(target_date)

        if sequential_data is None or len(sequential_data) == 0:
            # No sequential data for this date - return zeros
            sequential_data = np.zeros((1, self.n_sequential_features), dtype=np.float32)

        # C. Truncate (keep latest if too long)
        if len(sequential_data) > self.max_len:
            sequential_data = sequential_data[-self.max_len:]

        # D. Convert to Tensor - shape: (Seq_Len, n_features)
        sequential_tensor = torch.tensor(sequential_data)
        length = len(sequential_data)

        # E. Target
        target = torch.tensor(self.targets[idx])

        return summary_vec, sequential_tensor, target, length

class TripleDataset(Dataset):

    def __init__(self, summary_feats_df:pd.DataFrame, sequential_df:pd.DataFrame, spatial_data:np.array, seq_max_len=200, sequential_cols=None, summary_cols=None):

        self.spatial_data = spatial_data
        self.summary_df =summary_feats_df
        self.seq_df = sequential_df

        return
