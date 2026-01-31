import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, Mapping, List


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
        self.df_summary = self.df_summary.reset_index(drop=True)

        # Identify feature columns (exclude date/target/datetime columns)
        exclude_cols = [date_col, 'date', 'Target', target_col] if target_col else [date_col, 'date', 'Target']
        exclude_cols = [c for c in exclude_cols if c is not None]
        candidate_features = self.df_summary.select_dtypes(exclude=['datetime']).columns
        self.feature_cols = [c for c in candidate_features if c not in exclude_cols]
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


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Mapping


class TriModalDataset(Dataset):
    def __init__(self,
                 summary_data: pd.DataFrame,
                 sequential_data: pd.DataFrame,
                 spatial_data: np.ndarray,
                 spatial_dates: np.ndarray,
                 target_data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
                 max_len: int = 200,
                 sequential_cols: Optional[List[str]] = None,
                 target_col: Optional[str] = None,
                 date_col: str = 'Datetime',
                 sequential_date_col: str = 'date'):
        """
        Dataset for combined summary features, sequential (VPIN) data, and spatial (Market Profile) data.
        """
        # ==========================================
        # 1. Process Summary Features
        # ==========================================
        self.df_summary = summary_data.copy()

        # A. Handle Date Column (Extract from index if missing)
        if date_col not in self.df_summary.columns:
            if isinstance(self.df_summary.index, pd.DatetimeIndex):
                self.df_summary[date_col] = self.df_summary.index
            else:
                raise ValueError(f"Date column '{date_col}' not found in summary_data and index is not DatetimeIndex")

        # B. CRITICAL FIX: Drop DatetimeIndex to ensure pure integer indexing for subsequent operations
        self.df_summary = self.df_summary.reset_index(drop=True)

        # C. Standardize Dates
        self.df_summary[date_col] = pd.to_datetime(self.df_summary[date_col])
        self.df_summary['date'] = self.df_summary[date_col].dt.date

        # D. Identify Features
        # Exclude metadata columns
        exclude_cols = [date_col, 'date', 'Target', target_col] if target_col else [date_col, 'date', 'Target']
        exclude_cols = [c for c in exclude_cols if c is not None]

        self.feature_cols = [c for c in self.df_summary.columns if c not in exclude_cols]

        # Safe conversion: Index is now RangeIndex, so .values won't carry over any index baggage
        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)

        # ==========================================
        # 2. Process Target Data
        # ==========================================
        self.target_col = target_col

        if target_data is not None:
            if isinstance(target_data, pd.Series):
                # Align via Date Index if Series
                # If series index is datetime, map to summary dates
                if isinstance(target_data.index, pd.DatetimeIndex):
                    temp_map = target_data.copy()
                    temp_map.index = temp_map.index.date
                    self.targets = np.array([temp_map.get(d, np.nan) for d in self.df_summary['date']],
                                            dtype=np.float32)
                else:
                    self.targets = target_data.values.astype(np.float32)

            elif isinstance(target_data, pd.DataFrame):
                if target_col is None:
                    raise ValueError("target_col must be specified when target_data is a DataFrame")

                # Standardize Target Date
                target_df = target_data.copy()

                # Try to find a date column
                t_date_col = 'Datetime' if 'Datetime' in target_df.columns else date_col
                if t_date_col in target_df.columns:
                    target_df[t_date_col] = pd.to_datetime(target_df[t_date_col])
                    target_df['date'] = target_df[t_date_col].dt.date
                elif isinstance(target_df.index, pd.DatetimeIndex):
                    target_df['date'] = target_df.index.date

                # Create Lookup Map
                if 'date' in target_df.columns:
                    target_map = dict(zip(target_df['date'], target_df[target_col]))
                    self.targets = np.array([target_map.get(d, np.nan) for d in self.df_summary['date']],
                                            dtype=np.float32)
                else:
                    # Assume aligned by index
                    self.targets = target_df[target_col].values.astype(np.float32)
            else:
                self.targets = np.asarray(target_data, dtype=np.float32)

        elif target_col and target_col in self.df_summary.columns:
            self.targets = self.df_summary[target_col].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(self.df_summary), dtype=np.float32)

        # ==========================================
        # 3. Process Sequential Data (VPIN)
        # ==========================================
        self.df_sequential = sequential_data.copy()

        # Handle Sequential Date
        if sequential_date_col not in self.df_sequential.columns:
            if isinstance(self.df_sequential.index, pd.DatetimeIndex):
                self.df_sequential[sequential_date_col] = self.df_sequential.index
            else:
                # Try 'ts_end' fallback common in VPIN files
                if 'ts_end' in self.df_sequential.columns:
                    self.df_sequential[sequential_date_col] = pd.to_datetime(self.df_sequential['ts_end'])
                else:
                    raise ValueError(f"Date column '{sequential_date_col}' not found in sequential_data")

        # Select Columns
        if sequential_cols is None:
            # Exclude dates/metadata to find features
            ignore = [sequential_date_col, 'ts_end', 'ts_start', 'bucket', 'date']
            self.sequential_cols = [c for c in self.df_sequential.select_dtypes(include=[np.number]).columns if
                                    c not in ignore]
        else:
            self.sequential_cols = [c for c in sequential_cols if c in self.df_sequential.columns]

        # Reset Index for safety
        self.df_sequential = self.df_sequential.reset_index(drop=True)

        # Create Lookup Dictionary
        self.df_sequential['date'] = pd.to_datetime(self.df_sequential[sequential_date_col]).dt.date

        self.sequential_by_date = {}
        for d, group in self.df_sequential.groupby('date'):
            # Ensure float32 and replace NaNs/Infs
            arr = group[self.sequential_cols].values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            self.sequential_by_date[d] = arr

        self.max_len = max_len
        self.n_sequential_features = len(self.sequential_cols)

        # ==========================================
        # 4. Process Spatial Data (Profiles)
        # ==========================================
        spatial_dates_dt = pd.to_datetime(spatial_dates).date

        self.spatial_by_date = {
            d: spatial_data[i].astype(np.float32)
            for i, d in enumerate(spatial_dates_dt)
        }

        if len(self.spatial_by_date) > 0:
            self.spatial_shape = next(iter(self.spatial_by_date.values())).shape
        else:
            self.spatial_shape = (1, 128)  # Fallback

        # ==========================================
        # 5. Alignment
        # ==========================================
        self._align_to_common_dates()

    def _align_to_common_dates(self):
        """Ensure summary, sequential, spatial, and target all have exactly the same dates."""
        summary_dates = set(self.df_summary['date'].unique())
        sequential_dates = set(self.sequential_by_date.keys())
        spatial_dates = set(self.spatial_by_date.keys())

        # Intersection
        common_dates = summary_dates & sequential_dates & spatial_dates

        # If Targets contain NaNs (from mapping), remove those dates too
        # Attach target temporarily to check for NaNs
        self.df_summary['__target__'] = self.targets
        valid_target_dates = set(self.df_summary.dropna(subset=['__target__'])['date'])
        common_dates = common_dates & valid_target_dates

        if len(common_dates) == 0:
            raise ValueError("No common dates found across all data sources.")

        common_dates = sorted(common_dates)

        # Filter Summary & Target
        mask = self.df_summary['date'].isin(common_dates)
        self.df_summary = self.df_summary[mask].sort_values('date').reset_index(drop=True)

        # Extract Cleaned Features & Targets
        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)
        self.targets = self.df_summary['__target__'].values.astype(np.float32)

        # Cleanup
        self.df_summary = self.df_summary.drop(columns=['__target__'])

        # Filter Dictionaries
        self.sequential_by_date = {d: self.sequential_by_date[d] for d in common_dates}
        self.spatial_by_date = {d: self.spatial_by_date[d] for d in common_dates}

        print(f"✓ TriModalDataset aligned to {len(common_dates)} common dates")

    @classmethod
    def from_files(cls, summary_path, sequential_path, spatial_path, target_path=None, **kwargs):
        """Helper to load from file paths."""
        # Load Summary
        if summary_path.endswith('.parquet'):
            summary_data = pd.read_parquet(summary_path)
        else:
            summary_data = pd.read_csv(summary_path)

        # Load Sequential
        if sequential_path.endswith('.parquet'):
            sequential_data = pd.read_parquet(sequential_path)
        else:
            sequential_data = pd.read_csv(sequential_path)

        # Load Spatial
        try:
            npz_data = np.load(spatial_path, allow_pickle=True)
            spatial_data = npz_data['profiles']
            spatial_dates = npz_data['dates']
        except Exception as e:
            raise ValueError(f"Failed to load spatial NPZ: {e}")

        # Load Target
        target_data = None
        if target_path:
            if target_path.endswith('.parquet'):
                target_data = pd.read_parquet(target_path)
            else:
                target_data = pd.read_csv(target_path)

        return cls(summary_data, sequential_data, spatial_data, spatial_dates, target_data, **kwargs)

    def __len__(self):
        return len(self.df_summary)

    def __getitem__(self, idx):
        # A. Summary
        summary_vec = torch.tensor(self.features[idx])
        target_date = self.df_summary.iloc[idx]['date']

        # B. Sequential
        seq_data = self.sequential_by_date.get(target_date)
        # Safety check (though alignment handles this)
        if seq_data is None: seq_data = np.zeros((1, self.n_sequential_features), dtype=np.float32)

        if len(seq_data) > self.max_len:
            seq_data = seq_data[-self.max_len:]

        seq_tensor = torch.tensor(seq_data)
        seq_len = len(seq_data)

        # C. Spatial
        spatial_data = self.spatial_by_date.get(target_date)
        if spatial_data is None: spatial_data = np.zeros(self.spatial_shape, dtype=np.float32)
        spatial_tensor = torch.tensor(spatial_data)

        # D. Target
        target = torch.tensor(self.targets[idx])

        return summary_vec, seq_tensor, spatial_tensor, target, seq_len


class QuadModalDataset(Dataset):
    """
    Dataset for summary, sequential, spatial, and number bars modalities.

    Number bars are expected with shape (T_nb, BINS, C_nb) per date.
    """
    def __init__(self,
                 summary_data: pd.DataFrame,
                 sequential_data: pd.DataFrame,
                 spatial_data: np.ndarray,
                 spatial_dates: np.ndarray,
                 nb_data: Union[np.ndarray, Mapping, None],
                 nb_dates: Optional[np.ndarray] = None,
                 target_data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
                 max_len: int = 200,
                 sequential_cols: Optional[List[str]] = None,
                 target_col: Optional[str] = None,
                 date_col: str = 'Datetime',
                 sequential_date_col: str = 'date'):
        # ==========================================
        # 1. Process Summary Features
        # ==========================================
        self.df_summary = summary_data.copy()

        if date_col not in self.df_summary.columns:
            if isinstance(self.df_summary.index, pd.DatetimeIndex):
                self.df_summary[date_col] = self.df_summary.index
            else:
                raise ValueError(f"Date column '{date_col}' not found in summary_data and index is not DatetimeIndex")

        self.df_summary = self.df_summary.reset_index(drop=True)
        self.df_summary[date_col] = pd.to_datetime(self.df_summary[date_col])
        self.df_summary['date'] = self.df_summary[date_col].dt.date

        exclude_cols = [date_col, 'date', 'Target', target_col] if target_col else [date_col, 'date', 'Target']
        exclude_cols = [c for c in exclude_cols if c is not None]

        self.feature_cols = [c for c in self.df_summary.columns if c not in exclude_cols]
        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)

        # ==========================================
        # 2. Process Target Data
        # ==========================================
        self.target_col = target_col

        if target_data is not None:
            if isinstance(target_data, pd.Series):
                if isinstance(target_data.index, pd.DatetimeIndex):
                    temp_map = target_data.copy()
                    temp_map.index = temp_map.index.date
                    self.targets = np.array([temp_map.get(d, np.nan) for d in self.df_summary['date']],
                                            dtype=np.float32)
                else:
                    self.targets = target_data.values.astype(np.float32)

            elif isinstance(target_data, pd.DataFrame):
                if target_col is None:
                    raise ValueError("target_col must be specified when target_data is a DataFrame")

                target_df = target_data.copy()
                t_date_col = 'Datetime' if 'Datetime' in target_df.columns else date_col
                if t_date_col in target_df.columns:
                    target_df[t_date_col] = pd.to_datetime(target_df[t_date_col])
                    target_df['date'] = target_df[t_date_col].dt.date
                elif isinstance(target_df.index, pd.DatetimeIndex):
                    target_df['date'] = target_df.index.date

                if 'date' in target_df.columns:
                    target_map = dict(zip(target_df['date'], target_df[target_col]))
                    self.targets = np.array([target_map.get(d, np.nan) for d in self.df_summary['date']],
                                            dtype=np.float32)
                else:
                    self.targets = target_df[target_col].values.astype(np.float32)
            else:
                self.targets = np.asarray(target_data, dtype=np.float32)

        elif target_col and target_col in self.df_summary.columns:
            self.targets = self.df_summary[target_col].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(self.df_summary), dtype=np.float32)

        # ==========================================
        # 3. Process Sequential Data (VPIN)
        # ==========================================
        self.df_sequential = sequential_data.copy()

        if sequential_date_col not in self.df_sequential.columns:
            if isinstance(self.df_sequential.index, pd.DatetimeIndex):
                self.df_sequential[sequential_date_col] = self.df_sequential.index
            else:
                if 'ts_end' in self.df_sequential.columns:
                    self.df_sequential[sequential_date_col] = pd.to_datetime(self.df_sequential['ts_end'])
                else:
                    raise ValueError(f"Date column '{sequential_date_col}' not found in sequential_data")

        if sequential_cols is None:
            ignore = [sequential_date_col, 'ts_end', 'ts_start', 'bucket', 'date']
            self.sequential_cols = [c for c in self.df_sequential.select_dtypes(include=[np.number]).columns if
                                    c not in ignore]
        else:
            self.sequential_cols = [c for c in sequential_cols if c in self.df_sequential.columns]

        self.df_sequential = self.df_sequential.reset_index(drop=True)
        self.df_sequential['date'] = pd.to_datetime(self.df_sequential[sequential_date_col]).dt.date

        self.sequential_by_date = {}
        for d, group in self.df_sequential.groupby('date'):
            arr = group[self.sequential_cols].values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            self.sequential_by_date[d] = arr

        self.max_len = max_len
        self.n_sequential_features = len(self.sequential_cols)

        # ==========================================
        # 4. Process Spatial Data (Profiles)
        # ==========================================
        spatial_dates_dt = pd.to_datetime(spatial_dates).date

        self.spatial_by_date = {
            d: spatial_data[i].astype(np.float32)
            for i, d in enumerate(spatial_dates_dt)
        }

        if len(self.spatial_by_date) > 0:
            self.spatial_shape = next(iter(self.spatial_by_date.values())).shape
        else:
            self.spatial_shape = (1, 128)

        # ==========================================
        # 5. Process Number Bars
        # ==========================================
        if nb_data is None:
            raise ValueError("nb_data must be provided for QuadModalDataset.")

        if isinstance(nb_data, Mapping):
            self.nb_by_date = {pd.Timestamp(k).date(): np.asarray(v, dtype=np.float32) for k, v in nb_data.items()}
        else:
            if nb_dates is None:
                raise ValueError("nb_dates must be provided when nb_data is an array.")
            nb_dates_dt = pd.to_datetime(nb_dates).date
            self.nb_by_date = {
                d: np.asarray(nb_data[i], dtype=np.float32)
                for i, d in enumerate(nb_dates_dt)
            }

        if len(self.nb_by_date) > 0:
            self.nb_shape = next(iter(self.nb_by_date.values())).shape
        else:
            self.nb_shape = (1, 1, 1)

        # ==========================================
        # 6. Alignment
        # ==========================================
        self._align_to_common_dates()

    def _align_to_common_dates(self):
        """Ensure summary, sequential, spatial, number bars, and target have the same dates.

        Note: This method automatically handles filtered NumberBars data. If NumberBarCleaner
        removed dates with invalid data (all zeros in required channels), those dates will
        not be in nb_dates and will be excluded from the common date set.
        """
        summary_dates = set(self.df_summary['date'].unique())
        sequential_dates = set(self.sequential_by_date.keys())
        spatial_dates = set(self.spatial_by_date.keys())
        nb_dates = set(self.nb_by_date.keys())

        # Intersect all data sources - dates missing from any modality are excluded
        common_dates = summary_dates & sequential_dates & spatial_dates & nb_dates

        self.df_summary['__target__'] = self.targets
        valid_target_dates = set(self.df_summary.dropna(subset=['__target__'])['date'])
        common_dates = common_dates & valid_target_dates

        if len(common_dates) == 0:
            raise ValueError("No common dates found across all data sources.")

        common_dates = sorted(common_dates)

        mask = self.df_summary['date'].isin(common_dates)
        self.df_summary = self.df_summary[mask].sort_values('date').reset_index(drop=True)

        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)
        self.targets = self.df_summary['__target__'].values.astype(np.float32)

        self.df_summary = self.df_summary.drop(columns=['__target__'])

        self.sequential_by_date = {d: self.sequential_by_date[d] for d in common_dates}
        self.spatial_by_date = {d: self.spatial_by_date[d] for d in common_dates}
        self.nb_by_date = {d: self.nb_by_date[d] for d in common_dates}

        print(f"✓ QuadModalDataset aligned to {len(common_dates)} common dates")

    @classmethod
    def from_files(cls, summary_path, sequential_path, spatial_path, nb_path, target_path=None, **kwargs):
        """Helper to load from file paths."""
        if summary_path.endswith('.parquet'):
            summary_data = pd.read_parquet(summary_path)
        else:
            summary_data = pd.read_csv(summary_path)

        if sequential_path.endswith('.parquet'):
            sequential_data = pd.read_parquet(sequential_path)
        else:
            sequential_data = pd.read_csv(sequential_path)

        try:
            npz_data = np.load(spatial_path, allow_pickle=True)
            spatial_data = npz_data['profiles']
            spatial_dates = npz_data['dates']
        except Exception as e:
            raise ValueError(f"Failed to load spatial NPZ: {e}")

        try:
            nb_npz = np.load(nb_path, allow_pickle=True)
            nb_dates = nb_npz['dates'] if 'dates' in nb_npz else nb_npz['date']
            nb_arrays = nb_npz['arrays'] if 'arrays' in nb_npz else nb_npz['nb']
        except Exception as e:
            raise ValueError(f"Failed to load number bars NPZ: {e}")

        target_data = None
        if target_path:
            if target_path.endswith('.parquet'):
                target_data = pd.read_parquet(target_path)
            else:
                target_data = pd.read_csv(target_path)

        return cls(
            summary_data,
            sequential_data,
            spatial_data,
            spatial_dates,
            nb_arrays,
            nb_dates=nb_dates,
            target_data=target_data,
            **kwargs,
        )

    def __len__(self):
        return len(self.df_summary)

    def __getitem__(self, idx):
        summary_vec = torch.tensor(self.features[idx])
        target_date = self.df_summary.iloc[idx]['date']

        seq_data = self.sequential_by_date.get(target_date)
        if seq_data is None:
            seq_data = np.zeros((1, self.n_sequential_features), dtype=np.float32)

        if len(seq_data) > self.max_len:
            seq_data = seq_data[-self.max_len:]

        seq_tensor = torch.tensor(seq_data)
        seq_len = len(seq_data)

        spatial_data = self.spatial_by_date.get(target_date)
        if spatial_data is None:
            spatial_data = np.zeros(self.spatial_shape, dtype=np.float32)
        spatial_tensor = torch.tensor(spatial_data)

        nb_data = self.nb_by_date.get(target_date)
        if nb_data is None:
            nb_data = np.zeros(self.nb_shape, dtype=np.float32)
        nb_tensor = torch.tensor(nb_data)
        nb_len = nb_tensor.shape[0]

        target = torch.tensor(self.targets[idx])

        return summary_vec, seq_tensor, spatial_tensor, nb_tensor, target, seq_len, nb_len


class RasterizedModalDataset(Dataset):
    """
    Dataset for Summary + Sequential + Profile + Rasterized VPIN data.

    This dataset is designed to work with rasterized VPIN data created by
    SequenceRasterizer.parquet_to_npz(), which converts sequential VPIN
    buckets into spatial grid representations.

    Unlike QuadModalDataset which uses NumberBars, this dataset uses
    rasterized VPIN data with fixed shape (T, C, Bins) per date.

    Parameters
    ----------
    summary_data : pd.DataFrame
        Summary features with Datetime column or DatetimeIndex
    sequential_data : pd.DataFrame
        Sequential VPIN data with date column
    spatial_data : np.ndarray
        Profile arrays with shape (N, C, Bins)
    spatial_dates : np.ndarray
        Dates corresponding to spatial_data
    rasterized_data : Union[dict, str]
        Either:
        - dict mapping date strings to arrays of shape (T, C, Bins)
        - str path to .npz file created by SequenceRasterizer.parquet_to_npz()
    target_data : Optional[Union[pd.DataFrame, pd.Series, np.ndarray]]
        Target values (may be classification labels if binned)
    raw_returns : Optional[Union[pd.Series, np.ndarray]]
        Raw continuous returns before any classification binning.
        Used with profit-weighted losses like ExpectedPnLLoss.
    add_raw_returns : bool
        If True, __getitem__ returns raw_returns as additional output.
        Default False for backward compatibility.
    max_len : int
        Maximum sequence length for sequential data
    sequential_cols : Optional[List[str]]
        Columns to use from sequential data
    target_col : Optional[str]
        Target column name
    date_col : str
        Date column name in summary_data
    sequential_date_col : str
        Date column name in sequential_data
    """

    def __init__(
        self,
        summary_data: pd.DataFrame,
        sequential_data: pd.DataFrame,
        spatial_data: np.ndarray,
        spatial_dates: np.ndarray,
        rasterized_data: Union[Dict, str],
        target_data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        raw_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        add_raw_returns: bool = False,
        max_len: int = 200,
        sequential_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        date_col: str = 'Datetime',
        sequential_date_col: str = 'date'
    ):
        self.add_raw_returns = add_raw_returns
        self._raw_returns_input = raw_returns  # Store for alignment later
        # ==========================================
        # 1. Process Summary Features
        # ==========================================
        self.df_summary = summary_data.copy()

        if date_col not in self.df_summary.columns:
            if isinstance(self.df_summary.index, pd.DatetimeIndex):
                self.df_summary[date_col] = self.df_summary.index
            else:
                raise ValueError(f"Date column '{date_col}' not found")

        self.df_summary = self.df_summary.reset_index(drop=True)
        self.df_summary[date_col] = pd.to_datetime(self.df_summary[date_col])
        self.df_summary['date'] = self.df_summary[date_col].dt.date

        exclude_cols = [date_col, 'date', 'Target', target_col] if target_col else [date_col, 'date', 'Target']
        exclude_cols = [c for c in exclude_cols if c is not None]

        self.feature_cols = [c for c in self.df_summary.columns if c not in exclude_cols]
        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)

        # ==========================================
        # 2. Process Target Data
        # ==========================================
        self.target_col = target_col

        if target_data is not None:
            if isinstance(target_data, pd.Series):
                if isinstance(target_data.index, pd.DatetimeIndex):
                    # Group by date and take last value (handles duplicate timestamps on same day)
                    temp_series = target_data.copy()
                    temp_series.index = temp_series.index.date
                    # If duplicates exist, keep last value per date
                    if temp_series.index.duplicated().any():
                        temp_series = temp_series.groupby(level=0).last()
                    target_map = temp_series.to_dict()
                    self.targets = np.array(
                        [target_map.get(d, np.nan) for d in self.df_summary['date']],
                        dtype=np.float32
                    )
                else:
                    self.targets = target_data.values.astype(np.float32)
            elif isinstance(target_data, pd.DataFrame):
                if target_col is None:
                    raise ValueError("target_col must be specified")
                target_df = target_data.copy()
                t_date_col = 'Datetime' if 'Datetime' in target_df.columns else date_col
                if t_date_col in target_df.columns:
                    target_df[t_date_col] = pd.to_datetime(target_df[t_date_col])
                    target_df['date'] = target_df[t_date_col].dt.date
                elif isinstance(target_df.index, pd.DatetimeIndex):
                    target_df['date'] = target_df.index.date
                if 'date' in target_df.columns:
                    target_map = dict(zip(target_df['date'], target_df[target_col]))
                    self.targets = np.array(
                        [target_map.get(d, np.nan) for d in self.df_summary['date']],
                        dtype=np.float32
                    )
                else:
                    self.targets = target_df[target_col].values.astype(np.float32)
            else:
                self.targets = np.asarray(target_data, dtype=np.float32)
        elif target_col and target_col in self.df_summary.columns:
            self.targets = self.df_summary[target_col].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(self.df_summary), dtype=np.float32)

        # ==========================================
        # 2b. Process Raw Returns (for profit-weighted losses)
        # ==========================================
        if self._raw_returns_input is not None:
            if isinstance(self._raw_returns_input, pd.Series):
                if isinstance(self._raw_returns_input.index, pd.DatetimeIndex):
                    temp_map = self._raw_returns_input.copy()
                    temp_map.index = temp_map.index.date
                    self.raw_returns = np.array(
                        [temp_map.get(d, np.nan) for d in self.df_summary['date']],
                        dtype=np.float32
                    )
                else:
                    self.raw_returns = self._raw_returns_input.values.astype(np.float32)
            else:
                self.raw_returns = np.asarray(self._raw_returns_input, dtype=np.float32)
        else:
            # If no raw_returns provided, use targets as fallback (for regression tasks)
            self.raw_returns = None

        # ==========================================
        # 3. Process Sequential Data (VPIN)
        # ==========================================
        self.df_sequential = sequential_data.copy()

        if sequential_date_col not in self.df_sequential.columns:
            if isinstance(self.df_sequential.index, pd.DatetimeIndex):
                self.df_sequential[sequential_date_col] = self.df_sequential.index
            elif 'ts_end' in self.df_sequential.columns:
                self.df_sequential[sequential_date_col] = pd.to_datetime(self.df_sequential['ts_end'])
            else:
                raise ValueError(f"Date column '{sequential_date_col}' not found")

        if sequential_cols is None:
            ignore = [sequential_date_col, 'ts_end', 'ts_start', 'bucket', 'date']
            self.sequential_cols = [
                c for c in self.df_sequential.select_dtypes(include=[np.number]).columns
                if c not in ignore
            ]
        else:
            self.sequential_cols = [c for c in sequential_cols if c in self.df_sequential.columns]

        self.df_sequential = self.df_sequential.reset_index(drop=True)
        self.df_sequential['date'] = pd.to_datetime(self.df_sequential[sequential_date_col]).dt.date

        self.sequential_by_date = {}
        for d, group in self.df_sequential.groupby('date'):
            arr = group[self.sequential_cols].values.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            self.sequential_by_date[d] = arr

        self.max_len = max_len
        self.n_sequential_features = len(self.sequential_cols)

        # ==========================================
        # 4. Process Spatial Data (Profiles)
        # ==========================================
        spatial_dates_dt = pd.to_datetime(spatial_dates).date

        self.spatial_by_date = {
            d: spatial_data[i].astype(np.float32)
            for i, d in enumerate(spatial_dates_dt)
        }

        if len(self.spatial_by_date) > 0:
            self.spatial_shape = next(iter(self.spatial_by_date.values())).shape
        else:
            self.spatial_shape = (1, 128)

        # ==========================================
        # 5. Process Rasterized VPIN Data
        # ==========================================
        if isinstance(rasterized_data, str):
            # Load from npz file
            from CTAFlow.features.volume.vpin import SequenceRasterizer
            self.rasterized_by_date = SequenceRasterizer.load_npz(rasterized_data, as_tensor=False)
        elif isinstance(rasterized_data, Mapping):
            # Convert keys to date objects for consistency
            self.rasterized_by_date = {}
            for k, v in rasterized_data.items():
                if isinstance(k, str):
                    date_key = pd.to_datetime(k).date()
                else:
                    date_key = pd.to_datetime(k).date()
                self.rasterized_by_date[date_key] = np.asarray(v, dtype=np.float32)
        else:
            raise ValueError("rasterized_data must be dict or path to npz file")

        if len(self.rasterized_by_date) > 0:
            self.rasterized_shape = next(iter(self.rasterized_by_date.values())).shape
        else:
            self.rasterized_shape = (4, 4, 64)  # Default: (T, C, Bins)

        # ==========================================
        # 6. Alignment
        # ==========================================
        self._align_to_common_dates()

    def _align_to_common_dates(self):
        """Ensure all modalities have the same dates."""
        summary_dates = set(self.df_summary['date'].unique())
        sequential_dates = set(self.sequential_by_date.keys())
        spatial_dates = set(self.spatial_by_date.keys())

        # Convert rasterized date keys to match
        rasterized_dates = set()
        for k in self.rasterized_by_date.keys():
            if isinstance(k, str):
                rasterized_dates.add(pd.to_datetime(k).date())
            else:
                rasterized_dates.add(k)

        # Intersect all date sources
        common_dates = summary_dates & sequential_dates & spatial_dates & rasterized_dates

        # Also filter by valid targets
        self.df_summary['__target__'] = self.targets
        if self.raw_returns is not None:
            self.df_summary['__raw_returns__'] = self.raw_returns
        valid_target_dates = set(self.df_summary.dropna(subset=['__target__'])['date'])
        common_dates = common_dates & valid_target_dates

        if len(common_dates) == 0:
            raise ValueError("No common dates found across all data sources.")

        common_dates = sorted(common_dates)

        # Filter summary
        mask = self.df_summary['date'].isin(common_dates)
        self.df_summary = self.df_summary[mask].sort_values('date').reset_index(drop=True)
        self.features = self.df_summary[self.feature_cols].values.astype(np.float32)
        self.targets = self.df_summary['__target__'].values.astype(np.float32)
        if '__raw_returns__' in self.df_summary.columns:
            self.raw_returns = self.df_summary['__raw_returns__'].values.astype(np.float32)
            self.df_summary = self.df_summary.drop(columns=['__target__', '__raw_returns__'])
        else:
            self.df_summary = self.df_summary.drop(columns=['__target__'])

        # Filter other modalities
        self.sequential_by_date = {d: self.sequential_by_date[d] for d in common_dates}
        self.spatial_by_date = {d: self.spatial_by_date[d] for d in common_dates}

        # Rebuild rasterized with proper date keys
        new_rasterized = {}
        for d in common_dates:
            # Try both date object and string keys
            if d in self.rasterized_by_date:
                new_rasterized[d] = self.rasterized_by_date[d]
            elif str(d) in self.rasterized_by_date:
                new_rasterized[d] = self.rasterized_by_date[str(d)]
        self.rasterized_by_date = new_rasterized

    def __len__(self):
        return len(self.df_summary)

    def __getitem__(self, idx):
        target_date = self.df_summary.iloc[idx]['date']

        # Summary features
        summary_vec = torch.tensor(self.features[idx])

        # Sequential data
        seq_data = self.sequential_by_date.get(target_date)
        if seq_data is None:
            seq_data = np.zeros((1, self.n_sequential_features), dtype=np.float32)
        if len(seq_data) > self.max_len:
            seq_data = seq_data[:self.max_len]
        seq_tensor = torch.tensor(seq_data)
        seq_len = len(seq_data)

        # Spatial (profile) data
        spatial_data = self.spatial_by_date.get(target_date)
        if spatial_data is None:
            spatial_data = np.zeros(self.spatial_shape, dtype=np.float32)
        spatial_tensor = torch.tensor(spatial_data)

        # Rasterized VPIN data
        raster_data = self.rasterized_by_date.get(target_date)
        if raster_data is None:
            raster_data = np.zeros(self.rasterized_shape, dtype=np.float32)
        raster_tensor = torch.tensor(raster_data)

        # Target
        target = torch.tensor(self.targets[idx])

        # Optionally return raw returns for profit-weighted losses
        if self.add_raw_returns:
            if self.raw_returns is not None:
                raw_ret = torch.tensor(self.raw_returns[idx])
            else:
                # Fallback to target if no raw_returns provided
                raw_ret = target.clone()
            return summary_vec, seq_tensor, spatial_tensor, raster_tensor, target, seq_len, raw_ret

        return summary_vec, seq_tensor, spatial_tensor, raster_tensor, target, seq_len

# assumes RasterizedModalDataset is already defined above in this file


class TriModalWindowDataset(RasterizedModalDataset):
    """
    Windowed version of RasterizedModalDataset.

    Returns:
      summary_days : (D, F_sum)
      seq_days     : (D, max_len, F_seq)     (RIGHT padded inside each day)
      profile_days : (D, C_prof, B_prof)
      raster_days  : (D, T_nb, C_nb, B_nb)
      target       : scalar (target for last day in the window)
      seq_lens     : (D,) intraday lengths per day
    """

    def __init__(self, *args, window_days: int = 20, return_dates: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        self.window_days = int(window_days)
        self.return_dates = return_dates

    def __len__(self):
        n = len(self.df_summary)
        return max(0, n - self.window_days + 1)

    def __getitem__(self, idx):
        end = idx + self.window_days - 1

        # dates aligned by parent class
        window_dates = self.df_summary.iloc[idx : end + 1]["date"].tolist()

        # A) summary window (already float32 in self.features)
        summary_days = torch.tensor(self.features[idx : end + 1], dtype=torch.float32)  # (D, F_sum)

        # B) seq window: (D, max_len, F_seq) + lens (D,)
        D = self.window_days
        F_seq = self.n_sequential_features
        seq_days = torch.zeros((D, self.max_len, F_seq), dtype=torch.float32)
        seq_lens = torch.zeros((D,), dtype=torch.long)

        for j, d in enumerate(window_dates):
            arr = self.sequential_by_date.get(d)
            if arr is None or len(arr) == 0:
                arr = np.zeros((1, F_seq), dtype=np.float32)

            # keep most recent max_len rows
            if arr.shape[0] > self.max_len:
                arr = arr[-self.max_len :]

            T = arr.shape[0]
            seq_lens[j] = T

            # IMPORTANT: right-pad so pack_padded_sequence works (valid steps first)
            seq_days[j, :T, :] = torch.from_numpy(arr.astype(np.float32))

        # C) profile window
        prof_list = []
        for d in window_dates:
            prof = self.spatial_by_date.get(d)
            if prof is None:
                prof = np.zeros(self.spatial_shape, dtype=np.float32)
            prof_list.append(torch.tensor(prof, dtype=torch.float32))
        profile_days = torch.stack(prof_list, dim=0)  # (D, C_prof, B_prof)

        # D) raster window
        rast_list = []
        for d in window_dates:
            rast = self.rasterized_by_date.get(d)
            if rast is None:
                rast = np.zeros(self.rasterized_shape, dtype=np.float32)
            rast_list.append(torch.tensor(rast, dtype=torch.float32))
        raster_days = torch.stack(rast_list, dim=0)   # (D, T_nb, C_nb, B_nb)

        # E) target for *last day* in window
        target = torch.tensor(self.targets[end])

        # F) optional raw returns for profit-weighted losses
        if self.add_raw_returns:
            if self.raw_returns is not None:
                raw_ret = torch.tensor(self.raw_returns[end])
            else:
                raw_ret = target.clone()
            if self.return_dates:
                return summary_days, seq_days, profile_days, raster_days, target, seq_lens, raw_ret, window_dates
            return summary_days, seq_days, profile_days, raster_days, target, seq_lens, raw_ret

        if self.return_dates:
            return summary_days, seq_days, profile_days, raster_days, target, seq_lens, window_dates

        return summary_days, seq_days, profile_days, raster_days, target, seq_lens


class DualModalWindowDataset(RasterizedModalDataset):
    """
    Dataset for RecurrentDualModal model (Summary + Spatial Fusion).

    Creates rolling windows of:
    1. Summary Features (Macro/Daily stats)
    2. Spatial Data (Profile + Rasterized VPIN)

    Unlike TriModalWindowDataset, this DOES NOT return the variable-length
    intraday sequence data (seq_days), optimizing for the DualModal architecture.

    Returns per sample:
      summary_window : (Window, F_sum)
      profile_window : (Window, C_prof, Bins)
      raster_window  : (Window, T_bars, C_rast, Bins)
      target         : Scalar (for the last day in window)
    """

    def __init__(self, *args, window_days: int = 20, return_dates: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        self.window_days = int(window_days)
        self.return_dates = return_dates

    def __len__(self):
        n = len(self.df_summary)
        return max(0, n - self.window_days + 1)

    def __getitem__(self, idx):
        # Window range: [idx, idx + window_days - 1]
        end = idx + self.window_days - 1

        # 1. Get Dates for the window
        # self.df_summary is already aligned/sorted in parent class
        window_dates = self.df_summary.iloc[idx: end + 1]["date"].tolist()

        # 2. Summary Window
        # (Window, F_sum)
        summary_window = torch.tensor(
            self.features[idx: end + 1],
            dtype=torch.float32
        )

        # 3. Profile Window
        # (Window, C_prof, Bins)
        prof_list = []
        for d in window_dates:
            prof = self.spatial_by_date.get(d)
            if prof is None:
                prof = np.zeros(self.spatial_shape, dtype=np.float32)
            prof_list.append(torch.from_numpy(prof))

        profile_window = torch.stack(prof_list, dim=0)

        # 4. Raster Window
        # (Window, T_bars, C_rast, Bins)
        rast_list = []
        for d in window_dates:
            rast = self.rasterized_by_date.get(d)
            if rast is None:
                rast = np.zeros(self.rasterized_shape, dtype=np.float32)
            rast_list.append(torch.from_numpy(rast))

        raster_window = torch.stack(rast_list, dim=0)

        # 5. Target (for the last day in the window)
        target = torch.tensor(self.targets[end])

        # 6. Optional raw returns for profit-weighted losses
        if self.add_raw_returns:
            if self.raw_returns is not None:
                raw_ret = torch.tensor(self.raw_returns[end])
            else:
                raw_ret = target.clone()
            if self.return_dates:
                return summary_window, profile_window, raster_window, target, raw_ret, window_dates
            return summary_window, profile_window, raster_window, target, raw_ret

        if self.return_dates:
            return summary_window, profile_window, raster_window, target, window_dates

        return summary_window, profile_window, raster_window, target


class OnTheFlyRasterizedDataset(Dataset):
    """
    Dataset that rasterizes VPIN DataFrames on-the-fly during __getitem__.

    This dataset stores raw VPIN DataFrames and rasterizes them when fetched,
    allowing dynamic adjustment of rasterization parameters without pre-processing.

    Uses CupySequenceRasterizer (GPU) if available, otherwise SequenceRasterizer (CPU).

    Parameters
    ----------
    summaries : np.ndarray or torch.Tensor
        Summary features (N, F_sum)
    vpin_dfs : list of pd.DataFrame
        Raw VPIN DataFrames, one per sample. Each must have columns:
        'ts_end', 'close', 'profile_vwap', 'vol', 'imb_frac', 'bucket_return'
    profiles : np.ndarray or torch.Tensor
        Profile features (N, C, Bins)
    targets : np.ndarray or torch.Tensor
        Target values (N,) or (N, num_classes)
    num_bars : int
        Number of time bars to split each VPIN sequence into (default 4)
    n_bins : int
        Number of vertical price bins for rasterization (default 64)
    interval_mins : float, optional
        Minutes per bar. If None, auto-computed from each DataFrame's time range.
    use_gpu : bool
        Use CupySequenceRasterizer if available (default True)
    rasterizer_kwargs : dict, optional
        Additional arguments passed to rasterizer constructor:
        - span_pct: float (default 0.01) - vertical range +/- from VWAP
        - vol_scale: float (default 10.0) - divisor for log volume
        - price_scale: float (default 100.0) - multiplier for price normalization

    Example
    -------
    >>> dataset = OnTheFlyRasterizedDataset(
    ...     summaries=summary_features,
    ...     vpin_dfs=vpin_dataframes,  # list of raw DataFrames
    ...     profiles=profile_arrays,
    ...     targets=target_values,
    ...     num_bars=4,
    ...     n_bins=64
    ... )
    >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_rasterized_vpin)
    """

    def __init__(
        self,
        summaries,
        vpin_dfs: List[pd.DataFrame],
        profiles,
        targets,
        num_bars: int = 4,
        n_bins: int = 64,
        interval_mins: Optional[float] = None,
        use_gpu: bool = True,
        rasterizer_kwargs: Optional[Dict] = None
    ):
        self.summaries = torch.as_tensor(summaries, dtype=torch.float32)
        self.vpin_dfs = vpin_dfs  # List of DataFrames
        self.profiles = torch.as_tensor(profiles, dtype=torch.float32)
        self.targets = torch.as_tensor(targets)

        # Rasterization settings for rasterize() calls
        self.rasterize_kwargs = {
            'num_bars': num_bars,
            'interval_mins': interval_mins
        }

        # Create rasterizer once (reused for all samples)
        rasterizer_kwargs = rasterizer_kwargs or {}
        from CTAFlow.features.volume.vpin import get_rasterizer
        self.rasterizer = get_rasterizer(use_gpu=use_gpu, n_bins=n_bins, **rasterizer_kwargs)

        # Store for shape inference
        self.num_bars = num_bars
        self.n_bins = n_bins

        # Infer rasterized shape from first sample
        if len(vpin_dfs) > 0:
            sample_raster = self.rasterizer.rasterize(vpin_dfs[0], **self.rasterize_kwargs)
            self.rasterized_shape = sample_raster.shape
        else:
            self.rasterized_shape = (num_bars, 4, n_bins)

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        summary = self.summaries[idx]
        profile = self.profiles[idx]
        target = self.targets[idx]

        # Rasterize VPIN DataFrame on-the-fly
        vpin_df = self.vpin_dfs[idx]
        rasterized = self.rasterizer.rasterize(vpin_df, **self.rasterize_kwargs)

        # Return format matches collate_rasterized_vpin expectations
        return (summary, profile, rasterized, target)


class WSPRWindowDataset(RasterizedModalDataset):
    """
    Dataset for RecurrentWSPR / MultiAssetWSPR models.

    Returns windowed summary/profile data but only the MOST RECENT day's
    raster and sequential data (as these models process recent data separately).

    Returns per sample:
      summary_days : (W, F_sum)           - windowed summary features
      profile_days : (W, C_prof, Bins)    - windowed profile data
      raster_recent: (T_bars, C_rast, Bins) - most recent day's raster
      seq_recent   : (max_len, F_seq)     - most recent day's sequential (right-padded)
      target       : scalar               - target for last day in window
      seq_len_recent: scalar              - length of most recent day's sequence
      window_dates : List[date]           - dates in the window (if return_dates=True)

    The meta dict for MultiAssetWSPR is NOT included here; use WSPRCollate
    to add ticker/time metadata during batching.
    """

    def __init__(
        self,
        *args,
        window_days: int = 20,
        return_dates: bool = False,
        ticker_id: int = 0,
        asset_class_id: int = 0,
        asset_subclass_id: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        self.window_days = int(window_days)
        self.return_dates = return_dates

        # Store ticker metadata for collate function
        self.ticker_id = ticker_id
        self.asset_class_id = asset_class_id
        self.asset_subclass_id = asset_subclass_id

    def __len__(self):
        n = len(self.df_summary)
        return max(0, n - self.window_days + 1)

    def __getitem__(self, idx):
        end = idx + self.window_days - 1
        window_dates = self.df_summary.iloc[idx: end + 1]["date"].tolist()

        # 1. Summary Window (W, F_sum)
        summary_days = torch.tensor(
            self.features[idx: end + 1],
            dtype=torch.float32
        )

        # 2. Profile Window (W, C_prof, Bins)
        prof_list = []
        for d in window_dates:
            prof = self.spatial_by_date.get(d)
            if prof is None:
                prof = np.zeros(self.spatial_shape, dtype=np.float32)
            prof_list.append(torch.from_numpy(prof.astype(np.float32)))
        profile_days = torch.stack(prof_list, dim=0)

        # 3. Raster - ONLY most recent day (T_bars, C_rast, Bins)
        recent_date = window_dates[-1]
        rast = self.rasterized_by_date.get(recent_date)
        if rast is None:
            rast = np.zeros(self.rasterized_shape, dtype=np.float32)
        raster_recent = torch.from_numpy(rast.astype(np.float32))

        # 4. Sequential - ONLY most recent day (max_len, F_seq), right-padded
        seq_data = self.sequential_by_date.get(recent_date)
        if seq_data is None or len(seq_data) == 0:
            seq_data = np.zeros((1, self.n_sequential_features), dtype=np.float32)
        if len(seq_data) > self.max_len:
            seq_data = seq_data[-self.max_len:]

        seq_len_recent = len(seq_data)
        seq_recent = torch.zeros((self.max_len, self.n_sequential_features), dtype=torch.float32)
        seq_recent[:seq_len_recent, :] = torch.from_numpy(seq_data.astype(np.float32))

        # 5. Target (for last day in window)
        target = torch.tensor(self.targets[end])

        # 5b. Raw returns (for profit-weighted losses)
        if self.add_raw_returns:
            if self.raw_returns is not None:
                raw_ret = torch.tensor(self.raw_returns[end])
            else:
                raw_ret = target.clone()
        else:
            raw_ret = None

        # 6. Time features for meta (for each day in window)
        # month: (W,), dow: (W,), doy_sin: (W,), doy_cos: (W,)
        months = []
        dows = []
        doy_sins = []
        doy_coss = []
        for d in window_dates:
            dt = pd.Timestamp(d)
            months.append(dt.month)
            dows.append(dt.dayofweek)
            doy = dt.dayofyear
            doy_sin = np.sin(2 * np.pi * doy / 365.0)
            doy_cos = np.cos(2 * np.pi * doy / 365.0)
            doy_sins.append(doy_sin)
            doy_coss.append(doy_cos)

        time_features = {
            'month': torch.tensor(months, dtype=torch.long),
            'dow': torch.tensor(dows, dtype=torch.long),
            'doy_sin': torch.tensor(doy_sins, dtype=torch.float32),
            'doy_cos': torch.tensor(doy_coss, dtype=torch.float32),
        }

        # Identity features (scalars, will be expanded in collate)
        identity = {
            'ticker_id': torch.tensor(self.ticker_id, dtype=torch.long),
            'asset_class_id': torch.tensor(self.asset_class_id, dtype=torch.long),
            'asset_subclass_id': torch.tensor(self.asset_subclass_id, dtype=torch.long),
        }

        # Build return tuple based on options
        base_return = (
            summary_days, profile_days, raster_recent, seq_recent, target,
            torch.tensor(seq_len_recent, dtype=torch.long),
            time_features, identity
        )

        if self.add_raw_returns and self.return_dates:
            return base_return[:5] + (raw_ret,) + base_return[5:] + (window_dates,)
        elif self.add_raw_returns:
            return base_return[:5] + (raw_ret,) + base_return[5:]
        elif self.return_dates:
            return base_return + (window_dates,)
        else:
            return base_return


def wspr_collate_fn(batch):
    """
    Collate function for WSPRWindowDataset that builds the meta dict
    required by MultiAssetWSPR's MetaModalityEncoder.

    Args:
        batch: List of tuples from WSPRWindowDataset.__getitem__

    Returns:
        Tuple of:
        - summary_days: (B, W, F_sum)
        - profile_days: (B, W, C, Bins)
        - raster_recent: (B, T, C, Bins)
        - seq_recent: (B, max_len, F_seq)
        - seq_lens_recent: (B,)
        - targets: (B,)
        - meta: dict with keys:
            - ticker_id: (B,)
            - asset_class_id: (B,)
            - asset_subclass_id: (B,)
            - month: (B, W)
            - dow: (B, W)
            - doy_sin: (B, W)
            - doy_cos: (B, W)
    """
    # Handle both with and without return_dates
    if len(batch[0]) == 9:
        # With dates
        (summaries, profiles, rasters, seqs, targets,
         seq_lens, time_feats, identities, dates_list) = zip(*batch)
        has_dates = True
    else:
        (summaries, profiles, rasters, seqs, targets,
         seq_lens, time_feats, identities) = zip(*batch)
        has_dates = False

    # Stack tensors
    summary_days = torch.stack(summaries, dim=0)      # (B, W, F_sum)
    profile_days = torch.stack(profiles, dim=0)       # (B, W, C, Bins)
    raster_recent = torch.stack(rasters, dim=0)       # (B, T, C, Bins)
    seq_recent = torch.stack(seqs, dim=0)             # (B, max_len, F_seq)
    seq_lens_recent = torch.stack(seq_lens, dim=0)    # (B,)
    targets_tensor = torch.stack(targets, dim=0)      # (B,)

    # Build meta dict
    meta = {
        'ticker_id': torch.stack([id['ticker_id'] for id in identities]),           # (B,)
        'asset_class_id': torch.stack([id['asset_class_id'] for id in identities]), # (B,)
        'asset_subclass_id': torch.stack([id['asset_subclass_id'] for id in identities]),  # (B,)
        'month': torch.stack([tf['month'] for tf in time_feats]),      # (B, W)
        'dow': torch.stack([tf['dow'] for tf in time_feats]),          # (B, W)
        'doy_sin': torch.stack([tf['doy_sin'] for tf in time_feats]),  # (B, W)
        'doy_cos': torch.stack([tf['doy_cos'] for tf in time_feats]),  # (B, W)
    }

    if has_dates:
        return (summary_days, profile_days, raster_recent, seq_recent,
                seq_lens_recent, targets_tensor, meta, dates_list)

    return (summary_days, profile_days, raster_recent, seq_recent,
            seq_lens_recent, targets_tensor, meta)


def _sanity_check_quad_modal():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    summary = pd.DataFrame({"Datetime": dates, "feat": [1.0, 2.0, 3.0]})
    seq_rows = []
    for d in dates:
        for i in range(2):
            seq_rows.append({"date": d, "vpin": i * 0.1, "ret": i * 0.2})
    seq = pd.DataFrame(seq_rows)
    profiles = np.random.rand(3, 3, 8).astype(np.float32)
    nb = {
        d.date(): np.random.rand(2 + i, 5, 3).astype(np.float32)
        for i, d in enumerate(dates)
    }
    targets = pd.Series([0.1, 0.2, 0.3], index=dates)

    dataset = QuadModalDataset(
        summary_data=summary,
        sequential_data=seq,
        spatial_data=profiles,
        spatial_dates=dates,
        nb_data=nb,
        target_data=targets,
        max_len=5,
    )
    sample = dataset[0]
    print("QuadModal sample shapes:", [x.shape for x in sample[:-2]], sample[-2:])


if __name__ == "__main__":
    _sanity_check_quad_modal()
