"""
TFT-Aligned Prep Layer
======================

Converts raw files or DataFrames into TFTAlignedSample objects and
DataLoaders ready for TFTAlignedWSPR / TFTAlignedMamba training.

File Layout
-----------
Each ticker directory should contain:

    <root>/<TICKER>/
        features.csv      # Daily summary features (date index, float columns)
        profiles.npz      # Market profiles: 'profiles' or 'tensor' + 'dates'
        rasterized.npz    # Rasterized VPIN: 'data' or 'rasterized' + 'dates'
        vpin.parquet      # Sequential intraday features (DatetimeIndex)
        target.csv        # Target column (date index)

Usage
-----
    from CTAFlow.models.prep.tft_aligned import TFTAlignedPrepLayer

    prep = TFTAlignedPrepLayer(
        tickers=["GC", "ES", "CL"],
        window_size=10,
    )

    # Load from files
    prep.load_ticker_from_files("GC", root_dir="/path/to/data/GC")

    # Create DataLoader
    train_loader, val_loader = prep.get_loaders(
        val_cutoff_date=date(2023, 1, 1),
        batch_size=32,
    )
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from CTAFlow.data.datasets.tft import (
    COMMODITY_EVENT_TYPES,
    EVENTS_BY_TICKER_TYPE,
    UNIVERSAL_EVENTS,
    EventWindowBuilder,
    ScheduledEvent,
    TFTAlignedDataset,
    TFTAlignedSample,
    build_calendar_features,
    build_ticker_registry,
    events_from_dataframe,
    macro_from_dataframe,
    tft_aligned_collate_fn,
)


# ============================================================================
# File Loading Helpers
# ============================================================================

def _read_tabular(path: Union[str, Path]) -> pd.DataFrame:
    """Read CSV or Parquet with date index."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=True, index_col=0)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def _read_target(path: Union[str, Path], target_col: str = "target") -> pd.Series:
    """Read target from CSV, auto-detecting the column name."""
    df = _read_tabular(path)
    if target_col in df.columns:
        s = df[target_col]
    else:
        for c in ("y", "label", "labels", "ret", "return", "target"):
            if c in df.columns:
                s = df[c]
                break
        else:
            s = df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def _load_npz_arrays(
    path: Union[str, Path],
    array_keys: Sequence[str] = ("tensor", "profiles", "data", "rasterized", "arr_0"),
    date_keys: Sequence[str] = ("dates", "date", "dates_str", "arr_1"),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load array + dates from an NPZ file, trying multiple key names."""
    npz = np.load(str(path), allow_pickle=True)

    arr_key = None
    for k in array_keys:
        if k in npz.files:
            arr_key = k
            break
    if arr_key is None:
        raise ValueError(
            f"Could not find array key in {path}. Available: {npz.files}"
        )
    arr = np.asarray(npz[arr_key], dtype=np.float32)

    date_key = None
    for k in date_keys:
        if k in npz.files:
            date_key = k
            break
    dates = np.asarray(npz[date_key]) if date_key else None

    return arr, dates


def _dates_to_python(dates: np.ndarray) -> List[date]:
    """Convert numpy date array to list of python date objects."""
    out = []
    for d in dates:
        if isinstance(d, (np.datetime64, pd.Timestamp)):
            out.append(pd.Timestamp(d).date())
        elif isinstance(d, str):
            out.append(pd.Timestamp(d).date())
        elif isinstance(d, datetime):
            out.append(d.date())
        elif isinstance(d, date):
            out.append(d)
        else:
            out.append(pd.Timestamp(str(d)).date())
    return out


# ============================================================================
# Prep Layer
# ============================================================================

class TFTAlignedPrepLayer:
    """Converts raw data into TFTAlignedSample objects and DataLoaders.

    Handles:
    1. File loading (CSV, NPZ, Parquet)
    2. Ticker registry construction (static IDs)
    3. Calendar feature computation
    4. Event window building with ticker-type filtering
    5. Macro feature alignment to lookback windows
    6. Train/val splitting and DataLoader creation

    Parameters
    ----------
    tickers : list of str
        All tickers in the training universe.
    window_size : int
        Number of lookback days (W).
    events_by_date : dict, optional
        {date: [ScheduledEvent, ...]} for the full history.
    macro_features : dict, optional
        {date: np.array(f_macro)} daily macro feature vectors.
    f_macro : int
        Number of macro features per day (0 to skip macro).
    max_events_per_day : int
        Maximum events per day slot.
    n_outcome_features : int
        Number of event outcome features.
    custom_ticker_table : dict, optional
        Override the default ticker classification table.
    """

    def __init__(
        self,
        tickers: List[str],
        window_size: int = 10,
        events_by_date: Optional[Dict[date, List[ScheduledEvent]]] = None,
        macro_features: Optional[Dict[date, np.ndarray]] = None,
        f_macro: int = 10,
        max_events_per_day: int = 3,
        n_outcome_features: int = 4,
        surprise_threshold: float = 1.0,
        anticipation_horizon: int = 5,
        custom_ticker_table: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.window_size = window_size
        self.f_macro = f_macro
        self.events_by_date = events_by_date or {}
        self.macro_features = macro_features or {}

        self.ticker_registry = build_ticker_registry(
            tickers, custom_table=custom_ticker_table,
        )

        self._allowed_events: Dict[str, Set[str]] = {}
        for ticker, meta in self.ticker_registry.items():
            tt = meta.ticker_type
            self._allowed_events[ticker] = EVENTS_BY_TICKER_TYPE.get(
                tt, UNIVERSAL_EVENTS,
            )

        self.event_builder = EventWindowBuilder(
            event_registry=COMMODITY_EVENT_TYPES,
            max_events_per_day=max_events_per_day,
            n_outcome_features=n_outcome_features,
            surprise_threshold=surprise_threshold,
            anticipation_horizon=anticipation_horizon,
        )

        # Per-ticker loaded data
        self._summary: Dict[str, Dict[date, np.ndarray]] = {}
        self._profile: Dict[str, Dict[date, np.ndarray]] = {}
        self._raster: Dict[str, Dict[date, np.ndarray]] = {}
        self._seq: Dict[str, Dict[date, np.ndarray]] = {}
        self._seq_lens: Dict[str, Dict[date, int]] = {}
        self._targets: Dict[str, Dict[date, Union[int, float]]] = {}
        self._available_dates: Dict[str, List[date]] = {}

    # ----- Properties -----

    @property
    def n_tickers(self) -> int:
        return len(self.ticker_registry)

    @property
    def n_asset_classes(self) -> int:
        return max(m.asset_class_id for m in self.ticker_registry.values()) + 1

    @property
    def n_asset_subclasses(self) -> int:
        return max(m.asset_subclass_id for m in self.ticker_registry.values()) + 1

    # ----- File Loading -----

    def load_ticker_from_files(
        self,
        ticker: str,
        root_dir: Union[str, Path],
        summary_file: str = "features.csv",
        profile_file: str = "profiles.npz",
        raster_file: str = "rasterized.npz",
        seq_file: str = "vpin.parquet",
        target_file: str = "target.csv",
        target_col: str = "target",
        summary_cols: Optional[List[str]] = None,
    ) -> int:
        """Load all data for a ticker from its directory.

        Parameters
        ----------
        ticker : str
            Ticker symbol (must be in registry).
        root_dir : str or Path
            Directory containing the ticker's files.
        summary_file, profile_file, raster_file, seq_file, target_file : str
            File names within root_dir.
        target_col : str
            Column name for the target in target.csv.
        summary_cols : list of str, optional
            Specific summary columns to use. If None, uses all numeric.

        Returns
        -------
        int
            Number of valid sample dates loaded.
        """
        if ticker not in self.ticker_registry:
            raise ValueError(f"Ticker '{ticker}' not in registry.")

        root = Path(root_dir)

        # 1. Summary features (CSV)
        summary_df = _read_tabular(root / summary_file)
        if summary_cols:
            summary_df = summary_df[[c for c in summary_cols if c in summary_df.columns]]
        summary_df = summary_df.select_dtypes(include="number").astype(np.float32)
        summary_df = summary_df.fillna(0.0)

        summary_dates = [d.date() if isinstance(d, (datetime, pd.Timestamp)) else d
                         for d in summary_df.index]
        summary_dict: Dict[date, np.ndarray] = {}
        for d, row in zip(summary_dates, summary_df.values):
            summary_dict[d] = row

        # 2. Profiles (NPZ)
        profile_arr, profile_dates_raw = _load_npz_arrays(
            root / profile_file,
            array_keys=("tensor", "profiles", "data", "arr_0"),
        )
        if profile_dates_raw is not None:
            profile_dates = _dates_to_python(profile_dates_raw)
        else:
            profile_dates = summary_dates[:len(profile_arr)]

        profile_dict: Dict[date, np.ndarray] = {}
        for d, arr in zip(profile_dates, profile_arr):
            profile_dict[d] = arr

        # 3. Rasterized VPIN (NPZ)
        raster_arr, raster_dates_raw = _load_npz_arrays(
            root / raster_file,
            array_keys=("data", "rasterized", "tensor", "arr_0"),
        )
        if raster_dates_raw is not None:
            raster_dates = _dates_to_python(raster_dates_raw)
        else:
            raster_dates = summary_dates[:len(raster_arr)]

        raster_dict: Dict[date, np.ndarray] = {}
        for d, arr in zip(raster_dates, raster_arr):
            raster_dict[d] = arr

        # 4. Sequential VPIN (Parquet)
        seq_df = _read_tabular(root / seq_file)
        seq_numeric = seq_df.select_dtypes(include="number").astype(np.float32)
        seq_numeric = seq_numeric.fillna(0.0)

        seq_dict: Dict[date, np.ndarray] = {}
        seq_lens_dict: Dict[date, int] = {}
        if isinstance(seq_numeric.index, pd.DatetimeIndex):
            for d, grp in seq_numeric.groupby(seq_numeric.index.date):
                seq_dict[d] = grp.values
                seq_lens_dict[d] = len(grp)
        else:
            # Try grouping by date part of index
            seq_numeric.index = pd.to_datetime(seq_numeric.index)
            for d, grp in seq_numeric.groupby(seq_numeric.index.date):
                seq_dict[d] = grp.values
                seq_lens_dict[d] = len(grp)

        # 5. Targets (CSV)
        target_series = _read_target(root / target_file, target_col=target_col)
        target_dict: Dict[date, Union[int, float]] = {}
        for idx, val in target_series.items():
            d = idx.date() if isinstance(idx, (datetime, pd.Timestamp)) else idx
            target_dict[d] = val

        # Compute available dates (intersection of all modalities)
        all_date_sets = [
            set(summary_dict.keys()),
            set(profile_dict.keys()),
            set(target_dict.keys()),
        ]
        common_dates = sorted(set.intersection(*all_date_sets))

        self._summary[ticker] = summary_dict
        self._profile[ticker] = profile_dict
        self._raster[ticker] = raster_dict
        self._seq[ticker] = seq_dict
        self._seq_lens[ticker] = seq_lens_dict
        self._targets[ticker] = target_dict
        self._available_dates[ticker] = common_dates

        return len(common_dates)

    def load_events_from_dataframe(self, df: pd.DataFrame, **kwargs):
        """Load event calendar from a DataFrame."""
        self.events_by_date = events_from_dataframe(df, **kwargs)

    def load_macro_from_dataframe(self, df: pd.DataFrame, **kwargs):
        """Load macro features from a DataFrame."""
        self.macro_features = macro_from_dataframe(df, **kwargs)
        if self.macro_features:
            sample = next(iter(self.macro_features.values()))
            self.f_macro = len(sample)

    # ----- Macro Window -----

    def get_macro_window(
        self,
        window_dates: Sequence[date],
    ) -> np.ndarray:
        """Extract macro features for the lookback window (forward-filled)."""
        W = len(window_dates)
        out = np.zeros((W, self.f_macro), dtype=np.float32)
        last_valid = np.zeros(self.f_macro, dtype=np.float32)

        for i, d in enumerate(window_dates):
            if isinstance(d, datetime):
                d = d.date()
            if d in self.macro_features:
                last_valid = self.macro_features[d].astype(np.float32)
            out[i] = last_valid

        return out

    # ----- Sample Building -----

    def prepare_sample(
        self,
        ticker: str,
        window_dates: Sequence[date],
        summary_days: np.ndarray,
        profile_days: np.ndarray,
        raster_recent: np.ndarray,
        seq_recent: np.ndarray,
        seq_len: int,
        target: Union[int, float, np.ndarray],
        prediction_date: Optional[date] = None,
    ) -> TFTAlignedSample:
        """Build a single TFTAlignedSample from pre-computed market data."""
        if ticker not in self.ticker_registry:
            raise ValueError(f"Ticker '{ticker}' not in registry.")

        meta = self.ticker_registry[ticker]
        allowed = self._allowed_events[ticker]
        ref_date = prediction_date or (
            window_dates[-1] if isinstance(window_dates[-1], date)
            else window_dates[-1].date()
        )

        cal = build_calendar_features(window_dates)

        evt = self.event_builder.build_window(
            window_dates=window_dates,
            events_by_date=self.events_by_date,
            allowed_events=allowed,
            reference_date=ref_date,
        )

        macro = self.get_macro_window(window_dates)

        return TFTAlignedSample(
            summary_days=summary_days,
            profile_days=profile_days,
            raster_recent=raster_recent,
            seq_recent=seq_recent,
            seq_len=seq_len,
            macro_days=macro,
            ticker_id=meta.ticker_id,
            asset_class_id=meta.asset_class_id,
            asset_subclass_id=meta.asset_subclass_id,
            month=cal["month"],
            dow=cal["dow"],
            doy_sin=cal["doy_sin"],
            doy_cos=cal["doy_cos"],
            event_type_ids=evt["event_type_ids"],
            event_outcomes=evt["event_outcomes"],
            days_until_event=evt["days_until_event"],
            event_mask=evt["event_mask"],
            target=target,
            ticker=ticker,
            prediction_date=ref_date,
        )

    def build_samples(
        self,
        tickers: Optional[List[str]] = None,
    ) -> List[TFTAlignedSample]:
        """Build samples from all loaded data.

        Uses data loaded via ``load_ticker_from_files``.

        Parameters
        ----------
        tickers : list of str, optional
            Subset of tickers to build. If None, uses all loaded tickers.

        Returns
        -------
        list of TFTAlignedSample
        """
        if tickers is None:
            tickers = list(self._available_dates.keys())

        samples: List[TFTAlignedSample] = []
        W = self.window_size

        for ticker in tickers:
            if ticker not in self._available_dates:
                warnings.warn(f"No data loaded for ticker '{ticker}', skipping.")
                continue

            avail_dates = self._available_dates[ticker]
            summary_dict = self._summary[ticker]
            profile_dict = self._profile[ticker]
            raster_dict = self._raster[ticker]
            seq_dict = self._seq[ticker]
            seq_lens_dict = self._seq_lens[ticker]
            target_dict = self._targets[ticker]

            for idx in range(W - 1, len(avail_dates)):
                pred_date = avail_dates[idx]
                window_dates = avail_dates[idx - W + 1: idx + 1]

                if len(window_dates) != W:
                    continue

                # Check all dates have summary + profile data
                if not all(d in summary_dict for d in window_dates):
                    continue
                if not all(d in profile_dict for d in window_dates):
                    continue

                summary_stack = np.stack(
                    [summary_dict[d] for d in window_dates]
                )
                profile_stack = np.stack(
                    [profile_dict[d] for d in window_dates]
                )

                # Raster and seq: use prediction date's data
                raster = raster_dict.get(
                    pred_date,
                    np.zeros((1, 1, 1), dtype=np.float32),
                )
                seq = seq_dict.get(
                    pred_date,
                    np.zeros((1, 1), dtype=np.float32),
                )
                seq_len = seq_lens_dict.get(pred_date, seq.shape[0])

                target = target_dict.get(pred_date)
                if target is None:
                    continue

                sample = self.prepare_sample(
                    ticker=ticker,
                    window_dates=window_dates,
                    summary_days=summary_stack,
                    profile_days=profile_stack,
                    raster_recent=raster,
                    seq_recent=seq,
                    seq_len=seq_len,
                    target=target,
                    prediction_date=pred_date,
                )
                samples.append(sample)

        return samples

    # ----- DataLoader Creation -----

    def get_loaders(
        self,
        val_cutoff_date: Optional[date] = None,
        val_ratio: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 0,
        return_metadata: bool = False,
        tickers: Optional[List[str]] = None,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Build train and validation DataLoaders.

        Uses a date-based split (no lookahead) if val_cutoff_date is
        provided, otherwise splits by ratio from the end.

        Parameters
        ----------
        val_cutoff_date : date, optional
            All samples with prediction_date >= this become validation.
        val_ratio : float
            Fraction for validation if val_cutoff_date is None.
        batch_size : int
            Batch size for both loaders.
        num_workers : int
            DataLoader workers.
        return_metadata : bool
            Whether to include ticker/date metadata in batches.
        tickers : list of str, optional
            Subset of tickers.
        shuffle_train : bool
            Whether to shuffle the training set.

        Returns
        -------
        train_loader, val_loader : DataLoader pair
        """
        all_samples = self.build_samples(tickers=tickers)

        if len(all_samples) == 0:
            raise ValueError("No samples built. Did you call load_ticker_from_files?")

        if val_cutoff_date is not None:
            train_samples = [
                s for s in all_samples
                if s.prediction_date is not None and s.prediction_date < val_cutoff_date
            ]
            val_samples = [
                s for s in all_samples
                if s.prediction_date is not None and s.prediction_date >= val_cutoff_date
            ]
        else:
            # Sort by date, split by ratio
            all_samples.sort(key=lambda s: s.prediction_date or date.min)
            split_idx = int(len(all_samples) * (1 - val_ratio))
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]

        train_ds = TFTAlignedDataset(train_samples, return_metadata=return_metadata)
        val_ds = TFTAlignedDataset(val_samples, return_metadata=return_metadata)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=tft_aligned_collate_fn,
            num_workers=num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=tft_aligned_collate_fn,
            num_workers=num_workers,
        )

        return train_loader, val_loader

    @classmethod
    def from_directories(
        cls,
        root_dir: Union[str, Path],
        tickers: List[str],
        window_size: int = 10,
        f_macro: int = 10,
        events_df: Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
        custom_ticker_table: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ) -> "TFTAlignedPrepLayer":
        """Convenience constructor that loads all tickers from a root directory.

        Expected layout::

            root_dir/
                GC/
                    features.csv
                    profiles.npz
                    rasterized.npz
                    vpin.parquet
                    target.csv
                ES/
                    ...

        Parameters
        ----------
        root_dir : str or Path
            Parent directory containing per-ticker subdirectories.
        tickers : list of str
            Which tickers to load.
        window_size : int
            Lookback window size.
        f_macro : int
            Number of macro features.
        events_df : DataFrame, optional
            Event calendar DataFrame.
        macro_df : DataFrame, optional
            Daily macro features DataFrame.
        custom_ticker_table : dict, optional
            Override default ticker metadata.

        Returns
        -------
        TFTAlignedPrepLayer
            Fully loaded prep layer ready for get_loaders().
        """
        root = Path(root_dir)

        events_by_date = None
        if events_df is not None:
            events_by_date = events_from_dataframe(events_df)

        macro_features = None
        if macro_df is not None:
            macro_features = macro_from_dataframe(macro_df)

        prep = cls(
            tickers=tickers,
            window_size=window_size,
            events_by_date=events_by_date,
            macro_features=macro_features,
            f_macro=f_macro,
            custom_ticker_table=custom_ticker_table,
            **kwargs,
        )

        for ticker in tickers:
            ticker_dir = root / ticker
            if not ticker_dir.exists():
                warnings.warn(f"Directory not found for ticker '{ticker}': {ticker_dir}")
                continue
            n = prep.load_ticker_from_files(ticker, ticker_dir)
            print(f"  [{ticker}] Loaded {n} valid dates")

        return prep

    def get_feature_dims(self) -> Dict[str, int]:
        """Inspect loaded data to determine feature dimensions.

        Returns dict with f_sum, f_profile, f_raster, f_seq, f_macro.
        Useful for model construction.
        """
        dims: Dict[str, int] = {"f_macro": self.f_macro}

        for ticker in self._summary:
            sample_dates = list(self._summary[ticker].keys())
            if sample_dates:
                dims["f_sum"] = self._summary[ticker][sample_dates[0]].shape[-1]
                break

        for ticker in self._profile:
            sample_dates = list(self._profile[ticker].keys())
            if sample_dates:
                arr = self._profile[ticker][sample_dates[0]]
                dims["f_profile"] = arr.shape[0]  # channels
                dims["profile_bins"] = arr.shape[1] if arr.ndim > 1 else 1
                break

        for ticker in self._raster:
            sample_dates = list(self._raster[ticker].keys())
            if sample_dates:
                arr = self._raster[ticker][sample_dates[0]]
                if arr.ndim == 3:
                    dims["f_raster"] = arr.shape[1]  # channels
                elif arr.ndim == 2:
                    dims["f_raster"] = arr.shape[0]
                break

        for ticker in self._seq:
            sample_dates = list(self._seq[ticker].keys())
            if sample_dates:
                arr = self._seq[ticker][sample_dates[0]]
                dims["f_seq"] = arr.shape[-1]
                break

        return dims
