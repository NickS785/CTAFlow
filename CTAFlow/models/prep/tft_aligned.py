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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from CTAFlow.models.multi_asset import (
    SummarySelectionConfig,
    feature_signature,
    _filter_by_substrings,
)

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
# Target Transforms
# ============================================================================

def quantile_classify(
    n_classes: int = 3,
    expanding_min: int = 60,
) -> Callable[[pd.Series], pd.Series]:
    """Return a target transform that bins returns into classes via expanding quantiles.

    Uses expanding (non-forward-looking) quantile boundaries so that each
    day's thresholds are computed from *past data only*.

    Parameters
    ----------
    n_classes : int
        Number of output classes (2 or 3 typical).
    expanding_min : int
        Minimum observations before quantile boundaries stabilise.

    Returns
    -------
    callable
        ``f(series) -> series`` mapping continuous returns to int labels.
    """
    def _transform(s: pd.Series) -> pd.Series:
        labels = pd.Series(np.full(len(s), -1, dtype=np.int64), index=s.index)
        boundaries = np.linspace(0, 1, n_classes + 1)[1:-1]  # e.g. [0.333, 0.667]

        for i in range(expanding_min, len(s)):
            window = s.iloc[:i]
            thresholds = [window.quantile(q) for q in boundaries]
            val = s.iloc[i]
            cls = 0
            for t in thresholds:
                if val > t:
                    cls += 1
            labels.iloc[i] = cls

        # Drop the warm-up period
        return labels[labels >= 0]

    return _transform


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


def _load_npz_date_keyed(
    path: Union[str, Path],
) -> Dict[date, np.ndarray]:
    """Load date-string-keyed NPZ (e.g. from SequenceRasterizer.parquet_to_npz).

    These files store one array per date: keys like '2011-01-03' -> (T, C, B).
    Returns a dict mapping python date -> float32 array.
    """
    npz = np.load(str(path), allow_pickle=True)
    out: Dict[date, np.ndarray] = {}
    for key in npz.files:
        try:
            d = pd.Timestamp(key).date()
        except (ValueError, TypeError):
            continue
        out[d] = np.asarray(npz[key], dtype=np.float32)
    if not out:
        raise ValueError(
            f"No date-string keys found in {path}. Available keys: {npz.files[:10]}"
        )
    return out


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
        summary_config: Optional[SummarySelectionConfig] = None,
        target_transform: Optional[Callable[[pd.Series], pd.Series]] = None,
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

        self.summary_config = summary_config
        self.target_transform = target_transform

        # Per-ticker loaded data
        self._summary: Dict[str, Dict[date, np.ndarray]] = {}
        self._summary_cols: Dict[str, List[str]] = {}
        self._profile: Dict[str, Dict[date, np.ndarray]] = {}
        self._raster: Dict[str, Dict[date, np.ndarray]] = {}
        self._seq: Dict[str, Dict[date, np.ndarray]] = {}
        self._seq_cols: Dict[str, List[str]] = {}
        self._seq_lens: Dict[str, Dict[date, int]] = {}
        self._targets: Dict[str, Dict[date, Union[int, float]]] = {}
        self._daily_returns: Dict[str, Dict[date, np.ndarray]] = {}  # (f_ae,) per day
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

    # ----- Summary Alignment -----

    def align_summaries(
        self,
        config: Optional[SummarySelectionConfig] = None,
    ) -> None:
        """Align summary feature schemas across all loaded tickers.

        After calling ``load_ticker_from_files`` for multiple tickers, their
        ``_summary`` dicts may have different column counts.  This method
        rewrites every ticker's summary arrays to a shared schema.

        Parameters
        ----------
        config : SummarySelectionConfig, optional
            Alignment strategy.  Falls back to ``self.summary_config`` or
            the default ``SummarySelectionConfig(strategy="exact")``.
        """
        cfg = config or self.summary_config or SummarySelectionConfig(strategy="exact")
        tickers = [t for t in self._summary if self._summary[t]]
        if len(tickers) <= 1:
            return  # nothing to align

        if cfg.strategy == "recompute":
            self._recompute_all_summaries(cfg)
            return

        # Reconstruct per-ticker DataFrames
        per_ticker_dfs: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            cols = self._summary_cols[t]
            dates = sorted(self._summary[t].keys())
            arr = np.stack([self._summary[t][d] for d in dates])
            per_ticker_dfs[t] = pd.DataFrame(arr, columns=cols, index=dates)

        if cfg.strategy == "exact":
            aligned_cols = self._align_exact(per_ticker_dfs, cfg)
        elif cfg.strategy == "signature":
            aligned_cols = self._align_signature(per_ticker_dfs, cfg)
        else:
            raise ValueError(f"Unknown strategy: {cfg.strategy!r}")

        if not aligned_cols:
            warnings.warn(
                "Summary alignment produced 0 common columns. "
                "Consider strategy='recompute'."
            )
            return

        # Rewrite _summary arrays to aligned shape
        for t in tickers:
            df = per_ticker_dfs[t]
            # Keep only aligned columns, fill missing with 0
            aligned_df = df.reindex(columns=aligned_cols, fill_value=0.0)
            self._summary_cols[t] = list(aligned_cols)
            for d, row in zip(aligned_df.index, aligned_df.values):
                self._summary[t][d] = row.astype(np.float32)

    def _align_exact(
        self,
        dfs: Dict[str, pd.DataFrame],
        cfg: SummarySelectionConfig,
    ) -> List[str]:
        """Intersect column names across all tickers."""
        col_sets = [set(df.columns) for df in dfs.values()]
        common = sorted(set.intersection(*col_sets)) if col_sets else []
        if cfg.always_include:
            for c in cfg.always_include:
                if c not in common and all(c in df.columns for df in dfs.values()):
                    common.append(c)
        return common

    def _align_signature(
        self,
        dfs: Dict[str, pd.DataFrame],
        cfg: SummarySelectionConfig,
    ) -> List[str]:
        """Match features by (window, remainder) signatures, pick best window.

        Returns the list of *original* column names from the first ticker
        that matched, renamed to canonical ``window__remainder`` form so all
        tickers share identical column names.
        """
        # Build per-ticker: {(window, remainder): [col, ...]}
        sig_maps: Dict[str, Dict[tuple, List[str]]] = {}
        for t, df in dfs.items():
            m: Dict[tuple, List[str]] = {}
            for c in df.columns:
                sig = feature_signature(c)
                if sig is None:
                    continue
                window, remainder = sig
                if cfg.require_substrings and not _filter_by_substrings(
                    remainder, cfg.require_substrings,
                ):
                    continue
                m.setdefault((window, remainder), []).append(c)
            sig_maps[t] = m

        # Try preferred windows in order
        best_sigs: List[tuple] = []
        for w in cfg.prefer_windows:
            per_sets = []
            for t in dfs:
                keys = {k for k in sig_maps[t] if k[0] == w}
                per_sets.append(keys)
            if not per_sets:
                continue
            inter = set.intersection(*per_sets)
            if len(inter) >= cfg.min_common:
                best_sigs = sorted(inter)
                break
            if len(inter) > len(best_sigs):
                best_sigs = sorted(inter)

        if not best_sigs:
            if cfg.strict:
                raise ValueError("No shared signature features found.")
            return []

        # Build canonical column names and rewrite DataFrames
        canonical_cols: List[str] = []
        # Add always-include columns first
        if cfg.always_include:
            for c in cfg.always_include:
                if all(c in df.columns for df in dfs.values()):
                    canonical_cols.append(c)

        for window, remainder in best_sigs:
            canonical_cols.append(f"{window}__{remainder}")

        # Rename columns in each ticker's DataFrame
        for t, df in dfs.items():
            rename_map: Dict[str, str] = {}
            for window, remainder in best_sigs:
                candidates = sig_maps[t].get((window, remainder), [])
                if candidates:
                    rename_map[sorted(candidates)[0]] = f"{window}__{remainder}"

            # Keep always_include as-is, add renamed signature cols
            keep_cols: List[str] = []
            if cfg.always_include:
                for c in cfg.always_include:
                    if c in df.columns:
                        keep_cols.append(c)

            for orig, canon in rename_map.items():
                if orig in df.columns:
                    keep_cols.append(orig)

            new_df = df[keep_cols].rename(columns=rename_map)
            new_df = new_df.reindex(columns=canonical_cols, fill_value=0.0)
            dfs[t] = new_df

        return canonical_cols

    def _recompute_all_summaries(self, cfg: SummarySelectionConfig) -> None:
        """Recompute universal summary features from sequential close prices."""
        periods = list(cfg.recompute_periods) if cfg.recompute_periods else [1, 5, 10]
        canonical_cols = (
            [f"session_return_{p}" for p in periods]
            + [f"session_volatility_{p}" for p in periods]
        )

        tickers = [t for t in self._seq if self._seq[t]]
        for t in tickers:
            recomputed = self._recompute_summary_from_seq(t, periods=periods)
            self._summary_cols[t] = list(canonical_cols)
            # Overwrite summary dict with recomputed values
            new_summary: Dict[date, np.ndarray] = {}
            for d in sorted(recomputed.index):
                py_d = d.date() if isinstance(d, (datetime, pd.Timestamp)) else d
                new_summary[py_d] = recomputed.loc[d].values.astype(np.float32)
            self._summary[t] = new_summary

            # Update available dates to intersection with new summary
            if t in self._available_dates:
                old = set(self._available_dates[t])
                new = set(new_summary.keys())
                self._available_dates[t] = sorted(old & new)

    def _recompute_summary_from_seq(
        self,
        ticker: str,
        periods: Sequence[int] = (1, 5, 10),
    ) -> pd.DataFrame:
        """Build session return/volatility features from sequential close prices.

        Uses the already-loaded ``_seq`` data so no separate intraday file
        is needed.
        """
        seq_dict = self._seq.get(ticker, {})
        cols = self._seq_cols.get(ticker, [])
        if not seq_dict or not cols:
            raise ValueError(f"No sequential data loaded for {ticker}")

        # Find close column index
        close_idx = None
        for name in ("close", "Close", "last", "Last"):
            if name in cols:
                close_idx = cols.index(name)
                break

        if close_idx is None:
            raise ValueError(
                f"No close/price column in sequential data for {ticker}. "
                f"Available: {cols}"
            )

        # Extract daily session close (last bar's close per day)
        daily_close: Dict[date, float] = {}
        for d in sorted(seq_dict.keys()):
            arr = seq_dict[d]
            if arr.ndim == 2 and arr.shape[0] > 0:
                daily_close[d] = float(arr[-1, close_idx])

        if not daily_close:
            return pd.DataFrame(columns=[f"session_return_{p}" for p in periods]
                                + [f"session_volatility_{p}" for p in periods])

        dates = sorted(daily_close.keys())
        closes = pd.Series(
            [daily_close[d] for d in dates],
            index=pd.DatetimeIndex(dates),
            dtype=np.float64,
        )

        # Daily returns
        rets = closes.pct_change()

        # Build features: rolling cumulative return and rolling mean volatility
        features: Dict[str, pd.Series] = {}
        log_rets = np.log1p(rets)
        for p in periods:
            cum = log_rets.rolling(p).sum().shift(1)
            features[f"session_return_{p}"] = np.expm1(cum)
            features[f"session_volatility_{p}"] = rets.abs().rolling(p).mean().shift(1)

        df = pd.DataFrame(features, index=closes.index)
        df = df.fillna(0.0).astype(np.float32)
        return df

    # ----- Daily Returns from Intraday -----

    def compute_daily_returns_from_intraday(
        self,
        ticker: str,
        intraday_path: Union[str, Path],
        target_time: str = "10:00",
    ) -> int:
        """Compute daily return features from 5min intraday OHLCV data.

        Extracts the close price at ``target_time`` each day, computes
        10AM-to-10AM returns, and builds a 3-feature vector per day:
        ``[return_1d, abs_return_1d, cumulative_5d_return]``.

        No shift is applied because the prediction target is for returns
        from 10AM forward — the 10AM(T-1)->10AM(T) return is fully known
        at prediction time.

        Parameters
        ----------
        ticker : str
            Ticker symbol (must already be in registry).
        intraday_path : str or Path
            Path to the intraday CSV file (Sierra Chart 5min export).
        target_time : str
            Time of day to anchor the daily close (HH:MM format).

        Returns
        -------
        int
            Number of valid daily return dates produced.
        """
        from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df

        intraday_path = Path(intraday_path)
        if not intraday_path.exists():
            warnings.warn(
                f"Intraday file not found for {ticker}: {intraday_path}. "
                f"ae_input will be unavailable."
            )
            return 0

        df = read_exported_df(str(intraday_path))

        # Filter to bars at the target time (e.g. "10:00")
        hour, minute = (int(x) for x in target_time.split(":"))
        mask = (df.index.hour == hour) & (df.index.minute == minute)
        daily_at_target = df.loc[mask, "Close"].copy()

        if daily_at_target.empty:
            warnings.warn(
                f"No bars at {target_time} found for {ticker}. "
                f"ae_input will be unavailable."
            )
            return 0

        # One price per day — take the first if duplicates exist
        daily_at_target.index = daily_at_target.index.date
        daily_at_target = daily_at_target.groupby(level=0).first()
        daily_at_target = daily_at_target.sort_index()

        # 10AM(T-1) -> 10AM(T) return — no shift needed for 10AM-forward prediction
        ret_1d = daily_at_target.pct_change()
        abs_ret_1d = ret_1d.abs()
        log_rets = np.log1p(ret_1d)
        cum_5d = log_rets.rolling(5).sum().apply(np.expm1)

        features = pd.DataFrame({
            "return_1d": ret_1d,
            "abs_return_1d": abs_ret_1d,
            "cumulative_5d_return": cum_5d,
        }, index=daily_at_target.index)

        # Scale returns to basis points for numerical stability
        features["return_1d"] = (features["return_1d"] * 100.0).clip(-20, 20)
        features["abs_return_1d"] = (features["abs_return_1d"] * 100.0).clip(0, 20)
        features["cumulative_5d_return"] = (
            features["cumulative_5d_return"] * 100.0
        ).clip(-50, 50)

        features = features.fillna(0.0).astype(np.float32)

        returns_dict: Dict[date, np.ndarray] = {}
        for d, row in features.iterrows():
            returns_dict[d] = row.values

        self._daily_returns[ticker] = returns_dict

        # Narrow available_dates to intersection with daily returns so
        # ae_input is always aligned with the other modalities.
        if ticker in self._available_dates and returns_dict:
            old_dates = set(self._available_dates[ticker])
            ret_dates = set(returns_dict.keys())
            aligned = sorted(old_dates & ret_dates)
            n_dropped = len(old_dates) - len(aligned)
            if n_dropped > 0:
                warnings.warn(
                    f"[{ticker}] Dropped {n_dropped} dates missing from "
                    f"intraday returns ({len(aligned)} remain)"
                )
            self._available_dates[ticker] = aligned

        return len(returns_dict)

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

        summary_col_names = list(summary_df.columns)
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

        # 3. Rasterized VPIN (NPZ) — try stacked format first, fall back to date-keyed
        raster_dict: Dict[date, np.ndarray] = {}
        raster_path = root / raster_file
        try:
            raster_arr, raster_dates_raw = _load_npz_arrays(
                raster_path,
                array_keys=("data", "rasterized", "tensor", "arr_0"),
            )
            if raster_dates_raw is not None:
                raster_dates = _dates_to_python(raster_dates_raw)
            else:
                raster_dates = summary_dates[:len(raster_arr)]
            for d, arr in zip(raster_dates, raster_arr):
                raster_dict[d] = arr
        except ValueError:
            # Date-string-keyed format (e.g. from SequenceRasterizer.parquet_to_npz)
            raster_dict = _load_npz_date_keyed(raster_path)

        # 4. Sequential VPIN (Parquet)
        seq_df = _read_tabular(root / seq_file)
        seq_numeric = seq_df.select_dtypes(include="number").astype(np.float32)
        seq_numeric = seq_numeric.fillna(0.0)
        seq_col_names = list(seq_numeric.columns)

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

        # Apply target transform (e.g. quantile discretization for classification)
        if self.target_transform is not None:
            target_series = self.target_transform(target_series)

        target_dict: Dict[date, Union[int, float]] = {}
        for idx, val in target_series.items():
            d = idx.date() if isinstance(idx, (datetime, pd.Timestamp)) else idx
            target_dict[d] = val

        # Compute available dates (intersection of ALL modalities)
        all_date_sets = [
            set(summary_dict.keys()),
            set(profile_dict.keys()),
            set(raster_dict.keys()),
            set(seq_dict.keys()),
            set(target_dict.keys()),
        ]
        common_dates = sorted(set.intersection(*all_date_sets))

        self._summary[ticker] = summary_dict
        self._summary_cols[ticker] = summary_col_names
        self._profile[ticker] = profile_dict
        self._raster[ticker] = raster_dict
        self._seq[ticker] = seq_dict
        self._seq_cols[ticker] = seq_col_names
        self._seq_lens[ticker] = seq_lens_dict
        self._targets[ticker] = target_dict
        self._available_dates[ticker] = common_dates

        return len(common_dates)

    def load_events_from_dataframe(self, df: pd.DataFrame, **kwargs):
        """Load event calendar from a DataFrame."""
        self.events_by_date = events_from_dataframe(df, **kwargs)

    def load_macro_from_dataframe(self, df: pd.DataFrame, **kwargs):
        """Load pre-computed macro features from a DataFrame.

        For raw daily close data that needs feature engineering, use
        :meth:`build_macro_from_daily` instead.
        """
        self.macro_features = macro_from_dataframe(df, **kwargs)
        if self.macro_features:
            sample = next(iter(self.macro_features.values()))
            self.f_macro = len(sample)

    def build_macro_from_daily(
        self,
        daily_df: pd.DataFrame,
        *,
        macro_prep_kwargs: Optional[Dict] = None,
        scale: bool = True,
        clip_range: float = 10.0,
        exclude_tickers: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Compute macro features from raw daily close data via MacroFeaturePrep.

        The output is shifted by 1 day so that prediction-day T only sees
        features derived from day T-1's closes (avoids lookahead).

        Parameters
        ----------
        daily_df : DataFrame
            Raw daily data with columns like VIX, YIELD_10Y, YIELD_2Y,
            SPX, etc.  DatetimeIndex or parseable date index.
        macro_prep_kwargs : dict, optional
            Override kwargs for :class:`MacroFeaturePrep` (rate_cols,
            vix_col, return_windows, etc.).
        scale : bool
            If True, scale features to ~[-5, +5] range (matching the
            spatial data normalisation convention).
        clip_range : float
            Absolute clipping bound applied when *scale* is True.
        exclude_tickers : sequence of str, optional
            Columns to drop from *daily_df* before processing (e.g.
            ``["SPX"]`` when training ES to avoid target leakage).

        Returns
        -------
        DataFrame
            Processed & shifted macro features (for inspection).
            The features are also stored into ``self.macro_features``
            ready for sample building.
        """
        from CTAFlow.features.macro_prep import MacroFeaturePrep

        df = daily_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.normalize()

        # Drop excluded tickers before feature engineering
        if exclude_tickers:
            drop = [c for c in df.columns if c in set(exclude_tickers)]
            if drop:
                df = df.drop(columns=drop)

        prep = MacroFeaturePrep(**(macro_prep_kwargs or {}))
        processed = prep.process(df)

        # Keep raw yield levels alongside the derived changes/spread
        for col in prep.rate_cols:
            if col in df.columns:
                processed[col] = df[col]

        # Shift by 1 day: day T sees day T-1 close-derived features
        processed = processed.shift(1)
        processed = processed.ffill().bfill().fillna(0.0)

        if scale:
            processed = self._scale_macro_features(processed, clip_range=clip_range)

        # Store as {date -> np.array} dict
        self.macro_features = {}
        for idx, row in processed.iterrows():
            d = idx.date() if isinstance(idx, (datetime, pd.Timestamp)) else idx
            self.macro_features[d] = row.values.astype(np.float32)

        if self.macro_features:
            self.f_macro = len(next(iter(self.macro_features.values())))

        return processed

    def build_macro_from_engine(
        self,
        market_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Use a pre-built MarketFeatureEngine output as macro context.

        ``MarketFeatureEngine.build_features()`` already produces
        z-score-normalised, clipped features.  This method only applies
        the 1-day shift for lookahead prevention and stores the result.

        Parameters
        ----------
        market_df : DataFrame
            Output of ``MarketFeatureEngine.build_features()`` (daily
            z-scored features clipped to [-5, 5]).

        Returns
        -------
        DataFrame
            Shifted macro features.
        """
        df = market_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.normalize()

        # Shift by 1 day: day T sees day T-1 close-derived features
        df = df.shift(1).ffill().bfill().fillna(0.0)

        self.macro_features = {}
        for idx, row in df.iterrows():
            d = idx.date() if isinstance(idx, (datetime, pd.Timestamp)) else idx
            self.macro_features[d] = row.values.astype(np.float32)

        if self.macro_features:
            self.f_macro = len(next(iter(self.macro_features.values())))

        return df

    @staticmethod
    def _scale_macro_features(
        df: pd.DataFrame,
        clip_range: float = 10.0,
    ) -> pd.DataFrame:
        """Scale macro features to ~[-5, +5] matching spatial data scale.

        Lean feature set from MacroFeaturePrep (14 cols):
          REAL_RATE_10Y, REAL_RATE_10Y_chg  — already ~[-5,5], chg ×10
          TERM_SPREAD, TERM_SPREAD_chg      — already ~[-3,3], chg ×10
          FF_SPREAD_10Y                     — ~[-5,5], clip only
          REAL_FF_RATE                      — ~[-5,5], clip only
          CORE_INFLATION, CORE_INFLATION_chg — /2, chg ×10
          RGDP_YOY, RGDP_YOY_chg           — /2, chg ×10
          UNRATE, UNRATE_chg                — /2, chg ×10
          UMCSENT, UMCSENT_chg              — /20, chg ×10

        Also handles MarketFeatureEngine columns (returns, VIX) when present.
        """
        df = df.copy()
        for col in df.columns:
            cl = col.lower()
            if "_chg" in cl:
                # Release-day changes are tiny diffs — amplify
                df[col] = (df[col] * 10.0).clip(-clip_range, clip_range)
            elif "_ret_" in cl or "_rel_" in cl:
                # Market returns → basis points
                df[col] = (df[col] * 100.0).clip(-clip_range, clip_range)
            elif cl == "vix":
                df[col] = (df[col] / 10.0).clip(0, clip_range)
            elif cl == "umcsent":
                df[col] = (df[col] / 20.0).clip(0, clip_range)
            elif cl in ("unrate", "core_inflation", "rgdp_yoy"):
                df[col] = (df[col] / 2.0).clip(-clip_range, clip_range)
            else:
                # REAL_RATE_10Y, TERM_SPREAD, FF_SPREAD_10Y, REAL_FF_RATE
                # already in ~[-5, +5] range
                df[col] = df[col].clip(-clip_range, clip_range)
        return df

    # ----- Feature Scaling -----

    def scale_features(
        self,
        tickers: Optional[List[str]] = None,
        *,
        scale_summary: bool = True,
        scale_sequential: bool = True,
        scale_profiles: bool = True,
        rolling_window: int = 252,
        price_scale: float = 100.0,
        profile_log_vol_div: float = 15.0,
        clip_range: float = 5.0,
    ) -> None:
        """Scale all loaded modalities in place (no lookahead).

        Follows the same conventions as ``DeepIDMomentum``:

        **Summary** — pattern-based fixed multipliers for known feature
        types, rolling z-score for unbounded / drifting features.

        **Sequential** — price-like columns normalised as
        ``(value - close) / close * 100`` (basis points); orderflow
        columns scaled with fixed multipliers.

        **Profiles** — channel-wise scaling matching ``ProfileScaler``:
        vol-density as-is, imbalance clipped, log-volume ÷ divisor,
        price channel × ``price_scale``.

        Parameters
        ----------
        tickers : list of str, optional
            Which tickers to scale. ``None`` = all loaded.
        scale_summary : bool
            Scale summary feature vectors.
        scale_sequential : bool
            Scale sequential (VPIN) feature vectors.
        scale_profiles : bool
            Scale profile arrays channel-wise.
        rolling_window : int
            Window for rolling z-score on unbounded features.
        price_scale : float
            Multiplier for price channels (profiles).
        profile_log_vol_div : float
            Divisor for log-volume channel in profiles.
        clip_range : float
            Final clip bound for scaled features.
        """
        if tickers is None:
            tickers = list(self._available_dates.keys())

        for ticker in tickers:
            if scale_summary and ticker in self._summary:
                self._scale_summary_inplace(
                    ticker, rolling_window=rolling_window, clip_range=clip_range,
                )
            if scale_sequential and ticker in self._seq:
                self._scale_sequential_inplace(
                    ticker, clip_range=clip_range,
                )
            if scale_profiles and ticker in self._profile:
                self._scale_profiles_inplace(
                    ticker,
                    price_scale=price_scale,
                    log_vol_div=profile_log_vol_div,
                )

    # ---- summary ----

    def _scale_summary_inplace(
        self,
        ticker: str,
        rolling_window: int = 252,
        clip_range: float = 5.0,
    ) -> None:
        """Scale summary features using fixed multipliers or rolling z-score.

        Pattern-based rules (matches ``DeepIDMomentum.scale_summary_data``):

        Fixed multipliers (no lookahead):
        - ``*dist_prior*``, ``*session_dist_vwap*``: × 100 (to bps)
        - ``rsv_*``: × 500
        - ``*impact_coeff*``: × 500
        - ``*impact_vol*``: log1p(x) − 5
        - ``*norm_range_hist*``: (clip(0,3) − 1) × 5
        - ``*intraday_curve_slope_change*``: × 10
        - ``*intraday_rel_basis_change*``: × 5
        - ``*scaled_returns*``: clip(−10,10) / 2
        - ``*_ret_*``, ``*_rel_*``: × 100 (to bps)
        - ``*_chg_*``: × 10

        Rolling z-score (non-forward-looking):
        - Calendar / unbounded features not caught by the fixed rules

        Already scaled (clip only):
        - ``*deseasonalized*``, ``is_*``, ``*curve_slope*``, ``*rel_basis*``
        """
        cols = self._summary_cols.get(ticker, [])
        if not cols:
            return

        dates = sorted(self._summary[ticker].keys())
        if not dates:
            return

        # Reconstruct DataFrame for rolling operations
        arr = np.stack([self._summary[ticker][d] for d in dates])
        df = pd.DataFrame(arr, columns=cols, index=dates)

        scaled_cols: set = set()

        # --- FIXED MULTIPLIERS ---
        for col in df.columns:
            cl = col.lower()

            if any(p in cl for p in ("dist_prior", "session_dist_vwap")):
                df[col] = df[col].clip(-0.05, 0.05) * 100.0
                scaled_cols.add(col)
            elif cl.startswith("rsv_"):
                df[col] = df[col].clip(0, 0.01) * 500.0
                scaled_cols.add(col)
            elif "impact_coeff" in cl:
                df[col] = df[col].clip(0, 0.01) * 500.0
                scaled_cols.add(col)
            elif "impact_vol" in cl:
                df[col] = np.log1p(df[col]) - 5.0
                scaled_cols.add(col)
            elif "norm_range_hist" in cl:
                df[col] = (df[col].clip(0, 3) - 1.0) * 5.0
                scaled_cols.add(col)
            elif "intraday_curve_slope_change" in cl:
                df[col] = df[col].clip(-0.5, 0.5) * 10.0
                scaled_cols.add(col)
            elif "intraday_rel_basis_change" in cl:
                df[col] = df[col].clip(-1, 1) * 5.0
                scaled_cols.add(col)
            elif "scaled_returns" in cl:
                df[col] = df[col].clip(-10, 10) / 2.0
                scaled_cols.add(col)
            elif "_ret_" in cl or "_rel_" in cl:
                df[col] = (df[col] * 100.0).clip(-clip_range, clip_range)
                scaled_cols.add(col)
            elif "_chg_" in cl:
                df[col] = (df[col] * 10.0).clip(-clip_range, clip_range)
                scaled_cols.add(col)

        # --- ALREADY SCALED (clip only) ---
        for col in df.columns:
            if col in scaled_cols:
                continue
            cl = col.lower()
            if (
                "deseasonalized" in cl
                or cl.startswith("is_")
                or ("curve_slope" in cl and "intraday" not in cl)
                or ("rel_basis" in cl and "intraday" not in cl)
            ):
                df[col] = df[col].clip(-clip_range, clip_range)
                scaled_cols.add(col)

        # --- ROLLING Z-SCORE for remaining unbounded features ---
        remaining = [c for c in df.columns if c not in scaled_cols]
        if remaining:
            for col in remaining:
                r_mean = df[col].expanding(min_periods=20).mean()
                r_std = df[col].expanding(min_periods=20).std().replace(0, 1)
                if len(df) > rolling_window:
                    r_mean = df[col].rolling(
                        window=rolling_window, min_periods=20,
                    ).mean()
                    r_std = df[col].rolling(
                        window=rolling_window, min_periods=20,
                    ).std().replace(0, 1)
                df[col] = ((df[col] - r_mean) / r_std).clip(
                    -clip_range, clip_range,
                )
            df[remaining] = df[remaining].fillna(0.0)

        # Write back
        values = df.values.astype(np.float32)
        for i, d in enumerate(dates):
            self._summary[ticker][d] = values[i]

    # ---- sequential ----

    def _scale_sequential_inplace(
        self,
        ticker: str,
        clip_range: float = 5.0,
    ) -> None:
        """Scale sequential VPIN data with fixed multipliers (no lookahead).

        Matches ``DeepIDMomentum.normalize_sequential_features``:

        Price-like columns (vah, val, poc, profile_vwap, ib_high, ib_low):
            ``(value − close) / close × 100``   (basis points)

        Orderflow columns:
        - ``vpin``: ``(x − 0.5) × 10``          → ~[−5, 5]
        - ``signed_imbalance``: ``× 5``          → [−5, 5]
        - ``imb_frac``: ``(x − 0.5) × 10``      → ~[−5, 5]
        - ``vol``: ``log1p(x) − 2.5``            → ~[−2, 2]
        - ``bucket_return``: ``× 100``           → bps
        - ``log_duration``: ``÷ 2``              → ~[−3.5, 4.5]
        """
        cols = self._seq_cols.get(ticker, [])
        if not cols:
            return

        col_idx = {c: i for i, c in enumerate(cols)}

        price_like = {"vah", "val", "poc", "profile_vwap", "ib_high", "ib_low"}
        price_like_idx = [col_idx[c] for c in price_like if c in col_idx]
        close_idx = col_idx.get("close")

        for d, arr in self._seq[ticker].items():
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            arr = arr.copy()

            # Price-like → basis points relative to close
            if close_idx is not None and price_like_idx:
                close_vals = arr[:, close_idx : close_idx + 1]
                safe_close = np.where(
                    np.abs(close_vals) < 1e-8, 1.0, close_vals,
                )
                for ci in price_like_idx:
                    arr[:, ci] = (
                        (arr[:, ci : ci + 1] - close_vals) / safe_close * 100.0
                    ).squeeze(-1)

            # Orderflow fixed multipliers
            if "vpin" in col_idx:
                i = col_idx["vpin"]
                arr[:, i] = (arr[:, i] - 0.5) * 10.0
            if "signed_imbalance" in col_idx:
                i = col_idx["signed_imbalance"]
                arr[:, i] = arr[:, i] * 5.0
            if "imb_frac" in col_idx:
                i = col_idx["imb_frac"]
                arr[:, i] = (arr[:, i] - 0.5) * 10.0
            if "vol" in col_idx:
                i = col_idx["vol"]
                arr[:, i] = np.log1p(arr[:, i]) - 2.5
            if "bucket_return" in col_idx:
                i = col_idx["bucket_return"]
                arr[:, i] = arr[:, i] * 100.0
            if "log_duration" in col_idx:
                i = col_idx["log_duration"]
                arr[:, i] = arr[:, i] / 2.0

            self._seq[ticker][d] = arr.astype(np.float32)

    # ---- profiles ----

    def _scale_profiles_inplace(
        self,
        ticker: str,
        price_scale: float = 100.0,
        log_vol_div: float = 15.0,
    ) -> None:
        """Scale profile channels matching ``ProfileScaler``.

        Channel layout  (N, C, Bins):
        - 0: Volume density (0-1) — keep as-is
        - 1: Imbalance (−1 to 1) — clip
        - 2: Log volume (0-14) — ÷ ``log_vol_div``
        - 3: Relative price (~−0.02 to 0.02) — × ``price_scale``
        """
        for d, arr in self._profile[ticker].items():
            if arr.ndim < 2:
                continue
            arr = arr.copy()
            n_ch = arr.shape[0]
            # Ch 1: imbalance clip
            if n_ch > 1:
                arr[1] = np.clip(arr[1], -1.0, 1.0)
            # Ch 2: log volume normalise
            if n_ch > 2:
                arr[2] = arr[2] / log_vol_div
            # Ch 3: price → basis points
            if n_ch > 3:
                arr[3] = arr[3] * price_scale
            self._profile[ticker][d] = arr.astype(np.float32)

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
        ae_input: Optional[np.ndarray] = None,
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
            ae_input=ae_input,
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
            returns_dict = self._daily_returns.get(ticker, {})
            has_returns = bool(returns_dict)

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

                # Skip if raster or sequential data missing for prediction date
                if pred_date not in raster_dict or pred_date not in seq_dict:
                    continue

                summary_stack = np.stack(
                    [summary_dict[d] for d in window_dates]
                )
                profile_stack = np.stack(
                    [profile_dict[d] for d in window_dates]
                )

                raster = raster_dict[pred_date]
                seq = seq_dict[pred_date]
                seq_len = seq_lens_dict.get(pred_date, seq.shape[0])

                target = target_dict.get(pred_date)
                if target is None:
                    continue

                # Build ae_input from daily returns (if available)
                ae_input = None
                if has_returns:
                    # All window_dates should be in returns_dict because
                    # _available_dates was intersected with daily returns.
                    # Skip sample if any date is somehow missing.
                    if not all(d in returns_dict for d in window_dates):
                        continue
                    ae_input = np.stack(
                        [returns_dict[d] for d in window_dates]
                    )  # (W, f_ae)

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
                    ae_input=ae_input,
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
        macro_raw_df: Optional[pd.DataFrame] = None,
        market_engine_df: Optional[pd.DataFrame] = None,
        macro_prep_kwargs: Optional[Dict] = None,
        exclude_macro_tickers: Optional[Sequence[str]] = None,
        custom_ticker_table: Optional[Dict[str, Dict[str, str]]] = None,
        summary_config: Optional[SummarySelectionConfig] = None,
        target_transform: Optional[Callable[[pd.Series], pd.Series]] = None,
        intraday_file: Optional[str] = None,
        intraday_target_time: str = "10:00",
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
                    intraday.csv  (optional, for VAE daily returns)
                ES/
                    ...

        Macro context can be supplied in three ways (first match wins):

        1. ``macro_raw_df`` — raw daily closes (VIX, YIELD_10Y, SPX, …).
           Processed via :class:`MacroFeaturePrep`, shifted by 1 day.
        2. ``market_engine_df`` — output of
           ``MarketFeatureEngine.build_features()``. Already z-scored;
           only the 1-day shift is applied.
        3. ``macro_df`` — pre-computed feature DataFrame loaded as-is
           (no shifting applied — caller is responsible).

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
            Pre-computed daily macro features DataFrame (no shifting).
        macro_raw_df : DataFrame, optional
            Raw daily close data to run through MacroFeaturePrep
            (shifted + scaled automatically).
        market_engine_df : DataFrame, optional
            Output of ``MarketFeatureEngine.build_features()``
            (shifted automatically).
        macro_prep_kwargs : dict, optional
            Override kwargs for MacroFeaturePrep when using macro_raw_df.
        exclude_macro_tickers : sequence of str, optional
            Columns to drop from macro_raw_df before processing.
        custom_ticker_table : dict, optional
            Override default ticker metadata.
        target_transform : callable, optional
            ``f(series) -> series`` applied to raw targets before storing.
            Use ``quantile_classify(n_classes=3)`` for classification tasks.

        Returns
        -------
        TFTAlignedPrepLayer
            Fully loaded prep layer ready for get_loaders().
        """
        root = Path(root_dir)

        events_by_date = None
        if events_df is not None:
            events_by_date = events_from_dataframe(events_df)

        prep = cls(
            tickers=tickers,
            window_size=window_size,
            events_by_date=events_by_date,
            f_macro=f_macro,
            custom_ticker_table=custom_ticker_table,
            summary_config=summary_config,
            target_transform=target_transform,
            **kwargs,
        )

        # Macro context — priority: raw daily > engine output > pre-computed
        if macro_raw_df is not None:
            result = prep.build_macro_from_daily(
                macro_raw_df,
                macro_prep_kwargs=macro_prep_kwargs,
                exclude_tickers=exclude_macro_tickers,
            )
            print(f"  [Macro] Built {result.shape[1]} features from raw daily data (shifted 1d)")
        elif market_engine_df is not None:
            result = prep.build_macro_from_engine(market_engine_df)
            print(f"  [Macro] Loaded {result.shape[1]} MarketFeatureEngine features (shifted 1d)")
        elif macro_df is not None:
            prep.macro_features = macro_from_dataframe(macro_df)
            if prep.macro_features:
                prep.f_macro = len(next(iter(prep.macro_features.values())))
            print(f"  [Macro] Loaded {prep.f_macro} pre-computed macro features")

        for ticker in tickers:
            ticker_dir = root / ticker
            if not ticker_dir.exists():
                warnings.warn(f"Directory not found for ticker '{ticker}': {ticker_dir}")
                continue
            n = prep.load_ticker_from_files(ticker, ticker_dir)
            print(f"  [{ticker}] Loaded {n} valid dates")

            # Load intraday data for VAE daily returns (if requested)
            if intraday_file is not None:
                intraday_path = ticker_dir / intraday_file
                n_ret = prep.compute_daily_returns_from_intraday(
                    ticker, intraday_path, target_time=intraday_target_time,
                )
                if n_ret > 0:
                    print(f"  [{ticker}] Computed {n_ret} daily returns "
                          f"(target_time={intraday_target_time})")

        # Align summary schemas across tickers if config provided
        if summary_config is not None and len(prep._summary) > 1:
            prep.align_summaries(summary_config)
            sample_cols = next(
                (prep._summary_cols[t] for t in prep._summary_cols if prep._summary_cols[t]),
                [],
            )
            print(f"  [Summary] Aligned to {len(sample_cols)} features "
                  f"(strategy={summary_config.strategy!r})")

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
