"""
MacroGAT Dataset — bar-level intraday samples with macro context.

Aligns:
  - Previous-day macro node features (Asset + Economic nodes for GAT)
  - Intraday 5min bar technical features (from ContinuousIntradayPrep)
  - NumberBars spatial orderflow (timestamp-indexed)
  - Rasterized VPIN spatial data (previous day)
  - Target: 30min forward log return → quartile classification (4 classes)

Stride/patching controls overlap between samples.
"""
from __future__ import annotations

import bisect
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from CTAFlow.models.prep.intraday_continuous import (
    ContinuousIntradayPrep,
    SessionSpec,
)
from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df
from CTAFlow.data.datasets.v3_continuous import (
    _load_npz_date_keyed,
    _load_npz_arrays,
    _dates_to_python,
    _normalize_datetime_index,
    scale_numbars,
)
from CTAFlow.features.macro_gat_prep import (
    MacroGATPrep,
    NODE_ORDER,
    ASSET_NODE_ORDER,
    ECON_NODE_ORDER,
)


# ---------------------------------------------------------------------------
# Quartile target classification
# ---------------------------------------------------------------------------

def create_quartile_targets(
    returns: np.ndarray,
    rolling_window: int = 1000,
) -> np.ndarray:
    """Map continuous returns to 4 adaptive quartile classes.

    Uses rolling quantiles so thresholds track volatility regimes.

    Classes:
      0: Strong Negative (< 25th percentile)
      1: Weak Negative   (25th to 50th)
      2: Weak Positive   (50th to 75th)
      3: Strong Positive  (> 75th)

    Returns np.int64 array, NaN positions → -1 (filtered out downstream).
    """
    s = pd.Series(returns)
    q25 = s.rolling(rolling_window, min_periods=max(rolling_window // 4, 50)).quantile(0.25)
    q50 = s.rolling(rolling_window, min_periods=max(rolling_window // 4, 50)).quantile(0.50)
    q75 = s.rolling(rolling_window, min_periods=max(rolling_window // 4, 50)).quantile(0.75)

    targets = np.full(len(returns), -1, dtype=np.int64)
    valid = ~(np.isnan(returns) | np.isnan(q25.values))

    targets[valid & (s.values < q25.values)] = 0
    targets[valid & (s.values >= q25.values) & (s.values < q50.values)] = 1
    targets[valid & (s.values >= q50.values) & (s.values < q75.values)] = 2
    targets[valid & (s.values >= q75.values)] = 3

    return targets


# ---------------------------------------------------------------------------
# Prep layer — loads and aligns all modalities
# ---------------------------------------------------------------------------

class MacroGATContinuousPrep:
    """Load and align macro + intraday + spatial data for Gold GAT training.

    Per sample (one intraday bar):
      - macro_nodes: dict[node_name -> (macro_lookback, F_node)] from prev day
      - tech_features: (tech_lookback, F_tech) intraday technicals
      - numbars_recent: (nb_lookback, C, bins) timestamp-aligned NumberBars
      - raster_prev_day: (T, C, bins) previous day's rasterized VPIN
      - target: 30min forward log return
      - target_class: quartile class label (0-3)
    """

    def __init__(
        self,
        sessions: Optional[Sequence[SessionSpec]] = None,
        bar_minutes: int = 5,
        target_horizon_minutes: int = 30,
        macro_lookback: int = 20,
        quartile_window: int = 1000,
        fred_api_key: Optional[str] = None,
    ):
        self.sessions = sessions or [SessionSpec("USA", "08:30", "16:00")]
        self.bar_minutes = bar_minutes
        self.target_steps = target_horizon_minutes // bar_minutes  # 6 for 30min
        self.macro_lookback = macro_lookback
        self.quartile_window = quartile_window

        self.prep = ContinuousIntradayPrep(
            sessions=self.sessions, bar_minutes=bar_minutes,
        )
        self.macro_prep = MacroGATPrep(fred_api_key=fred_api_key)

        # Populated by load()
        self._tech_df: Optional[pd.DataFrame] = None
        self._tech_feature_cols: List[str] = []
        self._target_col: str = ""
        self._target_class_arr: Optional[np.ndarray] = None
        self._numbars_ts: Tuple[np.ndarray, np.ndarray] = (
            np.array([], dtype="datetime64[ns]"),
            np.empty((0, 4, 32), dtype=np.float32),
        )
        self._rasters: Dict[date, np.ndarray] = {}
        self._macro_nodes: Dict[str, pd.DataFrame] = {}

        # Auto-detected shapes
        self._numbars_bar_shape: Optional[Tuple[int, ...]] = None
        self._raster_shape: Optional[Tuple[int, ...]] = None

    def load(
        self,
        root_dir: Union[str, Path],
        ticker: str = "GC",
        intraday_file: str = "intraday.csv",
        numbars_file: str = "{TICKER}_numbars.npz",
        raster_file: str = "rasterized.npz",
        macro_start_date: Optional[str] = None,
    ) -> "MacroGATContinuousPrep":
        """Load all data from standard directory layout.

        Parameters
        ----------
        root_dir : path
            Parent directory containing ``<ticker>/`` subfolder.
        ticker : str
            Ticker name (default GC for Gold).
        macro_start_date : str, optional
            Override start date for macro data fetch. If None, derived
            from intraday data with extra lookback.
        """
        root = Path(root_dir) / ticker

        # 1. Intraday technical features + 30min target
        raw_df = read_exported_df(str(root / intraday_file))
        df_out, train_mask, target_cols = self.prep.prepare(
            raw_df,
            steps_60m=self.target_steps,
            keep_only_active=False,
            apply_scaling=True,
            scale_to_basis_points=True,
        )

        feat_cols = self.prep.get_feature_cols(
            steps_60m=self.target_steps,
            bar_minutes=self.bar_minutes,
        )
        feat_cols = [c for c in feat_cols if c in df_out.columns]

        # Target: y_fwd_{target_steps} = 30min forward return
        target_col = target_cols[-1]

        df_out[feat_cols] = df_out[feat_cols].ffill().bfill().fillna(0.0)
        self._tech_df = df_out
        self._tech_feature_cols = feat_cols
        self._target_col = target_col

        # Compute quartile classification targets
        raw_targets = df_out[target_col].values.astype(np.float32)
        self._target_class_arr = create_quartile_targets(
            raw_targets, rolling_window=self.quartile_window,
        )

        # 2. NumberBars (timestamp-indexed)
        nb_filename = numbars_file.replace("{TICKER}", ticker)
        nb_path = root / nb_filename
        if nb_path.exists():
            nb_npz = np.load(str(nb_path), allow_pickle=True)
            nb_data = nb_npz["data"].astype(np.float32)
            nb_idx = nb_npz["idx"]
            nb_data = scale_numbars(nb_data)
            sort_order = np.argsort(nb_idx)
            self._numbars_ts = (nb_idx[sort_order], nb_data[sort_order])
            self._numbars_bar_shape = nb_data.shape[1:]
            print(f"  NumberBars: {len(nb_idx)} frames, shape={nb_data.shape[1:]}")
        else:
            print(f"  NumberBars: {nb_path} not found, using zeros")

        # 3. Rasterized VPIN (date-keyed)
        rast_path = root / raster_file
        if rast_path.exists():
            self._rasters = self._load_spatial_npz(rast_path)
            if self._rasters:
                sample = next(iter(self._rasters.values()))
                self._raster_shape = sample.shape
            print(f"  Rasterized: {len(self._rasters)} dates, shape={self._raster_shape}")
        else:
            print(f"  Rasterized: {rast_path} not found, using zeros")

        # 4. Macro node features
        intra_start = df_out.index.min()
        intra_end = df_out.index.max()
        if macro_start_date:
            m_start = pd.Timestamp(macro_start_date).to_pydatetime()
        else:
            # Extra lookback for macro rolling features + lookback window
            m_start = (intra_start - pd.DateOffset(days=self.macro_lookback + 300)).to_pydatetime()
        m_end = intra_end.to_pydatetime()

        print(f"  Fetching macro data: {m_start.date()} → {m_end.date()}")
        self._macro_nodes = self.macro_prep.fetch_and_build(
            start_date=m_start, end_date=m_end,
        )

        # Summary
        n_bars = len(df_out)
        n_session = train_mask.sum() if hasattr(train_mask, "sum") else 0
        print(f"  Intraday: {n_bars} bars, {n_session} in-session, "
              f"{len(feat_cols)} features, target={target_col}")
        for name, mdf in self._macro_nodes.items():
            print(f"  Macro {name}: {mdf.shape[1]} features, {len(mdf)} days")

        return self

    @staticmethod
    def _load_spatial_npz(path: Path) -> Dict[date, np.ndarray]:
        """Load NPZ as date->array dict."""
        from collections import defaultdict
        try:
            arr, dates_raw = _load_npz_arrays(path)
            if dates_raw is not None:
                dates = _dates_to_python(dates_raw)
                groups: Dict[date, list] = defaultdict(list)
                for i, d in enumerate(dates):
                    if i < len(arr):
                        groups[d].append(i)
                out: Dict[date, np.ndarray] = {}
                for d, idxs in groups.items():
                    out[d] = arr[idxs[0]] if len(idxs) == 1 else arr[idxs]
                return out
            return {}
        except ValueError:
            return _load_npz_date_keyed(path)

    def get_dims(self) -> Dict[str, int]:
        """Return feature dimensions for model construction."""
        dims = {
            "f_tech": len(self._tech_feature_cols),
            "target_steps": self.target_steps,
            "macro_lookback": self.macro_lookback,
        }
        # Macro node dimensions
        for name, mdf in self._macro_nodes.items():
            dims[f"macro_{name}"] = mdf.shape[1]
        # Spatial shapes
        if self._numbars_bar_shape:
            dims["nb_channels"] = self._numbars_bar_shape[0]
            dims["nb_bins"] = self._numbars_bar_shape[1]
        if self._raster_shape:
            if len(self._raster_shape) == 3:
                dims["raster_T"] = self._raster_shape[0]
                dims["raster_C"] = self._raster_shape[1]
                dims["raster_bins"] = self._raster_shape[2]
            elif len(self._raster_shape) == 2:
                dims["raster_C"] = self._raster_shape[0]
                dims["raster_bins"] = self._raster_shape[1]
        return dims

    def build_samples(
        self,
        tech_lookback: int = 64,
        numbars_lookback: int = 8,
        session_only: bool = True,
        sample_session_start: Optional[str] = None,
        sample_session_end: Optional[str] = None,
        stride: int = 6,
        patch_size: int = 1,
    ) -> List[Dict]:
        """Build flat list of sample dicts.

        Parameters
        ----------
        tech_lookback : int
            Bars of intraday tech features per sample.
        numbars_lookback : int
            Max NumberBars frames to include (timestamp-aligned).
        session_only : bool
            Restrict to active session bars.
        sample_session_start, sample_session_end : str
            Custom session window as "HH:MM".
        stride : int
            Bar stride per day. stride=6 with 5min bars and 30min target
            gives non-overlapping windows. Default 6.
        patch_size : int
            Number of consecutive bars per patch. If >1, groups of
            ``patch_size`` bars are taken every ``stride`` bars, reducing
            overlap while keeping more data than stride alone.
        """
        df = self._tech_df
        if df is None or df.empty:
            raise RuntimeError("No data loaded. Call load() first.")

        feat_cols = self._tech_feature_cols
        target_col = self._target_col
        nb_ts_arr, nb_vals_arr = self._numbars_ts
        has_numbars = len(nb_ts_arr) > 0
        rasters = self._rasters
        macro_nodes = self._macro_nodes

        # Pre-extract arrays
        tech_arr = df[feat_cols].values.astype(np.float32)
        target_arr = df[target_col].values.astype(np.float32)
        target_class_arr = self._target_class_arr
        dates_arr = df.index.date
        df_index_values = df.index.values.astype("datetime64[ns]")

        # Session mask
        if sample_session_start is not None and sample_session_end is not None:
            t_start = time(*[int(x) for x in sample_session_start.split(":")])
            t_end = time(*[int(x) for x in sample_session_end.split(":")])
            bar_times = df.index.time
            session_mask = np.array(
                [(t >= t_start) & (t <= t_end) for t in bar_times], dtype=bool,
            )
        elif session_only and "is_active" in df.columns:
            session_mask = df["is_active"].values.astype(bool)
        else:
            session_mask = np.ones(len(df), dtype=bool)

        # Build macro date-keyed arrays for fast lookup
        # Convert macro DataFrames to date-keyed numpy arrays
        macro_daily: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, mdf in macro_nodes.items():
            m_dates = np.array([d.date() if hasattr(d, "date") else d for d in mdf.index])
            m_vals = mdf.values.astype(np.float32)
            macro_daily[name] = (m_dates, m_vals)

        # Unique trading dates for date indexing
        unique_dates = sorted(set(dates_arr))
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        # Collect eligible bars per day, then apply stride + patching
        day_bars: Dict[date, List[int]] = {}
        min_bar = max(tech_lookback, 1)

        for bar_idx in range(min_bar, len(df)):
            if not session_mask[bar_idx]:
                continue
            if np.isnan(target_arr[bar_idx]):
                continue
            if target_class_arr[bar_idx] < 0:
                continue  # Skip bars without valid quartile label
            d = dates_arr[bar_idx]
            day_bars.setdefault(d, []).append(bar_idx)

        # Apply stride + patching per day
        strided_bars: Dict[date, List[int]] = {}
        for d, indices in day_bars.items():
            selected = []
            i = 0
            while i < len(indices):
                # Take patch_size consecutive bars
                patch_end = min(i + patch_size, len(indices))
                selected.extend(indices[i:patch_end])
                i += stride
            strided_bars[d] = selected

        # Zero templates
        nb_zero_shape = (1,) + (nb_vals_arr.shape[1:] if len(nb_vals_arr) > 0 else (4, 32))
        raster_zero_shape = self._raster_shape or (12, 4, 64)

        # Stats
        n_total = sum(len(v) for v in strided_bars.values())
        n_skip_macro = 0
        n_skip_spatial = 0

        samples: List[Dict] = []

        for current_date, bar_indices in strided_bars.items():
            date_ord = date_to_idx.get(current_date, -1)
            if date_ord < 1:
                continue

            prev_date = unique_dates[date_ord - 1]

            # Check rasterized VPIN availability (previous day)
            raster = rasters.get(prev_date)

            # Build macro windows ending at prev_date (strictly causal)
            macro_window = self._get_macro_window(
                macro_daily, prev_date, self.macro_lookback,
            )
            if macro_window is None:
                n_skip_macro += 1
                continue

            # Batch searchsorted for NumberBars
            bar_ts_all = df_index_values[np.array(bar_indices)]
            if has_numbars:
                nb_cuts_all = np.searchsorted(nb_ts_arr, bar_ts_all, side="left")
            else:
                nb_cuts_all = np.zeros(len(bar_indices), dtype=np.intp)

            for bi, bar_idx in enumerate(bar_indices):
                start = bar_idx - tech_lookback
                tech_window = tech_arr[start:bar_idx]

                # NumberBars (timestamp-aligned, causal)
                if has_numbars:
                    nb_cut = int(nb_cuts_all[bi])
                    if nb_cut > 0:
                        nb_lo = max(0, nb_cut - numbars_lookback)
                        nb_data = nb_vals_arr[nb_lo:nb_cut]
                    else:
                        nb_data = np.zeros(nb_zero_shape, dtype=np.float32)
                else:
                    nb_data = np.zeros(nb_zero_shape, dtype=np.float32)

                sample = {
                    "tech_features": tech_window,
                    "numbars_recent": nb_data,
                    "raster_prev_day": raster,
                    "target": target_arr[bar_idx],
                    "target_class": target_class_arr[bar_idx],
                    "date": current_date,
                    "bar_idx": bar_idx,
                }

                # Add macro node tensors
                for name in NODE_ORDER:
                    if name in macro_window:
                        sample[f"macro_{name}"] = macro_window[name]

                samples.append(sample)

        print(f"  build_samples: {n_total} eligible bars, "
              f"{len(samples)} samples built, "
              f"skip_macro={n_skip_macro}, skip_spatial={n_skip_spatial}, "
              f"stride={stride}, patch_size={patch_size}")

        return samples

    @staticmethod
    def _get_macro_window(
        macro_daily: Dict[str, Tuple[np.ndarray, np.ndarray]],
        end_date: date,
        lookback: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract macro window ending at end_date for each node.

        Returns dict[node_name -> (lookback, F)] or None if insufficient data.
        """
        result: Dict[str, np.ndarray] = {}

        for name, (m_dates, m_vals) in macro_daily.items():
            # Find end_date position via searchsorted
            end_idx = np.searchsorted(m_dates, end_date, side="right")
            if end_idx < lookback:
                return None  # Not enough history for any node → skip
            start_idx = end_idx - lookback
            result[name] = m_vals[start_idx:end_idx]

        return result if result else None


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MacroGATDataset(Dataset):
    """PyTorch dataset for MacroGAT bar-level samples.

    Each sample returns:
      - tech_features: (tech_lookback, F_tech)
      - numbars_recent: (nb_T, C, bins)  [variable T]
      - raster_prev_day: (rT, rC, rBins)
      - macro_{node_name}: (macro_lookback, F_node) for each GAT node
      - target: raw 30min return (float)
      - target_class: quartile label 0-3 (long)
    """

    def __init__(
        self,
        samples: List[Dict],
        raster_shape: Tuple[int, ...] = (12, 4, 64),
        macro_lookback: int = 20,
        macro_node_dims: Optional[Dict[str, int]] = None,
    ):
        self.samples = samples
        self.raster_shape = raster_shape
        self.macro_lookback = macro_lookback
        self.macro_node_dims = macro_node_dims or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        out: Dict[str, torch.Tensor] = {
            "tech_features": torch.tensor(
                s["tech_features"], dtype=torch.float32,
            ),
            "target": torch.tensor(s["target"], dtype=torch.float32),
            "target_class": torch.tensor(s["target_class"], dtype=torch.long),
        }

        # NumberBars (variable length)
        out["numbars_recent"] = torch.tensor(
            np.asarray(s["numbars_recent"], dtype=np.float32),
            dtype=torch.float32,
        )

        # Rasterized VPIN (previous day)
        raster = s.get("raster_prev_day")
        if raster is not None:
            out["raster_prev_day"] = torch.tensor(
                np.asarray(raster, dtype=np.float32),
                dtype=torch.float32,
            )
        else:
            out["raster_prev_day"] = torch.zeros(
                self.raster_shape, dtype=torch.float32,
            )

        # Macro node tensors
        for name in NODE_ORDER:
            key = f"macro_{name}"
            if key in s:
                out[key] = torch.tensor(
                    s[key], dtype=torch.float32,
                )
            else:
                # Fallback zero tensor
                f_dim = self.macro_node_dims.get(name, 10)
                out[key] = torch.zeros(
                    self.macro_lookback, f_dim, dtype=torch.float32,
                )

        return out


# ---------------------------------------------------------------------------
# Collate — handles variable-length NumberBars
# ---------------------------------------------------------------------------

def macro_gat_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate MacroGAT samples, padding variable-length NumberBars."""
    out: Dict[str, torch.Tensor] = {}

    # Fixed-size tensors — stack directly
    fixed_keys = ["tech_features", "raster_prev_day", "target", "target_class"]
    for k in fixed_keys:
        out[k] = torch.stack([s[k] for s in batch])

    # Macro nodes — all fixed size (macro_lookback, F)
    for name in NODE_ORDER:
        key = f"macro_{name}"
        out[key] = torch.stack([s[key] for s in batch])

    # Variable-length NumberBars — pad T dimension
    max_nb = max(s["numbars_recent"].shape[0] for s in batch)
    nb_tail = batch[0]["numbars_recent"].shape[1:]
    padded_nb = torch.zeros(len(batch), max_nb, *nb_tail, dtype=torch.float32)
    nb_lens = torch.zeros(len(batch), dtype=torch.long)
    for i, s in enumerate(batch):
        T = s["numbars_recent"].shape[0]
        padded_nb[i, :T] = s["numbars_recent"]
        nb_lens[i] = T
    out["numbars_recent"] = padded_nb
    out["numbars_lens"] = nb_lens

    return out


# ---------------------------------------------------------------------------
# Unpack helper
# ---------------------------------------------------------------------------

def unpack_macro_gat_batch(
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Split collated batch into (intraday_inputs, macro_dict, targets, target_classes).

    Returns
    -------
    intraday : dict
        tech_features, numbars_recent, numbars_lens, raster_prev_day
    macro_dict : dict
        {node_name: (B, macro_lookback, F_node)} — ready for GAT forward()
    targets : tensor
        (B,) raw 30min forward returns (for profit-weighted loss)
    target_classes : tensor
        (B,) quartile class labels 0-3 (for cross-entropy)
    """
    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device) if device is not None else t

    intraday = {
        "tech_features": _to(batch["tech_features"]),
        "numbars_recent": _to(batch["numbars_recent"]),
        "numbars_lens": _to(batch["numbars_lens"]),
        "raster_prev_day": _to(batch["raster_prev_day"]),
    }

    macro_dict = {}
    for name in NODE_ORDER:
        key = f"macro_{name}"
        if key in batch:
            macro_dict[name] = _to(batch[key])

    targets = _to(batch["target"])
    target_classes = _to(batch["target_class"])
    return intraday, macro_dict, targets, target_classes


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_macro_gat_loaders(
    prep: MacroGATContinuousPrep,
    tech_lookback: int = 64,
    numbars_lookback: int = 8,
    stride: int = 6,
    patch_size: int = 1,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    val_cutoff_date: Optional[str] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    sample_session_start: Optional[str] = None,
    sample_session_end: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Build train/val DataLoaders from a prepared MacroGATContinuousPrep.

    Returns (train_loader, val_loader, dims_dict).
    """
    all_samples = prep.build_samples(
        tech_lookback=tech_lookback,
        numbars_lookback=numbars_lookback,
        session_only=True,
        sample_session_start=sample_session_start,
        sample_session_end=sample_session_end,
        stride=stride,
        patch_size=patch_size,
    )

    if not all_samples:
        raise ValueError("No samples built — check data availability")

    # Chronological split
    if val_cutoff_date:
        cutoff = pd.Timestamp(val_cutoff_date).date()
        train_samples = [s for s in all_samples if s["date"] < cutoff]
        val_samples = [s for s in all_samples if s["date"] >= cutoff]
    else:
        n_train = int(len(all_samples) * (1.0 - val_ratio))
        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:]

    dims = prep.get_dims()
    raster_shape = prep._raster_shape or (12, 4, 64)

    # Build node dim map for zero-fill fallback
    macro_node_dims = {}
    for name in NODE_ORDER:
        dim_key = f"macro_{name}"
        if dim_key in dims:
            macro_node_dims[name] = dims[dim_key]

    train_ds = MacroGATDataset(
        train_samples,
        raster_shape=raster_shape,
        macro_lookback=prep.macro_lookback,
        macro_node_dims=macro_node_dims,
    )
    val_ds = MacroGATDataset(
        val_samples,
        raster_shape=raster_shape,
        macro_lookback=prep.macro_lookback,
        macro_node_dims=macro_node_dims,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=macro_gat_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=macro_gat_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"  Dims: {dims}")

    return train_loader, val_loader, dims
