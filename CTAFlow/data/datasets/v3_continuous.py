"""
V3 Continuous Dataset for MMTFv3Core — bar-level intraday samples.

Aligns:
  - Technical features (from ContinuousIntradayPrep)
  - Spatial: NumberBars profiles + Rasterized VPIN (previous day)
  - Sequential: Tabular VPIN (recent bars)
  - AE input: Daily [return_1d, return_5d, return_21d, rv_1d]
  - Identity: ticker_id, asset_class_id, asset_subclass_id

Each sample is one intraday bar within an active session, producing
tensors matching MMTFv3Core.forward() signature.
"""
from __future__ import annotations

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
from CTAFlow.data.datasets.tft import build_ticker_registry

# ---------------------------------------------------------------------------
# NPZ helpers (mirror tft_aligned.py pattern)
# ---------------------------------------------------------------------------

def _load_npz_arrays(
    path: Union[str, Path],
    array_keys: Sequence[str] = ("tensor", "profiles", "data", "rasterized", "arr_0"),
    date_keys: Sequence[str] = ("dates", "date", "dates_str", "arr_1"),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    npz = np.load(str(path), allow_pickle=True)
    arr_key = next((k for k in array_keys if k in npz.files), None)
    if arr_key is None:
        raise ValueError(f"No array key in {path}. Available: {npz.files}")
    arr = np.asarray(npz[arr_key], dtype=np.float32)
    date_key = next((k for k in date_keys if k in npz.files), None)
    dates = np.asarray(npz[date_key]) if date_key else None
    return arr, dates


def _load_npz_date_keyed(path: Union[str, Path]) -> Dict[date, np.ndarray]:
    npz = np.load(str(path), allow_pickle=True)
    out: Dict[date, np.ndarray] = {}
    for key in npz.files:
        try:
            d = pd.Timestamp(key).date()
        except (ValueError, TypeError):
            continue
        out[d] = np.asarray(npz[key], dtype=np.float32)
    if not out:
        raise ValueError(f"No date-string keys in {path}. Keys: {npz.files[:10]}")
    return out


def _dates_to_python(raw: np.ndarray) -> List[date]:
    out = []
    for d in raw.flat:
        if isinstance(d, (pd.Timestamp, np.datetime64)):
            out.append(pd.Timestamp(d).date())
        else:
            out.append(pd.Timestamp(str(d)).date())
    return out


# ---------------------------------------------------------------------------
# AE daily feature computation
# ---------------------------------------------------------------------------

def compute_ae_daily_features(
    intraday_df: pd.DataFrame,
    target_time: str = "10:00",
    eps: float = 1e-8,
) -> Dict[date, np.ndarray]:
    """Compute daily AE features from intraday close prices.

    Returns dict[date -> np.array([ret_1d, ret_5d, ret_21d, rv_1d])].
    All values scaled to basis points and clipped.
    """
    df = intraday_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Standardize close column
    close_col = None
    for c in df.columns:
        if c.lower() in ("close", "last"):
            close_col = c
            break
    if close_col is None:
        raise KeyError(f"No 'Close'/'Last' column. Found: {list(df.columns)}")

    close = df[close_col].astype(float)

    # Extract close at target time each day
    tgt = time(int(target_time.split(":")[0]), int(target_time.split(":")[1]))
    mask = df.index.time == tgt
    daily = close[mask].copy()
    daily.index = daily.index.normalize()
    # Deduplicate (keep last)
    daily = daily[~daily.index.duplicated(keep="last")]

    if len(daily) < 22:
        return {}

    # Intraday log returns for RV
    logp = np.log(close.clip(lower=eps))
    intra_ret = logp.diff()
    daily_rv = intra_ret.groupby(df.index.normalize()).apply(
        lambda x: np.sqrt(np.nansum(x.values ** 2))
    )
    daily_rv.index = pd.to_datetime(daily_rv.index).normalize()

    # Daily returns
    ret_1d = daily.pct_change()
    ret_5d = daily.pct_change(5)
    ret_21d = daily.pct_change(21)

    out: Dict[date, np.ndarray] = {}
    for dt in daily.index:
        d = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
        r1 = ret_1d.get(dt, np.nan)
        r5 = ret_5d.get(dt, np.nan)
        r21 = ret_21d.get(dt, np.nan)
        rv = daily_rv.get(dt, np.nan)
        if any(np.isnan(x) for x in [r1, r5, r21, rv]):
            continue
        feat = np.array([
            np.clip(r1 * 100.0, -20, 20),
            np.clip(r5 * 100.0, -50, 50),
            np.clip(r21 * 100.0, -100, 100),
            np.clip(rv * 100.0, 0, 20),
        ], dtype=np.float32)
        out[d] = feat

    return out


# ---------------------------------------------------------------------------
# V3ContinuousPrep — multi-ticker alignment layer
# ---------------------------------------------------------------------------

class V3ContinuousPrep:
    """Prepare and align all modalities for MMTFv3Core across tickers.

    Per ticker, loads:
      - intraday.csv -> ContinuousIntradayPrep -> tech features + targets
      - profiles.npz -> daily NumberBars profiles
      - rasterized.npz -> daily rasterized VPIN
      - vpin.parquet -> intraday tabular VPIN (sequential branch)
      - intraday.csv -> AE daily features [ret_1d, ret_5d, ret_21d, rv_1d]

    All modalities aligned by date, NaNs forward-filled.
    """

    def __init__(
        self,
        tickers: List[str],
        sessions: Optional[Sequence[SessionSpec]] = None,
        bar_minutes: int = 5,
        target_horizon_minutes: int = 30,
        ae_window: int = 21,
        ae_target_time: str = "10:00",
    ):
        self.tickers = sorted(tickers)
        self.sessions = sessions or [SessionSpec("USA", "08:30", "16:00")]
        self.bar_minutes = bar_minutes
        self.target_steps = target_horizon_minutes // bar_minutes
        self.ae_window = ae_window
        self.ae_target_time = ae_target_time

        self.registry = build_ticker_registry(self.tickers)
        self.prep = ContinuousIntradayPrep(
            sessions=self.sessions, bar_minutes=bar_minutes,
        )

        # Per-ticker storage (populated by load_ticker / from_directories)
        self._tech_dfs: Dict[str, pd.DataFrame] = {}
        self._tech_feature_cols: List[str] = []
        self._target_col: str = ""
        self._profiles: Dict[str, Dict[date, np.ndarray]] = {}
        self._rasters: Dict[str, Dict[date, np.ndarray]] = {}
        self._seq_vpin: Dict[str, pd.DataFrame] = {}
        self._ae_features: Dict[str, Dict[date, np.ndarray]] = {}

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def n_asset_classes(self) -> int:
        return max(m.asset_class_id for m in self.registry.values()) + 1

    @property
    def n_asset_subclasses(self) -> int:
        return max(m.asset_subclass_id for m in self.registry.values()) + 1

    @property
    def f_tech(self) -> int:
        return len(self._tech_feature_cols)

    def load_ticker(
        self,
        ticker: str,
        root_dir: Union[str, Path],
        intraday_file: str = "intraday.csv",
        profile_file: str = "profiles.npz",
        raster_file: str = "rasterized.npz",
        vpin_file: str = "vpin.parquet",
    ) -> int:
        """Load and prepare all data for one ticker.

        Returns number of valid sample bars.
        """
        root = Path(root_dir) / ticker

        # 1. Intraday tech features
        intra_path = root / intraday_file
        raw_df = pd.read_csv(str(intra_path), parse_dates=True, index_col=0)
        if not isinstance(raw_df.index, pd.DatetimeIndex):
            raw_df.index = pd.to_datetime(raw_df.index)

        df_out, train_mask, target_cols = self.prep.prepare(
            raw_df,
            steps_60m=self.target_steps,
            keep_only_active=False,
            apply_scaling=True,
            scale_to_basis_points=True,
        )

        # Resolve feature columns
        feat_cols = self.prep.get_feature_cols(
            steps_60m=self.target_steps,
            bar_minutes=self.bar_minutes,
        )
        feat_cols = [c for c in feat_cols if c in df_out.columns]

        # Target: 30min forward return (single step)
        target_col = target_cols[-1]  # y_fwd_{target_steps} = 30min

        # FFill NaNs in features
        df_out[feat_cols] = df_out[feat_cols].ffill().bfill()
        df_out[feat_cols] = df_out[feat_cols].fillna(0.0)

        self._tech_dfs[ticker] = df_out
        self._tech_feature_cols = feat_cols
        self._target_col = target_col

        # 2. Profiles (NumberBars)
        prof_path = root / profile_file
        if prof_path.exists():
            self._profiles[ticker] = self._load_spatial_npz(prof_path)
        else:
            self._profiles[ticker] = {}

        # 3. Rasterized VPIN
        rast_path = root / raster_file
        if rast_path.exists():
            self._rasters[ticker] = self._load_spatial_npz(rast_path)
        else:
            self._rasters[ticker] = {}

        # 4. Sequential VPIN
        vpin_path = root / vpin_file
        if vpin_path.exists():
            vpin_df = pd.read_parquet(str(vpin_path))
            if not isinstance(vpin_df.index, pd.DatetimeIndex):
                vpin_df.index = pd.to_datetime(vpin_df.index)
            vpin_df = vpin_df.select_dtypes(include="number").astype(np.float32)
            vpin_df = vpin_df.ffill().bfill().fillna(0.0)
            self._seq_vpin[ticker] = vpin_df
        else:
            self._seq_vpin[ticker] = pd.DataFrame()

        # 5. AE daily features
        self._ae_features[ticker] = compute_ae_daily_features(
            raw_df, target_time=self.ae_target_time,
        )

        n_valid = train_mask.sum() if hasattr(train_mask, "sum") else 0
        return int(n_valid)

    @staticmethod
    def _load_spatial_npz(path: Path) -> Dict[date, np.ndarray]:
        """Load NPZ as date->array dict (handles both indexed and date-keyed)."""
        try:
            arr, dates_raw = _load_npz_arrays(path)
            if dates_raw is not None:
                dates = _dates_to_python(dates_raw)
                return {d: arr[i] for i, d in enumerate(dates) if i < len(arr)}
            else:
                return {}
        except ValueError:
            return _load_npz_date_keyed(path)

    @classmethod
    def from_directories(
        cls,
        root_dir: Union[str, Path],
        tickers: List[str],
        sessions: Optional[Sequence[SessionSpec]] = None,
        bar_minutes: int = 5,
        target_horizon_minutes: int = 30,
        ae_window: int = 21,
        ae_target_time: str = "10:00",
        intraday_file: str = "intraday.csv",
        profile_file: str = "profiles.npz",
        raster_file: str = "rasterized.npz",
        vpin_file: str = "vpin.parquet",
    ) -> "V3ContinuousPrep":
        """Load all tickers from standard directory layout."""
        obj = cls(
            tickers=tickers,
            sessions=sessions,
            bar_minutes=bar_minutes,
            target_horizon_minutes=target_horizon_minutes,
            ae_window=ae_window,
            ae_target_time=ae_target_time,
        )
        for ticker in obj.tickers:
            n = obj.load_ticker(
                ticker, root_dir,
                intraday_file=intraday_file,
                profile_file=profile_file,
                raster_file=raster_file,
                vpin_file=vpin_file,
            )
            print(f"  [{ticker}] {n} valid bars, "
                  f"profiles={len(obj._profiles.get(ticker, {}))}, "
                  f"rasters={len(obj._rasters.get(ticker, {}))}, "
                  f"ae_dates={len(obj._ae_features.get(ticker, {}))}")
        return obj

    def get_dims(self) -> Dict[str, int]:
        """Return feature dimensions for model construction."""
        f_seq = 0
        for vpin_df in self._seq_vpin.values():
            if len(vpin_df.columns) > 0:
                f_seq = len(vpin_df.columns)
                break
        return {
            "f_tech": len(self._tech_feature_cols),
            "f_seq": f_seq,
            "f_ae": 4,  # [ret_1d, ret_5d, ret_21d, rv_1d]
        }

    def build_samples(
        self,
        tech_lookback: int = 64,
        seq_lookback_bars: int = 12,
        session_only: bool = True,
        sample_session: Optional[str] = None,
        sample_session_start: Optional[str] = None,
        sample_session_end: Optional[str] = None,
        stride: int = 1,
    ) -> List[Dict]:
        """Build flat list of sample dicts for V3ContinuousDataset.

        Each sample is anchored at one intraday bar and contains
        lookback windows of tech features, sequential VPIN, plus
        the previous day's spatial data and AE window.

        Parameters
        ----------
        tech_lookback : int
            Number of bars in the tech feature lookback window.
        seq_lookback_bars : int
            Max tabular VPIN bars per sample.
        session_only : bool
            If True and no specific session filter is set, uses the
            ``is_active`` column (union of all prep sessions).
        sample_session : str, optional
            Filter to a named session. Recognized names:
            ``"usa"`` → is_usa, ``"london"`` → is_london,
            ``"overlap"`` → is_session_overlap.
            Overrides ``session_only`` when set.
        sample_session_start : str, optional
            Custom session start time as ``"HH:MM"``. Used together
            with ``sample_session_end`` to define an arbitrary window.
            Overrides ``sample_session`` and ``session_only``.
        sample_session_end : str, optional
            Custom session end time as ``"HH:MM"``.
        stride : int
            Take every *stride*-th eligible bar per day. Use stride=6
            with 5-min bars and a 30-min target to get non-overlapping
            samples.  Default 1 (every bar).
        """
        samples = []
        for ticker in self.tickers:
            df = self._tech_dfs.get(ticker)
            if df is None or df.empty:
                continue

            meta = self.registry[ticker]
            feat_cols = self._tech_feature_cols
            target_col = self._target_col
            profiles = self._profiles.get(ticker, {})
            rasters = self._rasters.get(ticker, {})
            ae_feats = self._ae_features.get(ticker, {})
            vpin_df = self._seq_vpin.get(ticker, pd.DataFrame())

            # Pre-extract arrays
            tech_arr = df[feat_cols].values.astype(np.float32)
            target_arr = df[target_col].values.astype(np.float32)
            dates = df.index.date
            bar_times = df.index.time

            # --- Session mask ---
            if sample_session_start is not None and sample_session_end is not None:
                # Custom time window
                t_start = time(
                    int(sample_session_start.split(":")[0]),
                    int(sample_session_start.split(":")[1]),
                )
                t_end = time(
                    int(sample_session_end.split(":")[0]),
                    int(sample_session_end.split(":")[1]),
                )
                session_mask = np.array(
                    [(t >= t_start) & (t <= t_end) for t in bar_times],
                    dtype=bool,
                )
            elif sample_session is not None:
                col_map = {
                    "usa": "is_usa",
                    "london": "is_london",
                    "overlap": "is_session_overlap",
                }
                col = col_map.get(sample_session.lower())
                if col is None or col not in df.columns:
                    raise ValueError(
                        f"Unknown sample_session={sample_session!r}. "
                        f"Use 'usa', 'london', 'overlap', or set "
                        f"sample_session_start/end for custom windows."
                    )
                session_mask = df[col].values.astype(bool)
            elif session_only:
                session_mask = (
                    df["is_active"].values.astype(bool)
                    if "is_active" in df.columns
                    else np.ones(len(df), dtype=bool)
                )
            else:
                session_mask = np.ones(len(df), dtype=bool)

            # Group VPIN by date for quick lookup
            vpin_by_date: Dict[date, np.ndarray] = {}
            if not vpin_df.empty:
                for d, grp in vpin_df.groupby(vpin_df.index.date):
                    vpin_by_date[d] = grp.values.astype(np.float32)

            # Sorted unique dates for AE window
            unique_dates = sorted(set(dates))
            date_to_idx = {d: i for i, d in enumerate(unique_dates)}

            # Collect eligible bar indices per day then apply stride
            day_bars: Dict[date, List[int]] = {}
            for bar_idx in range(tech_lookback, len(df)):
                if not session_mask[bar_idx]:
                    continue
                if np.isnan(target_arr[bar_idx]):
                    continue
                d = dates[bar_idx]
                day_bars.setdefault(d, []).append(bar_idx)

            # Apply stride per day
            if stride > 1:
                for d in day_bars:
                    day_bars[d] = day_bars[d][::stride]

            for current_date, bar_indices in day_bars.items():
                date_ord = date_to_idx.get(current_date, -1)
                prev_date = unique_dates[date_ord - 1] if date_ord > 0 else None
                if prev_date is None:
                    continue
                if prev_date not in profiles and prev_date not in rasters:
                    continue

                # AE window: ae_window consecutive days ending at prev_date
                ae_end_idx = date_ord
                ae_start_idx = ae_end_idx - self.ae_window
                if ae_start_idx < 0:
                    continue
                ae_dates = unique_dates[ae_start_idx:ae_end_idx]
                if not all(d in ae_feats for d in ae_dates):
                    continue
                ae_input = np.stack([ae_feats[d] for d in ae_dates])

                profile = profiles.get(prev_date)
                raster = rasters.get(prev_date)

                # Sequential VPIN for this day
                seq_day = vpin_by_date.get(
                    current_date,
                    np.zeros(
                        (1, max(1, vpin_df.shape[1] if not vpin_df.empty else 1)),
                        dtype=np.float32,
                    ),
                )

                for bar_idx in bar_indices:
                    start = bar_idx - tech_lookback
                    tech_window = tech_arr[start:bar_idx]
                    tech_len = tech_lookback

                    # Take last seq_lookback_bars of VPIN
                    seq_data = seq_day
                    if len(seq_data) > seq_lookback_bars:
                        seq_data = seq_data[-seq_lookback_bars:]
                    seq_len = len(seq_data)

                    samples.append({
                        "tech_features": tech_window,
                        "tech_len": tech_len,
                        "numbars_recent": profile,
                        "vpin_raster_recent": raster,
                        "seq_vpin": seq_data,
                        "seq_vpin_len": seq_len,
                        "ae_input": ae_input,
                        "ticker_id": meta.ticker_id,
                        "asset_class_id": meta.asset_class_id,
                        "asset_subclass_id": meta.asset_subclass_id,
                        "target": target_arr[bar_idx],
                        "ticker": ticker,
                        "date": current_date,
                    })

        return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

# Default spatial shapes (used as zero-fill when data missing for a day)
_DEFAULT_PROFILE_SHAPE = (4, 129, 4)   # (T_max, bins, channels)
_DEFAULT_RASTER_SHAPE = (24, 4, 64)    # (time, channels, bins)


class V3ContinuousDataset(Dataset):
    """PyTorch dataset producing bar-level samples for MMTFv3Core."""

    def __init__(
        self,
        samples: List[Dict],
        default_profile_shape: Tuple[int, ...] = _DEFAULT_PROFILE_SHAPE,
        default_raster_shape: Tuple[int, ...] = _DEFAULT_RASTER_SHAPE,
    ):
        self.samples = samples
        self.default_profile_shape = default_profile_shape
        self.default_raster_shape = default_raster_shape

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        tech = torch.tensor(s["tech_features"], dtype=torch.float32)
        tech_len = torch.tensor(s["tech_len"], dtype=torch.long)

        # Spatial — use zeros if missing
        if s["numbars_recent"] is not None:
            nb = torch.tensor(s["numbars_recent"], dtype=torch.float32)
        else:
            nb = torch.zeros(self.default_profile_shape, dtype=torch.float32)

        if s["vpin_raster_recent"] is not None:
            vr = torch.tensor(s["vpin_raster_recent"], dtype=torch.float32)
        else:
            vr = torch.zeros(self.default_raster_shape, dtype=torch.float32)

        seq = torch.tensor(s["seq_vpin"], dtype=torch.float32)
        seq_len = torch.tensor(s["seq_vpin_len"], dtype=torch.long)

        ae = torch.tensor(s["ae_input"], dtype=torch.float32)

        target = torch.tensor(s["target"], dtype=torch.float32)

        return {
            "tech_features": tech,
            "tech_lens": tech_len,
            "numbars_recent": nb,
            "vpin_raster_recent": vr,
            "seq_vpin": seq,
            "seq_vpin_lens": seq_len,
            "ae_input": ae,
            "ticker_id": torch.tensor(s["ticker_id"], dtype=torch.long),
            "asset_class_id": torch.tensor(s["asset_class_id"], dtype=torch.long),
            "asset_subclass_id": torch.tensor(s["asset_subclass_id"], dtype=torch.long),
            "target": target,
        }


# ---------------------------------------------------------------------------
# Collate + unpack
# ---------------------------------------------------------------------------

def v3_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate V3 samples, padding variable-length seq_vpin."""
    # Fixed-size tensors — just stack
    fixed_keys = [
        "tech_features", "tech_lens",
        "numbars_recent", "vpin_raster_recent",
        "ae_input",
        "ticker_id", "asset_class_id", "asset_subclass_id",
        "target",
    ]
    out = {}
    for k in fixed_keys:
        out[k] = torch.stack([s[k] for s in batch])

    # Variable-length: seq_vpin — pad to max length
    seq_lens = torch.stack([s["seq_vpin_lens"] for s in batch])
    max_seq = max(s["seq_vpin"].shape[0] for s in batch)
    f_seq = batch[0]["seq_vpin"].shape[1] if batch[0]["seq_vpin"].dim() == 2 else 1

    padded_seq = torch.zeros(len(batch), max_seq, f_seq, dtype=torch.float32)
    for i, s in enumerate(batch):
        L = s["seq_vpin"].shape[0]
        padded_seq[i, :L] = s["seq_vpin"]

    out["seq_vpin"] = padded_seq
    out["seq_vpin_lens"] = seq_lens

    return out


_MODEL_KEYS = [
    "tech_features", "tech_lens",
    "numbars_recent", "vpin_raster_recent",
    "seq_vpin", "seq_vpin_lens",
    "ae_input",
    "ticker_id", "asset_class_id", "asset_subclass_id",
]


def unpack_v3_batch(
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Split collated batch into (model_inputs, targets) for train_epoch_v3."""
    inputs = {}
    for k in _MODEL_KEYS:
        if k in batch:
            t = batch[k]
            inputs[k] = t.to(device) if device is not None else t
    targets = batch["target"]
    if device is not None:
        targets = targets.to(device)
    return inputs, targets


def build_v3_loaders(
    prep: V3ContinuousPrep,
    tech_lookback: int = 64,
    seq_lookback_bars: int = 12,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    val_cutoff_date: Optional[str] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    sample_session: Optional[str] = None,
    sample_session_start: Optional[str] = None,
    sample_session_end: Optional[str] = None,
    stride: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from a prepared V3ContinuousPrep.

    Parameters
    ----------
    sample_session : str, optional
        Named session filter: ``"usa"``, ``"london"``, ``"overlap"``.
    sample_session_start, sample_session_end : str, optional
        Custom time window as ``"HH:MM"`` strings. Overrides
        ``sample_session``.
    stride : int
        Bar stride per day to reduce target overlap. stride=6 with
        5-min bars yields non-overlapping 30-min windows.
    """
    all_samples = prep.build_samples(
        tech_lookback=tech_lookback,
        seq_lookback_bars=seq_lookback_bars,
        session_only=True,
        sample_session=sample_session,
        sample_session_start=sample_session_start,
        sample_session_end=sample_session_end,
        stride=stride,
    )
    if not all_samples:
        raise ValueError("No valid samples produced. Check data availability.")

    # Date-based split
    all_dates = sorted(set(s["date"] for s in all_samples))
    if val_cutoff_date is not None:
        cutoff = pd.Timestamp(val_cutoff_date).date()
    else:
        n_val = max(1, int(len(all_dates) * val_ratio))
        cutoff = all_dates[-n_val]

    train_samples = [s for s in all_samples if s["date"] < cutoff]
    val_samples = [s for s in all_samples if s["date"] >= cutoff]

    print(f"Samples: {len(train_samples)} train, {len(val_samples)} val "
          f"(cutoff={cutoff}, {len(all_dates)} total days)")

    train_ds = V3ContinuousDataset(train_samples)
    val_ds = V3ContinuousDataset(val_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        collate_fn=v3_collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=v3_collate_fn, num_workers=num_workers,
    )
    return train_loader, val_loader
