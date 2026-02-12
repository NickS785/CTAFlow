from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Sequence, Any, Union


def _build_sample_mask(
    df: pd.DataFrame,
    target_cols: List[str],
    lookback: int,
    sample_mode: str = "active",
    allow_overlap: bool = True,
) -> np.ndarray:
    """
    Returns boolean mask of eligible rows by index position.
    Requires: df has 'session_code' (0 none, 1 london, 2 usa, 3 overlap) and/or 'is_active'.
    """
    n = len(df)
    ok = np.ones(n, dtype=bool)

    # must have targets
    for c in target_cols:
        ok &= df[c].notna().to_numpy()

    # must have enough history
    ok[:lookback - 1] = False

    # session filtering
    if sample_mode == "all":
        return ok

    if sample_mode == "active":
        # London or USA including overlap
        if "is_active" in df.columns:
            ok &= df["is_active"].astype(bool).to_numpy()
        else:
            ok &= (df["session_code"].to_numpy() > 0)

    elif sample_mode == "overlap":
        ok &= (df["session_code"].to_numpy() == 3)

    elif sample_mode == "london":
        sc = df["session_code"].to_numpy()
        if allow_overlap:
            ok &= (sc == 1) | (sc == 3)
        else:
            ok &= (sc == 1)

    elif sample_mode == "usa":
        sc = df["session_code"].to_numpy()
        if allow_overlap:
            ok &= (sc == 2) | (sc == 3)
        else:
            ok &= (sc == 2)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    return ok


def _contiguous_5m_check(idx: pd.DatetimeIndex, lookback: int, bar_minutes: int = 5) -> np.ndarray:
    """
    Precompute whether each row i has a fully contiguous lookback window [i-lookback+1..i]
    at bar_minutes frequency.
    """
    n = len(idx)
    ok = np.ones(n, dtype=bool)
    ok[:lookback - 1] = False

    # diff in minutes between consecutive rows
    diffs = idx.to_series().diff().dt.total_seconds().div(60.0).to_numpy()
    diffs[0] = bar_minutes  # dummy

    # a window is contiguous if all diffs inside window are == bar_minutes
    # i is valid if diffs[i-lookback+1 .. i] are all bar_minutes (except the first element which corresponds to gap into window)
    good_step = (diffs == bar_minutes)
    # rolling min over good_step
    # for i, need all good_step in [i-lookback+1..i] True
    # easiest: cumulative sum of bad steps
    bad = (~good_step).astype(np.int32)
    cbad = np.cumsum(bad)
    for i in range(lookback - 1, n):
        left = i - lookback + 1
        bad_in_window = cbad[i] - (cbad[left - 1] if left > 0 else 0)
        ok[i] = (bad_in_window == 0)
    return ok


def _to_class_labels(values: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Convert continuous values into ordinal classes by threshold cuts."""
    arr = np.asarray(values, dtype=np.float32)
    cuts = sorted(float(t) for t in thresholds)
    labels = np.zeros(arr.shape[0], dtype=np.int64)
    for i, threshold in enumerate(cuts):
        labels[arr > threshold] = i + 1
    return labels


class ContinuousWindowDataset(Dataset):
    """
    Takes a continuous dataframe and produces:
      x: [lookback, n_features]
      y: [n_targets]  (e.g. 12 horizons for next 60m)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        lookback: int = 256,
        sample_mode: str = "active",        # "all"|"active"|"london"|"usa"|"overlap"
        allow_overlap: bool = True,
        enforce_contiguous: bool = True,
        bar_minutes: int = 5,
        return_meta: bool = False,
    ):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.lookback = int(lookback)
        self.return_meta = return_meta

        base_mask = _build_sample_mask(
            df=df,
            target_cols=target_cols,
            lookback=self.lookback,
            sample_mode=sample_mode,
            allow_overlap=allow_overlap,
        )

        if enforce_contiguous:
            cont = _contiguous_5m_check(df.index, lookback=self.lookback, bar_minutes=bar_minutes)
            base_mask &= cont

        self.sample_pos = np.flatnonzero(base_mask)
        if len(self.sample_pos) == 0:
            raise ValueError("No eligible samples found. Check your sample_mode / targets / lookback / contiguity.")

        # cache arrays for speed
        self.X = df[feature_cols].to_numpy(dtype=np.float32)
        self.Y = df[target_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sample_pos)

    def __getitem__(self, idx: int):
        i = self.sample_pos[idx]
        j0 = i - self.lookback + 1
        x = torch.from_numpy(self.X[j0:i + 1])     # [L, F]
        y = torch.from_numpy(self.Y[i])            # [T]

        if not self.return_meta:
            return x, y

        ts = self.df.index[i]
        sess = int(self.df["session_code"].iloc[i]) if "session_code" in self.df.columns else -1
        return x, y, {"timestamp": ts, "session_code": sess}


def _normalize_ts_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize index to tz-naive UTC for robust cross-source alignment."""
    ts = pd.to_datetime(idx)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(ts)


def _ffill_bfill_raster(day_raster: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Forward/backward fill missing raster rows inside one day."""
    out = day_raster.copy()
    t = out.shape[0]
    if t == 0:
        return out
    if not np.any(valid_mask):
        return out

    first = int(np.flatnonzero(valid_mask)[0])
    last = int(np.flatnonzero(valid_mask)[-1])

    # Fill leading gap from first valid
    for i in range(0, first):
        out[i] = out[first]

    # Fill trailing gap from last valid
    for i in range(last + 1, t):
        out[i] = out[last]

    # Forward-fill interior gaps
    prev = first
    for i in range(first + 1, t):
        if valid_mask[i]:
            prev = i
        else:
            out[i] = out[prev]
    return out


def _build_session_index(
    day: pd.Timestamp,
    session_start: str,
    session_end: str,
    freq: str,
) -> pd.DatetimeIndex:
    """Build per-day expected timestamps for the selected session window."""
    s = pd.Timestamp(f"{day.date()} {session_start}")
    e = pd.Timestamp(f"{day.date()} {session_end}")
    if e <= s:
        e = e + pd.Timedelta(days=1)
    return pd.date_range(s, e, freq=freq)


def _resample_frame(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample with OHLCV-aware aggregation and last for other numeric columns."""
    agg = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "open":
            agg[col] = "first"
        elif lc == "high":
            agg[col] = "max"
        elif lc == "low":
            agg[col] = "min"
        elif lc in ("close", "last"):
            agg[col] = "last"
        elif lc in ("volume", "vol"):
            agg[col] = "sum"
        else:
            agg[col] = "last"
    return df.resample(rule).agg(agg).sort_index()


class ContinuousRasterAlignedDataset(Dataset):
    """
    Continuous dataset for CMDMamba-style long/short streams with raster alignment.

    Produces per-sample:
      x_short:    [T_short, C, H]
      t_short:    [T_short, F_time]
      x_long:     [T_long, F_long]
      t_long:     [T_long, F_time]
      short_mask: [T_short]
      y:          [F_target]
    """

    DEFAULT_TIME_COLS = (
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "is_london_session",
        "is_usa_session",
        "is_overlap_session",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        raster_npz_path: str,
        feature_cols: List[str],
        target_cols: List[str],
        long_lookback: int = 256,
        short_len_range: Tuple[int, int] = (32, 128),
        resample_rule: str = "15min",
        session_start: str = "02:00",
        session_end: str = "11:00",
        sample_mode: str = "active",
        allow_overlap: bool = True,
        time_feature_cols: Optional[List[str]] = None,
        max_missing_rows_per_day: int = 6,
        max_missing_ratio_per_day: float = 0.20,
        random_short_len: bool = True,
        return_meta: bool = False,
        sample_stride: int = 1,
    ):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df must have DatetimeIndex")
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.long_lookback = int(long_lookback)
        self.short_lo = int(short_len_range[0])
        self.short_hi = int(short_len_range[1])
        self.random_short_len = bool(random_short_len)
        self.return_meta = bool(return_meta)

        if self.short_lo <= 0 or self.short_hi < self.short_lo:
            raise ValueError(f"Invalid short_len_range={short_len_range}")
        if self.long_lookback <= 0:
            raise ValueError("long_lookback must be > 0")

        work = df.copy()
        work.index = _normalize_ts_index(work.index)
        work = work.sort_index()
        work = _resample_frame(work, resample_rule)

        # Keep only required columns and make sure they exist
        missing_feats = [c for c in self.feature_cols if c not in work.columns]
        missing_tgts = [c for c in self.target_cols if c not in work.columns]
        if missing_feats:
            raise KeyError(f"Missing feature columns in df: {missing_feats}")
        if missing_tgts:
            raise KeyError(f"Missing target columns in df: {missing_tgts}")

        if time_feature_cols is None:
            self.time_feature_cols = [c for c in self.DEFAULT_TIME_COLS if c in work.columns]
        else:
            self.time_feature_cols = [c for c in time_feature_cols if c in work.columns]
        if not self.time_feature_cols:
            # Fallback to a constant channel if time features were not provided.
            work["_time_const"] = 1.0
            self.time_feature_cols = ["_time_const"]

        # Raster
        z = np.load(raster_npz_path, allow_pickle=False)
        if "data" not in z.files or "idx" not in z.files:
            raise KeyError(f"{raster_npz_path} must contain keys 'data' and 'idx'")
        raster = z["data"].astype(np.float32)
        ridx = pd.DatetimeIndex(pd.to_datetime(z["idx"]))
        ridx = _normalize_ts_index(ridx)

        if raster.ndim != 3:
            raise ValueError(f"Expected raster shape (N,C,H), got {raster.shape}")
        if raster.shape[0] != len(ridx):
            raise ValueError("Raster length does not match idx length")

        # Sort raster by timestamp (just in case)
        sort_idx = np.argsort(ridx.values)
        ridx = ridx[sort_idx]
        raster = raster[sort_idx]
        raster_index = pd.DatetimeIndex(ridx)

        # Per-day alignment and filtering
        common_days = sorted(set(work.index.normalize()) & set(raster_index.normalize()))
        aligned_frames: List[pd.DataFrame] = []
        aligned_rasters: List[np.ndarray] = []
        dropped_days: Dict[str, str] = {}

        for day in common_days:
            expected = _build_session_index(day, session_start=session_start, session_end=session_end, freq=resample_rule)
            if len(expected) == 0:
                continue

            day_raw = work.reindex(expected)
            feat_missing = day_raw[self.feature_cols].isna().all(axis=1)
            n_feat_missing = int(feat_missing.sum())

            # Fill features for continuity; keep targets unfilled.
            day_filled = day_raw.copy()
            day_filled[self.feature_cols] = day_filled[self.feature_cols].ffill().bfill()
            day_filled[self.time_feature_cols] = day_filled[self.time_feature_cols].ffill().bfill()
            for c in self.target_cols:
                day_filled[c] = day_raw[c]

            ridx_pos = raster_index.get_indexer(expected)
            valid = ridx_pos >= 0
            n_raster_missing = int((~valid).sum())

            if not np.any(valid):
                dropped_days[str(day.date())] = "no_raster_rows"
                continue

            tlen = len(expected)
            cdim = raster.shape[1]
            hdim = raster.shape[2]
            day_raster = np.zeros((tlen, cdim, hdim), dtype=np.float32)
            day_raster[valid] = raster[ridx_pos[valid]]
            day_raster = _ffill_bfill_raster(day_raster, valid_mask=valid)

            miss_limit = max(max_missing_rows_per_day, int(np.floor(max_missing_ratio_per_day * tlen)))
            if n_feat_missing > miss_limit:
                dropped_days[str(day.date())] = f"feature_missing={n_feat_missing}"
                continue
            if n_raster_missing > miss_limit:
                dropped_days[str(day.date())] = f"raster_missing={n_raster_missing}"
                continue

            aligned_frames.append(day_filled)
            aligned_rasters.append(day_raster)

        if not aligned_frames:
            raise ValueError("No aligned days left after missing-data filtering.")

        self.df = pd.concat(aligned_frames).sort_index()
        self.raster = np.concatenate(aligned_rasters, axis=0).astype(np.float32)
        if len(self.df) != self.raster.shape[0]:
            raise RuntimeError("Aligned feature/raster lengths do not match after day concat.")

        # Final fill pass for features/time only (targets remain untouched)
        self.df[self.feature_cols] = self.df[self.feature_cols].ffill().bfill()
        self.df[self.time_feature_cols] = self.df[self.time_feature_cols].ffill().bfill()

        self.X_long = np.nan_to_num(self.df[self.feature_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.T = np.nan_to_num(self.df[self.time_feature_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.Y = self.df[self.target_cols].to_numpy(dtype=np.float32)

        # Sample positions
        lookback = max(self.long_lookback, self.short_hi)
        sample_mask = _build_sample_mask(
            df=self.df,
            target_cols=self.target_cols,
            lookback=lookback,
            sample_mode=sample_mode,
            allow_overlap=allow_overlap,
        )
        self.sample_pos = np.flatnonzero(sample_mask)
        if len(self.sample_pos) == 0:
            raise ValueError("No eligible samples after alignment/masking.")

        # Per-day striding: take every N-th sample within each calendar day
        # to reduce autocorrelation and cut dataset size.
        self.sample_stride = int(max(1, sample_stride))
        if self.sample_stride > 1:
            sample_dates = self.df.index[self.sample_pos].normalize()
            thinned = []
            for day in sample_dates.unique():
                day_mask = sample_dates == day
                day_positions = self.sample_pos[day_mask]
                thinned.append(day_positions[:: self.sample_stride])
            self.sample_pos = np.concatenate(thinned)

        self.alignment_stats = {
            "num_days_in_common": len(common_days),
            "num_days_kept": len(aligned_frames),
            "num_days_dropped": len(dropped_days),
            "dropped_days": dropped_days,
            "num_samples": int(len(self.sample_pos)),
            "time_feature_cols": list(self.time_feature_cols),
        }

    def __len__(self) -> int:
        return len(self.sample_pos)

    def __getitem__(self, idx: int):
        i = int(self.sample_pos[idx])
        if self.random_short_len and self.short_hi > self.short_lo:
            t_short = int(np.random.randint(self.short_lo, self.short_hi + 1))
        else:
            t_short = self.short_hi

        s0 = i - t_short + 1
        l0 = i - self.long_lookback + 1

        x_short = torch.from_numpy(self.raster[s0:i + 1].copy())  # [T_short,C,H]
        t_short_feats = torch.from_numpy(self.T[s0:i + 1].copy())  # [T_short,Ft]
        x_long = torch.from_numpy(self.X_long[l0:i + 1].copy())  # [T_long,F]
        t_long_feats = torch.from_numpy(self.T[l0:i + 1].copy())  # [T_long,Ft]
        y = torch.from_numpy(self.Y[i].copy())  # [targets]
        short_mask = torch.ones((t_short,), dtype=torch.float32)

        if not self.return_meta:
            return x_short, t_short_feats, x_long, t_long_feats, short_mask, y

        meta = {
            "timestamp": self.df.index[i],
            "session_code": int(self.df["session_code"].iloc[i]) if "session_code" in self.df.columns else -1,
            "sample_pos": i,
        }
        return x_short, t_short_feats, x_long, t_long_feats, short_mask, y, meta


class ContinuousRasterAlignedTaskDataset(ContinuousRasterAlignedDataset):
    """
    Task-oriented extension of ContinuousRasterAlignedDataset.

    Adds optional on-the-fly classification targets while preserving the same
    sample/collate structure expected by collate_continuous_raster.
    """

    def __init__(
        self,
        *args,
        classification: bool = False,
        classification_target: Optional[Union[str, int]] = None,
        classification_thresholds: Sequence[float] = (-0.001, 0.001),
        classification_target_is_label: bool = False,
        include_regression_target: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classification = bool(classification)
        self.classification_target = classification_target
        self.classification_thresholds = tuple(float(t) for t in classification_thresholds)
        self.classification_target_is_label = bool(classification_target_is_label)
        self.include_regression_target = bool(include_regression_target)

        if self.include_regression_target and not self.classification:
            raise ValueError("include_regression_target=True requires classification=True.")

        self._class_target_index: Optional[int] = None
        self._class_targets: Optional[np.ndarray] = None

        if self.classification:
            self._class_target_index = self._resolve_target_index(classification_target)
            raw = self.Y[:, self._class_target_index]
            if self.classification_target_is_label:
                labels = np.rint(raw).astype(np.int64)
            else:
                labels = _to_class_labels(raw, self.classification_thresholds)
            self._class_targets = labels

            self.alignment_stats["classification"] = {
                "enabled": True,
                "target_col": self.target_cols[self._class_target_index],
                "target_index": int(self._class_target_index),
                "target_is_label": bool(self.classification_target_is_label),
                "thresholds": list(self.classification_thresholds),
                "include_regression_target": bool(self.include_regression_target),
                "num_classes": int(labels.max() + 1) if labels.size else 0,
            }
        else:
            self.alignment_stats["classification"] = {"enabled": False}

    def _resolve_target_index(self, target: Optional[Union[str, int]]) -> int:
        if target is None:
            return 0
        if isinstance(target, int):
            if target < 0 or target >= len(self.target_cols):
                raise IndexError(
                    f"classification_target index {target} out of range for "
                    f"{len(self.target_cols)} target columns."
                )
            return int(target)
        if isinstance(target, str):
            if target not in self.target_cols:
                raise KeyError(f"classification_target '{target}' not in target_cols={self.target_cols}.")
            return self.target_cols.index(target)
        raise TypeError("classification_target must be str, int, or None.")

    def _target_from_sample_pos(self, sample_i: int) -> torch.Tensor:
        if not self.classification:
            return torch.from_numpy(self.Y[sample_i].copy())

        assert self._class_targets is not None
        y_class = torch.tensor(int(self._class_targets[sample_i]), dtype=torch.long)
        if self.include_regression_target:
            assert self._class_target_index is not None
            y_reg = torch.tensor(float(self.Y[sample_i, self._class_target_index]), dtype=torch.float32)
            return torch.stack([y_class.to(torch.float32), y_reg], dim=0)
        return y_class

    def __getitem__(self, idx: int):
        base_item = super().__getitem__(idx)
        if not self.classification:
            return base_item

        sample_i = int(self.sample_pos[idx])
        y = self._target_from_sample_pos(sample_i)

        if self.return_meta:
            x_short, t_short_feats, x_long, t_long_feats, short_mask, _, meta = base_item
            return x_short, t_short_feats, x_long, t_long_feats, short_mask, y, meta

        x_short, t_short_feats, x_long, t_long_feats, short_mask, _ = base_item
        return x_short, t_short_feats, x_long, t_long_feats, short_mask, y


class FinMambaContinuousDataset(ContinuousRasterAlignedTaskDataset):
    """
    FinMamba-ready continuous dataset.

    Returns per-sample in this exact order:
      x_asset_short, x_asset_long, x_market, t_short, t_long, targets
    Optional meta payload is appended as the last element when return_meta=True.
    """

    def __init__(
        self,
        *args,
        market_feature_cols: List[str],
        market_lookback: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.market_feature_cols = list(market_feature_cols)
        if not self.market_feature_cols:
            raise ValueError("market_feature_cols must be non-empty for FinMambaContinuousDataset.")

        missing_market = [c for c in self.market_feature_cols if c not in self.df.columns]
        if missing_market:
            raise KeyError(f"Missing market_feature_cols in df: {missing_market}")

        self.market_lookback = int(market_lookback or self.long_lookback)
        if self.market_lookback != self.long_lookback:
            raise ValueError(
                "FinMambaContinuousDataset currently requires market_lookback == long_lookback "
                f"(got market_lookback={self.market_lookback}, long_lookback={self.long_lookback})."
            )

        self.X_market = np.nan_to_num(
            self.df[self.market_feature_cols].to_numpy(dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.alignment_stats["market_feature_cols"] = list(self.market_feature_cols)
        self.alignment_stats["market_lookback"] = int(self.market_lookback)

    def __getitem__(self, idx: int):
        i = int(self.sample_pos[idx])
        if self.random_short_len and self.short_hi > self.short_lo:
            t_short = int(np.random.randint(self.short_lo, self.short_hi + 1))
        else:
            t_short = self.short_hi

        s0 = i - t_short + 1
        l0 = i - self.long_lookback + 1
        m0 = i - self.market_lookback + 1

        x_asset_short = torch.from_numpy(self.raster[s0:i + 1].copy())  # [T_short,C,H]
        x_asset_long = torch.from_numpy(self.X_long[l0:i + 1].copy())  # [T_long,F_asset]
        x_market = torch.from_numpy(self.X_market[m0:i + 1].copy())  # [T_market,F_market]
        t_short_feats = torch.from_numpy(self.T[s0:i + 1].copy())  # [T_short,F_time]
        t_long_feats = torch.from_numpy(self.T[l0:i + 1].copy())  # [T_long,F_time]
        y = self._target_from_sample_pos(i)

        if not self.return_meta:
            return x_asset_short, x_asset_long, x_market, t_short_feats, t_long_feats, y

        meta = {
            "timestamp": self.df.index[i],
            "session_code": int(self.df["session_code"].iloc[i]) if "session_code" in self.df.columns else -1,
            "sample_pos": i,
        }
        return x_asset_short, x_asset_long, x_market, t_short_feats, t_long_feats, y, meta


def collate_continuous_raster(batch: Sequence[Tuple[Any, ...]]):
    """
    Collate for ContinuousRasterAlignedDataset with variable short sequence lengths.

    Returns
    -------
    tuple
      x_short_pad: (B, T_short_max, C, H)
      t_short_pad: (B, T_short_max, F_time)
      x_long:      (B, T_long, F_long)
      t_long:      (B, T_long, F_time)
      short_mask:  (B, T_short_max)
      y:           (B, F_target) or (B,)
      meta:        optional tuple of metadata dicts
    """
    has_meta = len(batch[0]) == 7

    if has_meta:
        x_short_list, t_short_list, x_long_list, t_long_list, m_short_list, y_list, meta_list = zip(*batch)
    else:
        x_short_list, t_short_list, x_long_list, t_long_list, m_short_list, y_list = zip(*batch)
        meta_list = None

    bsz = len(x_short_list)
    t_max = max(int(x.shape[0]) for x in x_short_list)
    cdim = int(x_short_list[0].shape[1])
    hdim = int(x_short_list[0].shape[2])
    ft_dim = int(t_short_list[0].shape[1])

    x_short_pad = torch.zeros((bsz, t_max, cdim, hdim), dtype=torch.float32)
    t_short_pad = torch.zeros((bsz, t_max, ft_dim), dtype=torch.float32)
    short_mask = torch.zeros((bsz, t_max), dtype=torch.float32)

    for b in range(bsz):
        t = int(x_short_list[b].shape[0])
        x_short_pad[b, :t] = x_short_list[b]
        t_short_pad[b, :t] = t_short_list[b]
        short_mask[b, :t] = m_short_list[b]

    x_long = torch.stack(x_long_list, dim=0)
    t_long = torch.stack(t_long_list, dim=0)
    y = torch.stack(y_list, dim=0)

    if has_meta:
        return x_short_pad, t_short_pad, x_long, t_long, short_mask, y, meta_list
    return x_short_pad, t_short_pad, x_long, t_long, short_mask, y


def collate_finmamba_continuous(batch: Sequence[Tuple[Any, ...]]):
    """
    Collate for FinMambaContinuousDataset.

    Returns in this exact order:
      x_asset_short, x_asset_long, x_market, t_short, t_long, targets
      (+ meta_list when dataset was built with return_meta=True)
    """
    has_meta = len(batch[0]) == 7

    if has_meta:
        x_short_list, x_long_list, x_market_list, t_short_list, t_long_list, y_list, meta_list = zip(*batch)
    else:
        x_short_list, x_long_list, x_market_list, t_short_list, t_long_list, y_list = zip(*batch)
        meta_list = None

    bsz = len(x_short_list)
    t_max = max(int(x.shape[0]) for x in x_short_list)
    cdim = int(x_short_list[0].shape[1])
    hdim = int(x_short_list[0].shape[2])
    ft_dim = int(t_short_list[0].shape[1])

    x_asset_short = torch.zeros((bsz, t_max, cdim, hdim), dtype=torch.float32)
    t_short = torch.zeros((bsz, t_max, ft_dim), dtype=torch.float32)
    for b in range(bsz):
        t = int(x_short_list[b].shape[0])
        x_asset_short[b, :t] = x_short_list[b]
        t_short[b, :t] = t_short_list[b]

    x_asset_long = torch.stack(x_long_list, dim=0)
    x_market = torch.stack(x_market_list, dim=0)
    t_long = torch.stack(t_long_list, dim=0)
    targets = torch.stack(y_list, dim=0)

    if has_meta:
        return x_asset_short, x_asset_long, x_market, t_short, t_long, targets, meta_list
    return x_asset_short, x_asset_long, x_market, t_short, t_long, targets


class VolMoEContinuousDataset(ContinuousRasterAlignedTaskDataset):
    """
    Dataset for VolRegimeAwareMoE training.

    Extends ContinuousRasterAlignedTaskDataset with:
    - ``x_vol``: a 1D return series (extracted from ``return_col``) that the
      DeepVol router consumes directly.
    - ``vol_target``: 1-day ahead realized volatility for the router's aux loss.

    No market features — the vol router only needs 1D returns, and the experts
    process asset features + raster directly.

    Returns per-sample:
      x_short, x_long, x_vol, t_short, t_long, y, vol_target
    With ``return_meta=True``, meta dict is appended as the last element.

    Parameters
    ----------
    return_col : str
        Column name in *feature_cols* that contains bar-level log returns.
        Used both as the router's 1D input (``x_vol``) and to compute the
        1-day ahead realized vol target.
    """

    def __init__(
        self,
        *args,
        return_col: str = "log_ret",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if return_col not in self.feature_cols:
            raise KeyError(
                f"return_col='{return_col}' not found in feature_cols. "
                f"Available: {self.feature_cols[:10]}..."
            )
        self.return_col = return_col
        self.return_col_idx = self.feature_cols.index(return_col)

        # Pre-extract the 1D return series for x_vol
        self.X_vol = np.nan_to_num(
            self.df[return_col].to_numpy(dtype=np.float32), nan=0.0,
        )

        # ---- compute 1-day ahead realized volatility ----
        ret_series = self.df[return_col]
        daily_rv = ret_series.groupby(ret_series.index.date).std()
        daily_rv.index = pd.to_datetime(daily_rv.index)
        fwd_rv = daily_rv.shift(-1)  # today → tomorrow's realized vol
        fwd_vol = self.df.index.normalize().map(fwd_rv).to_numpy(dtype=np.float32)
        # NaN on last day (no tomorrow) → fill with previous day's vol
        nan_mask = np.isnan(fwd_vol)
        if nan_mask.any() and not nan_mask.all():
            last_valid = fwd_vol[~nan_mask][-1]
            fwd_vol[nan_mask] = last_valid
        self.fwd_vol = np.nan_to_num(fwd_vol, nan=0.0)

        self.alignment_stats["vol_target"] = {
            "return_col": return_col,
            "return_col_idx": int(self.return_col_idx),
            "fwd_vol_mean": float(np.nanmean(self.fwd_vol)),
            "fwd_vol_std": float(np.nanstd(self.fwd_vol)),
        }

    def __getitem__(self, idx: int):
        i = int(self.sample_pos[idx])
        if self.random_short_len and self.short_hi > self.short_lo:
            t_short = int(np.random.randint(self.short_lo, self.short_hi + 1))
        else:
            t_short = self.short_hi

        s0 = i - t_short + 1
        l0 = i - self.long_lookback + 1

        x_short = torch.from_numpy(self.raster[s0:i + 1].copy())       # [T_short, C, H]
        x_long = torch.from_numpy(self.X_long[l0:i + 1].copy())        # [T_long, F]
        x_vol = torch.from_numpy(self.X_vol[l0:i + 1].copy())          # [T_long]
        t_short_feats = torch.from_numpy(self.T[s0:i + 1].copy())      # [T_short, Ft]
        t_long_feats = torch.from_numpy(self.T[l0:i + 1].copy())       # [T_long, Ft]
        y = self._target_from_sample_pos(i)
        vol_target = torch.tensor([self.fwd_vol[i]], dtype=torch.float32)  # [1]

        if not self.return_meta:
            return x_short, x_long, x_vol, t_short_feats, t_long_feats, y, vol_target

        meta = {
            "timestamp": self.df.index[i],
            "session_code": int(self.df["session_code"].iloc[i]) if "session_code" in self.df.columns else -1,
            "sample_pos": i,
        }
        return x_short, x_long, x_vol, t_short_feats, t_long_feats, y, vol_target, meta


def collate_vol_moe_continuous(batch: Sequence[Tuple[Any, ...]]):
    """
    Collate for VolMoEContinuousDataset.

    Returns in this exact order:
      x_short, x_long, x_vol, t_short, t_long, targets, vol_target
      (+ meta_list when dataset was built with return_meta=True)

    - ``x_vol``    is ``[B, T_long]``  — 1D return series for the DeepVol router.
    - ``vol_target`` is ``[B, 1]``     — 1-day ahead realized vol for aux loss.
    """
    has_meta = len(batch[0]) == 8

    if has_meta:
        (x_short_list, x_long_list, x_vol_list,
         t_short_list, t_long_list, y_list, vol_list, meta_list) = zip(*batch)
    else:
        (x_short_list, x_long_list, x_vol_list,
         t_short_list, t_long_list, y_list, vol_list) = zip(*batch)
        meta_list = None

    bsz = len(x_short_list)
    t_max = max(int(x.shape[0]) for x in x_short_list)
    cdim = int(x_short_list[0].shape[1])
    hdim = int(x_short_list[0].shape[2])
    ft_dim = int(t_short_list[0].shape[1])

    x_short_pad = torch.zeros((bsz, t_max, cdim, hdim), dtype=torch.float32)
    t_short_pad = torch.zeros((bsz, t_max, ft_dim), dtype=torch.float32)
    for b in range(bsz):
        t = int(x_short_list[b].shape[0])
        x_short_pad[b, :t] = x_short_list[b]
        t_short_pad[b, :t] = t_short_list[b]

    x_long = torch.stack(x_long_list, dim=0)
    x_vol = torch.stack(x_vol_list, dim=0)      # [B, T_long]
    t_long = torch.stack(t_long_list, dim=0)
    targets = torch.stack(y_list, dim=0)
    vol_target = torch.stack(vol_list, dim=0)    # [B, 1]

    if has_meta:
        return x_short_pad, x_long, x_vol, t_short_pad, t_long, targets, vol_target, meta_list
    return x_short_pad, x_long, x_vol, t_short_pad, t_long, targets, vol_target
