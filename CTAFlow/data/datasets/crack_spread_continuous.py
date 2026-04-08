"""Crack-spread continuous dataset with separate spread and orderflow branches."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import DataLoader, Dataset

from CTAFlow.models.prep.crack_spread_continuous import DEFAULT_CRACK_TICKERS


@dataclass
class _CrackSpreadArrayStore:
    """Shared arrays for lazy crack-spread sampling."""
    spread: np.ndarray
    known_temporal: Optional[np.ndarray]
    target: np.ndarray
    timestamps: np.ndarray
    orderflow_ts: Dict[str, np.ndarray]
    orderflow_vals: Dict[str, np.ndarray]


@dataclass
class _CrackSpreadSampleTable:
    """Compact sample index for crack-spread datasets."""
    bar_indices: np.ndarray
    anchor_ordinals: np.ndarray
    orderflow_cuts: np.ndarray

    def __len__(self) -> int:
        return int(len(self.bar_indices))


def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, tz-naive DataFrame with a strict DatetimeIndex."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    if out.index.hasnans:
        out = out[~out.index.isna()]
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _build_session_mask(
    df: pd.DataFrame,
    session_only: bool = True,
    sample_session: Optional[str] = None,
    sample_session_start: Optional[str] = None,
    sample_session_end: Optional[str] = None,
) -> np.ndarray:
    bar_times = df.index.time

    if sample_session_start is not None and sample_session_end is not None:
        t_start = time(
            int(sample_session_start.split(":")[0]),
            int(sample_session_start.split(":")[1]),
        )
        t_end = time(
            int(sample_session_end.split(":")[0]),
            int(sample_session_end.split(":")[1]),
        )
        return np.array([(t >= t_start) and (t <= t_end) for t in bar_times], dtype=bool)

    if sample_session is not None:
        col_map = {
            "usa": "is_usa",
            "london": "is_london",
            "overlap": "is_session_overlap",
        }
        col = col_map.get(sample_session.lower())
        if col is None or col not in df.columns:
            raise ValueError(
                f"Unknown sample_session={sample_session!r}. "
                "Use 'usa', 'london', 'overlap', or a custom time window."
            )
        session_mask = df[col].values.astype(bool)
        if not session_mask.any() and "is_active" in df.columns:
            session_mask = df["is_active"].values.astype(bool)
        return session_mask

    if session_only and "is_active" in df.columns:
        return df["is_active"].values.astype(bool)

    return np.ones(len(df), dtype=bool)


def _resolve_spread_feature_cols(
    df_out: pd.DataFrame,
    spread_feature_cols: Optional[Sequence[str]],
    target_col: str,
    exclude_known_temporal_features: bool = False,
) -> List[str]:
    if spread_feature_cols is not None:
        cols = [col for col in spread_feature_cols if col in df_out.columns]
        if not cols:
            raise ValueError("No requested spread_feature_cols were found in df_out.")
        return cols

    exclude = {"Open", "High", "Low", "Close", "Volume", "BidVolume", "AskVolume", target_col}
    cols: List[str] = []
    for col in df_out.columns:
        if col in exclude or col.startswith("y_fwd_"):
            continue
        if exclude_known_temporal_features and col.startswith("kt_"):
            continue
        dtype = df_out[col].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            cols.append(col)
    if not cols:
        raise ValueError("Could not infer any numeric spread feature columns from df_out.")
    return cols


def _resolve_known_temporal_cols(
    df_out: pd.DataFrame,
    known_temporal_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    if known_temporal_cols is not None:
        cols = [col for col in known_temporal_cols if col in df_out.columns]
        if not cols:
            raise ValueError("No requested known_temporal_cols were found in df_out.")
        return cols
    return [col for col in df_out.columns if col.startswith("kt_")]


def _align_orderflow_frames(
    orderflow_frames: Dict[str, pd.DataFrame],
    tickers: Sequence[str],
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    aligned: Dict[str, pd.DataFrame] = {}
    common_cols: Optional[set[str]] = None
    first_non_empty: Optional[pd.DataFrame] = None

    for ticker in tickers:
        frame = orderflow_frames.get(ticker, pd.DataFrame())
        if frame is None or frame.empty:
            aligned[ticker] = pd.DataFrame()
            continue
        work = _normalize_datetime_index(frame.select_dtypes(include="number")).astype(np.float32)
        aligned[ticker] = work
        cols = set(work.columns)
        common_cols = cols if common_cols is None else common_cols & cols
        if first_non_empty is None:
            first_non_empty = work

    if first_non_empty is None or not common_cols:
        raise ValueError("Orderflow frames must contain at least one shared numeric column.")

    ordered_cols = [col for col in first_non_empty.columns if col in common_cols]
    for ticker in tickers:
        frame = aligned.get(ticker, pd.DataFrame())
        aligned[ticker] = frame.loc[:, ordered_cols] if not frame.empty else pd.DataFrame(columns=ordered_cols)
    return aligned, ordered_cols


def build_crack_spread_indexed_dataset(
    df_out: pd.DataFrame,
    orderflow_frames: Dict[str, pd.DataFrame],
    target_col: str,
    spread_feature_cols: Optional[Sequence[str]] = None,
    known_temporal_cols: Optional[Sequence[str]] = None,
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    spread_lookback: int = 64,
    orderflow_lookback: int = 128,
    session_only: bool = True,
    sample_session: Optional[str] = None,
    sample_session_start: Optional[str] = None,
    sample_session_end: Optional[str] = None,
    stride: int = 1,
    require_all_assets: bool = True,
    include_known_temporal_features: bool = True,
) -> "CrackSpreadContinuousIndexedDataset":
    """Build a lazy crack-spread dataset backed by shared arrays."""
    if target_col not in df_out.columns:
        raise KeyError(f"Missing target column {target_col!r} in df_out")

    df = _normalize_datetime_index(df_out)
    tickers = tuple(t.upper() for t in tickers)
    spread_feature_cols = _resolve_spread_feature_cols(
        df,
        spread_feature_cols,
        target_col,
        exclude_known_temporal_features=include_known_temporal_features,
    )
    resolved_known_temporal_cols = (
        _resolve_known_temporal_cols(df, known_temporal_cols)
        if include_known_temporal_features
        else []
    )
    aligned_orderflow, _ = _align_orderflow_frames(orderflow_frames, tickers)

    feature_df = df.loc[:, spread_feature_cols].copy().ffill().bfill().fillna(0.0)
    spread_arr = np.asarray(feature_df.values, dtype=np.float32)
    known_temporal_arr = None
    if resolved_known_temporal_cols:
        kt_df = df.loc[:, resolved_known_temporal_cols].copy().ffill().bfill().fillna(0.0)
        known_temporal_arr = np.asarray(kt_df.values, dtype=np.float32)
    target_arr = pd.to_numeric(df[target_col], errors="coerce").values.astype(np.float32)
    timestamps = df.index.values.astype("datetime64[ns]")
    dates = df.index.date
    session_mask = _build_session_mask(
        df,
        session_only=session_only,
        sample_session=sample_session,
        sample_session_start=sample_session_start,
        sample_session_end=sample_session_end,
    )

    orderflow_ts: Dict[str, np.ndarray] = {}
    orderflow_vals: Dict[str, np.ndarray] = {}
    for ticker in tickers:
        frame = aligned_orderflow[ticker]
        orderflow_ts[ticker] = frame.index.values.astype("datetime64[ns]")
        orderflow_vals[ticker] = frame.values.astype(np.float32)

    day_bars: Dict[date, List[int]] = {}
    for bar_idx in range(spread_lookback, len(df)):
        if not session_mask[bar_idx]:
            continue
        if np.isnan(target_arr[bar_idx]):
            continue
        day_bars.setdefault(dates[bar_idx], []).append(bar_idx)

    if stride > 1:
        for current_date in day_bars:
            day_bars[current_date] = day_bars[current_date][::stride]

    bar_indices_all: List[np.ndarray] = []
    anchor_ordinals_all: List[np.ndarray] = []
    orderflow_cuts_all: List[np.ndarray] = []

    for current_date, current_bar_indices in day_bars.items():
        if not current_bar_indices:
            continue
        current_bar_indices_arr = np.asarray(current_bar_indices, dtype=np.int32)
        anchor_ts_all = timestamps[current_bar_indices_arr]
        cuts = np.zeros((len(current_bar_indices_arr), len(tickers)), dtype=np.int32)
        valid_mask = np.ones(len(current_bar_indices_arr), dtype=bool)

        for asset_idx, ticker in enumerate(tickers):
            ts = orderflow_ts[ticker]
            asset_cuts = np.searchsorted(ts, anchor_ts_all, side="left").astype(np.int32)
            cuts[:, asset_idx] = asset_cuts
            if require_all_assets:
                valid_mask &= asset_cuts > 0

        if require_all_assets:
            current_bar_indices_arr = current_bar_indices_arr[valid_mask]
            cuts = cuts[valid_mask]
        if len(current_bar_indices_arr) == 0:
            continue

        bar_indices_all.append(current_bar_indices_arr)
        anchor_ordinals_all.append(
            np.full(len(current_bar_indices_arr), current_date.toordinal(), dtype=np.int32)
        )
        orderflow_cuts_all.append(cuts)

    if bar_indices_all:
        sample_table = _CrackSpreadSampleTable(
            bar_indices=np.concatenate(bar_indices_all),
            anchor_ordinals=np.concatenate(anchor_ordinals_all),
            orderflow_cuts=np.concatenate(orderflow_cuts_all, axis=0),
        )
    else:
        sample_table = _CrackSpreadSampleTable(
            bar_indices=np.array([], dtype=np.int32),
            anchor_ordinals=np.array([], dtype=np.int32),
            orderflow_cuts=np.empty((0, len(tickers)), dtype=np.int32),
        )

    return CrackSpreadContinuousIndexedDataset(
        store=_CrackSpreadArrayStore(
            spread=spread_arr,
            known_temporal=known_temporal_arr,
            target=target_arr,
            timestamps=timestamps,
            orderflow_ts=orderflow_ts,
            orderflow_vals=orderflow_vals,
        ),
        sample_table=sample_table,
        tickers=tickers,
        spread_lookback=spread_lookback,
        orderflow_lookback=orderflow_lookback,
        target_steps=int(target_col.split("_")[-1]) if target_col.startswith("y_fwd_") and target_col.split("_")[-1].isdigit() else 0,
    )


def build_crack_spread_samples(
    df_out: pd.DataFrame,
    orderflow_frames: Dict[str, pd.DataFrame],
    target_col: str,
    spread_feature_cols: Optional[Sequence[str]] = None,
    known_temporal_cols: Optional[Sequence[str]] = None,
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    spread_lookback: int = 64,
    orderflow_lookback: int = 128,
    session_only: bool = True,
    sample_session: Optional[str] = None,
    sample_session_start: Optional[str] = None,
    sample_session_end: Optional[str] = None,
    stride: int = 1,
    require_all_assets: bool = True,
    include_known_temporal_features: bool = True,
) -> List[Dict]:
    """Build flat crack-spread samples keyed off the spread master clock."""
    if target_col not in df_out.columns:
        raise KeyError(f"Missing target column {target_col!r} in df_out")

    df = _normalize_datetime_index(df_out)
    tickers = tuple(t.upper() for t in tickers)
    spread_feature_cols = _resolve_spread_feature_cols(
        df,
        spread_feature_cols,
        target_col,
        exclude_known_temporal_features=include_known_temporal_features,
    )
    resolved_known_temporal_cols = _resolve_known_temporal_cols(df, known_temporal_cols) if include_known_temporal_features else []
    aligned_orderflow, orderflow_feature_cols = _align_orderflow_frames(orderflow_frames, tickers)

    feature_df = df.loc[:, spread_feature_cols].copy()
    feature_df = feature_df.ffill().bfill().fillna(0.0)
    spread_arr = feature_df.values.astype(np.float32)
    known_temporal_arr = None
    if include_known_temporal_features and resolved_known_temporal_cols:
        known_temporal_df = df.loc[:, resolved_known_temporal_cols].copy()
        known_temporal_df = known_temporal_df.ffill().bfill().fillna(0.0)
        known_temporal_arr = known_temporal_df.values.astype(np.float32)
    target_arr = pd.to_numeric(df[target_col], errors="coerce").values.astype(np.float32)
    timestamps = df.index.values.astype("datetime64[ns]")
    dates = df.index.date
    session_mask = _build_session_mask(
        df,
        session_only=session_only,
        sample_session=sample_session,
        sample_session_start=sample_session_start,
        sample_session_end=sample_session_end,
    )

    orderflow_ts: Dict[str, np.ndarray] = {}
    orderflow_vals: Dict[str, np.ndarray] = {}
    for ticker in tickers:
        frame = aligned_orderflow[ticker]
        orderflow_ts[ticker] = frame.index.values.astype("datetime64[ns]")
        orderflow_vals[ticker] = frame.values.astype(np.float32)

    day_bars: Dict[date, List[int]] = {}
    for bar_idx in range(spread_lookback, len(df)):
        if not session_mask[bar_idx]:
            continue
        if np.isnan(target_arr[bar_idx]):
            continue
        day_bars.setdefault(dates[bar_idx], []).append(bar_idx)

    if stride > 1:
        for current_date in day_bars:
            day_bars[current_date] = day_bars[current_date][::stride]

    target_steps = 0
    if target_col.startswith("y_fwd_"):
        try:
            target_steps = int(target_col.split("_")[-1])
        except ValueError:
            target_steps = 0

    samples: List[Dict] = []
    for current_date, bar_indices in day_bars.items():
        for bar_idx in bar_indices:
            anchor_ts64 = timestamps[bar_idx]

            asset_sequences: Dict[str, np.ndarray] = {}
            asset_lengths: Dict[str, int] = {}
            skip_sample = False

            for ticker in tickers:
                ts = orderflow_ts[ticker]
                vals = orderflow_vals[ticker]
                cut = int(np.searchsorted(ts, anchor_ts64, side="left"))
                lo = max(0, cut - orderflow_lookback)
                seq = vals[lo:cut]
                asset_sequences[ticker] = seq
                asset_lengths[ticker] = int(len(seq))
                if require_all_assets and len(seq) == 0:
                    skip_sample = True
                    break

            if skip_sample:
                continue

            sample: Dict[str, object] = {
                "spread_time_features": spread_arr[bar_idx - spread_lookback:bar_idx],
                "spread_time_len": spread_lookback,
                "target": target_arr[bar_idx],
                "date": current_date,
                "anchor_ts": pd.Timestamp(df.index[bar_idx]),
                "target_end_ts": (
                    pd.Timestamp(df.index[bar_idx + target_steps])
                    if target_steps > 0 and (bar_idx + target_steps) < len(df)
                    else pd.NaT
                ),
            }
            if known_temporal_arr is not None:
                sample["known_temporal_features"] = known_temporal_arr[bar_idx]
            for ticker in tickers:
                sample[f"orderflow_{ticker}"] = asset_sequences[ticker]
                sample[f"orderflow_{ticker}_len"] = asset_lengths[ticker]
            sample["_orderflow_feature_cols"] = list(orderflow_feature_cols)
            if known_temporal_arr is not None:
                sample["_known_temporal_cols"] = list(resolved_known_temporal_cols)
            samples.append(sample)

    return samples


class CrackSpreadContinuousIndexedDataset(Dataset):
    """Lazy crack-spread dataset backed by shared arrays plus cut indices."""

    def __init__(
        self,
        store: _CrackSpreadArrayStore,
        sample_table: _CrackSpreadSampleTable,
        tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
        spread_lookback: int = 64,
        orderflow_lookback: int = 128,
        target_steps: int = 0,
        return_metadata: bool = False,
    ):
        self.store = store
        self.sample_table = sample_table
        self.tickers = tuple(t.upper() for t in tickers)
        self.spread_lookback = spread_lookback
        self.orderflow_lookback = orderflow_lookback
        self.target_steps = target_steps
        self.return_metadata = return_metadata

    @property
    def sample_dates(self) -> List[date]:
        return [date.fromordinal(int(x)) for x in self.sample_table.anchor_ordinals]

    def __len__(self) -> int:
        return len(self.sample_table)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        bar_idx = int(self.sample_table.bar_indices[idx])
        anchor_ordinal = int(self.sample_table.anchor_ordinals[idx])
        sample: Dict[str, object] = {
            "spread_time_features": torch.tensor(
                self.store.spread[bar_idx - self.spread_lookback:bar_idx],
                dtype=torch.float32,
            ),
            "spread_time_lens": torch.tensor(self.spread_lookback, dtype=torch.long),
            "target": torch.tensor(self.store.target[bar_idx], dtype=torch.float32),
        }
        if self.store.known_temporal is not None:
            sample["known_temporal_features"] = torch.tensor(
                self.store.known_temporal[bar_idx],
                dtype=torch.float32,
            )
        for asset_idx, ticker in enumerate(self.tickers):
            cut = int(self.sample_table.orderflow_cuts[idx, asset_idx])
            lo = max(0, cut - self.orderflow_lookback)
            seq = self.store.orderflow_vals[ticker][lo:cut]
            sample[f"orderflow_{ticker}"] = torch.tensor(seq, dtype=torch.float32)
            sample[f"orderflow_{ticker}_len"] = torch.tensor(len(seq), dtype=torch.long)
        if self.return_metadata:
            sample["_date"] = date.fromordinal(anchor_ordinal)
            sample["_anchor_ts"] = pd.Timestamp(self.store.timestamps[bar_idx])
            sample["_target_end_ts"] = (
                pd.Timestamp(self.store.timestamps[bar_idx + self.target_steps])
                if self.target_steps > 0 and (bar_idx + self.target_steps) < len(self.store.timestamps)
                else pd.NaT
            )
        return sample


class CrackSpreadContinuousDataset(Dataset):
    """PyTorch dataset for crack-spread samples with packed orderflow streams."""

    def __init__(
        self,
        samples: List[Dict],
        tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
        return_metadata: bool = False,
    ):
        self.samples = samples
        self.tickers = tuple(t.upper() for t in tickers)
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        out: Dict[str, object] = {
            "spread_time_features": torch.tensor(sample["spread_time_features"], dtype=torch.float32),
            "spread_time_lens": torch.tensor(sample["spread_time_len"], dtype=torch.long),
            "target": torch.tensor(sample["target"], dtype=torch.float32),
        }
        if "known_temporal_features" in sample:
            out["known_temporal_features"] = torch.tensor(sample["known_temporal_features"], dtype=torch.float32)
        for ticker in self.tickers:
            out[f"orderflow_{ticker}"] = torch.tensor(sample[f"orderflow_{ticker}"], dtype=torch.float32)
            out[f"orderflow_{ticker}_len"] = torch.tensor(sample[f"orderflow_{ticker}_len"], dtype=torch.long)
        if self.return_metadata:
            out["_date"] = sample["date"]
            out["_anchor_ts"] = sample["anchor_ts"]
            out["_target_end_ts"] = sample["target_end_ts"]
        return out


def crack_spread_collate_fn(
    batch: List[Dict[str, object]],
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    orderflow_lookback: int = 128,
) -> Dict[str, object]:
    """Collate crack-spread samples with packed per-asset orderflow streams."""
    tickers = tuple(t.upper() for t in tickers)
    out: Dict[str, object] = {
        "spread_time_features": torch.stack([sample["spread_time_features"] for sample in batch]),
        "spread_time_lens": torch.stack([sample["spread_time_lens"] for sample in batch]),
        "target": torch.stack([sample["target"] for sample in batch]),
    }
    if "known_temporal_features" in batch[0]:
        out["known_temporal_features"] = torch.stack([sample["known_temporal_features"] for sample in batch])

    feature_dim = 0
    for ticker in tickers:
        for sample in batch:
            seq = sample[f"orderflow_{ticker}"]
            if seq.ndim == 2 and seq.shape[1] > 0:
                feature_dim = int(seq.shape[1])
                break
        if feature_dim > 0:
            break
    if feature_dim <= 0:
        raise ValueError("Unable to infer orderflow feature dimension from batch.")

    batch_size = len(batch)
    orderflow_cube = torch.zeros(batch_size, feature_dim, orderflow_lookback, len(tickers), dtype=torch.float32)
    orderflow_mask = torch.zeros(batch_size, orderflow_lookback, len(tickers), dtype=torch.bool)
    orderflow_lens = torch.zeros(batch_size, len(tickers), dtype=torch.long)
    orderflow_packed: Dict[str, PackedSequence] = {}

    for asset_idx, ticker in enumerate(tickers):
        seqs: List[torch.Tensor] = []
        for sample_idx, sample in enumerate(batch):
            seq = sample[f"orderflow_{ticker}"]
            seq_len = int(sample[f"orderflow_{ticker}_len"])
            if seq_len <= 0:
                raise ValueError(
                    f"Packed orderflow for {ticker} received an empty sequence. "
                    "Build samples with require_all_assets=True or filter empty anchors."
                )
            seq = seq[-orderflow_lookback:]
            seq_len = min(seq_len, orderflow_lookback)
            seqs.append(seq)
            orderflow_lens[sample_idx, asset_idx] = seq_len
            orderflow_cube[sample_idx, :, orderflow_lookback - seq_len:, asset_idx] = seq.transpose(0, 1)
            orderflow_mask[sample_idx, orderflow_lookback - seq_len:, asset_idx] = True
        orderflow_packed[ticker] = pack_sequence(seqs, enforce_sorted=False)

    out["orderflow_cube"] = orderflow_cube
    out["orderflow_mask"] = orderflow_mask
    out["orderflow_lens"] = orderflow_lens
    out["orderflow_packed"] = orderflow_packed

    if "_anchor_ts" in batch[0]:
        out["_anchor_ts"] = [sample["_anchor_ts"] for sample in batch]
        out["_date"] = [sample["_date"] for sample in batch]
        out["_target_end_ts"] = [sample["_target_end_ts"] for sample in batch]

    return out


def unpack_crack_spread_batch(
    batch: Dict[str, object],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, object], torch.Tensor]:
    """Split a collated batch into model inputs and targets."""
    inputs: Dict[str, object] = {}
    tensor_keys = [
        "spread_time_features",
        "spread_time_lens",
        "known_temporal_features",
        "orderflow_cube",
        "orderflow_mask",
        "orderflow_lens",
    ]
    for key in tensor_keys:
        if key not in batch:
            continue
        value = batch[key]
        inputs[key] = value.to(device) if device is not None else value

    if "orderflow_packed" in batch:
        packed = {}
        for ticker, value in batch["orderflow_packed"].items():
            packed[ticker] = value.to(device) if device is not None else value
        inputs["orderflow_packed"] = packed

    targets = batch["target"]
    if device is not None:
        targets = targets.to(device)
    return inputs, targets


def loaders_from_crack_samples(
    train_samples: List[Dict],
    val_samples: List[Dict],
    batch_size: int = 64,
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    shuffle_train: bool = True,
    num_workers: int = 0,
    return_metadata: bool = False,
    orderflow_lookback: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoaders from pre-built crack sample lists."""
    tickers = tuple(t.upper() for t in tickers)
    collate = partial(
        crack_spread_collate_fn,
        tickers=tickers,
        orderflow_lookback=orderflow_lookback,
    )
    train_ds = CrackSpreadContinuousDataset(
        train_samples,
        tickers=tickers,
        return_metadata=return_metadata,
    )
    val_ds = CrackSpreadContinuousDataset(
        val_samples,
        tickers=tickers,
        return_metadata=return_metadata,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    return train_loader, val_loader
