"""
Collate functions and dataset utilities for multi-modal deep learning models.

This module provides collate functions for DataLoader that handle:
- DualModal: Summary + Sequential
- TriModal: Summary + Sequential + Profile
- QuadModal: Summary + Sequential + Profile + NumberBars
- RasterizedModal: Summary + Sequential + Profile + Rasterized VPIN

All collate functions handle variable-length sequences with proper padding.
"""
import math
from datetime import date, datetime

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Tuple, Any, Optional, Dict


def collate_dual(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for DualBranchModel (Summary + Sequential).

    Input batch items: (summary, sequential_seq, target, seq_length)

    Returns
    -------
    tuple
        (summaries, sequential_padded, targets, lengths)
        - summaries: (B, F_sum)
        - sequential_padded: (B, max_T, F_seq)
        - targets: (B,) or (B, num_classes)
        - lengths: (B,)
    """
    summaries, sequential_seqs, targets, lengths = zip(*batch)
    summaries = torch.stack(summaries)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)
    sequential_padded = rnn_utils.pad_sequence(
        sequential_seqs, batch_first=True, padding_value=0.0
    )
    return summaries, sequential_padded, targets, lengths


def collate_tri(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for TriModalModel (Summary + Sequential + Profile).

    Input batch items: (summary, sequential_seq, profile, target, seq_length)

    Returns
    -------
    tuple
        (summaries, sequential_padded, profiles, targets, lengths)
        - summaries: (B, F_sum)
        - sequential_padded: (B, max_T, F_seq)
        - profiles: (B, C, Bins)
        - targets: (B,) or (B, num_classes)
        - lengths: (B,)
    """
    summaries, sequential_seqs, profiles, targets, lengths = zip(*batch)
    summaries = torch.stack(summaries)
    profiles = torch.stack(profiles)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)
    sequential_padded = rnn_utils.pad_sequence(
        sequential_seqs, batch_first=True, padding_value=0.0
    )
    return summaries, sequential_padded, profiles, targets, lengths



def collate_windowed_rasterized(batch):
    """
    Collate for TriModalWindowDataset.

    Produces:
      summary_days : (B, D, F_sum)
      seq_days     : (B, D, T_seq, F_seq)
      seq_lens     : (B, D)
      profile_days : (B, D, C_prof, B_prof)
      raster_days  : (B, D, T_nb, C_nb, B_nb)
      targets      : (B,)
      raw_returns  : (B,) - optional, if add_raw_returns=True
      dates        : tuple of lists - optional, if return_dates=True

    Handles variable outputs:
      - 6 items: base (no raw_returns, no dates)
      - 7 items: with raw_returns OR with dates (disambiguate by type)
      - 8 items: with raw_returns AND dates
    """
    n_items = len(batch[0])

    if n_items == 8:
        # All outputs: raw_returns AND dates
        summaries, seqs, profiles, rasters, targets, seq_lens, raw_rets, dates = zip(*batch)
        return (
            torch.stack(summaries, dim=0),
            torch.stack(seqs, dim=0),
            torch.stack(seq_lens, dim=0),
            torch.stack(profiles, dim=0),
            torch.stack(rasters, dim=0),
            torch.stack(targets, dim=0),
            torch.stack(raw_rets, dim=0),
            dates,
        )

    if n_items == 7:
        # Either raw_returns OR dates - disambiguate by checking if last element is a list
        last_elements = [b[-1] for b in batch]
        if isinstance(last_elements[0], (list, tuple)):
            # Last element is dates (list of date objects)
            summaries, seqs, profiles, rasters, targets, seq_lens, dates = zip(*batch)
            return (
                torch.stack(summaries, dim=0),
                torch.stack(seqs, dim=0),
                torch.stack(seq_lens, dim=0),
                torch.stack(profiles, dim=0),
                torch.stack(rasters, dim=0),
                torch.stack(targets, dim=0),
                dates,
            )
        else:
            # Last element is raw_returns (tensor)
            summaries, seqs, profiles, rasters, targets, seq_lens, raw_rets = zip(*batch)
            return (
                torch.stack(summaries, dim=0),
                torch.stack(seqs, dim=0),
                torch.stack(seq_lens, dim=0),
                torch.stack(profiles, dim=0),
                torch.stack(rasters, dim=0),
                torch.stack(targets, dim=0),
                torch.stack(raw_rets, dim=0),
            )

    # Base case: 6 items (no raw_returns, no dates)
    summaries, seqs, profiles, rasters, targets, seq_lens = zip(*batch)
    return (
        torch.stack(summaries, dim=0),
        torch.stack(seqs, dim=0),
        torch.stack(seq_lens, dim=0),
        torch.stack(profiles, dim=0),
        torch.stack(rasters, dim=0),
        torch.stack(targets, dim=0),
    )


def collate_recurrent_dual(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for DualModalWindowDataset.

    Handles variable outputs:
      - 4 items: base (no raw_returns, no dates)
      - 5 items: with raw_returns OR with dates
      - 6 items: with raw_returns AND dates

    Returns:
      summary_batch : (B, Window, F_sum)
      profile_batch : (B, Window, C_prof, Bins)
      raster_batch  : (B, Window, T_bars, C_rast, Bins)
      target_batch  : (B,)
      raw_returns   : (B,) - optional
      dates         : tuple of lists - optional
    """
    n_items = len(batch[0])
    unzipped = list(zip(*batch))

    summaries = torch.stack(unzipped[0], dim=0)
    profiles = torch.stack(unzipped[1], dim=0)
    rasters = torch.stack(unzipped[2], dim=0)
    targets = torch.stack(unzipped[3], dim=0)

    if n_items == 4:
        return summaries, profiles, rasters, targets

    if n_items == 5:
        # Disambiguate between dates (list) and raw_returns (tensor)
        if isinstance(unzipped[4][0], (list, tuple)):
            return summaries, profiles, rasters, targets, unzipped[4]  # Dates
        else:
            return summaries, profiles, rasters, targets, torch.stack(unzipped[4], dim=0)  # Raw returns

    if n_items == 6:
        # Order is fixed: raw_returns then dates
        raw_rets = torch.stack(unzipped[4], dim=0)
        dates = unzipped[5]
        return summaries, profiles, rasters, targets, raw_rets, dates

    # Should not be reached, but as a fallback
    return summaries, profiles, rasters, targets

def collate_quad(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for QuadModalModel (Summary + Sequential + Profile + NumberBars).

    Input batch items: (summary, sequential_seq, profile, nb_seq, target, seq_length, nb_length)

    NumberBars are expected with shape (T_nb, BINS, C_nb) per sample.
    Padding is applied along the T_nb dimension.

    Returns
    -------
    tuple
        (summaries, sequential_padded, profiles, nb_padded, targets, lengths, nb_lengths)
        - summaries: (B, F_sum)
        - sequential_padded: (B, max_T, F_seq)
        - profiles: (B, C, Bins)
        - nb_padded: (B, max_T_nb, BINS, C_nb)
        - targets: (B,) or (B, num_classes)
        - lengths: (B,)
        - nb_lengths: (B,)
    """
    summaries, sequential_seqs, profiles, nb_seqs, targets, lengths, nb_lengths = zip(*batch)
    summaries = torch.stack(summaries)
    profiles = torch.stack(profiles)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)
    nb_lengths = torch.tensor(nb_lengths)
    sequential_padded = rnn_utils.pad_sequence(
        sequential_seqs, batch_first=True, padding_value=0.0
    )

    # Pad NumberBars along time dimension
    max_nb_len = int(max(nb_lengths)) if nb_lengths.numel() > 0 else 0
    if max_nb_len > 0:
        nb_shape = nb_seqs[0].shape
        nb_bins = nb_shape[1]
        nb_channels = nb_shape[2]
        nb_padded = torch.zeros(
            (len(nb_seqs), max_nb_len, nb_bins, nb_channels),
            dtype=nb_seqs[0].dtype,
        )
        for i, nb_seq in enumerate(nb_seqs):
            nb_padded[i, :nb_seq.shape[0]] = nb_seq
    else:
        nb_padded = torch.zeros((len(nb_seqs), 0, 0, 0))

    return summaries, sequential_padded, profiles, nb_padded, targets, lengths, nb_lengths


def collate_rasterized(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for RasterizedModal (Summary + Sequential + Profile + Rasterized VPIN).

    Input batch items:
    - (summary, sequential_seq, profile, rasterized, target, seq_length)
    - (summary, sequential_seq, profile, rasterized, target, seq_length, raw_return) - if add_raw_returns=True

    Rasterized data has fixed shape (T, C, Bins) and does not need padding.

    Returns
    -------
    tuple
        (summaries, sequential_padded, profiles, rasterized, targets, lengths) OR
        (summaries, sequential_padded, profiles, rasterized, targets, lengths, raw_returns)
    """
    if len(batch[0]) == 7:
        summaries, sequential_seqs, profiles, rasterized_tensors, targets, lengths, raw_rets = zip(*batch)
    else:
        summaries, sequential_seqs, profiles, rasterized_tensors, targets, lengths = zip(*batch)
        raw_rets = None

    summaries = torch.stack(summaries)
    profiles = torch.stack(profiles)
    rasterized = torch.stack(rasterized_tensors)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)
    sequential_padded = rnn_utils.pad_sequence(
        sequential_seqs, batch_first=True, padding_value=0.0
    )

    if raw_rets is not None:
        return summaries, sequential_padded, profiles, rasterized, targets, lengths, torch.stack(raw_rets)

    return summaries, sequential_padded, profiles, rasterized, targets, lengths


def collate_quad_rasterized(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for QuadModal with Rasterized data instead of NumberBars.

    This is an alternative to collate_quad that handles rasterized VPIN data
    (fixed shape, no padding needed) instead of NumberBars.

    Input batch items: (summary, sequential_seq, profile, rasterized, target, seq_length)

    Returns
    -------
    tuple
        (summaries, sequential_padded, profiles, rasterized, targets, lengths, raster_lengths)
        - summaries: (B, F_sum)
        - sequential_padded: (B, max_T, F_seq)
        - profiles: (B, C_profile, Bins_profile)
        - rasterized: (B, T_bars, C_raster, Bins_raster)
        - targets: (B,) or (B, num_classes)
        - lengths: (B,)
        - raster_lengths: (B,) - Always equal to T_bars for rasterized data
    """
    summaries, sequential_seqs, profiles, rasterized_tensors, targets, lengths = zip(*batch)
    summaries = torch.stack(summaries)
    profiles = torch.stack(profiles)
    rasterized = torch.stack(rasterized_tensors)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)

    # Rasterized data has fixed length (num_bars)
    raster_lengths = torch.tensor([r.shape[0] for r in rasterized_tensors])

    sequential_padded = rnn_utils.pad_sequence(
        sequential_seqs, batch_first=True, padding_value=0.0
    )
    return summaries, sequential_padded, profiles, rasterized, targets, lengths, raster_lengths


def collate_rasterized_vpin(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for OnTheFlyRasterizedDataset.

    Input batch items: (summary, profile, rasterized, target)

    This is a simpler collate than collate_rasterized since there's no sequential
    data - just summary, profile, and rasterized tensors.

    Returns
    -------
    tuple
        (summaries, profiles, rasterized, targets)
        - summaries: (B, F_sum)
        - profiles: (B, C_profile, Bins_profile)
        - rasterized: (B, num_bars, C_raster, Bins_raster)
        - targets: (B,) or (B, num_classes)
    """
    summaries, profiles, rasterized_tensors, targets = zip(*batch)

    summaries = torch.stack(summaries)
    profiles = torch.stack(profiles)
    rasterized = torch.stack(rasterized_tensors)
    targets = torch.stack(targets)

    return summaries, profiles, rasterized, targets


def get_collate_fn(mode: str):
    """
    Get the appropriate collate function for a given mode.

    Parameters
    ----------
    mode : str
        One of: 'dual', 'tri', 'quad', 'rasterized', 'quad_rasterized', 'rasterized_vpin'

    Returns
    -------
    callable
        The collate function for the specified mode
    """
    collate_fns = {
        'dual': collate_dual,
        'tri': collate_tri,
        'quad': collate_quad,
        'rasterized': collate_rasterized,
        'quad_rasterized': collate_quad_rasterized,
        'rasterized_vpin': collate_rasterized_vpin,
        'recurrent_dual': collate_recurrent_dual,
    }

    if mode not in collate_fns:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(collate_fns.keys())}")

    return collate_fns[mode]


# Export all collate functions
__all__ = [
    'collate_dual',
    'collate_tri',
    'collate_quad',
    'collate_rasterized',
    'collate_quad_rasterized',
    'collate_rasterized_vpin',
    'get_collate_fn',
]


def _is_date_like(x: Any) -> bool:
    return isinstance(x, (date, datetime, np.datetime64)) or hasattr(x, "to_pydatetime")


def _to_pydate(x: Any) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if hasattr(x, "to_pydatetime"):
        return x.to_pydatetime().date()
    if isinstance(x, np.datetime64):
        s = np.datetime_as_string(x, unit="D")  # 'YYYY-MM-DD'
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    raise TypeError(f"Unsupported date type: {type(x)}")


def _parse_extras(extras: Tuple[Any, ...]) -> Tuple[Optional[torch.Tensor], Optional[List[Any]], Optional[torch.Tensor], Optional[Dict]]:
    """Parse optional extras from a sample.
    Returns: raw_ret, window_dates, meta_ids_tensor, meta_dict
    """
    raw_ret = None
    window_dates = None
    meta_ids = None
    meta_dict = None

    for e in extras:
        if isinstance(e, (list, tuple)) and len(e) > 0 and _is_date_like(e[0]):
            window_dates = list(e)
            continue
        if isinstance(e, dict):
            meta_dict = e
            continue
        if torch.is_tensor(e):
            # raw returns tends to be float scalar
            if e.dtype.is_floating_point and (e.ndim == 0 or e.numel() == 1):
                raw_ret = e.reshape(())
                continue
            # meta ids tends to be long vector (K,)
            if e.dtype in (torch.int64, torch.long) and e.ndim == 1 and 1 <= e.numel() <= 8:
                meta_ids = e
                continue

    return raw_ret, window_dates, meta_ids, meta_dict


def _time_tensors(dates_batch: List[List[Any]], device) -> Dict[str, torch.Tensor]:
    B = len(dates_batch)
    W = len(dates_batch[0])

    month = torch.zeros((B, W), dtype=torch.long, device=device)
    dow = torch.zeros((B, W), dtype=torch.long, device=device)
    doy = torch.zeros((B, W), dtype=torch.long, device=device)

    for i in range(B):
        for j in range(W):
            d = _to_pydate(dates_batch[i][j])
            month[i, j] = d.month  # 1..12
            dow[i, j] = d.weekday()  # 0..6
            doy[i, j] = d.timetuple().tm_yday  # 1..366

    angle = 2.0 * math.pi * (doy.float() / 365.25)
    doy_sin = torch.sin(angle)
    doy_cos = torch.cos(angle)

    return {"month": month, "dow": dow, "doy_sin": doy_sin, "doy_cos": doy_cos}


def collate_windowed_wspr_with_meta(batch):
    """Collate for training RecurrentWSPR-like models from TriModalWindowDataset outputs.

    Consumes per-sample tuples like:
      (summary_days, seq_days, profile_days, raster_days, target, seq_lens, [raw_ret], [window_dates], [meta_ids or meta_dict])

    Produces (core):
      summary_days     : (B, W, F_sum)
      profile_days     : (B, W, C_prof, B_prof)
      raster_recent    : (B, T_nb, C_nb, B_nb)   # last day
      seq_recent       : (B, T_seq, F_seq)       # last day
      seq_lens_recent  : (B,)                    # last day
      meta             : dict with ticker/class/subclass + time tensors (if dates present)
      targets          : (B,)

    Plus optional:
      raw_returns      : (B,) if present for all samples
      dates_batch      : tuple(list[date]) if present for all samples
    """

    summaries, seqs, profiles, rasters, targets, seq_lens = zip(*[b[:6] for b in batch])

    summary_days = torch.stack(summaries, dim=0)
    seq_days = torch.stack(seqs, dim=0)
    profile_days = torch.stack(profiles, dim=0)
    raster_days = torch.stack(rasters, dim=0)
    targets = torch.stack(targets, dim=0)
    seq_lens = torch.stack(seq_lens, dim=0)

    raster_recent = raster_days[:, -1]
    seq_recent = seq_days[:, -1]
    seq_lens_recent = seq_lens[:, -1]

    # parse extras
    raw_list, dates_list, meta_ids_list, meta_dict_list = [], [], [], []
    for b in batch:
        raw, dates, meta_ids, meta_dict = _parse_extras(tuple(b[6:]))
        raw_list.append(raw)
        dates_list.append(dates)
        meta_ids_list.append(meta_ids)
        meta_dict_list.append(meta_dict)

    has_raw = all(x is not None for x in raw_list)
    has_dates = all(x is not None for x in dates_list)

    # --- build meta dict ---
    device = summary_days.device
    meta: Dict[str, torch.Tensor] = {}

    # priority: meta_dict if provided (already keyed), else meta_ids vector [ticker, class, subclass]
    if any(md is not None for md in meta_dict_list):
        # stack per key
        md0 = next(md for md in meta_dict_list if md is not None)
        for k in md0.keys():
            vals = []
            for md in meta_dict_list:
                if md is None:
                    raise ValueError("Some samples missing meta_dict while others have it. Make it consistent.")
                v = md[k]
                if not torch.is_tensor(v):
                    v = torch.tensor(v, dtype=torch.long)
                vals.append(v)
            meta[k] = torch.stack(vals, dim=0).to(device=device)
    else:
        # meta_ids
        if all(mi is not None for mi in meta_ids_list):
            mt = torch.stack([mi for mi in meta_ids_list], dim=0).to(device=device).long()
            if mt.size(1) >= 1: meta["ticker_id"] = mt[:, 0]
            if mt.size(1) >= 2: meta["asset_class_id"] = mt[:, 1]
            if mt.size(1) >= 3: meta["asset_subclass_id"] = mt[:, 2]
        else:
            B = summary_days.size(0)
            meta["ticker_id"] = torch.zeros((B,), dtype=torch.long, device=device)
            meta["asset_class_id"] = torch.zeros((B,), dtype=torch.long, device=device)
            meta["asset_subclass_id"] = torch.zeros((B,), dtype=torch.long, device=device)

    if has_dates:
        meta.update(_time_tensors(dates_list, device=device))

    core = (summary_days, profile_days, raster_recent, seq_recent, seq_lens_recent, meta, targets)

    if has_raw and has_dates:
        raw_returns = torch.stack(raw_list, dim=0).to(device=device)
        return core + (raw_returns, tuple(dates_list))
    if has_raw:
        raw_returns = torch.stack(raw_list, dim=0).to(device=device)
        return core + (raw_returns,)
    if has_dates:
        return core + (tuple(dates_list),)
    return core
