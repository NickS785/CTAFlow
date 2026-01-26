"""
Collate functions and dataset utilities for multi-modal deep learning models.

This module provides collate functions for DataLoader that handle:
- DualModal: Summary + Sequential
- TriModal: Summary + Sequential + Profile
- QuadModal: Summary + Sequential + Profile + NumberBars
- RasterizedModal: Summary + Sequential + Profile + Rasterized VPIN

All collate functions handle variable-length sequences with proper padding.
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from typing import List, Tuple, Any


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
