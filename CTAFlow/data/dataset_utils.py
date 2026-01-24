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
    """
    if len(batch[0]) == 7:
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

    summaries, seqs, profiles, rasters, targets, seq_lens = zip(*batch)
    return (
        torch.stack(summaries, dim=0),
        torch.stack(seqs, dim=0),
        torch.stack(seq_lens, dim=0),
        torch.stack(profiles, dim=0),
        torch.stack(rasters, dim=0),
        torch.stack(targets, dim=0),
    )

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

    Input batch items: (summary, sequential_seq, profile, rasterized, target, seq_length)

    Rasterized data has fixed shape (T, C, Bins) from SequenceRasterizer,
    so no padding is needed - just stack.

    Returns
    -------
    tuple
        (summaries, sequential_padded, profiles, rasterized, targets, lengths)
        - summaries: (B, F_sum)
        - sequential_padded: (B, max_T, F_seq)
        - profiles: (B, C_profile, Bins_profile)
        - rasterized: (B, T_bars, C_raster, Bins_raster)
        - targets: (B,) or (B, num_classes)
        - lengths: (B,)
    """
    summaries, sequential_seqs, profiles, rasterized_tensors, targets, lengths = zip(*batch)
    summaries = torch.stack(summaries)
    profiles = torch.stack(profiles)
    rasterized = torch.stack(rasterized_tensors)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)
    sequential_padded = rnn_utils.pad_sequence(
        sequential_seqs, batch_first=True, padding_value=0.0
    )
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
