import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MultiScalePatcher(nn.Module):
    """
    x: [B, T, K] -> long_tok: [B, Nl, D], short_tok: [B, Ns, D]
    """
    def __init__(
        self,
        in_features: int,
        d_model: int,
        long_patch: int = 12,     # 12*5m = 60m
        long_stride: int = 12,
        short_patch: int = 2,     # 10m patch
        short_stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.long_patch, self.long_stride = long_patch, long_stride
        self.short_patch, self.short_stride = short_patch, short_stride

        self.long_proj = nn.Sequential(
            nn.Linear(long_patch * in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.short_proj = nn.Sequential(
            nn.Linear(short_patch * in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def _patchify(self, x: torch.Tensor, patch: int, stride: int) -> torch.Tensor:
        B, T, K = x.shape
        if T < patch:
            pad = patch - T
            x = F.pad(x, (0, 0, pad, 0))  # pad on the left
            T = x.shape[1]
        patches = x.unfold(dimension=1, size=patch, step=stride)  # [B, N, patch, K]
        patches = patches.contiguous().view(B, patches.shape[1], patch * K)
        return patches

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        long_p = self._patchify(x, self.long_patch, self.long_stride)
        short_p = self._patchify(x, self.short_patch, self.short_stride)
        long_tok = self.long_proj(long_p)
        short_tok = self.short_proj(short_p)
        return long_tok, short_tok


class DualLookbackPatcher(nn.Module):
    """
    Dual-lookback patcher for HierarchicalHybridMamba.

    Accepts two separate input tensors with different lookbacks:
    - x_long: [B, T_long, K] -> long_tok: [B, Nl, D]  (e.g., 256 bars -> 21 tokens)
    - x_short: [B, T_short, K] -> short_tok: [B, Ns, D] (e.g., 64 bars -> 63 tokens)

    Or a single tensor where short uses the tail:
    - x: [B, T_long, K] -> uses full for long, last short_lookback for short

    Parameters
    ----------
    in_features : int
        Number of input features per bar
    d_model : int
        Model dimension for output tokens
    long_lookback : int
        Number of bars for long branch (default: 256)
    long_patch : int
        Patch size for long branch (default: 12 = 60min for 5min bars)
    long_stride : int
        Stride for long branch (default: 12, non-overlapping)
    short_lookback : int
        Number of bars for short branch (default: 64)
    short_patch : int
        Patch size for short branch (default: 2 = 10min for 5min bars)
    short_stride : int
        Stride for short branch (default: 1, overlapping)
    dropout : float
        Dropout rate

    Example
    -------
    >>> patcher = DualLookbackPatcher(
    ...     in_features=20,
    ...     d_model=128,
    ...     long_lookback=256,  # 256 bars
    ...     long_patch=12,      # 12-bar patches -> 256/12 = 21 tokens
    ...     short_lookback=64,  # 64 bars
    ...     short_patch=2,      # 2-bar patches, stride=1 -> 63 tokens
    ... )
    >>> x_long = torch.randn(32, 256, 20)
    >>> x_short = torch.randn(32, 64, 20)
    >>> long_tok, short_tok = patcher(x_long, x_short)
    >>> long_tok.shape   # [32, 21, 128]
    >>> short_tok.shape  # [32, 63, 128]
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        long_lookback: int = 256,
        long_patch: int = 12,
        long_stride: int = 12,
        short_lookback: int = 64,
        short_patch: int = 2,
        short_stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.long_lookback = long_lookback
        self.short_lookback = short_lookback
        self.long_patch, self.long_stride = long_patch, long_stride
        self.short_patch, self.short_stride = short_patch, short_stride

        # Compute expected token counts
        self.n_long_tokens = (long_lookback - long_patch) // long_stride + 1
        self.n_short_tokens = (short_lookback - short_patch) // short_stride + 1

        self.long_proj = nn.Sequential(
            nn.Linear(long_patch * in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.short_proj = nn.Sequential(
            nn.Linear(short_patch * in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def _patchify(self, x: torch.Tensor, patch: int, stride: int) -> torch.Tensor:
        """Convert sequence to patches."""
        B, T, K = x.shape
        if T < patch:
            pad = patch - T
            x = F.pad(x, (0, 0, pad, 0))  # pad on the left
            T = x.shape[1]
        patches = x.unfold(dimension=1, size=patch, step=stride)  # [B, N, patch, K]
        patches = patches.contiguous().view(B, patches.shape[1], patch * K)
        return patches

    def forward(
        self,
        x_long: torch.Tensor,
        x_short: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x_long : torch.Tensor
            Long lookback input [B, T_long, K]. If x_short is None, the last
            short_lookback bars are used for the short branch.
        x_short : torch.Tensor, optional
            Short lookback input [B, T_short, K]. If None, sliced from x_long.

        Returns
        -------
        long_tok : torch.Tensor
            Long branch tokens [B, Nl, D]
        short_tok : torch.Tensor
            Short branch tokens [B, Ns, D]
        """
        # If x_short not provided, slice from x_long
        if x_short is None:
            x_short = x_long[:, -self.short_lookback:, :]

        # Patchify
        long_p = self._patchify(x_long, self.long_patch, self.long_stride)
        short_p = self._patchify(x_short, self.short_patch, self.short_stride)

        # Project to d_model
        long_tok = self.long_proj(long_p)
        short_tok = self.short_proj(short_p)

        return long_tok, short_tok
