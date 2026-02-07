"""
Resampled Dual-Scale Mamba Architecture.

Short branch: Raw 5-min bars for fine-grained recent signals
Long branch: Resampled (15/30/60min) bars for structural context
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from mamba_ssm import Mamba


class CausalConv1d(nn.Module):
    """1D Causal Convolution for the DualConvFFN."""

    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, groups=groups
        )

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class DualConvFFN(nn.Module):
    """Gated convolution FFN for filtering noisy short-term data."""

    def __init__(self, d_model, expansion=2, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expansion
        self.conv1 = nn.Conv1d(d_model, hidden_dim, kernel_size=1)
        self.act1 = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim)
        self.act2 = nn.SiLU()
        self.conv3 = nn.Conv1d(hidden_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.dropout(self.conv3(x))
        return x.transpose(1, 2)


class CMDMambaBlock(nn.Module):
    """Mamba + DualConvFFN for noisy short-term data."""

    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.dcffn = DualConvFFN(d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.dcffn(self.norm2(x))
        return x


class StandardMambaBlock(nn.Module):
    """Mamba + MLP for cleaner long-term data."""

    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResampledDualMamba(nn.Module):
    """
    Dual-scale Mamba with resampled long-term data.

    Architecture:
    - Short branch: Raw 5-min bars → project → CMDMambaBlocks
    - Long branch: Resampled bars (15/30/60min) → project → StandardMambaBlocks
    - Fusion: Concatenate last states → Classification head

    Parameters
    ----------
    input_dim : int
        Number of features per bar
    d_model : int
        Model dimension (default: 128)
    short_seq_len : int
        Number of 5-min bars for short branch (default: 128)
    long_seq_len : int
        Number of resampled bars for long branch (default: 128)
    n_short_layers : int
        Number of CMDMambaBlock layers (default: 2)
    n_long_layers : int
        Number of StandardMambaBlock layers (default: 3)
    d_state_short : int
        Mamba state dim for short branch (default: 16)
    d_state_long : int
        Mamba state dim for long branch (default: 64)
    num_classes : int
        Number of output classes (default: 3)
    dropout : float
        Dropout rate (default: 0.1)

    Forward Input
    -------------
    x_short : [B, short_seq_len, input_dim] - Raw 5-min bars
    x_long : [B, long_seq_len, input_dim] - Resampled bars

    Example
    -------
    >>> model = ResampledDualMamba(input_dim=20, d_model=128)
    >>> x_short = torch.randn(32, 128, 20)  # Last 128 5-min bars
    >>> x_long = torch.randn(32, 128, 20)   # Last 128 30-min bars (= 64 hours)
    >>> out = model(x_short, x_long)
    >>> out.shape  # [32, 3]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        short_seq_len: int = 128,
        long_seq_len: int = 128,
        n_short_layers: int = 2,
        n_long_layers: int = 3,
        d_state_short: int = 16,
        d_state_long: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.short_seq_len = short_seq_len
        self.long_seq_len = long_seq_len
        self.num_classes = num_classes

        # Input projections
        self.short_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.long_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Short branch: CMDMamba for noisy 5-min data
        short_layers = []
        for _ in range(n_short_layers):
            short_layers.append(CMDMambaBlock(d_model, d_state=d_state_short, dropout=dropout))
        short_layers.append(nn.LayerNorm(d_model))
        self.short_branch = nn.Sequential(*short_layers)

        # Long branch: StandardMamba for cleaner resampled data
        long_layers = []
        for _ in range(n_long_layers):
            long_layers.append(StandardMambaBlock(d_model, d_state=d_state_long, dropout=dropout))
        long_layers.append(nn.LayerNorm(d_model))
        self.long_branch = nn.Sequential(*long_layers)

        # Classification head
        head_out = num_classes if num_classes > 1 else 1
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, head_out),
        )

    def forward(
        self,
        x_short: torch.Tensor,
        x_long: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_short : torch.Tensor
            Short-term input [B, short_seq_len, input_dim] (raw 5-min bars)
        x_long : torch.Tensor
            Long-term input [B, long_seq_len, input_dim] (resampled bars)

        Returns
        -------
        torch.Tensor
            Logits [B, num_classes]
        """
        # Project inputs
        short_emb = self.short_proj(x_short)  # [B, S, D]
        long_emb = self.long_proj(x_long)     # [B, L, D]

        # Process branches
        short_out = self.short_branch(short_emb)  # [B, S, D]
        long_out = self.long_branch(long_emb)     # [B, L, D]

        # Pool last state from each branch
        combined = torch.cat([short_out[:, -1], long_out[:, -1]], dim=-1)

        return self.head(combined)

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            "short_seq_len": self.short_seq_len,
            "long_seq_len": self.long_seq_len,
            "num_classes": self.num_classes,
        }
