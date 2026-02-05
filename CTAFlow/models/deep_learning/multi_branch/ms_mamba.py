

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from ..temporal_encoders.multi_scale_patcher import MultiScalePatcher, DualLookbackPatcher


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution for the DualConvFFN.
    Ensures we don't leak future information in the short-term branch.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups
        )

    def forward(self, x):
        # x: [Batch, Dim, Seq]
        # Pad left only to maintain causality
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class DualConvFFN(nn.Module):
    """
    The 'CMD' part: Replaces standard MLP with gated convolutions.
    Great for filtering noise in the short-term branch.
    """

    def __init__(self, d_model, expansion=2, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expansion

        # 1. Expand (Gate)
        self.conv1 = nn.Conv1d(d_model, hidden_dim, kernel_size=1)
        self.act1 = nn.SiLU()

        # 2. Local Context (Depthwise Causal Conv)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim)
        self.act2 = nn.SiLU()

        # 3. Project Back
        self.conv3 = nn.Conv1d(hidden_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Mamba works in [B, L, D], Conv needs [B, D, L]
        x = x.transpose(1, 2)

        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))  # Local feature extraction
        x = self.dropout(self.conv3(x))

        return x.transpose(1, 2)


class CMDMambaBlock(nn.Module):
    """Mamba Mixer + DualConvFFN (Used for Short Branch)"""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.dcffn = DualConvFFN(d_model)  # <--- The upgrade

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.dcffn(self.norm2(x))
        return x


class StandardMambaBlock(nn.Module):
    """Mamba Mixer + Standard MLP (Used for Long Branch)"""

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HierarchicalHybridMamba(nn.Module):
    def __init__(self, input_dim, d_model=128, dropout=0.1):
        super().__init__()

        self.patcher = MultiScalePatcher(
            in_features=input_dim, d_model=d_model, dropout=dropout
        )

        # Branch 1: Long Term (Structural Trend)
        # Uses Standard Mamba because 60m data is cleaner
        self.long_branch = nn.Sequential(
            StandardMambaBlock(d_model, d_state=64),
            StandardMambaBlock(d_model, d_state=64),
            StandardMambaBlock(d_model, d_state=64),
            nn.LayerNorm(d_model)
        )

        # Branch 2: Short Term (Immediate Volatility)
        # Uses CMDMamba because 10m data is noisy and needs Conv filtering
        self.short_branch = nn.Sequential(
            CMDMambaBlock(d_model, d_state=16),
            CMDMambaBlock(d_model, d_state=16),
            nn.LayerNorm(d_model)
        )

        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # 1. Patchify
        long_toks, short_toks = self.patcher(x)

        # 2. Process Branches
        long_out = self.long_branch(long_toks)  # [B, N_l, D]
        short_out = self.short_branch(short_toks)  # [B, N_s, D]

        # 3. Pool (Last State)
        combined = torch.cat([long_out[:, -1], short_out[:, -1]], dim=-1)

        return self.head(combined)


class DualLookbackHierarchicalMamba(nn.Module):
    """
    HierarchicalHybridMamba with separate lookbacks for long/short branches.

    This model processes two different temporal scales:
    - Long branch: Full lookback (e.g., 256 bars) patched into coarse tokens
    - Short branch: Recent lookback (e.g., 32-128 bars) for fine-grained signals

    The long branch captures structural trends while the short branch
    captures immediate volatility and momentum.

    Parameters
    ----------
    input_dim : int
        Number of input features per bar
    d_model : int
        Model dimension (default: 128)
    long_lookback : int
        Number of bars for long branch (default: 256)
    long_patch : int
        Patch size for long branch (default: 12 = 60min for 5min bars)
    short_lookback : int
        Number of bars for short branch (default: 64)
    short_patch : int
        Patch size for short branch (default: 2 = 10min for 5min bars)
    n_long_layers : int
        Number of StandardMambaBlock layers in long branch (default: 3)
    n_short_layers : int
        Number of CMDMambaBlock layers in short branch (default: 2)
    d_state_long : int
        State dimension for long branch Mamba (default: 64)
    d_state_short : int
        State dimension for short branch Mamba (default: 16)
    num_classes : int
        Number of output classes. If 1, outputs regression value. (default: 1)
    dropout : float
        Dropout rate (default: 0.1)

    Example
    -------
    >>> model = DualLookbackHierarchicalMamba(
    ...     input_dim=20,
    ...     d_model=128,
    ...     long_lookback=256,   # 256 bars -> 21 tokens (patch=12, stride=12)
    ...     short_lookback=64,   # 64 bars -> 63 tokens (patch=2, stride=1)
    ...     num_classes=3,       # 3-class classification
    ... )
    >>> x = torch.randn(32, 256, 20)  # Full sequence
    >>> out = model(x)  # Uses full for long, last 64 for short
    >>> out.shape  # [32, 3]

    Or with separate inputs:
    >>> x_long = torch.randn(32, 256, 20)
    >>> x_short = torch.randn(32, 64, 20)
    >>> out = model(x_long, x_short)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        long_lookback: int = 256,
        long_patch: int = 12,
        short_lookback: int = 64,
        short_patch: int = 2,
        n_long_layers: int = 3,
        n_short_layers: int = 2,
        d_state_long: int = 64,
        d_state_short: int = 16,
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.long_lookback = long_lookback
        self.short_lookback = short_lookback
        self.num_classes = num_classes

        # Dual-lookback patcher
        self.patcher = DualLookbackPatcher(
            in_features=input_dim,
            d_model=d_model,
            long_lookback=long_lookback,
            long_patch=long_patch,
            long_stride=long_patch,  # Non-overlapping for efficiency
            short_lookback=short_lookback,
            short_patch=short_patch,
            short_stride=1,  # Overlapping for fine-grained
            dropout=dropout,
        )

        # Branch 1: Long Term (Structural Trend)
        # Uses Standard Mamba because aggregated data is cleaner
        long_layers = []
        for _ in range(n_long_layers):
            long_layers.append(StandardMambaBlock(d_model, d_state=d_state_long))
        long_layers.append(nn.LayerNorm(d_model))
        self.long_branch = nn.Sequential(*long_layers)

        # Branch 2: Short Term (Immediate Volatility)
        # Uses CMDMamba because raw short-term data is noisy
        short_layers = []
        for _ in range(n_short_layers):
            short_layers.append(CMDMambaBlock(d_model, d_state=d_state_short))
        short_layers.append(nn.LayerNorm(d_model))
        self.short_branch = nn.Sequential(*short_layers)

        # Output head
        head_out = num_classes if num_classes > 1 else 1
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, head_out),
        )

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_long : torch.Tensor
            Long lookback input [B, T_long, K].
            If x_short is None, last short_lookback bars are used for short branch.
        x_short : torch.Tensor, optional
            Short lookback input [B, T_short, K].

        Returns
        -------
        torch.Tensor
            Output logits/values of shape [B, num_classes] or [B, 1]
        """
        # Patchify with dual lookbacks
        long_toks, short_toks = self.patcher(x_long, x_short)

        # Process branches
        long_out = self.long_branch(long_toks)   # [B, N_l, D]
        short_out = self.short_branch(short_toks)  # [B, N_s, D]

        # Pool last state from each branch
        combined = torch.cat([long_out[:, -1], short_out[:, -1]], dim=-1)

        return self.head(combined)

    def get_token_counts(self) -> dict:
        """Return expected token counts for each branch."""
        return {
            "long_tokens": self.patcher.n_long_tokens,
            "short_tokens": self.patcher.n_short_tokens,
            "long_lookback": self.long_lookback,
            "short_lookback": self.short_lookback,
        }