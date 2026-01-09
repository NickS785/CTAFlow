import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ----------------------------
# Small blocks
# ----------------------------
class AttnPool(nn.Module):
    """Attention pooling over time: (B, T, D) -> (B, D)"""
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x, mask=None):
        # x: (B, T, D), mask: (B, T) 1=valid, 0=pad
        logits = self.score(x).squeeze(-1)  # (B, T)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        w = F.softmax(logits, dim=-1)  # (B, T)
        return torch.einsum("bt,btd->bd", w, x)

class GatedFusion(nn.Module):
    """Learns per-sample weights for [z_profile, z_seq, z_sum]."""
    def __init__(self, d: int, n_mod: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d * n_mod, d),
            nn.GELU(),
            nn.Linear(d, n_mod)
        )

    def forward(self, zs):
        # zs: list of (B, D)
        cat = torch.cat(zs, dim=-1)                  # (B, 3D)
        w = F.softmax(self.gate(cat), dim=-1)        # (B, 3)
        z = 0.0
        for i, zi in enumerate(zs):
            z = z + zi * w[:, i:i+1]
        return z, w

# ----------------------------
# Encoders
# ----------------------------
class ProfileEncoder(nn.Module):
    """
    Profile encoder using MarketProfileCNN architecture.

    Input: (B, in_ch, num_bins) -> Output: (B, d_out)
    Default: (B, 3, 96) with channels [total, imbalance, magnitude]

    Architecture adapted from MarketProfileCNN:
    - Layer 1: Detects small local structures (ledges, small nodes)
    - Layer 2: Detects larger structures (value areas, balance zones)
    - Layer 3: High-level shape recognition (P-shape, b-shape)
    - MaxPooling between layers for hierarchical feature extraction
    """
    def __init__(self, in_ch=3, d_out=128, dropout=0.1, num_bins=96):
        super().__init__()
        self.out_dim = d_out

        # Layer 1: Detect small local structures (ledges, small nodes)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduces size by half
        )

        # Layer 2: Detect larger structures (value areas, balance zones)
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduces size by half again
        )

        # Layer 3: High-level shape recognition (P-shape, b-shape)
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling: summarize entire profile
        )

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(64, d_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x_profile):
        # x_profile: (B, in_ch, num_bins) e.g., (B, 3, 96)
        h = self.conv1(x_profile)    # (B, 16, num_bins/2)
        h = self.conv2(h)            # (B, 32, num_bins/4)
        h = self.conv3(h)            # (B, 64, 1)
        h = h.flatten(1)             # (B, 64)
        z = self.fc(h)               # (B, d_out)
        return z

class SeqEncoder(nn.Module):
    """
    Sequential VPIN/bucketed flow:
      x_seq: (B, T, F_seq) padded
      seq_len: (B,)
    -> (B, D)
    """
    def __init__(self, f_in: int, d_out=128, d_conv=64, d_lstm=128, dropout=0.1, bidir=True):
        super().__init__()
        self.out_dim = d_out
        self.pre = nn.Sequential(
            nn.LayerNorm(f_in),
            nn.Linear(f_in, d_conv),
            nn.GELU(),
        )
        self.conv = nn.Conv1d(d_conv, d_conv, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(
            input_size=d_conv,
            hidden_size=d_lstm,
            num_layers=1,
            batch_first=True,
            bidirectional=bidir
        )
        lstm_dim = d_lstm * (2 if bidir else 1)
        self.post = nn.Sequential(
            nn.Linear(lstm_dim, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttnPool(d_out)

    def forward(self, x_seq, seq_len):
        # x_seq: (B, T, F)
        B, T, Fdim = x_seq.shape

        h = self.pre(x_seq)  # (B, T, d_conv)

        # Conv over time
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)  # (B, T, d_conv)

        # Pack for LSTM
        seq_len_cpu = seq_len.detach().to("cpu")
        packed = pack_padded_sequence(h, lengths=seq_len_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # (B, T, lstm_dim)

        out = self.post(out)  # (B, T, D)

        # Mask for padding positions
        device = x_seq.device
        mask = (torch.arange(T, device=device)[None, :] < seq_len[:, None]).int()  # (B, T)

        z = self.pool(out, mask=mask)  # (B, D)
        return z

class SummaryEncoder(nn.Module):
    """Summary: (B, F_sum) -> (B, D)"""
    def __init__(self, f_in: int, d_out=128, dropout=0.1):
        super().__init__()
        self.out_dim = d_out
        self.net = nn.Sequential(
            nn.LayerNorm(f_in),
            nn.Linear(f_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_out),
            nn.GELU(),
        )

    def forward(self, x_sum):
        return self.net(x_sum)


class SummaryMLPEnc(nn.Module):
    """Simple MLP encoder for summary features (matches DualBranchModel default).

    Uses BatchNorm + ReLU instead of LayerNorm + GELU for compatibility
    with the original DualBranchModel architecture.

    Parameters
    ----------
    f_in : int
        Input feature dimension
    d_hidden : int, default 128
        Hidden layer dimension
    dropout : float, default 0.3
        Dropout rate
    """
    def __init__(self, f_in: int, d_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.out_dim = d_hidden // 2
        self.net = nn.Sequential(
            nn.Linear(f_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, self.out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
