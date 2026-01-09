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
    Profile: (B, 3, 96) -> (B, D)
    Channels = [total, imbalance, magnitude]
    """
    def __init__(self, in_ch=3, d_out=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (B, 128, 1)
        self.proj = nn.Sequential(
            nn.Linear(128, d_out),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x_profile):
        # x_profile: (B, 3, 96)
        h = self.net(x_profile)          # (B, 128, 96)
        h = self.pool(h).squeeze(-1)     # (B, 128)
        z = self.proj(h)                 # (B, D)
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
