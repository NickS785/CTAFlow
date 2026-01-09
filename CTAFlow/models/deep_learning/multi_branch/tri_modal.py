from torch import nn as nn

from ..encoders import ProfileEncoder, SeqEncoder, SummaryEncoder, GatedFusion


class TriModalLiquidityModel(nn.Module):
    """
    Matches TriModalDataset outputs:
      summary_vec: (B, F_sum)
      seq_tensor:  (B, T, F_seq) [padded]
      spatial/profile: (B, 3, 96)
      seq_len: (B,)
    """
    def __init__(self, f_sum: int, f_seq: int, d=128, out_dim=1, dropout=0.1):
        super().__init__()
        self.profile_enc = ProfileEncoder(in_ch=3, d_out=d, dropout=dropout)
        self.seq_enc     = SeqEncoder(f_in=f_seq, d_out=d, dropout=dropout)
        self.sum_enc     = SummaryEncoder(f_in=f_sum, d_out=d, dropout=dropout)

        self.fuse = GatedFusion(d=d, n_mod=3)

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, summary_vec, seq_tensor, profile_tensor, seq_len):
        # summary_vec: (B, F_sum)
        # seq_tensor: (B, T, F_seq)
        # profile_tensor: (B, 3, 96)
        # seq_len: (B,)
        z_profile = self.profile_enc(profile_tensor)
        z_seq     = self.seq_enc(seq_tensor, seq_len)
        z_sum     = self.sum_enc(summary_vec)

        z, gate_w = self.fuse([z_profile, z_seq, z_sum])
        y = self.head(z)
        return y, gate_w
