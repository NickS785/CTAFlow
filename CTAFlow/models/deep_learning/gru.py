

import torch
import torch.nn as nn

class GRUAttnRegressor(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        h, _ = self.gru(x)                 # (B, L, H)
        w = self.attn(h).squeeze(-1)       # (B, L)
        w = torch.softmax(w, dim=-1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return self.head(pooled).squeeze(-1)
