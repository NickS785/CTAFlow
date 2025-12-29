import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ...data import MomentumWindowDataset

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)

class TCNRegressor(nn.Module):
    def __init__(self, in_channels: int, channels=(32, 32, 32), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            layers.append(
                TemporalBlock(
                    c_in, c_out,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout
                )
            )
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):
        # x: (B, C, L)
        z = self.tcn(x)              # (B, hidden, L)
        last = z[:, :, -1]           # (B, hidden)  use last timestep
        out = self.head(last).squeeze(-1)  # (B,)
        return out

def train_simple_tcn(ds: MomentumWindowDataset, in_channels: int, epochs: int = 30, model : nn.Module = None):
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    if model is None:
        model = TCNRegressor(in_channels=in_channels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"epoch {ep:02d} loss {tot/n:.6f}")

    return model
