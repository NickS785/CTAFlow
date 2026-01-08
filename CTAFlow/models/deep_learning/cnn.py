import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


# ==========================================
# 1. The New CNN Branch (Market Profile)
# ==========================================
class MarketProfileCNN(nn.Module):
    def __init__(self, num_bins=128, output_dim=32):
        super(MarketProfileCNN, self).__init__()

        # Input Shape: (Batch, 1, num_bins)
        # e.g., Batch of 128-tick wide histograms

        self.net = nn.Sequential(
            # Layer 1: Detect small local structures (ledges, small nodes)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduces size to 64

            # Layer 2: Detect larger structures (value areas, balance zones)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduces size to 32

            # Layer 3: High level shape recognition (P-shape, b-shape)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Pooling: Summarize the whole profile into 1 vector
        )

        self.fc = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, 128) -> needs (Batch, 1, 128) for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.net(x)
        x = x.flatten(1)  # Flatten (Batch, 64, 1) -> (Batch, 64)
        return self.relu(self.fc(x))