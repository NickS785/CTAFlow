import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Ensures that the output at time 't' only depends on inputs from 't' and earlier.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        # Slice the output to remove the 'future' padding
        x = self.conv(x)
        return x[:, :, :-self.padding]


class DeepVolResidualBlock(nn.Module):
    """
    A single residual layer for DeepVol featuring gated activations (Tanh/Sigmoid).
    """

    def __init__(self, channels, kernel_size, dilation):
        super(DeepVolResidualBlock, self).__init__()
        self.causal_conv = CausalConv1d(channels, channels, kernel_size, dilation)

        # Gated activation logic (common in WaveNet/DeepVol architectures)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # 1x1 conv to project back for the residual connection
        self.res_map = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        combined = self.causal_conv(x)

        # Gated Linear Unit (GLU) mechanism
        out = self.tanh(combined) * self.sigmoid(combined)
        out = self.res_map(out)

        return out + residual


class DeepVolEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=4):
        super(DeepVolEncoder, self).__init__()
        self.init_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Exponentially increasing dilation: 1, 2, 4, 8...
        self.res_blocks = nn.ModuleList([
            DeepVolResidualBlock(hidden_dim, kernel_size=3, dilation=2 ** i)
            for i in range(layers)
        ])

    def forward(self, x):
        # x shape: (Batch, Features, Seq_Length)
        x = self.init_conv(x)
        for block in self.res_blocks:
            x = block(x)
        return x  # Returns volatility embeddings