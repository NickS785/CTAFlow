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


class SpatialFuse(nn.Module):
    def __init__(self, d_spatial: int, mode: str = "gated", temperature: float = 2.0):
        super().__init__()
        self.mode = mode
        self.temperature = temperature  # Add temperature scaling
        self.last_importance = None

        if mode == "gated":
            # 1. ADD THIS: Normalize inputs so they compete fairly
            self.gate_norm = nn.LayerNorm(d_spatial * 2)

            self.gate = nn.Sequential(
                nn.Linear(d_spatial * 2, d_spatial),
                nn.GELU(),
                nn.Linear(d_spatial, 2),
            )
            # Initialize for fair 50/50 start
            nn.init.xavier_uniform_(self.gate[2].weight)
            nn.init.constant_(self.gate[2].bias, 0.0)

        elif mode != "mean":
            raise ValueError(f"Unknown SpatialFuse mode: {mode}")

    def forward(self, z_profile, z_nb=None, return_importance=False):
        if z_nb is None:
            # ... (Keep existing single-modality logic) ...
            if return_importance:
                batch_size = z_profile.size(0)
                importance = torch.tensor([[1.0, 0.0]], device=z_profile.device).expand(batch_size, 2)
                self.last_importance = importance
                return z_profile, importance
            return z_profile

        if self.mode == "mean":
            # ... (Keep existing mean logic) ...
            fused = 0.5 * (z_profile + z_nb)
            # Update tracking for mean mode too
            self.last_importance = torch.tensor([[0.5, 0.5]], device=z_profile.device).expand(z_profile.size(0), 2)
            return (fused, self.last_importance) if return_importance else fused

        # --- THE FIX ---

        # 1. Concatenate
        cat_input = torch.cat([z_profile, z_nb], dim=-1)

        # 2. Normalize BEFORE Gating (Crucial!)
        gate_input = self.gate_norm(cat_input)

        # 3. Calculate Logits
        logits = self.gate(gate_input)

        # 4. Apply Temperature Scaling (Softens the decision)
        weights = F.softmax(logits / self.temperature, dim=-1)

        self.last_importance = weights.detach()

        # 5. Fuse (Use original inputs, not normalized ones)
        fused = weights[:, 0:1] * z_profile + weights[:, 1:2] * z_nb

        if return_importance:
            return fused, weights
        return fused

    def get_importance_stats(self):
        """
        Get statistics about modality importance from last forward pass.

        Returns
        -------
        dict or None
            Dictionary with keys:
            - 'profile_mean': Mean importance of profile modality
            - 'nb_mean': Mean importance of number bars modality
            - 'profile_std': Std deviation of profile importance
            - 'nb_std': Std deviation of number bars importance
            Returns None if no forward pass has been made yet
        """
        if self.last_importance is None:
            return None

        profile_weights = self.last_importance[:, 0]
        nb_weights = self.last_importance[:, 1]

        return {
            'profile_mean': profile_weights.mean().item(),
            'nb_mean': nb_weights.mean().item(),
            'profile_std': profile_weights.std().item(),
            'nb_std': nb_weights.std().item(),
            'profile_dominant': (profile_weights > nb_weights).float().mean().item(),
        }

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

        # Pack for LSTM (clamp to at least 1 to avoid errors with zero-length sequences)
        seq_len_clamped = seq_len.clamp(min=1).detach().to("cpu")
        packed = pack_padded_sequence(h, lengths=seq_len_clamped, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # (B, T, lstm_dim)

        out = self.post(out)  # (B, T, D)

        # Mask for padding positions
        device = x_seq.device
        mask = (torch.arange(T, device=device)[None, :] < seq_len[:, None]).int()  # (B, T)

        z = self.pool(out, mask=mask)  # (B, D)
        return z


class NumberBarsEncoder(nn.Module):
    """
    CNN-Transformer hybrid to process NumbersBars
    x_nb: (B, T, BINS, C) -> (B, d_model)
    """
    def __init__(self, c_in=3, d_model=128, conv_ch=64, n_tf=2, n_heads=4, dropout=0.1, max_T=16):
        super().__init__()
        self.out_dim = d_model

        # per-slice (over price bins) conv encoder
        self.proj = nn.Conv1d(c_in, conv_ch, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(conv_ch, conv_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_bins = nn.AdaptiveAvgPool1d(1)
        self.slice_to_model = nn.Sequential(
            nn.Linear(conv_ch, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # temporal encoder across slices (T)
        self.pos = nn.Embedding(max_T, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_tf)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_nb, nb_lengths=None):
        """
        x_nb: (B, T, BINS, C)
        nb_lengths: (B,) lengths for each sample (optional)
        """
        B, T, BINS, C = x_nb.shape

        # (B*T, C, BINS)
        x = x_nb.reshape(B*T, BINS, C).transpose(1, 2)

        x = self.proj(x)            # (B*T, conv_ch, BINS)
        x = self.conv(x)            # (B*T, conv_ch, BINS)
        x = self.pool_bins(x).squeeze(-1)  # (B*T, conv_ch)

        x = self.slice_to_model(x).reshape(B, T, -1)  # (B, T, d_model)

        # key padding mask: True where PAD
        key_padding_mask = None
        if nb_lengths is not None:
            t = torch.arange(T, device=x.device)[None, :]
            key_padding_mask = t >= nb_lengths[:, None]  # (B,T) bool

        # add positions + transformer
        pos = torch.arange(T, device=x.device)[None, :].expand(B, T)
        x = x + self.pos(pos)
        x = self.tf(x, src_key_padding_mask=key_padding_mask)  # (B,T,d_model)

        # masked mean pool
        if nb_lengths is None:
            z = x.mean(dim=1)
        else:
            mask = (~key_padding_mask).float()  # (B,T)
            z = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        return self.norm(z)  # (B,d_model)

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


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block adapted for Financial Time-Price grids.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Conv2d over (Time, PriceBins)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RasterResNet(nn.Module):
    """
    "3D" ResNet implementation for Rasterized VPIN data.

    Treats the input (Batch, Time, Channels, Bins) as an image of shape
    (Batch, Channels, Height=Time, Width=Bins).

    This effectively performs Spatio-Temporal convolution:
    - Vertical patterns = Temporal evolution (velocity)
    - Horizontal patterns = Price structure (nodes/ledges)
    """

    def __init__(
            self,
            in_ch=4,  # Channels (Density, Vol, Imbal, Ret)
            d_model=128,  # Output dimension
            layers=[2, 2, 2],  # Depth of ResNet blocks
            base_filters=32
    ):
        super().__init__()
        self.in_planes = base_filters

        # 1. Stem: Initial processing
        # Note: We do NOT downsample Time immediately if it's short
        self.conv1 = nn.Conv2d(in_ch, base_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)

        # 2. ResNet Layers
        # Layer 1: Keep dimensions (capture fine details)
        self.layer1 = self._make_layer(BasicBlock, base_filters, layers[0], stride=1)

        # Layer 2: Downsample Price Bins (Width), keep Time (Height) or downsample both?
        # Stride (2, 2) reduces both Time and Price resolution
        self.layer2 = self._make_layer(BasicBlock, base_filters * 2, layers[1], stride=2)

        # Layer 3: Downsample again
        self.layer3 = self._make_layer(BasicBlock, base_filters * 4, layers[2], stride=2)

        # 3. Aggregation Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Pools remaining Time & Price
        self.fc = nn.Linear(base_filters * 4 * BasicBlock.expansion, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        """
        x: (Batch, T, C, Bins)
        lengths: Ignored (CNNs handle padding via masking or just learning zero-features)
        """
        # Permute to (Batch, Channels, Time, Bins) for Conv2d
        x = x.permute(0, 2, 1, 3)

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)  # (B, C_out, 1, 1)
        x = x.flatten(1)  # (B, C_out)
        x = self.fc(x)  # (B, d_model)
        return self.norm(x)



class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D.
    Allows the model to dynamically weight channels (e.g., focus on Delta vs Volume).
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional SE attention.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Squeeze-and-Excitation
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class MarketProfileResNet(nn.Module):
    """
    Enhanced Market Profile Encoder.
    Replaces the simple CNN with a ResNet-1D + SE Attention.

    Structure:
    1. Stem (Conv1d -> BN -> ReLU)
    2. Layer 1 (ResBlock - extract fine details like ledges)
    3. Layer 2 (ResBlock - extract shape like P/b profiles)
    4. Layer 3 (ResBlock - extract global balance/imbalance)
    5. Global Pool -> Output
    """

    def __init__(self, in_channels=3, d_model=128, layers=[2, 2, 2]):
        super().__init__()
        self.inplanes = 32

        # 1. Stem
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 2. ResNet Layers
        self.layer1 = self._make_layer(32, layers[0])
        self.layer2 = self._make_layer(64, layers[1], stride=2)
        self.layer3 = self._make_layer(128, layers[2], stride=2)

        # 3. Output Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(ResBlock1D(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock1D(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, Bins)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return self.norm(x)

# ----------------------------
# Spatial-Temporal Encoder (Rasterized VPIN)
# ----------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequences."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Seq, Feature)
        x = x + self.pe[:x.size(1), :]
        return x


class SpatialTemporalEncoder(nn.Module):
    """
    CNN-Transformer hybrid for rasterized VPIN sequences.

    This encoder processes rasterized VPIN data where each time bar (e.g., 15-min interval)
    is represented as a 2D grid with channels [Density, Volume, Imbalance, Returns].

    Architecture:
        1. **Spatial (CNN)**: Processes each time bar independently to extract shape features
           (e.g., "this is a trend bar", "this is a balanced bar")
        2. **Temporal (Transformer)**: Processes the sequence of bar embeddings to understand
           evolution (e.g., "value is migrating higher")

    Input: (B, T, C, Bins) where:
        - B: batch size
        - T: number of time bars (e.g., 4 x 15-min bars)
        - C: channels [Density, LogVolume, Imbalance, Returns]
        - Bins: price levels (e.g., 64)

    Output: (B, d_model)

    This is designed to work with SequenceRasterizer from CTAFlow.features.volume.vpin.

    Example:
        >>> rasterizer = SequenceRasterizer(bins=64, span_pct=0.01)
        >>> encoder = SpatialTemporalEncoder(num_bars=4, in_ch=4, num_bins=64, d_model=128)
        >>> # For each date's VPIN data:
        >>> rasterized = rasterizer.rasterize(vpin_df, num_bars=4)  # (4, 4, 64)
        >>> batch = rasterized.unsqueeze(0)  # (1, 4, 4, 64)
        >>> embedding = encoder(batch)  # (1, 128)
    """

    def __init__(
        self,
        num_bars: int = 4,
        in_ch: int = 4,
        num_bins: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Parameters
        ----------
        num_bars : int
            Number of time bars per sample (e.g., 4 for 4x15min bars)
        in_ch : int
            Number of input channels [Density, Volume, Imbalance, Returns]
        num_bins : int
            Number of price bins in the rasterized grid
        d_model : int
            Transformer/output dimension
        nhead : int
            Number of attention heads
        n_layers : int
            Number of transformer encoder layers
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.out_dim = d_model
        self.num_bars = num_bars
        self.num_bins = num_bins

        # --- 1. Spatial Encoder (Shared CNN) ---
        # Applies to each bar independently: (Batch*T, C, Bins)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),  # bins -> bins/2

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # bins/2 -> bins/4

            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            # Output: (B*T, d_model, bins/4)
        )

        # Flatten CNN output to get a vector per bar
        # bins/4 * d_model -> d_model
        cnn_out_size = d_model * (num_bins // 4)
        self.adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # --- 2. Temporal Encoder (Transformer) ---
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_bars)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final aggregation (Attention Pooling over time)
        self.attn_pool = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, C, Bins) - rasterized VPIN data
        lengths : torch.Tensor, optional
            Shape (B,) - actual sequence lengths for masking

        Returns
        -------
        torch.Tensor
            Shape (B, d_model) - aggregated embedding
        """
        b, t, c, h = x.shape

        # 1. Fold Time into Batch for CNN
        # (B*T, C, H)
        x_flat = x.view(b * t, c, h)

        # 2. Extract Spatial Features
        # (B*T, d_model, H/4)
        cnn_feat = self.cnn(x_flat)

        # 3. Create Bar Embeddings
        # (B*T, d_model)
        bar_embeds = self.adapter(cnn_feat)

        # 4. Unfold Time
        # (Batch, T, d_model)
        x_seq = bar_embeds.view(b, t, -1)

        # 5. Transformer Pass
        x_seq = self.pos_encoder(x_seq)

        # Create attention mask if lengths provided
        key_padding_mask = None
        if lengths is not None:
            time_idx = torch.arange(t, device=x.device)[None, :]
            key_padding_mask = time_idx >= lengths[:, None]  # (B, T) True where pad

        # (Batch, T, d_model)
        memory = self.transformer(x_seq, src_key_padding_mask=key_padding_mask)

        # 6. Aggregation (Attention pooling)
        # Let model decide which time bars are most important
        scores = self.attn_pool(memory)  # (B, T, 1)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(memory * weights, dim=1)  # (B, d_model)

        return self.norm(context)


class MarketProfileCNN(nn.Module):
    """
    A lightweight 1D CNN specifically designed for Market Profile (histogram) data.
    Input Shape: (Batch, Channels, Bins) -> e.g., (32, 3, 128)
    """

    def __init__(self, in_channels, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: Capture local shape (nodes/ledges)
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),  # 128 -> 64

            # Block 2: Capture structure (balance/imbalance)
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),  # 64 -> 32

            # Block 3: Global abstraction
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1)  # Flatten to (Batch, 64, 1)
        )

        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)  # (Batch, 64)
        return self.fc(x)  # (Batch, out_dim)


class IntradayRNN(nn.Module):
    """
    Encoder for the Sequential branch (Intraday VPIN time-series).
    """

    def __init__(self, input_dim, d_model=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths=None):
        # x: (B, SeqLen, Features)
        if lengths is not None:
            # Handle variable lengths if provided
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.rnn(x_packed)
        else:
            _, h_n = self.rnn(x)

        # Take last hidden state: (NumLayers, B, Hidden) -> (B, Hidden)
        embedding = h_n[-1]
        return self.norm(embedding)


class IntradayTransformer(nn.Module):
    """
    Transformer encoder for the Sequential branch (Intraday VPIN time-series).

    Drop-in replacement for IntradayRNN, better suited for sequence lengths
    of 24-96 where self-attention captures long-range dependencies more
    effectively than a GRU.

    Input:  (B, SeqLen, Features)
    Output: (B, d_model)
    """

    def __init__(self, input_dim, d_model=128, num_layers=2, dropout=0.2, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim

        # Learnable per-feature gate (softmax → interpretable importance)
        self.feature_gate_logits = nn.Parameter(torch.zeros(input_dim))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(d_model, max_len=256)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.pool_score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.last_tracker: dict = {}

    def forward(self, x, lengths=None):
        B, T, _ = x.shape

        # Per-feature gating (softmax over input_dim)
        feature_weights = F.softmax(self.feature_gate_logits, dim=-1)  # (input_dim,)
        x_gated = x * (feature_weights * self.input_dim)  # scale so mean gate ≈ 1

        h = self.input_proj(x_gated)
        h = self.pos_enc(h)

        key_padding_mask = None
        if lengths is not None:
            t_idx = torch.arange(T, device=x.device)[None, :]
            key_padding_mask = t_idx >= lengths[:, None]

        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Attention pooling with weight tracking
        logits = self.pool_score(h).squeeze(-1)  # (B, T)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask, torch.finfo(logits.dtype).min)
        pool_weights = F.softmax(logits, dim=-1)  # (B, T)
        embedding = torch.einsum("bt,btd->bd", pool_weights, h)

        # Track pool attention + feature importance
        self.last_tracker = {
            "pool_weights": pool_weights.detach(),
            "pool_entropy": -(pool_weights * (pool_weights + 1e-8).log()).sum(dim=-1).mean().item(),
            "pool_max_weight": pool_weights.max(dim=-1).values.mean().item(),
            "feature_weights": feature_weights.detach(),
            "feature_entropy": -(feature_weights * (feature_weights + 1e-8).log()).sum().item(),
            "feature_max_weight": feature_weights.max().item(),
        }

        return self.norm(embedding)


class MetaModalityEncoder(nn.Module):
    """Encodes asset identity + calendar time as its own modality.

    Expects a `meta` dict with (recommended):
      - ticker_id:         (B,) long
      - asset_class_id:    (B,) long
      - asset_subclass_id: (B,) long
      - month:             (B, W) long in [1..12] (0 allowed for unknown)
      - dow:               (B, W) long in [0..6]
      - doy_sin:           (B, W) float
      - doy_cos:           (B, W) float

    Returns:
      - z_meta_days:    (B, W, d_model)  per-day meta embeddings
      - z_meta_window:  (B, meta_hidden) pooled/window representation
    """

    def __init__(
        self,
        d_model: int = 128,
        ctx_dim: int = 64,
        time_dim: int = 64,
        meta_hidden: int = 64,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- identity / context ---
        self.ticker_emb = nn.Embedding(n_tickers, ctx_dim)
        self.class_emb = nn.Embedding(n_asset_classes, ctx_dim)
        self.subclass_emb = nn.Embedding(n_asset_subclasses, ctx_dim)

        # --- calendar time ---
        # month uses 1..12; reserve 0 for unknown/pad
        self.month_emb = nn.Embedding(13, time_dim, padding_idx=0)
        self.dow_emb = nn.Embedding(7, time_dim)
        self.doy_proj = nn.Sequential(
            nn.Linear(2, time_dim),
            nn.GELU(),
            nn.LayerNorm(time_dim),
        )

        # --- fuse ctx + time into per-day token ---
        self.fuse = nn.Sequential(
            nn.Linear(ctx_dim + time_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- temporal aggregator (seasonality detector) ---
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=meta_hidden, batch_first=True)

        self.out_dim = meta_hidden
        self.d_model = d_model
        self.meta_hidden = meta_hidden

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, meta: dict, W: int, device=None):
        # If meta is missing, emit zeros (still a modality, but blank)
        if meta is None:
            B = 1 if device is None else None  # cannot infer B safely
            raise ValueError("MetaModalityEncoder requires a meta dict (ticker/time).")

        # --- required ids ---
        ticker_id = meta.get("ticker_id", None)
        asset_class_id = meta.get("asset_class_id", None)
        asset_subclass_id = meta.get("asset_subclass_id", None)

        if ticker_id is None or asset_class_id is None or asset_subclass_id is None:
            raise ValueError("meta dict must include ticker_id, asset_class_id, asset_subclass_id")

        if device is None:
            device = ticker_id.device

        B = ticker_id.shape[0]


        # --- context embedding (B, ctx_dim) -> expand to (B, W, ctx_dim) ---
        z_ctx = (
            self.ticker_emb(ticker_id)
            + self.class_emb(asset_class_id)
            + self.subclass_emb(asset_subclass_id)
        )
        z_ctx = z_ctx.unsqueeze(1).expand(B, W, z_ctx.size(-1))

        # --- time embedding (B, W, time_dim) ---
        month = meta.get("month", None)
        dow = meta.get("dow", None)
        doy_sin = meta.get("doy_sin", None)
        doy_cos = meta.get("doy_cos", None)

        if month is None or dow is None or doy_sin is None or doy_cos is None:
            # allow identity-only meta if dates weren't provided
            z_time = torch.zeros((B, W, self.month_emb.embedding_dim), device=device)
        else:
            # shapes: month/dow (B,W), doy_sin/cos (B,W)
            z_time = self.month_emb(month.clamp(0, 12)) + self.dow_emb(dow.clamp(0, 6))
            doy = torch.stack([doy_sin, doy_cos], dim=-1).to(z_time.dtype)  # (B,W,2)
            z_time = z_time + self.doy_proj(doy)

        # --- per-day meta token + temporal pooling ---
        z_meta_days = self.fuse(torch.cat([z_ctx, z_time], dim=-1))  # (B,W,d_model)
        _, (h_n, _) = self.lstm(z_meta_days)                         # h_n: (1,B,meta_hidden)
        z_meta_window = h_n[-1]                                      # (B,meta_hidden)
        return z_meta_days, z_meta_window
