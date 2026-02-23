\This is a great design choice. By setting the stride to `(1, 2)` instead of the standard `2`, we can **preserve the temporal resolution** (keeping every time step) while still **compressing the price bins** (feature dimension).

This effectively turns the ResNet into a powerful **Feature Tokenizer**: it looks at the full 32-bin or 64-bin depth for every single time bar and extracts a dense embedding vector `d_model` representing the "shape" of the order flow at that specific moment.

Here is the implementation of the **Time-Preserving Raster ResNet** and its integration.

### 1. The Time-Preserving Encoder (`encoders.py`)

We modify the `SequentialRasterResNet` to use asymmetric strides.

* **Vertical Stride (Time):** 1 (No Downsampling)
* **Horizontal Stride (Price):** 2 (Downsampling)

```python
# In CTAFlow/models/deep_learning/encoders.py

class TimePreservingRasterResNet(nn.Module):
    """
    ResNet backbone that preserves Time resolution but compresses Price Bins.
    
    Input:  (B, T, C, Bins) 
    Output: (B, T, d_model)
    
    Strides are set to (1, 2) to downsample Price dimension only.
    """
    def __init__(
        self,
        in_ch=4, 
        d_model=128, 
        layers=[2, 2, 2], 
        base_filters=32,
        dropout=0.1
    ):
        super().__init__()
        self.in_planes = base_filters

        # 1. Stem
        # Stride=1 preserves both Time and Bins initially
        self.conv1 = nn.Conv2d(in_ch, base_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU(inplace=True)

        # 2. ResNet Layers
        # Layer 1: Stride 1 (No downsampling)
        self.layer1 = self._make_layer(BasicBlock, base_filters, layers[0], stride=1)
        
        # Layer 2: Stride (1, 2) -> Keep Time, Halve Bins
        self.layer2 = self._make_layer(BasicBlock, base_filters * 2, layers[1], stride=(1, 2))
        
        # Layer 3: Stride (1, 2) -> Keep Time, Halve Bins again (Total Price downsample /4)
        self.layer3 = self._make_layer(BasicBlock, base_filters * 4, layers[2], stride=(1, 2))

        # 3. Projection Head
        self.out_planes = base_filters * 4 * BasicBlock.expansion
        
        # We pool remaining Price Bins to 1x1, but since we kept Time, 
        # we treat Time as the "Batch" dimension for the linear layer or use Conv1d
        self.proj = nn.Sequential(
            nn.Linear(self.out_planes, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        # Input: (B, T, C, Bins) -> Permute to (B, C, T, Bins) for Conv2d
        # T is treated as 'Height', Bins as 'Width'
        x = x.permute(0, 2, 1, 3) 

        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        
        # x is now (B, C_out, T, Bins_new)
        
        # Collapse remaining Price Bins
        x = x.mean(dim=3)  # (B, C_out, T)

        # Permute for Projection: (B, T, C_out)
        x = x.permute(0, 2, 1) 
        
        return self.proj(x) # (B, T, d_model)

```

### 2. Integration into `CMDMamba` (`cmd_mamba.py`)

Since the encoder now preserves the sequence length, the integration is simpler (no need to interpolate masks).

**Step A: Update Config**

```python
@dataclass
class CMDMambaConfig:
    # ... existing ...
    short_encoder_type: str = "resnet_iso" # 'resnet_iso' for isometric time (no stride)
    # ... existing ...

```

**Step B: Update Model Initialization**

```python
# In CTAFlow/models/deep_learning/multi_branch/cmd_mamba.py

class CMDMamba(nn.Module):
    def __init__(self, ...):
        # ... 
        
        if cfg.short_encoder_type == "resnet_iso":
            from CTAFlow.models.deep_learning.encoders import TimePreservingRasterResNet
            self.short_encoder = TimePreservingRasterResNet(
                in_ch=cfg.raster_channels,
                d_model=d_model,
                dropout=cfg.dropout,
                layers=[2, 2, 2],
                base_filters=32
            )
        else:
            self.short_encoder = DenseRasterEncoder(...)
            
        # ...

```

**Step C: Update `forward**`
Because input  equals output , the masking logic is standard.

```python
    def forward(self, x_short, ..., short_mask=None, ...):
        # 1. Preprocess
        z_short = self.raster_pre(x_short)
        
        # 2. Encode (T is preserved)
        z_short = self.short_encoder(z_short)  # [B, T, D]
        
        # 3. Apply Mask (Direct matching)
        if short_mask is not None:
            # Mask shape [B, T] -> [B, T, 1] for broadcasting
            z_short = z_short * short_mask.unsqueeze(-1).to(z_short.dtype)

        # 4. Mamba Mixer
        for blk in self.short_blocks:
            z_short = blk(z_short)
            
        # ... rest of model ...

```

### Benefit of this Approach

You get the best of both worlds:

1. **Deep Visual Features:** The ResNet can learn complex spatial patterns in the order book (e.g., "Iceberg detected at level 5 while price pushes up").
2. **Precise Timing:** Mamba receives a token for *every* 5-minute bar (or whatever your raster resolution is), allowing it to react instantly to a change in market regime, rather than waiting for a pooled "summary" token covering 30 minutes.