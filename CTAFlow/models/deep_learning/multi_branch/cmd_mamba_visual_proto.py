from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .cmd_mamba import CMDMamba, CMDMambaConfig


class _RasterBasicBlock(nn.Module):
    """ResNet basic block for raster grids [Time x Bins]."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: Union[int, Tuple[int, int]] = 1,
    ):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride)

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != (1, 1) or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class TimePreservingRasterResNet(nn.Module):
    """
    Visual short encoder for CMDMamba.

    Input:  [B, T, C, Bins]
    Output: [B, T, d_model]

    Time is preserved while bins are compressed with stride (1, 2).
    """

    def __init__(
        self,
        in_ch: int = 4,
        d_model: int = 128,
        layers: Tuple[int, int, int] = (2, 2, 2),
        base_filters: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_planes = int(base_filters)

        self.stem_conv = nn.Conv2d(
            in_ch,
            base_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(base_filters)
        self.stem_act = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(base_filters, int(layers[0]), stride=1)
        self.layer2 = self._make_layer(base_filters * 2, int(layers[1]), stride=(1, 2))
        self.layer3 = self._make_layer(base_filters * 4, int(layers[2]), stride=(1, 2))

        out_planes = base_filters * 4 * _RasterBasicBlock.expansion
        self.proj = nn.Sequential(
            nn.Linear(out_planes, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: Union[int, Tuple[int, int]],
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(_RasterBasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * _RasterBasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x_short: torch.Tensor) -> torch.Tensor:
        if x_short.dim() != 4:
            raise ValueError(f"Expected [B,T,C,Bins], got {tuple(x_short.shape)}")

        # [B,T,C,Bins] -> [B,C,T,Bins]
        x = x_short.permute(0, 2, 1, 3).contiguous()

        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Pool only bins, keep time axis.
        x = x.mean(dim=3)  # [B,C_out,T]
        x = x.permute(0, 2, 1).contiguous()  # [B,T,C_out]
        return self.proj(x)  # [B,T,d_model]


@dataclass
class CMDMambaVisualConfig(CMDMambaConfig):
    """Prototype config: CMDMamba + visual short encoder."""

    short_resnet_layers: Tuple[int, int, int] = (2, 2, 2)
    short_resnet_base_filters: int = 32


class CMDMambaVisualProto(CMDMamba):
    """
    Prototype CMDMamba variant that uses a visual ResNet short encoder before Mamba.

    Long branch + DeepVol branch logic are inherited from CMDMamba.
    """

    def __init__(
        self,
        long_input_dim: int,
        time_feat_dim: int = 0,
        cfg: Optional[Union[CMDMambaVisualConfig, Dict]] = None,
        raster_norm_means: Optional[torch.Tensor] = None,
        raster_norm_stds: Optional[torch.Tensor] = None,
    ):
        if cfg is None:
            visual_cfg = CMDMambaVisualConfig()
        elif isinstance(cfg, dict):
            visual_cfg = CMDMambaVisualConfig(**cfg)
        elif isinstance(cfg, CMDMambaVisualConfig):
            visual_cfg = cfg
        else:
            # Accept base config objects and augment with visual defaults.
            cfg_dict = vars(cfg)
            visual_cfg = CMDMambaVisualConfig(**cfg_dict)

        # CMDMamba only accepts base config fields.
        base_cfg_kwargs = {
            f.name: getattr(visual_cfg, f.name)
            for f in fields(CMDMambaConfig)
        }
        base_cfg = CMDMambaConfig(**base_cfg_kwargs)

        super().__init__(
            long_input_dim=long_input_dim,
            time_feat_dim=time_feat_dim,
            cfg=base_cfg,
            raster_norm_means=raster_norm_means,
            raster_norm_stds=raster_norm_stds,
        )

        self.visual_cfg = visual_cfg
        self.short_encoder = TimePreservingRasterResNet(
            in_ch=base_cfg.raster_channels,
            d_model=base_cfg.d_model,
            layers=visual_cfg.short_resnet_layers,
            base_filters=visual_cfg.short_resnet_base_filters,
            dropout=base_cfg.dropout,
        )

