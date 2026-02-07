from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Mamba = None  # type: ignore


class TimePatcher(nn.Module):
    """
    Patchify [B, T, F] into [B, N, D] tokens.

    Defaults (patch=2, stride=2) are designed for 15-minute bars:
    256 bars -> 128 long tokens (30-minute stride-equivalent).
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        patch: int = 2,
        stride: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch = int(patch)
        self.stride = int(stride)
        self.proj = nn.Sequential(
            nn.Linear(self.patch * in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        if x.dim() != 3:
            raise ValueError(f"TimePatcher expects [B,T,F], got {tuple(x.shape)}")
        bsz, t, f = x.shape
        if t < self.patch:
            pad = self.patch - t
            x = F.pad(x, (0, 0, pad, 0))
            t = x.shape[1]
        windows = x.unfold(1, self.patch, self.stride).contiguous()  # [B,N,patch,F]
        windows = windows.view(bsz, windows.shape[1], self.patch * f)
        return self.proj(windows)

    def expected_tokens(self, t: int) -> int:
        if t <= 0:
            return 0
        if t < self.patch:
            return 1
        return 1 + (t - self.patch) // self.stride


class RasterPreprocessor(nn.Module):
    """
    Channel-aware raster preprocessing for channels:
      [Density, LogVolume, Imbalance, Returns]
    """

    def __init__(
        self,
        means: Optional[torch.Tensor] = None,
        stds: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        use_atanh_imbalance: bool = True,
        returns_scale: float = 1.0,
    ):
        super().__init__()
        self.eps = float(eps)
        self.use_atanh_imbalance = bool(use_atanh_imbalance)
        self.returns_scale = float(returns_scale)

        if means is None:
            means = torch.zeros(4, dtype=torch.float32)
        if stds is None:
            stds = torch.ones(4, dtype=torch.float32)
        self.register_buffer("means", means.float())
        self.register_buffer("stds", stds.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,4,H]
        if x.dim() != 4 or x.shape[2] != 4:
            raise ValueError(f"RasterPreprocessor expects [B,T,4,H], got {tuple(x.shape)}")

        density = x[:, :, 0]
        logvol = x[:, :, 1]
        imb = x[:, :, 2]
        ret = x[:, :, 3]

        density = torch.sqrt(torch.clamp(density, min=0.0))
        imb = torch.clamp(imb, -1.0, 1.0)
        if self.use_atanh_imbalance:
            imb = torch.atanh(0.999 * imb)

        ret = torch.clamp(ret * self.returns_scale, -10.0, 10.0)
        y = torch.stack([density, logvol, imb, ret], dim=2)
        y = (y - self.means[None, None, :, None]) / (self.stds[None, None, :, None] + self.eps)
        return y


class CausalConv1d(nn.Module):
    """Left-padded causal convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class DualConvFFN(nn.Module):
    """Convolutional FFN for noisy short-term branch."""

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.conv1 = nn.Conv1d(d_model, hidden, kernel_size=1)
        self.act1 = nn.SiLU()
        self.conv2 = CausalConv1d(hidden, hidden, kernel_size=3, groups=hidden)
        self.act2 = nn.SiLU()
        self.conv3 = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D] -> [B,D,T]
        z = x.transpose(1, 2)
        z = self.act1(self.conv1(z))
        z = self.act2(self.conv2(z))
        z = self.dropout(self.conv3(z))
        return z.transpose(1, 2)


class _MambaCore(nn.Module):
    """Mamba wrapper with depthwise-conv fallback when dependency is missing."""

    def __init__(self, d_model: int, d_state: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.is_fallback = Mamba is None
        if not self.is_fallback:
            self.core = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            warnings.warn(
                "mamba_ssm is not installed. Falling back to depthwise temporal conv.",
                RuntimeWarning,
            )
            k = max(3, int(d_conv) * 2 - 1)
            self.core = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fallback:
            return self.core(x.transpose(1, 2)).transpose(1, 2)
        return self.core(x)


class CMDMambaBlock(nn.Module):
    """Short-branch block: Mamba mixer + DualConvFFN."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mamba = _MambaCore(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ffn = DualConvFFN(d_model=d_model, expansion=2, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mamba(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class StandardMambaBlock(nn.Module):
    """Long-branch block: Mamba mixer + MLP."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mamba = _MambaCore(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mamba(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class DenseRasterEncoder(nn.Module):
    """
    Encode raster short-stream into one token per time step.

    Input:  [B, T, C, H]
    Output: [B, T, D]
    """

    def __init__(self, in_ch: int = 4, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_short: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H] -> [B,C,H,T]
        z = x_short.permute(0, 2, 3, 1).contiguous()
        z = self.net(z)  # [B,D,H',T]
        z = z.mean(dim=2)  # pool over H' -> [B,D,T]
        return z.permute(0, 2, 1)  # [B,T,D]


class LongShortRouter(nn.Module):
    """Soft router between long and short branch summaries."""

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, z_long: torch.Tensor, z_short: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(torch.cat([z_long, z_short], dim=-1))  # [B,2]
        w = torch.softmax(logits, dim=-1)
        fused = w[:, 0:1] * z_long + w[:, 1:2] * z_short
        return fused, w


@dataclass
class CMDMambaConfig:
    # Frequency semantics (defaults assume 15-minute bars)
    base_bar_minutes: int = 15

    # Long stream
    long_patch: int = 2
    long_stride: int = 2
    n_long_layers: int = 3

    # Short stream
    n_short_layers: int = 2
    raster_channels: int = 4
    raster_bins: int = 32

    # Shared model dimensions
    d_model: int = 128
    dropout: float = 0.1
    d_state_long: int = 64
    d_state_short: int = 16
    d_conv: int = 4
    expand: int = 2

    # Head
    task: str = "regression"  # "regression" or "classification"
    out_dim: int = 1


class CMDMamba(nn.Module):
    """
    Dual-stream CMDMamba using dense raster encoder only.

    Inputs
    ------
    x_short : [B, T_short, 4, 32]
    t_short : [B, T_short, F_time] (optional)
    x_long  : [B, T_long, F_long]
    t_long  : [B, T_long, F_time] (optional)
    short_mask : [B, T_short] (optional; 1 for valid positions)
    """

    def __init__(
        self,
        long_input_dim: int,
        time_feat_dim: int = 0,
        cfg: CMDMambaConfig = CMDMambaConfig(),
        raster_norm_means: Optional[torch.Tensor] = None,
        raster_norm_stds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        self.time_feat_dim = int(time_feat_dim)

        self.raster_pre = RasterPreprocessor(
            means=raster_norm_means,
            stds=raster_norm_stds,
            eps=1e-6,
            use_atanh_imbalance=True,
            returns_scale=1.0,
        )
        self.short_encoder = DenseRasterEncoder(
            in_ch=cfg.raster_channels,
            d_model=d_model,
            dropout=cfg.dropout,
        )

        self.long_patcher = TimePatcher(
            in_features=long_input_dim,
            d_model=d_model,
            patch=cfg.long_patch,
            stride=cfg.long_stride,
            dropout=cfg.dropout,
        )

        self.use_time = self.time_feat_dim > 0
        if self.use_time:
            self.time_short_proj = nn.Sequential(
                nn.Linear(self.time_feat_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(cfg.dropout),
            )
            self.time_long_patcher = TimePatcher(
                in_features=self.time_feat_dim,
                d_model=d_model,
                patch=cfg.long_patch,
                stride=cfg.long_stride,
                dropout=cfg.dropout,
            )

        self.long_blocks = nn.ModuleList(
            [
                StandardMambaBlock(
                    d_model=d_model,
                    d_state=cfg.d_state_long,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    dropout=cfg.dropout,
                )
                for _ in range(max(1, cfg.n_long_layers))
            ]
        )
        self.short_blocks = nn.ModuleList(
            [
                CMDMambaBlock(
                    d_model=d_model,
                    d_state=cfg.d_state_short,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    dropout=cfg.dropout,
                )
                for _ in range(max(1, cfg.n_short_layers))
            ]
        )
        self.long_norm = nn.LayerNorm(d_model)
        self.short_norm = nn.LayerNorm(d_model)

        self.router = LongShortRouter(d_model=d_model, hidden=max(64, d_model), dropout=cfg.dropout)

        if cfg.task not in {"regression", "classification"}:
            raise ValueError(f"Unknown task={cfg.task}")
        head_out = int(cfg.out_dim)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, head_out),
        )

        self.last_tracker: Dict[str, object] = {}

    @staticmethod
    def _last_valid(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # tokens: [B,T,D], mask: [B,T]
        if mask is None:
            return tokens[:, -1, :]
        lengths = mask.long().sum(dim=1).clamp(min=1)  # [B]
        idx = (lengths - 1).view(tokens.size(0), 1, 1).expand(tokens.size(0), 1, tokens.size(2))
        return tokens.gather(dim=1, index=idx).squeeze(1)

    def forward(
        self,
        x_short: torch.Tensor,
        x_long: torch.Tensor,
        t_short: Optional[torch.Tensor] = None,
        t_long: Optional[torch.Tensor] = None,
        short_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_features: bool = False,
    ):
        # Short branch
        z_short = self.raster_pre(x_short)
        z_short = self.short_encoder(z_short)  # [B,Ts,D]
        if self.use_time and t_short is not None:
            z_short = z_short + self.time_short_proj(t_short)
        if short_mask is not None:
            z_short = z_short * short_mask.unsqueeze(-1).to(z_short.dtype)

        for blk in self.short_blocks:
            z_short = blk(z_short)
        z_short = self.short_norm(z_short)
        z_short_last = self._last_valid(z_short, short_mask)

        # Long branch
        z_long = self.long_patcher(x_long)  # [B,Tl',D]
        if self.use_time and t_long is not None:
            z_long = z_long + self.time_long_patcher(t_long)
        for blk in self.long_blocks:
            z_long = blk(z_long)
        z_long = self.long_norm(z_long)
        z_long_last = z_long[:, -1, :]

        # Router + head
        z_fused, w = self.router(z_long_last, z_short_last)
        out = self.head(z_fused)
        if self.cfg.task == "classification" and return_probs:
            out = torch.softmax(out, dim=-1)

        self.last_tracker = {
            "router_w_mean": w.mean(dim=0).detach().cpu(),
            "router_w_std": w.std(dim=0).detach().cpu(),
            "long_tokens": int(z_long.shape[1]),
            "short_tokens": int(z_short.shape[1]),
            "base_bar_minutes": int(self.cfg.base_bar_minutes),
        }

        if not return_features:
            return out
        return out, {
            "z_long_last": z_long_last,
            "z_short_last": z_short_last,
            "z_fused": z_fused,
            "router_w": w,
        }
