from __future__ import annotations

from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders import MarketProfileResNet, RasterResNet, IntradayRNN, SpatialFuse, MetaModalityEncoder

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Mamba = None  # type: ignore


class DailyGraphLayer(nn.Module):
    """Lightweight day-to-day graph layer over sequence tokens."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model ** -0.5
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.scale, dim=-1)
        self.last_attn = attn.detach()
        agg = torch.matmul(attn, v)
        return self.out_proj(self.dropout(agg))


class MambaBlock(nn.Module):
    """Residual Mamba block with FFN and optional fallback."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_stats: Dict[str, float] = {}

        if Mamba is not None:
            self.core = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.is_fallback = False
        else:
            warnings.warn(
                "mamba_ssm is not installed. Falling back to a depthwise temporal conv block.",
                RuntimeWarning,
            )
            k = max(3, int(d_conv) * 2 - 1)
            self.core = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
            self.is_fallback = True

        hidden = d_model * max(2, int(expand))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x_in = x
        h = self.norm1(x)
        if self.is_fallback:
            h = self.core(h.transpose(1, 2)).transpose(1, 2)
        else:
            h = self.core(h)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        self.last_stats = {
            "in_norm": float(x_in.norm(dim=-1).mean().detach().cpu()),
            "out_norm": float(x.norm(dim=-1).mean().detach().cpu()),
        }
        return x


class SpatioTemporalMambaFusion(nn.Module):
    """
    Mamba-based spatio-temporal fusion over daily sequences.

    Inputs:
      - spatial_seq:  (B, W, D_spatial)
      - temporal_seq: (B, W, D_temporal)
    Outputs:
      - fused_seq: (B, W, D_model)
      - pooled:    (B, D_model)
    """

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_daily_graph: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, d_model)
        self.temporal_proj = nn.Linear(temporal_dim, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(max(1, n_layers))
            ]
        )
        self.daily_graph = DailyGraphLayer(d_model=d_model, dropout=dropout) if use_daily_graph else None
        self.norm = nn.LayerNorm(d_model)
        self.last_tracker: Dict[str, object] = {}

    def forward(self, spatial_seq: torch.Tensor, temporal_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_sp = self.spatial_proj(spatial_seq)
        z_tm = self.temporal_proj(temporal_seq)
        g = self.gate(torch.cat([z_sp, z_tm], dim=-1))
        x = g * z_sp + (1.0 - g) * z_tm

        for blk in self.blocks:
            x = blk(x)

        if self.daily_graph is not None:
            x = x + self.daily_graph(x)

        x = self.norm(x)
        pooled = x[:, -1, :]
        self.last_tracker = {
            "gate_mean": float(g.mean().detach().cpu()),
            "gate_std": float(g.std().detach().cpu()),
            "daily_graph_attn_mean": (
                float(self.daily_graph.last_attn.mean().detach().cpu())
                if (self.daily_graph is not None and self.daily_graph.last_attn is not None)
                else None
            ),
        }
        return x, pooled


class MambaFusionHead(nn.Module):
    """Main fusion head over a short token sequence."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        out_dim: int = 1,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(max(1, n_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )
        self.last_tracker: Dict[str, object] = {}

    def forward(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        # x: (B, T, input_dim)
        z = self.input_proj(x)
        token_scores = z.norm(dim=-1)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        if pool == "last":
            p = z[:, -1, :]
        else:
            p = z.mean(dim=1)
        self.last_tracker = {
            "token_score_mean": token_scores.mean(dim=0).detach().cpu(),
            "token_score_std": token_scores.std(dim=0).detach().cpu(),
            "pool_mode": pool,
        }
        return self.head(p)


class MultiModalMamba(nn.Module):
    """
    Mamba-based alternative to WSPR-style fusion.

    Current version fuses:
      1) Daily profile sequence (spatial)
      2) Daily context sequence from summary_days (temporal)
      3) Recent spatial fused token (profile_recent + raster_recent)
      4) Recent intraday sequential token
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        d_model: int = 128,
        task: str = "classification",
        num_classes: int = 3,
        n_fusion_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task = task

        self.profile_net = MarketProfileResNet(in_channels=f_profile, d_model=d_model)
        self.summary_proj = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.raster_net = RasterResNet(in_ch=f_raster, d_model=d_model)
        self.seq_net = IntradayRNN(input_dim=f_seq, d_model=d_model, num_layers=1)
        self.spatial_fuse = SpatialFuse(d_spatial=d_model, mode="gated")

        self.spatiotemporal = SpatioTemporalMambaFusion(
            spatial_dim=d_model,
            temporal_dim=d_model,
            d_model=d_model,
            n_layers=n_fusion_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_daily_graph=True,
            dropout=dropout,
        )
        out_dim = num_classes if task == "classification" else 1
        self.head = MambaFusionHead(
            input_dim=d_model,
            d_model=d_model,
            out_dim=out_dim,
            n_layers=n_fusion_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self._last_tracker: Dict[str, object] = {}

    def forward(
        self,
        summary_days: torch.Tensor,
        profile_days: torch.Tensor,
        raster_recent: torch.Tensor,
        seq_recent: torch.Tensor,
        seq_lens_recent: torch.Tensor,
        return_probs: bool = False,
        return_features: bool = False,
        return_tracker: bool = False,
    ):
        b, w = summary_days.shape[0:2]
        bw = b * w

        # Windowed daily tokens
        z_summary_seq = self.summary_proj(summary_days.reshape(bw, -1)).view(b, w, -1)
        z_profile_seq = self.profile_net(profile_days.reshape(bw, profile_days.size(2), -1)).view(b, w, -1)
        fused_daily_seq, fused_daily_token = self.spatiotemporal(z_profile_seq, z_summary_seq)

        # Recent tokens
        z_raster_recent = self.raster_net(raster_recent)
        z_seq_recent = self.seq_net(seq_recent, lengths=seq_lens_recent)
        z_profile_recent = z_profile_seq[:, -1, :]
        z_spatial_recent = self.spatial_fuse(z_profile_recent, z_raster_recent)

        # Main fusion token sequence for head
        token_seq = torch.stack(
            [fused_daily_token, z_spatial_recent, z_seq_recent],
            dim=1,
        )
        logits = self.head(token_seq, pool="mean")

        if self.task == "classification" and return_probs:
            logits = F.softmax(logits, dim=1)

        self._last_tracker = {
            "spatiotemporal": dict(self.spatiotemporal.last_tracker),
            "head": dict(self.head.last_tracker),
            "spatial_fuse": self.spatial_fuse.get_importance_stats(),
            "token_names": ["daily", "spatial_recent", "seq_recent"],
        }

        if return_features or return_tracker:
            payload = {
                "daily_seq": fused_daily_seq,
                "daily_token": fused_daily_token,
                "spatial_recent": z_spatial_recent,
                "seq_recent": z_seq_recent,
            }
            if return_tracker:
                payload["tracker"] = self.get_last_tracker()
            return logits, payload
        return logits

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)


class MultiAssetMamba(nn.Module):
    """
    Multi-asset variant of MultiModalMamba with explicit meta modality input.
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        d_model: int = 128,
        meta_hidden: int = 64,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        task: str = "classification",
        num_classes: int = 3,
        n_fusion_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task = task

        self.profile_net = MarketProfileResNet(in_channels=f_profile, d_model=d_model)
        self.summary_proj = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.raster_net = RasterResNet(in_ch=f_raster, d_model=d_model)
        self.seq_net = IntradayRNN(input_dim=f_seq, d_model=d_model, num_layers=1)
        self.spatial_fuse = SpatialFuse(d_spatial=d_model, mode="gated")
        self.meta_net = MetaModalityEncoder(
            d_model=d_model,
            ctx_dim=max(16, d_model // 2),
            time_dim=max(16, d_model // 2),
            meta_hidden=meta_hidden,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            dropout=dropout,
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.meta_day_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.spatiotemporal = SpatioTemporalMambaFusion(
            spatial_dim=d_model,
            temporal_dim=d_model,
            d_model=d_model,
            n_layers=n_fusion_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_daily_graph=True,
            dropout=dropout,
        )
        out_dim = num_classes if task == "classification" else 1
        self.head = MambaFusionHead(
            input_dim=d_model,
            d_model=d_model,
            out_dim=out_dim,
            n_layers=n_fusion_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self._last_tracker: Dict[str, object] = {}

    def forward(
        self,
        summary_days: torch.Tensor,
        profile_days: torch.Tensor,
        raster_recent: torch.Tensor,
        seq_recent: torch.Tensor,
        seq_lens_recent: torch.Tensor,
        meta: dict,
        return_probs: bool = False,
        return_features: bool = False,
        return_tracker: bool = False,
    ):
        b, w = summary_days.shape[0:2]
        bw = b * w

        z_summary_seq = self.summary_proj(summary_days.reshape(bw, -1)).view(b, w, -1)
        z_profile_seq = self.profile_net(profile_days.reshape(bw, profile_days.size(2), -1)).view(b, w, -1)
        z_meta_days, z_meta_window = self.meta_net(meta, W=w, device=summary_days.device)
        z_meta_days = self.meta_day_proj(z_meta_days)
        z_meta_window = self.meta_proj(z_meta_window)

        # Merge macro + meta before spatiotemporal scan.
        temporal_seq = z_summary_seq + z_meta_days
        fused_daily_seq, fused_daily_token = self.spatiotemporal(z_profile_seq, temporal_seq)

        z_raster_recent = self.raster_net(raster_recent)
        z_seq_recent = self.seq_net(seq_recent, lengths=seq_lens_recent)
        z_profile_recent = z_profile_seq[:, -1, :]
        z_spatial_recent = self.spatial_fuse(z_profile_recent, z_raster_recent)

        token_seq = torch.stack(
            [fused_daily_token, z_spatial_recent, z_seq_recent, z_meta_window],
            dim=1,
        )
        logits = self.head(token_seq, pool="mean")

        if self.task == "classification" and return_probs:
            logits = F.softmax(logits, dim=1)

        self._last_tracker = {
            "spatiotemporal": dict(self.spatiotemporal.last_tracker),
            "head": dict(self.head.last_tracker),
            "spatial_fuse": self.spatial_fuse.get_importance_stats(),
            "token_names": ["daily", "spatial_recent", "seq_recent", "meta"],
        }

        if return_features or return_tracker:
            payload = {
                "daily_seq": fused_daily_seq,
                "daily_token": fused_daily_token,
                "spatial_recent": z_spatial_recent,
                "seq_recent": z_seq_recent,
                "meta_window": z_meta_window,
            }
            if return_tracker:
                payload["tracker"] = self.get_last_tracker()
            return logits, payload
        return logits

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)


class MambaWeightTracker:
    """Collects per-step tracker payloads emitted by MultiModalMamba/MultiAssetMamba."""

    def __init__(self):
        self.history: list = []

    def reset(self) -> None:
        self.history.clear()

    def update(self, tracker: Optional[Dict[str, object]]) -> None:
        if tracker:
            self.history.append(tracker)

    def latest(self) -> Optional[Dict[str, object]]:
        if not self.history:
            return None
        return self.history[-1]
