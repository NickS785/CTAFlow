"""
MMTF: Multi-Modal Temporal Fusion Architecture
===============================================

Unified architecture for multi-modal temporal fusion supporting both
Transformer and Mamba backbones.

Architecture:
1. Static encoder: ticker/asset metadata → context vectors
2. Temporal encoder: calendar + event schedule → temporal embeddings
3. Multi-modal encoders: summary, profile, raster, sequential
4. Temporal fusion backbone: Transformer or Mamba
5. Branch selection with static context
6. Prediction head

Key Features:
- Modality-agnostic design (works with any subset of modalities)
- Plug-and-play backbone (Transformer or Mamba)
- Event-aware temporal encoding
- Static covariate conditioning
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...encoders import (
    IntradayRNN,
    MarketProfileResNet,
    RasterResNet,
    SpatialFuse,
)
from .tft_encoders import (
    StaticCovariateEncoder,
    TemporalKnownInputEncoder,
)
from ..market_context_models import (
    BranchVariableSelection,
    GatedResidualNetwork,
)

# Import N_EVENT_TYPES constant
try:
    from CTAFlow.data.datasets.tft import N_EVENT_TYPES
except ImportError:
    N_EVENT_TYPES = 15  # Fallback


class TemporalFusionBackbone(nn.Module):
    """Base class for temporal fusion backbones."""

    def forward(
        self,
        spatial_seq: torch.Tensor,
        temporal_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        spatial_seq : Tensor (B, W, d_model)
            Spatial sequence (e.g., profile embeddings).
        temporal_seq : Tensor (B, W, d_model)
            Temporal sequence (summary + calendar + static).

        Returns
        -------
        fused_seq : Tensor (B, W, d_model)
            Fused sequence over window.
        fused_token : Tensor (B, d_model)
            Aggregated representation (typically last token or pooled).
        """
        raise NotImplementedError


class TransformerFusionBackbone(TemporalFusionBackbone):
    """Transformer-based temporal fusion.

    Uses standard multi-head self-attention over the temporal window.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Projection for spatial-temporal fusion
        self.spatial_proj = nn.Linear(d_model, d_model)
        self.temporal_proj = nn.Linear(d_model, d_model)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.last_tracker = {}

    def forward(
        self,
        spatial_seq: torch.Tensor,
        temporal_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, w, d = temporal_seq.shape

        # Fuse spatial and temporal
        z_spatial = self.spatial_proj(spatial_seq)
        z_temporal = self.temporal_proj(temporal_seq)

        # Gated fusion
        gate = self.fusion_gate(torch.cat([z_spatial, z_temporal], dim=-1))
        fused_input = gate * z_spatial + (1 - gate) * z_temporal

        # Apply positional dropout
        fused_input = self.pos_dropout(fused_input)

        # Transformer encoding
        fused_seq = self.transformer(fused_input)

        # Aggregate (use last token)
        fused_token = fused_seq[:, -1, :]

        self.last_tracker = {
            "gate_mean": gate.mean().item(),
            "spatial_weight": gate.mean().item(),
            "temporal_weight": (1 - gate).mean().item(),
        }

        return fused_seq, fused_token


class MambaFusionBackbone(TemporalFusionBackbone):
    """Mamba-based temporal fusion.

    Uses state-space model for efficient long-range dependency modeling.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        from ..mamba_model import SpatioTemporalMambaFusion

        self.mamba = SpatioTemporalMambaFusion(
            spatial_dim=d_model,
            temporal_dim=d_model,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_daily_graph=True,
            dropout=dropout,
        )

        self.last_tracker = {}

    def forward(
        self,
        spatial_seq: torch.Tensor,
        temporal_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_seq, fused_token = self.mamba(spatial_seq, temporal_seq)

        self.last_tracker = dict(self.mamba.last_tracker)

        return fused_seq, fused_token


class MMTFCore(nn.Module):
    """Multi-Modal Temporal Fusion Architecture (Core).

    Unified multi-modal architecture supporting both Transformer and Mamba backbones.

    Parameters
    ----------
    f_sum : int
        Summary feature dimension.
    f_profile : int
        Profile channels.
    f_raster : int
        Rasterized channels.
    f_seq : int
        Sequential feature dimension.
    n_tickers : int
        Number of tickers in universe.
    n_asset_classes : int
        Number of asset classes.
    n_asset_subclasses : int
        Number of asset subclasses.
    n_event_types : int
        Number of event types.
    max_events_per_day : int
        Max events per day slot.
    d_model : int
        Model dimension.
    d_static_emb : int
        Static embedding dimension.
    d_calendar : int
        Calendar embedding dimension.
    d_event_emb : int
        Event embedding dimension.
    backbone : str
        'transformer' or 'mamba'.
    n_heads : int
        Number of attention heads (Transformer only).
    n_layers : int
        Number of backbone layers.
    d_ff : int
        Feed-forward dimension (Transformer only).
    d_state : int
        State dimension (Mamba only).
    d_conv : int
        Convolution kernel size (Mamba only).
    expand : int
        Expansion factor (Mamba only).
    task : str
        'classification' or 'regression'.
    num_classes : int
        Number of classes (for classification).
    dropout : float
        General dropout rate.
    grn_dropout : float, optional
        GRN/BVS-specific dropout.
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        n_event_types: int = N_EVENT_TYPES,
        max_events_per_day: int = 3,
        d_model: int = 128,
        d_static_emb: int = 64,
        d_calendar: int = 32,
        d_event_emb: int = 32,
        backbone: Literal["transformer", "mamba"] = "mamba",
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.backbone_type = backbone

        _grn_drop = grn_dropout if grn_dropout is not None else dropout

        # Static Encoder
        self.static_encoder = StaticCovariateEncoder(
            d_model=d_model,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            d_emb=d_static_emb,
            dropout=_grn_drop,
        )

        self.static_daily_proj = GatedResidualNetwork(
            d_model=d_model,
            dropout=_grn_drop,
        )

        # Temporal Known Encoder
        self.known_encoder = TemporalKnownInputEncoder(
            d_model=d_model,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            dropout=dropout,
        )

        # Multi-Modal Encoders
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

        # Temporal Fusion Backbone
        if backbone == "transformer":
            self.fusion_backbone = TransformerFusionBackbone(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
            )
        elif backbone == "mamba":
            self.fusion_backbone = MambaFusionBackbone(
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'transformer' or 'mamba'.")

        # Branch Selection
        self.branch_selector = BranchVariableSelection(
            n_branches=3,  # daily, spatial, sequential
            d_branch=d_model,
            d_context=d_model,
            dropout=_grn_drop,
        )

        # Enrichment context
        self.enrichment_fuse = GatedResidualNetwork(
            d_model=d_model,
            d_input=d_model * 2,  # c_e + latest event token
            dropout=_grn_drop,
        )

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_ln = nn.LayerNorm(d_model)

        # Prediction Head
        out_dim = num_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )

        self._last_tracker: Dict[str, object] = {}

    def forward(
        self,
        summary_days: torch.Tensor,
        profile_days: torch.Tensor,
        raster_recent: torch.Tensor,
        seq_recent: torch.Tensor,
        seq_lens_recent: torch.Tensor,
        ticker_id: torch.Tensor,
        asset_class_id: torch.Tensor,
        asset_subclass_id: torch.Tensor,
        month: torch.Tensor,
        dow: torch.Tensor,
        doy_sin: torch.Tensor,
        doy_cos: torch.Tensor,
        event_type_ids: Optional[torch.Tensor] = None,
        days_until_event: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_tracker: bool = False,
    ):
        """Forward pass.

        Parameters
        ----------
        summary_days : Tensor (B, W, f_sum)
        profile_days : Tensor (B, W, C_profile, profile_bins)
        raster_recent : Tensor (B, C_raster, T, bins)
        seq_recent : Tensor (B, seq_len, f_seq)
        seq_lens_recent : Tensor (B,)
        ticker_id, asset_class_id, asset_subclass_id : Tensor (B,)
        month, dow, doy_sin, doy_cos : Tensor (B, W)
        event_type_ids : Tensor (B, W, max_events), optional
        days_until_event : Tensor (B, W, max_events), optional
        return_probs : bool
        return_tracker : bool

        Returns
        -------
        logits : Tensor (B, num_classes) or (B, 1)
        tracker : dict, optional
        """
        b, w = summary_days.shape[0:2]
        bw = b * w

        # Static Context
        c_s, c_e, c_c, c_h = self.static_encoder(
            ticker_id, asset_class_id, asset_subclass_id,
        )

        z_static_daily = self.static_daily_proj(c_h)
        z_static_daily = z_static_daily.unsqueeze(1).expand(b, w, -1)

        # Known Future
        z_known = self.known_encoder(
            month=month,
            dow=dow,
            doy_sin=doy_sin,
            doy_cos=doy_cos,
            event_type_ids=event_type_ids,
            days_until_event=days_until_event,
        )

        # Multi-Modal Encoding
        z_summary_seq = self.summary_proj(summary_days.reshape(bw, -1)).view(b, w, -1)
        z_profile_seq = self.profile_net(
            profile_days.reshape(bw, profile_days.size(2), -1)
        ).view(b, w, -1)

        # Temporal Fusion
        temporal_seq = z_summary_seq + z_known + z_static_daily
        fused_daily_seq, fused_daily_token = self.fusion_backbone(
            z_profile_seq, temporal_seq,
        )

        # Recent Modalities
        z_raster = self.raster_net(raster_recent)
        z_seq = self.seq_net(seq_recent, lengths=seq_lens_recent)
        z_prof_recent = z_profile_seq[:, -1, :]
        z_spatial = self.spatial_fuse(z_prof_recent, z_raster)

        # Branch Selection
        branch_outputs = [fused_daily_token, z_spatial, z_seq]
        z_selected, branch_weights = self.branch_selector(
            branch_outputs=branch_outputs,
            context=c_s,
        )

        # Enrichment
        z_event_recent = z_known[:, -1, :]
        enrichment_ctx = self.enrichment_fuse(
            torch.cat([c_e, z_event_recent], dim=-1)
        )

        # Temporal Attention
        query = enrichment_ctx.unsqueeze(1)
        attn_out, attn_weights = self.temporal_attn(
            query=query,
            key=fused_daily_seq,
            value=fused_daily_seq,
            need_weights=True,
        )
        z_temporal = self.temporal_ln(attn_out.squeeze(1) + enrichment_ctx)

        # Final Fusion & Prediction
        z_final = torch.cat([z_temporal, z_selected], dim=-1)
        logits = self.head(z_final)

        if self.task == "classification" and return_probs:
            logits = F.softmax(logits, dim=1)

        self._last_tracker = {
            "branch_weights": {
                name: branch_weights[:, i].mean().item()
                for i, name in enumerate(["daily", "spatial", "sequential"])
            },
            "static_var_weights": {
                name: self.static_encoder.last_var_weights[:, i].mean().item()
                for i, name in enumerate(["ticker", "asset_class", "asset_subclass"])
            } if self.static_encoder.last_var_weights is not None else None,
            "temporal_attn": attn_weights.squeeze(1),
            "backbone": dict(self.fusion_backbone.last_tracker),
        }

        if return_tracker:
            return logits, self._last_tracker
        return logits

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)
