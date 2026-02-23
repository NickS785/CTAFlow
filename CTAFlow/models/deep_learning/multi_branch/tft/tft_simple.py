"""
Simplified TFT-Aligned Mamba Model
===================================

Simplified variant without macro context encoder. Uses only:
- Static covariates (ticker, asset class)
- Temporal covariates (calendar features)
- Event schedule (days_until_event)

Conditioning via static + event context only.
"""

from __future__ import annotations

from typing import Dict, Optional

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
from ...market_context_models import (
    BranchVariableSelection,
    GatedResidualNetwork,
)

# Import N_EVENT_TYPES constant
try:
    from CTAFlow.data.datasets.tft import N_EVENT_TYPES
except ImportError:
    N_EVENT_TYPES = 15  # Fallback


class TFTAlignedMambaSimple(nn.Module):
    """Simplified TFT-aligned Mamba without macro context.

    Architecture:
    1. Static encoder: ticker/asset class → context vectors (c_s, c_e, c_c, c_h)
    2. Temporal encoder: calendar + event schedule → daily tokens
    3. Branch encoders: summary, profile, raster, sequential
    4. Mamba backbone: spatiotemporal fusion
    5. Branch selection: context = static + event (no macro)
    6. Prediction head

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
        Number of event types (for calendar encoding).
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
    n_attn_heads : int
        Number of attention heads.
    n_fusion_layers : int
        Number of Mamba fusion layers.
    d_state : int
        Mamba state dimension.
    d_conv : int
        Mamba convolution kernel size.
    expand : int
        Mamba expansion factor.
    task : str
        'classification' or 'regression'.
    num_classes : int
        Number of classes (for classification).
    dropout : float
        General dropout rate.
    grn_dropout : float, optional
        GRN/BVS-specific dropout (defaults to dropout).
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
        n_attn_heads: int = 4,
        n_fusion_layers: int = 2,
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

        _grn_drop = grn_dropout if grn_dropout is not None else dropout

        from ..mamba_model import SpatioTemporalMambaFusion

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

        # Temporal Known Encoder (calendar + events)
        self.known_encoder = TemporalKnownInputEncoder(
            d_model=d_model,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            dropout=dropout,
        )

        # Branch Encoders
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

        # Mamba Backbone
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

        # Branch Selection (context = static selection vector c_s)
        self.branch_selector = BranchVariableSelection(
            n_branches=3,  # daily, spatial, sequential
            d_branch=d_model,
            d_context=d_model,
            dropout=_grn_drop,
        )

        # Enrichment context from static + event
        self.enrichment_fuse = GatedResidualNetwork(
            d_model=d_model,
            d_input=d_model * 2,  # c_e + latest event token
            dropout=_grn_drop,
        )

        # Temporal attention over daily sequence
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_ln = nn.LayerNorm(d_model)

        # Head
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
            Windowed daily summary features.
        profile_days : Tensor (B, W, C_profile, profile_bins)
            Windowed market profiles.
        raster_recent : Tensor (B, C_raster, T, bins)
            Most recent day's rasterized VPIN.
        seq_recent : Tensor (B, seq_len, f_seq)
            Most recent day's sequential features.
        seq_lens_recent : Tensor (B,)
            Sequence lengths.
        ticker_id, asset_class_id, asset_subclass_id : Tensor (B,)
            Static identifiers.
        month, dow, doy_sin, doy_cos : Tensor (B, W)
            Calendar features.
        event_type_ids : Tensor (B, W, max_events), optional
            Event type IDs per day.
        days_until_event : Tensor (B, W, max_events), optional
            Days until each event.
        return_probs : bool
            If True, return softmax probabilities for classification.
        return_tracker : bool
            If True, return (logits, tracker_dict).

        Returns
        -------
        logits : Tensor (B, num_classes) or (B, 1)
            Model predictions.
        tracker : dict, optional
            Tracking dict if return_tracker=True.
        """
        b, w = summary_days.shape[0:2]
        bw = b * w

        # Static Context
        c_s, c_e, c_c, c_h = self.static_encoder(
            ticker_id, asset_class_id, asset_subclass_id,
        )

        z_static_daily = self.static_daily_proj(c_h)
        z_static_daily = z_static_daily.unsqueeze(1).expand(b, w, -1)

        # Known Future (calendar + events)
        z_known = self.known_encoder(
            month=month,
            dow=dow,
            doy_sin=doy_sin,
            doy_cos=doy_cos,
            event_type_ids=event_type_ids,
            days_until_event=days_until_event,
        )

        # Branch Encoding
        z_summary_seq = self.summary_proj(summary_days.reshape(bw, -1)).view(b, w, -1)
        z_profile_seq = self.profile_net(
            profile_days.reshape(bw, profile_days.size(2), -1)
        ).view(b, w, -1)

        # Mamba Spatiotemporal Scan
        temporal_seq = z_summary_seq + z_known + z_static_daily
        fused_daily_seq, fused_daily_token = self.spatiotemporal(
            z_profile_seq, temporal_seq,
        )

        # Recent modalities
        z_raster = self.raster_net(raster_recent)
        z_seq = self.seq_net(seq_recent, lengths=seq_lens_recent)
        z_prof_recent = z_profile_seq[:, -1, :]
        z_spatial = self.spatial_fuse(z_prof_recent, z_raster)

        # Branch Selection (context = static selection vector)
        branch_outputs = [fused_daily_token, z_spatial, z_seq]
        z_selected, branch_weights = self.branch_selector(
            branch_outputs=branch_outputs,
            context=c_s,
        )

        # Enrichment context (static enrichment + latest event)
        z_event_recent = z_known[:, -1, :]  # Latest day's event embedding
        enrichment_ctx = self.enrichment_fuse(
            torch.cat([c_e, z_event_recent], dim=-1)
        )

        # Temporal attention over daily sequence
        # Query: enrichment_ctx, K/V: fused_daily_seq
        query = enrichment_ctx.unsqueeze(1)  # (B, 1, d_model)
        attn_out, attn_weights = self.temporal_attn(
            query=query,
            key=fused_daily_seq,
            value=fused_daily_seq,
            need_weights=True,
        )
        z_temporal = self.temporal_ln(attn_out.squeeze(1) + enrichment_ctx)

        # Final fusion
        z_final = torch.cat([z_temporal, z_selected], dim=-1)  # (B, d_model*2)
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
            "temporal_attn": attn_weights.squeeze(1),  # (B, W)
            "spatiotemporal": dict(self.spatiotemporal.last_tracker),
        }

        if return_tracker:
            return logits, self._last_tracker
        return logits

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)
