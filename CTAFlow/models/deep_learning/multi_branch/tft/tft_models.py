"""
TFT-Aligned Models
==================

Full TFT-aligned model architectures:

  TFTAlignedWSPR  : WSPR variant with windowed LSTMs
  TFTAlignedMamba : Mamba variant with SpatioTemporalMambaFusion backbone

Both use the TFT input taxonomy:
  - Static covariates condition branch selection + temporal processing
  - Known future inputs (calendar + event schedule) enrich daily tokens
  - Past-observed inputs feed branch encoders + macro encoder
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..market_context_models import (
    GatedResidualNetwork,
    BranchVariableSelection,
    MacroEnrichedTemporalAttention,
    GatedCrossAttentionFusion,
)
from ..macro_event_encoding import N_EVENT_TYPES
from ...encoders import (
    MarketProfileResNet,
    RasterResNet,
    IntradayRNN,
    SpatialFuse,
)
from .tft_encoders import (
    StaticCovariateEncoder,
    TemporalKnownInputEncoder,
    MacroPastObservedEncoder,
)


# ============================================================================
# TFTAlignedWSPR
# ============================================================================

class TFTAlignedWSPR(nn.Module):
    """Fully TFT-aligned WSPR model with windowed LSTMs.

    5 branches (summary, profile, raster, sequential, spatial):
      - Static context c_s conditions BranchVariableSelection
      - Static context c_e + macro context enrich temporal attention
      - Static context c_c, c_h initialize summary/profile LSTMs
      - Known future inputs (calendar + events) add to daily tokens
      - Macro past-observed encoder handles VIX/yields + event outcomes
    """
    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        f_macro: int,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        n_event_types: int = N_EVENT_TYPES,
        n_outcome_features: int = 4,
        max_events_per_day: int = 3,
        d_model: int = 128,
        d_static_emb: int = 64,
        d_calendar: int = 32,
        d_event_emb: int = 32,
        sum_lstm_hidden: int = 64,
        prof_lstm_hidden: int = 128,
        n_attn_heads: int = 4,
        use_event_decay: bool = True,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.sum_lstm_hidden = sum_lstm_hidden
        self.prof_lstm_hidden = prof_lstm_hidden

        # Static Covariate Encoder
        self.static_encoder = StaticCovariateEncoder(
            d_model=d_model,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            d_emb=d_static_emb,
            dropout=dropout,
        )

        # Project c_c and c_h to match LSTM hidden sizes
        self.cc_to_sum_lstm = nn.Linear(d_model, sum_lstm_hidden)
        self.ch_to_sum_lstm = nn.Linear(d_model, sum_lstm_hidden)
        self.cc_to_prof_lstm = nn.Linear(d_model, prof_lstm_hidden)
        self.ch_to_prof_lstm = nn.Linear(d_model, prof_lstm_hidden)

        # Known Future Input Encoder
        self.known_encoder = TemporalKnownInputEncoder(
            d_model=d_model,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            dropout=dropout,
        )

        # Branch Encoders
        self.summary_net = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.profile_net = MarketProfileResNet(in_channels=f_profile, d_model=d_model)
        self.raster_net = RasterResNet(in_ch=f_raster, d_model=d_model)
        self.seq_net = IntradayRNN(input_dim=f_seq, d_model=d_model, num_layers=1)
        self.spatial_fuse = SpatialFuse(d_spatial=d_model, mode="gated")

        # Windowed LSTMs (initialized by static context)
        self.summary_lstm = nn.LSTM(
            input_size=d_model, hidden_size=sum_lstm_hidden,
            num_layers=1, batch_first=True,
        )
        self.profile_lstm = nn.LSTM(
            input_size=d_model, hidden_size=prof_lstm_hidden,
            num_layers=1, batch_first=True,
        )

        # Macro Past-Observed Encoder
        self.macro_encoder = MacroPastObservedEncoder(
            f_macro=f_macro,
            d_model=d_model,
            n_event_types=n_event_types,
            n_outcome_features=n_outcome_features,
            max_events_per_day=max_events_per_day,
            n_heads=n_attn_heads,
            n_layers=2,
            dropout=dropout,
            use_event_decay=use_event_decay,
        )

        # Fusion Components
        self.sum_proj = nn.Linear(sum_lstm_hidden, d_model)
        self.prof_proj = nn.Linear(prof_lstm_hidden, d_model)

        self.branch_selector = BranchVariableSelection(
            n_branches=5,
            d_branch=d_model,
            d_context=d_model,
            dropout=dropout,
        )

        self.temporal_attn = MacroEnrichedTemporalAttention(
            d_model=d_model,
            d_context=d_model,
            n_heads=n_attn_heads,
            dropout=dropout,
        )

        self.enrichment_ctx_fuse = GatedResidualNetwork(
            d_model=d_model,
            d_input=d_model * 2,
            dropout=dropout,
        )

        self.cross_attn_fusion = GatedCrossAttentionFusion(
            d_model=d_model,
            n_heads=n_attn_heads,
            dropout=dropout,
        )

        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes if task == "classification" else 1),
        )

        self._last_tracker: Dict[str, object] = {}
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        summary_days: torch.Tensor,
        profile_days: torch.Tensor,
        raster_recent: torch.Tensor,
        seq_recent: torch.Tensor,
        seq_lens_recent: torch.Tensor,
        macro_days: torch.Tensor,
        ticker_id: torch.Tensor,
        asset_class_id: torch.Tensor,
        asset_subclass_id: torch.Tensor,
        month: torch.Tensor,
        dow: torch.Tensor,
        doy_sin: torch.Tensor,
        doy_cos: torch.Tensor,
        event_type_ids: Optional[torch.Tensor] = None,
        event_outcomes: Optional[torch.Tensor] = None,
        days_until_event: Optional[torch.Tensor] = None,
        event_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_tracker: bool = False,
    ):
        B, W = summary_days.shape[0:2]
        BW = B * W

        # Phase 0: Static Context
        c_s, c_e, c_c, c_h = self.static_encoder(
            ticker_id, asset_class_id, asset_subclass_id,
        )

        h0_sum = self.ch_to_sum_lstm(c_h).unsqueeze(0)
        c0_sum = self.cc_to_sum_lstm(c_c).unsqueeze(0)
        h0_prof = self.ch_to_prof_lstm(c_h).unsqueeze(0)
        c0_prof = self.cc_to_prof_lstm(c_c).unsqueeze(0)

        # Phase 1: Known Future Inputs
        z_known = self.known_encoder(
            month=month, dow=dow, doy_sin=doy_sin, doy_cos=doy_cos,
            event_type_ids=event_type_ids,
            days_until_event=days_until_event,
        )

        # Phase 2: Encode Past-Observed Branches
        flat_sum = summary_days.reshape(BW, -1)
        z_sum_all = self.summary_net(flat_sum).view(B, W, -1)
        _, (h_sum, _) = self.summary_lstm(z_sum_all, (h0_sum, c0_sum))
        z_sum = self.sum_proj(h_sum[-1])

        flat_prof = profile_days.reshape(BW, profile_days.size(2), -1)
        z_prof_all = self.profile_net(flat_prof).view(B, W, -1)
        _, (h_prof, _) = self.profile_lstm(z_prof_all, (h0_prof, c0_prof))
        z_prof = self.prof_proj(h_prof[-1])

        z_raster = self.raster_net(raster_recent)
        z_seq = self.seq_net(seq_recent, lengths=seq_lens_recent)

        z_prof_recent = z_prof_all[:, -1, :]
        z_spatial = self.spatial_fuse(z_prof_recent, z_raster)

        z_macro_days, z_macro_context = self.macro_encoder(
            macro_days=macro_days,
            event_type_ids=event_type_ids,
            event_outcomes=event_outcomes,
            event_mask=event_mask,
        )

        # Phase 3: Context-Conditioned Branch Selection
        branch_outputs = [z_sum, z_prof, z_raster, z_seq, z_spatial]
        z_selected, branch_weights = self.branch_selector(
            branch_outputs=branch_outputs,
            context=c_s,
        )

        # Phase 4: Macro-Enriched Temporal Attention
        daily_tokens = z_sum_all + z_prof_all + z_known + z_macro_days

        enrichment_ctx = self.enrichment_ctx_fuse(
            torch.cat([c_e, z_macro_context], dim=-1)
        )

        attended_seq, attn_weights = self.temporal_attn(
            daily_tokens=daily_tokens,
            macro_context=enrichment_ctx,
            causal_mask=True,
        )
        z_temporal = attended_seq[:, -1, :]

        # Phase 5: Gated Cross-Attention Fusion
        kv_tokens = torch.stack([z_selected, z_temporal], dim=1)
        z_fused = self.cross_attn_fusion(
            macro_query=z_macro_context,
            branch_kv=kv_tokens,
        )

        # Phase 6: Prediction
        z_final = torch.cat([z_fused, z_selected], dim=-1)
        logits = self.head(z_final)

        if self.task == "classification" and return_probs:
            logits = F.softmax(logits, dim=1)

        self._last_tracker = {
            "branch_weights": {
                name: branch_weights[:, i].mean().item()
                for i, name in enumerate([
                    "summary", "profile", "raster", "sequential", "spatial"
                ])
            },
            "static_var_weights": {
                name: self.static_encoder.last_var_weights[:, i].mean().item()
                for i, name in enumerate(["ticker", "asset_class", "asset_subclass"])
            } if self.static_encoder.last_var_weights is not None else None,
            "temporal_attn": attn_weights,
            "cross_attn_gate": (
                self.cross_attn_fusion.last_gate_value.mean().item()
                if self.cross_attn_fusion.last_gate_value is not None
                else None
            ),
            "spatial_fuse": self.spatial_fuse.get_importance_stats(),
        }

        if return_tracker:
            return logits, self._last_tracker
        return logits

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)

    def get_regime_distance(self, attn_weights: torch.Tensor) -> torch.Tensor:
        avg_pattern = attn_weights.mean(dim=0, keepdim=True)
        eps = 1e-8
        bc_coeff = torch.sqrt(attn_weights * avg_pattern + eps).sum(dim=-1)
        bc_coeff = bc_coeff.clamp(max=1.0)
        dist = torch.sqrt(1.0 - bc_coeff + eps)
        return dist.mean(dim=-1)


# ============================================================================
# TFTAlignedMamba
# ============================================================================

class TFTAlignedMamba(nn.Module):
    """TFT-aligned variant using Mamba backbone instead of windowed LSTMs.

    Same TFT input taxonomy as TFTAlignedWSPR. Static context is
    additively merged into daily tokens before the Mamba scan (no LSTMs).
    """
    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        f_macro: int,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        n_event_types: int = N_EVENT_TYPES,
        n_outcome_features: int = 4,
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
        use_event_decay: bool = True,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model

        # GRN/BVS dropout defaults to main dropout if not specified
        _grn_drop = grn_dropout if grn_dropout is not None else dropout

        from ..mamba_model import SpatioTemporalMambaFusion

        # Static Encoder (uses GRNs internally)
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

        # Known Future Encoder
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
            nn.Linear(f_sum, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.raster_net = RasterResNet(in_ch=f_raster, d_model=d_model)
        self.seq_net = IntradayRNN(input_dim=f_seq, d_model=d_model, num_layers=1)
        self.spatial_fuse = SpatialFuse(d_spatial=d_model, mode="gated")

        # Macro Encoder
        self.macro_encoder = MacroPastObservedEncoder(
            f_macro=f_macro,
            d_model=d_model,
            n_event_types=n_event_types,
            n_outcome_features=n_outcome_features,
            max_events_per_day=max_events_per_day,
            n_heads=n_attn_heads,
            n_layers=2,
            dropout=dropout,
            use_event_decay=use_event_decay,
        )

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

        # Fusion (GRN/BVS components use grn_dropout)
        self.branch_selector = BranchVariableSelection(
            n_branches=4,
            d_branch=d_model,
            d_context=d_model,
            dropout=_grn_drop,
        )

        self.enrichment_ctx_fuse = GatedResidualNetwork(
            d_model=d_model,
            d_input=d_model * 2,
            dropout=_grn_drop,
        )

        self.temporal_attn = MacroEnrichedTemporalAttention(
            d_model=d_model,
            d_context=d_model,
            n_heads=n_attn_heads,
            dropout=dropout,
        )

        self.cross_attn_fusion = GatedCrossAttentionFusion(
            d_model=d_model,
            n_heads=n_attn_heads,
            dropout=dropout,
        )

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
        macro_days: torch.Tensor,
        ticker_id: torch.Tensor,
        asset_class_id: torch.Tensor,
        asset_subclass_id: torch.Tensor,
        month: torch.Tensor,
        dow: torch.Tensor,
        doy_sin: torch.Tensor,
        doy_cos: torch.Tensor,
        event_type_ids: Optional[torch.Tensor] = None,
        event_outcomes: Optional[torch.Tensor] = None,
        days_until_event: Optional[torch.Tensor] = None,
        event_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_tracker: bool = False,
    ):
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
            month=month, dow=dow, doy_sin=doy_sin, doy_cos=doy_cos,
            event_type_ids=event_type_ids,
            days_until_event=days_until_event,
        )

        # Branch Encoding
        z_summary_seq = self.summary_proj(summary_days.reshape(bw, -1)).view(b, w, -1)
        z_profile_seq = self.profile_net(
            profile_days.reshape(bw, profile_days.size(2), -1)
        ).view(b, w, -1)

        z_macro_days, z_macro_context = self.macro_encoder(
            macro_days=macro_days,
            event_type_ids=event_type_ids,
            event_outcomes=event_outcomes,
            event_mask=event_mask,
        )

        # Mamba Spatiotemporal Scan
        temporal_seq = z_summary_seq + z_known + z_macro_days + z_static_daily
        fused_daily_seq, fused_daily_token = self.spatiotemporal(
            z_profile_seq, temporal_seq,
        )

        z_raster = self.raster_net(raster_recent)
        z_seq = self.seq_net(seq_recent, lengths=seq_lens_recent)
        z_prof_recent = z_profile_seq[:, -1, :]
        z_spatial = self.spatial_fuse(z_prof_recent, z_raster)

        # Context-Conditioned Fusion
        branch_outputs = [fused_daily_token, z_spatial, z_seq]
        z_selected, branch_weights = self.branch_selector(
            branch_outputs=branch_outputs + [z_macro_context],
            context=c_s,
        )

        enrichment_ctx = self.enrichment_ctx_fuse(
            torch.cat([c_e, z_macro_context], dim=-1)
        )
        attended_seq, attn_weights = self.temporal_attn(
            daily_tokens=fused_daily_seq,
            macro_context=enrichment_ctx,
            causal_mask=True,
        )
        z_temporal = attended_seq[:, -1, :]

        kv_tokens = torch.stack([z_selected, z_temporal], dim=1)
        z_fused = self.cross_attn_fusion(
            macro_query=z_macro_context,
            branch_kv=kv_tokens,
        )

        # Predict
        z_final = torch.cat([z_fused, z_selected], dim=-1)
        logits = self.head(z_final)

        if self.task == "classification" and return_probs:
            logits = F.softmax(logits, dim=1)

        self._last_tracker = {
            "branch_weights": {
                name: branch_weights[:, i].mean().item()
                for i, name in enumerate(["daily", "spatial", "sequential", "macro_ctx"])
            },
            "static_var_weights": {
                name: self.static_encoder.last_var_weights[:, i].mean().item()
                for i, name in enumerate(["ticker", "asset_class", "asset_subclass"])
            } if self.static_encoder.last_var_weights is not None else None,
            "temporal_attn": attn_weights,
            "spatiotemporal": dict(self.spatiotemporal.last_tracker),
            "cross_attn_gate": (
                self.cross_attn_fusion.last_gate_value.mean().item()
                if self.cross_attn_fusion.last_gate_value is not None
                else None
            ),
        }

        if return_tracker:
            return logits, self._last_tracker
        return logits

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)
