"""
Anti-Collapse Branch Variable Selection + Updated MMTF
======================================================

Fixes two compounding problems in the original MMTFCore:

Problem 1 — BVS softmax saturation:
  With 3 branches and raw softmax, logits like [0.1, 0.1, 5.0] give
  weights [0.007, 0.007, 0.986] → dead branches. No mechanism prevents
  this, and high grn_dropout accelerates it via "rich get richer" dynamics.

  Fix: Temperature scaling + entropy regularization + minimum weight floor
  + LayerNorm on branch outputs to prevent scale mismatches.

Problem 2 — Reduced branch set:
  MMTFCore collapsed 5 branches (TFTAlignedWSPR) to 3 by fusing summary
  into the daily path. This means:
  - Summary's windowed temporal patterns get diluted through the backbone
  - Spatial only sees last day's profile + raster (no windowed evolution)
  - Sequential has the cleanest signal path → dominates

  Fix: Restore dedicated windowed summary and spatial branches alongside
  the backbone-fused daily representation, giving 5 branches with distinct
  signal paths.

Drop-in replacement: MMTFv2Mamba / MMTFv2Transformer
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTAFlow.models.deep_learning.multi_branch.market_context_models import (
    GatedResidualNetwork,
    GLU,
)
from CTAFlow.models.deep_learning.multi_branch.tft.tft_encoders import (
    StaticCovariateEncoder,
    TemporalKnownInputEncoder,
)
from CTAFlow.models.deep_learning.multi_branch.tft.mmtf_core import (
    TransformerFusionBackbone,
    MambaFusionBackbone,
)
from CTAFlow.models.deep_learning.encoders import (
    IntradayRNN,
    MarketProfileResNet,
    RasterResNet,
    SpatialFuse,
)

try:
    from CTAFlow.data.datasets.tft import N_EVENT_TYPES
except ImportError:
    N_EVENT_TYPES = 15


# ============================================================================
# Anti-Collapse Branch Variable Selection
# ============================================================================

class BranchVariableSelectionV2(nn.Module):
    """BVS with anti-collapse mechanisms.

    Improvements over original BVS:

    1. **Temperature scaling**: Controls softmax sharpness.
       - temperature=1.0 → standard softmax (can saturate with few branches)
       - temperature=2.0 → softer distribution (recommended for 3 branches)
       - temperature=0.5 → sharper (use when you WANT winner-take-all)
       Optuna range: [0.5, 3.0]

    2. **Entropy regularization**: Auxiliary loss that penalizes collapsed
       distributions. Added to total loss during training.
       - entropy_weight=0.0 → no regularization (original behavior)
       - entropy_weight=0.1 → gentle push toward uniform
       - entropy_weight=0.5 → strong anti-collapse
       Optuna range: [0.01, 0.3]

    3. **Minimum weight floor**: Hard lower bound on branch weights.
       After softmax, any weight below `min_weight` is clamped up, and the
       distribution is renormalized. Guarantees every branch gets gradient.
       - min_weight=0.0 → no floor (original behavior)
       - min_weight=0.05 → each branch gets at least 5%
       Optuna range: [0.0, 0.1]

    4. **Pre-selection LayerNorm**: Normalizes branch outputs before
       feeding to the selection GRN. Prevents scale mismatches where
       one branch's raw magnitude dominates selection.

    Parameters
    ----------
    n_branches : int
        Number of branches to select over.
    d_branch : int
        Dimension of each branch output.
    d_context : int
        Dimension of the context vector.
    dropout : float
        Dropout rate for GRNs.
    temperature : float
        Softmax temperature. Higher = softer distribution.
    entropy_weight : float
        Weight for entropy regularization loss.
    min_weight : float
        Minimum per-branch weight (0.0 to disable).
    use_pre_norm : bool
        Whether to LayerNorm branch outputs before selection.
    """

    def __init__(
        self,
        n_branches: int,
        d_branch: int,
        d_context: int,
        dropout: float = 0.1,
        temperature: float = 1.5,
        entropy_weight: float = 0.1,
        min_weight: float = 0.05,
        use_pre_norm: bool = True,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.d_branch = d_branch
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.min_weight = min_weight

        # Optional pre-selection normalization
        if use_pre_norm:
            self.pre_norms = nn.ModuleList([
                nn.LayerNorm(d_branch) for _ in range(n_branches)
            ])
        else:
            self.pre_norms = None

        # Per-branch GRN for non-linear processing
        self.branch_grns = nn.ModuleList([
            GatedResidualNetwork(d_model=d_branch, dropout=dropout)
            for _ in range(n_branches)
        ])

        # Selection network: flattened branches + context → logits
        self.selection_grn = GatedResidualNetwork(
            d_model=n_branches,
            d_input=n_branches * d_branch,
            d_context=d_context,
            dropout=dropout,
        )

        # Interpretability storage
        self.last_weights: Optional[torch.Tensor] = None
        self.last_entropy: Optional[torch.Tensor] = None
        self.last_entropy_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        branch_outputs: List[torch.Tensor],
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        branch_outputs : list of Tensor (B, d_branch) each
        context : Tensor (B, d_context)

        Returns
        -------
        z_selected : Tensor (B, d_branch)
        weights : Tensor (B, n_branches)
        """
        assert len(branch_outputs) == self.n_branches

        # 1. Optional pre-normalization (prevents scale mismatch)
        if self.pre_norms is not None:
            normed = [norm(z) for norm, z in zip(self.pre_norms, branch_outputs)]
        else:
            normed = branch_outputs

        # 2. Process each branch through its own GRN
        processed = [grn(z) for grn, z in zip(self.branch_grns, normed)]

        # 3. Compute selection weights with temperature scaling
        flattened = torch.cat(normed, dim=-1)  # (B, n_branches * d_branch)
        weight_logits = self.selection_grn(flattened, context=context)  # (B, n_branches)

        # Temperature-scaled softmax
        weights = F.softmax(weight_logits / self.temperature, dim=-1)  # (B, n_branches)

        # 4. Apply minimum weight floor
        if self.min_weight > 0:
            weights = self._apply_weight_floor(weights)

        # 5. Compute entropy for regularization
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # (B,)
        max_entropy = torch.log(torch.tensor(
            float(self.n_branches), device=weights.device,
        ))
        # Normalized entropy: 0 = collapsed, 1 = uniform
        norm_entropy = entropy / max_entropy

        # Entropy loss: penalize LOW entropy (collapsed distributions)
        # Loss = weight * (1 - normalized_entropy) → 0 when uniform
        entropy_loss = self.entropy_weight * (1.0 - norm_entropy).mean()

        self.last_weights = weights.detach()
        self.last_entropy = norm_entropy.mean().detach()
        self.last_entropy_loss = entropy_loss.detach()

        # 6. Weighted combination
        stacked = torch.stack(processed, dim=1)  # (B, n_branches, d_branch)
        z_selected = torch.einsum("bn,bnd->bd", weights, stacked)  # (B, d_branch)

        return z_selected, weights

    def _apply_weight_floor(self, weights: torch.Tensor) -> torch.Tensor:
        """Clamp minimum and renormalize."""
        floored = weights.clamp(min=self.min_weight)
        return floored / floored.sum(dim=-1, keepdim=True)

    def get_entropy_loss(self) -> torch.Tensor:
        """Get the entropy regularization loss for adding to total loss.

        Call this AFTER forward() and add to your training loss:
            total_loss = task_loss + model.branch_selector.get_entropy_loss()
        """
        if self.last_entropy_loss is not None:
            return self.last_entropy_loss
        return torch.tensor(0.0)


# ============================================================================
# MMTFv2 Core: 5 Branches + Anti-Collapse BVS
# ============================================================================

class MMTFv2Core(nn.Module):
    """Multi-Modal Temporal Fusion v2 with anti-collapse branch selection.

    Restores dedicated windowed branches that were lost in the original
    MMTFCore simplification:

    Branch 1: daily_fused  — backbone output (Mamba/Transformer over the
              full window of summary + profile + known + static)
    Branch 2: summary_wind — dedicated windowed LSTM over summary features
              (captures trending summary stats independently)
    Branch 3: spatial      — fused profile + raster (most recent day)
    Branch 4: sequential   — IntradayRNN over intraday sequence
    Branch 5: spatial_wind — windowed attention over daily profile sequence
              (captures spatial evolution across the window)

    The BVS uses BranchVariableSelectionV2 with temperature, entropy
    regularization, and minimum weight floor to prevent collapse.

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
        Number of attention heads.
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
    sum_lstm_hidden : int
        Hidden dim for windowed summary LSTM.
    task : str
        'classification' or 'regression'.
    num_classes : int
        Number of output classes.
    dropout : float
        General dropout rate.
    grn_dropout : float, optional
        GRN/BVS-specific dropout.
    bvs_temperature : float
        BVS softmax temperature. Higher = softer selection.
    bvs_entropy_weight : float
        BVS entropy regularization weight.
    bvs_min_weight : float
        BVS minimum per-branch weight floor.
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
        sum_lstm_hidden: int = 64,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
        # --- Anti-collapse BVS params ---
        bvs_temperature: float = 1.5,
        bvs_entropy_weight: float = 0.1,
        bvs_min_weight: float = 0.05,
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.backbone_type = backbone
        self.sum_lstm_hidden = sum_lstm_hidden

        _grn_drop = grn_dropout if grn_dropout is not None else dropout

        # ================================================================
        # STATIC ENCODER
        # ================================================================
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

        # ================================================================
        # TEMPORAL KNOWN ENCODER
        # ================================================================
        self.known_encoder = TemporalKnownInputEncoder(
            d_model=d_model,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            dropout=dropout,
        )

        # ================================================================
        # MULTI-MODAL ENCODERS
        # ================================================================
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

        # ================================================================
        # WINDOWED SUMMARY LSTM (Branch 2 — dedicated temporal summary)
        # ================================================================
        # Static context c_c/c_h initialize the LSTM, so the windowed
        # processing is identity-conditioned from the start.
        self.summary_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=sum_lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.cc_to_sum_lstm = nn.Linear(d_model, sum_lstm_hidden)
        self.ch_to_sum_lstm = nn.Linear(d_model, sum_lstm_hidden)
        self.sum_lstm_proj = nn.Linear(sum_lstm_hidden, d_model)

        # ================================================================
        # WINDOWED SPATIAL ATTENTION (Branch 5 — spatial evolution)
        # ================================================================
        # Attends over the daily profile sequence to capture how spatial
        # structure evolves across the lookback window.
        self.spatial_wind_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_wind_norm = nn.LayerNorm(d_model)
        self.spatial_wind_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ================================================================
        # TEMPORAL FUSION BACKBONE
        # ================================================================
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
            raise ValueError(f"Unknown backbone: {backbone}")

        # ================================================================
        # ANTI-COLLAPSE BRANCH SELECTION (5 branches)
        # ================================================================
        self.branch_selector = BranchVariableSelectionV2(
            n_branches=5,
            d_branch=d_model,
            d_context=d_model,
            dropout=_grn_drop,
            temperature=bvs_temperature,
            entropy_weight=bvs_entropy_weight,
            min_weight=bvs_min_weight,
        )

        # ================================================================
        # ENRICHMENT + TEMPORAL ATTENTION
        # ================================================================
        self.enrichment_fuse = GatedResidualNetwork(
            d_model=d_model,
            d_input=d_model * 2,
            dropout=_grn_drop,
        )

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_ln = nn.LayerNorm(d_model)

        # ================================================================
        # PREDICTION HEAD
        # ================================================================
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
        summary_days: torch.Tensor,       # (B, W, f_sum)
        profile_days: torch.Tensor,       # (B, W, C_profile, profile_bins)
        raster_recent: torch.Tensor,      # (B, C_raster, T, bins)
        seq_recent: torch.Tensor,         # (B, seq_len, f_seq)
        seq_lens_recent: torch.Tensor,    # (B,)
        ticker_id: torch.Tensor,          # (B,)
        asset_class_id: torch.Tensor,     # (B,)
        asset_subclass_id: torch.Tensor,  # (B,)
        month: torch.Tensor,              # (B, W)
        dow: torch.Tensor,                # (B, W)
        doy_sin: torch.Tensor,            # (B, W)
        doy_cos: torch.Tensor,            # (B, W)
        event_type_ids: Optional[torch.Tensor] = None,
        days_until_event: Optional[torch.Tensor] = None,
        macro_days: Optional[torch.Tensor] = None,
        event_outcomes: Optional[torch.Tensor] = None,
        event_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_tracker: bool = False,
    ):
        """Forward pass with 5 branches and anti-collapse BVS.

        Returns
        -------
        logits : Tensor (B, num_classes) or (B, 1)
        tracker : dict, optional
        """
        b, w = summary_days.shape[0:2]
        bw = b * w

        # ============================================================
        # PHASE 0: STATIC CONTEXT
        # ============================================================
        c_s, c_e, c_c, c_h = self.static_encoder(
            ticker_id, asset_class_id, asset_subclass_id,
        )

        z_static_daily = self.static_daily_proj(c_h)
        z_static_daily = z_static_daily.unsqueeze(1).expand(b, w, -1)

        # ============================================================
        # PHASE 1: KNOWN FUTURE
        # ============================================================
        z_known = self.known_encoder(
            month=month, dow=dow, doy_sin=doy_sin, doy_cos=doy_cos,
            event_type_ids=event_type_ids,
            days_until_event=days_until_event,
        )

        # ============================================================
        # PHASE 2: MULTI-MODAL ENCODING
        # ============================================================
        z_summary_seq = self.summary_proj(
            summary_days.reshape(bw, -1)
        ).view(b, w, -1)

        z_profile_seq = self.profile_net(
            profile_days.reshape(bw, profile_days.size(2), -1)
        ).view(b, w, -1)

        # ============================================================
        # PHASE 3: TEMPORAL FUSION BACKBONE
        # ============================================================
        # Branch 1: daily_fused — backbone over full multimodal sequence
        temporal_seq = z_summary_seq + z_known + z_static_daily
        fused_daily_seq, fused_daily_token = self.fusion_backbone(
            z_profile_seq, temporal_seq,
        )

        # ============================================================
        # PHASE 4: DEDICATED WINDOWED BRANCHES
        # ============================================================

        # Branch 2: summary_wind — LSTM over windowed summary features
        # Initialized by static context (identity-conditioned temporal processing)
        h0_sum = self.ch_to_sum_lstm(c_h).unsqueeze(0)  # (1, B, hidden)
        c0_sum = self.cc_to_sum_lstm(c_c).unsqueeze(0)
        _, (h_sum, _) = self.summary_lstm(z_summary_seq, (h0_sum, c0_sum))
        z_summary_wind = self.sum_lstm_proj(h_sum[-1])   # (B, d_model)

        # Branch 3: spatial — fused profile + raster (most recent)
        z_raster = self.raster_net(raster_recent)
        z_prof_recent = z_profile_seq[:, -1, :]
        z_spatial = self.spatial_fuse(z_prof_recent, z_raster)

        # Branch 4: sequential — intraday sequence
        z_seq = self.seq_net(seq_recent, lengths=seq_lens_recent)

        # Branch 5: spatial_wind — attention over daily profile evolution
        # Query: last day's profile, K/V: full window profile sequence
        # Captures how the spatial structure has evolved
        query = z_prof_recent.unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.spatial_wind_attn(
            query=query,
            key=z_profile_seq,
            value=z_profile_seq,
        )
        z_spatial_wind = self.spatial_wind_norm(
            attn_out.squeeze(1) + z_prof_recent
        )
        z_spatial_wind = self.spatial_wind_proj(z_spatial_wind)

        # ============================================================
        # PHASE 5: ANTI-COLLAPSE BRANCH SELECTION
        # ============================================================
        branch_outputs = [
            fused_daily_token,   # Branch 1: backbone-fused daily
            z_summary_wind,      # Branch 2: windowed summary temporal
            z_spatial,           # Branch 3: most-recent spatial
            z_seq,               # Branch 4: intraday sequential
            z_spatial_wind,      # Branch 5: windowed spatial evolution
        ]
        z_selected, branch_weights = self.branch_selector(
            branch_outputs=branch_outputs,
            context=c_s,
        )

        # ============================================================
        # PHASE 6: ENRICHMENT + TEMPORAL ATTENTION
        # ============================================================
        z_event_recent = z_known[:, -1, :]
        enrichment_ctx = self.enrichment_fuse(
            torch.cat([c_e, z_event_recent], dim=-1)
        )

        query = enrichment_ctx.unsqueeze(1)
        attn_out, attn_weights = self.temporal_attn(
            query=query,
            key=fused_daily_seq,
            value=fused_daily_seq,
            need_weights=True,
        )
        z_temporal = self.temporal_ln(attn_out.squeeze(1) + enrichment_ctx)

        # ============================================================
        # PHASE 7: PREDICTION
        # ============================================================
        z_final = torch.cat([z_temporal, z_selected], dim=-1)
        logits = self.head(z_final)

        if self.task == "classification" and return_probs:
            logits = F.softmax(logits, dim=1)

        # ============================================================
        # TRACKING
        # ============================================================
        branch_names = [
            "daily_fused", "summary_wind", "spatial",
            "sequential", "spatial_wind",
        ]
        self._last_tracker = {
            "branch_weights": {
                name: branch_weights[:, i].mean().item()
                for i, name in enumerate(branch_names)
            },
            "branch_entropy": (
                self.branch_selector.last_entropy.item()
                if self.branch_selector.last_entropy is not None
                else None
            ),
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

    def get_entropy_loss(self) -> torch.Tensor:
        """Get BVS entropy regularization loss.

        Add this to your training loss:
            total_loss = task_loss + model.get_entropy_loss()
        """
        return self.branch_selector.get_entropy_loss()


# ============================================================================
# Convenience Variants
# ============================================================================

class MMTFv2Transformer(MMTFv2Core):
    """Transformer-backed MMTF v2 with anti-collapse BVS and 5 branches."""

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
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        sum_lstm_hidden: int = 64,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
        bvs_temperature: float = 1.5,
        bvs_entropy_weight: float = 0.1,
        bvs_min_weight: float = 0.05,
    ):
        super().__init__(
            f_sum=f_sum,
            f_profile=f_profile,
            f_raster=f_raster,
            f_seq=f_seq,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            d_model=d_model,
            d_static_emb=d_static_emb,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            backbone="transformer",
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            sum_lstm_hidden=sum_lstm_hidden,
            task=task,
            num_classes=num_classes,
            dropout=dropout,
            grn_dropout=grn_dropout,
            bvs_temperature=bvs_temperature,
            bvs_entropy_weight=bvs_entropy_weight,
            bvs_min_weight=bvs_min_weight,
        )


class MMTFv2Mamba(MMTFv2Core):
    """Mamba-backed MMTF v2 with anti-collapse BVS and 5 branches."""

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
        n_heads: int = 4,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        sum_lstm_hidden: int = 64,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
        bvs_temperature: float = 1.5,
        bvs_entropy_weight: float = 0.1,
        bvs_min_weight: float = 0.05,
    ):
        super().__init__(
            f_sum=f_sum,
            f_profile=f_profile,
            f_raster=f_raster,
            f_seq=f_seq,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            d_model=d_model,
            d_static_emb=d_static_emb,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            backbone="mamba",
            n_heads=n_heads,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            sum_lstm_hidden=sum_lstm_hidden,
            task=task,
            num_classes=num_classes,
            dropout=dropout,
            grn_dropout=grn_dropout,
            bvs_temperature=bvs_temperature,
            bvs_entropy_weight=bvs_entropy_weight,
            bvs_min_weight=bvs_min_weight,
        )
