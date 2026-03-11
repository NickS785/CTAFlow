"""
MMTFv3: Continuous Trading with TFT/Mamba Backbone + VAE Regime Conditioning
=============================================================================

Proper Temporal Fusion architecture:

  Phase 0: VAE on long-term return/vol → z_regime
           AutoencoderConditionedEncoder(identity + z_regime) → c_s, c_e, c_c, c_h

  Phase 1: Technical features → projection → (B, L, d_model)
           + regime context broadcast per bar (static enrichment)

  Phase 2: Mamba/Transformer backbone over enriched tech sequence
           → fused_seq (B, L, d_model), fused_token (B, d_model)

  Phase 3: Spatial: NumberBars + VPIN raster → SpatialFuse → z_spatial
           Sequential: tabular VPIN buckets → IntradayRNN → z_seq

  Phase 4: BVS([fused_token, z_spatial, z_seq], context=c_s) → z_selected

  Phase 5: Enrichment + Temporal Attentioni
           enrichment_ctx = GRN(c_e)
           temporal_attn(query=enrichment_ctx, K/V=fused_seq) → z_temporal

  Phase 6: head(cat[z_temporal, z_selected]) → tanh → position ∈ [-1, 1]

The backbone IS the model — it processes the primary temporal stream
(intraday technical indicators from ContinuousIntradayPrep), while the
other two branches provide cross-modal microstructure signals through
regime-conditioned branch selection.

Data shapes:
  tech_features:    (B, tech_seq_len, f_tech) — intraday technical indicators
  numbars_recent:   (B, T_max=4, 129, 4)      — previous day number bars
  vpin_recent:      (B, 24, 4, 64)             — previous day VPIN raster
  seq_vpin:         (B, vpin_seq_len, f_seq)    — recent ~1h tabular VPIN
  ae_input:         (B, ae_window, f_ae)        — long-term OHLCV for VAE
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Literal, Optional, Tuple

import torch
import torch.nn as nn

# --- Existing CTAFlow components ---
from CTAFlow.models.deep_learning.multi_branch.market_context_models import (
    BranchVariableSelection,
    GatedResidualNetwork,
)
from CTAFlow.models.deep_learning.multi_branch.tft.mmtf_v2_models import (
    BranchVariableSelectionV2,
)
from CTAFlow.models.deep_learning.encoders import (
    IntradayRNN,
    IntradayTransformer,
    SpatialFuse,
)

# --- VAE + conditioning from auto_mmtft ---
from CTAFlow.models.deep_learning.multi_branch.tft.auto_mmtft import (
    DeterministicRegimeAE,
    VariationalRegimeAE,
    VQRegimeAE,
    AutoencoderConditionedEncoder,
)
from CTAFlow.models.deep_learning.multi_branch.tft.tft_encoders import (
    TransformerTemporalBackbone,
    MambaTemporalBackbone,
    NumberBarEncoder,
    FusedSpatialEncoder,
    VPINRasterEncoder,
)
from CTAFlow.models.deep_learning.training.loss.clf import ContinuousTradingLoss


def _default_unpack_v3(batch, device):
    """Lazy-import fallback for unpack_fn when None is passed."""
    from CTAFlow.data.datasets.v3_continuous import unpack_v3_batch
    return unpack_v3_batch(batch, device=device)


# ============================================================================
# MMTFv3 Core — Temporal Fusion Transformer/Mamba
# ============================================================================

class MMTFv3Core(nn.Module):
    """Multi-Modal Temporal Fusion v3 — Continuous Trading.

    Follows the TFT pattern from MMTFAutoEncoderCore:

    1. VAE encodes long-term return/vol into regime latent
    2. AutoencoderConditionedEncoder fuses identity + regime → 4 context vecs
    3. Technical features are projected, enriched with regime context,
       and processed through a Mamba/Transformer temporal backbone
       → (fused_seq, fused_token)
    4. Spatial (numbars + VPIN raster) and Sequential (VPIN buckets)
       provide cross-modal branch outputs
    5. BranchVariableSelection (regime-conditioned) weights branches
    6. Enrichment context queries over backbone fused_seq via
       temporal self-attention (interpretable TFT attention)
    7. Position head: cat[z_temporal, z_selected] → tanh → [-1, 1]

    3 branches for BVS:
      1. backbone_fused — Mamba/Transformer summary of intraday technicals
      2. spatial        — NumberBars + VPIN raster (previous day, SpatialFuse'd)
      3. sequential     — tabular VPIN buckets (~1h, IntradayRNN)

    Parameters
    ----------
    f_tech : int
        Technical indicator features per bar (from ContinuousIntradayPrep).
    f_seq : int
        Tabular VPIN feature dimension per timestep.
    f_ae : int
        Autoencoder input features per timestep.
    ae_type : str
        'deterministic', 'vae', or 'vqvae'.
    d_latent : int
        Autoencoder bottleneck dimension.
    d_ae_hidden : int
        Autoencoder GRU hidden size.
    ae_n_layers : int
        Autoencoder GRU layers.
    n_codes : int
        VQ-VAE codebook size.
    kl_weight : float
        VAE KL weight.
    recon_weight : float
        Weight of AE reconstruction loss in total loss.
    numbars_channels : int
        Number bar input channels (default 4).
    vpin_channels, vpin_bins, vpin_time : int
        VPIN raster dimensions.
    n_tickers, n_asset_classes, n_asset_subclasses : int
        Identity dimensions.
    d_model : int
        Main model dimension.
    d_static_emb : int
        Static embedding dimension.
    backbone : str
        'transformer' or 'mamba'.
    n_heads : int
        Attention heads (backbone, VPIN encoder, temporal attention).
    n_layers : int
        Backbone layers.
    d_ff : int
        Transformer FFN dimension.
    d_state, d_conv, expand : int
        Mamba configuration.
    spatial_fuse_mode : str
        'gated' or 'mean' for SpatialFuse.
    spatial_fuse_temp : float
        Temperature for gated SpatialFuse.
    dropout : float
        General dropout.
    grn_dropout : float, optional
        GRN/BVS-specific dropout.
    """

    def __init__(
        self,
        # Primary features
        f_tech: int,
        f_seq: int,
        # Autoencoder
        f_ae: int = 6,
        ae_type: Literal["deterministic", "vae", "vqvae"] = "vae",
        d_latent: int = 64,
        d_ae_hidden: int = 128,
        ae_n_layers: int = 2,
        n_codes: int = 16,
        kl_weight: float = 0.01,
        recon_weight: float = 0.1,
        # Spatial dims
        numbars_channels: int = 4,
        vpin_channels: int = 4,
        vpin_bins: int = 64,
        vpin_time: int = 24,
        # Identity
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        # Architecture
        d_model: int = 128,
        d_static_emb: int = 64,
        backbone: Literal["transformer", "mamba"] = "mamba",
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        spatial_encoder: Literal["separate", "fused"] = "separate",
        spatial_fuse_mode: str = "gated",
        spatial_fuse_temp: float = 2.0,
        # Sequential encoder
        seq_layers: int = 2,
        seq_nheads: int = 4,
        # Training
        dropout: float = 0.2,
        grn_dropout: float | None = None,
        # Anti-collapse (BVS V2)
        bvs_temperature: float = 1.5,
        bvs_entropy_weight: float = 0.1,
        bvs_min_weight: float = 0.05,
        bvs_pre_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.ae_type = ae_type
        self.recon_weight = recon_weight
        self.backbone_type = backbone
        self.spatial_encoder_type = spatial_encoder
        self.dropout = dropout

        _grn_drop = grn_dropout if grn_dropout is not None else dropout

        # ==============================================================
        # AUTOENCODER (long-term return/volatility → regime latent)
        # ==============================================================
        if ae_type == "deterministic":
            self.autoencoder = DeterministicRegimeAE(
                f_input=f_ae, d_hidden=d_ae_hidden,
                d_latent=d_latent, n_layers=ae_n_layers,
                dropout=dropout,
            )
        elif ae_type == "vae":
            self.autoencoder = VariationalRegimeAE(
                f_input=f_ae, d_hidden=d_ae_hidden,
                d_latent=d_latent, n_layers=ae_n_layers,
                kl_weight=kl_weight, dropout=dropout,
            )
        elif ae_type == "vqvae":
            self.autoencoder = VQRegimeAE(
                f_input=f_ae, d_hidden=d_ae_hidden,
                d_latent=d_latent, n_codes=n_codes,
                n_layers=ae_n_layers, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown ae_type: {ae_type}")

        # ==============================================================
        # AUTOENCODER-CONDITIONED STATIC ENCODER
        # Produces 4 context vectors from identity + z_regime:
        #   c_s (selection), c_e (enrichment), c_c (cell), c_h (hidden)
        # ==============================================================
        self.regime_encoder = AutoencoderConditionedEncoder(
            d_model=d_model,
            d_latent=d_latent,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            d_emb=d_static_emb,
            dropout=_grn_drop,
        )

        # Project c_h to per-bar regime context for backbone input
        self.regime_bar_proj = GatedResidualNetwork(
            d_model=d_model, dropout=_grn_drop,
        )

        # ==============================================================
        # TECHNICAL FEATURE PROJECTION (backbone input)
        # Projects raw tech features to d_model, then adds regime context
        # ==============================================================
        self.tech_proj = nn.Sequential(
            nn.LayerNorm(f_tech),
            nn.Linear(f_tech, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ==============================================================
        # TEMPORAL FUSION BACKBONE (Mamba or Transformer)
        # Processes enriched technical sequence → (fused_seq, fused_token)
        # ==============================================================
        if backbone == "transformer":
            self.fusion_backbone = TransformerTemporalBackbone(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
            )
        elif backbone == "mamba":
            self.fusion_backbone = MambaTemporalBackbone(
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ==============================================================
        # BRANCH 2: SPATIAL (number bars + VPIN raster → fused)
        # ==============================================================
        if spatial_encoder == "separate":
            self.numbar_encoder = NumberBarEncoder(
                in_channels=numbars_channels, d_model=d_model,
            )
            self.vpin_encoder = VPINRasterEncoder(
                in_channels=vpin_channels, n_bins=vpin_bins,
                n_time=vpin_time, d_model=d_model,
                n_heads=n_heads, dropout=dropout,
            )
            self.spatial_fuse = SpatialFuse(
                d_spatial=d_model, mode=spatial_fuse_mode,
                temperature=spatial_fuse_temp,
            )
            self.fused_spatial_encoder = None
        elif spatial_encoder == "fused":
            self.numbar_encoder = None
            self.vpin_encoder = None
            self.spatial_fuse = None
            self.fused_spatial_encoder = FusedSpatialEncoder(
                in_channels=numbars_channels + 3, d_model=d_model,
            )
        else:
            raise ValueError(f"Unknown spatial_encoder: {spatial_encoder}")

        # ==============================================================
        # BRANCH 3: SEQUENTIAL (tabular VPIN buckets, ~1h)
        # ==============================================================
        self.seq_net = IntradayTransformer(
            input_dim=f_seq, d_model=d_model, num_layers=seq_layers,
            dropout=dropout, nhead=seq_nheads,
        )

        # ==============================================================
        # BRANCH SELECTION (regime-conditioned via c_s)
        # ==============================================================
        self.branch_selector = BranchVariableSelectionV2(
            n_branches=3,
            d_branch=d_model,
            d_context=d_model,
            dropout=_grn_drop,
            temperature=bvs_temperature,
            entropy_weight=bvs_entropy_weight,
            min_weight=bvs_min_weight,
            use_pre_norm=bvs_pre_norm,
        )

        # ==============================================================
        # ENRICHMENT + TEMPORAL ATTENTION (TFT pattern)
        # Enrichment context queries over backbone sequence
        # ==============================================================
        self.enrichment_grn = GatedResidualNetwork(
            d_model=d_model, dropout=_grn_drop,
        )

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_ln = nn.LayerNorm(d_model)

        # ==============================================================
        # CONTINUOUS POSITION HEAD
        # ==============================================================
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),  # position ∈ [-1, 1]
        )

        self._last_tracker: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        # Technical indicators (backbone input — primary temporal stream)
        tech_features: torch.Tensor,        # (B, L, f_tech)
        tech_lens: torch.Tensor,            # (B,)
        # Spatial (previous day)
        numbars_recent: Optional[torch.Tensor] = None,
        vpin_raster_recent: Optional[torch.Tensor] = None,
        fused_spatial: Optional[torch.Tensor] = None,
        # Sequential VPIN (~1h lookback)
        seq_vpin: Optional[torch.Tensor] = None,
        seq_vpin_lens: Optional[torch.Tensor] = None,
        # Autoencoder input (long-term window)
        ae_input: Optional[torch.Tensor] = None,
        # Identity
        ticker_id: Optional[torch.Tensor] = None,
        asset_class_id: Optional[torch.Tensor] = None,
        asset_subclass_id: Optional[torch.Tensor] = None,
        # Control
        return_tracker: bool = False,
        return_ae_losses: bool = True,
    ):
        """
        Returns
        -------
        position : Tensor (B, 1) — continuous position ∈ [-1, 1]
        ae_losses : dict (if return_ae_losses)
        tracker : dict (if return_tracker)
        """
        B, L, _ = tech_features.shape
        if ae_input is None or seq_vpin is None or seq_vpin_lens is None:
            raise ValueError("ae_input, seq_vpin, and seq_vpin_lens are required")
        if ticker_id is None or asset_class_id is None or asset_subclass_id is None:
            raise ValueError("ticker_id, asset_class_id, and asset_subclass_id are required")

        # ==========================================================
        # PHASE 0: AUTOENCODER → REGIME LATENT
        # ==========================================================
        z_regime, _x_recon, ae_losses = self.autoencoder(ae_input)
        if "total_ae_loss" not in ae_losses and "recon_loss" in ae_losses:
            ae_losses["total_ae_loss"] = ae_losses["recon_loss"]

        # ==========================================================
        # PHASE 1: REGIME-CONDITIONED CONTEXT
        # ==========================================================
        c_s, c_e, c_c, c_h = self.regime_encoder(
            ticker_id=ticker_id,
            asset_class_id=asset_class_id,
            asset_subclass_id=asset_subclass_id,
            z_regime=z_regime,
        )

        # ==========================================================
        # PHASE 2: BACKBONE — TEMPORAL FUSION OVER TECHNICAL STREAM
        # Project tech features → d_model, add regime context per bar,
        # then run through Mamba/Transformer backbone.
        # ==========================================================
        z_tech = self.tech_proj(tech_features)                      # (B, L, d_model)
        z_regime_bar = self.regime_bar_proj(c_h)                    # (B, d_model)
        z_tech_enriched = z_tech + z_regime_bar.unsqueeze(1)        # (B, L, d_model)

        fused_seq, fused_token = self.fusion_backbone(z_tech_enriched)
        # fused_seq:   (B, L, d_model) — for temporal attention
        # fused_token: (B, d_model)    — backbone summary for BVS

        # ==========================================================
        # PHASE 3: CROSS-MODAL BRANCH ENCODING
        # ==========================================================
        # Branch 2: Spatial — number bars + VPIN raster
        if self.spatial_encoder_type == "separate":
            if numbars_recent is None or vpin_raster_recent is None:
                raise ValueError(
                    "numbars_recent and vpin_raster_recent are required when spatial_encoder='separate'"
                )
            z_numbars = self.numbar_encoder(numbars_recent)
            z_vpin = self.vpin_encoder(vpin_raster_recent)
            z_spatial = self.spatial_fuse(z_numbars, z_vpin)
            spatial_importance = self.spatial_fuse.get_importance_stats()
        else:
            if fused_spatial is None:
                raise ValueError("fused_spatial is required when spatial_encoder='fused'")
            z_spatial = self.fused_spatial_encoder(fused_spatial)
            spatial_importance = {
                "mode": "fused",
                "channels": int(fused_spatial.shape[2]) if fused_spatial.dim() == 4 else None,
            }

        # Branch 3: Sequential — tabular VPIN
        z_seq = self.seq_net(seq_vpin, lengths=seq_vpin_lens)

        # ==========================================================
        # PHASE 4: REGIME-CONDITIONED BRANCH SELECTION
        # ==========================================================
        branch_outputs = [fused_token, z_spatial, z_seq]
        z_selected, branch_weights = self.branch_selector(
            branch_outputs=branch_outputs,
            context=c_s,
        )

        # ==========================================================
        # PHASE 5: ENRICHMENT + TEMPORAL ATTENTION (TFT pattern)
        # The enrichment context vector queries over the backbone
        # sequence — this is the interpretable temporal self-attention
        # from the original TFT paper.
        # ==========================================================
        enrichment_ctx = self.enrichment_grn(c_e)                   # (B, d_model)

        query = enrichment_ctx.unsqueeze(1)                         # (B, 1, d_model)
        attn_out, attn_weights = self.temporal_attn(
            query=query,
            key=fused_seq,
            value=fused_seq,
            need_weights=True,
        )
        z_temporal = self.temporal_ln(
            attn_out.squeeze(1) + enrichment_ctx
        )                                                           # (B, d_model)

        # ==========================================================
        # PHASE 6: PREDICTION
        # ==========================================================
        z_final = torch.cat([z_temporal, z_selected], dim=-1)       # (B, 2*d_model)
        ptp_context = torch.cat([c_h, z_temporal], dim=-1)          # (B, 2*d_model)

        logits = None
        if isinstance(self.head, QuantilePositionHead):
            position, logits = self.head(
                z_final, context=ptp_context,
            )                                                       # (B,1), (B,5)
        else:
            position = self.head(z_final)                           # (B, 1)

        # ==========================================================
        # TRACKING
        # ==========================================================
        # --- Branch weight stats ---
        bw_entropy = -(branch_weights * (branch_weights + 1e-8).log()).sum(dim=-1).mean().item()

        self._last_tracker = {
            "branch_weights": {
                name: branch_weights[:, i].mean().item()
                for i, name in enumerate(["backbone_fused", "spatial", "sequential"])
            },
            "branch_weights_std": {
                name: branch_weights[:, i].std().item()
                for i, name in enumerate(["backbone_fused", "spatial", "sequential"])
            },
            "branch_weights_entropy": bw_entropy,
            "branch_weights_min": branch_weights.min(dim=-1).values.mean().item(),
            "bvs_entropy_loss": (
                self.branch_selector.last_entropy_loss.item()
                if hasattr(self.branch_selector, "last_entropy_loss")
                and self.branch_selector.last_entropy_loss is not None
                else 0.0
            ),
            "regime_var_weights": {
                name: self.regime_encoder.last_var_weights[:, i].mean().item()
                for i, name in enumerate(self.regime_encoder.var_names)
            } if self.regime_encoder.last_var_weights is not None else None,
            "spatial_fuse": spatial_importance,
            "temporal_attn": attn_weights.detach().squeeze(1),
            "backbone": dict(self.fusion_backbone.last_tracker),
            "seq_net": dict(self.seq_net.last_tracker),
            "seq_feature_weights": self.seq_net.last_tracker.get("feature_weights"),
            "ae_losses": {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in ae_losses.items()
                if k not in ("code_indices", "codebook_usage")
            },
            "avg_position": position.detach().mean().item(),
            "avg_abs_position": position.detach().abs().mean().item(),
            "ptp_context_norm": ptp_context.detach().norm(dim=-1).mean().item(),
        }
        if logits is not None and hasattr(self.head, "get_last_stats"):
            self._last_tracker["ptp"] = self.head.get_last_stats()

        # Build return
        results = [position]
        if return_ae_losses:
            results.append(ae_losses)
        if logits is not None:
            results.append(logits)
        if return_tracker:
            results.append(self._last_tracker)
        return results[0] if len(results) == 1 else tuple(results)

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)

    def get_regime_latent(self, ae_input: torch.Tensor) -> torch.Tensor:
        """Extract regime latent without full forward — for analysis."""
        return self.autoencoder.encode(ae_input)


# ============================================================================
# Convenience Variants
# ============================================================================

class MMTFv3Mamba(MMTFv3Core):
    """MMTFv3 with Mamba backbone + VAE regime conditioning."""
    def __init__(self, **kwargs):
        kwargs.setdefault("backbone", "mamba")
        kwargs.setdefault("ae_type", "vae")
        super().__init__(**kwargs)


class MMTFv3Transformer(MMTFv3Core):
    """MMTFv3 with Transformer backbone + VAE regime conditioning."""
    def __init__(self, **kwargs):
        kwargs.setdefault("backbone", "transformer")
        kwargs.setdefault("ae_type", "vae")
        super().__init__(**kwargs)


class MMTFv3MambaVQVAE(MMTFv3Core):
    """MMTFv3 with Mamba backbone + VQ-VAE regime conditioning."""
    def __init__(self, **kwargs):
        kwargs.setdefault("backbone", "mamba")
        kwargs.setdefault("ae_type", "vqvae")
        super().__init__(**kwargs)


# ============================================================================
# PredictionToPosition — 5-class quantile → continuous position
# ============================================================================

def returns_to_classes(
    returns: torch.Tensor,
    outer: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Bucket forward returns into 4 conviction classes using σ-based thresholds.

    Classes: 0=strong_neg, 1=weak_neg, 2=weak_pos, 3=strong_pos
    No neutral class — the market is rarely flat.

    Parameters
    ----------
    returns : Tensor (B,) or (B, 1)
        Forward returns.
    outer : float
        Threshold in σ units for strong conviction boundary. Default 1.0.

    Returns
    -------
    labels : Tensor (B,) long in {0, 1, 2, 3}
    """
    r = returns.detach().view(-1)
    sigma = r.std().clamp(min=1e-8)

    labels = torch.full_like(r, 1, dtype=torch.long)          # weak neg default
    labels[r <= -outer * sigma] = 0                            # strong neg
    labels[(r > -outer * sigma) & (r < 0)] = 1                 # weak neg
    labels[(r >= 0) & (r < outer * sigma)] = 2                 # weak pos
    labels[r >= outer * sigma] = 3                             # strong pos
    return labels


class PredictionToPosition(nn.Module):
    """Convert 4-class quantile logits → continuous position ∈ [-1, 1].

    Classes: 0=strong_neg, 1=weak_neg, 2=weak_pos, 3=strong_pos

    Position = tanh(temperature * sum(p_i * anchor_i))
    where anchors are learnable, initialized to [-1.0, -0.33, +0.33, +1.0].
    """

    N_CLASSES = 4

    def __init__(
        self,
        temperature: float = 1.5,
        context_dim: int = 0,
        max_anchor_shift: float = 0.35,
        max_temp_scale: float = 0.30,
    ):
        super().__init__()
        self.temperature = temperature
        self.context_dim = context_dim
        self.max_anchor_shift = max_anchor_shift
        self.max_temp_scale = max_temp_scale
        self.anchors = nn.Parameter(
            torch.tensor([-1.0, -0.33, 0.33, 1.0]),
        )
        if context_dim > 0:
            ctx_hidden = max(16, min(128, context_dim))
            self.context_to_anchor = nn.Sequential(
                nn.Linear(context_dim, ctx_hidden),
                nn.LayerNorm(ctx_hidden),
                nn.GELU(),
                nn.Linear(ctx_hidden, self.N_CLASSES),
            )
            self.context_to_temp = nn.Sequential(
                nn.Linear(context_dim, ctx_hidden),
                nn.GELU(),
                nn.Linear(ctx_hidden, 1),
                nn.Tanh(),
            )
        else:
            self.context_to_anchor = None
            self.context_to_temp = None
        self._last_stats: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        logits : Tensor (B, 4)

        Returns
        -------
        position : Tensor (B, 1)
        logits : Tensor (B, 4) — passed through for multi-loss
        """
        batch_size = logits.size(0)
        anchors = self.anchors.unsqueeze(0).expand(batch_size, -1)
        temperature = torch.full(
            (batch_size, 1),
            fill_value=self.temperature,
            dtype=logits.dtype,
            device=logits.device,
        )
        anchor_shift = torch.zeros_like(anchors)

        if context is not None and self.context_to_anchor is not None:
            anchor_shift = self.max_anchor_shift * torch.tanh(
                self.context_to_anchor(context),
            )
            anchors = anchors + anchor_shift
            temperature = temperature * (
                1.0 + self.max_temp_scale * self.context_to_temp(context)
            ).clamp(min=0.5, max=1.5)

        probs = torch.softmax(logits / temperature, dim=-1)         # (B, 5)
        weighted = (probs * anchors).sum(dim=-1)                    # (B,)
        position = torch.tanh(weighted).unsqueeze(-1)                # (B, 1)
        self._last_stats = {
            "anchor_shift_mean": anchor_shift.detach().abs().mean().item(),
            "temperature_mean": temperature.detach().mean().item(),
            "anchor_mean": anchors.detach().mean().item(),
        }
        return position, logits

    def get_last_stats(self) -> Dict[str, float]:
        return dict(self._last_stats)


class QuantilePositionHead(nn.Module):
    """MLP → 4-class logits → PredictionToPosition → continuous position.

    Classes: 0=strong_neg, 1=weak_neg, 2=weak_pos, 3=strong_pos.
    Replaces the standard Tanh head in MMTFv3Core when quantile_head=True.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        dropout: float = 0.2,
        temperature: float = 1.5,
        context_dim: int = 0,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_proj = None
        if context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.Linear(context_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        self.hidden_to_logits = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, PredictionToPosition.N_CLASSES),
        )
        self.ptp = PredictionToPosition(
            temperature=temperature,
            context_dim=context_dim,
        )

    def forward(
        self,
        z_final: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_final : Tensor (B, d_input) — typically 2*d_model

        Returns
        -------
        position : Tensor (B, 1)
        logits : Tensor (B, 4)
        """
        hidden = self.feature_proj(z_final)
        if context is not None and self.context_proj is not None:
            hidden = hidden + self.context_proj(context)
        logits = self.hidden_to_logits(hidden)                       # (B, 4)
        position, logits = self.ptp(logits, context=context)        # (B, 1), (B, 4)
        return position, logits

    def quantile_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.feature_proj.parameters()
        if self.context_proj is not None:
            yield from self.context_proj.parameters()
        yield from self.hidden_to_logits.parameters()

    def ptp_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.ptp.parameters()

    def get_last_stats(self) -> Dict[str, float]:
        return self.ptp.get_last_stats()


# ============================================================================
# Stateful Position Layer
# ============================================================================

class TickerPositionStateLayer(nn.Module):
    """Online ticker-level position state and fusion layer."""

    def __init__(
        self,
        n_tickers: int,
        hidden_dim: int = 16,
        momentum: float = 0.9,
    ):
        super().__init__()
        if n_tickers <= 0:
            raise ValueError(f"n_tickers must be > 0, got {n_tickers}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        self.n_tickers = n_tickers
        self.momentum = momentum
        self.fuser = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("position_state", torch.zeros(n_tickers, 1))

    @torch.no_grad()
    def reset(self, value: float = 0.0) -> None:
        self.position_state.fill_(value)

    @torch.no_grad()
    def _update_state(
        self,
        ticker_id: torch.Tensor,
        next_position: torch.Tensor,
    ) -> None:
        ticker_flat = ticker_id.view(-1).long()
        next_pos = next_position.detach().view(-1, 1)

        for tid in ticker_flat.unique(sorted=False):
            mask = ticker_flat == tid
            mean_pos = next_pos[mask].mean(dim=0)
            idx = int(tid.item())
            prev = self.position_state[idx]
            self.position_state[idx] = self.momentum * prev + (1.0 - self.momentum) * mean_pos

    def forward(
        self,
        base_position: torch.Tensor,
        ticker_id: torch.Tensor,
        update_state: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ticker_flat = ticker_id.view(-1).long()
        prev_state = self.position_state.index_select(0, ticker_flat).to(base_position.dtype)

        fused = torch.cat([base_position, prev_state], dim=-1)
        state_delta = self.fuser(fused)
        position = torch.tanh(base_position + state_delta)

        if update_state:
            self._update_state(ticker_flat, position)

        return position, prev_state, state_delta


class StatefulMMTFv3Core(nn.Module):
    """MMTFv3Core wrapped with a ticker-aware online position state layer.

    Parameters
    ----------
    base_model : MMTFv3Core
        The base MMTFv3 model.
    n_tickers : int
        Number of tickers for position state tracking.
    quantile_head : bool
        If True, replace the base model's head with a QuantilePositionHead
        that produces 5-class logits → continuous position via PTP.
        Forward returns ``(position, ae_losses, logits)`` when True.
    ptp_temperature : float
        Temperature for PredictionToPosition sharpness. Default 1.5.
    state_hidden_dim : int
        Hidden dim for the state fusion MLP. Default 16.
    state_momentum : float
        EMA momentum for position state. Default 0.9.
    update_on_eval : bool
        Whether to update position state during evaluation. Default True.
    """

    def __init__(
        self,
        base_model: MMTFv3Core,
        n_tickers: int,
        quantile_head: bool = False,
        ptp_temperature: float = 1.5,
        state_hidden_dim: int = 16,
        state_momentum: float = 0.9,
        update_on_eval: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.recon_weight = base_model.recon_weight
        self.update_on_eval = update_on_eval
        self.quantile_head = quantile_head

        if quantile_head:
            base_model.head = QuantilePositionHead(
                d_input=base_model.d_model * 2,
                d_model=base_model.d_model,
                dropout=base_model.dropout,
                temperature=ptp_temperature,
                context_dim=base_model.d_model * 2,
            )

        self.state_layer = TickerPositionStateLayer(
            n_tickers=n_tickers,
            hidden_dim=state_hidden_dim,
            momentum=state_momentum,
        )
        self._last_tracker: Dict[str, object] = {}

    @torch.no_grad()
    def reset_position_state(self, value: float = 0.0) -> None:
        self.state_layer.reset(value=value)

    def forward(
        self,
        return_tracker: bool = False,
        return_ae_losses: bool = True,
        **inputs,
    ):
        base_out = self.base_model(
            **inputs,
            return_tracker=return_tracker,
            return_ae_losses=True,
        )

        # Unpack base model outputs — logits present when using quantile head
        logits = None
        if self.quantile_head:
            if return_tracker:
                base_position, ae_losses, logits, tracker = base_out
            else:
                base_position, ae_losses, logits = base_out
                tracker = {}
        else:
            if return_tracker:
                base_position, ae_losses, tracker = base_out
            else:
                base_position, ae_losses = base_out
                tracker = {}

        should_update_state = self.training or self.update_on_eval
        position, prev_state, state_delta = self.state_layer(
            base_position=base_position,
            ticker_id=inputs["ticker_id"],
            update_state=should_update_state,
        )

        tracker = dict(tracker)
        tracker.update(
            {
                "avg_prev_state": prev_state.detach().mean().item(),
                "avg_state_delta": state_delta.detach().mean().item(),
                "avg_stateful_position": position.detach().mean().item(),
                "avg_abs_stateful_position": position.detach().abs().mean().item(),
            }
        )
        self._last_tracker = tracker

        outputs = [position]
        if return_ae_losses:
            outputs.append(ae_losses)
        if logits is not None:
            outputs.append(logits)
        if return_tracker:
            outputs.append(tracker)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)


# ============================================================================
# Training Helpers
# ============================================================================

def _maybe_reset_position_state(model) -> None:
    if hasattr(model, "reset_position_state") and callable(model.reset_position_state):
        model.reset_position_state()

def train_epoch_v3(
    model: MMTFv3Core,
    loader,
    loss_fn: ContinuousTradingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
    unpack_fn=None,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch with continuous trading loss + AE reconstruction."""
    model.train()
    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        _unpack = unpack_fn or _default_unpack_v3
        inputs, targets = _unpack(batch, device=device)
        targets = targets.float()

        optimizer.zero_grad()
        position, ae_losses = model(**inputs, return_ae_losses=True)

        trading_loss, metrics = loss_fn(position, targets)
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"]
        bvs_entropy_loss = _get_bvs_entropy_loss(model)
        loss = trading_loss + ae_loss + bvs_entropy_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += loss.item()
        for k, v in metrics.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v
        metric_accum["ae_recon"] = metric_accum.get("ae_recon", 0.0) + ae_losses["recon_loss"].item()
        if "kl_loss" in ae_losses:
            metric_accum["ae_kl"] = metric_accum.get("ae_kl", 0.0) + ae_losses["kl_loss"].item()
        if bvs_entropy_loss.item() > 0:
            metric_accum["bvs_entropy"] = metric_accum.get("bvs_entropy", 0.0) + bvs_entropy_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, {k: v / n for k, v in metric_accum.items()}


def train_epoch_v3_stateful(
    model: nn.Module,
    loader,
    loss_fn: ContinuousTradingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
    unpack_fn=None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch with online ticker state updates.

    Pass ``scaler=torch.amp.GradScaler()`` to enable mixed-precision training.
    """
    model.train()
    _maybe_reset_position_state(model)
    use_amp = scaler is not None

    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        _unpack = unpack_fn or _default_unpack_v3
        inputs, targets = _unpack(batch, device=device)
        targets = targets.float()

        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_amp):
            position, ae_losses = model(**inputs, return_ae_losses=True)

        # Compute loss in fp32 — Sharpe-like ratios underflow in fp16
        position = position.float()
        trading_loss, metrics = loss_fn(position, targets)
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"].float()
        bvs_entropy_loss = _get_bvs_entropy_loss(model)
        loss = trading_loss + ae_loss + bvs_entropy_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        total_loss += loss.item()
        for k, v in metrics.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v
        metric_accum["ae_recon"] = metric_accum.get("ae_recon", 0.0) + ae_losses["recon_loss"].item()
        if "kl_loss" in ae_losses:
            metric_accum["ae_kl"] = metric_accum.get("ae_kl", 0.0) + ae_losses["kl_loss"].item()
        if bvs_entropy_loss.item() > 0:
            metric_accum["bvs_entropy"] = metric_accum.get("bvs_entropy", 0.0) + bvs_entropy_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, {k: v / n for k, v in metric_accum.items()}


def _exposure_adjusted_sharpe(
    raw_sharpe: float,
    avg_exposure: float,
    min_exposure: float = 0.15,
) -> float:
    """Penalize low-exposure Sharpe so conservative models don't dominate.

    Below ``min_exposure`` the Sharpe is quadratically discounted toward 0,
    preventing "do-nothing" strategies from winning Optuna selection.
    """
    if avg_exposure < min_exposure:
        penalty = (avg_exposure / min_exposure) ** 2
    else:
        penalty = 1.0
    return raw_sharpe * penalty


@torch.no_grad()
def evaluate_v3(
    model: MMTFv3Core,
    loader,
    loss_fn: ContinuousTradingLoss,
    device: torch.device,
    unpack_fn=None,
) -> Dict[str, float]:
    """Evaluate with continuous trading metrics."""
    model.eval()
    all_positions, all_returns = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        _unpack = unpack_fn or _default_unpack_v3
        inputs, targets = _unpack(batch, device=device)
        targets = targets.float()
        position, ae_losses = model(**inputs, return_ae_losses=True)
        trading_loss, _ = loss_fn(position, targets)
        total_loss += (trading_loss + model.recon_weight * ae_losses["total_ae_loss"]).item()
        n_batches += 1
        all_positions.append(position.squeeze().cpu())
        all_returns.append(targets.squeeze().cpu())

    positions = torch.cat(all_positions)
    returns = torch.cat(all_returns)
    strategy_ret = positions * returns

    n = max(n_batches, 1)
    mean_ret = strategy_ret.mean().item()
    std_ret = strategy_ret.std().item() + 1e-8

    gross_profit = strategy_ret[strategy_ret > 0].sum().item()
    gross_loss = strategy_ret[strategy_ret < 0].abs().sum().item() + 1e-8
    cum_ret = strategy_ret.cumsum(dim=0)

    correct_dir = ((positions > 0) & (returns > 0)) | ((positions < 0) & (returns < 0))
    non_flat = positions.abs() > 0.05
    dir_acc = (correct_dir & non_flat).float().sum().item() / max(non_flat.float().sum().item(), 1)

    avg_exposure = positions.abs().mean().item()
    raw_sharpe = mean_ret / std_ret
    adj_sharpe = _exposure_adjusted_sharpe(raw_sharpe, avg_exposure)

    return {
        "loss": total_loss / n,
        "sharpe": adj_sharpe,
        "sharpe_raw": raw_sharpe,
        "sortino": mean_ret / (strategy_ret.clamp(max=0.0).pow(2).mean().sqrt().item() + 1e-8),
        "mean_strategy_ret": mean_ret,
        "win_rate": (strategy_ret > 0).float().mean().item() * 100.0,
        "dir_accuracy": dir_acc * 100.0,
        "profit_factor": gross_profit / gross_loss,
        "max_drawdown": (cum_ret.cummax(dim=0)[0] - cum_ret).max().item(),
        "avg_exposure": avg_exposure,
        "avg_position": positions.mean().item(),
        "n_samples": len(positions),
    }


@torch.no_grad()
def evaluate_v3_stateful(
    model: nn.Module,
    loader,
    loss_fn: ContinuousTradingLoss,
    device: torch.device,
    unpack_fn=None,
) -> Dict[str, float]:
    """Evaluate with state reset and online stateful inference."""
    model.eval()
    _maybe_reset_position_state(model)
    all_positions, all_returns = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        _unpack = unpack_fn or _default_unpack_v3
        inputs, targets = _unpack(batch, device=device)
        targets = targets.float()
        position, ae_losses = model(**inputs, return_ae_losses=True)
        trading_loss, _ = loss_fn(position, targets)
        total_loss += (trading_loss + model.recon_weight * ae_losses["total_ae_loss"]).item()
        n_batches += 1
        all_positions.append(position.detach().view(-1).cpu())
        all_returns.append(targets.detach().view(-1).cpu())

    positions = torch.cat(all_positions)
    returns = torch.cat(all_returns)
    strategy_ret = positions * returns

    n = max(n_batches, 1)
    mean_ret = strategy_ret.mean().item()
    std_ret = strategy_ret.std().item() + 1e-8

    gross_profit = strategy_ret[strategy_ret > 0].sum().item()
    gross_loss = strategy_ret[strategy_ret < 0].abs().sum().item() + 1e-8
    cum_ret = strategy_ret.cumsum(dim=0)

    correct_dir = ((positions > 0) & (returns > 0)) | ((positions < 0) & (returns < 0))
    non_flat = positions.abs() > 0.05
    dir_acc = (correct_dir & non_flat).float().sum().item() / max(non_flat.float().sum().item(), 1)

    avg_exposure = positions.abs().mean().item()
    raw_sharpe = mean_ret / std_ret
    adj_sharpe = _exposure_adjusted_sharpe(raw_sharpe, avg_exposure)

    return {
        "loss": total_loss / n,
        "sharpe": adj_sharpe,
        "sharpe_raw": raw_sharpe,
        "sortino": mean_ret / (strategy_ret.clamp(max=0.0).pow(2).mean().sqrt().item() + 1e-8),
        "mean_strategy_ret": mean_ret,
        "win_rate": (strategy_ret > 0).float().mean().item() * 100.0,
        "dir_accuracy": dir_acc * 100.0,
        "profit_factor": gross_profit / gross_loss,
        "max_drawdown": (cum_ret.cummax(dim=0)[0] - cum_ret).max().item(),
        "avg_exposure": avg_exposure,
        "avg_position": positions.mean().item(),
        "n_samples": len(positions),
    }


def print_v3_diagnostics(tracker: dict, eval_metrics: dict, epoch: int = 0) -> None:
    """Pretty-print backbone, branch, spatial, regime, and trading diagnostics."""
    print(f"\n{'='*60}")
    print(f"  Epoch {epoch} Diagnostics")
    print(f"{'='*60}")

    bw = tracker.get("branch_weights", {})
    bw_std = tracker.get("branch_weights_std", {})
    if bw:
        bw_ent = tracker.get("branch_weights_entropy", 0)
        max_w = max(bw.values()) if bw else 1
        print(f"\n  Branch Importance (entropy={bw_ent:.3f}):")
        for name, weight in sorted(bw.items(), key=lambda x: -x[1]):
            std = bw_std.get(name, 0)
            bar = "#" * int(weight / max(max_w, 1e-8) * 40)
            print(f"    {name:<16s}: {weight:.4f} +/- {std:.4f} {bar}")

    sf = tracker.get("spatial_fuse")
    if sf:
        print(f"\n  Spatial Fuse (numbar vs vpin):")
        print(f"    numbar: {sf.get('profile_mean', 0):.3f} +/- {sf.get('profile_std', 0):.3f}")
        print(f"    vpin:   {sf.get('nb_mean', 0):.3f} +/- {sf.get('nb_std', 0):.3f}")

    rw = tracker.get("regime_var_weights")
    if rw:
        print(f"\n  Regime Variable Selection:")
        for name, weight in sorted(rw.items(), key=lambda x: -x[1]):
            print(f"    {name:<16s}: {weight:.4f}")

    bb = tracker.get("backbone", {})
    if bb:
        bb_ent = bb.get("pool_weights_entropy", 0)
        bb_max = bb.get("pool_max_weight", 0)
        print(f"\n  Backbone: entropy={bb_ent:.3f}, max_pool_wt={bb_max:.3f}")

    sn = tracker.get("seq_net", {})
    if sn:
        sn_ent = sn.get("pool_entropy", 0)
        sn_max = sn.get("pool_max_weight", 0)
        sn_feat_ent = sn.get("feature_entropy", 0)
        sn_feat_max = sn.get("feature_max_weight", 0)
        print(f"  Seq Net:  pool_entropy={sn_ent:.3f}, max_pool_wt={sn_max:.3f}, "
              f"feat_entropy={sn_feat_ent:.3f}, feat_max_wt={sn_feat_max:.3f}")

    seq_fw = tracker.get("seq_feature_weights")
    if seq_fw is not None:
        fw = seq_fw if isinstance(seq_fw, dict) else None
        if fw is None and hasattr(seq_fw, "cpu"):
            # Raw tensor — print top-5 and bottom-5 by weight
            vals = seq_fw.cpu().numpy()
            ranked = sorted(enumerate(vals), key=lambda x: -x[1])
            print(f"\n  Seq Feature Importance (top-5 / bottom-5):")
            for idx, w in ranked[:5]:
                bar = "#" * int(w / max(vals.max(), 1e-8) * 30)
                print(f"    feat[{idx:>3d}]: {w:.4f} {bar}")
            print(f"    ...")
            for idx, w in ranked[-5:]:
                print(f"    feat[{idx:>3d}]: {w:.4f}")

    ae = tracker.get("ae_losses", {})
    if ae:
        parts = [f"recon={ae.get('recon_loss', 0):.4f}"]
        if "kl_loss" in ae:
            parts.append(f"kl={ae['kl_loss']:.4f}")
        print(f"\n  AE: {', '.join(parts)}")

    print("\n  Trading Metrics:")
    for key in ["sharpe", "sortino", "win_rate", "dir_accuracy",
                "profit_factor", "max_drawdown", "avg_exposure", "mean_strategy_ret"]:
        if key in eval_metrics:
            val = eval_metrics[key]
            fmt = f"{val:.4f}" if abs(val) < 10 else f"{val:.1f}"
            print(f"    {key:<22s}: {fmt}")
    print()


# ============================================================================
# PTP Composite Loss
# ============================================================================

class PTPLoss(nn.Module):
    """Combined loss for PredictionToPosition training.

    Components:
      1. Profit-weighted CE on 4-class logits (classification quality)
      2. ContinuousTradingLoss on continuous position (PnL/Sharpe)

    The CE uses 4-class direction mapping: classes {0,1} → negative,
    classes {2,3} → positive. No neutral class.

    Parameters
    ----------
    ce_weight : float
        Weight for the classification loss component. Default 1.0.
    pnl_weight : float
        Weight for the continuous trading loss component. Default 1.0.
    profit_scale : float
        Scale factor for return-magnitude weighting in CE. Default 100.0.
    direction_penalty : float
        Extra penalty for wrong-direction predictions in CE. Default 1.0.
    outer_threshold : float
        σ threshold for strong conviction in returns_to_classes. Default 1.0.
    **trading_kwargs
        Passed to ContinuousTradingLoss (tc_cost, direction_weight, etc.).
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        pnl_weight: float = 1.0,
        profit_scale: float = 100.0,
        direction_penalty: float = 1.0,
        outer_threshold: float = 1.0,
        inner_threshold: float = None,  # unused, kept for backward compat
        **trading_kwargs,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.pnl_weight = pnl_weight
        self.profit_scale = profit_scale
        self.direction_penalty = direction_penalty
        self.outer_threshold = outer_threshold
        self.trading_loss = ContinuousTradingLoss(**trading_kwargs)

    def _profit_weighted_ce(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Profit-weighted CE with 4-class direction mapping."""
        ce = nn.functional.cross_entropy(logits, y_true, reduction="none")

        # Weight by return magnitude
        weights = (returns.abs() * self.profit_scale).clamp(0.1, 10.0)

        # 4-class direction: {0,1}→neg, {2,3}→pos
        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)
            pred_dir = (y_hat >= 2).float() - (y_hat < 2).float()
            true_dir = (y_true >= 2).float() - (y_true < 2).float()
            wrong_dir = (pred_dir * true_dir) < 0
            weights = weights + wrong_dir.float() * self.direction_penalty

        return (ce * weights).mean()

    def forward(
        self,
        position: torch.Tensor,
        logits: torch.Tensor,
        forward_return: torch.Tensor,
        prev_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        position : Tensor (B, 1)
            Continuous position from PTP.
        logits : Tensor (B, 4)
            Raw class logits for CE loss.
        forward_return : Tensor (B,) or (B, 1)
            Actual forward returns.
        prev_position : Tensor (B, 1), optional
            Previous position for turnover / TC-adjusted Sharpe.

        Returns
        -------
        total_loss : Tensor
        metrics : dict
        """
        fwd = forward_return.view(-1)

        # Derive class labels from returns
        class_labels = returns_to_classes(
            fwd, outer=self.outer_threshold,
        )

        # 1. Classification loss (profit-weighted CE on 4-class logits)
        ce = self._profit_weighted_ce(logits, class_labels, fwd)

        # 2. Trading loss (PnL/Sharpe on continuous position)
        trading, trading_metrics = self.trading_loss(
            position, fwd, prev_position=prev_position,
        )

        total = self.ce_weight * ce + self.pnl_weight * trading

        # Classification metrics
        with torch.no_grad():
            pred_cls = logits.argmax(dim=-1)
            cls_acc = (pred_cls == class_labels).float().mean().item() * 100.0
            # Direction accuracy: classes 0,1 = negative, 2,3 = positive
            pred_dir = (pred_cls >= 2).long() - (pred_cls < 2).long()
            true_dir = (class_labels >= 2).long() - (class_labels < 2).long()
            dir_match = (pred_dir == true_dir).float().mean().item() * 100.0

        metrics = {
            **trading_metrics,
            "ce_loss": ce.item(),
            "trading_loss": trading.item(),
            "cls_accuracy": cls_acc,
            "cls_dir_accuracy": dir_match,
        }
        return total, metrics


# ============================================================================
# PTP Training / Evaluation Loops
# ============================================================================

def _get_bvs_entropy_loss(model: nn.Module) -> torch.Tensor:
    """Extract BVS entropy regularization loss from model (any wrapper depth)."""
    # Unwrap StatefulMMTFv3Core → base_model
    core = getattr(model, "base_model", model)
    selector = getattr(core, "branch_selector", None)
    if selector is not None and hasattr(selector, "get_entropy_loss"):
        return selector.get_entropy_loss()
    return torch.tensor(0.0, device=next(model.parameters()).device)


def train_epoch_v3_ptp(
    model: nn.Module,
    loader,
    ptp_loss: PTPLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
    unpack_fn=None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch with PTP multi-loss (CE + PnL + AE).

    Expects model to return ``(position, ae_losses, logits)`` — i.e.,
    a ``StatefulMMTFv3Core`` with ``quantile_head=True``.
    Pass ``scaler=torch.amp.GradScaler()`` to enable mixed-precision training.
    """
    model.train()
    _maybe_reset_position_state(model)
    use_amp = scaler is not None

    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0
    prev_pos = None  # track position across batches for turnover / TC

    for batch in loader:
        _unpack = unpack_fn or _default_unpack_v3
        inputs, targets = _unpack(batch, device=device)
        targets = targets.float()

        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_amp):
            position, ae_losses, logits = model(**inputs, return_ae_losses=True)

        # Compute loss in fp32 — Sharpe-like ratios underflow in fp16
        position = position.float()
        logits = logits.float()

        # PTP composite loss (CE + trading)
        ptp_total, ptp_metrics = ptp_loss(
            position, logits, targets, prev_position=prev_pos,
        )

        # AE reconstruction
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"].float()

        # BVS entropy regularization (anti-collapse)
        bvs_entropy_loss = _get_bvs_entropy_loss(model)
        loss = ptp_total + ae_loss + bvs_entropy_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        total_loss += loss.item()
        for k, v in ptp_metrics.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v
        metric_accum["ae_recon"] = metric_accum.get("ae_recon", 0.0) + ae_losses["recon_loss"].item()
        if "kl_loss" in ae_losses:
            metric_accum["ae_kl"] = metric_accum.get("ae_kl", 0.0) + ae_losses["kl_loss"].item()
        if bvs_entropy_loss.item() > 0:
            metric_accum["bvs_entropy"] = metric_accum.get("bvs_entropy", 0.0) + bvs_entropy_loss.item()
        n_batches += 1
        prev_pos = position.detach()

    n = max(n_batches, 1)
    return total_loss / n, {k: v / n for k, v in metric_accum.items()}


@torch.no_grad()
def evaluate_v3_ptp(
    model: nn.Module,
    loader,
    ptp_loss: PTPLoss,
    device: torch.device,
    unpack_fn=None,
) -> Dict[str, float]:
    """Evaluate PTP model with classification + trading metrics."""
    model.eval()
    _maybe_reset_position_state(model)

    all_positions, all_returns, all_logits = [], [], []
    metric_accum: Dict[str, float] = {}
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        _unpack = unpack_fn or _default_unpack_v3
        inputs, targets = _unpack(batch, device=device)
        targets = targets.float()

        position, ae_losses, logits = model(**inputs, return_ae_losses=True)

        ptp_total, ptp_metrics = ptp_loss(position, logits, targets)
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"]
        total_loss += (ptp_total + ae_loss).item()
        n_batches += 1
        for k, v in ptp_metrics.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v

        all_positions.append(position.detach().view(-1).cpu())
        all_returns.append(targets.detach().view(-1).cpu())
        all_logits.append(logits.detach().cpu())

    positions = torch.cat(all_positions)
    returns = torch.cat(all_returns)
    all_logits_cat = torch.cat(all_logits)
    strategy_ret = positions * returns

    n = max(n_batches, 1)
    mean_ret = strategy_ret.mean().item()
    std_ret = strategy_ret.std().item() + 1e-8

    gross_profit = strategy_ret[strategy_ret > 0].sum().item()
    gross_loss = strategy_ret[strategy_ret < 0].abs().sum().item() + 1e-8
    cum_ret = strategy_ret.cumsum(dim=0)

    correct_dir = ((positions > 0) & (returns > 0)) | ((positions < 0) & (returns < 0))
    non_flat = positions.abs() > 0.05
    dir_acc = (correct_dir & non_flat).float().sum().item() / max(non_flat.float().sum().item(), 1)

    # Classification metrics
    class_labels = returns_to_classes(
        returns,
        outer=ptp_loss.outer_threshold,
    )
    pred_cls = all_logits_cat.argmax(dim=-1)
    cls_acc = (pred_cls == class_labels).float().mean().item() * 100.0

    # Per-class accuracy
    cls_names = {0: "strong_neg", 1: "weak_neg", 2: "weak_pos", 3: "strong_pos"}
    per_class = {}
    for c in range(PredictionToPosition.N_CLASSES):
        mask = class_labels == c
        if mask.any():
            per_class[f"cls_{c}_acc"] = (pred_cls[mask] == c).float().mean().item() * 100.0
            per_class[f"cls_{c}_count"] = int(mask.sum().item())

    avg_exposure = positions.abs().mean().item()
    raw_sharpe = mean_ret / std_ret

    # Exposure-adjusted Sharpe: penalize low participation.
    # Below min_exposure (0.15) the score is heavily discounted so
    # "do-nothing" models can never win Optuna selection.
    _min_exp = 0.15
    if avg_exposure < _min_exp:
        exposure_penalty = (avg_exposure / _min_exp) ** 2   # quadratic ramp
    else:
        exposure_penalty = 1.0
    adj_sharpe = raw_sharpe * exposure_penalty

    metrics = {
        "loss": total_loss / n,
        "sharpe": adj_sharpe,
        "sharpe_raw": raw_sharpe,
        "sortino": mean_ret / (strategy_ret.clamp(max=0.0).pow(2).mean().sqrt().item() + 1e-8),
        "mean_strategy_ret": mean_ret,
        "win_rate": (strategy_ret > 0).float().mean().item() * 100.0,
        "dir_accuracy": dir_acc * 100.0,
        "profit_factor": gross_profit / gross_loss,
        "max_drawdown": (cum_ret.cummax(dim=0)[0] - cum_ret).max().item(),
        "avg_exposure": avg_exposure,
        "avg_position": positions.mean().item(),
        "cls_accuracy": cls_acc,
        **per_class,
        "n_samples": len(positions),
    }
    metrics.update({k: v / n for k, v in metric_accum.items()})
    return metrics


def build_ptp_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    quantile_lr_scale: float = 1.0,
    ptp_lr_scale: float = 1.0,
) -> list[dict]:
    """Build optimizer groups for trunk, quantile logits head, and PTP mapper."""
    core_model = getattr(model, "base_model", model)
    head = getattr(core_model, "head", None)

    if not isinstance(head, QuantilePositionHead):
        return [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
            "group_name": "trunk",
        }]

    quantile_params = list(head.quantile_parameters())
    ptp_params = list(head.ptp_parameters())
    tracked = {id(p) for p in quantile_params + ptp_params}
    trunk_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in tracked
    ]

    param_groups = []
    if trunk_params:
        param_groups.append({
            "params": trunk_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "group_name": "trunk",
        })
    if quantile_params:
        param_groups.append({
            "params": quantile_params,
            "lr": base_lr * quantile_lr_scale,
            "weight_decay": weight_decay,
            "group_name": "quantile_head",
        })
    if ptp_params:
        param_groups.append({
            "params": ptp_params,
            "lr": base_lr * ptp_lr_scale,
            "weight_decay": weight_decay,
            "group_name": "ptp_head",
        })
    return param_groups


class HeadAwarePTPScheduler:
    """Adaptive LR control for trunk, quantile logits head, and PTP mapper.

    The trunk follows a cosine decay. The quantile and PTP groups are reduced
    independently when their validation objective plateaus or starts to overfit,
    and they can recover gradually after fresh improvement.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        trunk_group: str = "trunk",
        quantile_group: str = "quantile_head",
        ptp_group: str = "ptp_head",
        trunk_min_scale: float = 0.2,
        head_decay: float = 0.6,
        head_recovery: float = 1.05,
        min_head_scale: float = 0.1,
        quantile_patience: int = 2,
        ptp_patience: int = 2,
        improvement_delta: float = 1e-4,
        overfit_tolerance: float = 0.03,
        ptp_downside_weight: float = 1.0,
    ):
        self.optimizer = optimizer
        self.total_epochs = max(total_epochs, 1)
        self.trunk_group = trunk_group
        self.quantile_group = quantile_group
        self.ptp_group = ptp_group
        self.trunk_min_scale = trunk_min_scale
        self.head_decay = head_decay
        self.head_recovery = head_recovery
        self.min_head_scale = min_head_scale
        self.improvement_delta = improvement_delta
        self.overfit_tolerance = overfit_tolerance
        self.ptp_downside_weight = ptp_downside_weight

        self.group_map = {
            group.get("group_name", f"group_{idx}"): group
            for idx, group in enumerate(self.optimizer.param_groups)
        }
        self.initial_lrs = {
            name: group["lr"] for name, group in self.group_map.items()
        }
        self.head_state = {
            self.quantile_group: {
                "best_train": float("inf"),
                "best_val": float("inf"),
                "bad_epochs": 0,
                "patience": quantile_patience,
            },
            self.ptp_group: {
                "best_train": float("inf"),
                "best_val": float("inf"),
                "bad_epochs": 0,
                "patience": ptp_patience,
            },
        }
        self.last_actions: Dict[str, str] = {}

    def _set_group_lr(self, group_name: str, new_lr: float) -> None:
        group = self.group_map.get(group_name)
        if group is None:
            return
        min_lr = self.initial_lrs[group_name] * self.min_head_scale
        max_lr = self.initial_lrs[group_name]
        group["lr"] = min(max(new_lr, min_lr), max_lr)

    def _apply_trunk_lr(self, epoch: int) -> None:
        group = self.group_map.get(self.trunk_group)
        if group is None:
            return
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * min(epoch, self.total_epochs - 1) / self.total_epochs)
        )
        scale = self.trunk_min_scale + (1.0 - self.trunk_min_scale) * cosine
        group["lr"] = self.initial_lrs[self.trunk_group] * scale

    def _update_head(
        self,
        group_name: str,
        train_value: Optional[float],
        val_value: Optional[float],
    ) -> str:
        if group_name not in self.group_map or train_value is None or val_value is None:
            return "missing"

        state = self.head_state[group_name]
        improved = val_value < (state["best_val"] - self.improvement_delta)
        overfit = (
            train_value < (state["best_train"] - self.improvement_delta)
            and val_value > state["best_val"] * (1.0 + self.overfit_tolerance)
        )

        if improved:
            state["best_val"] = val_value
            state["best_train"] = min(state["best_train"], train_value)
            state["bad_epochs"] = 0
            current_lr = self.group_map[group_name]["lr"]
            self._set_group_lr(group_name, current_lr * self.head_recovery)
            return "improved"

        state["best_train"] = min(state["best_train"], train_value)
        state["bad_epochs"] += 2 if overfit else 1
        if state["bad_epochs"] >= state["patience"]:
            current_lr = self.group_map[group_name]["lr"]
            self._set_group_lr(group_name, current_lr * self.head_decay)
            state["bad_epochs"] = 0
            return "reduced_overfit" if overfit else "reduced_plateau"
        return "overfit_watch" if overfit else "plateau_watch"

    def step(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, float]:
        self._apply_trunk_lr(epoch)

        quantile_action = self._update_head(
            self.quantile_group,
            metrics.get("train_ce_loss"),
            metrics.get("val_ce_loss"),
        )
        ptp_train = None
        ptp_val = None
        if metrics.get("train_trading_loss") is not None:
            ptp_train = metrics["train_trading_loss"] + self.ptp_downside_weight * metrics.get(
                "train_downside_vol",
                0.0,
            )
        if metrics.get("val_trading_loss") is not None:
            ptp_val = metrics["val_trading_loss"] + self.ptp_downside_weight * metrics.get(
                "val_downside_vol",
                0.0,
            )
        ptp_action = self._update_head(self.ptp_group, ptp_train, ptp_val)

        self.last_actions = {
            self.quantile_group: quantile_action,
            self.ptp_group: ptp_action,
        }
        return self.get_lrs()

    def get_lrs(self) -> Dict[str, float]:
        return {
            name: group["lr"]
            for name, group in self.group_map.items()
        }


