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

  Phase 5: Enrichment + Temporal Attention
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

from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn

# --- Existing CTAFlow components ---
from CTAFlow.models.deep_learning.multi_branch.market_context_models import (
    BranchVariableSelection,
    GatedResidualNetwork,
)
from CTAFlow.models.deep_learning.encoders import (
    IntradayRNN,
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
    VPINRasterEncoder,
)
from CTAFlow.models.deep_learning.training.loss.clf import ContinuousTradingLoss



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
        spatial_fuse_mode: str = "gated",
        spatial_fuse_temp: float = 2.0,
        # Training
        dropout: float = 0.2,
        grn_dropout: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.ae_type = ae_type
        self.recon_weight = recon_weight
        self.backbone_type = backbone

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

        # ==============================================================
        # BRANCH 3: SEQUENTIAL (tabular VPIN buckets, ~1h)
        # ==============================================================
        self.seq_net = IntradayRNN(
            input_dim=f_seq, d_model=d_model, num_layers=1,
            dropout=dropout,
        )

        # ==============================================================
        # BRANCH SELECTION (regime-conditioned via c_s)
        # ==============================================================
        self.branch_selector = BranchVariableSelection(
            n_branches=3,
            d_branch=d_model,
            d_context=d_model,
            dropout=_grn_drop,
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
        numbars_recent: torch.Tensor,       # (B, T=4, 129, 4)
        vpin_raster_recent: torch.Tensor,   # (B, 24, 4, 64)
        # Sequential VPIN (~1h lookback)
        seq_vpin: torch.Tensor,             # (B, vpin_seq_len, f_seq)
        seq_vpin_lens: torch.Tensor,        # (B,)
        # Autoencoder input (long-term window)
        ae_input: torch.Tensor,             # (B, ae_window, f_ae)
        # Identity
        ticker_id: torch.Tensor,            # (B,)
        asset_class_id: torch.Tensor,       # (B,)
        asset_subclass_id: torch.Tensor,    # (B,)
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
        z_numbars = self.numbar_encoder(numbars_recent)
        z_vpin = self.vpin_encoder(vpin_raster_recent)
        z_spatial = self.spatial_fuse(z_numbars, z_vpin)

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

        logits = None
        if isinstance(self.head, QuantilePositionHead):
            position, logits = self.head(z_final)                   # (B,1), (B,5)
        else:
            position = self.head(z_final)                           # (B, 1)

        # ==========================================================
        # TRACKING
        # ==========================================================
        spatial_importance = self.spatial_fuse.get_importance_stats()
        self._last_tracker = {
            "branch_weights": {
                name: branch_weights[:, i].mean().item()
                for i, name in enumerate(["backbone_fused", "spatial", "sequential"])
            },
            "regime_var_weights": {
                name: self.regime_encoder.last_var_weights[:, i].mean().item()
                for i, name in enumerate(self.regime_encoder.var_names)
            } if self.regime_encoder.last_var_weights is not None else None,
            "spatial_fuse": spatial_importance,
            "temporal_attn": attn_weights.detach().squeeze(1),
            "backbone": dict(self.fusion_backbone.last_tracker),
            "ae_losses": {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in ae_losses.items()
                if k not in ("code_indices", "codebook_usage")
            },
            "avg_position": position.detach().mean().item(),
            "avg_abs_position": position.detach().abs().mean().item(),
        }

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
    inner: float = 0.25,
    outer: float = 1.0,
) -> torch.Tensor:
    """Bucket forward returns into 5 conviction classes using σ-based thresholds.

    Classes: 0=strong_neg, 1=weak_neg, 2=neutral, 3=weak_pos, 4=strong_pos

    Parameters
    ----------
    returns : Tensor (B,) or (B, 1)
        Forward returns.
    inner : float
        Inner threshold in σ units for neutral zone boundary. Default 0.25.
    outer : float
        Outer threshold in σ units for strong conviction boundary. Default 1.0.

    Returns
    -------
    labels : Tensor (B,) long in {0, 1, 2, 3, 4}
    """
    r = returns.detach().view(-1)
    sigma = r.std().clamp(min=1e-8)

    labels = torch.full_like(r, 2, dtype=torch.long)          # neutral
    labels[r <= -outer * sigma] = 0                            # strong neg
    labels[(r > -outer * sigma) & (r <= -inner * sigma)] = 1   # weak neg
    labels[(r >= inner * sigma) & (r < outer * sigma)] = 3     # weak pos
    labels[r >= outer * sigma] = 4                             # strong pos
    return labels


class PredictionToPosition(nn.Module):
    """Convert 5-class quantile logits → continuous position ∈ [-1, 1].

    Classes: 0=strong_neg, 1=weak_neg, 2=neutral, 3=weak_pos, 4=strong_pos

    Position = tanh(temperature * sum(p_i * anchor_i))
    where anchors are learnable, initialized to [-1.0, -0.5, 0.0, +0.5, +1.0].
    """

    def __init__(self, temperature: float = 1.5):
        super().__init__()
        self.temperature = temperature
        self.anchors = nn.Parameter(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
        )

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        logits : Tensor (B, 5)

        Returns
        -------
        position : Tensor (B, 1)
        logits : Tensor (B, 5) — passed through for multi-loss
        """
        probs = torch.softmax(logits / self.temperature, dim=-1)    # (B, 5)
        weighted = (probs * self.anchors.unsqueeze(0)).sum(dim=-1)   # (B,)
        position = torch.tanh(weighted).unsqueeze(-1)                # (B, 1)
        return position, logits


class QuantilePositionHead(nn.Module):
    """MLP → 5-class logits → PredictionToPosition → continuous position.

    Replaces the standard Tanh head in MMTFv3Core when quantile_head=True.
    """

    def __init__(self, d_input: int, d_model: int, dropout: float = 0.2, temperature: float = 1.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 5),
        )
        self.ptp = PredictionToPosition(temperature=temperature)

    def forward(self, z_final: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_final : Tensor (B, d_input) — typically 2*d_model

        Returns
        -------
        position : Tensor (B, 1)
        logits : Tensor (B, 5)
        """
        logits = self.mlp(z_final)           # (B, 5)
        position, logits = self.ptp(logits)  # (B, 1), (B, 5)
        return position, logits


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
                dropout=0.2,
                temperature=ptp_temperature,
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
        inputs, targets = unpack_fn(batch, device=device)
        targets = targets.float()

        optimizer.zero_grad()
        position, ae_losses = model(**inputs, return_ae_losses=True)

        trading_loss, metrics = loss_fn(position, targets)
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"]
        loss = trading_loss + ae_loss

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
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch with online ticker state updates."""
    model.train()
    _maybe_reset_position_state(model)

    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        inputs, targets = unpack_fn(batch, device=device)
        targets = targets.float()

        optimizer.zero_grad()
        position, ae_losses = model(**inputs, return_ae_losses=True)

        trading_loss, metrics = loss_fn(position, targets)
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"]
        loss = trading_loss + ae_loss

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
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, {k: v / n for k, v in metric_accum.items()}


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
        inputs, targets = unpack_fn(batch, device=device)
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

    return {
        "loss": total_loss / n,
        "sharpe": mean_ret / std_ret,
        "sortino": mean_ret / (strategy_ret.clamp(max=0.0).pow(2).mean().sqrt().item() + 1e-8),
        "mean_strategy_ret": mean_ret,
        "win_rate": (strategy_ret > 0).float().mean().item() * 100.0,
        "dir_accuracy": dir_acc * 100.0,
        "profit_factor": gross_profit / gross_loss,
        "max_drawdown": (cum_ret.cummax(dim=0)[0] - cum_ret).max().item(),
        "avg_exposure": positions.abs().mean().item(),
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
        inputs, targets = unpack_fn(batch, device=device)
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

    return {
        "loss": total_loss / n,
        "sharpe": mean_ret / std_ret,
        "sortino": mean_ret / (strategy_ret.clamp(max=0.0).pow(2).mean().sqrt().item() + 1e-8),
        "mean_strategy_ret": mean_ret,
        "win_rate": (strategy_ret > 0).float().mean().item() * 100.0,
        "dir_accuracy": dir_acc * 100.0,
        "profit_factor": gross_profit / gross_loss,
        "max_drawdown": (cum_ret.cummax(dim=0)[0] - cum_ret).max().item(),
        "avg_exposure": positions.abs().mean().item(),
        "avg_position": positions.mean().item(),
        "n_samples": len(positions),
    }


def print_v3_diagnostics(tracker: dict, eval_metrics: dict, epoch: int = 0) -> None:
    """Pretty-print backbone, branch, spatial, regime, and trading diagnostics."""
    print(f"\n{'='*60}")
    print(f"  Epoch {epoch} Diagnostics")
    print(f"{'='*60}")

    bw = tracker.get("branch_weights", {})
    if bw:
        max_w = max(bw.values()) if bw else 1
        print("\n  Branch Importance:")
        for name, weight in sorted(bw.items(), key=lambda x: -x[1]):
            bar = "#" * int(weight / max(max_w, 1e-8) * 40)
            print(f"    {name:<16s}: {weight:.4f} {bar}")

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
        print(f"\n  Backbone: {bb}")

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
      1. Profit-weighted CE on 5-class logits (classification quality)
      2. ContinuousTradingLoss on continuous position (PnL/Sharpe)

    The CE uses 5-class direction mapping: classes {0,1} → negative,
    class {2} → neutral, classes {3,4} → positive.

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
    inner_threshold : float
        Inner σ threshold for neutral zone in returns_to_classes. Default 0.25.
    outer_threshold : float
        Outer σ threshold for strong conviction in returns_to_classes. Default 1.0.
    **trading_kwargs
        Passed to ContinuousTradingLoss (tc_cost, direction_weight, etc.).
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        pnl_weight: float = 1.0,
        profit_scale: float = 100.0,
        direction_penalty: float = 1.0,
        inner_threshold: float = 0.25,
        outer_threshold: float = 1.0,
        **trading_kwargs,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.pnl_weight = pnl_weight
        self.profit_scale = profit_scale
        self.direction_penalty = direction_penalty
        self.inner_threshold = inner_threshold
        self.outer_threshold = outer_threshold
        self.trading_loss = ContinuousTradingLoss(**trading_kwargs)

    def _profit_weighted_ce_5class(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Profit-weighted CE with correct 5-class direction mapping."""
        ce = nn.functional.cross_entropy(logits, y_true, reduction="none")

        # Weight by return magnitude
        weights = (returns.abs() * self.profit_scale).clamp(0.1, 10.0)

        # 5-class direction: {0,1}→neg, {2}→neutral, {3,4}→pos
        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)
            pred_dir = (y_hat > 2).float() - (y_hat < 2).float()
            true_dir = (y_true > 2).float() - (y_true < 2).float()
            wrong_dir = (pred_dir * true_dir) < 0
            weights = weights + wrong_dir.float() * self.direction_penalty

        return (ce * weights).mean()

    def forward(
        self,
        position: torch.Tensor,
        logits: torch.Tensor,
        forward_return: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        position : Tensor (B, 1)
            Continuous position from PTP.
        logits : Tensor (B, 5)
            Raw class logits for CE loss.
        forward_return : Tensor (B,) or (B, 1)
            Actual forward returns.

        Returns
        -------
        total_loss : Tensor
        metrics : dict
        """
        fwd = forward_return.view(-1)

        # Derive class labels from returns
        class_labels = returns_to_classes(
            fwd, inner=self.inner_threshold, outer=self.outer_threshold,
        )

        # 1. Classification loss (profit-weighted CE on 5-class logits)
        ce = self._profit_weighted_ce_5class(logits, class_labels, fwd)

        # 2. Trading loss (PnL/Sharpe on continuous position)
        trading, trading_metrics = self.trading_loss(position, fwd)

        total = self.ce_weight * ce + self.pnl_weight * trading

        # Classification metrics
        with torch.no_grad():
            pred_cls = logits.argmax(dim=-1)
            cls_acc = (pred_cls == class_labels).float().mean().item() * 100.0
            # Direction accuracy: classes 0,1 = negative, 3,4 = positive
            pred_dir = (pred_cls > 2).long() - (pred_cls < 2).long()
            true_dir = (class_labels > 2).long() - (class_labels < 2).long()
            directional_mask = true_dir != 0
            if directional_mask.any():
                dir_match = (pred_dir[directional_mask] == true_dir[directional_mask]).float().mean().item() * 100.0
            else:
                dir_match = 0.0

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

def train_epoch_v3_ptp(
    model: nn.Module,
    loader,
    ptp_loss: PTPLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
    unpack_fn=None,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch with PTP multi-loss (CE + PnL + AE).

    Expects model to return ``(position, ae_losses, logits)`` — i.e.,
    a ``StatefulMMTFv3Core`` with ``quantile_head=True``.
    """
    model.train()
    _maybe_reset_position_state(model)

    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        inputs, targets = unpack_fn(batch, device=device)
        targets = targets.float()

        optimizer.zero_grad()
        position, ae_losses, logits = model(**inputs, return_ae_losses=True)

        # PTP composite loss (CE + trading)
        ptp_total, ptp_metrics = ptp_loss(position, logits, targets)

        # AE reconstruction
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"]
        loss = ptp_total + ae_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += loss.item()
        for k, v in ptp_metrics.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v
        metric_accum["ae_recon"] = metric_accum.get("ae_recon", 0.0) + ae_losses["recon_loss"].item()
        if "kl_loss" in ae_losses:
            metric_accum["ae_kl"] = metric_accum.get("ae_kl", 0.0) + ae_losses["kl_loss"].item()
        n_batches += 1

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
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inputs, targets = unpack_fn(batch, device=device)
        targets = targets.float()

        position, ae_losses, logits = model(**inputs, return_ae_losses=True)

        ptp_total, _ = ptp_loss(position, logits, targets)
        ae_loss = model.recon_weight * ae_losses["total_ae_loss"]
        total_loss += (ptp_total + ae_loss).item()
        n_batches += 1

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
        inner=ptp_loss.inner_threshold,
        outer=ptp_loss.outer_threshold,
    )
    pred_cls = all_logits_cat.argmax(dim=-1)
    cls_acc = (pred_cls == class_labels).float().mean().item() * 100.0

    # Per-class accuracy
    per_class = {}
    for c in range(5):
        mask = class_labels == c
        if mask.any():
            per_class[f"cls_{c}_acc"] = (pred_cls[mask] == c).float().mean().item() * 100.0
            per_class[f"cls_{c}_count"] = int(mask.sum().item())

    return {
        "loss": total_loss / n,
        "sharpe": mean_ret / std_ret,
        "sortino": mean_ret / (strategy_ret.clamp(max=0.0).pow(2).mean().sqrt().item() + 1e-8),
        "mean_strategy_ret": mean_ret,
        "win_rate": (strategy_ret > 0).float().mean().item() * 100.0,
        "dir_accuracy": dir_acc * 100.0,
        "profit_factor": gross_profit / gross_loss,
        "max_drawdown": (cum_ret.cummax(dim=0)[0] - cum_ret).max().item(),
        "avg_exposure": positions.abs().mean().item(),
        "avg_position": positions.mean().item(),
        "cls_accuracy": cls_acc,
        **per_class,
        "n_samples": len(positions),
    }


