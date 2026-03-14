"""HybridTCN — Temporal Convolutional Network + Spatial Branch + VAE Regime.

Drop-in replacement for ``MMTFv3Core`` that uses dilated causal convolutions
instead of Transformer/Mamba for the primary temporal stream.  Accepts the
same ``unpack_v3_batch`` dictionary so it works with ``StatefulMMTFv3Core``,
``PTPLoss``, and the existing training loops without modification.

Architecture:
  1. **TCN backbone** — dilated causal convolutions over ``tech_features``
  2. **Spatial branch** — FusedSpatialEncoder *or* NumberBar + VPIN encoders
  3. **Sequential branch** — lightweight Transformer encoder over ``seq_vpin``
  4. **VAE regime encoder** — same AE path as MMTFv3Core for regime context
  5. **Static context** — ticker / asset-class embeddings
  6. **Regime-gated tanh head** — position ∈ [-1, 1] scaled by regime weight
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTAFlow.models.deep_learning.encoders import SpatialFuse
from CTAFlow.models.deep_learning.multi_branch.tft.tft_encoders import (
    FusedSpatialEncoder,
    NumberBarEncoder,
    VPINRasterEncoder,
)


# ============================================================================
# TCN primitives
# ============================================================================

class Chomp1d(nn.Module):
    """Remove right-side padding to enforce strict causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single residual block with dilated causal convolutions."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.act1, self.drop1,
            self.conv2, self.chomp2, self.act2, self.drop2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs else None
        )
        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    """Stack of ``TemporalBlock`` layers with exponentially increasing dilation."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channels[i - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size,
                stride=1, dilation=dilation, padding=padding, dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** len(channels) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, L) → (B, C_out, L)"""
        return self.network(x)


# ============================================================================
# Spatial-Temporal encoder (optional: TCN over per-frame spatial features)
# ============================================================================

class SpatialTemporalEncoder(nn.Module):
    """2D CNN per spatial frame → sequence → TCN → last-step embedding."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        tcn_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.spatial_proj = nn.Linear(32, d_model)
        self.tcn = TCNBackbone(d_model, tcn_channels, kernel_size, dropout)

    def forward(self, spatial_seq: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) → (B, d_out)"""
        B, T, C, H, W = spatial_seq.shape
        x = spatial_seq.view(B * T, C, H, W)
        x = self.spatial_proj(self.spatial_cnn(x))  # (B*T, d_model)
        x = x.view(B, T, -1).transpose(1, 2)       # (B, d_model, T)
        return self.tcn(x)[:, :, -1]                 # (B, d_out)


# ============================================================================
# VAE Regime Encoder (mirrors MMTFv3Core's autoencoder path)
# ============================================================================

class _VAERegimeEncoder(nn.Module):
    """Encode [ret_1d, ret_5d, ret_21d, rv_1d] window → z_regime + AE losses."""

    def __init__(
        self,
        f_ae: int = 4,
        ae_window: int = 21,
        d_latent: int = 32,
        d_hidden: int = 128,
        kl_weight: float = 0.01,
        recon_weight: float = 0.1,
    ):
        super().__init__()
        flat_dim = f_ae * ae_window
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
        )
        self.fc_mu = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, flat_dim),
        )
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.f_ae = f_ae
        self.ae_window = ae_window

    def forward(
        self, ae_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """(B, ae_window, f_ae) → z_regime (B, d_latent), ae_losses dict."""
        B = ae_input.shape[0]
        x_flat = ae_input.reshape(B, -1)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, x_flat)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        total = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        ae_losses = {
            "total_ae_loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
        return z, ae_losses


# ============================================================================
# Sequential branch (lightweight Transformer for VPIN sequence)
# ============================================================================

class _SeqEncoder(nn.Module):
    """Small Transformer encoder over ``seq_vpin``."""

    def __init__(
        self,
        f_seq: int,
        d_model: int = 128,
        n_layers: int = 1,
        n_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Linear(f_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(
        self,
        seq_vpin: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(B, L, f_seq) → (B, d_model)"""
        B, L, _ = seq_vpin.shape
        x = self.proj(seq_vpin)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        # Key padding mask
        if seq_lens is not None:
            pad_mask = torch.arange(L, device=x.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        else:
            pad_mask = None

        out = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)
        # Gather last valid timestep per sample
        if seq_lens is not None:
            idx = (seq_lens - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(-1, -1, self.d_model)
            return out.gather(1, idx).squeeze(1)
        return out[:, -1, :]


# ============================================================================
class RegimeGatedFusion(nn.Module):
    """Use the regime embedding to route sequential vs spatial features."""

    def __init__(self, d_model: int, regime_dim: int, dropout: float = 0.2):
        super().__init__()
        self.regime_gate = nn.Sequential(
            nn.Linear(regime_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        z_spatial: torch.Tensor,
        z_regime: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gates = F.softmax(self.regime_gate(z_regime), dim=-1)
        gate_seq = gates[:, 0:1]
        gate_spatial = gates[:, 1:2]

        z_seq_weighted = z_seq * gate_seq
        z_spatial_weighted = z_spatial * gate_spatial
        z_fused = self.fusion_proj(
            torch.cat([z_seq_weighted, z_spatial_weighted], dim=-1)
        )
        return z_fused, gate_seq, gate_spatial


# ============================================================================
# HybridTCN — main model
# ============================================================================

class HybridTCN(nn.Module):
    """Hybrid TCN + Spatial + Sequential + VAE for trend-following.

    Designed as a drop-in replacement for ``MMTFv3Core``.  Accepts the same
    keyword arguments from ``unpack_v3_batch`` so it plugs directly into
    ``StatefulMMTFv3Core``, ``PTPLoss``, and the v3 training loops.

    Parameters
    ----------
    f_tech : int
        Number of technical features per bar.
    f_seq : int
        Number of sequential VPIN features per bar.
    f_ae : int
        Number of autoencoder input features.
    d_model : int
        Shared embedding dimension.
    tcn_channels : list[int]
        Channel widths for TCN layers (length = number of layers).
    kernel_size : int
        TCN convolution kernel size.
    spatial_encoder : {"separate", "fused"}
        Which spatial encoder to use.
    """

    # Expose recon_weight for StatefulMMTFv3Core compatibility
    recon_weight: float

    def __init__(
        self,
        # Feature dims (from prep.get_dims())
        f_tech: int,
        f_seq: int,
        f_ae: int = 4,
        # Autoencoder
        ae_type: str = "vae",  # kept for interface compat; always VAE
        d_latent: int = 32,
        d_ae_hidden: int = 128,
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
        tcn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        spatial_encoder: Literal["separate", "fused"] = "fused",
        spatial_fuse_mode: str = "gated",
        spatial_fuse_temp: float = 2.0,
        # Sequential encoder
        seq_layers: int = 1,
        seq_nheads: int = 2,
        # Regime gate
        regime_gate: bool = True,
        regime_floor: float = 0.15,
        # Training
        dropout: float = 0.2,
        # Variable Selection Network for tech features
        use_vsn: bool = False,
        # Ignored kwargs for MMTFv3Core interface compat
        **_ignored,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [d_model] * 4

        self.d_model = d_model
        self.recon_weight = recon_weight
        self.regime_gate = regime_gate
        self.regime_floor = regime_floor
        self.spatial_encoder_type = spatial_encoder
        self.use_vsn = use_vsn

        # ── 1. TCN backbone ──────────────────────────────────────────
        if use_vsn:
            from CTAFlow.models.deep_learning.multi_branch.market_context_models import (
                VariableSelectionNetwork,
            )
            self.tech_vsn = VariableSelectionNetwork(
                n_vars=f_tech,
                d_model=tcn_channels[0],
                d_context=d_model,   # conditioned on regime context
                dropout=dropout,
            )
            self.tech_proj = None
        else:
            self.tech_vsn = None
            self.tech_proj = nn.Linear(f_tech, tcn_channels[0])
        self.tcn = TCNBackbone(
            in_channels=tcn_channels[0],
            channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        tcn_out_dim = tcn_channels[-1]

        # ── 2. Spatial branch ────────────────────────────────────────
        if spatial_encoder == "separate":
            self.numbar_encoder = NumberBarEncoder(
                in_channels=numbars_channels, d_model=d_model,
            )
            self.vpin_encoder = VPINRasterEncoder(
                in_channels=vpin_channels, n_bins=vpin_bins,
                n_time=vpin_time, d_model=d_model, dropout=dropout,
            )
            self.spatial_fuse = SpatialFuse(
                d_spatial=d_model, mode=spatial_fuse_mode,
                temperature=spatial_fuse_temp,
            )
            self.fused_spatial_encoder = None
        else:
            fused_ch = numbars_channels + vpin_channels - 1  # typically 7
            self.fused_spatial_encoder = FusedSpatialEncoder(
                in_channels=fused_ch, d_model=d_model,
            )
            self.numbar_encoder = None
            self.vpin_encoder = None
            self.spatial_fuse = None

        # ── 3. Sequential branch (VPIN) ──────────────────────────────
        self.seq_encoder = _SeqEncoder(
            f_seq=f_seq, d_model=d_model,
            n_layers=seq_layers, n_heads=seq_nheads, dropout=dropout,
        )

        # ── 4. VAE regime encoder ────────────────────────────────────
        self.vae = _VAERegimeEncoder(
            f_ae=f_ae, d_latent=d_latent, d_hidden=d_ae_hidden,
            kl_weight=kl_weight, recon_weight=recon_weight,
        )

        # ── 5. Static context (ticker identity) ─────────────────────
        self.ticker_emb = nn.Embedding(max(n_tickers, 1), d_static_emb)
        self.class_emb = nn.Embedding(max(n_asset_classes, 1), d_static_emb)
        self.subclass_emb = nn.Embedding(max(n_asset_subclasses, 1), d_static_emb)
        self.static_proj = nn.Linear(3 * d_static_emb, d_model)

        # ── 6. Branch fusion ─────────────────────────────────────────
        # TCN (tcn_out_dim) + spatial (d_model) + sequential (d_model) + regime (d_latent) + static (d_model)
        self.regime_fusion = RegimeGatedFusion(
            d_model=d_model,
            regime_dim=d_latent,
            dropout=dropout,
        )
        fusion_dim = tcn_out_dim + d_model + d_latent + d_model
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 7. Position head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

        # ── 8. Regime gate ───────────────────────────────────────────
        if regime_gate:
            self.regime_scaler = nn.Sequential(
                nn.Linear(d_latent, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )

        self._last_tracker: Dict = {}

    def forward(
        self,
        # Technical indicators (TCN input)
        tech_features: torch.Tensor,                         # (B, L, f_tech)
        tech_lens: torch.Tensor,                             # (B,)
        # Spatial
        numbars_recent: Optional[torch.Tensor] = None,
        vpin_raster_recent: Optional[torch.Tensor] = None,
        fused_spatial: Optional[torch.Tensor] = None,
        # Sequential VPIN
        seq_vpin: Optional[torch.Tensor] = None,
        seq_vpin_lens: Optional[torch.Tensor] = None,
        # Autoencoder
        ae_input: Optional[torch.Tensor] = None,
        # Identity
        ticker_id: Optional[torch.Tensor] = None,
        asset_class_id: Optional[torch.Tensor] = None,
        asset_subclass_id: Optional[torch.Tensor] = None,
        # Control
        return_tracker: bool = False,
        return_ae_losses: bool = True,
    ):
        B = tech_features.shape[0]
        device = tech_features.device

        # ── 0. VAE regime (computed first so context is available for VSN)
        if ae_input is not None:
            z_regime, ae_losses = self.vae(ae_input)
        else:
            z_regime = torch.zeros(B, self.vae.fc_mu.out_features, device=device)
            ae_losses = {
                "total_ae_loss": torch.tensor(0.0, device=device),
                "recon_loss": torch.tensor(0.0, device=device),
                "kl_loss": torch.tensor(0.0, device=device),
            }

        # ── 1. TCN path ──────────────────────────────────────────────
        vsn_weights = None
        if self.tech_vsn is not None:
            # Project regime to d_model for VSN context
            regime_ctx = self.static_proj(
                torch.cat([
                    self.ticker_emb(
                        ticker_id if ticker_id is not None
                        else torch.zeros(B, dtype=torch.long, device=device)
                    ),
                    self.class_emb(
                        asset_class_id if asset_class_id is not None
                        else torch.zeros(B, dtype=torch.long, device=device)
                    ),
                    self.subclass_emb(
                        asset_subclass_id if asset_subclass_id is not None
                        else torch.zeros(B, dtype=torch.long, device=device)
                    ),
                ], dim=-1)
            )  # (B, d_model) — use static embedding as context
            x_tech, vsn_weights = self.tech_vsn(tech_features, context=regime_ctx)
        else:
            x_tech = self.tech_proj(tech_features)          # (B, L, C)
        x_tech = x_tech.transpose(1, 2)                 # (B, C, L)
        tcn_out = self.tcn(x_tech)                       # (B, C, L)
        # Last valid timestep per sample
        last_idx = (tech_lens - 1).clamp(min=0)
        z_temporal = tcn_out[
            torch.arange(B, device=device), :, last_idx
        ]  # (B, C)

        # ── 2. Spatial path ──────────────────────────────────────────
        if self.spatial_encoder_type == "separate":
            z_nb = self.numbar_encoder(numbars_recent)
            z_vpin = self.vpin_encoder(vpin_raster_recent)
            z_spatial = self.spatial_fuse(z_nb, z_vpin)
        elif fused_spatial is not None:
            z_spatial = self.fused_spatial_encoder(fused_spatial)
        else:
            z_spatial = torch.zeros(B, self.d_model, device=device)

        # ── 3. Sequential path ───────────────────────────────────────
        if seq_vpin is not None:
            z_seq = self.seq_encoder(seq_vpin, seq_vpin_lens)
        else:
            z_seq = torch.zeros(B, self.d_model, device=device)

        # ── 4. Static context ────────────────────────────────────────
        tid = ticker_id if ticker_id is not None else torch.zeros(B, dtype=torch.long, device=device)
        acid = asset_class_id if asset_class_id is not None else torch.zeros(B, dtype=torch.long, device=device)
        asid = asset_subclass_id if asset_subclass_id is not None else torch.zeros(B, dtype=torch.long, device=device)
        z_static = self.static_proj(torch.cat([
            self.ticker_emb(tid),
            self.class_emb(acid),
            self.subclass_emb(asid),
        ], dim=-1))

        # ── 6. Fusion ────────────────────────────────────────────────
        z_branch, gate_seq, gate_spatial = self.regime_fusion(
            z_seq=z_seq,
            z_spatial=z_spatial,
            z_regime=z_regime,
        )
        z_cat = torch.cat([z_temporal, z_branch, z_regime, z_static], dim=-1)
        z_fused = self.fusion(z_cat)

        # ── 7. Position ──────────────────────────────────────────────
        raw_position = self.head(z_fused)  # (B, 1)

        # ── 8. Regime gating ─────────────────────────────────────────
        if self.regime_gate:
            regime_weight = self.regime_scaler(z_regime)  # (B, 1)
            # Floor prevents zeroing out entirely
            regime_weight = regime_weight * (1.0 - self.regime_floor) + self.regime_floor
            position = raw_position * regime_weight
        else:
            position = raw_position
            regime_weight = torch.ones(B, 1, device=device)

        # ── Tracker ──────────────────────────────────────────────────
        self._last_tracker = {
            "avg_position": position.detach().mean().item(),
            "avg_regime_weight": regime_weight.detach().mean().item(),
            "weight_seq_branch": gate_seq.detach().mean().item(),
            "weight_spatial_branch": gate_spatial.detach().mean().item(),
            "tcn_receptive_field": self.tcn.receptive_field,
            "vsn_weights": vsn_weights.detach().mean(dim=1) if vsn_weights is not None else None,
        }

        # Match MMTFv3Core return signature
        if return_tracker and return_ae_losses:
            return position, ae_losses, self._last_tracker
        elif return_ae_losses:
            return position, ae_losses
        elif return_tracker:
            return position, {}, self._last_tracker
        return position, {}
