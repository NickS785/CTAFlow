"""
Autoencoder-Conditioned MMTF
=============================

Replaces hand-crafted trend/vol features with a LEARNED regime
representation from a temporal autoencoder. The autoencoder compresses
a window of recent price/feature data into a compact latent vector,
which then conditions the MMTF model exactly where static covariates
normally would (branch selection, enrichment, daily bias).

Why this works:
- Hand-crafted features (returns, slopes, rvol) capture KNOWN regime
  dimensions but miss complex patterns (e.g., pre-breakout compression,
  correlated momentum-vol shifts, seasonal regime transitions)
- The autoencoder discovers regime structure from data, learning to
  compress the most SALIENT state information into the bottleneck
- The reconstruction objective forces the latent to capture enough
  information to explain the input window — not just what's useful
  for the downstream task, which acts as a regularizer

Architecture options:
  1. Deterministic AE  — simpler, latent = encoder(window)
  2. Variational AE    — KL-regularized latent, structured space
  3. VQ-VAE            — discrete regime codes (finite regime vocabulary)

We implement all three; the VAE is recommended for most cases because
the KL regularization prevents latent collapse and gives you a
meaningful continuous regime space.

Training:
  - Reconstruction loss (MSE) is added as an AUXILIARY loss
  - Joint training: total_loss = task_loss + recon_weight * recon_loss
  - The autoencoder sees the SAME window the model sees (no leakage)
  - Pretrain option: train AE alone first, then fine-tune jointly

Data flow:
  OHLCV window (B, W, f_ae)
       ↓
  TemporalEncoder (GRU/Conv1D)
       ↓
  z_regime (B, d_latent)  ←─── THIS is the conditioning signal
       ↓                         (replaces trend_features + vol_features)
  TemporalDecoder
       ↓
  reconstruction (B, W, f_ae)  → MSE loss (auxiliary)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTAFlow.models.deep_learning.multi_branch.market_context_models import (
    GatedResidualNetwork,
    GLU,
)
from CTAFlow.models.deep_learning.multi_branch.tft.mmtf_v2_models import (
    BranchVariableSelectionV2,
)
from CTAFlow.models.deep_learning.multi_branch.tft.tft_encoders import (
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
# Temporal Autoencoder Variants
# ============================================================================

class TemporalEncoder(nn.Module):
    """Encode a feature window into a latent vector.

    Uses a GRU (captures sequential dependencies in the window) followed
    by a projection to the latent dimension. The GRU's final hidden state
    is the natural "summary" of the temporal window.

    Parameters
    ----------
    f_input : int
        Input feature dimension per timestep.
    d_hidden : int
        GRU hidden dimension.
    d_latent : int
        Output latent dimension.
    n_layers : int
        Number of GRU layers.
    dropout : float
        Dropout between GRU layers.
    """

    def __init__(
        self,
        f_input: int,
        d_hidden: int = 128,
        d_latent: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=f_input,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_hidden, d_latent),
            nn.LayerNorm(d_latent),
            nn.Tanh(),  # Bound latent to [-1, 1] for stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, W, f_input)
        Returns: (B, d_latent)
        """
        _, h_n = self.gru(x)       # h_n: (n_layers, B, d_hidden)
        h_last = h_n[-1]           # (B, d_hidden) — last layer
        return self.proj(h_last)   # (B, d_latent)


class TemporalDecoder(nn.Module):
    """Decode a latent vector back to a feature window.

    Broadcasts the latent to each timestep, then uses a GRU to
    reconstruct the sequential structure.

    Parameters
    ----------
    d_latent : int
        Latent dimension.
    d_hidden : int
        GRU hidden dimension.
    f_output : int
        Output feature dimension per timestep.
    n_layers : int
        Number of GRU layers.
    dropout : float
        Dropout between GRU layers.
    """

    def __init__(
        self,
        d_latent: int = 64,
        d_hidden: int = 128,
        f_output: int = 6,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_latent,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(d_hidden, f_output)

    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        z : (B, d_latent)
        seq_len : int — window length to reconstruct
        Returns: (B, seq_len, f_output)
        """
        # Broadcast latent to each timestep
        z_seq = z.unsqueeze(1).expand(-1, seq_len, -1)  # (B, W, d_latent)
        out, _ = self.gru(z_seq)                         # (B, W, d_hidden)
        return self.proj(out)                             # (B, W, f_output)


class DeterministicRegimeAE(nn.Module):
    """Standard deterministic temporal autoencoder.

    Encodes → latent → decodes. Simple and effective.
    Good when you have enough data that latent collapse isn't an issue.

    Parameters
    ----------
    f_input : int
        Features per timestep (OHLCV = 5, or add indicators).
    d_hidden : int
        Encoder/decoder GRU hidden size.
    d_latent : int
        Bottleneck dimension. This becomes the regime vector.
        Recommended: 32-64 for regime conditioning.
    n_layers : int
        GRU layers in encoder/decoder.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        f_input: int = 6,
        d_hidden: int = 128,
        d_latent: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent

        self.encoder = TemporalEncoder(
            f_input=f_input,
            d_hidden=d_hidden,
            d_latent=d_latent,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.decoder = TemporalDecoder(
            d_latent=d_latent,
            d_hidden=d_hidden,
            f_output=f_input,
            n_layers=n_layers,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode window to latent. Use this for conditioning."""
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        x : (B, W, f_input)

        Returns
        -------
        z : (B, d_latent) — regime latent vector
        x_recon : (B, W, f_input) — reconstruction
        ae_losses : dict with 'recon_loss'
        """
        z = self.encoder(x)
        x_recon = self.decoder(z, seq_len=x.size(1))
        recon_loss = F.mse_loss(x_recon, x)

        return z, x_recon, {"recon_loss": recon_loss}


class VariationalRegimeAE(nn.Module):
    """Variational autoencoder for regime representation.

    Adds KL regularization to the latent space, which:
    1. Prevents latent collapse (common with deterministic AE + joint training)
    2. Creates a STRUCTURED latent space (nearby points = similar regimes)
    3. Enables sampling for regime-conditional generation/scenario analysis

    The KL term acts as an information bottleneck, forcing the latent to
    capture only the most informative aspects of the market state.

    Parameters
    ----------
    f_input : int
        Features per timestep.
    d_hidden : int
        GRU hidden dimension.
    d_latent : int
        Latent dimension (both mean and logvar are this size).
    n_layers : int
        GRU layers.
    kl_weight : float
        Weight for KL divergence term. Start low (0.001-0.01) and
        increase with KL annealing. Too high → posterior collapse.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        f_input: int = 6,
        d_hidden: int = 128,
        d_latent: int = 64,
        n_layers: int = 2,
        kl_weight: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.kl_weight = kl_weight

        # Encoder outputs 2x latent dim (mean + logvar)
        self.encoder_gru = nn.GRU(
            input_size=f_input,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

        self.decoder = TemporalDecoder(
            d_latent=d_latent,
            d_hidden=d_hidden,
            f_output=f_input,
            n_layers=n_layers,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to mean (no sampling). Use for conditioning at inference."""
        _, h_n = self.encoder_gru(x)
        h_last = h_n[-1]
        return self.fc_mu(h_last)

    def _encode_distributional(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to (mu, logvar) for training."""
        _, h_n = self.encoder_gru(x)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        # Clamp logvar for numerical stability
        logvar = logvar.clamp(-10, 2)
        return mu, logvar

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample z = mu + eps * sigma (reparameterization trick)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic at inference

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        x : (B, W, f_input)

        Returns
        -------
        z : (B, d_latent) — sampled regime latent
        x_recon : (B, W, f_input) — reconstruction
        ae_losses : dict with 'recon_loss', 'kl_loss', 'total_ae_loss'
        """
        mu, logvar = self._encode_distributional(x)
        z = self._reparameterize(mu, logvar)
        x_recon = self.decoder(z, seq_len=x.size(1))

        recon_loss = F.mse_loss(x_recon, x)
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon_loss + self.kl_weight * kl_loss

        return z, x_recon, {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_ae_loss": total,
        }


class TransformerVariationalRegimeAE(nn.Module):
    """Transformer-based variational autoencoder for regime representation.

    Replaces the GRU encoder/decoder with transformer blocks, which can
    capture non-local temporal dependencies in the return window more
    effectively. Uses learned positional encoding and mean-pooling over
    time to produce the latent distribution parameters.

    Parameters
    ----------
    f_input : int
        Features per timestep (e.g., 3 for [return, abs_return, cum_return]).
    d_hidden : int
        Transformer model dimension.
    d_latent : int
        Latent dimension (both mean and logvar are this size).
    n_layers : int
        Number of transformer encoder/decoder layers.
    n_heads : int
        Number of attention heads.
    kl_weight : float
        Weight for KL divergence term.
    max_seq_len : int
        Maximum window length (for positional encoding).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        f_input: int = 3,
        d_hidden: int = 128,
        d_latent: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        kl_weight: float = 0.01,
        max_seq_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.kl_weight = kl_weight

        # --- Encoder ---
        self.enc_input_proj = nn.Linear(f_input, d_hidden)
        self.enc_pos = nn.Parameter(torch.randn(1, max_seq_len, d_hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.enc_norm = nn.LayerNorm(d_hidden)
        self.fc_mu = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

        # --- Decoder ---
        self.dec_latent_proj = nn.Linear(d_latent, d_hidden)
        self.dec_pos = nn.Parameter(torch.randn(1, max_seq_len, d_hidden) * 0.02)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.dec_norm = nn.LayerNorm(d_hidden)
        self.dec_output_proj = nn.Linear(d_hidden, f_input)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to (mu, logvar)."""
        W = x.size(1)
        h = self.enc_input_proj(x) + self.enc_pos[:, :W, :]
        h = self.encoder(h)
        h = self.enc_norm(h)
        # Mean-pool over time dimension
        h_pooled = h.mean(dim=1)  # (B, d_hidden)
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled).clamp(-10, 2)
        return mu, logvar

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent to reconstructed sequence."""
        z_proj = self.dec_latent_proj(z)  # (B, d_hidden)
        # Broadcast to (B, W, d_hidden) and add positional encoding
        z_seq = z_proj.unsqueeze(1).expand(-1, seq_len, -1)
        z_seq = z_seq + self.dec_pos[:, :seq_len, :]
        h = self.decoder(z_seq)
        h = self.dec_norm(h)
        return self.dec_output_proj(h)  # (B, W, f_input)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to mean (no sampling). Use for conditioning at inference."""
        mu, _ = self._encode(x)
        return mu

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        x : (B, W, f_input)

        Returns
        -------
        z : (B, d_latent) — sampled regime latent
        x_recon : (B, W, f_input) — reconstruction
        ae_losses : dict with 'recon_loss', 'kl_loss', 'total_ae_loss'
        """
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        x_recon = self._decode(z, seq_len=x.size(1))

        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.kl_weight * kl_loss

        return z, x_recon, {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_ae_loss": total,
        }


class VQRegimeAE(nn.Module):
    """Vector-Quantized autoencoder for DISCRETE regime codes.

    Instead of a continuous latent space, this learns a finite vocabulary
    of K regime prototypes. The encoder output is snapped to the nearest
    codebook vector, giving you:
    - Discrete regime labels (which codebook entry was selected)
    - Interpretable clusters (each codebook vector = a regime type)
    - Natural regime counting (how often each code is used)

    Ideal when you believe markets have a finite number of distinct states
    (trending-up-low-vol, ranging-high-vol, crash, etc.).

    Parameters
    ----------
    f_input : int
        Features per timestep.
    d_hidden : int
        GRU hidden dimension.
    d_latent : int
        Codebook vector dimension.
    n_codes : int
        Number of regime prototypes (codebook size).
        Recommended: 8-32 for market regimes.
    commitment_cost : float
        VQ commitment loss weight. Default 0.25.
    n_layers : int
        GRU layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        f_input: int = 6,
        d_hidden: int = 128,
        d_latent: int = 64,
        n_codes: int = 16,
        commitment_cost: float = 0.25,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_codes = n_codes
        self.commitment_cost = commitment_cost

        self.encoder = TemporalEncoder(
            f_input=f_input,
            d_hidden=d_hidden,
            d_latent=d_latent,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.decoder = TemporalDecoder(
            d_latent=d_latent,
            d_hidden=d_hidden,
            f_output=f_input,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Codebook: K prototype regime vectors
        self.codebook = nn.Embedding(n_codes, d_latent)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

        # Usage tracking for codebook health monitoring
        self.register_buffer("code_usage", torch.zeros(n_codes))

    def _quantize(
        self, z_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Snap encoder output to nearest codebook vector.

        Returns
        -------
        z_q : quantized vector (B, d_latent) — straight-through gradient
        code_idx : selected codebook indices (B,)
        vq_loss : commitment + codebook loss
        """
        # Distances to codebook vectors: ||z_e - e_k||^2
        # z_e: (B, d_latent), codebook: (K, d_latent)
        dists = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=-1)
            - 2 * z_e @ self.codebook.weight.T
        )  # (B, K)

        code_idx = dists.argmin(dim=-1)            # (B,)
        z_q = self.codebook(code_idx)               # (B, d_latent)

        # Track usage
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(code_idx, self.n_codes).float()
                self.code_usage = 0.99 * self.code_usage + 0.01 * onehot.sum(dim=0)

        # VQ losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)      # Move encoder toward codes
        commitment_loss = F.mse_loss(z_q, z_e.detach())     # Move codes toward encoder
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through: gradient flows through z_q as if it were z_e
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, code_idx, vq_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and quantize. Use for conditioning."""
        z_e = self.encoder(x)
        z_q_st, _, _ = self._quantize(z_e)
        return z_q_st

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns
        -------
        z_q : (B, d_latent) — quantized regime vector
        x_recon : (B, W, f_input) — reconstruction
        ae_losses : dict with 'recon_loss', 'vq_loss', 'total_ae_loss',
                    'code_indices', 'codebook_usage'
        """
        z_e = self.encoder(x)
        z_q, code_idx, vq_loss = self._quantize(z_e)
        x_recon = self.decoder(z_q, seq_len=x.size(1))

        recon_loss = F.mse_loss(x_recon, x)
        total = recon_loss + vq_loss

        return z_q, x_recon, {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "total_ae_loss": total,
            "code_indices": code_idx.detach(),
            "codebook_usage": self.code_usage.detach(),
        }

    def get_regime_label(self, x: torch.Tensor) -> torch.Tensor:
        """Get discrete regime label for each sample.

        Useful for regime analysis: cluster trades by regime code
        and analyze performance per regime.
        """
        z_e = self.encoder(x)
        dists = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=-1)
            - 2 * z_e @ self.codebook.weight.T
        )
        return dists.argmin(dim=-1)


# ============================================================================
# Autoencoder-Conditioned Encoder (replaces RegimeConditionedEncoder)
# ============================================================================

class AutoencoderConditionedEncoder(nn.Module):
    """Fuses identity embeddings with autoencoder regime latent.

    Same interface as RegimeConditionedEncoder but the continuous regime
    signal comes from a learned autoencoder rather than hand-crafted
    trend/vol features.

    Variable selection over 4 inputs:
      [ticker_emb, class_emb, subclass_emb, z_regime]

    Parameters
    ----------
    d_model : int
        Output dimension for context vectors.
    d_latent : int
        Autoencoder latent dimension.
    n_tickers : int
        Number of tickers.
    n_asset_classes : int
        Number of asset classes.
    n_asset_subclasses : int
        Number of asset subclasses.
    d_emb : int
        Internal embedding dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_latent: int = 64,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        d_emb: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb

        # Identity embeddings
        self.ticker_emb = nn.Embedding(n_tickers, d_emb)
        self.class_emb = nn.Embedding(n_asset_classes, d_emb)
        self.subclass_emb = nn.Embedding(n_asset_subclasses, d_emb)

        # Project AE latent into embedding space
        self.regime_proj = nn.Sequential(
            nn.Linear(d_latent, d_emb),
            nn.LayerNorm(d_emb),
            nn.GELU(),
        )

        # Variable selection: identity (3) + regime (1)
        n_vars = 4
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model=d_emb, dropout=dropout)
            for _ in range(n_vars)
        ])
        self.var_selection = nn.Sequential(
            nn.Linear(n_vars * d_emb, n_vars),
            nn.Softmax(dim=-1),
        )

        # 4 context encoders
        self.encoder_cs = GatedResidualNetwork(
            d_model=d_model, d_input=d_emb, dropout=dropout,
        )
        self.encoder_ce = GatedResidualNetwork(
            d_model=d_model, d_input=d_emb, dropout=dropout,
        )
        self.encoder_cc = GatedResidualNetwork(
            d_model=d_model, d_input=d_emb, dropout=dropout,
        )
        self.encoder_ch = GatedResidualNetwork(
            d_model=d_model, d_input=d_emb, dropout=dropout,
        )

        self.last_var_weights: Optional[torch.Tensor] = None
        self.var_names = ["ticker", "asset_class", "asset_subclass", "regime"]

        self._init_weights()

    def _init_weights(self):
        for emb in [self.ticker_emb, self.class_emb, self.subclass_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ticker_id: torch.Tensor,
        asset_class_id: torch.Tensor,
        asset_subclass_id: torch.Tensor,
        z_regime: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        ticker_id, asset_class_id, asset_subclass_id : Tensor (B,) long
        z_regime : Tensor (B, d_latent) float
            Autoencoder latent vector.

        Returns
        -------
        c_s, c_e, c_c, c_h : each Tensor (B, d_model)
        """
        embs = [
            self.ticker_emb(ticker_id),
            self.class_emb(asset_class_id),
            self.subclass_emb(asset_subclass_id),
            self.regime_proj(z_regime),
        ]

        processed = [grn(e) for grn, e in zip(self.var_grns, embs)]
        flat = torch.cat(embs, dim=-1)
        weights = self.var_selection(flat)
        self.last_var_weights = weights.detach()

        stacked = torch.stack(processed, dim=1)
        z_fused = torch.einsum("bn,bnd->bd", weights, stacked)

        c_s = self.encoder_cs(z_fused)
        c_e = self.encoder_ce(z_fused)
        c_c = self.encoder_cc(z_fused)
        c_h = self.encoder_ch(z_fused)

        return c_s, c_e, c_c, c_h


# ============================================================================
# MMTF with Autoencoder Conditioning
# ============================================================================

class MMTFAutoEncoderCore(nn.Module):
    """Multi-Modal Temporal Fusion with Autoencoder Regime Conditioning.

    The autoencoder is an INTERNAL component — it receives the AE input
    window in the forward pass, produces the latent, and the latent
    conditions the MMTF pipeline. Reconstruction loss is returned
    alongside the main logits for joint training.

    Parameters
    ----------
    f_sum, f_profile, f_raster, f_seq : int
        Multi-modal feature dimensions.
    f_ae : int
        Autoencoder input features per timestep.
        Typically OHLCV (5) or OHLCV + returns + vol (8-12).
    ae_type : str
        'deterministic', 'vae', or 'vqvae'.
    d_latent : int
        Autoencoder bottleneck dimension.
    d_ae_hidden : int
        Autoencoder GRU hidden size.
    ae_n_layers : int
        Autoencoder GRU layers.
    n_codes : int
        VQ-VAE codebook size (only used if ae_type='vqvae').
    kl_weight : float
        VAE KL weight (only used if ae_type='vae').
    recon_weight : float
        Weight of reconstruction loss in total loss.
    n_tickers, n_asset_classes, n_asset_subclasses : int
        Identity dimensions.
    n_event_types, max_events_per_day : int
        Event dimensions.
    d_model : int
        Main model dimension.
    d_static_emb : int
        Embedding dimension.
    d_calendar, d_event_emb : int
        Temporal encoder dimensions.
    backbone : str
        'transformer' or 'mamba'.
    n_heads, n_layers : int
        Backbone config.
    d_ff : int
        Transformer FFN dim.
    d_state, d_conv, expand : int
        Mamba config.
    task : str
        'classification' or 'regression'.
    num_classes : int
        Output classes.
    dropout : float
        General dropout.
    grn_dropout : float, optional
        GRN/BVS dropout.
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        f_ae: int = 6,
        ae_type: Literal["deterministic", "vae", "transformer_vae", "vqvae"] = "vae",
        d_latent: int = 64,
        d_ae_hidden: int = 128,
        ae_n_layers: int = 2,
        ae_n_heads: int = 4,
        n_codes: int = 16,
        kl_weight: float = 0.01,
        recon_weight: float = 0.1,
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
        # --- Anti-collapse BVS + windowed branch params ---
        sum_lstm_hidden: int = 64,
        bvs_temperature: float = 1.5,
        bvs_entropy_weight: float = 0.1,
        bvs_min_weight: float = 0.05,
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.backbone_type = backbone
        self.ae_type = ae_type
        self.recon_weight = recon_weight
        self.sum_lstm_hidden = sum_lstm_hidden

        _grn_drop = grn_dropout if grn_dropout is not None else dropout

        # ================================================================
        # AUTOENCODER
        # ================================================================
        if ae_type == "deterministic":
            self.autoencoder = DeterministicRegimeAE(
                f_input=f_ae,
                d_hidden=d_ae_hidden,
                d_latent=d_latent,
                n_layers=ae_n_layers,
                dropout=dropout,
            )
        elif ae_type == "vae":
            self.autoencoder = VariationalRegimeAE(
                f_input=f_ae,
                d_hidden=d_ae_hidden,
                d_latent=d_latent,
                n_layers=ae_n_layers,
                kl_weight=kl_weight,
                dropout=dropout,
            )
        elif ae_type == "transformer_vae":
            self.autoencoder = TransformerVariationalRegimeAE(
                f_input=f_ae,
                d_hidden=d_ae_hidden,
                d_latent=d_latent,
                n_layers=ae_n_layers,
                n_heads=ae_n_heads,
                kl_weight=kl_weight,
                dropout=dropout,
            )
        elif ae_type == "vqvae":
            self.autoencoder = VQRegimeAE(
                f_input=f_ae,
                d_hidden=d_ae_hidden,
                d_latent=d_latent,
                n_codes=n_codes,
                n_layers=ae_n_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown ae_type: {ae_type}")

        # ================================================================
        # AUTOENCODER-CONDITIONED ENCODER
        # ================================================================
        self.regime_encoder = AutoencoderConditionedEncoder(
            d_model=d_model,
            d_latent=d_latent,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            d_emb=d_static_emb,
            dropout=_grn_drop,
        )

        self.regime_daily_proj = GatedResidualNetwork(
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
        # HEAD
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
        # Multi-modal market data
        summary_days: torch.Tensor,
        profile_days: torch.Tensor,
        raster_recent: torch.Tensor,
        seq_recent: torch.Tensor,
        seq_lens_recent: torch.Tensor,
        # Identity
        ticker_id: torch.Tensor,
        asset_class_id: torch.Tensor,
        asset_subclass_id: torch.Tensor,
        # Autoencoder input window
        ae_input: torch.Tensor,           # (B, W, f_ae) — OHLCV window
        # Known temporal
        month: torch.Tensor,
        dow: torch.Tensor,
        doy_sin: torch.Tensor,
        doy_cos: torch.Tensor,
        event_type_ids: Optional[torch.Tensor] = None,
        days_until_event: Optional[torch.Tensor] = None,
        # Compatibility
        macro_days: Optional[torch.Tensor] = None,
        event_outcomes: Optional[torch.Tensor] = None,
        event_mask: Optional[torch.Tensor] = None,
        trend_features: Optional[torch.Tensor] = None,
        vol_features: Optional[torch.Tensor] = None,
        # Control
        return_probs: bool = False,
        return_tracker: bool = False,
        return_ae_losses: bool = True,
    ):
        """
        Parameters
        ----------
        ae_input : Tensor (B, W, f_ae)
            Window of price/feature data for the autoencoder.
            Should be the SAME window the model sees (no leakage).
            Typically: normalized OHLCV or OHLCV + returns + vol indicators.

        Returns
        -------
        logits : Tensor (B, num_classes) or (B, 1)
        ae_losses : dict, optional (if return_ae_losses=True)
            Contains reconstruction loss + AE-specific losses.
        tracker : dict, optional (if return_tracker=True)
        """
        b, w = summary_days.shape[0:2]
        bw = b * w

        # ============================================================
        # PHASE 0: AUTOENCODER → REGIME LATENT
        # ============================================================
        z_regime, x_recon, ae_losses = self.autoencoder(ae_input)

        # ============================================================
        # PHASE 1: REGIME-CONDITIONED CONTEXT
        # ============================================================
        c_s, c_e, c_c, c_h = self.regime_encoder(
            ticker_id=ticker_id,
            asset_class_id=asset_class_id,
            asset_subclass_id=asset_subclass_id,
            z_regime=z_regime,
        )

        z_regime_daily = self.regime_daily_proj(c_h)
        z_regime_daily = z_regime_daily.unsqueeze(1).expand(b, w, -1)

        # ============================================================
        # PHASE 2: KNOWN FUTURE
        # ============================================================
        z_known = self.known_encoder(
            month=month, dow=dow, doy_sin=doy_sin, doy_cos=doy_cos,
            event_type_ids=event_type_ids,
            days_until_event=days_until_event,
        )

        # ============================================================
        # PHASE 3: MULTI-MODAL ENCODING
        # ============================================================
        z_summary_seq = self.summary_proj(
            summary_days.reshape(bw, -1)
        ).view(b, w, -1)
        z_profile_seq = self.profile_net(
            profile_days.reshape(bw, profile_days.size(2), -1)
        ).view(b, w, -1)

        # ============================================================
        # PHASE 4: TEMPORAL FUSION
        # ============================================================
        temporal_seq = z_summary_seq + z_known + z_regime_daily
        fused_daily_seq, fused_daily_token = self.fusion_backbone(
            z_profile_seq, temporal_seq,
        )

        # ============================================================
        # PHASE 5: DEDICATED WINDOWED + RECENT BRANCHES
        # ============================================================

        # Branch 2: summary_wind — LSTM over windowed summary features
        h0_sum = self.ch_to_sum_lstm(c_h).unsqueeze(0)
        c0_sum = self.cc_to_sum_lstm(c_c).unsqueeze(0)
        _, (h_sum, _) = self.summary_lstm(z_summary_seq, (h0_sum, c0_sum))
        z_summary_wind = self.sum_lstm_proj(h_sum[-1])

        # Branch 3: spatial — fused profile + raster (most recent)
        z_raster = self.raster_net(raster_recent)
        z_prof_recent = z_profile_seq[:, -1, :]
        z_spatial = self.spatial_fuse(z_prof_recent, z_raster)

        # Branch 4: sequential — intraday sequence
        z_seq = self.seq_net(seq_recent, lengths=seq_lens_recent)

        # Branch 5: spatial_wind — attention over daily profile evolution
        query = z_prof_recent.unsqueeze(1)
        attn_out_sw, _ = self.spatial_wind_attn(
            query=query,
            key=z_profile_seq,
            value=z_profile_seq,
        )
        z_spatial_wind = self.spatial_wind_norm(
            attn_out_sw.squeeze(1) + z_prof_recent
        )
        z_spatial_wind = self.spatial_wind_proj(z_spatial_wind)

        # ============================================================
        # PHASE 6: ANTI-COLLAPSE BRANCH SELECTION
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
        # PHASE 6: PREDICTION
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
            "regime_var_weights": {
                name: self.regime_encoder.last_var_weights[:, i].mean().item()
                for i, name in enumerate(self.regime_encoder.var_names)
            } if self.regime_encoder.last_var_weights is not None else None,
            "temporal_attn": attn_weights.squeeze(1),
            "backbone": dict(self.fusion_backbone.last_tracker),
            "ae_losses": {k: v.item() if torch.is_tensor(v) else v
                         for k, v in ae_losses.items()
                         if k != "code_indices" and k != "codebook_usage"},
        }

        # Build return values
        results = [logits]
        if return_ae_losses:
            results.append(ae_losses)
        if return_tracker:
            results.append(self._last_tracker)

        return results[0] if len(results) == 1 else tuple(results)

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)

    def get_regime_latent(self, ae_input: torch.Tensor) -> torch.Tensor:
        """Extract regime latent without full forward pass.

        Useful for regime analysis, clustering, and visualization.
        """
        return self.autoencoder.encode(ae_input)

    def get_entropy_loss(self) -> torch.Tensor:
        """Get BVS entropy regularization loss.

        Add this to your training loss:
            total_loss = task_loss + recon_loss + model.get_entropy_loss()
        """
        return self.branch_selector.get_entropy_loss()


# ============================================================================
# Convenience Variants
# ============================================================================

class MMTFAutoEncoderTransformer(MMTFAutoEncoderCore):
    """Transformer-backed MMTF with autoencoder conditioning."""
    def __init__(self, **kwargs):
        kwargs["backbone"] = "transformer"
        super().__init__(**kwargs)


class MMTFAutoEncoderMamba(MMTFAutoEncoderCore):
    """Mamba-backed MMTF with autoencoder conditioning."""
    def __init__(self, **kwargs):
        kwargs["backbone"] = "mamba"
        super().__init__(**kwargs)


# ============================================================================
# Training Helper
# ============================================================================

def train_step_with_ae(
    model: MMTFAutoEncoderCore,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
    recon_weight: Optional[float] = None,
) -> Dict[str, float]:
    """Single training step with joint AE + task loss.

    Parameters
    ----------
    model : MMTFAutoEncoderCore
    batch : dict from DataLoader
    optimizer : optimizer
    criterion : task loss (e.g., RoundTripTradeLoss)
    device : torch device
    clip_grad : gradient clipping norm
    recon_weight : override model's recon_weight

    Returns
    -------
    dict with 'task_loss', 'recon_loss', 'total_loss', etc.
    """
    from CTAFlow.data.datasets.tft import unpack_batch_for_model

    model.train()
    optimizer.zero_grad()

    inputs, targets = unpack_batch_for_model(batch, device=device)
    logits, ae_losses = model(**inputs, return_ae_losses=True)

    # Task loss
    if "raw_returns" in batch:
        task_loss = criterion(logits, targets, returns=batch["raw_returns"].to(device))
    else:
        task_loss = criterion(logits, targets)

    # AE reconstruction loss
    rw = recon_weight if recon_weight is not None else model.recon_weight
    ae_loss = ae_losses["total_ae_loss"]

    # BVS entropy regularization
    entropy_loss = model.get_entropy_loss()

    total_loss = task_loss + rw * ae_loss + entropy_loss
    total_loss.backward()

    if clip_grad > 0:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()

    metrics = {
        "task_loss": task_loss.item(),
        "recon_loss": ae_losses["recon_loss"].item(),
        "entropy_loss": entropy_loss.item() if torch.is_tensor(entropy_loss) else entropy_loss,
        "total_loss": total_loss.item(),
    }
    if "kl_loss" in ae_losses:
        metrics["kl_loss"] = ae_losses["kl_loss"].item()
    if "vq_loss" in ae_losses:
        metrics["vq_loss"] = ae_losses["vq_loss"].item()

    return metrics
