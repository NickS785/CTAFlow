"""
Regime-Aware Heterogeneous Mixture of Experts for Natural Gas Forecasting
==========================================================================

Dual-target: next-day log-return + next-day realized std.

Expert types:
  - TCN experts — dilated causal convolutions, multi-scale temporal patterns
  - MDN experts — mixture density networks, multimodal/fat-tailed density

Regime conditioning (from HybridTCN's VAE path):
  The VAERegimeEncoder processes a window of regime-defining features:
    [ret_1d, ret_5d, ret_21d, rv_5d, rv_21d,
     pct_in_5y_band, dev_from_5y_mean_zscore, band_width_pct,
     forecast_vs_seasonal_zscore, change_vs_seasonal_zscore,
     is_injection_season, dev_x_season]
  into z_regime → feeds router directly + conditions the shared expert.

Loss:
  L = λ_ret * Huber(return) + λ_vol * MSE(log_std)
    + λ_nll * MDN_NLL + λ_bal * load_balance + λ_ent * entropy + AE_loss
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class MoEConfig:

    # --- Input ---
    n_features: int = 60
    seq_len: int = 20

    # --- VAE Regime Encoder ---
    f_ae: int = 12
    ae_window: int = 21
    d_latent: int = 32
    d_ae_hidden: int = 128
    kl_weight: float = 0.01
    recon_weight: float = 0.1

    # --- Expert pool ---
    n_tcn_experts: int = 3
    n_mdn_experts: int = 3
    top_k: int = 3

    # --- Shared expert (DeepSeekMoE) ---
    shared_expert_dim: int = 64

    # --- TCN expert config ---
    tcn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    tcn_kernel_size: int = 3
    stride: int = 1  # causal downsampling before TCN layers (reduces seq_len by this factor)

    # --- MDN expert config ---
    mdn_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    mdn_n_components: int = 4
    mdn_min_sigma: float = 1e-4

    # --- Router ---
    router_hidden_dim: int = 64
    router_noise_std: float = 0.1

    # --- Training ---
    dropout: float = 0.2
    load_balance_weight: float = 0.01
    entropy_reg_weight: float = 0.01
    nll_weight: float = 0.1
    return_loss_weight: float = 1.0
    vol_loss_weight: float = 1.0

    # --- Grouped Variable Selection ---
    use_vsn: bool = False
    vsn_d_model: int = 32           # output dim per group after GRN
    vsn_temperature: float = 1.5    # softmax temperature (higher = softer)
    vsn_entropy_weight: float = 0.1 # anti-collapse entropy regularization
    vsn_min_weight: float = 0.05    # minimum per-group weight floor
    # feature_group_sizes set at runtime from NGMoEDataBuilder.feature_groups

    # --- Classification / Positioning head ---
    n_classes: int = 0              # 0 = regression only, 4 = quartile classification
    use_positioning_head: bool = False  # tanh exposure head on top of class logits
    positioning_hidden_dim: int = 32
    ce_weight: float = 1.0         # cross-entropy loss weight
    positioning_pnl_weight: float = 1.0  # PnL-based positioning loss weight
    tc_cost: float = 0.0           # transaction cost for positioning loss

    @property
    def n_routed_experts(self) -> int:
        return self.n_tcn_experts + self.n_mdn_experts


# ============================================================================
# 2. TCN PRIMITIVES
# ============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        return self.act(self.net(x) + res)


class CausalStride(nn.Module):
    """Causal average pooling for sequence length reduction.

    Reduces temporal dimension by ``stride`` factor while maintaining
    causality — each output depends only on current + past inputs.
    Applied once before the TCN layers so all subsequent computation
    operates on the shorter sequence.
    """

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
        self.pad = stride - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C, ceil(L / stride))"""
        if self.stride <= 1:
            return x
        x = F.pad(x, (self.pad, 0))  # left-pad for causality
        return F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)


class TCNBackbone(nn.Module):
    def __init__(self, in_channels: int, channels: List[int],
                 kernel_size: int = 3, dropout: float = 0.2,
                 stride: int = 1):
        super().__init__()
        self.stride_layer = CausalStride(stride) if stride > 1 else None
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = in_channels if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride_layer is not None:
            x = self.stride_layer(x)
        return self.network(x)


# ============================================================================
# 3. VAE REGIME ENCODER
# ============================================================================

class VAERegimeEncoder(nn.Module):
    """Encode a window of regime-defining features -> z_regime + AE losses.

    Regime feature vector (per timestep, f_ae channels):
      Returns & Vol:  ret_1d, ret_5d, ret_21d, rv_5d, rv_21d
      Storage state:  pct_in_5y_band, dev_from_5y_mean_zscore, band_width_pct,
                      forecast_vs_seasonal_zscore, change_vs_seasonal_zscore
      Seasonal:       is_injection_season, dev_x_season
    """

    def __init__(self, f_ae: int = 12, ae_window: int = 21, d_latent: int = 32,
                 d_hidden: int = 128, kl_weight: float = 0.01,
                 recon_weight: float = 0.1):
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
        self.d_latent = d_latent

    def forward(self, ae_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = ae_input.shape[0]
        x_flat = ae_input.reshape(B, -1)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.training:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        else:
            z = mu

        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, x_flat)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        total = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        return z, {"total_ae_loss": total, "recon_loss": recon_loss, "kl_loss": kl_loss}


# ============================================================================
# 4. EXPERT DEFINITIONS
# ============================================================================

class ExpertOutput:
    """Standardized output: return + std (+ optional MDN params)."""
    def __init__(self, pred_return: torch.Tensor, pred_std: torch.Tensor,
                 mdn_params: Optional[Tuple[torch.Tensor, ...]] = None):
        self.pred_return = pred_return   # (B,)
        self.pred_std = pred_std         # (B,)
        self.mdn_params = mdn_params     # (pi, mu, sigma) if MDN


class TCNExpert(nn.Module):
    """TCN expert: sequence -> (return, std)."""

    def __init__(self, n_features: int, channels: List[int], kernel_size: int,
                 dropout: float = 0.2, stride: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, channels[0])
        self.tcn = TCNBackbone(channels[0], channels, kernel_size, dropout,
                               stride=stride)
        d = channels[-1]
        self.return_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.std_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1), nn.Softplus())

    def forward(self, x_seq: torch.Tensor) -> ExpertOutput:
        h = self.input_proj(x_seq).transpose(1, 2)   # (B, C, L)
        z = self.tcn(h)[:, :, -1]                     # (B, C)
        return ExpertOutput(
            self.return_head(z).squeeze(-1),
            self.std_head(z).squeeze(-1) + 1e-6,
        )


class MDNExpert(nn.Module):
    """Mixture Density Network expert: flat features -> Gaussian mixture.

    Derives return = E[r] and std = Std[r] analytically from the mixture.
    The NLL on the density provides a richer training signal than MSE alone,
    as it penalises the model for being overconfident (narrow sigma) when
    the realised return falls in the tails.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], n_components: int,
                 min_sigma: float = 1e-4, dropout: float = 0.2):
        super().__init__()
        self.n_components = n_components
        self.min_sigma = min_sigma
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                nn.SiLU(), nn.Dropout(dropout),
            ])
        self.trunk = nn.Sequential(*layers)
        d = hidden_dims[-1]
        self.pi_head = nn.Linear(d, n_components)
        self.mu_head = nn.Linear(d, n_components)
        self.sigma_head = nn.Linear(d, n_components)

        nn.init.zeros_(self.pi_head.bias)
        nn.init.normal_(self.pi_head.weight, std=0.01)
        nn.init.constant_(self.sigma_head.bias, math.log(math.exp(0.5) - 1))

    def forward(self, x_flat: torch.Tensor) -> ExpertOutput:
        h = self.trunk(x_flat)
        pi = torch.softmax(self.pi_head(h), dim=-1)       # (B, K)
        mu = self.mu_head(h)                                # (B, K)
        sigma = F.softplus(self.sigma_head(h)) + self.min_sigma  # (B, K)

        # Analytic mixture moments
        pred_ret = (pi * mu).sum(dim=-1)                    # E[r]
        mixture_var = (pi * (sigma**2 + mu**2)).sum(dim=-1) - pred_ret**2
        pred_std = torch.sqrt(mixture_var.clamp(min=1e-8))  # Std[r]

        return ExpertOutput(pred_ret, pred_std, mdn_params=(pi, mu, sigma))

    def compute_nll(self, pi: torch.Tensor, mu: torch.Tensor,
                    sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood (log-sum-exp stable)."""
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        log_pi = torch.log(pi + 1e-10)
        log_n = (-0.5 * math.log(2 * math.pi)
                 - torch.log(sigma)
                 - 0.5 * ((y - mu) / sigma) ** 2)
        return -torch.logsumexp(log_pi + log_n, dim=-1).mean()


# ============================================================================
# 5. SHARED EXPERT (always-on, regime-conditioned)
# ============================================================================

class SharedExpert(nn.Module):
    """Always-active expert conditioned on z_regime."""

    def __init__(self, input_dim: int, regime_dim: int, hidden_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim + regime_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(hidden_dim, 1)
        self.std_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, x_flat: torch.Tensor, z_regime: torch.Tensor) -> ExpertOutput:
        z = self.trunk(torch.cat([x_flat, z_regime], dim=-1))
        return ExpertOutput(
            self.return_head(z).squeeze(-1),
            self.std_head(z).squeeze(-1) + 1e-6,
        )


# ============================================================================
# 5b. GROUPED VARIABLE SELECTION NETWORK
# ============================================================================

class GroupedFeatureVSN(nn.Module):
    """Variable Selection over logical feature groups, conditioned on z_regime.

    Each group (returns, volatility, momentum, microstructure, temporal,
    storage, weather) is projected through its own GRN into a common d_model
    space. A regime-conditioned selection network then produces per-group
    softmax weights with anti-collapse mechanisms (temperature scaling,
    entropy regularization, minimum weight floor).

    Output: (B, L, d_model) selected representation replacing raw features.

    Parameters
    ----------
    group_sizes : dict  {group_name: n_features_in_group}
        Ordered dict mapping group names to feature counts.
    d_model : int
        Common output dimension per group.
    d_context : int
        Dimension of conditioning context (z_regime).
    dropout : float
    temperature : float
        Softmax temperature for anti-collapse.
    entropy_weight : float
        Weight for entropy regularization loss.
    min_weight : float
        Minimum per-group weight floor.
    """

    def __init__(
        self,
        group_sizes: Dict[str, int],
        d_model: int = 32,
        d_context: int = 32,
        dropout: float = 0.2,
        temperature: float = 1.5,
        entropy_weight: float = 0.1,
        min_weight: float = 0.05,
    ):
        super().__init__()
        self.group_names = list(group_sizes.keys())
        self.group_sizes = list(group_sizes.values())
        self.n_groups = len(self.group_names)
        self.d_model = d_model
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.min_weight = min_weight

        # Per-group GRN: projects group features -> d_model
        self.group_grns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_feat, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            for n_feat in self.group_sizes
        ])

        # Selection network: [concat group embeddings + context] -> group weights
        selection_input_dim = self.n_groups * d_model + d_context
        self.selection_net = nn.Sequential(
            nn.Linear(selection_input_dim, self.n_groups * 2),
            nn.SiLU(),
            nn.Linear(self.n_groups * 2, self.n_groups),
        )

        # Interpretability
        self.last_weights: Optional[torch.Tensor] = None
        self.last_entropy_loss: Optional[torch.Tensor] = None

        # Precompute split indices
        self._split_sizes = self.group_sizes

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, L, F) raw feature sequence
        context : (B, d_context) regime vector

        Returns
        -------
        selected : (B, L, d_model) weighted group combination
        weights : (B, n_groups) selection weights
        """
        B, L, _ = x.shape

        # Split features into groups
        groups = torch.split(x, self._split_sizes, dim=-1)

        # Process each group through its GRN
        processed = []
        for grn, g in zip(self.group_grns, groups):
            processed.append(grn(g))  # (B, L, d_model)

        # Stack for weighted combination: (B, L, n_groups, d_model)
        stacked = torch.stack(processed, dim=2)

        # Selection weights from last-timestep group embeddings + context
        # Use mean-pool over time for more stable selection signal
        group_summaries = torch.stack(
            [p.mean(dim=1) for p in processed], dim=1
        )  # (B, n_groups, d_model)
        flat = group_summaries.reshape(B, -1)  # (B, n_groups * d_model)
        sel_input = torch.cat([flat, context], dim=-1)  # (B, n_groups*d_model + d_ctx)
        logits = self.selection_net(sel_input)  # (B, n_groups)

        # Temperature-scaled softmax
        weights = F.softmax(logits / self.temperature, dim=-1)

        # Minimum weight floor
        if self.min_weight > 0:
            weights = weights.clamp(min=self.min_weight)
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # Entropy regularization loss
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # (B,)
        max_entropy = math.log(self.n_groups)
        norm_entropy = entropy / max_entropy
        entropy_loss = self.entropy_weight * (1.0 - norm_entropy).mean()

        self.last_weights = weights.detach()
        self.last_entropy_loss = entropy_loss.detach()

        # Weighted combination: (B, L, n_groups, d_model) * (B, 1, n_groups, 1)
        w_expanded = weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, n_groups, 1)
        selected = (stacked * w_expanded).sum(dim=2)  # (B, L, d_model)

        return selected, entropy_loss


# ============================================================================
# 5c. CLASSIFICATION + POSITIONING HEADS
# ============================================================================

class ClassificationHead(nn.Module):
    """Maps MoE return/std embeddings to class logits."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, n_classes)


class PositioningHead(nn.Module):
    """Maps class logits -> tanh exposure in [-1, +1].

    Architecture: softmax(logits) -> MLP -> tanh
    The class probabilities encode the return distribution belief;
    the MLP learns the optimal exposure mapping from that belief.
    """

    def __init__(self, n_classes: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_classes, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)  # (B, n_classes)
        return torch.tanh(self.net(probs)).squeeze(-1)  # (B,)


# ============================================================================
# 6. REGIME-CONDITIONED ROUTER
# ============================================================================

class RegimeConditionedRouter(nn.Module):
    """Sparse top-k gating on [z_trunk || z_regime]."""

    def __init__(self, trunk_dim: int, regime_dim: int, n_experts: int,
                 top_k: int, hidden_dim: int = 64, noise_std: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.gate = nn.Sequential(
            nn.Linear(trunk_dim + regime_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, z_trunk: torch.Tensor, z_regime: torch.Tensor):
        x = torch.cat([z_trunk, z_regime], dim=-1)
        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(1, topk_idx, topk_weights)

        # Load-balance loss (Switch Transformer)
        f = (weights > 0).float().mean(dim=0)
        P = F.softmax(logits, dim=-1).mean(dim=0)
        balance_loss = self.n_experts * (f * P).sum()

        return weights, topk_idx, balance_loss


# ============================================================================
# 7. MOE CORE
# ============================================================================

class NatGasMoE(nn.Module):
    """Heterogeneous MoE for NG return + vol forecasting.

    Data flow:
      x_seq (B,L,F)           ae_input (B,ae_win,f_ae)
           |                         |
      SharedTrunk              VAERegimeEncoder
       z_trunk (64)           z_regime (d_latent)
           |                   /           \\
           +--- concat --> Router      SharedExpert
           |             top-k |       (always active)
      [TCN_0  TCN_1  TCN_2  MDN_0  MDN_1  MDN_2]
           |
      Weighted agg  +  shared expert
           |
      (pred_return, pred_std)

    TCN experts process the raw sequence through causal convolutions.
    MDN experts operate on the trunk embedding and output full Gaussian
    mixture densities, providing an auxiliary NLL training signal that
    penalises overconfidence in the tails.
    """

    def __init__(self, config: MoEConfig,
                 feature_group_sizes: Optional[Dict[str, int]] = None):
        super().__init__()
        self.config = config
        trunk_dim = 64

        # --- Grouped Variable Selection (optional) ---
        self.vsn = None
        if config.use_vsn and feature_group_sizes is not None:
            self.vsn = GroupedFeatureVSN(
                group_sizes=feature_group_sizes,
                d_model=config.vsn_d_model,
                d_context=config.d_latent,
                dropout=config.dropout,
                temperature=config.vsn_temperature,
                entropy_weight=config.vsn_entropy_weight,
                min_weight=config.vsn_min_weight,
            )
            # VSN projects features: n_features -> vsn_d_model
            feat_dim = config.vsn_d_model
        else:
            feat_dim = config.n_features

        # Shared trunk
        self.trunk_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(128, trunk_dim),
            nn.LayerNorm(trunk_dim), nn.SiLU(),
        )

        # VAE regime encoder
        self.vae = VAERegimeEncoder(
            f_ae=config.f_ae, ae_window=config.ae_window,
            d_latent=config.d_latent, d_hidden=config.d_ae_hidden,
            kl_weight=config.kl_weight, recon_weight=config.recon_weight,
        )

        # Shared expert (regime-conditioned)
        self.shared_expert = SharedExpert(
            input_dim=trunk_dim, regime_dim=config.d_latent,
            hidden_dim=config.shared_expert_dim, dropout=config.dropout,
        )

        # Router (conditioned on trunk + regime)
        self.router = RegimeConditionedRouter(
            trunk_dim=trunk_dim, regime_dim=config.d_latent,
            n_experts=config.n_routed_experts, top_k=config.top_k,
            hidden_dim=config.router_hidden_dim, noise_std=config.router_noise_std,
        )

        # Routed experts
        self.experts = nn.ModuleList()
        for _ in range(config.n_tcn_experts):
            self.experts.append(TCNExpert(
                feat_dim, config.tcn_channels,
                config.tcn_kernel_size, config.dropout,
                stride=config.stride,
            ))
        for _ in range(config.n_mdn_experts):
            self.experts.append(MDNExpert(
                trunk_dim, config.mdn_hidden_dims,
                config.mdn_n_components, config.mdn_min_sigma, config.dropout,
            ))

        self.expert_types = (
            ["tcn"] * config.n_tcn_experts
            + ["mdn"] * config.n_mdn_experts
        )

        # Learned shared-vs-routed mixing weight
        self.shared_gate = nn.Parameter(torch.tensor(0.5))

        # --- Classification + Positioning heads (optional) ---
        self.class_head = None
        self.pos_head = None
        if config.n_classes > 0:
            # Input: concatenated [pred_return, pred_std, z_regime] -> class logits
            cls_input_dim = 2 + config.d_latent
            self.class_head = ClassificationHead(
                cls_input_dim, config.n_classes, config.dropout,
            )
            if config.use_positioning_head:
                self.pos_head = PositioningHead(
                    config.n_classes, config.positioning_hidden_dim, config.dropout,
                )

    def forward(
        self,
        x_seq: torch.Tensor,       # (B, L, F)
        ae_input: torch.Tensor,     # (B, ae_window, f_ae)
    ) -> Dict[str, torch.Tensor]:

        B = x_seq.shape[0]
        device = x_seq.device

        # 1. VAE regime encoding
        z_regime, ae_losses = self.vae(ae_input)

        # 1b. Grouped variable selection (optional)
        vsn_entropy_loss = torch.tensor(0.0, device=device)
        if self.vsn is not None:
            x_selected, vsn_entropy_loss = self.vsn(x_seq, z_regime)
        else:
            x_selected = x_seq

        # 2. Shared trunk (operates on last timestep)
        z_trunk = self.trunk_encoder(x_selected[:, -1, :])

        # 3. Shared expert (regime-conditioned)
        shared_out = self.shared_expert(z_trunk, z_regime)

        # 4. Route (on [z_trunk || z_regime])
        weights, topk_idx, balance_loss = self.router(z_trunk, z_regime)

        # 5. Execute activated experts, collect MDN NLL
        agg_return = torch.zeros(B, device=device)
        agg_std = torch.zeros(B, device=device)
        mdn_nll_accum = torch.tensor(0.0, device=device)
        n_mdn = 0

        for expert_idx in range(self.config.n_routed_experts):
            w = weights[:, expert_idx]
            mask = w > 0
            if not mask.any():
                continue

            expert = self.experts[expert_idx]
            etype = self.expert_types[expert_idx]

            if etype == "mdn":
                out = expert(z_trunk[mask])
            else:
                out = expert(x_selected[mask])

            w_active = w[mask]
            agg_return[mask] += out.pred_return * w_active
            agg_std[mask] += out.pred_std * w_active

            if etype == "mdn" and out.mdn_params is not None:
                n_mdn += 1

        # 6. Combine shared + routed
        alpha = torch.sigmoid(self.shared_gate)
        pred_return = alpha * shared_out.pred_return + (1 - alpha) * agg_return
        pred_std = alpha * shared_out.pred_std + (1 - alpha) * agg_std

        # Entropy regularization
        pi_avg = weights.mean(dim=0)
        entropy = -(pi_avg * torch.log(pi_avg + 1e-10)).sum()
        entropy_loss = torch.relu(0.5 * math.log(self.config.n_routed_experts) - entropy)

        result = {
            "pred_return": pred_return,
            "pred_std": pred_std,
            "router_weights": weights,
            "balance_loss": balance_loss,
            "entropy_loss": entropy_loss,
            "mdn_nll_loss": mdn_nll_accum / max(n_mdn, 1),
            "ae_losses": ae_losses,
            "shared_gate": alpha,
            "z_regime": z_regime,
            "vsn_entropy_loss": vsn_entropy_loss,
        }
        if self.vsn is not None and self.vsn.last_weights is not None:
            result["vsn_weights"] = self.vsn.last_weights

        # --- Classification head ---
        if self.class_head is not None:
            cls_input = torch.cat([
                pred_return.unsqueeze(-1),
                pred_std.unsqueeze(-1),
                z_regime,
            ], dim=-1)
            result["class_logits"] = self.class_head(cls_input)  # (B, n_classes)

            if self.pos_head is not None:
                result["position"] = self.pos_head(result["class_logits"])  # (B,)

        return result

    @torch.no_grad()
    def compute_mdn_nll(
        self,
        x_seq: torch.Tensor,
        ae_input: torch.Tensor,
        target_return: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MDN NLL across all MDN experts for a batch (for eval/logging)."""
        self.eval()
        z_regime, _ = self.vae(ae_input)
        z_trunk = self.trunk_encoder(x_seq[:, -1, :])
        weights, _, _ = self.router(z_trunk, z_regime)

        total_nll = torch.tensor(0.0)
        n = 0
        for expert_idx in range(self.config.n_routed_experts):
            if self.expert_types[expert_idx] != "mdn":
                continue
            w = weights[:, expert_idx]
            mask = w > 0
            if not mask.any():
                continue
            out = self.experts[expert_idx](z_trunk[mask])
            pi, mu, sigma = out.mdn_params
            nll = self.experts[expert_idx].compute_nll(pi, mu, sigma, target_return[mask])
            total_nll += nll * mask.sum()
            n += mask.sum()

        return total_nll / max(n, 1)


# ============================================================================
# 8. LOSS COMPUTATION
# ============================================================================

class MoELoss(nn.Module):
    """Multi-task loss for TCN + MDN MoE.

    Regression mode:
      L = λ_ret * Huber(return) + λ_vol * MSE(log_std)
        + λ_nll * MDN_NLL + λ_bal * balance + λ_ent * entropy + AE_loss

    Classification mode (n_classes > 0):
      Adds λ_ce * CrossEntropy(class_logits, target_class)

    Positioning mode (use_positioning_head):
      Adds λ_pnl * PositioningPnL  (sharpe-aware PnL from tanh exposure)
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

    def forward(self, model_out, target_return, target_std,
                target_class=None):
        # Return loss (Huber for robustness to storage-surprise spikes)
        return_loss = F.huber_loss(
            model_out["pred_return"], target_return, delta=0.5,
        )

        # Vol loss (log-space for scale invariance across regimes)
        vol_loss = F.mse_loss(
            torch.log(model_out["pred_std"] + 1e-6),
            torch.log(target_std + 1e-6),
        )

        cfg = self.config
        ae = model_out["ae_losses"]

        vsn_entropy = model_out.get("vsn_entropy_loss", torch.tensor(0.0))

        total = (
            cfg.return_loss_weight * return_loss
            + cfg.vol_loss_weight * vol_loss
            + cfg.nll_weight * model_out["mdn_nll_loss"]
            + cfg.load_balance_weight * model_out["balance_loss"]
            + cfg.entropy_reg_weight * model_out["entropy_loss"]
            + ae["total_ae_loss"]
            + vsn_entropy
        )

        losses = {
            "total_loss": total,
            "return_loss": return_loss,
            "vol_loss": vol_loss,
            "mdn_nll_loss": model_out["mdn_nll_loss"],
            "balance_loss": model_out["balance_loss"],
            "entropy_loss": model_out["entropy_loss"],
            "ae_recon_loss": ae["recon_loss"],
            "ae_kl_loss": ae["kl_loss"],
            "vsn_entropy_loss": vsn_entropy,
        }

        # --- Classification loss ---
        if "class_logits" in model_out and target_class is not None:
            ce_loss = F.cross_entropy(model_out["class_logits"], target_class)
            losses["ce_loss"] = ce_loss
            losses["total_loss"] = losses["total_loss"] + cfg.ce_weight * ce_loss

            # Classification accuracy (for logging)
            with torch.no_grad():
                preds = model_out["class_logits"].argmax(dim=-1)
                losses["class_accuracy"] = (preds == target_class).float().mean()

        # --- Positioning PnL loss ---
        if "position" in model_out:
            position = model_out["position"]  # (B,) in [-1, 1]
            strategy_ret = position * target_return

            # Transaction cost penalty (approx position change)
            tc_penalty = cfg.tc_cost * position.abs().mean()

            # Negative Sharpe-like objective
            mean_ret = strategy_ret.mean()
            std_ret = strategy_ret.std().clamp(min=1e-6)
            neg_sharpe = -(mean_ret - tc_penalty) / std_ret

            losses["positioning_loss"] = neg_sharpe
            losses["total_loss"] = losses["total_loss"] + cfg.positioning_pnl_weight * neg_sharpe

            with torch.no_grad():
                losses["mean_position"] = position.mean()
                losses["mean_strategy_ret"] = strategy_ret.mean()

        return losses


# ============================================================================
# 9. HYBRID MIXTURE NETWORK (simplified single TCN+MDN)
# ============================================================================

@dataclass
class HybridConfig:
    """Config for HybridMixtureNetwork — single TCN + MDN with regime AE."""

    # --- Input ---
    n_features: int = 60
    seq_len: int = 20

    # --- VAE Regime Encoder ---
    f_ae: int = 12
    ae_window: int = 21
    d_latent: int = 32
    d_ae_hidden: int = 128
    kl_weight: float = 0.01
    recon_weight: float = 0.1

    # --- TCN ---
    tcn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    tcn_kernel_size: int = 3
    stride: int = 1  # causal downsampling before TCN layers

    # --- MDN ---
    mdn_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    mdn_n_components: int = 4
    mdn_min_sigma: float = 1e-4

    # --- MLP head ---
    head_hidden_dim: int = 128

    # --- Training ---
    dropout: float = 0.2
    nll_weight: float = 0.1
    return_loss_weight: float = 1.0
    vol_loss_weight: float = 1.0

    # --- Grouped Variable Selection ---
    use_vsn: bool = False
    vsn_d_model: int = 32
    vsn_temperature: float = 1.5
    vsn_entropy_weight: float = 0.1
    vsn_min_weight: float = 0.05

    # --- Classification / Positioning head ---
    n_classes: int = 0
    use_positioning_head: bool = False
    positioning_hidden_dim: int = 32
    ce_weight: float = 1.0
    positioning_pnl_weight: float = 1.0
    tc_cost: float = 0.0


class HybridMixtureNetwork(nn.Module):
    """Single TCN + MDN with regime autoencoder and MLP head.

    Simplified alternative to NatGasMoE — no router, no expert pool.

    Data flow:
      x_seq (B,L,F)           ae_input (B,ae_win,f_ae)
           |                         |
      [opt VSN]              VAERegimeEncoder
           |                   z_regime (d_latent)
      TCNBackbone                    |
       z_tcn (C)                     |
           |                         |
           +---------- concat -------+
           |                         |
      MDN(z_tcn, z_regime)     MLP Head(z_tcn, z_regime)
       └─ NLL loss (aux)        └─ (pred_return, pred_std)
       └─ density moments          final prediction
    """

    def __init__(self, config: HybridConfig,
                 feature_group_sizes: Optional[Dict[str, int]] = None):
        super().__init__()
        self.config = config

        # --- Grouped Variable Selection (optional) ---
        self.vsn = None
        if config.use_vsn and feature_group_sizes is not None:
            self.vsn = GroupedFeatureVSN(
                group_sizes=feature_group_sizes,
                d_model=config.vsn_d_model,
                d_context=config.d_latent,
                dropout=config.dropout,
                temperature=config.vsn_temperature,
                entropy_weight=config.vsn_entropy_weight,
                min_weight=config.vsn_min_weight,
            )
            feat_dim = config.vsn_d_model
        else:
            feat_dim = config.n_features

        # --- VAE regime encoder ---
        self.vae = VAERegimeEncoder(
            f_ae=config.f_ae, ae_window=config.ae_window,
            d_latent=config.d_latent, d_hidden=config.d_ae_hidden,
            kl_weight=config.kl_weight, recon_weight=config.recon_weight,
        )

        # --- Input projection + TCN backbone ---
        tcn_in_ch = config.tcn_channels[0]
        self.input_proj = nn.Linear(feat_dim, tcn_in_ch)
        self.tcn = TCNBackbone(
            tcn_in_ch, config.tcn_channels,
            config.tcn_kernel_size, config.dropout,
            stride=config.stride,
        )
        tcn_out_dim = config.tcn_channels[-1]

        # --- MDN (density estimation on TCN features + regime) ---
        mdn_input_dim = tcn_out_dim + config.d_latent
        self.mdn = MDNExpert(
            mdn_input_dim, config.mdn_hidden_dims,
            config.mdn_n_components, config.mdn_min_sigma, config.dropout,
        )

        # --- MLP head: [z_tcn, z_regime] -> (return, std) ---
        head_in = tcn_out_dim + config.d_latent
        self.head = nn.Sequential(
            nn.Linear(head_in, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.return_head = nn.Linear(config.head_hidden_dim, 1)
        self.std_head = nn.Sequential(
            nn.Linear(config.head_hidden_dim, 1), nn.Softplus(),
        )

        # --- Classification + Positioning heads (optional) ---
        self.class_head = None
        self.pos_head = None
        if config.n_classes > 0:
            cls_input_dim = 2 + config.d_latent
            self.class_head = ClassificationHead(
                cls_input_dim, config.n_classes, config.dropout,
            )
            if config.use_positioning_head:
                self.pos_head = PositioningHead(
                    config.n_classes, config.positioning_hidden_dim, config.dropout,
                )

    def forward(
        self,
        x_seq: torch.Tensor,       # (B, L, F)
        ae_input: torch.Tensor,     # (B, ae_window, f_ae)
    ) -> Dict[str, torch.Tensor]:

        device = x_seq.device

        # 1. VAE regime encoding
        z_regime, ae_losses = self.vae(ae_input)

        # 2. Variable selection (optional)
        vsn_entropy_loss = torch.tensor(0.0, device=device)
        if self.vsn is not None:
            x_selected, vsn_entropy_loss = self.vsn(x_seq, z_regime)
        else:
            x_selected = x_seq

        # 3. TCN backbone
        h = self.input_proj(x_selected).transpose(1, 2)  # (B, C, L)
        z_tcn = self.tcn(h)[:, :, -1]                     # (B, C)

        # 4. MDN density estimation (auxiliary)
        mdn_input = torch.cat([z_tcn, z_regime], dim=-1)
        mdn_out = self.mdn(mdn_input)

        # 5. MLP head prediction
        head_input = torch.cat([z_tcn, z_regime], dim=-1)
        h_head = self.head(head_input)
        pred_return = self.return_head(h_head).squeeze(-1)
        pred_std = self.std_head(h_head).squeeze(-1) + 1e-6

        result = {
            "pred_return": pred_return,
            "pred_std": pred_std,
            "mdn_pred_return": mdn_out.pred_return,
            "mdn_pred_std": mdn_out.pred_std,
            "mdn_params": mdn_out.mdn_params,
            "ae_losses": ae_losses,
            "z_regime": z_regime,
            "vsn_entropy_loss": vsn_entropy_loss,
        }
        if self.vsn is not None and self.vsn.last_weights is not None:
            result["vsn_weights"] = self.vsn.last_weights

        # 6. Classification head (optional)
        if self.class_head is not None:
            cls_input = torch.cat([
                pred_return.unsqueeze(-1),
                pred_std.unsqueeze(-1),
                z_regime,
            ], dim=-1)
            result["class_logits"] = self.class_head(cls_input)
            if self.pos_head is not None:
                result["position"] = self.pos_head(result["class_logits"])

        return result


class HybridLoss(nn.Module):
    """Loss for HybridMixtureNetwork.

    L = λ_ret * Huber(return) + λ_vol * MSE(log_std)
      + λ_nll * MDN_NLL + AE_loss + VSN_entropy
      + (optional) CE + positioning PnL
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self._mdn_nll_fn = MDNExpert.__dict__["compute_nll"]

    def forward(self, model_out, target_return, target_std,
                target_class=None):
        cfg = self.config

        # Return loss
        return_loss = F.huber_loss(
            model_out["pred_return"], target_return, delta=0.5,
        )

        # Vol loss (log-space)
        vol_loss = F.mse_loss(
            torch.log(model_out["pred_std"] + 1e-6),
            torch.log(target_std + 1e-6),
        )

        # MDN NLL
        mdn_nll = torch.tensor(0.0, device=target_return.device)
        if model_out["mdn_params"] is not None:
            pi, mu, sigma = model_out["mdn_params"]
            y = target_return
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            log_pi = torch.log(pi + 1e-10)
            log_n = (-0.5 * math.log(2 * math.pi)
                     - torch.log(sigma)
                     - 0.5 * ((y - mu) / sigma) ** 2)
            mdn_nll = -torch.logsumexp(log_pi + log_n, dim=-1).mean()

        ae = model_out["ae_losses"]
        vsn_entropy = model_out.get("vsn_entropy_loss", torch.tensor(0.0))

        total = (
            cfg.return_loss_weight * return_loss
            + cfg.vol_loss_weight * vol_loss
            + cfg.nll_weight * mdn_nll
            + ae["total_ae_loss"]
            + vsn_entropy
        )

        losses = {
            "total_loss": total,
            "return_loss": return_loss,
            "vol_loss": vol_loss,
            "mdn_nll_loss": mdn_nll,
            "ae_recon_loss": ae["recon_loss"],
            "ae_kl_loss": ae["kl_loss"],
            "vsn_entropy_loss": vsn_entropy,
        }

        # Classification loss
        if "class_logits" in model_out and target_class is not None:
            ce_loss = F.cross_entropy(model_out["class_logits"], target_class)
            losses["ce_loss"] = ce_loss
            losses["total_loss"] = losses["total_loss"] + cfg.ce_weight * ce_loss
            with torch.no_grad():
                preds = model_out["class_logits"].argmax(dim=-1)
                losses["class_accuracy"] = (preds == target_class).float().mean()

        # Positioning PnL loss
        if "position" in model_out:
            position = model_out["position"]
            strategy_ret = position * target_return
            tc_penalty = cfg.tc_cost * position.abs().mean()
            mean_ret = strategy_ret.mean()
            std_ret = strategy_ret.std().clamp(min=1e-6)
            neg_sharpe = -(mean_ret - tc_penalty) / std_ret
            losses["positioning_loss"] = neg_sharpe
            losses["total_loss"] = losses["total_loss"] + cfg.positioning_pnl_weight * neg_sharpe
            with torch.no_grad():
                losses["mean_position"] = position.mean()
                losses["mean_strategy_ret"] = strategy_ret.mean()

        return losses


# ============================================================================
# 10. DEMO
# ============================================================================

def run_demo():
    print("=" * 70)
    print("NatGas MoE (TCN + MDN) — Smoke Test")
    print("=" * 70)

    B = 32

    # --- Regression mode ---
    config = MoEConfig(
        n_features=40, seq_len=20,
        f_ae=12, ae_window=21, d_latent=32,
        n_tcn_experts=3, n_mdn_experts=3, top_k=3,
    )
    model = NatGasMoE(config)
    loss_fn = MoELoss(config)

    x_seq = torch.randn(B, config.seq_len, config.n_features)
    ae_input = torch.randn(B, config.ae_window, config.f_ae)
    target_ret = torch.randn(B) * 0.03
    target_std = torch.rand(B) * 0.02 + 0.01

    print("\n[1] MoE Regression forward pass...")
    out = model(x_seq, ae_input)
    print(f"    pred_return:    {out['pred_return'].shape}")
    print(f"    pred_std:       {out['pred_std'].shape}")
    print(f"    shared_gate:    {out['shared_gate'].item():.3f}")

    losses = loss_fn(out, target_ret, target_std)
    for k, v in losses.items():
        print(f"    {k}: {v.item():.6f}")

    # --- MoE Classification + Positioning + VSN mode ---
    print("\n[2] MoE Classification + Positioning + VSN mode...")
    group_sizes = {
        "returns": 12, "volatility": 8, "momentum": 10,
        "temporal": 5, "storage": 5,
    }
    n_feat_total = sum(group_sizes.values())
    cls_config = MoEConfig(
        n_features=n_feat_total, seq_len=20,
        f_ae=12, ae_window=21, d_latent=32,
        n_tcn_experts=3, n_mdn_experts=3, top_k=3,
        n_classes=4, use_positioning_head=True,
        ce_weight=1.0, positioning_pnl_weight=0.5,
        use_vsn=True, vsn_d_model=32,
    )
    cls_model = NatGasMoE(cls_config, feature_group_sizes=group_sizes)
    cls_loss_fn = MoELoss(cls_config)

    x_seq_vsn = torch.randn(B, cls_config.seq_len, n_feat_total)
    out = cls_model(x_seq_vsn, ae_input)
    target_class = torch.randint(0, 4, (B,))

    print(f"    class_logits:   {out['class_logits'].shape}")
    print(f"    position:       {out['position'].shape}")
    print(f"    position range: [{out['position'].min().item():.3f}, {out['position'].max().item():.3f}]")
    print(f"    vsn_weights:    {out['vsn_weights'].shape}  groups={list(group_sizes.keys())}")
    print(f"    vsn avg weights: {out['vsn_weights'].mean(0).tolist()}")

    losses = cls_loss_fn(out, target_ret, target_std, target_class)
    for k, v in losses.items():
        print(f"    {k}: {v.item():.6f}")

    print("\n[3] MoE Backward pass...")
    losses["total_loss"].backward()
    n_params = sum(p.numel() for p in cls_model.parameters())
    n_grad = sum(p.numel() for p in cls_model.parameters() if p.grad is not None)
    print(f"    Total params:   {n_params:,}")
    print(f"    Params w/ grad: {n_grad:,}")

    # =================================================================
    # HybridMixtureNetwork smoke test
    # =================================================================
    print("\n" + "=" * 70)
    print("HybridMixtureNetwork (TCN + MDN + RegimeAE) — Smoke Test")
    print("=" * 70)

    # --- Regression ---
    print("\n[4] Hybrid regression...")
    h_cfg = HybridConfig(n_features=40, seq_len=20)
    h_model = HybridMixtureNetwork(h_cfg)
    h_loss_fn = HybridLoss(h_cfg)

    h_x = torch.randn(B, h_cfg.seq_len, h_cfg.n_features)
    h_ae = torch.randn(B, h_cfg.ae_window, h_cfg.f_ae)

    h_out = h_model(h_x, h_ae)
    print(f"    pred_return:      {h_out['pred_return'].shape}")
    print(f"    pred_std:         {h_out['pred_std'].shape}")
    print(f"    mdn_pred_return:  {h_out['mdn_pred_return'].shape}")

    h_losses = h_loss_fn(h_out, target_ret, target_std)
    for k, v in h_losses.items():
        print(f"    {k}: {v.item():.6f}")

    # --- Classification + VSN ---
    print("\n[5] Hybrid classification + VSN...")
    h_cls_cfg = HybridConfig(
        n_features=n_feat_total, seq_len=20,
        n_classes=4, use_positioning_head=True,
        use_vsn=True, vsn_d_model=32,
    )
    h_cls_model = HybridMixtureNetwork(h_cls_cfg, feature_group_sizes=group_sizes)
    h_cls_loss_fn = HybridLoss(h_cls_cfg)

    h_x2 = torch.randn(B, h_cls_cfg.seq_len, n_feat_total)
    h_out2 = h_cls_model(h_x2, h_ae)

    print(f"    class_logits:   {h_out2['class_logits'].shape}")
    print(f"    position:       {h_out2['position'].shape}")
    print(f"    position range: [{h_out2['position'].min().item():.3f}, {h_out2['position'].max().item():.3f}]")
    print(f"    vsn avg weights: {h_out2['vsn_weights'].mean(0).tolist()}")

    h_losses2 = h_cls_loss_fn(h_out2, target_ret, target_std, target_class)
    for k, v in h_losses2.items():
        print(f"    {k}: {v.item():.6f}")

    print("\n[6] Hybrid backward pass...")
    h_losses2["total_loss"].backward()
    n_params = sum(p.numel() for p in h_cls_model.parameters())
    n_grad = sum(p.numel() for p in h_cls_model.parameters() if p.grad is not None)
    print(f"    Total params:   {n_params:,}")
    print(f"    Params w/ grad: {n_grad:,}")

    print("\n" + "=" * 70)
    print("All smoke tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
