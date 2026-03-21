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


class TCNBackbone(nn.Module):
    def __init__(self, in_channels: int, channels: List[int],
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = in_channels if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                 dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, channels[0])
        self.tcn = TCNBackbone(channels[0], channels, kernel_size, dropout)
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

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        trunk_dim = 64

        # Shared trunk
        self.trunk_encoder = nn.Sequential(
            nn.Linear(config.n_features, 128),
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
                config.n_features, config.tcn_channels,
                config.tcn_kernel_size, config.dropout,
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

    def forward(
        self,
        x_seq: torch.Tensor,       # (B, L, F)
        ae_input: torch.Tensor,     # (B, ae_window, f_ae)
    ) -> Dict[str, torch.Tensor]:

        B = x_seq.shape[0]
        device = x_seq.device

        # 1. VAE regime encoding
        z_regime, ae_losses = self.vae(ae_input)

        # 2. Shared trunk
        z_trunk = self.trunk_encoder(x_seq[:, -1, :])

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
                out = expert(x_seq[mask])

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

        return {
            "pred_return": pred_return,
            "pred_std": pred_std,
            "router_weights": weights,
            "balance_loss": balance_loss,
            "entropy_loss": entropy_loss,
            "mdn_nll_loss": mdn_nll_accum / max(n_mdn, 1),
            "ae_losses": ae_losses,
            "shared_gate": alpha,
            "z_regime": z_regime,
        }

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

    L = λ_ret * Huber(return) + λ_vol * MSE(log_std)
      + λ_nll * MDN_NLL + λ_bal * balance + λ_ent * entropy + AE_loss
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

    def forward(self, model_out, target_return, target_std):
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

        total = (
            cfg.return_loss_weight * return_loss
            + cfg.vol_loss_weight * vol_loss
            + cfg.nll_weight * model_out["mdn_nll_loss"]
            + cfg.load_balance_weight * model_out["balance_loss"]
            + cfg.entropy_reg_weight * model_out["entropy_loss"]
            + ae["total_ae_loss"]
        )

        return {
            "total_loss": total,
            "return_loss": return_loss,
            "vol_loss": vol_loss,
            "mdn_nll_loss": model_out["mdn_nll_loss"],
            "balance_loss": model_out["balance_loss"],
            "entropy_loss": model_out["entropy_loss"],
            "ae_recon_loss": ae["recon_loss"],
            "ae_kl_loss": ae["kl_loss"],
        }


# ============================================================================
# 9. DEMO
# ============================================================================

def run_demo():
    print("=" * 70)
    print("NatGas MoE (TCN + MDN) — Smoke Test")
    print("=" * 70)

    config = MoEConfig(
        n_features=40, seq_len=20,
        f_ae=12, ae_window=21, d_latent=32,
        n_tcn_experts=3, n_mdn_experts=3, top_k=3,
    )
    model = NatGasMoE(config)
    loss_fn = MoELoss(config)

    B = 32
    x_seq = torch.randn(B, config.seq_len, config.n_features)
    ae_input = torch.randn(B, config.ae_window, config.f_ae)
    target_ret = torch.randn(B) * 0.03
    target_std = torch.rand(B) * 0.02 + 0.01

    print("\n[1] Forward pass...")
    out = model(x_seq, ae_input)
    print(f"    pred_return:    {out['pred_return'].shape}")
    print(f"    pred_std:       {out['pred_std'].shape}")
    print(f"    z_regime:       {out['z_regime'].shape}")
    print(f"    shared_gate:    {out['shared_gate'].item():.3f}")

    w = out["router_weights"]
    for etype in ["tcn", "mdn"]:
        mask = [i for i, t in enumerate(model.expert_types) if t == etype]
        print(f"    avg_weight_{etype}: {w[:, mask].sum(dim=-1).mean().item():.3f}")

    print("\n[2] Loss computation...")
    losses = loss_fn(out, target_ret, target_std)
    for k, v in losses.items():
        print(f"    {k}: {v.item():.6f}")

    print("\n[3] Backward pass...")
    losses["total_loss"].backward()
    n_params = sum(p.numel() for p in model.parameters())
    n_grad = sum(p.numel() for p in model.parameters() if p.grad is not None)
    print(f"    Total params:   {n_params:,}")
    print(f"    Params w/ grad: {n_grad:,}")

    vae_grad = model.vae.encoder[0].weight.grad
    router_grad = model.router.gate[0].weight.grad
    print(f"    VAE encoder grad norm:  {vae_grad.norm().item():.6f}")
    print(f"    Router gate grad norm:  {router_grad.norm().item():.6f}")
    print(f"    Router input dim:       {model.router.gate[0].in_features}"
          f"  (trunk=64 + regime={config.d_latent})")

    print("\n" + "=" * 70)
    print("Smoke test PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
