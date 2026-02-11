"""
Regime-Aware Mixture of Experts built on CMDMamba.

Flow
----
1. RegimeRouter inspects x_market → expert weights  (regime detection)
2. Each Expert = MarketGatingUnit → CMDMamba          (asset processing)
3. Weighted fusion of expert embeddings → unified head (prediction)

The market state is consumed *before* asset features hit the backbone,
making the model explicitly regime-aware prior to prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTAFlow.models.deep_learning.multi_branch.cmd_mamba import (
    CMDMamba,
    CMDMambaConfig,
)
from CTAFlow.models.deep_learning.multi_branch.fin_mamba import MarketGatingUnit


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RegimeMoEConfig:
    """Top-level config for the Regime-Aware MoE wrapper."""

    # Expert count
    num_experts: int = 3

    # Router
    router_hidden_dim: int = 64
    router_dropout: float = 0.1
    router_temperature: float = 1.0          # softmax temperature (< 1 → sharper)

    # Sparse routing (set top_k < num_experts to activate)
    top_k: int = 0                            # 0 = soft (all experts), >0 = sparse

    # Load-balancing auxiliary loss coefficient
    aux_loss_coeff: float = 0.01

    # Unified head
    head_hidden_dim: int = 0                  # 0 → inherit d_model from CMDMambaConfig


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class RegimeRouter(nn.Module):
    """
    Computes per-expert routing weights from the market-state stream.

    Supports both **soft** gating (all experts weighted) and **top-k sparse**
    gating.  Optionally returns an auxiliary load-balancing loss to prevent
    expert collapse during training.
    """

    def __init__(
        self,
        market_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
        top_k: int = 0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = float(temperature)
        self.top_k = int(top_k) if top_k > 0 else 0

        self.net = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self, x_market: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_market : [B, T, D_market]

        Returns
        -------
        weights    : [B, K]   routing probabilities
        aux_loss   : scalar   load-balancing loss
        """
        # Use final time-step for routing (most current regime signal)
        state = x_market[:, -1, :]                        # [B, D_market]
        logits = self.net(state) / self.temperature        # [B, K]

        if self.top_k > 0 and self.top_k < self.num_experts:
            # Sparse: zero out all but top-k experts
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
            logits = logits.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(logits, dim=-1)                # [B, K]

        # ---- auxiliary load-balancing loss (Switch Transformer style) ----
        # Encourages uniform expert utilisation across the batch.
        #   f_i = fraction of batch routed to expert i  (mean of weights)
        #   P_i = mean router probability for expert i   (same here for soft)
        # loss = K * sum(f_i * P_i)
        f = weights.mean(dim=0)                            # [K]
        aux_loss = self.num_experts * (f * f).sum()

        return weights, aux_loss


# ---------------------------------------------------------------------------
# Expert wrapper (MarketGate + CMDMamba backbone, head-less)
# ---------------------------------------------------------------------------
class _Expert(nn.Module):
    """
    Single expert: MarketGatingUnit → CMDMamba backbone.

    Returns the fused embedding (z_fused) rather than the head output so
    the MoE container can blend embeddings before a shared head.
    """

    def __init__(
        self,
        asset_dim: int,
        market_dim: int,
        cmd_config: CMDMambaConfig,
        time_feat_dim: int = 0,
        raster_norm_means: Optional[torch.Tensor] = None,
        raster_norm_stds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gate = MarketGatingUnit(
            asset_dim=asset_dim,
            market_dim=market_dim,
            hidden_dim=max(64, asset_dim),
            dropout=cmd_config.dropout,
        )
        self.backbone = CMDMamba(
            long_input_dim=asset_dim,
            time_feat_dim=time_feat_dim,
            cfg=cmd_config,
            raster_norm_means=raster_norm_means,
            raster_norm_stds=raster_norm_stds,
        )

    def forward(
        self,
        x_short: torch.Tensor,
        x_long: torch.Tensor,
        x_market: torch.Tensor,
        t_short: Optional[torch.Tensor] = None,
        t_long: Optional[torch.Tensor] = None,
        short_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns z_fused [B, D] — the blended long/short embedding."""
        x_long_gated = self.gate(x_long, x_market)

        _, feats = self.backbone(
            x_short=x_short,
            x_long=x_long_gated,
            t_short=t_short,
            t_long=t_long,
            short_mask=short_mask,
            return_features=True,
        )
        return feats["z_fused"]                            # [B, D]


# ---------------------------------------------------------------------------
# Main MoE model
# ---------------------------------------------------------------------------
class RegimeAwareMoE(nn.Module):
    """
    Regime-Aware Mixture of Experts.

    Architecture
    ------------
    ┌──────────┐
    │ x_market ├──► RegimeRouter ──► weights [B, K]
    └──────────┘
    ┌──────────┐  ┌──────────┐
    │ x_long   │  │ x_short  │
    └────┬─────┘  └────┬─────┘
         │             │
         ▼             ▼
    ┌─ Expert 0 (Gate → CMDMamba) ──► z_0 ─┐
    ├─ Expert 1 (Gate → CMDMamba) ──► z_1 ─┤  weighted
    └─ Expert 2 (Gate → CMDMamba) ──► z_2 ─┘  sum → z_fused
                                                    │
                                              Unified Head
                                                    │
                                                prediction
    """

    def __init__(
        self,
        asset_dim: int,
        market_dim: int,
        cmd_config: Optional[Union[CMDMambaConfig, Dict]] = None,
        moe_config: Optional[Union[RegimeMoEConfig, Dict]] = None,
        time_feat_dim: int = 0,
        classification: bool = False,
        num_classes: int = 3,
        raster_norm_means: Optional[torch.Tensor] = None,
        raster_norm_stds: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # ---- resolve configs ----
        if cmd_config is None:
            cfg = CMDMambaConfig()
        elif isinstance(cmd_config, dict):
            cfg = CMDMambaConfig(**cmd_config)
        else:
            cfg = CMDMambaConfig(**vars(cmd_config))

        if classification:
            cfg.task = "classification"
            cfg.out_dim = int(num_classes)

        if moe_config is None:
            moe_cfg = RegimeMoEConfig()
        elif isinstance(moe_config, dict):
            moe_cfg = RegimeMoEConfig(**moe_config)
        else:
            moe_cfg = moe_config

        self.cfg = cfg
        self.moe_cfg = moe_cfg
        self.num_experts = moe_cfg.num_experts
        self.aux_loss_coeff = moe_cfg.aux_loss_coeff
        d_model = cfg.d_model

        # ---- router (processes market state FIRST) ----
        self.router = RegimeRouter(
            market_dim=market_dim,
            num_experts=self.num_experts,
            hidden_dim=moe_cfg.router_hidden_dim,
            dropout=moe_cfg.router_dropout,
            temperature=moe_cfg.router_temperature,
            top_k=moe_cfg.top_k,
        )

        # ---- experts ----
        # Each expert gets its own MarketGatingUnit + CMDMamba so weights
        # can specialise to different regime dynamics.
        self.experts = nn.ModuleList(
            [
                _Expert(
                    asset_dim=asset_dim,
                    market_dim=market_dim,
                    cmd_config=cfg,
                    time_feat_dim=time_feat_dim,
                    raster_norm_means=raster_norm_means,
                    raster_norm_stds=raster_norm_stds,
                )
                for _ in range(self.num_experts)
            ]
        )

        # ---- unified head ----
        head_hidden = moe_cfg.head_hidden_dim or d_model
        head_out = cfg.out_dim
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_hidden, head_out),
        )

        # ---- diagnostics ----
        self.last_tracker: Dict[str, object] = {}

    # ------------------------------------------------------------------
    def forward(
        self,
        x_short: torch.Tensor,
        x_long: torch.Tensor,
        x_market: torch.Tensor,
        t_short: Optional[torch.Tensor] = None,
        t_long: Optional[torch.Tensor] = None,
        short_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_features: bool = False,
    ):
        """
        Parameters
        ----------
        x_short   : [B, T_short, 4, H]    raster microstructure
        x_long    : [B, T_long, F_asset]   target-asset features
        x_market  : [B, T_long, F_market]  macro / sector state
        t_short   : optional time features for short branch
        t_long    : optional time features for long branch
        short_mask: [B, T_short] validity mask

        Returns
        -------
        prediction : [B, out_dim]
        extras     : dict (when return_features=True) containing:
            weights, expert_embeddings, z_fused, aux_loss, tracker
        """

        # 1. REGIME DETECTION — market state processed first
        weights, aux_loss = self.router(x_market)          # [B, K], scalar

        # 2. RUN EXPERTS — each gets market-gated asset features
        expert_embeddings: List[torch.Tensor] = []
        expert_run_mask = weights.sum(dim=0) > 0           # [K] skip zeroed experts

        for i, expert in enumerate(self.experts):
            if self.moe_cfg.top_k > 0 and not expert_run_mask[i]:
                # Sparse mode: skip experts that got zero weight
                expert_embeddings.append(
                    torch.zeros(
                        x_long.shape[0],
                        self.cfg.d_model,
                        device=x_long.device,
                        dtype=x_long.dtype,
                    )
                )
                continue

            z_i = expert(
                x_short=x_short,
                x_long=x_long,
                x_market=x_market,
                t_short=t_short,
                t_long=t_long,
                short_mask=short_mask,
            )                                               # [B, D]
            expert_embeddings.append(z_i)

        # Stack: [B, K, D]
        stacked = torch.stack(expert_embeddings, dim=1)

        # 3. WEIGHTED FUSION
        # weights [B, K] → [B, K, 1]  ·  stacked [B, K, D]  → sum → [B, D]
        z_fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)

        # 4. PREDICTION
        prediction = self.head(z_fused)
        if self.cfg.task == "classification" and return_probs:
            prediction = torch.softmax(prediction, dim=-1)

        # ---- diagnostics ----
        self.last_tracker = {
            "router_weights_mean": weights.mean(dim=0).detach().cpu(),
            "router_weights_std": weights.std(dim=0).detach().cpu(),
            "aux_loss": float(aux_loss.detach().cpu()),
            "dominant_expert": int(weights.mean(dim=0).argmax()),
        }

        if not return_features:
            return prediction

        return prediction, {
            "weights": weights,
            "expert_embeddings": stacked,
            "z_fused": z_fused,
            "aux_loss": aux_loss,
            "tracker": self.get_last_tracker(),
        }

    # ------------------------------------------------------------------
    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self.last_tracker)

    def aux_loss(self, features_dict: Dict) -> torch.Tensor:
        """Convenience: extract scaled aux loss for adding to task loss."""
        return self.aux_loss_coeff * features_dict["aux_loss"]


# ---------------------------------------------------------------------------
# Example usage / quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cpu"

    cmd_cfg = CMDMambaConfig(
        d_model=64,
        n_long_layers=2,
        n_short_layers=1,
        d_state_long=32,
        d_state_short=16,
        raster_bins=32,
        raster_channels=4,
        task="regression",
        out_dim=1,
    )
    moe_cfg = RegimeMoEConfig(
        num_experts=3,
        router_hidden_dim=32,
        top_k=0,              # soft routing
        aux_loss_coeff=0.01,
    )

    model = RegimeAwareMoE(
        asset_dim=24,
        market_dim=50,
        cmd_config=cmd_cfg,
        moe_config=moe_cfg,
        time_feat_dim=0,
    ).to(device)

    B, T_long, T_short = 4, 64, 64
    x_long   = torch.randn(B, T_long, 24, device=device)
    x_market = torch.randn(B, T_long, 50, device=device)
    x_short  = torch.randn(B, T_short, 4, 32, device=device)

    pred, extras = model(
        x_short=x_short,
        x_long=x_long,
        x_market=x_market,
        return_features=True,
    )

    print(f"prediction : {pred.shape}")       # [4, 1]
    print(f"weights    : {extras['weights']}")
    print(f"aux_loss   : {extras['aux_loss']:.4f}")
    print(f"tracker    : {model.get_last_tracker()}")

    # Training loop sketch:
    # task_loss = criterion(pred, targets)
    # total_loss = task_loss + model.aux_loss(extras)
    # total_loss.backward()
