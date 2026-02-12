"""
Volatility-Regime Mixture of Experts built on CMDMamba.

Flow
----
1. DeepVolRouter inspects 1D returns (x_vol) → predicts σ → expert weights
2. Each Expert = CMDMamba                                (asset processing)
3. Weighted fusion of expert embeddings → unified head   (prediction)

The router specialises in volatility (using 1D returns),
while the experts specialise in alpha (using asset features + raster).
No market gating — the vol router handles regime separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from CTAFlow.models.deep_learning.routers.deep_vol import (
    DeepVolRouter,
    DeepVolRouterConfig,
)
from CTAFlow.models.deep_learning.multi_branch.cmd_mamba import (
    CMDMamba,
    CMDMambaConfig,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class VolRegimeMoEConfig:
    """Top-level config for the Vol-Regime MoE wrapper."""

    num_experts: int = 3

    # DeepVol Router
    router_hidden_channels: int = 32
    router_layers: int = 4
    router_sharpness: float = 2.0        # softmax temperature
    router_momentum: float = 0.01        # EMA for running vol stats

    # Sparse routing (set top_k < num_experts to activate)
    top_k: int = 0                        # 0 = soft (all experts)

    # Aux loss coefficient for vol-prediction loss
    aux_loss_coeff: float = 0.01

    # Unified head
    head_hidden_dim: int = 0              # 0 → inherit d_model


# ---------------------------------------------------------------------------
# Expert wrapper (CMDMamba, head-less)
# ---------------------------------------------------------------------------
class _VolExpert(nn.Module):
    """Single expert: CMDMamba → z_fused embedding."""

    def __init__(
        self,
        asset_dim: int,
        cmd_config: CMDMambaConfig,
        time_feat_dim: int = 0,
        raster_norm_means: Optional[torch.Tensor] = None,
        raster_norm_stds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
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
        t_short: Optional[torch.Tensor] = None,
        t_long: Optional[torch.Tensor] = None,
        short_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns z_fused [B, D]."""
        _, feats = self.backbone(
            x_short=x_short,
            x_long=x_long,
            t_short=t_short,
            t_long=t_long,
            short_mask=short_mask,
            return_features=True,
        )
        return feats["z_fused"]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class VolRegimeAwareMoE(nn.Module):
    """
    Volatility-Regime Mixture of Experts.

    Architecture
    ------------
    ┌─────────────┐
    │  x_vol (1D) ├──► DeepVolRouter ──► weights [B,K] + pred_vol
    └─────────────┘
    ┌──────────┐  ┌──────────┐
    │ x_long   │  │ x_short  │
    └────┬─────┘  └────┬─────┘
         │             │
         ▼             ▼
    ┌─ Expert 0 (CMDMamba) ──► z_0 ─┐
    ├─ Expert 1 (CMDMamba) ──► z_1 ─┤  weighted
    └─ Expert 2 (CMDMamba) ──► z_2 ─┘  sum → z_fused
                                              │
                                        Unified Head
                                              │
                                          prediction
    """

    def __init__(
        self,
        asset_dim: int,
        cmd_config: Optional[Union[CMDMambaConfig, Dict]] = None,
        moe_config: Optional[Union[VolRegimeMoEConfig, Dict]] = None,
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
            moe_cfg = VolRegimeMoEConfig()
        elif isinstance(moe_config, dict):
            moe_cfg = VolRegimeMoEConfig(**moe_config)
        else:
            moe_cfg = moe_config

        self.cfg = cfg
        self.moe_cfg = moe_cfg
        self.num_experts = moe_cfg.num_experts
        self.aux_loss_coeff = moe_cfg.aux_loss_coeff
        d_model = cfg.d_model

        # ---- DeepVol router (processes 1D returns ONLY) ----
        router_cfg = DeepVolRouterConfig(
            num_experts=self.num_experts,
            hidden_channels=moe_cfg.router_hidden_channels,
            encoder_layers=moe_cfg.router_layers,
            sharpness=moe_cfg.router_sharpness,
            momentum=moe_cfg.router_momentum,
        )
        self.router = DeepVolRouter(cfg=router_cfg)

        # ---- experts (asset features + raster, no market gating) ----
        self.experts = nn.ModuleList(
            [
                _VolExpert(
                    asset_dim=asset_dim,
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
        x_vol: torch.Tensor,
        t_short: Optional[torch.Tensor] = None,
        t_long: Optional[torch.Tensor] = None,
        short_mask: Optional[torch.Tensor] = None,
        vol_target: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_features: bool = False,
    ):
        """
        Parameters
        ----------
        x_short    : [B, T_short, 4, H]   raster microstructure
        x_long     : [B, T_long, F_asset]  target-asset features
        x_vol      : [B, T_long]           1D return series for DeepVolRouter
        t_short    : optional time features for short branch
        t_long     : optional time features for long branch
        short_mask : [B, T_short] validity mask
        vol_target : [B, 1]  optional realised-vol target for aux loss
        """
        # 1. VOL-REGIME DETECTION
        weights, vol_loss, pred_vol = self.router(x_vol, vol_target)

        # 2. APPLY TOP-K SPARSITY (optional)
        if self.moe_cfg.top_k > 0 and self.moe_cfg.top_k < self.num_experts:
            topk_vals, topk_idx = weights.topk(self.moe_cfg.top_k, dim=-1)
            mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. RUN EXPERTS
        expert_embeddings: List[torch.Tensor] = []
        expert_run_mask = weights.sum(dim=0) > 0

        for i, expert in enumerate(self.experts):
            if self.moe_cfg.top_k > 0 and not expert_run_mask[i]:
                expert_embeddings.append(
                    torch.zeros(
                        x_long.shape[0], self.cfg.d_model,
                        device=x_long.device, dtype=x_long.dtype,
                    )
                )
                continue

            z_i = expert(
                x_short=x_short, x_long=x_long,
                t_short=t_short, t_long=t_long, short_mask=short_mask,
            )
            expert_embeddings.append(z_i)

        stacked = torch.stack(expert_embeddings, dim=1)       # [B, K, D]

        # 4. WEIGHTED FUSION
        z_fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # [B, D]

        # 5. PREDICTION
        prediction = self.head(z_fused)
        if self.cfg.task == "classification" and return_probs:
            prediction = torch.softmax(prediction, dim=-1)

        # ---- diagnostics ----
        self.last_tracker = {
            "router_weights_mean": weights.mean(dim=0).detach().cpu(),
            "router_weights_std": weights.std(dim=0).detach().cpu(),
            "vol_loss": float(vol_loss.detach().cpu()),
            "pred_vol_mean": float(pred_vol.mean().detach().cpu()),
            "pred_vol_std": float(pred_vol.std().detach().cpu()),
            "dominant_expert": int(weights.mean(dim=0).argmax()),
        }

        if not return_features:
            return prediction

        return prediction, {
            "weights": weights,
            "expert_embeddings": stacked,
            "z_fused": z_fused,
            "aux_loss": vol_loss,
            "pred_vol": pred_vol,
            "tracker": self.get_last_tracker(),
        }

    # ------------------------------------------------------------------
    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self.last_tracker)

    def aux_loss(self, features_dict: Dict) -> torch.Tensor:
        """Convenience: extract scaled aux loss for adding to task loss."""
        return self.aux_loss_coeff * features_dict["aux_loss"]
