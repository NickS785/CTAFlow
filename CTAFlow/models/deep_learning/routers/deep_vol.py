"""
DeepVol Router — volatility-based regime routing.

Takes a 1D return series, predicts realised volatility via causal dilated
convolutions (reusing the existing ``DeepVolEncoder``), then maps the
predicted σ to per-expert routing weights through z-score anchors.

The router **only** sees returns; experts see the full multi-modal features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTAFlow.models.deep_learning.deep_vol import DeepVolEncoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DeepVolRouterConfig:
    """Parameters for the DeepVol-based regime router."""

    num_experts: int = 3
    hidden_channels: int = 32
    encoder_layers: int = 4
    vol_head_dim: int = 16
    sharpness: float = 2.0          # softmax temperature (higher → sharper)
    momentum: float = 0.01          # EMA momentum for running vol stats


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class DeepVolRouter(nn.Module):
    """
    1D Returns → Predicted σ → Regime Weights.

    Architecture
    ------------
    1. ``DeepVolEncoder`` (gated causal conv stack) extracts temporal features.
    2. A lightweight head predicts scalar volatility (Softplus output).
    3. The predicted σ is z-scored against running batch statistics and
       mapped to regime anchors via distance → softmax.

    Parameters
    ----------
    cfg : DeepVolRouterConfig  (or use keyword defaults)
    """

    def __init__(self, cfg: Optional[DeepVolRouterConfig] = None, **kw):
        super().__init__()

        if cfg is None:
            cfg = DeepVolRouterConfig(**kw)
        self.cfg = cfg
        self.num_experts = cfg.num_experts

        # ---- backbone (reuses existing DeepVol encoder) ----
        self.encoder = DeepVolEncoder(
            input_dim=1,
            hidden_dim=cfg.hidden_channels,
            layers=cfg.encoder_layers,
        )

        # ---- volatility head ----
        self.vol_head = nn.Sequential(
            nn.Linear(cfg.hidden_channels, cfg.vol_head_dim),
            nn.ReLU(),
            nn.Linear(cfg.vol_head_dim, 1),
            nn.Softplus(),                           # σ must be > 0
        )

        # ---- regime anchors (evenly spaced z-scores) ----
        anchors = torch.linspace(-1.0, 1.0, cfg.num_experts)
        self.register_buffer("anchors", anchors)
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.sharpness = cfg.sharpness
        self.momentum = cfg.momentum

    # ------------------------------------------------------------------
    def forward(
        self,
        x_returns: torch.Tensor,
        vol_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_returns  : [B, T]  1-D return series
        vol_target : [B, 1]  optional realised-vol target for aux loss

        Returns
        -------
        weights    : [B, K]   routing probabilities
        aux_loss   : scalar   vol-prediction MSE (0 if no target)
        pred_vol   : [B, 1]   predicted volatility
        """
        # (B, T) → (B, 1, T) — single-channel input for encoder
        if x_returns.dim() == 2:
            x = x_returns.unsqueeze(1)
        else:
            x = x_returns

        # ---- temporal features → vol prediction ----
        h = self.encoder(x)                          # [B, H, T']
        embedding = h.mean(dim=-1)                   # [B, H]
        pred_vol = self.vol_head(embedding)           # [B, 1]

        # ---- update running statistics (train only) ----
        if self.training:
            with torch.no_grad():
                self.running_mean.lerp_(pred_vol.mean(), self.momentum)
                self.running_var.lerp_(pred_vol.var(), self.momentum)

        # ---- z-score → distance to anchors → softmax ----
        std = torch.sqrt(self.running_var + 1e-6)
        z = (pred_vol - self.running_mean) / std     # [B, 1]
        dists = -torch.abs(z - self.anchors)         # [B, K]
        weights = F.softmax(dists * self.sharpness, dim=-1)

        # ---- auxiliary vol-prediction loss ----
        aux_loss = torch.tensor(0.0, device=x_returns.device, dtype=x_returns.dtype)
        if vol_target is not None:
            aux_loss = F.mse_loss(pred_vol, vol_target)

        return weights, aux_loss, pred_vol

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"experts={self.num_experts}, "
            f"hidden={self.cfg.hidden_channels}, "
            f"layers={self.cfg.encoder_layers}, "
            f"sharpness={self.sharpness}"
        )
