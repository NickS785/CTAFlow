from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .cmd_mamba import CMDMamba, CMDMambaConfig


class MarketGatingUnit(nn.Module):
    """
    Fuse target-asset features with market-state features via gated modulation.

    Inputs
    ------
    x_asset  : [B, T, D_asset]
    x_market : [B, T, D_market]
    Output
    ------
    [B, T, D_asset]
    """

    def __init__(
        self,
        asset_dim: int,
        market_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = max(8, int(hidden_dim))
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, asset_dim),
            nn.Tanh(),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(asset_dim * 2, asset_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(asset_dim)
        self._last_stats: Dict[str, float] = {}

    def forward(self, x_asset: torch.Tensor, x_market: torch.Tensor) -> torch.Tensor:
        if x_asset.dim() != 3 or x_market.dim() != 3:
            raise ValueError(
                f"MarketGatingUnit expects 3D tensors, got "
                f"x_asset={tuple(x_asset.shape)}, x_market={tuple(x_market.shape)}"
            )
        if x_asset.shape[:2] != x_market.shape[:2]:
            raise ValueError(
                "x_asset and x_market must have matching [B,T] dimensions, got "
                f"{tuple(x_asset.shape[:2])} vs {tuple(x_market.shape[:2])}"
            )

        m_encoded = self.market_encoder(x_market)
        gate = self.gate_net(torch.cat([x_asset, m_encoded], dim=-1))
        out = self.norm(x_asset + gate * m_encoded)
        self._last_stats = {
            "gate_mean": float(gate.mean().detach().cpu()),
            "gate_std": float(gate.std().detach().cpu()),
            "encoded_market_mean": float(m_encoded.mean().detach().cpu()),
            "encoded_market_std": float(m_encoded.std().detach().cpu()),
        }
        return out

    def get_last_stats(self) -> Dict[str, float]:
        return dict(self._last_stats)


class FinMambaCMD(nn.Module):
    """
    Market-aware CMDMamba.

    Flow:
      x_long (target-asset features) + x_market (macro/sector state)
        -> MarketGatingUnit
        -> CMDMamba long/short backbone
        -> regression or classification output
    """

    def __init__(
        self,
        asset_dim: int,
        market_dim: int,
        cmd_config: Optional[Union[CMDMambaConfig, Dict[str, object]]] = None,
        time_feat_dim: int = 0,
        classification: bool = False,
        num_classes: int = 3,
        raster_norm_means: Optional[torch.Tensor] = None,
        raster_norm_stds: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if cmd_config is None:
            cfg = CMDMambaConfig()
        elif isinstance(cmd_config, CMDMambaConfig):
            cfg = CMDMambaConfig(**vars(cmd_config))
        elif isinstance(cmd_config, dict):
            cfg = CMDMambaConfig(**cmd_config)
        else:
            raise TypeError("cmd_config must be CMDMambaConfig, dict, or None.")

        if classification:
            cfg.task = "classification"
            cfg.out_dim = int(num_classes)

        self.cfg = cfg
        self.task = cfg.task
        self.market_gate = MarketGatingUnit(
            asset_dim=int(asset_dim),
            market_dim=int(market_dim),
            hidden_dim=max(64, int(asset_dim)),
            dropout=float(cfg.dropout),
        )
        self.backbone = CMDMamba(
            long_input_dim=int(asset_dim),
            time_feat_dim=int(time_feat_dim),
            cfg=cfg,
            raster_norm_means=raster_norm_means,
            raster_norm_stds=raster_norm_stds,
        )
        self.last_tracker: Dict[str, object] = {}

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
        x_long_fused = self.market_gate(x_long, x_market)

        out = self.backbone(
            x_short=x_short,
            x_long=x_long_fused,
            t_short=t_short,
            t_long=t_long,
            short_mask=short_mask,
            return_probs=return_probs,
            return_features=return_features,
        )
        self.last_tracker = {
            "market_gate": self.market_gate.get_last_stats(),
            "cmd_backbone": dict(self.backbone.last_tracker),
        }

        if not return_features:
            return out

        logits, feats = out
        payload = dict(feats)
        payload["x_long_fused"] = x_long_fused
        payload["tracker"] = self.get_last_tracker()
        return logits, payload

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self.last_tracker)
