from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTAFlow.models.deep_learning.multi_asset_qlstm import QLSTMConfig, QLSTMModel


DEFAULT_DQ_QUANTILES: Tuple[float, ...] = (
    0.00005,
    0.00025,
    0.00075,
    0.00125,
    0.00175,
    0.0025,
    0.005,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.98,
    0.99,
    0.995,
    0.9975,
    0.99925,
    0.99975,
    0.99995,
)


@dataclass
class QuantileLSTMConfig(QLSTMConfig):
    """
    Same architecture as CTAFlow.models.deep_learning.multi_asset_qlstm.QLSTMConfig,
    but with dense-vs-lstm paper quantile grid as default.
    """

    quantiles: Tuple[float, ...] = DEFAULT_DQ_QUANTILES


class QuantileLSTMModel(QLSTMModel):
    """Alias model for the dense-vs-lstm pipeline (same implementation as QLSTMModel)."""



def pinball_loss(error: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Pinball loss averaged across batch/horizon/quantiles."""
    tau = tau.view(1, 1, -1).to(error.dtype)
    return torch.where(error >= 0, tau * error, (tau - 1.0) * error).mean()


@dataclass
class DenseQuantileConfig:
    """Configuration for the two-stage dense quantile network."""

    quantiles: Tuple[float, ...] = DEFAULT_DQ_QUANTILES
    asset_input_dim: int = 27
    asset_hidden_dims: Tuple[int, ...] = (128, 128, 128, 128)
    bottleneck_dim: int = 4
    market_input_dim: int = 15
    market_hidden_dims: Tuple[int, ...] = (32, 16)
    num_asset_classes: int = 6
    asset_embed_dim: int = 8
    use_asset_embedding: bool = True
    dropout: float = 0.2

    # Optional regularization terms used in the paper reference script.
    l1_lambda_layer1: float = 1e-4
    l1_lambda_layer2: float = 1e-5
    l2_lambda_market: float = 1e-5

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)


class DenseBlock(nn.Module):
    """Linear -> BatchNorm -> LeakyReLU -> Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


class TemporalAttentionPool(nn.Module):
    """
    Attention pooling over time.

    Note:
      This intentionally avoids variable names that shadow torch.nn.functional,
      which can break softmax calls.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        hidden = max(feature_dim // 2, 4)
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TemporalAttentionPool expects [B,T,F], got {tuple(x.shape)}")
        batch_size, seq_len, _ = x.shape

        if seq_lengths is None:
            seq_lengths = torch.full((batch_size,), seq_len, device=x.device, dtype=torch.long)
        seq_lengths = seq_lengths.clamp(min=1, max=seq_len)

        scores = self.attn(x).squeeze(-1)  # [B,T]
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B,T,1]
        return (x * weights).sum(dim=1)  # [B,F]


class DenseQuantileModel(nn.Module):
    """
    Two-stage dense quantile network.

    Stage 1: asset sequence -> pooled -> dense stack -> monotonic quantiles.
    Stage 2: market sequence -> pooled -> dense stack -> positive scaling.

    Final output:
      q = q_norm * sigma_market
      if group_vol provided: q = q_norm * sigma_market * group_vol
    """

    def __init__(self, config: DenseQuantileConfig):
        super().__init__()
        self.config = config
        self.register_buffer("quantile_levels", torch.tensor(config.quantiles, dtype=torch.float32))

        self.asset_pool = TemporalAttentionPool(config.asset_input_dim)
        self.market_pool = TemporalAttentionPool(config.market_input_dim)

        asset_in = config.asset_input_dim
        if config.use_asset_embedding:
            self.asset_embed = nn.Embedding(max(1, config.num_asset_classes), config.asset_embed_dim)
            asset_in += config.asset_embed_dim
        else:
            self.asset_embed = None

        layers: List[nn.Module] = []
        prev = asset_in
        for hidden_dim in config.asset_hidden_dims:
            layers.append(DenseBlock(prev, hidden_dim, config.dropout))
            prev = hidden_dim
        self.asset_hidden = nn.Sequential(*layers)

        self.bottleneck = nn.Sequential(
            nn.Linear(prev, config.bottleneck_dim),
            nn.Tanh(),
        )
        self.quantile_head = nn.Linear(config.bottleneck_dim, config.num_quantiles)

        mkt_layers: List[nn.Module] = []
        prev_m = config.market_input_dim
        for hidden_dim in config.market_hidden_dims:
            mkt_layers.append(DenseBlock(prev_m, hidden_dim, config.dropout))
            prev_m = hidden_dim
        mkt_layers.append(nn.Linear(prev_m, 1))
        self.market_net = nn.Sequential(*mkt_layers)

    @staticmethod
    def _enforce_monotonicity(q: torch.Tensor) -> torch.Tensor:
        first = q[:, :1]
        deltas = F.softplus(q[:, 1:])
        return torch.cat([first, first + torch.cumsum(deltas, dim=1)], dim=1)

    def forward(
        self,
        asset_features: torch.Tensor,
        market_features: torch.Tensor,
        asset_class_ids: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        group_vol: Optional[torch.Tensor] = None,
        **_: Dict,
    ) -> torch.Tensor:
        if seq_lengths is None:
            seq_lengths = torch.full(
                (asset_features.shape[0],),
                asset_features.shape[1],
                device=asset_features.device,
                dtype=torch.long,
            )

        x = self.asset_pool(asset_features, seq_lengths)
        z = self.market_pool(market_features, seq_lengths)

        if self.asset_embed is not None and asset_class_ids is not None:
            emb = self.asset_embed(asset_class_ids.to(torch.long))
            x = torch.cat([x, emb], dim=-1)

        h = self.asset_hidden(x)
        b = self.bottleneck(h)
        q_norm = self._enforce_monotonicity(self.quantile_head(b))

        sigma_market = F.softplus(self.market_net(z)) + 1e-6  # [B,1]
        q = q_norm * sigma_market
        if group_vol is not None:
            q = q * group_vol.unsqueeze(-1)
        return q

    def compute_loss(
        self,
        predicted_quantiles: torch.Tensor,
        raw_returns: torch.Tensor,
        normalized_returns: Optional[torch.Tensor] = None,
        predicted_normalized_quantiles: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tau = self.quantile_levels

        if raw_returns.dim() == 1:
            raw_returns = raw_returns.unsqueeze(1)
        if predicted_quantiles.dim() == 2:
            predicted_quantiles = predicted_quantiles.unsqueeze(1)

        raw_error = raw_returns.unsqueeze(-1) - predicted_quantiles
        raw_loss = pinball_loss(raw_error, tau)
        total = raw_loss
        norm_loss = raw_loss.new_zeros(())

        if normalized_returns is not None and predicted_normalized_quantiles is not None:
            if normalized_returns.dim() == 1:
                normalized_returns = normalized_returns.unsqueeze(1)
            if predicted_normalized_quantiles.dim() == 2:
                predicted_normalized_quantiles = predicted_normalized_quantiles.unsqueeze(1)
            norm_error = normalized_returns.unsqueeze(-1) - predicted_normalized_quantiles
            norm_loss = pinball_loss(norm_error, tau)
            total = total + norm_loss

        return {
            "total_loss": total,
            "raw_loss": raw_loss,
            "normalized_loss": norm_loss,
        }

    def get_l1_l2_penalty(self) -> torch.Tensor:
        """
        Optional regularization matching the reference script:
          - L1 on first two asset linear layers
          - L2 on first market linear layer
        """
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)

        asset_linears = [m for m in self.asset_hidden.modules() if isinstance(m, nn.Linear)]
        if len(asset_linears) >= 1:
            penalty = penalty + self.config.l1_lambda_layer1 * asset_linears[0].weight.abs().sum()
        if len(asset_linears) >= 2:
            penalty = penalty + self.config.l1_lambda_layer2 * asset_linears[1].weight.abs().sum()

        market_linears = [m for m in self.market_net.modules() if isinstance(m, nn.Linear)]
        if len(market_linears) >= 1:
            penalty = penalty + self.config.l2_lambda_market * (market_linears[0].weight ** 2).sum()

        return penalty
