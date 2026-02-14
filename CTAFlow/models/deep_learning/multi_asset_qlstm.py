from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    dims = [int(in_dim)] + [int(d) for d in hidden_dims if int(d) > 0]
    if len(dims) <= 1:
        return nn.Sequential(nn.Identity())

    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@dataclass
class QLSTMConfig:
    asset_input_dim: int
    market_input_dim: int

    asset_lstm_hidden: int = 64
    asset_lstm_layers: int = 1
    asset_dense_dims: List[int] = field(default_factory=lambda: [128, 64])

    market_lstm_hidden: int = 32
    market_lstm_layers: int = 1
    market_dense_dims: List[int] = field(default_factory=lambda: [64])

    num_asset_classes: int = 6
    asset_embed_dim: int = 8

    fusion_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1

    quantiles: Tuple[float, ...] = (
        0.01,
        0.05,
        0.10,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.90,
        0.95,
        0.99,
    )

    monotonic_eps: float = 1e-4
    norm_loss_weight: float = 0.0
    monotonic_penalty_weight: float = 0.0

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)


class _SequenceEncoder(nn.Module):
    """
    LSTM sequence encoder with optional post-LSTM dense projection.
    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        dense_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(lstm_hidden),
            num_layers=int(lstm_layers),
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.post = _build_mlp(int(lstm_hidden), dense_dims, dropout)
        self.output_dim = int(dense_dims[-1]) if len(dense_dims) > 0 else int(lstm_hidden)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B,T,F], got {tuple(x.shape)}")
        if seq_lengths.dim() != 1:
            raise ValueError(f"expected [B] seq_lengths, got {tuple(seq_lengths.shape)}")

        bsz, seq_len, _ = x.shape
        lengths = seq_lengths.clamp(min=1, max=seq_len).to(torch.long)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # [B, H]
        z = self.post(h_last)
        return self.norm(z)


class QLSTMModel(nn.Module):
    """
    Multi-asset qLSTM model that outputs monotonic quantiles for future returns.
    """

    def __init__(self, config: QLSTMConfig):
        super().__init__()
        self.config = config

        self.asset_encoder = _SequenceEncoder(
            input_dim=config.asset_input_dim,
            lstm_hidden=config.asset_lstm_hidden,
            lstm_layers=config.asset_lstm_layers,
            dense_dims=config.asset_dense_dims,
            dropout=config.dropout,
        )
        self.market_encoder = _SequenceEncoder(
            input_dim=config.market_input_dim,
            lstm_hidden=config.market_lstm_hidden,
            lstm_layers=config.market_lstm_layers,
            dense_dims=config.market_dense_dims,
            dropout=config.dropout,
        )

        self.asset_embed = nn.Embedding(
            num_embeddings=max(1, int(config.num_asset_classes)),
            embedding_dim=int(config.asset_embed_dim),
        )

        fused_dim = (
            self.asset_encoder.output_dim
            + self.market_encoder.output_dim
            + int(config.asset_embed_dim)
        )
        self.fusion = _build_mlp(fused_dim, config.fusion_hidden_dims, config.dropout)
        fusion_out_dim = int(config.fusion_hidden_dims[-1]) if len(config.fusion_hidden_dims) else fused_dim
        self.fusion_norm = nn.LayerNorm(fusion_out_dim)
        self.fusion_drop = nn.Dropout(config.dropout)

        self.base_head = nn.Linear(fusion_out_dim, 1)
        self.delta_head = nn.Linear(fusion_out_dim, max(0, config.num_quantiles - 1))

        taus = torch.tensor(config.quantiles, dtype=torch.float32)
        self.register_buffer("taus", taus)

    def forward(
        self,
        asset_features: torch.Tensor,
        market_features: torch.Tensor,
        asset_class_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
          quantiles: [B, Q]
        """
        z_asset = self.asset_encoder(asset_features, seq_lengths)
        z_market = self.market_encoder(market_features, seq_lengths)

        class_ids = asset_class_ids.to(torch.long).clamp(min=0, max=self.asset_embed.num_embeddings - 1)
        z_class = self.asset_embed(class_ids)

        z = torch.cat([z_asset, z_market, z_class], dim=-1)
        z = self.fusion(z)
        z = self.fusion_drop(self.fusion_norm(z))

        base = self.base_head(z)  # [B,1]
        if self.config.num_quantiles == 1:
            return base

        deltas = F.softplus(self.delta_head(z)) + float(self.config.monotonic_eps)
        q_rest = base + torch.cumsum(deltas, dim=-1)
        quantiles = torch.cat([base, q_rest], dim=-1)
        return quantiles

    def _pinball_loss(self, quantiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Quantile (pinball) loss for scalar targets.

        Args:
            quantiles: [B, Q] predicted quantile values.
            target: [B] scalar target returns.
        """
        if target.dim() == 1:
            target = target.unsqueeze(-1)  # [B, 1]

        # target: [B, 1], quantiles: [B, Q]
        tau = self.taus.unsqueeze(0).to(quantiles.dtype)  # [1, Q]
        err = target - quantiles  # [B, Q]
        loss = torch.maximum(tau * err, (tau - 1.0) * err)
        return loss.mean()

    def compute_loss(
        self,
        quantiles: torch.Tensor,
        target_return: torch.Tensor,
        target_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined pinball loss for the next-day return prediction.

        Args:
            quantiles: [B, Q] predicted quantile values.
            target_return: [B] scalar next-day raw log return r_{t+1}.
            target_norm: [B] optional vol-normalised return r_{t+1} / σ̄_t.
        """
        raw_loss = self._pinball_loss(quantiles, target_return)

        if target_norm is not None and self.config.norm_loss_weight > 0.0:
            norm_loss = self._pinball_loss(quantiles, target_norm)
        else:
            norm_loss = quantiles.new_zeros(())

        mono_violation = F.relu(quantiles[:, :-1] - quantiles[:, 1:]).mean()

        total = raw_loss
        if self.config.norm_loss_weight > 0.0:
            total = total + self.config.norm_loss_weight * norm_loss
        if self.config.monotonic_penalty_weight > 0.0:
            total = total + self.config.monotonic_penalty_weight * mono_violation

        return {
            "total_loss": total,
            "raw_loss": raw_loss,
            "norm_loss": norm_loss,
            "monotonic_penalty": mono_violation,
        }

