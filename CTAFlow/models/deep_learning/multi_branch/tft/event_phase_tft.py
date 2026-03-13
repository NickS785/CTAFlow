from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..market_context_models import (
    BranchVariableSelection,
    GatedCrossAttentionFusion,
    GatedResidualNetwork,
    MacroEnrichedTemporalAttention,
)
from ...density_quantile import pinball_loss
from ...training.loss.clf import ContinuousTradingLoss


DEFAULT_EVENT_PHASE_QUANTILES: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
DEFAULT_PHASE_NAMES: Tuple[str, ...] = ("pre", "event", "post")


def _validate_quantiles(quantiles: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(q) for q in quantiles)
    if not values:
        raise ValueError("quantiles must not be empty")
    if any(q <= 0.0 or q >= 1.0 for q in values):
        raise ValueError(f"quantiles must lie in (0, 1), got {values}")
    if tuple(sorted(values)) != values:
        raise ValueError(f"quantiles must be sorted ascending, got {values}")
    return values


@dataclass
class EventPhaseTFTConfig:
    input_dim: int
    known_future_dim: int = 0
    static_real_dim: int = 0
    technical_input_dim: int = 0
    technical_seq_len: int = 0
    single_point_num_classes: int = 0
    num_phases: int = 3
    phase_names: Tuple[str, ...] = DEFAULT_PHASE_NAMES
    phase_seq_len: int = 128
    horizon_steps: int = 6
    quantiles: Tuple[float, ...] = DEFAULT_EVENT_PHASE_QUANTILES
    d_model: int = 128
    d_hidden: int = 128
    d_phase_emb: int = 32
    d_static_emb: int = 32
    n_heads: int = 4
    dropout: float = 0.1
    n_contract_months: int = 13
    n_vol_regimes: int = 8
    causal_phase_attention: bool = False

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)


@dataclass
class QuantileExposureConfig:
    horizon_steps: int = 6
    quantiles: Tuple[float, ...] = DEFAULT_EVENT_PHASE_QUANTILES
    d_model: int = 64
    dropout: float = 0.1
    base_temperature: float = 1.5
    context_dim: int = 0
    target_horizon_weights: Optional[Tuple[float, ...]] = None


@dataclass
class JointEventPhaseTFTConfig:
    trunk: EventPhaseTFTConfig
    exposure: QuantileExposureConfig

    def __post_init__(self):
        if self.trunk.horizon_steps != self.exposure.horizon_steps:
            raise ValueError("trunk and exposure horizon_steps must match")
        if tuple(self.trunk.quantiles) != tuple(self.exposure.quantiles):
            raise ValueError("trunk and exposure quantiles must match")
        if self.exposure.target_horizon_weights is not None:
            if len(self.exposure.target_horizon_weights) != self.trunk.horizon_steps:
                raise ValueError("target_horizon_weights must match trunk.horizon_steps")


class EventStaticContextEncoder(nn.Module):
    """Encode static event context into TFT-style conditioning vectors."""

    def __init__(
        self,
        *,
        d_model: int,
        static_real_dim: int = 0,
        d_static_emb: int = 32,
        n_contract_months: int = 13,
        n_vol_regimes: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.static_real_dim = int(static_real_dim)
        self.month_emb = nn.Embedding(max(1, n_contract_months), d_static_emb)
        self.regime_emb = nn.Embedding(max(1, n_vol_regimes), d_static_emb)
        self.real_proj = (
            nn.Sequential(
                nn.Linear(self.static_real_dim, d_static_emb),
                nn.LayerNorm(d_static_emb),
                nn.GELU(),
            )
            if self.static_real_dim > 0
            else None
        )

        parts = 2 + (1 if self.real_proj is not None else 0)
        self.fuse = nn.Sequential(
            nn.Linear(parts * d_static_emb, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_cs = GatedResidualNetwork(d_model=d_model, dropout=dropout)
        self.to_ce = GatedResidualNetwork(d_model=d_model, dropout=dropout)
        self.to_cc = GatedResidualNetwork(d_model=d_model, dropout=dropout)
        self.to_ch = GatedResidualNetwork(d_model=d_model, dropout=dropout)

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        contract_month: Optional[torch.Tensor] = None,
        vol_regime: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        month_ids = (
            contract_month.to(device=device, dtype=torch.long)
            if contract_month is not None
            else torch.zeros(batch_size, device=device, dtype=torch.long)
        )
        regime_ids = (
            vol_regime.to(device=device, dtype=torch.long)
            if vol_regime is not None
            else torch.zeros(batch_size, device=device, dtype=torch.long)
        )

        parts = [
            self.month_emb(month_ids.clamp(0, self.month_emb.num_embeddings - 1)),
            self.regime_emb(regime_ids.clamp(0, self.regime_emb.num_embeddings - 1)),
        ]

        if self.real_proj is not None:
            if static_real is None:
                static_real = torch.zeros(batch_size, self.static_real_dim, device=device)
            else:
                static_real = static_real.to(device=device, dtype=torch.float32)
            parts.append(self.real_proj(static_real))

        fused = self.fuse(torch.cat(parts, dim=-1))
        return self.to_cs(fused), self.to_ce(fused), self.to_cc(fused), self.to_ch(fused)


class PhaseSpecificVariableSelection(nn.Module):
    """Per-phase variable selection network operating on one phase chunk."""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        d_context: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)

        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(self.input_dim)]
        )
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(d_model=d_model, dropout=dropout) for _ in range(self.input_dim)]
        )
        self.weight_grn = GatedResidualNetwork(
            d_model=d_model,
            d_input=self.input_dim,
            d_context=d_context,
            dropout=dropout,
        )
        self.weight_proj = nn.Linear(d_model, self.input_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"PhaseSpecificVariableSelection expects [B,T,F], got {tuple(x.shape)}")
        batch_size, seq_len, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {feat_dim}")

        feature_tokens = []
        for feature_idx in range(self.input_dim):
            feature_slice = x[..., feature_idx:feature_idx + 1]
            projected = self.feature_projections[feature_idx](feature_slice)
            encoded = self.feature_grns[feature_idx](projected)
            feature_tokens.append(encoded)

        stacked = torch.stack(feature_tokens, dim=2)

        x_flat = x.reshape(batch_size * seq_len, feat_dim)
        if context is not None:
            context_expanded = context.unsqueeze(1).expand(batch_size, seq_len, -1).reshape(batch_size * seq_len, -1)
        else:
            context_expanded = None
        weight_hidden = self.weight_grn(x_flat, context=context_expanded)
        weights = F.softmax(self.weight_proj(weight_hidden), dim=-1).view(batch_size, seq_len, feat_dim)

        selected = torch.einsum("btf,btfd->btd", weights, stacked)
        return selected, weights


class PhaseTemporalContextEncoder(nn.Module):
    """Encode known-future inputs and phase IDs into phase tokens."""

    def __init__(
        self,
        *,
        num_phases: int,
        known_future_dim: int,
        d_model: int,
        d_phase_emb: int = 32,
        d_context: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.known_future_dim = int(known_future_dim)
        self.phase_emb = nn.Embedding(max(1, num_phases), d_phase_emb)
        input_dim = d_phase_emb + self.known_future_dim
        self.encoder = GatedResidualNetwork(
            d_model=d_model,
            d_input=input_dim,
            d_context=d_context,
            dropout=dropout,
        )

    def forward(
        self,
        *,
        phase_ids: torch.Tensor,
        known_future: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_phases = phase_ids.shape
        phase_embed = self.phase_emb(phase_ids)

        if self.known_future_dim > 0:
            if known_future is None:
                known_future = torch.zeros(
                    batch_size,
                    num_phases,
                    self.known_future_dim,
                    device=phase_ids.device,
                    dtype=torch.float32,
                )
            elif known_future.shape != (batch_size, num_phases, self.known_future_dim):
                raise ValueError(
                    f"known_future must have shape {(batch_size, num_phases, self.known_future_dim)}, "
                    f"got {tuple(known_future.shape)}"
                )
            encoder_input = torch.cat([phase_embed, known_future], dim=-1)
        else:
            encoder_input = phase_embed

        flat_input = encoder_input.reshape(batch_size * num_phases, -1)
        if context is not None:
            flat_context = context.unsqueeze(1).expand(batch_size, num_phases, -1).reshape(batch_size * num_phases, -1)
        else:
            flat_context = None
        encoded = self.encoder(flat_input, context=flat_context)
        return encoded.view(batch_size, num_phases, -1)


class TechnicalIndicatorBranchEncoder(nn.Module):
    """Encode aligned technical-indicator sequences into a dense branch token."""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        d_context: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pre_lstm_grn = GatedResidualNetwork(
            d_model=d_model,
            d_context=d_context,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"technical_features must have shape [B,T,F], got {tuple(x.shape)}")
        batch_size, seq_len, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected technical feature dim {self.input_dim}, got {feat_dim}")
        hidden = self.proj(x)
        flat_hidden = hidden.reshape(batch_size * seq_len, -1)
        if context is not None:
            flat_context = context.unsqueeze(1).expand(batch_size, seq_len, -1).reshape(batch_size * seq_len, -1)
        else:
            flat_context = None
        encoded = self.pre_lstm_grn(flat_hidden, context=flat_context).view(batch_size, seq_len, -1)
        init_state = None
        if h0 is not None and c0 is not None:
            init_state = (h0.unsqueeze(0), c0.unsqueeze(0))
        _, (h_n, _) = self.lstm(encoded, init_state)
        return h_n[-1]


class MultiHorizonQuantileHead(nn.Module):
    """Project a fused token into horizon-specific monotonic quantile paths."""

    def __init__(
        self,
        *,
        d_input: int,
        d_model: int,
        horizon_steps: int,
        quantiles: Sequence[float],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.quantiles = _validate_quantiles(quantiles)
        self.horizon_emb = nn.Embedding(self.horizon_steps, d_model)
        self.context_proj = nn.Linear(d_input, d_model)
        self.token_grn = GatedResidualNetwork(d_model=d_model, dropout=dropout)
        self.out_proj = nn.Linear(d_model, len(self.quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        base = self.context_proj(x).unsqueeze(1).expand(batch_size, self.horizon_steps, -1)
        horizon_ids = torch.arange(self.horizon_steps, device=x.device, dtype=torch.long)
        horizon_tokens = base + self.horizon_emb(horizon_ids).unsqueeze(0)
        flat_tokens = horizon_tokens.reshape(batch_size * self.horizon_steps, -1)
        raw = self.out_proj(self.token_grn(flat_tokens)).view(batch_size, self.horizon_steps, -1)

        first = raw[..., :1]
        if raw.shape[-1] == 1:
            return first

        deltas = F.softplus(raw[..., 1:])
        return torch.cat([first, first + torch.cumsum(deltas, dim=-1)], dim=-1)


class QuantileToExposureHead(nn.Module):
    """Trainable mapper from multi-horizon quantiles to a tanh exposure."""

    def __init__(self, config: QuantileExposureConfig):
        super().__init__()
        self.config = config
        self.config.quantiles = _validate_quantiles(self.config.quantiles)
        self.horizon_steps = int(self.config.horizon_steps)
        self.num_quantiles = len(self.config.quantiles)
        self.raw_feature_dim = self.num_quantiles + 6

        self.horizon_emb = nn.Embedding(self.horizon_steps, self.config.d_model)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.raw_feature_dim, self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.horizon_grn = GatedResidualNetwork(
            d_model=self.config.d_model,
            d_context=self.config.context_dim if self.config.context_dim > 0 else None,
            dropout=self.config.dropout,
        )
        self.horizon_score = nn.Linear(self.config.d_model, 1)
        self.horizon_weight = nn.Linear(self.config.d_model, 1)
        self.context_to_temp = None
        if self.config.context_dim > 0:
            hidden = max(16, min(128, self.config.context_dim))
            self.context_to_temp = nn.Sequential(
                nn.Linear(self.config.context_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            )
        self._last_stats: Dict[str, float] = {}

    @staticmethod
    def _derive_features(quantiles: torch.Tensor) -> torch.Tensor:
        q_lo = quantiles[..., 0]
        q_hi = quantiles[..., -1]
        q_mid_lo = quantiles[..., max(0, quantiles.shape[-1] // 2 - 1)]
        q_mid_hi = quantiles[..., min(quantiles.shape[-1] - 1, quantiles.shape[-1] // 2)]
        center = 0.5 * (q_mid_lo + q_mid_hi)
        spread = q_hi - q_lo
        upside = F.relu(q_hi)
        downside = F.relu(-q_lo)
        skew = (q_hi + q_lo) - (q_mid_lo + q_mid_hi)
        confidence = center / (spread.abs() + 1e-6)
        return torch.cat(
            [
                quantiles,
                center.unsqueeze(-1),
                spread.unsqueeze(-1),
                upside.unsqueeze(-1),
                downside.unsqueeze(-1),
                skew.unsqueeze(-1),
                confidence.unsqueeze(-1),
            ],
            dim=-1,
        )

    def forward(
        self,
        quantiles: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if quantiles.dim() != 3:
            raise ValueError(f"quantiles must have shape [B,H,Q], got {tuple(quantiles.shape)}")
        batch_size, horizon_steps, num_quantiles = quantiles.shape
        if horizon_steps != self.horizon_steps:
            raise ValueError(f"Expected horizon_steps={self.horizon_steps}, got {horizon_steps}")
        if num_quantiles != self.num_quantiles:
            raise ValueError(f"Expected {self.num_quantiles} quantiles, got {num_quantiles}")

        derived = self._derive_features(quantiles)
        hidden = self.feature_proj(derived)
        horizon_ids = torch.arange(horizon_steps, device=quantiles.device, dtype=torch.long)
        hidden = hidden + self.horizon_emb(horizon_ids).unsqueeze(0)

        flat_hidden = hidden.reshape(batch_size * horizon_steps, -1)
        if context is not None and self.config.context_dim > 0:
            flat_context = context.unsqueeze(1).expand(batch_size, horizon_steps, -1).reshape(batch_size * horizon_steps, -1)
        else:
            flat_context = None
        encoded = self.horizon_grn(flat_hidden, context=flat_context).view(batch_size, horizon_steps, -1)

        horizon_scores = self.horizon_score(encoded).squeeze(-1)
        horizon_logits = self.horizon_weight(encoded).squeeze(-1)
        horizon_weights = F.softmax(horizon_logits, dim=-1)

        score = torch.einsum("bh,bh->b", horizon_weights, horizon_scores)
        if context is not None and self.context_to_temp is not None:
            temp_scale = 1.0 + 0.30 * self.context_to_temp(context).squeeze(-1)
        else:
            temp_scale = torch.ones_like(score)
        temperature = (self.config.base_temperature * temp_scale).clamp(min=0.5, max=3.0)
        position = torch.tanh(temperature * score).unsqueeze(-1)

        self._last_stats = {
            "temperature_mean": temperature.detach().mean().item(),
            "score_mean": score.detach().mean().item(),
            "avg_abs_position": position.detach().abs().mean().item(),
        }
        return position, score.unsqueeze(-1), horizon_weights

    def get_last_stats(self) -> Dict[str, float]:
        return dict(self._last_stats)


class SinglePointClassHead(nn.Module):
    """Auxiliary 4-class head for single-point 30-minute cumulative return classification."""

    def __init__(
        self,
        *,
        d_input: int,
        d_model: int,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.net = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class EventPhaseTFT(nn.Module):
    """
    Event-specific TFT variant with separate Pre/Event/Post VSNs.

    Expected inputs:
      - phase_features: [B, P, T, F]
      - known_future: [B, P, K] (optional)
      - phase_ids: [B, P] (optional; defaults to 0..P-1)
      - target_returns: [B, H] for quantile regression loss
    """

    def __init__(self, config: EventPhaseTFTConfig):
        super().__init__()
        self.config = config
        if self.config.num_phases != len(self.config.phase_names):
            raise ValueError(
                f"num_phases={self.config.num_phases} does not match "
                f"len(phase_names)={len(self.config.phase_names)}"
            )
        if self.config.phase_seq_len < 1:
            raise ValueError("phase_seq_len must be >= 1")
        self.config.quantiles = _validate_quantiles(self.config.quantiles)

        self.register_buffer(
            "quantile_levels",
            torch.tensor(self.config.quantiles, dtype=torch.float32),
        )

        self.static_encoder = EventStaticContextEncoder(
            d_model=self.config.d_model,
            static_real_dim=self.config.static_real_dim,
            d_static_emb=self.config.d_static_emb,
            n_contract_months=self.config.n_contract_months,
            n_vol_regimes=self.config.n_vol_regimes,
            dropout=self.config.dropout,
        )

        self.phase_vsns = nn.ModuleList(
            [
                PhaseSpecificVariableSelection(
                    input_dim=self.config.input_dim,
                    d_model=self.config.d_model,
                    d_context=self.config.d_model,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_phases)
            ]
        )
        self.phase_lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=self.config.d_model,
                    hidden_size=self.config.d_model,
                    batch_first=True,
                )
                for _ in range(self.config.num_phases)
            ]
        )

        self.temporal_context = PhaseTemporalContextEncoder(
            num_phases=self.config.num_phases,
            known_future_dim=self.config.known_future_dim,
            d_model=self.config.d_model,
            d_phase_emb=self.config.d_phase_emb,
            d_context=self.config.d_model,
            dropout=self.config.dropout,
        )
        self.technical_encoder = None
        if self.config.technical_input_dim > 0:
            if self.config.technical_seq_len < 1:
                raise ValueError("technical_seq_len must be >= 1 when technical_input_dim > 0")
            self.technical_encoder = TechnicalIndicatorBranchEncoder(
                input_dim=self.config.technical_input_dim,
                d_model=self.config.d_model,
                d_context=self.config.d_model,
                dropout=self.config.dropout,
            )

        self.phase_selector = BranchVariableSelection(
            n_branches=self.config.num_phases,
            d_branch=self.config.d_model,
            d_context=self.config.d_model,
            dropout=self.config.dropout,
        )

        self.enrichment_ctx = GatedResidualNetwork(
            d_model=self.config.d_model,
            d_input=self.config.d_model * 2,
            dropout=self.config.dropout,
        )
        self.temporal_attention = MacroEnrichedTemporalAttention(
            d_model=self.config.d_model,
            d_context=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
        )
        self.phase_pool = nn.Linear(self.config.d_model, 1)
        self.cross_fusion = GatedCrossAttentionFusion(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
        )
        self.output_head = MultiHorizonQuantileHead(
            d_input=self.config.d_model * 2,
            d_model=self.config.d_hidden,
            horizon_steps=self.config.horizon_steps,
            quantiles=self.config.quantiles,
            dropout=self.config.dropout,
        )
        self.single_point_head = None
        if self.config.single_point_num_classes > 0:
            self.single_point_head = SinglePointClassHead(
                d_input=self.config.d_model * 2,
                d_model=self.config.d_hidden,
                num_classes=self.config.single_point_num_classes,
                dropout=self.config.dropout,
            )

        self._last_tracker: Dict[str, object] = {}

    def _forward_core(
        self,
        *,
        phase_features: torch.Tensor,
        technical_features: Optional[torch.Tensor] = None,
        known_future: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,
        contract_month: Optional[torch.Tensor] = None,
        vol_regime: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if phase_features.dim() != 4:
            raise ValueError(
                f"phase_features must have shape [B,P,T,F], got {tuple(phase_features.shape)}"
            )
        batch_size, num_phases, seq_len, feat_dim = phase_features.shape
        if num_phases != self.config.num_phases:
            raise ValueError(f"Expected {self.config.num_phases} phases, got {num_phases}")
        if feat_dim != self.config.input_dim:
            raise ValueError(f"Expected input_dim={self.config.input_dim}, got {feat_dim}")
        if seq_len != self.config.phase_seq_len:
            raise ValueError(f"Expected phase_seq_len={self.config.phase_seq_len}, got {seq_len}")
        if self.technical_encoder is not None:
            if technical_features is None:
                raise ValueError("technical_features are required when technical_input_dim > 0")
            if technical_features.shape[0] != batch_size:
                raise ValueError("technical_features batch dimension must match phase_features")
            if technical_features.shape[1] != self.config.technical_seq_len:
                raise ValueError(
                    f"Expected technical_seq_len={self.config.technical_seq_len}, "
                    f"got {technical_features.shape[1]}"
                )
            if technical_features.shape[2] != self.config.technical_input_dim:
                raise ValueError(
                    f"Expected technical_input_dim={self.config.technical_input_dim}, "
                    f"got {technical_features.shape[2]}"
                )

        if phase_ids is None:
            phase_ids = torch.arange(num_phases, device=phase_features.device, dtype=torch.long)
            phase_ids = phase_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            phase_ids = phase_ids.to(device=phase_features.device, dtype=torch.long)

        c_s, c_e, c_c, c_h = self.static_encoder(
            batch_size=batch_size,
            device=phase_features.device,
            contract_month=contract_month,
            vol_regime=vol_regime,
            static_real=static_real,
        )

        phase_tokens = []
        avg_vsn_weights: Dict[str, torch.Tensor] = {}
        for phase_idx, phase_name in enumerate(self.config.phase_names):
            selected_seq, feature_weights = self.phase_vsns[phase_idx](
                phase_features[:, phase_idx],
                context=c_s,
            )
            lstm = self.phase_lstms[phase_idx]
            h0 = c_h.unsqueeze(0)
            c0 = c_c.unsqueeze(0)
            _, (h_n, _) = lstm(selected_seq, (h0, c0))
            phase_tokens.append(h_n[-1])
            avg_vsn_weights[phase_name] = feature_weights.mean(dim=(0, 1)).detach()

        local_phase_tokens = torch.stack(phase_tokens, dim=1)
        context_tokens = self.temporal_context(
            phase_ids=phase_ids,
            known_future=known_future,
            context=c_e,
        )
        phase_sequence = local_phase_tokens + context_tokens

        selected_phase, phase_weights = self.phase_selector(
            branch_outputs=[local_phase_tokens[:, idx, :] for idx in range(num_phases)],
            context=c_s,
        )
        technical_token = None
        if self.technical_encoder is not None:
            technical_token = self.technical_encoder(
                technical_features,
                context=c_s,
                h0=c_h,
                c0=c_c,
            )

        enrichment_ctx = self.enrichment_ctx(
            torch.cat([c_e, context_tokens.mean(dim=1)], dim=-1)
        )
        attended_seq, attn_weights = self.temporal_attention(
            daily_tokens=phase_sequence,
            macro_context=enrichment_ctx,
            causal_mask=self.config.causal_phase_attention,
        )

        pool_logits = self.phase_pool(attended_seq).squeeze(-1)
        pool_weights = F.softmax(pool_logits, dim=-1)
        temporal_token = torch.einsum("bp,bpd->bd", pool_weights, attended_seq)

        kv_list = [selected_phase, temporal_token]
        if technical_token is not None:
            kv_list.append(technical_token)
        kv_tokens = torch.stack(kv_list, dim=1)
        fused = self.cross_fusion(
            macro_query=enrichment_ctx,
            branch_kv=kv_tokens,
        )

        head_input = torch.cat([fused, selected_phase], dim=-1)
        quantiles = self.output_head(head_input)
        single_point_logits = None
        single_point_probs = None
        if self.single_point_head is not None:
            single_point_logits, single_point_probs = self.single_point_head(head_input)

        self._last_tracker = {
            "phase_weights": {
                name: phase_weights[:, idx].mean().item()
                for idx, name in enumerate(self.config.phase_names)
            },
            "phase_vsn_feature_weights": {
                name: weights.cpu()
                for name, weights in avg_vsn_weights.items()
            },
            "phase_pool_weights": {
                name: pool_weights[:, idx].mean().item()
                for idx, name in enumerate(self.config.phase_names)
            },
            "temporal_attention": attn_weights.detach(),
            "cross_attn_gate": (
                self.cross_fusion.last_gate_value.mean().item()
                if self.cross_fusion.last_gate_value is not None
                else None
            ),
            "has_technical_branch": technical_token is not None,
            "has_single_point_head": self.single_point_head is not None,
        }
        return {
            "quantiles": quantiles,
            "single_point_logits": single_point_logits,
            "single_point_probs": single_point_probs,
            "head_input": head_input,
            "selected_phase": selected_phase,
            "temporal_token": temporal_token,
            "technical_token": technical_token,
            "enrichment_context": enrichment_ctx,
            "fused": fused,
        }

    def forward(
        self,
        *,
        phase_features: torch.Tensor,
        technical_features: Optional[torch.Tensor] = None,
        known_future: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,
        contract_month: Optional[torch.Tensor] = None,
        vol_regime: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
        return_tracker: bool = False,
    ):
        outputs = self._forward_core(
            phase_features=phase_features,
            technical_features=technical_features,
            known_future=known_future,
            phase_ids=phase_ids,
            contract_month=contract_month,
            vol_regime=vol_regime,
            static_real=static_real,
        )
        if return_tracker:
            return outputs["quantiles"], self._last_tracker
        return outputs["quantiles"]

    def compute_loss(
        self,
        predicted_quantiles: torch.Tensor,
        target_returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if target_returns.dim() != 2:
            raise ValueError(
                f"target_returns must have shape [B,H], got {tuple(target_returns.shape)}"
            )
        if predicted_quantiles.dim() != 3:
            raise ValueError(
                f"predicted_quantiles must have shape [B,H,Q], got {tuple(predicted_quantiles.shape)}"
            )
        if predicted_quantiles.shape[:2] != target_returns.shape:
            raise ValueError(
                f"predicted_quantiles {tuple(predicted_quantiles.shape[:2])} "
                f"must align with target_returns {tuple(target_returns.shape)}"
            )
        if predicted_quantiles.shape[-1] != self.quantile_levels.numel():
            raise ValueError(
                f"Expected {self.quantile_levels.numel()} quantiles, got {predicted_quantiles.shape[-1]}"
            )

        error = target_returns.unsqueeze(-1) - predicted_quantiles
        raw_loss = pinball_loss(error, self.quantile_levels)
        monotonicity_penalty = F.relu(
            predicted_quantiles[..., :-1] - predicted_quantiles[..., 1:]
        ).mean()
        total_loss = raw_loss + 0.05 * monotonicity_penalty
        return {
            "total_loss": total_loss,
            "raw_loss": raw_loss,
            "monotonicity_penalty": monotonicity_penalty,
        }

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)


class JointEventPhaseTFT(nn.Module):
    """EventPhaseTFT trunk plus trainable quantile-to-exposure mapper."""

    def __init__(self, config: JointEventPhaseTFTConfig):
        super().__init__()
        self.config = config
        self.trunk = EventPhaseTFT(config.trunk)
        exposure_cfg = QuantileExposureConfig(
            horizon_steps=config.exposure.horizon_steps,
            quantiles=config.exposure.quantiles,
            d_model=config.exposure.d_model,
            dropout=config.exposure.dropout,
            base_temperature=config.exposure.base_temperature,
            context_dim=config.trunk.d_model * 2,
        )
        self.exposure_head = QuantileToExposureHead(exposure_cfg)
        self._last_tracker: Dict[str, object] = {}

    def forward(
        self,
        *,
        phase_features: torch.Tensor,
        technical_features: Optional[torch.Tensor] = None,
        known_future: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,
        contract_month: Optional[torch.Tensor] = None,
        vol_regime: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
        return_tracker: bool = False,
    ):
        outputs = self.trunk._forward_core(
            phase_features=phase_features,
            technical_features=technical_features,
            known_future=known_future,
            phase_ids=phase_ids,
            contract_month=contract_month,
            vol_regime=vol_regime,
            static_real=static_real,
        )
        position, score, horizon_weights = self.exposure_head(
            outputs["quantiles"],
            context=outputs["head_input"],
        )

        self._last_tracker = {
            **self.trunk.get_last_tracker(),
            "exposure_head": {
                **self.exposure_head.get_last_stats(),
                "horizon_weights": horizon_weights.detach(),
            },
        }
        result = {
            "quantiles": outputs["quantiles"],
            "position": position,
            "score": score,
            "head_input": outputs["head_input"],
            "single_point_logits": outputs["single_point_logits"],
            "single_point_probs": outputs["single_point_probs"],
        }
        if return_tracker:
            return result, self._last_tracker
        return result

    def _resolve_exposure_returns(
        self,
        target_returns: torch.Tensor,
        exposure_returns: Optional[torch.Tensor] = None,
        horizon_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if exposure_returns is not None:
            return exposure_returns.reshape(-1)

        if horizon_weights is None:
            if self.config.exposure.target_horizon_weights is not None:
                horizon_weights = torch.tensor(
                    self.config.exposure.target_horizon_weights,
                    device=target_returns.device,
                    dtype=target_returns.dtype,
                )
            else:
                horizon_weights = torch.arange(
                    1,
                    target_returns.shape[1] + 1,
                    device=target_returns.device,
                    dtype=target_returns.dtype,
                )
        weights = horizon_weights / horizon_weights.sum().clamp(min=1e-8)
        return torch.einsum("bh,h->b", target_returns, weights)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_returns: torch.Tensor,
        *,
        exposure_returns: Optional[torch.Tensor] = None,
        horizon_weights: Optional[torch.Tensor] = None,
        prev_position: Optional[torch.Tensor] = None,
        single_point_targets: Optional[torch.Tensor] = None,
        quantile_weight: float = 1.0,
        exposure_weight: float = 1.0,
        single_point_weight: float = 0.0,
        trading_loss: Optional[nn.Module] = None,
        single_point_loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        quantile_terms = self.trunk.compute_loss(outputs["quantiles"], target_returns)
        realized_exposure_returns = self._resolve_exposure_returns(
            target_returns,
            exposure_returns=exposure_returns,
            horizon_weights=horizon_weights,
        )
        trading_module = trading_loss if trading_loss is not None else ContinuousTradingLoss()
        exposure_loss, trading_metrics = trading_module(
            outputs["position"],
            realized_exposure_returns,
            prev_position=prev_position,
        )
        single_point_loss = outputs["position"].new_zeros(())
        single_point_accuracy = outputs["position"].new_zeros(())
        if single_point_targets is not None:
            logits = outputs.get("single_point_logits")
            if logits is None:
                raise ValueError("single_point_targets provided but model has no single-point head enabled")
            target_labels = single_point_targets.view(-1).to(logits.device, dtype=torch.long)
            cls_loss_fn = single_point_loss_fn if single_point_loss_fn is not None else nn.CrossEntropyLoss()
            single_point_loss = cls_loss_fn(logits, target_labels)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                single_point_accuracy = (preds == target_labels).float().mean()

        total_loss = (
            quantile_weight * quantile_terms["total_loss"]
            + exposure_weight * exposure_loss
            + single_point_weight * single_point_loss
        )
        loss_dict = {
            "total_loss": total_loss,
            "quantile_loss": quantile_terms["total_loss"],
            "raw_quantile_loss": quantile_terms["raw_loss"],
            "monotonicity_penalty": quantile_terms["monotonicity_penalty"],
            "exposure_loss": exposure_loss,
            "realized_exposure_returns": realized_exposure_returns,
            "trading_metrics": trading_metrics,
        }
        if single_point_targets is not None:
            loss_dict["single_point_loss"] = single_point_loss
            loss_dict["single_point_accuracy"] = single_point_accuracy
        return loss_dict

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)

    def set_exposure_trainable(self, trainable: bool = True) -> None:
        for param in self.exposure_head.parameters():
            param.requires_grad = bool(trainable)

    def freeze_exposure_head(self) -> None:
        self.set_exposure_trainable(False)

    def unfreeze_exposure_head(self) -> None:
        self.set_exposure_trainable(True)


# =====================================================================
# Classification-Primary Model for Continuous Intraday Trading
# =====================================================================


@dataclass
class ClassificationEventTFTConfig:
    """Config for ClassificationEventTFT.

    The trunk config controls the phase-orderflow encoder (VSNs, LSTMs,
    attention).  The classification head processes per-bar features
    conditioned on the shared event context.

    Parameters
    ----------
    trunk : EventPhaseTFTConfig
        Controls phase-orderflow encoder.  ``horizon_steps`` and
        ``quantiles`` are ignored — the quantile head is not built.
    num_classes : int
        Ordinal classification buckets (e.g. 5 from 4 quantile thresholds).
    bar_feature_dim : int
        Per-bar feature dimension (technicals + price context).
    session_close_hour : int
        Hour (ET) at which trading stops.  Bars after this are dropped.
    position_weights : tuple of float or None
        Per-class position map, length ``num_classes``.  Default: linear
        from −1 to +1.  E.g. for 5 classes: (−1, −0.5, 0, 0.5, 1).
    tc_cost : float
        Transaction cost for ``ContinuousTradingLoss``.
    direction_weight : float
        Direction accuracy weight in ``ContinuousTradingLoss``.
    reg_weight : float
        Exposure regularization weight.
    target_exposure : float
        Target average |position|.
    ce_weight : float
        Weight of classification CE in the combined loss.
    trading_weight : float
        Weight of ``ContinuousTradingLoss`` in the combined loss.
    use_sortino : bool
        Use Sortino instead of Sharpe in the trading loss.
    """

    trunk: EventPhaseTFTConfig
    num_classes: int = 5
    bar_feature_dim: int = 9
    session_close_hour: int = 17
    position_weights: Optional[Tuple[float, ...]] = None
    tc_cost: float = 0.001
    direction_weight: float = 0.5
    reg_weight: float = 0.1
    target_exposure: float = 0.3
    ce_weight: float = 1.0
    trading_weight: float = 1.0
    use_sortino: bool = False

    def __post_init__(self):
        if self.position_weights is None:
            n = self.num_classes
            self.position_weights = tuple(
                -1.0 + 2.0 * i / (n - 1) for i in range(n)
            )
        if len(self.position_weights) != self.num_classes:
            raise ValueError(
                f"position_weights length {len(self.position_weights)} "
                f"!= num_classes {self.num_classes}"
            )


class BarClassificationHead(nn.Module):
    """Per-bar classifier conditioned on shared event context.

    Takes (event_context, bar_features) → class logits.  Accepts both
    single-bar ``(B, F_bar)`` and multi-bar ``(B, N, F_bar)`` inputs.
    """

    def __init__(
        self,
        *,
        d_event: int,
        bar_feature_dim: int,
        d_hidden: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bar_proj = nn.Sequential(
            nn.Linear(bar_feature_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = GatedResidualNetwork(
            d_model=d_hidden,
            d_input=d_hidden + d_event,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, num_classes),
        )

    def forward(
        self,
        event_context: torch.Tensor,
        bar_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        event_context : (B, d_event)
        bar_features : (B, F_bar) single bar per sample, or (B, N, F_bar)

        Returns
        -------
        logits : (B, C) or (B, N, C) matching input rank.
        """
        squeezed = bar_features.dim() == 2
        if squeezed:
            bar_features = bar_features.unsqueeze(1)  # (B, 1, F)

        B, N, _ = bar_features.shape
        bar_hidden = self.bar_proj(bar_features)
        ctx = event_context.unsqueeze(1).expand(B, N, -1)
        fused = self.fuse(
            torch.cat([bar_hidden, ctx], dim=-1).reshape(B * N, -1)
        ).view(B, N, -1)
        logits = self.head(fused)

        if squeezed:
            return logits.squeeze(1)  # (B, C)
        return logits


class ClassificationEventTFT(nn.Module):
    """Classification-primary EventPhaseTFT for continuous intraday trading.

    Each sample is **one bar** carrying its event day's phase orderflow as
    shared context.  The batch contains many bars (potentially from
    different event days).  ``ContinuousTradingLoss`` optimises the
    Sharpe ratio across the batch, identical to the v3 pipeline.

    Architecture
    ------------
    1. **Trunk** — ``EventPhaseTFT._forward_core`` encodes phase orderflow
       ``[B, P, 128, F]`` → ``event_context [B, d_model*2]``.
    2. **Per-bar head** — ``BarClassificationHead`` takes
       ``(event_context, bar_features [B, F_bar])`` → logits ``[B, C]``.
    3. **Position** — ``Σ weight_i × p_i`` maps class probs to [−1, +1].
    4. **Loss** — ``CE + ContinuousTradingLoss`` over the batch.

    At end-of-day the position must flatten to zero.  Pass
    ``is_last_bar=True`` in the batch to apply a forced-flat with
    transaction cost penalty.
    """

    def __init__(self, config: ClassificationEventTFTConfig):
        super().__init__()
        self.config = config

        # Build trunk — quantile head will be created but unused during
        # forward; we only call _forward_core for the fused representation.
        self.trunk = EventPhaseTFT(config.trunk)

        # Per-bar classification head
        self.bar_head = BarClassificationHead(
            d_event=config.trunk.d_model * 2,
            bar_feature_dim=config.bar_feature_dim,
            d_hidden=config.trunk.d_hidden,
            num_classes=config.num_classes,
            dropout=config.trunk.dropout,
        )

        # Position weights: maps class probs → continuous position in [-1, 1]
        self.register_buffer(
            "position_weights",
            torch.tensor(config.position_weights, dtype=torch.float32),
        )

        # Trading loss
        self.trading_loss_fn = ContinuousTradingLoss(
            tc_cost=config.tc_cost,
            direction_weight=config.direction_weight,
            reg_weight=config.reg_weight,
            target_exposure=config.target_exposure,
            use_sortino=config.use_sortino,
        )

        self._last_tracker: Dict[str, object] = {}

    def forward(
        self,
        *,
        phase_features: torch.Tensor,
        bar_features: torch.Tensor,
        technical_features: Optional[torch.Tensor] = None,
        known_future: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,
        contract_month: Optional[torch.Tensor] = None,
        vol_regime: Optional[torch.Tensor] = None,
        static_real: Optional[torch.Tensor] = None,
        return_tracker: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        phase_features : (B, P, 128, F)
            Orderflow chunks — each sample carries its event day's phases.
        bar_features : (B, F_bar)
            Per-bar technical + price context features.
        technical_features : (B, T_tech, F_tech), optional
            Sequential technical window for the trunk encoder.
        known_future, phase_ids, contract_month, vol_regime, static_real :
            Passed through to trunk.

        Returns
        -------
        dict with logits (B, C), probs (B, C), position (B,),
        event_context (B, d_event).
        """
        # 1. Trunk — encode phase orderflow
        trunk_out = self.trunk._forward_core(
            phase_features=phase_features,
            technical_features=technical_features,
            known_future=known_future,
            phase_ids=phase_ids,
            contract_month=contract_month,
            vol_regime=vol_regime,
            static_real=static_real,
        )
        event_context = trunk_out["head_input"]  # (B, d_model*2)

        # 2. Per-bar classification
        logits = self.bar_head(event_context, bar_features)  # (B, C)
        probs = F.softmax(logits, dim=-1)

        # 3. Position from class probs
        position = (probs * self.position_weights).sum(dim=-1)  # (B,)

        self._last_tracker = {
            **self.trunk.get_last_tracker(),
            "avg_abs_position": position.detach().abs().mean().item(),
            "mean_position": position.detach().mean().item(),
        }

        result = {
            "logits": logits,
            "probs": probs,
            "position": position,
            "event_context": event_context,
        }
        if return_tracker:
            return result, self._last_tracker
        return result

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_returns: torch.Tensor,
        target_classes: torch.Tensor,
        *,
        prev_position: Optional[torch.Tensor] = None,
        is_last_bar: Optional[torch.Tensor] = None,
        ce_loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined classification + trading loss.

        Parameters
        ----------
        outputs : dict from ``forward()``.
        target_returns : (B,) rolling 30-min forward log return.
        target_classes : (B,) ordinal class labels.
        prev_position : (B,), optional
            Position at previous bar (for turnover penalty).
        is_last_bar : (B,) bool, optional
            Marks end-of-day bars.  Position is forced to 0 and the
            flattening turnover incurs transaction cost.
        ce_loss_fn : nn.Module, optional
            Custom CE loss.  Default: standard cross-entropy.
        """
        logits = outputs["logits"]  # (B, C)
        position = outputs["position"]  # (B,)
        B = logits.shape[0]

        # ── End-of-day flatten ─────────────────────────────────────
        # On last-bar samples, force position to 0 so the trading loss
        # sees the cost of unwinding.
        if is_last_bar is not None:
            eod_mask = is_last_bar.bool()
            position = torch.where(eod_mask, torch.zeros_like(position), position)

        # ── Classification loss ────────────────────────────────────
        targets_long = target_classes.long()
        if ce_loss_fn is not None:
            ce_loss = ce_loss_fn(logits, targets_long)
            if ce_loss.dim() > 0:
                ce_loss = ce_loss.mean()
        else:
            ce_loss = F.cross_entropy(logits, targets_long)

        # ── Trading loss (across batch) ────────────────────────────
        trading_loss, trading_metrics = self.trading_loss_fn(
            position,
            target_returns,
            prev_position=prev_position,
        )

        # ── Combined ───────────────────────────────────────────────
        total = (
            self.config.ce_weight * ce_loss
            + self.config.trading_weight * trading_loss
        )

        # ── Diagnostics ───────────────────────────────────────────
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == targets_long).float().mean()

        return {
            "total_loss": total,
            "ce_loss": ce_loss,
            "trading_loss": trading_loss,
            "accuracy": accuracy,
            **{f"trading/{k}": v for k, v in trading_metrics.items()},
        }

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)
