"""TFT-style crack spread classifier with anti-collapse regularization."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalized_entropy(
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return entropy normalized to [0, 1] along the last dimension."""
    n = weights.shape[-1]
    if n <= 1:
        return torch.ones_like(weights[..., 0])
    entropy = -(weights * (weights.clamp_min(eps)).log()).sum(dim=-1)
    max_entropy = math.log(float(n))
    return entropy / max(max_entropy, eps)


def _apply_weight_floor(
    weights: torch.Tensor,
    min_weight: float,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Clamp a minimum weight floor and renormalize."""
    if min_weight <= 0:
        return weights

    out = weights
    if valid_mask is not None:
        valid_mask_f = valid_mask.to(dtype=weights.dtype)
        valid_count = valid_mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        max_floor = (1.0 / valid_count) - eps
        floor = torch.minimum(
            torch.full_like(valid_count, float(min_weight)),
            max_floor,
        )
        out = torch.where(valid_mask, torch.maximum(out, floor), torch.zeros_like(out))
        denom = out.sum(dim=-1, keepdim=True).clamp_min(eps)
        return out / denom

    out = out.clamp(min=min_weight)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(eps)


class GatedResidualNetwork(nn.Module):
    """Core TFT-style building block with optional context injection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate_fc = nn.Linear(hidden_dim, output_dim)
        self.ctx_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim is not None else None
        self.skip_proj = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else None
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x if self.skip_proj is None else self.skip_proj(x)
        h = self.fc1(x)
        if context is not None and self.ctx_proj is not None:
            h = h + self.ctx_proj(context)
        h = F.elu(h)
        h = self.dropout(h)
        v = self.fc2(h)
        g = torch.sigmoid(self.gate_fc(h))
        return self.layer_norm(residual + g * v)


class AntiCollapseVariableSelectionNetwork(nn.Module):
    """Per-time-step variable selection with entropy and minimum-weight regularization."""

    def __init__(
        self,
        n_vars: int,
        var_dim: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        temperature: float = 1.5,
        min_weight: float = 0.02,
        entropy_weight: float = 0.02,
        use_pre_norm: bool = True,
    ):
        super().__init__()
        self.n_vars = int(n_vars)
        self.temperature = float(temperature)
        self.min_weight = float(min_weight)
        self.entropy_weight = float(entropy_weight)

        self.pre_norms = nn.ModuleList([
            nn.LayerNorm(var_dim) for _ in range(self.n_vars)
        ]) if use_pre_norm else None
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=var_dim,
                hidden_dim=hidden_dim,
                output_dim=var_dim,
                context_dim=context_dim,
                dropout=dropout,
            )
            for _ in range(self.n_vars)
        ])
        self.selection_grn = GatedResidualNetwork(
            input_dim=self.n_vars * var_dim,
            hidden_dim=hidden_dim,
            output_dim=self.n_vars,
            context_dim=context_dim,
            dropout=dropout,
        )

        self.last_weights: Optional[torch.Tensor] = None
        self.last_entropy: Optional[torch.Tensor] = None
        self.last_entropy_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        inputs: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(inputs) != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {len(inputs)}")

        if self.pre_norms is not None:
            normed_inputs = [norm(x) for norm, x in zip(self.pre_norms, inputs)]
        else:
            normed_inputs = inputs

        processed = [
            grn(inp, context)
            for grn, inp in zip(self.var_grns, normed_inputs)
        ]
        flat = torch.cat(normed_inputs, dim=-1)
        logits = self.selection_grn(flat, context)
        weights = F.softmax(logits / self.temperature, dim=-1)
        weights = _apply_weight_floor(weights, self.min_weight)

        entropy = _normalized_entropy(weights)
        entropy_loss = self.entropy_weight * (1.0 - entropy).mean()
        self.last_weights = weights.detach()
        self.last_entropy = entropy.mean().detach()
        self.last_entropy_loss = entropy_loss

        fused = (weights.unsqueeze(-1) * torch.stack(processed, dim=-2)).sum(dim=-2)
        return fused, weights

    def get_entropy_loss(self) -> torch.Tensor:
        if self.last_entropy_loss is None:
            param = next(self.parameters())
            return param.new_zeros(())
        return self.last_entropy_loss


class CrossAssetOrderflowFusion(nn.Module):
    """Fuse CL / HO / RB orderflow with masked cross-asset attention and entropy regularization."""

    def __init__(
        self,
        n_assets: int,
        n_orderflow_feats: int,
        d_model: int,
        n_heads: int = 2,
        dropout: float = 0.0,
        temperature: float = 1.5,
        min_weight: float = 0.05,
        entropy_weight: float = 0.02,
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_orderflow_feats = int(n_orderflow_feats)
        self.d_model = int(d_model)
        self.temperature = float(temperature)
        self.min_weight = float(min_weight)
        self.entropy_weight = float(entropy_weight)

        self.asset_proj = nn.Linear(n_orderflow_feats, d_model)
        self.asset_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.asset_scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.out_grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model * 2,
            output_dim=d_model,
            dropout=dropout,
        )

        self.last_asset_weights: Optional[torch.Tensor] = None
        self.last_asset_entropy: Optional[torch.Tensor] = None
        self.last_entropy_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        orderflow: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if orderflow.dim() != 4:
            raise ValueError(f"orderflow must have shape [B,T,A,F], got {tuple(orderflow.shape)}")
        bsz, steps, assets, feats = orderflow.shape
        if assets != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} assets, got {assets}")
        if feats != self.n_orderflow_feats:
            raise ValueError(f"Expected {self.n_orderflow_feats} orderflow features, got {feats}")

        x = self.asset_norm(self.asset_proj(orderflow))
        x_flat = x.view(bsz * steps, assets, self.d_model)

        key_padding_mask = None
        valid_mask_flat = None
        if valid_mask is not None:
            if valid_mask.shape != (bsz, steps, assets):
                raise ValueError(
                    f"valid_mask must have shape {(bsz, steps, assets)}, got {tuple(valid_mask.shape)}"
                )
            valid_mask_flat = valid_mask.reshape(bsz * steps, assets)
            key_padding_mask = ~valid_mask_flat

        attn_out, _ = self.cross_attn(
            x_flat,
            x_flat,
            x_flat,
            key_padding_mask=key_padding_mask,
        )
        x_flat = self.post_norm(x_flat + attn_out)

        logits = self.asset_scorer(x_flat).squeeze(-1)
        if valid_mask_flat is not None:
            logits = logits.masked_fill(~valid_mask_flat, float("-inf"))
        weights = F.softmax(logits / self.temperature, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = _apply_weight_floor(weights, self.min_weight, valid_mask=valid_mask_flat)

        entropy = _normalized_entropy(weights)
        entropy_loss = self.entropy_weight * (1.0 - entropy).mean()
        self.last_asset_weights = weights.view(bsz, steps, assets).detach()
        self.last_asset_entropy = entropy.mean().detach()
        self.last_entropy_loss = entropy_loss

        fused = torch.einsum("ba,bad->bd", weights, x_flat)
        fused = self.out_grn(fused).view(bsz, steps, self.d_model)
        return fused, weights.view(bsz, steps, assets)

    def get_entropy_loss(self) -> torch.Tensor:
        if self.last_entropy_loss is None:
            param = next(self.parameters())
            return param.new_zeros(())
        return self.last_entropy_loss


class InterpretableMultiHeadAttention(nn.Module):
    """TFT-style interpretable attention with a shared value projection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.d_k = d_model // n_heads
        self.w_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_heads)])
        self.w_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_heads)])
        self.w_v = nn.Linear(d_model, self.d_k, bias=False)
        self.w_h = nn.Linear(self.d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(float(self.d_k))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.w_v(value)
        head_outputs = []
        head_weights = []
        for q_proj, k_proj in zip(self.w_q, self.w_k):
            q = q_proj(query)
            k = k_proj(key)
            scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
            attn = self.dropout(attn)
            head_weights.append(attn)
            head_outputs.append(torch.bmm(attn, v))
        combined = torch.stack(head_outputs, dim=0).mean(dim=0)
        output = self.w_h(combined)
        weights = torch.stack(head_weights, dim=0).mean(dim=0).detach()
        return output, weights


class TemporalClassificationHead(nn.Module):
    """Sequence classification head for decoder steps."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=d_model,
            dropout=dropout,
        )
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.grn(x))


class CrackSpreadTFT(nn.Module):
    """Temporal Fusion Transformer variant for crack spread classification."""

    def __init__(
        self,
        n_past_features: int,
        n_known_features: int,
        n_orderflow_feats: int,
        n_assets: int = 3,
        encoder_steps: int = 60,
        decoder_steps: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.10,
        n_static_features: int = 0,
        num_classes: int = 3,
        vsn_temperature: float = 1.5,
        vsn_min_weight: float = 0.02,
        vsn_entropy_weight: float = 0.02,
        asset_temperature: float = 1.5,
        asset_min_weight: float = 0.05,
        asset_entropy_weight: float = 0.02,
    ):
        super().__init__()
        if n_past_features < 1:
            raise ValueError("n_past_features must be >= 1")
        if n_known_features < 1:
            raise ValueError("n_known_features must be >= 1")

        self.encoder_steps = int(encoder_steps)
        self.decoder_steps = int(decoder_steps)
        self.n_assets = int(n_assets)
        self.n_orderflow_feats = int(n_orderflow_feats)
        self.n_past_features = int(n_past_features)
        self.n_known_features = int(n_known_features)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        hidden_dim = d_model * 2

        self.of_fusion = CrossAssetOrderflowFusion(
            n_assets=n_assets,
            n_orderflow_feats=n_orderflow_feats,
            d_model=d_model,
            n_heads=max(1, n_heads // 2),
            dropout=dropout,
            temperature=asset_temperature,
            min_weight=asset_min_weight,
            entropy_weight=asset_entropy_weight,
        )

        self.past_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_past_features)])
        self.known_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_known_features)])
        self.n_past_vars = n_past_features + 1

        if n_static_features > 0:
            self.static_proj = nn.Linear(n_static_features, d_model)
            static_ctx_dim = d_model
        else:
            self.static_proj = None
            static_ctx_dim = None

        self.past_vsn = AntiCollapseVariableSelectionNetwork(
            n_vars=self.n_past_vars,
            var_dim=d_model,
            hidden_dim=hidden_dim,
            context_dim=static_ctx_dim,
            dropout=dropout,
            temperature=vsn_temperature,
            min_weight=vsn_min_weight,
            entropy_weight=vsn_entropy_weight,
        )
        self.known_vsn = AntiCollapseVariableSelectionNetwork(
            n_vars=n_known_features,
            var_dim=d_model,
            hidden_dim=hidden_dim,
            context_dim=static_ctx_dim,
            dropout=dropout,
            temperature=vsn_temperature,
            min_weight=vsn_min_weight,
            entropy_weight=vsn_entropy_weight,
        )

        self.encoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.lstm_norm = nn.LayerNorm(d_model)

        self.static_enrichment_grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=d_model,
            context_dim=static_ctx_dim,
            dropout=dropout,
        )
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_gate = nn.Linear(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.pos_grn = GatedResidualNetwork(d_model, hidden_dim, d_model, dropout=dropout)
        self.pos_norm = nn.LayerNorm(d_model)
        self.output_head = TemporalClassificationHead(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.last_tracker: Dict[str, float] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "lstm" in name:
                if param.dim() == 2:
                    nn.init.orthogonal_(param)
                elif param.dim() == 1:
                    nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _project_scalars(
        self,
        x: torch.Tensor,
        projections: nn.ModuleList,
    ) -> List[torch.Tensor]:
        return [proj(x[..., i:i + 1]) for i, proj in enumerate(projections)]

    def _encode_static(self, static_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if static_context is None or self.static_proj is None:
            return None
        return self.static_proj(static_context)

    def _normalize_orderflow_input(
        self,
        orderflow: torch.Tensor,
        orderflow_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if orderflow.dim() != 4:
            raise ValueError(f"orderflow must have rank 4, got {tuple(orderflow.shape)}")

        if orderflow.shape[2] == self.n_assets and orderflow.shape[3] == self.n_orderflow_feats:
            normalized = orderflow
        elif orderflow.shape[1] == self.n_orderflow_feats and orderflow.shape[3] == self.n_assets:
            normalized = orderflow.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(
                "orderflow must be [B,T,A,F] or [B,F,T,A] with matching asset/feature dims; "
                f"got {tuple(orderflow.shape)}"
            )

        if orderflow_mask is None:
            return normalized, None
        if orderflow_mask.shape != normalized.shape[:3]:
            raise ValueError(
                f"orderflow_mask must have shape {tuple(normalized.shape[:3])}, got {tuple(orderflow_mask.shape)}"
            )
        return normalized, orderflow_mask.bool()

    def _normalize_known_future(
        self,
        known_future: torch.Tensor,
        encoder_steps: int,
    ) -> torch.Tensor:
        if known_future.dim() != 3:
            raise ValueError(f"known_future must have shape [B,T,F], got {tuple(known_future.shape)}")
        if known_future.size(-1) != self.n_known_features:
            raise ValueError(
                f"Expected {self.n_known_features} known features, got {known_future.size(-1)}"
            )
        if known_future.size(1) == self.decoder_steps:
            zeros = known_future.new_zeros(known_future.size(0), encoder_steps, known_future.size(2))
            return torch.cat([zeros, known_future], dim=1)
        if known_future.size(1) < encoder_steps:
            raise ValueError(
                f"known_future length must be >= encoder_steps ({encoder_steps}), got {known_future.size(1)}"
            )
        return known_future

    def _causal_mask(self, total_steps: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(total_steps, total_steps, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(0)

    def get_aux_loss(self) -> torch.Tensor:
        loss = self.past_vsn.get_entropy_loss()
        loss = loss + self.known_vsn.get_entropy_loss()
        loss = loss + self.of_fusion.get_entropy_loss()
        return loss

    def forward(
        self,
        past_observed: torch.Tensor,
        known_future: torch.Tensor,
        orderflow: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
        orderflow_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        return_dict: bool = False,
    ):
        if past_observed.dim() != 3:
            raise ValueError(f"past_observed must have shape [B,T,F], got {tuple(past_observed.shape)}")
        if past_observed.size(-1) != self.n_past_features:
            raise ValueError(
                f"Expected {self.n_past_features} past features, got {past_observed.size(-1)}"
            )

        encoder_steps = past_observed.size(1)
        known_future = self._normalize_known_future(known_future, encoder_steps=encoder_steps)
        decoder_steps = known_future.size(1) - encoder_steps
        if decoder_steps <= 0:
            raise ValueError("known_future must include at least one decoder step")

        normalized_orderflow, normalized_mask = self._normalize_orderflow_input(
            orderflow,
            orderflow_mask=orderflow_mask,
        )
        if normalized_orderflow.size(1) != encoder_steps:
            raise ValueError(
                f"orderflow encoder length {normalized_orderflow.size(1)} does not match past_observed {encoder_steps}"
            )

        static_ctx = self._encode_static(static_context)
        static_ctx_t = static_ctx.unsqueeze(1) if static_ctx is not None else None

        of_fused, asset_weights = self.of_fusion(normalized_orderflow, valid_mask=normalized_mask)
        past_vars = self._project_scalars(past_observed, self.past_proj)
        past_vars.append(of_fused)
        past_selected, past_weights = self.past_vsn(past_vars, static_ctx_t)

        known_vars = self._project_scalars(known_future, self.known_proj)
        known_selected, known_weights = self.known_vsn(known_vars, static_ctx_t)
        known_enc = known_selected[:, :encoder_steps, :]
        known_dec = known_selected[:, encoder_steps:, :]

        enc_input = past_selected + known_enc
        enc_out, (h_n, c_n) = self.encoder_lstm(enc_input)
        enc_gate = self.lstm_gate(enc_out)
        enc_gated = self.lstm_norm(enc_gate * enc_out + enc_input)

        dec_out, _ = self.decoder_lstm(known_dec, (h_n, c_n))
        dec_gate = self.lstm_gate(dec_out)
        dec_gated = self.lstm_norm(dec_gate * dec_out + known_dec)

        seq = torch.cat([enc_gated, dec_gated], dim=1)
        seq = self.static_enrichment_grn(seq, static_ctx_t)

        causal_mask = self._causal_mask(seq.size(1), seq.device)
        attn_out, attn_weights = self.attn(seq, seq, seq, mask=causal_mask)
        attn_gated = torch.sigmoid(self.attn_gate(attn_out)) * attn_out
        seq = self.attn_norm(attn_gated + seq)
        seq = self.pos_norm(self.pos_grn(seq) + seq)

        dec_seq = seq[:, encoder_steps:, :]
        logits = self.output_head(dec_seq)
        probs = torch.softmax(logits, dim=-1) if return_probs or return_dict else None
        dec_attn = attn_weights[:, encoder_steps:, :]

        self.last_tracker = {
            "past_vsn_entropy": float(self.past_vsn.last_entropy.item()) if self.past_vsn.last_entropy is not None else 0.0,
            "known_vsn_entropy": float(self.known_vsn.last_entropy.item()) if self.known_vsn.last_entropy is not None else 0.0,
            "asset_entropy": float(self.of_fusion.last_asset_entropy.item()) if self.of_fusion.last_asset_entropy is not None else 0.0,
            "aux_loss": float(self.get_aux_loss().detach().item()),
            "asset_weight_max": float(asset_weights.detach().max().item()),
            "decoder_steps": float(decoder_steps),
        }

        if return_dict:
            return {
                "logits": logits,
                "probs": probs if probs is not None else torch.softmax(logits, dim=-1),
                "attn_weights": dec_attn,
                "past_selection_weights": past_weights.detach(),
                "known_selection_weights": known_weights.detach(),
                "asset_weights": asset_weights.detach(),
                "tracker": dict(self.last_tracker),
            }
        if return_probs:
            return logits, probs, dec_attn
        return logits, dec_attn


def crack_spread_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    ordinal_alpha: float = 0.5,
    anti_collapse_weight: float = 0.05,
    target_probs: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Classification loss with ordinal distance and batch-level anti-collapse regularization."""
    if logits.dim() != 3:
        raise ValueError(f"logits must have shape [B,T,C], got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"targets must have shape [B,T], got {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logits leading dims {tuple(logits.shape[:2])} must match targets {tuple(targets.shape)}"
        )

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1).long()
    valid_mask = flat_targets != ignore_index
    if not torch.any(valid_mask):
        return flat_logits.new_zeros(())

    flat_logits = flat_logits[valid_mask]
    flat_targets = flat_targets[valid_mask]
    ce = F.cross_entropy(
        flat_logits,
        flat_targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )

    with torch.no_grad():
        pred = flat_logits.argmax(dim=-1)
        dist = (pred - flat_targets).abs().float()
        weights = 1.0 + ordinal_alpha * dist

    main_loss = (ce * weights).mean()
    probs = torch.softmax(flat_logits, dim=-1)
    mean_pred = probs.mean(dim=0).clamp_min(eps)
    if target_probs is None:
        target_probs = torch.full_like(mean_pred, 1.0 / mean_pred.numel())
    else:
        target_probs = target_probs.to(device=mean_pred.device, dtype=mean_pred.dtype)
        target_probs = target_probs / target_probs.sum().clamp_min(eps)
    kl = (target_probs * (target_probs.clamp_min(eps).log() - mean_pred.log())).sum()
    return main_loss + anti_collapse_weight * kl
