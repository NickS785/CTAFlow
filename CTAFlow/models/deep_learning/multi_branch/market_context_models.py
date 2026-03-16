"""
TFT Building Blocks
====================

Core components from Temporal Fusion Transformers (Lim et al., 2021)
adapted for multi-branch commodity models.

Classes
-------
GLU : Gated Linear Unit activation
GatedResidualNetwork : Variable-dimension GRN with optional context
EventDecayEncoding : Temporal decay encoding for event recency
BranchVariableSelection : Context-conditioned branch/variable selection
InterpretableMultiHeadAttention : Interpretable multi-head attention
MacroEnrichedTemporalAttention : Self-attention with macro context enrichment
GatedCrossAttentionFusion : Cross-attention fusion with learnable gate
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# GLU (Gated Linear Unit)
# ============================================================================

class GLU(nn.Module):
    """Gated Linear Unit: splits input in half, applies sigmoid gate.

    If input dim == d_model (not 2*d_model), projects to 2*d_model first.

    Parameters
    ----------
    d_model : int
        Expected feature dimension of the input.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


# ============================================================================
# Gated Residual Network (TFT Eq. 3-5)
# ============================================================================

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network from TFT.

    Applies: LayerNorm(x + GLU(Dropout(W2 * ELU(W1 * x + b1) + b2)))

    Supports optional context conditioning and input projection when
    d_input != d_model.

    Parameters
    ----------
    d_model : int
        Output (and residual) dimension.
    d_input : int, optional
        Input dimension. If None, assumes d_input == d_model.
    d_hidden : int, optional
        Hidden layer dimension. If None, uses d_model.
    d_context : int, optional
        Context vector dimension for conditioning. If None, no context.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_input: Optional[int] = None,
        d_hidden: Optional[int] = None,
        d_context: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_input = d_input or d_model
        d_hidden = d_hidden or d_model

        self.fc1 = nn.Linear(d_input, d_hidden)

        if d_context is not None:
            self.context_proj = nn.Linear(d_context, d_hidden, bias=False)
        else:
            self.context_proj = None

        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)

        # Skip connection projection when dimensions differ
        if d_input != d_model:
            self.skip_proj = nn.Linear(d_input, d_model)
        else:
            self.skip_proj = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x if self.skip_proj is None else self.skip_proj(x)

        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)

        return self.norm(residual + h)


# ============================================================================
# Variable Selection Network (TFT-style per-feature gating)
# ============================================================================

class VariableSelectionNetwork(nn.Module):
    """TFT-style Variable Selection Network for individual features.

    Each input variable is processed through its own GRN, then a softmax
    selection network produces per-variable weights conditioned on an
    optional context vector.  The output is the weighted sum of the
    transformed variables.

    Parameters
    ----------
    n_vars : int
        Number of input variables (features).
    d_model : int
        Output dimension per variable after GRN transformation.
    d_context : int, optional
        Context vector dimension for conditioning the selection weights.
        If None, selection is unconditional.
    dropout : float
        Dropout rate for GRNs and the flattened input projection.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        d_context: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        # Per-variable GRN: each scalar feature → d_model vector
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                d_model=d_model, d_input=1, d_hidden=d_model, dropout=dropout,
            )
            for _ in range(n_vars)
        ])

        # Selection network: operates on flattened raw input + optional context
        # to produce per-variable softmax weights
        self.selection_grn = GatedResidualNetwork(
            d_model=n_vars,
            d_input=n_vars,
            d_hidden=n_vars,
            d_context=d_context,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            (B, L, n_vars) or (B, n_vars) raw feature values.
        context : Tensor, optional
            (B, d_context) context vector for conditioned selection.
            Broadcast to match sequence length if x is 3D.

        Returns
        -------
        selected : Tensor
            (B, L, d_model) or (B, d_model) weighted combination.
        weights : Tensor
            (B, L, n_vars) or (B, n_vars) softmax selection weights.
        """
        has_seq = x.dim() == 3
        if has_seq:
            B, L, _ = x.shape
        else:
            B = x.shape[0]
            L = None

        # 1. Per-variable GRN transformation
        # Split each feature as (*, 1) → GRN → (*, d_model)
        transformed = []
        for i, grn in enumerate(self.var_grns):
            xi = x[..., i : i + 1]            # (B, L, 1) or (B, 1)
            transformed.append(grn(xi))        # (B, L, d_model) or (B, d_model)

        # Stack: (B, L, n_vars, d_model) or (B, n_vars, d_model)
        transformed = torch.stack(transformed, dim=-2)

        # 2. Selection weights from flattened input
        ctx = context
        if has_seq and ctx is not None and ctx.dim() == 2:
            ctx = ctx.unsqueeze(1).expand(-1, L, -1)

        weight_logits = self.selection_grn(x, context=ctx)  # (B, [L], n_vars)
        weights = F.softmax(weight_logits, dim=-1)

        # 3. Weighted sum: (*, n_vars, d_model) × (*, n_vars, 1) → (*, d_model)
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=-2)

        return selected, weights


class SharedVariableSelectionNetwork(nn.Module):
    """Variable selector with a shared scalar encoder across features.

    This keeps TFT-style per-feature selection weights but replaces the
    ``n_vars`` independent 1D GRNs with one shared GRN. A learned feature
    embedding conditions the shared encoder so each input can still learn a
    distinct transformation without paying for a separate subnetwork.

    Parameters
    ----------
    n_vars : int
        Number of input variables (features).
    d_model : int
        Output dimension per variable after transformation.
    d_context : int, optional
        Context vector dimension for conditioning the selection weights.
    dropout : float
        Dropout rate for the shared encoder and selection GRN.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        d_context: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        self.shared_var_grn = GatedResidualNetwork(
            d_model=d_model,
            d_input=1,
            d_hidden=d_model,
            d_context=d_model,
            dropout=dropout,
        )
        self.feature_embeddings = nn.Parameter(torch.empty(n_vars, d_model))
        nn.init.normal_(self.feature_embeddings, mean=0.0, std=d_model ** -0.5)

        self.selection_grn = GatedResidualNetwork(
            d_model=n_vars,
            d_input=n_vars,
            d_hidden=n_vars,
            d_context=d_context,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply shared feature transforms and context-conditioned selection."""
        has_seq = x.dim() == 3
        if has_seq:
            _, seq_len, _ = x.shape
        else:
            seq_len = None

        feature_ctx = self.feature_embeddings.view(
            *([1] * (x.dim() - 1)),
            self.n_vars,
            self.d_model,
        )
        transformed = self.shared_var_grn(
            x.unsqueeze(-1),
            context=feature_ctx,
        )

        ctx = context
        if has_seq and ctx is not None and ctx.dim() == 2:
            ctx = ctx.unsqueeze(1).expand(-1, seq_len, -1)

        weight_logits = self.selection_grn(x, context=ctx)
        weights = F.softmax(weight_logits, dim=-1)
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=-2)
        return selected, weights


# ============================================================================
# Event Decay Encoding
# ============================================================================

class EventDecayEncoding(nn.Module):
    """Temporal decay positional encoding for event recency.

    Produces a per-timestep embedding modulated by how recently a
    significant event occurred. Uses learned decay rates so the model
    can learn different persistence patterns for different latent
    dimensions.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_decay : float
        Maximum decay rate (fastest forgetting).
    """

    def __init__(self, d_model: int, max_decay: float = 5.0):
        super().__init__()
        self.d_model = d_model
        # Learnable decay rates per dimension
        self.log_decay = nn.Parameter(
            torch.linspace(0.0, math.log(max_decay), d_model)
        )
        self.scale = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(
        self,
        event_mask: torch.Tensor,
        W: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        event_mask : Tensor (B, W)
            Binary mask: 1.0 where significant event occurred.
        W : int
            Window length (should match event_mask.shape[1]).

        Returns
        -------
        Tensor (B, W, d_model)
            Decay-modulated positional encoding.
        """
        B = event_mask.shape[0]
        device = event_mask.device

        # Compute days-since-event for each position
        # For each position, find the most recent event
        positions = torch.arange(W, device=device).float()
        days_since = torch.zeros(B, W, device=device)

        for t in range(W):
            # Look backward from position t for the most recent event
            mask_up_to_t = event_mask[:, :t + 1]  # (B, t+1)
            # Distance from each past position to t
            dists = positions[t] - positions[:t + 1]  # (t+1,)
            dists = dists.unsqueeze(0).expand(B, -1)  # (B, t+1)

            # Mask out non-event positions
            valid = mask_up_to_t > 0
            dists_masked = dists.clone()
            dists_masked[~valid] = float('inf')

            # Minimum distance to an event
            min_dist, _ = dists_masked.min(dim=1)  # (B,)
            min_dist = min_dist.clamp(max=W)
            days_since[:, t] = min_dist

        # Apply learned decay: exp(-decay_rate * days_since)
        decay_rates = self.log_decay.exp()  # (d_model,)
        # (B, W, 1) * (1, 1, d_model)
        decay = torch.exp(
            -days_since.unsqueeze(-1) * decay_rates.unsqueeze(0).unsqueeze(0)
        )

        return decay * self.scale.unsqueeze(0).unsqueeze(0)


# ============================================================================
# Branch Variable Selection (TFT Eq. 6)
# ============================================================================

class BranchVariableSelection(nn.Module):
    """Context-conditioned variable/branch selection.

    Given N branch outputs and a context vector, produces a
    weighted combination of the branches. The weights are
    conditioned on the context, so different assets can learn
    different branch importance patterns.

    Parameters
    ----------
    n_branches : int
        Number of input branches.
    d_branch : int
        Dimension of each branch output.
    d_context : int, optional
        Context vector dimension for conditioning.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_branches: int,
        d_branch: int,
        d_context: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.d_branch = d_branch

        # Per-branch GRN processing
        self.branch_grns = nn.ModuleList([
            GatedResidualNetwork(d_model=d_branch, dropout=dropout)
            for _ in range(n_branches)
        ])

        # Softmax variable selection weights
        # Input: flattened branches + optional context
        selection_input_dim = n_branches * d_branch
        if d_context is not None:
            selection_input_dim += d_context

        self.selection_network = nn.Sequential(
            nn.Linear(selection_input_dim, n_branches * d_branch),
            nn.GELU(),
            nn.Linear(n_branches * d_branch, n_branches),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        branch_outputs: list,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        branch_outputs : list of Tensor
            Each (B, d_branch).
        context : Tensor (B, d_context), optional
            Static context for conditioning.

        Returns
        -------
        z_selected : Tensor (B, d_branch)
            Weighted combination of branch outputs.
        weights : Tensor (B, n_branches)
            Selection weights per branch.
        """
        # Process each branch through its GRN
        processed = [
            grn(x) for grn, x in zip(self.branch_grns, branch_outputs)
        ]

        # Compute selection weights
        flat = torch.cat(branch_outputs, dim=-1)  # (B, n_branches * d_branch)
        if context is not None:
            flat = torch.cat([flat, context], dim=-1)

        weights = self.selection_network(flat)  # (B, n_branches)

        # Weighted combination
        stacked = torch.stack(processed, dim=1)  # (B, n_branches, d_branch)
        z_selected = torch.einsum("bn,bnd->bd", weights, stacked)

        return z_selected, weights


# ============================================================================
# Interpretable Multi-Head Attention
# ============================================================================

class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention from TFT.

    Each head computes its own attention pattern, but all heads
    share a single set of value weights. This makes the attention
    patterns directly interpretable as temporal importance weights.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_head)  # Shared values
        self.W_o = nn.Linear(self.d_head, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : Tensor (B, T_q, d_model)
        key : Tensor (B, T_k, d_model)
        value : Tensor (B, T_k, d_model)
        mask : Tensor (T_q, T_k), optional
            Additive attention mask (e.g., causal).

        Returns
        -------
        output : Tensor (B, T_q, d_model)
        attn_weights : Tensor (B, n_heads, T_q, T_k)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Multi-head Q, K
        q = self.W_q(query).view(B, T_q, self.n_heads, self.d_head)
        k = self.W_k(key).view(B, T_k, self.n_heads, self.d_head)
        # Shared V
        v = self.W_v(value)  # (B, T_k, d_head)

        # Attention scores
        q = q.permute(0, 2, 1, 3)  # (B, H, T_q, d_head)
        k = k.permute(0, 2, 1, 3)  # (B, H, T_k, d_head)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to shared values
        # attn_weights: (B, H, T_q, T_k), v: (B, T_k, d_head)
        # Average across heads
        attn_avg = attn_weights.mean(dim=1)  # (B, T_q, T_k)
        output = torch.bmm(attn_avg, v)  # (B, T_q, d_head)
        output = self.W_o(output)  # (B, T_q, d_model)

        return output, attn_weights


# ============================================================================
# Macro-Enriched Temporal Attention
# ============================================================================

class MacroEnrichedTemporalAttention(nn.Module):
    """Self-attention over daily tokens enriched by macro context.

    Applies GRN-based enrichment (conditioned on macro/static context)
    before interpretable self-attention, followed by a gated residual.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_context : int
        Context vector dimension (c_e + macro_context combined).
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_context: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Pre-attention enrichment GRN (conditioned on context)
        self.enrichment_grn = GatedResidualNetwork(
            d_model=d_model,
            d_context=d_context,
            dropout=dropout,
        )

        # Self-attention
        self.self_attn = InterpretableMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Post-attention gate + residual
        self.gate = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        daily_tokens: torch.Tensor,
        macro_context: torch.Tensor,
        causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        daily_tokens : Tensor (B, W, d_model)
            Per-day token sequence.
        macro_context : Tensor (B, d_model)
            Combined enrichment context.
        causal_mask : bool
            Whether to apply causal masking.

        Returns
        -------
        attended : Tensor (B, W, d_model)
            Attended sequence.
        attn_weights : Tensor (B, H, W, W)
            Attention weight matrices.
        """
        B, W, D = daily_tokens.shape

        # Enrich each day's token with context
        ctx_expanded = macro_context.unsqueeze(1).expand(B, W, -1)
        enriched = daily_tokens.clone()
        for t in range(W):
            enriched[:, t, :] = self.enrichment_grn(
                daily_tokens[:, t, :],
                context=macro_context,
            )

        # Build causal mask
        mask = None
        if causal_mask:
            mask = torch.triu(
                torch.full((W, W), float('-inf'), device=daily_tokens.device),
                diagonal=1,
            )

        # Self-attention
        attn_out, attn_weights = self.self_attn(
            query=enriched,
            key=enriched,
            value=enriched,
            mask=mask,
        )

        # Gated residual
        gated = self.gate(attn_out)
        attended = self.norm(enriched + gated)

        return attended, attn_weights


# ============================================================================
# Gated Cross-Attention Fusion
# ============================================================================

class GatedCrossAttentionFusion(nn.Module):
    """Cross-attention fusion with a learnable gate.

    Uses the macro context as query and branch/temporal tokens as
    key-value. A learned gate controls how much the cross-attention
    output modifies the macro query, allowing the model to "trust"
    macro vs. branch signals adaptively.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable gate
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_model)
        self.last_gate_value: Optional[torch.Tensor] = None

    def forward(
        self,
        macro_query: torch.Tensor,
        branch_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        macro_query : Tensor (B, d_model)
            Macro context as query.
        branch_kv : Tensor (B, N_kv, d_model)
            Branch/temporal tokens as key-value.

        Returns
        -------
        z_fused : Tensor (B, d_model)
            Fused output.
        """
        # Expand query for cross-attention: (B, 1, d_model)
        q = macro_query.unsqueeze(1)

        # Cross-attention
        attn_out, _ = self.cross_attn(q, branch_kv, branch_kv)
        attn_out = attn_out.squeeze(1)  # (B, d_model)

        # Gated combination
        gate = self.gate_proj(
            torch.cat([macro_query, attn_out], dim=-1)
        )
        self.last_gate_value = gate.detach()

        z_fused = self.norm(macro_query + gate * attn_out)
        return z_fused
