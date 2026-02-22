"""
Macro Event Encoding
====================

Event type registry and neural encoder for macro-economic event
outcomes in multi-branch commodity models.

Classes
-------
MacroEventEncoder : Encodes event type + outcomes into d_model tokens

Constants
---------
COMMODITY_EVENT_TYPES : dict mapping event name -> integer ID
N_EVENT_TYPES : total number of event types (for embedding tables)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .market_context_models import GLU, GatedResidualNetwork


# ============================================================================
# Event Type Registry
# ============================================================================

ENERGY_EVENT_TYPES = {
    "none":              0,
    "fomc_decision":     1,
    "fomc_minutes":      2,
    "cpi":               3,
    "ppi":               4,
    "nfp":               5,
    "ism_manufacturing":  6,
    "ism_services":      7,
    "gdp":               8,
    "pce":               9,
    "retail_sales":     10,
    "eia_petroleum":    11,
    "eia_natgas":       12,
    "api_weekly":       13,
    "baker_hughes_rig": 14,
    "opec_meeting":     15,
    "opec_plus_meeting": 16,
    "china_pmi":        17,
    "china_trade":      18,
    "ecb_decision":     19,
    "boj_decision":     20,
    "geopolitical":     21,
    "weather_event":    22,
    "supply_disruption": 23,
    "emergency_opec":   24,
}

COMMODITY_EVENT_TYPES = {
    **ENERGY_EVENT_TYPES,
    "usda_wasde":       25,
    "usda_crop_report": 26,
    "usda_export_sales": 27,
    "lme_warehouse":    28,
    "comex_delivery":   29,
    "china_reserves":   30,
}

N_EVENT_TYPES = max(COMMODITY_EVENT_TYPES.values()) + 1


# ============================================================================
# Macro Event Encoder
# ============================================================================

class MacroEventEncoder(nn.Module):
    """Encodes event type embeddings + continuous outcomes into d_model tokens.

    For each day in the window, encodes up to max_events_per_day events.
    Each event has:
      - A categorical type (embedded)
      - Continuous outcomes: surprise, direction, magnitude, revision

    Events are pooled per day via attention-weighted aggregation, then
    gated into the macro representation.

    Parameters
    ----------
    d_model : int
        Output dimension per day.
    n_event_types : int
        Total event types for the embedding table.
    n_outcome_features : int
        Number of continuous outcome features per event.
    max_events_per_day : int
        Maximum events per day slot.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_event_types: int = N_EVENT_TYPES,
        n_outcome_features: int = 4,
        max_events_per_day: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_outcomes = n_outcome_features
        self.max_events = max_events_per_day

        # Event type embedding
        d_event = d_model // 2
        self.event_type_emb = nn.Embedding(
            n_event_types, d_event, padding_idx=0,
        )

        # Outcome projection
        self.outcome_proj = nn.Sequential(
            nn.Linear(n_outcome_features, d_event),
            nn.GELU(),
        )

        # Combine type + outcome per event
        self.event_fuse = GatedResidualNetwork(
            d_model=d_model,
            d_input=d_event * 2,
            dropout=dropout,
        )

        # Attention-pool across events on the same day
        self.pool_score = nn.Linear(d_model, 1)

        # Gate for merging event signal into macro
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.gate_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        event_type_ids: torch.Tensor,
        event_outcomes: torch.Tensor,
        days_until_event: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        event_type_ids : Tensor (B, W, E) long
        event_outcomes : Tensor (B, W, E, n_outcomes) float
        days_until_event : Tensor (B, W, E) long, optional
            Not used for past-observed outcomes, included for API compat.

        Returns
        -------
        z_events : Tensor (B, W, d_model)
            Per-day event encoding.
        """
        B, W, E = event_type_ids.shape

        # Embed event types
        z_type = self.event_type_emb(event_type_ids)  # (B, W, E, d_event)

        # Project outcomes
        z_outcome = self.outcome_proj(event_outcomes)  # (B, W, E, d_event)

        # Fuse type + outcome
        z_combined = torch.cat([z_type, z_outcome], dim=-1)
        z_combined = z_combined.view(B * W * E, -1)
        z_fused = self.event_fuse(z_combined).view(B, W, E, -1)

        # Attention-pool across events per day
        event_valid = (event_type_ids != 0).float()  # (B, W, E)
        scores = self.pool_score(z_fused).squeeze(-1)  # (B, W, E)
        scores = scores.masked_fill(event_valid == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # (B, W, E)

        z_pooled = torch.einsum("bwe,bwed->bwd", weights, z_fused)

        # Zero out event-free days
        has_event = event_valid.sum(dim=-1, keepdim=True).clamp(max=1)
        z_pooled = z_pooled * has_event

        return z_pooled

    def gate_into_macro(
        self,
        z_macro: torch.Tensor,
        z_events: torch.Tensor,
    ) -> torch.Tensor:
        """Gated addition of event signal into macro representation.

        Parameters
        ----------
        z_macro : Tensor (B, W, d_model)
        z_events : Tensor (B, W, d_model)

        Returns
        -------
        Tensor (B, W, d_model)
        """
        gate = self.gate_proj(
            torch.cat([z_macro, z_events], dim=-1)
        )
        return self.gate_norm(z_macro + gate * z_events)
