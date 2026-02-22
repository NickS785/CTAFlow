"""
TFT-Aligned Encoders
====================

Encoder components that map raw inputs to the TFT input taxonomy:

  StaticCovariateEncoder     : Identity -> 4 context vectors (c_s, c_e, c_c, c_h)
  TemporalKnownInputEncoder  : Calendar + event schedule -> per-day tokens
  MacroPastObservedEncoder   : Continuous macro + event outcomes -> daily + context
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..market_context_models import (
    GLU,
    GatedResidualNetwork,
    EventDecayEncoding,
)
from ..macro_event_encoding import MacroEventEncoder, N_EVENT_TYPES


# ============================================================================
# Static Covariate Encoder (TFT Section 4.3)
# ============================================================================

class StaticCovariateEncoder(nn.Module):
    """Encodes time-invariant identity features into 4 context vectors.

    Replaces MetaModalityEncoder's identity handling. Produces 4 SPECIALIZED
    context vectors that condition different parts of the network:

      c_s: Branch/variable selection context
      c_e: Static enrichment context
      c_c: LSTM cell state initialization
      c_h: LSTM hidden state initialization

    Parameters
    ----------
    d_model : int
        Output dimension for each context vector.
    n_tickers : int
        Number of unique tickers.
    n_asset_classes : int
        Number of asset classes.
    n_asset_subclasses : int
        Number of sub-classes.
    d_emb : int
        Embedding dimension per categorical feature.
    dropout : float
        Dropout rate.
    """
    def __init__(
        self,
        d_model: int = 128,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        d_emb: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.ticker_emb = nn.Embedding(n_tickers, d_emb)
        self.class_emb = nn.Embedding(n_asset_classes, d_emb)
        self.subclass_emb = nn.Embedding(n_asset_subclasses, d_emb)

        n_static_vars = 3
        self.static_var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model=d_emb, dropout=dropout)
            for _ in range(n_static_vars)
        ])
        self.static_var_selection = nn.Sequential(
            nn.Linear(n_static_vars * d_emb, n_static_vars),
            nn.Softmax(dim=-1),
        )

        self.encoder_cs = GatedResidualNetwork(d_model=d_model, d_input=d_emb, dropout=dropout)
        self.encoder_ce = GatedResidualNetwork(d_model=d_model, d_input=d_emb, dropout=dropout)
        self.encoder_cc = GatedResidualNetwork(d_model=d_model, d_input=d_emb, dropout=dropout)
        self.encoder_ch = GatedResidualNetwork(d_model=d_model, d_input=d_emb, dropout=dropout)

        self.last_var_weights: Optional[torch.Tensor] = None
        self._init_weights()

    def _init_weights(self):
        for emb in [self.ticker_emb, self.class_emb, self.subclass_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ticker_id: torch.Tensor,
        asset_class_id: torch.Tensor,
        asset_subclass_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        ticker_id : Tensor (B,) long
        asset_class_id : Tensor (B,) long
        asset_subclass_id : Tensor (B,) long

        Returns
        -------
        c_s, c_e, c_c, c_h : each Tensor (B, d_model)
        """
        embs = [
            self.ticker_emb(ticker_id),
            self.class_emb(asset_class_id),
            self.subclass_emb(asset_subclass_id),
        ]

        processed = [grn(e) for grn, e in zip(self.static_var_grns, embs)]
        flat = torch.cat(embs, dim=-1)
        weights = self.static_var_selection(flat)
        self.last_var_weights = weights.detach()

        stacked = torch.stack(processed, dim=1)
        z_static = torch.einsum("bn,bnd->bd", weights, stacked)

        c_s = self.encoder_cs(z_static)
        c_e = self.encoder_ce(z_static)
        c_c = self.encoder_cc(z_static)
        c_h = self.encoder_ch(z_static)

        return c_s, c_e, c_c, c_h


# ============================================================================
# Temporal Known Input Encoder (Calendar + Scheduled Events)
# ============================================================================

class TemporalKnownInputEncoder(nn.Module):
    """Encodes known-in-advance temporal features into per-day tokens.

    Known inputs:
      - Calendar: month, day-of-week, day-of-year (sin/cos)
      - Scheduled events: event_type_ids, days_until_event

    Parameters
    ----------
    d_model : int
        Output dimension per day.
    d_calendar : int
        Calendar embedding dimension.
    d_event_emb : int
        Event type embedding dimension.
    n_event_types : int
        Number of event types.
    max_events_per_day : int
        Max simultaneous events.
    dropout : float
        Dropout rate.
    """
    def __init__(
        self,
        d_model: int = 128,
        d_calendar: int = 32,
        d_event_emb: int = 32,
        n_event_types: int = N_EVENT_TYPES,
        max_events_per_day: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.month_emb = nn.Embedding(13, d_calendar, padding_idx=0)
        self.dow_emb = nn.Embedding(7, d_calendar)
        self.doy_proj = nn.Sequential(
            nn.Linear(2, d_calendar),
            nn.GELU(),
        )

        self.event_type_emb = nn.Embedding(n_event_types, d_event_emb, padding_idx=0)
        self.days_until_emb = nn.Embedding(7, d_event_emb)
        self.event_pool = nn.Linear(d_event_emb * 2, 1)
        self.max_events = max_events_per_day

        total_input = d_calendar + d_event_emb * 2
        self.fuse = GatedResidualNetwork(
            d_model=d_model,
            d_input=total_input,
            dropout=dropout,
        )

        self.output_gate = GLU(d_model)

    def forward(
        self,
        month: torch.Tensor,
        dow: torch.Tensor,
        doy_sin: torch.Tensor,
        doy_cos: torch.Tensor,
        event_type_ids: Optional[torch.Tensor] = None,
        days_until_event: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        month : Tensor (B, W) long, 1-12
        dow : Tensor (B, W) long, 0-6
        doy_sin, doy_cos : Tensor (B, W) float
        event_type_ids : Tensor (B, W, max_events) long, optional
        days_until_event : Tensor (B, W, max_events) long, optional

        Returns
        -------
        Tensor (B, W, d_model)
        """
        B, W = month.shape

        z_cal = (
            self.month_emb(month.clamp(0, 12))
            + self.dow_emb(dow.clamp(0, 6))
            + self.doy_proj(torch.stack([doy_sin, doy_cos], dim=-1))
        )

        if event_type_ids is not None:
            z_etype = self.event_type_emb(event_type_ids)

            if days_until_event is not None:
                z_days = self.days_until_emb(days_until_event.clamp(0, 6))
            else:
                z_days = torch.zeros_like(z_etype)

            z_events = torch.cat([z_etype, z_days], dim=-1)

            event_valid = (event_type_ids != 0).float()
            scores = self.event_pool(z_events).squeeze(-1)
            scores = scores.masked_fill(event_valid == 0, -1e9)
            weights = F.softmax(scores, dim=-1)
            z_event_pooled = torch.einsum("bwe,bwed->bwd", weights, z_events)

            has_event = event_valid.sum(dim=-1, keepdim=True).clamp(max=1)
            z_event_pooled = z_event_pooled * has_event
        else:
            d_evt = self.event_type_emb.embedding_dim
            z_event_pooled = torch.zeros(B, W, d_evt * 2, device=month.device)

        z_known = torch.cat([z_cal, z_event_pooled], dim=-1)
        z_known_flat = z_known.reshape(B * W, -1)
        z_out = self.fuse(z_known_flat).view(B, W, -1)

        return self.output_gate(z_out)


# ============================================================================
# Macro Past-Observed Encoder
# ============================================================================

class MacroPastObservedEncoder(nn.Module):
    """Encodes past-observed macro features + event OUTCOMES.

    Processes:
    - Continuous macro features: VIX, yields, DXY, spreads (daily)
    - Event outcomes: surprise, direction, magnitude (post-release only)

    Parameters
    ----------
    f_macro : int
        Number of continuous macro features per day.
    d_model : int
        Model dimension.
    n_event_types : int
        Event types for the outcome encoder.
    n_outcome_features : int
        Outcome features per event.
    max_events_per_day : int
        Max simultaneous events.
    n_heads : int
        Attention heads for temporal processing.
    n_layers : int
        Transformer layers.
    dropout : float
        Dropout rate.
    use_event_decay : bool
        Whether to add EventDecayEncoding.
    max_window : int
        Maximum window length for positional encoding.
    """
    def __init__(
        self,
        f_macro: int,
        d_model: int = 128,
        n_event_types: int = N_EVENT_TYPES,
        n_outcome_features: int = 4,
        max_events_per_day: int = 3,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_event_decay: bool = True,
        max_window: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_event_decay = use_event_decay

        self.macro_grn = GatedResidualNetwork(
            d_model=d_model,
            d_input=f_macro,
            dropout=dropout,
        )

        self.event_outcome_encoder = MacroEventEncoder(
            d_model=d_model,
            n_event_types=n_event_types,
            n_outcome_features=n_outcome_features,
            max_events_per_day=max_events_per_day,
            dropout=dropout,
        )

        if use_event_decay:
            self.event_decay = EventDecayEncoding(d_model=d_model)

        self.pos_emb = nn.Embedding(max_window, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.context_pool = nn.Linear(d_model, 1)
        self.out_dim = d_model

    def forward(
        self,
        macro_days: torch.Tensor,
        event_type_ids: Optional[torch.Tensor] = None,
        event_outcomes: Optional[torch.Tensor] = None,
        event_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        macro_days : Tensor (B, W, f_macro)
        event_type_ids : Tensor (B, W, max_events) long, optional
        event_outcomes : Tensor (B, W, max_events, n_outcomes) float, optional
        event_mask : Tensor (B, W) float, optional

        Returns
        -------
        z_macro_days : Tensor (B, W, d_model)
        z_macro_context : Tensor (B, d_model)
        """
        B, W, _ = macro_days.shape

        z = self.macro_grn(macro_days)

        if event_type_ids is not None and event_outcomes is not None:
            z_events = self.event_outcome_encoder(
                event_type_ids=event_type_ids,
                event_outcomes=event_outcomes,
                days_until_event=None,
            )
            z = self.event_outcome_encoder.gate_into_macro(z, z_events)

        positions = torch.arange(W, device=macro_days.device).unsqueeze(0).expand(B, W)
        z = z + self.pos_emb(positions)

        if self.use_event_decay and event_mask is not None:
            z = z + self.event_decay(event_mask, W)

        z = self.temporal(z)
        z_macro_days = self.norm(z)

        scores = self.context_pool(z_macro_days)
        weights = F.softmax(scores, dim=1)
        z_macro_context = (z_macro_days * weights).sum(dim=1)

        return z_macro_days, z_macro_context
