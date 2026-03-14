"""
MacroGAT-Mamba: 3-branch model for Gold intraday prediction.

Branches:
  1. Spatial: NumberBarsEncoder + RasterResNet → SpatialFuse
  2. Macro:   HeterogeneousMacroGraph (Asset/Econ GRU nodes → GAT → Gold extraction)
  3. Tech:    MLP over intraday technical features

Head: MambaFusionHead with 4-class quartile classification.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders import (
    NumberBarsEncoder,
    RasterResNet,
    SpatialFuse,
)
from .mamba_model import MambaBlock, MambaFusionHead


# ---------------------------------------------------------------------------
# Short-sequence fusion head (replaces Mamba for 3-token sequences)
# ---------------------------------------------------------------------------

class BranchFusionHead(nn.Module):
    """Attention + MLP head for fusing a small number of branch tokens.

    mamba_ssm's CUDA selective-scan kernel is unreliable for very short
    sequences (L <= 4).  This module uses multi-head self-attention
    instead, which is well-suited for 3-token fusion and fully stable.
    """

    def __init__(
        self,
        d_model: int = 128,
        out_dim: int = 4,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )
        self.last_tracker: Dict[str, object] = {}

    def forward(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """x: (B, T, d_model) → (B, out_dim)"""
        attn_out, attn_w = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        if pool == "last":
            p = x[:, -1, :]
        else:
            p = x.mean(dim=1)
        self.last_tracker = {
            "attn_weights": attn_w.detach() if attn_w is not None else None,
            "pool_mode": pool,
        }
        return self.head(p)


# ---------------------------------------------------------------------------
# GAT Node Encoders (from MACRO_GAT.md design)
# ---------------------------------------------------------------------------

class AssetNodeEncoder(nn.Module):
    """Temporal encoder for traded assets (Gold, DXY, Yields, TIPS).

    Projects 10-feature daily vectors through a small MLP then GRU.
    Output: (B, temporal_hidden) per node.
    """

    def __init__(
        self,
        in_features: int = 10,
        temporal_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, temporal_hidden),
            nn.LayerNorm(temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(temporal_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, in_features) → (B, temporal_hidden)"""
        x_proj = self.feature_proj(x)
        _, h_n = self.gru(x_proj)
        return self.norm(h_n[-1])


class EconomicNodeEncoder(nn.Module):
    """Temporal encoder for macro indicators (Inflation, Labor, Growth).

    Uses a wider projection to amplify sparse release-day signals.
    Output: (B, temporal_hidden) per node.
    """

    def __init__(
        self,
        in_features: int = 2,
        temporal_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shock_amplifier = nn.Sequential(
            nn.Linear(in_features, temporal_hidden // 2),
            nn.GELU(),
            nn.Linear(temporal_hidden // 2, temporal_hidden),
            nn.LayerNorm(temporal_hidden),
        )
        self.gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, in_features) → (B, temporal_hidden)"""
        x_amp = self.shock_amplifier(x)
        _, h_n = self.gru(x_amp)
        return h_n[-1]


# ---------------------------------------------------------------------------
# Heterogeneous Macro Graph (GAT)
# ---------------------------------------------------------------------------

# Default node configuration: name → (encoder_type, in_features)
DEFAULT_NODE_CONFIG: Dict[str, Tuple[str, int]] = {
    "Gold":      ("asset", 10),
    "DXY":       ("asset", 10),
    "US_2Y":     ("asset", 10),
    "US_10Y":    ("asset", 10),
    "TIPS":      ("asset", 10),
    "Inflation": ("econ", 2),
    "Labor":     ("econ", 2),
    "Growth":    ("econ", 2),
}


class HeterogeneousMacroGraph(nn.Module):
    """Graph Attention Network over heterogeneous macro nodes.

    Constructs per-node GRU temporal encoders, projects all into shared
    embedding space, runs multi-head self-attention, and extracts the
    Gold node embedding as the macro context vector.

    Parameters
    ----------
    node_config : dict
        Maps node name → (encoder_type, in_features).
        encoder_type is "asset" or "econ".
    temporal_hidden : int
        Hidden dimension for all node GRUs and the graph.
    num_heads : int
        Number of attention heads in the graph layer.
    macro_out_dim : int
        Output dimension of the extracted macro vector.
    target_node : str
        Which node to extract (default "Gold").
    """

    def __init__(
        self,
        node_config: Optional[Dict[str, Tuple[str, int]]] = None,
        temporal_hidden: int = 64,
        num_heads: int = 4,
        macro_out_dim: int = 128,
        target_node: str = "Gold",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = node_config or DEFAULT_NODE_CONFIG
        self.temporal_hidden = temporal_hidden
        self.target_node = target_node
        self.node_order = list(self.config.keys())
        self.target_idx = self.node_order.index(target_node)

        # Build per-node encoders
        self.node_encoders = nn.ModuleDict()
        for name, (enc_type, in_feat) in self.config.items():
            if enc_type == "asset":
                self.node_encoders[name] = AssetNodeEncoder(
                    in_features=in_feat,
                    temporal_hidden=temporal_hidden,
                    dropout=dropout,
                )
            else:
                self.node_encoders[name] = EconomicNodeEncoder(
                    in_features=in_feat,
                    temporal_hidden=temporal_hidden,
                    dropout=max(0.1, dropout * 0.5),
                )

        # Graph attention (fully connected, all nodes attend to all)
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=temporal_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_norm = nn.LayerNorm(temporal_hidden)

        # Extraction head: project target node to output dim
        self.target_extraction = nn.Sequential(
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden, macro_out_dim),
        )

        self.out_dim = macro_out_dim

    def forward(
        self,
        macro_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        macro_dict : dict
            Maps node name → (B, seq_len, F_node) tensors.

        Returns
        -------
        macro_vector : (B, macro_out_dim)
            Extracted Gold-centric macro context.
        attention_weights : (B, num_nodes, num_nodes)
            Graph attention weights for interpretability.
        """
        latent_nodes = []
        for name in self.node_order:
            x_raw = macro_dict[name]  # (B, seq_len, F_node)
            z_node = self.node_encoders[name](x_raw)  # (B, temporal_hidden)
            latent_nodes.append(z_node.unsqueeze(1))  # (B, 1, H)

        # Assemble graph: (B, num_nodes, temporal_hidden)
        graph_input = torch.cat(latent_nodes, dim=1)

        # Multi-head self-attention (message passing)
        attn_out, attn_weights = self.graph_attention(
            query=graph_input,
            key=graph_input,
            value=graph_input,
        )

        # Residual + norm
        fused_graph = self.graph_norm(graph_input + attn_out)

        # Extract target node
        target_context = fused_graph[:, self.target_idx, :]
        macro_vector = self.target_extraction(target_context)

        return macro_vector, attn_weights


# ---------------------------------------------------------------------------
# Technical Feature Encoder
# ---------------------------------------------------------------------------

class TechEncoder(nn.Module):
    """MLP encoder for windowed intraday technical features.

    Takes (B, lookback, F_tech) → pools → (B, d_model).
    Uses a small LSTM to capture temporal structure before projection.
    """

    def __init__(
        self,
        f_tech: int,
        d_model: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(f_tech, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lookback, F_tech) → (B, d_model)"""
        h = self.proj(x)
        _, (h_n, _) = self.lstm(h)
        return self.norm(h_n[-1])


# ---------------------------------------------------------------------------
# Full Model: MacroGATMamba
# ---------------------------------------------------------------------------

class MacroGATMamba(nn.Module):
    """3-branch Gold intraday model with Mamba classification head.

    Branch 1 — Spatial (orderflow):
      NumberBarsEncoder + RasterResNet → SpatialFuse → z_spatial

    Branch 2 — Macro (graph):
      HeterogeneousMacroGraph → z_macro

    Branch 3 — Technical (intraday):
      TechEncoder (LSTM over 5min bar features) → z_tech

    Head:
      Stack [z_spatial, z_macro, z_tech] → MambaFusionHead → 4-class logits

    Parameters
    ----------
    f_tech : int
        Number of intraday technical features per bar.
    nb_channels : int
        NumberBars channels (default 4).
    nb_bins : int
        NumberBars price bins (default 32).
    raster_channels : int
        Rasterized VPIN channels (default 4).
    d_model : int
        Shared embedding dimension across all branches.
    num_classes : int
        Number of quartile classes (default 4).
    node_config : dict, optional
        GAT node configuration. If None, uses DEFAULT_NODE_CONFIG.
    macro_temporal_hidden : int
        Hidden dim for macro node GRUs (default 64).
    macro_num_heads : int
        Attention heads in the GAT (default 4).
    n_mamba_layers : int
        Number of Mamba blocks in the fusion head.
    dropout : float
        Global dropout rate.
    """

    def __init__(
        self,
        f_tech: int,
        nb_channels: int = 4,
        nb_bins: int = 32,
        raster_channels: int = 4,
        d_model: int = 128,
        num_classes: int = 4,
        node_config: Optional[Dict[str, Tuple[str, int]]] = None,
        macro_temporal_hidden: int = 64,
        macro_num_heads: int = 4,
        nb_max_T: int = 32,
        n_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # --- Branch 1: Spatial (orderflow) ---
        self.numbars_enc = NumberBarsEncoder(
            c_in=nb_channels,
            d_model=d_model,
            dropout=dropout,
            max_T=nb_max_T,
        )
        self.raster_enc = RasterResNet(
            in_ch=raster_channels,
            d_model=d_model,
        )
        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model,
            mode="gated",
        )

        # --- Branch 2: Macro (graph) ---
        self.macro_graph = HeterogeneousMacroGraph(
            node_config=node_config,
            temporal_hidden=macro_temporal_hidden,
            num_heads=macro_num_heads,
            macro_out_dim=d_model,
            target_node="Gold",
            dropout=dropout,
        )

        # --- Branch 3: Technical (intraday) ---
        self.tech_enc = TechEncoder(
            f_tech=f_tech,
            d_model=d_model,
            dropout=dropout,
        )

        # --- Mamba Fusion Head ---
        # Token sequence is exactly 3 (spatial, macro, tech).
        self.head = MambaFusionHead(
            input_dim=d_model,
            d_model=d_model,
            out_dim=num_classes,
            n_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        self._last_tracker: Dict[str, object] = {}

    def forward(
        self,
        tech_features: torch.Tensor,
        numbars_recent: torch.Tensor,
        numbars_lens: torch.Tensor,
        raster_prev_day: torch.Tensor,
        macro_dict: Dict[str, torch.Tensor],
        return_probs: bool = False,
        return_tracker: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tech_features : (B, lookback, F_tech)
        numbars_recent : (B, T_nb, bins, channels) — note: permuted internally
        numbars_lens : (B,) actual NumberBars lengths
        raster_prev_day : (B, T_raster, C_raster, bins)
        macro_dict : dict[str, (B, macro_lookback, F_node)]

        Returns
        -------
        logits : (B, num_classes) raw logits (or probabilities if return_probs)
        """
        # Branch 1: Spatial
        # NumberBarsEncoder expects (B, T, BINS, C) — dataset gives (B, T, C, BINS)
        nb_input = numbars_recent.permute(0, 1, 3, 2)
        z_nb = self.numbars_enc(nb_input, nb_lengths=numbars_lens)
        z_raster = self.raster_enc(raster_prev_day)
        z_spatial = self.spatial_fuse(z_nb, z_raster)

        # Branch 2: Macro
        z_macro, attn_weights = self.macro_graph(macro_dict)

        # Branch 3: Technical
        z_tech = self.tech_enc(tech_features)

        # Fuse: stack 3 branch tokens → Mamba sequence
        token_seq = torch.stack([z_spatial, z_macro, z_tech], dim=1)  # (B, 3, d_model)
        logits = self.head(token_seq, pool="mean")  # (B, num_classes)

        if return_probs:
            logits = F.softmax(logits, dim=-1)

        self._last_tracker = {
            "spatial_fuse": self.spatial_fuse.get_importance_stats(),
            "macro_attn": attn_weights.detach(),
            "head": dict(self.head.last_tracker),
            "token_names": ["spatial", "macro", "tech"],
        }

        if return_tracker:
            return logits, self._last_tracker
        return logits

    def get_last_tracker(self) -> Dict[str, object]:
        return dict(self._last_tracker)

    @classmethod
    def from_prep_dims(
        cls,
        dims: Dict[str, int],
        d_model: int = 128,
        num_classes: int = 4,
        **kwargs,
    ) -> "MacroGATMamba":
        """Construct from dims dict returned by MacroGATContinuousPrep.get_dims()."""
        from CTAFlow.features.macro_gat_prep import NODE_ORDER, ASSET_NODE_ORDER, ECON_NODE_ORDER

        node_config = {}
        for name in NODE_ORDER:
            key = f"macro_{name}"
            if key in dims:
                enc_type = "asset" if name in ASSET_NODE_ORDER else "econ"
                node_config[name] = (enc_type, dims[key])

        return cls(
            f_tech=dims["f_tech"],
            nb_channels=dims.get("nb_channels", 4),
            nb_bins=dims.get("nb_bins", 32),
            raster_channels=dims.get("raster_C", 4),
            d_model=d_model,
            num_classes=num_classes,
            node_config=node_config,
            **kwargs,
        )
