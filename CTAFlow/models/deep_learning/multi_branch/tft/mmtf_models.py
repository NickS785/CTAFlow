"""
MMTF Model Variants
===================

Convenient wrappers for Multi-Modal Temporal Fusion Transformer/Mamba variants.
"""

from __future__ import annotations

from .mmtf_core import MMTFCore

try:
    from CTAFlow.data.datasets.tft import N_EVENT_TYPES
except ImportError:
    N_EVENT_TYPES = 15


class MMTFTransformer(MMTFCore):
    """Multi-Modal Temporal Fusion Transformer.

    Transformer-based variant of MMTF using multi-head self-attention
    for temporal fusion over the lookback window.

    Parameters
    ----------
    f_sum : int
        Summary feature dimension.
    f_profile : int
        Profile channels.
    f_raster : int
        Rasterized channels.
    f_seq : int
        Sequential feature dimension.
    n_tickers : int
        Number of tickers.
    n_asset_classes : int
        Number of asset classes.
    n_asset_subclasses : int
        Number of asset subclasses.
    n_event_types : int
        Number of event types.
    max_events_per_day : int
        Max events per day.
    d_model : int
        Model dimension.
    d_static_emb : int
        Static embedding dimension.
    d_calendar : int
        Calendar embedding dimension.
    d_event_emb : int
        Event embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer layers.
    d_ff : int
        Feed-forward dimension.
    task : str
        'classification' or 'regression'.
    num_classes : int
        Number of classes.
    dropout : float
        Dropout rate.
    grn_dropout : float, optional
        GRN/BVS dropout.
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        n_event_types: int = N_EVENT_TYPES,
        max_events_per_day: int = 3,
        d_model: int = 128,
        d_static_emb: int = 64,
        d_calendar: int = 32,
        d_event_emb: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
    ):
        super().__init__(
            f_sum=f_sum,
            f_profile=f_profile,
            f_raster=f_raster,
            f_seq=f_seq,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            d_model=d_model,
            d_static_emb=d_static_emb,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            backbone="transformer",
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            task=task,
            num_classes=num_classes,
            dropout=dropout,
            grn_dropout=grn_dropout,
        )


class MMTFMamba(MMTFCore):
    """Multi-Modal Temporal Fusion Mamba.

    Mamba-based variant of MMTF using state-space models for efficient
    long-range temporal modeling.

    Parameters
    ----------
    f_sum : int
        Summary feature dimension.
    f_profile : int
        Profile channels.
    f_raster : int
        Rasterized channels.
    f_seq : int
        Sequential feature dimension.
    n_tickers : int
        Number of tickers.
    n_asset_classes : int
        Number of asset classes.
    n_asset_subclasses : int
        Number of asset subclasses.
    n_event_types : int
        Number of event types.
    max_events_per_day : int
        Max events per day.
    d_model : int
        Model dimension.
    d_static_emb : int
        Static embedding dimension.
    d_calendar : int
        Calendar embedding dimension.
    d_event_emb : int
        Event embedding dimension.
    n_heads : int
        Number of attention heads (for final temporal attention only).
    n_layers : int
        Number of Mamba layers.
    d_state : int
        Mamba state dimension.
    d_conv : int
        Mamba convolution kernel size.
    expand : int
        Mamba expansion factor.
    task : str
        'classification' or 'regression'.
    num_classes : int
        Number of classes.
    dropout : float
        Dropout rate.
    grn_dropout : float, optional
        GRN/BVS dropout.
    """

    def __init__(
        self,
        f_sum: int,
        f_profile: int,
        f_raster: int,
        f_seq: int,
        n_tickers: int = 1,
        n_asset_classes: int = 1,
        n_asset_subclasses: int = 1,
        n_event_types: int = N_EVENT_TYPES,
        max_events_per_day: int = 3,
        d_model: int = 128,
        d_static_emb: int = 64,
        d_calendar: int = 32,
        d_event_emb: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
    ):
        super().__init__(
            f_sum=f_sum,
            f_profile=f_profile,
            f_raster=f_raster,
            f_seq=f_seq,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            n_event_types=n_event_types,
            max_events_per_day=max_events_per_day,
            d_model=d_model,
            d_static_emb=d_static_emb,
            d_calendar=d_calendar,
            d_event_emb=d_event_emb,
            backbone="mamba",
            n_heads=n_heads,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            task=task,
            num_classes=num_classes,
            dropout=dropout,
            grn_dropout=grn_dropout,
        )
