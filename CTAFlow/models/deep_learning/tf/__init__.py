"""
TensorFlow/Keras implementations of CTAFlow deep learning models.

This module provides TensorFlow/Keras versions of the PyTorch models
in CTAFlow.models.deep_learning for compatibility with TensorFlow workflows.
"""

from .encoders import (
    AttnPool,
    GatedFusion,
    SpatialFuse,
    ProfileEncoder,
    SeqEncoder,
    NumberBarsEncoder,
    SummaryEncoder,
    SummaryMLPEnc,
    RasterResNet,
    MarketProfileResNet,
    SpatialTemporalEncoder,
    MarketProfileCNN,
    IntradayRNN,
)

from .recurrent_models import (
    RecurrentDualModal,
    RecurrentWSPR,
    RecurrentTriModal,
)

__all__ = [
    # Encoders
    "AttnPool",
    "GatedFusion",
    "SpatialFuse",
    "ProfileEncoder",
    "SeqEncoder",
    "NumberBarsEncoder",
    "SummaryEncoder",
    "SummaryMLPEnc",
    "RasterResNet",
    "MarketProfileResNet",
    "SpatialTemporalEncoder",
    "MarketProfileCNN",
    "IntradayRNN",
    # Models
    "RecurrentDualModal",
    "RecurrentWSPR",
    "RecurrentTriModal",
]
