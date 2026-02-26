"""Backward-compatible re-export for MMTF v3 core.

Use `c_mmtft.py` as the primary module.
"""

from .c_mmtft import (
    MMTFv3Core,
    MMTFv3Mamba,
    MMTFv3Transformer,
    MMTFv3MambaVQVAE,
    TickerPositionStateLayer,
    StatefulMMTFv3Core,
    train_epoch_v3,
    train_epoch_v3_stateful,
    evaluate_v3,
    evaluate_v3_stateful,
    print_v3_diagnostics,
)
from ...training.loss.clf import ContinuousTradingLoss, SharpeScheduler

__all__ = [
    "MMTFv3Core",
    "MMTFv3Mamba",
    "MMTFv3Transformer",
    "MMTFv3MambaVQVAE",
    "TickerPositionStateLayer",
    "StatefulMMTFv3Core",
    "train_epoch_v3",
    "train_epoch_v3_stateful",
    "evaluate_v3",
    "evaluate_v3_stateful",
    "print_v3_diagnostics",
]
