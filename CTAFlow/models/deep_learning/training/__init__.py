from .training import compute_class_weights, create_classification_targets, mixup_criterion, mixup_data
from .loops import (
    evaluate,
    TrainingMetrics,
    compute_trading_metrics,
    train_classification_epoch,
    evaluate_classification,
    train_regression_epoch,
    evaluate_regression,
)
from .backtest import BacktestAttributionResult, backtest_from_predictions, predictions_to_positions
from .loss import (
    ConfidencePenalizedCE,
    CostAwareCE,
    ExpectedPnLLoss,
    ProfitWeightedCE,
    OrdinalCEWithAntiCollapse,
    HierarchicalDirectionalLoss,
    SharpePenaltyLoss,
    TradingPnLLoss,
)

__all__ = [
    # Training utilities
    "create_classification_targets",
    "compute_class_weights",
    "mixup_data",
    "mixup_criterion",
    # Training loops
    "TrainingMetrics",
    "compute_trading_metrics",
    "train_classification_epoch",
    "evaluate_classification",
    "train_regression_epoch",
    "evaluate_regression",
    "evaluate",  # Legacy
    "BacktestAttributionResult",
    "predictions_to_positions",
    "backtest_from_predictions",
    # Classification losses
    "ConfidencePenalizedCE",
    "CostAwareCE",
    "ExpectedPnLLoss",
    "ProfitWeightedCE",
    "OrdinalCEWithAntiCollapse",
    "HierarchicalDirectionalLoss",
    # Regression losses
    "SharpePenaltyLoss",
    "TradingPnLLoss",
]
