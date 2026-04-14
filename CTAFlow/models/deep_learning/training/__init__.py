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
from .backtest import (
    BacktestAttributionResult,
    TradingMode,
    backtest_from_predictions,
    backtest_eod_momentum,
    predictions_to_positions,
)
from .detailed_backtest import (
    DetailedBacktestResult,
    run_detailed_backtest,
    save_detailed_backtest_artifacts,
)
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
from .optimization import AccuracyWeightedPnL

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
    "DetailedBacktestResult",
    "TradingMode",
    "predictions_to_positions",
    "backtest_from_predictions",
    "backtest_eod_momentum",
    "run_detailed_backtest",
    "save_detailed_backtest_artifacts",
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
    # Optimization objectives
    "AccuracyWeightedPnL",
]
