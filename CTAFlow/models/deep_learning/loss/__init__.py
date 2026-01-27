"""
Custom Loss Functions for Deep Learning Models.

This module provides specialized loss functions for trading applications:

Classification Losses (Ordinal & Trading-Inspired)
---------------------------------------------------
Base Ordinal:
- DistanceWeightedCE: CE with penalty scaled by class distance
- OrdinalCEWithAntiCollapse: Ordinal CE + anti-collapse regularization

Trading-Inspired Classification:
- TradingCostAwareCE: Incorporates transaction/opportunity costs
- ProfitWeightedCE: Weights samples by potential P&L
- ConfidencePenalizedCE: Penalizes overconfident wrong predictions
- FocalDirectionalLoss: Focal loss with direction awareness
- ExpectedPnLLoss: Directly optimizes expected profit
- MarginOrdinalLoss: Requires margin between class logits
- AsymmetricDirectionalCE: Different costs for bullish/bearish mistakes
- SharpeInspiredCE: Variance penalty + direction awareness
- HierarchicalDirectionalLoss: Two-stage (direction then full class)

Regression Losses (with Directional Penalties)
----------------------------------------------
Base Directional:
- DirectionalMSE: MSE + wrong sign penalty
- DirectionalMAE: MAE + wrong sign penalty
- DirectionalHuber: Huber + wrong sign penalty

Asymmetric & Quantile:
- AsymmetricMSE: Different weights for over/under prediction
- QuantileDirectional: Quantile loss + wrong sign penalty

Sign-Focused:
- SignAccuracyLoss: Directly optimizes sign prediction
- SignAwareLoss: Composite loss balancing magnitude + sign

Trading-Inspired Regression:
- SharpePenaltyLoss: Sharpe-inspired with variance penalty
- TradingPnLLoss: Direct PnL optimization
- WeightedDirectionalMSE: Sample-weighted with direction penalty

Flexible:
- CombinedRegressionLoss: Multi-component configurable loss

Usage Examples
--------------
>>> from CTAFlow.models.deep_learning.loss import (
...     DirectionalMSE,
...     SignAwareLoss,
...     CostAwareCE,
...     ExpectedPnLLoss,
... )

>>> # Regression: penalize wrong direction
>>> reg_criterion = DirectionalMSE(direction_penalty=2.0)
>>> loss = reg_criterion(pred, target)

>>> # Classification: trading cost aware
>>> clf_criterion = CostAwareCE(direction_cost=3.0)
>>> loss = clf_criterion(logits, labels)

>>> # Classification: optimize expected P&L directly
>>> pnl_criterion = ExpectedPnLLoss(ce_weight=0.1)
>>> loss = pnl_criterion(logits, labels, returns=actual_returns)
"""

# Classification losses - Base
from .clf import (
    DistanceWeightedCE,
    OrdinalCEWithAntiCollapse,
)

# Classification losses - Trading-Inspired
from .clf import (
    CostAwareCE,
    ProfitWeightedCE,
    ConfidencePenalizedCE,
    FocalDirectionalLoss,
    ExpectedPnLLoss,
    MarginOrdinalLoss,
    AsymmetricDirectionalCE,
    SharpeInspiredCE,
    HierarchicalDirectionalLoss,
)

# Regression losses with directional penalties
from .regression import (
    DirectionalMSE,
    DirectionalMAE,
    DirectionalHuber,
    AsymmetricMSE,
    QuantileDirectional,
    SignAccuracyLoss,
    SignAwareLoss,
    SharpePenaltyLoss,
    TradingPnLLoss,
    WeightedDirectionalMSE,
    CombinedRegressionLoss,
)

__all__ = [
    # Classification - Base
    'DistanceWeightedCE',
    'OrdinalCEWithAntiCollapse',
    # Classification - Trading
    'CostAwareCE',
    'ProfitWeightedCE',
    'ConfidencePenalizedCE',
    'FocalDirectionalLoss',
    'ExpectedPnLLoss',
    'MarginOrdinalLoss',
    'AsymmetricDirectionalCE',
    'SharpeInspiredCE',
    'HierarchicalDirectionalLoss',
    # Regression
    'DirectionalMSE',
    'DirectionalMAE',
    'DirectionalHuber',
    'AsymmetricMSE',
    'QuantileDirectional',
    'SignAccuracyLoss',
    'SignAwareLoss',
    'SharpePenaltyLoss',
    'TradingPnLLoss',
    'WeightedDirectionalMSE',
    'CombinedRegressionLoss',
]
