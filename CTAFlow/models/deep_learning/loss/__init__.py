"""
Custom Loss Functions for Deep Learning Models.

This module provides specialized loss functions for:
1. Classification with ordinal awareness and anti-collapse regularization
2. Regression with directional penalties for trading applications

Classification Losses
---------------------
- DistanceWeightedCE: CE with penalty scaled by class distance
- OrdinalCEWithAntiCollapse: Ordinal CE + anti-collapse regularization

Regression Losses (with directional penalties)
----------------------------------------------
- DirectionalMSE: MSE + wrong sign penalty
- DirectionalMAE: MAE + wrong sign penalty
- DirectionalHuber: Huber + wrong sign penalty
- AsymmetricMSE: Different weights for over/under prediction
- QuantileDirectional: Quantile loss + wrong sign penalty
- SignAccuracyLoss: Directly optimizes sign prediction
- SignAwareLoss: Composite loss balancing magnitude + sign
- SharpePenaltyLoss: Sharpe-inspired loss with variance penalty
- TradingPnLLoss: Direct PnL optimization
- WeightedDirectionalMSE: Sample-weighted MSE with direction penalty
- CombinedRegressionLoss: Flexible multi-component loss

Usage Examples
--------------
>>> from CTAFlow.models.deep_learning.loss import DirectionalMSE, SignAwareLoss

>>> # Simple directional MSE
>>> criterion = DirectionalMSE(direction_penalty=1.0)
>>> loss = criterion(predictions, targets)

>>> # Composite loss prioritizing sign accuracy
>>> criterion = SignAwareLoss(mse_weight=0.3, sign_weight=0.7)
>>> loss = criterion(predictions, targets)

>>> # Classification with ordinal awareness
>>> criterion = OrdinalCEWithAntiCollapse(alpha=1.0, reg_lambda=0.05)
>>> loss = criterion(logits, labels)
"""

# Classification losses
from .clf import (
    DistanceWeightedCE,
    OrdinalCEWithAntiCollapse,
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
    # Classification
    'DistanceWeightedCE',
    'OrdinalCEWithAntiCollapse',
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
