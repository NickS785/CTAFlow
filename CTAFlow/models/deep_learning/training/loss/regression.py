"""
Custom Regression Losses with Directional Penalties.

These losses extend standard regression losses (MSE, MAE, Huber) with:
1. Wrong direction penalties - extra cost when predicted sign differs from target
2. Magnitude-weighted penalties - scale penalty by how wrong the direction is
3. Asymmetric losses - different weights for over/under prediction
4. Regularization terms - L1/L2 on predictions, anti-collapse, Sharpe-inspired

Usage:
    from CTAFlow.models.deep_learning.loss import DirectionalMSE, SignAwareLoss

    # Simple directional MSE
    criterion = DirectionalMSE(direction_penalty=1.0)
    loss = criterion(predictions, targets)

    # Composite loss with sign focus
    criterion = SignAwareLoss(mse_weight=0.3, sign_weight=0.7)
    loss = criterion(predictions, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class DirectionalMSE(nn.Module):
    """
    MSE loss with penalty for wrong direction (sign) predictions.

    The loss is: MSE + direction_penalty * wrong_sign_indicator * |error|

    When the prediction and target have opposite signs, an additional
    penalty proportional to the error magnitude is applied.

    Parameters
    ----------
    direction_penalty : float
        Multiplier for the directional penalty term. Higher values
        penalize wrong direction more severely. Default 1.0.
    magnitude_weighted : bool
        If True, weight the direction penalty by target magnitude.
        Larger moves are more important to get right. Default False.
    reduction : str
        Reduction mode: 'mean', 'sum', or 'none'. Default 'mean'.

    Examples
    --------
    >>> criterion = DirectionalMSE(direction_penalty=2.0)
    >>> pred = torch.tensor([0.5, -0.3, 0.1])
    >>> target = torch.tensor([0.3, 0.2, -0.1])  # middle pred has wrong sign
    >>> loss = criterion(pred, target)
    """

    def __init__(
            self,
            direction_penalty: float = 1.0,
            magnitude_weighted: bool = False,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.direction_penalty = direction_penalty
        self.magnitude_weighted = magnitude_weighted
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (B,) or (B, 1) predictions
        target: (B,) or (B, 1) targets
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # Base MSE
        mse = (pred - target) ** 2

        # Direction penalty: activated when signs differ
        wrong_sign = (pred * target) < 0  # True when opposite signs
        error_magnitude = (pred - target).abs()

        if self.magnitude_weighted:
            # Weight by target magnitude - bigger moves matter more
            direction_term = wrong_sign.float() * error_magnitude * target.abs()
        else:
            direction_term = wrong_sign.float() * error_magnitude

        loss = mse + self.direction_penalty * direction_term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DirectionalMAE(nn.Module):
    """
    MAE loss with penalty for wrong direction predictions.

    Similar to DirectionalMSE but uses L1 loss as the base.
    MAE is more robust to outliers than MSE.

    Parameters
    ----------
    direction_penalty : float
        Multiplier for wrong sign penalty. Default 1.0.
    smooth : float
        Smoothing parameter. If > 0, uses smooth L1 loss. Default 0.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            direction_penalty: float = 1.0,
            smooth: float = 0.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.direction_penalty = direction_penalty
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        # Base MAE (or smooth L1)
        if self.smooth > 0:
            mae = F.smooth_l1_loss(pred, target, reduction="none", beta=self.smooth)
        else:
            mae = (pred - target).abs()

        # Direction penalty
        wrong_sign = (pred * target) < 0
        direction_term = wrong_sign.float() * mae

        loss = mae + self.direction_penalty * direction_term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DirectionalHuber(nn.Module):
    """
    Huber loss with directional penalty.

    Huber loss is quadratic for small errors (|error| < delta) and
    linear for large errors, making it robust to outliers while
    maintaining smoothness near zero.

    Parameters
    ----------
    delta : float
        Threshold between quadratic and linear regions. Default 1.0.
    direction_penalty : float
        Multiplier for wrong sign penalty. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            delta: float = 1.0,
            direction_penalty: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.delta = delta
        self.direction_penalty = direction_penalty
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        # Huber loss
        error = pred - target
        abs_error = error.abs()
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        huber = torch.where(abs_error <= self.delta, quadratic, linear)

        # Direction penalty
        wrong_sign = (pred * target) < 0
        direction_term = wrong_sign.float() * abs_error

        loss = huber + self.direction_penalty * direction_term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricMSE(nn.Module):
    """
    MSE with asymmetric weights for over/under prediction.

    Useful when over-predicting vs under-predicting has different
    costs (e.g., in risk management or trading).

    Parameters
    ----------
    over_weight : float
        Weight for over-predictions (pred > target). Default 1.0.
    under_weight : float
        Weight for under-predictions (pred < target). Default 1.0.
    direction_penalty : float
        Additional penalty for wrong sign. Default 0.0.
    reduction : str
        Reduction mode. Default 'mean'.

    Examples
    --------
    >>> # Penalize over-predictions more (conservative)
    >>> criterion = AsymmetricMSE(over_weight=2.0, under_weight=1.0)
    """

    def __init__(
            self,
            over_weight: float = 1.0,
            under_weight: float = 1.0,
            direction_penalty: float = 0.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.over_weight = over_weight
        self.under_weight = under_weight
        self.direction_penalty = direction_penalty
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        error = pred - target
        mse = error ** 2

        # Asymmetric weights
        over_mask = (error > 0).float()
        under_mask = (error <= 0).float()
        weights = over_mask * self.over_weight + under_mask * self.under_weight

        loss = weights * mse

        # Optional direction penalty
        if self.direction_penalty > 0:
            wrong_sign = (pred * target) < 0
            loss = loss + self.direction_penalty * wrong_sign.float() * error.abs()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class QuantileDirectional(nn.Module):
    """
    Quantile (pinball) loss with directional penalty.

    Quantile loss asymmetrically penalizes errors based on quantile.
    tau=0.5 gives MAE, tau<0.5 penalizes over-prediction more,
    tau>0.5 penalizes under-prediction more.

    Parameters
    ----------
    tau : float
        Quantile in (0, 1). Default 0.5 (median).
    direction_penalty : float
        Penalty for wrong sign predictions. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            tau: float = 0.5,
            direction_penalty: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        assert 0 < tau < 1, "tau must be in (0, 1)"
        self.tau = tau
        self.direction_penalty = direction_penalty
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        error = target - pred
        quantile_loss = torch.maximum(
            self.tau * error,
            (self.tau - 1) * error
        )

        # Direction penalty
        wrong_sign = (pred * target) < 0
        direction_term = wrong_sign.float() * error.abs()

        loss = quantile_loss + self.direction_penalty * direction_term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SignAccuracyLoss(nn.Module):
    """
    Loss that directly optimizes sign (direction) accuracy.

    Uses a differentiable approximation to sign accuracy via
    soft sign matching. The gradient exists because we use
    sigmoid/tanh approximations rather than hard thresholds.

    Parameters
    ----------
    temperature : float
        Temperature for sigmoid approximation. Lower values make
        the approximation sharper. Default 1.0.
    margin : float
        Margin around zero where predictions are uncertain. Default 0.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            temperature: float = 1.0,
            margin: float = 0.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        # Soft sign using tanh
        soft_sign_pred = torch.tanh(pred / self.temperature)
        soft_sign_target = torch.tanh(target / self.temperature)

        # Loss: 1 - (agreement between soft signs)
        # Agreement is high when both are same sign, low when opposite
        agreement = soft_sign_pred * soft_sign_target
        loss = 1.0 - agreement

        # Optional margin: reduce loss for predictions near zero
        if self.margin > 0:
            near_zero = (pred.abs() < self.margin).float()
            loss = loss * (1.0 - 0.5 * near_zero)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SignAwareLoss(nn.Module):
    """
    Composite loss combining magnitude accuracy and sign accuracy.

    This loss balances:
    1. Getting the magnitude right (MSE/MAE)
    2. Getting the direction right (sign accuracy)
    3. Optional regularization

    Parameters
    ----------
    mse_weight : float
        Weight for MSE component. Default 0.5.
    sign_weight : float
        Weight for sign accuracy component. Default 0.5.
    l2_reg : float
        L2 regularization on predictions. Default 0.0.
    use_huber : bool
        Use Huber instead of MSE for magnitude. Default False.
    huber_delta : float
        Delta for Huber loss. Default 1.0.
    sign_temperature : float
        Temperature for soft sign. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.

    Examples
    --------
    >>> # Prioritize getting the direction right
    >>> criterion = SignAwareLoss(mse_weight=0.3, sign_weight=0.7)
    >>> loss = criterion(pred, target)
    """

    def __init__(
            self,
            mse_weight: float = 0.5,
            sign_weight: float = 0.5,
            l2_reg: float = 0.0,
            use_huber: bool = False,
            huber_delta: float = 1.0,
            sign_temperature: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.sign_weight = sign_weight
        self.l2_reg = l2_reg
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.sign_temperature = sign_temperature
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        # Magnitude component
        if self.use_huber:
            error = pred - target
            abs_error = error.abs()
            quadratic = 0.5 * error ** 2
            linear = self.huber_delta * (abs_error - 0.5 * self.huber_delta)
            magnitude_loss = torch.where(abs_error <= self.huber_delta, quadratic, linear)
        else:
            magnitude_loss = (pred - target) ** 2

        # Sign component (soft sign accuracy)
        soft_sign_pred = torch.tanh(pred / self.sign_temperature)
        soft_sign_target = torch.tanh(target / self.sign_temperature)
        sign_loss = 1.0 - soft_sign_pred * soft_sign_target

        # Combine
        loss = self.mse_weight * magnitude_loss + self.sign_weight * sign_loss

        # L2 regularization on predictions
        if self.l2_reg > 0:
            loss = loss + self.l2_reg * pred ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SharpePenaltyLoss(nn.Module):
    """
    Loss inspired by Sharpe ratio optimization.

    Penalizes predictions that would result in poor risk-adjusted returns.
    The loss considers both the return prediction error and the variance
    of errors (risk).

    Parameters
    ----------
    mse_weight : float
        Weight for base MSE. Default 1.0.
    variance_penalty : float
        Penalty for prediction variance (encourages consistency). Default 0.1.
    direction_penalty : float
        Penalty for wrong sign. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.

    Notes
    -----
    This loss operates on batches and uses batch statistics to approximate
    the variance penalty. For true Sharpe optimization, consider using
    the returns directly in a custom training loop.
    """

    def __init__(
            self,
            mse_weight: float = 1.0,
            variance_penalty: float = 0.1,
            direction_penalty: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.variance_penalty = variance_penalty
        self.direction_penalty = direction_penalty
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        # Base MSE
        error = pred - target
        mse = error ** 2

        # Variance penalty on predictions (encourages consistent predictions)
        pred_var = pred.var()

        # Direction penalty
        wrong_sign = (pred * target) < 0
        direction_term = wrong_sign.float() * error.abs()

        # Per-sample loss
        sample_loss = self.mse_weight * mse + self.direction_penalty * direction_term

        if self.reduction == "mean":
            return sample_loss.mean() + self.variance_penalty * pred_var
        elif self.reduction == "sum":
            return sample_loss.sum() + self.variance_penalty * pred_var
        return sample_loss


class TradingPnLLoss(nn.Module):
    """
    Loss that directly optimizes trading PnL.

    Simulates trading based on predictions:
    - Long when pred > threshold
    - Short when pred < -threshold
    - Flat otherwise

    Loss is negative realized return (we minimize loss = maximize return).

    Parameters
    ----------
    threshold : float
        Prediction threshold for taking positions. Default 0.0.
    transaction_cost : float
        Cost per trade as fraction. Default 0.0.
    wrong_direction_penalty : float
        Extra penalty when position loses money. Default 1.0.
    temperature : float
        Softness of position sizing (differentiable approximation). Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.

    Notes
    -----
    This loss uses soft position sizing (tanh) to maintain differentiability.
    Actual trading would use hard thresholds.
    """

    def __init__(
            self,
            threshold: float = 0.0,
            transaction_cost: float = 0.0,
            wrong_direction_penalty: float = 1.0,
            temperature: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.wrong_direction_penalty = wrong_direction_penalty
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (B,) predicted returns
        target: (B,) actual returns
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # Soft position sizing using tanh
        # position in [-1, 1] based on prediction strength
        position = torch.tanh((pred - self.threshold) / self.temperature)

        # PnL = position * actual_return
        pnl = position * target

        # Transaction cost (proportional to position size)
        cost = self.transaction_cost * position.abs()

        # Net return
        net_return = pnl - cost

        # Extra penalty when position loses money (wrong direction)
        wrong_direction = (position * target) < 0
        penalty = wrong_direction.float() * self.wrong_direction_penalty * (position * target).abs()

        # Loss = negative return + penalty (minimize loss = maximize return)
        loss = -net_return + penalty

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedDirectionalMSE(nn.Module):
    """
    MSE with sample-level weights and directional penalty.

    Useful when some samples are more important than others
    (e.g., high volatility days, regime changes).

    Parameters
    ----------
    direction_penalty : float
        Penalty for wrong direction. Default 1.0.
    base_weight : float
        Base weight for all samples. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            direction_penalty: float = 1.0,
            base_weight: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.direction_penalty = direction_penalty
        self.base_weight = base_weight
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred: (B,) predictions
        target: (B,) targets
        weights: (B,) sample weights, optional
        """
        pred = pred.view(-1)
        target = target.view(-1)

        if weights is None:
            weights = torch.ones_like(pred) * self.base_weight
        else:
            weights = weights.view(-1) * self.base_weight

        # MSE
        mse = (pred - target) ** 2

        # Direction penalty
        wrong_sign = (pred * target) < 0
        direction_term = wrong_sign.float() * (pred - target).abs()

        loss = weights * (mse + self.direction_penalty * direction_term)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedRegressionLoss(nn.Module):
    """
    Flexible combined loss with multiple components.

    Allows mixing different loss functions with configurable weights.

    Parameters
    ----------
    mse_weight : float
        Weight for MSE component. Default 1.0.
    mae_weight : float
        Weight for MAE component. Default 0.0.
    huber_weight : float
        Weight for Huber component. Default 0.0.
    huber_delta : float
        Delta for Huber loss. Default 1.0.
    direction_penalty : float
        Penalty for wrong direction. Default 0.0.
    sign_weight : float
        Weight for sign accuracy loss. Default 0.0.
    l1_pred_reg : float
        L1 regularization on predictions. Default 0.0.
    l2_pred_reg : float
        L2 regularization on predictions. Default 0.0.
    reduction : str
        Reduction mode. Default 'mean'.

    Examples
    --------
    >>> # MSE + MAE + direction penalty
    >>> criterion = CombinedRegressionLoss(
    ...     mse_weight=0.5,
    ...     mae_weight=0.3,
    ...     direction_penalty=0.5,
    ... )
    """

    def __init__(
            self,
            mse_weight: float = 1.0,
            mae_weight: float = 0.0,
            huber_weight: float = 0.0,
            huber_delta: float = 1.0,
            direction_penalty: float = 0.0,
            sign_weight: float = 0.0,
            l1_pred_reg: float = 0.0,
            l2_pred_reg: float = 0.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        self.huber_delta = huber_delta
        self.direction_penalty = direction_penalty
        self.sign_weight = sign_weight
        self.l1_pred_reg = l1_pred_reg
        self.l2_pred_reg = l2_pred_reg
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        error = pred - target
        loss = torch.zeros_like(pred)

        # MSE
        if self.mse_weight > 0:
            loss = loss + self.mse_weight * error ** 2

        # MAE
        if self.mae_weight > 0:
            loss = loss + self.mae_weight * error.abs()

        # Huber
        if self.huber_weight > 0:
            abs_error = error.abs()
            quadratic = 0.5 * error ** 2
            linear = self.huber_delta * (abs_error - 0.5 * self.huber_delta)
            huber = torch.where(abs_error <= self.huber_delta, quadratic, linear)
            loss = loss + self.huber_weight * huber

        # Direction penalty
        if self.direction_penalty > 0:
            wrong_sign = (pred * target) < 0
            loss = loss + self.direction_penalty * wrong_sign.float() * error.abs()

        # Sign accuracy
        if self.sign_weight > 0:
            soft_sign_pred = torch.tanh(pred)
            soft_sign_target = torch.tanh(target)
            sign_loss = 1.0 - soft_sign_pred * soft_sign_target
            loss = loss + self.sign_weight * sign_loss

        # Regularization
        if self.l1_pred_reg > 0:
            loss = loss + self.l1_pred_reg * pred.abs()
        if self.l2_pred_reg > 0:
            loss = loss + self.l2_pred_reg * pred ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss



# ---------------------------------------------------------------------------
# MDN-specific losses
# ---------------------------------------------------------------------------

import math as _math


class MDNNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for a Gaussian Mixture Density Network.

    p(y | x) = Σ_k π_k · N(y; μ_k, σ_k²)
    Loss = -log p(y | x), computed via log-sum-exp for numerical stability.

    Parameters
    ----------
    reduction : str
        'mean' or 'sum'. Default 'mean'.

    Usage
    -----
        criterion = MDNNLLLoss()
        pi, mu, sigma = model(x)
        loss = criterion(pi, mu, sigma, y)
    """

    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pi:    (B, K) mixing weights
            mu:    (B, K) component means
            sigma: (B, K) component std devs
            y:     (B,) or (B, 1) targets

        Returns:
            Scalar NLL
        """
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # (B, 1)

        log_pi = torch.log(pi + 1e-10)
        log_normal = (
            -0.5 * _math.log(2 * _math.pi)
            - torch.log(sigma)
            - 0.5 * ((y - mu) / sigma) ** 2
        )
        log_mix = torch.logsumexp(log_pi + log_normal, dim=-1)  # (B,)

        if self.reduction == "mean":
            return -log_mix.mean()
        return -log_mix.sum()


class MDNEntropyRegularizer(nn.Module):
    """
    Entropy regularization for MDN mixing weights.

    Encourages component utilization by penalizing low entropy in π.
    Prevents mode collapse where one component absorbs all the weight.

    The penalty is a hinge loss: max(0, target_entropy − batch_entropy),
    so it only activates when entropy falls below the target fraction of
    the theoretical maximum log(K).

    Parameters
    ----------
    target_entropy_frac : float
        Target entropy as a fraction of max entropy log(K). Default 0.5.

    Usage
    -----
        nll_loss  = MDNNLLLoss()
        ent_reg   = MDNEntropyRegularizer(target_entropy_frac=0.5)

        pi, mu, sigma = model(x)
        loss = nll_loss(pi, mu, sigma, y) + 0.1 * ent_reg(pi)
    """

    def __init__(self, target_entropy_frac: float = 0.5):
        super().__init__()
        self.target_entropy_frac = target_entropy_frac

    def forward(self, pi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pi: (B, K) mixing weights

        Returns:
            Scalar hinge penalty
        """
        K = pi.shape[1]
        max_entropy = _math.log(K)
        target = self.target_entropy_frac * max_entropy

        pi_avg = pi.mean(dim=0)                               # (K,)
        entropy = -(pi_avg * torch.log(pi_avg + 1e-10)).sum()

        return torch.relu(target - entropy)


# Convenience exports
__all__ = [
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
    # MDN
    'MDNNLLLoss',
    'MDNEntropyRegularizer',
]
