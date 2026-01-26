"""
Classification Losses for Trading Applications.

These losses extend standard cross-entropy with:
1. Ordinal awareness - penalize predicting opposite direction more
2. Trading cost awareness - incorporate transaction costs
3. Profit weighting - scale by potential PnL
4. Confidence calibration - penalize overconfident wrong predictions
5. Anti-collapse regularization - prevent degenerate predictions

Classes are typically ordinal: 0=down/short, 1=flat/neutral, 2=up/long
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, List


class DistanceWeightedCE(nn.Module):
    """
    Penalize misclassifications more when predicted class is farther from the true class.
    Classes must be ordinal (e.g., down=0, flat=1, up=2).
    """
    def __init__(self, alpha: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        y_true: (B,) long
        """
        ce = F.cross_entropy(logits, y_true, reduction="none")  # (B,)

        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)                       # (B,)
            dist = (y_hat - y_true).abs().float()               # (B,)
            w = 1.0 + self.alpha * dist                         # (B,)

        loss = ce * w
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class OrdinalCEWithAntiCollapse(nn.Module):
    """
    3-class ordinal loss with:
      - distance-weighted CE (bigger penalty for down<->up mistakes)
      - anti-collapse regularizer (match batch-average predicted distribution to target)
    """
    def __init__(
        self,
        alpha: float = 1.0,          # distance penalty strength
        reg_lambda: float = 0.05,    # anti-collapse strength
        target_probs=None,           # tensor shape (3,) or None -> uniform
        reduction: str = "mean",
        eps: float = 1e-8
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.reg_lambda = float(reg_lambda)
        self.reduction = reduction
        self.eps = eps

        if target_probs is None:
            target_probs = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
        else:
            target_probs = torch.tensor(target_probs, dtype=torch.float32)
            target_probs = target_probs / target_probs.sum()

        self.register_buffer("target_probs", target_probs)

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 3)
        y_true: (B,) long in {0,1,2}
        """
        # --- base CE per sample ---
        ce = F.cross_entropy(logits, y_true, reduction="none")  # (B,)

        # --- distance penalty weight using argmax ---
        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)                       # (B,)
            dist = (y_hat - y_true).abs().float()               # (B,) in {0,1,2}
            w = 1.0 + self.alpha * dist                         # (B,)

        main = (ce * w).mean() if self.reduction == "mean" else (ce * w).sum()

        # --- anti-collapse: KL( target || mean_pred ) ---
        probs = logits.softmax(dim=-1)                          # (B,3)
        mean_pred = probs.mean(dim=0).clamp(self.eps, 1.0)      # (3,)
        target = self.target_probs.clamp(self.eps, 1.0)

        # KL(target || mean_pred) = sum target * log(target/mean_pred)
        kl = (target * (target.log() - mean_pred.log())).sum()

        return main + self.reg_lambda * kl


# =============================================================================
# Trading-Inspired Classification Losses
# =============================================================================


class TradingCostAwareCE(nn.Module):
    """
    Cross-entropy that incorporates trading/transaction costs.

    For a 3-class setup (down=0, flat=1, up=2):
    - Predicting flat when should trade: opportunity cost
    - Predicting trade when should be flat: transaction cost
    - Predicting wrong direction: transaction cost + loss

    Parameters
    ----------
    transaction_cost : float
        Cost of entering/exiting a position. Default 0.001 (10 bps).
    opportunity_cost : float
        Cost of missing a trade. Default 0.5 (relative weight).
    direction_cost : float
        Extra cost for wrong direction (on top of transaction). Default 2.0.
    class_weights : list or tensor, optional
        Base weights for each class.
    reduction : str
        Reduction mode. Default 'mean'.

    Examples
    --------
    >>> criterion = TradingCostAwareCE(transaction_cost=0.001, direction_cost=3.0)
    >>> loss = criterion(logits, labels)
    """

    def __init__(
            self,
            transaction_cost: float = 0.001,
            opportunity_cost: float = 0.5,
            direction_cost: float = 2.0,
            class_weights: Optional[List[float]] = None,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.transaction_cost = transaction_cost
        self.opportunity_cost = opportunity_cost
        self.direction_cost = direction_cost
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 3) for classes [down, flat, up]
        y_true: (B,) long in {0, 1, 2}
        """
        # Base CE
        if self.class_weights is not None:
            ce = F.cross_entropy(logits, y_true, weight=self.class_weights, reduction="none")
        else:
            ce = F.cross_entropy(logits, y_true, reduction="none")

        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)

            # Build cost multiplier based on prediction vs truth
            cost_mult = torch.ones_like(ce)

            # Case 1: Predicted flat (1) when should trade (0 or 2)
            # Opportunity cost
            should_trade = (y_true != 1)
            pred_flat = (y_hat == 1)
            opportunity_mask = should_trade & pred_flat
            cost_mult = cost_mult + opportunity_mask.float() * self.opportunity_cost

            # Case 2: Predicted trade (0 or 2) when should be flat (1)
            # Transaction cost incurred for nothing
            should_flat = (y_true == 1)
            pred_trade = (y_hat != 1)
            false_trade_mask = should_flat & pred_trade
            cost_mult = cost_mult + false_trade_mask.float() * self.transaction_cost * 100

            # Case 3: Wrong direction (predicted 0 when true 2, or vice versa)
            # This is the worst: transaction cost + directional loss
            wrong_direction = ((y_true == 0) & (y_hat == 2)) | ((y_true == 2) & (y_hat == 0))
            cost_mult = cost_mult + wrong_direction.float() * self.direction_cost

        loss = ce * cost_mult

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ProfitWeightedCE(nn.Module):
    """
    Cross-entropy weighted by potential profit/loss.

    Samples with larger potential moves are weighted more heavily.
    Useful when you have access to the actual returns for each sample.

    Parameters
    ----------
    profit_scale : float
        Scale factor for profit weighting. Default 1.0.
    min_weight : float
        Minimum sample weight (prevents zero weights). Default 0.1.
    max_weight : float
        Maximum sample weight (prevents outlier dominance). Default 10.0.
    direction_penalty : float
        Extra penalty for wrong direction. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.

    Examples
    --------
    >>> criterion = ProfitWeightedCE(profit_scale=100.0)
    >>> # returns: actual returns for weighting
    >>> loss = criterion(logits, labels, returns=actual_returns)
    """

    def __init__(
            self,
            profit_scale: float = 1.0,
            min_weight: float = 0.1,
            max_weight: float = 10.0,
            direction_penalty: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.profit_scale = profit_scale
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.direction_penalty = direction_penalty
        self.reduction = reduction

    def forward(
            self,
            logits: torch.Tensor,
            y_true: torch.Tensor,
            returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        logits: (B, C)
        y_true: (B,)
        returns: (B,) optional actual returns for profit weighting
        """
        ce = F.cross_entropy(logits, y_true, reduction="none")

        # Compute sample weights based on return magnitude
        if returns is not None:
            returns = returns.view(-1)
            weights = (returns.abs() * self.profit_scale).clamp(self.min_weight, self.max_weight)
        else:
            weights = torch.ones_like(ce)

        # Direction penalty for wrong sign predictions
        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)
            # Map classes to directions: 0->-1, 1->0, 2->1
            pred_dir = y_hat.float() - 1.0
            true_dir = y_true.float() - 1.0
            wrong_dir = (pred_dir * true_dir) < 0
            weights = weights + wrong_dir.float() * self.direction_penalty

        loss = ce * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ConfidencePenalizedCE(nn.Module):
    """
    Cross-entropy with penalty for overconfident wrong predictions.

    High confidence wrong predictions are penalized more than
    low confidence wrong predictions. Encourages calibration.

    Parameters
    ----------
    confidence_penalty : float
        Multiplier for confidence-based penalty. Default 1.0.
    temperature : float
        Temperature for confidence calculation. Default 1.0.
    label_smoothing : float
        Label smoothing factor. Default 0.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            confidence_penalty: float = 1.0,
            temperature: float = 1.0,
            label_smoothing: float = 0.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.confidence_penalty = confidence_penalty
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, y_true,
            reduction="none",
            label_smoothing=self.label_smoothing
        )

        with torch.no_grad():
            # Get prediction confidence (max probability)
            probs = F.softmax(logits / self.temperature, dim=-1)
            confidence = probs.max(dim=-1).values  # (B,)

            # Check if prediction is wrong
            y_hat = logits.argmax(dim=-1)
            wrong = (y_hat != y_true).float()

            # Penalty: high confidence + wrong = bad
            penalty = wrong * confidence * self.confidence_penalty

        loss = ce * (1.0 + penalty)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalDirectionalLoss(nn.Module):
    """
    Focal loss with directional awareness for ordinal classes.

    Focal loss focuses on hard examples by down-weighting easy ones.
    This variant adds extra focus on directionally-wrong predictions.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values focus more on hard examples.
        Default 2.0.
    alpha : float or list
        Class balancing weights. Default None (uniform).
    direction_gamma : float
        Extra gamma for wrong direction predictions. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", 2017
    """

    def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[float] = None,
            direction_gamma: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.direction_gamma = direction_gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            else:
                self.register_buffer("alpha", torch.tensor([alpha], dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)

        # Get probability of true class
        y_true_onehot = F.one_hot(y_true, n_classes).float()
        pt = (probs * y_true_onehot).sum(dim=-1)  # (B,)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Direction penalty: extra gamma for wrong direction
        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)
            wrong_direction = ((y_true == 0) & (y_hat == 2)) | ((y_true == 2) & (y_hat == 0))
            direction_weight = 1.0 + wrong_direction.float() * self.direction_gamma

        # Base CE
        ce = F.cross_entropy(logits, y_true, reduction="none")

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[y_true]
            loss = alpha_t * focal_weight * direction_weight * ce
        else:
            loss = focal_weight * direction_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ExpectedPnLLoss(nn.Module):
    """
    Optimize expected P&L based on class probabilities.

    Instead of minimizing classification error, directly optimizes
    the expected profit assuming positions are sized by probability.

    For classes [down, flat, up] with returns r:
    - Expected PnL = p(down)*(-r) + p(flat)*0 + p(up)*(+r)
    - Loss = -Expected PnL (we minimize, so maximize PnL)

    Parameters
    ----------
    ce_weight : float
        Weight for cross-entropy regularization. Default 0.1.
    transaction_cost : float
        Transaction cost deducted from PnL. Default 0.0.
    temperature : float
        Temperature for probability sharpening. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.

    Examples
    --------
    >>> criterion = ExpectedPnLLoss(ce_weight=0.1)
    >>> loss = criterion(logits, labels, returns=actual_returns)
    """

    def __init__(
            self,
            ce_weight: float = 0.1,
            transaction_cost: float = 0.0,
            temperature: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.transaction_cost = transaction_cost
        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            logits: torch.Tensor,
            y_true: torch.Tensor,
            returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        logits: (B, 3) for [down, flat, up]
        y_true: (B,)
        returns: (B,) actual returns (required for PnL calculation)
        """
        probs = F.softmax(logits / self.temperature, dim=-1)  # (B, 3)

        if returns is not None:
            returns = returns.view(-1)

            # Position based on probability: p(up) - p(down)
            # This gives position in [-1, 1]
            position = probs[:, 2] - probs[:, 0]  # (B,)

            # PnL = position * return - transaction_cost * |position|
            pnl = position * returns - self.transaction_cost * position.abs()

            # Loss = negative PnL (minimize loss = maximize PnL)
            pnl_loss = -pnl
        else:
            # No returns provided, fall back to pure CE
            pnl_loss = torch.zeros(logits.size(0), device=logits.device)

        # CE regularization
        ce = F.cross_entropy(logits, y_true, reduction="none")

        loss = pnl_loss + self.ce_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MarginOrdinalLoss(nn.Module):
    """
    Ordinal loss with margin requirement between classes.

    Requires a minimum margin between the logit of the true class
    and adjacent/distant classes. Encourages confident predictions.

    Parameters
    ----------
    margin_adjacent : float
        Required margin between true class and adjacent class. Default 0.5.
    margin_distant : float
        Required margin between true class and distant class (e.g., 0 vs 2).
        Default 1.0.
    margin_loss_weight : float
        Weight for margin violation penalty. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            margin_adjacent: float = 0.5,
            margin_distant: float = 1.0,
            margin_loss_weight: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.margin_adjacent = margin_adjacent
        self.margin_distant = margin_distant
        self.margin_loss_weight = margin_loss_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 3)
        y_true: (B,) in {0, 1, 2}
        """
        B = logits.size(0)

        # Base CE
        ce = F.cross_entropy(logits, y_true, reduction="none")

        # Margin loss
        margin_loss = torch.zeros(B, device=logits.device)

        # Get logits for true class
        true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)  # (B,)

        for c in range(3):
            mask = (y_true == c)
            if not mask.any():
                continue

            # Compare against other classes
            for other_c in range(3):
                if other_c == c:
                    continue

                # Determine required margin based on distance
                dist = abs(c - other_c)
                required_margin = self.margin_distant if dist == 2 else self.margin_adjacent

                # Margin violation: true_logit should be > other_logit + margin
                other_logits = logits[:, other_c]
                violation = F.relu(other_logits + required_margin - true_logits)

                margin_loss = margin_loss + mask.float() * violation

        loss = ce + self.margin_loss_weight * margin_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricDirectionalCE(nn.Module):
    """
    Cross-entropy with asymmetric penalties for bullish vs bearish mistakes.

    Useful when the cost of missing upside differs from the cost of
    catching downside (e.g., long-only strategies, hedging).

    Parameters
    ----------
    bullish_weight : float
        Weight for bullish mistakes (predicting down when true is up). Default 1.0.
    bearish_weight : float
        Weight for bearish mistakes (predicting up when true is down). Default 1.0.
    flat_miss_weight : float
        Weight for missing trading opportunities. Default 0.5.
    reduction : str
        Reduction mode. Default 'mean'.

    Examples
    --------
    >>> # Long-only: penalize missing upside more
    >>> criterion = AsymmetricDirectionalCE(bullish_weight=2.0, bearish_weight=1.0)
    """

    def __init__(
            self,
            bullish_weight: float = 1.0,
            bearish_weight: float = 1.0,
            flat_miss_weight: float = 0.5,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.bullish_weight = bullish_weight
        self.bearish_weight = bearish_weight
        self.flat_miss_weight = flat_miss_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, y_true, reduction="none")

        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)

            weights = torch.ones_like(ce)

            # Bullish mistake: predicted down (0) when true is up (2)
            bullish_miss = (y_true == 2) & (y_hat == 0)
            weights = weights + bullish_miss.float() * (self.bullish_weight - 1.0)

            # Bearish mistake: predicted up (2) when true is down (0)
            bearish_miss = (y_true == 0) & (y_hat == 2)
            weights = weights + bearish_miss.float() * (self.bearish_weight - 1.0)

            # Flat miss: predicted flat (1) when should have traded
            flat_when_should_trade = (y_hat == 1) & (y_true != 1)
            weights = weights + flat_when_should_trade.float() * (self.flat_miss_weight - 1.0)

        loss = ce * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SharpeInspiredCE(nn.Module):
    """
    Cross-entropy with Sharpe ratio-inspired regularization.

    Penalizes predictions that would lead to high variance in returns.
    Encourages consistent predictions across the batch.

    Parameters
    ----------
    variance_penalty : float
        Penalty for prediction variance. Default 0.1.
    consistency_bonus : float
        Bonus for consistent (stable) predictions. Default 0.05.
    direction_penalty : float
        Penalty for wrong direction. Default 1.0.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            variance_penalty: float = 0.1,
            consistency_bonus: float = 0.05,
            direction_penalty: float = 1.0,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.variance_penalty = variance_penalty
        self.consistency_bonus = consistency_bonus
        self.direction_penalty = direction_penalty
        self.reduction = reduction

    def forward(
            self,
            logits: torch.Tensor,
            y_true: torch.Tensor,
            returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, y_true, reduction="none")

        probs = F.softmax(logits, dim=-1)

        with torch.no_grad():
            y_hat = logits.argmax(dim=-1)

            # Direction penalty
            wrong_dir = ((y_true == 0) & (y_hat == 2)) | ((y_true == 2) & (y_hat == 0))
            dir_weight = 1.0 + wrong_dir.float() * self.direction_penalty

        loss = ce * dir_weight

        # Variance penalty: penalize high variance in predicted positions
        # Position = p(up) - p(down)
        positions = probs[:, 2] - probs[:, 0] if probs.size(1) == 3 else probs[:, 1] - probs[:, 0]
        position_variance = positions.var()

        # Consistency: reward when prediction matches batch consensus
        consensus = positions.mean()
        consistency_loss = (positions - consensus).abs().mean()

        batch_loss = loss.mean() if self.reduction == "mean" else loss.sum()
        total_loss = batch_loss + self.variance_penalty * position_variance + self.consistency_bonus * consistency_loss

        if self.reduction == "none":
            return loss
        return total_loss


class HierarchicalDirectionalLoss(nn.Module):
    """
    Two-stage hierarchical loss for ordinal classification.

    Stage 1: Binary direction (up vs down, ignoring flat)
    Stage 2: Full 3-class classification

    This helps the model first learn direction, then refine.

    Parameters
    ----------
    direction_weight : float
        Weight for binary direction loss. Default 0.5.
    full_weight : float
        Weight for full 3-class loss. Default 0.5.
    reduction : str
        Reduction mode. Default 'mean'.
    """

    def __init__(
            self,
            direction_weight: float = 0.5,
            full_weight: float = 0.5,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.full_weight = full_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 3) for [down, flat, up]
        y_true: (B,) in {0, 1, 2}
        """
        # Full 3-class CE
        ce_full = F.cross_entropy(logits, y_true, reduction="none")

        # Binary direction: collapse to up vs down (ignore flat samples)
        # Create binary target: 0->0, 2->1
        directional_mask = (y_true != 1)  # Exclude flat

        if directional_mask.any():
            # Binary logits: [logit_down, logit_up]
            binary_logits = torch.stack([logits[:, 0], logits[:, 2]], dim=-1)
            binary_target = (y_true == 2).long()  # 0 for down, 1 for up

            ce_direction = F.cross_entropy(binary_logits, binary_target, reduction="none")
            ce_direction = ce_direction * directional_mask.float()
        else:
            ce_direction = torch.zeros_like(ce_full)

        loss = self.full_weight * ce_full + self.direction_weight * ce_direction

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# Convenience exports
__all__ = [
    # Base ordinal losses
    'DistanceWeightedCE',
    'OrdinalCEWithAntiCollapse',
    # Trading-inspired losses
    'TradingCostAwareCE',
    'ProfitWeightedCE',
    'ConfidencePenalizedCE',
    'FocalDirectionalLoss',
    'ExpectedPnLLoss',
    'MarginOrdinalLoss',
    'AsymmetricDirectionalCE',
    'SharpeInspiredCE',
    'HierarchicalDirectionalLoss',
]