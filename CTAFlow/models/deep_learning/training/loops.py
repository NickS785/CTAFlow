"""
Training and evaluation loops for deep learning models.

Provides specialized loops for:
- Classification with trading-aware losses (ProfitWeightedCE, ExpectedPnLLoss, etc.)
- Regression with directional penalties
- PnL tracking and trading metrics
- Multi-modal model support (WSPR, TriModal, etc.)
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CTAFlow.models.deep_learning.training.loss.clf import (
    ProfitWeightedCE,
    ExpectedPnLLoss,
    HierarchicalDirectionalLoss,
    CostAwareCE,
)


# =============================================================================
# Batch Unpacking Utilities
# =============================================================================


def _unpack_batch(batch_data: Union[tuple, list]) -> Tuple[Union[torch.Tensor, tuple], torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
    """
    Unpack batch data from various dataloader formats.

    Handles:
    - Standard: (inputs, targets) or (inputs, targets, returns)
    - Multi-modal TriModal: (summary, seq, seq_lens, profile, raster, targets) or + returns
    - MultiAssetWSPR: (summary, profile, raster, seq, seq_lens, targets, meta)
    - With dates: (..., dates_list) - dates are ignored

    Parameters
    ----------
    batch_data : tuple or list
        Batch from DataLoader.

    Returns
    -------
    inputs : torch.Tensor or tuple
        Model inputs (single tensor or tuple for multi-modal).
    targets : torch.Tensor
        Target labels or values.
    returns : torch.Tensor or None
        Actual returns for PnL calculation (if present).
    meta : dict or None
        Meta dictionary for MultiAssetWSPR (if present).
    """
    # Filter out dates (last element as list)
    if len(batch_data) > 0 and isinstance(batch_data[-1], list):
        batch_data = batch_data[:-1]

    # Check for meta dict (MultiAssetWSPR format)
    meta = None
    has_meta = False
    if len(batch_data) > 0 and isinstance(batch_data[-1], dict):
        meta = batch_data[-1]
        batch_data = batch_data[:-1]
        has_meta = True

    # Now batch_data contains only tensors

    # MultiAssetWSPR format: (summary, profile, raster, seq, seq_lens, targets, returns?)
    # After meta removed, we have 6 or 7 elements
    if has_meta:
        if len(batch_data) == 6:
            # No returns
            inputs = tuple(batch_data[:5])  # (summary, profile, raster, seq, seq_lens)
            targets = batch_data[5]
            returns = None
        elif len(batch_data) == 7:
            # With returns
            inputs = tuple(batch_data[:5])  # (summary, profile, raster, seq, seq_lens)
            targets = batch_data[5]
            returns = batch_data[6]
        else:
            # Unexpected format with meta
            *inputs, targets = batch_data
            if len(inputs) == 1:
                inputs = inputs[0]
            else:
                inputs = tuple(inputs)
            returns = None
    # Standard/TriModal formats
    else:
        if len(batch_data) == 2:
            # Standard: (inputs, targets)
            inputs, targets = batch_data
            returns = None
        elif len(batch_data) == 3:
            # Standard: (inputs, targets, returns)
            inputs, targets, returns = batch_data
        else:
            # Multi-modal: TriModal/Dual windowed datasets
            # Format: (summary, seq, seq_lens, profile, raster, targets) or + returns
            # Target is always second-to-last (if no returns) or third-to-last (if returns)
            # Returns (if present) is last element and is a 1D tensor (batch_size,)

            # Check if last element is returns
            # Returns are 1D tensors matching target shape
            if batch_data[-1].ndim == 1 and batch_data[-2].ndim == 1:
                # Both last and second-to-last are 1D: last is returns, second-to-last is targets
                *inputs, targets, returns = batch_data
            else:
                # Last element is target (1D), everything before is inputs
                *inputs, targets = batch_data
                returns = None

            # Package inputs as tuple
            if len(inputs) == 1:
                inputs = inputs[0]
            else:
                inputs = tuple(inputs)

    return inputs, targets, returns, meta


# =============================================================================
# Configuration and Metrics
# =============================================================================


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    accuracy: Optional[float] = None
    pnl: Optional[float] = None
    sharpe: Optional[float] = None
    directional_accuracy: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


def compute_trading_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    returns: Optional[np.ndarray] = None,
    transaction_cost: float = 0.0,
    task: str = 'classification',
) -> Dict[str, float]:
    """
    Compute trading metrics from predictions and targets.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions. For classification: class labels or probabilities.
        For regression: predicted returns.
    targets : np.ndarray
        True labels (classification) or returns (regression).
    returns : np.ndarray, optional
        Actual returns for PnL calculation. Required for classification.
    transaction_cost : float
        Transaction cost per trade (e.g., 0.001 for 10 bps).
    task : str
        'classification' or 'regression'.

    Returns
    -------
    Dict[str, float]
        Dictionary of trading metrics.
    """
    metrics = {}

    if task == 'classification':
        # Extract class predictions if probabilities provided
        if predictions.ndim > 1:
            class_preds = predictions.argmax(axis=1)
        else:
            class_preds = predictions.astype(int)

        # Accuracy
        metrics['accuracy'] = (class_preds == targets).mean() * 100

        # Directional accuracy (excluding flat predictions/targets)
        if predictions.ndim > 1 and predictions.shape[1] == 3:
            trade_mask = (class_preds != 1) & (targets != 1)
            if trade_mask.any():
                dir_correct = (class_preds[trade_mask] == targets[trade_mask]).sum()
                metrics['directional_accuracy'] = (dir_correct / trade_mask.sum()) * 100

        # PnL calculation if returns provided
        if returns is not None:
            if predictions.ndim > 1 and predictions.shape[1] == 3:
                # Probabilistic position: p(up) - p(down)
                probs = predictions
                positions = probs[:, 2] - probs[:, 0]
            else:
                # Discrete position: -1, 0, +1
                positions = class_preds.astype(float) - 1.0

            # PnL = position * return - transaction_cost * |position|
            pnl_per_sample = positions * returns - transaction_cost * np.abs(positions)
            metrics['pnl'] = pnl_per_sample.mean()

            # Sharpe ratio (annualized, assuming daily data)
            if len(pnl_per_sample) > 1:
                pnl_std = pnl_per_sample.std()
                if pnl_std > 0:
                    metrics['sharpe'] = (pnl_per_sample.mean() / pnl_std) * np.sqrt(252)

            # Win rate and avg win/loss
            winning_trades = pnl_per_sample > 0
            losing_trades = pnl_per_sample < 0

            if winning_trades.any():
                metrics['win_rate'] = winning_trades.mean() * 100
                metrics['avg_win'] = pnl_per_sample[winning_trades].mean()

            if losing_trades.any():
                metrics['avg_loss'] = pnl_per_sample[losing_trades].mean()

            # Maximum drawdown
            cumulative_pnl = np.cumsum(pnl_per_sample)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            metrics['max_drawdown'] = drawdown.max() if len(drawdown) > 0 else 0.0

    else:  # regression
        # Directional accuracy
        pred_sign = np.sign(predictions)
        true_sign = np.sign(targets)
        metrics['directional_accuracy'] = (pred_sign == true_sign).mean() * 100

        # PnL using predicted returns as position sizing
        positions = np.clip(predictions, -1, 1)  # Clip to [-1, 1] for safety
        pnl_per_sample = positions * targets - transaction_cost * np.abs(positions)
        metrics['pnl'] = pnl_per_sample.mean()

        # MSE and MAE
        metrics['mse'] = ((predictions - targets) ** 2).mean()
        metrics['mae'] = np.abs(predictions - targets).mean()

        # Sharpe
        if len(pnl_per_sample) > 1:
            pnl_std = pnl_per_sample.std()
            if pnl_std > 0:
                metrics['sharpe'] = (pnl_per_sample.mean() / pnl_std) * np.sqrt(252)

    return metrics


# =============================================================================
# Classification Training Loops
# =============================================================================


def train_classification_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    transaction_cost: float = 0.0,
    gradient_clip: Optional[float] = 1.0,
    use_amp: bool = False,
    return_predictions: bool = False,
) -> Union[TrainingMetrics, Tuple[TrainingMetrics, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Training loop for classification tasks with trading-aware losses.

    Supports losses that require returns (ProfitWeightedCE, ExpectedPnLLoss)
    and tracks trading metrics (PnL, Sharpe, directional accuracy).

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : DataLoader
        Training data loader. Should return (inputs, targets) or
        (inputs, targets, returns) for trading-aware losses.
    criterion : nn.Module
        Loss function. Can be ProfitWeightedCE, ExpectedPnLLoss, etc.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : str or torch.device
        Device to run on.
    transaction_cost : float
        Transaction cost for PnL calculation.
    gradient_clip : float, optional
        Gradient clipping value. None to disable.
    use_amp : bool
        Use automatic mixed precision.
    return_predictions : bool
        If True, return predictions, targets, and returns arrays.

    Returns
    -------
    TrainingMetrics or tuple
        Training metrics, optionally with predictions/targets/returns.

    Examples
    --------
    >>> criterion = ProfitWeightedCE(profit_scale=50.0)
    >>> metrics = train_classification_epoch(model, train_loader, criterion, optimizer, device)
    >>> print(f"Loss: {metrics.loss:.4f}, PnL: {metrics.pnl:.6f}")
    """
    model.train()
    device = torch.device(device) if isinstance(device, str) else device

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_returns = []

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Determine if criterion needs returns
    needs_returns = isinstance(criterion, (ProfitWeightedCE, ExpectedPnLLoss))

    for batch_data in loader:
        # Unpack batch - handle different data formats
        inputs, targets, returns, meta = _unpack_batch(batch_data)

        # Move to device
        if isinstance(inputs, (tuple, list)):
            inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
        else:
            inputs = inputs.to(device)

        targets = targets.to(device).long()

        if returns is not None:
            returns = returns.to(device).float()

        optimizer.zero_grad()

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Handle meta dict for MultiAssetWSPR
            if isinstance(inputs, tuple):
                if meta is not None:
                    outputs = model(*inputs, meta)
                else:
                    outputs = model(*inputs)
            else:
                outputs = model(inputs)

            # Compute loss
            if needs_returns and returns is not None:
                loss = criterion(outputs, targets, returns=returns)
            else:
                loss = criterion(outputs, targets)

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Track metrics
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Collect predictions for metrics
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            if returns is not None:
                all_returns.append(returns.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    returns_np = np.concatenate(all_returns, axis=0) if all_returns else None

    trading_metrics = compute_trading_metrics(
        predictions, targets_np, returns_np,
        transaction_cost=transaction_cost,
        task='classification'
    )

    metrics = TrainingMetrics(
        loss=avg_loss,
        accuracy=trading_metrics.get('accuracy'),
        pnl=trading_metrics.get('pnl'),
        sharpe=trading_metrics.get('sharpe'),
        directional_accuracy=trading_metrics.get('directional_accuracy'),
        max_drawdown=trading_metrics.get('max_drawdown'),
        win_rate=trading_metrics.get('win_rate'),
        avg_win=trading_metrics.get('avg_win'),
        avg_loss=trading_metrics.get('avg_loss'),
    )

    if return_predictions:
        return metrics, predictions, targets_np, returns_np
    return metrics


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    transaction_cost: float = 0.0,
    use_amp: bool = False,
    return_predictions: bool = False,
) -> Union[TrainingMetrics, Tuple[TrainingMetrics, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Evaluation loop for classification tasks.

    Same as train_classification_epoch but without gradient updates.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Validation/test data loader.
    criterion : nn.Module
        Loss function.
    device : str or torch.device
        Device to run on.
    transaction_cost : float
        Transaction cost for PnL calculation.
    use_amp : bool
        Use automatic mixed precision.
    return_predictions : bool
        If True, return predictions, targets, and returns arrays.

    Returns
    -------
    TrainingMetrics or tuple
        Validation metrics, optionally with predictions/targets/returns.
    """
    was_training = model.training
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_returns = []

    needs_returns = isinstance(criterion, (ProfitWeightedCE, ExpectedPnLLoss))

    with torch.no_grad():
        for batch_data in loader:
            # Unpack batch - handle different data formats
            inputs, targets, returns, meta = _unpack_batch(batch_data)

            # Move to device
            if isinstance(inputs, (tuple, list)):
                inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
            else:
                inputs = inputs.to(device)

            targets = targets.to(device).long()

            if returns is not None:
                returns = returns.to(device).float()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Handle meta dict for MultiAssetWSPR
                if isinstance(inputs, tuple):
                    if meta is not None:
                        outputs = model(*inputs, meta)
                    else:
                        outputs = model(*inputs)
                else:
                    outputs = model(inputs)

                if needs_returns and returns is not None:
                    loss = criterion(outputs, targets, returns=returns)
                else:
                    loss = criterion(outputs, targets)

            # Track metrics
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            if returns is not None:
                all_returns.append(returns.cpu().numpy())

    # Restore training state
    if was_training:
        model.train()

    # Compute metrics
    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    returns_np = np.concatenate(all_returns, axis=0) if all_returns else None

    trading_metrics = compute_trading_metrics(
        predictions, targets_np, returns_np,
        transaction_cost=transaction_cost,
        task='classification'
    )

    metrics = TrainingMetrics(
        loss=avg_loss,
        accuracy=trading_metrics.get('accuracy'),
        pnl=trading_metrics.get('pnl'),
        sharpe=trading_metrics.get('sharpe'),
        directional_accuracy=trading_metrics.get('directional_accuracy'),
        max_drawdown=trading_metrics.get('max_drawdown'),
        win_rate=trading_metrics.get('win_rate'),
        avg_win=trading_metrics.get('avg_win'),
        avg_loss=trading_metrics.get('avg_loss'),
    )

    if return_predictions:
        return metrics, predictions, targets_np, returns_np
    return metrics


# =============================================================================
# Regression Training Loops
# =============================================================================


def train_regression_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    transaction_cost: float = 0.0,
    gradient_clip: Optional[float] = 1.0,
    use_amp: bool = False,
    return_predictions: bool = False,
) -> Union[TrainingMetrics, Tuple[TrainingMetrics, np.ndarray, np.ndarray]]:
    """
    Training loop for regression tasks with directional penalties.

    Supports DirectionalMSE, DirectionalMAE, DirectionalHuber, etc.
    Tracks directional accuracy and trading PnL.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : DataLoader
        Training data loader. Returns (inputs, targets).
    criterion : nn.Module
        Regression loss function with directional penalties.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : str or torch.device
        Device to run on.
    transaction_cost : float
        Transaction cost for PnL calculation.
    gradient_clip : float, optional
        Gradient clipping value.
    use_amp : bool
        Use automatic mixed precision.
    return_predictions : bool
        If True, return predictions and targets arrays.

    Returns
    -------
    TrainingMetrics or tuple
        Training metrics, optionally with predictions/targets.
    """
    model.train()
    device = torch.device(device) if isinstance(device, str) else device

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_data in loader:
        # Unpack batch - handle different data formats
        inputs, targets, returns, meta = _unpack_batch(batch_data)

        # Move to device
        if isinstance(inputs, (tuple, list)):
            inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
        else:
            inputs = inputs.to(device)

        targets = targets.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Handle meta dict for MultiAssetWSPR
            if isinstance(inputs, tuple):
                if meta is not None:
                    outputs = model(*inputs, meta)
                else:
                    outputs = model(*inputs)
            else:
                outputs = model(inputs)
            outputs = outputs.squeeze(-1) if outputs.ndim > 1 else outputs
            loss = criterion(outputs, targets)

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Track metrics
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        with torch.no_grad():
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    trading_metrics = compute_trading_metrics(
        predictions, targets_np,
        transaction_cost=transaction_cost,
        task='regression'
    )

    metrics = TrainingMetrics(
        loss=avg_loss,
        pnl=trading_metrics.get('pnl'),
        sharpe=trading_metrics.get('sharpe'),
        directional_accuracy=trading_metrics.get('directional_accuracy'),
    )

    if return_predictions:
        return metrics, predictions, targets_np
    return metrics


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    transaction_cost: float = 0.0,
    use_amp: bool = False,
    return_predictions: bool = False,
) -> Union[TrainingMetrics, Tuple[TrainingMetrics, np.ndarray, np.ndarray]]:
    """
    Evaluation loop for regression tasks.

    Same as train_regression_epoch but without gradient updates.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Validation/test data loader.
    criterion : nn.Module
        Regression loss function.
    device : str or torch.device
        Device to run on.
    transaction_cost : float
        Transaction cost for PnL calculation.
    use_amp : bool
        Use automatic mixed precision.
    return_predictions : bool
        If True, return predictions and targets arrays.

    Returns
    -------
    TrainingMetrics or tuple
        Validation metrics, optionally with predictions/targets.
    """
    was_training = model.training
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in loader:
            # Unpack batch - handle different data formats
            inputs, targets, returns, meta = _unpack_batch(batch_data)

            # Move to device
            if isinstance(inputs, (tuple, list)):
                inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
            else:
                inputs = inputs.to(device)

            targets = targets.to(device).float()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Handle meta dict for MultiAssetWSPR
                if isinstance(inputs, tuple):
                    if meta is not None:
                        outputs = model(*inputs, meta)
                    else:
                        outputs = model(*inputs)
                else:
                    outputs = model(inputs)
                outputs = outputs.squeeze(-1) if outputs.ndim > 1 else outputs
                loss = criterion(outputs, targets)

            # Track metrics
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Restore training state
    if was_training:
        model.train()

    # Compute metrics
    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    trading_metrics = compute_trading_metrics(
        predictions, targets_np,
        transaction_cost=transaction_cost,
        task='regression'
    )

    metrics = TrainingMetrics(
        loss=avg_loss,
        pnl=trading_metrics.get('pnl'),
        sharpe=trading_metrics.get('sharpe'),
        directional_accuracy=trading_metrics.get('directional_accuracy'),
    )

    if return_predictions:
        return metrics, predictions, targets_np
    return metrics


# =============================================================================
# Legacy compatibility
# =============================================================================


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    metrics_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    use_amp: bool = False,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Legacy evaluation function for backward compatibility.

    For new code, use evaluate_classification() or evaluate_regression().
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n = 0

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    batch_count = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
            out = model(xb).float()
            loss = loss_fn(out, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        preds.append(out.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())

        if debug and batch_count == 0:
            print(f"[DEBUG] First batch: out min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}")
            print(f"[DEBUG] Model params sum: {sum(p.sum().item() for p in model.parameters()):.6f}")
        batch_count += 1

    if debug:
        print(f"[DEBUG] Total batches processed: {batch_count}")

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    results = {"loss": float(total_loss / max(n, 1))}
    if metrics_fn is not None:
        results.update(metrics_fn(y_true, y_pred))

    # Restore training state
    if was_training:
        model.train()

    return results


# Convenience exports
__all__ = [
    'TrainingMetrics',
    'compute_trading_metrics',
    'train_classification_epoch',
    'evaluate_classification',
    'train_regression_epoch',
    'evaluate_regression',
    'evaluate',  # Legacy
]
