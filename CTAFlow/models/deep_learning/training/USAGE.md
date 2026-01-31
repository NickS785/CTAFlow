# Training Loops Usage Guide

This guide demonstrates how to use the CTAFlow training loops for classification and regression tasks with trading-aware losses.

## Quick Start

### Classification with ProfitWeightedCE

```python
import torch
from torch.utils.data import DataLoader
from CTAFlow.models.deep_learning.training import (
    train_classification_epoch,
    evaluate_classification,
    ProfitWeightedCE,
)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Trading-aware loss
criterion = ProfitWeightedCE(
    profit_scale=50.0,      # Scale profit weighting
    min_weight=0.1,         # Minimum sample weight
    max_weight=10.0,        # Maximum sample weight
    direction_penalty=2.0,  # Extra penalty for wrong direction
)

# Training loop
for epoch in range(num_epochs):
    # Train
    train_metrics = train_classification_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        transaction_cost=0.001,  # 10 bps
        gradient_clip=1.0,
    )

    # Evaluate
    val_metrics = evaluate_classification(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        transaction_cost=0.001,
    )

    # Print metrics
    print(f"Epoch {epoch+1}")
    print(f"  Train - Loss: {train_metrics.loss:.4f}, "
          f"Acc: {train_metrics.accuracy:.2f}%, "
          f"PnL: {train_metrics.pnl:.6f}, "
          f"Sharpe: {train_metrics.sharpe:.2f}")
    print(f"  Val   - Loss: {val_metrics.loss:.4f}, "
          f"Acc: {val_metrics.accuracy:.2f}%, "
          f"PnL: {val_metrics.pnl:.6f}, "
          f"Sharpe: {val_metrics.sharpe:.2f}")
```

### Regression with DirectionalMSE

```python
from CTAFlow.models.deep_learning.training import (
    train_regression_epoch,
    evaluate_regression,
)
from CTAFlow.models.deep_learning.training.loss.regression import DirectionalMSE

# Directional MSE loss
criterion = DirectionalMSE(
    direction_penalty=2.0,      # Penalty for wrong sign
    magnitude_weighted=True,    # Weight by target magnitude
)

# Training loop
for epoch in range(num_epochs):
    train_metrics = train_regression_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        transaction_cost=0.001,
    )

    val_metrics = evaluate_regression(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Epoch {epoch+1}")
    print(f"  Train - Loss: {train_metrics.loss:.4f}, "
          f"Dir Acc: {train_metrics.directional_accuracy:.2f}%, "
          f"PnL: {train_metrics.pnl:.6f}")
```

## DataLoader Requirements

### Classification with Returns

For trading-aware losses (ProfitWeightedCE, ExpectedPnLLoss), your DataLoader should return:
- `(inputs, targets, returns)` - returns are actual price returns for PnL weighting

```python
# Example dataset
class TradingDataset(Dataset):
    def __init__(self, features, labels, returns):
        self.features = features
        self.labels = labels
        self.returns = returns

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.returns[idx]

    def __len__(self):
        return len(self.features)

# Create loader
dataset = TradingDataset(X_train, y_train, returns_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Standard Classification

For standard losses (CrossEntropy, OrdinalCE), returns are optional:
- `(inputs, targets)` - standard format

### Multi-Modal Models

The training loops support multi-modal models (WSPR, TriModal, etc.):

```python
# Multi-modal batch format (example for WSPR)
# Returns: (summary_days, profile_days, raster, seq, seq_lens, targets, returns)
batch = next(iter(loader))
summary, profile, raster, seq, lens, targets, returns = batch

# The loop automatically unpacks and handles this
train_metrics = train_classification_epoch(
    model=model,
    loader=loader,  # Multi-modal loader
    criterion=criterion,
    optimizer=optimizer,
    device=device,
)
```

## Available Metrics

### TrainingMetrics Object

All training functions return a `TrainingMetrics` object with:

**Classification:**
- `loss`: Average loss
- `accuracy`: Overall accuracy (%)
- `directional_accuracy`: Accuracy excluding flat predictions (%)
- `pnl`: Average PnL per sample
- `sharpe`: Annualized Sharpe ratio
- `max_drawdown`: Maximum drawdown
- `win_rate`: Percentage of winning trades
- `avg_win`: Average profit per winning trade
- `avg_loss`: Average loss per losing trade

**Regression:**
- `loss`: Average loss
- `directional_accuracy`: Sign prediction accuracy (%)
- `pnl`: Average PnL per sample
- `sharpe`: Annualized Sharpe ratio

Access metrics:
```python
metrics = train_classification_epoch(...)
print(f"Accuracy: {metrics.accuracy:.2f}%")
print(f"Sharpe: {metrics.sharpe:.2f}")

# Convert to dictionary
metrics_dict = metrics.to_dict()
```

## Loss Functions Reference

### Classification Losses

**ProfitWeightedCE** - Weights samples by potential profit
```python
from CTAFlow.models.deep_learning.training import ProfitWeightedCE

criterion = ProfitWeightedCE(
    profit_scale=50.0,       # Higher = more weight on big moves
    min_weight=0.1,          # Minimum sample weight
    max_weight=10.0,         # Maximum sample weight
    direction_penalty=2.0,   # Extra penalty for wrong sign
)
# Requires returns: loss = criterion(logits, targets, returns=returns)
```

**ExpectedPnLLoss** - Directly optimizes expected PnL
```python
from CTAFlow.models.deep_learning.training import ExpectedPnLLoss

criterion = ExpectedPnLLoss(
    ce_weight=0.1,           # CrossEntropy regularization weight
    transaction_cost=0.001,  # Deduct transaction costs from PnL
    temperature=1.0,         # Probability sharpening
)
# Requires returns: loss = criterion(logits, targets, returns=returns)
```

**HierarchicalDirectionalLoss** - Two-stage: direction then full classification
```python
from CTAFlow.models.deep_learning.training import HierarchicalDirectionalLoss

criterion = HierarchicalDirectionalLoss(
    direction_weight=0.5,  # Weight for binary direction loss
    full_weight=0.5,       # Weight for 3-class loss
)
```

**CostAwareCE** - Incorporates trading costs
```python
from CTAFlow.models.deep_learning.training import CostAwareCE

criterion = CostAwareCE(
    transaction_cost=0.001,  # Cost per trade
    opportunity_cost=0.5,    # Cost of missing a trade
    direction_cost=2.0,      # Extra cost for wrong direction
)
```

**OrdinalCEWithAntiCollapse** - Distance-weighted with anti-collapse
```python
from CTAFlow.models.deep_learning.training import OrdinalCEWithAntiCollapse

criterion = OrdinalCEWithAntiCollapse(
    alpha=1.0,            # Distance penalty strength
    reg_lambda=0.05,      # Anti-collapse regularization
    target_probs=None,    # Target distribution (None = uniform)
)
```

### Regression Losses

**DirectionalMSE** - MSE with wrong-sign penalty
```python
from CTAFlow.models.deep_learning.training.loss.regression import DirectionalMSE

criterion = DirectionalMSE(
    direction_penalty=2.0,      # Penalty for wrong sign
    magnitude_weighted=True,    # Weight by target magnitude
)
```

**DirectionalMAE** - MAE with wrong-sign penalty
```python
from CTAFlow.models.deep_learning.training.loss.regression import DirectionalMAE

criterion = DirectionalMAE(
    direction_penalty=2.0,
    smooth=0.0,  # If > 0, uses smooth L1
)
```

**DirectionalHuber** - Huber loss with directional penalty
```python
from CTAFlow.models.deep_learning.training.loss.regression import DirectionalHuber

criterion = DirectionalHuber(
    delta=1.0,                  # Threshold between quadratic/linear
    direction_penalty=2.0,
)
```

## Advanced Usage

### Getting Predictions

```python
# Return predictions along with metrics
metrics, preds, targets, returns = evaluate_classification(
    model=model,
    loader=val_loader,
    criterion=criterion,
    device=device,
    return_predictions=True,  # Enable prediction return
)

# Analyze predictions
import numpy as np
class_preds = preds.argmax(axis=1)
conf_matrix = confusion_matrix(targets, class_preds)
```

### Mixed Precision Training

```python
# Enable AMP for faster training
train_metrics = train_classification_epoch(
    model=model,
    loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    use_amp=True,  # Enable automatic mixed precision
)
```

### Custom Transaction Costs

```python
# Different transaction costs per asset
transaction_cost = 0.001  # 10 bps for equities
# transaction_cost = 0.0005  # 5 bps for futures

train_metrics = train_classification_epoch(
    model=model,
    loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    transaction_cost=transaction_cost,
)
```

### Gradient Clipping

```python
# Adjust gradient clipping
train_metrics = train_classification_epoch(
    model=model,
    loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    gradient_clip=0.5,  # Lower for stability
    # gradient_clip=None,  # Disable clipping
)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CTAFlow.models.deep_learning.training import (
    train_classification_epoch,
    evaluate_classification,
    ProfitWeightedCE,
    TrainingMetrics,
)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    device: str = "cuda",
):
    # Setup
    criterion = ProfitWeightedCE(profit_scale=50.0, direction_penalty=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_pnl': [], 'val_pnl': [],
        'train_sharpe': [], 'val_sharpe': [],
    }

    best_sharpe = -float('inf')
    best_state = None

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_classification_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            transaction_cost=0.001,
            use_amp=True,
        )

        # Validate
        val_metrics = evaluate_classification(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            transaction_cost=0.001,
            use_amp=True,
        )

        scheduler.step()

        # Save history
        history['train_loss'].append(train_metrics.loss)
        history['val_loss'].append(val_metrics.loss)
        history['train_acc'].append(train_metrics.accuracy)
        history['val_acc'].append(val_metrics.accuracy)
        history['train_pnl'].append(train_metrics.pnl)
        history['val_pnl'].append(val_metrics.pnl)
        history['train_sharpe'].append(train_metrics.sharpe)
        history['val_sharpe'].append(val_metrics.sharpe)

        # Track best model
        if val_metrics.sharpe and val_metrics.sharpe > best_sharpe:
            best_sharpe = val_metrics.sharpe
            best_state = model.state_dict().copy()

        # Log
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={train_metrics.loss:.4f}, "
              f"Acc={train_metrics.accuracy:.2f}%, "
              f"PnL={train_metrics.pnl:.6f}, "
              f"Sharpe={train_metrics.sharpe:.2f}")
        print(f"  Val:   Loss={val_metrics.loss:.4f}, "
              f"Acc={val_metrics.accuracy:.2f}%, "
              f"PnL={val_metrics.pnl:.6f}, "
              f"Sharpe={val_metrics.sharpe:.2f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    return model, history

# Usage
model = YourModel()
trained_model, history = train_model(model, train_loader, val_loader)
```

## See Also

- [Loss Functions Documentation](loss/README.md)
- [Model Architecture Guide](../multi_branch/README.md)
- [Multi-Asset Training](../../multi_asset.py)
