# Dual Branch Classification Guide

## Overview

The DualBranchModel has been enhanced to support **classification tasks with PDF outputs** using Cross Entropy loss. This guide shows how to use the new features for return classification.

## Key Features

1. **DualBranchModel Classification Mode** - Outputs class probabilities (PDF)
2. **_make_clf_targets()** - Converts continuous returns into 3 classes
3. **get_loaders()** - Creates PyTorch DataLoaders with DualDataset

---

## 1. DualBranchModel Updates

### Regression Mode (Default)
```python
from CTAFlow.models.deep_learning.multi_branch.dual_model import DualBranchModel

# Regression model (original behavior)
model = DualBranchModel(
    summary_input_dim=33,
    vpin_input_dim=3,
    lstm_hidden_dim=64,
    dense_hidden_dim=128,
    task='regression',  # Default
)

# Forward pass returns shape (batch, 1)
output = model(summary, vpin, lengths)
```

### Classification Mode (NEW)
```python
# Classification model with 3 classes
model = DualBranchModel(
    summary_input_dim=33,
    vpin_input_dim=3,
    lstm_hidden_dim=64,
    dense_hidden_dim=128,
    task='classification',  # NEW: Classification mode
    num_classes=3,          # NEW: Number of classes
)

# Forward pass with logits (for training with CrossEntropyLoss)
logits = model(summary, vpin, lengths, return_probs=False)  # Shape: (batch, 3)

# Forward pass with probabilities (for inference/PDF)
probs = model(summary, vpin, lengths, return_probs=True)   # Shape: (batch, 3)
# probs[i] = [P(class_0), P(class_1), P(class_2)]
```

---

## 2. Creating Classification Targets

### Using `_make_clf_targets()`

The `_make_clf_targets()` method divides continuous returns into 3 classes:

| Class | Description | Condition |
|-------|-------------|-----------|
| **0** | Negative returns (Down) | `returns < 0` |
| **1** | Small absolute returns (Neutral) | Low volatility / noise |
| **2** | Large positive returns (Strong Up) | `returns > 0` AND high magnitude |

### Default Usage
```python
from CTAFlow.models.intraday_momentum import DeepIDMomentum

# Load model with continuous returns
model = DeepIDMomentum.from_files(
    intraday_path='data/cl_intraday.csv',
    features_path='data/cl_features.csv',
    sequential_path='data/cl_vpin.parquet',
    target_col='return_1d'
)

# Convert to classification targets
model._make_clf_targets(
    upper_threshold=0.6,   # Top 40% of absolute returns → class 2 (if positive)
    lower_threshold=0.3,   # Not used in current implementation
    inplace=True
)

# Now model.target_data contains classes: 0, 1, 2
print(model.target_data.value_counts())
# 0    1234  (negative returns)
# 1     890  (small positive returns)
# 2     876  (large positive returns)
```

### Custom Thresholds
```python
# More aggressive: only classify top 20% as strong up-moves
model._make_clf_targets(upper_threshold=0.8, inplace=True)

# More conservative: classify top 50% as strong moves
model._make_clf_targets(upper_threshold=0.5, inplace=True)

# Without modifying original (returns new targets)
clf_targets = model._make_clf_targets(upper_threshold=0.7, inplace=False)
```

### Class Distribution Logic

```python
# Pseudocode for how classes are assigned:
abs_returns = returns.abs()
upper_q = abs_returns.quantile(0.6)  # e.g., 0.015 (1.5%)

# Class 0: All negative returns
classes[returns < 0] = 0

# Class 2: Large positive returns above threshold
classes[(returns > 0) & (abs_returns >= upper_q)] = 2

# Class 1: Small positive returns below threshold
classes[(returns >= 0) & (abs_returns < upper_q)] = 1
```

---

## 3. Creating DataLoaders with `get_loaders()`

The `get_loaders()` method wraps `get_xy()` and returns PyTorch DataLoaders using DualDataset.

### Basic Usage
```python
# Single DataLoader (all data)
loader = model.get_loaders(
    batch_size=32,
    max_seq_len=200,
    sequential_cols=['vpin', 'bucket_return', 'log_duration'],
)

# Iterate
for summary, sequential, target, lengths in loader:
    # summary: (batch, n_features)
    # sequential: (batch, max_seq_len, n_seq_features) - padded
    # target: (batch,) - class labels (0, 1, 2)
    # lengths: (batch,) - actual sequence lengths before padding
    pass
```

### Train/Val Split
```python
# Train/val split (80/20 by default)
train_loader, val_loader = model.get_loaders(
    val_split=True,
    val_split_size=0.2,
    batch_size=32,
    shuffle_train=True,
    max_seq_len=200,
    sequential_cols=['vpin', 'bucket_return', 'log_duration'],
)
```

### All Parameters
```python
loaders = model.get_loaders(
    # Data filtering
    start_date='2020-01-01',
    end_date='2023-12-31',
    dropna=True,
    remove_outliers=False,
    outlier_threshold=0.01,

    # Train/val split
    val_split=True,
    val_split_size=0.2,

    # DataLoader config
    batch_size=64,
    shuffle_train=True,
    num_workers=4,

    # Sequential data config
    max_seq_len=200,
    sequential_cols=['vpin', 'bucket_return', 'log_duration'],

    # Debug
    verbose=True,
)
```

---

## 4. Complete Classification Training Example

### Preparing Data
```python
from CTAFlow.models.intraday_momentum import DeepIDMomentum
from CTAFlow.models.deep_learning.multi_branch.dual_model import DualBranchModel
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Load data
model_prep = DeepIDMomentum.from_files(
    intraday_path='data/cl_intraday.csv',
    features_path='data/cl_features.csv',
    sequential_path='data/cl_vpin.parquet',
    target_col='return_1d'
)

# 2. Convert to classification targets
model_prep._make_clf_targets(upper_threshold=0.6, inplace=True)

# 3. Get DataLoaders
train_loader, val_loader = model_prep.get_loaders(
    val_split=True,
    val_split_size=0.2,
    batch_size=32,
    shuffle_train=True,
    max_seq_len=200,
    sequential_cols=['vpin', 'bucket_return', 'log_duration'],
    verbose=True
)
```

### Training Loop
```python
# 4. Initialize classification model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DualBranchModel(
    summary_input_dim=len(model_prep.feature_names),
    vpin_input_dim=3,
    lstm_hidden_dim=64,
    dense_hidden_dim=128,
    task='classification',
    num_classes=3,
).to(device)

# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 6. Training loop
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for summary, sequential, target, lengths in loader:
        summary = summary.to(device)
        sequential = sequential.to(device)
        target = target.to(device).long()  # CrossEntropyLoss expects Long
        lengths = lengths.to(device)

        optimizer.zero_grad()

        # Get logits (not probabilities) for CrossEntropyLoss
        logits = model(summary, sequential, lengths, return_probs=False)
        loss = criterion(logits, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for summary, sequential, target, lengths in loader:
            summary = summary.to(device)
            sequential = sequential.to(device)
            target = target.to(device).long()
            lengths = lengths.to(device)

            # Get logits for loss
            logits = model(summary, sequential, lengths, return_probs=False)
            loss = criterion(logits, target)
            total_loss += loss.item()

            # Get probabilities for analysis
            probs = model(summary, sequential, lengths, return_probs=True)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return total_loss / len(loader), correct / total, all_probs, all_targets


# 7. Train
EPOCHS = 50
best_val_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_probs, val_targets = validate(model, val_loader, criterion, device)

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_classification_model.pt')

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```

### Inference with PDF Output
```python
# Load best model
model.load_state_dict(torch.load('best_classification_model.pt'))
model.eval()

# Get probabilities (PDF)
with torch.no_grad():
    for summary, sequential, target, lengths in val_loader:
        summary = summary.to(device)
        sequential = sequential.to(device)
        lengths = lengths.to(device)

        # Get probability distribution
        probs = model(summary, sequential, lengths, return_probs=True)

        # probs[i] = [P(down), P(neutral), P(up)]
        for i in range(len(probs)):
            print(f"Sample {i}:")
            print(f"  P(Down)    = {probs[i, 0]:.3f}")
            print(f"  P(Neutral) = {probs[i, 1]:.3f}")
            print(f"  P(Up)      = {probs[i, 2]:.3f}")
            print(f"  Prediction = Class {torch.argmax(probs[i]).item()}")

        break  # Just show first batch
```

---

## 5. Evaluation Metrics

### Classification Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get all predictions
model.eval()
all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for summary, sequential, target, lengths in val_loader:
        summary = summary.to(device)
        sequential = sequential.to(device)
        lengths = lengths.to(device)

        probs = model(summary, sequential, lengths, return_probs=True)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
all_probs = np.concatenate(all_probs)

# Classification report
print(classification_report(
    all_targets,
    all_preds,
    target_names=['Down (0)', 'Neutral (1)', 'Up (2)']
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(all_targets, all_preds))

# Probability calibration check
print("\nMean predicted probabilities per class:")
for i in range(3):
    mask = all_targets == i
    if mask.sum() > 0:
        print(f"  Class {i}: {all_probs[mask, i].mean():.3f}")
```

---

## 6. Comparison: Regression vs Classification

### Regression Model
- **Output**: Continuous value (return)
- **Loss**: MSELoss
- **Use Case**: Precise return prediction
- **Target**: `model.target_data` (continuous)

```python
model_reg = DualBranchModel(task='regression', ...)
output = model_reg(summary, vpin, lengths)  # Shape: (batch, 1)
loss = nn.MSELoss()(output, target)
```

### Classification Model
- **Output**: Class probabilities (PDF)
- **Loss**: CrossEntropyLoss
- **Use Case**: Directional prediction, trading signals
- **Target**: `model._make_clf_targets()` (classes 0, 1, 2)

```python
model_clf = DualBranchModel(task='classification', num_classes=3, ...)
logits = model_clf(summary, vpin, lengths)  # Shape: (batch, 3)
loss = nn.CrossEntropyLoss()(logits, target)

# For inference:
probs = model_clf(summary, vpin, lengths, return_probs=True)
```

---

## 7. Tips and Best Practices

### Class Balance
```python
# Check class distribution before training
print(model_prep.target_data.value_counts(normalize=True))

# If severely imbalanced, use class weights
class_counts = model_prep.target_data.value_counts().sort_index()
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights.values, dtype=torch.float32).to(device)
)
```

### Threshold Tuning
```python
# Experiment with different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8]
for thresh in thresholds:
    model_prep._make_clf_targets(upper_threshold=thresh, inplace=False)
    print(f"Threshold {thresh}: {model_prep.target_data.value_counts()}")
```

### Early Stopping
```python
best_val_acc = 0
patience = 10
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc, _, _ = validate(...)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

---

## 8. Troubleshooting

### Issue: RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long
**Solution**: Convert targets to `.long()` for CrossEntropyLoss
```python
target = target.to(device).long()  # Not .float()
```

### Issue: Class imbalance leading to poor performance
**Solution**: Use class weights or focal loss
```python
from torch.nn import CrossEntropyLoss

class_weights = compute_class_weights(model_prep.target_data)
criterion = CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
```

### Issue: get_loaders() requires torch
**Solution**: Install PyTorch
```bash
pip install torch
```

---

## Summary

1. **DualBranchModel** now supports `task='classification'` with `num_classes` parameter
2. **_make_clf_targets()** converts continuous returns into 3 classes (0=down, 1=neutral, 2=up)
3. **get_loaders()** creates PyTorch DataLoaders with same API as `get_xy()`
4. Use **CrossEntropyLoss** for training classification models
5. Use `return_probs=True` in forward pass to get probability distributions (PDF)

This enables directional return prediction with confidence estimates!
