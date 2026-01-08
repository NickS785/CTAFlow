"""
Dual Branch Classification - Quick Start Example

This script demonstrates the complete workflow for training a classification
model using DualBranchModel with DeepIDMomentum.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from CTAFlow.models.intraday_momentum import DeepIDMomentum
from CTAFlow.models.deep_learning.multi_branch.dual_model import DualBranchModel

# ============================================================================
# 1. LOAD DATA AND CREATE CLASSIFICATION TARGETS
# ============================================================================

print("Loading data...")
model_prep = DeepIDMomentum.from_files(
    intraday_path='data/cl_intraday.csv',
    features_path='data/cl_features.csv',
    sequential_path='data/cl_vpin.parquet',
    target_col='return_1d'
)

print("Converting to classification targets...")
model_prep._make_clf_targets(
    upper_threshold=0.6,  # Top 40% of absolute returns → class 2
    inplace=True
)

print("Class distribution:")
print(model_prep.target_data.value_counts())

# ============================================================================
# 2. CREATE DATALOADERS
# ============================================================================

print("\nCreating DataLoaders...")
train_loader, val_loader = model_prep.get_loaders(
    val_split=True,
    val_split_size=0.2,
    batch_size=32,
    shuffle_train=True,
    max_seq_len=200,
    sequential_cols=['vpin', 'bucket_return', 'log_duration'],
    verbose=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ============================================================================
# 3. INITIALIZE MODEL
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = DualBranchModel(
    summary_input_dim=len(model_prep.feature_names),
    vpin_input_dim=3,
    lstm_hidden_dim=64,
    dense_hidden_dim=128,
    task='classification',  # Classification mode
    num_classes=3,          # 3 classes: down, neutral, up
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 4. TRAINING SETUP
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for summary, sequential, target, lengths in loader:
        summary = summary.to(device)
        sequential = sequential.to(device)
        target = target.to(device).long()
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
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

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

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return total_loss / len(loader), correct / total


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

print("\nStarting training...")
EPOCHS = 50
best_val_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    scheduler.step(val_loss)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_classification_model.pt')

    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}")

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

# ============================================================================
# 6. INFERENCE WITH PDF OUTPUT
# ============================================================================

print("\nRunning inference with PDF output...")
model.load_state_dict(torch.load('best_classification_model.pt'))
model.eval()

# Get probability distributions for first batch
with torch.no_grad():
    for summary, sequential, target, lengths in val_loader:
        summary = summary.to(device)
        sequential = sequential.to(device)
        lengths = lengths.to(device)

        # Get probability distribution (PDF)
        probs = model(summary, sequential, lengths, return_probs=True)

        # Show first 5 samples
        print("\nSample predictions (first 5):")
        for i in range(min(5, len(probs))):
            print(f"\nSample {i}:")
            print(f"  P(Down)    = {probs[i, 0]:.3f}")
            print(f"  P(Neutral) = {probs[i, 1]:.3f}")
            print(f"  P(Up)      = {probs[i, 2]:.3f}")
            print(f"  Predicted class: {torch.argmax(probs[i]).item()}")
            print(f"  Actual class: {target[i].item()}")

        break  # Only show first batch

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

print("\nCalculating evaluation metrics...")
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

all_preds = []
all_targets = []

with torch.no_grad():
    for summary, sequential, target, lengths in val_loader:
        summary = summary.to(device)
        sequential = sequential.to(device)
        lengths = lengths.to(device)

        probs = model(summary, sequential, lengths, return_probs=True)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# Classification report
print("\nClassification Report:")
print(classification_report(
    all_targets,
    all_preds,
    target_names=['Down (0)', 'Neutral (1)', 'Up (2)']
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(all_targets, all_preds))

print("\nDone!")
