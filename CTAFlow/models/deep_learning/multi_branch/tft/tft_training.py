"""
TFT-Aligned Training & Evaluation Utilities
============================================

Training loop, evaluation, and Optuna objective helpers for
TFTAlignedWSPR / TFTAlignedMamba models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from CTAFlow.data.datasets.tft import unpack_batch_for_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[Any] = None,
    clip_grad: float = 1.0,
    return_raw_returns: bool = False,
) -> Dict[str, float]:
    """Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        TFTAlignedWSPR or TFTAlignedMamba.
    loader : DataLoader
        Training DataLoader from TFTAlignedPrepLayer.
    optimizer : Optimizer
    criterion : loss function
        Should accept (logits, targets) or (logits, targets, returns=...).
    device : torch.device
    scheduler : optional LR scheduler (stepped per batch)
    clip_grad : float
        Max gradient norm.
    return_raw_returns : bool
        If True, pass 'raw_returns' from batch to criterion.

    Returns
    -------
    dict with 'loss', 'accuracy', 'n_samples'
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs, targets = unpack_batch_for_model(batch, device=device)

        optimizer.zero_grad()
        logits = model(**inputs)

        if return_raw_returns and "raw_returns" in batch:
            raw_ret = batch["raw_returns"].to(device)
            loss = criterion(logits, targets, returns=raw_ret)
        else:
            loss = criterion(logits, targets)

        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * targets.size(0)
        if logits.dim() == 2 and logits.size(1) > 1:
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
        total += targets.size(0)

    metrics = {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1) * 100.0,
        "n_samples": total,
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on a validation set.

    Returns dict with 'loss', 'accuracy', 'n_samples'.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs, targets = unpack_batch_for_model(batch, device=device)
        logits = model(**inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        if logits.dim() == 2 and logits.size(1) > 1:
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
        total += targets.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1) * 100.0,
        "n_samples": total,
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Collect predictions and targets from a DataLoader.

    Returns dict with 'predictions', 'targets', 'probabilities'.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in loader:
        inputs, targets = unpack_batch_for_model(batch, device=device)
        logits = model(**inputs)

        if logits.dim() == 2 and logits.size(1) > 1:
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        else:
            probs = logits
            preds = logits.squeeze(-1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return {
        "predictions": np.concatenate(all_preds),
        "targets": np.concatenate(all_targets),
        "probabilities": np.concatenate(all_probs),
    }


@torch.no_grad()
def get_tracker_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 5,
) -> Dict[str, Any]:
    """Collect interpretability stats from model tracker.

    Runs a few batches with return_tracker=True and averages
    branch weights, static var weights, etc.

    Returns dict with averaged tracker statistics.
    """
    model.eval()
    branch_weights_accum: Dict[str, List[float]] = {}
    static_weights_accum: Dict[str, List[float]] = {}
    gate_values: List[float] = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        inputs, _ = unpack_batch_for_model(batch, device=device)
        _, tracker = model(**inputs, return_tracker=True)

        if "branch_weights" in tracker and tracker["branch_weights"]:
            for name, val in tracker["branch_weights"].items():
                branch_weights_accum.setdefault(name, []).append(val)

        if "static_var_weights" in tracker and tracker["static_var_weights"]:
            for name, val in tracker["static_var_weights"].items():
                static_weights_accum.setdefault(name, []).append(val)

        if "cross_attn_gate" in tracker and tracker["cross_attn_gate"] is not None:
            gate_values.append(tracker["cross_attn_gate"])

    stats: Dict[str, Any] = {}
    if branch_weights_accum:
        stats["branch_weights"] = {
            k: np.mean(v) for k, v in branch_weights_accum.items()
        }
    if static_weights_accum:
        stats["static_var_weights"] = {
            k: np.mean(v) for k, v in static_weights_accum.items()
        }
    if gate_values:
        stats["cross_attn_gate_mean"] = np.mean(gate_values)

    return stats
