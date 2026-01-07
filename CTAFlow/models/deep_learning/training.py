from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ...data import make_window_dataset, MomentumWindowDataset
from ..intraday_momentum import IntradayMomentum


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 2e-3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    use_amp: bool = True
    device: Optional[str] = None  # "cuda", "cpu", or None to auto
    log_every: int = 1

    # Early stopping
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0  # improvement threshold

    # Scheduler
    scheduler: Optional[str] = "plateau"  # None | "step" | "plateau"
    step_size: int = 10
    gamma: float = 0.5
    plateau_factor: float = 0.5
    plateau_patience: int = 5

def convert_IM(model_prep:IntradayMomentum, lookback_period=10, batch_size=16, train_test_split=True, train_size=0.7, val_split=False, val_size=0.15):
    """
    Convert IntradayMomentum model to windowed datasets for deep learning.

    Parameters:
    -----------
    model_prep : IntradayMomentum
        Prepared model with features
    lookback_period : int
        Number of time steps to look back
    batch_size : int
        Batch size for DataLoader
    train_test_split : bool
        Whether to split into train/test or train/val/test
    train_size : float
        Proportion of data for training (used if val_split=False)
    val_split : bool
        Whether to create separate validation set
    val_size : float
        Proportion of data for validation (from train+val combined)

    Returns:
    --------
    If val_split=True: (train_ds, train_dl), (val_ds, val_dl)
    If train_test_split=True and val_split=False: (train_ds, train_dl), (test_ds, test_dl)
    Otherwise: ds, dl
    """
    X, y = model_prep.get_xy()
    if scaler is None:
        scaler = StandardScaler()

    if val_split:
        # Split into train and validation (no test set in this mode)
        train_len = int(len(X) * (1 - val_size))
        train_X, val_X = X[:train_len], X[train_len:]
        train_y, val_y = y[:train_len], y[train_len:]

        # Create train dataset and get normalization stats
        train_ds = MomentumWindowDataset(train_X, train_y, lookback=lookback_period, normalize=True)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # Use train stats for validation dataset
        norm_stats = train_ds.get_normalization_stats()
        val_ds = MomentumWindowDataset(val_X, val_y, lookback=lookback_period, normalize=True, fit_stats=norm_stats)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        return (train_ds, train_dl), (val_ds, val_dl)
    elif train_test_split:
        train_len = int(len(X) * train_size)
        train_X, test_X = X[:train_len], X[train_len:]
        train_y, test_y = y[:train_len], y[train_len:]

        # Create train dataset and get normalization stats
        train_ds = MomentumWindowDataset(train_X, train_y, lookback=lookback_period, normalize=True)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # Use train stats for test dataset
        norm_stats = train_ds.get_normalization_stats()
        test_ds = MomentumWindowDataset(test_X, test_y, lookback=lookback_period, normalize=True, fit_stats=norm_stats)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        return (train_ds, train_dl), (test_ds, test_dl)
    else:
        ds, dl = make_window_dataset(X, y, lookback=lookback_period, batch_size=batch_size)
        return ds, dl

def default_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    # correlation can be NaN if constant
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if (np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12) else float("nan")
    # direction accuracy (sign)
    dir_acc = float(np.mean((np.sign(y_true) == np.sign(y_pred)).astype(np.float32)))
    return {"mse": mse, "mae": mae, "corr": corr, "dir_acc": dir_acc}


def default_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (integers)
    y_pred : np.ndarray
        Predicted class labels (integers) or logits/probabilities

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy and per-class metrics
    """
    # Handle case where y_pred might be logits (2D array)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    accuracy = float(np.mean(y_true == y_pred))

    # Per-class accuracy
    classes = np.unique(y_true)
    per_class = {}
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            per_class[f"acc_class_{c}"] = float(np.mean(y_pred[mask] == c))

    return {"accuracy": accuracy, **per_class}


def compute_class_weights(
    labels: np.ndarray,
    method: str = "balanced",
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer class labels
    method : str, default "balanced"
        Weighting method:
        - "balanced": Inverse frequency weighting (sklearn-style)
        - "effective": Effective number of samples (for long-tail distributions)
        - "sqrt": Square root of inverse frequency (softer than balanced)
    smoothing : float, default 0.0
        Label smoothing factor (0 to 1). Higher values make weights more uniform.

    Returns
    -------
    torch.Tensor
        Class weights tensor of shape (num_classes,)

    Example
    -------
    >>> labels = np.array([0, 0, 0, 1, 1, 2])  # Imbalanced
    >>> weights = compute_class_weights(labels)
    >>> loss_fn = nn.CrossEntropyLoss(weight=weights)
    """
    labels = np.asarray(labels).astype(int)
    classes = np.unique(labels)
    num_classes = len(classes)
    n_samples = len(labels)

    # Count samples per class
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    if method == "balanced":
        # sklearn-style balanced weights: n_samples / (n_classes * n_samples_per_class)
        weights = n_samples / (num_classes * counts)
    elif method == "effective":
        # Effective number of samples (for long-tail)
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    elif method == "sqrt":
        # Square root of inverse frequency (softer weighting)
        weights = np.sqrt(n_samples / (num_classes * counts))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced', 'effective', or 'sqrt'")

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    # Apply smoothing (interpolate towards uniform)
    if smoothing > 0:
        uniform = np.ones(num_classes)
        weights = (1 - smoothing) * weights + smoothing * uniform

    return torch.tensor(weights, dtype=torch.float32)


def create_classification_targets(
    returns: np.ndarray,
    thresholds: Tuple[float, ...] = (-0.001, 0.001),
) -> np.ndarray:
    """Convert continuous returns to classification targets.

    Parameters
    ----------
    returns : np.ndarray
        Continuous return values
    thresholds : tuple of float
        Boundaries for classification. E.g., (-0.001, 0.001) creates 3 classes:
        - Class 0: returns < -0.001 (down)
        - Class 1: -0.001 <= returns <= 0.001 (flat)
        - Class 2: returns > 0.001 (up)

    Returns
    -------
    np.ndarray
        Integer class labels

    Example
    -------
    >>> returns = np.array([-0.02, 0.0, 0.015, -0.005])
    >>> labels = create_classification_targets(returns, thresholds=(-0.01, 0.01))
    >>> # labels = [0, 1, 2, 0]  # down, flat, up, down
    """
    returns = np.asarray(returns)
    thresholds = sorted(thresholds)

    labels = np.zeros(len(returns), dtype=np.int64)
    for i, threshold in enumerate(thresholds):
        labels[returns > threshold] = i + 1

    return labels



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    metrics_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = default_regression_metrics,
    use_amp: bool = False,
    debug: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n = 0

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    batch_count = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)  # xb: (B, C, L)
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


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: TrainConfig,
    metrics_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = default_regression_metrics,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[torch.nn.Module, Dict[str, List[Dict[str, float]]]]:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.startswith("cuda")))

    # Scheduler
    scheduler = None
    if cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience
        )

    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}

    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.startswith("cuda"))):
                out = model(xb).float()
                loss = loss_fn(out, yb)

            scaler.scale(loss).backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n += bs

        train_loss = float(total_loss / max(n, 1))
        train_metrics = {"loss": train_loss}
        history["train"].append(train_metrics)

        # Validation
        if val_loader is not None:
            # Enable debug for first 3 epochs to diagnose frozen metrics
            debug_mode = epoch <= 3
            val_metrics = evaluate(
                model, val_loader, loss_fn=loss_fn, device=device, metrics_fn=metrics_fn, use_amp=cfg.use_amp, debug=debug_mode
            )
            history["val"].append(val_metrics)

            # Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            elif scheduler is not None:
                scheduler.step()

            # Early stopping (minimize val loss)
            improved = (best_val - val_metrics["loss"]) > cfg.early_stop_min_delta
            if improved:
                best_val = val_metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if (epoch % cfg.log_every) == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch:03d} | lr {lr:.2e} | train {train_loss:.6f} | val {val_metrics}")

            if bad_epochs >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val loss={best_val:.6f})")
                break
        else:
            # no val loader
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            if (epoch % cfg.log_every) == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch:03d} | lr {lr:.2e} | train {train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
