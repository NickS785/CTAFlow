from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
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

def convert_IM(model_prep:IntradayMomentum, lookback_period=20, batch_size=5, val_split=True, val_size=0.2):

    X, y = model_prep.get_xy()

    if val_split:
        train_len = int(len(X) * (1.0 - val_size))
        train_X, val_X = X[:train_len], X[train_len:]
        train_y, val_y = y[:train_len], y[train_len:]
        (train_ds, train_dl) = make_window_dataset(train_X, train_y, lookback=lookback_period, batch_size=batch_size)
        (val_ds, val_dl) = make_window_dataset(val_X, val_y, lookback=lookback_period, batch_size=batch_size)
        return (train_ds, train_dl), (val_ds, val_dl)
    else:
        ds, dl = make_window_dataset(X, y, lookback=lookback_period,batch_size=batch_size)

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



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    metrics_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = default_regression_metrics,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

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

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    results = {"loss": float(total_loss / max(n, 1))}
    if metrics_fn is not None:
        results.update(metrics_fn(y_true, y_pred))
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
            val_metrics = evaluate(
                model, val_loader, loss_fn=loss_fn, device=device, metrics_fn=metrics_fn, use_amp=cfg.use_amp
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
