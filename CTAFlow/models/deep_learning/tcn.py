from typing import Optional, Tuple, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from ...data import MomentumWindowDataset
except ImportError:
    MomentumWindowDataset = None


class Chomp1d(nn.Module):
    """Remove trailing elements to maintain causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolutions and residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)


class TCNRegressor(nn.Module):
    """Temporal Convolutional Network for regression tasks."""

    def __init__(self, in_channels: int, channels=(32, 32, 32), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            layers.append(
                TemporalBlock(
                    c_in, c_out,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout
                )
            )
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):
        # x: (B, C, L)
        z = self.tcn(x)              # (B, hidden, L)
        last = z[:, :, -1]           # (B, hidden)  use last timestep
        out = self.head(last).squeeze(-1)  # (B,)
        return out


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network for multiclass classification.

    Parameters
    ----------
    in_channels : int
        Number of input features per timestep
    num_classes : int
        Number of output classes
    channels : tuple, default (32, 32, 32)
        Number of channels in each temporal block
    kernel_size : int, default 3
        Convolution kernel size
    dropout : float, default 0.1
        Dropout probability

    Example
    -------
    >>> model = TCNClassifier(in_channels=10, num_classes=3)
    >>> x = torch.randn(32, 10, 50)  # (batch, features, seq_len)
    >>> logits = model(x)  # (32, 3)
    >>> probs = torch.softmax(logits, dim=-1)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: Sequence[int] = (32, 32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            layers.append(
                TemporalBlock(
                    c_in, c_out,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            c_in = c_out

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, seq_len)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes)
        """
        z = self.tcn(x)  # (B, hidden, L)
        last = z[:, :, -1]  # (B, hidden) - use last timestep
        return self.head(last)  # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def train_simple_tcn(ds, in_channels: int, epochs: int = 30, model: nn.Module = None):
    """Train TCN regressor (legacy function for backward compatibility)."""
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    if model is None:
        model = TCNRegressor(in_channels=in_channels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"epoch {ep:02d} loss {tot/n:.6f}")

    return model


def train_tcn_classifier(
    train_ds: Dataset,
    in_channels: int,
    num_classes: int,
    val_ds: Optional[Dataset] = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
    channels: Sequence[int] = (32, 32, 32),
    kernel_size: int = 3,
    dropout: float = 0.1,
    patience: int = 5,
    model: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, List[dict]]:
    """Train a TCN classifier with early stopping.

    Parameters
    ----------
    train_ds : Dataset
        Training dataset yielding (x, y) pairs
    in_channels : int
        Number of input features
    num_classes : int
        Number of output classes
    val_ds : Dataset, optional
        Validation dataset for early stopping
    epochs : int, default 30
        Maximum training epochs
    batch_size : int, default 64
        Batch size
    lr : float, default 2e-3
        Learning rate
    weight_decay : float, default 1e-3
        L2 regularization
    class_weights : torch.Tensor, optional
        Weights for imbalanced classes, shape (num_classes,)
    channels : tuple, default (32, 32, 32)
        Channels per temporal block
    kernel_size : int, default 3
        Convolution kernel size
    dropout : float, default 0.1
        Dropout probability
    patience : int, default 5
        Early stopping patience (0 to disable)
    model : nn.Module, optional
        Pre-initialized model (if None, creates new one)
    verbose : bool, default True
        Print training progress

    Returns
    -------
    model : nn.Module
        Trained model
    history : List[dict]
        Training history with loss and accuracy per epoch
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = TCNClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=verbose
    )

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(dim=-1) == yb).sum().item()
            train_total += xb.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        val_loss = None
        val_acc = None
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).long()
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

                    val_loss_sum += loss.item() * xb.size(0)
                    val_correct += (logits.argmax(dim=-1) == yb).sum().item()
                    val_total += xb.size(0)

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if verbose:
            msg = f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} train_acc: {train_acc:.3f}"
            if val_loss is not None:
                msg += f" | val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}"
            print(msg)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history
