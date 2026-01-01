from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class GRUAttnRegressor(nn.Module):
    """GRU with attention for regression tasks."""

    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        h, _ = self.gru(x)                 # (B, L, H)
        w = self.attn(h).squeeze(-1)       # (B, L)
        w = torch.softmax(w, dim=-1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return self.head(pooled).squeeze(-1)


class GRUAttnClassifier(nn.Module):
    """GRU with attention for multiclass classification.

    Parameters
    ----------
    in_channels : int
        Number of input features per timestep
    num_classes : int
        Number of output classes
    hidden : int, default 64
        Hidden dimension of GRU
    layers : int, default 2
        Number of GRU layers
    dropout : float, default 0.15
        Dropout probability

    Example
    -------
    >>> model = GRUAttnClassifier(in_channels=10, num_classes=3)
    >>> x = torch.randn(32, 10, 50)  # (batch, features, seq_len)
    >>> logits = model(x)  # (32, 3)
    >>> probs = torch.softmax(logits, dim=-1)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
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
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        h, _ = self.gru(x)  # (B, L, H)

        # Attention pooling
        w = self.attn(h).squeeze(-1)  # (B, L)
        w = torch.softmax(w, dim=-1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, H)

        return self.head(pooled)  # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def train_gru_classifier(
    train_ds: Dataset,
    in_channels: int,
    num_classes: int,
    val_ds: Optional[Dataset] = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.15,
    patience: int = 5,
    model: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, List[dict]]:
    """Train a GRU classifier with early stopping.

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
    hidden : int, default 64
        GRU hidden dimension
    layers : int, default 2
        Number of GRU layers
    dropout : float, default 0.15
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
        model = GRUAttnClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden=hidden,
            layers=layers,
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
