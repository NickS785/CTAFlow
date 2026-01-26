import torch
import torch.nn as nn
import torch.nn.functional as F

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