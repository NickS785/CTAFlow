import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import ProfileEncoder, SeqEncoder, SummaryEncoder, GatedFusion


class TriModalLiquidityModel(nn.Module):
    """
    Tri-modal model combining summary, sequential, and spatial (profile) data.

    Supports interchangeable encoders and configurable fusion/head modes.

    Parameters
    ----------
    f_sum : int
        Summary feature dimension (ignored if sum_encoder provided)
    f_seq : int
        Sequential feature dimension (ignored if seq_encoder provided)
    d : int, default 128
        Encoder output dimension
    out_dim : int, default 1
        Output dimension (1 for regression, num_classes for classification)
    dropout : float, default 0.1
        Dropout rate
    task : str, default 'regression'
        Task type: 'regression' or 'classification'
    num_classes : int, default 3
        Number of classes (only used if task='classification' and out_dim not set)
    profile_encoder : nn.Module, optional
        Custom profile encoder. Must have `out_dim` attribute.
    seq_encoder : nn.Module, optional
        Custom sequential encoder. Must accept (x, lengths) and have `out_dim`.
    sum_encoder : nn.Module, optional
        Custom summary encoder. Must have `out_dim` attribute.
    fusion_mode : str, default 'gated'
        Fusion strategy: 'gated' (learned weights), 'concat' (concatenation), 'mean' (average)
    head_mode : str, default 'default'
        Head architecture: 'default' (LayerNorm+GELU), 'classification' (ReLU, no LayerNorm),
        'simple' (single linear layer)
    """

    def __init__(
        self,
        f_sum: int,
        f_seq: int,
        d: int = 128,
        out_dim: int = 1,
        dropout: float = 0.1,
        task: str = 'regression',
        num_classes: int = 3,
        profile_encoder=None,
        seq_encoder=None,
        sum_encoder=None,
        fusion_mode: str = 'gated',
        head_mode: str = 'default',
    ):
        super().__init__()

        self.task = task
        self.fusion_mode = fusion_mode
        self.head_mode = head_mode

        # Determine output dimension
        if task == 'classification' and out_dim == 1:
            out_dim = num_classes
        self.out_dim = out_dim
        self.num_classes = num_classes if task == 'classification' else None

        # --- ENCODERS ---
        if profile_encoder is not None:
            self.profile_enc = profile_encoder
            profile_d = profile_encoder.out_dim
        else:
            self.profile_enc = ProfileEncoder(in_ch=3, d_out=d, dropout=dropout)
            profile_d = d

        if seq_encoder is not None:
            self.seq_enc = seq_encoder
            seq_d = seq_encoder.out_dim
        else:
            self.seq_enc = SeqEncoder(f_in=f_seq, d_out=d, dropout=dropout)
            seq_d = d

        if sum_encoder is not None:
            self.sum_enc = sum_encoder
            sum_d = sum_encoder.out_dim
        else:
            self.sum_enc = SummaryEncoder(f_in=f_sum, d_out=d, dropout=dropout)
            sum_d = d

        # --- FUSION ---
        if fusion_mode == 'gated':
            # All encoders must output same dimension for gated fusion
            assert profile_d == seq_d == sum_d, \
                f"Gated fusion requires equal encoder dims: {profile_d}, {seq_d}, {sum_d}"
            self.fuse = GatedFusion(d=d, n_mod=3)
            fused_dim = d
        elif fusion_mode == 'concat':
            self.fuse = None
            fused_dim = profile_d + seq_d + sum_d
        elif fusion_mode == 'mean':
            assert profile_d == seq_d == sum_d, \
                f"Mean fusion requires equal encoder dims: {profile_d}, {seq_d}, {sum_d}"
            self.fuse = None
            fused_dim = d
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self._fused_dim = fused_dim

        # --- HEAD ---
        self.head = self._build_head(fused_dim, out_dim, dropout, head_mode, task)

    def _build_head(self, in_dim, out_dim, dropout, head_mode, task):
        """Build the output head based on mode and task."""
        if head_mode == 'simple':
            return nn.Linear(in_dim, out_dim)

        elif head_mode == 'classification' or (head_mode == 'default' and task == 'classification'):
            # Classification head: ReLU activations, no LayerNorm before logits
            return nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, out_dim)
            )

        else:  # 'default' regression head
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim)
            )

    def forward(self, summary_vec, seq_tensor, profile_tensor, seq_len, return_probs=False):
        """
        Forward pass.

        Parameters
        ----------
        summary_vec : torch.Tensor
            Summary features (B, F_sum)
        seq_tensor : torch.Tensor
            Sequential features (B, T, F_seq)
        profile_tensor : torch.Tensor
            Profile/spatial features (B, 3, 96)
        seq_len : torch.Tensor
            Sequence lengths (B,)
        return_probs : bool, default False
            For classification: return probabilities instead of logits

        Returns
        -------
        output : torch.Tensor
            Predictions (B, out_dim)
        gate_weights : torch.Tensor or None
            Gate weights if fusion_mode='gated', else None
        """
        # Encode each modality
        z_profile = self.profile_enc(profile_tensor)
        z_seq = self.seq_enc(seq_tensor, seq_len)
        z_sum = self.sum_enc(summary_vec)

        # Fuse
        gate_w = None
        if self.fusion_mode == 'gated':
            z, gate_w = self.fuse([z_profile, z_seq, z_sum])
        elif self.fusion_mode == 'concat':
            z = torch.cat([z_profile, z_seq, z_sum], dim=-1)
        elif self.fusion_mode == 'mean':
            z = (z_profile + z_seq + z_sum) / 3.0

        # Output
        output = self.head(z)

        if self.task == 'classification' and return_probs:
            output = F.softmax(output, dim=-1)

        return output, gate_w


class TriModalClassifier(nn.Module):
    """
    Classification-optimized tri-modal model.

    Uses concatenation fusion (preserves all information) and a deeper
    classification head with batch normalization.

    Parameters
    ----------
    f_sum : int
        Summary feature dimension
    f_seq : int
        Sequential feature dimension
    num_classes : int, default 3
        Number of output classes
    d : int, default 128
        Encoder output dimension
    dropout : float, default 0.2
        Dropout rate (higher default for classification)
    profile_encoder : nn.Module, optional
        Custom profile encoder
    seq_encoder : nn.Module, optional
        Custom sequential encoder
    sum_encoder : nn.Module, optional
        Custom summary encoder
    """

    def __init__(
        self,
        f_sum: int,
        f_seq: int,
        num_classes: int = 3,
        d: int = 128,
        dropout: float = 0.2,
        profile_encoder=None,
        seq_encoder=None,
        sum_encoder=None,
    ):
        super().__init__()

        self.num_classes = num_classes

        # --- ENCODERS ---
        if profile_encoder is not None:
            self.profile_enc = profile_encoder
            profile_d = profile_encoder.out_dim
        else:
            self.profile_enc = ProfileEncoder(in_ch=3, d_out=d, dropout=dropout)
            profile_d = d

        if seq_encoder is not None:
            self.seq_enc = seq_encoder
            seq_d = seq_encoder.out_dim
        else:
            self.seq_enc = SeqEncoder(f_in=f_seq, d_out=d, dropout=dropout)
            seq_d = d

        if sum_encoder is not None:
            self.sum_enc = sum_encoder
            sum_d = sum_encoder.out_dim
        else:
            self.sum_enc = SummaryEncoder(f_in=f_sum, d_out=d, dropout=dropout)
            sum_d = d

        # Store dimensions for external access
        self.profile_d = profile_d
        self.seq_d = seq_d
        self.sum_d = sum_d
        fused_dim = profile_d + seq_d + sum_d

        # --- CLASSIFICATION HEAD ---
        # Deeper head with BatchNorm for better gradient flow
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),

            nn.Linear(64, num_classes)
        )

        # Initialize final layer with smaller weights
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, summary_vec, seq_tensor, profile_tensor, seq_len, return_probs=False):
        """
        Forward pass.

        Returns
        -------
        output : torch.Tensor
            Class logits (B, num_classes) or probabilities if return_probs=True
        encoder_outputs : dict
            Dictionary with individual encoder outputs for analysis
        """
        # Encode each modality
        z_profile = self.profile_enc(profile_tensor)
        z_seq = self.seq_enc(seq_tensor, seq_len)
        z_sum = self.sum_enc(summary_vec)

        # Concatenate (preserves all class-discriminative information)
        z = torch.cat([z_profile, z_seq, z_sum], dim=-1)

        # Classification head
        logits = self.head(z)

        if return_probs:
            output = F.softmax(logits, dim=-1)
        else:
            output = logits

        # Return encoder outputs for analysis (e.g., which modality contributes most)
        encoder_outputs = {
            'profile': z_profile,
            'sequential': z_seq,
            'summary': z_sum,
        }

        return output, encoder_outputs

    def get_attention_weights(self, summary_vec, seq_tensor, profile_tensor, seq_len):
        """
        Compute importance of each modality using gradient-based attribution.

        Returns normalized importance scores for each modality.
        """
        self.eval()
        summary_vec.requires_grad_(True)
        seq_tensor.requires_grad_(True)
        profile_tensor.requires_grad_(True)

        z_profile = self.profile_enc(profile_tensor)
        z_seq = self.seq_enc(seq_tensor, seq_len)
        z_sum = self.sum_enc(summary_vec)

        z = torch.cat([z_profile, z_seq, z_sum], dim=-1)
        logits = self.head(z)

        # Get predicted class
        pred_class = logits.argmax(dim=-1)

        # Compute gradient w.r.t. predicted class
        importance = []
        for i, z_mod in enumerate([z_profile, z_seq, z_sum]):
            grad = torch.autograd.grad(
                logits.gather(1, pred_class.unsqueeze(1)).sum(),
                z_mod,
                retain_graph=True
            )[0]
            importance.append((grad * z_mod).abs().sum(dim=-1))

        importance = torch.stack(importance, dim=-1)  # (B, 3)
        importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)

        return importance  # (B, 3) - [profile, seq, summary] importance
