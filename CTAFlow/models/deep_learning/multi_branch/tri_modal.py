import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


from ..encoders import (
    GatedFusion,
    NumberBarsEncoder,
    ProfileEncoder,
    SeqEncoder,
    SpatialFuse,
    SpatialTemporalEncoder,
    SummaryEncoder, MarketProfileCNN,
)


class TriModalModel(nn.Module):
    """
    Tri-modal model combining summary, sequential, and spatial (profile + number bars) data.

    Supports two modes for the spatial/number bars encoder:
    - 'numberbars': Uses NumberBarsEncoder for pre-extracted number bars data (T, BINS, C)
    - 'rasterized': Uses SpatialTemporalEncoder for rasterized VPIN data (T, C, BINS)

    The rasterized mode is designed to work with SequenceRasterizer from CTAFlow.features.volume.vpin,
    which converts sequential VPIN buckets into a spatial grid representation.

    Parameters
    ----------
    f_sum : int
        Number of summary features
    f_seq : int
        Number of sequential features (default: 3 for VPIN, Return, Duration)
    f_spatial : int
        Number of profile channels (default: 4)
    f_nb : int
        Number of number bars / rasterized channels (default: 4)
    d_model : int
        Hidden dimension size for summary/spatial branches (default: 64)
    lstm_hidden_dim : int
        LSTM hidden dimension for sequential branch (default: 64)
    lstm_num_layers : int
        Number of LSTM layers (default: 1)
    transformer_d_model : int
        Transformer output dimension for rasterized encoder (default: 32)
    transformer_n_layers : int
        Number of transformer encoder layers in rasterized encoder (default: 2)
    transformer_nhead : int
        Number of attention heads in transformer (default: 4)
    summary_dropout : float
        Dropout for summary branch (default: 0.1)
    head_dropout : float
        Dropout for fusion head (default: 0.3)
    nb_dropout : float
        Dropout for number bars encoder (default: 0.2)
    task : str
        'regression' or 'classification' (default: 'regression')
    num_classes : int
        Number of classes for classification (default: 3)
    spatial_fuse_mode : str
        Mode for fusing profile and number bars ('gated' or 'mean')
    spatial_encoder_type : str
        Type of spatial encoder for number bars:
        - 'numberbars': NumberBarsEncoder for (B, T, BINS, C) data
        - 'rasterized': SpatialTemporalEncoder for (B, T, C, BINS) data from SequenceRasterizer
    num_bars : int
        Number of time bars for rasterized encoder (default: 4)
    num_bins : int
        Number of price bins for rasterized encoder (default: 64)
    """

    def __init__(
            self,
            f_sum: int,  # Number of summary features
            f_seq: int = 3,  # Number of sequential features (VPIN, Return, Dur)
            f_spatial: int = 4,  # Number of profile channels (Bid/Ask/Total)
            f_nb: int = 4,  # Number bars channels
            d_model: int = 64,  # Hidden dimension size for summary/spatial
            lstm_hidden_dim: int = 64,  # LSTM hidden dimension
            lstm_num_layers: int = 1,  # Number of LSTM layers
            transformer_d_model: int = 32,  # Transformer output dimension for rasterized encoder
            transformer_n_layers: int = 2,  # Number of transformer layers
            transformer_nhead: int = 4,  # Number of attention heads in transformer
            summary_dropout: float = 0.1,
            head_dropout: float = 0.3,
            nb_dropout: float = 0.2,
            task: str = 'regression',
            num_classes: int = 3,
            spatial_fuse_mode: str = "gated",
            spatial_encoder_type: str = "numberbars",
            num_bars: int = 4,
            num_bins: int = 64,
    ):
        super().__init__()
        self.task = task
        self.spatial_encoder_type = spatial_encoder_type

        # --- BRANCH 1: MACRO (Summary MLP) ---
        self.summary_net = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(summary_dropout),
            nn.Linear(d_model, d_model // 2)  # Compression
        )

        # --- BRANCH 2: MICRO (Sequential LSTM) ---
        self.lstm = nn.LSTM(
            input_size=f_seq,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # --- BRANCH 3: SPATIAL (Profile CNN) ---
        self.spatial_net = MarketProfileCNN(in_channels=f_spatial, out_dim=d_model // 2)

        # --- BRANCH 4: NUMBER BARS / RASTERIZED VPIN (Optional) ---
        if spatial_encoder_type == "rasterized":
            # Use SpatialTemporalEncoder for rasterized VPIN data
            # Input shape: (B, T, C, BINS) from SequenceRasterizer
            self.nb_net = SpatialTemporalEncoder(
                num_bars=num_bars,
                in_ch=f_nb,
                num_bins=num_bins,
                d_model=transformer_d_model,
                n_layers=transformer_n_layers,
                nhead=transformer_nhead,
                dropout=nb_dropout
            )
            nb_out_dim = transformer_d_model

            # Project nb output to match spatial dimension for fusion
            if transformer_d_model != d_model // 2:
                self.nb_proj = nn.Sequential(
                    nn.Linear(transformer_d_model, d_model // 2),
                    nn.GELU()
                )
            else:
                self.nb_proj = nn.Identity()
        else:
            # Default: NumberBarsEncoder for pre-extracted number bars
            # Input shape: (B, T, BINS, C)
            self.nb_net = NumberBarsEncoder(c_in=f_nb, d_model=d_model // 2, dropout=nb_dropout)
            nb_out_dim = d_model // 2
            self.nb_proj = nn.Identity()

        self.spatial_fuse = SpatialFuse(d_spatial=d_model // 2, mode=spatial_fuse_mode)

        # --- FUSION HEAD ---
        # Concatenate: Summary(d_model//2) + LSTM(lstm_hidden_dim) + Spatial(d_model//2)
        # Note: spatial_fuse outputs d_model//2 regardless of nb_out_dim
        fusion_dim = (d_model // 2) + lstm_hidden_dim + (d_model // 2)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes if task == 'classification' else 1)
        )

        # Store branch dimensions for importance tracking
        self.summary_dim = d_model // 2
        self.seq_dim = lstm_hidden_dim
        self.spatial_dim = d_model // 2

        # Storage for last forward pass intermediate representations
        self.last_branch_embeddings = None

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        summary_vec,
        seq_tensor,
        seq_lengths,
        profile_tensor,
        nb_tensor=None,
        nb_lengths=None,
        return_probs=False,
        return_importance=False,
    ):
        """
        Forward pass through the tri-modal model.

        Parameters
        ----------
        summary_vec : torch.Tensor
            Summary features (Batch, f_sum)
        seq_tensor : torch.Tensor
            Sequential features (Batch, Max_Len, f_seq)
        seq_lengths : torch.Tensor
            Sequence lengths (Batch,)
        profile_tensor : torch.Tensor
            Profile features (Batch, f_spatial, 128)
        nb_tensor : torch.Tensor, optional
            Number bars/rasterized tensor (Batch, T_nb, BINS, C_nb)
        nb_lengths : torch.Tensor, optional
            Lengths for number bars slices (Batch,)
        return_probs : bool, default False
            For classification: return probabilities instead of logits
        return_importance : bool, default False
            If True, return (output, importance_dict) tuple where importance_dict
            contains:
            - 'spatial_importance' (B, 2): [profile_weight, nb_weight]
            - 'branch_importance' (B, 3): [summary_weight, sequential_weight, spatial_weight]
            - 'branch_embeddings' dict: Raw embeddings from each branch

        Returns
        -------
        torch.Tensor or tuple
            If return_importance=False: model output (B, num_classes) or (B, 1)
            If return_importance=True: (output, importance_dict) tuple
        """
        # 1. Macro Branch
        z_sum = self.summary_net(summary_vec)

        # 2. Micro Branch (Packed Sequence)
        # Clamp lengths to at least 1 to avoid pack_padded_sequence errors
        seq_lengths_clamped = seq_lengths.clamp(min=1).cpu()
        packed_seq = rnn_utils.pack_padded_sequence(
            seq_tensor,
            seq_lengths_clamped,
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM returns (output, (h_n, c_n))
        # We only want the last hidden state (h_n)
        _, (h_n, _) = self.lstm(packed_seq)
        z_seq = h_n[-1]  # Shape: (Batch, lstm_hidden_dim)

        # 3. Spatial Branch
        z_profile = self.spatial_net(profile_tensor)
        z_nb = None
        if nb_tensor is not None:
            z_nb = self.nb_net(nb_tensor, nb_lengths)
            z_nb = self.nb_proj(z_nb)  # Project to match spatial dimension

        # Spatial fusion with optional importance tracking
        if return_importance:
            z_spatial, spatial_importance = self.spatial_fuse(z_profile, z_nb, return_importance=True)
        else:
            z_spatial = self.spatial_fuse(z_profile, z_nb, return_importance=False)

        # Store branch embeddings for analysis
        self.last_branch_embeddings = {
            'summary': z_sum.detach(),
            'sequential': z_seq.detach(),
            'spatial': z_spatial.detach()
        }

        # 4. Concatenation Fusion
        z_fused = torch.cat([z_sum, z_seq, z_spatial], dim=1)

        # 5. Output Head
        logits = self.head(z_fused)

        if self.task == 'classification' and return_probs:
            output = F.softmax(logits, dim=1)
        else:
            output = logits

        if return_importance:
            # Compute branch-level importance
            branch_importance = self._compute_branch_importance(
                z_sum, z_seq, z_spatial, logits
            )

            importance_dict = {
                'spatial_importance': spatial_importance,  # (B, 2): [profile_weight, nb_weight]
                'branch_importance': branch_importance,  # (B, 3): [summary, sequential, spatial]
                'branch_embeddings': {
                    'summary': z_sum.detach(),
                    'sequential': z_seq.detach(),
                    'spatial': z_spatial.detach()
                }
            }
            return output, importance_dict

        return output

    def _compute_branch_importance(self, z_sum, z_seq, z_spatial, logits):
        """
        Compute importance/contribution of each branch to the final prediction.

        Uses magnitude-based importance: L2 norm of each branch's embedding
        normalized across branches.

        Parameters
        ----------
        z_sum : torch.Tensor
            Summary branch embedding (B, summary_dim)
        z_seq : torch.Tensor
            Sequential branch embedding (B, seq_dim)
        z_spatial : torch.Tensor
            Spatial branch embedding (B, spatial_dim)
        logits : torch.Tensor
            Model output (B, num_classes) or (B, 1)

        Returns
        -------
        torch.Tensor
            Branch importance weights (B, 3) with values summing to 1
            [summary_importance, sequential_importance, spatial_importance]
        """
        # Compute L2 norm of each branch
        sum_magnitude = torch.norm(z_sum, p=2, dim=1, keepdim=True)  # (B, 1)
        seq_magnitude = torch.norm(z_seq, p=2, dim=1, keepdim=True)  # (B, 1)
        spatial_magnitude = torch.norm(z_spatial, p=2, dim=1, keepdim=True)  # (B, 1)

        # Stack and normalize to get importance weights
        magnitudes = torch.cat([sum_magnitude, seq_magnitude, spatial_magnitude], dim=1)  # (B, 3)
        importance = F.softmax(magnitudes, dim=1)  # Normalize to sum to 1

        return importance

    def get_spatial_importance_stats(self):
        """
        Get statistics about spatial modality importance from the last forward pass.

        Returns
        -------
        dict or None
            Statistics about profile vs number bars importance, or None if no
            forward pass has been made yet. Contains:
            - 'profile_mean': Mean importance of profile modality
            - 'nb_mean': Mean importance of number bars/rasterized modality
            - 'profile_std': Std deviation of profile importance
            - 'nb_std': Std deviation of nb importance
            - 'profile_dominant': Fraction of samples where profile > nb
        """
        return self.spatial_fuse.get_importance_stats()

    def get_branch_importance_stats(self):
        """
        Get statistics about branch-level importance from the last forward pass.

        Requires that the last forward pass was done with return_importance=True,
        or that the branch embeddings are stored.

        Returns
        -------
        dict or None
            Statistics about summary/sequential/spatial branch importance, or None
            if no forward pass with stored embeddings has been made yet. Contains:
            - 'summary_mean': Mean importance of summary branch
            - 'sequential_mean': Mean importance of sequential branch
            - 'spatial_mean': Mean importance of spatial branch
            - 'summary_std': Std deviation of summary importance
            - 'sequential_std': Std deviation of sequential importance
            - 'spatial_std': Std deviation of spatial importance
            - 'dominant_branch': Index of most frequently dominant branch (0=summary, 1=seq, 2=spatial)
        """
        if self.last_branch_embeddings is None:
            return None

        # Recompute importance from stored embeddings
        z_sum = self.last_branch_embeddings['summary']
        z_seq = self.last_branch_embeddings['sequential']
        z_spatial = self.last_branch_embeddings['spatial']

        # Dummy logits (not used in magnitude-based importance)
        dummy_logits = torch.zeros(z_sum.size(0), 1, device=z_sum.device)

        importance = self._compute_branch_importance(z_sum, z_seq, z_spatial, dummy_logits)

        summary_weights = importance[:, 0].cpu().numpy()
        seq_weights = importance[:, 1].cpu().numpy()
        spatial_weights = importance[:, 2].cpu().numpy()

        # Find most frequently dominant branch
        dominant_indices = importance.argmax(dim=1).cpu().numpy()
        dominant_branch = np.bincount(dominant_indices).argmax()

        return {
            'summary_mean': summary_weights.mean(),
            'sequential_mean': seq_weights.mean(),
            'spatial_mean': spatial_weights.mean(),
            'summary_std': summary_weights.std(),
            'sequential_std': seq_weights.std(),
            'spatial_std': spatial_weights.std(),
            'dominant_branch': int(dominant_branch),
            'dominant_branch_name': ['summary', 'sequential', 'spatial'][dominant_branch],
        }

    def compute_gradient_based_importance(self,
                                          summary_vec,
                                          seq_tensor,
                                          seq_lengths,
                                          profile_tensor,
                                          nb_tensor=None,
                                          nb_lengths=None,
                                          target_class=None):
        """
        Compute gradient-based importance for each branch.

        This uses integrated gradients-like approach: computes gradients of
        the output w.r.t. each branch's embedding to measure sensitivity.

        Parameters
        ----------
        summary_vec, seq_tensor, etc.
            Same as forward() method
        target_class : int, optional
            For classification: which class to compute importance for.
            If None, uses predicted class.

        Returns
        -------
        dict
            Contains gradient-based importance for each branch:
            - 'summary_grad_importance': (B,) importance scores
            - 'sequential_grad_importance': (B,) importance scores
            - 'spatial_grad_importance': (B,) importance scores
        """
        # Enable gradients for embeddings
        summary_vec.requires_grad_(True)
        seq_tensor.requires_grad_(True)
        profile_tensor.requires_grad_(True)
        if nb_tensor is not None:
            nb_tensor.requires_grad_(True)

        # Forward pass
        outputs, importance_dict = self.forward(
            summary_vec=summary_vec,
            seq_tensor=seq_tensor,
            seq_lengths=seq_lengths,
            profile_tensor=profile_tensor,
            nb_tensor=nb_tensor,
            nb_lengths=nb_lengths,
            return_importance=True
        )

        batch_size = outputs.size(0)

        # Get embeddings (with gradients enabled)
        z_sum = importance_dict['branch_embeddings']['summary'].requires_grad_(True)
        z_seq = importance_dict['branch_embeddings']['sequential'].requires_grad_(True)
        z_spatial = importance_dict['branch_embeddings']['spatial'].requires_grad_(True)

        # Determine target for gradient computation
        if target_class is None:
            if self.task == 'classification':
                target_class = outputs.argmax(dim=1)
            else:
                # For regression, use the output directly
                target_output = outputs.squeeze()
        else:
            target_output = outputs.gather(1, target_class.unsqueeze(1)).squeeze()

        # Compute gradients for each branch
        summary_grads = []
        seq_grads = []
        spatial_grads = []

        for i in range(batch_size):
            if self.task == 'classification':
                output_val = outputs[i, target_class[i]]
            else:
                output_val = outputs[i, 0]

            # Compute gradients
            grad_sum = torch.autograd.grad(output_val, z_sum, retain_graph=True, create_graph=False)[0][i]
            grad_seq = torch.autograd.grad(output_val, z_seq, retain_graph=True, create_graph=False)[0][i]
            grad_spatial = torch.autograd.grad(output_val, z_spatial, retain_graph=True, create_graph=False)[0][i]

            # Importance = gradient * embedding (approximation of contribution)
            summary_importance = (grad_sum * z_sum[i]).abs().sum()
            seq_importance = (grad_seq * z_seq[i]).abs().sum()
            spatial_importance = (grad_spatial * z_spatial[i]).abs().sum()

            summary_grads.append(summary_importance.item())
            seq_grads.append(seq_importance.item())
            spatial_grads.append(spatial_importance.item())

        # Normalize to sum to 1 per sample
        summary_grads = np.array(summary_grads)
        seq_grads = np.array(seq_grads)
        spatial_grads = np.array(spatial_grads)

        total = summary_grads + seq_grads + spatial_grads + 1e-8
        summary_grads = summary_grads / total
        seq_grads = seq_grads / total
        spatial_grads = spatial_grads / total

        return {
            'summary_grad_importance': summary_grads,
            'sequential_grad_importance': seq_grads,
            'spatial_grad_importance': spatial_grads,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from CTAFlow.models.deep_learning.encoders import (
    NumberBarsEncoder,
    SpatialTemporalEncoder,
    SpatialFuse,
)



class TriModalLSTM(nn.Module):
    """
    Windowed (day-sequence) tri-modal model:
      - Per-day encoders: summary, intraday-seq, profile, rasterized
      - Per-day spatial fuse: profile + rasterized -> spatial embedding
      - Per-day tri-modal fuse: [summary, intraday, spatial] -> day embedding
      - Day LSTM over window -> final prediction from last hidden state

    Expected batch tensors:
      summary_days : (B, D, F_sum)
      seq_days     : (B, D, T_seq, F_seq)      (right-padded within each day)
      seq_lens     : (B, D)                    intraday lengths per day
      profile_days : (B, D, C_prof, B_prof)
      raster_days  : (B, D, T_nb, C_nb, B_nb)  (SequenceRasterizer style: (T, C, Bins))
    """

    def __init__(
        self,
        f_sum: int,
        f_seq: int,
        f_profile: int = 3,
        profile_bins: int = 96,
        f_nb: int = 4,
        nb_bins: int = 64,
        num_bars: int = 4,
        d_model: int = 256,
        lstm_hidden_dim: int = 128,         # intraday LSTM hidden
        day_lstm_hidden: int = 256,         # day LSTM hidden
        day_lstm_layers: int = 1,
        task: str = "classification",
        num_classes: int = 3,
        spatial_fuse_mode: str = "gated",
        spatial_encoder_type: str = "rasterized",  # "rasterized" or "numberbars"
        transformer_d_model: int = 128,
        transformer_n_layers: int = 2,
        transformer_nhead: int = 4,
        dropout: float = 0.2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.task = task

        # -------- per-day summary encoder --------
        self.summary_net = nn.Sequential(
            nn.Linear(f_sum, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
        )

        # -------- per-day intraday sequential encoder (micro) --------
        self.seq_lstm = nn.LSTM(
            input_size=f_seq,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.seq_post = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.GELU(),
        )

        # -------- per-day profile encoder --------
        self.profile_net = MarketProfileCNN(in_channels=f_profile, out_dim=d_model // 2)

        # -------- per-day raster/nb encoder --------
        if spatial_encoder_type == "rasterized":
            # Input: (B, T, C, BINS) from SequenceRasterizer
            self.nb_net = SpatialTemporalEncoder(
                num_bars=num_bars,
                in_ch=f_nb,
                num_bins=nb_bins,
                d_model=transformer_d_model,
                n_layers=transformer_n_layers,
                nhead=transformer_nhead,
                dropout=dropout,
            )
            self.nb_proj = (
                nn.Linear(transformer_d_model, d_model // 2)
                if transformer_d_model != d_model // 2
                else nn.Identity()
            )
        else:
            # Input: (B, T, BINS, C)
            self.nb_net = NumberBarsEncoder(c_in=f_nb, d_model=d_model // 2, dropout=dropout)
            self.nb_proj = nn.Identity()

        # fuse profile + nb into a single spatial embedding
        self.spatial_fuse = SpatialFuse(d_spatial=d_model // 2, mode=spatial_fuse_mode)

        # -------- per-day tri-modal fusion -> day embedding --------
        per_day_dim = (d_model // 2) + lstm_hidden_dim + (d_model // 2)
        self.day_fuse = nn.Sequential(
            nn.LayerNorm(per_day_dim),
            nn.Linear(per_day_dim, per_day_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -------- day LSTM over window --------
        self.day_lstm = nn.LSTM(
            input_size=per_day_dim,
            hidden_size=day_lstm_hidden,
            num_layers=day_lstm_layers,
            batch_first=True,
        )

        out_dim = num_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(day_lstm_hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # keep consistent with your other models
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        summary_days: torch.Tensor,
        seq_days: torch.Tensor,
        seq_lens: torch.Tensor,
        profile_days: torch.Tensor,
        raster_days: torch.Tensor,
        return_probs: bool = False,
    ):
        """
        Returns:
          logits/probs: (B, num_classes) or (B, 1)
        """
        B, D, _, _ = seq_days.shape
        BD = B * D

        # ---- flatten day dimension so we can reuse your per-day encoders ----
        sum_flat = summary_days.reshape(BD, -1)                     # (BD, F_sum)
        seq_flat = seq_days.reshape(BD, seq_days.size(2), -1)       # (BD, T_seq, F_seq)
        lens_flat = seq_lens.reshape(BD).clamp(min=1).cpu()         # (BD,)
        prof_flat = profile_days.reshape(BD, profile_days.size(2), profile_days.size(3))  # (BD, C_prof, B_prof)
        rast_flat = raster_days.reshape(BD, raster_days.size(2), raster_days.size(3), raster_days.size(4))  # (BD, T, C, Bins)

        # ---- per-day: summary ----
        z_sum = self.summary_net(sum_flat)                          # (BD, d/2)

        # ---- per-day: intraday seq ----
        packed = rnn_utils.pack_padded_sequence(seq_flat, lens_flat, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.seq_lstm(packed)
        z_seq = self.seq_post(h_n[-1])                              # (BD, lstm_hidden)

        # ---- per-day: spatial = fuse(profile, raster) ----
        z_prof = self.profile_net(prof_flat)                        # (BD, d/2)
        z_nb = self.nb_proj(self.nb_net(rast_flat, None))            # (BD, d/2)
        z_spatial = self.spatial_fuse(z_prof, z_nb, return_importance=False)  # (BD, d/2)

        # ---- per-day fused embedding ----
        z_day = torch.cat([z_sum, z_seq, z_spatial], dim=-1)         # (BD, per_day_dim)
        z_day = self.day_fuse(z_day).reshape(B, D, -1)               # (B, D, per_day_dim)

        # ---- day LSTM ----
        _, (h_n_day, _) = self.day_lstm(z_day)
        z_ctx = h_n_day[-1]                                          # (B, day_lstm_hidden)

        logits = self.head(z_ctx)
        if self.task == "classification" and return_probs:
            return F.softmax(logits, dim=-1)
        return logits


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
        Fusion strategy: 'gated' (learned weights), 'concat' (concatenation), 'mean' (average).
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
