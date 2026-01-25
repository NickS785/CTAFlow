import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from ..encoders import RasterResNet, MarketProfileResNet, SpatialFuse, IntradayRNN



class DualBranchModel(nn.Module):
    """Dual Branch Model for regression or classification.

    Supports interchangeable summary and sequential encoders, with configurable
    fusion modes.

    Parameters
    ----------
    summary_input_dim : int
        Dimension of summary features (ignored if summary_encoder is provided)
    vpin_input_dim : int, default 3
        Dimension of sequential features (ignored if seq_encoder is provided)
    lstm_hidden_dim : int, default 64
        Hidden dimension for LSTM (ignored if seq_encoder is provided)
    dense_hidden_dim : int, default 128
        Hidden dimension for dense layers (ignored if summary_encoder is provided)
    task : str, default 'regression'
        Task type: 'regression' or 'classification'
    num_classes : int, default 3
        Number of classes for classification task (ignored for regression)
    summary_encoder : nn.Module, optional
        Custom encoder for summary features. Must have `out_dim` attribute.
        If None, uses default MLP.
    seq_encoder : nn.Module, optional
        Custom encoder for sequential features. Must accept (x, lengths) and
        have `out_dim` attribute. If None, uses default LSTM with packing.
    fusion_mode : str, default 'default'
        Fusion strategy: 'default' (MLP), 'simple' (linear), 'gated' (learned weights)
    """

    def __init__(
        self,
        summary_input_dim,
        vpin_input_dim=3,
        lstm_hidden_dim=64,
        dense_hidden_dim=128,
        task='regression',
        num_classes=3,
        summary_encoder=None,
        seq_encoder=None,
        fusion_mode='default',
        seq_dropout = 0.1,
        fusion_dropout = 0.3
    ):
        super(DualBranchModel, self).__init__()

        self.task = task
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self._use_custom_seq_encoder = seq_encoder is not None
        self.seq_dropout = nn.Dropout(seq_dropout)

        # --- BRANCH A: MACRO SUMMARY (Static) ---
        if summary_encoder is not None:
            self.summary_net = summary_encoder
            summary_out_dim = summary_encoder.out_dim
        else:
            # Default: simple MLP to process summary features
            self.summary_net = nn.Sequential(
                nn.Linear(summary_input_dim, dense_hidden_dim),
                nn.BatchNorm1d(dense_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(dense_hidden_dim, dense_hidden_dim // 2),
                nn.ReLU()
            )
            summary_out_dim = dense_hidden_dim // 2

        # --- BRANCH B: MICRO SEQUENCE (Time-Series) ---
        if seq_encoder is not None:
            self.seq_encoder = seq_encoder
            seq_out_dim = seq_encoder.out_dim
            self.vpin_lstm = None  # Not used with custom encoder
        else:
            # Default: LSTM to process variable-length VPIN buckets
            self.vpin_lstm = nn.LSTM(
                input_size=vpin_input_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.seq_encoder = None
            seq_out_dim = lstm_hidden_dim

        # --- FUSION HEAD ---
        fusion_input_dim = summary_out_dim + seq_out_dim

        if fusion_mode == 'gated':
            # Gated fusion: learned weights for each branch
            self.gate = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim // 2),
                nn.GELU(),
                nn.Linear(fusion_input_dim // 2, 2),
                nn.Softmax(dim=-1)
            )
            # Project branches to same dim for weighted sum
            self.summary_proj = nn.Linear(summary_out_dim, 64)
            self.seq_proj = nn.Linear(seq_out_dim, 64)
            head_input_dim = 64
        elif fusion_mode == 'simple':
            # Simple: direct linear projection from concatenation
            head_input_dim = fusion_input_dim
        else:  # 'default'
            head_input_dim = fusion_input_dim

        # Output head
        if task == 'classification':
            if fusion_mode == 'simple':
                self.fusion_net = nn.Linear(head_input_dim, num_classes)
            else:
                self.fusion_net = nn.Sequential(
                    nn.Linear(head_input_dim, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(fusion_dropout),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 1 if task == 'regression' else num_classes)
                )

        else:
            if fusion_mode == 'simple':
                self.fusion_net = nn.Linear(head_input_dim, 1)
            else:
                self.fusion_net = nn.Sequential(
                    nn.Linear(head_input_dim, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(fusion_dropout),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 1 if task == 'regression' else num_classes)
                )

    def forward(self, summary_data, vpin_sequence, vpin_lengths, return_probs=False):
        """Forward pass through the dual branch model.

        Parameters
        ----------
        summary_data : torch.Tensor
            Summary features, shape (batch, summary_input_dim)
        vpin_sequence : torch.Tensor
            Sequential features, shape (batch, seq_len, vpin_input_dim)
        vpin_lengths : torch.Tensor
            Actual lengths of sequences (before padding), shape (batch,)
        return_probs : bool, default False
            For classification: if True, return probabilities (softmax applied).
            If False, return raw logits. Ignored for regression.

        Returns
        -------
        torch.Tensor
            For regression: shape (batch, 1)
            For classification with return_probs=False: shape (batch, num_classes) - logits
            For classification with return_probs=True: shape (batch, num_classes) - probabilities
        """
        # 1. Process Summary Data
        summary_out = self.summary_net(summary_data)

        # 2. Process Sequential Data
        if self._use_custom_seq_encoder:
            seq_out = self.seq_encoder(vpin_sequence, vpin_lengths)
        else:
            # Default LSTM with packing (clamp to at least 1 to avoid errors)
            vpin_lengths_clamped = vpin_lengths.clamp(min=1).cpu()
            packed_input = rnn_utils.pack_padded_sequence(
                vpin_sequence,
                vpin_lengths_clamped,
                batch_first=True,
                enforce_sorted=False
            )
            _, (hidden_state, _) = self.vpin_lstm(packed_input)
            seq_out = hidden_state[-1]
            seq_out = self.seq_dropout(seq_out)

        # 3. Fuse
        if self.fusion_mode == 'gated':
            # Compute gate weights from concatenated features
            combined = torch.cat((summary_out, seq_out), dim=1)
            gate_weights = self.gate(combined)  # (batch, 2)
            # Project and weight
            summary_proj = self.summary_proj(summary_out)
            seq_proj = self.seq_proj(seq_out)
            fused = gate_weights[:, 0:1] * summary_proj + gate_weights[:, 1:2] * seq_proj
        else:
            # Default or simple: concatenate
            fused = torch.cat((summary_out, seq_out), dim=1)

        output = self.fusion_net(fused)

        # 4. Apply softmax for classification if requested
        if self.task == 'classification' and return_probs:
            output = F.softmax(output, dim=1)

        return output


class RecurrentDualModal(nn.Module):
    """
    State-of-the-art Sequence Model for Market Data.

    Components:
    1. Macro: Summary MLP
    2. Structure (Static): MarketProfileResNet (1D ResNet + SE)
    3. Flow (Dynamic): RasterResNet (Pseudo-3D ResNet)
    4. Sequence: Window LSTM
    """

    def __init__(
            self,
            f_sum: int,
            f_profile: int = 3,
            f_raster: int = 4,
            num_bins: int = 64,
            d_model: int = 128,
            lstm_hidden: int = 128,
            spatial_fuse_mode: str = "gated",
            task: str = "classification",
            num_classes: int = 3,
            dropout: float = 0.2
    ):
        super().__init__()
        self.task = task

        # --- 1. Summary Encoder ---
        self.summary_net = nn.Sequential(
            nn.Linear(f_sum, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
        )

        # --- 2. Enhanced Spatial Encoders ---

        # A. Profile (Static Structure) - NOW UPGRADED
        self.profile_net = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model // 2,
            layers=[2, 2, 2]  # 2 blocks per layer = 6 ResBlocks total
        )

        # B. Raster (Dynamic Flow) - 3D ResNet logic
        self.raster_net = RasterResNet(
            in_ch=f_raster,
            d_model=d_model // 2,
            layers=[2, 2, 2],
            base_filters=32
        )

        # C. Fusion
        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model // 2,
            mode=spatial_fuse_mode
        )

        # --- 3. Day Fusion & LSTM ---
        self.day_fuse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # The LSTM now tracks the evolution of these high-quality embeddings
        self.window_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes if task == 'classification' else 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for ResNets
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, summary_days, profile_days, raster_days, return_probs=False):
        """
        Input Shapes:
        - summary_days: (Batch, Window, f_sum)
        - profile_days: (Batch, Window, f_prof, bins)
        - raster_days:  (Batch, Window, T_bars, f_rast, bins)
        """
        B, W, _, _ = profile_days.shape
        BW = B * W

        # 1. Flatten Time for shared encoders
        flat_sum = summary_days.reshape(BW, -1)
        flat_prof = profile_days.reshape(BW, profile_days.size(2), -1)
        flat_rast = raster_days.reshape(BW, raster_days.size(2), raster_days.size(3), -1)

        # 2. Run Encoders
        z_sum = self.summary_net(flat_sum)  # (BW, d/2)
        z_prof = self.profile_net(flat_prof)  # (BW, d/2) -> Uses ResNet
        z_rast = self.raster_net(flat_rast)  # (BW, d/2) -> Uses RasterResNet

        # 3. Spatial Fusion (Static + Dynamic)
        z_spatial = self.spatial_fuse(z_prof, z_rast)  # (BW, d/2)

        # 4. Day Fusion (Macro + Spatial)
        z_day_cat = torch.cat([z_sum, z_spatial], dim=1)  # (BW, d)
        z_day = self.day_fuse(z_day_cat)  # (BW, d)

        # 5. Window LSTM
        z_seq = z_day.view(B, W, -1)  # (B, W, d)
        _, (h_n, _) = self.window_lstm(z_seq)
        z_final = h_n[-1]  # (B, lstm_hidden)

        # 6. Prediction
        logits = self.head(z_final)

        if self.task == "classification" and return_probs:
            return F.softmax(logits, dim=1)

        return logits


class RecurrentWSPR(nn.Module):
    """
    Recurrent Windowed Summary, Profile, and recent Raster/Sequential/Fused model.

    This model processes inputs through multiple parallel paths as per user spec:
    - Path 1 (Windowed Summary): LSTM over a window of summary embeddings.
    - Path 2 (Windowed Profile): LSTM over a window of profile embeddings.
    - Path 3 (Recent Raster): Encoder processes only the most recent raster data.
    - Path 4 (Recent Sequential): Encoder processes only the most recent sequential data.
    - Path 5 (Recent Spatial Fusion): Spatially fuses the most recent profile and raster data.

    The outputs of all five paths are concatenated for final prediction.
    """
    def __init__(
            self,
            f_sum: int,
            f_profile: int,
            f_raster: int,
            f_seq: int,
            d_model: int = 128,
            sum_lstm_hidden: int = 64,
            prof_lstm_hidden: int = 128,
            task: str = "classification",
            num_classes: int = 3,
            dropout: float = 0.3
    ):
        super().__init__()
        self.task = task
        self.d_model = d_model

        # --- Branch Encoders ---
        self.summary_net = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.profile_net = MarketProfileResNet(in_channels=f_profile, d_model=d_model)
        self.raster_net = RasterResNet(in_ch=f_raster, d_model=d_model)
        self.seq_net = IntradayRNN(input_dim=f_seq, d_model=d_model, num_layers=1)
        self.spatial_fuse = SpatialFuse(d_spatial=d_model, mode="gated")

        # --- Path 1 & 2: Windowed LSTMs ---
        self.summary_lstm = nn.LSTM(input_size=d_model, hidden_size=sum_lstm_hidden, batch_first=True)
        self.profile_lstm = nn.LSTM(input_size=d_model, hidden_size=prof_lstm_hidden, batch_first=True)

        # --- Final Fusion Head ---
        final_fusion_dim = (
            sum_lstm_hidden      # Path 1
            + prof_lstm_hidden   # Path 2
            + d_model            # Path 3 (Recent Raster)
            + d_model            # Path 4 (Recent Seq)
            + d_model            # Path 5 (Recent Spatial Fuse)
        )
        self.head = nn.Sequential(
            nn.Linear(final_fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes if task == 'classification' else 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, summary_days, profile_days, raster_recent, seq_recent, seq_lens_recent, return_probs=False):
        B, W = summary_days.shape[0:2]
        BW = B * W

        # --- Encode all days for windowed paths ---
        flat_sum = summary_days.reshape(BW, -1)
        z_sum_all = self.summary_net(flat_sum) # (BW, d)

        flat_prof = profile_days.reshape(BW, profile_days.size(2), -1)
        z_prof_all = self.profile_net(flat_prof) # (BW, d)

        # --- Path 1: Windowed Summary LSTM ---
        z_sum_seq = z_sum_all.view(B, W, -1)
        _, (h_n_sum, _) = self.summary_lstm(z_sum_seq)
        z_summary_temporal = h_n_sum[-1] # (B, sum_lstm_hidden)

        # --- Path 2: Windowed Profile LSTM ---
        z_prof_seq = z_prof_all.view(B, W, -1)
        _, (h_n_prof, _) = self.profile_lstm(z_prof_seq)
        z_profile_temporal = h_n_prof[-1] # (B, prof_lstm_hidden)

        # --- Path 3: Recent Raster ---
        z_raster_recent = self.raster_net(raster_recent) # (B, d)

        # --- Path 4: Recent Sequential ---
        z_seq_recent = self.seq_net(seq_recent, lengths=seq_lens_recent) # (B, d)

        # --- Path 5: Recent Spatial Fusion ---
        # We need the profile embedding for the most recent day
        z_prof_recent = z_prof_seq[:, -1, :] # (B, d)
        # We need a raster embedding for fusion. We can reuse the one from Path 3.
        z_spatial_fused_recent = self.spatial_fuse(z_prof_recent, z_raster_recent) # (B, d)

        # --- Final Concatenation ---
        z_final_cat = torch.cat([
            z_summary_temporal,
            z_profile_temporal,
            z_raster_recent,
            z_seq_recent,
            z_spatial_fused_recent
        ], dim=1)

        # --- Prediction Head ---
        logits = self.head(z_final_cat)

        if self.task == "classification" and return_probs:
            return F.softmax(logits, dim=1)
        return logits