import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


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
        fusion_mode='default'
    ):
        super(DualBranchModel, self).__init__()

        self.task = task
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self._use_custom_seq_encoder = seq_encoder is not None

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
                    nn.Linear(head_input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_classes)
                )
        else:
            if fusion_mode == 'simple':
                self.fusion_net = nn.Linear(head_input_dim, 1)
            else:
                self.fusion_net = nn.Sequential(
                    nn.Linear(head_input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
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
            # Default LSTM with packing
            packed_input = rnn_utils.pack_padded_sequence(
                vpin_sequence,
                vpin_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            _, (hidden_state, _) = self.vpin_lstm(packed_input)
            seq_out = hidden_state[-1]

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
