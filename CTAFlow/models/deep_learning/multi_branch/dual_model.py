import torch
import torch.nn as nn
import torch.nn.functional as F
from ..encoders import SeqEncoder


class DualBranchModel(nn.Module):
    def __init__(self,
                 summary_input_dim,
                 vpin_input_dim=3,  # (VPIN, Return, LogDuration)
                 lstm_hidden_dim=64,
                 dense_hidden_dim=128,
                 task='regression',
                 num_classes=3):
        """Dual Branch Model for regression or classification.

        Parameters
        ----------
        summary_input_dim : int
            Dimension of summary features
        vpin_input_dim : int, default 3
            Dimension of sequential features
        lstm_hidden_dim : int, default 64
            Hidden dimension for LSTM
        dense_hidden_dim : int, default 128
            Hidden dimension for dense layers
        task : str, default 'regression'
            Task type: 'regression' or 'classification'
        num_classes : int, default 3
            Number of classes for classification task (ignored for regression)
        """
        super(DualBranchModel, self).__init__()

        self.task = task
        self.num_classes = num_classes

        # --- BRANCH A: MACRO SUMMARY (Static) ---
        # A simple MLP to process the ~33 CL features
        self.summary_net = nn.Sequential(
            nn.Linear(summary_input_dim, dense_hidden_dim),
            nn.BatchNorm1d(dense_hidden_dim),  # Helps with unscaled inputs
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dense_hidden_dim, dense_hidden_dim //2),
            nn.ReLU()
        )

        # --- BRANCH B: MICRO SEQUENCE (Time-Series) ---
        # SeqEncoder to process the variable-length VPIN buckets
        # Handles packing/unpacking internally
        self.vpin_lstm = SeqEncoder(
            f_in=vpin_input_dim,
            d_out=lstm_hidden_dim,
            d_conv=64,
            d_lstm=128,
            dropout=0.1,
            bidir=True
        )

        # --- FUSION HEAD ---
        # Concatenate (64 from Summary) + (64 from LSTM)
        fusion_input_dim = (dense_hidden_dim // 2) + lstm_hidden_dim

        if task == 'classification':
            # Classification: output logits for each class
            self.fusion_net = nn.Sequential(
                nn.Linear(fusion_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)  # Output logits for num_classes
            )
        else:
            # Regression: output single value
            self.fusion_net = nn.Sequential(
                nn.Linear(fusion_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)  # Final Prediction (e.g., Return)
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
        # 1. Process Macro Data
        summary_out = self.summary_net(summary_data)

        # 2. Process Micro Data
        # SeqEncoder handles packing/unpacking internally
        # Input: (batch, seq_len, features), seq_lengths
        # Output: (batch, d_out)
        lstm_out = self.vpin_lstm(vpin_sequence, vpin_lengths)

        # 3. Fuse
        combined = torch.cat((summary_out, lstm_out), dim=1)
        output = self.fusion_net(combined)

        # 4. Apply softmax for classification if requested
        if self.task == 'classification' and return_probs:
            output = F.softmax(output, dim=1)

        return output