import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Assuming these are importable from your project structure
from ..encoders import (
    MarketProfileResNet,
    RasterResNet,
    IntradayRNN,
    SpatialFuse
)


class WSPRExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible Feature Extractor using CTAFlow Encoders.

    Architecture based on RecurrentWSPR:
    1. Windowed Summary -> MLP -> LSTM -> History Embedding
    2. Windowed Profile -> ResNet -> LSTM -> Structure History Embedding
    3. Current Raster -> ResNet -> Flow Embedding
    4. Current Sequential -> RNN -> Intraday Embedding
    5. Spatial Fusion -> Gate(Profile_Current, Raster_Current) -> Fused Embedding

    Args:
        observation_space (gym.spaces.Dict): The observation space from the environment.
        d_model (int): Base embedding dimension for encoders.
        sum_lstm_hidden (int): Hidden size for the Summary LSTM.
        prof_lstm_hidden (int): Hidden size for the Profile LSTM.
        f_sum (int): Number of summary features.
        f_profile (int): Number of profile channels (e.g., 3).
        f_raster (int): Number of raster channels (e.g., 4).
        f_seq (int): Number of sequential features.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            d_model: int = 128,
            sum_lstm_hidden: int = 64,
            prof_lstm_hidden: int = 128,
            f_sum: int = 10,  # Adjust based on your data
            f_profile: int = 3,  # Adjust based on your data
            f_raster: int = 4,  # Adjust based on your data
            f_seq: int = 5,  # Adjust based on your data
            dropout: float = 0.0
    ):
        # Calculate total output dimension
        features_dim = (
                sum_lstm_hidden  # Windowed Summary History
                + prof_lstm_hidden  # Windowed Profile History
                + d_model  # Current Raster
                + d_model  # Current Sequential
                + d_model  # Current Spatial Fusion
        )

        super().__init__(observation_space, features_dim)

        # --- 1. Summary Branch (Windowed) ---
        # Encoder applied to every step in the window
        self.summary_enc = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        # LSTM to aggregate the window history
        self.summary_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=sum_lstm_hidden,
            batch_first=True
        )

        # --- 2. Profile Branch (Windowed) ---
        # Shared ResNet applied to every step in the window
        self.profile_enc = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model
        )
        # LSTM to aggregate the window history
        self.profile_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=prof_lstm_hidden,
            batch_first=True
        )

        # --- 3. Raster Branch (Current Only) ---
        # "3D" ResNet for the current time step's flow
        self.raster_enc = RasterResNet(
            in_ch=f_raster,
            d_model=d_model
        )

        # --- 4. Sequential Branch (Current Only) ---
        # RNN for the current intraday tick/bar sequence
        self.seq_enc = IntradayRNN(
            input_dim=f_seq,
            d_model=d_model,
            num_layers=1
        )

        # --- 5. Spatial Fusion ---
        # Fuses the Current Profile (from window) with Current Raster
        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model,
            mode="gated"
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, observations):
        # SB3 passes a Dict of Tensors.
        # Expected keys matching the environment definition:
        # 'summary_window': (B, W, F_sum)
        # 'profile_window': (B, W, C, Bins)
        # 'raster_current': (B, T_bars, C, Bins)
        # 'seq_current':    (B, SeqLen, F_seq)
        # 'seq_lens':       (B,)

        summary_win = observations['summary_window']
        profile_win = observations['profile_window']
        raster_curr = observations['raster_current']
        seq_curr = observations['seq_current']
        # Retrieve seq_lens if available, else None
        seq_lens = observations.get('seq_lens', None)

        B, W = summary_win.shape[0], summary_win.shape[1]

        # --- 1. Process Windowed Data (Summary & Profile) ---

        # Flatten Batch and Window dims to pass through shared encoders
        # Summary: (B, W, F) -> (B*W, F)
        flat_sum = summary_win.reshape(B * W, -1)

        # Profile: (B, W, C, Bins) -> (B*W, C, Bins)
        # Note: Check if Environment provides (B, W, Bins, C) or (B, W, C, Bins)
        # MarketProfileResNet expects (Batch, Channels, Bins)
        flat_prof = profile_win.reshape(B * W, profile_win.shape[2], profile_win.shape[3])

        # Encode
        z_sum_all = self.summary_enc(flat_sum)  # (B*W, d_model)
        z_prof_all = self.profile_enc(flat_prof)  # (B*W, d_model)

        # Unflatten back to sequences: (B, W, d_model)
        z_sum_seq = z_sum_all.view(B, W, -1)
        z_prof_seq = z_prof_all.view(B, W, -1)

        # Run LSTMs over the windows
        # We only care about the final hidden state (history context)
        # Output is (out, (h, c)). h is (NumLayers, B, Hidden).
        _, (h_sum, _) = self.summary_lstm(z_sum_seq)
        z_sum_hist = h_sum[-1]  # (B, sum_lstm_hidden)

        _, (h_prof, _) = self.profile_lstm(z_prof_seq)
        z_prof_hist = h_prof[-1]  # (B, prof_lstm_hidden)

        # --- 2. Process Current Data (Raster & Sequential) ---

        # Raster ResNet
        # Input: (B, T, C, Bins)
        z_rast_curr = self.raster_enc(raster_curr)  # (B, d_model)

        # Intraday RNN
        # Input: (B, SeqLen, F)
        # IntradayRNN in encoders.py handles lengths internally
        z_seq_curr = self.seq_enc(seq_curr, lengths=seq_lens)  # (B, d_model)

        # --- 3. Spatial Fusion ---

        # We need the profile embedding specifically for the *current* day
        # (the last step in the window) to fuse with the current raster.
        z_prof_curr = z_prof_seq[:, -1, :]  # (B, d_model)

        z_spatial_curr = self.spatial_fuse(z_prof_curr, z_rast_curr)  # (B, d_model)

        # --- 4. Concatenate All Features ---
        features = torch.cat([
            z_sum_hist,  # Macro Context
            z_prof_hist,  # Structural Context
            z_rast_curr,  # Current Flow (Grid)
            z_seq_curr,  # Current Flow (Sequence)
            z_spatial_curr  # Current Structure+Flow Alignment
        ], dim=1)

        features = self.layer_norm(features)
        features = self.dropout(features)

        return features