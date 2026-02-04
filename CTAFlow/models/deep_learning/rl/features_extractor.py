import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Assuming these are importable from your project structure
from ..encoders import (
    MarketProfileResNet,
    RasterResNet,
    IntradayRNN,
    SpatialFuse,
    MetaModalityEncoder,
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


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for conditioning on meta embeddings.

    Learns to scale (gamma) and shift (beta) feature maps based on
    conditioning information (e.g., ticker identity, asset class).

    This allows the model to learn ticker-specific or asset-class-specific
    adjustments to the learned representations.
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)

        # Initialize near identity: gamma=1, beta=0
        nn.init.ones_(self.gamma.weight.data.mean(dim=1))
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features to modulate (B, D) or (B, T, D)
            cond: Conditioning vector (B, cond_dim)
        """
        gamma = self.gamma(cond)
        beta = self.beta(cond)

        if x.dim() == 3:
            # (B, T, D) case: expand conditioning
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


class WSPRExtractorV2(BaseFeaturesExtractor):
    """
    Multi-Ticker SB3-compatible Feature Extractor with Meta Modality.

    Enhanced architecture that supports learning across multiple assets:
    1. Windowed Summary -> MLP -> LSTM -> History Embedding
    2. Windowed Profile -> ResNet -> LSTM -> Structure History Embedding
    3. Current Raster -> ResNet -> Flow Embedding
    4. Current Sequential -> RNN -> Intraday Embedding
    5. Spatial Fusion -> Gate(Profile_Current, Raster_Current) -> Fused Embedding
    6. **Meta Modality** -> Ticker/Asset Class Embeddings + Calendar Features
    7. **FiLM Conditioning** -> Meta modulates other branches for ticker-specific behavior

    The meta modality allows the model to:
    - Learn ticker-specific patterns (e.g., ES vs CL behave differently)
    - Learn asset-class patterns (e.g., equities vs commodities)
    - Incorporate calendar effects (month, day-of-week, seasonality)

    Args:
        observation_space (gym.spaces.Dict): The observation space from the environment.
        d_model (int): Base embedding dimension for encoders.
        sum_lstm_hidden (int): Hidden size for the Summary LSTM.
        prof_lstm_hidden (int): Hidden size for the Profile LSTM.
        meta_hidden (int): Hidden size for the Meta LSTM output.
        f_sum (int): Number of summary features.
        f_profile (int): Number of profile channels (e.g., 3).
        f_raster (int): Number of raster channels (e.g., 4).
        f_seq (int): Number of sequential features.
        n_tickers (int): Number of unique tickers.
        n_asset_classes (int): Number of asset classes.
        n_asset_subclasses (int): Number of asset subclasses.
        use_film_conditioning (bool): Use FiLM to modulate branches with meta.
        dropout (float): Dropout rate.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            d_model: int = 128,
            sum_lstm_hidden: int = 64,
            prof_lstm_hidden: int = 128,
            meta_hidden: int = 64,
            f_sum: int = 10,
            f_profile: int = 3,
            f_raster: int = 4,
            f_seq: int = 5,
            n_tickers: int = 10,
            n_asset_classes: int = 5,
            n_asset_subclasses: int = 10,
            use_film_conditioning: bool = True,
            dropout: float = 0.1
    ):
        # Calculate total output dimension
        features_dim = (
                sum_lstm_hidden  # Windowed Summary History
                + prof_lstm_hidden  # Windowed Profile History
                + d_model  # Current Raster
                + d_model  # Current Sequential
                + d_model  # Current Spatial Fusion
                + meta_hidden  # Meta Modality
        )

        super().__init__(observation_space, features_dim)

        self.d_model = d_model
        self.use_film = use_film_conditioning

        # --- 1. Summary Branch (Windowed) ---
        self.summary_enc = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.summary_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=sum_lstm_hidden,
            batch_first=True
        )

        # --- 2. Profile Branch (Windowed) ---
        self.profile_enc = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model
        )
        self.profile_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=prof_lstm_hidden,
            batch_first=True
        )

        # --- 3. Raster Branch (Current Only) ---
        self.raster_enc = RasterResNet(
            in_ch=f_raster,
            d_model=d_model
        )

        # --- 4. Sequential Branch (Current Only) ---
        self.seq_enc = IntradayRNN(
            input_dim=f_seq,
            d_model=d_model,
            num_layers=1
        )

        # --- 5. Spatial Fusion ---
        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model,
            mode="gated"
        )

        # --- 6. Meta Modality Encoder ---
        self.meta_enc = MetaModalityEncoder(
            d_model=d_model,
            ctx_dim=64,
            time_dim=64,
            meta_hidden=meta_hidden,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            dropout=dropout,
        )

        # --- 7. FiLM Conditioning (optional) ---
        if use_film_conditioning:
            # Meta conditions each branch after encoding
            self.film_summary = FiLMLayer(sum_lstm_hidden, meta_hidden)
            self.film_profile = FiLMLayer(prof_lstm_hidden, meta_hidden)
            self.film_raster = FiLMLayer(d_model, meta_hidden)
            self.film_seq = FiLMLayer(d_model, meta_hidden)
            self.film_spatial = FiLMLayer(d_model, meta_hidden)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, observations):
        """
        Expected observation keys:
        - 'summary_window': (B, W, F_sum)
        - 'profile_window': (B, W, C, Bins)
        - 'raster_current': (B, T_bars, C, Bins)
        - 'seq_current':    (B, SeqLen, F_seq)
        - 'seq_lens':       (B,) or (B, 1)
        - 'ticker_id':      (B,) long
        - 'asset_class_id': (B,) long
        - 'asset_subclass_id': (B,) long
        - 'month':          (B, W) long [optional]
        - 'dow':            (B, W) long [optional]
        - 'doy_sin':        (B, W) float [optional]
        - 'doy_cos':        (B, W) float [optional]
        """
        summary_win = observations['summary_window']
        profile_win = observations['profile_window']
        raster_curr = observations['raster_current']
        seq_curr = observations['seq_current']
        seq_lens = observations.get('seq_lens', None)

        B, W = summary_win.shape[0], summary_win.shape[1]

        # Handle seq_lens shape
        if seq_lens is not None and seq_lens.dim() == 2:
            seq_lens = seq_lens.squeeze(-1)

        # --- 6. Process Meta Modality First (for conditioning) ---
        meta_dict = {
            'ticker_id': observations['ticker_id'],
            'asset_class_id': observations['asset_class_id'],
            'asset_subclass_id': observations['asset_subclass_id'],
            'month': observations.get('month'),
            'dow': observations.get('dow'),
            'doy_sin': observations.get('doy_sin'),
            'doy_cos': observations.get('doy_cos'),
        }

        z_meta_days, z_meta_window = self.meta_enc(meta_dict, W=W)

        # --- 1. Process Windowed Summary ---
        flat_sum = summary_win.reshape(B * W, -1)
        z_sum_all = self.summary_enc(flat_sum)
        z_sum_seq = z_sum_all.view(B, W, -1)
        _, (h_sum, _) = self.summary_lstm(z_sum_seq)
        z_sum_hist = h_sum[-1]

        # --- 2. Process Windowed Profile ---
        flat_prof = profile_win.reshape(B * W, profile_win.shape[2], profile_win.shape[3])
        z_prof_all = self.profile_enc(flat_prof)
        z_prof_seq = z_prof_all.view(B, W, -1)
        _, (h_prof, _) = self.profile_lstm(z_prof_seq)
        z_prof_hist = h_prof[-1]

        # --- 3. Process Current Raster ---
        z_rast_curr = self.raster_enc(raster_curr)

        # --- 4. Process Current Sequential ---
        z_seq_curr = self.seq_enc(seq_curr, lengths=seq_lens)

        # --- 5. Spatial Fusion ---
        z_prof_curr = z_prof_seq[:, -1, :]
        z_spatial_curr = self.spatial_fuse(z_prof_curr, z_rast_curr)

        # --- 7. Apply FiLM Conditioning (if enabled) ---
        if self.use_film:
            z_sum_hist = self.film_summary(z_sum_hist, z_meta_window)
            z_prof_hist = self.film_profile(z_prof_hist, z_meta_window)
            z_rast_curr = self.film_raster(z_rast_curr, z_meta_window)
            z_seq_curr = self.film_seq(z_seq_curr, z_meta_window)
            z_spatial_curr = self.film_spatial(z_spatial_curr, z_meta_window)

        # --- 8. Concatenate All Features ---
        features = torch.cat([
            z_sum_hist,
            z_prof_hist,
            z_rast_curr,
            z_seq_curr,
            z_spatial_curr,
            z_meta_window,
        ], dim=1)

        features = self.layer_norm(features)
        features = self.dropout(features)

        return features

    def get_meta_embedding(self, observations) -> torch.Tensor:
        """Extract just the meta embedding for analysis/visualization."""
        B = observations['ticker_id'].shape[0]
        W = observations['summary_window'].shape[1]
        meta_dict = {
            'ticker_id': observations['ticker_id'],
            'asset_class_id': observations['asset_class_id'],
            'asset_subclass_id': observations['asset_subclass_id'],
            'month': observations.get('month'),
            'dow': observations.get('dow'),
            'doy_sin': observations.get('doy_sin'),
            'doy_cos': observations.get('doy_cos'),
        }
        _, z_meta_window = self.meta_enc(meta_dict, W=W)
        return z_meta_window
