import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Existing CTAFlow encoder components
from ..encoders import (
    MarketProfileResNet,
    RasterResNet,
    IntradayRNN,
    SpatialFuse,
    MetaModalityEncoder,
)
from ..multi_branch.tft.tft_encoders import (
    NumberBarEncoder,
    VPINRasterEncoder,
    TransformerTemporalBackbone,
    MambaTemporalBackbone,
)
from ..multi_branch.market_context_models import (
    BranchVariableSelection,
    GatedResidualNetwork,
)
from ..multi_branch.tft.auto_mmtft import (
    DeterministicRegimeAE,
    VariationalRegimeAE,
    VQRegimeAE,
    AutoencoderConditionedEncoder,
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


class V3ContinuousExtractor(BaseFeaturesExtractor):
    """SB3 extractor mirroring MMTFv3Core architecture.

    Follows the full MMTFv3 pipeline:
      Phase 0: VAE on ae_input → z_regime
      Phase 1: AutoencoderConditionedEncoder(identity + z_regime) → c_s, c_e, c_c, c_h
      Phase 2: Tech features → projection + regime enrichment → Mamba/Transformer backbone
               → (fused_seq, fused_token)
      Phase 3: Spatial (NumberBars + VPIN raster) → SpatialFuse → z_spatial
               Sequential (tabular VPIN) → IntradayRNN → z_seq
      Phase 4: BranchVariableSelection([fused_token, z_spatial, z_seq], context=c_s) → z_selected
      Phase 5: Enrichment + Temporal Attention
               enrichment_ctx = GRN(c_e), temporal_attn(query=ctx, K/V=fused_seq) → z_temporal
      Output:  cat[z_temporal, z_selected] → (B, 2*d_model)

    Parameters
    ----------
    observation_space : gym.spaces.Dict
        From V3ContinuousPPOEnv.
    d_model : int
        Main model dimension (matches MMTFv3Core).
    d_static_emb : int
        Identity embedding dimension.
    d_latent : int
        AE bottleneck dimension.
    d_ae_hidden : int
        AE GRU hidden size.
    ae_type : str
        'deterministic', 'vae', or 'vqvae'.
    ae_n_layers : int
        AE GRU layers.
    kl_weight : float
        VAE KL weight.
    backbone : str
        'transformer' or 'mamba'.
    n_heads : int
        Attention heads.
    n_layers : int
        Backbone layers.
    d_ff : int
        Transformer FFN dimension.
    d_state, d_conv, expand : int
        Mamba configuration.
    spatial_fuse_mode : str
        'gated' or 'mean'.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        d_model: int = 128,
        d_static_emb: int = 64,
        d_latent: int = 64,
        d_ae_hidden: int = 128,
        ae_type: str = "vae",
        ae_n_layers: int = 2,
        kl_weight: float = 0.01,
        n_codes: int = 16,
        backbone: str = "mamba",
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        spatial_fuse_mode: str = "gated",
        spatial_fuse_temp: float = 2.0,
        dropout: float = 0.2,
        grn_dropout: float | None = None,
    ):
        # Output is cat[z_temporal, z_selected] = 2 * d_model
        features_dim = d_model * 2
        super().__init__(observation_space, features_dim)

        self.d_model = d_model

        # Infer dimensions from observation space
        tech_shape = observation_space["tech_features"].shape     # (L_tech, f_tech)
        nb_shape = observation_space["numbars_recent"].shape      # (T_nb, bins, C)
        vr_shape = observation_space["vpin_raster_recent"].shape  # (T_vpin, C, bins)
        seq_shape = observation_space["seq_vpin"].shape           # (max_seq_len, f_seq)
        ae_shape = observation_space["ae_input"].shape            # (ae_window, f_ae)

        f_tech = int(tech_shape[-1])
        nb_channels = int(nb_shape[-1])
        vr_time = int(vr_shape[0])
        vr_channels = int(vr_shape[1])
        vr_bins = int(vr_shape[2])
        f_seq = int(seq_shape[-1])
        f_ae = int(ae_shape[-1])

        _grn_drop = grn_dropout if grn_dropout is not None else dropout

        # ==============================================================
        # PHASE 0: AUTOENCODER (long-term → regime latent)
        # ==============================================================
        if ae_type == "deterministic":
            self.autoencoder = DeterministicRegimeAE(
                f_input=f_ae, d_hidden=d_ae_hidden,
                d_latent=d_latent, n_layers=ae_n_layers,
                dropout=dropout,
            )
        elif ae_type == "vae":
            self.autoencoder = VariationalRegimeAE(
                f_input=f_ae, d_hidden=d_ae_hidden,
                d_latent=d_latent, n_layers=ae_n_layers,
                kl_weight=kl_weight, dropout=dropout,
            )
        elif ae_type == "vqvae":
            self.autoencoder = VQRegimeAE(
                f_input=f_ae, d_hidden=d_ae_hidden,
                d_latent=d_latent, n_codes=n_codes,
                n_layers=ae_n_layers, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown ae_type: {ae_type}")

        # ==============================================================
        # PHASE 1: REGIME-CONDITIONED STATIC ENCODER
        # identity + z_regime → c_s, c_e, c_c, c_h
        # ==============================================================
        n_tickers = max(observation_space["ticker_id"].n, 1)
        n_asset_classes = max(observation_space["asset_class_id"].n, 1)
        n_asset_subclasses = max(observation_space["asset_subclass_id"].n, 1)

        self.regime_encoder = AutoencoderConditionedEncoder(
            d_model=d_model,
            d_latent=d_latent,
            n_tickers=n_tickers,
            n_asset_classes=n_asset_classes,
            n_asset_subclasses=n_asset_subclasses,
            d_emb=d_static_emb,
            dropout=_grn_drop,
        )
        self.regime_bar_proj = GatedResidualNetwork(
            d_model=d_model, dropout=_grn_drop,
        )

        # ==============================================================
        # PHASE 2: TECHNICAL FEATURE PROJECTION + BACKBONE
        # ==============================================================
        self.tech_proj = nn.Sequential(
            nn.LayerNorm(f_tech),
            nn.Linear(f_tech, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if backbone == "transformer":
            self.fusion_backbone = TransformerTemporalBackbone(
                d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            )
        elif backbone == "mamba":
            self.fusion_backbone = MambaTemporalBackbone(
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv,
                expand=expand, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ==============================================================
        # PHASE 3: SPATIAL + SEQUENTIAL BRANCHES
        # ==============================================================
        self.numbar_encoder = NumberBarEncoder(
            in_channels=nb_channels, d_model=d_model,
        )
        self.vpin_encoder = VPINRasterEncoder(
            in_channels=vr_channels, n_bins=vr_bins,
            n_time=vr_time, d_model=d_model,
            n_heads=n_heads, dropout=dropout,
        )
        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model, mode=spatial_fuse_mode,
            temperature=spatial_fuse_temp,
        )
        self.seq_net = IntradayRNN(
            input_dim=f_seq, d_model=d_model,
            num_layers=1, dropout=dropout,
        )

        # ==============================================================
        # PHASE 4: BRANCH VARIABLE SELECTION (regime-conditioned)
        # ==============================================================
        self.branch_selector = BranchVariableSelection(
            n_branches=3,
            d_branch=d_model,
            d_context=d_model,
            dropout=_grn_drop,
        )

        # ==============================================================
        # PHASE 5: ENRICHMENT + TEMPORAL ATTENTION
        # ==============================================================
        self.enrichment_grn = GatedResidualNetwork(
            d_model=d_model, dropout=_grn_drop,
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.temporal_ln = nn.LayerNorm(d_model)

        # Tracking for diagnostics
        self._last_branch_weights = None
        self._last_regime_var_weights = None

    @staticmethod
    def _obs_to_index(x: torch.Tensor) -> torch.Tensor:
        """Convert SB3 Discrete observation (may be one-hot) to long index."""
        if x.dim() == 1:
            return x.long().view(-1)
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        if x.shape[1] == 1:
            return x[:, 0].long()
        return torch.argmax(x, dim=1).long()

    def forward(self, observations):
        tech = observations["tech_features"].float()       # (B, L, f_tech)
        nb = observations["numbars_recent"].float()        # (B, T_nb, bins, C)
        vr = observations["vpin_raster_recent"].float()    # (B, T_vpin, C, bins)
        seq = observations["seq_vpin"].float()             # (B, max_seq_len, f_seq)
        ae = observations["ae_input"].float()              # (B, ae_window, f_ae)

        # Sequence lengths
        seq_lens = observations.get("seq_vpin_lens")
        if seq_lens is None:
            seq_lens = torch.full(
                (seq.shape[0],), seq.shape[1],
                dtype=torch.long, device=seq.device,
            )
        else:
            if seq_lens.dim() == 2:
                seq_lens = seq_lens.squeeze(-1)
            seq_lens = seq_lens.long().clamp(min=1, max=seq.shape[1])

        # Identity indices
        ticker_id = self._obs_to_index(observations["ticker_id"])
        class_id = self._obs_to_index(observations["asset_class_id"])
        subclass_id = self._obs_to_index(observations["asset_subclass_id"])

        B, L, _ = tech.shape

        # ==============================================================
        # PHASE 0: AUTOENCODER → REGIME LATENT
        # ==============================================================
        z_regime, _x_recon, _ae_losses = self.autoencoder(ae)

        # ==============================================================
        # PHASE 1: REGIME-CONDITIONED CONTEXT
        # ==============================================================
        c_s, c_e, c_c, c_h = self.regime_encoder(
            ticker_id=ticker_id,
            asset_class_id=class_id,
            asset_subclass_id=subclass_id,
            z_regime=z_regime,
        )
        self._last_regime_var_weights = self.regime_encoder.last_var_weights

        # ==============================================================
        # PHASE 2: BACKBONE — TEMPORAL FUSION OVER TECH STREAM
        # ==============================================================
        z_tech = self.tech_proj(tech)                          # (B, L, d_model)
        z_regime_bar = self.regime_bar_proj(c_h)               # (B, d_model)
        z_tech_enriched = z_tech + z_regime_bar.unsqueeze(1)   # (B, L, d_model)

        fused_seq, fused_token = self.fusion_backbone(z_tech_enriched)
        # fused_seq:   (B, L, d_model)
        # fused_token: (B, d_model)

        # ==============================================================
        # PHASE 3: CROSS-MODAL BRANCH ENCODING
        # ==============================================================
        z_numbars = self.numbar_encoder(nb)
        z_vpin = self.vpin_encoder(vr)
        z_spatial = self.spatial_fuse(z_numbars, z_vpin)
        z_seq = self.seq_net(seq, lengths=seq_lens)

        # ==============================================================
        # PHASE 4: REGIME-CONDITIONED BRANCH SELECTION
        # ==============================================================
        branch_outputs = [fused_token, z_spatial, z_seq]
        z_selected, branch_weights = self.branch_selector(
            branch_outputs=branch_outputs,
            context=c_s,
        )
        self._last_branch_weights = branch_weights.detach()

        # ==============================================================
        # PHASE 5: ENRICHMENT + TEMPORAL ATTENTION
        # ==============================================================
        enrichment_ctx = self.enrichment_grn(c_e)              # (B, d_model)
        query = enrichment_ctx.unsqueeze(1)                    # (B, 1, d_model)
        attn_out, _attn_weights = self.temporal_attn(
            query=query, key=fused_seq, value=fused_seq,
            need_weights=False,
        )
        z_temporal = self.temporal_ln(
            attn_out.squeeze(1) + enrichment_ctx
        )                                                      # (B, d_model)

        # ==============================================================
        # OUTPUT: cat[z_temporal, z_selected] → (B, 2*d_model)
        # ==============================================================
        features = torch.cat([z_temporal, z_selected], dim=-1)
        return features

    def get_diagnostics(self) -> dict:
        """Return last-step diagnostic info for logging."""
        diag = {}
        if self._last_branch_weights is not None:
            names = ["backbone_fused", "spatial", "sequential"]
            diag["branch_weights"] = {
                name: self._last_branch_weights[:, i].mean().item()
                for i, name in enumerate(names)
            }
        if self._last_regime_var_weights is not None:
            names = self.regime_encoder.var_names
            diag["regime_var_weights"] = {
                name: self._last_regime_var_weights[:, i].mean().item()
                for i, name in enumerate(names)
            }
        diag["spatial_fuse"] = self.spatial_fuse.get_importance_stats()
        return diag
