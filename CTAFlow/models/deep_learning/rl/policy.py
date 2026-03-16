"""
Custom RL Policies for Multi-Modal Trading with Stable-Baselines3.

This module provides CnnLSTM and MultiInputLSTM policy architectures for PPO
that work with the MultiModalTradingEnv environment.

Usage:
    from CTAFlow.models.deep_learning.rl.policy import (
        CnnLstmPolicy,
        MultiInputLstmPolicy,
        make_ppo_policy,
    )
    from stable_baselines3 import PPO

    # Option 1: Use factory function
    model = make_ppo_policy(
        env,
        policy_type="cnn_lstm",
        dims=model_data.dims,
    )

    # Option 2: Direct instantiation with custom config
    policy_kwargs = dict(
        features_extractor_class=WSPRExtractor,
        features_extractor_kwargs=dict(
            d_model=128,
            f_sum=dims.summary_dim,
            f_seq=dims.seq_dim,
            f_profile=dims.profile_channels,
            f_raster=dims.raster_channels,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
"""

from typing import Any, Dict, List, Optional, Type

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .features_extractor import WSPRExtractor, V3ContinuousExtractor
from ..encoders import (
    MarketProfileResNet,
    RasterResNet,
    IntradayRNN,
    SpatialFuse,
)


class CnnLstmExtractor(BaseFeaturesExtractor):
    """
    CNN-LSTM Feature Extractor for spatial + temporal data.

    Architecture:
    1. Profile CNN -> profile embedding
    2. Raster CNN -> raster embedding
    3. Concat [profile, raster] -> LSTM -> temporal embedding
    4. Summary MLP -> summary embedding
    5. Concat all -> final features

    This is a simplified version of WSPRExtractor that uses a single
    LSTM to process the concatenated spatial features over the window.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            d_model: int = 128,
            lstm_hidden: int = 128,
            f_sum: int = 10,
            f_profile: int = 3,
            f_raster: int = 4,
            f_seq: int = 5,
            dropout: float = 0.1,
    ):
        # Calculate total output dimension
        # summary_out + lstm_hidden (spatial temporal) + seq_out
        features_dim = d_model + lstm_hidden + d_model

        super().__init__(observation_space, features_dim)

        self.d_model = d_model
        self.lstm_hidden = lstm_hidden

        # Summary branch
        self.summary_enc = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Profile CNN
        self.profile_cnn = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model,
        )

        # Raster CNN
        self.raster_cnn = RasterResNet(
            in_ch=f_raster,
            d_model=d_model,
        )

        # Spatial fusion LSTM - processes [profile_emb, raster_emb] over window
        self.spatial_lstm = nn.LSTM(
            input_size=d_model * 2,  # profile + raster concatenated
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Sequential encoder for intraday data
        self.seq_enc = IntradayRNN(
            input_dim=f_seq,
            d_model=d_model,
            num_layers=1,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        summary_win = observations['summary_window']  # (B, W, F_sum)
        profile_win = observations['profile_window']  # (B, W, C, Bins)
        raster_curr = observations['raster_current']  # (B, T, C, Bins)
        seq_curr = observations['seq_current']  # (B, SeqLen, F_seq)
        seq_lens = observations.get('seq_lens', None)

        B, W = summary_win.shape[0], summary_win.shape[1]

        # 1. Summary: Use only the current day (last in window)
        summary_curr = summary_win[:, -1, :]  # (B, F_sum)
        z_sum = self.summary_enc(summary_curr)  # (B, d_model)

        # 2. Profile CNN over window
        flat_prof = profile_win.reshape(B * W, profile_win.shape[2], profile_win.shape[3])
        z_prof_all = self.profile_cnn(flat_prof)  # (B*W, d_model)
        z_prof_seq = z_prof_all.view(B, W, -1)  # (B, W, d_model)

        # 3. Raster CNN for current day
        z_rast = self.raster_cnn(raster_curr)  # (B, d_model)

        # 4. Create spatial sequence by appending current raster to profile history
        # Broadcast raster to match profile sequence, then concatenate
        z_rast_expanded = z_rast.unsqueeze(1).expand(-1, W, -1)  # (B, W, d_model)
        spatial_seq = torch.cat([z_prof_seq, z_rast_expanded], dim=-1)  # (B, W, d_model*2)

        # 5. LSTM over spatial sequence
        _, (h_spatial, _) = self.spatial_lstm(spatial_seq)
        z_spatial = h_spatial[-1]  # (B, lstm_hidden)

        # 6. Sequential encoder for intraday
        z_seq = self.seq_enc(seq_curr, lengths=seq_lens)  # (B, d_model)

        # 7. Concatenate all features
        features = torch.cat([z_sum, z_spatial, z_seq], dim=-1)
        features = self.layer_norm(features)
        features = self.dropout(features)

        return features


class MultiInputLstmExtractor(BaseFeaturesExtractor):
    """
    Multi-Input LSTM Feature Extractor with separate encoders per modality.

    Architecture:
    1. Summary Window -> MLP per step -> LSTM -> summary history
    2. Profile Window -> CNN per step -> LSTM -> profile history
    3. Raster Current -> CNN -> raster embedding
    4. Seq Current -> RNN -> seq embedding
    5. Spatial Fusion (profile[-1] + raster) -> spatial embedding
    6. Cross-Modal LSTM: [summary_h, profile_h, spatial, seq] -> final

    This provides more explicit modeling of cross-modal interactions.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            d_model: int = 128,
            sum_lstm_hidden: int = 64,
            prof_lstm_hidden: int = 64,
            cross_lstm_hidden: int = 128,
            f_sum: int = 10,
            f_profile: int = 3,
            f_raster: int = 4,
            f_seq: int = 5,
            use_attention: bool = True,
            dropout: float = 0.1,
    ):
        # Output: cross_lstm_hidden (final aggregated representation)
        features_dim = cross_lstm_hidden

        super().__init__(observation_space, features_dim)

        self.d_model = d_model
        self.use_attention = use_attention

        # --- Per-modality encoders ---

        # Summary branch
        self.summary_enc = nn.Sequential(
            nn.Linear(f_sum, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.summary_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=sum_lstm_hidden,
            batch_first=True,
        )

        # Profile branch
        self.profile_enc = MarketProfileResNet(
            in_channels=f_profile,
            d_model=d_model,
        )
        self.profile_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=prof_lstm_hidden,
            batch_first=True,
        )

        # Raster branch
        self.raster_enc = RasterResNet(
            in_ch=f_raster,
            d_model=d_model,
        )

        # Sequential branch
        self.seq_enc = IntradayRNN(
            input_dim=f_seq,
            d_model=d_model,
            num_layers=1,
            dropout=dropout,
        )

        # Spatial fusion
        self.spatial_fuse = SpatialFuse(
            d_spatial=d_model,
            mode="gated",
        )

        # --- Cross-modal aggregation ---

        # Input to cross-modal: sum_lstm + prof_lstm + spatial + seq
        cross_input_dim = sum_lstm_hidden + prof_lstm_hidden + d_model + d_model

        if use_attention:
            # Project each modality to same dim for attention
            self.modal_proj = nn.ModuleDict({
                'summary': nn.Linear(sum_lstm_hidden, d_model),
                'profile': nn.Linear(prof_lstm_hidden, d_model),
                'spatial': nn.Identity(),  # already d_model
                'seq': nn.Identity(),  # already d_model
            })
            # Multi-head attention over modalities
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                batch_first=True,
                dropout=dropout,
            )
            self.cross_lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=cross_lstm_hidden,
                batch_first=True,
            )
        else:
            self.cross_lstm = nn.LSTM(
                input_size=cross_input_dim,
                hidden_size=cross_lstm_hidden,
                batch_first=True,
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        summary_win = observations['summary_window']  # (B, W, F_sum)
        profile_win = observations['profile_window']  # (B, W, C, Bins)
        raster_curr = observations['raster_current']  # (B, T, C, Bins)
        seq_curr = observations['seq_current']  # (B, SeqLen, F_seq)
        seq_lens = observations.get('seq_lens', None)

        B, W = summary_win.shape[0], summary_win.shape[1]

        # 1. Summary window -> LSTM
        flat_sum = summary_win.reshape(B * W, -1)
        z_sum_all = self.summary_enc(flat_sum)
        z_sum_seq = z_sum_all.view(B, W, -1)
        _, (h_sum, _) = self.summary_lstm(z_sum_seq)
        z_sum_hist = h_sum[-1]  # (B, sum_lstm_hidden)

        # 2. Profile window -> LSTM
        flat_prof = profile_win.reshape(B * W, profile_win.shape[2], profile_win.shape[3])
        z_prof_all = self.profile_enc(flat_prof)
        z_prof_seq = z_prof_all.view(B, W, -1)
        _, (h_prof, _) = self.profile_lstm(z_prof_seq)
        z_prof_hist = h_prof[-1]  # (B, prof_lstm_hidden)

        # 3. Current raster
        z_rast = self.raster_enc(raster_curr)  # (B, d_model)

        # 4. Current sequential
        z_seq = self.seq_enc(seq_curr, lengths=seq_lens)  # (B, d_model)

        # 5. Spatial fusion (current profile + raster)
        z_prof_curr = z_prof_seq[:, -1, :]
        z_spatial = self.spatial_fuse(z_prof_curr, z_rast)  # (B, d_model)

        # 6. Cross-modal aggregation
        if self.use_attention:
            # Project to common dimension
            z_sum_proj = self.modal_proj['summary'](z_sum_hist)
            z_prof_proj = self.modal_proj['profile'](z_prof_hist)
            z_spatial_proj = self.modal_proj['spatial'](z_spatial)
            z_seq_proj = self.modal_proj['seq'](z_seq)

            # Stack as sequence for attention: (B, 4, d_model)
            modal_seq = torch.stack([z_sum_proj, z_prof_proj, z_spatial_proj, z_seq_proj], dim=1)

            # Self-attention over modalities
            attn_out, _ = self.cross_attn(modal_seq, modal_seq, modal_seq)

            # LSTM over attended modalities
            _, (h_cross, _) = self.cross_lstm(attn_out)
            features = h_cross[-1]
        else:
            # Simple concatenation + single-step LSTM
            modal_cat = torch.cat([z_sum_hist, z_prof_hist, z_spatial, z_seq], dim=-1)
            modal_cat = modal_cat.unsqueeze(1)  # (B, 1, total_dim)
            _, (h_cross, _) = self.cross_lstm(modal_cat)
            features = h_cross[-1]

        features = self.layer_norm(features)
        features = self.dropout(features)

        return features


class RecurrentActorCriticPolicy(MultiInputActorCriticPolicy):
    """
    Actor-Critic Policy with recurrent state tracking.

    This wraps the feature extractor to support LSTM state persistence
    across environment steps within an episode.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            features_extractor_class: Type[BaseFeaturesExtractor] = WSPRExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs or {},
            **kwargs,
        )


# --- Policy Type Aliases ---
CnnLstmPolicy = "CnnLstmPolicy"
MultiInputLstmPolicy = "MultiInputLstmPolicy"
WSPRPolicy = "WSPRPolicy"


def get_policy_kwargs(
        policy_type: str,
        dims: Any,  # MultiModalDim from DeepIDMomentum
        d_model: int = 128,
        lstm_hidden: int = 128,
        dropout: float = 0.1,
        net_arch: Optional[Dict[str, List[int]]] = None,
        **kwargs,
) -> Dict[str, Any]:
    """
    Get policy_kwargs for PPO based on policy type and data dimensions.

    Parameters
    ----------
    policy_type : str
        One of 'cnn_lstm', 'multi_input_lstm', 'wspr'
    dims : MultiModalDim
        Dimensions from DeepIDMomentum.dims
    d_model : int
        Base embedding dimension
    lstm_hidden : int
        LSTM hidden size
    dropout : float
        Dropout rate
    net_arch : dict, optional
        Network architecture for policy/value heads
    **kwargs
        Additional kwargs passed to feature extractor

    Returns
    -------
    dict
        Policy kwargs for PPO constructor
    """
    if net_arch is None:
        net_arch = dict(pi=[128, 64], vf=[128, 64])

    base_kwargs = dict(
        d_model=d_model,
        f_sum=dims.summary_dim,
        f_profile=dims.profile_channels,
        f_raster=dims.raster_channels,
        f_seq=dims.seq_dim,
        dropout=dropout,
    )
    base_kwargs.update(kwargs)

    if policy_type.lower() in ('cnn_lstm', 'cnnlstm'):
        return dict(
            features_extractor_class=CnnLstmExtractor,
            features_extractor_kwargs=dict(
                lstm_hidden=lstm_hidden,
                **base_kwargs,
            ),
            net_arch=net_arch,
        )

    elif policy_type.lower() in ('multi_input_lstm', 'multiinputlstm'):
        return dict(
            features_extractor_class=MultiInputLstmExtractor,
            features_extractor_kwargs=dict(
                sum_lstm_hidden=lstm_hidden // 2,
                prof_lstm_hidden=lstm_hidden // 2,
                cross_lstm_hidden=lstm_hidden,
                use_attention=kwargs.get('use_attention', True),
                **base_kwargs,
            ),
            net_arch=net_arch,
        )

    elif policy_type.lower() in ('wspr', 'wsprextractor'):
        return dict(
            features_extractor_class=WSPRExtractor,
            features_extractor_kwargs=dict(
                sum_lstm_hidden=lstm_hidden // 2,
                prof_lstm_hidden=lstm_hidden,
                **base_kwargs,
            ),
            net_arch=net_arch,
        )

    elif policy_type.lower() in ('v3', 'mmtfv3', 'v3continuous'):
        # V3 infers all dims from observation_space, no `dims` needed
        v3_kwargs = dict(
            d_model=d_model,
            dropout=dropout,
        )
        # Pass through V3-specific params
        for k in (
            'backbone', 'use_vsn', 'ae_type',
            'd_static_emb', 'd_latent', 'd_ae_hidden',
            'ae_n_layers', 'kl_weight', 'n_codes',
            'n_heads', 'n_layers', 'd_ff',
            'd_state', 'd_conv', 'expand',
            'spatial_fuse_mode', 'spatial_fuse_temp',
            'grn_dropout',
            'bvs_temperature', 'bvs_entropy_weight',
            'bvs_min_weight', 'bvs_pre_norm',
        ):
            if k in kwargs:
                v3_kwargs[k] = kwargs.pop(k)
        return dict(
            features_extractor_class=V3ContinuousExtractor,
            features_extractor_kwargs=v3_kwargs,
            net_arch=net_arch,
        )

    else:
        raise ValueError(f"Unknown policy type: {policy_type}. "
                         f"Choose from: cnn_lstm, multi_input_lstm, wspr, v3")


def make_ppo_policy(
        env: gym.Env,
        policy_type: str = "wspr",
        dims: Any = None,
        d_model: int = 128,
        lstm_hidden: int = 128,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        **kwargs,
) -> PPO:
    """
    Create a PPO model with the specified policy architecture.

    Parameters
    ----------
    env : gym.Env
        MultiModalTradingEnv or similar multi-input environment
    policy_type : str
        Policy architecture: 'cnn_lstm', 'multi_input_lstm', or 'wspr'
    dims : MultiModalDim, optional
        Data dimensions from DeepIDMomentum.dims. If None, extracted from env.
    d_model : int
        Base embedding dimension
    lstm_hidden : int
        LSTM hidden size
    learning_rate : float
        Learning rate for optimizer
    n_steps : int
        Number of steps to collect per environment per update
    batch_size : int
        Minibatch size
    n_epochs : int
        Number of epochs per update
    gamma : float
        Discount factor
    gae_lambda : float
        GAE lambda parameter
    clip_range : float
        PPO clip range
    ent_coef : float
        Entropy coefficient for exploration
    vf_coef : float
        Value function coefficient
    max_grad_norm : float
        Maximum gradient norm for clipping
    device : str
        Device to use ('auto', 'cpu', 'cuda')
    verbose : int
        Verbosity level
    tensorboard_log : str, optional
        Path for TensorBoard logging
    **kwargs
        Additional kwargs for policy_kwargs

    Returns
    -------
    PPO
        Configured PPO model ready for training

    Examples
    --------
    >>> from CTAFlow.models.deep_learning.rl.env import MultiModalTradingEnv
    >>> from CTAFlow.models.deep_learning.rl.policy import make_ppo_policy
    >>>
    >>> # Create environment
    >>> env = MultiModalTradingEnv(model_data, window_size=5)
    >>>
    >>> # Create PPO with CNN-LSTM policy
    >>> model = make_ppo_policy(
    ...     env,
    ...     policy_type="cnn_lstm",
    ...     dims=model_data.dims,
    ...     learning_rate=1e-4,
    ... )
    >>>
    >>> # Train
    >>> model.learn(total_timesteps=100_000)
    """
    # V3 infers dims from observation_space directly — skip dims extraction
    _is_v3 = policy_type.lower() in ('v3', 'mmtfv3', 'v3continuous')

    # Extract dimensions from env if not provided (non-V3 policies only)
    if dims is None and not _is_v3:
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            # Infer from observation space shapes
            from ...intraday_momentum import MultiModalDim
            summary_shape = obs_space['summary_window'].shape
            profile_shape = obs_space['profile_window'].shape
            raster_shape = obs_space['raster_current'].shape
            seq_shape = obs_space['seq_current'].shape

            dims = MultiModalDim(
                summary_dim=summary_shape[-1],  # (W, F) -> F
                seq_dim=seq_shape[-1],  # (SeqLen, F) -> F
                profile_channels=profile_shape[-2],  # (W, C, Bins) -> C
                profile_bins=profile_shape[-1],
                raster_bars=raster_shape[-3] if len(raster_shape) > 2 else raster_shape[0],
                raster_channels=raster_shape[-2],
                raster_bins=raster_shape[-1],
            )

    # Get policy kwargs
    policy_kwargs = get_policy_kwargs(
        policy_type=policy_type,
        dims=dims,
        d_model=d_model,
        lstm_hidden=lstm_hidden,
        **kwargs,
    )

    # Create PPO model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )

    return model


def make_v3_ppo(
    prep,
    backbone: str = "transformer",
    use_vsn: bool = True,
    ae_type: str = "vae",
    d_model: int = 128,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    device: str = "auto",
    verbose: int = 1,
    tensorboard_log: Optional[str] = None,
    tech_lookback: int = 64,
    seq_lookback_bars: int = 12,
    val_ratio: float = 0.2,
    val_cutoff_date: Optional[str] = None,
    **kwargs,
):
    """Create a V3 PPO model from V3ContinuousPrep in one call.

    Builds train/val environments via ``build_v3_rl_envs``, then
    constructs a PPO with ``V3ContinuousExtractor`` (transformer
    backbone + shared tech feature selection by default).

    Parameters
    ----------
    prep : V3ContinuousPrep
        Prepared data object.
    backbone : str
        'transformer' (default) or 'mamba'.
    use_vsn : bool
        Use shared per-feature selection for tech features (default True).
    ae_type : str
        Autoencoder type: 'deterministic', 'vae', 'vqvae'.
    d_model : int
        Base embedding dimension.
    learning_rate, n_steps, batch_size, n_epochs, gamma, ent_coef
        Standard PPO hyperparameters.
    tech_lookback, seq_lookback_bars, val_ratio, val_cutoff_date
        Passed to ``build_v3_rl_envs``.
    **kwargs
        Additional kwargs forwarded to ``make_ppo_policy`` / extractor.

    Returns
    -------
    tuple[PPO, V3ContinuousPPOEnv, V3ContinuousPPOEnv, dict]
        ``(ppo_model, train_env, val_env, info)``
    """
    from .env import build_v3_rl_envs

    train_env, val_env, info = build_v3_rl_envs(
        prep,
        tech_lookback=tech_lookback,
        seq_lookback_bars=seq_lookback_bars,
        val_ratio=val_ratio,
        val_cutoff_date=val_cutoff_date,
        **{k: kwargs.pop(k) for k in list(kwargs) if k in (
            'sample_session', 'sample_session_start',
            'sample_session_end', 'stride',
            'transaction_cost_bps', 'reward_scale',
            'max_episode_steps',
        )},
    )

    model = make_ppo_policy(
        env=train_env,
        policy_type="v3",
        d_model=d_model,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        device=device,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        backbone=backbone,
        use_vsn=use_vsn,
        ae_type=ae_type,
        **kwargs,
    )

    return model, train_env, val_env, info


# Convenience exports
__all__ = [
    'CnnLstmExtractor',
    'MultiInputLstmExtractor',
    'WSPRExtractor',
    'V3ContinuousExtractor',
    'RecurrentActorCriticPolicy',
    'CnnLstmPolicy',
    'MultiInputLstmPolicy',
    'WSPRPolicy',
    'get_policy_kwargs',
    'make_ppo_policy',
    'make_v3_ppo',
]
