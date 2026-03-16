"""
Reinforcement Learning module for CTAFlow.

This module provides RL environments and policies for training trading agents
using multi-modal market data (summary, profile, raster, sequential).

Components:
- Environments: EndOfDayTradingEnv, MultiModalTradingEnv, MultiTickerTradingEnv
- Feature Extractors: WSPRExtractor, WSPRExtractorV2, CnnLstmExtractor, MultiInputLstmExtractor
- Policy Factory: make_ppo_policy, get_policy_kwargs

Example Usage (Single Ticker):
    >>> from CTAFlow.models.deep_learning.rl import (
    ...     MultiModalTradingEnv,
    ...     make_ppo_policy,
    ... )
    >>> from CTAFlow.models.intraday_momentum import DeepIDMomentum
    >>>
    >>> # Load data
    >>> model_data = DeepIDMomentum.from_files(...)
    >>>
    >>> # Create environment
    >>> env = MultiModalTradingEnv(model_data, window_size=5)
    >>>
    >>> # Create and train PPO agent
    >>> model = make_ppo_policy(env, policy_type="wspr", dims=model_data.dims)
    >>> model.learn(total_timesteps=100_000)

Example Usage (Multi-Ticker with Meta Modality):
    >>> from CTAFlow.models.deep_learning.rl import (
    ...     MultiTickerTradingEnv,
    ...     WSPRExtractorV2,
    ... )
    >>> from CTAFlow.models.multi_asset import MultiAssetMomentum
    >>>
    >>> # Load multi-asset data
    >>> mam = MultiAssetMomentum(root_dir='data/', target_dir='targets/')
    >>> ticker_data = {t: mam.get_model(t) for t in mam.tickers}
    >>> ticker_meta = {
    ...     'ES': {'ticker_id': 0, 'asset_class_id': 0, 'asset_subclass_id': 0},
    ...     'CL': {'ticker_id': 1, 'asset_class_id': 1, 'asset_subclass_id': 0},
    ... }
    >>>
    >>> # Create multi-ticker environment
    >>> env = MultiTickerTradingEnv(
    ...     ticker_data=ticker_data,
    ...     ticker_meta=ticker_meta,
    ...     sampling_strategy='sequential',
    ...     common_dates_only=True,  # Prevents lookahead bias
    ... )
"""

from .env import (
    EndOfDayTradingEnv,
    MultiModalTradingEnv,
    MultiTickerTradingEnv,
    V3ContinuousPPOEnv,
    build_v3_rl_envs,
)
from .callbacks import TradingMetricsCallback, EarlyStoppingCallback
from .features_extractor import WSPRExtractor, WSPRExtractorV2, V3ContinuousExtractor, FiLMLayer
from .policy import (
    CnnLstmExtractor,
    MultiInputLstmExtractor,
    RecurrentActorCriticPolicy,
    CnnLstmPolicy,
    MultiInputLstmPolicy,
    WSPRPolicy,
    get_policy_kwargs,
    make_ppo_policy,
    make_v3_ppo,
)

__all__ = [
    # Environments
    'EndOfDayTradingEnv',
    'MultiModalTradingEnv',
    'MultiTickerTradingEnv',
    'V3ContinuousPPOEnv',
    'build_v3_rl_envs',
    # Feature Extractors
    'WSPRExtractor',
    'WSPRExtractorV2',
    'V3ContinuousExtractor',
    'FiLMLayer',
    'TradingMetricsCallback',
    'EarlyStoppingCallback',
    'CnnLstmExtractor',
    'MultiInputLstmExtractor',
    # Policies
    'RecurrentActorCriticPolicy',
    'CnnLstmPolicy',
    'MultiInputLstmPolicy',
    'WSPRPolicy',
    # Factory functions
    'get_policy_kwargs',
    'make_ppo_policy',
    'make_v3_ppo',
]
