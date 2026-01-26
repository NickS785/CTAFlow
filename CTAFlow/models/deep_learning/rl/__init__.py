"""
Reinforcement Learning module for CTAFlow.

This module provides RL environments and policies for training trading agents
using multi-modal market data (summary, profile, raster, sequential).

Components:
- Environments: EndOfDayTradingEnv, MultiModalTradingEnv
- Feature Extractors: WSPRExtractor, CnnLstmExtractor, MultiInputLstmExtractor
- Policy Factory: make_ppo_policy, get_policy_kwargs

Example Usage:
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
"""

from .env import EndOfDayTradingEnv, MultiModalTradingEnv
from .features_extractor import WSPRExtractor
from .policy import (
    CnnLstmExtractor,
    MultiInputLstmExtractor,
    RecurrentActorCriticPolicy,
    CnnLstmPolicy,
    MultiInputLstmPolicy,
    WSPRPolicy,
    get_policy_kwargs,
    make_ppo_policy,
)

__all__ = [
    # Environments
    'EndOfDayTradingEnv',
    'MultiModalTradingEnv',
    # Feature Extractors
    'WSPRExtractor',
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
]
