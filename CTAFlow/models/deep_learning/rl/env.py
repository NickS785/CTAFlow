import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class EndOfDayTradingEnv(gym.Env):
    def __init__(self, features, returns, transaction_cost_bps=0.0):
        super(EndOfDayTradingEnv, self).__init__()

        self.features = features.values.astype(np.float32)
        self.returns = returns.values.astype(np.float32)
        self.dates = features.index
        self.transaction_cost = transaction_cost_bps / 10000.0

        # Action Space: 0=Short, 1=Neutral, 2=Long
        self.action_space = spaces.Discrete(3)

        # Observation Space: The feature vector size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.features.shape[1],),
            dtype=np.float32
        )

        self.current_step = 0
        self.last_action = 1  # Start flat (mapped to 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action = 1

        # Return first observation and info
        return self.features[self.current_step], {}

    def step(self, action):
        # Map: 0 -> -1 (Short), 1 -> 0 (Flat), 2 -> 1 (Long)
        position = action - 1

        # 1. Get Market Return
        market_return = self.returns[self.current_step]

        # 2. Calculate PnL (Gross)
        gross_return = position * market_return

        # 3. Transaction Costs (Round Trip)
        # Since we close automatically, every non-zero action incurs
        # entry AND exit costs.
        # If position is 0, cost is 0.
        # If position is 1 or -1, cost is applied twice (entry + exit).

        # Multiply by 2.0 for round-trip (entry + exit)
        cost = abs(position) * (self.transaction_cost * 2.0)

        reward = gross_return - cost

        # 4. Advance Step
        self.current_step += 1

        # 5. Check Termination
        terminated = (self.current_step >= len(self.features) - 1)
        truncated = False

        # 6. Get Next Observation
        next_obs = self.features[self.current_step]

        info = {
            'date': self.dates[self.current_step],
            'reward': reward,
            'position': position
        }

        return next_obs, reward, terminated, truncated, info
