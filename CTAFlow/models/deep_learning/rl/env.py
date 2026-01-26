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


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch


class MultiModalTradingEnv(gym.Env):
    """
    RL Environment that consumes a CTAFlow DeepIDMomentum object.

    It serves observations compatible with MultiModalRLFeatureExtractor:
    - 'summary_window': (Window, F_sum) -> History of macro features
    - 'profile_window': (Window, C, Bins) -> History of market structure
    - 'raster_current': (T_bars, C, Bins) -> Today's Vol/Price flow (Rasterized VPIN)
    - 'seq_current':    (SeqLen, F_seq) -> Today's Intraday Sequence
    - 'seq_lens':       (1,) -> Actual length of today's sequence

    Args:
        model_data (DeepIDMomentum): Pre-processed data object from CTAFlow.
        window_size (int): Lookback window for Summary and Profile branches.
        transaction_cost_bps (float): Transaction costs in basis points.
        max_seq_len (int): Maximum length for intraday sequential data (truncates/pads).
        reward_scale (float): Scaling factor for rewards (e.g., 100.0 for %)
    """

    def __init__(
            self,
            model_data,
            window_size=5,
            transaction_cost_bps=1.0,
            max_seq_len=200,
            reward_scale=1.0
    ):
        super().__init__()

        self.window_size = window_size
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.reward_scale = reward_scale
        self.max_seq_len = max_seq_len

        # --- 1. Data Ingestion & Alignment ---
        # We align everything to the target_data index (Trading Days)
        self.dates = model_data.target_data.index
        valid_dates_set = set(self.dates.date)

        # A. Summary Data (DataFrame -> Float32 Array)
        # Ensure we select only the features intended for the model
        feature_cols = model_data.feature_names
        summary_df = model_data.training_data['summary'][feature_cols].reindex(self.dates).fillna(0.0)
        self.data_summary = summary_df.values.astype(np.float32)

        # B. Profile Data (Array -> Aligned Array)
        # DeepIDMomentum stores profiles separately. We assume they are aligned or provide a date map.
        if model_data.profile_array is not None:
            # If profile_dates provided, we must align.
            # If not, we assume 1:1 mapping (risky but standard if generated together)
            if model_data.profile_dates is not None:
                # Build lookup
                prof_lookup = {pd.to_datetime(d).date(): model_data.profile_array[i]
                               for i, d in enumerate(model_data.profile_dates)}

                # Reconstruct aligned array
                input_shape = model_data.profile_array.shape[1:]  # (C, Bins)
                self.data_profile = np.zeros((len(self.dates), *input_shape), dtype=np.float32)

                for i, date in enumerate(self.dates.date):
                    if date in prof_lookup:
                        self.data_profile[i] = prof_lookup[date]
            else:
                self.data_profile = model_data.profile_array.astype(np.float32)
        else:
            # Fallback: Zero profiles
            self.data_profile = np.zeros((len(self.dates), 3, 64), dtype=np.float32)

        # C. Rasterized Data (Dict -> Aligned List of Arrays)
        # Raster data is (T_bars, C, Bins) per day
        self.data_raster = []
        # Get shape from first valid entry
        first_raster = next(iter(model_data.rasterized_data.values()))
        raster_shape = first_raster.shape

        for date in self.dates.date:
            if date in model_data.rasterized_data:
                self.data_raster.append(model_data.rasterized_data[date].astype(np.float32))
            else:
                self.data_raster.append(np.zeros(raster_shape, dtype=np.float32))

        # D. Sequential Data (DataFrame -> List of Padded Arrays)
        self.data_seq = []
        self.data_seq_lens = []

        # Group sequential data by date
        seq_df = model_data.sequential_data
        # Ensure date column exists (handle DatetimeIndex or column)
        if not 'date' in seq_df.columns:
            seq_df = seq_df.copy()
            seq_df['date'] = seq_df.index.date

        # Use dims.seq_cols if available (filters non-training columns)
        # Otherwise fall back to numeric-only filtering
        if hasattr(model_data, 'dims') and model_data.dims.seq_cols:
            seq_features = [c for c in model_data.dims.seq_cols if c in seq_df.columns]
        else:
            # Legacy fallback: numeric columns only, excluding identifiers
            seq_features = seq_df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove known non-training columns
            exclude_cols = {'date', 'bucket_id', 'bucket_idx', 'bar_id', 'session_id'}
            seq_features = [c for c in seq_features if c.lower() not in exclude_cols]

        seq_groups = seq_df.groupby('date')
        f_seq = len(seq_features)

        for date in self.dates.date:
            if date in seq_groups.groups:
                group = seq_groups.get_group(date)
                arr = group[seq_features].values.astype(np.float32)

                # Truncate or Pad
                length = min(len(arr), max_seq_len)
                padded = np.zeros((max_seq_len, f_seq), dtype=np.float32)
                padded[:length, :] = arr[:length, :]

                self.data_seq.append(padded)
                self.data_seq_lens.append(length)
            else:
                self.data_seq.append(np.zeros((max_seq_len, f_seq), dtype=np.float32))
                self.data_seq_lens.append(0)

        # Returns for reward calculation
        self.returns = model_data.target_data.reindex(self.dates).fillna(0.0).values.astype(np.float32)

        # --- 2. Space Definitions ---

        # Dimensions
        f_sum = self.data_summary.shape[1]
        prof_shape = self.data_profile.shape[1:]  # (C, Bins)
        rast_shape = raster_shape  # (T, C, Bins)

        self.observation_space = spaces.Dict({
            'summary_window': spaces.Box(-np.inf, np.inf, shape=(window_size, f_sum), dtype=np.float32),
            'profile_window': spaces.Box(-np.inf, np.inf, shape=(window_size, *prof_shape), dtype=np.float32),
            'raster_current': spaces.Box(-np.inf, np.inf, shape=rast_shape, dtype=np.float32),
            'seq_current': spaces.Box(-np.inf, np.inf, shape=(max_seq_len, f_seq), dtype=np.float32),
            'seq_lens': spaces.Box(0, max_seq_len, shape=(1,), dtype=np.int32)
        })

        self.action_space = spaces.Discrete(3)  # 0=Short, 1=Neutral, 2=Long

        # State vars
        self._idx = 0
        self._position = 0  # 0=Neutral, 1=Long, -1=Short

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start at window_size so we have enough history
        self._idx = self.window_size
        self._position = 0

        return self._get_obs(), {}

    def step(self, action):
        # Map action: 0->-1, 1->0, 2->1
        target_pos = action - 1

        # 1. Calculate PnL
        # We trade at the CLOSE of the current step (t).
        # The return for this decision is realized at t+1 (Close to Close).
        # Note: Be careful with alignment. If target_data[t] is return from t-1 to t,
        # then action at t-1 determines reward at t.
        # Here we assume self.returns[t] is the return achieved by holding through day t.

        step_return = self.returns[self._idx]

        # Transaction Cost (on position change)
        delta_pos = abs(target_pos - self._position)
        cost = delta_pos * self.transaction_cost

        gross_pnl = self._position * step_return  # PnL based on position HELD coming into this step
        net_reward = (gross_pnl - cost) * self.reward_scale

        # Update state
        self._position = target_pos
        self._idx += 1

        # Termination
        terminated = self._idx >= len(self.dates) - 1
        truncated = False

        obs = self._get_obs() if not terminated else self._get_obs_at(self._idx - 1)

        info = {
            'date': self.dates[self._idx - 1 if terminated else self._idx],
            'gross_pnl': gross_pnl,
            'cost': cost,
            'position': self._position
        }

        return obs, net_reward, terminated, truncated, info

    def _get_obs(self):
        return self._get_obs_at(self._idx)

    def _get_obs_at(self, idx):
        # Window slices: [t - w + 1 : t + 1] -> includes current step t as last element
        # This implies we use Today's Close data to make a decision for Tomorrow
        w = self.window_size
        start = idx - w + 1
        end = idx + 1

        return {
            'summary_window': self.data_summary[start:end],
            'profile_window': self.data_profile[start:end],
            'raster_current': self.data_raster[idx],  # Current day only
            'seq_current': self.data_seq[idx],  # Current day only
            'seq_lens': np.array([self.data_seq_lens[idx]], dtype=np.int32)
        }