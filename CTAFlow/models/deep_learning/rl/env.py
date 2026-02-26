import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


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


class MultiTickerTradingEnv(gym.Env):
    """
    Multi-Ticker RL Environment for training across multiple assets.

    This environment supports training a single policy across multiple tickers,
    enabling the model to learn:
    - Cross-asset patterns and correlations
    - Ticker-specific behaviors via meta embeddings
    - Calendar/seasonality effects

    Compatible with WSPRExtractorV2 which uses MetaModalityEncoder.

    Observation Space includes:
    - Standard WSPR modalities (summary_window, profile_window, raster_current, seq_current)
    - Meta modalities (ticker_id, asset_class_id, asset_subclass_id)
    - Calendar features (month, dow, doy_sin, doy_cos)

    Args:
        ticker_data (Dict[str, DeepIDMomentum]): Dict mapping ticker -> model_data
        ticker_meta (Dict[str, Dict]): Dict mapping ticker -> {'ticker_id': int, 'asset_class_id': int, ...}
        window_size (int): Lookback window for Summary and Profile branches.
        transaction_cost_bps (float): Transaction costs in basis points.
        max_seq_len (int): Maximum length for intraday sequential data.
        reward_scale (float): Scaling factor for rewards.
        sampling_strategy (str): How to sample tickers:
            - 'sequential': Go through each ticker's full history before moving to next
            - 'interleaved': Alternate between tickers each step
            - 'random': Random ticker each episode
        common_dates_only (bool): If True, only use dates present in ALL tickers (prevents lookahead).
    """

    def __init__(
            self,
            ticker_data,
            ticker_meta=None,
            window_size=5,
            transaction_cost_bps=1.0,
            max_seq_len=200,
            reward_scale=1.0,
            sampling_strategy='sequential',
            common_dates_only=True,
    ):
        super().__init__()

        self.window_size = window_size
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.reward_scale = reward_scale
        self.max_seq_len = max_seq_len
        self.sampling_strategy = sampling_strategy

        self.tickers = list(ticker_data.keys())
        self.n_tickers = len(self.tickers)

        # --- Build ticker metadata mapping ---
        if ticker_meta is None:
            # Auto-generate IDs
            ticker_meta = {
                t: {'ticker_id': i, 'asset_class_id': 0, 'asset_subclass_id': 0}
                for i, t in enumerate(self.tickers)
            }
        self.ticker_meta = ticker_meta

        # Get max IDs for observation space
        self.max_ticker_id = max(m.get('ticker_id', 0) for m in ticker_meta.values()) + 1
        self.max_class_id = max(m.get('asset_class_id', 0) for m in ticker_meta.values()) + 1
        self.max_subclass_id = max(m.get('asset_subclass_id', 0) for m in ticker_meta.values()) + 1

        # --- Process each ticker's data ---
        self._ticker_data = {}  # ticker -> processed data dict

        # First pass: collect all dates
        all_dates_per_ticker = {}
        for ticker, model_data in ticker_data.items():
            dates = model_data.target_data.index
            all_dates_per_ticker[ticker] = set(d.date() if hasattr(d, 'date') else d for d in dates)

        # Compute common dates if requested
        if common_dates_only and len(self.tickers) > 1:
            common_dates = set.intersection(*all_dates_per_ticker.values())
            self.common_dates = sorted(common_dates)
        else:
            # Union of all dates (may have gaps per ticker)
            self.common_dates = None

        # Second pass: process data
        first_ticker = True
        for ticker, model_data in ticker_data.items():
            processed = self._process_ticker_data(
                ticker, model_data, self.common_dates
            )
            self._ticker_data[ticker] = processed

            # Get dimensions from first ticker
            if first_ticker:
                self.f_sum = processed['data_summary'].shape[1]
                self.prof_shape = processed['data_profile'].shape[1:]
                self.rast_shape = processed['raster_shape']
                self.f_seq = processed['f_seq']
                first_ticker = False

        # --- Build observation space ---
        self.observation_space = spaces.Dict({
            # Standard WSPR modalities
            'summary_window': spaces.Box(
                -np.inf, np.inf, shape=(window_size, self.f_sum), dtype=np.float32
            ),
            'profile_window': spaces.Box(
                -np.inf, np.inf, shape=(window_size, *self.prof_shape), dtype=np.float32
            ),
            'raster_current': spaces.Box(
                -np.inf, np.inf, shape=self.rast_shape, dtype=np.float32
            ),
            'seq_current': spaces.Box(
                -np.inf, np.inf, shape=(max_seq_len, self.f_seq), dtype=np.float32
            ),
            'seq_lens': spaces.Box(0, max_seq_len, shape=(1,), dtype=np.int32),

            # Meta modalities
            'ticker_id': spaces.Discrete(self.max_ticker_id),
            'asset_class_id': spaces.Discrete(self.max_class_id),
            'asset_subclass_id': spaces.Discrete(self.max_subclass_id),

            # Calendar features (per window day)
            'month': spaces.Box(0, 12, shape=(window_size,), dtype=np.int32),
            'dow': spaces.Box(0, 6, shape=(window_size,), dtype=np.int32),
            'doy_sin': spaces.Box(-1, 1, shape=(window_size,), dtype=np.float32),
            'doy_cos': spaces.Box(-1, 1, shape=(window_size,), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(3)  # 0=Short, 1=Neutral, 2=Long

        # --- Episode state ---
        self._current_ticker = None
        self._current_ticker_idx = 0
        self._idx = 0
        self._position = 0
        self._episode_ticker_order = list(range(self.n_tickers))

    def _process_ticker_data(self, ticker, model_data, common_dates=None):
        """Process a single ticker's data into arrays."""
        # Get dates
        dates = model_data.target_data.index

        if common_dates is not None:
            # Filter to common dates only
            date_set = set(common_dates)
            mask = [d.date() if hasattr(d, 'date') else d in date_set for d in dates]
            dates = dates[mask]

        # A. Summary Data
        feature_cols = model_data.feature_names
        summary_df = model_data.training_data['summary'][feature_cols].reindex(dates).fillna(0.0)
        data_summary = summary_df.values.astype(np.float32)

        # B. Profile Data
        if model_data.profile_array is not None:
            if model_data.profile_dates is not None:
                prof_lookup = {
                    pd.to_datetime(d).date(): model_data.profile_array[i]
                    for i, d in enumerate(model_data.profile_dates)
                }
                input_shape = model_data.profile_array.shape[1:]
                data_profile = np.zeros((len(dates), *input_shape), dtype=np.float32)
                for i, date in enumerate(dates):
                    d = date.date() if hasattr(date, 'date') else date
                    if d in prof_lookup:
                        data_profile[i] = prof_lookup[d]
            else:
                data_profile = model_data.profile_array.astype(np.float32)
        else:
            data_profile = np.zeros((len(dates), 3, 64), dtype=np.float32)

        # C. Rasterized Data
        data_raster = []
        first_raster = next(iter(model_data.rasterized_data.values()))
        raster_shape = first_raster.shape

        for date in dates:
            d = date.date() if hasattr(date, 'date') else date
            if d in model_data.rasterized_data:
                data_raster.append(model_data.rasterized_data[d].astype(np.float32))
            else:
                data_raster.append(np.zeros(raster_shape, dtype=np.float32))

        # D. Sequential Data
        data_seq = []
        data_seq_lens = []

        seq_df = model_data.sequential_data.copy()
        if 'date' not in seq_df.columns:
            seq_df['date'] = seq_df.index.date

        if hasattr(model_data, 'dims') and model_data.dims.seq_cols:
            seq_features = [c for c in model_data.dims.seq_cols if c in seq_df.columns]
        else:
            seq_features = seq_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = {'date', 'bucket_id', 'bucket_idx', 'bar_id', 'session_id'}
            seq_features = [c for c in seq_features if c.lower() not in exclude_cols]

        seq_groups = seq_df.groupby('date')
        f_seq = len(seq_features)

        for date in dates:
            d = date.date() if hasattr(date, 'date') else date
            if d in seq_groups.groups:
                group = seq_groups.get_group(d)
                arr = group[seq_features].values.astype(np.float32)
                length = min(len(arr), self.max_seq_len)
                padded = np.zeros((self.max_seq_len, f_seq), dtype=np.float32)
                padded[:length, :] = arr[:length, :]
                data_seq.append(padded)
                data_seq_lens.append(length)
            else:
                data_seq.append(np.zeros((self.max_seq_len, f_seq), dtype=np.float32))
                data_seq_lens.append(0)

        # E. Returns
        returns = model_data.target_data.reindex(dates).fillna(0.0).values.astype(np.float32)

        # F. Calendar features (precompute for efficiency)
        calendar = self._compute_calendar_features(dates)

        return {
            'dates': dates,
            'data_summary': data_summary,
            'data_profile': data_profile,
            'data_raster': data_raster,
            'data_seq': data_seq,
            'data_seq_lens': data_seq_lens,
            'returns': returns,
            'calendar': calendar,
            'raster_shape': raster_shape,
            'f_seq': f_seq,
            'n_samples': len(dates),
        }

    def _compute_calendar_features(self, dates):
        """Precompute calendar features for all dates."""
        months = np.array([d.month for d in dates], dtype=np.int32)
        dows = np.array([d.dayofweek for d in dates], dtype=np.int32)

        # Day of year as sin/cos (cyclical encoding)
        doys = np.array([d.dayofyear for d in dates], dtype=np.float32)
        doy_sin = np.sin(2 * np.pi * doys / 365.25).astype(np.float32)
        doy_cos = np.cos(2 * np.pi * doys / 365.25).astype(np.float32)

        return {
            'month': months,
            'dow': dows,
            'doy_sin': doy_sin,
            'doy_cos': doy_cos,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Select ticker based on strategy
        if self.sampling_strategy == 'random':
            self._current_ticker_idx = self.np_random.integers(0, self.n_tickers)
        elif self.sampling_strategy == 'sequential':
            # Cycle through tickers
            self._current_ticker_idx = (self._current_ticker_idx + 1) % self.n_tickers
        else:  # interleaved handled in step
            self._current_ticker_idx = 0

        self._current_ticker = self.tickers[self._current_ticker_idx]
        self._idx = self.window_size
        self._position = 0

        return self._get_obs(), {'ticker': self._current_ticker}

    def step(self, action):
        # Get current ticker's data
        ticker = self._current_ticker
        data = self._ticker_data[ticker]

        # Map action: 0->-1, 1->0, 2->1
        target_pos = action - 1

        # Calculate PnL
        step_return = data['returns'][self._idx]
        delta_pos = abs(target_pos - self._position)
        cost = delta_pos * self.transaction_cost
        gross_pnl = self._position * step_return
        net_reward = (gross_pnl - cost) * self.reward_scale

        # Update state
        self._position = target_pos
        self._idx += 1

        # Check termination
        terminated = self._idx >= data['n_samples'] - 1
        truncated = False

        # Handle interleaved strategy
        if self.sampling_strategy == 'interleaved' and not terminated:
            self._current_ticker_idx = (self._current_ticker_idx + 1) % self.n_tickers
            self._current_ticker = self.tickers[self._current_ticker_idx]

        obs = self._get_obs() if not terminated else self._get_obs_at(self._idx - 1)

        info = {
            'ticker': ticker,
            'date': data['dates'][self._idx - 1 if terminated else self._idx],
            'gross_pnl': gross_pnl,
            'cost': cost,
            'position': self._position,
        }

        return obs, net_reward, terminated, truncated, info

    def _get_obs(self):
        return self._get_obs_at(self._idx)

    def _get_obs_at(self, idx):
        ticker = self._current_ticker
        data = self._ticker_data[ticker]
        meta = self.ticker_meta[ticker]

        w = self.window_size
        start = max(0, idx - w + 1)
        end = idx + 1

        # Handle edge case where we don't have enough history
        if end - start < w:
            # Pad with zeros at the beginning
            pad_size = w - (end - start)
            summary_window = np.concatenate([
                np.zeros((pad_size, self.f_sum), dtype=np.float32),
                data['data_summary'][start:end]
            ])
            profile_window = np.concatenate([
                np.zeros((pad_size, *self.prof_shape), dtype=np.float32),
                data['data_profile'][start:end]
            ])
            month = np.concatenate([
                np.zeros(pad_size, dtype=np.int32),
                data['calendar']['month'][start:end]
            ])
            dow = np.concatenate([
                np.zeros(pad_size, dtype=np.int32),
                data['calendar']['dow'][start:end]
            ])
            doy_sin = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                data['calendar']['doy_sin'][start:end]
            ])
            doy_cos = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                data['calendar']['doy_cos'][start:end]
            ])
        else:
            summary_window = data['data_summary'][start:end]
            profile_window = data['data_profile'][start:end]
            month = data['calendar']['month'][start:end]
            dow = data['calendar']['dow'][start:end]
            doy_sin = data['calendar']['doy_sin'][start:end]
            doy_cos = data['calendar']['doy_cos'][start:end]

        return {
            # Standard WSPR modalities
            'summary_window': summary_window,
            'profile_window': profile_window,
            'raster_current': data['data_raster'][idx],
            'seq_current': data['data_seq'][idx],
            'seq_lens': np.array([data['data_seq_lens'][idx]], dtype=np.int32),

            # Meta modalities (scalars for this sample)
            'ticker_id': np.array(meta.get('ticker_id', 0), dtype=np.int64),
            'asset_class_id': np.array(meta.get('asset_class_id', 0), dtype=np.int64),
            'asset_subclass_id': np.array(meta.get('asset_subclass_id', 0), dtype=np.int64),

            # Calendar features (per window day)
            'month': month,
            'dow': dow,
            'doy_sin': doy_sin,
            'doy_cos': doy_cos,
        }

    @property
    def current_ticker(self):
        """Get the current ticker being traded."""
        return self._current_ticker

    def get_ticker_stats(self):
        """Get statistics about each ticker's data."""
        stats = {}
        for ticker, data in self._ticker_data.items():
            stats[ticker] = {
                'n_samples': data['n_samples'],
                'date_range': (data['dates'][0], data['dates'][-1]),
                'mean_return': float(data['returns'].mean()),
                'std_return': float(data['returns'].std()),
            }
        return stats


class V3ContinuousPPOEnv(gym.Env):
    """Gymnasium environment sourced from V3ContinuousPrep samples.

    Observations are the V3 multi-modal payload at each bar:
      - tech_window:        (L_tech, f_tech)
      - numbars_recent:     (T_nb, nb_bins, nb_channels)
      - vpin_raster_recent: (T_vpin, vpin_channels, vpin_bins)
      - seq_vpin:           (max_seq_len, f_seq)
      - seq_vpin_len:       (1,)
      - ae_input:           (ae_window, f_ae)
      - ticker_id / asset ids (discrete scalars)

    Action space:
      - Discrete(3): 0=Short (-1), 1=Flat (0), 2=Long (+1)

    Reward:
      reward_t = (prev_position_ticker * forward_return_t - tc_cost * |delta_position_ticker|) * reward_scale

    Position state is tracked independently per ticker.
    """

    def __init__(
        self,
        samples: List[Dict],
        max_seq_len: int = 24,
        transaction_cost_bps: float = 1.0,
        reward_scale: float = 100.0,
        numbars_channels: int = 4,
        vpin_channels: int = 4,
        max_episode_steps: Optional[int] = None,
        random_start: bool = False,
    ):
        super().__init__()
        if not samples:
            raise ValueError("V3ContinuousPPOEnv requires non-empty samples.")

        self.samples = samples
        self.max_seq_len = int(max_seq_len)
        self.transaction_cost = float(transaction_cost_bps) / 10000.0
        self.reward_scale = float(reward_scale)
        self.numbars_channels = int(max(1, numbars_channels))
        self.vpin_channels = int(max(1, vpin_channels))
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start

        self._processed = [self._process_sample(s) for s in self.samples]
        self._n_steps = len(self._processed)

        # Infer dimensions from first sample
        first = self._processed[0]
        self._tech_shape = first["tech_window"].shape
        self._nb_shape = first["numbars_recent"].shape
        self._vr_shape = first["vpin_raster_recent"].shape
        self._seq_shape = first["seq_vpin"].shape
        self._ae_shape = first["ae_input"].shape

        self.max_ticker_id = max(int(p["ticker_id"]) for p in self._processed) + 1
        self.max_class_id = max(int(p["asset_class_id"]) for p in self._processed) + 1
        self.max_subclass_id = max(int(p["asset_subclass_id"]) for p in self._processed) + 1

        self.observation_space = spaces.Dict({
            "tech_window": spaces.Box(-np.inf, np.inf, shape=self._tech_shape, dtype=np.float32),
            "numbars_recent": spaces.Box(-np.inf, np.inf, shape=self._nb_shape, dtype=np.float32),
            "vpin_raster_recent": spaces.Box(-np.inf, np.inf, shape=self._vr_shape, dtype=np.float32),
            "seq_vpin": spaces.Box(-np.inf, np.inf, shape=self._seq_shape, dtype=np.float32),
            "seq_vpin_len": spaces.Box(0, self.max_seq_len, shape=(1,), dtype=np.int32),
            "ae_input": spaces.Box(-np.inf, np.inf, shape=self._ae_shape, dtype=np.float32),
            "ticker_id": spaces.Discrete(max(self.max_ticker_id, 1)),
            "asset_class_id": spaces.Discrete(max(self.max_class_id, 1)),
            "asset_subclass_id": spaces.Discrete(max(self.max_subclass_id, 1)),
        })

        self.action_space = spaces.Discrete(3)

        self._idx = 0
        self._episode_steps = 0
        self._position_by_ticker = np.zeros(self.max_ticker_id, dtype=np.float32)

    def _normalize_numbars(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        c = self.numbars_channels

        if x.ndim == 2:
            if x.shape[-1] == c:       # (bins, C)
                x = x[None, :, :]      # (1, bins, C)
            elif x.shape[0] == c:      # (C, bins)
                x = x.T[None, :, :]    # (1, bins, C)
            else:
                raise ValueError(f"Unsupported numbars 2D shape={x.shape} with channels={c}")
        elif x.ndim == 3:
            if x.shape[-1] == c:       # (T, bins, C)
                pass
            elif x.shape[1] == c:      # (T, C, bins)
                x = np.transpose(x, (0, 2, 1))
            elif x.shape[0] == c:      # (C, T, bins)
                x = np.transpose(x, (1, 2, 0))
            else:
                raise ValueError(f"Unsupported numbars 3D shape={x.shape} with channels={c}")
        else:
            raise ValueError(f"numbars_recent must be 2D/3D, got shape={x.shape}")

        return x.astype(np.float32, copy=False)

    def _normalize_raster(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        c = self.vpin_channels

        if x.ndim == 2:
            if x.shape[0] == c:          # (C, bins)
                x = x[None, :, :]        # (1, C, bins)
            elif x.shape[-1] == c:       # (bins, C)
                x = np.transpose(x, (1, 0))[None, :, :]  # (1, C, bins)
            else:
                raise ValueError(f"Unsupported raster 2D shape={x.shape} with channels={c}")
        elif x.ndim == 3:
            if x.shape[1] == c:          # (T, C, bins)
                pass
            elif x.shape[-1] == c:       # (T, bins, C)
                x = np.transpose(x, (0, 2, 1))
            elif x.shape[0] == c:        # (C, T, bins)
                x = np.transpose(x, (1, 0, 2))
            else:
                raise ValueError(f"Unsupported raster 3D shape={x.shape} with channels={c}")
        else:
            raise ValueError(f"vpin_raster_recent must be 2D/3D, got shape={x.shape}")

        return x.astype(np.float32, copy=False)

    def _process_sample(self, sample: Dict) -> Dict:
        tech = np.asarray(sample["tech_features"], dtype=np.float32)
        nb = self._normalize_numbars(np.asarray(sample["numbars_recent"], dtype=np.float32))
        vr = self._normalize_raster(np.asarray(sample["vpin_raster_recent"], dtype=np.float32))
        ae = np.asarray(sample["ae_input"], dtype=np.float32)

        seq_raw = np.asarray(sample["seq_vpin"], dtype=np.float32)
        if seq_raw.ndim == 1:
            seq_raw = seq_raw[:, None]

        f_seq = int(seq_raw.shape[1]) if seq_raw.size > 0 else 1
        seq = np.zeros((self.max_seq_len, f_seq), dtype=np.float32)
        seq_len = int(min(len(seq_raw), self.max_seq_len))
        if seq_len > 0:
            seq[:seq_len, :] = seq_raw[:seq_len, :]

        return {
            "tech_window": tech,
            "numbars_recent": nb,
            "vpin_raster_recent": vr,
            "seq_vpin": seq,
            "seq_vpin_len": np.array([seq_len], dtype=np.int32),
            "ae_input": ae,
            "ticker_id": np.array(int(sample["ticker_id"]), dtype=np.int64),
            "asset_class_id": np.array(int(sample["asset_class_id"]), dtype=np.int64),
            "asset_subclass_id": np.array(int(sample["asset_subclass_id"]), dtype=np.int64),
            "target": float(sample["target"]),
            "ticker": sample.get("ticker"),
            "date": sample.get("date"),
        }

    def _get_obs(self, idx: int) -> Dict[str, np.ndarray]:
        x = self._processed[idx]
        return {
            "tech_window": x["tech_window"],
            "numbars_recent": x["numbars_recent"],
            "vpin_raster_recent": x["vpin_raster_recent"],
            "seq_vpin": x["seq_vpin"],
            "seq_vpin_len": x["seq_vpin_len"],
            "ae_input": x["ae_input"],
            "ticker_id": x["ticker_id"],
            "asset_class_id": x["asset_class_id"],
            "asset_subclass_id": x["asset_subclass_id"],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start and self._n_steps > 2:
            self._idx = int(self.np_random.integers(0, self._n_steps - 2))
        else:
            self._idx = 0

        self._episode_steps = 0
        self._position_by_ticker.fill(0.0)

        obs = self._get_obs(self._idx)
        info = {"idx": self._idx, "date": self._processed[self._idx].get("date")}
        return obs, info

    def step(self, action: int):
        action = int(action)
        target_pos = float(action - 1)  # 0->-1, 1->0, 2->+1

        cur = self._processed[self._idx]
        tid = int(cur["ticker_id"])
        prev_pos = float(self._position_by_ticker[tid])
        step_return = float(cur["target"])

        gross_pnl = prev_pos * step_return
        delta = abs(target_pos - prev_pos)
        cost = delta * self.transaction_cost
        reward = (gross_pnl - cost) * self.reward_scale

        # Update ticker-specific position state after reward realization.
        self._position_by_ticker[tid] = target_pos

        self._idx += 1
        self._episode_steps += 1

        terminated = self._idx >= self._n_steps - 1
        truncated = False
        if self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps:
            truncated = True

        next_idx = min(self._idx, self._n_steps - 1)
        obs = self._get_obs(next_idx)

        info = {
            "idx": next_idx,
            "ticker": cur.get("ticker"),
            "ticker_id": tid,
            "date": cur.get("date"),
            "position_prev": prev_pos,
            "position_target": target_pos,
            "gross_pnl": gross_pnl,
            "cost": cost,
            "step_return": step_return,
            "reward_unscaled": gross_pnl - cost,
        }

        return obs, float(reward), terminated, truncated, info


def build_v3_rl_envs(
    prep,
    tech_lookback: int = 64,
    seq_lookback_bars: int = 12,
    val_ratio: float = 0.2,
    val_cutoff_date: Optional[str] = None,
    sample_session: Optional[str] = None,
    sample_session_start: Optional[str] = None,
    sample_session_end: Optional[str] = None,
    stride: int = 1,
    transaction_cost_bps: float = 1.0,
    reward_scale: float = 100.0,
    max_episode_steps: Optional[int] = None,
) -> Tuple[V3ContinuousPPOEnv, V3ContinuousPPOEnv, Dict[str, object]]:
    """Build train/val V3 RL environments from V3ContinuousPrep."""
    all_samples = prep.build_samples(
        tech_lookback=tech_lookback,
        seq_lookback_bars=seq_lookback_bars,
        session_only=True,
        sample_session=sample_session,
        sample_session_start=sample_session_start,
        sample_session_end=sample_session_end,
        stride=stride,
    )

    if not all_samples:
        raise ValueError("No V3 samples produced for RL env construction.")

    all_dates = sorted(set(s["date"] for s in all_samples))
    if val_cutoff_date is not None:
        cutoff = pd.Timestamp(val_cutoff_date).date()
    else:
        n_val = max(1, int(len(all_dates) * val_ratio))
        cutoff = all_dates[-n_val]

    train_samples = [s for s in all_samples if s["date"] < cutoff]
    val_samples = [s for s in all_samples if s["date"] >= cutoff]

    if not train_samples or not val_samples:
        raise ValueError(
            f"Invalid split for RL envs: train={len(train_samples)}, val={len(val_samples)}, cutoff={cutoff}"
        )

    dims = prep.get_dims()
    env_kwargs = dict(
        max_seq_len=seq_lookback_bars,
        transaction_cost_bps=transaction_cost_bps,
        reward_scale=reward_scale,
        numbars_channels=dims.get("numbars_channels", 4),
        vpin_channels=dims.get("vpin_channels", 4),
        max_episode_steps=max_episode_steps,
    )

    train_env = V3ContinuousPPOEnv(
        samples=train_samples,
        random_start=True,
        **env_kwargs,
    )
    val_env = V3ContinuousPPOEnv(
        samples=val_samples,
        random_start=False,
        **env_kwargs,
    )

    info = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "cutoff_date": cutoff,
        "n_unique_dates": len(all_dates),
        "train_obs_shapes": {
            k: tuple(v.shape) for k, v in train_env.observation_space.spaces.items()
            if hasattr(v, "shape")
        },
        "n_tickers": train_env.max_ticker_id,
        "n_asset_classes": train_env.max_class_id,
        "n_asset_subclasses": train_env.max_subclass_id,
        "env_kwargs": env_kwargs,
    }

    return train_env, val_env, info
