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
