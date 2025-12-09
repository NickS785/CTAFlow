"""Intraday LightGBM model focused on session-end momentum."""

from datetime import timedelta, time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from CTAFlow.models.base_models import CTALight


class IntradayMomentumLight(CTALight):
    """LightGBM variant tailored to intraday momentum and volatility patterns."""

    def __init__(
        self,
        *,
        session_end: time = time(15, 0),
        closing_length: timedelta = timedelta(minutes=30),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_end = session_end
        self.closing_length = closing_length

    def _group_sessions(self, intraday_df: pd.DataFrame) -> Iterable:
        return intraday_df.groupby(intraday_df.index.normalize())

    def opening_range_volatility(
        self,
        intraday_df: pd.DataFrame,
        price_col: str = 'Close',
        period_length: Optional[timedelta] = None,
    ) -> pd.DataFrame:
        """Calculate opening window realized volatility and semivariance."""

        window = period_length or self.closing_length
        bars = max(int(window.total_seconds() // 60), 1)
        records: Dict[pd.Timestamp, Dict[str, float]] = {}

        for session_date, frame in self._group_sessions(intraday_df):
            session_slice = frame.sort_index().iloc[:bars]
            if session_slice.empty or price_col not in session_slice.columns:
                continue

            log_ret = np.log(session_slice[price_col]).diff().dropna()
            if log_ret.empty:
                continue

            records[session_date] = {
                'open_rv': float((log_ret ** 2).sum()),
                'open_rsv_pos': float((log_ret[log_ret >= 0] ** 2).sum()),
                'open_rsv_neg': float((log_ret[log_ret < 0] ** 2).sum()),
            }

        return pd.DataFrame.from_dict(records, orient='index')

    def target_time_return_features(
        self,
        intraday_df: pd.DataFrame,
        target_time: time,
        price_col: str = 'Close',
        period_length: Optional[timedelta] = None,
    ) -> pd.DataFrame:
        """Aggregate returns around a target time for training."""

        mask = intraday_df.index.time == target_time
        returns = pd.Series(dtype=float)

        if period_length:
            window = max(int(period_length.total_seconds() // 60), 1)
            grouped = self._group_sessions(intraday_df)
            for session_date, frame in grouped:
                frame = frame.sort_index()
                centered = frame.loc[frame.index.time == target_time]
                if centered.empty or price_col not in centered.columns:
                    continue
                anchor = centered.index[0]
                start = anchor - timedelta(minutes=window)
                end = anchor + timedelta(minutes=window)
                window_frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
                log_ret = np.log(window_frame[price_col]).diff().dropna()
                if not log_ret.empty:
                    returns.loc[session_date] = float(log_ret.sum())
        else:
            if price_col in intraday_df.columns:
                log_ret = np.log(intraday_df[price_col]).diff().dropna()
                returns = log_ret[mask]
                returns.index = returns.index.normalize()

        return pd.DataFrame({'target_return': returns})

    def har_volatility_features(
        self,
        daily_close: pd.Series,
        target_horizon: int = 1,
    ) -> pd.DataFrame:
        """HAR-style volatility regressors from prior daily closes."""

        log_ret = np.log(daily_close).diff().dropna()
        realized_var = log_ret.pow(2)

        features = pd.DataFrame(index=realized_var.index)
        features['har_vol_1d'] = realized_var.shift(1)
        features['har_vol_5d'] = realized_var.rolling(5).mean().shift(1)
        features['har_vol_22d'] = realized_var.rolling(22).mean().shift(1)
        features['target_volatility'] = realized_var.shift(-target_horizon)

        return features.dropna(how='all')

    def closing_window_return(
        self,
        intraday_df: pd.DataFrame,
        price_col: str = 'Close',
        period_length: Optional[timedelta] = None,
    ) -> pd.Series:
        """Return over the final portion of the trading session."""

        window = period_length or self.closing_length
        bars = max(int(window.total_seconds() // 60), 1)
        returns = pd.Series(dtype=float)

        for session_date, frame in self._group_sessions(intraday_df):
            session_slice = frame[frame.index.time <= self.session_end].sort_index()
            tail = session_slice.tail(bars)
            if tail.empty or price_col not in tail.columns:
                continue

            log_ret = np.log(tail[price_col]).diff().dropna()
            if not log_ret.empty:
                returns.loc[session_date] = float(log_ret.sum())

        return returns

    def market_structure_features(
        self,
        daily_df: pd.DataFrame,
        lookbacks: Sequence[int] = (5, 10, 20),
        vah_col: str = 'VAH',
        val_col: str = 'VAL',
    ) -> pd.DataFrame:
        """Rolling market structure and microstructure style features."""

        features = pd.DataFrame(index=daily_df.index)
        if 'High' in daily_df.columns:
            for lb in lookbacks:
                features[f'high_{lb}d'] = daily_df['High'].rolling(lb).max()
        if 'Low' in daily_df.columns:
            for lb in lookbacks:
                features[f'low_{lb}d'] = daily_df['Low'].rolling(lb).min()

        if vah_col in daily_df.columns:
            features['prev_vah'] = daily_df[vah_col].shift(1)
        if val_col in daily_df.columns:
            features['prev_val'] = daily_df[val_col].shift(1)

        if 'Close' in daily_df.columns and 'Open' in daily_df.columns:
            features['session_range'] = daily_df['High'] - daily_df['Low'] if 'High' in daily_df and 'Low' in daily_df else np.nan
            features['close_to_open'] = np.log(daily_df['Close']) - np.log(daily_df['Open'])

        return features.dropna(how='all')

    def build_training_frame(
        self,
        intraday_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        target_time: time,
        price_col: str = 'Close',
        period_length: Optional[timedelta] = None,
    ) -> pd.DataFrame:
        """Assemble a feature frame combining intraday and daily structure."""

        frames = [
            self.opening_range_volatility(intraday_df, price_col=price_col, period_length=period_length),
            self.target_time_return_features(intraday_df, target_time, price_col=price_col, period_length=period_length),
        ]

        closing_returns = self.closing_window_return(intraday_df, price_col=price_col, period_length=period_length)
        frames.append(pd.DataFrame({'closing_return': closing_returns}))
        frames.append(self.market_structure_features(daily_df))

        if price_col in daily_df.columns:
            frames.append(self.har_volatility_features(daily_df[price_col]))

        feature_frame = pd.concat(frames, axis=1)
        return feature_frame.dropna(how='all')
