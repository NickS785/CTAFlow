from __future__ import annotations

from datetime import time, timedelta
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .base_models import CTALight
from .volatility import RVForecast


class IntradayMomentumLight(CTALight):
    """LightGBM wrapper focused on intraday momentum and volatility features.

    The model is tailored to predict session-end behaviour (default 15:00 CST)
    and exposes convenience helpers for constructing intraday-aware training
    matrices.
    """

    def __init__(
            self,
            intraday_data: pd.DataFrame,
            session_end: time = time(15, 0),
            session_open: time = time(8, 30),
            closing_length: timedelta = timedelta(minutes=60),
            tz="America/Chicago",  # TZ is necessary to track timestamps accurately
            price_col="Close",
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.intraday_data = intraday_data
        self.price = intraday_data[price_col]
        self.training_data = None  # Daily df which contains the training variables
        self.session_end = session_end
        self.session_open = session_open
        self.closing_length = closing_length
        self.feature_names = []  # Track feature names for model training
        self.tz = tz

    def _add_feature(self, data: pd.Series, feature_name: str, tf='1d'):
        """Add a feature to the training dataset and track it.

        Parameters
        ----------
        data : pd.Series
            Feature data with datetime index
        feature_name : str
            Name for the feature column
        tf : str, default '1d'
            Timeframe: 'intraday' for intraday_data, '1d' for training_data
        """
        if tf == 'intraday':
            self.intraday_data[feature_name] = data
            if feature_name not in self.feature_names:
                self.feature_names.append(feature_name)
        else:
            if isinstance(self.training_data, pd.DataFrame):
                dates = self.training_data.index.normalize()
                common = set(dates).intersection(set(data.index.normalize()))
                self.training_data[feature_name] = data.loc[common]
                if feature_name not in self.feature_names:
                    self.feature_names.append(feature_name)
        return

    @staticmethod
    def _coerce_price(df: pd.DataFrame, price_col: str) -> pd.Series:
        if price_col not in df.columns:
            raise KeyError(f"Price column '{price_col}' not found")
        return df[price_col].copy()

    def opening_range_volatility(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
            period_length: Optional[timedelta] = None,
    ) -> pd.DataFrame:
        """Compute opening range realised volatility and semivariance.

        Uses vectorized operations for improved performance compared to iterative approach.

        Parameters
        ----------
        intraday_df : pd.DataFrame
            Intraday OHLCV data with a DatetimeIndex.
        price_col : str, default "Close"
            Column containing the price series used for returns.
        period_length : Optional[timedelta]
            Optional override for the opening window. Defaults to
            ``closing_length`` if not provided.

        Returns
        -------
        pd.DataFrame
            Daily features with columns: rv_open, rsv_pos_open, rsv_neg_open

        Notes
        -----
        Vectorized implementation avoids explicit loops over days, improving
        performance for large datasets. Uses pandas groupby aggregations.
        """

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for opening range calculations")

        window = period_length or self.closing_length
        prices = self._coerce_price(data, price_col)
        returns = prices.pct_change().dropna()

        # Vectorized approach: compute session times for all rows at once
        session_open_offset = pd.Timedelta(
            hours=self.session_open.hour,
            minutes=self.session_open.minute,
            seconds=self.session_open.second,
        )

        # Create working dataframe with date and time offsets
        work_df = pd.DataFrame({'returns': returns})
        work_df['date'] = work_df.index.normalize()
        work_df['session_start'] = work_df['date'] + session_open_offset
        work_df['session_end'] = work_df['session_start'] + window

        # Filter to opening range in one vectorized operation
        in_opening_range = (work_df.index >= work_df['session_start']) & (work_df.index < work_df['session_end'])
        opening_data = work_df[in_opening_range].copy()

        if opening_data.empty:
            return pd.DataFrame(columns=['rv_open', 'rsv_pos_open', 'rsv_neg_open'])

        # Prepare columns for aggregation
        opening_data['ret_squared'] = opening_data['returns'] ** 2
        opening_data['ret_pos_squared'] = np.where(opening_data['returns'] > 0, opening_data['ret_squared'], 0)
        opening_data['ret_neg_squared'] = np.where(opening_data['returns'] < 0, opening_data['ret_squared'], 0)
        opening_data['has_pos'] = opening_data['returns'] > 0
        opening_data['has_neg'] = opening_data['returns'] < 0

        # Vectorized aggregation by date
        agg_dict = {
            'ret_squared': 'sum',
            'ret_pos_squared': 'sum',
            'ret_neg_squared': 'sum',
            'has_pos': 'any',
            'has_neg': 'any'
        }
        grouped = opening_data.groupby('date').agg(agg_dict)

        # Compute volatility measures
        grouped['rv_open'] = np.sqrt(grouped['ret_squared'])
        grouped['rsv_pos_open'] = np.where(grouped['has_pos'], np.sqrt(grouped['ret_pos_squared']), np.nan)
        grouped['rsv_neg_open'] = np.where(grouped['has_neg'], np.sqrt(grouped['ret_neg_squared']), np.nan)

        # Return only the feature columns
        feature_df = grouped[['rv_open', 'rsv_pos_open', 'rsv_neg_open']]
        return feature_df

    def har_volatility_features(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            horizons: Sequence[int] = (1, 5, 22),
    ) -> pd.DataFrame:
        """Generate HAR-style realised volatility features for 1d ahead forecasts."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        forecaster = RVForecast(intraday_df=data)
        return forecaster.har_features(horizons=horizons)

    def target_time_returns(
            self,
            target_time: time,
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            price_col: str = "Close",
    ) -> pd.Series:
        """Return series for a specific target time window."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for target time returns")

        window = period_length or self.closing_length
        prices = self._coerce_price(data, price_col)
        ret = prices.pct_change().dropna()

        mask = ret.index.time == target_time
        if window:
            bars = max(int(window.total_seconds() // 60 // 5), 1)
            ret = ret.rolling(bars).sum()
        return ret[mask]

    def market_structure_features(
            self,
            daily_df: pd.DataFrame,
            lookbacks: Iterable[int] = (5, 10, 20),
            price_col: str = "Close",
    ) -> pd.DataFrame:
        """Build multi-horizon market structure context (high/low/VAH/VAL)."""

        prices = self._coerce_price(daily_df, price_col)
        feats: Dict[str, pd.Series] = {}
        for lb in lookbacks:
            feats[f"hh_{lb}"] = prices.rolling(lb).max()
            feats[f"ll_{lb}"] = prices.rolling(lb).min()
            feats[f"range_{lb}"] = feats[f"hh_{lb}"] - feats[f"ll_{lb}"]

        if {"High", "Low"}.issubset(daily_df.columns):
            vah = daily_df["High"].rolling(1).max()
            val = daily_df["Low"].rolling(1).min()
            feats["vah_prev"] = vah.shift(1)
            feats["val_prev"] = val.shift(1)

        return pd.DataFrame(feats)

    def microstructure_features(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
    ) -> pd.DataFrame:
        """Add microstructure proxies such as imbalance and realised range."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for microstructure features")

        prices = self._coerce_price(data, price_col)
        diffs = prices.diff().fillna(0)

        features = pd.DataFrame(index=data.index)
        features["signed_volume"] = np.sign(diffs) * data.get("Volume", 0)
        features["price_range"] = data.get("High", prices).fillna(prices) - data.get("Low", prices).fillna(prices)
        features["abs_ret"] = diffs.abs()
        return features

    def assemble_training_frame(
            self,
            daily_df: pd.DataFrame,
            target_times: Sequence[time],
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
    ) -> pd.DataFrame:
        """Convenience constructor for a full training feature matrix."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        har = self.har_volatility_features(data)
        market_struct = self.market_structure_features(daily_df, price_col=price_col)

        intraday_features = []
        for t in target_times:
            intraday_returns = self.target_time_returns(
                data,
                target_time=t,
                price_col=price_col,
            )
            intraday_features.append(intraday_returns.rename(f"ret_{t.strftime('%H%M')}"))

        intraday_frame = pd.concat(intraday_features, axis=1)
        combined = pd.concat([har, market_struct, intraday_frame], axis=1)
        return combined.dropna()
