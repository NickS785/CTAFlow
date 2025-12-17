from __future__ import annotations

from datetime import time, timedelta
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from ..features.dt_features import build_datetime_features
from ..features.session_features import (
    cumulative_session_returns,
    cumulative_session_volatility,
)
from ..utils.session import DEFAULT_SESSION_TZ
from . import base_models
from .base_models import CTALight


class IntradayMomentum:
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
            session_target="close",
            supplementary_intraday_data: Optional[Mapping[str, pd.DataFrame]] = None,
            base_model: Union[str, Type[object], object] = CTALight,
            **kwargs,
    ) -> None:
        self.base_model = base_model
        self.model_kwargs = kwargs
        self.model: Optional[object] = None
        if not isinstance(base_model, (str, type)):
            self.model = base_model
            self.base_model = base_model.__class__
        self.intraday_data = intraday_data
        self.price = intraday_data[price_col]
        self.session_end = session_end
        self.session_open = session_open
        self.closing_length = closing_length
        self.supplementary_intraday_data = supplementary_intraday_data or {}
        self.feature_names = []  # Track feature names for model training
        self.tz = tz

        # Get valid trading dates from intraday data (excludes weekends/holidays)
        self.valid_trading_dates = self._get_valid_trading_dates(intraday_data)

        if session_target == "open":
            # Calculate target_time as session_open + closing_length
            # Use timedelta arithmetic to handle any period length correctly
            start_datetime = pd.Timestamp('1970-01-01') + pd.Timedelta(
                hours=session_open.hour,
                minutes=session_open.minute,
                seconds=session_open.second
            )
            end_datetime = start_datetime + closing_length
            self.target_time = end_datetime.time()
        else:
            self.target_time = session_end

        self.target_data = self.target_time_returns(
            self.target_time,
            intraday_data,
            period_length=closing_length,
        )
        self.training_data = pd.DataFrame(index=self.target_data.index)

        return

    def _resolve_model_class(self) -> Type[object]:
        if isinstance(self.base_model, str):
            try:
                return getattr(base_models, self.base_model)
            except AttributeError as exc:
                raise ValueError(f"Unknown base_model '{self.base_model}'") from exc
        if isinstance(self.base_model, type):
            return self.base_model
        raise TypeError("base_model must be a model class or the name of a base_models class")

    def _get_model(self) -> object:
        if self.model is None:
            model_cls = self._resolve_model_class()
            self.model = model_cls(**self.model_kwargs)
        return self.model

    @staticmethod
    def _get_valid_trading_dates(intraday_data: pd.DataFrame) -> pd.DatetimeIndex:
        """Extract valid trading dates from intraday data (excludes weekends/holidays).

        Parameters
        ----------
        intraday_data : pd.DataFrame
            Intraday OHLCV data with DatetimeIndex

        Returns
        -------
        pd.DatetimeIndex
            Normalized dates that have actual trading data
        """
        if not isinstance(intraday_data.index, pd.DatetimeIndex):
            intraday_data = intraday_data.copy()
            intraday_data.index = pd.to_datetime(intraday_data.index)

        # Get unique trading dates by normalizing and dropping duplicates
        valid_dates = pd.DatetimeIndex(intraday_data.index.normalize().unique()).sort_values()
        return valid_dates

    def _filter_to_trading_dates(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Filter a series or dataframe to only include valid trading dates.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Data with DatetimeIndex to filter

        Returns
        -------
        pd.Series or pd.DataFrame
            Filtered data containing only valid trading dates
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)

        # Normalize the data index
        normalized_idx = data.index.normalize()

        # Keep only dates that are in valid_trading_dates
        mask = normalized_idx.isin(self.valid_trading_dates)
        return data[mask]

    def _needs_shift(self, feature_time: time) -> bool:
        """Determine if a feature at feature_time needs to be shifted to avoid lookahead bias.

        Parameters
        ----------
        feature_time : time
            The time of day when the feature is measured

        Returns
        -------
        bool
            True if feature should be shifted (lagged by 1 day)

        Notes
        -----
        Shift is needed when feature_time is at or after target_time within the same trading session.
        For overnight sessions (session_open > session_end), handles midnight wrap-around correctly.
        """
        # Check if we're in an overnight session (crosses midnight)
        if self.session_open > self.session_end:
            # Overnight session (e.g., 18:00 open, 16:00 close next day)
            # Session timeline: [session_open, 23:59] -> midnight -> [00:00, session_end]

            # Determine if times are in session
            feature_in_session = (feature_time >= self.session_open) or (feature_time <= self.session_end)
            target_in_session = (self.target_time >= self.session_open) or (self.target_time <= self.session_end)

            # If feature is outside session hours, it's from previous trading day
            if not feature_in_session:
                return True

            # Both in session - determine temporal order accounting for midnight wrap
            feature_is_evening = feature_time >= self.session_open  # In [session_open, 23:59]
            target_is_evening = self.target_time >= self.session_open  # In [session_open, 23:59]

            if feature_is_evening and target_is_evening:
                # Both in evening part - simple comparison
                return feature_time >= self.target_time
            elif not feature_is_evening and not target_is_evening:
                # Both in morning part (next calendar day) - simple comparison
                return feature_time >= self.target_time
            elif feature_is_evening and not target_is_evening:
                # Feature is in evening (e.g., 20:00), target in morning (e.g., 16:00)
                # Feature comes BEFORE target in session timeline
                return False
            else:
                # Feature is in morning (e.g., 08:00), target in evening (e.g., 20:00)
                # Feature comes AFTER target in session timeline
                return True
        else:
            # Standard session (no midnight crossing)
            return feature_time >= self.target_time

    def _make_feature_name(
        self,
        base_name: str,
        feature_time: time,
        period_str: str = "",
        ticker: str = ""
    ) -> str:
        """Generate feature name with proper prefix for shifted features.

        Parameters
        ----------
        base_name : str
            Base descriptor (e.g., 'return', 'volume', 'ba_imbalance_norm')
        feature_time : time
            Time of day when feature is measured
        period_str : str, optional
            Period length string (e.g., '60min')
        ticker : str, optional
            Ticker symbol for cross-ticker features

        Returns
        -------
        str
            Feature name with 'pd_' prefix if shifted, otherwise standard format
        """
        time_str = feature_time.strftime("%H%M")
        needs_shift = self._needs_shift(feature_time)

        # Build components
        parts = []
        if ticker:
            parts.append(ticker)
        parts.append(time_str)
        if period_str:
            parts.append(period_str)
        parts.append(base_name)

        feature_name = '_'.join(parts)

        # Add prefix if shifted
        if needs_shift:
            feature_name = f"pd_{feature_name}"

        return feature_name

    @staticmethod
    def _filter_kwargs(method, kwargs: Dict) -> Dict:
        import inspect

        signature = inspect.signature(method)
        return {k: v for k, v in kwargs.items() if k in signature.parameters}

    def _normalize_predictions(self, predictions: object) -> np.ndarray:
        preds = np.asarray(predictions)
        if preds.ndim > 1:
            if preds.shape[1] == 1:
                return preds.ravel()
            if hasattr(self.model, "task") and getattr(self.model, "task", None) in {"binary_classification", "multiclass"}:
                return np.argmax(preds, axis=1)
        return preds


    @staticmethod
    def _format_period_length(period_length: Optional[timedelta]) -> str:
        """Format period_length as a string for feature names.

        Parameters
        ----------
        period_length : Optional[timedelta]
            Period length to format

        Returns
        -------
        str
            Formatted string like "60min", "30min", etc.
        """
        if period_length is None:
            return ""
        total_minutes = int(period_length.total_seconds() / 60)
        return f"{total_minutes}min"

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
        # Check for duplicates before adding
        if feature_name in self.feature_names:
            import warnings
            warnings.warn(f"Feature '{feature_name}' already exists, skipping duplicate", UserWarning)
            return

        # Convert boolean features to integers (0/1) for ML compatibility
        if data.dtype == bool:
            data = data.astype(int)

        if tf == 'intraday':
            self.intraday_data[feature_name] = data
            self.feature_names.append(feature_name)
        else:
            if isinstance(self.training_data, pd.DataFrame):
                # Use pandas Index.intersection instead of set operations
                dates = self.training_data.index.normalize()
                data_dates = pd.DatetimeIndex(data.index).normalize()
                common = dates.intersection(data_dates)
                self.training_data[feature_name] = data.loc[common]
                self.feature_names.append(feature_name)
        return

    @staticmethod
    def _coerce_price(df: pd.DataFrame, price_col: str) -> pd.Series:
        if price_col not in df.columns:
            raise KeyError(f"Price column '{price_col}' not found")
        return df[price_col].copy()

    def prev_hl(self, horizon=5, add_as_feature=True, normalize=False):
        # Resample to daily and drop NaN rows (weekends/holidays)
        daily_ohlc = self.intraday_data.resample('1d', offset=f"-{self.target_time.hour}h").agg({"Open":"first", "High":"max", "Low":"min", "Close":"last"}).dropna()

        h = daily_ohlc['High'].shift(1).rolling(horizon).max()
        l = daily_ohlc["Low"].shift(1).rolling(horizon).min()

        if normalize or add_as_feature:
            # Use daily_ohlc Close prices for perfect alignment with h and l
            prices = daily_ohlc['Close']

            h_norm = (h - prices)/prices
            l_norm = (prices - l)/prices
            if add_as_feature:
                self._add_feature(h_norm, f"{horizon}_high")
                self._add_feature(l_norm, feature_name=f"{horizon}_low")

            return h_norm, l_norm
        else:
            return h, l

    def opening_range_volatility(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
            period_length: Optional[timedelta] = None,
            add_as_feature: bool = True,
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

        # Return only the feature columns with normalized daily index
        feature_df = grouped[['rv_open', 'rsv_pos_open', 'rsv_neg_open']]
        feature_df.index = pd.to_datetime(feature_df.index).normalize()

        window_end_time = (
            pd.Timestamp("1970-01-01")
            + pd.Timedelta(
                hours=self.session_open.hour,
                minutes=self.session_open.minute,
                seconds=self.session_open.second,
            )
            + window
        ).time()
        if window_end_time > self.target_time:
            feature_df = feature_df.shift(1)

        if add_as_feature and isinstance(self.training_data, pd.DataFrame):
            for col in feature_df.columns:
                self._add_feature(feature_df[col], col, tf='1d')

        return feature_df

    def create_clf_target(
            self,
            n_classes: int = 2,
            lower_bound: float = 0.0,
            upper_bound: float = 0.0,
            add_as_feature: bool = False,
            feature_name: str = "target",
    ) -> pd.Series:
        """Create a classification target from session returns.

        Parameters
        ----------
        n_classes : int, default 2
            Number of classes to generate (2 or 3).
        lower_bound : float, default 0.0
            Lower threshold for the neutral band when ``n_classes=3``.
        upper_bound : float, default 0.0
            Upper threshold for the neutral band when ``n_classes=3``.
        add_as_feature : bool, default False
            When True, attach the generated target to ``training_data``.
        feature_name : str, default "target"
            Column name used when adding the target to ``training_data``.

        Returns
        -------
        pd.Series
            Classification labels aligned to ``target_data`` index.
        """
        if self.target_data is None or self.target_data.empty:
            raise ValueError("target_data is required to create classification targets")

        returns = self.target_data

        if n_classes == 2:
            labels = np.where(np.sign(returns) > 0, 1, 0)
        elif n_classes == 3:
            if upper_bound < lower_bound:
                raise ValueError("upper_bound must be greater than or equal to lower_bound")
            labels = np.where(returns >= upper_bound, 2, np.where(returns <= lower_bound, 0, 1))
        else:
            raise ValueError("n_classes must be either 2 or 3")

        target = pd.Series(labels, index=returns.index, name=feature_name)

        if add_as_feature and isinstance(self.training_data, pd.DataFrame):
            self._add_feature(target, feature_name, tf='1d')

        return target

    def har_volatility_features(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            horizons: Sequence[int] = (1, 5, 22),
            add_as_feature: bool = True,
    ) -> pd.DataFrame:
        """Generate HAR-style realised volatility features for 1d ahead forecasts."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for HAR volatility features")

        prices = self._coerce_price(data, "Close")
        returns = prices.pct_change().dropna()
        if returns.empty:
            return pd.DataFrame({f"rv_lag_{h}d": pd.Series(dtype=float) for h in horizons})

        rv = returns.pow(2).groupby(pd.Grouper(freq="1D")).sum()
        rv.index = rv.index.normalize()

        # Filter to only valid trading dates (removes weekends/holidays created by Grouper)
        rv = self._filter_to_trading_dates(rv)

        uses_post_target = (returns.index.time > self.target_time).any()

        features: Dict[str, pd.Series] = {}
        for horizon in horizons:
            feature_name = f"rv_lag_{horizon}d"
            feature_series = rv.rolling(horizon).mean()
            if uses_post_target:
                feature_series = feature_series.shift(1)
            features[feature_name] = feature_series
            if add_as_feature and isinstance(self.training_data, pd.DataFrame):
                self._add_feature(feature_series, feature_name, tf='1d')

        return pd.DataFrame(features)


    def target_time_returns(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            price_col: str = "Close",
            add_as_feature=False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Return series for a specific target time window.

        Parameters
        ----------
        target_time : time or List[time]
            Single time or list of times to extract returns for
        intraday_df : Optional[pd.DataFrame]
            Intraday data with DatetimeIndex
        period_length : Optional[timedelta]
            Rolling window length
        price_col : str
            Price column name
        add_as_feature : bool
            If True, add to training_data with proper lagging

        Returns
        -------
        pd.Series or pd.DataFrame
            If single time: pd.Series
            If list of times: pd.DataFrame with one column per time
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.target_time_returns(
                    t, intraday_df, period_length, price_col, add_as_feature
                )
                results[f'{t.strftime("%H%M")}'] = result
            return pd.DataFrame(results)

        # Single time handling
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
        target_returns = ret[mask]
        target_returns.index = pd.to_datetime(target_returns.index).normalize()

        # Filter to only valid trading dates (removes weekends/holidays)
        target_returns = self._filter_to_trading_dates(target_returns)

        if add_as_feature:
            # Shift if needed to avoid lookahead bias
            needs_shift = self._needs_shift(target_time)
            feature_time_return = target_returns.shift(1) if needs_shift else target_returns

            # Generate feature name with 'pd_' prefix if shifted
            period_str = self._format_period_length(window)
            feature_name = self._make_feature_name('return', target_time, period_str)
            self._add_feature(feature_time_return, feature_name)

        return target_returns

    def target_time_volume(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            volume_col: str = "Volume",
            add_as_feature: bool = False,
            aggregation: str = "sum",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate volume at a specific target time over a given period.

        Parameters
        ----------
        target_time : time or List[time]
            Single time or list of times to extract volume for
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex. Uses self.intraday_data if None.
        period_length : Optional[timedelta]
            Length of the period to aggregate volume over. Uses self.closing_length if None.
        volume_col : str, default "Volume"
            Column name containing volume data
        add_as_feature : bool, default False
            If True, add the volume series to training_data with proper lagging
        aggregation : str, default "sum"
            Aggregation method: "sum" for total volume, "mean" for average

        Returns
        -------
        pd.Series or pd.DataFrame
            If single time: pd.Series with daily volume
            If list of times: pd.DataFrame with one column per time

        Notes
        -----
        - Aggregates volume over a rolling window ending at target_time
        - Properly lags the feature if target_time is before self.target_time
        - Useful for analyzing volume patterns at specific times (e.g., opening/closing volume)
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.target_time_volume(
                    t, intraday_df, period_length, volume_col, add_as_feature, aggregation
                )
                results[f'{t.strftime("%H%M")}'] = result
            return pd.DataFrame(results)

        # Single time handling
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for target time volume")

        if volume_col not in data.columns:
            raise KeyError(f"Volume column '{volume_col}' not found in data")

        window = period_length or self.closing_length
        volume = data[volume_col].copy()

        # Calculate rolling aggregation over the specified window
        if window:
            bars = max(int(window.total_seconds() // 60 // 5), 1)
            if aggregation == "sum":
                volume = volume.rolling(bars).sum()
            elif aggregation == "mean":
                volume = volume.rolling(bars).mean()
            else:
                raise ValueError(f"Aggregation must be 'sum' or 'mean', got '{aggregation}'")

        # Extract volume at target time
        mask = volume.index.time == target_time
        target_volume = volume[mask]
        target_volume.index = pd.to_datetime(target_volume.index).normalize()

        # Filter to only valid trading dates (removes weekends/holidays)
        target_volume = self._filter_to_trading_dates(target_volume)

        if add_as_feature:
            # Shift if needed to avoid lookahead bias
            needs_shift = self._needs_shift(target_time)
            feature_volume = target_volume.shift(1) if needs_shift else target_volume

            # Generate feature name with 'pd_' prefix if shifted
            period_str = self._format_period_length(window)
            feature_name = self._make_feature_name(f'volume_{aggregation}', target_time, period_str)
            self._add_feature(feature_volume, feature_name)

        return target_volume

    def bid_ask_volume_imbalance(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            bid_vol_col: str = "BidVolume",
            ask_vol_col: str = "AskVolume",
            add_as_feature: bool = False,
            use_proxy: bool = False,
            return_components: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate bid-ask volume imbalance at a specific target time.

        Parameters
        ----------
        target_time : time or List[time]
            Single time or list of times to extract imbalance for
        intraday_df : Optional[pd.DataFrame]
            Intraday data with DatetimeIndex. Uses self.intraday_data if None.
        period_length : Optional[timedelta]
            Length of rolling window to aggregate imbalance. Uses self.closing_length if None.
        bid_vol_col : str, default "BidVolume"
            Column name for bid volume data
        ask_vol_col : str, default "AskVolume"
            Column name for ask volume data
        add_as_feature : bool, default False
            If True, add imbalance to training_data with proper lagging
        use_proxy : bool, default False
            If True and bid/ask columns not found, use price change to proxy direction:
            - Volume on upticks attributed to ask (buyers)
            - Volume on downticks attributed to bid (sellers)
        return_components : bool, default False
            If True, return DataFrame with imbalance, ratio, and normalized imbalance.
            If False, return Series with just the normalized imbalance.

        Returns
        -------
        pd.Series or pd.DataFrame
            If single time + return_components=False: Series with normalized imbalance
            If single time + return_components=True: DataFrame with all metrics
            If list of times: DataFrame with columns for each time

        Notes
        -----
        - Calculates imbalance over rolling window ending at target_time
        - Properly lags the feature if target_time is before self.target_time
        - Positive imbalance: more bid volume (selling pressure)
        - Negative imbalance: more ask volume (buying pressure)
        - Similar API to target_time_returns() for consistency
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.bid_ask_volume_imbalance(
                    t, intraday_df, period_length, bid_vol_col, ask_vol_col,
                    add_as_feature, use_proxy, return_components
                )
                # If return_components=False, we get a Series
                if isinstance(result, pd.Series):
                    results[f'{t.strftime("%H%M")}'] = result
                else:
                    # If return_components=True, we get a DataFrame with multiple columns
                    for col in result.columns:
                        results[f'{t.strftime("%H%M")}_{col}'] = result[col]
            return pd.DataFrame(results)

        # Single time handling
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for bid-ask volume imbalance")

        window = period_length or self.closing_length

        # Check if bid/ask columns exist, otherwise use proxy if enabled
        has_bid_ask = bid_vol_col in data.columns and ask_vol_col in data.columns

        if not has_bid_ask and not use_proxy:
            raise KeyError(
                f"Bid/ask volume columns '{bid_vol_col}' and '{ask_vol_col}' not found. "
                "Set use_proxy=True to estimate from price movements."
            )

        if has_bid_ask:
            bid_volume = data[bid_vol_col].copy()
            ask_volume = data[ask_vol_col].copy()
        else:
            # Proxy: classify volume by price direction
            if "Volume" not in data.columns or "Close" not in data.columns:
                raise KeyError("Volume and Close columns required for proxy estimation")

            price_change = data["Close"].diff()
            volume = data["Volume"]

            # Upticks = ask volume (buyers), downticks = bid volume (sellers)
            ask_volume = np.where(price_change > 0, volume, 0)
            bid_volume = np.where(price_change < 0, volume, 0)
            # Neutral ticks split evenly
            neutral_mask = price_change == 0
            ask_volume = np.where(neutral_mask, volume / 2, ask_volume)
            bid_volume = np.where(neutral_mask, volume / 2, bid_volume)

            ask_volume = pd.Series(ask_volume, index=data.index)
            bid_volume = pd.Series(bid_volume, index=data.index)

        # Calculate rolling imbalance over the window
        if window:
            bars = max(int(window.total_seconds() // 60 // 5), 1)
            bid_rolling = bid_volume.rolling(bars).sum()
            ask_rolling = ask_volume.rolling(bars).sum()
        else:
            bid_rolling = bid_volume
            ask_rolling = ask_volume

        # Calculate imbalance metrics
        imbalance = bid_rolling - ask_rolling
        total_vol = bid_rolling + ask_rolling

        # Handle ratio calculation: NaN when ask_rolling is NaN or 0
        ratio = bid_rolling / ask_rolling  # Will be NaN when ask_rolling is 0 or NaN

        # Handle normalized imbalance:
        # - Keep NaN when total_vol is NaN (from rolling window at start)
        # - Set to 0 when total_vol is exactly 0 (no volume)
        # - Calculate imbalance/total_vol when total_vol > 0
        imbalance_norm = np.where(
            pd.isna(total_vol),
            np.nan,  # Preserve NaN from rolling window
            np.where(total_vol > 0, imbalance / total_vol, 0)  # 0 when no volume
        )

        # Extract at target time
        mask = data.index.time == target_time

        if return_components:
            # Return all components as DataFrame
            # Create series first to preserve index
            imbalance_series = pd.Series(imbalance, index=data.index)[mask]
            ratio_series = pd.Series(ratio, index=data.index)[mask]
            imbalance_norm_series = pd.Series(imbalance_norm, index=data.index)[mask]

            components_df = pd.DataFrame({
                'ba_imbalance': imbalance_series,
                'ba_ratio': ratio_series,
                'ba_imbalance_norm': imbalance_norm_series
            })
            components_df.index = pd.to_datetime(components_df.index).normalize()

            # Filter to only valid trading dates (removes weekends/holidays)
            components_df = self._filter_to_trading_dates(components_df)

            if add_as_feature:
                # Shift if needed to avoid lookahead bias
                needs_shift = self._needs_shift(target_time)
                period_str = self._format_period_length(window)

                # Add each component with proper naming
                for col in components_df.columns:
                    feature_series = components_df[col].shift(1) if needs_shift else components_df[col]
                    feature_name = self._make_feature_name(col, target_time, period_str)
                    self._add_feature(feature_series, feature_name)

            return components_df
        else:
            # Return just normalized imbalance as Series (default, like target_time_returns)
            # Create series with proper index before masking
            imbalance_norm_series = pd.Series(imbalance_norm, index=data.index)
            target_imbalance = imbalance_norm_series[mask]
            target_imbalance.index = pd.to_datetime(target_imbalance.index).normalize()

            # Filter to only valid trading dates (removes weekends/holidays)
            target_imbalance = self._filter_to_trading_dates(target_imbalance)

            if add_as_feature:
                # Shift if needed to avoid lookahead bias
                needs_shift = self._needs_shift(target_time)
                feature_imbalance = target_imbalance.shift(1) if needs_shift else target_imbalance

                # Generate feature name with 'pd_' prefix if shifted
                period_str = self._format_period_length(window)
                feature_name = self._make_feature_name('ba_imbalance_norm', target_time, period_str)
                self._add_feature(feature_imbalance, feature_name)

            return target_imbalance

    def intraday_correlation(
            self,
            tickers: Union[str, List[str]],
            target_time: time,
            n_bars: int,
            resample: bool = False,
            resample_period: str = '1h',
            price_col: str = "Close",
            add_as_feature: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling correlation between main instrument and supplementary tickers.

        Computes the rolling correlation of returns between the main intraday_data
        and each ticker in supplementary_intraday_data, extracting the correlation
        at a specific target_time each day.

        Args:
            tickers: Single ticker symbol or list of ticker symbols from supplementary_intraday_data
            target_time: Time of day to extract correlation (e.g., time(14, 30))
            n_bars: Number of bars to use for rolling correlation window
            resample: If True, resample both series to resample_period before correlation
            resample_period: Pandas resample frequency string (e.g., '1h', '30min')
            price_col: Column name for prices (default: "Close")
            add_as_feature: If True, add correlation(s) to feature_names

        Returns:
            pd.Series: If single ticker, returns Series indexed by date with correlations
            pd.DataFrame: If multiple tickers, returns DataFrame with one column per ticker

        Raises:
            ValueError: If supplementary_intraday_data was not provided
            KeyError: If requested ticker not found in supplementary_intraday_data

        Example:
            >>> # Calculate 60-bar correlation with ES at 2:30 PM
            >>> corr = model.intraday_correlation('ES', time(14, 30), n_bars=60)
            >>>
            >>> # Multiple tickers with resampling to hourly
            >>> corr_df = model.intraday_correlation(
            ...     ['ES', 'NQ', 'GC'],
            ...     time(14, 30),
            ...     n_bars=4,  # 4 hourly bars
            ...     resample=True,
            ...     resample_period='1h'
            ... )
        """
        if not self.supplementary_intraday_data:
            raise ValueError(
                "supplementary_intraday_data must be provided to IntradayMomentumLight "
                "to use intraday_correlation()"
            )

        # Handle single ticker vs list of tickers
        if isinstance(tickers, str):
            tickers = [tickers]
            return_series = True
        else:
            return_series = False

        # Validate all tickers exist
        missing_tickers = [t for t in tickers if t not in self.supplementary_intraday_data]
        if missing_tickers:
            raise KeyError(
                f"Tickers {missing_tickers} not found in supplementary_intraday_data. "
                f"Available: {list(self.supplementary_intraday_data.keys())}"
            )

        # Get main instrument data
        if not isinstance(self.intraday_data.index, pd.DatetimeIndex):
            main_data = self.intraday_data.copy()
            main_data.index = pd.to_datetime(main_data.index)
        else:
            main_data = self.intraday_data

        # Calculate main returns
        main_prices = main_data[price_col]
        if resample:
            main_prices = main_prices.resample(resample_period).last().dropna()
        main_returns = main_prices.pct_change()

        # Dictionary to store correlation series for each ticker
        correlations = {}

        for ticker in tickers:
            # Get supplementary ticker data
            supp_data = self.supplementary_intraday_data[ticker]
            if not isinstance(supp_data.index, pd.DatetimeIndex):
                supp_data = supp_data.copy()
                supp_data.index = pd.to_datetime(supp_data.index)

            # Calculate supplementary returns
            supp_prices = supp_data[price_col]
            if resample:
                supp_prices = supp_prices.resample(resample_period).last().dropna()
            supp_returns = supp_prices.pct_change()

            # Align the two return series
            aligned_main, aligned_supp = main_returns.align(supp_returns, join='inner')

            # Calculate rolling correlation
            rolling_corr = aligned_main.rolling(window=n_bars).corr(aligned_supp)

            # Extract correlation at target_time for each day
            target_corr = []
            dates = []

            # Group by date
            for date, group in rolling_corr.groupby(rolling_corr.index.date):
                # Find the closest time to target_time on this date
                group_times = group.index.time
                target_idx = None

                # Find exact match or closest time at or after target_time
                for idx, t in enumerate(group_times):
                    if t >= target_time:
                        target_idx = idx
                        break

                # If no time >= target_time, use last time of day
                if target_idx is None:
                    target_idx = len(group) - 1

                target_corr.append(group.iloc[target_idx])
                dates.append(pd.Timestamp(date))

            # Create Series for this ticker
            corr_series = pd.Series(target_corr, index=pd.DatetimeIndex(dates))

            # Filter to only valid trading dates (removes weekends/holidays)
            corr_series = self._filter_to_trading_dates(corr_series)

            correlations[ticker] = corr_series

            # Add as feature if requested
            if add_as_feature:
                # Shift if needed to avoid lookahead bias
                needs_shift = self._needs_shift(target_time)
                feature_corr = corr_series.shift(1) if needs_shift else corr_series

                # Build feature name with proper prefix
                period_str = f'{n_bars}bar'
                if resample:
                    period_str = f'{n_bars}bar_{resample_period}'

                # Use standard naming with ticker
                base_name = f'corr_{ticker}'
                feature_name = self._make_feature_name(base_name, target_time, period_str)
                self._add_feature(feature_corr, feature_name)

        # Return Series if single ticker, DataFrame if multiple
        if return_series:
            return correlations[tickers[0]]
        else:
            return pd.DataFrame(correlations)

    def daily_correlation(
            self,
            ticker: str,
            length: int,
            price_col: str = "Close",
            add_as_feature: bool = False,
    ) -> pd.Series:
        """Calculate rolling daily correlation between main and supplementary ticker.

        Computes rolling correlation of daily returns between the main instrument
        and a supplementary ticker over a specified window. Automatically offsets
        by one day to prevent lookahead bias.

        Args:
            ticker: Ticker symbol from supplementary_intraday_data
            length: Rolling window length in days
            price_col: Column name for prices (default: "Close")
            add_as_feature: If True, add correlation to training_data

        Returns:
            pd.Series: Daily correlation values indexed by date, shifted by 1 day

        Raises:
            ValueError: If supplementary_intraday_data was not provided
            KeyError: If ticker not found in supplementary_intraday_data

        Example:
            >>> # Calculate 20-day rolling correlation with ES
            >>> corr = model.daily_correlation('ES', length=20)
        """
        if not self.supplementary_intraday_data:
            raise ValueError(
                "supplementary_intraday_data must be provided to IntradayMomentumLight "
                "to use daily_correlation()"
            )

        if ticker not in self.supplementary_intraday_data:
            raise KeyError(
                f"Ticker '{ticker}' not found in supplementary_intraday_data. "
                f"Available: {list(self.supplementary_intraday_data.keys())}"
            )

        # Get main instrument daily returns
        if not isinstance(self.intraday_data.index, pd.DatetimeIndex):
            main_data = self.intraday_data.copy()
            main_data.index = pd.to_datetime(main_data.index)
        else:
            main_data = self.intraday_data

        # Resample to daily using session close time
        main_daily = main_data[price_col].resample('1D').last().dropna()
        main_daily_returns = main_daily.pct_change()

        # Get supplementary ticker daily returns
        supp_data = self.supplementary_intraday_data[ticker]
        if not isinstance(supp_data.index, pd.DatetimeIndex):
            supp_data = supp_data.copy()
            supp_data.index = pd.to_datetime(supp_data.index)

        supp_daily = supp_data[price_col].resample('1D').last().dropna()
        supp_daily_returns = supp_daily.pct_change()

        # Align the two return series
        aligned_main, aligned_supp = main_daily_returns.align(supp_daily_returns, join='inner')

        # Calculate rolling correlation
        rolling_corr = aligned_main.rolling(window=length).corr(aligned_supp)

        # Offset by 1 day to prevent lookahead bias
        rolling_corr = rolling_corr.shift(1)

        if add_as_feature:
            feature_name = f'daily_{length}d_corr_{ticker}'
            self._add_feature(rolling_corr, feature_name)

        return rolling_corr

    def target_ticker_returns(
            self,
            ticker: str,
            target_time: Union[time, List[time]],
            period_length: Optional[timedelta] = None,
            price_col: str = "Close",
            add_as_feature: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate returns for a supplementary ticker at target time(s).

        Similar to target_time_returns() but uses supplementary_intraday_data.
        Automatically prevents lookahead bias by shifting returns if target_time
        is at or after the session target time.

        Args:
            ticker: Ticker symbol from supplementary_intraday_data
            target_time: Single time or list of times to extract returns for
            period_length: Rolling window length for returns calculation
            price_col: Column name for prices (default: "Close")
            add_as_feature: If True, add to training_data with proper lagging

        Returns:
            pd.Series: If single time, returns Series indexed by date
            pd.DataFrame: If list of times, returns DataFrame with one column per time

        Raises:
            ValueError: If supplementary_intraday_data was not provided
            KeyError: If ticker not found in supplementary_intraday_data

        Example:
            >>> # Get ES returns at 2:30 PM over 60 min window
            >>> es_ret = model.target_ticker_returns('ES', time(14, 30),
            ...                                       timedelta(minutes=60))
        """
        if not self.supplementary_intraday_data:
            raise ValueError(
                "supplementary_intraday_data must be provided to IntradayMomentumLight "
                "to use target_ticker_returns()"
            )

        if ticker not in self.supplementary_intraday_data:
            raise KeyError(
                f"Ticker '{ticker}' not found in supplementary_intraday_data. "
                f"Available: {list(self.supplementary_intraday_data.keys())}"
            )

        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.target_ticker_returns(
                    ticker, t, period_length, price_col, add_as_feature
                )
                results[f'{ticker}_{t.strftime("%H%M")}'] = result
            return pd.DataFrame(results)

        # Get supplementary ticker data
        supp_data = self.supplementary_intraday_data[ticker]
        if not isinstance(supp_data.index, pd.DatetimeIndex):
            supp_data = supp_data.copy()
            supp_data.index = pd.to_datetime(supp_data.index)

        if supp_data is None or supp_data.empty:
            raise ValueError(f"Supplementary data for ticker '{ticker}' is empty")

        window = period_length or self.closing_length
        prices = self._coerce_price(supp_data, price_col)
        ret = prices.pct_change().dropna()

        # Extract returns at target_time
        mask = ret.index.time == target_time
        if window:
            bars = max(int(window.total_seconds() // 60 // 5), 1)
            ret = ret.rolling(bars).sum()
        target_returns = ret[mask]
        target_returns.index = pd.to_datetime(target_returns.index).normalize()

        # Filter to only valid trading dates (removes weekends/holidays)
        target_returns = self._filter_to_trading_dates(target_returns)

        if add_as_feature:
            # Shift if needed to avoid lookahead bias
            needs_shift = self._needs_shift(target_time)
            feature_ticker_return = target_returns.shift(1) if needs_shift else target_returns

            # Generate feature name with 'pd_' prefix if shifted, include ticker
            period_str = self._format_period_length(window)
            feature_name = self._make_feature_name('return', target_time, period_str, ticker=ticker)
            self._add_feature(feature_ticker_return, feature_name)

        return target_returns

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

    def fit(self, X, y, **kwargs):
        """Fit the configured base model with compatible parameters."""
        model = self._get_model()
        fit_kwargs = self._filter_kwargs(model.fit, kwargs)
        return model.fit(X, y, **fit_kwargs)

    def predict(self, X, **kwargs):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        predict_kwargs = self._filter_kwargs(self.model.predict, kwargs)
        predictions = self.model.predict(X, **predict_kwargs)
        return self._normalize_predictions(predictions)

    def evaluate(self, X_test, y_test, **kwargs):
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        if not hasattr(self.model, "evaluate"):
            raise AttributeError(f"Model {self.model.__class__.__name__} does not support evaluation")
        eval_kwargs = self._filter_kwargs(self.model.evaluate, kwargs)
        return self.model.evaluate(X_test, y_test, **eval_kwargs)

    def fit_with_grid_search(self, X, y, **kwargs):
        model = self._get_model()
        if hasattr(model, "fit_with_grid_search"):
            fit_grid_kwargs = self._filter_kwargs(model.fit_with_grid_search, kwargs)
            return model.fit_with_grid_search(X, y, **fit_grid_kwargs)
        if hasattr(model, "grid_search"):
            grid_kwargs = self._filter_kwargs(model.grid_search, kwargs)
            return model.grid_search(X, y, **grid_kwargs)
        raise AttributeError(f"Model {model.__class__.__name__} does not support grid search")

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

    def add_daily_momentum_features(
            self,
            daily_df: pd.DataFrame,
            lookbacks: Sequence[int] = (1, 5, 10, 20),
            price_col: str = "Close",
    ) -> pd.DataFrame:
        """Add lagged daily momentum/returns features.

        Calculates cumulative log returns over various lookback periods and lags
        them by 1 day to avoid lookahead bias. Features are named as momentum_{d}d
        where d is the lookback period in days.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Daily OHLCV data with DatetimeIndex
        lookbacks : Sequence[int], default (1, 5, 10, 20)
            Lookback periods in days for momentum calculation
        price_col : str, default "Close"
            Column containing the price series

        Returns
        -------
        pd.DataFrame
            DataFrame with momentum features named as momentum_{d}d

        Notes
        -----
        All features are lagged by 1 day to avoid lookahead bias - they do not
        include the target date's (present day) return. For example, momentum_5d
        on day T represents the cumulative return from day T-6 to day T-1.

        Examples
        --------
        >>> model = IntradayMomentumLight(intraday_data, ...)
        >>> momentum_feats = model.add_daily_momentum_features(daily_df, lookbacks=[5, 10, 20])
        >>> # momentum_5d contains returns from T-6 to T-1 (5 days, lagged by 1)
        """
        prices = self._coerce_price(daily_df, price_col)
        feats: Dict[str, pd.Series] = {}

        for lb in lookbacks:
            # Calculate total log return over lookback period
            momentum = np.log(prices / prices.shift(lb))
            # Lag by 1 to avoid using current day's return
            momentum_lagged = momentum.shift(1)
            feature_name = f"momentum_{lb}d"
            feats[feature_name] = momentum_lagged

            # Add to training data if it exists
            if isinstance(self.training_data, pd.DataFrame):
                self._add_feature(momentum_lagged, feature_name, tf='1d')

        return pd.DataFrame(feats)

    def add_basic_datetime_features(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
        add_day_of_week: bool = True,
        add_month: bool = True,
        add_quarter: bool = True,
    ) -> pd.DataFrame:
        """Attach simple calendar breakdowns (DOW, month, quarter)."""

        idx = dates if dates is not None else self.training_data.index
        normalized = pd.DatetimeIndex(idx).normalize()
        feats: Dict[str, pd.Series] = {}

        if add_day_of_week:
            feats["day_of_week"] = normalized.dayofweek
        if add_month:
            feats["month"] = normalized.month
        if add_quarter:
            feats["quarter"] = normalized.quarter

        feature_df = pd.DataFrame(feats, index=normalized)
        for name, series in feature_df.items():
            self._add_feature(series, name)
        return feature_df

    def add_calendar_datetime_features(
        self,
        ticker: str,
        include_last_trading_week: bool = True,
        include_opex_week: bool = True,
        include_days_since_opex: bool = True,
        include_expiration_week: bool = True,
        include_weeks_until_expiration: bool = True,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Add calendar-based features such as opex and expiry timing."""

        idx = dates if dates is not None else self.training_data.index
        cal_feats = build_datetime_features(idx, ticker)

        keep_cols = []
        if include_last_trading_week:
            keep_cols.append("is_last_trading_week")
        if include_opex_week:
            keep_cols.append("is_opex_week")
        if include_days_since_opex:
            keep_cols.append("days_since_opex")
        if include_expiration_week:
            keep_cols.append("is_expiration_week")
        if include_weeks_until_expiration:
            keep_cols.append("weeks_until_expiration")

        feature_df = cal_feats[keep_cols].copy()

        # Convert boolean columns to integers (0/1) for ML compatibility
        for col in feature_df.columns:
            if feature_df[col].dtype == bool:
                feature_df[col] = feature_df[col].astype(int)

        for name, series in feature_df.items():
            self._add_feature(series, name)
        return feature_df

    def add_session_features(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        price_col: str = "Close",
        return_periods: Sequence[int] = (1, 5, 10),
        volatility_periods: Sequence[int] = (1, 5, 10),
        add_returns: bool = True,
        add_volatility: bool = True,
        session_start: Optional[time] = None,
        session_end: Optional[time] = None,
        tz: str = DEFAULT_SESSION_TZ,
    ) -> pd.DataFrame:
        """Create rolling session return and volatility features."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for session features")

        start = session_start or self.session_open
        end = session_end or self.session_end

        features: Dict[str, pd.Series] = {}
        if add_returns:
            ret_df = cumulative_session_returns(
                intraday_df=data,
                n_periods=return_periods,
                price_col=price_col,
                session_start=start,
                session_end=end,
                tz=tz,
            )
            features.update(ret_df.to_dict(orient="series"))

        if add_volatility:
            vol_df = cumulative_session_volatility(
                intraday_df=data,
                n_periods=volatility_periods,
                price_col=price_col,
                session_start=start,
                session_end=end,
                tz=tz,
            )
            features.update(vol_df.to_dict(orient="series"))

        for name, series in features.items():
            self._add_feature(series, name)

        return pd.DataFrame(features)

    def assemble_training_frame(
            self,
            target_times : Sequence[str]= None,
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
    ) -> pd.DataFrame:
        """Convenience constructor for a full training feature matrix."""

        return combined.dropna()


