from __future__ import annotations

from datetime import time, timedelta
from typing import Dict, Iterable, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from . import base_models
from .base_models import CTALight


class IntradayMomentumLight:
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
        self.feature_names = []  # Track feature names for model training
        self.tz = tz

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
                # Use pandas Index.intersection instead of set operations
                dates = self.training_data.index.normalize()
                data_dates = pd.DatetimeIndex(data.index).normalize()
                common = dates.intersection(data_dates)
                self.training_data[feature_name] = data.loc[common]
                if feature_name not in self.feature_names:
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
            target_time: time,
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            price_col: str = "Close",
            add_as_feature=False,
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
        target_returns = ret[mask]
        target_returns.index = pd.to_datetime(target_returns.index).normalize()

        if add_as_feature:
            feature_time_return = target_returns if target_time.hour < self.target_time.hour else target_returns.shift(1)
            self._add_feature(feature_time_return, f'{target_time.strftime("%H%M")}_return')

        return target_returns

    def target_time_volume(
            self,
            target_time: time,
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            volume_col: str = "Volume",
            add_as_feature: bool = False,
            aggregation: str = "sum",
    ) -> pd.Series:
        """Calculate volume at a specific target time over a given period.

        Parameters
        ----------
        target_time : time
            The target time to extract volume for (e.g., time(9, 30) for 9:30 AM)
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
        pd.Series
            Daily volume series aligned to the target time, indexed by date

        Notes
        -----
        - Aggregates volume over a rolling window ending at target_time
        - Properly lags the feature if target_time is before self.target_time
        - Useful for analyzing volume patterns at specific times (e.g., opening/closing volume)
        """
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

        if add_as_feature:
            # Lag if target_time is before the session target time to avoid lookahead bias
            feature_volume = target_volume if target_time.hour < self.target_time.hour else target_volume.shift(1)
            feature_name = f'{target_time.strftime("%H%M")}_volume_{aggregation}'
            self._add_feature(feature_volume, feature_name)

        return target_volume

    def bid_ask_volume_imbalance(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            bid_vol_col: str = "BidVolume",
            ask_vol_col: str = "AskVolume",
            add_as_feature: bool = True,
            use_proxy: bool = False,
    ) -> pd.DataFrame:
        """Calculate cumulative bid-ask volume imbalance over a period.

        Parameters
        ----------
        intraday_df : Optional[pd.DataFrame]
            Intraday data with DatetimeIndex. Uses self.intraday_data if None.
        period_length : Optional[timedelta]
            Length of period to aggregate imbalance. Uses self.closing_length if None.
        bid_vol_col : str, default "BidVolume"
            Column name for bid volume data
        ask_vol_col : str, default "AskVolume"
            Column name for ask volume data
        add_as_feature : bool, default True
            If True, add imbalance features to training_data
        use_proxy : bool, default False
            If True and bid/ask columns not found, use price change to proxy direction:
            - Volume on upticks attributed to ask (buyers)
            - Volume on downticks attributed to bid (sellers)

        Returns
        -------
        pd.DataFrame
            Daily features with columns:
            - ba_imbalance: cumulative (bid_vol - ask_vol) over period
            - ba_ratio: bid_vol / ask_vol ratio
            - ba_imbalance_norm: imbalance normalized by total volume

        Notes
        -----
        Bid-ask imbalance measures buying vs selling pressure:
        - Positive imbalance: more bid volume (selling pressure)
        - Negative imbalance: more ask volume (buying pressure)

        If bid/ask volume columns are not available and use_proxy=True, the method
        will classify volume by price movement direction.
        """
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

        # Vectorized approach: compute session times
        session_open_offset = pd.Timedelta(
            hours=self.session_open.hour,
            minutes=self.session_open.minute,
            seconds=self.session_open.second,
        )

        work_df = pd.DataFrame({
            'bid_vol': bid_volume,
            'ask_vol': ask_volume,
        })
        work_df['date'] = work_df.index.normalize()
        work_df['session_start'] = work_df['date'] + session_open_offset
        work_df['session_end'] = work_df['session_start'] + window

        # Filter to target period
        in_period = (work_df.index >= work_df['session_start']) & (work_df.index < work_df['session_end'])
        period_data = work_df[in_period].copy()

        if period_data.empty:
            return pd.DataFrame(columns=['ba_imbalance', 'ba_ratio', 'ba_imbalance_norm'])

        # Aggregate by date
        daily_agg = period_data.groupby('date').agg({
            'bid_vol': 'sum',
            'ask_vol': 'sum'
        })

        # Calculate imbalance metrics
        daily_agg['ba_imbalance'] = daily_agg['bid_vol'] - daily_agg['ask_vol']
        daily_agg['total_vol'] = daily_agg['bid_vol'] + daily_agg['ask_vol']
        daily_agg['ba_ratio'] = np.where(
            daily_agg['ask_vol'] > 0,
            daily_agg['bid_vol'] / daily_agg['ask_vol'],
            np.nan
        )
        daily_agg['ba_imbalance_norm'] = np.where(
            daily_agg['total_vol'] > 0,
            daily_agg['ba_imbalance'] / daily_agg['total_vol'],
            0
        )

        # Return only feature columns
        feature_df = daily_agg[['ba_imbalance', 'ba_ratio', 'ba_imbalance_norm']].copy()
        feature_df.index = pd.to_datetime(feature_df.index).normalize()

        # Check if we need to lag the features
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

    def assemble_training_frame(
            self,
            target_times : Sequence[str]= None,
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
    ) -> pd.DataFrame:
        """Convenience constructor for a full training feature matrix."""

        return combined.dropna()


