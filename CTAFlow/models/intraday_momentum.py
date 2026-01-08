from __future__ import annotations

import datetime
from datetime import time, timedelta
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from ..features.dt_features import build_datetime_features
from ..features.session_features import (
    cumulative_session_returns,
    cumulative_session_volatility,
)
from .feature_selection import FeatureSelector, FeatureSelectionConfig
from ..utils.session import DEFAULT_SESSION_TZ
from . import base_models
from .base_models import CTALight
from ..features.curve.curve_features import CurveFeatures
from ..features import deseasonalize_volatility, deseasonalize_volume
from ..utils.tenor_interpolation import TenorInterpolator, create_tenor_grid
from ..data import read_exported_df


# noinspection PyDefaultArgument
class IntradayMomentum:
    """LightGBM wrapper focused on intraday momentum and volatility features.

    The model is tailored to predict session-end behaviour (default 15:00 CST)
    and exposes convenience helpers for constructing intraday-aware training
    matrices.

    Parameters
    ----------

    scale_target : bool, default False
        If True, scale target_data by volatility_scale (EWMA of realized volatility).
        This produces volatility-adjusted returns as the target, useful for normalizing
        returns by their expected volatility.
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
            vol_scale_ewm_halflife=21,
            vol_scale_ewm_span=63,
            scale_target: bool = False,
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
        self.diurnal_factor = None

        # Cache for deseasonalized calculations to avoid redundant computation
        self._deseas_cache = {}

        # Get valid trading dates from intraday data (excludes weekends/holidays)
        # Must be set before calling _get_volatility_scale() which uses _filter_to_trading_dates()
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

        # Volatility scale for normalizing returns (calculated by _get_volatility_scale)
        # Must be set after target_data since _get_volatility_scale() needs to reindex to target_data
        self.volatility_scale = self._get_volatility_scale(halflife=vol_scale_ewm_halflife, lookback_days=vol_scale_ewm_span)

        # Apply volatility scaling to target if requested
        if scale_target:
            # Align volatility scale with target data
            common_idx = self.target_data.index.intersection(self.volatility_scale.index)
            if len(common_idx) > 0:
                self.target_data = self.target_data.loc[common_idx] / self.volatility_scale.loc[common_idx]
            else:
                raise ValueError("No common dates between target_data and volatility_scale")

        self.training_data = pd.DataFrame(index=self.target_data.index)

        return

    @classmethod
    def from_files(
            cls,
            intraday_path: str,
            features_path: str,
            target_path: Optional[str] = None,
            target_col: str = "target",
            **kwargs,
    ) -> "IntradayMomentum":
        """Load IntradayMomentum from file paths.

        Parameters
        ----------
        intraday_path : str
            Path to intraday data (CSV or Parquet with DatetimeIndex)
        features_path : str
            Path to pre-computed features (CSV or Parquet)
        target_path : str, optional
            Path to target data. If None, expects target_col in features file.
        target_col : str, default "target"
            Column name for target variable
        **kwargs
            Additional arguments passed to IntradayMomentum.__init__

        Returns
        -------
        IntradayMomentum
            Initialized model with loaded data and features
        """
        # Load intraday data
        if intraday_path.endswith('.parquet'):
            intraday_data = pd.read_parquet(intraday_path)
        else:
            intraday_data = pd.read_csv(intraday_path, parse_dates=True, index_col=0)

        if not isinstance(intraday_data.index, pd.DatetimeIndex):
            intraday_data.index = pd.to_datetime(intraday_data.index)

        # Load features
        if features_path.endswith('.parquet'):
            features_df = pd.read_parquet(features_path)
        else:
            features_df = pd.read_csv(features_path, parse_dates=True, index_col=0)

        if not isinstance(features_df.index, pd.DatetimeIndex):
            features_df.index = pd.to_datetime(features_df.index)

        # Load or extract target
        if target_path is not None:
            if target_path.endswith('.parquet'):
                target_df = pd.read_parquet(target_path)
            else:
                target_df = pd.read_csv(target_path, parse_dates=True, index_col=0)
            if target_col in target_df.columns:
                target_data = target_df[target_col]
            else:
                target_data = target_df.iloc[:, 0]
        elif target_col in features_df.columns:
            target_data = features_df[target_col]
            features_df = features_df.drop(columns=[target_col])
        else:
            target_data = None

        # Create instance
        instance = cls(intraday_data=intraday_data, **kwargs)

        # Override training_data and feature_names with loaded data
        instance.training_data = features_df.copy()
        instance.feature_names = list(features_df.columns)

        # Override target_data if provided
        if target_data is not None:
            instance.target_data = target_data

        return instance

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
                return feature_time > self.target_time
            elif not feature_is_evening and not target_is_evening:
                # Both in morning part (next calendar day) - simple comparison
                return feature_time > self.target_time
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

    def _get_target_time_mask(
            self,
            index: pd.DatetimeIndex,
            target_time: time,
            use_nearest: bool = True
    ) -> pd.Series:
        """Get mask for extracting data at target_time, with fallback to nearest available time.

        Parameters
        ----------
        index : pd.DatetimeIndex
            DatetimeIndex to create mask for
        target_time : time
            Target time to extract data for
        use_nearest : bool, default True
            If True and exact time not found for a day, use nearest available time <= target_time.
            If False, use exact time matching only (may result in missing data).

        Returns
        -------
        pd.Series
            Boolean mask with same index, True for rows to extract

        Notes
        -----
        This method makes target_time methods more robust to missing data by:
        1. First trying exact time match
        2. If use_nearest=True and a day has no exact match, finds the latest available
           time <= target_time for that day
        3. Returns a mask that selects one row per day (or none if no valid time found)

        Examples
        --------
        >>> # If target_time=14:00 but data only has 13:55, 14:05 for a day,
        >>> # will select 13:55 (nearest time <= 14:00)
        """
        if not use_nearest:
            # Simple exact matching (original behavior)
            return index.time == target_time

        # Try exact match first
        exact_mask = index.time == target_time

        # Create helper dataframe
        df = pd.DataFrame({
            'date': index.normalize(),
            'time': index.time
        }, index=index)

        # Check which days have exact matches
        exact_dates = set(df[exact_mask]['date'].unique())
        all_dates = set(df['date'].unique())
        missing_dates = all_dates - exact_dates

        if not missing_dates:
            # All days have exact time match
            return exact_mask

        # For missing dates, find nearest time <= target_time
        result_mask = pd.Series(False, index=index)
        result_mask[exact_mask] = True

        for missing_date in missing_dates:
            # Get data for this day
            day_mask = df['date'] == missing_date
            day_data = df[day_mask]

            # Find times <= target_time
            valid_times_mask = day_data['time'] <= target_time

            if valid_times_mask.any():
                # Get the latest valid time
                latest_time = day_data[valid_times_mask]['time'].max()
                # Find the row(s) with this time on this date
                nearest_mask = (df['date'] == missing_date) & (df['time'] == latest_time)
                # Take only the first occurrence if multiple bars at same time
                nearest_idx = df[nearest_mask].index[0]
                result_mask.loc[nearest_idx] = True

        return result_mask

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
            if hasattr(self.model, "task") and getattr(self.model, "task", None) in {"binary_classification",
                                                                                     "multiclass"}:
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

    def _get_volatility_scale(
            self,
            lookback_days: int = 63,
            halflife: Optional[int] = 21,
            min_periods: int = 20,
            epsilon: float = 1e-8,
            use_intraday: bool = False,
    ) -> pd.Series:
        """Calculate volatility scale for normalizing returns.

        Uses the formula: scale = exp(EWMA(log(σ_d))) + ε
        where σ_d is daily realized volatility.

        This scale can be used to normalize returns: z_t = r_t / scale_t

        Parameters
        ----------
        lookback_days : int, default 252
            Number of days to use for volatility calculation
        halflife : int, optional, default 60
            Halflife for EWMA in days. If None, uses simple moving average
        min_periods : int, default 20
            Minimum periods required for EWMA calculation
        epsilon : float, default 1e-8
            Small constant added to scale to avoid division by zero
        use_intraday : bool, default False
            If True, calculate volatility from intraday returns.
            If False, use daily close-to-close returns (faster, standard approach)

        Returns
        -------
        pd.Series
            Volatility scale indexed by date, to be stored in self.volatility_scale

        Notes
        -----
        - Scale is calculated using only past data to avoid lookahead bias
        - Returns are scaled as: scaled_return = return / volatility_scale
        - Higher volatility periods get downweighted (returns divided by higher scale)
        - Lower volatility periods get upweighted (returns divided by lower scale)

        Example
        -------
        >>> model = IntradayMomentum(...)
        >>> model._get_volatility_scale(lookback_days=252, halflife=60)
        >>> # Now use self.volatility_scale to normalize returns
        """
        if use_intraday:
            # Calculate realized volatility from intraday returns
            prices = self._coerce_price(self.intraday_data, "Close")
            returns = prices.pct_change().dropna()

            # Group by date and calculate daily realized volatility (std of intraday returns)
            daily_vol = returns.groupby(returns.index.normalize()).std()
        else:
            # Calculate daily close-to-close volatility (standard approach)
            # Get daily closes
            prices = self._coerce_price(self.intraday_data, "Close")
            daily_prices = prices.resample('D').last().dropna()
            daily_returns = daily_prices.pct_change().dropna()

            # Calculate rolling volatility
            daily_vol = daily_returns.abs()  # Use absolute returns as proxy for volatility

        # Apply the formula: scale = exp(EWMA(log(σ_d))) + ε
        # Take log of volatility (add small constant to handle zeros)
        log_vol = np.log(daily_vol + epsilon)

        # Apply EWMA or SMA to log volatility
        if halflife is not None:
            # EWMA with specified halflife
            ewma_log_vol = log_vol.ewm(halflife=halflife, min_periods=min_periods).mean()
        else:
            # Simple moving average
            ewma_log_vol = log_vol.rolling(lookback_days, min_periods=min_periods).mean()

        # Exponentiate back and add epsilon
        volatility_scale = np.exp(ewma_log_vol) + epsilon

        # Filter to trading dates and align with target_data index
        volatility_scale = self._filter_to_trading_dates(volatility_scale)
        volatility_scale = volatility_scale.reindex(self.target_data.index)

        # Forward fill any missing values (in case of gaps)
        volatility_scale = volatility_scale.fillna(method='ffill')

        # Store in instance variable
        self.volatility_scale = volatility_scale

        return volatility_scale

    def _broadcast_volatility_scale_to_intraday(
            self,
            intraday_index: pd.DatetimeIndex,
            smooth_method: str = 'none',
            smooth_window: Optional[int] = None,
    ) -> pd.Series:
        """Broadcast daily volatility scale to intraday frequency with optional smoothing.

        Parameters
        ----------
        intraday_index : pd.DatetimeIndex
            Intraday timestamps to broadcast the scale to
        smooth_method : str, default 'none'
            Smoothing method to apply:
            - 'none': Use same daily value for all bars in a day (step function)
            - 'interpolate': Linear interpolation between days
            - 'rolling': Rolling average of intraday realized volatility
            - 'ewma': EWMA of intraday realized volatility
        smooth_window : int, optional
            Window size for 'rolling' or halflife for 'ewma' (in number of bars)

        Returns
        -------
        pd.Series
            Volatility scale at intraday frequency, aligned with intraday_index

        Notes
        -----
        - 'none': Fastest, same value throughout the day (creates steps at day boundaries)
        - 'interpolate': Smooth transition between days
        - 'rolling'/'ewma': Adapts to intraday volatility changes (slower, more responsive)

        Example
        -------
        >>> model._get_volatility_scale()  # Calculate daily scale first
        >>> intraday_scale = model._broadcast_volatility_scale_to_intraday(
        ...     model.intraday_data.index,
        ...     smooth_method='interpolate'
        ... )
        """
        if self.volatility_scale is None:
            raise ValueError(
                "volatility_scale is not calculated. Call _get_volatility_scale() first."
            )

        if not isinstance(intraday_index, pd.DatetimeIndex):
            intraday_index = pd.to_datetime(intraday_index)

        if smooth_method == 'none':
            # Simple broadcast: map each intraday bar to its date's volatility scale
            dates = intraday_index.normalize()
            intraday_scale = self.volatility_scale.reindex(dates)
            intraday_scale.index = intraday_index
            return intraday_scale

        elif smooth_method == 'interpolate':
            # Linear interpolation between daily values
            # Create a series at daily frequency, then interpolate to intraday
            daily_scale = self.volatility_scale.copy()

            # Reindex to include all intraday timestamps
            combined_index = daily_scale.index.union(intraday_index)
            combined_scale = daily_scale.reindex(combined_index)

            # Interpolate missing values
            combined_scale = combined_scale.interpolate(method='time')

            # Select only intraday timestamps
            intraday_scale = combined_scale.reindex(intraday_index)
            return intraday_scale

        elif smooth_method in ['rolling', 'ewma']:
            # Calculate intraday realized volatility and smooth it
            prices = self._coerce_price(self.intraday_data, "Close")
            intraday_returns = prices.pct_change().abs()

            if smooth_method == 'rolling':
                if smooth_window is None:
                    smooth_window = 12  # Default: 1 hour for 5-min bars
                intraday_vol = intraday_returns.rolling(
                    smooth_window, min_periods=max(1, smooth_window // 4)
                ).mean()
            else:  # ewma
                if smooth_window is None:
                    smooth_window = 12  # Default halflife: 1 hour for 5-min bars
                intraday_vol = intraday_returns.ewm(
                    halflife=smooth_window, min_periods=max(1, smooth_window // 4)
                ).mean()

            # Apply log-exp transform to match daily scale methodology
            log_vol = np.log(intraday_vol + 1e-8)
            smoothed_log_vol = log_vol.ewm(halflife=smooth_window or 12).mean()
            intraday_scale = np.exp(smoothed_log_vol) + 1e-8

            # Align with intraday_index
            intraday_scale = intraday_scale.reindex(intraday_index)
            return intraday_scale

        else:
            raise ValueError(
                f"Unknown smooth_method: {smooth_method}. "
                f"Choose from: 'none', 'interpolate', 'rolling', 'ewma'"
            )

    def vol_scaled_deseasonalized_returns(
            self,
            period_length: timedelta = timedelta(minutes=5),
            rolling_days: int = 252,
            deseasonalize: bool = True,
            vol_scale: bool = True,
            smooth_method: str = 'interpolate',
            smooth_window: Optional[int] = None,
            return_components: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate returns that are both volatility-scaled and deseasonalized.

        This method combines two normalization techniques:
        1. Volatility scaling: Normalize by recent volatility level
        2. Deseasonalization: Remove time-of-day patterns

        The result is returns that are normalized for both volatility regime and
        intraday seasonal patterns.

        Parameters
        ----------
        period_length : timedelta, default 5 minutes
            Bar frequency for return calculation
        rolling_days : int, default 252
            Lookback for deseasonalization
        deseasonalize : bool, default True
            If True, remove seasonal time-of-day patterns
        vol_scale : bool, default True
            If True, scale by volatility
        smooth_method : str, default 'interpolate'
            Method for broadcasting daily vol scale to intraday:
            'none', 'interpolate', 'rolling', 'ewma'
        smooth_window : int, optional
            Window for smoothing (if using 'rolling' or 'ewma')
        return_components : bool, default False
            If True, return DataFrame with raw returns, scaled, and deseasonalized

        Returns
        -------
        pd.Series or pd.DataFrame
            If return_components=False: Series of normalized returns
            If return_components=True: DataFrame with columns:
                - 'raw_returns': Original returns
                - 'vol_scaled': Returns scaled by volatility
                - 'deseasonalized': Deseasonalized returns
                - 'both': Both vol-scaled and deseasonalized

        Notes
        -----
        Order of operations:
        1. Calculate raw returns at period_length frequency
        2. Apply volatility scaling (if vol_scale=True)
        3. Apply deseasonalization (if deseasonalize=True)

        Example
        -------
        >>> model = IntradayMomentum(...)
        >>> model._get_volatility_scale(halflife=60)
        >>>
        >>> # Get fully normalized returns
        >>> normalized = model.vol_scaled_deseasonalized_returns(
        ...     period_length=timedelta(minutes=5),
        ...     smooth_method='interpolate'
        ... )
        >>>
        >>> # Or see all components
        >>> components = model.vol_scaled_deseasonalized_returns(
        ...     return_components=True
        ... )
        """
        # Calculate raw returns
        freq_minutes = int(period_length.total_seconds() / 60)
        resample_rule = f"{freq_minutes}min"

        prices = self._coerce_price(self.intraday_data, "Close")
        resampled_prices = prices.resample(
            resample_rule, closed='right', label='right'
        ).last()
        returns = resampled_prices.pct_change().dropna()

        result = pd.DataFrame({'raw_returns': returns})

        # Apply volatility scaling if requested
        if vol_scale:
            if self.volatility_scale is None:
                raise ValueError(
                    "volatility_scale not calculated. Call _get_volatility_scale() first."
                )

            # Broadcast daily scale to intraday
            intraday_scale = self._broadcast_volatility_scale_to_intraday(
                returns.index,
                smooth_method=smooth_method,
                smooth_window=smooth_window,
            )

            vol_scaled = returns / intraday_scale
            result['vol_scaled'] = vol_scaled
        else:
            vol_scaled = returns
            result['vol_scaled'] = vol_scaled

        # Apply deseasonalization if requested
        if deseasonalize:
            from CTAFlow.features.volume_seasonality import deseasonalize_volume

            # Build intraday index for deseasonalization
            intraday_idx = pd.Series(returns.index).groupby(
                returns.index.normalize()
            ).cumcount()
            intraday_idx.index = returns.index

            # Deseasonalize the volatility (absolute returns)
            volatility = returns.abs()
            deseas_result = deseasonalize_volume(
                volatility,
                intraday_idx=intraday_idx,
                rolling_days=rolling_days,
            )
            deseasonalized = deseas_result["adjusted"]

            # Restore sign (deseasonalization works on magnitudes)
            deseasonalized = deseasonalized * np.sign(returns)
            result['deseasonalized'] = deseasonalized

            # Apply both vol scaling and deseasonalization
            if vol_scale:
                both = vol_scaled.abs()
                deseas_both = deseasonalize_volume(
                    both,
                    intraday_idx=intraday_idx,
                    rolling_days=rolling_days,
                )["adjusted"]
                deseas_both = deseas_both * np.sign(vol_scaled)
                result['both'] = deseas_both
            else:
                result['both'] = deseasonalized
        else:
            result['deseasonalized'] = vol_scaled
            result['both'] = vol_scaled

        if return_components:
            return result
        else:
            return result['both']

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

    def _extract_at_time_daily(
            self,
            series: pd.Series,
            target_time: time,
            window_minutes: int = 5,
            adaptive_window: bool = True,
            max_window_minutes: int = 120,
            fallback_to_nearest: bool = True,
    ) -> pd.Series:
        """Extract values at target_time each day using a time window.

        More robust than exact time matching - finds bars within window_minutes of target_time.
        Groups by date and takes the closest bar to target_time per day.

        Parameters
        ----------
        series : pd.Series
            Intraday series with DatetimeIndex
        target_time : time
            Target time to extract each day
        window_minutes : int, default 5
            Initial time window in minutes (looks ± window_minutes around target_time)
        adaptive_window : bool, default True
            If True and no bars found within initial window, automatically expands
            the window up to max_window_minutes to find the nearest bar.
        max_window_minutes : int, default 120
            Maximum window size for adaptive expansion (2 hours default)
        fallback_to_nearest : bool, default True
            If True and no bars found even with expanded window, falls back to
            the nearest bar <= target_time for each day (prevents complete data loss)

        Returns
        -------
        pd.Series
            Daily series with normalized date index

        Notes
        -----
        The adaptive window handles data with different bar frequencies (5-min, 15-min,
        hourly, etc.) without manual configuration. The fallback ensures that even if
        the exact window has no data, we use the most recent available observation.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")

        # Detect data frequency for smarter window sizing
        if adaptive_window and len(series) > 1:
            time_diffs = series.index.to_series().diff().dropna()
            if not time_diffs.empty:
                median_freq_minutes = time_diffs.median().total_seconds() / 60
                # Ensure window is at least 1.5x the data frequency
                window_minutes = max(window_minutes, int(median_freq_minutes * 1.5))

        # Create target timestamp for comparison
        target_dt = pd.Timestamp('1970-01-01') + pd.Timedelta(
            hours=target_time.hour,
            minutes=target_time.minute,
            seconds=target_time.second
        )

        # Try initial window
        current_window = window_minutes
        series_in_window = pd.Series(dtype=float)

        while current_window <= max_window_minutes:
            lower = (target_dt - pd.Timedelta(minutes=current_window)).time()
            upper = (target_dt + pd.Timedelta(minutes=current_window)).time()

            # Handle midnight wrap-around for overnight sessions
            if lower > upper:
                mask = (series.index.time >= lower) | (series.index.time <= upper)
            else:
                mask = (series.index.time >= lower) & (series.index.time <= upper)

            series_in_window = series[mask]

            if not series_in_window.empty or not adaptive_window:
                break

            # Expand window and try again
            current_window = min(current_window * 2, max_window_minutes + 1)

        # If still empty and fallback enabled, use nearest bar <= target_time per day
        if series_in_window.empty and fallback_to_nearest:
            df = pd.DataFrame({'value': series})
            df['date'] = df.index.normalize()
            df['time'] = df.index.time

            results = []
            for date, group in df.groupby('date'):
                # Find bars <= target_time
                valid = group[group['time'] <= target_time]
                if not valid.empty:
                    # Take the latest bar <= target_time
                    results.append(valid.iloc[-1]['value'])
                else:
                    # No bars <= target_time, take earliest bar of the day as last resort
                    results.append(group.iloc[0]['value'])

            if results:
                dates = df.groupby('date').first().index
                return pd.Series(results, index=dates)

        if series_in_window.empty:
            import warnings
            warnings.warn(
                f"No data found within {max_window_minutes} minutes of {target_time}. "
                "Returning empty series - this will cause NaNs.",
                UserWarning
            )
            return pd.Series(dtype=float)

        # Group by date and take bar closest to target_time
        df = pd.DataFrame({'value': series_in_window})
        df['date'] = df.index.normalize()
        df['time_diff'] = abs(
            df.index.hour * 60 + df.index.minute -
            (target_time.hour * 60 + target_time.minute)
        )

        # For each date, take bar closest to target_time
        result = df.sort_values('time_diff').groupby('date')['value'].first()

        return result

    @staticmethod
    def _coerce_price(df: pd.DataFrame, price_col: str) -> pd.Series:
        if price_col not in df.columns:
            raise KeyError(f"Price column '{price_col}' not found")
        return df[price_col].copy()

    def _detect_bar_frequency_minutes(self, data: Optional[pd.DataFrame] = None) -> float:
        """Detect the median bar frequency in minutes from data.

        Parameters
        ----------
        data : Optional[pd.DataFrame]
            Data with DatetimeIndex. Uses self.intraday_data if None.

        Returns
        -------
        float
            Median bar frequency in minutes. Returns 5.0 as default if cannot detect.
        """
        df = data if data is not None else self.intraday_data
        if df is None or df.empty or len(df) < 2:
            return 5.0  # Default assumption

        if not isinstance(df.index, pd.DatetimeIndex):
            return 5.0

        time_diffs = df.index.to_series().diff().dropna()
        if time_diffs.empty:
            return 5.0

        median_minutes = time_diffs.median().total_seconds() / 60
        return max(1.0, median_minutes)  # At least 1 minute

    def _period_to_bars(
            self,
            period: timedelta,
            data: Optional[pd.DataFrame] = None,
            min_bars: int = 1,
    ) -> int:
        """Convert a timedelta period to number of bars based on detected data frequency.

        Parameters
        ----------
        period : timedelta
            Time period to convert
        data : Optional[pd.DataFrame]
            Data to detect frequency from. Uses self.intraday_data if None.
        min_bars : int, default 1
            Minimum number of bars to return

        Returns
        -------
        int
            Number of bars corresponding to the period
        """
        freq_minutes = self._detect_bar_frequency_minutes(data)
        period_minutes = period.total_seconds() / 60
        bars = int(period_minutes / freq_minutes)
        return max(min_bars, bars)

    def prev_hl(
            self,
            target_time: Optional[time] = None,
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
            add_as_feature: bool = False,
    ) -> pd.DataFrame:
        """Calculate distance from current price to previous session high/low.

        Parameters
        ----------
        target_time : Optional[time]
            Time of day to measure distance. Uses self.target_time if None.
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex. Uses self.intraday_data if None.
        price_col : str, default "Close"
            Column name for price data
        add_as_feature : bool, default False
            If True, add features to training_data with proper lagging

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'dist_prior_high' and 'dist_prior_low'

        Notes
        -----
        - Calculates (price - prior_high) / prior_high and (price - prior_low) / prior_low
        - Prior session = previous trading day's session high/low
        - Properly lags features if target_time is at or after self.target_time
        - Filtered to valid trading dates (removes weekends/holidays)

        Examples
        --------
        >>> model = IntradayMomentum(intraday_data, session_open=time(8, 30), session_end=time(15, 0))
        >>> dist_df = model.prev_hl(target_time=time(14, 0), add_as_feature=True)
        """
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for prev_hl")

        t_time = target_time or self.target_time

        # Get required columns
        if 'High' not in data.columns or 'Low' not in data.columns:
            raise KeyError("'High' and 'Low' columns required for prev_hl")

        prices = self._coerce_price(data, price_col)

        # Resample to daily to get session highs and lows
        daily_ohlc = data.resample('1D').agg({
            'High': 'max',
            'Low': 'min',
            price_col: 'last'
        }).dropna()

        # Filter to valid trading dates
        daily_ohlc.index = daily_ohlc.index.normalize()
        daily_ohlc = self._filter_to_trading_dates(daily_ohlc)

        # Get prior session high/low (shifted by 1 day)
        prior_high = daily_ohlc['High'].shift(1)
        prior_low = daily_ohlc['Low'].shift(1)
        prior_close = daily_ohlc[price_col].shift(1)

        # Get current price at target_time
        mask = data.index.time == t_time
        target_prices = prices[mask]
        target_prices.index = pd.to_datetime(target_prices.index).normalize()
        target_prices = self._filter_to_trading_dates(target_prices)

        # Calculate distances
        dist_high = (target_prices - prior_high) / prior_high
        dist_low = (target_prices - prior_low) / prior_low
        dist_close = (target_prices - prior_close) / prior_close

        result_df = pd.DataFrame({
            'dist_prior_high': dist_high,
            'dist_prior_low': dist_low,
            'dist_prior_close': dist_close
        })

        if add_as_feature:
            # Determine if we need to shift
            needs_shift = self._needs_shift(t_time)

            for col in result_df.columns:
                feature_series = result_df[col].shift(1) if needs_shift else result_df[col]
                feature_name = self._make_feature_name(col, t_time, "")
                self._add_feature(feature_series, feature_name)

        return result_df

    def vwap_distance(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            price_col: str = "Close",
            volume_col: str = "Volume",
            add_as_feature: bool = False,
            use_nearest: bool = True,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate distance from VWAP at target time(s).

        Computes volume-weighted average price from session open to target_time,
        then calculates (price - vwap) / vwap as a normalized distance metric.

        Parameters
        ----------
        target_time : time or List[time]
            Single time or list of times to calculate VWAP distance for
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex. Uses self.intraday_data if None.
        period_length : Optional[timedelta]
            Lookback window for VWAP calculation. If None, uses from session_open to target_time.
        price_col : str, default "Close"
            Column name for price data
        volume_col : str, default "Volume"
            Column name for volume data
        add_as_feature : bool, default False
            If True, add to training_data with proper lagging
        use_nearest : bool, default True
            If True, use nearest available time <= target_time when exact time not found.
            Prevents missing data from early closes or missing bars.

        Returns
        -------
        pd.Series or pd.DataFrame
            If single time: Series with VWAP distance
            If list of times: DataFrame with one column per time

        Notes
        -----
        - VWAP = sum(price * volume) / sum(volume) over the lookback period
        - Properly lags feature if target_time is at or after self.target_time
        - Filtered to valid trading dates (removes weekends/holidays)

        Examples
        --------
        >>> model = IntradayMomentum(intraday_data, session_open=time(8, 30))
        >>> # Distance from VWAP at 2:00 PM
        >>> vwap_dist = model.vwap_distance(time(14, 0), add_as_feature=True)
        >>> # Multiple times
        >>> vwap_dist_df = model.vwap_distance([time(10, 0), time(14, 0)])
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.vwap_distance(
                    t, intraday_df, period_length, price_col, volume_col, add_as_feature, use_nearest
                )
                results[f'{t.strftime("%H%M")}'] = result
            return pd.DataFrame(results)

        # Single time handling
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for VWAP distance")

        if volume_col not in data.columns:
            raise KeyError(f"Volume column '{volume_col}' not found in data")

        prices = self._coerce_price(data, price_col)
        volume = data[volume_col].copy()

        # Calculate typical price (for more accurate VWAP)
        if 'High' in data.columns and 'Low' in data.columns:
            typical_price = (data['High'] + data['Low'] + prices) / 3
        else:
            typical_price = prices

        # If period_length specified, use rolling VWAP
        if period_length:
            # Use detected bar frequency instead of hardcoded 5-minute assumption
            bars = self._period_to_bars(period_length, data)
            pv = (typical_price * volume).rolling(bars, min_periods=1).sum()
            v = volume.rolling(bars, min_periods=1).sum()
            vwap = pv / v.replace(0, np.nan)
        else:
            # Use session-based VWAP (from session_open to target_time)
            work_df = pd.DataFrame({
                'price': typical_price,
                'volume': volume,
                'close': prices
            })
            work_df['date'] = work_df.index.normalize()

            # Calculate session start time for each row
            session_open_offset = pd.Timedelta(
                hours=self.session_open.hour,
                minutes=self.session_open.minute,
                seconds=self.session_open.second,
            )
            work_df['session_start_time'] = work_df['date'] + session_open_offset

            # Filter to session open -> target_time window
            target_offset = pd.Timedelta(
                hours=target_time.hour,
                minutes=target_time.minute,
                seconds=target_time.second,
            )
            work_df['target_time_dt'] = work_df['date'] + target_offset

            # Keep only bars within session window
            in_window = (work_df.index >= work_df['session_start_time']) & (work_df.index <= work_df['target_time_dt'])
            window_data = work_df[in_window].copy()

            # Calculate cumulative VWAP within each session
            window_data['pv'] = window_data['price'] * window_data['volume']
            cum_pv = window_data.groupby('date')['pv'].cumsum()
            cum_vol = window_data.groupby('date')['volume'].cumsum()
            window_data['vwap'] = cum_pv / cum_vol.replace(0, np.nan)

            # Broadcast VWAP back to full dataframe
            vwap = pd.Series(index=data.index, dtype=float)
            vwap.loc[window_data.index] = window_data['vwap']

        # Extract VWAP at target_time
        mask = self._get_target_time_mask(data.index, target_time, use_nearest)
        target_vwap = vwap[mask]
        target_prices_at_time = prices[mask]

        # Calculate distance: (price - vwap) / vwap
        vwap_distance = (target_prices_at_time - target_vwap) / target_vwap
        vwap_distance.index = pd.to_datetime(vwap_distance.index).normalize()

        # Filter to valid trading dates
        vwap_distance = self._filter_to_trading_dates(vwap_distance)

        if add_as_feature:
            # Shift if needed to avoid lookahead bias
            needs_shift = self._needs_shift(target_time)
            feature_vwap_dist = vwap_distance.shift(1) if needs_shift else vwap_distance

            # Generate feature name with 'pd_' prefix if shifted
            period_str = self._format_period_length(period_length) if period_length else "session"
            feature_name = self._make_feature_name('dist_vwap', target_time, period_str)
            self._add_feature(feature_vwap_dist, feature_name)

        return vwap_distance

    def overnight_returns(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            price_col: str = "Close",
            add_as_feature: bool = False,
    ) -> pd.Series:
        """Calculate overnight returns from previous session close to current session open.

        Computes the return from the previous trading day's session close to the
        current trading day's session open, representing the overnight gap.

        Parameters
        ----------
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex. Uses self.intraday_data if None.
        price_col : str, default "Close"
            Column name for price data
        add_as_feature : bool, default False
            If True, add to training_data with proper lagging

        Returns
        -------
        pd.Series
            Daily overnight returns indexed by date

        Notes
        -----
        - Overnight return = (session_open[t] - session_close[t-1]) / session_close[t-1]
        - Properly lags feature if session_open is at or after target_time
        - Filtered to only valid trading dates (removes weekends/holidays)

        Examples
        --------
        >>> model = IntradayMomentum(intraday_data, session_open=time(8, 30), session_end=time(15, 0))
        >>> overnight_ret = model.overnight_returns(add_as_feature=True)
        """
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for overnight returns")

        prices = self._coerce_price(data, price_col)

        # Extract session close prices (last price at or before session_end each day)
        session_end_offset = pd.Timedelta(
            hours=self.session_end.hour,
            minutes=self.session_end.minute,
            seconds=self.session_end.second,
        )

        # Extract session open prices (first price at or after session_open each day)
        session_open_offset = pd.Timedelta(
            hours=self.session_open.hour,
            minutes=self.session_open.minute,
            seconds=self.session_open.second,
        )

        # Create working dataframe
        work_df = pd.DataFrame({'price': prices})
        work_df['date'] = work_df.index.normalize()

        # Get close prices (at session_end)
        work_df['session_end_time'] = work_df['date'] + session_end_offset
        close_mask = work_df.index.time == self.session_end
        session_closes = work_df[close_mask].set_index('date')['price']

        # Get open prices (at session_open)
        work_df['session_open_time'] = work_df['date'] + session_open_offset
        open_mask = work_df.index.time == self.session_open
        session_opens = work_df[open_mask].set_index('date')['price']

        # Calculate overnight returns: (open[t] - close[t-1]) / close[t-1]
        prev_closes = session_closes.shift(1)
        overnight_ret = (session_opens - prev_closes) / prev_closes

        # Normalize index
        overnight_ret.index = pd.to_datetime(overnight_ret.index).normalize()

        # Filter to only valid trading dates (removes weekends/holidays)
        overnight_ret = self._filter_to_trading_dates(overnight_ret)

        if add_as_feature:
            # Determine if we need to shift based on session_open time
            needs_shift = self._needs_shift(self.session_open)
            feature_ret = overnight_ret.shift(1) if needs_shift else overnight_ret

            # Generate feature name with proper prefix
            feature_name = self._make_feature_name('overnight_return', self.session_open, "")
            self._add_feature(feature_ret, feature_name)

        return overnight_ret

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

    def gk_vol(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            lookbacks: Sequence[int] = (1, 5, 10, 20),
            add_as_feature: bool = False,
    ) -> pd.DataFrame:
        """Calculate Garman-Klass volatility at daily level.

        The Garman-Klass volatility estimator is more efficient than close-to-close
        volatility as it uses OHLC data. It provides better estimates with fewer
        observations.

        Parameters
        ----------
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex
        lookbacks : Sequence[int], default (1, 5, 10, 20)
            Lookback periods in days for rolling GK volatility
        add_as_feature : bool, default False
            If True, add features to training_data with proper lagging

        Returns
        -------
        pd.DataFrame
            DataFrame with GK volatility columns for each lookback

        Notes
        -----
        - GK estimator: 0.5 * log(H/L)^2 - (2*log(2) - 1) * log(C/O)^2
        - All features are lagged by 1 day to avoid lookahead bias
        - More efficient than realized volatility for intraday data
        - Annualized assuming 252 trading days

        Examples
        --------
        >>> model = IntradayMomentum(intraday_data)
        >>> gk_vol_df = model.gk_vol(lookbacks=[5, 20], add_as_feature=True)
        """
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for GK volatility")

        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns for GK volatility: {missing_cols}")

        # Resample to daily OHLC
        daily_ohlc = data.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        daily_ohlc.index = daily_ohlc.index.normalize()
        daily_ohlc = self._filter_to_trading_dates(daily_ohlc)

        # Calculate log ratios
        log_hl = np.log(daily_ohlc['High'] / daily_ohlc['Low'])
        log_co = np.log(daily_ohlc['Close'] / daily_ohlc['Open'])

        # GK estimator
        gk_term = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

        features: Dict[str, pd.Series] = {}
        for lookback in lookbacks:
            # Rolling average of GK term, then annualize
            gk_vol = np.sqrt(gk_term.rolling(lookback).mean() * 252)

            # Lag by 1 to avoid lookahead bias
            gk_vol_lagged = gk_vol.shift(1)

            feature_name = f"gk_vol_{lookback}d"
            features[feature_name] = gk_vol_lagged

            if add_as_feature and isinstance(self.training_data, pd.DataFrame):
                self._add_feature(gk_vol_lagged, feature_name, tf='1d')

        return pd.DataFrame(features)

    def intraday_range(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            lookback_days: int = 20,
            add_as_feature: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate current intraday range relative to historical average.

        Measures the current session's high-low range at target_time and normalizes
        it by the average range over the lookback period.

        Parameters
        ----------
        target_time : time or List[time]
            Time(s) to measure the current range
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex
        lookback_days : int, default 20
            Number of days to use for historical average range
        add_as_feature : bool, default False
            If True, add to training_data with proper lagging

        Returns
        -------
        pd.Series or pd.DataFrame
            If single time: Series with range ratio
            If list of times: DataFrame with one column per time

        Notes
        -----
        - Range ratio = current_range / avg_range
        - Values > 1 indicate above-average range (higher volatility)
        - Values < 1 indicate below-average range (lower volatility)
        - Properly lags if target_time is at or after self.target_time

        Examples
        --------
        >>> model = IntradayMomentum(intraday_data)
        >>> range_ratio = model.intraday_range(time(14, 0), lookback_days=20, add_as_feature=True)
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.intraday_range(t, intraday_df, lookback_days, add_as_feature)
                results[f'{t.strftime("%H%M")}'] = result
            return pd.DataFrame(results)

        # Single time handling
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for intraday range")

        if 'High' not in data.columns or 'Low' not in data.columns:
            raise KeyError("'High' and 'Low' columns required for intraday range")

        # Calculate cumulative session high/low up to target_time
        work_df = pd.DataFrame({
            'high': data['High'],
            'low': data['Low']
        })
        work_df['date'] = work_df.index.normalize()

        # Filter to bars up to target_time each day
        work_df['time'] = work_df.index.time
        mask = work_df['time'] <= target_time
        filtered_data = work_df[mask].copy()

        # Get session high/low up to target_time
        session_stats = filtered_data.groupby('date').agg({
            'high': 'max',
            'low': 'min'
        })
        session_stats['range'] = session_stats['high'] - session_stats['low']

        # Filter to valid trading dates
        session_stats.index = pd.to_datetime(session_stats.index).normalize()
        session_stats = self._filter_to_trading_dates(session_stats)

        # Calculate rolling average range
        avg_range = session_stats['range'].rolling(lookback_days).mean()

        # Range ratio
        range_ratio = session_stats['range'] / avg_range

        if add_as_feature:
            # Shift if needed to avoid lookahead bias
            needs_shift = self._needs_shift(target_time)
            feature_range = range_ratio.shift(1) if needs_shift else range_ratio

            # Generate feature name
            feature_name = self._make_feature_name(f'range_ratio_{lookback_days}d', target_time, "")
            self._add_feature(feature_range, feature_name)

        return range_ratio

    def vol_ratios(
            self,
            intraday_df: Optional[pd.DataFrame] = None,
            short_long_pairs: Sequence[Tuple[int, int]] = ((5, 20), (10, 60), (20, 120)),
            price_col: str = "Close",
            add_as_feature: bool = False,
    ) -> pd.DataFrame:
        """Calculate short-term vs long-term volatility ratios.

        Compares short-term realized volatility to long-term volatility to identify
        regime changes. High ratios indicate recent volatility expansion.

        Parameters
        ----------
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex
        short_long_pairs : Sequence[Tuple[int, int]], default ((5, 20), (10, 60), (20, 120))
            Pairs of (short_window, long_window) in days for volatility comparison
        price_col : str, default "Close"
            Column name for price data
        add_as_feature : bool, default False
            If True, add features to training_data

        Returns
        -------
        pd.DataFrame
            DataFrame with volatility ratio columns for each pair

        Notes
        -----
        - Ratio = short_term_vol / long_term_vol
        - Values > 1 indicate volatility expansion (recent vol higher than average)
        - Values < 1 indicate volatility contraction (recent vol lower than average)
        - All features lagged by 1 day to avoid lookahead bias
        - Uses close-to-close returns for simplicity

        Examples
        --------
        >>> model = IntradayMomentum(intraday_data)
        >>> vol_ratios_df = model.vol_ratios(short_long_pairs=[(5, 20), (10, 60)], add_as_feature=True)
        """
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for volatility ratios")

        prices = self._coerce_price(data, price_col)

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Resample to daily
        daily_returns = returns.groupby(pd.Grouper(freq="1D")).sum()
        daily_returns.index = daily_returns.index.normalize()
        daily_returns = self._filter_to_trading_dates(daily_returns)

        features: Dict[str, pd.Series] = {}
        for short_window, long_window in short_long_pairs:
            # Calculate rolling volatility for both windows
            short_vol = daily_returns.rolling(short_window).std()
            long_vol = daily_returns.rolling(long_window).std()

            # Ratio
            vol_ratio = short_vol / long_vol.replace(0, np.nan)

            # Lag by 1 to avoid lookahead bias
            vol_ratio_lagged = vol_ratio.shift(1)

            feature_name = f"vol_ratio_{short_window}_{long_window}d"
            features[feature_name] = vol_ratio_lagged

            if add_as_feature and isinstance(self.training_data, pd.DataFrame):
                self._add_feature(vol_ratio_lagged, feature_name, tf='1d')

        return pd.DataFrame(features)

    def normalized_range(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            lookback_days: int = 20,
            normalize_by: str = "historical",
            add_as_feature: bool = False,
            use_nearest: bool = True,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate normalized range over a specific period at target time.

        Calculates the high-low range over a rolling window ending at target_time,
        normalized by either historical average range or current price level.

        Parameters
        ----------
        target_time : time or List[time]
            Time(s) to extract range for
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex. Uses self.intraday_data if None.
        period_length : Optional[timedelta]
            Rolling window length for range calculation. Uses self.closing_length if None.
        lookback_days : int, default 20
            Number of days for historical average (only used if normalize_by='historical')
        normalize_by : str, default "historical"
            Normalization method:
            - "historical": Divide by average range over lookback_days
            - "price": Divide by current price (percentage range)
        add_as_feature : bool, default False
            If True, add to training_data with proper lagging
        use_nearest : bool, default True
            If True, use nearest available time <= target_time when exact time not found.
            Prevents missing data from early closes or missing bars.

        Returns
        -------
        pd.Series or pd.DataFrame
            If single time: Series with normalized range
            If list of times: DataFrame with one column per time

        Notes
        -----
        - Calculates range = (high - low) over the rolling window
        - normalize_by='historical': range / avg_historical_range
        - normalize_by='price': range / price (gives percentage range)
        - Properly lags if target_time is at or after self.target_time

        Examples
        --------
        >>> # 60-minute normalized range at 2:00 PM
        >>> norm_range = model.normalized_range(
        ...     time(14, 0),
        ...     period_length=timedelta(hours=1),
        ...     normalize_by='historical',
        ...     add_as_feature=True
        ... )
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.normalized_range(
                    t, intraday_df, period_length, lookback_days, normalize_by, add_as_feature, use_nearest
                )
                results[f'{t.strftime("%H%M")}'] = result
            return pd.DataFrame(results)

        # Single time handling
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for normalized range")

        if 'High' not in data.columns or 'Low' not in data.columns:
            raise KeyError("'High' and 'Low' columns required for normalized range")

        window = period_length or self.closing_length
        high = data['High'].copy()
        low = data['Low'].copy()

        # Calculate rolling range over the window
        if window:
            # Use detected bar frequency instead of hardcoded 5-minute assumption
            bars = self._period_to_bars(window, data)
            rolling_high = high.rolling(bars, min_periods=1).max()
            rolling_low = low.rolling(bars, min_periods=1).min()
        else:
            rolling_high = high
            rolling_low = low

        range_series = rolling_high - rolling_low

        # Extract at target time
        mask = self._get_target_time_mask(data.index, target_time, use_nearest)
        target_range = range_series[mask]

        # Handle case where mask returns no data
        if target_range.empty:
            import warnings
            warnings.warn(
                f"No data found at target_time {target_time}. Using fallback extraction.",
                UserWarning
            )
            # Use the improved _extract_at_time_daily which has adaptive window
            target_range = self._extract_at_time_daily(range_series, target_time)
        else:
            target_range.index = pd.to_datetime(target_range.index).normalize()

        # Filter to valid trading dates
        target_range = self._filter_to_trading_dates(target_range)

        # Normalize
        if normalize_by == "historical":
            # Calculate historical average range
            avg_range = target_range.rolling(lookback_days).mean()
            normalized = target_range / avg_range.replace(0, np.nan)
        elif normalize_by == "price":
            # Normalize by current price level
            if 'Close' not in data.columns:
                raise KeyError("'Close' column required for price normalization")
            close = data['Close'][mask]
            close.index = pd.to_datetime(close.index).normalize()
            close = self._filter_to_trading_dates(close)
            normalized = target_range / close.replace(0, np.nan)
        else:
            raise ValueError(f"normalize_by must be 'historical' or 'price', got '{normalize_by}'")

        if add_as_feature:
            # Shift if needed to avoid lookahead bias
            needs_shift = self._needs_shift(target_time)
            feature_series = normalized.shift(1) if needs_shift else normalized

            # Generate feature name
            period_str = self._format_period_length(window)
            norm_suffix = "hist" if normalize_by == "historical" else "pct"
            feature_name = self._make_feature_name(f'norm_range_{norm_suffix}', target_time, period_str)
            self._add_feature(feature_series, feature_name)

        return normalized

    def relative_volume(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            volume_col: str = "Volume",
            lookback_days: int = 20,
            aggregation: str = "sum",
            add_as_feature: bool = False,
            use_nearest: bool = True,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate volume relative to historical average at target time.

        Computes volume over a rolling window ending at target_time and normalizes
        by the historical average volume for the same time window.

        Parameters
        ----------
        target_time : time or List[time]
            Time(s) to extract volume for
        intraday_df : Optional[pd.DataFrame]
            Intraday OHLCV data with DatetimeIndex. Uses self.intraday_data if None.
        period_length : Optional[timedelta]
            Rolling window length for volume aggregation. Uses self.closing_length if None.
        volume_col : str, default "Volume"
            Column name for volume data
        lookback_days : int, default 20
            Number of days for historical average calculation
        aggregation : str, default "sum"
            Aggregation method: "sum" for total volume, "mean" for average
        add_as_feature : bool, default False
            If True, add to training_data with proper lagging
        use_nearest : bool, default True
            If True, use nearest available time <= target_time when exact time not found.
            Prevents missing data from early closes or missing bars.

        Returns
        -------
        pd.Series or pd.DataFrame
            If single time: Series with relative volume (current / historical avg)
            If list of times: DataFrame with one column per time

        Notes
        -----
        - relative_volume = current_volume / avg_historical_volume
        - Values > 1 indicate above-average volume
        - Values < 1 indicate below-average volume
        - Properly lags if target_time is at or after self.target_time
        - Useful for identifying unusual volume patterns

        Examples
        --------
        >>> # Opening 30-min volume relative to 20-day average
        >>> rel_vol = model.relative_volume(
        ...     time(9, 30),
        ...     period_length=timedelta(minutes=30),
        ...     lookback_days=20,
        ...     add_as_feature=True
        ... )
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            results = {}
            for t in target_time:
                result = self.relative_volume(
                    t, intraday_df, period_length, volume_col, lookback_days, aggregation, add_as_feature, use_nearest
                )
                results[f'{t.strftime("%H%M")}'] = result
            return pd.DataFrame(results)

        # Single time handling
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for relative volume")

        if volume_col not in data.columns:
            raise KeyError(f"Volume column '{volume_col}' not found in data")

        window = period_length or self.closing_length
        volume = data[volume_col].copy()

        # Calculate rolling aggregation over the window
        if window:
            # Use detected bar frequency instead of hardcoded 5-minute assumption
            bars = self._period_to_bars(window, data)
            if aggregation == "sum":
                volume_agg = volume.rolling(bars, min_periods=1).sum()
            elif aggregation == "mean":
                volume_agg = volume.rolling(bars, min_periods=1).mean()
            else:
                raise ValueError(f"Aggregation must be 'sum' or 'mean', got '{aggregation}'")
        else:
            volume_agg = volume

        # Extract at target time
        mask = self._get_target_time_mask(volume_agg.index, target_time, use_nearest)
        target_volume = volume_agg[mask]

        # Handle case where mask returns no data
        if target_volume.empty:
            target_volume = self._extract_at_time_daily(volume_agg, target_time)
        else:
            target_volume.index = pd.to_datetime(target_volume.index).normalize()

        # Filter to valid trading dates
        target_volume = self._filter_to_trading_dates(target_volume)

        # CRITICAL FIX: Reindex to continuous date range to include ALL calendar days
        # This ensures rolling windows are calendar-day based, not trading-day based
        if not target_volume.empty:
            # Create complete date range (including weekends/holidays)
            full_date_range = pd.date_range(
                start=target_volume.index.min(),
                end=target_volume.index.max(),
                freq='D'
            )

            # Reindex to full range, NaN for non-trading days
            target_volume_full = target_volume.reindex(full_date_range)

            # Calculate historical average on CALENDAR days (not trading days)
            # This makes lookback_days=20 actually mean 20 calendar days
            min_periods = max(1, lookback_days // 4)  # Require at least 25% of data
            avg_volume_full = target_volume_full.rolling(
                window=lookback_days,
                min_periods=min_periods
            ).mean()

            # Remove non-trading days from both series
            avg_volume = avg_volume_full.loc[target_volume.index]
        else:
            # Empty data case
            avg_volume = pd.Series(dtype=float, index=target_volume.index)

        # Handle zero/NaN volumes gracefully
        # Replace zeros in avg_volume with NaN to avoid division by zero
        avg_volume_safe = avg_volume.replace(0, np.nan)

        # Also handle NaN in target_volume (missing bars)
        # Fill missing target volumes with 0 (no volume = neutral signal)
        target_volume_safe = target_volume.fillna(0)

        # Calculate relative volume with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_vol = target_volume_safe / avg_volume_safe

        # Replace inf values with NaN
        relative_vol = relative_vol.replace([np.inf, -np.inf], np.nan)

        # Optional: For days where avg_volume is NaN (insufficient history),
        # we could use a global average as fallback (uncomment if needed)
        # if relative_vol.isna().any():
        #     global_avg = target_volume.mean()
        #     if global_avg > 0:
        #         relative_vol = relative_vol.fillna(target_volume / global_avg)

        if add_as_feature:
            # Shift if needed to avoid lookahead bias
            needs_shift = self._needs_shift(target_time)
            feature_series = relative_vol.shift(1) if needs_shift else relative_vol

            # Generate feature name
            period_str = self._format_period_length(window)
            feature_name = self._make_feature_name(f'rel_vol_{lookback_days}d', target_time, period_str)
            self._add_feature(feature_series, feature_name)

        return relative_vol

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
            use_nearest: bool = True,
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
        use_nearest : bool, default True
            If True, use nearest available time <= target_time when exact time not found.
            Prevents missing data from early closes or missing bars.

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
                    t, intraday_df, period_length, price_col, add_as_feature, use_nearest
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

        if window:
            # Use detected bar frequency instead of hardcoded 5-minute assumption
            bars = self._period_to_bars(window, data)
            ret = ret.rolling(bars, min_periods=1).sum()

        mask = self._get_target_time_mask(ret.index, target_time, use_nearest)
        target_returns = ret[mask]

        # Handle case where mask returns no data
        if target_returns.empty:
            target_returns = self._extract_at_time_daily(ret, target_time)
        else:
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
            use_nearest: bool = True,
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
        use_nearest : bool, default True
            If True, use nearest available time <= target_time when exact time not found.
            Prevents missing data from early closes or missing bars.

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
                    t, intraday_df, period_length, volume_col, add_as_feature, aggregation, use_nearest
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
            # Use detected bar frequency instead of hardcoded 5-minute assumption
            bars = self._period_to_bars(window, data)
            if aggregation == "sum":
                volume = volume.rolling(bars, min_periods=1).sum()
            elif aggregation == "mean":
                volume = volume.rolling(bars, min_periods=1).mean()
            else:
                raise ValueError(f"Aggregation must be 'sum' or 'mean', got '{aggregation}'")

        # Extract volume at target time
        mask = self._get_target_time_mask(volume.index, target_time, use_nearest)
        target_volume = volume[mask]

        # Handle case where mask returns no data
        if target_volume.empty:
            target_volume = self._extract_at_time_daily(volume, target_time)
        else:
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
            use_nearest: bool = True,
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
        use_nearest : bool, default True
            If True, use nearest available time <= target_time when exact time not found.
            Prevents missing data from early closes or missing bars.

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
                    add_as_feature, use_proxy, return_components, use_nearest
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
            # Use detected bar frequency instead of hardcoded 5-minute assumption
            bars = self._period_to_bars(window, data)
            bid_rolling = bid_volume.rolling(bars, min_periods=1).sum()
            ask_rolling = ask_volume.rolling(bars, min_periods=1).sum()
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
        mask = self._get_target_time_mask(data.index, target_time, use_nearest)

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
            # Use closed='right', label='right' to avoid lookahead bias
            main_prices = main_prices.resample(
                resample_period, closed='right', label='right'
            ).last().dropna()
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
                # Use closed='right', label='right' to avoid lookahead bias
                supp_prices = supp_prices.resample(
                    resample_period, closed='right', label='right'
                ).last().dropna()
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
            use_nearest: bool = True,
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
            use_nearest: If True, use nearest available time <= target_time when exact time not found.
                Prevents missing data from early closes or missing bars.

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
                    ticker, t, period_length, price_col, add_as_feature, use_nearest
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

        if window:
            # Use detected bar frequency instead of hardcoded 5-minute assumption
            bars = self._period_to_bars(window, supp_data)
            ret = ret.rolling(bars, min_periods=1).sum()

        # Extract returns at target_time
        mask = self._get_target_time_mask(ret.index, target_time, use_nearest)
        target_returns = ret[mask]

        # Handle case where mask returns no data
        if target_returns.empty:
            target_returns = self._extract_at_time_daily(ret, target_time)
        else:
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

    def curve_levels(self,
                     fwd_curve_df: pd.DataFrame,
                     target_time: Union[time, List[time]],
                     slope: bool = True,
                     slope_mos: Tuple[int, int] = (1, 4),
                     normalized_basis: bool = False,
                     basis_mos: Tuple[int, int] = (1, 3),
                     relative_basis: bool = True,
                     spread_pair_1: Tuple[int, int] = (1, 2),
                     spread_pair_2: Tuple[int, int] = (2, 3),
                     const_maturity: bool = False,
                     expiry_data: Optional[pd.Series] = None,
                     add_as_feature: bool = True) -> pd.DataFrame:
        """Extract curve shape features at specific intraday time(s).

        Measures the forward curve structure (slope, basis, relative basis) at target_time each day.
        Features are properly lagged to avoid lookahead bias.

        Parameters
        ----------
        fwd_curve_df : pd.DataFrame
            Continuous contract prices (M1, M2, M3, ...) with intraday DatetimeIndex
        target_time : time or List[time]
            Intraday time(s) to extract curve features
        slope : bool, default True
            Calculate curve slope between slope_mos contracts
        slope_mos : Tuple[int, int], default (1, 4)
            (front, back) contract months for slope calculation
        normalized_basis : bool, default False
            Calculate normalized spread (basis) between contracts.
            If relative_basis=False, uses spread_pair_1 for calculation.
            Formula: (M_back - M_front) / M_front
        basis_mos : Tuple[int, int], default (1, 3)
            DEPRECATED: Use spread_pair_1 instead. Kept for backward compatibility.
        relative_basis : bool, default True
            Calculate relative basis (annualized spread difference).
            Formula: (Spread1/dt1) - (Spread2/dt2) where dt = months between contracts
        spread_pair_1 : Tuple[int, int], default (1, 2)
            First spread pair (front, back) for relative basis. Also used for normalized_basis
            when relative_basis=False. E.g., (1, 2) means M2 - M1.
        spread_pair_2 : Tuple[int, int], default (2, 3)
            Second spread pair (front, back) for relative basis. E.g., (2, 3) means M3 - M2.
        const_maturity : bool, default False
            Use constant maturity tenor interpolation. If True, requires expiry_data.
        expiry_data : pd.Series, optional
            Mapping of contract names (M1, M2, ...) to expiry dates.
            Required if const_maturity=True. Can be obtained from SpreadEngine
            with return_expiries=True.
        add_as_feature : bool, default True
            If True, add features to training_data with proper lagging

        Returns
        -------
        pd.DataFrame
            DataFrame with curve level features for each target_time

        Notes
        -----
        - Features are lagged if target_time is at or after self.target_time
        - Slope measures average price change per month: (M_back - M_front) / (back - front)
        - Normalized basis: (M_back - M_front) / M_front (percentage spread)
        - Relative basis: (Spread1/dt1) - (Spread2/dt2) where Spread = M_back - M_front

        Examples
        --------
        >>> # Curve slope and relative basis at 10am and 2pm
        >>> curve_feats = model.curve_levels(
        ...     fwd_curve_df=continuous_contracts,
        ...     target_time=[time(10, 0), time(14, 0)],
        ...     slope=True,
        ...     relative_basis=True,
        ...     spread_pair_1=(1, 2),
        ...     spread_pair_2=(2, 3)
        ... )
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            all_features = []
            for t in target_time:
                feats = self.curve_levels(
                    fwd_curve_df, t, slope, slope_mos, normalized_basis, basis_mos,
                    relative_basis, spread_pair_1, spread_pair_2,
                    const_maturity, expiry_data, add_as_feature
                )
                all_features.append(feats)
            return pd.concat(all_features, axis=1)

        # Single time handling
        if const_maturity:
            # Use constant maturity interpolation
            if expiry_data is None:
                raise ValueError(
                    "expiry_data is required when const_maturity=True. "
                    "Obtain from SpreadEngine with return_expiries=True"
                )

            tenor_grid = create_tenor_grid(min_tau=1/12, max_tau=1.0)
            ti = TenorInterpolator(tenor_grid=tenor_grid, min_contracts=4)

            # Extract data at target_time using adaptive windowing
            # Use _get_target_time_mask with fallback instead of exact matching
            mask = self._get_target_time_mask(fwd_curve_df.index, target_time, use_nearest=True)
            prices_at_time = fwd_curve_df[mask]

            if prices_at_time.empty:
                # Try extracting with adaptive window as fallback
                import warnings
                warnings.warn(
                    f"No exact match at target_time {target_time}. "
                    "Using adaptive window extraction for curve data.",
                    UserWarning
                )
                # Extract first column to check if data exists
                test_series = fwd_curve_df.iloc[:, 0]
                test_extract = self._extract_at_time_daily(test_series, target_time)
                if test_extract.empty:
                    raise ValueError(f"No data found at or near target_time {target_time}")
                # Re-extract all columns using the identified indices
                prices_at_time = fwd_curve_df.loc[
                    fwd_curve_df.index.normalize().isin(test_extract.index)
                ]

            if not prices_at_time.empty:
                # Interpolate to constant maturity
                const_mat_panel = ti.interpolate(prices_at_time, expiry_data)

                # Use constant maturity panel for curve features
                cf = CurveFeatures(continuous_df=const_mat_panel)
            else:
                raise ValueError(f"No data found at target_time {target_time}")
        else:
            # Use raw contract-based features
            cf = CurveFeatures(continuous_df=fwd_curve_df)

        features_dict = {}

        # Extract curve slope at target_time
        if slope:
            # Calculate full slope series
            slope_series = cf.curve_slope(front=slope_mos[0], back=slope_mos[1])

            # Detect curve data frequency to adjust extraction parameters
            curve_freq_minutes = self._detect_bar_frequency_minutes(fwd_curve_df)

            # Extract at target_time using adaptive window
            # Curve data is often less frequent, so use larger window
            max_window = max(120, int(curve_freq_minutes * 3))  # At least 2 hours or 3x frequency
            slope_at_time = self._extract_at_time_daily(
                slope_series,
                target_time,
                adaptive_window=True,
                max_window_minutes=max_window
            )

            if slope_at_time.empty:
                import warnings
                warnings.warn(
                    f"No curve slope data found at {target_time}. "
                    f"Curve data frequency: {curve_freq_minutes:.1f} min. "
                    "Slope feature will be all NaN.",
                    UserWarning
                )
                slope_feature = pd.Series(dtype=float, index=self.training_data.index.normalize())
            else:
                # Determine appropriate forward fill limit based on curve data frequency
                # If curve updates daily, allow 7 days fill; if hourly, allow 2 days
                if curve_freq_minutes >= 1440:  # Daily or less frequent
                    fill_limit = 7
                elif curve_freq_minutes >= 60:  # Hourly
                    fill_limit = 3
                else:  # Intraday
                    fill_limit = 2

                # Reindex to training_data dates with frequency-aware forward fill
                slope_at_time = slope_at_time.reindex(
                    self.training_data.index.normalize(),
                    method='ffill',
                    limit=fill_limit
                )

                # Lag if needed to avoid lookahead bias
                needs_shift = self._needs_shift(target_time)
                slope_feature = slope_at_time.shift(1) if needs_shift else slope_at_time

            # Generate feature name
            feature_name = self._make_feature_name(
                f'curve_slope_M{slope_mos[0]}_M{slope_mos[1]}',
                target_time,
                ""
            )
            features_dict[feature_name] = slope_feature

            if add_as_feature:
                self._add_feature(slope_feature, feature_name)

        # Extract normalized basis at target_time
        # When relative_basis=False and normalized_basis=True, use spread_pair_1 for normal basis
        if normalized_basis and not relative_basis:
            # Use spread_pair_1 for normalized basis calculation
            front_col = f"M{spread_pair_1[0]}"
            back_col = f"M{spread_pair_1[1]}"

            if front_col in fwd_curve_df.columns and back_col in fwd_curve_df.columns:
                # Calculate normalized spread: (M_back - M_front) / M_front
                basis_series = (fwd_curve_df[back_col] - fwd_curve_df[front_col]) / fwd_curve_df[front_col]

                # Detect curve data frequency to adjust extraction parameters
                curve_freq_minutes = self._detect_bar_frequency_minutes(fwd_curve_df)

                # Extract at target_time using adaptive window
                max_window = max(120, int(curve_freq_minutes * 3))
                basis_at_time = self._extract_at_time_daily(
                    basis_series,
                    target_time,
                    adaptive_window=True,
                    max_window_minutes=max_window
                )

                if basis_at_time.empty:
                    import warnings
                    warnings.warn(
                        f"No curve basis data found at {target_time}. "
                        f"Curve data frequency: {curve_freq_minutes:.1f} min. "
                        "Basis feature will be all NaN.",
                        UserWarning
                    )
                    basis_feature = pd.Series(dtype=float, index=self.training_data.index.normalize())
                else:
                    # Determine appropriate forward fill limit based on curve data frequency
                    if curve_freq_minutes >= 1440:  # Daily or less frequent
                        fill_limit = 7
                    elif curve_freq_minutes >= 60:  # Hourly
                        fill_limit = 3
                    else:  # Intraday
                        fill_limit = 2

                    # Reindex to training_data dates with frequency-aware forward fill
                    basis_at_time = basis_at_time.reindex(
                        self.training_data.index.normalize(),
                        method='ffill',
                        limit=fill_limit
                    )

                    # Lag if needed to avoid lookahead bias
                    needs_shift = self._needs_shift(target_time)
                    basis_feature = basis_at_time.shift(1) if needs_shift else basis_at_time

                # Generate feature name
                feature_name = self._make_feature_name(
                    f'norm_basis_M{spread_pair_1[0]}_M{spread_pair_1[1]}',
                    target_time,
                    ""
                )
                features_dict[feature_name] = basis_feature

                if add_as_feature:
                    self._add_feature(basis_feature, feature_name)

        # Extract relative basis at target_time
        # Relative basis = (Spread1/dt1) - (Spread2/dt2)
        if relative_basis:
            # Get column names for both spread pairs
            s1_front_col = f"M{spread_pair_1[0]}"
            s1_back_col = f"M{spread_pair_1[1]}"
            s2_front_col = f"M{spread_pair_2[0]}"
            s2_back_col = f"M{spread_pair_2[1]}"

            # Calculate time differences (months between contracts)
            dt1 = abs(spread_pair_1[1] - spread_pair_1[0])
            dt2 = abs(spread_pair_2[1] - spread_pair_2[0])

            required_cols = [s1_front_col, s1_back_col, s2_front_col, s2_back_col]
            if all(col in fwd_curve_df.columns for col in required_cols):
                # Calculate spreads
                spread_1 = fwd_curve_df[s1_back_col] - fwd_curve_df[s1_front_col]
                spread_2 = fwd_curve_df[s2_back_col] - fwd_curve_df[s2_front_col]

                # Annualized relative basis
                rel_basis_series = (spread_1 / dt1) - (spread_2 / dt2)

                # Detect curve data frequency to adjust extraction parameters
                curve_freq_minutes = self._detect_bar_frequency_minutes(fwd_curve_df)

                # Extract at target_time using adaptive window
                max_window = max(120, int(curve_freq_minutes * 3))
                rel_basis_at_time = self._extract_at_time_daily(
                    rel_basis_series,
                    target_time,
                    adaptive_window=True,
                    max_window_minutes=max_window
                )

                if rel_basis_at_time.empty:
                    import warnings
                    warnings.warn(
                        f"No relative basis data found at {target_time}. "
                        f"Curve data frequency: {curve_freq_minutes:.1f} min. "
                        "Relative basis feature will be all NaN.",
                        UserWarning
                    )
                    rel_basis_feature = pd.Series(dtype=float, index=self.training_data.index.normalize())
                else:
                    # Determine appropriate forward fill limit based on curve data frequency
                    if curve_freq_minutes >= 1440:  # Daily or less frequent
                        fill_limit = 7
                    elif curve_freq_minutes >= 60:  # Hourly
                        fill_limit = 3
                    else:  # Intraday
                        fill_limit = 2

                    # Reindex to training_data dates with frequency-aware forward fill
                    rel_basis_at_time = rel_basis_at_time.reindex(
                        self.training_data.index.normalize(),
                        method='ffill',
                        limit=fill_limit
                    )

                    # Lag if needed to avoid lookahead bias
                    needs_shift = self._needs_shift(target_time)
                    rel_basis_feature = rel_basis_at_time.shift(1) if needs_shift else rel_basis_at_time

                # Generate feature name: rel_basis_M1M2_vs_M2M3
                feature_name = self._make_feature_name(
                    f'rel_basis_M{spread_pair_1[0]}M{spread_pair_1[1]}_vs_M{spread_pair_2[0]}M{spread_pair_2[1]}',
                    target_time,
                    ""
                )
                features_dict[feature_name] = rel_basis_feature

                if add_as_feature:
                    self._add_feature(rel_basis_feature, feature_name)
            else:
                missing = [c for c in required_cols if c not in fwd_curve_df.columns]
                import warnings
                warnings.warn(
                    f"Missing columns for relative basis: {missing}. "
                    "Skipping relative basis calculation.",
                    UserWarning
                )

        return pd.DataFrame(features_dict)



    def daily_curve_changes(self,
                           fwd_curve_df: pd.DataFrame,
                           target_time: Union[time, List[time]],
                           lookback_days: int,
                           slope_change: bool = True,
                           slope_mos: Tuple[int, int] = (1, 4),
                           relative_basis_change: bool = True,
                           spread_pair_1: Tuple[int, int] = (1, 2),
                           spread_pair_2: Tuple[int, int] = (2, 3),
                           add_as_feature: bool = True) -> pd.DataFrame:
        """Calculate daily changes in curve shape (day-over-day).

        Measures how the forward curve structure at target_time has changed over
        the past lookback_days trading days. Features are properly lagged to avoid lookahead bias.

        Parameters
        ----------
        fwd_curve_df : pd.DataFrame
            Continuous contract prices (M1, M2, M3, ...) with intraday DatetimeIndex
        target_time : time or List[time]
            Intraday time(s) to extract curve for comparison
        lookback_days : int
            Lookback period in days. E.g., 5 = change from 5 days ago to today
        slope_change : bool, default True
            Calculate change in curve slope over lookback_days
        slope_mos : Tuple[int, int], default (1, 4)
            (front, back) contract months for slope calculation
        relative_basis_change : bool, default True
            Calculate change in relative basis over lookback_days.
            Relative basis = (Spread1 / dt1) - (Spread2 / dt2)
        spread_pair_1 : Tuple[int, int], default (1, 2)
            First spread pair (front, back). E.g., (1, 2) means M2 - M1
        spread_pair_2 : Tuple[int, int], default (2, 3)
            Second spread pair (front, back). E.g., (2, 3) means M3 - M2
        add_as_feature : bool, default True
            If True, add features to training_data with proper lagging

        Returns
        -------
        pd.DataFrame
            DataFrame with daily curve change features

        Notes
        -----
        - Features are lagged if target_time is at or after self.target_time
        - Slope change: slope[t] - slope[t - lookback_days]
        - Relative basis change: rel_basis[t] - rel_basis[t - lookback_days]
          where rel_basis = (Spread1/dt1) - (Spread2/dt2)
        - Positive values indicate steepening/widening, negative values indicate flattening/narrowing

        Examples
        --------
        >>> # 5-day relative basis change at 2pm
        >>> daily_change_feats = model.daily_curve_changes(
        ...     fwd_curve_df=continuous_contracts,
        ...     target_time=time(14, 0),
        ...     lookback_days=5,
        ...     relative_basis_change=True,
        ...     spread_pair_1=(1, 2),  # M2 - M1
        ...     spread_pair_2=(2, 3),  # M3 - M2
        ... )
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            all_features = []
            for t in target_time:
                feats = self.daily_curve_changes(
                    fwd_curve_df, t, lookback_days, slope_change, slope_mos,
                    relative_basis_change, spread_pair_1, spread_pair_2, add_as_feature
                )
                all_features.append(feats)
            return pd.concat(all_features, axis=1)

        # Single time handling
        cf = CurveFeatures(continuous_df=fwd_curve_df)
        features_dict = {}

        # Detect curve data frequency once for both slope and basis
        curve_freq_minutes = self._detect_bar_frequency_minutes(fwd_curve_df)

        # Determine forward fill limit based on curve frequency
        if curve_freq_minutes >= 1440:  # Daily or less frequent
            fill_limit = 7
        elif curve_freq_minutes >= 60:  # Hourly
            fill_limit = 3
        else:  # Intraday
            fill_limit = 2

        max_window = max(120, int(curve_freq_minutes * 3))

        # Calculate slope change over lookback_days
        if slope_change:
            # Calculate full slope series
            slope_series = cf.curve_slope(front=slope_mos[0], back=slope_mos[1])

            # Extract at target_time using adaptive window
            slope_at_time = self._extract_at_time_daily(
                slope_series,
                target_time,
                adaptive_window=True,
                max_window_minutes=max_window
            )

            if slope_at_time.empty:
                import warnings
                warnings.warn(
                    f"No curve slope data for daily changes at {target_time}. "
                    f"Curve data frequency: {curve_freq_minutes:.1f} min.",
                    UserWarning
                )
                slope_change_feature = pd.Series(dtype=float, index=self.training_data.index.normalize())
            else:
                # Reindex to training_data dates with frequency-aware forward fill
                slope_at_time = slope_at_time.reindex(
                    self.training_data.index.normalize(),
                    method='ffill',
                    limit=fill_limit
                )

                # Calculate change: current slope minus slope lookback_days ago
                slope_change_series = slope_at_time - slope_at_time.shift(lookback_days)

                # Lag if needed to avoid lookahead bias
                needs_shift = self._needs_shift(target_time)
                slope_change_feature = slope_change_series.shift(1) if needs_shift else slope_change_series

            # Generate feature name
            feature_name = self._make_feature_name(
                f'daily_curve_slope_change_{lookback_days}d_M{slope_mos[0]}_M{slope_mos[1]}',
                target_time,
                ""
            )
            features_dict[feature_name] = slope_change_feature

            if add_as_feature:
                self._add_feature(slope_change_feature, feature_name)

        # Calculate relative basis change over lookback_days
        # Relative basis = (Spread1 / dt1) - (Spread2 / dt2)
        # where Spread = M_back - M_front, dt = months between contracts
        if relative_basis_change:
            # Spread 1: e.g., M2 - M1
            s1_front_col = f"M{spread_pair_1[0]}"
            s1_back_col = f"M{spread_pair_1[1]}"
            dt1 = abs(spread_pair_1[1] - spread_pair_1[0])

            # Spread 2: e.g., M3 - M2
            s2_front_col = f"M{spread_pair_2[0]}"
            s2_back_col = f"M{spread_pair_2[1]}"
            dt2 = abs(spread_pair_2[1] - spread_pair_2[0])

            required_cols = [s1_front_col, s1_back_col, s2_front_col, s2_back_col]
            if all(col in fwd_curve_df.columns for col in required_cols):
                # Calculate spreads
                spread_1 = fwd_curve_df[s1_back_col] - fwd_curve_df[s1_front_col]
                spread_2 = fwd_curve_df[s2_back_col] - fwd_curve_df[s2_front_col]

                # Annualized relative basis
                rel_basis_series = (spread_1 / dt1) - (spread_2 / dt2)

                # Extract at target_time using adaptive window
                rel_basis_at_time = self._extract_at_time_daily(
                    rel_basis_series,
                    target_time,
                    adaptive_window=True,
                    max_window_minutes=max_window
                )

                if rel_basis_at_time.empty:
                    import warnings
                    warnings.warn(
                        f"No relative basis data for daily changes at {target_time}. "
                        f"Curve data frequency: {curve_freq_minutes:.1f} min.",
                        UserWarning
                    )
                    rel_basis_change_feature = pd.Series(dtype=float, index=self.training_data.index.normalize())
                else:
                    # Reindex to training_data dates with frequency-aware forward fill
                    rel_basis_at_time = rel_basis_at_time.reindex(
                        self.training_data.index.normalize(),
                        method='ffill',
                        limit=fill_limit
                    )

                    # Calculate change: current rel_basis minus rel_basis lookback_days ago
                    rel_basis_change_series = rel_basis_at_time - rel_basis_at_time.shift(lookback_days)

                    # Lag if needed to avoid lookahead bias
                    needs_shift = self._needs_shift(target_time)
                    rel_basis_change_feature = rel_basis_change_series.shift(1) if needs_shift else rel_basis_change_series

                # Generate feature name: daily_rel_basis_change_5d_M1M2_vs_M2M3
                feature_name = self._make_feature_name(
                    f'daily_rel_basis_change_{lookback_days}d_M{spread_pair_1[0]}M{spread_pair_1[1]}_vs_M{spread_pair_2[0]}M{spread_pair_2[1]}',
                    target_time,
                    ""
                )
                features_dict[feature_name] = rel_basis_change_feature

                if add_as_feature:
                    self._add_feature(rel_basis_change_feature, feature_name)

        return pd.DataFrame(features_dict)

    def intraday_curve_changes(self,
                              fwd_curve_df: pd.DataFrame,
                              time_1: time,
                              time_2: Optional[time] = None,
                              period_length: Optional[timedelta] = None,
                              slope_change: bool = True,
                              slope_mos: Tuple[int, int] = (1, 4),
                              relative_basis_change: bool = True,
                              spread_pair_1: Tuple[int, int] = (1, 2),
                              spread_pair_2: Tuple[int, int] = (2, 3),
                              add_as_feature: bool = True) -> pd.DataFrame:
        """Calculate intraday changes in curve shape (within-day).

        Measures how the forward curve structure changed within the trading day,
        either between two specific times or over a period from time_1.

        Parameters
        ----------
        fwd_curve_df : pd.DataFrame
            Continuous contract prices (M1, M2, M3, ...) with intraday DatetimeIndex
        time_1 : time
            First time point or starting time for comparison
        time_2 : time, optional
            Second time point for comparison. If None, must provide period_length.
        period_length : timedelta, optional
            Period from time_1 to calculate change. E.g., timedelta(hours=4) = 4 hours after time_1.
            If None, must provide time_2.
        slope_change : bool, default True
            Calculate change in curve slope between times
        slope_mos : Tuple[int, int], default (1, 4)
            (front, back) contract months for slope calculation
        relative_basis_change : bool, default True
            Calculate change in relative basis between times.
            Relative basis = (Spread1 / dt1) - (Spread2 / dt2)
        spread_pair_1 : Tuple[int, int], default (1, 2)
            First spread pair (front, back). E.g., (1, 2) means M2 - M1
        spread_pair_2 : Tuple[int, int], default (2, 3)
            Second spread pair (front, back). E.g., (2, 3) means M3 - M2
        add_as_feature : bool, default True
            If True, add features to training_data with proper lagging

        Returns
        -------
        pd.DataFrame
            DataFrame with intraday curve change features

        Notes
        -----
        - Features are lagged if the later time is at or after self.target_time
        - Slope change: slope[time_2] - slope[time_1] (or slope[time_1 + period_length] - slope[time_1])
        - Relative basis change: rel_basis[time_2] - rel_basis[time_1]
          where rel_basis = (Spread1/dt1) - (Spread2/dt2)
        - Positive values indicate steepening/widening, negative values indicate flattening/narrowing

        Examples
        --------
        >>> # Curve change from 10am to 2pm (same day)
        >>> intraday_change_feats = model.intraday_curve_changes(
        ...     fwd_curve_df=continuous_contracts,
        ...     time_1=time(10, 0),
        ...     time_2=time(14, 0),
        ...     slope_change=True,
        ...     relative_basis_change=True,
        ...     spread_pair_1=(1, 2),  # M2 - M1
        ...     spread_pair_2=(2, 3),  # M3 - M2
        ... )
        >>>
        >>> # Curve change over 4 hours from 10am
        >>> intraday_change_feats = model.intraday_curve_changes(
        ...     fwd_curve_df=continuous_contracts,
        ...     time_1=time(10, 0),
        ...     period_length=timedelta(hours=4),
        ...     slope_change=True
        ... )
        """
        if time_2 is None and period_length is None:
            raise ValueError("Must provide either time_2 or period_length")

        # Determine the second time
        if time_2 is None:
            # Calculate time_2 from time_1 + period_length
            dt_base = pd.Timestamp('1970-01-01') + pd.Timedelta(
                hours=time_1.hour,
                minutes=time_1.minute,
                seconds=time_1.second
            )
            dt_end = dt_base + period_length
            time_2 = dt_end.time()

        cf = CurveFeatures(continuous_df=fwd_curve_df)
        features_dict = {}

        # Detect curve data frequency for adaptive extraction
        curve_freq_minutes = self._detect_bar_frequency_minutes(fwd_curve_df)

        # Determine forward fill limit based on curve frequency
        if curve_freq_minutes >= 1440:  # Daily or less frequent
            fill_limit = 7
        elif curve_freq_minutes >= 60:  # Hourly
            fill_limit = 3
        else:  # Intraday
            fill_limit = 2

        max_window = max(120, int(curve_freq_minutes * 3))

        # Calculate slope change between time_1 and time_2
        if slope_change:
            # Calculate full slope series
            slope_series = cf.curve_slope(front=slope_mos[0], back=slope_mos[1])

            # Extract at time_1 and time_2 using adaptive window
            slope_at_time_1 = self._extract_at_time_daily(
                slope_series,
                time_1,
                adaptive_window=True,
                max_window_minutes=max_window
            )
            slope_at_time_2 = self._extract_at_time_daily(
                slope_series,
                time_2,
                adaptive_window=True,
                max_window_minutes=max_window
            )

            # Check if either extraction failed
            if slope_at_time_1.empty or slope_at_time_2.empty:
                import warnings
                warnings.warn(
                    f"Insufficient curve slope data for intraday changes at {time_1} to {time_2}. "
                    f"Curve data frequency: {curve_freq_minutes:.1f} min.",
                    UserWarning
                )
                slope_change_feature = pd.Series(dtype=float, index=self.training_data.index.normalize())
            else:
                # Reindex both to training_data dates with frequency-aware forward fill
                slope_at_time_1 = slope_at_time_1.reindex(
                    self.training_data.index.normalize(),
                    method='ffill',
                    limit=fill_limit
                )
                slope_at_time_2 = slope_at_time_2.reindex(
                    self.training_data.index.normalize(),
                    method='ffill',
                    limit=fill_limit
                )

                # Calculate intraday change: slope at time_2 minus slope at time_1 (same day)
                # Align indices and compute difference
                common_dates = slope_at_time_1.index.intersection(slope_at_time_2.index)
                slope_change_series = slope_at_time_2.loc[common_dates] - slope_at_time_1.loc[common_dates]

                # Lag if needed to avoid lookahead bias (based on the later time)
                needs_shift = self._needs_shift(time_2)
                slope_change_feature = slope_change_series.shift(1) if needs_shift else slope_change_series

            # Generate feature name
            time_1_str = time_1.strftime("%H%M")
            time_2_str = time_2.strftime("%H%M")
            feature_name = self._make_feature_name(
                f'intraday_curve_slope_change_{time_1_str}_to_{time_2_str}_M{slope_mos[0]}_M{slope_mos[1]}',
                time_2,  # Use later time for shift logic
                ""
            )
            features_dict[feature_name] = slope_change_feature

            if add_as_feature:
                self._add_feature(slope_change_feature, feature_name)

        # Calculate relative basis change between time_1 and time_2
        # Relative basis = (Spread1 / dt1) - (Spread2 / dt2)
        # where Spread = M_back - M_front, dt = months between contracts
        if relative_basis_change:
            # Spread 1: e.g., M2 - M1
            s1_front_col = f"M{spread_pair_1[0]}"
            s1_back_col = f"M{spread_pair_1[1]}"
            dt1 = abs(spread_pair_1[1] - spread_pair_1[0])

            # Spread 2: e.g., M3 - M2
            s2_front_col = f"M{spread_pair_2[0]}"
            s2_back_col = f"M{spread_pair_2[1]}"
            dt2 = abs(spread_pair_2[1] - spread_pair_2[0])

            required_cols = [s1_front_col, s1_back_col, s2_front_col, s2_back_col]
            if all(col in fwd_curve_df.columns for col in required_cols):
                # Calculate spreads
                spread_1 = fwd_curve_df[s1_back_col] - fwd_curve_df[s1_front_col]
                spread_2 = fwd_curve_df[s2_back_col] - fwd_curve_df[s2_front_col]

                # Annualized relative basis
                rel_basis_series = (spread_1 / dt1) - (spread_2 / dt2)

                # Extract at time_1 and time_2 using adaptive window
                rel_basis_at_time_1 = self._extract_at_time_daily(
                    rel_basis_series,
                    time_1,
                    adaptive_window=True,
                    max_window_minutes=max_window
                )
                rel_basis_at_time_2 = self._extract_at_time_daily(
                    rel_basis_series,
                    time_2,
                    adaptive_window=True,
                    max_window_minutes=max_window
                )

                # Check if either extraction failed
                if rel_basis_at_time_1.empty or rel_basis_at_time_2.empty:
                    import warnings
                    warnings.warn(
                        f"Insufficient relative basis data for intraday changes at {time_1} to {time_2}. "
                        f"Curve data frequency: {curve_freq_minutes:.1f} min.",
                        UserWarning
                    )
                    rel_basis_change_feature = pd.Series(dtype=float, index=self.training_data.index.normalize())
                else:
                    # Reindex both to training_data dates with frequency-aware forward fill
                    rel_basis_at_time_1 = rel_basis_at_time_1.reindex(
                        self.training_data.index.normalize(),
                        method='ffill',
                        limit=fill_limit
                    )
                    rel_basis_at_time_2 = rel_basis_at_time_2.reindex(
                        self.training_data.index.normalize(),
                        method='ffill',
                        limit=fill_limit
                    )

                    # Calculate intraday change: rel_basis at time_2 minus rel_basis at time_1 (same day)
                    common_dates = rel_basis_at_time_1.index.intersection(rel_basis_at_time_2.index)
                    rel_basis_change_series = rel_basis_at_time_2.loc[common_dates] - rel_basis_at_time_1.loc[common_dates]

                    # Lag if needed to avoid lookahead bias (based on the later time)
                    needs_shift = self._needs_shift(time_2)
                    rel_basis_change_feature = rel_basis_change_series.shift(1) if needs_shift else rel_basis_change_series

                # Generate feature name: intraday_rel_basis_change_1000_to_1400_M1M2_vs_M2M3
                time_1_str = time_1.strftime("%H%M")
                time_2_str = time_2.strftime("%H%M")
                feature_name = self._make_feature_name(
                    f'intraday_rel_basis_change_{time_1_str}_to_{time_2_str}_M{spread_pair_1[0]}M{spread_pair_1[1]}_vs_M{spread_pair_2[0]}M{spread_pair_2[1]}',
                    time_2,  # Use later time for shift logic
                    ""
                )
                features_dict[feature_name] = rel_basis_change_feature

                if add_as_feature:
                    self._add_feature(rel_basis_change_feature, feature_name)

        return pd.DataFrame(features_dict)

    def liquidity_impact(
            self,
            target_time: Union[time, List[time]],
            intraday_df: Optional[pd.DataFrame] = None,
            period_length: Optional[timedelta] = None,
            price_col: str = "Close",
            bid_vol_col: str = "BidVolume",
            ask_vol_col: str = "AskVolume",
            use_proxy: bool = False,
            add_as_feature: bool = False,
            use_nearest: bool = True,
    ) -> Union[pd.DataFrame, pd.DataFrame]:
        """Calculate average liquidity impact metrics over a time interval.

        Measures the average price impact of orders crossing the spread (aggressive flow)
        and the average magnitude of that flow over a rolling window.

        Parameters
        ----------
        target_time : time or List[time]
            Time(s) to extract metrics for.
        intraday_df : Optional[pd.DataFrame]
            Intraday data. Uses self.intraday_data if None.
        period_length : Optional[timedelta]
            Length of the rolling window for averaging. Uses self.closing_length if None.
        price_col : str, default "Close"
            Price column name.
        bid_vol_col : str, default "BidVolume"
            Bid volume column name.
        ask_vol_col : str, default "AskVolume"
            Ask volume column name.
        use_proxy : bool, default False
            If True, estimate aggressive volume using price direction if bid/ask cols missing.
        add_as_feature : bool, default False
            If True, add features to training_data with proper lagging.
        use_nearest : bool, default True
            Use nearest bar if exact target_time is missing.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - impact_coeff: Average |PriceChange| / |NetVolume| (Price impact per unit of volume)
            - impact_vol: Average |NetVolume| (Volume crossing the spread)
        """
        # Handle list of times
        if isinstance(target_time, (list, tuple)):
            all_features = []
            for t in target_time:
                feats = self.liquidity_impact(
                    t, intraday_df, period_length, price_col, bid_vol_col,
                    ask_vol_col, use_proxy, add_as_feature, use_nearest
                )
                # Rename columns to include time for uniqueness
                time_str = t.strftime("%H%M")
                feats.columns = [f"{c}_{time_str}" for c in feats.columns]
                all_features.append(feats)
            return pd.concat(all_features, axis=1)

        # Single time handling
        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for liquidity impact")

        window = period_length or self.closing_length
        prices = self._coerce_price(data, price_col)

        # Calculate Price Changes
        price_diff = prices.diff().fillna(0)

        # Calculate Net Aggressor Volume (Signed Volume)
        has_bid_ask = bid_vol_col in data.columns and ask_vol_col in data.columns

        if has_bid_ask:
            bid_vol = data[bid_vol_col].fillna(0)
            ask_vol = data[ask_vol_col].fillna(0)
            # Net Aggressor Volume: Ask (Buy) - Bid (Sell)
            signed_vol = ask_vol - bid_vol
        elif use_proxy:
            # Proxy: Volume * Sign of Price Change
            vol = data.get("Volume", pd.Series(0, index=data.index))
            signed_vol = vol * np.sign(price_diff)
        else:
            raise KeyError(
                f"Bid/Ask columns '{bid_vol_col}/{ask_vol_col}' not found. "
                "Set use_proxy=True to estimate."
            )

        # 1. Calculate Per-Bar Impact Coefficient (Amihud-like but for aggressive flow)
        # Ratio = |PriceChange| / |NetVolume|
        # Use abs() because we care about the magnitude of impact per unit of volume,
        # regardless of direction. Avoid division by zero.
        abs_vol = signed_vol.abs()
        impact_ratio = price_diff.abs() / abs_vol.replace(0, np.nan)

        # 2. Calculate rolling averages
        if window:
            bars = self._period_to_bars(window, data)
            # Average Impact Coefficient
            avg_impact = impact_ratio.rolling(bars, min_periods=1).mean()
            # Average Aggressive Volume (Impact Volume)
            avg_impact_vol = abs_vol.rolling(bars, min_periods=1).mean()
        else:
            avg_impact = impact_ratio
            avg_impact_vol = abs_vol

        # Extract at target time
        mask = self._get_target_time_mask(data.index, target_time, use_nearest)

        # Create result DataFrame
        result = pd.DataFrame({
            'impact_coeff': avg_impact[mask],
            'impact_vol': avg_impact_vol[mask]
        })

        # Handle empty extraction fallback
        if result.empty:
            avg_impact_extract = self._extract_at_time_daily(avg_impact, target_time)
            avg_vol_extract = self._extract_at_time_daily(avg_impact_vol, target_time)

            result = pd.DataFrame({
                'impact_coeff': avg_impact_extract,
                'impact_vol': avg_vol_extract
            })
        else:
            result.index = pd.to_datetime(result.index).normalize()

        result = self._filter_to_trading_dates(result)

        if add_as_feature:
            needs_shift = self._needs_shift(target_time)
            period_str = self._format_period_length(window)

            for col in result.columns:
                feature_series = result[col].shift(1) if needs_shift else result[col]
                feature_name = self._make_feature_name(col, target_time, period_str)
                self._add_feature(feature_series, feature_name)

        return result


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

    def fit_selected(
            self,
            selector: Optional[FeatureSelector] = None,
            selector_config: Optional[FeatureSelectionConfig] = None,
            drop_na_target: bool = True,
            drop_na_features: Optional[str] = None,
            force_keep_dates: Optional[list] = None,
            fillna_features: Optional[dict] = None,
            X:Optional[np.array]=None,
            y:Optional[np.array]=None,
            **fit_kwargs,
    ):
        """Fit model with feature selection

        Args:
            selector: Optional FeatureSelector instance to use
            selector_config: Optional FeatureSelectionConfig to create selector
            drop_na_target: If True, drop rows where target is NaN (default: True)
            drop_na_features: How to handle NaN in features:
                - 'any': drop rows with any NaN in features
                - 'all': drop rows only if all features are NaN
                - None: don't drop rows based on feature NaNs (default)
            force_keep_dates: Optional list of dates/indices to keep even if they have NaNs
            fillna_features: Optional dict of {feature: value} to fill NaNs before selection
            **fit_kwargs: Additional arguments passed to fit()

        Returns:
            self
        """

        if X is None:
            X = self.training_data.copy()
        if y is None:
            y = self.target_data.copy()

        # Ensure X and y are aligned by index (use inner join)
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        # Track which rows to keep
        keep_mask = pd.Series(True, index=X.index)

        # 1. Handle NaN in target
        if drop_na_target:
            target_valid = y.notna()
            keep_mask &= target_valid

        # 2. Handle NaN in features
        if drop_na_features == 'any':
            # Drop rows with any NaN in features
            features_valid = X.notna().all(axis=1)
            keep_mask &= features_valid
        elif drop_na_features == 'all':
            # Drop rows only if all features are NaN
            features_valid = X.notna().any(axis=1)
            keep_mask &= features_valid
        # If None, don't drop based on features

        # 3. Force keep specific dates/indices
        if force_keep_dates is not None:
            force_keep_mask = X.index.isin(force_keep_dates)
            keep_mask |= force_keep_mask

        # Apply the combined mask
        X = X.loc[keep_mask]
        y = y.loc[keep_mask]

        # 4. Fill NaN in features if requested (before selection)
        if fillna_features is not None:
            for feature, value in fillna_features.items():
                if feature in X.columns:
                    X[feature] = X[feature].fillna(value)

        base_model = self._get_model()

        # Use the base_model wrapper directly - FeatureSelector now handles CTA models
        # (CTALight, CTAXGBoost, etc.) as well as sklearn estimators
        est = base_model

        if selector is None:
            selector = FeatureSelector(selector_config or FeatureSelectionConfig())

        Xs = selector.fit_transform(X, y, estimator=est)

        # cache for later
        self.selector_ = selector
        self.selected_features_ = selector.selected_features_
        self.feature_names = selector.selected_features_

        # fit the actual wrapper model using reduced X
        self.fit(Xs, y, **fit_kwargs)

        return self

    def predict_selected(self, X: pd.DataFrame, **predict_kwargs):
        if not hasattr(self, "selector_"):
            raise ValueError("No selector fitted. Call fit_selected() first.")
        Xs = self.selector_.transform(X)
        return self.predict(Xs, **predict_kwargs)

    def prune_features(
            self,
            selector: Optional[FeatureSelector] = None,
            selector_config: Optional[FeatureSelectionConfig] = None,
            drop_na_target: bool = True,
            drop_na_features: Optional[str] = None,
            force_keep_dates: Optional[list] = None,
            fillna_features: Optional[dict] = None    ) -> "IntradayMomentum":
        """Prune features using feature selection and update training_data and feature_names.

        This method applies feature selection to the current training_data and target_data,
        then updates the model's internal state so that subsequent calls to fit() or
        fit_with_grid_search() will automatically use only the selected features.

        Parameters
        ----------
        selector : Optional[FeatureSelector]
            Pre-configured FeatureSelector instance. If None, creates one from selector_config.
        selector_config : Optional[FeatureSelectionConfig]
            Configuration for feature selection. If None, uses default config.
        drop_na_target : bool, default True
            If True, drop rows where target is NaN
        drop_na_features : Optional[str], default None
            How to handle NaN in features:
            - 'any': drop rows with any NaN in features
            - 'all': drop rows only if all features are NaN
            - None: don't drop rows based on feature NaNs
        force_keep_dates : Optional[list], default None
            List of dates/indices to keep even if they have NaNs
        fillna_features : Optional[dict], default None
            Dict of {feature: value} to fill NaNs before selection

        Returns
        -------
        self
            Returns self for method chaining

        Examples
        --------
        >>> # Prune features, then train with grid search
        >>> model = IntradayMomentum(intraday_data, ...)
        >>> # Build features...
        >>> model.prune_features(selector_config=FeatureSelectionConfig(max_features=50))
        >>> # Now fit with only selected features
        >>> model.fit_with_grid_search(X_train, y_train, param_grid={...})
        >>>
        >>> # Or prune with custom config and NaN handling
        >>> config = FeatureSelectionConfig(
        ...     max_features=80,
        ...     per_group_max=3,
        ...     corr_threshold=0.95
        ... )
        >>> model.prune_features(
        ...     selector_config=config,
        ...     drop_na_features='all',  # Only drop if all features are NaN
        ...     fillna_features={'feature1': 0, 'feature2': -999}
        ... )
        >>> model.fit(X_train, y_train)

        Notes
        -----
        - This permanently modifies training_data and feature_names
        - The selector and selected features are cached in selector_ and selected_features_
        - To see what was selected, use the feature_report_ attribute on the selector
        - After pruning, you can use normal fit() or fit_with_grid_search()
        """
        X = self.training_data.copy()
        y = self.target_data.copy()

        # Ensure X and y are aligned by index (use inner join)
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        # Track which rows to keep
        keep_mask = pd.Series(True, index=X.index)

        # 1. Handle NaN in target
        if drop_na_target:
            target_valid = y.notna()
            keep_mask &= target_valid

        # 2. Handle NaN in features
        if drop_na_features == 'any':
            # Drop rows with any NaN in features
            features_valid = X.notna().all(axis=1)
            keep_mask &= features_valid
        elif drop_na_features == 'all':
            # Drop rows only if all features are NaN
            features_valid = X.notna().any(axis=1)
            keep_mask &= features_valid
        # If None, don't drop based on features

        # 3. Force keep specific dates/indices
        if force_keep_dates is not None:
            force_keep_mask = X.index.isin(force_keep_dates)
            keep_mask |= force_keep_mask

        # Apply the combined mask
        X = X.loc[keep_mask]
        y = y.loc[keep_mask]

        # 4. Fill NaN in features if requested (before selection)
        if fillna_features is not None:
            for feature, value in fillna_features.items():
                if feature in X.columns:
                    X[feature] = X[feature].fillna(value)

        # Get the base model for importance calculations
        base_model = self._get_model()
        # Use the base_model wrapper directly - FeatureSelector now handles CTA models
        est = base_model

        # Create or use provided selector
        if selector is None:
            selector = FeatureSelector(selector_config or FeatureSelectionConfig())

        # Fit the selector and transform the data
        selector.fit(X, y, estimator=est)

        # Cache selector and selected features
        self.selector_ = selector
        self.selected_features_ = selector.selected_features_

        # Update training_data to only include selected features
        self.training_data = self.training_data[self.selected_features_].copy()

        # Update feature_names to match
        self.feature_names = self.selected_features_.copy()

        return self

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
            handle_missing: str = "drop",
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
        handle_missing : str, default "drop"
            How to handle missing/invalid values:
            - "drop": Drop rows with NaN/inf (safe, reduces data)
            - "fill": Forward fill NaNs, replace inf with NaN then forward fill
            - "zero": Replace NaN/inf with 0 (use with caution)

        Returns
        -------
        pd.DataFrame
            DataFrame with momentum features named as momentum_{d}d

        Notes
        -----
        All features are lagged by 1 day to avoid lookahead bias - they do not
        include the target date's (present day) return. For example, momentum_5d
        on day T represents the cumulative return from day T-6 to day T-1.

        The method handles edge cases:
        - Zero or negative prices (log is undefined) -> NaN
        - Missing data at start due to shift operations
        - Index alignment with training_data

        Examples
        --------
        >>> model = IntradayMomentumLight(intraday_data, ...)
        >>> momentum_feats = model.add_daily_momentum_features(daily_df, lookbacks=[5, 10, 20])
        >>> # momentum_5d contains returns from T-6 to T-1 (5 days, lagged by 1)
        """
        prices = self._coerce_price(daily_df, price_col)

        # Ensure index is normalized DatetimeIndex
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        prices.index = prices.index.normalize()

        # Validate prices (log requires positive values)
        if (prices <= 0).any():
            import warnings
            n_invalid = (prices <= 0).sum()
            warnings.warn(
                f"Found {n_invalid} non-positive prices in daily_df. "
                "Log returns will be NaN for these periods.",
                UserWarning
            )

        feats: Dict[str, pd.Series] = {}

        for lb in lookbacks:
            # Calculate total log return over lookback period
            # Use np.where to handle edge cases gracefully
            price_ratio = prices / prices.shift(lb)

            # Calculate log return, handling invalid values
            with np.errstate(invalid='ignore', divide='ignore'):
                momentum = np.log(price_ratio)

            # Replace inf with NaN (from division by zero or log of zero)
            momentum = momentum.replace([np.inf, -np.inf], np.nan)

            # Lag by 1 to avoid using current day's return
            momentum_lagged = momentum.shift(1)

            # Handle missing values based on strategy
            if handle_missing == "fill":
                # Forward fill NaNs (use with caution - may propagate stale data)
                momentum_lagged = momentum_lagged.fillna(method='ffill')
            elif handle_missing == "zero":
                # Replace NaNs with 0 (neutral return)
                momentum_lagged = momentum_lagged.fillna(0)
            # "drop" strategy: leave NaNs as-is, will be handled by _add_feature

            feature_name = f"momentum_{lb}d"
            feats[feature_name] = momentum_lagged

            # Add to training data if it exists
            if isinstance(self.training_data, pd.DataFrame):
                # Align with training_data index before adding
                aligned_feature = momentum_lagged.reindex(
                    self.training_data.index.normalize(),
                    method=None  # No forward/backward fill during reindex
                )
                self._add_feature(aligned_feature, feature_name, tf='1d')

        return pd.DataFrame(feats)

    def deseasonalized_vol(
            self,
            target_time: Union[datetime.time, List[datetime.time]],
            period_length: timedelta,
            rolling_days: int = 252,
            return_volume: bool = True,
            return_volatility: bool = True,
            return_scaled_returns: bool = False,
            use_session_times: bool = True,
            add_as_feature: bool = False,
            use_cache: bool = True,
            bid_ask_volume: bool = False,
            refit_interval: int = 10,
    ):
        """Extract deseasonalized volatility and/or volume at target time(s).

        Computes deseasonalization once, then extracts at multiple times efficiently.
        Caches deseasonalized results to reuse across multiple calls with same parameters.

        Parameters
        ----------
        target_time : time or list of time
            Target time(s) to extract features at
        period_length : timedelta
            Resampling frequency (e.g., timedelta(minutes=5))
        rolling_days : int, default 252
            Rolling window for deseasonalization (avoids look-ahead bias)
        return_volume : bool, default True
            Return deseasonalized volume
        return_volatility : bool, default True
            Return deseasonalized volatility
        return_scaled_returns : bool, default False
            Return returns scaled by seasonal volatility factor (r / s_t).
            Useful for standardizing returns by time-of-day volatility.
        use_session_times : bool, default True
            Filter to session hours
        add_as_feature : bool, default False
            Add extracted series as features to the model
        use_cache : bool, default True
            Use cached deseasonalized data if available. Set to False to force recalculation.
        bid_ask_volume : bool, default False
            If True, deseasonalize bid and ask volumes separately plus their imbalance.
            Requires 'BidVolume' and 'AskVolume' columns in intraday_data.
            Returns additional columns: 'bid_volume', 'ask_volume', 'volume_imbalance'.
        refit_interval : int, default 1
            Refit the seasonal model every N days instead of every day. Higher values
            (e.g., 5-10) provide significant speedup with minimal accuracy loss.
            Use 1 for maximum accuracy, 5 for balanced speed/accuracy, 10 for maximum speed.
        """
        # Helper to convert various formats to time object
        def _to_time(t):
            if isinstance(t, time):
                return t
            elif isinstance(t, int):
                return time(t, 0)
            elif isinstance(t, tuple) and len(t) in (2, 3) and all(isinstance(x, int) for x in t):
                # Tuple (hour, minute[, second]) -> time object
                return time(*t)
            else:
                raise TypeError(f"target_time must be time, int, or (hour, min) tuple, got {type(t)}: {t}")

        # Normalize target_time to list for uniform processing
        # Special case: if tuple of 2-3 ints, treat as single time (hour, minute)
        if isinstance(target_time, tuple) and len(target_time) in (2, 3) and all(isinstance(x, int) for x in target_time):
            times_list = [_to_time(target_time)]
        elif isinstance(target_time, (list, tuple)):
            times_list = [_to_time(t) for t in target_time]
        else:
            times_list = [_to_time(target_time)]

        single_time = len(times_list) == 1

        # Create cache key based on parameters that affect deseasonalization
        freq_minutes = int(period_length.total_seconds() / 60)
        cache_key = (
            freq_minutes,
            rolling_days,
            use_session_times,
            return_volume,
            return_volatility or return_scaled_returns,  # Both need volatility deseasonalization
            bid_ask_volume,
            refit_interval,  # OPTIMIZATION: Include in cache key
        )

        # Check if we can reuse cached deseasonalized data
        if use_cache and cache_key in self._deseas_cache:
            cached = self._deseas_cache[cache_key]
            deseasonalized_vol_full = cached.get('deseasonalized_vol_full')
            seasonal_factor = cached.get('seasonal_factor')
            deseasonalized_vol_full_volume = cached.get('deseasonalized_vol_full_volume')
            intraday_idx = cached.get('intraday_idx')
            returns = cached.get('returns')
            scaled_returns_full = cached.get('scaled_returns_full')
            deseasonalized_bid_volume = cached.get('deseasonalized_bid_volume')
            deseasonalized_ask_volume = cached.get('deseasonalized_ask_volume')
            deseasonalized_imbalance = cached.get('deseasonalized_imbalance')
        else:
            # Compute deseasonalized data from scratch
            # OPTIMIZATION: Only copy when filtering is needed
            resample_rule = f"{freq_minutes}min"

            if use_session_times:
                data = self.intraday_data.between_time(
                    self.session_open, self.session_end, inclusive='both'
                ).copy()  # Only copy the filtered subset
            else:
                data = self.intraday_data  # No copy needed - work with view

            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            prices = self._coerce_price(data, "Close")

            # Use closed='right', label='right' to avoid lookahead bias:
            # - Bar labeled 10:00 contains data from 09:01 to 10:00 (not 10:00 to 10:59)
            # - The return at 10:00 = (close_10:00 - close_09:00) / close_09:00
            resampled_prices = prices.resample(
                resample_rule, closed='right', label='right'
            ).last()
            returns = resampled_prices.pct_change().dropna()

            volatility = returns.abs()

            if return_volume:
                if "Volume" not in data.columns:
                    raise KeyError("Volume column not found in intraday data")
                volume = data["Volume"].resample(
                    resample_rule, closed='right', label='right'
                ).sum()
            else:
                volume = None

            # Build intraday index for deseasonalization (once)
            resampled_idx = resampled_prices.index
            intraday_idx = pd.Series(resampled_idx).groupby(resampled_idx.normalize()).cumcount()
            intraday_idx.index = resampled_idx

            # Compute deseasonalized volatility ONCE
            # Also get seasonal factor if we need scaled returns
            seasonal_factor = None
            deseasonalized_vol_full = None
            scaled_returns_full = None
            if return_volatility or return_scaled_returns:
                vol_idx = intraday_idx.loc[volatility.index]
                deseas_result = deseasonalize_volatility(
                    volatility,
                    intraday_idx=vol_idx,
                    rolling_days=rolling_days,
                    refit_interval=refit_interval,  # OPTIMIZATION: Pass through refit interval
                )
                deseasonalized_vol_full = deseas_result["adjusted"]

                # Get seasonal factor for scaling returns (convert from log-space to linear)
                if return_scaled_returns:
                    seasonal_factor = np.exp(deseas_result["seasonal"])
                    # Pre-compute scaled returns for caching
                    common_idx = returns.index.intersection(seasonal_factor.index)
                    scaled_returns_full = returns.loc[common_idx] / seasonal_factor.loc[common_idx]

            # Compute deseasonalized volume ONCE
            deseasonalized_vol_full_volume = None
            if return_volume and volume is not None:
                vol_idx = intraday_idx.loc[volume.index]
                deseasonalized_vol_full_volume = deseasonalize_volume(
                    volume,
                    intraday_idx=vol_idx,
                    rolling_days=rolling_days,
                    refit_interval=refit_interval,  # OPTIMIZATION: Pass through refit interval
                )["adjusted"]

            # Compute deseasonalized bid/ask volumes and imbalance
            deseasonalized_bid_volume = None
            deseasonalized_ask_volume = None
            deseasonalized_imbalance = None
            if bid_ask_volume:
                if "BidVolume" not in data.columns or "AskVolume" not in data.columns:
                    raise KeyError("BidVolume and AskVolume columns required when bid_ask_volume=True")

                # Resample bid and ask volumes
                bid_vol = data["BidVolume"].resample(
                    resample_rule, closed='right', label='right'
                ).sum()
                ask_vol = data["AskVolume"].resample(
                    resample_rule, closed='right', label='right'
                ).sum()

                # Get intraday indices for the resampled volumes
                bid_idx = intraday_idx.loc[bid_vol.index]
                ask_idx = intraday_idx.loc[ask_vol.index]

                # Deseasonalize bid, ask, and imbalance
                bid_ask_result = deseasonalize_volume(
                    volume=None,
                    bid_ask_volume=True,
                    bid_volume=bid_vol,
                    ask_volume=ask_vol,
                    intraday_idx=bid_idx,
                    rolling_days=rolling_days,
                    refit_interval=refit_interval,  # OPTIMIZATION: Pass through refit interval
                )

                deseasonalized_bid_volume = bid_ask_result["bid_adjusted"]
                deseasonalized_ask_volume = bid_ask_result["ask_adjusted"]
                deseasonalized_imbalance = bid_ask_result["imbalance_adjusted"]

            # Cache the results for future calls
            if use_cache:
                self._deseas_cache[cache_key] = {
                    'deseasonalized_vol_full': deseasonalized_vol_full,
                    'seasonal_factor': seasonal_factor,
                    'deseasonalized_vol_full_volume': deseasonalized_vol_full_volume,
                    'intraday_idx': intraday_idx,
                    'returns': returns,
                    'scaled_returns_full': scaled_returns_full,
                    'deseasonalized_bid_volume': deseasonalized_bid_volume,
                    'deseasonalized_ask_volume': deseasonalized_ask_volume,
                    'deseasonalized_imbalance': deseasonalized_imbalance,
                }

        period_str = self._format_period_length(period_length)
        all_results = {}

        # Extract at target times from cached/computed deseasonalized data
        if return_volatility and deseasonalized_vol_full is not None:
            for t in times_list:
                time_key = t.strftime("%H%M")
                mask = self._get_target_time_mask(deseasonalized_vol_full.index, t)
                target_series = deseasonalized_vol_full[mask]

                if target_series.empty:
                    target_series = self._extract_at_time_daily(deseasonalized_vol_full, t)
                else:
                    target_series.index = pd.to_datetime(target_series.index).normalize()

                target_series = self._filter_to_trading_dates(target_series)

                if add_as_feature:
                    needs_shift = self._needs_shift(t)
                    feature_series = target_series.shift(1) if needs_shift else target_series
                    feature_name = self._make_feature_name(
                        "deseasonalized_volatility", t, period_str,
                    )
                    self._add_feature(feature_series, feature_name)

                col_name = "volatility" if single_time else f"volatility_{time_key}"
                all_results[col_name] = target_series

        # Extract scaled returns from cached data
        if return_scaled_returns and scaled_returns_full is not None:

            for t in times_list:
                time_key = t.strftime("%H%M")
                mask = self._get_target_time_mask(scaled_returns_full.index, t)
                target_series = scaled_returns_full[mask]

                if target_series.empty:
                    target_series = self._extract_at_time_daily(scaled_returns_full, t)
                else:
                    target_series.index = pd.to_datetime(target_series.index).normalize()

                target_series = self._filter_to_trading_dates(target_series)

                if add_as_feature:
                    needs_shift = self._needs_shift(t)
                    feature_series = target_series.shift(1) if needs_shift else target_series
                    feature_name = self._make_feature_name(
                        "scaled_returns", t, period_str,
                    )
                    self._add_feature(feature_series, feature_name)

                col_name = "scaled_returns" if single_time else f"scaled_returns_{time_key}"
                all_results[col_name] = target_series

        # Extract deseasonalized volume from cached data
        if return_volume and deseasonalized_vol_full_volume is not None:
            for t in times_list:
                time_key = t.strftime("%H%M")
                mask = self._get_target_time_mask(deseasonalized_vol_full_volume.index, t)
                target_series = deseasonalized_vol_full_volume[mask]

                if target_series.empty:
                    target_series = self._extract_at_time_daily(deseasonalized_vol_full_volume, t)
                else:
                    target_series.index = pd.to_datetime(target_series.index).normalize()

                target_series = self._filter_to_trading_dates(target_series)

                if add_as_feature:
                    needs_shift = self._needs_shift(t)
                    feature_series = target_series.shift(1) if needs_shift else target_series
                    feature_name = self._make_feature_name(
                        "deseasonalized_volume", t, period_str,
                    )
                    self._add_feature(feature_series, feature_name)

                col_name = "volume" if single_time else f"volume_{time_key}"
                all_results[col_name] = target_series

        # Extract deseasonalized bid/ask volumes and imbalance from cached data
        if bid_ask_volume and deseasonalized_bid_volume is not None:
            for t in times_list:
                time_key = t.strftime("%H%M")

                # Extract bid volume
                mask = self._get_target_time_mask(deseasonalized_bid_volume.index, t)
                bid_series = deseasonalized_bid_volume[mask]

                if bid_series.empty:
                    bid_series = self._extract_at_time_daily(deseasonalized_bid_volume, t)
                else:
                    bid_series.index = pd.to_datetime(bid_series.index).normalize()

                bid_series = self._filter_to_trading_dates(bid_series)

                if add_as_feature:
                    needs_shift = self._needs_shift(t)
                    feature_series = bid_series.shift(1) if needs_shift else bid_series
                    feature_name = self._make_feature_name(
                        "deseasonalized_bid_volume", t, period_str,
                    )
                    self._add_feature(feature_series, feature_name)

                col_name = "bid_volume" if single_time else f"bid_volume_{time_key}"
                all_results[col_name] = bid_series

                # Extract ask volume
                mask = self._get_target_time_mask(deseasonalized_ask_volume.index, t)
                ask_series = deseasonalized_ask_volume[mask]

                if ask_series.empty:
                    ask_series = self._extract_at_time_daily(deseasonalized_ask_volume, t)
                else:
                    ask_series.index = pd.to_datetime(ask_series.index).normalize()

                ask_series = self._filter_to_trading_dates(ask_series)

                if add_as_feature:
                    needs_shift = self._needs_shift(t)
                    feature_series = ask_series.shift(1) if needs_shift else ask_series
                    feature_name = self._make_feature_name(
                        "deseasonalized_ask_volume", t, period_str,
                    )
                    self._add_feature(feature_series, feature_name)

                col_name = "ask_volume" if single_time else f"ask_volume_{time_key}"
                all_results[col_name] = ask_series

                # Extract volume imbalance
                mask = self._get_target_time_mask(deseasonalized_imbalance.index, t)
                imbalance_series = deseasonalized_imbalance[mask]

                if imbalance_series.empty:
                    imbalance_series = self._extract_at_time_daily(deseasonalized_imbalance, t)
                else:
                    imbalance_series.index = pd.to_datetime(imbalance_series.index).normalize()

                imbalance_series = self._filter_to_trading_dates(imbalance_series)

                if add_as_feature:
                    needs_shift = self._needs_shift(t)
                    feature_series = imbalance_series.shift(1) if needs_shift else imbalance_series
                    feature_name = self._make_feature_name(
                        "deseasonalized_volume_imbalance", t, period_str,
                    )
                    self._add_feature(feature_series, feature_name)

                col_name = "volume_imbalance" if single_time else f"volume_imbalance_{time_key}"
                all_results[col_name] = imbalance_series

        # Return format based on inputs
        if len(all_results) == 1:
            return next(iter(all_results.values()))

        return pd.DataFrame(all_results)

    def clear_deseas_cache(self):
        """Clear the deseasonalized data cache.

        Call this method if you've updated the intraday_data and want to force
        recalculation of deseasonalized volatility/volume.
        """
        self._deseas_cache = {}

    def add_basic_datetime_features(
            self,
            dates: Optional[pd.DatetimeIndex] = None,
            add_day_of_week: bool = True,
            add_month: bool = True,
            add_quarter: bool = True,
            use_sincos: bool = False,
    ) -> pd.DataFrame:
        """Attach simple calendar breakdowns (DOW, month, quarter).

        Parameters
        ----------
        dates : Optional[pd.DatetimeIndex]
            Date index to create features for. Uses self.training_data.index if None.
        add_day_of_week : bool, default True
            Add day of week features
        add_month : bool, default True
            Add month features
        add_quarter : bool, default True
            Add quarter features
        use_sincos : bool, default False
            If True, encode cyclical features using sin/cos instead of raw integers.
            This better represents the cyclical nature (e.g., Dec->Jan, Sun->Mon).
            Creates two features per cycle: {name}_sin and {name}_cos

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime features
        """

        idx = dates if dates is not None else self.training_data.index
        normalized = pd.DatetimeIndex(idx).normalize()
        feats: Dict[str, pd.Series] = {}

        if add_day_of_week:
            dow = normalized.dayofweek
            if use_sincos:
                # Day of week: 0-6, period = 7
                feats["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
                feats["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
            else:
                feats["day_of_week"] = dow

        if add_month:
            month = normalized.month
            if use_sincos:
                # Month: 1-12, period = 12
                feats["month_sin"] = np.sin(2 * np.pi * month / 12)
                feats["month_cos"] = np.cos(2 * np.pi * month / 12)
            else:
                feats["month"] = month

        if add_quarter:
            quarter = normalized.quarter
            if use_sincos:
                # Quarter: 1-4, period = 4
                feats["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
                feats["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
            else:
                feats["quarter"] = quarter

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
            use_sincos: bool = False,
            expiration_cycle_weeks: int = 13,
    ) -> pd.DataFrame:
        """Add calendar-based features such as opex and expiry timing.

        Parameters
        ----------
        ticker : str
            Ticker symbol to determine contract cycle
        include_last_trading_week : bool, default True
            Add is_last_trading_week feature
        include_opex_week : bool, default True
            Add is_opex_week feature
        include_days_since_opex : bool, default True
            Add days_since_opex feature
        include_expiration_week : bool, default True
            Add is_expiration_week feature
        include_weeks_until_expiration : bool, default True
            Add weeks_until_expiration feature
        dates : Optional[pd.DatetimeIndex]
            Date index to create features for. Uses self.training_data.index if None.
        use_sincos : bool, default False
            If True, encode cyclical features using sin/cos instead of raw values.
            Applies to: days_since_opex (30-day cycle), weeks_until_expiration (contract cycle)
        expiration_cycle_weeks : int, default 13
            Period for weeks_until_expiration cycle when using sin/cos encoding.
            Use 13 for quarterly contracts, 4 for monthly contracts.

        Returns
        -------
        pd.DataFrame
            DataFrame with calendar features
        """

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

        # Apply sin/cos encoding to cyclical features if requested
        if use_sincos:
            if "days_since_opex" in feature_df.columns:
                # Monthly OPEX cycle (approximately 30 days, but use 21 trading days)
                days = feature_df["days_since_opex"].fillna(0)
                feature_df["days_since_opex_sin"] = np.sin(2 * np.pi * days / 21)
                feature_df["days_since_opex_cos"] = np.cos(2 * np.pi * days / 21)
                feature_df = feature_df.drop(columns=["days_since_opex"])

            if "weeks_until_expiration" in feature_df.columns:
                # Contract expiration cycle (default 13 weeks for quarterly)
                weeks = feature_df["weeks_until_expiration"]
                feature_df["weeks_until_expiration_sin"] = np.sin(2 * np.pi * weeks / expiration_cycle_weeks)
                feature_df["weeks_until_expiration_cos"] = np.cos(2 * np.pi * weeks / expiration_cycle_weeks)
                feature_df = feature_df.drop(columns=["weeks_until_expiration"])

        for name, series in feature_df.items():
            self._add_feature(series, name)
        return feature_df

    def add_eia_release_features(
            self,
            ticker: str,
            dates: Optional[pd.DatetimeIndex] = None,
            add_as_feature: bool = True,
    ) -> pd.DataFrame:
        """Add EIA (Energy Information Administration) release day features.

        EIA releases weekly petroleum and natural gas inventory reports which are
        major market-moving events for energy futures.

        Parameters
        ----------
        ticker : str
            Ticker symbol to determine product type:
            - Petroleum products (CL, RB, HO, BZ): First Tuesday + every Wednesday
            - Natural Gas products (NG): First Tuesday + every Thursday
        dates : Optional[pd.DatetimeIndex]
            Date index to create features for. Uses self.training_data.index if None.
        add_as_feature : bool, default True
            If True, add features to training_data

        Returns
        -------
        pd.DataFrame
            DataFrame with EIA release features:
            - is_eia_release_day: Binary flag (1 on release days, 0 otherwise)
            - days_since_eia: Days since last EIA release
            - days_until_eia: Days until next EIA release

        Notes
        -----
        EIA Release Schedule:
        - Petroleum Status Report (CL, RB, HO, BZ):
          * First Tuesday of each month (if market open)
          * Every Wednesday at 10:30 AM ET
        - Natural Gas Storage Report (NG):
          * First Tuesday of each month (if market open)
          * Every Thursday at 10:30 AM ET

        Release days can cause increased volatility and volume.

        Examples
        --------
        >>> model = IntradayMomentum(intraday_data)
        >>> # For crude oil
        >>> eia_feats = model.add_eia_release_features('CL', add_as_feature=True)
        >>> # For natural gas
        >>> eia_feats = model.add_eia_release_features('NG', add_as_feature=True)
        """
        idx = dates if dates is not None else self.training_data.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)

        # Normalize to dates
        normalized = pd.DatetimeIndex(idx).normalize()

        # Determine product type
        petroleum_products = ['CL', 'RB', 'HO', 'BZ', 'WTI', 'BRENT']
        natural_gas_products = ['NG', 'NGAS', 'NATGAS']

        ticker_upper = ticker.upper()

        # Initialize release day mask
        is_eia_day = pd.Series(False, index=normalized)

        # Get day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)
        day_of_week = normalized.dayofweek

        # First Tuesday of each month (applies to both petroleum and NG)
        # Check if it's a Tuesday (day 1) and day of month is 1-7
        is_first_tuesday = (day_of_week == 1) & (normalized.day <= 7)
        is_eia_day |= is_first_tuesday

        if ticker_upper in petroleum_products:
            # Every Wednesday (day 2)
            is_wednesday = day_of_week == 2
            is_eia_day |= is_wednesday

        elif ticker_upper in natural_gas_products:
            # Every Thursday (day 3)
            is_thursday = day_of_week == 3
            is_eia_day |= is_thursday

        else:
            # Unknown ticker - use petroleum schedule as default
            import warnings
            warnings.warn(
                f"Ticker '{ticker}' not recognized for EIA releases. "
                f"Using petroleum schedule (First Tuesday + Wednesday). "
                f"Known tickers: {petroleum_products + natural_gas_products}",
                UserWarning
            )
            is_wednesday = day_of_week == 2
            is_eia_day |= is_wednesday

        # Calculate days since last EIA release
        eia_dates = normalized[is_eia_day]
        days_since = pd.Series(np.nan, index=normalized)

        for i, date in enumerate(normalized):
            past_releases = eia_dates[eia_dates < date]
            if len(past_releases) > 0:
                days_since.iloc[i] = (date - past_releases[-1]).days
            else:
                days_since.iloc[i] = np.nan

        # Calculate days until next EIA release
        days_until = pd.Series(np.nan, index=normalized)

        for i, date in enumerate(normalized):
            future_releases = eia_dates[eia_dates > date]
            if len(future_releases) > 0:
                days_until.iloc[i] = (future_releases[0] - date).days
            else:
                days_until.iloc[i] = np.nan

        # Create feature DataFrame
        feature_df = pd.DataFrame({
            'is_eia_release_day': is_eia_day.astype(int),
            'days_since_eia': days_since,
            'days_until_eia': days_until
        }, index=normalized)

        if add_as_feature and isinstance(self.training_data, pd.DataFrame):
            for col in feature_df.columns:
                self._add_feature(feature_df[col], col)

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

    def get_xy(
            self,
            start_date=None,
            end_date=None,
            val_split: bool = False,
            val_split_size: float = 0.2,
            dropna: bool = True,
            verbose: bool = False,
            remove_outliers: bool = False,
            outlier_threshold: float = 0.01,
    ):
        """Convenience constructor for a full training feature matrix.

        Performs the following cleaning steps:
        1. Selects features from self.feature_names
        2. Drops columns with >90% NaN values (keeps columns with at least 10% valid data)
        3. Drops rows with any remaining NaN values (if dropna=True)
        4. Removes outliers from target data (if remove_outliers=True)
        5. Aligns with target data by date intersection
        6. Optionally filters by date range
        7. Optionally splits into train/validation sets

        Parameters
        ----------
        start_date : str, pd.Timestamp, or datetime, optional
            Filter data from this date onwards (inclusive)
        end_date : str, pd.Timestamp, or datetime, optional
            Filter data up to this date (inclusive)
        val_split : bool, default False
            If True, split data into train and validation sets
        val_split_size : float, default 0.2
            Proportion of data to use for validation (only used if val_split=True)
        dropna : bool, default True
            If True, drop rows with NaN values. Set to False to keep all rows
            (useful for debugging or when you want to handle NaN yourself).
        verbose : bool, default False
            If True, print diagnostic info showing where rows are dropped
        remove_outliers : bool, default False
            If True, remove extreme return outliers from target data before alignment.
            Removes returns below the `outlier_threshold` percentile and above
            the `1 - outlier_threshold` percentile.
        outlier_threshold : float, default 0.01
            Percentile threshold for outlier removal (0.01 = remove top/bottom 1%).
            Only used if remove_outliers=True.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series] or Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            If val_split=False: (X, y) tuple with aligned features and target
            If val_split=True: (X_train, X_val, y_train, y_val) tuple
        """
        # Select feature columns
        x_data = self.training_data[self.feature_names].copy()
        initial_rows = len(x_data)
        initial_cols = len(x_data.columns)

        if verbose:
            print(f"Initial: {initial_rows} rows, {initial_cols} features")

        # Step 1: Drop columns that are mostly NaN (>90% NaN)
        # Keep columns with at least 10% non-NaN values
        min_valid_count = max(1, int(len(x_data) * 0.1))
        x_data = x_data.dropna(axis=1, thresh=min_valid_count)

        if verbose:
            dropped_cols = initial_cols - len(x_data.columns)
            if dropped_cols > 0:
                print(f"After dropping sparse columns: {len(x_data.columns)} features ({dropped_cols} dropped)")

        # Step 2: Drop rows with any remaining NaN values
        if dropna:
            rows_before = len(x_data)

            if verbose:
                # Identify which columns have NaN in remaining rows
                nan_per_col = x_data.isna().sum()
                cols_with_nan = nan_per_col[nan_per_col > 0].sort_values(ascending=False)
                if len(cols_with_nan) > 0:
                    print(f"Columns with NaN values (causing row drops):")
                    for col, count in cols_with_nan.head(10).items():
                        print(f"  {col}: {count} NaN ({100*count/rows_before:.1f}%)")
                    if len(cols_with_nan) > 10:
                        print(f"  ... and {len(cols_with_nan) - 10} more columns")

            x_data = x_data.dropna(axis=0)

            if verbose:
                dropped_rows = rows_before - len(x_data)
                if dropped_rows > 0:
                    print(f"After dropna: {len(x_data)} rows ({dropped_rows} dropped, {100*dropped_rows/rows_before:.1f}%)")

        # Step 3: Remove outliers from target data
        y_data = self.target_data.copy()

        if remove_outliers:
            y_before = len(y_data)
            lower_pct = outlier_threshold
            upper_pct = 1 - outlier_threshold
            lower_bound = y_data.quantile(lower_pct)
            upper_bound = y_data.quantile(upper_pct)

            # Remove rows outside bounds
            y_data = y_data[(y_data >= lower_bound) & (y_data <= upper_bound)]

            if verbose:
                removed = y_before - len(y_data)
                print(f"Outlier removal: {removed} rows removed ({100*removed/y_before:.1f}%)")
                print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}] (p{100*lower_pct:.0f}-p{100*upper_pct:.0f})")

        # Step 4: Align with target data
        rows_before = len(x_data)

        if verbose:
            print(f"Target data has {len(y_data)} rows")
            x_only = x_data.index.difference(y_data.index)
            y_only = y_data.index.difference(x_data.index)
            if len(x_only) > 0:
                print(f"  Dates in X but not in target: {len(x_only)} (first: {x_only.min()}, last: {x_only.max()})")
            if len(y_only) > 0:
                print(f"  Dates in target but not in X: {len(y_only)} (first: {y_only.min()}, last: {y_only.max()})")

        val_dates = x_data.index.intersection(y_data.index)
        x_data = x_data.loc[val_dates]
        y_data = y_data.loc[val_dates]

        if verbose:
            dropped_rows = rows_before - len(x_data)
            if dropped_rows > 0:
                print(f"After target alignment: {len(x_data)} rows ({dropped_rows} dropped)")

        # Step 5: Filter by date range if provided
        if start_date is not None:
            rows_before = len(x_data)
            start_date = pd.to_datetime(start_date)
            x_data = x_data[x_data.index >= start_date]
            y_data = y_data[y_data.index >= start_date]
            if verbose and rows_before - len(x_data) > 0:
                print(f"After start_date filter: {len(x_data)} rows ({rows_before - len(x_data)} dropped)")

        if end_date is not None:
            rows_before = len(x_data)
            end_date = pd.to_datetime(end_date)
            x_data = x_data[x_data.index <= end_date]
            y_data = y_data[y_data.index <= end_date]
            if verbose and rows_before - len(x_data) > 0:
                print(f"After end_date filter: {len(x_data)} rows ({rows_before - len(x_data)} dropped)")

        if verbose:
            print(f"Final: {len(x_data)} rows, {len(x_data.columns)} features")

        # Step 6: Optionally split into train/validation sets
        if val_split:
            # Time-series split: use earlier data for training, later for validation
            split_idx = int(len(x_data) * (1 - val_split_size))

            X_train = x_data.iloc[:split_idx]
            X_val = x_data.iloc[split_idx:]
            y_train = y_data.iloc[:split_idx]
            y_val = y_data.iloc[split_idx:]

            return X_train, X_val, y_train, y_val
        else:
            return x_data, y_data

class DeepIDMomentum(IntradayMomentum):
    """Deep learning variant of IntradayMomentum with sequential and optional profile data.

    Extends IntradayMomentum to support multiple data modalities:
    - Summary features (inherited from parent)
    - Sequential data (required)
    - Profile array data (optional)

    Parameters
    ----------
    sequential_data : pd.DataFrame
        Sequential/time-series data with DatetimeIndex. Typically intraday orderflow
        or market profile data that needs to be processed as sequences.
    profile_array : np.ndarray, optional
        Optional 3D array of profile data (n_samples, height, width) for CNN processing.
    **kwargs
        All other arguments passed to IntradayMomentum.__init__
    """

    def __init__(
            self,
            sequential_data: pd.DataFrame,
            profile_array: Optional[np.ndarray] = None,
            **kwargs
    ) -> None:
        # Initialize parent class
        super().__init__(**kwargs)

        # Store original training_data (summary features)
        summary_data = self.training_data.copy()

        # Convert training_data to dictionary structure
        self.training_data = {
            'summary': summary_data,
            'sequential': sequential_data
        }

        # Add profile array if provided
        if profile_array is not None:
            self.training_data['profile'] = profile_array

        # Store reference to sequential data for easy access
        self.sequential_data = sequential_data
        self.profile_array = profile_array

    @classmethod
    def from_files(
            cls,
            intraday_path: str,
            features_path: str,
            sequential_path: str,
            profile_path: Optional[str] = None,
            target_path: Optional[str] = None,
            target_col: str = "target",
            **kwargs,
    ) -> "DeepIDMomentum":
        """Load DeepIDMomentum from file paths.

        Parameters
        ----------
        intraday_path : str
            Path to intraday OHLCV data (CSV or Parquet with DatetimeIndex)
        features_path : str
            Path to pre-computed summary features (CSV or Parquet)
        sequential_path : str
            Path to sequential data (CSV or Parquet with DatetimeIndex)
        profile_path : str, optional
            Path to profile array data (.npy file)
        target_path : str, optional
            Path to target data. If None, expects target_col in features file.
        target_col : str, default "target"
            Column name for target variable
        **kwargs
            Additional arguments passed to DeepIDMomentum.__init__

        Returns
        -------
        DeepIDMomentum
            Initialized model with loaded data and features
        """
        # Load intraday data
        if intraday_path.endswith('.parquet'):
            intraday_data = pd.read_parquet(intraday_path)
        else:
            intraday_data = read_exported_df(intraday_path)

        if not isinstance(intraday_data.index, pd.DatetimeIndex):
            intraday_data.index = pd.to_datetime(intraday_data.index)

        # Load features
        if features_path.endswith('.parquet'):
            features_df = pd.read_parquet(features_path)
        else:
            features_df = pd.read_csv(features_path, parse_dates=True, index_col=0)

        if not isinstance(features_df.index, pd.DatetimeIndex):
            features_df.index = pd.to_datetime(features_df.index)

        # Load sequential data
        if sequential_path.endswith('.parquet'):
            sequential_df = pd.read_parquet(sequential_path)
        else:
            sequential_df = pd.read_csv(sequential_path, parse_dates=True, index_col=0)

        if not isinstance(sequential_df.index, pd.DatetimeIndex):
            sequential_df.index = pd.to_datetime(sequential_df.index)

        # Load profile array if provided
        profile_array = None
        if profile_path is not None:
            profile_array = np.load(profile_path)

        # Load or extract target
        if target_path is not None:
            if target_path.endswith('.parquet'):
                target_df = pd.read_parquet(target_path)
            else:
                target_df = pd.read_csv(target_path, parse_dates=True, index_col=0)
            if target_col in target_df.columns:
                target_data = target_df[target_col]
            else:
                target_data = target_df.iloc[:, 0]
        elif target_col in features_df.columns:
            target_data = features_df[target_col]
            features_df = features_df.drop(columns=[target_col])
        else:
            target_data = None

        # Remove intraday_data from kwargs if present (will be passed explicitly)
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('intraday_data', None)
        kwargs_copy.pop('sequential_data', None)
        kwargs_copy.pop('profile_array', None)

        # Create instance - pass intraday_data to parent, sequential_data to child
        instance = cls(
            intraday_data=intraday_data,
            sequential_data=sequential_df,
            profile_array=profile_array,
            **kwargs_copy
        )

        # Override training_data summary component with loaded features
        instance.training_data['summary'] = features_df.copy()
        instance.feature_names = list(features_df.columns)

        # Override target_data if provided
        if target_data is not None:
            instance.target_data = target_data

        return instance

    def normalize_sequential_features(
            self,
            price_col: str = 'close',
            vah_col: str = 'vah',
            val_col: str = 'val',
            poc_col: str = 'poc',
            ib_high_col: Optional[str] = 'ib_high',
            ib_low_col: Optional[str] = 'ib_low',
            additional_cols: Optional[List[str]] = None,
            inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Normalize/make stationary the sequential data columns.

        Normalizes price-like columns (market profile, initial balance, etc.) by
        expressing them relative to the close price: (value - close) / close

        This makes the features scale-invariant and stationary.

        Parameters
        ----------
        price_col : str, default 'close'
            Column name for the close price to normalize against
        vah_col : str, default 'vah'
            Column name for Value Area High
        val_col : str, default 'val'
            Column name for Value Area Low
        poc_col : str, default 'poc'
            Column name for Point of Control
        ib_high_col : str, optional, default 'ib_high'
            Column name for Initial Balance High. Set to None to skip.
        ib_low_col : str, optional, default 'ib_low'
            Column name for Initial Balance Low. Set to None to skip.
        additional_cols : list of str, optional
            Additional column names to normalize using the same formula.
            Useful for custom price levels, support/resistance, etc.
        inplace : bool, default True
            If True, modify self.sequential_data in place and update training_data dict.
            If False, return normalized copy without modifying original.

        Returns
        -------
        pd.DataFrame or None
            If inplace=True, returns None (modifies in place).
            If inplace=False, returns normalized DataFrame.

        Notes
        -----
        - Normalization formula: (value - close) / close
        - Resulting values represent percentage distance from close price
        - Positive values mean above close, negative means below close
        - Original columns are replaced with normalized versions
        - Columns that don't exist in data are silently skipped

        Examples
        --------
        >>> model = DeepIDMomentum(sequential_data=seq_df, intraday_data=intra_df)
        >>> # Normalize market profile columns only
        >>> model.normalize_sequential_features(price_col='close')
        >>>
        >>> # Include initial balance columns
        >>> model.normalize_sequential_features(
        ...     price_col='close',
        ...     ib_high_col='ib_high',
        ...     ib_low_col='ib_low'
        ... )
        >>>
        >>> # Skip IB columns if not present
        >>> model.normalize_sequential_features(
        ...     price_col='close',
        ...     ib_high_col=None,
        ...     ib_low_col=None
        ... )
        >>>
        >>> # Add custom columns
        >>> model.normalize_sequential_features(
        ...     price_col='close',
        ...     additional_cols=['prev_high', 'prev_low', 'support', 'resistance']
        ... )
        """
        # Determine which dataframe to work with
        if inplace:
            data = self.sequential_data
        else:
            data = self.sequential_data.copy()

        # Check price column exists
        if price_col not in data.columns:
            raise KeyError(f"Price column '{price_col}' not found in sequential_data. Available: {list(data.columns)}")

        # Build list of all columns to normalize
        cols_to_normalize = []

        # Add profile columns
        profile_cols = [vah_col, val_col, poc_col]
        for col in profile_cols:
            if col in data.columns:
                cols_to_normalize.append(col)

        # Add initial balance columns if specified
        if ib_high_col is not None and ib_high_col in data.columns:
            cols_to_normalize.append(ib_high_col)
        if ib_low_col is not None and ib_low_col in data.columns:
            cols_to_normalize.append(ib_low_col)

        # Add additional columns if specified
        if additional_cols is not None:
            for col in additional_cols:
                if col in data.columns:
                    cols_to_normalize.append(col)

        # Ensure we have at least one column to normalize
        if not cols_to_normalize:
            raise KeyError(
                f"No columns found to normalize. Checked: {profile_cols}" +
                f"([ib_high_col, ib_low_col] if provided) + (additional_cols if provided). "
                f"Available: {list(data.columns)}"
            )

        # Get close prices
        close_prices = data[price_col].copy()

        # Normalize all columns: (value - close) / close
        for col in cols_to_normalize:
            data[col] = (data[col] - close_prices) / close_prices

        if inplace:
            # Update the sequential_data reference
            self.sequential_data = data
            # Update training_data dict
            self.training_data['sequential'] = data
            return None
        else:
            return data

    def _make_clf_targets(
            self,
            upper_threshold: float = 0.6,
            lower_threshold: float = 0.3,
            inplace: bool = True
    ) -> Optional[pd.Series]:
        """Create classification targets from continuous returns.

        Divides returns into 3 classes based on quantile thresholds:
        - Class 0: Negative returns (returns < 0)
        - Class 1: Small absolute returns (neutral, devalued returns)
        - Class 2: Large positive returns (returns >= upper quantile)

        The quantile thresholds are computed on the absolute values of returns
        to identify returns with the smallest magnitude.

        Parameters
        ----------
        upper_threshold : float, default 0.6
            Upper quantile threshold (0-1) for class 2. Returns above this
            quantile of absolute values are classified as class 2.
        lower_threshold : float, default 0.3
            Lower quantile threshold (0-1) for class 1. Returns between
            lower_threshold and upper_threshold are classified as class 1.
        inplace : bool, default True
            If True, replace self.target_data with classification targets.
            If False, return new targets without modifying original.

        Returns
        -------
        pd.Series or None
            If inplace=True, returns None (modifies in place).
            If inplace=False, returns classification targets as Series.

        Examples
        --------
        >>> model = DeepIDMomentum(...)
        >>> # Create 3-class targets with default thresholds
        >>> model._make_clf_targets()
        >>> # Class distribution: 0 (down), 1 (neutral), 2 (strong up)

        >>> # More granular: keep more in neutral class
        >>> model._make_clf_targets(upper_threshold=0.7, lower_threshold=0.3)

        Notes
        -----
        - Quantiles are computed on absolute values, so symmetric thresholds
        - Class 1 captures low-volatility / small-return periods
        - Useful for filtering noise in high-frequency data
        """
        returns = self.target_data.copy()

        # Compute quantiles on absolute values
        abs_returns = returns.abs()
        upper_q = abs_returns.quantile(upper_threshold)
        lower_q = abs_returns.quantile(lower_threshold)

        # Initialize all as class 1 (neutral/small returns)
        clf_targets = pd.Series(1, index=returns.index, dtype=np.int64)

        # Class 0: Negative returns
        clf_targets[returns < 0] = 0

        # Class 2: Large positive returns (above upper quantile of absolute values)
        # Must be both positive AND above threshold
        clf_targets[(returns > 0) & (abs_returns >= upper_q)] = 2

        # Class 1: Everything else (small absolute returns)
        # These are returns with absolute value below upper_q
        # This includes:
        # - Small positive returns (0 < return < upper_q)
        # Already set to 1 by default, but explicitly for clarity:
        clf_targets[(returns >= 0) & (abs_returns < upper_q)] = 1

        if inplace:
            self.target_data = clf_targets
            return None
        else:
            return clf_targets

    def get_loaders(
            self,
            start_date=None,
            end_date=None,
            val_split: bool = False,
            val_split_size: float = 0.2,
            dropna: bool = True,
            verbose: bool = False,
            remove_outliers: bool = False,
            outlier_threshold: float = 0.01,
            batch_size: int = 32,
            shuffle_train: bool = True,
            num_workers: int = 0,
            max_seq_len: int = 200,
            sequential_cols: Optional[List[str]] = None,
    ):
        """Create PyTorch DataLoaders for training dual-branch models.

        This method wraps get_xy() functionality and returns DataLoader objects
        using DualDataset for combined summary + sequential data.

        Parameters
        ----------
        start_date : str, pd.Timestamp, or datetime, optional
            Filter data from this date onwards (inclusive)
        end_date : str, pd.Timestamp, or datetime, optional
            Filter data up to this date (inclusive)
        val_split : bool, default False
            If True, split data into train and validation sets
        val_split_size : float, default 0.2
            Proportion of data to use for validation (only used if val_split=True)
        dropna : bool, default True
            If True, drop rows with NaN values
        verbose : bool, default False
            If True, print diagnostic info
        remove_outliers : bool, default False
            If True, remove extreme outliers from target data
        outlier_threshold : float, default 0.01
            Percentile threshold for outlier removal (0.01 = remove top/bottom 1%)
        batch_size : int, default 32
            Batch size for DataLoaders
        shuffle_train : bool, default True
            Whether to shuffle training data. Validation is never shuffled.
        num_workers : int, default 0
            Number of workers for DataLoader (0 = single process)
        max_seq_len : int, default 200
            Maximum sequence length for sequential data
        sequential_cols : list of str, optional
            Columns to use from sequential data. If None, uses all numeric columns.

        Returns
        -------
        torch.utils.data.DataLoader or Tuple[DataLoader, DataLoader]
            If val_split=False: single DataLoader for all data
            If val_split=True: (train_loader, val_loader) tuple

        Examples
        --------
        >>> model = DeepIDMomentum.from_files(...)
        >>> # Single loader for all data
        >>> loader = model.get_loaders(batch_size=64)
        >>>
        >>> # Train/val split
        >>> train_loader, val_loader = model.get_loaders(
        ...     val_split=True,
        ...     val_split_size=0.2,
        ...     batch_size=32,
        ...     shuffle_train=True
        ... )
        """
        # Import here to avoid requiring torch for all IntradayMomentum usage
        try:
            import torch
            from torch.utils.data import DataLoader
            from ..data.model_datasets import DualDataset
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for get_loaders(). "
                "Install with: pip install torch"
            ) from e

        # Custom collate function for variable-length sequences
        def collate_fn(batch):
            import torch.nn.utils.rnn as rnn_utils
            summaries, sequential_seqs, targets, lengths = zip(*batch)
            summaries = torch.stack(summaries)
            targets = torch.stack(targets)
            lengths = torch.tensor(lengths)
            sequential_padded = rnn_utils.pad_sequence(
                sequential_seqs, batch_first=True, padding_value=0.0
            )
            return summaries, sequential_padded, targets, lengths

        # Use get_xy to get aligned, cleaned data
        result = self.get_xy(
            start_date=start_date,
            end_date=end_date,
            val_split=val_split,
            val_split_size=val_split_size,
            dropna=dropna,
            verbose=verbose,
            remove_outliers=remove_outliers,
            outlier_threshold=outlier_threshold,
        )

        if val_split:
            X_train, X_val, y_train, y_val = result

            # Create train dataset
            train_dataset = DualDataset(
                summary_data=X_train,
                sequential_data=self.sequential_data,
                target_data=y_train,
                max_len=max_seq_len,
                sequential_cols=sequential_cols,
                target_col=None,  # Targets already aligned via y_train
            )

            # Create validation dataset
            val_dataset = DualDataset(
                summary_data=X_val,
                sequential_data=self.sequential_data,
                target_data=y_val,
                max_len=max_seq_len,
                sequential_cols=sequential_cols,
                target_col=None,
            )

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,  # Never shuffle validation
                collate_fn=collate_fn,
                num_workers=num_workers,
            )

            return train_loader, val_loader

        else:
            X, y = result

            # Create dataset
            dataset = DualDataset(
                summary_data=X,
                sequential_data=self.sequential_data,
                target_data=y,
                max_len=max_seq_len,
                sequential_cols=sequential_cols,
                target_col=None,
            )

            # Create DataLoader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )

            return loader


