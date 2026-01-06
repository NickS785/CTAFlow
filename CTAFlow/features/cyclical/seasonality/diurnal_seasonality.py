"""
Volatility deseasonalization using Flexible Fourier Form (FFF).

This module provides tools to remove intraday diurnal patterns (e.g., the U-shape
in volume and volatility) from high-frequency financial time series.

The Flexible Fourier Form, popularized by Andersen and Bollerslev (1997), uses
sine/cosine basis functions to model smooth periodic patterns without requiring
dummy variables for each time interval.

Key Classes:
    DiurnalAdjuster: Main class for FFF-based deseasonalization

Key Functions:
    fft_spectrum: Analyze frequency content to identify dominant cycles
    compute_realized_volatility: Calculate realized volatility from returns
"""
from __future__ import annotations

from datetime import time
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression, Ridge


class DiurnalAdjuster:
    """Remove intraday diurnal seasonality using Flexible Fourier Form (FFF).

    The FFF models the seasonal component s(t) as a sum of sinusoids:
        s(t) = sum_{k=1}^{P} [a_k * sin(2*pi*k*t/N) + b_k * cos(2*pi*k*t/N)]

    where:
        - N: Number of intervals per day (e.g., 78 for 5-min bars in 6.5h session)
        - P: Order of the expansion (number of harmonic pairs)
        - a_k, b_k: Coefficients estimated via OLS

    Parameters
    ----------
    bins_per_day : int
        Number of time intervals per trading day. E.g., 78 for 5-min bars
        in a 6.5-hour session, 390 for 1-min bars.
    order : int, default 3
        Number of Fourier harmonic pairs (P). Higher values capture more
        complex patterns but risk overfitting. Typically 2-4 is sufficient.
    use_log : bool, default True
        If True, work in log-space (recommended for volatility/volume).
        Converts multiplicative seasonality to additive.
    regularization : float, default 0.0
        Ridge regression regularization strength. Set > 0 to prevent
        overfitting when order is high.

    Attributes
    ----------
    model_ : LinearRegression or Ridge
        Fitted regression model
    seasonal_pattern_ : np.ndarray
        Estimated seasonal pattern for one complete day (length = bins_per_day)
    coefficients_ : Dict[str, float]
        Fitted sine/cosine coefficients
    is_fitted : bool
        Whether the model has been fitted

    Examples
    --------
    >>> adjuster = DiurnalAdjuster(bins_per_day=78, order=3)
    >>> adjuster.fit(df['realized_vol'], df['intraday_idx'])
    >>> deseasonalized = adjuster.transform(df['realized_vol'], df['intraday_idx'])

    References
    ----------
    Andersen, T.G. and Bollerslev, T. (1997). "Intraday periodicity and
    volatility persistence in financial markets." Journal of Empirical Finance.
    """

    def __init__(
        self,
        bins_per_day: int,
        order: int = 3,
        use_log: bool = True,
        regularization: float = 0.0,
    ):
        self.bins_per_day = bins_per_day
        self.order = order
        self.use_log = use_log
        self.regularization = regularization

        self.model_: Optional[Union[LinearRegression, Ridge]] = None
        self.seasonal_pattern_: Optional[np.ndarray] = None
        self.coefficients_: Dict[str, float] = {}
        self.is_fitted = False

    def _create_fourier_features(self, intraday_idx: np.ndarray) -> np.ndarray:
        """Create sine/cosine basis features for FFF regression.

        Parameters
        ----------
        intraday_idx : np.ndarray
            Intraday time index (0 to bins_per_day-1, repeating each day)

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2*order) with sin/cos features
        """
        t = np.asarray(intraday_idx, dtype=np.float64)
        n_samples = len(t)
        n_features = 2 * self.order

        # Pre-allocate array for speed
        X = np.empty((n_samples, n_features), dtype=np.float64)

        # Vectorized computation of all harmonics
        for k in range(1, self.order + 1):
            phase = 2 * np.pi * k * t / self.bins_per_day
            X[:, 2*(k-1)] = np.sin(phase)
            X[:, 2*(k-1) + 1] = np.cos(phase)

        return X

    def fit(
        self,
        y: Union[pd.Series, np.ndarray],
        intraday_idx: Union[pd.Series, np.ndarray],
    ) -> 'DiurnalAdjuster':
        """Fit the FFF model to estimate the diurnal pattern.

        Parameters
        ----------
        y : array-like
            Target variable (e.g., realized volatility, volume)
        intraday_idx : array-like
            Intraday time index (0 to bins_per_day-1)

        Returns
        -------
        self
            Fitted DiurnalAdjuster instance
        """
        y = np.asarray(y)
        intraday_idx = np.asarray(intraday_idx)

        # Handle missing values
        valid_mask = ~np.isnan(y) & ~np.isinf(y)
        if self.use_log:
            valid_mask &= (y > 0)

        y_clean = y[valid_mask]
        idx_clean = intraday_idx[valid_mask]

        if len(y_clean) == 0:
            raise ValueError("No valid data points after filtering NaN/Inf/non-positive values")

        # Log transform if requested
        if self.use_log:
            y_clean = np.log(y_clean)

        # Create Fourier features
        X = self._create_fourier_features(idx_clean)

        # Fit regression
        if self.regularization > 0:
            self.model_ = Ridge(alpha=self.regularization, copy_X=False)
        else:
            self.model_ = LinearRegression(copy_X=False)

        self.model_.fit(X, y_clean)

        # Store coefficients
        self.coefficients_ = {'intercept': float(self.model_.intercept_)}
        for k in range(1, self.order + 1):
            self.coefficients_[f'sin_{k}'] = float(self.model_.coef_[2*(k-1)])
            self.coefficients_[f'cos_{k}'] = float(self.model_.coef_[2*(k-1) + 1])

        # Compute seasonal pattern for one full day
        full_day_idx = np.arange(self.bins_per_day)
        X_full = self._create_fourier_features(full_day_idx)
        self.seasonal_pattern_ = self.model_.predict(X_full)

        if self.use_log:
            # Convert back to original scale for visualization
            self.seasonal_pattern_exp_ = np.exp(self.seasonal_pattern_)
        else:
            self.seasonal_pattern_exp_ = self.seasonal_pattern_

        self.is_fitted = True
        return self

    def transform(
        self,
        y: Union[pd.Series, np.ndarray],
        intraday_idx: Union[pd.Series, np.ndarray],
        return_seasonal: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Remove the diurnal pattern from the data.

        Parameters
        ----------
        y : array-like
            Target variable to deseasonalize
        intraday_idx : array-like
            Intraday time index (0 to bins_per_day-1)
        return_seasonal : bool, default False
            If True, also return the estimated seasonal component

        Returns
        -------
        y_adj : np.ndarray
            Deseasonalized data
        seasonal : np.ndarray (optional)
            Estimated seasonal component (if return_seasonal=True)
        """
        if not self.is_fitted:
            raise RuntimeError("DiurnalAdjuster must be fitted before transform")

        y = np.asarray(y)
        intraday_idx = np.asarray(intraday_idx)

        # Predict seasonal component
        X = self._create_fourier_features(intraday_idx)
        seasonal = self.model_.predict(X)

        if self.use_log:
            # In log space: y_adj = log(y) - seasonal
            # Then exp(y_adj) = y / exp(seasonal)
            y_log = np.where(y > 0, np.log(y), np.nan)
            y_adj = y_log - seasonal
        else:
            # In raw space: y_adj = y - seasonal
            y_adj = y - seasonal

        if return_seasonal:
            return y_adj, seasonal
        return y_adj

    def fit_transform(
        self,
        y: Union[pd.Series, np.ndarray],
        intraday_idx: Union[pd.Series, np.ndarray],
        return_seasonal: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Fit the model and transform in one step.

        Parameters
        ----------
        y : array-like
            Target variable
        intraday_idx : array-like
            Intraday time index
        return_seasonal : bool, default False
            If True, also return the seasonal component

        Returns
        -------
        y_adj : np.ndarray
            Deseasonalized data
        seasonal : np.ndarray (optional)
            Seasonal component (if return_seasonal=True)
        """
        self.fit(y, intraday_idx)
        return self.transform(y, intraday_idx, return_seasonal)

    def get_seasonal_at_idx(self, intraday_idx: Union[int, np.ndarray]) -> np.ndarray:
        """Get the seasonal adjustment factor for specific intraday indices.

        Parameters
        ----------
        intraday_idx : int or array-like
            Intraday time index (0 to bins_per_day-1)

        Returns
        -------
        np.ndarray
            Seasonal factor(s). If use_log=True, these are multiplicative
            factors (exp of the log-space pattern).
        """
        if not self.is_fitted:
            raise RuntimeError("DiurnalAdjuster must be fitted first")

        intraday_idx = np.atleast_1d(intraday_idx)
        X = self._create_fourier_features(intraday_idx)
        seasonal_log = self.model_.predict(X)

        if self.use_log:
            return np.exp(seasonal_log)
        return seasonal_log


class RollingDiurnalAdjuster:
    """Rolling window FFF adjustment to avoid look-ahead bias.

    For trading applications, you should not fit the FFF on future data.
    This class maintains a rolling window of historical data to estimate
    the diurnal pattern.

    Parameters
    ----------
    bins_per_day : int
        Number of intervals per trading day
    lookback_days : int, default 20
        Number of historical days to use for fitting
    order : int, default 3
        Number of Fourier harmonic pairs
    use_log : bool, default True
        Work in log-space
    min_days : int, default 5
        Minimum days required before producing estimates

    Examples
    --------
    >>> adjuster = RollingDiurnalAdjuster(bins_per_day=78, lookback_days=20)
    >>> df['vol_adj'] = adjuster.fit_transform(
    ...     df['realized_vol'],
    ...     df['intraday_idx'],
    ...     df['date']
    ... )
    """

    def __init__(
        self,
        bins_per_day: int,
        lookback_days: int = 20,
        order: int = 3,
        use_log: bool = True,
        min_days: int = 5,
    ):
        self.bins_per_day = bins_per_day
        self.lookback_days = lookback_days
        self.order = order
        self.use_log = use_log
        self.min_days = min_days

    def fit_transform(
        self,
        y: pd.Series,
        intraday_idx: pd.Series,
        dates: Union[pd.Series, pd.DatetimeIndex],
        return_seasonal: bool = False,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """Apply rolling FFF adjustment.

        Parameters
        ----------
        y : pd.Series
            Target variable with datetime index
        intraday_idx : pd.Series
            Intraday time index (0 to bins_per_day-1)
        dates : pd.Series or pd.DatetimeIndex
            Date component (normalized dates)
        return_seasonal : bool, default False
            If True, also return the seasonal component

        Returns
        -------
        pd.Series or Tuple[pd.Series, pd.Series]
            Deseasonalized data, and optionally the seasonal component
        """
        # Convert to numpy for faster operations
        y_vals = y.values.astype(np.float64)
        idx_vals = intraday_idx.values.astype(np.int32)

        # Convert dates to integer day codes for fast comparison
        if isinstance(dates, pd.DatetimeIndex):
            date_vals = dates.normalize()
        elif isinstance(dates, pd.Series):
            date_vals = pd.DatetimeIndex(dates.values).normalize()
        else:
            date_vals = pd.DatetimeIndex(dates).normalize()

        # Get unique dates and create mapping
        unique_dates = np.sort(date_vals.unique())
        n_days = len(unique_dates)

        # Pre-compute day boundaries (start/end indices for each day)
        # This avoids repeated boolean mask creation
        date_codes = pd.Categorical(date_vals, categories=unique_dates).codes
        day_boundaries = []
        for day_idx in range(n_days):
            mask = date_codes == day_idx
            indices = np.where(mask)[0]
            if len(indices) > 0:
                day_boundaries.append((indices[0], indices[-1] + 1))
            else:
                day_boundaries.append((0, 0))

        # Pre-allocate result arrays
        result_vals = np.full(len(y_vals), np.nan, dtype=np.float64)
        seasonal_vals = np.full(len(y_vals), np.nan, dtype=np.float64) if return_seasonal else None

        # Reuse single adjuster instance
        adjuster = DiurnalAdjuster(
            bins_per_day=self.bins_per_day,
            order=self.order,
            use_log=self.use_log,
        )

        for i in range(self.min_days, n_days):
            # Get lookback window indices
            start_day = max(0, i - self.lookback_days)

            # Gather training data from lookback window
            train_slices = []
            for j in range(start_day, i):
                s, e = day_boundaries[j]
                if e > s:
                    train_slices.append((s, e))

            if not train_slices:
                continue

            # Concatenate training data efficiently
            train_y = np.concatenate([y_vals[s:e] for s, e in train_slices])
            train_idx = np.concatenate([idx_vals[s:e] for s, e in train_slices])

            # Current day boundaries
            curr_start, curr_end = day_boundaries[i]
            if curr_end <= curr_start:
                continue

            try:
                # Fit on training data
                adjuster.fit(train_y, train_idx)

                # Transform current day
                curr_y = y_vals[curr_start:curr_end]
                curr_idx = idx_vals[curr_start:curr_end]

                if return_seasonal:
                    y_adj, seas = adjuster.transform(curr_y, curr_idx, return_seasonal=True)
                    seasonal_vals[curr_start:curr_end] = seas
                else:
                    y_adj = adjuster.transform(curr_y, curr_idx)

                result_vals[curr_start:curr_end] = y_adj

            except Exception:
                # If fitting fails, leave as NaN
                continue

        # Convert back to Series
        result = pd.Series(result_vals, index=y.index)
        if return_seasonal:
            seasonal = pd.Series(seasonal_vals, index=y.index)
            return result, seasonal
        return result

    def batch_fit_transform(
        self,
        y_dict: Dict[str, pd.Series],
        intraday_idx: pd.Series,
        dates: Union[pd.Series, pd.DatetimeIndex],
        return_seasonal: bool = False,
    ) -> Dict[str, Union[pd.Series, Tuple[pd.Series, pd.Series]]]:
        """Apply rolling FFF adjustment to multiple series with single fit per day.

        Significantly faster than calling fit_transform multiple times when
        deseasonalizing related series (e.g., bid/ask/imbalance volumes).

        Parameters
        ----------
        y_dict : Dict[str, pd.Series]
            Dictionary of series to deseasonalize (e.g., {'bid': bid_vol, 'ask': ask_vol})
        intraday_idx : pd.Series
            Intraday time index (0 to bins_per_day-1)
        dates : pd.Series or pd.DatetimeIndex
            Date component (normalized dates)
        return_seasonal : bool, default False
            If True, also return the seasonal components

        Returns
        -------
        Dict[str, Union[pd.Series, Tuple[pd.Series, pd.Series]]]
            Dictionary with same keys, containing deseasonalized series
            (and seasonal if return_seasonal=True)
        """
        # Use first series as reference for index
        ref_key = next(iter(y_dict))
        ref_series = y_dict[ref_key]

        # Convert all to numpy
        y_arrays = {k: v.values.astype(np.float64) for k, v in y_dict.items()}
        idx_vals = intraday_idx.values.astype(np.int32)

        # Convert dates to integer day codes for fast comparison
        if isinstance(dates, pd.DatetimeIndex):
            date_vals = dates.normalize()
        elif isinstance(dates, pd.Series):
            date_vals = pd.DatetimeIndex(dates.values).normalize()
        else:
            date_vals = pd.DatetimeIndex(dates).normalize()

        # Get unique dates and create mapping
        unique_dates = np.sort(date_vals.unique())
        n_days = len(unique_dates)

        # Pre-compute day boundaries
        date_codes = pd.Categorical(date_vals, categories=unique_dates).codes
        day_boundaries = []
        for day_idx in range(n_days):
            mask = date_codes == day_idx
            indices = np.where(mask)[0]
            if len(indices) > 0:
                day_boundaries.append((indices[0], indices[-1] + 1))
            else:
                day_boundaries.append((0, 0))

        # Pre-allocate result arrays for all series
        n_samples = len(ref_series)
        result_arrays = {k: np.full(n_samples, np.nan, dtype=np.float64) for k in y_dict}
        seasonal_arrays = {k: np.full(n_samples, np.nan, dtype=np.float64) for k in y_dict} if return_seasonal else None

        # Reuse single adjuster instance
        adjuster = DiurnalAdjuster(
            bins_per_day=self.bins_per_day,
            order=self.order,
            use_log=self.use_log,
        )

        for i in range(self.min_days, n_days):
            start_day = max(0, i - self.lookback_days)

            # Gather training data from lookback window
            train_slices = []
            for j in range(start_day, i):
                s, e = day_boundaries[j]
                if e > s:
                    train_slices.append((s, e))

            if not train_slices:
                continue

            # Use reference series to fit (all series share same seasonal pattern)
            train_y = np.concatenate([y_arrays[ref_key][s:e] for s, e in train_slices])
            train_idx = np.concatenate([idx_vals[s:e] for s, e in train_slices])

            curr_start, curr_end = day_boundaries[i]
            if curr_end <= curr_start:
                continue

            try:
                # Fit ONCE on reference series
                adjuster.fit(train_y, train_idx)

                # Transform ALL series using the same fitted model
                curr_idx = idx_vals[curr_start:curr_end]

                for key, y_arr in y_arrays.items():
                    curr_y = y_arr[curr_start:curr_end]

                    if return_seasonal:
                        y_adj, seas = adjuster.transform(curr_y, curr_idx, return_seasonal=True)
                        seasonal_arrays[key][curr_start:curr_end] = seas
                    else:
                        y_adj = adjuster.transform(curr_y, curr_idx)

                    result_arrays[key][curr_start:curr_end] = y_adj

            except Exception:
                continue

        # Convert back to Series
        results = {}
        for key in y_dict:
            result_series = pd.Series(result_arrays[key], index=ref_series.index)
            if return_seasonal:
                seasonal_series = pd.Series(seasonal_arrays[key], index=ref_series.index)
                results[key] = (result_series, seasonal_series)
            else:
                results[key] = result_series

        return results


def fft_spectrum(
    signal: np.ndarray,
    sampling_rate: float,
    max_frequency: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT spectrum to identify dominant cycles.

    Use this to discover periodic patterns before applying FFF adjustment.
    A spike at frequency=1.0 indicates a daily cycle.

    Parameters
    ----------
    signal : np.ndarray
        Time series data
    sampling_rate : float
        Samples per cycle unit. E.g., bins_per_day for daily cycles.
    max_frequency : float, optional
        Maximum frequency to return. Useful for focusing on low frequencies.

    Returns
    -------
    frequencies : np.ndarray
        Frequency values (cycles per unit)
    amplitudes : np.ndarray
        Amplitude at each frequency

    Examples
    --------
    >>> # Analyze for daily cycles (sampling_rate = bars_per_day)
    >>> freqs, amps = fft_spectrum(df['log_vol'].values, sampling_rate=78)
    >>> # Spike at freq=1.0 indicates daily cycle
    """
    N = len(signal)

    # Handle NaN values
    signal_clean = np.nan_to_num(signal, nan=np.nanmean(signal))

    # Compute FFT
    yf = fft(signal_clean)
    xf = fftfreq(N, 1 / sampling_rate)

    # Get positive frequencies only
    positive_mask = xf >= 0
    frequencies = xf[positive_mask]
    amplitudes = 2.0 / N * np.abs(yf[positive_mask])

    if max_frequency is not None:
        freq_mask = frequencies <= max_frequency
        frequencies = frequencies[freq_mask]
        amplitudes = amplitudes[freq_mask]

    return frequencies, amplitudes


def compute_realized_volatility(
    returns: pd.Series,
    window: Optional[int] = None,
    annualize: bool = False,
    trading_days: int = 252,
) -> pd.Series:
    """Compute realized volatility from returns.

    Parameters
    ----------
    returns : pd.Series
        Return series (can be intraday)
    window : int, optional
        Rolling window size. If None, computes for entire series.
    annualize : bool, default False
        Whether to annualize the volatility
    trading_days : int, default 252
        Trading days per year (for annualization)

    Returns
    -------
    pd.Series
        Realized volatility
    """
    if window is not None:
        rv = returns.pow(2).rolling(window).sum().apply(np.sqrt)
    else:
        rv = returns.pow(2).groupby(returns.index.normalize()).sum().apply(np.sqrt)

    if annualize:
        # Estimate number of observations per year
        if window is not None:
            obs_per_day = len(returns) / len(returns.index.normalize().unique())
            obs_per_year = obs_per_day * trading_days
            rv = rv * np.sqrt(obs_per_year / window)

    return rv


def deseasonalize_volatility(
    volatility: pd.Series,
    intraday_idx: Optional[pd.Series] = None,
    bins_per_day: Optional[int] = None,
    order: int = 3,
    use_log: bool = True,
    rolling_days: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience function to deseasonalize volatility.

    Parameters
    ----------
    volatility : pd.Series
        Realized volatility series with DatetimeIndex
    intraday_idx : pd.Series, optional
        Intraday time index. If None, computed from volatility index.
    bins_per_day : int, optional
        Number of intervals per day. If None, estimated from data.
    order : int, default 3
        FFF order (number of harmonic pairs)
    use_log : bool, default True
        Work in log-space
    rolling_days : int, optional
        If provided, use rolling window adjustment (avoids look-ahead bias).
        If None, fit on entire dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'original': Original volatility
        - 'seasonal': Estimated seasonal component
        - 'adjusted': Deseasonalized volatility

    Examples
    --------
    >>> result = deseasonalize_volatility(df['rv'], rolling_days=20)
    >>> df['rv_adj'] = result['adjusted']
    """
    if not isinstance(volatility.index, pd.DatetimeIndex):
        raise ValueError("volatility must have DatetimeIndex")

    # Estimate bins_per_day if not provided
    if bins_per_day is None:
        dates = volatility.index.normalize()
        counts = dates.value_counts()
        bins_per_day = int(counts.median())

    # Compute intraday_idx if not provided
    if intraday_idx is None:
        # Compute within-day position
        df_temp = pd.DataFrame({'vol': volatility})
        df_temp['date'] = df_temp.index.normalize()
        df_temp['intraday_idx'] = df_temp.groupby('date').cumcount()
        intraday_idx = df_temp['intraday_idx']

    # Ensure intraday_idx is a Series with the same index
    if not isinstance(intraday_idx, pd.Series):
        intraday_idx = pd.Series(intraday_idx, index=volatility.index)
    elif not intraday_idx.index.equals(volatility.index):
        intraday_idx = pd.Series(intraday_idx.values, index=volatility.index)

    result = pd.DataFrame({
        'original': volatility,
        'seasonal': np.nan,
        'adjusted': np.nan,
    }, index=volatility.index)

    if rolling_days is not None:
        # Rolling window adjustment
        dates = pd.Series(volatility.index.normalize(), index=volatility.index)
        adjuster = RollingDiurnalAdjuster(
            bins_per_day=bins_per_day,
            lookback_days=rolling_days,
            order=order,
            use_log=use_log,
        )
        adjusted, seasonal = adjuster.fit_transform(
            volatility, intraday_idx, dates, return_seasonal=True
        )
        result['adjusted'] = adjusted
        result['seasonal'] = seasonal
    else:
        # Fit on entire dataset
        adjuster = DiurnalAdjuster(
            bins_per_day=bins_per_day,
            order=order,
            use_log=use_log,
        )
        y_adj, seasonal = adjuster.fit_transform(
            volatility.values,
            intraday_idx.values,
            return_seasonal=True,
        )
        result['adjusted'] = y_adj
        result['seasonal'] = seasonal

    return result


def deseasonalize_volume(
    volume: pd.Series,
    intraday_idx: Optional[pd.Series] = None,
    bins_per_day: Optional[int] = None,
    order: int = 4,
    rolling_days: Optional[int] = None,
    bid_ask_volume: bool = False,
    bid_volume: Optional[pd.Series] = None,
    ask_volume: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Convenience function to deseasonalize volume.

    Volume typically has stronger U-shape patterns than volatility,
    so a higher order (4) is used by default.

    Parameters
    ----------
    volume : pd.Series
        Volume series with DatetimeIndex (used when bid_ask_volume=False)
    intraday_idx : pd.Series, optional
        Intraday time index
    bins_per_day : int, optional
        Intervals per day
    order : int, default 4
        FFF order
    rolling_days : int, optional
        Rolling window size (None = fit on all data)
    bid_ask_volume : bool, default False
        If True, deseasonalize bid and ask volumes separately plus their difference.
        Requires bid_volume and ask_volume parameters.
    bid_volume : pd.Series, optional
        Bid volume series (required when bid_ask_volume=True)
    ask_volume : pd.Series, optional
        Ask volume series (required when bid_ask_volume=True)

    Returns
    -------
    pd.DataFrame
        If bid_ask_volume=False:
            DataFrame with 'original', 'seasonal', 'adjusted' columns
        If bid_ask_volume=True:
            DataFrame with columns for bid, ask, and imbalance:
            - 'bid_original', 'bid_seasonal', 'bid_adjusted'
            - 'ask_original', 'ask_seasonal', 'ask_adjusted'
            - 'imbalance_original', 'imbalance_seasonal', 'imbalance_adjusted'
    """
    if bid_ask_volume:
        if bid_volume is None or ask_volume is None:
            raise ValueError("bid_volume and ask_volume must be provided when bid_ask_volume=True")

        # Calculate volume imbalance (absolute difference)
        imbalance = (bid_volume - ask_volume).abs()

        # Estimate bins_per_day if not provided
        if bins_per_day is None:
            dates = bid_volume.index.normalize()
            counts = dates.value_counts()
            bins_per_day = int(counts.median())

        # Compute intraday_idx if not provided
        if intraday_idx is None:
            df_temp = pd.DataFrame({'vol': bid_volume})
            df_temp['date'] = df_temp.index.normalize()
            df_temp['intraday_idx'] = df_temp.groupby('date').cumcount()
            intraday_idx = df_temp['intraday_idx']

        # Ensure intraday_idx is a Series with the same index
        if not isinstance(intraday_idx, pd.Series):
            intraday_idx = pd.Series(intraday_idx, index=bid_volume.index)
        elif not intraday_idx.index.equals(bid_volume.index):
            intraday_idx = pd.Series(intraday_idx.values, index=bid_volume.index)

        if rolling_days is not None:
            # Use batch method for 3x speedup with rolling window
            dates = pd.Series(bid_volume.index.normalize(), index=bid_volume.index)
            adjuster = RollingDiurnalAdjuster(
                bins_per_day=bins_per_day,
                lookback_days=rolling_days,
                order=order,
                use_log=True,
            )

            # Batch transform all three series with single fit per day
            batch_results = adjuster.batch_fit_transform(
                y_dict={'bid': bid_volume, 'ask': ask_volume, 'imbalance': imbalance},
                intraday_idx=intraday_idx,
                dates=dates,
                return_seasonal=True,
            )

            # Unpack results
            bid_adjusted, bid_seasonal = batch_results['bid']
            ask_adjusted, ask_seasonal = batch_results['ask']
            imbalance_adjusted, imbalance_seasonal = batch_results['imbalance']

            combined = pd.DataFrame(index=bid_volume.index)
            combined['bid_original'] = bid_volume
            combined['bid_seasonal'] = bid_seasonal
            combined['bid_adjusted'] = bid_adjusted
            combined['ask_original'] = ask_volume
            combined['ask_seasonal'] = ask_seasonal
            combined['ask_adjusted'] = ask_adjusted
            combined['imbalance_original'] = imbalance
            combined['imbalance_seasonal'] = imbalance_seasonal
            combined['imbalance_adjusted'] = imbalance_adjusted

        else:
            # Non-rolling: use existing method (fit on all data)
            adjuster = DiurnalAdjuster(
                bins_per_day=bins_per_day,
                order=order,
                use_log=True,
            )

            # Fit on bid volume (representative of volume pattern)
            adjuster.fit(bid_volume.values, intraday_idx.values)

            # Transform all series with same fitted model
            bid_adj, bid_seas = adjuster.transform(bid_volume.values, intraday_idx.values, return_seasonal=True)
            ask_adj, ask_seas = adjuster.transform(ask_volume.values, intraday_idx.values, return_seasonal=True)
            imb_adj, imb_seas = adjuster.transform(imbalance.values, intraday_idx.values, return_seasonal=True)

            combined = pd.DataFrame(index=bid_volume.index)
            combined['bid_original'] = bid_volume
            combined['bid_seasonal'] = bid_seas
            combined['bid_adjusted'] = bid_adj
            combined['ask_original'] = ask_volume
            combined['ask_seasonal'] = ask_seas
            combined['ask_adjusted'] = ask_adj
            combined['imbalance_original'] = imbalance
            combined['imbalance_seasonal'] = imb_seas
            combined['imbalance_adjusted'] = imb_adj

        return combined

    else:
        return deseasonalize_volatility(
            volume,
            intraday_idx=intraday_idx,
            bins_per_day=bins_per_day,
            order=order,
            use_log=True,  # Volume is strictly positive
            rolling_days=rolling_days,
        )


def estimate_diurnal_pattern(
    data: pd.Series,
    bins_per_day: int,
    order: int = 3,
    use_log: bool = True,
) -> pd.DataFrame:
    """Estimate and return the average diurnal pattern.

    Useful for visualization and understanding the seasonal structure.

    Parameters
    ----------
    data : pd.Series
        Intraday data with DatetimeIndex
    bins_per_day : int
        Number of intervals per day
    order : int, default 3
        FFF order
    use_log : bool, default True
        Work in log-space

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by intraday position (0 to bins_per_day-1) with:
        - 'empirical': Average value at each time (simple mean)
        - 'fff_smooth': FFF-smoothed pattern
    """
    # Compute intraday index
    df = pd.DataFrame({'value': data})
    df['date'] = df.index.normalize()
    df['intraday_idx'] = df.groupby('date').cumcount() % bins_per_day

    # Empirical average by time
    if use_log:
        df['log_value'] = np.log(df['value'].clip(lower=1e-10))
        empirical = df.groupby('intraday_idx')['log_value'].mean()
        empirical_orig = np.exp(empirical)
    else:
        empirical = df.groupby('intraday_idx')['value'].mean()
        empirical_orig = empirical

    # FFF smooth pattern
    adjuster = DiurnalAdjuster(
        bins_per_day=bins_per_day,
        order=order,
        use_log=use_log,
    )
    adjuster.fit(data.values, df['intraday_idx'].values)

    result = pd.DataFrame({
        'empirical': empirical_orig,
        'fff_smooth': adjuster.seasonal_pattern_exp_,
    }, index=range(bins_per_day))
    result.index.name = 'intraday_idx'

    return result
