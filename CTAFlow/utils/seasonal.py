"""Seasonal and normalization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def deseasonalize_monthly(data: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Remove simple monthly seasonal component from a data matrix.

    Parameters
    ----------
    data : np.ndarray
        Matrix with shape ``(n_dates, n_features)``.
    dates : pd.DatetimeIndex
        Corresponding datetime index for ``data``.
    """
    if data.size == 0:
        return data

    df = pd.DataFrame(data, index=dates)
    result = np.full_like(data, np.nan, dtype=float)
    for col in df.columns:
        series = df[col]
        if series.notna().any():
            monthly_means = series.groupby(series.index.month).transform('mean')
            result[:, col] = (series - monthly_means).values
    return result


def zscore_normalize(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score normalize an array along a given axis."""
    mean = np.nanmean(data, axis=axis, keepdims=True)
    std = np.nanstd(data, axis=axis, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def _kalman_filter_1d(series: np.ndarray, process_var: float, obs_var: float) -> np.ndarray:
    n = len(series)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhatminus = np.zeros(n)
    Pminus = np.zeros(n)
    K = np.zeros(n)

    xhat[0] = series[0]
    P[0] = 1.0

    for k in range(1, n):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + process_var
        K[k] = Pminus[k] / (Pminus[k] + obs_var)
        xhat[k] = xhatminus[k] + K[k] * (series[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat


class SeasonalAnalysis:
    """Utility class for seasonality handling and diagnostics."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._deseasonalized = None
        self._monthly_means = None

    def deseasonalize(self) -> pd.DataFrame:
        if self._deseasonalized is None:
            deseason = deseasonalize_monthly(self.data.values, self.data.index)
            self._deseasonalized = pd.DataFrame(deseason, index=self.data.index, columns=self.data.columns)
        return self._deseasonalized

    def kalman_filter(self, process_var: float = 1e-5, obs_var: float = 1e-1) -> pd.DataFrame:
        filtered = {}
        for col in self.data.columns:
            series = self.data[col].to_numpy(dtype=float)
            mask = np.isfinite(series)
            if not mask.any():
                filtered[col] = series
                continue
            filled = pd.Series(series).ffill().bfill().to_numpy()
            filtered_series = _kalman_filter_1d(filled, process_var, obs_var)
            filtered_series[~mask] = np.nan
            filtered[col] = filtered_series
        return pd.DataFrame(filtered, index=self.data.index)

    def deseasonalized_pca(self, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        data = self.deseasonalize().dropna()
        if data.empty:
            return np.empty((0, n_components)), np.empty((data.shape[1], n_components))
        matrix = data.values
        matrix -= matrix.mean(axis=0, keepdims=True)
        cov = np.cov(matrix, rowvar=False)
        cov = np.atleast_2d(cov)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx][:, :n_components]
        scores = matrix @ eigvecs
        return scores, eigvecs

    def fit_seasonal_model(self) -> None:
        self._monthly_means = self.data.groupby(self.data.index.month).mean()

    def test_seasonal_model(self) -> float:
        if self._monthly_means is None:
            raise RuntimeError("Call fit_seasonal_model before testing.")
        month_idx = self.data.index.month
        preds = self._monthly_means.reindex(month_idx).to_numpy()
        diff = self.data.to_numpy() - preds
        return float(np.sqrt(np.nanmean(diff ** 2)))
