"""Seasonal and normalization utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from typing import Tuple, Dict

from scipy import stats
from dataclasses import dataclass


@dataclass
class IntradayPairSpec:
    """
    Defines two points or windows within a day to measure correlation.

    Examples:
      - time_a/time_b: Use a single-bar return at those times (per day).
      - If window_* > 1, aggregate consecutive bars starting at the time.
    """
    time_a: str  # e.g., "09:30", "13:45" (HH:MM in local tz of index)
    time_b: str  # e.g., "10:00" or next period's start
    window_a: int = 1  # number of bars to aggregate for A
    window_b: int = 1  # number of bars to aggregate for B


def log_returns(series: pd.Series) -> pd.Series:
    """Calculate log returns from a price series."""
    return np.log(series).diff()


def tod_mask(idx: pd.DatetimeIndex, hhmm: str) -> pd.Series:
    """Create boolean mask for specific time-of-day."""
    hh, mm = map(int, hhmm.split(":"))
    return (idx.hour == hh) & (idx.minute == mm)


def aggregate_window(ret: pd.Series, start_mask: pd.Series, window: int) -> pd.Series:
    """
    Aggregate window returns (log-additive) starting at masked bars.
    Returns a Series indexed by the start timestamp.
    """
    starts = ret.index[start_mask]
    out_idx = []
    out_vals = []
    for t in starts:
        # Slice next `window` bars including t
        sl = ret.loc[t:]
        vals = sl.iloc[:window]
        if len(vals) == window and not np.isnan(vals).any():
            out_idx.append(t)
            out_vals.append(np.nansum(vals))
    return pd.Series(out_vals, index=out_idx)


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


def intraday_autocorr_between_times(
    df: pd.DataFrame,
    price_col: str,
    spec: IntradayPairSpec
) -> Dict[str, float]:
    """
    Compute correlation between per-day window returns at two times-of-day.

    Steps:
      1) Compute log returns for all bars.
      2) Build daily series for A and B windows using time-of-day anchors.
      3) Inner-join on dates; report Pearson r and t-stat.

    Returns:
    --------
    Dict[str, float]
        Dictionary with keys: n, r, t_stat, p_value
    """
    px = df[price_col]
    r = log_returns(px).dropna()

    mA = tod_mask(r.index, spec.time_a)
    mB = tod_mask(r.index, spec.time_b)

    RA = aggregate_window(r, mA, spec.window_a)
    RB = aggregate_window(r, mB, spec.window_b)

    # Map to dates to ensure same session comparison
    A = RA.groupby(RA.index.date).sum()
    B = RB.groupby(RB.index.date).sum()
    J = pd.concat([A.rename("A"), B.rename("B")], axis=1).dropna()

    if len(J) < 10:
        return {"n": float(len(J)), "r": np.nan, "t_stat": np.nan, "p_value": np.nan}

    r_val, p_val = stats.pearsonr(J["A"], J["B"])
    # t-stat for correlation (large-sample approximation)
    n = len(J)
    t_stat = r_val * math.sqrt((n - 2) / (1 - r_val**2)) if abs(r_val) < 1 else np.inf
    return {"n": float(n), "r": float(r_val), "t_stat": float(t_stat), "p_value": float(p_val)}


def intraday_lag_autocorr(df: pd.DataFrame, price_col: str, k: int = 1) -> Dict[str, float]:
    """
    Classic lag-k autocorrelation of intraday log-returns.
    Use k>0 for forward correlation of r_t with r_{t+k}.

    Returns:
    --------
    Dict[str, float]
        Dictionary with keys: n, r, t_stat, p_value
    """
    r = log_returns(df[price_col]).dropna()
    X = r.iloc[:-k]
    Y = r.iloc[k:]
    if len(Y) < 100:
        return {"n": float(len(Y)), "r": np.nan, "t_stat": np.nan, "p_value": np.nan}
    r_val, p_val = stats.pearsonr(X.values, Y.values)
    n = len(Y)
    t_stat = r_val * math.sqrt((n - 2) / (1 - r_val**2)) if abs(r_val) < 1 else np.inf
    return {"n": float(n), "r": float(r_val), "t_stat": float(t_stat), "p_value": float(p_val)}


def monthly_returns(
    df: pd.DataFrame,
    price_col: str,
    use_log_returns: bool = True,
    month_agg: str = "M"
) -> pd.Series:
    """
    Calculate monthly returns from price data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data indexed by datetime
    price_col : str
        Column name for prices
    use_log_returns : bool
        If True, use log returns; otherwise simple returns
    month_agg : str
        'M' for month-end, 'MS' for month-start

    Returns:
    --------
    pd.Series
        Monthly returns indexed by month boundary dates
    """
    px = df[price_col].asfreq(None)  # keep original; resample at monthly boundaries
    monthly_px = px.resample(month_agg).last().dropna()
    if use_log_returns:
        mr = np.log(monthly_px).diff()
    else:
        mr = monthly_px.pct_change()
    return mr.dropna().rename("mret")


def abnormal_months(mr: pd.Series) -> pd.DataFrame:
    """
    Compute per-month mean return, t-stat vs overall mean, and z-score.

    Parameters:
    -----------
    mr : pd.Series
        Monthly returns series

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by month (1..12) with columns:
        - n: number of observations
        - mean: average monthly return
        - std: standard deviation
        - t_stat: t-statistic vs overall mean
        - p_value: p-value for t-test
        - z_score: z-score vs overall distribution
    """
    df = mr.to_frame()
    df["month"] = df.index.month
    overall_mu = df["mret"].mean()
    overall_sd = df["mret"].std(ddof=1)

    rows = []
    for m, g in df.groupby("month"):
        mu = g["mret"].mean()
        sd = g["mret"].std(ddof=1)
        n = len(g)
        # One-sample t-test against overall mean
        if n > 1 and not np.isnan(sd) and sd > 0:
            t_stat = (mu - overall_mu) / (sd / math.sqrt(n))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        else:
            t_stat, p_val = np.nan, np.nan
        z = (mu - overall_mu) / overall_sd if overall_sd and not np.isnan(overall_sd) and overall_sd > 0 else np.nan
        rows.append({"month": m, "n": n, "mean": mu, "std": sd, "t_stat": t_stat, "p_value": p_val, "z_score": z})

    out = pd.DataFrame(rows).set_index("month").sort_index()
    return out


def last_year_predicts_this_year(mr: pd.Series) -> pd.DataFrame:
    """
    For each calendar month, regress current year's month return on last year's same-month return.

    Parameters:
    -----------
    mr : pd.Series
        Monthly returns series

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by month with columns: n, slope, r, p_value
    """
    df = mr.to_frame()
    df["month"] = df.index.month
    df["year"] = df.index.year

    results = []
    for m, g in df.groupby("month"):
        g = g.sort_index()
        # Align by year: y_t on y_{t-1} for same month
        y = g["mret"].iloc[1:].values
        x = g["mret"].shift(1).dropna().values
        n = min(len(x), len(y))
        if n < 8:
            results.append({"month": m, "n": n, "slope": np.nan, "r": np.nan, "p_value": np.nan})
            continue
        slope, intercept, r_val, p_val, stderr = stats.linregress(x[:n], y[:n])
        results.append({"month": m, "n": n, "slope": slope, "r": r_val, "p_value": p_val})

    return pd.DataFrame(results).set_index("month").sort_index()


def prewindow_feature(
    df: pd.DataFrame,
    month_agg: str = "M",
    start_days_before: int = 90,
    end_days_before: int = 60,
    price_col: str = "Close",
    use_log_returns: bool = True
) -> pd.DataFrame:
    """
    Compute a feature as the cumulative return in the pre-window [T-90, T-60] days
    before each month's start (or end depending on month_agg).

    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    month_agg : str
        'M' or 'MS' for month boundary
    start_days_before : int
        Start of pre-window (days before month boundary)
    end_days_before : int
        End of pre-window (days before month boundary)
    price_col : str
        Price column name
    use_log_returns : bool
        Use log returns if True

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by month-end (or start) with columns:
          - mret: the month return
          - prewin: cumulative log return in the pre-window
    """
    px = df[price_col]
    mr = monthly_returns(df, price_col=price_col, use_log_returns=use_log_returns, month_agg=month_agg)

    boundary = mr.index  # monthly timestamps
    r_daily = np.log(px).diff().dropna() if use_log_returns else px.pct_change().dropna()
    daily = r_daily.asfreq("D").fillna(0.0)  # approximate daily calendar agg

    prevals = []
    for t in boundary:
        start = (t - pd.Timedelta(days=start_days_before)).normalize()
        end = (t - pd.Timedelta(days=end_days_before)).normalize()
        window = daily.loc[start:end]
        if len(window) == 0:
            prevals.append(np.nan)
        else:
            prevals.append(window.sum())  # log returns add

    feat = pd.DataFrame({"mret": mr.values, "prewin": prevals}, index=boundary)
    return feat.dropna()


def prewindow_predicts_month(
    df: pd.DataFrame,
    price_col: str = "Close",
    use_log_returns: bool = True,
    month_agg: str = "M"
) -> Dict[str, float]:
    """
    Test if pre-window (60-90 days before) returns predict monthly returns.

    Returns:
    --------
    Dict[str, float]
        Dictionary with keys: n, slope, r, p_value
    """
    feat = prewindow_feature(df, month_agg=month_agg, price_col=price_col, use_log_returns=use_log_returns)
    y = feat["mret"].values
    x = feat["prewin"].values
    if len(x) < 24:
        return {"n": float(len(x)), "slope": np.nan, "r": np.nan, "p_value": np.nan}
    slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
    return {"n": float(len(x)), "slope": float(slope), "r": float(r_val), "p_value": float(p_val)}
