"""
Integrated Feature Engineering Framework for Commodity Futures

Combines:
1. Intraday microstructure features (volume, realized variance, delta)
2. Seasonal anomaly detection and monthly cyclicity analysis
3. Intraday autocorrelation patterns
4. Pre-window predictive features

Author: Quant Commodities Engineer
"""

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import warnings
from scipy import interpolate, stats
from scipy.stats import skew, kurtosis
import calendar

from ..data.data_client import DataClient

# Dataloading class
dclient = DataClient()

# Standard futures month code mappings (will be imported from futures_curve_manager)
MONTH_CODE_MAP = {
    'F': 1,  # January
    'G': 2,  # February
    'H': 3,  # March
    'J': 4,  # April
    'K': 5,  # May
    'M': 6,  # June
    'N': 7,  # July
    'Q': 8,  # August
    'U': 9,  # September
    'V': 10,  # October
    'X': 11,  # November
    'Z': 12  # December
}

MONTH_CODE_ORDER = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']


class IntradayFeatures:

    def __init__(self, ticker_symbol, close_col="Last", bid_volume="BidVolume", ask_volume="AskVolume", volume="Volume"):

        print('Loading Intraday Data')
        self.data = dclient.query_market_data(ticker_symbol)
        self.returns = np.log(self.data[close_col]) - np.log(self.data[close_col].shift(1))
        self.buy_vol = self.data[ask_volume]
        self.sell_vol = self.data[bid_volume]
        self.volume = self.data[volume]

        return

    def historical_rv(self, window=21, average=True, annualize=False):
        returns = self.returns

        dr = pd.bdate_range(returns.index.date[0], returns.index.date[-1])
        rv = np.zeros(len(dr))
        rv[:] = np.nan
        if average:
            denom = window
        else:
            denom = 1

        for idx in range(window, len(dr)):
            d_start = dr[idx - window]
            d_end = dr[idx]

            rv[idx] = np.sqrt(np.sum(returns.loc[d_start:d_end] ** 2)) / denom

        if annualize:
            rv *= np.sqrt(252)

        hrv = pd.Series(data=rv,
                        index=dr,
                        name=f'{window}_rv')

        return hrv

    def realized_semivariance(self,window=1, average=True):
        returns = self.returns
        data = []
        dr = pd.bdate_range(returns.index.date[0], returns.index.date[-1])
        denom = 1
        if average:
            denom = window

        for idx in range(window, len(dr)):
            start = dr[idx - window]
            end = dr[idx]
            rets = returns.loc[start:end]
            rs_neg = np.sqrt((rets[rets < 0].sum() ** 2)) / denom
            rs_pos = np.sqrt((rets[rets > 0].sum() ** 2)) / denom
            data.append((end, rs_pos, rs_neg))

        # Create a DataFrame from collected rows
        rs_df = pd.DataFrame(data, columns=['date', 'RS_pos', 'RS_neg'])
        rs_df.set_index('date', inplace=True)
        return rs_df

    def cumulative_delta(self, window=1):
        """Calculate cumulative delta (buy volume - sell volume) over a rolling window

        Args:
            window: Number of days to calculate cumulative delta over

        Returns:
            pd.Series: Cumulative delta values indexed by date
        """
        returns = self.returns
        buy_vol = self.buy_vol
        sell_vol = self.sell_vol

        # Get unique dates from the index
        dr = pd.bdate_range(returns.index.date[0], returns.index.date[-1])
        cumulative_delta = np.zeros(len(dr))
        cumulative_delta[:] = np.nan

        for idx in range(window, len(dr)):
            d_start = dr[idx - window]
            d_end = dr[idx]

            # Calculate delta for the window period
            buy_volume_window = buy_vol.loc[d_start:d_end].sum()
            sell_volume_window = sell_vol.loc[d_start:d_end].sum()
            cumulative_delta[idx] = buy_volume_window - sell_volume_window

        cd_series = pd.Series(data=cumulative_delta,
                             index=dr,
                             name=f'{window}d_cumulative_delta')

        return cd_series


# ============================================================================
# Seasonal Anomaly Detection & Intraday Autocorrelation
# ============================================================================

def log_returns(series: pd.Series) -> pd.Series:
    """Calculate log returns from a price series."""
    return np.log(series).diff()


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


@dataclass
class SeasonalSettings:
    """Settings for seasonal analysis."""
    price_col: str = "Close"
    use_log_returns: bool = True
    month_agg: str = "M"  # pandas offset alias: 'M' month-end, 'MS' month-start


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


# ============================================================================
# Monthly Seasonality Scanner
# ============================================================================

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


# ============================================================================
# Seasonal Analysis Class (Unified Interface)
# ============================================================================

class SeasonalAnalyzer:
    """
    Unified interface for seasonal anomaly detection and intraday autocorrelation analysis.

    Combines monthly seasonality scanning with intraday pattern detection.
    """

    def __init__(self, df: pd.DataFrame, price_col: str = "Close"):
        """
        Initialize SeasonalAnalyzer.

        Parameters:
        -----------
        df : pd.DataFrame
            Price data indexed by datetime
        price_col : str
            Column name for prices
        """
        self.df = df
        self.price_col = price_col

    def analyze_intraday_patterns(
        self,
        time_a: str = "09:30",
        time_b: str = "10:00",
        window_a: int = 1,
        window_b: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze intraday autocorrelation patterns.

        Parameters:
        -----------
        time_a : str
            First time window (HH:MM format)
        time_b : str
            Second time window (HH:MM format)
        window_a : int
            Number of bars for first window
        window_b : int
            Number of bars for second window

        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary with 'between_times' and 'lag_1' autocorrelation results
        """
        spec = IntradayPairSpec(time_a=time_a, time_b=time_b, window_a=window_a, window_b=window_b)
        between = intraday_autocorr_between_times(self.df, self.price_col, spec)
        lag1 = intraday_lag_autocorr(self.df, self.price_col, k=1)

        return {"between_times": between, "lag_1": lag1}

    def analyze_monthly_seasonality(
        self,
        use_log_returns: bool = True,
        month_agg: str = "M"
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze monthly seasonality patterns.

        Parameters:
        -----------
        use_log_returns : bool
            Use log returns if True
        month_agg : str
            'M' for month-end, 'MS' for month-start

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with 'abnormal_months' and 'year_over_year' DataFrames
        """
        mr = monthly_returns(self.df, self.price_col, use_log_returns, month_agg)
        abnormal = abnormal_months(mr)
        yoy = last_year_predicts_this_year(mr)

        return {"abnormal_months": abnormal, "year_over_year": yoy, "monthly_returns": mr}

    def analyze_prewindow_predictive(
        self,
        use_log_returns: bool = True,
        month_agg: str = "M"
    ) -> Dict[str, float]:
        """
        Test if 60-90 day pre-window returns predict monthly performance.

        Returns:
        --------
        Dict[str, float]
            Dictionary with n, slope, r, p_value
        """
        return prewindow_predicts_month(self.df, self.price_col, use_log_returns, month_agg)

    def full_seasonal_report(
        self,
        use_log_returns: bool = True,
        month_agg: str = "M",
        include_intraday: bool = False
    ) -> Dict[str, Any]:
        """
        Generate comprehensive seasonal analysis report.

        Parameters:
        -----------
        use_log_returns : bool
            Use log returns for monthly analysis
        month_agg : str
            Monthly aggregation method
        include_intraday : bool
            Include intraday autocorrelation if True

        Returns:
        --------
        Dict[str, Any]
            Comprehensive report with all seasonal metrics
        """
        report = {}

        # Monthly seasonality
        monthly_analysis = self.analyze_monthly_seasonality(use_log_returns, month_agg)
        report["monthly_seasonality"] = monthly_analysis

        # Pre-window predictive power
        report["prewindow_predictive"] = self.analyze_prewindow_predictive(use_log_returns, month_agg)

        # Intraday patterns (optional)
        if include_intraday:
            report["intraday_patterns"] = self.analyze_intraday_patterns()

        return report


