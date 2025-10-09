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
from dataclasses import dataclass
from typing import Dict, Any

from ..data.data_client import DataClient
from ..utils.seasonal import intraday_autocorr_between_times, intraday_lag_autocorr, monthly_returns, abnormal_months, IntradayPairSpec, last_year_predicts_this_year, prewindow_predicts_month

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


