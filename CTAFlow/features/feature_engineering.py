"""
Integrated Feature Engineering Framework for Commodity Futures

Combines:
1. Intraday microstructure features (volume, realized variance, delta)
2. Seasonal anomaly detection and monthly cyclicity analysis
3. Intraday autocorrelation patterns
4. Pre-window predictive features

Author: Quant Commodities Engineer
"""
import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Literal, Optional

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

class IntradaySignals:

    def __init__(self, data, open="Open", high="High", low="Low", close="Close"):
        self.data = data
        self.open, self.high, self.low, self.close = open, high, low, close

        return

    def intraday_breakout_hl(self, time_start:datetime.time, time_end:datetime.time, bidirectional:bool=True, bias:Literal["long", "short"]="long"):
        """Calculate intraday breakout signals based on high/low range during specified time window.

        For each day, computes the high and low within the time_start to time_end window.
        Then generates signals for the rest of the day:
        - Returns 1 when price breaks above the high
        - Returns -1 when price breaks below the low
        - Returns 0 when price remains within the range

        Parameters
        ----------
        time_start : datetime.time
            Start time of the range calculation window
        time_end : datetime.time
            End time of the range calculation window
        bidirectional : bool
            If True, generates both long (+1) and short (-1) signals
            If False, only generates signals in the direction of bias
        bias : Literal["long", "short"]
            Direction preference when bidirectional=False

        Returns
        -------
        pd.DataFrame
            Copy of data with added columns: breakout_h, breakout_l, signal_breakout
        """
        if not isinstance(time_start, datetime.time) or not isinstance(time_end, datetime.time):
            raise ValueError("time_start and time_end must be datetime.time objects")

        data = self.data.copy()

        # Filter data to the specified time window
        filtered = data.between_time(time_start, time_end)

        # Calculate high and low for each day within the time window
        hl_signals = filtered.groupby(filtered.index.date).agg({
            self.high: "max",
            self.low: "min"
        }).rename(columns={self.high: "breakout_h", self.low: "breakout_l"})

        # Add breakout columns to data by aligning on date
        data['_date'] = data.index.date
        data['breakout_h'] = data['_date'].map(hl_signals['breakout_h'])
        data['breakout_l'] = data['_date'].map(hl_signals['breakout_l'])
        data.drop(columns=['_date'], inplace=True)


        # Forward fill breakout levels within each day (for bars after the window)
        data[['breakout_h', 'breakout_l']] = data.groupby(data.index.date)[['breakout_h', 'breakout_l']].ffill()

        # Initialize signal column
        data['signal_breakout'] = 0

        # Generate signals based on price relative to breakout levels
        if bidirectional:
            # Both long and short signals
            data.loc[data[self.close] > data['breakout_h'], 'signal_breakout'] = 1
            data.loc[data[self.close] < data['breakout_l'], 'signal_breakout'] = -1
        else:
            # Single direction based on bias
            if bias == "long":
                data.loc[data[self.close] > data['breakout_h'], 'signal_breakout'] = 1
            elif bias == "short":
                data.loc[data[self.close] < data['breakout_l'], 'signal_breakout'] = -1

        return data

    def intraday_momentum(self, session_open: datetime.time, session_close: datetime.time,
                         return_x_period: Optional[datetime.time] = None,
                         returns_period_length: timedelta = timedelta(hours=2),
                         pre_close: bool = False):
        """Calculate intraday momentum returns for specified time periods.

        Parameters
        ----------
        session_open : datetime.time
            Session start time
        session_close : datetime.time
            Session end time
        return_x_period : Optional[datetime.time]
            Custom start time for return calculation. If None, uses session_open
        returns_period_length : timedelta
            Length of the return measurement window (default 2 hours)
        pre_close : bool
            If True, measures returns ending at session_close instead of starting at open

        Returns
        -------
        pd.Series
            Momentum returns indexed by datetime (same index as self.data).
            The momentum value is inserted at the last timestamp used in the calculation
            and forward filled for the rest of each day.

        Notes
        -----
        Return calculation modes:
        1. Default (return_x_period=None, pre_close=False):
           Returns from session_open to (session_open + returns_period_length)
           Value inserted at time_end timestamp and forward filled

        2. Custom start (return_x_period given, pre_close=False):
           Returns from return_x_period to (return_x_period + returns_period_length)
           Value inserted at time_end timestamp and forward filled

        3. Pre-close (pre_close=True, return_x_period=None):
           Returns from (session_close - returns_period_length) to session_close
           Value inserted at session_close timestamp and forward filled

        4. Pre-close with custom (pre_close=True, return_x_period given):
           Returns from (session_close - returns_period_length) to session_close
           Value inserted at session_close timestamp and forward filled
           (return_x_period is ignored in this case)
        """
        # Initialize result series with same index as original data
        momentum_series = pd.Series(index=self.data.index, dtype=float, name='intraday_momentum')
        momentum_series[:] = np.nan

        # Filter to session times
        session_data = self.data.between_time(session_open, session_close)

        if session_data.empty:
            return momentum_series

        # Determine start and end times for the momentum window
        if pre_close:
            # Calculate returns ending at session close
            total_seconds = returns_period_length.total_seconds()
            hours_delta = int(total_seconds // 3600)
            minutes_delta = int((total_seconds % 3600) // 60)

            # Create start time by subtracting from session_close
            start_hour = session_close.hour - hours_delta
            start_minute = session_close.minute - minutes_delta

            # Handle minute underflow
            if start_minute < 0:
                start_minute += 60
                start_hour -= 1

            # Handle hour underflow (cross midnight - unlikely for intraday)
            if start_hour < 0:
                start_hour += 24

            time_start = datetime.time(start_hour, start_minute)
            time_end = session_close
        else:
            # Calculate returns starting from open or custom period
            if return_x_period is not None:
                time_start = return_x_period
            else:
                time_start = session_open

            # Calculate end time by adding period to start
            total_seconds = returns_period_length.total_seconds()
            hours_delta = int(total_seconds // 3600)
            minutes_delta = int((total_seconds % 3600) // 60)

            end_hour = time_start.hour + hours_delta
            end_minute = time_start.minute + minutes_delta

            # Handle minute overflow
            if end_minute >= 60:
                end_minute -= 60
                end_hour += 1

            # Handle hour overflow (unlikely for intraday but handle anyway)
            if end_hour >= 24:
                end_hour -= 24

            time_end = datetime.time(end_hour, end_minute)

        # Extract data within the calculated window
        window_data = session_data.between_time(time_start, time_end)

        if window_data.empty:
            return momentum_series

        # Calculate log returns for each day and get the last timestamp in the window
        def calc_daily_return_and_timestamp(group):
            if len(group) < 2:
                return pd.Series({'return': np.nan, 'timestamp': group.index[-1]})
            first_price = group[self.close].iloc[0]
            last_price = group[self.close].iloc[-1]
            if first_price <= 0 or last_price <= 0:
                return pd.Series({'return': np.nan, 'timestamp': group.index[-1]})
            log_return = np.log(last_price) - np.log(first_price)
            return pd.Series({'return': log_return, 'timestamp': group.index[-1]})

        # Get returns and their timestamps
        daily_results = window_data.groupby(window_data.index.date).apply(calc_daily_return_and_timestamp)

        # Insert momentum values at the appropriate timestamps
        for date, row in daily_results.iterrows():
            if pd.notna(row['return']):
                timestamp = row['timestamp']
                momentum_series.loc[timestamp] = row['return']

        # Forward fill within each day
        momentum_series = momentum_series.groupby(momentum_series.index.date).ffill()

        return momentum_series






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


