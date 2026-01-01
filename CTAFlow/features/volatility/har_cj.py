import numpy as np
import pandas as pd
import statsmodels.api as sm
from ..diurnal_seasonality import deseasonalize_volatility
from typing import List, Optional


class HarCJModel:
    def __init__(self):
        self.model = None
        self.params = None

    def calculate_daily_stats(self, intraday_df, price_col='Close', date_col='Date'):
        """
        Expects a DataFrame with intraday data (e.g., 5-min bars).
        Must have a 'Date' column to group by day.
        """
        # 1. Calculate Log Returns
        intraday_df['log_ret'] = np.log(intraday_df[price_col] / intraday_df[price_col].shift(1))

        # Drop the first NaN created by differencing
        intraday_df = intraday_df.dropna()

        # 2. Group by Day to calculate RV and BV
        stats = []
        grouped = intraday_df.groupby(date_col)

        for date, group in grouped:
            returns = group['log_ret'].values

            # Realized Volatility (RV): Sum of squared returns
            rv = np.sum(returns ** 2)

            # Bipower Variation (BV): Sum of adjacent absolute returns * (Pi/2)
            # This estimator is robust to jumps.
            term1 = np.abs(returns[1:])
            term2 = np.abs(returns[:-1])
            bv = (np.pi / 2) * np.sum(term1 * term2)

            # Ensure BV doesn't exceed RV (mathematical constraint)
            if bv > rv:
                bv = rv

            # Jump Component (J)
            j = max(rv - bv, 0)

            stats.append({'Date': date, 'RV': rv, 'C': bv, 'J': j})

        return pd.DataFrame(stats).set_index('Date')

    def prepare_har_features(self, daily_stats):
        """
        Creates the Heterogeneous lags (Daily, Weekly, Monthly)
        separated for Continuous (C) and Jump (J) components.
        """
        df = daily_stats.copy()

        # 1. Continuous Lags
        df['C_d'] = df['C'].shift(1)  # Yesterday
        df['C_w'] = df['C'].rolling(window=5).mean().shift(1)  # Past Week
        df['C_m'] = df['C'].rolling(window=22).mean().shift(1)  # Past Month

        # 2. Jump Lags
        # Research (Andersen et al) suggests Weekly/Monthly Jumps are often noise.
        # We usually only keep the Daily Jump lag.
        df['J_d'] = df['J'].shift(1)
        df['J_w'] = df['J'].rolling(window=5).mean().shift(1)

        # Target: Tomorrow's Total Volatility (RV)
        # Note: You can also predict just 'C' if you want to ignore jump risk
        df['Target_RV'] = df['RV']

        return df.dropna()

    def fit(self, daily_stats):
        data = self.prepare_har_features(daily_stats)

        # Define X (Features) and Y (Target)
        # Standard HAR-CJ usually includes C_d, C_w, C_m, and J_d
        features = ['C_d', 'C_w', 'C_m', 'J_d', 'J_w']
        X = data[features]
        Y = data['Target_RV']

        # Add constant for OLS
        X = sm.add_constant(X)

        # Fit OLS
        self.model = sm.OLS(Y, X).fit()
        self.params = self.model.params
        return self.model.summary()



class IntradayHARCJ:
    def __init__(self, start_time="02:00", cutoff_time="10:00", close_cutoff="11:00", close_col="Close", rolling_window=None):
        self.cutoff_time = pd.to_datetime(cutoff_time).time()
        self.start_time = pd.to_datetime(start_time).time()
        self.close_cutoff = pd.to_datetime(close_cutoff).time()
        self.price_col = close_col
        self.rolling_window = rolling_window or 252
        self.seasonal_factor = None  # Will hold the time-of-day multipliers
        self.model = None

    def _calculate_log_returns(self, df):
        """Helper to ensure safe log-return calculation"""
        # Log(Price_t / Price_t-1)
        return np.log(df[self.price_col] / df[self.price_col].shift(1)).fillna(0)

    def fit_seasonality(self, df):
        """
        Calculates the Time-of-Day Seasonality (U-Shape).
        """
        # 1. Calc Returns & Vol Proxy
        df = df.copy()
        df['log_ret'] = self._calculate_log_returns(df)
        df['vol_proxy'] = np.abs(df['log_ret'])

        # Filter to trading hours if needed
        # df = df.between_time(self.start_time, self.cutoff_time) # Optional

        # 2. Use library function to get seasonal component
        # We assume bins_per_day is auto-detected or passed if needed
        results = deseasonalize_volatility(df['vol_proxy'])

        # 3. Extract the Generic Daily Pattern (Time -> Factor)
        # The 'seasonal' column is in Log-Space. We need Linear Space for division.
        results['seasonal_linear'] = np.exp(results['seasonal'])

        # Create a lookup map: Time -> Seasonal Factor
        # We group by time and take the mean (it should be constant per time bin if static fit)
        self.seasonal_factor = results.groupby(results.index.time)['seasonal_linear'].mean()

        return self.seasonal_factor

    def deseasonalize(self, df):
        """
        Applies the seasonal factor to create r_star (Deseasonalized Returns).
        """
        data = df.copy()
        if 'log_ret' not in data.columns:
            data['log_ret'] = self._calculate_log_returns(data)

        # Map the factor based on the time of each row
        # map() is much faster than iterating
        data['s_i'] = data.index.time
        data['s_i'] = data['s_i'].map(self.seasonal_factor)

        # r* = r / s_i
        data['r_star'] = data['log_ret'] / data['s_i']
        return data

    def _process_daily_data(self, day_df):
        """
        Helper to process a single day's data.
        Calculates Morning C/J features and Last Hour Target RV.
        """
        # 1. Split Day into Observation (Morning) and Target (Last Hour)
        morning = day_df[day_df.index.time < self.cutoff_time]
        target = day_df[(day_df.index.time >= self.cutoff_time) & (day_df.index.time < self.close_cutoff)]

        # Validation: Ensure we have enough data
        # FIX: Return NaNs with correct keys instead of empty Series
        if len(morning) < 10 or len(target) == 0:
            return pd.Series({
                'C_morning': np.nan,
                'J_morning': np.nan,
                'Target_LastHour_RV': np.nan
            })

        # 2. Calculate Morning Features (using Deseasonalized Volatility)
        # r_star = Deseasonalized Volatility Magnitude (approx |ret| / seasonal_factor)
        r_star = morning['r_star'].values
        r_star_open = r_star[:12] # Assuming 5-min bars
        rv_star_open = np.sum(r_star_open ** 2)
        term_1 = np.abs(r_star_open[1:])
        term_2 = np.abs(r_star_open[:-1])
        bv_star_open = (np.pi / 2 ) * np.sum(term_1 * term_2)
        if bv_star_open > rv_star_open:
            bv_star_open = rv_star_open

        jump_star_open = max(rv_star_open - bv_star_open, 0)


        # Realized Variance (RV*)
        rv_star = np.sum(r_star ** 2)

        # Bipower Variation (BV*)
        term1 = np.abs(r_star[1:])
        term2 = np.abs(r_star[:-1])
        bv_star = (np.pi / 2) * np.sum(term1 * term2)

        # Jump Component (J*)
        if bv_star > rv_star:
            bv_star = rv_star
        jump_star = max(rv_star - bv_star, 0)

        # 3. Calculate Target Variable (using RAW Volatility)
        target_rv = np.sum(target['vol_proxy'] ** 2)

        return pd.Series({
            'C_morning': bv_star,
            'J_morning': jump_star,
            'Target_LastHour_RV': target_rv,
            'C_open':bv_star_open,
            'J_open':jump_star_open,
            "RV_open":rv_star_open
        })
    def prepare_training_data(self, df):
        """
        Full pipeline: Calculate Vol -> Deseasonalize -> GroupBy Day -> Lag Features
        """
        data = df.copy()

        # 1. Calculate Volatility Proxy
        # We use absolute log returns as the proxy for intraday volatility
        data['log_ret'] = np.log(data[self.price_col] / data[self.price_col].shift(1)).fillna(0)
        data['vol_proxy'] = np.abs(data['log_ret'])

        # Replace 0s with epsilon to avoid log errors in the adjuster
        # (The adjuster uses log-space by default)
        vol_clean = data['vol_proxy'].replace(0, 1e-8)

        # 2. Deseasonalize
        # This returns a DF with ['original', 'seasonal', 'adjusted']
        # 'adjusted' is log(vol) - log(seasonal)
        print("Running deseasonalization...")
        ds_results = deseasonalize_volatility(
            vol_clean,
            rolling_days=self.rolling_window,
            use_log=True
        )

        # 3. Transform back to Linear Space
        # We exponentiate 'adjusted' to get the Deseasonalized Volatility Magnitude (r_star)
        data['r_star'] = np.exp(ds_results['adjusted'])

        # 4. Apply GroupBy Logic
        # This replaces the manual loop
        print("Calculating daily HAR-CJ features...")
        daily_stats = data.groupby(data.index.date).apply(
            self._process_daily_data, include_groups=False
        )

        # Unstack from MultiIndex Series to DataFrame with proper columns
        if isinstance(daily_stats, pd.Series):
            daily_stats = daily_stats.unstack()

        # Clean up empty days
        daily_stats = daily_stats.dropna()

        # 5. Create Lags (Yesterday's Volatility)
        # RV_yesterday = C_yesterday + J_yesterday
        daily_stats['RV_yesterday'] = (
                daily_stats['C_morning'].shift(1) + daily_stats['J_morning'].shift(1)
        )
        daily_stats['RV_5ma'] = daily_stats['RV_yesterday'].rolling(5).mean()
        daily_stats['RV_21ma'] = daily_stats['RV_yesterday'].rolling(21).mean()

        return daily_stats

    def train_model(self, training_data, features: Optional[List] = None, use_log: bool = False):
        """
        Train HAR-CJ model.

        Parameters
        ----------
        training_data : pd.DataFrame
            Output from prepare_training_data()
        features : list, optional
            Feature columns to use. Defaults to ['C_morning', 'J_morning']
        use_log : bool, default False
            If True, apply log1p transform to features and target.
            Reduces condition number and handles skewness.

        Returns
        -------
        statsmodels OLS summary
        """
        minimal_features = ['C_morning', 'J_morning']

        if features is None:
            features = minimal_features
        else:
            features = list(features) + minimal_features

        # Drop rows with NaN in features or target
        data = training_data[features + ['Target_LastHour_RV']].dropna()

        X = data[features].copy()
        y = data['Target_LastHour_RV'].copy()

        if use_log:
            X = np.log1p(X)
            y = np.log1p(y)

        X = sm.add_constant(X)
        self.model = sm.OLS(y, X).fit()
        self._use_log = use_log
        self._features = features
        return self.model.summary()
# --- Usage Note ---
# 1. Load 5-min data
# df = pd.read_csv(...)
# forecaster = IntradayHARCJ(cutoff_time="15:00")
# train_df = forecaster.prepare_training_data(df)
# print(forecaster.train_model(train_df))

# --- Example Workflow ---
# 1. Load your 5-min data for Crude Oil (CL) or Gold (GC)
# df_5min = pd.read_csv("CL_5min.csv")

# 2. Initialize and Run
# har_cj = HarCJModel()
# daily_metrics = har_cj.calculate_daily_stats(df_5min)
# print(har_cj.fit(daily_metrics))