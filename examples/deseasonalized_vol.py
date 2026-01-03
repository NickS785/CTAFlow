from CTAFlow.data import read_exported_df
from CTAFlow.config import INTRADAY_DATA_PATH, OHLC_DICT
from CTAFlow.features.volatility import deseasonalize_volatility, estimate_diurnal_pattern
import pandas as pd
import numpy as np

ticker = "CL"
df = read_exported_df(INTRADAY_DATA_PATH / f"CSV/{ticker}_5min.csv")
start_hr = 2
end_hr = 15
interval_mins = 5
total_hrs = end_hr-start_hr

df = df.loc["2012-01-01":].dropna()

df['returns']= np.log(df['Close']/df['Close'].shift(1))

# Volatility proxy: absolute log-return (always positive)
df['vol_proxy'] = np.abs(df['returns']) + 1e-6

df = df.loc[(df.index.hour >=  start_hr) & (df.index.hour <= end_hr)]  # Filter out low-information-bars
# Filter invalid values (zeros cause issues with log transform)
cleaned_df = df[(df['vol_proxy'] > 0.0) & np.isfinite(df['vol_proxy'])].copy()

n_bins = total_hrs * (60 // interval_mins)  # 108 bins for 9-hour session at 5-min intervals
print(f"Data shape: {cleaned_df.shape}")
print(f"Date range: {cleaned_df.index.min()} to {cleaned_df.index.max()}")
print(f"Bins per day: {n_bins}")

print("\n--- Testing without rolling window (full dataset fit) ---")
results_static = deseasonalize_volatility(
    cleaned_df['vol_proxy'] + 1e-5,
    bins_per_day=n_bins,
    rolling_days=252  # Fit on entire dataset
)

print("Results shape:", results_static.shape)
print("\nNaN counts:")
print(results_static.isna().sum())
print("\nAdjusted stats:")
print(results_static['adjusted'].describe())




