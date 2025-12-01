from CTAFlow.screeners import ScreenParams, HistoricalScreener
from CTAFlow.data import read_exported_df
from CTAFlow.config import INTRADAY_DATA_PATH
import pandas as pd
import traceback
from datetime import time

# Load one ticker
data = {'RB': read_exported_df(INTRADAY_DATA_PATH / 'CSV/RB_5min.csv')}
print(f"Initial data index tz: {data['RB'].index.tz}")

# Test localization manually
hs = HistoricalScreener(data, verbose=True)

# Test the _localize_dataframe method
localized = hs._localize_dataframe(data['RB'], "America/Chicago")
print(f"After _localize_dataframe tz: {localized.index.tz}")

# Test _filter_by_months
filtered = hs._filter_by_months(localized, [12])
print(f"After _filter_by_months tz: {filtered.index.tz}")
print(f"Filtered data shape: {filtered.shape}")

# Test _ensure_intraday_time_metadata
metadata_added = hs._ensure_intraday_time_metadata(filtered)
print(f"After _ensure_intraday_time_metadata tz: {metadata_added.index.tz}")

# Now test _extract_session_data
try:
    session_data = hs._extract_session_data(
        metadata_added,
        time(8, 30),
        time(15, 0),
        'Close'
    )
    print(f"Session data extracted successfully! Shape: {session_data.shape}")
    print(f"Session data tz: {session_data.index.tz}")
except Exception as e:
    print(f"\n{'='*60}")
    print('ERROR in _extract_session_data:')
    print('='*60)
    traceback.print_exc()
    print('='*60)