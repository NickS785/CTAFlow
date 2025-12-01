from CTAFlow.screeners import ScreenParams, HistoricalScreener
from CTAFlow.data import read_exported_df
from CTAFlow.config import INTRADAY_DATA_PATH
import pandas as pd
import traceback
import sys

# Load one ticker
data = {'RB': read_exported_df(INTRADAY_DATA_PATH / 'CSV/RB_5min.csv')}
print(f"Data columns: {data['RB'].columns.tolist()}")
print(f"Data index tz: {data['RB'].index.tz}")
print(f"Data shape: {data['RB'].shape}")

hs = HistoricalScreener(data, verbose=True)

# Simple screen
screen_params = ScreenParams(
    screen_type='momentum',
    months=[12],
    session_starts=['08:30'],
    session_ends=['15:00']
)

# Set the exception hook to get full traceback
def exception_hook(exc_type, exc_value, exc_traceback):
    print("\n" + "="*80)
    print("FULL EXCEPTION TRACEBACK:")
    print("="*80)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("="*80)

sys.excepthook = exception_hook

try:
    result = hs.run_screens([screen_params])
    print('Success! Result:', result.keys())
except Exception as e:
    print(f"\nCaught exception: {type(e).__name__}: {e}")
    traceback.print_exc()
