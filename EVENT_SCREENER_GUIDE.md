# Event Screener Guide

## Overview

The Event Screener analyzes futures price reactions around scheduled economic data releases (CPI, NFP, FOMC, etc.). It computes returns in event windows and identifies statistical patterns like trend continuation or reversals.

## Quick Start

```python
from CTAFlow.screeners.event_screener import run_event_screener
from CTAFlow.screeners.params import EventParams
import pandas as pd

# 1. Prepare event calendar
events = pd.DataFrame({
    'release_ts': pd.to_datetime([
        '2023-01-15 13:30:00',
        '2023-02-15 13:30:00',
        '2023-03-15 13:30:00'
    ]),
    'event_code': ['CPI', 'CPI', 'CPI'],
    'value': [6.5, 6.4, 5.0],        # Actual release
    'consensus': [6.2, 6.3, 5.2]     # Market expectation
})

# 2. Configure parameters
params = EventParams(
    event_code='CPI',
    event_window_pre_minutes=30,   # Look 30min before release
    event_window_post_minutes=60,  # Look 60min after release
    value_col='value',
    consensus_col='consensus',
    surprise_mode='diff'            # surprise = actual - consensus
)

# 3. Run screener
result = run_event_screener(
    bars=bars_df,              # Your OHLCV data
    events=events,
    params=params,
    symbol='ES',
    instrument_tz='America/Chicago'
)

# 4. Analyze results
print(result.summary)          # Statistical analysis
print(result.patterns)         # Detected patterns
```

## Event Calendar Format

### Required Column

**`release_ts`** (datetime): Exact timestamp of the scheduled data release

Must be:
- Timezone-aware, OR
- Will be localized to `instrument_tz`
- Accurate to the minute (e.g., CPI releases at 13:30 ET)

### Optional Columns for Enhanced Analysis

**`event_code`** (str): Event type identifier
- Examples: 'CPI', 'NFP', 'FOMC', 'PPI', 'GDP'
- Used for grouping and filtering events

**`value`** / **`actual`** (float): Actual released value
- The economic indicator reading (e.g., CPI of 6.5%)
- Required for surprise analysis

**`consensus`** / **`forecast`** (float): Market consensus
- Expected value before release
- Required for surprise analysis

**`stdev`** (float): Standard deviation for z-score normalization
- Optional, used when `surprise_mode='z'`

### Example Event Calendars

#### Minimal (No Surprise Analysis)
```python
events = pd.DataFrame({
    'release_ts': pd.to_datetime([
        '2023-01-15 13:30',
        '2023-02-15 13:30'
    ])
})
```

#### Full Featured (With Surprise Analysis)
```python
events = pd.DataFrame({
    'release_ts': pd.to_datetime([
        '2023-01-15 13:30:00',
        '2023-02-15 13:30:00',
        '2023-03-15 13:30:00'
    ]),
    'event_code': ['CPI', 'CPI', 'CPI'],
    'value': [6.5, 6.4, 5.0],
    'consensus': [6.2, 6.3, 5.2],
    'stdev': [0.3, 0.3, 0.3]
})
```

## Data Sources for Event Calendars

### Free/Public Sources

1. **Federal Reserve Economic Data (FRED)**
   - URL: https://fred.stlouisfed.org/
   - Coverage: US economic indicators
   - API: Free with registration
   ```python
   # Example: Fetch CPI releases from FRED
   from fredapi import Fred
   fred = Fred(api_key='your_key')
   cpi_data = fred.get_series('CPIAUCSL')
   ```

2. **Trading Economics**
   - URL: https://tradingeconomics.com/calendar
   - Coverage: Global economic calendar
   - API: Paid subscription required

3. **Investing.com Economic Calendar**
   - URL: https://www.investing.com/economic-calendar/
   - Coverage: Comprehensive global calendar
   - Scraping: Possible but check ToS

### Premium Sources

4. **Bloomberg Terminal**
   - Command: `ECST <GO>`
   - Coverage: Real-time global calendar
   - Export: Excel/CSV export available

5. **Refinitiv Eikon**
   - Coverage: Professional-grade calendar
   - API: Available with subscription

### Custom Scrapers

Build custom scrapers for specific needs:
```python
# Example: Scrape from a public source
import requests
from bs4 import BeautifulSoup

def scrape_economic_calendar(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Parse table and extract events
    # Return DataFrame with release_ts, event_code, etc.
    pass
```

## Common Event Codes

| Code | Event Name | Release Time (ET) | Frequency |
|------|-----------|-------------------|-----------|
| **FOMC** | Federal Open Market Committee | 14:00 | 8x/year |
| **CPI** | Consumer Price Index | 08:30 | Monthly |
| **NFP** | Non-Farm Payrolls | 08:30 | Monthly |
| **PPI** | Producer Price Index | 08:30 | Monthly |
| **GDP** | Gross Domestic Product | 08:30 | Quarterly |
| **RETAIL** | Retail Sales | 08:30 | Monthly |
| **JOBLESS** | Initial Jobless Claims | 08:30 | Weekly |
| **PCE** | Personal Consumption Expenditures | 08:30 | Monthly |
| **PMI** | Purchasing Managers' Index | 09:45 | Monthly |
| **HOUSING** | Housing Starts | 08:30 | Monthly |

## EventParams Configuration

### Core Window Settings

```python
params = EventParams(
    event_code='CPI',                     # Event identifier
    event_window_pre_minutes=30,          # Minutes before release (default: 30)
    event_window_post_minutes=60,         # Minutes after release (default: 60)
    min_events=10,                        # Min events for stats (default: 10)
)
```

### Horizon Analysis

```python
params = EventParams(
    # ...
    include_t1_close=True,                # Include T+1 returns (default: True)
    extra_daily_horizons=[2, 3, 5],      # Additional horizons (T+2, T+3, T+5)
)
```

### Surprise Analysis

```python
params = EventParams(
    # ...
    value_col='value',                    # Column with actual values
    consensus_col='consensus',            # Column with consensus
    surprise_mode='diff',                 # 'diff', 'pct', or 'z'
)
```

Surprise modes:
- **`'diff'`**: `surprise = actual - consensus` (e.g., 6.5 - 6.2 = 0.3)
- **`'pct'`**: `surprise = (actual - consensus) / |consensus|` (percentage)
- **`'z'`**: `surprise = (actual - consensus) / stdev` (z-score)

### Pattern Detection

```python
params = EventParams(
    # ...
    corr_threshold=0.3,                   # Min correlation for patterns (default: 0.3)
)
```

### Orderflow Integration

```python
params = EventParams(
    # ...
    use_orderflow=True,                   # Enable orderflow analysis
    orderflow_window_pre_minutes=5,       # Orderflow window before (default: 5)
    orderflow_window_post_minutes=5,      # Orderflow window after (default: 5)
)
```

## Output Structure

### EventScreenerResult

The `run_event_screener()` function returns an `EventScreenerResult` with three components:

#### 1. `events` DataFrame

One row per event with computed returns:

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Instrument identifier |
| `event_code` | str | Event type |
| `release_ts` | datetime | Scheduled release time |
| `session_date` | date | Trading session date |
| `pre_ts` | datetime | Pre-window timestamp |
| `post_ts` | datetime | Post-window timestamp |
| `pre_close` | float | Price before event |
| `post_close` | float | Price after event |
| `close_T0` | float | Session close price |
| `close_T1` | float | T+1 session close |
| `r_event` | float | Log return (pre → post) |
| `r_T0` | float | Log return (post → T0) |
| `r_T1` | float | Log return (T0 → T1) |
| `surprise` | float | Computed surprise metric |

#### 2. `summary` DataFrame

Aggregated statistics per event code:

| Column | Type | Description |
|--------|------|-------------|
| `n_events` | int | Number of events |
| `mean_r_event` | float | Average event return |
| `p_event` | float | T-test p-value |
| `mean_r_T0` | float | Average T0 return |
| `p_T0` | float | T-test p-value |
| `rho_event_T0` | float | Correlation: event × T0 |
| `p_rho_event_T0` | float | Correlation p-value |
| `rho_surprise_event` | float | Correlation: surprise × event |

#### 3. `patterns` List

Detected patterns for trading:

```python
{
    'symbol': 'ES',
    'screen_type': 'SCREEN_EVENT',
    'pattern_type': 'event_trend_T0',     # or 'event_reversal_T0'
    'event_code': 'CPI',
    'gate_direction': 1,                   # +1 trend, -1 reversal
    'strength': 0.75,                      # Correlation strength
    'description': 'CPI event_trend_T0 from event-window to T0 close'
}
```

Pattern types:
- **`event_trend_T0`**: Event return predicts same-direction T0 return (continuation)
- **`event_reversal_T0`**: Event return predicts opposite-direction T0 return (mean reversion)
- **`event_flow_trend_T0`**: Orderflow imbalance predicts same-direction T0 return
- **`event_flow_reversal_T0`**: Orderflow imbalance predicts opposite-direction T0 return

## Complete Example

```python
from CTAFlow.screeners.event_screener import run_event_screener
from CTAFlow.screeners.params import EventParams
import pandas as pd

# Load your bar data (ES futures)
bars = pd.read_parquet('es_5min_bars.parquet')
# Ensure: bars has 'ts' and 'close' columns

# Create event calendar (CPI releases)
events = pd.DataFrame({
    'release_ts': pd.to_datetime([
        '2023-01-12 08:30',
        '2023-02-14 08:30',
        '2023-03-14 08:30',
        '2023-04-12 08:30',
        '2023-05-10 08:30',
        '2023-06-13 08:30',
    ]),
    'event_code': ['CPI'] * 6,
    'value': [6.5, 6.4, 6.0, 5.0, 4.9, 3.0],
    'consensus': [6.5, 6.4, 6.0, 5.2, 5.0, 3.1]
})

# Configure parameters
params = EventParams(
    event_code='CPI',
    event_window_pre_minutes=30,
    event_window_post_minutes=60,
    include_t1_close=True,
    value_col='value',
    consensus_col='consensus',
    surprise_mode='diff',
    corr_threshold=0.4,
    min_events=5
)

# Run screener
result = run_event_screener(
    bars=bars,
    events=events,
    params=params,
    symbol='ES',
    instrument_tz='America/Chicago'
)

# Analyze results
print("Event-level returns:")
print(result.events[['release_ts', 'r_event', 'r_T0', 'surprise']])

print("\nSummary statistics:")
print(result.summary)

print("\nDetected patterns:")
for pattern in result.patterns:
    print(f"  {pattern['pattern_type']}: strength={pattern['strength']:.2f}")
```

## Error Handling

The event screener validates inputs and provides helpful error messages:

### Missing `release_ts` Column
```python
ValueError: Event calendar missing required columns: ['release_ts']
Required: ['release_ts']
Provided columns: ['event_code', 'value']

Event calendar must contain:
  - 'release_ts' (datetime): Timestamp of scheduled release
...
```

### Empty Event Calendar
```python
ValueError: Event calendar is empty. Provide a DataFrame with at least one event.
```

### Missing Bar Data
```python
KeyError: Bars DataFrame missing required columns: ['close']
Required: ['ts' or DatetimeIndex, 'close']
Provided columns: ['open', 'high', 'low', 'volume']
```

## Tips and Best Practices

### 1. Timezone Handling
Always ensure your `release_ts` is in the correct timezone:
```python
# Localize to Eastern Time
events['release_ts'] = pd.to_datetime(events['release_ts']).dt.tz_localize('America/New_York')

# Or convert if already tz-aware
events['release_ts'] = events['release_ts'].dt.tz_convert('America/New_York')
```

### 2. Data Quality
- Use high-frequency bars (1-5 minute) for accurate event windows
- Ensure bar data covers event timestamps +/- windows
- Remove data errors (missing bars, bad ticks) before analysis

### 3. Statistical Significance
- Use `min_events >= 10` for reliable statistics
- Check p-values before trusting correlations
- Higher `corr_threshold` (0.4-0.5) reduces false positives

### 4. Surprise Analysis
- Only use surprise when you have reliable consensus data
- Consider `surprise_mode='z'` for normalized comparisons
- Different events have different surprise sensitivities

### 5. Pattern Usage
- Higher `strength` (correlation) = more reliable pattern
- Test patterns out-of-sample before trading
- Combine event patterns with other signals

## Integration with Pipeline

Event screener integrates with the broader CTAFlow workflow:

```python
from CTAFlow.screeners.event_screener import run_event_screener
from CTAFlow.strategy.screener_pipeline import ScreenerPipeline

# Run event screener
event_result = run_event_screener(bars, events, params, symbol='ES', instrument_tz='America/Chicago')

# Extract patterns
event_patterns = event_result.patterns

# Use patterns in pipeline
pipeline = ScreenerPipeline(use_gpu=True)
backtest_results = pipeline.concurrent_pattern_backtests(
    bars=bars,
    patterns=event_patterns,
    verbose=True
)
```

## Troubleshooting

**Q: Why am I getting no events in the output?**
- Check that `release_ts` falls within your bar data time range
- Verify `event_window_pre/post_minutes` don't exceed bar coverage
- Ensure bars are sorted by timestamp

**Q: Why are correlations all NaN?**
- Need at least `min_events` (default 10) occurrences
- Check that returns are computable (non-zero prices)
- Verify your surprise columns have valid data

**Q: How do I handle multiple event types?**
- Either run screener separately per event_code
- Or pass all events and group results by event_code

**Q: Can I use this for non-US events?**
- Yes! Just set appropriate `instrument_tz`
- Ensure event calendar uses correct timezone
- Example: European events use 'Europe/London'

## Summary

✅ **Event calendar**: Must have `release_ts`, optional `event_code`, `value`, `consensus`
✅ **Data sources**: FRED, Trading Economics, Bloomberg, custom scrapers
✅ **Configuration**: Use EventParams to set windows, horizons, surprise mode
✅ **Validation**: Automatic validation with helpful error messages
✅ **Output**: Events, summary statistics, and tradeable patterns
✅ **Integration**: Works seamlessly with ScreenerPipeline

For more details, see the docstrings in `event_screener.py`.
