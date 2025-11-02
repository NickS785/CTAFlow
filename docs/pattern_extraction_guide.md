# Pattern Extraction Guide

## Overview

The Pattern Extraction system processes results from `HistoricalScreener` and `OrderflowScanner` to identify significant recurring patterns (weekly, monthly, seasonal, intraday) and generate time series for each pattern.

## Key Features

- **Unified Interface**: Works with both orderflow and historical screener results
- **Flexible Filtering**: Filter patterns by type, significance, weekday, week-of-month, and more
- **Time Series Generation**: Convert patterns to pandas Series for further analysis
- **Statistical Classification**: Automatic significance classification (high/medium/low/exploratory)
- **Summary Reports**: Generate comprehensive DataFrame summaries of all patterns

## Installation

The pattern extraction module is included in `CTAFlow.screeners`:

```python
from CTAFlow.screeners import (
    PatternExtractor,
    PatternResult,
    extract_patterns_from_orderflow,
    extract_patterns_from_historical,
)
```

## Quick Start

### 1. Loading Data with from_parquet

The new `HistoricalScreener.from_parquet()` method loads intraday data using `AsyncParquetWriter`:

```python
from CTAFlow.screeners import HistoricalScreener

# Load 1-minute data for multiple symbols
screener = HistoricalScreener.from_parquet(
    symbols=['CL_F', 'NG_F', 'ZC_F'],
    timeframe='1T',
    start='2023-01-01',
    end='2024-01-01',
    max_concurrent=5  # Async loading with 5 concurrent tasks
)

# Load volume bucket data
screener = HistoricalScreener.from_parquet(
    symbols=['CL_F'],
    volume_bucket_size=500,
    start='2023-01-01'
)

# Load from custom path
screener = HistoricalScreener.from_parquet(
    symbols=['CL_F'],
    parquet_path='F:/Data/intraday/',
    timeframe='5T'
)
```

### 2. Extracting Patterns from Orderflow Results

```python
from CTAFlow.screeners import OrderflowScanner, PatternExtractor

# Run orderflow scan
scanner = OrderflowScanner(params)
results = scanner.scan(tick_data)

# Extract patterns
extractor = PatternExtractor(
    min_observations=10,
    significance_threshold=0.05
)
patterns = extractor.extract_orderflow_patterns(
    results,
    include_exploratory=False
)

print(f"Extracted {len(patterns)} significant patterns")
```

### 3. Extracting Patterns from Historical Screener Results

```python
from CTAFlow.screeners import HistoricalScreener, PatternExtractor

# Load data and run screens
screener = HistoricalScreener.from_parquet(['CL_F'], timeframe='1T')
results = screener.run_screens([momentum_params, seasonality_params])

# Extract patterns
extractor = PatternExtractor()
patterns = extractor.extract_historical_patterns(
    results,
    min_correlation=0.3,
    include_exploratory=False
)
```

### 4. Filtering and Analyzing Patterns

```python
# Filter by pattern type
weekly_patterns = extractor.filter_patterns(pattern_type='weekly')
wom_patterns = extractor.filter_patterns(pattern_type='wom_weekday')

# Filter by significance
high_sig = extractor.filter_patterns(significance='high')

# Filter by weekday
monday_patterns = extractor.filter_patterns(weekday='Monday')

# Filter by week of month
week1_patterns = extractor.filter_patterns(week_of_month=1)

# Combine filters
monday_week1_high = extractor.filter_patterns(
    weekday='Monday',
    week_of_month=1,
    significance='high'
)

# Get summary DataFrame
summary = extractor.get_pattern_summary()
print(summary[['symbol', 'pattern_type', 'significance', 'mean', 'p_value']])
```

### 5. Converting Patterns to Time Series

```python
# Convert individual pattern to Series
for pattern in patterns:
    if len(pattern.dates) > 0:
        series = pattern.to_series()
        print(f"{pattern.description}")
        print(series.head())

        # Save to CSV
        series.to_csv(f"{pattern.symbol}_{pattern.pattern_type}.csv")
```

## Pattern Types

### Orderflow Patterns

#### Weekly Patterns
Day-of-week seasonality patterns:
- **Pattern Type**: `'weekly'`
- **Frequency**: `'weekly'`
- **Trigger Conditions**: `{'weekday': 'Monday'}`
- **Example**: "Monday buy_pressure pattern"

#### Week-of-Month + Weekday Patterns
Combined week-of-month and day-of-week patterns:
- **Pattern Type**: `'wom_weekday'`
- **Frequency**: `'monthly'`
- **Trigger Conditions**: `{'week_of_month': 1, 'weekday': 'Monday'}`
- **Example**: "Week 1 Monday buy_pressure pattern"

#### Peak Pressure Patterns
Time-of-day pressure patterns by weekday:
- **Pattern Type**: `'peak_pressure'`
- **Frequency**: `'weekly'`
- **Trigger Conditions**: `{'weekday': 'Monday', 'clock_time': '09:30:00', 'pressure_bias': 'buy'}`
- **Example**: "Monday peak buy pressure at 09:30:00"

#### Event Patterns
Significant intraday pressure events:
- **Pattern Type**: `'event'`
- **Frequency**: `'intraday'`
- **Trigger Conditions**: `{'direction': 'positive', 'time_range': (start, end)}`
- **Example**: "positive buy_pressure event at 2023-01-15 09:30"

### Historical Patterns

#### Momentum Patterns
Session momentum and correlation patterns:
- **Pattern Type**: `'momentum'`
- **Frequency**: `'daily'`
- **Trigger Conditions**: `{'session': 'london'}`
- **Example**: "winter_momentum london correlation pattern"

#### Seasonality Patterns
Time-of-day and seasonal patterns:
- **Pattern Type**: `'seasonality'`
- **Frequency**: `'intraday'`
- **Trigger Conditions**: `{'time_period': '09:30'}`
- **Example**: "spring_seasonality 09:30 seasonal pattern"

## Significance Classification

Patterns are automatically classified based on p-values:

| P-Value | Classification |
|---------|---------------|
| < 0.001 | `'high'` |
| < 0.05 | `'medium'` |
| ≥ 0.05 | `'low'` |
| Any (with low sample size) | `'exploratory'` |

## PatternResult Object

Each pattern is represented by a `PatternResult` object:

```python
@dataclass
class PatternResult:
    pattern_type: str          # 'weekly', 'wom_weekday', 'event', etc.
    symbol: str                # Trading symbol
    metric: str                # 'buy_pressure', 'momentum', etc.
    description: str           # Human-readable description
    trigger_conditions: Dict   # Conditions that trigger this pattern
    dates: pd.DatetimeIndex   # Dates when pattern occurred
    values: np.ndarray        # Values for each date
    statistics: Dict          # Statistical measures
    significance: str         # 'high', 'medium', 'low', 'exploratory'
    frequency: str            # 'weekly', 'monthly', 'daily', 'intraday'
    metadata: Dict            # Additional metadata
```

### Available Statistics

Statistics vary by pattern type but commonly include:

- `mean`: Average value
- `std`: Standard deviation
- `t_stat`: T-statistic
- `p_value`: P-value
- `n`: Number of observations
- `correlation`: Correlation coefficient (momentum patterns)
- `max_abs_z`: Maximum absolute z-score (event patterns)

## Complete Workflow Example

```python
from CTAFlow.screeners import (
    HistoricalScreener,
    OrderflowScanner,
    OrderflowParams,
    ScreenParams,
    PatternExtractor,
)
from datetime import timedelta, time

# Step 1: Load data from Parquet
screener = HistoricalScreener.from_parquet(
    symbols=['CL_F', 'NG_F'],
    timeframe='1T',
    start='2023-01-01',
    end='2024-01-01',
    max_concurrent=5
)

# Step 2: Define screening parameters
momentum_params = ScreenParams(
    screen_type='momentum',
    season='winter',
    session_starts=["02:30", "08:30"],
    session_ends=["10:30", "15:00"],
    sess_start_hrs=1,
    sess_start_minutes=30
)

seasonality_params = ScreenParams(
    screen_type='seasonality',
    season='winter',
    target_times=["09:30", "14:00"],
    period_length=timedelta(hours=2),
    dayofweek_screen=True,
    seasonality_session_start="08:30",
    seasonality_session_end="15:30"
)

orderflow_params = OrderflowParams(
    session_start="08:30",
    session_end="15:30",
    bucket_size="auto",
    vpin_window=50,
    threshold_z=2.0,
    min_days=30
)

# Step 3: Run screens
hist_results = screener.run_screens([momentum_params, seasonality_params])

# For orderflow (requires tick data)
# of_scanner = OrderflowScanner(orderflow_params)
# of_results = of_scanner.scan(tick_data)

# Step 4: Extract patterns
extractor = PatternExtractor(
    min_observations=10,
    significance_threshold=0.05
)

hist_patterns = extractor.extract_historical_patterns(
    hist_results,
    min_correlation=0.3,
    include_exploratory=False
)

# of_patterns = extractor.extract_orderflow_patterns(
#     of_results,
#     include_exploratory=False
# )

# Step 5: Filter and analyze
high_sig_patterns = extractor.filter_patterns(significance='high')
monday_patterns = extractor.filter_patterns(weekday='Monday')
cl_patterns = extractor.filter_patterns(symbol='CL_F')

# Step 6: Generate summary
summary = extractor.get_pattern_summary()
print("\nHigh Significance Patterns:")
print(summary[summary['significance'] == 'high'])

# Step 7: Export patterns
for pattern in high_sig_patterns:
    if len(pattern.dates) > 0:
        series = pattern.to_series()
        filename = f"{pattern.symbol}_{pattern.pattern_type}_{pattern.metric}.csv"
        series.to_csv(filename)
        print(f"Exported: {filename}")
```

## Convenience Functions

For quick pattern extraction without creating an extractor object:

```python
from CTAFlow.screeners import (
    extract_patterns_from_orderflow,
    extract_patterns_from_historical,
)

# Extract from orderflow results
patterns = extract_patterns_from_orderflow(
    orderflow_results,
    min_observations=10,
    include_exploratory=False
)

# Extract from historical results
patterns = extract_patterns_from_historical(
    historical_results,
    min_observations=10,
    min_correlation=0.3,
    include_exploratory=False
)
```

## Working with Real Scanner Results

The pattern extractor works seamlessly with saved scanner results:

```python
import pandas as pd
from pathlib import Path
from CTAFlow.screeners import PatternExtractor

# Load results from CSV files (as in screens/scanner_results/)
results_path = Path("screens/scanner_results")
orderflow_results = {}

for symbol in ['ZC', 'ZS']:
    orderflow_results[symbol] = {}

    # Load weekly patterns
    weekly_file = results_path / f"{symbol.lower()}_df_weekly.csv"
    if weekly_file.exists():
        orderflow_results[symbol]['df_weekly'] = pd.read_csv(weekly_file, index_col=0)

    # Load week-of-month patterns
    wom_file = results_path / f"{symbol.lower()}_df_wom_weekday.csv"
    if wom_file.exists():
        orderflow_results[symbol]['df_wom_weekday'] = pd.read_csv(wom_file, index_col=0)

    # Load peak pressure patterns
    peak_file = results_path / f"{symbol.lower()}_df_weekly_peak_pressure.csv"
    if peak_file.exists():
        orderflow_results[symbol]['df_weekly_peak_pressure'] = pd.read_csv(peak_file, index_col=0)

    # Load events
    events_file = results_path / f"{symbol.lower()}_df_events.csv"
    if events_file.exists():
        orderflow_results[symbol]['df_events'] = pd.read_csv(
            events_file,
            index_col=0,
            parse_dates=['ts_start', 'ts_end']
        )

# Extract all patterns
extractor = PatternExtractor()
patterns = extractor.extract_orderflow_patterns(orderflow_results)

print(f"Extracted {len(patterns)} total patterns")
summary = extractor.get_pattern_summary()
print(summary)
```

## Performance Tips

1. **Use Async Loading**: `from_parquet()` uses async I/O for fast data loading
2. **Set max_concurrent**: Adjust based on your system (default: 5)
3. **Filter Early**: Use `include_exploratory=False` to exclude low-sample patterns
4. **Batch Processing**: Process multiple symbols in a single screener instance
5. **Reuse Extractors**: Add patterns incrementally to the same extractor

## API Reference

### PatternExtractor

```python
PatternExtractor(
    min_observations: int = 10,
    significance_threshold: float = 0.05
)
```

#### Methods

- `extract_orderflow_patterns(results, include_exploratory=False) -> List[PatternResult]`
- `extract_historical_patterns(results, min_correlation=0.3, include_exploratory=False) -> List[PatternResult]`
- `filter_patterns(**kwargs) -> List[PatternResult]`
- `get_pattern_summary() -> pd.DataFrame`

### PatternResult

```python
PatternResult(
    pattern_type: str,
    symbol: str,
    metric: str,
    description: str,
    trigger_conditions: Dict[str, Any],
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    statistics: Dict[str, float],
    significance: str,
    frequency: str,
    metadata: Dict[str, Any] = {}
)
```

#### Methods

- `to_series() -> pd.Series`: Convert pattern to pandas Series

### HistoricalScreener.from_parquet

```python
HistoricalScreener.from_parquet(
    symbols: List[str],
    parquet_path: Optional[str] = None,
    timeframe: Optional[str] = None,
    volume_bucket_size: Optional[int] = None,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    max_concurrent: int = 5
) -> HistoricalScreener
```

## Troubleshooting

### No patterns extracted
- Check `include_exploratory=True` to see low-sample patterns
- Verify significance_threshold isn't too strict
- Ensure screener results have the expected structure

### Empty dates in patterns
- Weekly/wom_weekday patterns don't have specific dates (summary statistics)
- Event patterns have specific timestamps
- Use pattern statistics instead of time series for aggregate patterns

### Memory issues with large datasets
- Reduce `max_concurrent` in `from_parquet()`
- Filter patterns early with strict criteria
- Process symbols in smaller batches

## See Also

- [HistoricalScreener Documentation](./historical_screener.md)
- [OrderflowScanner Documentation](./orderflow_scanner.md)
- [Example Scripts](../examples/pattern_extraction_example.py)
- [Tests](../tests/test_pattern_extraction_standalone.py)
