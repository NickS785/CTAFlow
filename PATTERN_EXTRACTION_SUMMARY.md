# Pattern Extraction System - Implementation Summary

## Overview

This implementation adds comprehensive pattern extraction capabilities to CTAFlow's screening framework, enabling automated identification and analysis of recurring patterns from both orderflow and historical screener results.

## What Was Implemented

### 1. HistoricalScreener.from_parquet() Method

**Location**: `CTAFlow/screeners/historical_screener.py:3120-3260`

A new class method that loads intraday data from Parquet files using `AsyncParquetWriter`:

```python
screener = HistoricalScreener.from_parquet(
    symbols=['CL_F', 'NG_F', 'ZC_F'],
    timeframe='1T',
    start='2023-01-01',
    end='2024-01-01',
    max_concurrent=5
)
```

**Key Features**:
- Async I/O for fast parallel data loading
- Supports time-based resampling (e.g., '1T', '5T') and volume buckets
- Automatic timezone-aware datetime filtering
- Configurable concurrency for optimal performance
- Compatible with existing `HistoricalScreener` workflows

### 2. PatternExtractor Module

**Location**: `CTAFlow/screeners/pattern_extractor.py`

A comprehensive pattern extraction and analysis framework:

```python
from CTAFlow.screeners import PatternExtractor

extractor = PatternExtractor(min_observations=10, significance_threshold=0.05)
patterns = extractor.extract_orderflow_patterns(results, include_exploratory=False)
```

**Key Components**:

#### PatternResult Class
Container for extracted patterns with:
- Pattern metadata (type, symbol, metric, description)
- Trigger conditions (weekday, week_of_month, time_range, etc.)
- Time series data (dates and values)
- Statistics (mean, t_stat, p_value, correlation, etc.)
- Automatic significance classification
- Conversion to pandas Series

#### PatternExtractor Class
Main extraction engine with:
- `extract_orderflow_patterns()`: Process orderflow scan results
- `extract_historical_patterns()`: Process historical screener results
- `filter_patterns()`: Multi-criteria filtering
- `get_pattern_summary()`: Generate summary DataFrames

### 3. Supported Pattern Types

#### Orderflow Patterns
- **Weekly**: Day-of-week seasonality (e.g., "Monday buy_pressure pattern")
- **WOM Weekday**: Week-of-month + weekday (e.g., "Week 1 Monday buy_pressure")
- **Peak Pressure**: Time-of-day by weekday (e.g., "Monday peak buy at 09:30")
- **Events**: Significant intraday pressure events with timestamps

#### Historical Patterns
- **Momentum**: Session momentum and correlations
- **Seasonality**: Time-of-day and seasonal patterns

### 4. Pattern Analysis Features

#### Filtering System
```python
# Multi-criteria filtering
high_sig = extractor.filter_patterns(significance='high')
monday_patterns = extractor.filter_patterns(weekday='Monday')
week1_patterns = extractor.filter_patterns(week_of_month=1)
cl_patterns = extractor.filter_patterns(symbol='CL_F')

# Combined filters
monday_week1_high = extractor.filter_patterns(
    weekday='Monday',
    week_of_month=1,
    significance='high'
)
```

#### Summary Reports
```python
summary = extractor.get_pattern_summary()
# Returns DataFrame with:
# - symbol, pattern_type, metric, frequency, significance
# - n_observations, mean, std, t_stat, p_value
# - trigger conditions (weekday, week_of_month, etc.)
```

#### Time Series Conversion
```python
for pattern in patterns:
    series = pattern.to_series()  # Convert to pandas Series
    series.to_csv(f"{pattern.symbol}_{pattern.pattern_type}.csv")
```

### 5. Significance Classification

Automatic classification based on p-values:
- **High**: p < 0.001
- **Medium**: p < 0.05
- **Low**: p ≥ 0.05
- **Exploratory**: Low sample size

### 6. Testing & Documentation

**Tests**: `tests/test_pattern_extraction_standalone.py`
- All 6 tests passing
- Real data integration tested with ZC scanner results
- 18 patterns extracted from actual orderflow data

**Documentation**:
- `docs/pattern_extraction_guide.md`: Complete user guide
- `examples/pattern_extraction_example.py`: 5 comprehensive examples
- Inline docstrings with examples for all classes and methods

## Integration Points

### 1. With OrderflowScanner
```python
scanner = OrderflowScanner(params)
results = scanner.scan(tick_data)

extractor = PatternExtractor()
patterns = extractor.extract_orderflow_patterns(results)
```

### 2. With HistoricalScreener
```python
screener = HistoricalScreener.from_parquet(['CL_F'], timeframe='1T')
results = screener.run_screens([momentum_params, seasonality_params])

extractor = PatternExtractor()
patterns = extractor.extract_historical_patterns(results)
```

### 3. With Existing Scanner Results
```python
# Load CSV results from screens/scanner_results/
orderflow_results = {
    'ZC': {
        'df_weekly': pd.read_csv('zc_df_weekly.csv'),
        'df_wom_weekday': pd.read_csv('zc_df_wom_weekday.csv'),
        'df_events': pd.read_csv('zc_df_events.csv'),
    }
}

patterns = extract_patterns_from_orderflow(orderflow_results)
```

## API Additions

### New Exports from CTAFlow.screeners

```python
from CTAFlow.screeners import (
    # New classes
    PatternExtractor,
    PatternResult,

    # New convenience functions
    extract_patterns_from_orderflow,
    extract_patterns_from_historical,

    # Enhanced existing class
    HistoricalScreener,  # Now has from_parquet() method
)
```

## Usage Examples

### Example 1: Quick Pattern Extraction
```python
from CTAFlow.screeners import extract_patterns_from_orderflow

# Load and extract in one step
patterns = extract_patterns_from_orderflow(
    orderflow_results,
    min_observations=10,
    include_exploratory=False
)

# Filter and analyze
high_sig = [p for p in patterns if p.significance == 'high']
for p in high_sig:
    print(f"{p.description}: p={p.statistics['p_value']:.4f}")
```

### Example 2: Complete Workflow
```python
# 1. Load data
screener = HistoricalScreener.from_parquet(
    symbols=['CL_F', 'NG_F'],
    timeframe='1T',
    start='2023-01-01'
)

# 2. Run screens
results = screener.run_screens([momentum_params, seasonality_params])

# 3. Extract patterns
extractor = PatternExtractor()
patterns = extractor.extract_historical_patterns(results)

# 4. Filter and export
monday_patterns = extractor.filter_patterns(weekday='Monday')
summary = extractor.get_pattern_summary()
summary.to_csv('pattern_summary.csv')
```

### Example 3: Time Series Generation
```python
extractor = PatternExtractor()
patterns = extractor.extract_orderflow_patterns(orderflow_results)

# Generate date series for significant patterns
date_series = {}
for pattern in patterns:
    if pattern.significance == 'high' and len(pattern.dates) > 0:
        series = pattern.to_series()
        date_series[pattern.description] = series

# Combine into DataFrame
df = pd.DataFrame(date_series)
```

## Performance Characteristics

### from_parquet() Loading
- **Async I/O**: Concurrent loading of multiple symbols
- **Tested**: 2 symbols, 1-minute data, ~1 year
- **Configurable**: `max_concurrent` parameter (default: 5)

### Pattern Extraction
- **Orderflow**: Processes weekly, wom_weekday, peak_pressure, events
- **Historical**: Processes momentum and seasonality screens
- **Filtering**: O(n) linear time for all filter operations
- **Memory**: Minimal - stores only extracted patterns, not raw data

### Real Data Test Results
- **Input**: ZC orderflow results (weekly + wom_weekday)
- **Output**: 18 significant patterns extracted
- **High Significance**: 10 patterns (p < 0.001)
- **Processing Time**: < 1 second

## Files Created/Modified

### New Files
1. `CTAFlow/screeners/pattern_extractor.py` - Main module (700+ lines)
2. `tests/test_pattern_extraction_standalone.py` - Comprehensive tests
3. `examples/pattern_extraction_example.py` - 5 usage examples
4. `docs/pattern_extraction_guide.md` - Complete documentation
5. `tests/test_pattern_extraction.py` - Pytest tests (optional)

### Modified Files
1. `CTAFlow/screeners/historical_screener.py` - Added `from_parquet()` method
2. `CTAFlow/screeners/__init__.py` - Added new exports

## Backward Compatibility

All changes are **fully backward compatible**:
- No existing APIs modified
- Only additions (new method, new module)
- Existing code continues to work unchanged

## Testing

All tests passing:
```
================================================================================
PATTERN EXTRACTION TESTS
================================================================================

[TEST] PatternExtractor initialization...
  PASSED

[TEST] Weekly pattern extraction...
  PASSED - Extracted 6 weekly patterns

[TEST] Pattern filtering...
  PASSED - Filters working correctly

[TEST] Pattern summary generation...
  PASSED - Summary DataFrame generated with 2 rows

[TEST] Convenience function...
  PASSED - Convenience function working

[TEST] Real data integration...
  PASSED - Extracted 18 patterns from real ZC data
    - High significance patterns: 10

================================================================================
RESULTS: 6 passed, 0 failed
================================================================================
```

## Future Enhancements

Possible extensions:
1. Pattern visualization (plotly charts)
2. Pattern backtesting framework
3. Machine learning pattern classification
4. Calendar date mapping for patterns
5. Pattern combination/interaction detection
6. Export to trading signals

## Dependencies

All existing CTAFlow dependencies (no new requirements):
- pandas
- numpy
- asyncio (standard library)

## Documentation

Complete documentation provided:
- **API Reference**: Inline docstrings with examples
- **User Guide**: `docs/pattern_extraction_guide.md`
- **Examples**: `examples/pattern_extraction_example.py`
- **Tests**: `tests/test_pattern_extraction_standalone.py`

## Conclusion

This implementation provides a complete, production-ready pattern extraction system that:
- ✅ Integrates seamlessly with existing screeners
- ✅ Loads data efficiently from Parquet files
- ✅ Extracts patterns from all screener result types
- ✅ Provides flexible filtering and analysis
- ✅ Generates time series for further analysis
- ✅ Includes comprehensive testing and documentation
- ✅ Maintains backward compatibility
- ✅ Follows CTAFlow architectural patterns

The system is ready for immediate use with both new and existing screener results.
