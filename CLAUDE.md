# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CTA (Commodity Trading Advisor) positioning prediction system that uses technical indicators and COT (Commitment of Traders) data to forecast week-over-week changes in institutional positioning in commodity futures markets. The system combines quantitative finance, machine learning, and systematic trading behavior analysis.

## Core Architecture

The codebase follows a modular pipeline architecture with four main components:

### Data Layer (`data/`)
- **`retrieval.py`**: Asynchronous data loading from HDF5 stores using `fetch_market_cot_data()`
- **`signals_processing.py`**: Technical analysis and COT signal generation
  - `DataProcessor`: Handles COT data cleaning and positioning metrics
  - `TechnicalAnalysis`: Calculates selective technical indicators and volatility normalization
- **`futures_mappings.toml`**: Maps futures ticker symbols to COT commodity codes

### Forecasting Engine (`forecaster/`)
- **`forecast.py`**: Main `Forecaster` class orchestrates the entire workflow
  - Selective technical indicator calculation (only computes requested groups)
  - Weekly resampling with configurable day-of-week (default Friday for COT alignment)
  - Feature engineering combining COT positioning with technical signals
  - Maintains `self.data`, `self.features`, and `self.target` for training/validation

### Strategy Layer (`strategy/`)
- **`strategy.py`**: Trading strategy implementation using the forecasting framework

### Configuration
- **`config.py`**: HDF5 data paths and environment setup
- **`project_goals.md`**: Detailed research methodology and indicator selection rationale

## Key Technical Concepts

### COT Data Integration
The system maps futures tickers to COT commodity codes via TOML configuration:
```python
# ZC_F (Corn futures) -> CORN -> "002602" (COT code)
```

### Technical Indicator Selection
Available indicator groups: `['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence', 'vol_normalized']`

Key indicators based on research:
- 50/200 MA crossovers (Golden Cross/Death Cross)
- MACD + RSI confluence signals  
- ATR-based volatility targeting
- VWAP deviations for institutional flow detection

### Volatility Normalization
Uses exponentially weighted 63-day standard deviation for risk-adjusted returns:
```python
vol_normalized_returns = raw_returns / (ewm_vol_63d * sqrt(period))
```

## Data Flow

1. **Load**: `fetch_market_cot_data()` loads both market and COT data asynchronously
2. **Process**: `TechnicalAnalysis.calculate_enhanced_indicators()` with selective calculation
3. **Engineer**: `Forecaster.prepare_forecasting_features()` combines all feature types
4. **Resample**: Optional weekly resampling (post-calculation or pre-calculation)
5. **Target**: `create_target_variable()` generates prediction targets

## Common Usage Patterns

### Basic Forecasting Workflow

```python
forecaster = Forecaster()
data = forecaster.prepare_data('CL_F',
                               selected_indicators=['moving_averages', 'rsi'],
                               normalize_momentum=True,
                               resample_weekly=True)
```

### Post-Calculation Resampling

```python
# Calculate on daily data, then resample features
forecaster.prepare_data('CL_F', resample_weekly=False)
weekly_features = forecaster.resample_existing_features(day_of_week='Friday')
```

### Selective Technical Indicators
Only calculates requested indicator groups to avoid computational waste:
```python
# Only calculates RSI, ATR, and normalized momentum
tech_features = forecaster.get_technical_features_only(df, 
                                                      selected_indicators=['rsi', 'atr'],
                                                      normalize_momentum=True)
```

## Data Requirements

- Market data in HDF5 format at `MARKET_DATA_PATH` with OHLCV columns
- COT data in HDF5 format at `COT_DATA_PATH` with standard COT report columns
- Data should be indexed by datetime for proper resampling

## Development Notes

- All technical indicator calculations are selective - only requested groups are computed
- Weekly resampling defaults to Friday to align with COT reporting schedule (Tuesday data published Friday)
- The system maintains separation between COT features and technical features for flexible model development
- Volatility normalization uses 63-day exponentially weighted standard deviation as the baseline risk measure
- @forecaster\forecast.py CTAForecast class works off of self.data and self.features. Ensure that it uses them throughout in order to maintain one dataset per model
- When testing use standard ASCII characters and avoid using emojis/special characters