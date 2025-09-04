# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CTA (Commodity Trading Advisor) positioning prediction system that uses technical indicators and COT (Commitment of Traders) data to forecast week-over-week changes in institutional positioning in commodity futures markets. The system combines quantitative finance, machine learning, and systematic trading behavior analysis.

## Package Installation & Development Commands

The project is structured as an installable Python package. For development:

```bash
# Install in development mode (editable)
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Run tests (requires dev dependencies)
python -m pytest tests/

# Run a specific test
python -m pytest tests/test_cot_date_conversion.py

# Run tests directly (fallback if pytest not available)
python tests/test_cot_date_conversion.py
```

The package structure follows standard Python conventions with all modules under `CTAFlow/`:

## Core Architecture

The codebase follows a modular pipeline architecture with four main components:

### Data Layer (`CTAFlow/data/`)
- **`retrieval.py`**: Asynchronous data loading from HDF5 stores using `fetch_data_sync()` 
- **`signals_processing.py`**: Technical analysis and COT signal generation
  - `COTProcessor`: Handles COT data cleaning and positioning metrics
  - `TechnicalAnalysis`: Calculates selective technical indicators and volatility normalization
- **`futures_mappings.toml`**: Maps futures ticker symbols to COT commodity codes
- **`data_client.py`**: Main data interface class

### Forecasting Engine (`CTAFlow/forecaster/`)
- **`forecast.py`**: Contains all forecasting classes (`CTAForecast`, `CTALinear`, `CTALight`, `CTAXGBoost`, `CTARForest`)
  - Selective technical indicator calculation (only computes requested groups)
  - Weekly resampling with configurable day-of-week (default Friday for COT alignment)
  - Feature engineering combining COT positioning with technical signals
  - Each model class maintains `self.data`, `self.features`, and `self.target` for training/validation

### Strategy Layer (`CTAFlow/strategy/`)
- **`strategy.py`**: `RegimeStrategy` class implementing trading strategies using forecasting framework

### Features (`CTAFlow/features/`)
- **`feature_engineering.py`**: Feature creation and transformation utilities

### Configuration
- **`CTAFlow/config.py`**: HDF5 data paths and environment setup
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

### Package Import and Basic Usage

```python
import CTAFlow

# Main forecasting class (preferred interface)
forecaster = CTAFlow.CTAForecast('CL_F')
data = forecaster.prepare_data(selected_indicators=['moving_averages', 'rsi'],
                               normalize_momentum=True,
                               resample_weekly=True)

# Train models
result = forecaster.train_model(model_type='lightgbm', target_type='return')
print(result['test_metrics'])
```

### Direct Model Access

```python
# Access specific model classes
linear_model = CTAFlow.CTALinear('CL_F')
lightgbm_model = CTAFlow.CTALight('CL_F') 
xgb_model = CTAFlow.CTAXGBoost('CL_F')
rf_model = CTAFlow.CTARForest('CL_F')

# Access utility classes
data_client = CTAFlow.DataClient()
cot_processor = CTAFlow.COTProcessor()
tech_analysis = CTAFlow.TechnicalAnalysis()
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
- Each model class (`CTAForecast`, `CTALinear`, etc.) works off of `self.data` and `self.features`. Ensure that these attributes are used consistently throughout to maintain one dataset per model instance
- When testing use standard ASCII characters and avoid using emojis/special characters
- The package must be installed in development mode (`pip install -e .`) to work properly from external directories

## Available Utility Functions

```python
# Package metadata and configuration
CTAFlow.get_version()                    # Returns package version
CTAFlow.get_supported_models()           # Lists all supported ML models  
CTAFlow.get_supported_indicators()       # Lists technical indicator groups
CTAFlow.get_supported_cot_features()     # Lists COT feature categories

# Configuration access (from config.py)
CTAFlow.MARKET_DATA_PATH                 # HDF5 market data location
CTAFlow.COT_DATA_PATH                    # HDF5 COT data location  
CTAFlow.MODEL_DATA_PATH                  # Saved models location
```