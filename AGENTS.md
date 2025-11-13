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
- **`data_client.py`**: Main data interface class for HDF5 operations
- **`retrieval.py`**: Asynchronous data loading from HDF5 stores using `fetch_data_sync()` 
- **`futures_curve_manager.py`**: Manages futures curve data and contract relationships
- **`futures_mappings.toml`**: Maps futures ticker symbols to COT commodity codes
- **`ticker_classifier.py`**: Automatic classification of tickers into commodity/financial types
- **`classifications_reference.py`**: Reference mappings for all supported tickers

### Features & Signal Processing (`CTAFlow/features/`) **[PRIMARY PROCESSING LOCATION]**
**This is where all signal processing and feature engineering is concentrated:**
- **`signals_processing.py`**: Core technical analysis and COT signal generation
  - `COTProcessor`: Handles COT data cleaning and positioning metrics
  - `TechnicalAnalysis`: Calculates selective technical indicators and volatility normalization
- **`feature_engineering.py`**: Intraday microstructure features
  - `IntradayFeatures`: Microstructure and intraday feature extraction
- **`curve_analysis.py`**: Advanced futures curve analysis framework
  - `CurveShapeAnalyzer`: Curve shape analysis and feature extraction
  - `CurveEvolution`: Time series evolution tracking using FuturesCurve snapshots
  - `SpreadAnalyzer`: Spread analysis, seasonality, and Lévy areas

### Data Container Classes (`CTAFlow/data/contract_handling/`)
**Core numpy-based data structures for futures analysis:**
- **`SpreadFeature`**: Base numpy array container with directional analysis capabilities
  - Horizontal + Sequential: Curve shape analysis (slopes across contracts)
  - Vertical + Sequential: Time series analysis (trends over time)
- **`SeqData`**: Sequential data container for ordered contract data
- **`FuturesCurve`**: Single curve snapshot with term structure metrics
- **`SpreadData`**: Main data container (uses sequentialized data only)
- **`Contract`**: Individual contract data with expiry tracking

### Forecasting Engine (`CTAFlow/forecaster/`)
- **`forecast.py`**: Contains all forecasting classes (`CTAForecast`, `CTALinear`, `CTALight`, `CTAXGBoost`, `CTARForest`)
  - Selective technical indicator calculation (only computes requested groups)
  - Weekly resampling with configurable day-of-week (default Friday for COT alignment)
  - Feature engineering combining COT positioning with technical signals
  - Each model class maintains `self.data`, `self.features`, and `self.target` for training/validation

### Strategy Layer (`CTAFlow/strategy/`)
- **`strategy.py`**: `RegimeStrategy` class implementing trading strategies using forecasting framework
- **`screener_pipeline.py`**:
  - `ScreenerPipeline`: normalises screener payloads into sparse gate columns. Keep `_items_from_patterns` compatible with nested mappings, `(key, pattern)` tuples, generator inputs, and `PatternExtractor.concat_many` outputs because both the screener pipeline and notebooks rely on those shapes.
  - `HorizonMapper`: aligns realised returns with generated gates. `build_xy` requires timezone-aware `ts`, `open`, `close`, and `session_id` columns so horizon calculations remain stable. Expect `PatternExtractor` summaries with canonical column names when wiring horizons.

### Screeners (`CTAFlow/screeners/`)
- **`pattern_extractor.py`**:
  - `PatternExtractor`: restructures screener payloads into canonical `PatternSummary` frames, supports arithmetic/concatenation helpers, and persists ranked signals. Preserve `SUMMARY_COLUMNS`, `_strength_raw`, and async loaders (`load_summaries_from_results_async`) because downstream notebooks and the pipeline consume those exact shapes.
  - Utility helpers compute significance scores, merge seasonal/orderflow frames, and serialise outputs. Keep scoring weights configurable via keyword arguments even though defaults live in `_strength_raw`.
- **`__init__.py`**: re-exports `PatternExtractor` and `PatternSummary` for external import paths (used by strategy notebooks and tests).

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
import CTAFlow.data.contract_handling.curve_manager

import CTAFlow.data.contract_handling.futures

linear_model = CTAFlow.CTALinear('CL_F')
lightgbm_model = CTAFlow.CTALight('CL_F')
xgb_model = CTAFlow.CTAXGBoost('CL_F')
rf_model = CTAFlow.CTARForest('CL_F')

# Access utility classes
data_client = CTAFlow.DataClient()

# Feature processing classes (from CTAFlow/features/)
cot_processor = CTAFlow.COTAnalyzer()
tech_analysis = CTAFlow.TechnicalAnalysis()
intraday_features = CTAFlow.IntradayFeatures('CL_F')
spread_data = CTAFlow.data.contract_handling.futures_curve_manager.SpreadData('CL_F')
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

### Data Architecture Principles
- **Numpy-First Approach**: All data stored as `np.ndarray` except for `index`/`timestamps` (pandas DatetimeIndex)
- **Sequentialized Data Priority**: All calculations use sequentialized data from `seq_data`, NOT raw `self.curve` which is unordered
- **Directional Analysis**: 
  - Horizontal + Sequential: Curve shape analysis across contracts
  - Vertical + Sequential: Time series analysis over time
- **Analysis Classes**: Located in `CTAFlow/features/curve_analysis.py` for curve-specific analysis

### Technical Implementation
- **Signal processing is centralized in `CTAFlow/features/`**: All technical analysis, COT processing, and feature engineering should be concentrated in this module
- All technical indicator calculations are selective - only requested groups are computed
- Weekly resampling defaults to Friday to align with COT reporting schedule (Tuesday data published Friday)
- The system maintains separation between COT features and technical features for flexible model development
- Volatility normalization uses 63-day exponentially weighted standard deviation as the baseline risk measure
- Each model class (`CTAForecast`, `CTALinear`, etc.) works off of `self.data` and `self.features`. Ensure that these attributes are used consistently throughout to maintain one dataset per model instance

### Development Best Practices
- When testing use standard ASCII characters and avoid using emojis/special characters
- The package must be installed in development mode (`pip install -e .`) to work properly from external directories
- Always validate data types: primary data should be `np.ndarray`, avoid lists of lists or mixed types
- Use `seq_data.seq_prices`, `seq_data.seq_spreads` for analysis, not raw curve data

## Available Utility Functions

```python
# Package metadata and configuration
CTAFlow.get_version()                    # Returns package version
CTAFlow.get_supported_models()           # Lists all supported ML models  
CTAFlow.get_supported_indicators()       # Lists technical indicator groups
CTAFlow.get_supported_cot_features()     # Lists COT feature ticker_categories

# Configuration access (from config.py)
CTAFlow.MARKET_DATA_PATH                 # HDF5 market data location
CTAFlow.COT_DATA_PATH                    # HDF5 COT data location  
CTAFlow.MODEL_DATA_PATH                  # Saved models location
```

## New Data Architecture (Post-Refactoring)

### Core Data Classes Hierarchy

```python
# Base data container with directional analysis
from CTAFlow.data.contract_handling.curve_manager import SpreadFeature, SeqData, SpreadData, FuturesCurve, Contract

# Analysis classes for curve-specific operations  
from CTAFlow.features import CurveShapeAnalyzer, CurveEvolution, SpreadAnalyzer, CurveEvolutionAnalyzer

# Example Usage:
# 1. Horizontal Sequential Analysis (Curve Shape)
horizontal_feat = SpreadFeature(data=curve_data, direction='horizontal', sequential=True)
slopes = horizontal_feat.get_slopes()  # Contango/backwardation analysis
structures = horizontal_feat.get_market_structure()

# 2. Vertical Sequential Analysis (Time Series)  
vertical_feat = SpreadFeature(data=price_series, direction='vertical', sequential=True)
ma = vertical_feat.calculate_moving_average(20)
trend = vertical_feat.analyze_trend()

# 3. Sequential Data Container
seq_data = SeqData(seq_prices=prices_array, seq_labels=labels_array)
front_month_prices = seq_data[datetime(2024, 1, 1), 0]  # Date, contract position

# 4. Main Data Container (uses sequentialized data only)
spread_data = SpreadData(symbol='CL_F', curve=raw_data, index=dates)
futures_curve = spread_data[datetime(2024, 1, 1)]  # Get curve snapshot
analysis = spread_data[0:10]  # Enhanced slice with curve shape analysis
```

### Data Type Standards
- **Primary Data**: Always `np.ndarray` (no lists, no DataFrames)
- **Time Indexes**: `pd.DatetimeIndex` only (sole exception to numpy-first rule)
- **Sequentialized Priority**: Use `seq_data.seq_prices` not `self.curve` (unordered)
- **Analysis Output**: Structured arrays or dictionaries with array values

### Key Features
- FuturesCurve is a snapshot of the Futures Curve containing days to expiration, prices, spreads, etc.
- SpreadFeature supports both horizontal (curve shape) and vertical (time series) analysis
- SeqData provides advanced indexing: `[datetime, contract_position]`  
- All analysis classes work exclusively with sequentialized data for accuracy

## Recent Framework Enhancements

### Enhanced SpreadData.__getitem__ Method (Latest Update)
The SpreadData indexing system now supports comprehensive access patterns:

```python
spread_data = SpreadData('CL_F')

# String Access Patterns:
contract_f = spread_data['F']                    # Contract for F expiration month
contract_m0 = spread_data['M0']                  # Front month Contract (continuous=True)
contract_m1 = spread_data['M1']                  # Second month Contract (continuous=True)

# Date String Access (NEW):
curve_jan = spread_data['2023-01-15']            # FuturesCurve for date string
curve_time = spread_data['2023-01-15 10:30']     # Supports time components

# Integer Access:
curve_first = spread_data[0]                     # First date's FuturesCurve
curve_last = spread_data[-1]                     # Last date (negative indexing)

# Enhanced Slice Access (NEW):
curves_str = spread_data['2023-01-01':'2023-06-30']         # Date string slice
curves_dt = spread_data[datetime(2023,1,1):datetime(2023,6,30)]  # Datetime slice  
curves_mixed = spread_data['2023-01-01':100]                # Mixed bound types
```

**Key Enhancements:**
- **Continuous Contracts**: Sequential contracts (M0, M1, M2) now return `Contract` objects with `continuous=True`
- **Date String Support**: Full support for ISO date strings with regex validation
- **Flexible Slicing**: Slice bounds can mix datetime, date strings, and integers
- **Robust Parsing**: Uses `dateutil.parser` with nearest-neighbor index matching

### Optimized SpreadAnalyzer with Performance Enhancements

**New File**: `CTAFlow/features/spread_analyzer_optimized.py`

Major performance optimizations addressing computational bottlenecks:

```python
from CTAFlow.features.spread_analyzer_optimized import OptimizedSpreadAnalyzer

analyzer = OptimizedSpreadAnalyzer(spread_data)

# Optimized seasonal statistics (O(n) vs O(n²))
seasonal_stats = analyzer.calculate_seasonal_statistics_optimized('spreads', 'month', 60)

# JIT-compiled Lévy area calculation (~10-100x speedup)
levy_areas = analyzer.calculate_levy_areas_optimized(window=20)

# Vectorized seasonal strength
strength = analyzer.calculate_seasonal_strength_optimized('spreads')

# New: Regime change visualization
fig = analyzer.plot_regime_changes_f0(
    regime_window=63,
    regime_threshold=2.0,
    show_seasonal=True
)
```

**Performance Improvements:**
- **O(n) Seasonal Statistics**: Sliding window approach replaces nested loops
- **JIT Compilation**: Numba-optimized Lévy area calculations
- **Smart Caching**: Avoids repeated DataFrame conversions
- **Vectorized Operations**: NumPy-based autocorrelation and aggregations

### Regime Change Analysis & Visualization (NEW)

**New Method**: `plot_regime_changes_f0()` in OptimizedSpreadAnalyzer

Multi-dimensional regime detection system:

```python
# Generate comprehensive regime analysis
fig = analyzer.plot_regime_changes_f0(
    regime_window=63,          # 3-month analysis window
    regime_threshold=2.0,      # 2-sigma threshold sensitivity
    min_regime_length=10,      # Minimum regime duration
    show_seasonal=True,        # Include seasonal overlay
    height=800
)

# Interactive 4-panel visualization:
# 1. F0 prices with regime highlighting
# 2. Volatility regime scores  
# 3. Trend regime analysis
# 4. Statistical outlier detection
```

**Regime Types Detected:**
- **Volatility Regimes**: High/low volatility periods using rolling vol Z-scores
- **Trend Regimes**: Bullish/bearish trends via rolling regression slopes
- **Statistical Outliers**: Price deviations using rolling Z-scores  
- **Combined Intensity**: Aggregate regime score across all dimensions

**Visual Features:**
- **Color-coded shading**: Red (high vol), Green (bull), Orange (bear), Purple (outliers)
- **Interactive tooltips**: Hover data with regime details
- **Seasonal overlay**: Optional monthly pattern background
- **Threshold lines**: Configurable sensitivity boundaries

### Enhanced SpreadReturns with Comprehensive PnL Tracking

Sophisticated spread trading PnL calculation system:

```python
spread_returns = SpreadReturns(spread_data)

# Calculate realistic PnL with roll costs and fees
pnl_result = spread_returns.calculate_spread_pnl(
    entry_date=datetime(2023, 1, 1),
    exit_date=datetime(2023, 6, 30),
    include_roll_costs=True,      # Account for roll yield
    include_fees=True             # Transaction costs per contract
)

# Returns comprehensive metrics:
# - total_pnl, cumulative_pnl, daily_pnl
# - sharpe_ratio, max_drawdown, win_rate
# - roll_costs, transaction_fees
# - risk_adjusted_returns
```

### CurveEvolution with pd.Series Integration

**Enhanced Architecture**: `CTAFlow/features/curve_analysis.py`

```python
# Bulk loading from SpreadData (NEW)
curve_evolution = CurveEvolution.from_spread_data(
    spread_data, 
    date_range=slice('2023-01-01', '2023-12-31')
)

# pd.Series with datetime indices and FuturesCurve values
history_series = curve_evolution.history  # pd.Series[datetime, FuturesCurve]

# Broadcast analysis methods (no date iteration)
contango_analysis = curve_evolution.analyze_contango_broadcast()
volatility_analysis = curve_evolution.analyze_volatility_broadcast()
```

**Performance Features:**
- **Bulk Loading**: Fast pipeline from SpreadData to CurveEvolution
- **pd.Series Integration**: DateTime-indexed FuturesCurve storage
- **Broadcast Methods**: Vectorized analysis across all dates
- **Factory Methods**: Flexible construction from various data sources

### Comprehensive Plotly Visualization Framework

**New File**: `CTAFlow/features/curve_visualization.py`

Interactive visualization methods for all curve classes:

```python
from CTAFlow.features.curve_visualization import (
    plot_futures_curve, plot_spread_data, plot_curve_evolution,
    plot_spread_returns_pnl, plot_contract_analysis
)

# Curve snapshots with technical indicators
fig = plot_futures_curve(futures_curve, show_technicals=True)

# 3D surface plots of curve evolution
fig = plot_spread_data(spread_data, plot_type='surface_3d')

# PnL analysis with drawdown visualization  
fig = plot_spread_returns_pnl(spread_returns, show_drawdown=True)
```

**Visualization Types:**
- **Term Structure Plots**: Curve shapes with contango/backwardation analysis
- **3D Surface Charts**: Time evolution of entire curves
- **PnL Analysis**: Returns, drawdown, and risk metrics
- **Heatmaps**: Seasonal patterns and regime changes
- **Interactive Features**: Hover data, zoom, pan, export capabilities

## Updated Usage Patterns

### Optimized Performance Analysis Workflow

```python
# 1. Load data with enhanced indexing
spread_data = SpreadData('CL_F')
front_month = spread_data['M0']  # Continuous contract
historical_slice = spread_data['2023-01-01':'2023-12-31']

# 2. Performance-optimized analysis
analyzer = OptimizedSpreadAnalyzer(spread_data)
seasonal_stats = analyzer.calculate_seasonal_statistics_optimized('spreads', 'month')
levy_areas = analyzer.calculate_levy_areas_optimized(window=20)

# 3. Regime analysis and visualization
regime_fig = analyzer.plot_regime_changes_f0(
    regime_threshold=1.5,
    show_seasonal=True
)

# 4. Advanced curve evolution analysis
curve_evolution = CurveEvolution.from_spread_data(spread_data)
broadcast_analysis = curve_evolution.analyze_contango_broadcast()
```

### Enhanced Data Access Patterns

```python
# Flexible date-based access
curve_today = spread_data[datetime.now()]
curve_str = spread_data['2023-06-15']
curve_slice = spread_data['2023-01-01':'2023-12-31':5]  # Every 5th day

# Continuous contract access (NEW)
m0_contract = spread_data['M0']  # continuous=True, is_front_month=True
m1_contract = spread_data['M1']  # continuous=True, is_front_month=False

# Mixed slice bounds (NEW)
mixed_slice = spread_data[datetime(2023,1,1):'2023-06-30']
```

### Example Data for Testing and Development (NEW)

**Simple Example SpreadData**: `SpreadData.example()`

For testing and development without requiring market data files:

```python
from CTAFlow.data import SpreadData
from CTAFlow.features import CurveEvolutionAnalyzer

# Create example data with preloaded synthetic curves
spread_data = SpreadData.example()

# All standard operations work normally
sd_slice = spread_data[-50:-1]  # Test slicing
curve = spread_data[0]          # Individual curve access
m0_contract = spread_data['M0'] # Contract access
cea = CurveEvolutionAnalyzer(spread_data)  # Analysis tools
```

**Example Data Specifications:**
- **100 business days** of synthetic data (2023-01-02 to 2023-05-19)
- **6 contracts** (M0 through M5) with realistic contango structure
- **Realistic patterns**: Volume decreases with contract distance, proper DTE progression
- **Full compatibility**: Works with all SpreadData methods and analysis tools
- **Fast generation**: Creates test data in milliseconds
- **Consistent data**: Deterministic patterns for reliable testing

### Crack-spread CSV for screener triage (NEW)

- Real intraday sample stored at `docs/example.csv`.
- Load **crack spread OHLC** bars via `CTAFlow.data.read_synthetic_csv("docs/example.csv")` – returns a dataframe keyed by `timestamp` with crack spread open/high/low/close columns.
- Load **heating oil / RBOB + crack spread** data via `CTAFlow.data.read_exported_df("docs/example.csv")`; outright columns include the original name and the crack spread columns are suffixed with `.1`.
- Useful when reproducing HistoricalScreener issues locally without HDF5 access.

### Workflow reminders (UPDATED)

- Always commit fixes locally and then use the `make_pr` tool to draft the PR message after running tests.
- Preferred smoke tests: `pytest tests/test_historical_screener_regressions.py`, `pytest tests/test_screener_pipeline.py`, and any targeted screener tests touched by your change set.

**Benefits:**
- **No external dependencies**: Test functionality without HDF5 market data files
- **Development friendly**: Quick setup for testing new features
- **Educational**: Clear example of proper data structure and patterns
- **Debugging**: Predictable data for isolating issues

## Framework Architecture Updates

### Performance Optimization Hierarchy
- **Level 1**: Smart caching and vectorized operations
- **Level 2**: JIT compilation with Numba for critical paths
- **Level 3**: Broadcast methods replacing iteration patterns
- **Level 4**: Memory-efficient numpy array operations

### Enhanced Data Type Consistency
- **Contract Objects**: Now properly support `continuous` flag for sequential data
- **Date String Support**: Robust parsing with multiple format support
- **Slice Flexibility**: Mixed type bounds with intelligent conversion
- **Performance Caching**: Automatic caching of expensive computations

### Visualization Integration
- **Plotly Integration**: Full interactive visualization framework
- **Regime Analysis**: Multi-dimensional regime detection and visualization
- **PnL Tracking**: Comprehensive spread trading analysis
- **3D Capabilities**: Surface plots and advanced chart types

## Unified CurveEvolutionAnalyzer (Latest Integration)

### Complete Merger of CurveEvolution and SpreadAnalyzer
The new `CurveEvolutionAnalyzer` class in `CTAFlow/features/curve_analysis.py` provides a unified interface combining all previous functionality with advanced path signature analysis:

```python
from CTAFlow.features.curve_analysis import CurveEvolutionAnalyzer

# Complete pipeline using SpreadData.get_seq_curves()
spread_data = SpreadData('CL_F')
seq_curves = spread_data.get_seq_curves('2023-01-01':'2023-12-31')

# Create unified analyzer
analyzer = CurveEvolutionAnalyzer.from_spread_data(spread_data)

# Advanced driver analysis using log price Lévy areas
driver_analysis = analyzer.analyze_curve_evolution_drivers()
print(f"Primary driver: {driver_analysis['summary_statistics']['primary_driver']}")

# Comprehensive visualization
evolution_fig = analyzer.plot_curve_evolution_analysis(
    show_drivers=True,
    show_regimes=True,
    show_levy_areas=True
)

# F0 regime analysis
regime_fig = analyzer.plot_regime_changes_f0(
    regime_window=63,
    regime_threshold=2.0
)
```

### Key Innovations
- **Log Price Lévy Areas**: Detects fundamental curve drivers using rotational path analysis
- **Path Signatures**: Multi-level mathematical signatures for regime identification
- **Driver Detection**: Identifies contango, term structure, front/back-end, volatility, and momentum drivers
- **Advanced Regime Detection**: Uses log price dynamics for structural break identification
- **Performance Optimized**: JIT-compiled calculations with smart caching

### Driver Analysis Results
```python
driver_analysis = analyzer.analyze_curve_evolution_drivers()

# Available drivers detected from log price Lévy areas:
drivers = {
    'front_end_changes': "Near-term contract dynamics and front month pressure",
    'back_end_changes': "Long-term contract structural changes and back month flows", 
    'seasonal_deviations': "Deviations from typical seasonal curve patterns",
    'momentum': "Rate of change in curve evolution and directional momentum"
}
```