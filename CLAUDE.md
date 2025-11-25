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
**Primary Data Processing (Post-Phase 3 Simplification):**
- **`simple_processor.py`**: **PRIMARY DATA PROCESSOR** - Unified data processing pipeline
  - `SimpleDataProcessor`: Main processor for all data operations using update files approach
  - COT data refresh via DataClient methods
  - Market data updates from CSV files with previous data combination and verification
  - Futures curve processing using DlyContractManager patterns
- **`data_client.py`**: Core data interface class for HDF5 operations and COT processing
  - Comprehensive COT methods: `refresh_latest_cot_year()`, `download_cot()`, `query_cot_metrics()`
  - Market data storage and retrieval with proper data combination
- **`retrieval.py`**: Asynchronous data loading from HDF5 stores using `fetch_data_sync()`

**Data Management & Classification:**
- **`contract_handling/dly_contract_manager.py`**: DLY file processing and HDF5 storage blueprint
  - `DLYContractManager.save_hdf()`: Standard pattern for curve storage
  - Contract expiry calculations and curve building from DLY files
- **`ticker_classifier.py`**: Automatic classification of tickers into commodity/financial types
- **`classifications_reference.py`**: Reference mappings for all supported tickers
- **`futures_mappings.toml`**: Maps futures ticker symbols to COT commodity codes

**Legacy Processors (Deprecated):**
- **`data_processor.py`**: Complex threaded processor (use SimpleDataProcessor instead)
- **`enhanced_curve_processor.py`**: Enhanced curve processing (use SimpleDataProcessor.batch_update_futures_curves instead)

### Features & Signal Processing (`CTAFlow/features/`) **[PRIMARY PROCESSING LOCATION]**
**This is where all signal processing and feature engineering is concentrated:**
- **`signals_processing.py`**: Core technical analysis and COT signal generation
  - `COTAnalyzer`: Handles COT data cleaning and positioning metrics
  - `TechnicalAnalysis`: Calculates selective technical indicators and volatility normalization
- **`feature_engineering.py`**: Intraday microstructure features
  - `IntradayFeatures`: Microstructure and intraday feature extraction
- **`curve_analysis.py`**: Unified futures curve analysis framework (Post-Phase 2 Consolidation)
  - `CurveShapeAnalyzer`: Curve shape analysis and feature extraction (includes evolution tracking from former CurveEvolution)
  - `CurveEvolutionAnalyzer`: Advanced curve evolution analysis (includes seasonal methods from former SpreadAnalyzer)
  - Unified interface combining path signature analysis, Lévy areas, and regime detection

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
**Core Trading Strategy Components:**

- **`screener_pipeline.py`**: Pattern materialisation and feature engineering pipeline
  - `ScreenerPipeline`: Transforms screener patterns into sparse gate columns on bar data
    - `build_features(bars, patterns)`: Main feature generation method, supports parallel processing via max_workers
    - `backtest_threshold()`: Quick backtest using HorizonMapper for return calculation
    - `build_and_backtest()`: Convenience wrapper combining feature generation and backtesting
    - `concurrent_pattern_backtests()`: Parallel backtesting of individual patterns for ranking
    - Pattern dispatch handles seasonal, orderflow, and momentum pattern types
    - Keep `_items_from_patterns()` compatible with nested mappings, (key, pattern) tuples, and generator inputs

  - `HorizonMapper`: Return calculation engine aligning gates with realized returns
    - `build_xy(bars_with_features, patterns)`: Creates (returns_x, returns_y) pairs for backtest
    - Requires timezone-aware `ts`, `open`, `close`, and `session_id` columns for stability
    - Supports multiple horizon types: intraday minutes, session close, next-day, next-week, Monday close
    - Weekend exit policies: 'first', 'last', 'average' for Friday-to-Monday patterns
    - Configurable NaN policies: 'drop', 'zero', 'ffill' for cleaning realized returns

- **`backtester.py`**: Backtesting engines for pattern validation and strategy evaluation
  - `ScreenerBacktester`: Lightweight backtester for ScreenerPipeline.build_xy outputs
    - `threshold(xy, threshold=0.0)`: Simple threshold-based backtest on returns_x signal
    - Returns: pnl, positions, cumulative, summary (BacktestSummary), monthly, daily_pnl
    - Supports group_field for multi-instrument aggregation
    - Uses PredictionToPosition for collision resolution when multiple patterns fire

  - `FeatureBacktester`: Full-featured backtester with configurable entry/exit logic
    - `run(data, signal, execution)`: Complete backtest with SignalSpec + ExecutionPolicy
    - SignalSpec: signal_col, trigger threshold, allow_short flag
    - ExecutionPolicy: price_in/out, bias (long/short/signed), exit_policy, holding_bars
    - Exit policies: 'bars', 'session_close', 'next_day_close', 'monday_close', 'target_pct'
    - Returns: pnl, positions, cumulative, trades DataFrame, summary

  - `MomentumBacktester`: Specialized backtester for intraday momentum screen results
    - `build_xy(results, session_key, predictor, target)`: Converts nested momentum results to XY frame
    - `threshold_backtest()`: Quick threshold test on momentum signals
    - Designed for HistoricalScreener.intraday_momentum_screen outputs

  - `BacktestSummary`: Standardized metrics container
    - total_return, mean_return, hit_rate, sharpe, max_drawdown, trades

  - Helper Functions:
    - `build_overlay_frame()`: Align cumulative PnL with price for visualization
    - `plot_backtest_results()`: Matplotlib plotting of PnL and price overlay
    - `diagnose_alignment()`: Debugging tool for timestamp alignment issues

- **`prediction_to_position.py`**: Collision resolution for overlapping pattern signals
  - `PredictionToPosition`: Aggregates multiple concurrent pattern signals
    - `aggregate(xy)`: Combines signals firing at same timestamp
    - `resolve(xy, group_field)`: Selects best signal when multiple patterns collide
    - Default selection: highest abs(returns_x), fallback to correlation, then pattern_type

### Screeners (`CTAFlow/screeners/`)
**Pattern Discovery and Statistical Validation:**

- **`pattern_extractor.py`**: Screener output restructuring and pattern ranking
  - `PatternExtractor`: Main class for organizing and analyzing screener results
    - Constructor: `PatternExtractor(screener, results, screen_params, metadata)`
    - `patterns`: Property returning nested dict of pattern summaries by symbol
    - `concat(other, conflict='prefer_strong')`: Merge multiple extractor outputs
    - `top_patterns(symbol, n=5)`: Ranked patterns for a ticker by strength score
    - `top_patterns_global(n=10)`: Top patterns across all tickers
    - `filter_patterns(symbol, pattern_types, screen_names)`: Subset extraction
    - `get_filtered_months()`: Returns canonical set of active months (1-12)
    - `concat_many(extractors)`: Static method for bulk merging
    - Preserves SUMMARY_COLUMNS for downstream consumption
    - Async loaders: `load_summaries_from_results_async()` for parallel loading

  - `PatternSummary`: Dataclass describing a single pattern
    - key, symbol, source_screen, screen_params, pattern_type, description, strength
    - payload: Dict containing pattern-specific data (mean, p_value, correlation, etc.)
    - metadata: Dict with session params, momentum_type, bias, etc.
    - `as_dict()`: Serializes to JSON-compatible format for pipeline consumption
    - `get_session_params()`: Extracts canonical session parameters

  - Strength Scoring:
    - `_strength_raw()`: Computes significance score from p-value, effect size, sample size
    - Configurable weights: p_weight, effect_weight, n_weight
    - Used for deterministic pattern ranking

- **`historical_screener.py`**: Main screener engine for seasonal and momentum patterns
  - `HistoricalScreener`: Primary pattern discovery class
    - Constructor: `HistoricalScreener(ticker_data, file_mgr, results_client, auto_write_results)`
    - `intraday_momentum_screen()`: Detects intraday momentum patterns across sessions
      - Parameters: session_starts/ends lists, st_momentum_days, sess_start/end hours/minutes
      - Supports season filters ('winter', 'spring', 'summer', 'fall')
      - Supports month filters: specific months list (1-12)
      - Regime filtering: use_regime_filtering, regime_col, target_regimes, regime_settings
      - Returns nested dict: {ticker: {session_key: {momentum_type: stats}}}

    - `intraday_seasonality_screen()`: Time-of-day and day-of-week seasonality
      - Parameters: target_times, period_length, dayofweek_screen, session_start/end
      - Detects predictive time-of-day patterns (next-day, next-week correlations)
      - Weekend hedging detection: Friday-to-Monday correlation analysis
      - Returns: time_predictive_nextday, time_predictive_nextweek, weekday_returns, weekend_hedging

    - `composite_screen(screen_params_list)`: Multi-screen orchestration
      - Takes List[ScreenParams] for parallel execution of multiple screens
      - Returns consolidated dict with all screen results

    - `write_results_to_store()`: Persist results via ResultsClient
    - `set_results_client()`: Configure persistence backend

  - `ScreenParams`: Dataclass for screen configuration
    - screen_type: 'momentum' or 'seasonality'
    - name: Optional custom name for the screen
    - season: 'winter', 'spring', 'summer', 'fall' (auto-mapped to months)
    - months: List[int] for custom month selection
    - Momentum params: session_starts/ends, st_momentum_days, sess_start/end hrs/minutes, test_vol
    - Seasonality params: target_times, period_length, dayofweek_screen, session_start/end, tz
    - Regime filtering: use_regime_filtering, regime_col, target_regimes, regime_settings
    - Auto-generates name if not provided

- **`orderflow_scan.py`**: Volume-based orderflow pressure analysis
  - `OrderflowScanner`: Tick-level orderflow analysis class
    - `scan_tickers()`: Main method for multi-ticker orderflow analysis
    - Computes buy/sell pressure, VPIN, net_pressure from tick data
    - Detects seasonal orderflow patterns (weekly, week-of-month, peak pressure)
    - Parallel processing with configurable max_workers
    - Returns: df_weekly, df_wom_weekday, df_weekly_peak_pressure

  - `OrderflowParams`: Configuration dataclass
    - session_start/end: Session window for tick filtering
    - tz: Timezone for session localization
    - bucket_size: Volume bucket size ('auto' or integer)
    - vpin_window: VPIN calculation window
    - min_days: Minimum sample size for statistical tests
    - cadence_target: Target buckets per day for auto bucket sizing
    - grid_multipliers: Multipliers for grid search in auto mode
    - month_filter/season_filter: Period filtering options

  - `orderflow_scan()`: Functional interface for single-ticker analysis
    - Alternative to OrderflowScanner class-based API
    - Takes tick data, OrderflowParams, returns analysis results

- **`session_first_hours.py`**: First hours momentum and volume analysis
  - `run_session_first_hours()`: Analyzes opening hour momentum patterns
  - `SessionFirstHoursParams`: Configuration for first hours analysis
  - Detects opening momentum, relative volume patterns

- **`__init__.py`**: Package exports
  - Re-exports: PatternExtractor, PatternSummary, HistoricalScreener, ScreenParams
  - Re-exports: OrderflowParams, OrderflowScanner, orderflow_scan
  - Re-exports: SessionFirstHoursParams, run_session_first_hours

### Screener Workflow Integration
**Complete Pattern Discovery to Backtest Pipeline:**

```python
from CTAFlow.screeners import HistoricalScreener, ScreenParams, PatternExtractor
from CTAFlow.strategy import ScreenerPipeline, ScreenerBacktester

# 1. Configure screens
winter_momentum = ScreenParams(
    screen_type='momentum',
    season='winter',
    session_starts=["02:30", "08:30"],
    session_ends=["10:30", "13:30"]
)

spring_seasonality = ScreenParams(
    screen_type='seasonality',
    season='spring',
    target_times=["09:30", "14:00"],
    dayofweek_screen=True
)

# 2. Run screener
screener = HistoricalScreener(ticker_data)
results = screener.composite_screen([winter_momentum, spring_seasonality])

# 3. Extract and rank patterns
extractor = PatternExtractor(screener, results, [winter_momentum, spring_seasonality])
top_patterns = extractor.top_patterns_global(n=20)

# 4. Materialize features on bar data
pipeline = ScreenerPipeline(tz="America/Chicago")
bars_with_features = pipeline.build_features(bars, extractor.patterns)

# 5. Backtest patterns
backtester = ScreenerBacktester(annualisation=252)
results = pipeline.backtest_threshold(
    bars_with_features,
    extractor.patterns,
    threshold=0.0,
    use_side_hint=True
)

print(f"Sharpe: {results['summary'].sharpe:.2f}")
print(f"Total Return: {results['summary'].total_return:.4f}")
print(f"Trades: {results['summary'].trades}")
```

### Data Pipeline (`CTAFlow/data/`)
**Simplified Processing Pipeline (Post-Phase 3):**
```python
# Primary workflow using SimpleDataProcessor
processor = SimpleDataProcessor()

# 1. Update all data types in coordinated fashion
results = processor.update_all_tickers(
    dly_folder="path/to/dly/files",
    selected_cot_features=['positioning', 'flows'],
    max_workers=4
)

# 2. Individual market updates using CSV files
market_result = processor.update_market_from_csv(
    symbol='CL_F',
    csv_folder="F:/charts/"
)

# 3. Futures curve updates with data combination and verification
curve_result = processor.update_futures_curve_from_dly(
    symbol='CL_F',
    dly_folder="path/to/dly/files"
)
```

**Key Pipeline Features:**
- **Update Files Approach**: Uses CSV update files (pattern: `{symbol}-update-*.csv`)
- **Data Combination**: Combines previous and new data with verification before saving
- **DlyContractManager Integration**: Uses proven HDF5 storage patterns
- **COT Integration**: Coordinated COT data refresh using DataClient methods
- **Error Handling**: Comprehensive error tracking and progress reporting

### Dashboard & Visualization
**Interactive Analysis Framework:**
- **Plotly Integration**: Full interactive visualization capabilities
- **Regime Analysis**: Multi-dimensional regime detection with color-coded visualization
- **3D Curve Evolution**: Surface plots showing curve evolution over time
- **PnL Tracking**: Comprehensive spread trading analysis with drawdown visualization

```python
# Regime change analysis
from CTAFlow.features.curve_analysis import CurveEvolutionAnalyzer

analyzer = CurveEvolutionAnalyzer.from_spread_data(spread_data)
regime_fig = analyzer.plot_regime_changes_f0(
    regime_window=63,
    regime_threshold=2.0,
    show_seasonal=True
)

# Driver analysis visualization
evolution_fig = analyzer.plot_curve_evolution_analysis(
    show_drivers=True,
    show_regimes=True,
    show_levy_areas=True
)
```

### Position Forecaster (`CTAFlow/forecaster/`)
**Unified Forecasting Interface:**
- **CTAForecast**: Main forecasting class with selective indicator calculation
- **Model Classes**: CTALinear, CTALight, CTAXGBoost, CTARForest
- **Feature Engineering**: Combines COT positioning with technical signals
- **Weekly Resampling**: Aligned with COT reporting schedule (Friday default)

```python
# Complete forecasting workflow
forecaster = CTAFlow.CTAForecast('CL_F')

# Selective indicator calculation for efficiency
data = forecaster.prepare_data(
    selected_indicators=['moving_averages', 'rsi', 'atr'],
    normalize_momentum=True,
    resample_weekly=True
)

# Train and evaluate models
result = forecaster.train_model(model_type='lightgbm', target_type='return')
print(f"Test Accuracy: {result['test_metrics']['accuracy']:.3f}")
```

### Utilities & Configuration
- **`CTAFlow/config.py`**: HDF5 data paths and environment setup
- **`project_goals.md`**: Detailed research methodology and indicator selection rationale
- **Package Utilities**: Version info, supported models/indicators, data paths

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

### Simplified Pipeline (Post-Phase 3)
1. **Update**: `SimpleDataProcessor.update_all_tickers()` - Coordinated data refresh using update files
2. **Combine**: Previous and new data combination with verification before saving
3. **Load**: `fetch_market_cot_data()` loads both market and COT data asynchronously
4. **Process**: `TechnicalAnalysis.calculate_enhanced_indicators()` with selective calculation
5. **Engineer**: `Forecaster.prepare_forecasting_features()` combines all feature types
6. **Resample**: Optional weekly resampling (post-calculation or pre-calculation)
7. **Target**: `create_target_variable()` generates prediction targets

### Pipeline Integration Points
- **Data Ingestion**: CSV update files → SimpleDataProcessor → HDF5 storage
- **COT Processing**: DataClient.refresh_latest_cot_year() → COT metrics calculation
- **Curve Management**: DLY files → DlyContractManager patterns → Sequentialized curves
- **Feature Engineering**: Market + COT data → TechnicalAnalysis + COTAnalyzer → ML features

## Common Usage Patterns

### Package Import and Basic Usage

```python
import CTAFlow

# Primary data processing (Post-Phase 3)
processor = CTAFlow.SimpleDataProcessor()

# Update all data coordinated refresh
results = processor.update_all_tickers(
    dly_folder="path/to/dly/files",
    selected_cot_features=['positioning', 'flows']
)

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
# Primary data processing classes (Post-Phase 3)
processor = CTAFlow.SimpleDataProcessor()
data_client = CTAFlow.DataClient()

# Access specific model classes
linear_model = CTAFlow.CTALinear('CL_F')
lightgbm_model = CTAFlow.CTALight('CL_F')
xgb_model = CTAFlow.CTAXGBoost('CL_F')
rf_model = CTAFlow.CTARForest('CL_F')

# Feature processing classes (from CTAFlow/features/)
cot_analyzer = CTAFlow.COTAnalyzer()
tech_analysis = CTAFlow.TechnicalAnalysis()
intraday_features = CTAFlow.IntradayFeatures('CL_F')

# Analysis classes (Post-Phase 2 Consolidation)
curve_shape_analyzer = CTAFlow.CurveShapeAnalyzer()
curve_evolution_analyzer = CTAFlow.CurveEvolutionAnalyzer()

# Data container classes
spread_data = CTAFlow.SpreadData('CL_F')
futures_curve_manager = CTAFlow.FuturesCurveManager()
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

### Simplified Data Processing Workflow (Post-Phase 3)

```python
# Complete data processing pipeline using SimpleDataProcessor
processor = CTAFlow.SimpleDataProcessor()

# 1. Coordinated data refresh (COT + Market + Curves)
update_results = processor.update_all_tickers(
    dly_folder="data/dly/",
    pattern="CL|NG|ZC",  # Focus on specific tickers
    selected_cot_features=['positioning', 'flows', 'extremes'],
    max_workers=4,
    cot_progress=True
)

# 2. Individual market data updates using CSV files
market_update = processor.update_market_from_csv(
    symbol='CL_F',
    csv_folder="F:/charts/",
    resample_rule="1T"
)

# 3. Futures curve processing with data combination
curve_update = processor.update_futures_curve_from_dly(
    symbol='CL_F',
    dly_folder="data/dly/"
)

# 4. Batch processing multiple symbols
batch_results = processor.batch_update_futures_curves(
    symbols=['CL_F', 'NG_F', 'ZC_F'],
    dly_folder="data/dly/",
    max_workers=2
)

print(f"Updated {update_results['processed_symbols']} symbols successfully")
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