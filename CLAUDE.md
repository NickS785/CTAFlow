# CLAUDE.md

Concise guide for working on CTAFlow, a CTA positioning and orderflow analysis toolkit that combines COT data, technical signals, and screener pipelines.

## Setup quickstart
- Install in editable mode: `pip install -e .`
- Dev dependencies: `pip install -e .[dev]`
- Run tests: `python -m pytest tests/`

## Architecture snapshot
- **Data (`CTAFlow/data/`)**: `data_client.py` handles HDF5 I/O and COT refresh; `retrieval.py` exposes async loaders; contract utilities live under `contract_handling/`.
- **Features (`CTAFlow/features/`)**: `signals_processing.py` builds COT + technical indicators; `feature_engineering.py` covers intraday microstructure; `curve_analysis.py` unifies curve shape/evolution analysis.
- **Containers (`CTAFlow/data/contract_handling/`)**: `SpreadData`, `FuturesCurve`, `Contract`, and friends provide numpy-backed curve slices.
- **Models (`CTAFlow/models/`)**:
  - `base_models.py`: Wrapper classes for ML models - `CTALight` (LightGBM), `CTAXGBoost`, `CTARForest`. Support regression and classification tasks with common interface (fit/predict/evaluate).
  - `intraday_momentum.py`: `IntradayMomentumLight` wraps base models with intraday feature engineering. Key methods: `add_daily_momentum_features()`, `har_volatility_features()`, `opening_range_volatility()`, `prev_hl()`, `target_time_volume()`, `bid_ask_volume_imbalance()`. All features properly lag to avoid lookahead bias. Use `model.target_data` for consistent target calculation.
- **Forecasting (`CTAFlow/forecaster/forecast.py`)**: family of CTA models with selective indicator calculation and weekly resampling.
- **Strategy (`CTAFlow/strategy/`)**: `screener_pipeline.py` normalises screener payloads into gate columns; keep `_items_from_patterns` compatible with nested mappings and `PatternExtractor.concat_many` outputs. `HorizonMapper.build_xy` expects timezone-aware `ts`, `open`, `close`, `session_id` columns.
- **Screeners (`CTAFlow/screeners/`)**:
  - `pattern_extractor.py`: Extracts patterns from price/volume data. Preserves `SUMMARY_COLUMNS`, `_strength_raw`, and async loaders. Use `PatternExtractor.concat_many()` for combining multiple pattern outputs.
  - `historical_screener.py`: `ScreenerBacktester` for backtesting screener signals. Supports GPU-accelerated batch processing, calendar patterns, and cross-ticker alignment. Returns performance metrics and pattern statistics.
  - `orderflow_screen.py`: Provides orderflow seasonality metrics and volume-based signals.
  - `generic.py`: Generic screening utilities that work across different pattern types.

## Module Details

### Cyclical Features (`CTAFlow/features/cyclical/`)
- `CEEMDAN.py`: Rolling CEEMDAN cycle analysis with volatility segmentation. Main class `CEEMDANCycleAnalyzer` decomposes price into IMFs using ensemble EMD, extracts cycle prevalence by frequency bands, supports year/month/vol-quantile segmentation.
- `ls_periodogram.py`: **EMPTY FILE - UNUSED**

### Curve Features (`CTAFlow/features/curve/`)
- `curve_features.py`: `CurveFeatures` class for futures term structure analysis. Methods: `relative_basis()`, `butterfly()`, `condor()`, `carry_return()`, `curve_slope()`, `m1_anchored_slopes()`, `all_features()`.
- `advanced_features.py`: `CurveShapeAnalyzer` for PCA-based curve shape analysis; `CurveEvolutionAnalyzer` for curve dynamics. Also includes `calculate_seasonal_decomposition()`, `calculate_information_flow()`, `calculate_microstructure_features()`.

### Models (`CTAFlow/models/`)
- `base_models.py`: Core ML wrappers - `CTALight`, `CTAXGBoost`, `CTARForest`, `CTALinear`. Common interface with `fit/predict/evaluate`.
- `intraday_momentum.py`: `IntradayMomentum` class for intraday feature engineering + model training.
- `volatility.py`: `RVForecast` class extending `CTALinear` for realized volatility forecasting.
- `feature_selection.py`: `FeatureSelector` and `FeatureXplainer` for SHAP-based feature importance and selection.
- `pattern_forecast.py`: `PatternMLBuilder` for building ML models from pattern data.
- `positioning.py`: **EMPTY FILE - UNUSED**

### Screeners (`CTAFlow/screeners/`)
**Main Entry Point**: `historical_screener_v2.py` - `HistoricalScreenerV2` is the primary screener class.

**Core Files**:
- `historical_screener_v2.py`: Modern screener implementation with engine-based architecture.
- `pattern_extractor.py`: Pattern extraction and ranking from screener results.
- `orderflow_screen.py`: `OrderflowScanner` for tick-level orderflow seasonality analysis.
- `calendar_effects.py`: `run_calendar_edge_tests()` for statistical testing of calendar patterns.
- `generic.py`: Factory functions for creating screen params: `make_seasonality_screen()`, `make_momentum_screen()`, `make_orderflow_screen()`.

**Engine Architecture**:
- `base_engine.py`: `BaseScreenEngine` abstract base class.
- `momentum_engine.py`: `MomentumScreenEngine` for momentum-based screens.
- `seasonality_engine.py`: `SeasonalityScreenEngine` for time-of-day patterns.
- `orderflow_engine.py`: `OrderflowScreenEngine` wrapper.
- `event_engine.py`: `EventScreenEngine` for data release events.

**Support Files**:
- `params.py`: Parameter dataclasses for all screen types.
- `screener_types.py`: Screen type constants.
- `pattern_calendar.py`: `PatternVault`, `ActivePatternCalendar` for pattern storage.
- `event_screener.py`: `run_event_screener()` for event-based analysis.
- `event_presets.py`: Preset event configurations (used by `event_engine.py`).
- `gpu_stats.py`: GPU-accelerated statistics (used by `historical_screener.py`).
- `session_first_hours.py`: Session opening analysis.

**Legacy/Compatibility Wrappers**:
- `orderflow_scan.py`: Compatibility wrapper → forwards to `orderflow_screen.py`
- `data_release_screener.py`: Compatibility wrapper → forwards to `event_screener.py`
- `historical_screener.py`: Legacy screener (imports still used, but `historical_screener_v2.py` is preferred)
- `regime_screens.py`: Example script with preset configurations (not a reusable module)

## Unused/Redundant Files

### DELETE (Empty):
- `CTAFlow/features/cyclical/ls_periodogram.py` - Empty file
- `CTAFlow/models/positioning.py` - Empty file

### CONSIDER REMOVING (Compatibility Wrappers):
- `CTAFlow/screeners/orderflow_scan.py` - Just re-exports from `orderflow_screen.py`
- `CTAFlow/screeners/data_release_screener.py` - Just re-exports from `event_screener.py`

### CONSIDER DEPRECATING:
- `CTAFlow/screeners/historical_screener.py` - Legacy, replaced by `historical_screener_v2.py` (but still has imports)
- `CTAFlow/screeners/regime_screens.py` - Example script with hardcoded paths, not a reusable module

### LOW USAGE (Review for Removal):
- `CTAFlow/models/pattern_forecast.py` - Only referenced in documentation, not imported in code

## Working notes
- Seasonal/orderflow outputs feed downstream notebooks; avoid changing canonical column names or shapes.
- Keep scoring weights configurable in `PatternExtractor` helpers and maintain compatibility of screener pipelines with existing notebooks.
- Plotting, strategy backtesting, and GPU helpers exist throughout; prefer existing utilities over ad-hoc implementations.
- **Models**: Always use `model.target_data` for targets in `IntradayMomentum` to ensure consistency and proper lagging. When adding features, use `_add_feature()` for automatic tracking. LightGBM classification returns probabilities - convert to class labels with threshold (binary) or argmax (multiclass).
- **Screeners/Backtesting**: `HistoricalScreenerV2` is the preferred screener. Legacy `HistoricalScreener` still has some imports but should be migrated away from. Use vectorized operations for performance. Calendar patterns need proper gate attachment to decision timestamps.
