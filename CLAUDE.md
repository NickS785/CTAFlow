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

## Working notes
- Seasonal/orderflow outputs feed downstream notebooks; avoid changing canonical column names or shapes.
- Keep scoring weights configurable in `PatternExtractor` helpers and maintain compatibility of screener pipelines with existing notebooks.
- Plotting, strategy backtesting, and GPU helpers exist throughout; prefer existing utilities over ad-hoc implementations.
- **Models**: Always use `model.target_data` for targets in `IntradayMomentumLight` to ensure consistency and proper lagging. When adding features, use `_add_feature()` for automatic tracking. LightGBM classification returns probabilities - convert to class labels with threshold (binary) or argmax (multiclass).
- **Screeners/Backtesting**: `ScreenerBacktester` expects aligned data across tickers when doing multi-ticker backtests. Use vectorized operations for performance. Calendar patterns need proper gate attachment to decision timestamps.
