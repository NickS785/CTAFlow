# CLAUDE.md

Concise guide for working on CTAFlow, a CTA positioning and orderflow analysis toolkit that combines COT data, technical signals, and screener pipelines.

## Setup quickstart
- Install in editable mode: `pip install -e .`
- Dev dependencies: `pip install -e .[dev]`
- Run tests: `python -m pytest tests/`

## Architecture snapshot
- **Data (`CTAFlow/data/`)**: `data_client.py` handles HDF5 I/O and COT refresh; `retrieval.py` exposes async loaders; contract utilities live under `contract_handling/`. `model_datasets.py` provides PyTorch datasets for multi-modal training: `DualModalWindowDataset` (summary + spatial windows for RecurrentDualModal), `TriModalWindowDataset` (summary + seq + spatial windows for RecurrentTriModal), both extending `RasterizedModalDataset` with rolling window support. `storage/` module provides production-ready storage infrastructure:
  - `model_manager.py`: `ModelManager` for model lifecycle management with training/backtest/production modes. Handles model versioning, checkpointing, and metadata tracking.
  - `aws_client.py`: `AWSClient` for S3 operations with configurable endpoints, automatic caching, and bulk ticker data downloads. `S3Config` dataclass for configuration.
  - `storage_backend.py`: Pluggable storage backends - `LocalStorage` for filesystem, `S3Storage` for cloud. Abstract `StorageBackend` interface enables custom implementations.
- **Features (`CTAFlow/features/`)**: `signals_processing.py` builds COT + technical indicators; `feature_engineering.py` covers intraday microstructure; `curve_analysis.py` unifies curve shape/evolution analysis. **CRITICAL**: Profile and NumberBars now include explicit price labels (4th channel) using unified normalization `(price - reference) / reference` for tri-modal alignment.
- **Containers (`CTAFlow/data/contract_handling/`)**: `SpreadData`, `FuturesCurve`, `Contract`, and friends provide numpy-backed curve slices.
- **Models (`CTAFlow/models/`)**:
  - `base_models.py`: Wrapper classes for ML models - `CTALight` (LightGBM), `CTAXGBoost`, `CTARForest`. Support regression and classification tasks with common interface (fit/predict/evaluate).
  - `intraday_momentum.py`: `IntradayMomentumLight` wraps base models with intraday feature engineering. Key methods: `add_daily_momentum_features()`, `har_volatility_features()`, `opening_range_volatility()`, `prev_hl()`, `target_time_volume()`, `bid_ask_volume_imbalance()`. All features properly lag to avoid lookahead bias. Use `model.target_data` for consistent target calculation. `DeepIDMomentum` extends this for deep learning with feature scaling methods: `normalize_sequential_features(scale_to_basis_points=True, scale_orderflow=True)` scales VPIN/sequential features to match spatial data scale (basis points); `scale_summary_data(rolling_window=252)` applies feature-specific scaling to summary features using fixed constants (no lookahead). **Shortcut**: Use `DeepIDMomentum.get_loaders()` to quickly create train/val DataLoaders for dual/tri/quad-modal models.
  - `deep_learning/multi_branch/`: Multi-modal deep learning models for orderflow prediction:
    - `dual_model.py`:
      - `DualBranchModel`: Combines summary (MLP) + sequential (LSTM) branches
      - `RecurrentDualModal`: State-of-the-art windowed model with Summary MLP + MarketProfileResNet (static) + RasterResNet (dynamic) + Window LSTM for temporal modeling
      - `RecurrentWSPR`: **Windowed Spatial-Profile-Raster** 5-path model for single-ticker prediction. Architecture: (1) Summary LSTM processes windowed daily features, (2) Profile LSTM processes windowed market profiles, (3) Recent Raster encoder (RasterResNet) processes most recent day's rasterized VPIN, (4) Recent Sequential encoder processes most recent day's intraday sequence, (5) Recent Spatial Fusion (gated) combines recent profile+raster. All paths merge for final prediction. Key insight: only most recent day's fine-grained data (raster/sequential) needed for prediction, while coarse features use full window history.
      - `MultiAssetWSPR`: **Multi-Ticker WSPR** 6-path extension adding MetaModalityEncoder (path 6) for cross-asset training. Encodes ticker identity (ticker_id, asset_class_id, asset_subclass_id) and calendar features (month, dow, doy_sin, doy_cos) into rich embeddings. Enables transfer learning across correlated markets (e.g., HE+LE livestock futures). Use with `MultiAssetMomentum.get_loaders(use_wspr=True, ticker_id_map={...})` for automatic metadata injection via `wspr_collate_fn`. Forward signature: `model(summary_days, profile_days, raster_recent, seq_recent, seq_lens_recent, meta: dict)`. Returns raw logits/values or probabilities with `return_probs=True`.
    - `tri_modal.py`: Tri-modal architectures with spatial fusion:
      - `TriModalModel`: Main model with optional `nb_tensor` parameter for NumberBars
      - `TriModalLiquidityModel`: Configurable with `fusion_mode` ('gated'/'concat'/'mean') and custom encoders
      - `TriModalClassifier`: Classification-optimized with concatenation fusion and deeper head
      - `RecurrentTriModal`: Advanced windowed model processing Summary + Profile + Raster + Intraday Seq through spatial fusion, day fusion, then Window LSTM
  - `deep_learning/encoders.py`: Modular encoder components:
    - Basic encoders: `ProfileEncoder`, `NumberBarsEncoder`, `SeqEncoder`, `SummaryEncoder` for feature extraction
    - Advanced spatial encoders: `RasterResNet` (pseudo-3D ResNet for rasterized VPIN with spatio-temporal convolutions), `MarketProfileResNet` (1D ResNet + SE attention for volume profiles)
    - Fusion modules: `SpatialFuse` (combines profile + raster/NumberBars with optional gating), `GatedFusion` (learnable multi-modal weighting)
  - `multi_asset.py`: `MultiAssetMomentum` for multi-ticker training with automatic schema alignment. Expects directory structure `<root>/<TICKER>/` with `features.csv`, `profiles.npz`, `vpin.parquet`, `rasterized.npz`, `target.csv`. Three summary alignment strategies:
    - `"exact"`: Strict column intersection (fast, may be small)
    - `"signature"`: Match features by (window, remainder) signatures across different anchors (e.g., `0930_60min_vol` matches `1000_60min_vol`)
    - `"recompute"`: Ignore features.csv, recompute universal session return/volatility features
    - Supports parallel loading (`parallel_load=True`, `max_workers=4`) and dataset caching (`save_cache=True`, `cache_path=...`)
- **Macro Pipeline (`CTAFlow/data/ext/macro_client.py` + `CTAFlow/features/macro_prep.py`)**: Two-source macro feature pipeline for model training.
  - `MacroClient`: Fetches FRED yields (DGS10, DGS2) and economic indicators (CPI, Core CPI, PCE, Core PCE, GDP nominal/real, unemployment, fed funds, nonfarm payrolls, UMich sentiment) via `pandas_datareader`. YoY transforms applied at native frequency; `include_sectors=False` by default excludes sector ETFs. Key methods: `fetch_fred_data()`, `fetch_econ_data()`, `get_macro_context()`.
  - `MacroFeaturePrep`: Processes raw macro context into features. Produces: rate diffs (1d/30d), term spread, econ YoY levels + release-day changes (`_chg` = `diff(1)` on forward-filled data, non-zero only on FRED release dates), derived composites: `REAL_RATE_10Y/2Y` (yield - CPI_YOY), `REAL_FF_RATE` (FEDFUNDS - CPI_YOY), `FF_SPREAD_10Y/2Y` (yield - FEDFUNDS). Output: 30 econ feature columns.
  - **TFT notebook pipeline**: `MarketFeatureEngine` (z-scored market returns) + `MacroClient.fetch_fred_data()` / `fetch_econ_data()` → `MacroFeaturePrep.process()` → rolling z-score → merge → `TFTAlignedPrepLayer` (shifted 1d). Market tickers (SPX/VIX/DXY) handled exclusively by `MarketFeatureEngine`; FRED data handled exclusively by `MacroFeaturePrep` — no overlap.
  - `TFTAlignedPrepLayer._scale_macro_features()` handles econ columns: `_yoy` suffix /2, `UMCSENT` /20, `FEDFUNDS` /4, `UNRATE` /2, `_chg` columns ×10.
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
- `multi_asset.py`: `MultiAssetMomentum` for training across multiple tickers with schema alignment. Key classes: `GenericFiles` (filename specs), `SummarySelectionConfig` (alignment strategy). Uses parallel file loading (`ThreadPoolExecutor`) and supports dataset caching to avoid recomputation.
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
- **Models**: Always use `model.target_data` for targets in `IntradayMomentum` to ensure consistency and proper lagging. When adding features, use `_add_feature()` for automatic tracking. LightGBM classification returns probabilities - convert to class labels with threshold (binary) or argmax (multiclass). For deep learning: call `normalize_sequential_features(scale_to_basis_points=True, scale_orderflow=True)` and `scale_summary_data()` to ensure all features are on comparable scales (~[-5, +5] basis points) matching the spatial data normalization `(price - vwap) / vwap * 100`. Rasterization: `SequenceRasterizer` should use `session_start=None` to process all VPIN data by time bounds (VPIN extraction already filters session times).
- **Screeners/Backtesting**: `HistoricalScreenerV2` is the preferred screener. Legacy `HistoricalScreener` still has some imports but should be migrated away from. Use vectorized operations for performance. Calendar patterns need proper gate attachment to decision timestamps.
- **Macro Features**: `MacroClient(include_sectors=False)` is the default — sector ETFs excluded. For TFT training, fetch FRED data directly via `fetch_fred_data()` + `fetch_econ_data()`, process with `MacroFeaturePrep`, z-score scale, then merge with `MarketFeatureEngine` output. Do NOT pass yfinance market tickers through `MacroFeaturePrep` — that creates redundant return features. The `FREDDataFetcher` in `CTAFlow/data/ext/fred.py` is a standalone utility not integrated into the pipeline; use `MacroClient` instead.
- **Storage & Deployment**: Use `ModelManager` for all model operations - it handles versioning, metadata, and multi-backend storage. Training mode: `ModelManager(storage, mode=ModelMode.TRAINING)` for checkpointing; Backtest mode: `ModelMode.BACKTEST` for historical simulations; Production mode: `ModelMode.PRODUCTION` for live inference with strict validation. `AWSClient` handles S3 with automatic caching (`cache_dir`), configurable endpoints (`endpoint_url`), and bulk ticker downloads (`download_ticker_data()`). Expected S3 structure: `s3://bucket/data/{TICKER}/features.csv`, `s3://bucket/models/{model_name}.pth`. Use `create_model_manager(mode='production', use_s3=True, s3_config={...})` for quick setup.
