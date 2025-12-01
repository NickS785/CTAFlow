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
- **Forecasting (`CTAFlow/forecaster/forecast.py`)**: family of CTA models with selective indicator calculation and weekly resampling.
- **Strategy (`CTAFlow/strategy/`)**: `screener_pipeline.py` normalises screener payloads into gate columns; keep `_items_from_patterns` compatible with nested mappings and `PatternExtractor.concat_many` outputs. `HorizonMapper.build_xy` expects timezone-aware `ts`, `open`, `close`, `session_id` columns.
- **Screeners (`CTAFlow/screeners/`)**: `pattern_extractor.py` preserves `SUMMARY_COLUMNS`, `_strength_raw`, and async loaders; `orderflow_screen.py` provides orderflow seasonality metrics.

## Working notes
- Seasonal/orderflow outputs feed downstream notebooks; avoid changing canonical column names or shapes.
- Keep scoring weights configurable in `PatternExtractor` helpers and maintain compatibility of screener pipelines with existing notebooks.
- Plotting, strategy backtesting, and GPU helpers exist throughout; prefer existing utilities over ad-hoc implementations.
