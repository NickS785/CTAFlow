# AGENTS.md

Quick orientation for contributors (see `CLAUDE.md` for more detail).

- Project: CTA positioning and orderflow forecasting toolkit with data loaders, feature engines, and screeners.
- Setup: `pip install -e .` for editable installs, `pip install -e .[dev]` for dev deps, tests via `python -m pytest tests/`.
- Key modules: data access (`CTAFlow/data/`), feature engineering (`CTAFlow/features/`), screener/strategy pipeline (`CTAFlow/strategy/screener_pipeline.py`, `CTAFlow/screeners/`), and forecasting models (`CTAFlow/forecaster/forecast.py`).
- Preserve canonical screener shapes: `_items_from_patterns` must handle nested mappings; `PatternExtractor` keeps `SUMMARY_COLUMNS` and `_strength_raw`; `HorizonMapper.build_xy` relies on timezone-aware OHLC columns.
- Use existing GPU/backtesting utilities instead of ad-hoc implementations; downstream notebooks expect stable column names and payload formats.
