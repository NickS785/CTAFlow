# AGENTS.md

This file is a fast handoff for coding agents. `CLAUDE.md` is the full architecture + conventions reference; use this file to stay aligned while making focused changes.

- Project: CTA positioning and orderflow forecasting toolkit (data loaders, features, screeners, forecasting, deep learning).
- Setup: `pip install -e .`; dev extras with `pip install -e .[dev]`; tests with `python -m pytest tests/`.
- Core paths: `CTAFlow/data/`, `CTAFlow/features/`, `CTAFlow/models/`, `CTAFlow/screeners/`, `CTAFlow/strategy/`.
- Compatibility guardrails: keep screener payload shapes stable (`_items_from_patterns` nested mappings, `PatternExtractor.SUMMARY_COLUMNS`, `_strength_raw`, timezone-aware OHLC in `HorizonMapper.build_xy`).
- Model/data guardrails: downstream notebooks depend on stable dataset column names and modal payload formats (summary/seq/profile/raster).
- Implementation preference: reuse existing GPU/backtesting/storage utilities before introducing new abstractions.
