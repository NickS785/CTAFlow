# Model Extraction Plan

## Goal

Slim the repository to the modeling stack that supports the core purpose:

- `CTAFlow/models/deep_learning/`
- `CTAFlow/models/prep/`
- `CTAFlow/models/intraday_momentum.py`
- `CTAFlow/models/multi_asset.py`

and only the data/feature/utils dependencies required to keep those workflows working.

This is a pruning plan, not a deletion list. Several current imports are broad package-level exports or stale alias imports, so the first job is to reduce ambiguity before removing files.

## Current MMTFv3 Split

### 1. Stateful supervised variant

Primary files:

- `CTAFlow/models/deep_learning/multi_branch/tft/c_mmtft.py`
- `CTAFlow/models/deep_learning/multi_branch/tft/mmtf_v3_core.py`
- `notebooks/mmtfv3_cl_gc_optuna.ipynb`

Key behavior:

- `MMTFv3Core` builds AE regime conditioning, temporal backbone, cross-modal fusion, and output head input.
- `StatefulMMTFv3Core` adds `TickerPositionStateLayer` and is the main supervised trading model.
- Optional PTP path uses `QuantilePositionHead -> PredictionToPosition -> continuous position`.
- PTP now receives AE hidden context and temporal context directly.

Training behavior:

- `PTPLoss = profit-weighted CE + ContinuousTradingLoss + AE reconstruction`.
- `ContinuousTradingLoss` now supports `downside_vol_weight`.
- `SharpeScheduler` adjusts the inner trading-loss weights over epochs.
- `HeadAwarePTPScheduler` adjusts LR separately for trunk, quantile head, and PTP mapper.

### 2. RL feature-extractor variant

Primary file:

- `CTAFlow/models/deep_learning/rl/features_extractor.py`

Key behavior:

- `V3ContinuousExtractor` reuses AE/static-context/backbone ideas for PPO.
- It is not stateful and does not use `StatefulMMTFv3Core`.
- It depends on `gymnasium` and the SB3 RL stack, which should be treated as optional if supervised-only extraction is desired.

## Verified dependency buckets

The following buckets were found by a recursive static import walk starting from:

- `CTAFlow/models/deep_learning/`
- `CTAFlow/models/prep/`
- `CTAFlow/models/intraday_momentum.py`
- `CTAFlow/models/multi_asset.py`

### Keep candidates inside `CTAFlow/models`

- `CTAFlow/models/base_models.py`
- `CTAFlow/models/feature_selection.py`
- `CTAFlow/models/tft_aligned.py`
- `CTAFlow/models/prep/*`
- `CTAFlow/models/deep_learning/*`
- `CTAFlow/models/intraday_momentum.py`
- `CTAFlow/models/multi_asset.py`

Notes:

- Some imports flow through `CTAFlow/models/__init__.py` or other barrel modules and may be removable once direct imports replace package-level imports.
- `CTAFlow/models/futures_models/` looks like a likely non-core candidate, but do not delete it until notebook and grep verification confirm no retained workflow depends on it.

### Required `CTAFlow/data` dependencies discovered

High-confidence dependencies:

- `CTAFlow/data/dataset_utils.py`
- `CTAFlow/data/datasets/continuous.py`
- `CTAFlow/data/datasets/tft.py`
- `CTAFlow/data/model_datasets.py`
- `CTAFlow/data/raw_formatting/intraday_manager.py`

Additional dependencies reached through broader model/data flows:

- `CTAFlow/data/data_client.py`
- `CTAFlow/data/data_processor.py`
- `CTAFlow/data/raw_formatting/synthetic.py`
- `CTAFlow/data/raw_formatting/spread_manager.py`
- `CTAFlow/data/raw_formatting/contract_specs.py`
- `CTAFlow/data/raw_formatting/dly_contract_manager.py`
- `CTAFlow/data/contract_expiry_rules.py`
- `CTAFlow/data/retrieval.py`
- `CTAFlow/data/ticker_classifier.py`
- `CTAFlow/data/update_management.py`

Action:

- Treat the first group as initial keep-set.
- Audit the second group by direct grep and runtime smoke tests before deciding whether they remain core.

### Required `CTAFlow/features` dependencies discovered

High-confidence dependencies:

- `CTAFlow/features/dt_features.py`
- `CTAFlow/features/session_features.py`
- `CTAFlow/features/curve/curve_features.py`
- `CTAFlow/features/volume/profile.py`
- `CTAFlow/features/volume/vpin.py`
- `CTAFlow/features/macro_prep.py`
- `CTAFlow/features/cyclical/seasonality/diurnal_seasonality.py`

Likely indirect/export dependencies:

- `CTAFlow/features/__init__.py` exports used by `intraday_momentum.py`
- `CTAFlow/features/base_extractor.py`
- `CTAFlow/features/signals_processing.py`

Action:

- Replace broad `from ..features import ...` usage with direct-file imports where possible.
- Re-evaluate `features/__init__.py` after direct imports are in place.

### Required `CTAFlow/utils` dependencies discovered

- `CTAFlow/utils/session.py`
- `CTAFlow/utils/tenor_interpolation.py`
- `CTAFlow/utils/unit_conversions.py`

### Non-model packages currently reached

- `CTAFlow/config`

This likely stays only if downstream retained modules still read configuration there after import cleanup.

## Cleanup risks already visible

### 1. Barrel imports are inflating the dependency graph

Problem:

- `CTAFlow/models/deep_learning/__init__.py`
- `CTAFlow/models/deep_learning/multi_branch/__init__.py`
- `CTAFlow/models/deep_learning/multi_branch/tft/__init__.py`
- `CTAFlow/models/prep/__init__.py`

These cause broad transitive imports and make unused modules look required.

Action:

- Replace broad package imports with direct module imports in retained notebooks/tests first.
- After that, shrink or remove package-level exports that are no longer needed.

### 2. There are stale/legacy import aliases

The static walk surfaced many paths that do not resolve cleanly as real modules. These are signs of legacy alias imports, outdated re-exports, or historical path moves.

Action:

- Fix import paths before pruning anything.
- Prefer explicit imports from concrete files over convenience aliases.

### 3. The RL and TensorFlow stacks should be treated separately

Likely optional stacks:

- `CTAFlow/models/deep_learning/rl/*` requires `gymnasium` and SB3.
- `CTAFlow/models/deep_learning/tf/*` requires TensorFlow.

If the retained core purpose is supervised PyTorch forecasting/trading only, these stacks should be evaluated as separate removable bundles.

## Execution plan

### Phase 1. Freeze the retained workflows

Define the exact workflows that must survive:

- Supervised stateful MMTFv3 notebook flow.
- PTP training loop importability.
- `prep` dataset/build pipeline.
- `IntradayMomentum` import and basic feature assembly.
- `multi_asset` import and dataset construction.
- Optional: RL `V3ContinuousExtractor` / PPO workflow if RL is still in scope.

Exit criteria:

- One smoke test or notebook cell path per retained workflow is identified.

### Phase 2. Replace barrel imports with direct imports

Targets:

- retained notebooks
- retained tests
- `CTAFlow/models/prep/*`
- `CTAFlow/models/intraday_momentum.py`
- `CTAFlow/models/multi_asset.py`
- `CTAFlow/models/deep_learning/*` entrypoints

Actions:

- Replace `from CTAFlow.models.deep_learning import ...` with direct module imports.
- Replace `from ..features import ...` with direct-file imports where possible.
- Remove reliance on `CTAFlow/models/__init__.py` side effects.

Exit criteria:

- Recursive import walk shrinks noticeably.
- Missing/stale alias imports are eliminated or explicitly quarantined.

### Phase 3. Establish the minimal keep-set

Produce a keep manifest with three buckets:

- Required now
- Optional but still referenced
- Candidate for quarantine

Recommended first keep-set:

- target model modules
- `CTAFlow/data/dataset_utils.py`
- `CTAFlow/data/datasets/continuous.py`
- `CTAFlow/data/datasets/tft.py`
- `CTAFlow/data/model_datasets.py`
- `CTAFlow/data/raw_formatting/intraday_manager.py`
- `CTAFlow/features/dt_features.py`
- `CTAFlow/features/session_features.py`
- `CTAFlow/features/curve/curve_features.py`
- `CTAFlow/features/volume/profile.py`
- `CTAFlow/features/volume/vpin.py`
- `CTAFlow/features/macro_prep.py`
- `CTAFlow/features/cyclical/seasonality/diurnal_seasonality.py`
- `CTAFlow/utils/session.py`
- `CTAFlow/utils/tenor_interpolation.py`
- `CTAFlow/utils/unit_conversions.py`
- supporting `models` files: `base_models.py`, `feature_selection.py`, `tft_aligned.py`

Exit criteria:

- Keep manifest is explicit enough that deletion becomes a reviewable diff.

### Phase 4. Add extraction smoke tests

Add lightweight tests for:

- `MMTFv3Core` / `StatefulMMTFv3Core` import and single forward pass with dummy tensors
- `PTPLoss` and `HeadAwarePTPScheduler`
- `V3ContinuousExtractor` import if RL remains in scope
- `build_v3_loaders(...)` or equivalent prep-loader assembly
- `IntradayMomentum` import and one minimal constructor path
- `multi_asset` import and one minimal dataset path

Exit criteria:

- Tests fail fast when a supposedly unused dependency is actually still required.

### Phase 5. Quarantine, then delete

Recommended approach:

- Move non-core candidates to a temporary quarantine branch or `_attic` branch first.
- Re-run smoke tests and retained notebooks.
- Only then delete from the main branch.

Likely early review candidates:

- `CTAFlow/models/futures_models/`
- legacy TensorFlow modules if not retained
- obsolete deep-learning architectures not imported after Phase 2
- unrelated notebooks/docs/scripts that do not touch the retained keep-set

Exit criteria:

- No retained workflow imports a quarantined module.

## Suggested commands

Inventory:

- `rg -n "from CTAFlow.models|import CTAFlow.models|from CTAFlow.features|from CTAFlow.data" CTAFlow tests notebooks`
- `python -m py_compile <retained python modules>`

Validation:

- `python -m pytest tests/`
- targeted notebook smoke runs for `mmtfv3_cl_gc_optuna.ipynb`

## Recommendation

Do not start deletion from the folder tree. Start from retained workflow entrypoints, collapse imports to direct references, add smoke tests, then prune by observed keep-set. That sequence will remove genuinely unused files without breaking the stateful MMTFv3/PTP path that is currently improving.
