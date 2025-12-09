from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

SUPPORTED_TASKS = {"regression", "binary_classification", "multiclass"}


_BASE_DEFAULTS: Dict[str, Any] = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}


_TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "regression": {"objective": "regression", "metric": "rmse"},
    "binary_classification": {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
    },
    "multiclass": {"objective": "multiclass", "metric": "multi_logloss"},
}


_GPU_HINTS: Dict[str, Any] = {
    "device": "gpu",
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "max_bin": 255,
}


def _load_json_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.is_file():
        with path.open("r") as handle:
            return json.load(handle)
    return {}


def infer_task_from_target(
    y: Optional[Iterable[Any]],
    preferred: Optional[str] = None,
) -> str:
    preferred_norm = (preferred or "").lower()
    if preferred_norm in SUPPORTED_TASKS:
        return preferred_norm

    if y is None:
        return "regression"

    values = pd.Series(y).dropna()
    if values.empty:
        return "regression"

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "regression"

    if not np.all(np.equal(np.mod(numeric, 1), 0)):
        # Continuous series -> regression
        return "regression"

    unique_vals = np.unique(numeric.astype(int))
    n_unique = len(unique_vals)
    if n_unique <= 2:
        return "binary_classification"
    if n_unique < 4:
        return "multiclass"
    return "regression"


def build_default_params(
    *,
    task: str = "regression",
    use_gpu: bool = False,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_task = task if task in SUPPORTED_TASKS else "regression"

    params = {**_BASE_DEFAULTS, **_TASK_DEFAULTS.get(resolved_task, {})}
    if use_gpu:
        params.update(_GPU_HINTS)

    params.update(_load_json_config(config_path))
    return params
