"""Base model definitions for CTAFlow machine learning models."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from CTAFlow.models.config_lightgbm_cpu import DEFAULT_LGBM_CPU_CONFIG
from CTAFlow.models.config_lightgbm_gpu import DEFAULT_LGBM_GPU_CONFIG
from CTAFlow.strategy.gpu_acceleration import GPU_AVAILABLE


@dataclass
class TrainingResult:
    """Container for training artifacts."""

    evaluation_results: Dict[str, Any]
    best_iteration: Optional[int]
    feature_importance: Optional[pd.Series]


class CTALight:
    """LightGBM model with GPU-aware configuration helpers."""

    def __init__(self, *, use_gpu: bool = False, config: Optional[Dict[str, Any]] = None, **lgb_params):
        base_config = DEFAULT_LGBM_GPU_CONFIG if use_gpu and GPU_AVAILABLE else DEFAULT_LGBM_CPU_CONFIG
        self.params: Dict[str, Any] = {**base_config}
        if config:
            self.params.update(config)
        self.params.update(lgb_params)

        self.use_gpu = bool(use_gpu and GPU_AVAILABLE)
        if use_gpu and not GPU_AVAILABLE:
            # Fall back to CPU settings when GPU is requested but unavailable
            self.params.update(DEFAULT_LGBM_CPU_CONFIG)
            self.params['device'] = 'cpu'

        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[Iterable[str]] = None
        self.feature_importance: Optional[pd.Series] = None
        self.train_history: Optional[Dict[str, Any]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        num_boost_round: int = 1000,
    ) -> TrainingResult:
        """Fit the LightGBM model with optional validation set."""

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_values = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_values = np.asarray(X)

        y_values = y.values if isinstance(y, pd.Series) else np.asarray(y)

        mask = ~(np.isnan(X_values).any(axis=1) | np.isnan(y_values))
        X_clean = X_values[mask]
        y_clean = y_values[mask]

        train_data = lgb.Dataset(X_clean, label=y_clean, feature_name=self.feature_names)

        valid_sets = [train_data]
        valid_names = ['train']
        eval_result: Dict[str, Any] = {}
        callbacks = []

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_values = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
            y_val_values = y_val.values if isinstance(y_val, pd.Series) else np.asarray(y_val)

            val_mask = ~(np.isnan(X_val_values).any(axis=1) | np.isnan(y_val_values))
            X_val_clean = X_val_values[val_mask]
            y_val_clean = y_val_values[val_mask]

            if len(X_val_clean) > 0:
                valid_data = lgb.Dataset(
                    X_val_clean,
                    label=y_val_clean,
                    reference=train_data,
                    feature_name=self.feature_names,
                )
                valid_sets.append(valid_data)
                valid_names.append('valid')
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
                callbacks.append(lgb.record_evaluation(eval_result))

        self.model = lgb.train(
            self.params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        if self.model is not None:
            importance = self.model.feature_importance(importance_type='gain')
            self.feature_importance = pd.Series(importance, index=self.feature_names)

        self.train_history = eval_result

        return TrainingResult(
            evaluation_results=eval_result,
            best_iteration=self.model.best_iteration if self.model is not None else None,
            feature_importance=self.feature_importance,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before predicting.")

        X_values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_values, num_iteration=self.model.best_iteration)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        predictions = self.predict(X)
        y_values = y_true.values if isinstance(y_true, pd.Series) else np.asarray(y_true)

        mask = ~(np.isnan(predictions) | np.isnan(y_values))
        pred_clean = predictions[mask]
        y_clean = y_values[mask]

        mse = np.mean((pred_clean - y_clean) ** 2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(pred_clean - y_clean)))
        directional_accuracy = float(np.mean(np.sign(pred_clean) == np.sign(y_clean))) if len(y_clean) else 0.0

        return {
            'mse': float(mse),
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the resolved LightGBM parameter set."""

        resolved = dict(self.params)
        resolved['device'] = 'gpu' if self.use_gpu else resolved.get('device', 'cpu')
        return resolved
