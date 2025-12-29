
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Iterable, Any

import numpy as np
import pandas as pd
import torch
import shap

from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression


@dataclass
class FeatureSelectionConfig:
    # Hard filters
    min_non_null_frac: float = 0.98
    variance_eps: float = 1e-12

    # Collinearity
    corr_threshold: float = 0.98

    # CV / stability
    n_splits: int = 5
    importance: Literal["permutation", "model"] = "permutation"
    n_perm_repeats: int = 5
    random_state: int = 42

    # Mutual Information
    use_mutual_info: bool = True
    mi_neighbors: int = 3

    # Weighting for final selection score
    # Default: 50% Stability, 30% Importance Mean, 20% Mutual Information
    stability_weight: float = 0.5
    importance_weight: float = 0.3
    mi_weight: float = 0.2

    # Budget / grouping
    max_features: int = 80
    per_group_max: Optional[int] = None
    group_mode: Literal["base_name", "base_name_time", "none"] = "base_name"


class FeatureXplanation:
    """
    Handles SHAP-based explanations for Deep Learning models (TCN/GRU).
    Identifies feature importance by analyzing model gradients or perturbations.
    """

    def __init__(self, model: torch.nn.Module, background_data: torch.Tensor):
        """
        Args:
            model: The PyTorch model to explain.
            background_data: A representative subset of data (Tensor) used as SHAP reference.
                             Expected shape: (Batch, Channels, Length)
        """
        self.model = model
        self.model.eval()
        self.background_data = background_data
        # Use DeepExplainer for PyTorch models
        self.explainer = shap.DeepExplainer(self.model, self.background_data)

    def compute_importances(self, test_data: torch.Tensor, feature_names: List[str]) -> pd.Series:
        """
        Step-by-step SHAP importance calculation:
        1. Calculate SHAP values for the 3D input (B, C, L).
        2. Take absolute values to measure magnitude of contribution.
        3. Average across the time dimension (L) and batch dimension (B).
        4. Map back to feature names.
        """
        # 1. SHAP values: list of arrays (one per output, here typically [1])
        shap_values = self.explainer.shap_values(test_data)

        # If regression (1 output), shap_values might be a single array or list of 1 array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # 2. Absolute magnitude (B, C, L)
        abs_shap = np.abs(shap_values)

        # 3. Average across time (dim 2) then batch (dim 0)
        # Resulting shape: (C,)
        mean_abs_shap = abs_shap.mean(axis=(0, 2))

        # 4. Create Series
        importance_series = pd.Series(mean_abs_shap, index=feature_names)
        return importance_series.sort_values(ascending=False)


class FeatureSelector:
    """
    Time-series aware feature selector updated with Mutual Information
    and flexible scoring weights.
    """

    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.selected_features_: List[str] = []
        self.feature_report_: Optional[pd.DataFrame] = None
        self.imputer_values_: Dict[str, float] = {}
        self.groups_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, estimator) -> "FeatureSelector":
        X = self._ensure_df(X)
        y = self._ensure_y(y, X.index)

        # A) Hard filters + Impute
        X1, report = self._hard_filter_and_impute(X)

        # B) Correlation pruning
        X2, report = self._prune_correlated(X1, y, report)

        # C) Mutual Information Regression
        if self.config.use_mutual_info:
            mi_scores = self._compute_mutual_info(X2, y)
            report = report.join(mi_scores, how="left")

        # D) Stability importance via TimeSeriesSplits
        imp_df = self._stability_importance(X2, y, estimator)
        report = report.join(imp_df, how="left")

        # E) Budget + Group caps
        selected = self._select_with_budget(report)

        self.selected_features_ = selected
        self.feature_report_ = report.sort_values(["selected", "stability_score"], ascending=[False, False])
        return self

    def _compute_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Computes Mutual Information between features and target to capture
        non-linear dependencies.
        """
        mi = mutual_info_regression(
            X, y,
            n_neighbors=self.config.mi_neighbors,
            random_state=self.config.random_state
        )
        return pd.DataFrame({"mi_score": mi}, index=X.columns)

    def _select_with_budget(self, report: pd.DataFrame) -> List[str]:
        cfg = self.config
        r = report.copy()

        r["selected"] = False
        r["group"] = [self._infer_group(c) for c in r.index]

        # Candidates survived filters
        cand = r[(r["hard_keep"] == True) & (r.get("corr_keep", True) == True)].copy()

        # Fill NaNs for scores
        for col in ["stability_top_frac", "importance_mean", "mi_score"]:
            if col not in cand.columns:
                cand[col] = 0.0
            cand[col] = cand[col].fillna(0.0)

        # Integrated Selection Criterion
        # Combine normalized ranks of stability, model importance, and MI
        s_rank = cand["stability_top_frac"].rank(pct=True)
        i_rank = cand["importance_mean"].rank(pct=True)
        m_rank = cand["mi_score"].rank(pct=True)

        cand["stability_score"] = (
                s_rank * cfg.stability_weight +
                i_rank * cfg.importance_weight +
                m_rank * cfg.mi_weight
        )

        cand = cand.sort_values("stability_score", ascending=False)

        selected = []
        group_counts: Dict[str, int] = {}

        for feat in cand.index:
            if len(selected) >= cfg.max_features:
                break
            g = cand.loc[feat, "group"]
            if cfg.per_group_max is not None:
                if group_counts.get(g, 0) >= cfg.per_group_max:
                    continue
            selected.append(feat)
            group_counts[g] = group_counts.get(g, 0) + 1

        r.loc[selected, "selected"] = True
        # Update the report with the calculated score
        r["stability_score"] = cand["stability_score"].reindex(r.index)
        self.feature_report_ = r
        return selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise ValueError("FeatureSelector is not fitted (selected_features_ is empty).")
        X = self._ensure_df(X).copy()

        # apply stored imputer
        for c, v in self.imputer_values_.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)

        # keep only selected
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise KeyError(f"Missing selected features in transform(): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, estimator) -> pd.DataFrame:
        self.fit(X, y, estimator=estimator)
        return self.transform(X)

    # ---------- internals ----------
    @staticmethod
    def _ensure_df(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame (so we can preserve feature names).")
        return X

    @staticmethod
    def _ensure_y(y, index) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y must be a Series or single-column DataFrame.")
            y = y.iloc[:, 0]
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=index)
        y = y.loc[index]
        return y

    def _hard_filter_and_impute(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        report = pd.DataFrame(index=X.columns)

        non_null_frac = X.notna().mean(axis=0)
        report["non_null_frac"] = non_null_frac
        keep = non_null_frac >= cfg.min_non_null_frac

        # keep numeric-ish only
        Xn = X.loc[:, keep].copy()
        Xn = Xn.apply(pd.to_numeric, errors="coerce")

        # impute medians (store for transform)
        med = Xn.median(axis=0, skipna=True)
        self.imputer_values_ = med.to_dict()
        Xn = Xn.fillna(med)

        var = Xn.var(axis=0)
        report["variance"] = var.reindex(report.index)
        keep2 = var >= cfg.variance_eps
        Xn = Xn.loc[:, keep2].copy()

        report["hard_keep"] = False
        report.loc[Xn.columns, "hard_keep"] = True

        report["dropped_reason"] = None
        report.loc[~keep, "dropped_reason"] = "missingness"
        report.loc[keep & ~keep2.reindex(keep.index).fillna(False), "dropped_reason"] = "low_variance"
        return Xn, report

    def _prune_correlated(self, X: pd.DataFrame, y: pd.Series, report: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        if X.shape[1] <= 1:
            report["corr_keep"] = True
            return X, report

        # simple “who to keep” rule: abs corr(feature, y) is the tiebreaker
        with np.errstate(invalid="ignore"):
            ycorr = X.corrwith(y).abs().fillna(0.0)

        corr = X.corr().abs()
        cols = list(X.columns)
        keep = set(cols)

        # greedy prune
        for i, ci in enumerate(cols):
            if ci not in keep:
                continue
            # look for later columns too correlated with ci
            for cj in cols[i + 1 :]:
                if cj not in keep:
                    continue
                if corr.loc[ci, cj] >= cfg.corr_threshold:
                    # drop the weaker one vs target
                    drop = cj if ycorr[ci] >= ycorr[cj] else ci
                    keep.discard(drop)

        kept_cols = [c for c in cols if c in keep]
        dropped_cols = [c for c in cols if c not in keep]

        report["corr_keep"] = False
        report.loc[kept_cols, "corr_keep"] = True
        for c in dropped_cols:
            if report.loc[c, "dropped_reason"] is None:
                report.loc[c, "dropped_reason"] = f"collinear(>={cfg.corr_threshold})"
        return X[kept_cols].copy(), report

    def _stability_importance(self, X: pd.DataFrame, y: pd.Series, estimator) -> pd.DataFrame:
        cfg = self.config
        tscv = TimeSeriesSplit(n_splits=min(cfg.n_splits, max(2, len(X) // 50)))

        importances = []
        for fold, (tr, va) in enumerate(tscv.split(X)):
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            Xva, yva = X.iloc[va], y.iloc[va]

            est = self._clone_estimator(estimator)
            est.fit(Xtr, ytr)

            if cfg.importance == "permutation":
                perm = permutation_importance(
                    est,
                    Xva,
                    yva,
                    n_repeats=cfg.n_perm_repeats,
                    random_state=cfg.random_state,
                    n_jobs=-1,
                )
                imp = pd.Series(perm.importances_mean, index=X.columns)
            else:
                # model-based (LightGBM/XGB etc.) if it exists
                if hasattr(est, "feature_importances_"):
                    imp = pd.Series(est.feature_importances_, index=X.columns)
                else:
                    # fallback: abs coef
                    coef = getattr(est, "coef_", None)
                    if coef is None:
                        raise AttributeError("Estimator has no feature_importances_ or coef_.")
                    imp = pd.Series(np.abs(np.ravel(coef)), index=X.columns)

            imp.name = f"fold_{fold}"
            importances.append(imp)

        imp_df = pd.concat(importances, axis=1).fillna(0.0)

        mean_imp = imp_df.mean(axis=1)
        # stability: how often a feature lands in the top-K per fold
        K = max(5, int(0.2 * X.shape[1]))
        top_hits = (imp_df.rank(ascending=False, axis=0) <= K).mean(axis=1)

        out = pd.DataFrame(
            {
                "importance_mean": mean_imp,
                "stability_top_frac": top_hits,
            }
        )
        # combine into a single sortable score
        out["stability_score"] = out["importance_mean"].rank(pct=True) * 0.7 + out["stability_top_frac"] * 0.3
        return out

    @staticmethod
    def _clone_estimator(estimator):
        # Works for sklearn estimators; for wrappers, we just reuse the same class if possible.
        try:
            from sklearn.base import clone
            return clone(estimator)
        except Exception:
            # last resort
            import copy
            return copy.deepcopy(estimator)

    def _infer_group(self, feature_name: str) -> str:
        # uses IntradayMomentum naming scheme: [pd_] [ticker_] HHMM [_period] base_name
        name = feature_name[3:] if feature_name.startswith("pd_") else feature_name
        parts = name.split("_")

        # find the first 4-digit token => time
        time_tok = next((p for p in parts if len(p) == 4 and p.isdigit()), None)
        base_name = parts[-1]

        if self.config.group_mode == "base_name_time" and time_tok:
            return f"{base_name}@{time_tok}"
        if self.config.group_mode == "base_name":
            return base_name
        return "__all__"

    def _select_with_budget(self, report: pd.DataFrame) -> List[str]:
        cfg = self.config
        r = report.copy()

        r["selected"] = False
        r["group"] = [self._infer_group(c) for c in r.index]
        self.groups_ = r.groupby("group").apply(lambda d: list(d.index)).to_dict()

        # candidates: only those surviving hard + corr keep
        cand = r[(r["hard_keep"] == True) & (r.get("corr_keep", True) == True)].copy()

        # default importance missing -> 0
        for col in ["stability_score", "importance_mean", "stability_top_frac"]:
            if col not in cand.columns:
                cand[col] = 0.0
            cand[col] = cand[col].fillna(0.0)

        cand = cand.sort_values("stability_score", ascending=False)

        selected = []
        group_counts: Dict[str, int] = {}

        for feat in cand.index:
            if len(selected) >= cfg.max_features:
                break
            g = cand.loc[feat, "group"]
            if cfg.per_group_max is not None:
                if group_counts.get(g, 0) >= cfg.per_group_max:
                    continue
            selected.append(feat)
            group_counts[g] = group_counts.get(g, 0) + 1

        r.loc[selected, "selected"] = True
        self.feature_report_ = r
        return selected
