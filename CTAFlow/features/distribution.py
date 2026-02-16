from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # optional import
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_monotone_quantiles(q: np.ndarray) -> np.ndarray:
    """
    Enforce non-crossing quantiles per row with cumulative max.
    """
    if q.ndim != 2:
        raise ValueError(f"Expected [N,Q] quantile array, got shape={q.shape}")
    return np.maximum.accumulate(q, axis=1)


def prob_up_from_quantiles(
    quantiles: ArrayLike,
    tau_levels: Sequence[float],
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Estimate P(X > threshold) from quantile function Q(tau).
    """
    q = _as_monotone_quantiles(_to_numpy(quantiles).astype(np.float64))
    tau = np.asarray(tau_levels, dtype=np.float64)
    if q.shape[1] != tau.shape[0]:
        raise ValueError(f"Quantile width mismatch: q={q.shape[1]} vs tau={tau.shape[0]}")

    out = np.zeros(q.shape[0], dtype=np.float64)
    for i in range(q.shape[0]):
        row = q[i]
        if threshold <= row[0]:
            out[i] = 1.0
            continue
        if threshold >= row[-1]:
            out[i] = 0.0
            continue
        tau_at_thr = float(np.interp(threshold, row, tau))
        out[i] = 1.0 - tau_at_thr
    return np.clip(out, 0.0, 1.0)


def quantile_moment_features(
    quantiles: ArrayLike,
    tau_levels: Sequence[float],
    eps: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Compute moment-style statistics from predicted quantiles using integration over tau.
    """
    q = _as_monotone_quantiles(_to_numpy(quantiles).astype(np.float64))
    tau = np.asarray(tau_levels, dtype=np.float64)

    if q.shape[1] != tau.shape[0]:
        raise ValueError(f"Quantile width mismatch: q={q.shape[1]} vs tau={tau.shape[0]}")

    mean = np.trapz(q, tau, axis=1)
    ex2 = np.trapz(q ** 2, tau, axis=1)
    var = np.maximum(ex2 - mean ** 2, 0.0)
    std = np.sqrt(var + eps)

    centered = q - mean[:, None]
    m3 = np.trapz(centered ** 3, tau, axis=1)
    m4 = np.trapz(centered ** 4, tau, axis=1)
    skew = m3 / (std ** 3 + eps)
    kurt_excess = m4 / (var ** 2 + eps) - 3.0

    return {
        "mean": mean,
        "var": var,
        "std": std,
        "skew": skew,
        "kurt_excess": kurt_excess,
    }


def quantile_tail_features(
    quantiles: ArrayLike,
    tau_levels: Sequence[float],
    eps: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Tail/risk descriptors from quantile predictions.
    """
    q = _as_monotone_quantiles(_to_numpy(quantiles).astype(np.float64))
    tau = np.asarray(tau_levels, dtype=np.float64)

    idx01 = min(np.searchsorted(tau, 0.01), len(tau) - 1)
    idx05 = min(np.searchsorted(tau, 0.05), len(tau) - 1)
    idx10 = min(np.searchsorted(tau, 0.10), len(tau) - 1)
    idx25 = min(np.searchsorted(tau, 0.25), len(tau) - 1)
    idx50 = min(np.searchsorted(tau, 0.50), len(tau) - 1)
    idx75 = min(np.searchsorted(tau, 0.75), len(tau) - 1)
    idx90 = min(np.searchsorted(tau, 0.90), len(tau) - 1)
    idx95 = min(np.searchsorted(tau, 0.95), len(tau) - 1)
    idx99 = min(np.searchsorted(tau, 0.99), len(tau) - 1)

    q01 = q[:, idx01]
    q05 = q[:, idx05]
    q10 = q[:, idx10]
    q25 = q[:, idx25]
    q50 = q[:, idx50]
    q75 = q[:, idx75]
    q90 = q[:, idx90]
    q95 = q[:, idx95]
    q99 = q[:, idx99]

    iqr = q75 - q25
    left_tail = q50 - q05
    right_tail = q95 - q50
    tail_ratio = right_tail / (np.abs(left_tail) + eps)
    bowley_skew = (q75 + q25 - 2.0 * q50) / (iqr + eps)
    central_spread = q90 - q10
    extreme_spread = q99 - q01

    cvar_left = np.zeros(q.shape[0], dtype=np.float64)
    cvar_right = np.zeros(q.shape[0], dtype=np.float64)
    left_mask = tau <= 0.05
    right_mask = tau >= 0.95
    if np.any(left_mask):
        cvar_left = q[:, left_mask].mean(axis=1)
    if np.any(right_mask):
        cvar_right = q[:, right_mask].mean(axis=1)

    return {
        "q01": q01,
        "q05": q05,
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "iqr": iqr,
        "left_tail": left_tail,
        "right_tail": right_tail,
        "tail_ratio": tail_ratio,
        "bowley_skew": bowley_skew,
        "central_spread": central_spread,
        "extreme_spread": extreme_spread,
        "cvar_left_5": cvar_left,
        "cvar_right_95": cvar_right,
    }


@dataclass
class DistributionFeatureConfig:
    tau_levels: Sequence[float]
    threshold_levels: Tuple[float, ...] = (0.0,)
    prefix: str = "dist_"
    add_temporal_deltas: bool = True
    delta_lags: Tuple[int, ...] = (1, 5)


class DistributionFeatureExtractor:
    """
    Extract tabular features from predicted quantile distributions.

    Primary use-case:
      Feed model-predicted distribution summaries into downstream sequential/RL agents.
    """

    def __init__(self, config: DistributionFeatureConfig):
        self.config = config
        self.tau = np.asarray(config.tau_levels, dtype=np.float64)
        if self.tau.ndim != 1:
            raise ValueError("tau_levels must be 1D")
        if np.any(np.diff(self.tau) <= 0):
            raise ValueError("tau_levels must be strictly increasing")

    def extract_matrix(
        self,
        quantiles: ArrayLike,
    ) -> pd.DataFrame:
        q = _as_monotone_quantiles(_to_numpy(quantiles).astype(np.float64))
        if q.shape[1] != len(self.tau):
            raise ValueError(f"Quantile width mismatch: q={q.shape[1]} vs tau={len(self.tau)}")

        feats: Dict[str, np.ndarray] = {}
        feats.update(quantile_moment_features(q, self.tau))
        feats.update(quantile_tail_features(q, self.tau))

        for thr in self.config.threshold_levels:
            p_up = prob_up_from_quantiles(q, self.tau, threshold=float(thr))
            suffix = str(thr).replace(".", "p").replace("-", "m")
            feats[f"p_up_{suffix}"] = p_up
            feats[f"logit_p_up_{suffix}"] = np.log((p_up + 1e-6) / (1.0 - p_up + 1e-6))

        df = pd.DataFrame(feats)
        return df.add_prefix(self.config.prefix)

    def extract_frame(
        self,
        quantiles: ArrayLike,
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        df = self.extract_matrix(quantiles)
        if index is not None and len(index) == len(df):
            df.index = index

        if self.config.add_temporal_deltas:
            for lag in self.config.delta_lags:
                for col in list(df.columns):
                    df[f"{col}_d{lag}"] = df[col] - df[col].shift(lag)
            df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return df


def extract_distribution_features(
    quantiles: ArrayLike,
    tau_levels: Sequence[float],
    threshold_levels: Sequence[float] = (0.0,),
    prefix: str = "dist_",
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for one-shot feature extraction.
    """
    cfg = DistributionFeatureConfig(
        tau_levels=tuple(float(t) for t in tau_levels),
        threshold_levels=tuple(float(t) for t in threshold_levels),
        prefix=prefix,
    )
    extractor = DistributionFeatureExtractor(cfg)
    return extractor.extract_frame(quantiles=quantiles, index=index)

