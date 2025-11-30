"""False discovery rate adjustment utilities.

This module implements the Benjamini–Hochberg (BH) and Benjamini–Yekutieli
(BY) procedures for controlling the false discovery rate (FDR).  The functions
are implemented in pure NumPy and support NaN aware computations while
preserving the original ordering of the input p-values.
"""
from __future__ import annotations

from typing import Iterable, Literal, NamedTuple

import numpy as np


class FDRResult(NamedTuple):
    """Container for the output of an FDR adjustment.

    Attributes
    ----------
    q_values:
        Adjusted p-values ("q-values") with the same shape as the provided
        ``p_values``.
    rejected:
        Boolean array indicating which hypotheses are rejected at the supplied
        ``alpha`` level after the FDR correction.
    alpha:
        The significance level supplied to :func:`fdr_bh`.
    method:
        Either ``"BH"`` for Benjamini–Hochberg or ``"BY"`` for
        Benjamini–Yekutieli.
    m:
        Number of valid (non-NaN and finite) hypotheses that were evaluated.
    """

    q_values: np.ndarray
    rejected: np.ndarray
    alpha: float
    method: str
    m: int


def _as_numeric_array(values: Iterable[float]) -> np.ndarray:
    """Convert ``values`` to a ``float64`` NumPy array.

    ``np.asarray`` is used to avoid copying whenever the input is already a
    NumPy array.  A one dimensional view is returned irrespective of the shape
    of ``values`` so that ``fdr_bh`` works with any iterable input while still
    being able to broadcast the output back to the original shape.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _harmonic_number(n: int) -> float:
    """Return the nth harmonic number ``H_n``."""

    if n < 1:
        return 0.0
    # Vectorised computation is faster and more stable than iterative loops for
    # large ``n``.
    return np.sum(1.0 / np.arange(1, n + 1, dtype=float))


def fdr_bh(
    p_values: Iterable[float],
    alpha: float = 0.05,
    method: Literal["BH", "BY"] = "BH",
) -> FDRResult:
    """Benjamini–Hochberg/Yekutieli false discovery rate adjustment.

    Parameters
    ----------
    p_values:
        Iterable of p-values.  ``NaN`` and ``inf`` values are ignored for
        ranking but preserved (as ``NaN``) in the returned q-values and
        rejection mask.
    alpha:
        Significance level in the interval ``(0, 1]``.  Defaults to ``0.05``.
    method:
        Either ``"BH"`` (default) for the Benjamini–Hochberg procedure or
        ``"BY"`` for the Benjamini–Yekutieli procedure which is robust under
        arbitrary dependence via the harmonic number correction.

    Returns
    -------
    FDRResult
        Named tuple containing q-values, rejection mask, and metadata.

    Raises
    ------
    ValueError
        If ``alpha`` lies outside ``(0, 1]`` or an unknown ``method`` is
        supplied.

    Examples
    --------
    >>> from stats_utils.fdr import fdr_bh
    >>> res = fdr_bh([0.01, 0.04, 0.03, 0.002, 0.5])
    >>> res.q_values.round(3)
    array([0.025, 0.05 , 0.05 , 0.01 , 0.5  ])
    >>> res.rejected
    array([ True,  True,  True,  True, False])
    """

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in the interval (0, 1]")

    method = method.upper()
    if method not in {"BH", "BY"}:
        raise ValueError("method must be either 'BH' or 'BY'")

    arr = _as_numeric_array(p_values)
    flat = arr.reshape(-1)
    q_values = np.full_like(flat, np.nan, dtype=float)
    rejected = np.zeros_like(flat, dtype=bool)

    valid_mask = np.isfinite(flat)
    valid_values = flat[valid_mask]
    m = valid_values.size

    if m == 0:
        q_values = q_values.reshape(arr.shape)
        rejected = rejected.reshape(arr.shape)
        return FDRResult(q_values, rejected, alpha, method, m)

    # Sort the valid p-values using a stable algorithm to preserve the ranking
    # of tied values and to keep the indices deterministic for NaN handling.
    order = np.argsort(valid_values, kind="mergesort")
    ranked = valid_values[order]

    ranks = np.arange(1, m + 1, dtype=float)
    harmonic = _harmonic_number(m) if method == "BY" else 1.0

    adjusted = ranked * (float(m) / ranks)
    if method == "BY":
        adjusted *= harmonic

    # Enforce monotonicity from the top (largest rank) downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    q_valid = adjusted[inv_order]

    q_values[valid_mask] = q_valid

    threshold_scale = float(m) * harmonic

    thresholds = ranks * (alpha / threshold_scale)
    significant = ranked <= thresholds
    if np.any(significant):
        max_idx = np.max(np.nonzero(significant))
        cutoff = ranked[max_idx]
        rejected[valid_mask] = flat[valid_mask] <= cutoff

    q_values = q_values.reshape(arr.shape)
    rejected = rejected.reshape(arr.shape)
    return FDRResult(q_values, rejected, alpha, method, m)


__all__ = ["FDRResult", "fdr_bh"]
