"""Special mathematical functions used by CTAFlow utilities."""
from __future__ import annotations

import math

_FPMIN = 1e-300
_MAX_ITER = 200
_EPS = 3e-7


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction evaluation for the incomplete beta function."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``.

    Parameters
    ----------
    a, b:
        Shape parameters, both strictly positive.
    x:
        Evaluation point in ``[0, 1]``.

    Returns
    -------
    float
        Value of the regularized incomplete beta function.
    """

    if not (0.0 <= x <= 1.0):
        raise ValueError("x must lie in [0, 1]")
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


__all__ = ["regularized_beta"]
