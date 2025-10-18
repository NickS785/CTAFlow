"""Event study utilities for synthetic spreads and other features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class EventSpec:
    """Specification describing an event study."""

    name: str
    event_fn: Callable[[pd.DataFrame], pd.Series]
    pre: int = 60
    post: int = 60
    baseline_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None


def build_abret(
    df: pd.DataFrame,
    ret_col: str,
    baseline: Optional[pd.Series],
) -> pd.Series:
    """Compute abnormal returns given an optional baseline series."""

    if ret_col not in df.columns:
        return pd.Series(dtype="float64", index=df.index)

    ret = df[ret_col].astype(float)
    if baseline is None:
        return ret

    baseline = baseline.reindex(df.index)
    return ret.sub(baseline.fillna(0.0))


def event_matrix(
    df: pd.DataFrame,
    events: pd.Series,
    abret: pd.Series,
    pre: int,
    post: int,
) -> pd.DataFrame:
    """Align abnormal returns around each event time."""

    if events.empty:
        return pd.DataFrame()

    valid_events = events.fillna(False)
    event_times = valid_events.index[valid_events.astype(bool)]
    if len(event_times) == 0:
        return pd.DataFrame()

    windows = {}
    total_len = len(abret)
    for ts in event_times:
        try:
            loc = abret.index.get_loc(ts)
        except KeyError:
            continue
        start = loc - pre
        end = loc + post
        if start < 0 or end >= total_len:
            continue
        window = abret.iloc[start : end + 1]
        window.index = range(-pre, post + 1)
        windows[ts] = window

    if not windows:
        return pd.DataFrame()

    return pd.DataFrame(windows)


def aar_car(M: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return AAR and CAR series."""

    if M.empty:
        empty = pd.Series(dtype="float64")
        return {"aar": empty, "car": empty}

    aar = M.mean(axis=1)
    car = aar.cumsum()
    return {"aar": aar, "car": car}


def bootstrap_cis(
    M: pd.DataFrame,
    horizons: Iterable[int] = (0, 5, 30, 60),
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[int, tuple]:
    """Bootstrap confidence intervals for cumulative abnormal returns."""

    if M.empty:
        return {}

    rng = np.random.default_rng(seed)
    horizons = list(horizons)
    car = M.cumsum(axis=0)
    cis: Dict[int, tuple] = {}
    event_count = M.shape[1]

    for h in horizons:
        if h not in car.index:
            continue
        samples = np.empty(n_boot)
        for i in range(n_boot):
            cols = rng.choice(event_count, size=event_count, replace=True)
            sampled = car.iloc[:, cols]
            samples[i] = sampled.loc[h].mean()
        lower, upper = np.percentile(samples, [2.5, 97.5])
        cis[h] = (lower, upper)

    return cis


def aar_hac_tstats(M: pd.DataFrame, maxlags: int = 10) -> pd.Series:
    """Compute HAC t-stats for average abnormal returns at each horizon."""

    if M.empty:
        return pd.Series(dtype="float64")

    tstats = {}
    for tau, values in M.iterrows():
        series = values.dropna()
        n_obs = len(series)
        if n_obs <= 1:
            tstats[tau] = np.nan
            continue
        maxlag = min(maxlags, n_obs - 1)
        X = np.ones((n_obs, 1))
        model = sm.OLS(series.values, X, hasconst=True)
        try:
            results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlag})
        except ValueError:
            tstats[tau] = np.nan
            continue
        tstats[tau] = results.tvalues[0]

    return pd.Series(tstats)


def run_event_study(
    df: pd.DataFrame,
    spec: EventSpec,
    ret_col: str = "ret",
) -> Dict[str, object]:
    """Execute an event study according to ``spec``."""

    baseline = spec.baseline_fn(df) if spec.baseline_fn else None
    abret = build_abret(df, ret_col, baseline)
    events = spec.event_fn(df)
    matrix = event_matrix(df, events, abret, spec.pre, spec.post)

    stats = aar_car(matrix)
    cis = bootstrap_cis(matrix) if not matrix.empty else {}
    tstats = aar_hac_tstats(matrix)

    return {
        "matrix": matrix,
        "aar": stats["aar"],
        "car": stats["car"],
        "cis": cis,
        "tstats": tstats,
        "n_events": matrix.shape[1] if not matrix.empty else 0,
    }


__all__ = [
    "EventSpec",
    "build_abret",
    "event_matrix",
    "aar_car",
    "bootstrap_cis",
    "aar_hac_tstats",
    "run_event_study",
]

