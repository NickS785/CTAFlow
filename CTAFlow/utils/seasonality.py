"""Seasonality utilities covering intraday, weekly, and monthly effects."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def add_seasonal_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, week-of-month, and month keys."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex")

    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")

    result = df.copy()
    result["hour"] = idx.hour
    result["dow"] = idx.dayofweek
    result["month"] = idx.month
    result["wom"] = ((idx.day - 1) // 7) + 1
    return result


def seasonality_regression(
    returns: pd.Series,
    design: pd.DataFrame,
    hac_lags: int = 30,
) -> Dict[str, object]:
    """Run an OLS regression with Newey-West standard errors."""

    if design.empty or returns.empty:
        return {
            "params": pd.Series(dtype="float64"),
            "tvalues": pd.Series(dtype="float64"),
            "pvalues": pd.Series(dtype="float64"),
            "rsq": np.nan,
            "n": 0,
        }

    aligned = pd.concat([returns, design], axis=1, join="inner").dropna()
    if aligned.empty:
        return {
            "params": pd.Series(dtype="float64"),
            "tvalues": pd.Series(dtype="float64"),
            "pvalues": pd.Series(dtype="float64"),
            "rsq": np.nan,
            "n": 0,
        }

    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    if X.empty:
        return {
            "params": pd.Series(dtype="float64"),
            "tvalues": pd.Series(dtype="float64"),
            "pvalues": pd.Series(dtype="float64"),
            "rsq": np.nan,
            "n": len(y),
        }

    maxlags = min(int(hac_lags), max(len(y) - 1, 0))
    model = sm.OLS(y.values, X.values, hasconst=False)
    try:
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    except ValueError:
        return {
            "params": pd.Series(dtype="float64"),
            "tvalues": pd.Series(dtype="float64"),
            "pvalues": pd.Series(dtype="float64"),
            "rsq": np.nan,
            "n": len(y),
        }

    params = pd.Series(results.params, index=X.columns)
    tvalues = pd.Series(results.tvalues, index=X.columns)
    pvalues = pd.Series(results.pvalues, index=X.columns)
    rsq = getattr(results, "rsquared", np.nan)
    return {"params": params, "tvalues": tvalues, "pvalues": pvalues, "rsq": rsq, "n": len(y)}


def seasonality_summary(
    dfs: Dict[str, pd.DataFrame],
    ret_col: str = "ret",
    hac_lags: int = 30,
    keys: Tuple[str, ...] = ("hour", "dow", "wom", "month"),
) -> pd.DataFrame:
    """Return a tidy seasonality summary for multiple symbols."""

    records = []

    for symbol, df in dfs.items():
        if ret_col not in df.columns or df.empty:
            continue

        enriched = add_seasonal_keys(df)
        design_parts = []
        for key in keys:
            if key not in enriched.columns:
                continue
            categories = pd.Categorical(enriched[key])
            dummies = pd.get_dummies(categories, prefix=key)
            dummies.index = enriched.index
            design_parts.append(dummies)

        if not design_parts:
            continue

        design = pd.concat(design_parts, axis=1)
        res = seasonality_regression(enriched[ret_col], design, hac_lags=hac_lags)

        params = res["params"]
        tvals = res["tvalues"]
        pvals = res["pvalues"]
        rsq = res["rsq"]
        n = res["n"]

        for col in params.index:
            if "_" not in col:
                factor, level = col, ""
            else:
                factor, level = col.split("_", 1)
            records.append(
                {
                    "symbol": symbol,
                    "factor": factor,
                    "level": level,
                    "coef": params[col],
                    "t": tvals.get(col, np.nan),
                    "p": pvals.get(col, np.nan),
                    "rsq": rsq,
                    "n": n,
                }
            )

    return pd.DataFrame.from_records(records, columns=["symbol", "factor", "level", "coef", "t", "p", "rsq", "n"])


__all__ = [
    "add_seasonal_keys",
    "seasonality_regression",
    "seasonality_summary",
]

