"""
Calendar effects testing for futures returns.

Analyzes returns around calendar boundaries (month-end, quarter-end, week-of-month)
and tests for lead-lag predictability based on previous periods.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CalendarEffectParams:
    """
    Configuration for calendar-effects tests.

    Attributes:
        price_col: Column name for price data (default: 'close')
        horizons: Dictionary mapping horizon labels to trading days
                 (default: {'1d': 1, '3d': 3, '5d': 5})
        min_obs: Minimum observations required for reliable statistics
                Below this threshold, patterns are flagged as exploratory
        week_len: Number of trading days per 'week of month' bucket (default: 5)
    """
    price_col: str = "close"
    horizons: Dict[str, int] = None
    min_obs: int = 50
    week_len: int = 5

    def __post_init__(self):
        if self.horizons is None:
            # 1, 3, 5 trading-day forward returns
            self.horizons = {"1d": 1, "3d": 3, "5d": 5}


# ---------- Statistical Helpers ----------

def _one_sample_ttest_against_zero(x: pd.Series) -> Dict[str, float]:
    """
    Simple t-test of mean(x) vs 0. Returns dict with mean, t, p, n.
    NaNs are dropped.
    """
    arr = pd.to_numeric(x, errors="coerce").dropna().values.astype(float)
    n = arr.size
    if n < 2:
        return {"mean": np.nan, "t_stat": np.nan, "p_value": np.nan, "n_obs": n}

    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0.0:
        return {"mean": mean, "t_stat": np.nan, "p_value": np.nan, "n_obs": n}

    se = std / math.sqrt(n)
    t_stat = mean / se
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    return {"mean": mean, "t_stat": t_stat, "p_value": p_value, "n_obs": n}


def _bh_fdr_q_values(p: pd.Series) -> pd.Series:
    """
    Benjamini–Hochberg FDR adjustment for multiple testing.
    """
    p = pd.to_numeric(p, errors="coerce")
    mask = p.notna()
    if not mask.any():
        return pd.Series(np.nan, index=p.index)

    p_valid = p[mask]
    m = float(p_valid.size)
    order = p_valid.sort_values().index
    ranks = np.arange(1, p_valid.size + 1, dtype=float)

    p_sorted = p_valid.loc[order].values
    q_sorted = p_sorted * m / ranks

    # Enforce monotone decreasing q-values when going from largest to smallest p
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]

    q = pd.Series(np.nan, index=p.index)
    q.loc[order] = q_sorted
    return q


def _univariate_regression(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """
    OLS regression y = a + b x + eps; returns slope b, t_stat, p_value, n, r2.
    """
    df = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    n = df.shape[0]
    if n < 3:
        return {
            "slope": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_obs": n,
            "r2": np.nan,
        }

    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)

    x_mean = x.mean()
    y_mean = y.mean()
    x_c = x - x_mean
    y_c = y - y_mean

    sxx = float(np.sum(x_c * x_c))
    if sxx <= 0.0:
        return {
            "slope": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_obs": n,
            "r2": np.nan,
        }

    sxy = float(np.sum(x_c * y_c))
    beta = sxy / sxx

    # fitted values with intercept
    y_hat = beta * x_c + y_mean
    resid = y - y_hat
    sse = float(np.sum(resid * resid))
    sst = float(np.sum((y - y_mean) ** 2))

    if n - 2 <= 0 or sxx == 0.0:
        return {
            "slope": beta,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_obs": n,
            "r2": 1.0 - sse / sst if sst > 0 else np.nan,
        }

    sigma2 = sse / (n - 2)
    se_beta = math.sqrt(sigma2 / sxx)
    if se_beta == 0.0:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = beta / se_beta
        p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 2)

    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return {
        "slope": beta,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_obs": n,
        "r2": r2,
    }


# ---------- Calendar Feature Engineering ----------

def _attach_calendar_flags(df: pd.DataFrame, params: CalendarEffectParams) -> pd.DataFrame:
    """
    Add month / quarter / week-of-month position flags for calendar tests.
    Index is expected to be a DatetimeIndex (or we will try to use 'date' column).
    """
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" not in out.columns:
            raise ValueError("DataFrame must have DatetimeIndex or a 'date' column.")
        out = out.set_index("date")

    out = out.sort_index()
    dates = out.index

    # Remove timezone info before converting to period to avoid warnings
    dates_tz_naive = dates.tz_localize(None) if hasattr(dates, 'tz') and dates.tz is not None else dates

    month_key = dates_tz_naive.to_period("M")
    quarter_key = dates_tz_naive.to_period("Q")

    out["month_key"] = month_key
    out["quarter_key"] = quarter_key

    # Position within month / quarter (0-based)
    out["month_pos"] = out.groupby("month_key").cumcount()
    out["month_cnt"] = out.groupby("month_key")["month_key"].transform("size")
    out["month_pos_rev"] = out["month_cnt"] - 1 - out["month_pos"]

    out["quarter_pos"] = out.groupby("quarter_key").cumcount()
    out["quarter_cnt"] = out.groupby("quarter_key")["quarter_key"].transform("size")
    out["quarter_pos_rev"] = out["quarter_cnt"] - 1 - out["quarter_pos"]

    # First / last 1, 3, 5 trading days of month/quarter
    for k in (1, 3, 5):
        out[f"month_start_{k}d"] = out["month_pos"] < k
        out[f"month_end_{k}d"] = out["month_pos_rev"] < k

        out[f"quarter_start_{k}d"] = out["quarter_pos"] < k
        out[f"quarter_end_{k}d"] = out["quarter_pos_rev"] < k

    # Week-of-month (trading weeks, length = params.week_len)
    week_len = int(params.week_len)
    out["month_week_index"] = (out["month_pos"] // week_len).astype(int)
    out["month_week_max"] = out.groupby("month_key")["month_week_index"].transform("max")
    out["month_first_week"] = out["month_week_index"] == 0
    out["month_last_week"] = out["month_week_index"] == out["month_week_max"]

    return out


def _attach_forward_returns(df: pd.DataFrame, params: CalendarEffectParams) -> pd.DataFrame:
    """
    Compute log forward returns over the configured horizons, using price_col.
    """
    out = df.copy()
    price = pd.to_numeric(out[params.price_col], errors="coerce")

    for label, steps in params.horizons.items():
        # log forward return over <steps> trading days from t to t+steps
        col = f"ret_fwd_{label}"
        out[col] = np.log(price.shift(-steps) / price)

    return out


# ---------- Calendar Edge Tests ----------

def run_calendar_edge_tests(
    df: pd.DataFrame,
    params: CalendarEffectParams,
    group_label: str = "calendar_edge",
) -> pd.DataFrame:
    """
    For each symbol's daily df:
      * Compute 1/3/5-day forward returns.
      * Test whether mean forward returns are significantly non-zero
        within:
          - first/last 1, 3, 5 trading days of the month
          - first/last 1, 3, 5 trading days of the quarter
          - first / last week of the month (5-day trading week)

    Returns a tidy DataFrame with one row per (pattern, horizon).

    Args:
        df: Daily OHLC data with DatetimeIndex or 'date' column
        params: CalendarEffectParams configuration
        group_label: Pattern group label for categorization

    Returns:
        DataFrame with columns:
            - pattern: Pattern identifier (e.g., 'calendar_month_start_1d')
            - event: Event window name
            - horizon: Forward return horizon ('1d', '3d', '5d')
            - group: Pattern group label
            - mean: Average forward return in window
            - t_stat: T-statistic
            - p_value: P-value for mean != 0
            - q_value: BH-FDR adjusted q-value
            - n_obs: Number of observations
            - sig_fdr_5pct: Boolean, significant at 5% FDR
            - exploratory: Boolean, flagged if n_obs < min_obs
    """
    df_flags = _attach_calendar_flags(df, params)
    df_ret = _attach_forward_returns(df_flags, params)

    event_cols: List[str] = []

    # month / quarter edge flags
    for k in (1, 3, 5):
        event_cols.extend(
            [
                f"month_start_{k}d",
                f"month_end_{k}d",
                f"quarter_start_{k}d",
                f"quarter_end_{k}d",
            ]
        )

    # first/last week-of-month flags (defined via trading weeks)
    event_cols.extend(["month_first_week", "month_last_week"])

    records: List[Dict[str, object]] = []

    for event_col in event_cols:
        mask = df_ret[event_col].astype(bool)
        if not mask.any():
            continue

        for label, _steps in params.horizons.items():
            ret_col = f"ret_fwd_{label}"
            if ret_col not in df_ret.columns:
                continue

            stats_dict = _one_sample_ttest_against_zero(df_ret.loc[mask, ret_col])

            rec = {
                "pattern": f"calendar_{event_col}",
                "event": event_col,
                "horizon": label,
                "group": group_label,
                "mean": stats_dict["mean"],
                "t_stat": stats_dict["t_stat"],
                "p_value": stats_dict["p_value"],
                "n_obs": stats_dict["n_obs"],
            }
            records.append(rec)

    if not records:
        return pd.DataFrame(columns=[
            "pattern", "event", "horizon", "group",
            "mean", "t_stat", "p_value", "q_value",
            "n_obs", "sig_fdr_5pct", "exploratory",
        ])

    out = pd.DataFrame.from_records(records)
    out["q_value"] = _bh_fdr_q_values(out["p_value"])
    out["sig_fdr_5pct"] = out["q_value"] <= 0.05
    out["exploratory"] = out["n_obs"] < params.min_obs

    return out.sort_values(["pattern", "horizon"]).reset_index(drop=True)


# ---------- Lead/Lag Tests ----------

def _aggregate_monthly_week_returns(
    df: pd.DataFrame,
    params: CalendarEffectParams,
) -> pd.DataFrame:
    """
    Build a monthly panel with:
      * first-week return (sum of 1d log returns)
      * last-week return
      * full-month return
    Using 1d log returns implied by price_col.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            raise ValueError("DataFrame must have DatetimeIndex or a 'date' column.")
        df = df.set_index("date")

    df = df.sort_index()
    df_flags = _attach_calendar_flags(df, params)

    # 1-day log returns
    price = pd.to_numeric(df_flags[params.price_col], errors="coerce")
    ret_1d = np.log(price / price.shift(1))
    df_flags["ret_1d"] = ret_1d

    # Note: first return for each month will be NaN; summations will drop it automatically
    grp = df_flags.groupby("month_key")

    month_first_week_ret = grp.apply(
        lambda g: pd.to_numeric(
            g.loc[g["month_first_week"], "ret_1d"], errors="coerce"
        ).dropna().sum()
    )
    month_last_week_ret = grp.apply(
        lambda g: pd.to_numeric(
            g.loc[g["month_last_week"], "ret_1d"], errors="coerce"
        ).dropna().sum()
    )
    month_full_ret = grp["ret_1d"].apply(lambda x: pd.to_numeric(x, errors="coerce").dropna().sum())
    month_start_date = grp.apply(lambda g: g.index.min())

    df_month = pd.DataFrame(
        {
            "month_start_date": month_start_date,
            "month_first_week_ret": month_first_week_ret,
            "month_last_week_ret": month_last_week_ret,
            "month_full_ret": month_full_ret,
        }
    ).sort_values("month_start_date")

    return df_month


def run_calendar_lead_lag_tests(
    df: pd.DataFrame,
    params: CalendarEffectParams,
    group_label: str = "calendar_lead_lag",
) -> pd.DataFrame:
    """
    Lead/lag tests:
      * previous month's first / last week returns -> this month's first week return
      * previous year's same month first / last week returns -> this month's first week return

    Returns tidy DataFrame with slope, t, p, q, n, r2 for each regression.

    Args:
        df: Daily OHLC data with DatetimeIndex or 'date' column
        params: CalendarEffectParams configuration
        group_label: Pattern group label for categorization

    Returns:
        DataFrame with columns:
            - pattern: Pattern identifier
            - predictor: Predictor variable name
            - response: Response variable name
            - group: Pattern group label
            - slope: Regression slope coefficient
            - t_stat: T-statistic for slope
            - p_value: P-value for slope != 0
            - q_value: BH-FDR adjusted q-value
            - n_obs: Number of observations
            - r2: R-squared
            - sig_fdr_5pct: Boolean, significant at 5% FDR
            - exploratory: Boolean, flagged if n_obs < min_obs
    """
    df_month = _aggregate_monthly_week_returns(df, params)

    # Previous month (lag 1 month)
    df_month["prev_month_first_week_ret"] = df_month["month_first_week_ret"].shift(1)
    df_month["prev_month_last_week_ret"] = df_month["month_last_week_ret"].shift(1)

    # Previous year same month (lag 12 months)
    df_month["prev_year_first_week_ret"] = df_month["month_first_week_ret"].shift(12)
    df_month["prev_year_last_week_ret"] = df_month["month_last_week_ret"].shift(12)

    target = df_month["month_first_week_ret"]

    tests: List[Dict[str, object]] = []

    combos = [
        ("prev_month_first_week_ret", "prev_month_first_week -> this_first_week"),
        ("prev_month_last_week_ret", "prev_month_last_week -> this_first_week"),
        ("prev_year_first_week_ret", "prev_year_first_week -> this_first_week"),
        ("prev_year_last_week_ret", "prev_year_last_week -> this_first_week"),
    ]

    for pred_col, label in combos:
        stats_dict = _univariate_regression(df_month[pred_col], target)
        tests.append(
            {
                "pattern": f"calendar_lead_lag_{label}",
                "predictor": pred_col,
                "response": "month_first_week_ret",
                "group": group_label,
                "slope": stats_dict["slope"],
                "t_stat": stats_dict["t_stat"],
                "p_value": stats_dict["p_value"],
                "n_obs": stats_dict["n_obs"],
                "r2": stats_dict["r2"],
            }
        )

    if not tests:
        return pd.DataFrame(
            columns=[
                "pattern",
                "predictor",
                "response",
                "group",
                "slope",
                "t_stat",
                "p_value",
                "q_value",
                "n_obs",
                "r2",
                "sig_fdr_5pct",
                "exploratory",
            ]
        )

    out = pd.DataFrame.from_records(tests)
    out["q_value"] = _bh_fdr_q_values(out["p_value"])
    out["sig_fdr_5pct"] = out["q_value"] <= 0.05
    out["exploratory"] = out["n_obs"] < params.min_obs

    return out.sort_values("pattern").reset_index(drop=True)
