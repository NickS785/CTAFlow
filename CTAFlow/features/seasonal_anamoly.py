"""
SeasonalScanner & Intraday Autocorrelation for Commodity Futures

Author: Quant Commodities Engineer (educational use only)

Overview
--------
Utilities to:
1) Locate intraday autocorrelation between two points/measurements in time.
2) Scan for monthly seasonality/cyclicity and abnormal returns.
3) Test whether last year's same-month return predicts this year's.
4) Extend tests to a pre-window 60–90 days before the target month.

Assumptions
-----------
- Input is a CSV with at least: `timestamp, Open, High, Low, Close, Volume`.
- `timestamp` is in UTC or a consistent timezone (set --tz if you want to convert).
- Data is a single continuous contract or a back-adjusted series. If you use
  individual expiries, concatenate externally or extend this script to roll logic.

Notes
-----
- This script is model-agnostic and uses simple linear statistics (linregress, t-stats).
- For intraday work, use 1–60s bars if possible; for months, log-returns are aggregated.
- Not financial advice. Validate on your own data and execution costs.
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from ..utils.vol_weighted_returns import log_returns
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------
# Data loading & preprocessing
# ----------------------------


# ----------------------------
# Intraday returns & autocorr
# ----------------------------



@dataclass
class IntradayPairSpec:
    """Defines two points or windows within a day to measure correlation.

    Examples:
      - time_a/time_b: Use a single-bar return at those times (per day).
      - If window_* > 1, aggregate consecutive bars starting at the time.
    """
    time_a: str  # e.g., "09:30", "13:45" (HH:MM in local tz of index)
    time_b: str  # e.g., "10:00" or next period's start
    window_a: int = 1  # number of bars to aggregate for A
    window_b: int = 1  # number of bars to aggregate for B


def tod_mask(idx: pd.DatetimeIndex, hhmm: str) -> pd.Series:
    hh, mm = map(int, hhmm.split(":"))
    return (idx.hour == hh) & (idx.minute == mm)


def aggregate_window(ret: pd.Series, start_mask: pd.Series, window: int) -> pd.Series:
    """Aggregate window returns (log-additive) starting at masked bars.
    Returns a Series indexed by the start timestamp.
    """
    starts = ret.index[start_mask]
    out_idx = []
    out_vals = []
    for t in starts:
        # Slice next `window` bars including t
        sl = ret.loc[t:]
        vals = sl.iloc[:window]
        if len(vals) == window and not np.isnan(vals).any():
            out_idx.append(t)
            out_vals.append(np.nansum(vals))
    return pd.Series(out_vals, index=out_idx)


def intraday_autocorr_between_times(df: pd.DataFrame, price_col: str, spec: IntradayPairSpec) -> Dict[str, float]:
    """Compute correlation between per-day window returns at two times-of-day.

    Steps:
      1) Compute log returns for all bars.
      2) Build daily series for A and B windows using time-of-day anchors.
      3) Inner-join on dates; report Pearson r and t-stat.
    """
    px = df[price_col]
    r = log_returns(px).dropna()

    mA = tod_mask(r.index, spec.time_a)
    mB = tod_mask(r.index, spec.time_b)

    RA = aggregate_window(r, mA, spec.window_a)
    RB = aggregate_window(r, mB, spec.window_b)

    # Map to dates to ensure same session comparison
    A = RA.groupby(RA.index.date).sum()
    B = RB.groupby(RB.index.date).sum()
    J = pd.concat([A.rename("A"), B.rename("B")], axis=1).dropna()

    if len(J) < 10:
        return {"n": float(len(J)), "r": np.nan, "t_stat": np.nan, "p_value": np.nan}

    r_val, p_val = stats.pearsonr(J["A"], J["B"])
    # t-stat for correlation (large-sample approximation)
    n = len(J)
    t_stat = r_val * math.sqrt((n - 2) / (1 - r_val**2)) if abs(r_val) < 1 else np.inf
    return {"n": float(n), "r": float(r_val), "t_stat": float(t_stat), "p_value": float(p_val)}


def intraday_lag_autocorr(df: pd.DataFrame, price_col: str, k: int = 1) -> Dict[str, float]:
    """Classic lag-k autocorrelation of intraday log-returns.
    Use k>0 for forward correlation of r_t with r_{t+k}.
    """
    r = log_returns(df[price_col]).dropna()
    X = r.iloc[:-k]
    Y = r.iloc[k:]
    if len(Y) < 100:
        return {"n": float(len(Y)), "r": np.nan, "t_stat": np.nan, "p_value": np.nan}
    r_val, p_val = stats.pearsonr(X.values, Y.values)
    n = len(Y)
    t_stat = r_val * math.sqrt((n - 2) / (1 - r_val**2)) if abs(r_val) < 1 else np.inf
    return {"n": float(n), "r": float(r_val), "t_stat": float(t_stat), "p_value": float(p_val)}

# ----------------------------
# Monthly seasonality scanner
# ----------------------------

@dataclass
class SeasonalSettings:
    price_col: str = "Close"
    use_log_returns: bool = True
    month_agg: str = "M"  # pandas offset alias: 'M' month-end, 'MS' month-start


def monthly_returns(df: pd.DataFrame, price_col: str, use_log_returns: bool = True, month_agg: str = "M") -> pd.Series:
    px = df[price_col].asfreq(None)  # keep original; resample at monthly boundaries
    monthly_px = px.resample(month_agg).last().dropna()
    if use_log_returns:
        mr = np.log(monthly_px).diff()
    else:
        mr = monthly_px.pct_change()
    return mr.dropna().rename("mret")


def abnormal_months(mr: pd.Series) -> pd.DataFrame:
    """Compute per-month mean return, t-stat vs overall mean, and z-score.
    Returns a DataFrame indexed by month (1..12).
    """
    df = mr.to_frame()
    df["month"] = df.index.month
    overall_mu = df["mret"].mean()
    overall_sd = df["mret"].std(ddof=1)

    rows = []
    for m, g in df.groupby("month"):
        mu = g["mret"].mean()
        sd = g["mret"].std(ddof=1)
        n = len(g)
        # One-sample t-test against overall mean
        if n > 1 and not np.isnan(sd) and sd > 0:
            t_stat = (mu - overall_mu) / (sd / math.sqrt(n))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        else:
            t_stat, p_val = np.nan, np.nan
        z = (mu - overall_mu) / overall_sd if overall_sd and not np.isnan(overall_sd) and overall_sd > 0 else np.nan
        rows.append({"month": m, "n": n, "mean": mu, "std": sd, "t_stat": t_stat, "p_value": p_val, "z_score": z})

    out = pd.DataFrame(rows).set_index("month").sort_index()
    return out


def last_year_predicts_this_year(mr: pd.Series) -> pd.DataFrame:
    """For each calendar month, regress current year's month return on last year's same-month return.
    Returns slope, r, p-value per month.
    """
    df = mr.to_frame()
    df["month"] = df.index.month
    df["year"] = df.index.year

    results = []
    for m, g in df.groupby("month"):
        g = g.sort_index()
        # Align by year: y_t on y_{t-1} for same month
        y = g["mret"].iloc[1:].values
        x = g["mret"].shift(1).dropna().values
        n = min(len(x), len(y))
        if n < 8:
            results.append({"month": m, "n": n, "slope": np.nan, "r": np.nan, "p_value": np.nan})
            continue
        slope, intercept, r_val, p_val, stderr = stats.linregress(x[:n], y[:n])
        results.append({"month": m, "n": n, "slope": slope, "r": r_val, "p_value": p_val})

    return pd.DataFrame(results).set_index("month").sort_index()


def prewindow_feature(df: pd.DataFrame, month_agg: str = "M", start_days_before: int = 90, end_days_before: int = 60, price_col: str = "Close", use_log_returns: bool = True) -> pd.DataFrame:
    """Compute a feature as the cumulative return in the pre-window [T-90, T-60] days before each month's start (or end depending on month_agg).

    Returns a DataFrame indexed by month-end (or start) with columns:
      - mret: the month return
      - prewin: cumulative log return in the 60–90 day window before the month boundary
    """
    px = df[price_col]
    mr = monthly_returns(df, price_col=price_col, use_log_returns=use_log_returns, month_agg=month_agg)

    boundary = mr.index  # monthly timestamps
    r_daily = np.log(px).diff().dropna() if use_log_returns else px.pct_change().dropna()
    daily = r_daily.asfreq("D").fillna(0.0)  # approximate daily calendar agg

    prevals = []
    for t in boundary:
        start = (t - pd.Timedelta(days=start_days_before)).normalize()
        end = (t - pd.Timedelta(days=end_days_before)).normalize()
        window = daily.loc[start:end]
        if len(window) == 0:
            prevals.append(np.nan)
        else:
            prevals.append(window.sum())  # log returns add

    feat = pd.DataFrame({"mret": mr.values, "prewin": prevals}, index=boundary)
    return feat.dropna()


def prewindow_predicts_month(df: pd.DataFrame, price_col: str = "Close", use_log_returns: bool = True, month_agg: str = "M") -> Dict[str, float]:
    feat = prewindow_feature(df, month_agg=month_agg, price_col=price_col, use_log_returns=use_log_returns)
    y = feat["mret"].values
    x = feat["prewin"].values
    if len(x) < 24:
        return {"n": float(len(x)), "slope": np.nan, "r": np.nan, "p_value": np.nan}
    slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
    return {"n": float(len(x)), "slope": float(slope), "r": float(r_val), "p_value": float(p_val)}

# ----------------------------
# Reporting helpers
# ----------------------------

def fmt_float(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "nan"
    return f"{x:.4f}"


def print_intraday_report(spec: IntradayPairSpec, basic: Dict[str, float], lag1: Dict[str, float]):
    print("\nIntraday Autocorrelation Report")
    print("-------------------------------")
    print(f"Windows: A={spec.time_a} (x{spec.window_a}) vs B={spec.time_b} (x{spec.window_b})")
    print(f"n={int(basic['n'])}, r={fmt_float(basic['r'])}, t={fmt_float(basic['t_stat'])}, p={fmt_float(basic['p_value'])}")
    print(f"Lag-1 autocorr (bar): n={int(lag1['n'])}, r={fmt_float(lag1['r'])}, t={fmt_float(lag1['t_stat'])}, p={fmt_float(lag1['p_value'])}")


def print_seasonality_report(mr: pd.Series):
    print("\nMonthly Seasonality Report")
    print("--------------------------")
    ab = abnormal_months(mr)
    print(ab.to_string(float_format=lambda x: f"{x: .4f}"))

    ly = last_year_predicts_this_year(mr)
    print("\nLast-Year-Predicts-This-Year (per month)")
    print(ly.to_string(float_format=lambda x: f"{x: .4f}"))


def print_prewindow_report(df: pd.DataFrame, price_col: str, use_log_returns: bool, month_agg: str):
    print("\nPre-Window 60–90 Days Predictive Test")
    print("------------------------------------")
    res = prewindow_predicts_month(df, price_col=price_col, use_log_returns=use_log_returns, month_agg=month_agg)
    print(f"n={int(res['n'])}, slope={fmt_float(res['slope'])}, r={fmt_float(res['r'])}, p={fmt_float(res['p_value'])}")

# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Intraday autocorr & seasonal scanner for commodity data")
    p.add_argument("--csv", required=True, help="Path to CSV with intraday data")
    p.add_argument("--price-col", default="Close", help="Price column to use (default Close)")
    p.add_argument("--tz", default=None, help="Timezone name to convert timestamps to (e.g., 'America/New_York')")
    # Intraday pair
    p.add_argument("--time-a", default="09:30", help="HH:MM time-of-day for window A start")
    p.add_argument("--time-b", default="10:00", help="HH:MM time-of-day for window B start")
    p.add_argument("--window-a", type=int, default=1, help="Bars to aggregate for window A")
    p.add_argument("--window-b", type=int, default=1, help="Bars to aggregate for window B")
    # Seasonality
    p.add_argument("--month-agg", default="M", choices=["M", "MS"], help="Monthly boundary: 'M'=month-end, 'MS'=month-start")
    p.add_argument("--arith", action="store_true", help="Use arithmetic pct returns instead of log")

    args = p.parse_args()

    df = load_prices(args.csv, tz=args.tz, price_col=args.price_col)
    use_log = not args.arith

    # Intraday pair analysis
    spec = IntradayPairSpec(time_a=args.time_a, time_b=args.time_b, window_a=args.window_a, window_b=args.window_b)
    basic = intraday_autocorr_between_times(df, args.price_col, spec)
    lag1 = intraday_lag_autocorr(df, args.price_col, k=1)
    print_intraday_report(spec, basic, lag1)

    # Monthly seasonality & last-year predictive power
    mr = monthly_returns(df, price_col=args.price_col, use_log_returns=use_log, month_agg=args.month_agg)
    print_seasonality_report(mr)

    # Pre-window predictive test (60–90 days)
    print_prewindow_report(df, price_col=args.price_col, use_log_returns=use_log, month_agg=args.month_agg)


if __name__ == "__main__":
    main()
