"""Liquidity and volume diagnostics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from CTAFlow.stats_utils import fdr_bh
from CTAFlow.stats_utils import regularized_beta

LOGGER = logging.getLogger(__name__)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df


def _extract_ticker(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    ticker = None
    for col in ("ticker", "symbol"):
        if col in df.columns:
            unique = pd.unique(df[col].dropna())
            if len(unique) > 1:
                raise ValueError("Multiple tickers detected; provide one ticker at a time")
            if len(unique) == 1:
                ticker = unique[0]
            df = df.drop(columns=col)
            break
    return df, ticker


def _apply_tz(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if tz is None:
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert(tz)


def _resample_intraday(df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if rule is None:
        return df
    agg = {
        col: ("sum" if col.lower().startswith("vol") else "last")
        for col in df.columns
    }
    agg.setdefault("volume", "sum")
    return df.resample(rule).agg(agg).dropna(how="all")


def _filter_calendar(
    df: pd.DataFrame,
    months: Optional[Iterable[int]] = None,
    weekdays: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    if months is not None:
        months = {int(m) for m in months}
        df = df[df.index.month.isin(months)]
    if weekdays is not None:
        weekdays = {int(w) for w in weekdays}
        df = df[df.index.weekday().isin(weekdays)]
    return df


def _winsorize(series: pd.Series, proportion: float = 0.01) -> pd.Series:
    if series.empty:
        return series
    lower = series.quantile(proportion)
    upper = series.quantile(1 - proportion)
    return series.clip(lower, upper)


def compute_liquidity_intraday(
    bars: pd.DataFrame,
    tz: Optional[str] = None,
    resample: Optional[str] = None,
    months: Optional[Iterable[int]] = None,
    weekdays: Optional[Iterable[int]] = None,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Compute intraday volume distribution diagnostics.

    Parameters
    ----------
    bars:
        Intraday bar data with a :class:`~pandas.DatetimeIndex`, ``volume`` and
        ``close`` columns.
    tz:
        Optional timezone string for reporting clock times.
    resample:
        Optional pandas offset string used to resample the bars prior to
        aggregation.
    months, weekdays:
        Optional filters limiting the analysis to selected calendar months or
        weekdays (``0`` = Monday).
    min_samples:
        Minimum number of trading days required for a clock-time bucket to be
        retained in the output.

    Returns
    -------
    pandas.DataFrame
        Table with average/median volume per clock time as well as peak/trough
        flags.

    Examples
    --------
    >>> df = pd.DataFrame({"volume": [100, 200]}, index=pd.date_range("2023-01-01", periods=2, freq="5min"))
    >>> compute_liquidity_intraday(df, min_samples=1)[["clock_time", "pct_of_daily_vol"]].head(1)
      clock_time  pct_of_daily_vol
    0       00:00               0.5
    """

    if "volume" not in bars.columns:
        raise KeyError("bars DataFrame must include a 'volume' column")
    df = bars.copy()
    df, ticker = _extract_ticker(df)
    df = _ensure_datetime_index(df)
    df = _apply_tz(df, tz)
    df = _resample_intraday(df, resample)
    df = _filter_calendar(df, months, weekdays)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker",
            "session",
            "clock_time",
            "avg_volume",
            "median_volume",
            "pct_of_daily_vol",
            "peak_flag",
            "trough_flag",
            "n_days",
        ])

    df["volume"] = df["volume"].fillna(0.0)
    day_groups = df.groupby(df.index.normalize())
    records = []
    for day, day_df in day_groups:
        total_vol = day_df["volume"].sum()
        if total_vol <= 0:
            continue
        shares = day_df["volume"] / total_vol
        shares = _winsorize(shares, 0.01)
        day_df = day_df.assign(pct_of_daily_vol=shares)
        day_df = day_df.assign(clock_time=day_df.index.strftime("%H:%M"))
        records.append(day_df)
    if not records:
        return pd.DataFrame()
    stacked = pd.concat(records)
    agg = stacked.groupby("clock_time")
    summary = agg.agg(
        avg_volume=("volume", "mean"),
        median_volume=("volume", "median"),
        pct_of_daily_vol=("pct_of_daily_vol", "mean"),
        n_days=("pct_of_daily_vol", "count"),
    ).reset_index()
    summary = summary[summary["n_days"] >= min_samples]
    if summary.empty:
        return summary
    summary["ticker"] = ticker
    summary["session"] = None
    max_idx = summary["pct_of_daily_vol"].idxmax()
    min_idx = summary["pct_of_daily_vol"].idxmin()
    summary["peak_flag"] = False
    summary["trough_flag"] = False
    if pd.notna(max_idx):
        summary.loc[max_idx, "peak_flag"] = True
    if pd.notna(min_idx):
        summary.loc[min_idx, "trough_flag"] = True
    return summary[
        [
            "ticker",
            "session",
            "clock_time",
            "avg_volume",
            "median_volume",
            "pct_of_daily_vol",
            "peak_flag",
            "trough_flag",
            "n_days",
        ]
    ]


def compute_volume_seasonality(daily: pd.DataFrame) -> pd.DataFrame:
    """Evaluate monthly and weekday volume seasonality.

    Parameters
    ----------
    daily:
        Daily volume totals with a datetime index and optional ``ticker``
        column.

    Returns
    -------
    pandas.DataFrame
        Seasonality diagnostics by month and weekday including BH adjusted
        p-values.
    """

    if "volume" not in daily.columns:
        raise KeyError("daily DataFrame must include a 'volume' column")
    df = daily.copy()
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df.index = pd.to_datetime(df.pop("date"))
    else:
        df.index = pd.to_datetime(df.index)
    df, ticker = _extract_ticker(df)
    df = df.sort_index()
    df["volume"] = _winsorize(df["volume"], 0.01)
    df["month"] = df.index.month
    df["weekday"] = df.index.weekday()
    outputs = []
    for bucket, col in ("month", "month"), ("weekday", "weekday"):
        grouped = df.groupby(col)
        overall_mean = df["volume"].mean()
        for bucket_value, grp in grouped:
            n = len(grp)
            mean_vol = grp["volume"].mean()
            median_vol = grp["volume"].median()
            std = grp["volume"].std(ddof=1)
            if n <= 1 or std == 0 or np.isnan(std):
                t_stat = np.nan
                p_value = np.nan
            else:
                se = std / np.sqrt(n)
                t_stat = (mean_vol - overall_mean) / se
                p_value = _student_t_two_sided_pvalue(t_stat, n - 1)
            outputs.append(
                {
                    "ticker": ticker,
                    "bucket": bucket,
                    "bucket_value": int(bucket_value),
                    "avg_daily_volume": mean_vol,
                    "median_daily_volume": median_vol,
                    "n_days": n,
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(outputs)
    if result.empty:
        return result
    result["q_value"] = np.nan
    result["sig_fdr_5pct"] = False
    for bucket in result["bucket"].unique():
        mask = result["bucket"] == bucket
        res = fdr_bh(result.loc[mask, "p_value"].to_numpy(), alpha=0.05)
        result.loc[mask, "q_value"] = res.q_values
        result.loc[mask, "sig_fdr_5pct"] = res.rejected
    return result


def _student_t_two_sided_pvalue(t_stat: float, dfree: float) -> float:
    if dfree <= 0 or not np.isfinite(dfree):
        return np.nan
    if t_stat is None or not np.isfinite(t_stat):
        return np.nan
    x = dfree / (dfree + float(t_stat) ** 2)
    try:
        pval = regularized_beta(0.5 * dfree, 0.5, x)
    except ValueError:
        return np.nan
    return min(max(pval, 0.0), 1.0)


def compute_volume_effects(bars: pd.DataFrame) -> pd.DataFrame:
    """Correlate volume concentration features with returns and volatility.

    Parameters
    ----------
    bars:
        Intraday bar data containing ``volume`` and ``close`` columns.

    Returns
    -------
    pandas.DataFrame
        Correlation diagnostics for volume distribution features versus next
        day returns and realised volatility.
    """

    if "volume" not in bars.columns or "close" not in bars.columns:
        raise KeyError("bars must include 'volume' and 'close' columns")
    df = bars.copy()
    df, ticker = _extract_ticker(df)
    df = _ensure_datetime_index(df)
    df = df.sort_index()
    df["volume"] = df["volume"].fillna(0.0)
    df["close"] = df["close"].ffill()

    day_groups = df.groupby(df.index.normalize())
    features = []
    closes = []
    dates = []
    for day, day_df in day_groups:
        total_vol = day_df["volume"].sum()
        if total_vol <= 0:
            continue
        minutes = (day_df.index - day_df.index[0]).total_seconds() / 60.0
        minutes = pd.Series(minutes, index=day_df.index)
        close_time = minutes.iloc[-1]
        share = day_df["volume"] / total_vol
        open_share = share[minutes <= 30].sum()
        close_share = share[(close_time - minutes) <= 30].sum()
        lunch_share = share[(minutes >= 180) & (minutes <= 240)].sum()
        intraday_returns = day_df["close"].pct_change().dropna()
        realized_vol = np.sqrt((intraday_returns ** 2).sum()) if not intraday_returns.empty else np.nan
        features.append(
            {
                "date": day,
                "vol_share_open30": float(open_share),
                "vol_share_close30": float(close_share),
                "vol_share_lunch": float(lunch_share),
                "volatility": realized_vol,
            }
        )
        closes.append(day_df["close"].iloc[-1])
        dates.append(day)
    if not features:
        return pd.DataFrame()
    feat_df = pd.DataFrame(features).set_index("date")
    close_series = pd.Series(closes, index=pd.to_datetime(dates)).sort_index()
    daily_returns = close_series.pct_change()
    feat_df["return_next"] = daily_returns.shift(-1)
    feat_df = feat_df.dropna()
    rows = []
    for feature in ["vol_share_open30", "vol_share_close30", "vol_share_lunch", "volatility"]:
        for target in ["return_next", "volatility"]:
            if feature == target:
                continue
            valid = feat_df[[feature, target]].dropna()
            if len(valid) <= 2:
                r = np.nan
                p_value = np.nan
                n = len(valid)
            else:
                r = valid[feature].corr(valid[target])
                if np.isnan(r):
                    p_value = np.nan
                else:
                    dfree = len(valid) - 2
                    if abs(r) >= 1:
                        p_value = 0.0
                    else:
                        t_stat = r * np.sqrt(dfree / (1 - r ** 2))
                        p_value = _student_t_two_sided_pvalue(t_stat, dfree)
                n = len(valid)
            rows.append(
                {
                    "ticker": ticker,
                    "feature": feature,
                    "target": target,
                    "r": r,
                    "n": n,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value"] = np.nan
    result["sig_fdr_5pct"] = False
    for feature in result["feature"].unique():
        mask = result["feature"] == feature
        res = fdr_bh(result.loc[mask, "p_value"].to_numpy(), alpha=0.05)
        result.loc[mask, "q_value"] = res.q_values
        result.loc[mask, "sig_fdr_5pct"] = res.rejected
    return result


def export_liquidity_tidy(data: Mapping[str, pd.DataFrame] | pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run all liquidity diagnostics.

    Parameters
    ----------
    data:
        Either a mapping containing ``bars``/``daily`` entries or a bare
        intraday DataFrame.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Intraday, seasonality and effects tidy outputs.
    """

    if isinstance(data, Mapping):
        bars = data.get("bars")
        daily = data.get("daily")
    else:
        bars = data
        daily = None
    if bars is None or bars.empty:
        raise ValueError("Intraday bars are required for liquidity diagnostics")
    intraday = compute_liquidity_intraday(bars)
    if daily is None:
        bars_copy = bars.copy()
        ticker = None
        if isinstance(bars_copy, pd.DataFrame) and "ticker" in bars_copy.columns:
            unique = pd.unique(bars_copy["ticker"].dropna())
            if len(unique) == 1:
                ticker = unique[0]
            bars_copy = bars_copy.drop(columns="ticker")
        daily_df = bars_copy.resample("1D").agg({"volume": "sum"})
        if ticker is not None:
            daily_df["ticker"] = ticker
    else:
        daily_df = daily
    seasonality = compute_volume_seasonality(daily_df)
    effects = compute_volume_effects(bars)
    return {
        "intraday": intraday,
        "seasonality": seasonality,
        "effects": effects,
    }


def _load_bars(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raise ValueError(f"Unsupported bars format: {path}")


def _main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Liquidity diagnostics")
    parser.add_argument("--bars", required=True, type=Path, help="Path to intraday bars file")
    parser.add_argument("--out", required=True, type=Path, help="Output folder")
    parser.add_argument("--ticker", type=str, default=None, help="Optional ticker override")
    args = parser.parse_args(list(argv) if argv is not None else None)
    bars = _load_bars(args.bars)
    if args.ticker is not None:
        bars["ticker"] = args.ticker
    diagnostics = export_liquidity_tidy({"bars": bars})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in diagnostics.items():
        if df.empty:
            continue
        df.to_csv(args.out / f"liquidity_{name}.csv", index=False)
    summary = {name: len(df) for name, df in diagnostics.items()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI hook
    raise SystemExit(_main())
