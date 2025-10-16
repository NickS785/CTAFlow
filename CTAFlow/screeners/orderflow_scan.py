"""Orderflow seasonality and event detection on volume buckets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.session import filter_session_ticks
from ..utils.volume_bucket import auto_bucket_size, ticks_to_volume_buckets
from stats_utils.fdr import fdr_bh

__all__ = ["OrderflowParams", "orderflow_scan"]


@dataclass(slots=True)
class OrderflowParams:
    session_start: str
    session_end: str
    tz: str = "America/Chicago"
    bucket_size: int | str = "auto"
    vpin_window: int = 50
    threshold_z: float = 2.0
    min_days: int = 30
    cadence_target: int = 50
    grid_multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5)


_TimeLike = str | time


def _parse_time(value: _TimeLike) -> time:
    if isinstance(value, time):
        return value
    parsed = pd.to_datetime(value).time()
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _time_to_micros(value: time) -> int:
    return (
        ((value.hour * 60 + value.minute) * 60 + value.second) * 1_000_000
        + value.microsecond
    )


def _time_in_session(target: time, start: time, end: time) -> bool:
    target_us = _time_to_micros(target)
    start_us = _time_to_micros(start)
    end_us = _time_to_micros(end)
    if start_us <= end_us:
        return start_us <= target_us <= end_us
    return target_us >= start_us or target_us <= end_us


def _robust_daily_zscores(df: pd.DataFrame, metric: str, date_col: str) -> pd.Series:
    z = pd.Series(np.nan, index=df.index, dtype=float)
    grouped = df.groupby(date_col)[metric]
    for _, values in grouped:
        valid = values.dropna()
        if valid.empty:
            continue
        median = float(np.nanmedian(valid))
        abs_dev = (valid - median).abs()
        mad = float(np.nanmedian(abs_dev))
        if mad > 0:
            scaled = 0.67448975 * (valid - median) / mad
        else:
            std = float(np.nanstd(valid))
            if std > 0:
                scaled = (valid - median) / std
            else:
                scaled = pd.Series(0.0, index=valid.index)
        z.loc[valid.index] = scaled
    return z


def _collapse_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in ticks.columns if col != "ts"]
    collapsed = (
        ticks.groupby("ts", as_index=False)[numeric_cols]
        .sum(numeric_only=True)
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return collapsed


def _intraday_pressure_table(
    df: pd.DataFrame,
    metrics: Dict[str, pd.Series],
    ticker: str,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    df = df.copy()
    df["clock_time"] = df["ts_end"].dt.time
    for metric_name, series in metrics.items():
        metric_df = pd.DataFrame({"value": series, "clock_time": df["clock_time"]}).dropna()
        if metric_df.empty:
            continue
        grouped = metric_df.groupby("clock_time")["value"]
        for clock, values in grouped:
            records.append(
                {
                    "ticker": ticker,
                    "clock_time": clock,
                    "metric": metric_name,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "n": int(values.count()),
                }
            )
    return pd.DataFrame(records)


def _event_runs(
    df: pd.DataFrame,
    metric: str,
    z_col: str,
    threshold: float,
    ticker: str,
) -> pd.DataFrame:
    mask = df[z_col].abs() >= threshold
    if not mask.any():
        return pd.DataFrame(columns=[
            "ticker",
            "metric",
            "ts_start",
            "ts_end",
            "run_len",
            "max_abs_z",
            "direction",
        ])

    runs = df.loc[mask, ["bucket", "ts_start", "ts_end", z_col]]
    runs = runs.assign(
        group=(runs["bucket"].diff().fillna(1) != 1).cumsum()
    )
    records: List[Dict[str, object]] = []
    for _, group_df in runs.groupby("group"):
        max_idx = group_df[z_col].abs().idxmax()
        max_z = float(df.loc[max_idx, z_col])
        records.append(
            {
                "ticker": ticker,
                "metric": metric,
                "ts_start": df.loc[group_df.index[0], "ts_start"],
                "ts_end": df.loc[group_df.index[-1], "ts_end"],
                "run_len": int(len(group_df)),
                "max_abs_z": float(abs(max_z)),
                "direction": "positive" if max_z >= 0 else "negative",
            }
        )
    return pd.DataFrame(records)


def _seasonality_table(
    df: pd.DataFrame,
    group_cols: List[str],
    metric: str,
    ticker: str,
    exploratory: bool,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    subset = df.dropna(subset=[metric])
    if subset.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "ticker",
                "metric",
                "mean",
                "t_stat",
                "p_value",
                "q_value",
                "sig_fdr_5pct",
                "n",
                "exploratory",
            ]
        )
    grouped = subset.groupby(group_cols)[metric]
    for key, values in grouped:
        n = int(values.count())
        mean_val = float(values.mean())
        std_val = float(values.std(ddof=1)) if n > 1 else np.nan
        if n > 1 and std_val > 0:
            t_stat = mean_val / (std_val / np.sqrt(n))
            p_val = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        elif n > 1:
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = np.nan
            p_val = np.nan
        record = {
            **{col: val for col, val in zip(group_cols, key if isinstance(key, tuple) else (key,))},
            "ticker": ticker,
            "metric": metric,
            "mean": mean_val,
            "t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
            "p_value": p_val,
            "n": n,
            "exploratory": exploratory,
        }
        records.append(record)
    table = pd.DataFrame(records)
    if table.empty:
        return table
    fdr_result = fdr_bh(table["p_value"].to_numpy(), alpha=0.05)
    table["q_value"] = fdr_result.q_values
    table["sig_fdr_5pct"] = table["q_value"] <= 0.05
    return table


def orderflow_scan(
    tick_source: Callable[[str], pd.DataFrame],
    symbols: Iterable[str],
    params: OrderflowParams,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    session_start = _parse_time(params.session_start)
    session_end = _parse_time(params.session_end)

    for symbol in symbols:
        try:
            ticks = tick_source(symbol)
        except Exception as exc:  # pragma: no cover - tick loader failure path
            results[symbol] = {"error": str(exc)}
            continue

        if ticks is None or ticks.empty:
            results[symbol] = {"error": "No tick data"}
            continue

        filtered = filter_session_ticks(ticks, params.tz, session_start, session_end)
        if filtered.empty:
            results[symbol] = {
                "error": "No ticks within session window",
                "metadata": {
                    "session_start": params.session_start,
                    "session_end": params.session_end,
                    "tz": params.tz,
                    "n_sessions": 0,
                    "n_buckets": 0,
                },
            }
            continue

        collapsed = _collapse_ticks(filtered)
        bucket_param: Optional[int]
        if isinstance(params.bucket_size, str) and params.bucket_size.lower() == "auto":
            bucket_param = auto_bucket_size(
                collapsed,
                cadence_target=params.cadence_target,
                grid_multipliers=params.grid_multipliers,
            )
            selection_method = "auto"
        else:
            bucket_param = int(params.bucket_size)
            if bucket_param <= 0:
                raise ValueError("bucket_size must be positive")
            selection_method = "manual"

        bucket_df = ticks_to_volume_buckets(collapsed, bucket_param)
        if bucket_df.empty:
            results[symbol] = {
                "error": "No buckets generated",
                "metadata": {
                    "session_start": params.session_start,
                    "session_end": params.session_end,
                    "tz": params.tz,
                    "n_sessions": 0,
                    "n_buckets": 0,
                    "bucket_size": bucket_param,
                    "bucket_selection": selection_method,
                },
            }
            continue

        bucket_df = bucket_df.copy()
        bucket_df["ticker"] = symbol
        bucket_df["vpin"] = (
            bucket_df["Imbalance"].abs() / bucket_param
        ).rolling(window=params.vpin_window, min_periods=1).mean()
        bucket_df["buy_pressure"] = bucket_df["AskShare"] - 0.5
        bucket_df["ask_pressure"] = bucket_df["AskQuoteShare"] - 0.5

        bucket_df["session_date"] = bucket_df["ts_end"].dt.tz_convert(params.tz).dt.date
        bucket_df["weekday"] = bucket_df["ts_end"].dt.day_name()
        bucket_df["weekday_number"] = bucket_df["ts_end"].dt.dayofweek
        bucket_df["week_of_month"] = ((bucket_df["ts_end"].dt.day - 1) // 7) + 1

        metrics = {"buy_pressure": bucket_df["buy_pressure"]}
        if bucket_df["ask_pressure"].notna().any():
            metrics["ask_pressure"] = bucket_df["ask_pressure"]

        intraday = _intraday_pressure_table(bucket_df, metrics, symbol)

        events_list: List[pd.DataFrame] = []
        for metric_name in metrics:
            z_col = f"{metric_name}_z"
            bucket_df[z_col] = _robust_daily_zscores(bucket_df, metric_name, "session_date")
            events_list.append(
                _event_runs(
                    bucket_df,
                    metric_name,
                    z_col,
                    params.threshold_z,
                    symbol,
                )
            )
        events = pd.concat(events_list, ignore_index=True) if events_list else pd.DataFrame()

        n_sessions = int(bucket_df["session_date"].nunique())
        exploratory_flag = n_sessions < params.min_days

        weekly_tables: List[pd.DataFrame] = []
        for metric_name in metrics:
            weekly_tables.append(
                _seasonality_table(
                    bucket_df,
                    ["weekday"],
                    metric_name,
                    symbol,
                    exploratory=exploratory_flag,
                )
            )
        df_weekly = pd.concat(weekly_tables, ignore_index=True) if weekly_tables else pd.DataFrame()

        wom_tables: List[pd.DataFrame] = []
        for metric_name in metrics:
            table = _seasonality_table(
                bucket_df,
                ["week_of_month", "weekday"],
                metric_name,
                symbol,
                exploratory=exploratory_flag,
            )
            wom_tables.append(table)
        df_wom = pd.concat(wom_tables, ignore_index=True) if wom_tables else pd.DataFrame()

        metadata = {
            "session_start": params.session_start,
            "session_end": params.session_end,
            "tz": params.tz,
            "bucket_size": bucket_param,
            "bucket_selection": selection_method,
            "n_sessions": n_sessions,
            "days": n_sessions,
            "n_buckets": int(len(bucket_df)),
            "exploratory": n_sessions < params.min_days,
        }

        bucket_cols = [
            "ticker",
            "bucket",
            "ts_start",
            "ts_end",
            "AskVolume",
            "BidVolume",
            "TotalVolume",
            "n_ticks",
            "AskShare",
            "AskQuoteShare",
            "Imbalance",
            "ImbalanceFraction",
            "vpin",
            "buy_pressure",
            "ask_pressure",
        ]
        if "AskQuoteVolume" in bucket_df.columns:
            bucket_cols.append("AskQuoteVolume")
        if "BidQuoteVolume" in bucket_df.columns:
            bucket_cols.append("BidQuoteVolume")

        results[symbol] = {
            "df_buckets": bucket_df[bucket_cols].copy(),
            "df_intraday_pressure": intraday,
            "df_events": events,
            "df_weekly": df_weekly,
            "df_wom_weekday": df_wom,
            "metadata": metadata,
        }

    return results


def _default_tick_loader_factory(path_template: str) -> Callable[[str], pd.DataFrame]:
    from pathlib import Path

    def _loader(symbol: str) -> pd.DataFrame:
        path = Path(path_template.format(symbol=symbol))
        if not path.exists():
            raise FileNotFoundError(f"No tick file found for {symbol} at {path}")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported tick format: {path.suffix}")

    return _loader


def _build_arg_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(description="Run orderflow scan on tick data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to scan", required=True)
    parser.add_argument("--session", nargs=2, metavar=("START", "END"), help="Session start/end HH:MM")
    parser.add_argument("--tz", default="America/Chicago", help="Session timezone")
    parser.add_argument("--bucket", default="auto", help="Bucket size (int or 'auto')")
    parser.add_argument("--z", type=float, default=2.0, help="Z-score threshold for events")
    parser.add_argument("--vpin-window", type=int, default=50, help="VPIN rolling window")
    parser.add_argument("--min-days", type=int, default=30, help="Minimum days for full significance")
    parser.add_argument("--cadence", type=int, default=50, help="Target buckets/day for auto tuning")
    parser.add_argument(
        "--grid",
        nargs="*",
        type=float,
        default=(0.5, 0.75, 1.0, 1.25, 1.5),
        help="Grid multipliers for auto bucket selection",
    )
    parser.add_argument(
        "--ticks-path",
        default="{symbol}.csv",
        help="Path template for tick files (use {symbol} placeholder)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    bucket = args.bucket
    try:
        bucket_value: int | str = int(bucket)
    except ValueError:
        bucket_value = bucket

    params = OrderflowParams(
        session_start=args.session[0],
        session_end=args.session[1],
        tz=args.tz,
        bucket_size=bucket_value,
        vpin_window=args.vpin_window,
        threshold_z=args.z,
        min_days=args.min_days,
        cadence_target=args.cadence,
        grid_multipliers=tuple(args.grid) if args.grid else (0.5, 0.75, 1.0, 1.25, 1.5),
    )

    tick_loader = _default_tick_loader_factory(args.ticks_path)
    scan_results = orderflow_scan(tick_loader, args.symbols, params)
    for symbol, payload in scan_results.items():
        print(f"=== {symbol} ===")
        if "error" in payload:
            print(f"Error: {payload['error']}")
            continue
        meta = payload.get("metadata", {})
        print(
            f"Buckets: {meta.get('n_buckets', 0)}, Sessions: {meta.get('n_sessions', 0)}, "
            f"Bucket size: {meta.get('bucket_size', 'n/a')} ({meta.get('bucket_selection', 'n/a')})"
        )
        print("Weekly signals:")
        weekly = payload.get("df_weekly")
        if isinstance(weekly, pd.DataFrame) and not weekly.empty:
            preview = weekly.loc[weekly["sig_fdr_5pct"], ["weekday", "metric", "mean", "q_value"]]
            if preview.empty:
                print("  No significant weekday effects at 5% FDR")
            else:
                print(preview.to_string(index=False))
        else:
            print("  No weekly stats available")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
