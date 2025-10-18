"""Orderflow seasonality and event detection on volume buckets."""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.session import filter_session_ticks
from ..utils.volume_bucket import auto_bucket_size, ticks_to_volume_buckets
from stats_utils.fdr import fdr_bh
from CTAFlow.features.event_study import EventSpec, run_event_study

__all__ = ["OrderflowParams", "OrderflowScanner", "orderflow_scan"]


_MONTH_TO_SEASON = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
    12: "winter",
}

_SEASON_ALIASES = {
    "winter": "winter",
    "spring": "spring",
    "summer": "summer",
    "fall": "fall",
    "autumn": "fall",
}

_QUARTER_ALIASES = {
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
    "quarter1": 1,
    "quarter2": 2,
    "quarter3": 3,
    "quarter4": 4,
    "quarter_1": 1,
    "quarter_2": 2,
    "quarter_3": 3,
    "quarter_4": 4,
}


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
    month_filter: Optional[Sequence[int]] = None
    season_filter: Optional[Sequence[str]] = None
    name: Optional[str] = None  # Optional name for identifying scan results
    event_pre_bars: int = 5
    event_post_bars: int = 20


@dataclass(frozen=True, slots=True)
class _SessionKey:
    session_start: str
    session_end: str
    tz: str


@dataclass(frozen=True, slots=True)
class _BucketKey:
    mode: str
    bucket_size: Optional[int]
    cadence_target: Optional[int]
    grid_multipliers: Optional[Tuple[float, ...]]


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


def _normalize_time_str(value: _TimeLike) -> str:
    return _parse_time(value).strftime("%H:%M:%S")


def _make_session_key(params: OrderflowParams) -> _SessionKey:
    return _SessionKey(
        session_start=_normalize_time_str(params.session_start),
        session_end=_normalize_time_str(params.session_end),
        tz=params.tz,
    )


def _make_bucket_key(params: OrderflowParams) -> _BucketKey:
    if isinstance(params.bucket_size, str) and params.bucket_size.lower() == "auto":
        multipliers = tuple(float(x) for x in params.grid_multipliers)
        return _BucketKey(
            mode="auto",
            bucket_size=None,
            cadence_target=int(params.cadence_target),
            grid_multipliers=multipliers,
        )

    bucket_size = int(params.bucket_size)
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    return _BucketKey(
        mode="manual",
        bucket_size=bucket_size,
        cadence_target=None,
        grid_multipliers=None,
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
    if "clock_time" not in df.columns:
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


def _event_flagger(event_times: pd.DatetimeIndex):
    """Return an event function compatible with :class:`EventSpec`."""

    if event_times.empty:
        def _no_events(df: pd.DataFrame) -> pd.Series:
            return pd.Series(False, index=df.index)

        return _no_events

    times = pd.DatetimeIndex(event_times)
    times = times[~times.duplicated()].sort_values()
    if times.tz is None:
        times = times.tz_localize("UTC")
    else:
        times = times.tz_convert("UTC")

    encoded = times.view("int64")

    def _flag(df: pd.DataFrame) -> pd.Series:
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        mask = np.in1d(idx.view("int64"), encoded)
        return pd.Series(mask, index=df.index)

    return _flag


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


def _normalize_season_filters(values: Sequence[str]) -> Tuple[set[str], set[int], List[str]]:
    seasons: set[str] = set()
    quarters: set[int] = set()
    normalized: List[str] = []
    for raw in values:
        if raw is None:
            continue
        label = str(raw).strip().lower()
        if not label:
            continue
        if label in _SEASON_ALIASES:
            canonical = _SEASON_ALIASES[label]
            seasons.add(canonical)
            normalized.append(canonical)
        elif label in _QUARTER_ALIASES:
            quarter = _QUARTER_ALIASES[label]
            quarters.add(_QUARTER_ALIASES[label])
            normalized.append(f"Q{_QUARTER_ALIASES[label]}")
        else:
            raise ValueError(f"Unsupported season/quarter label: {raw}")
    # Preserve input order but drop duplicates
    normalized_unique = list(dict.fromkeys(normalized))
    return seasons, quarters, normalized_unique


def _build_period_filter(
    df: pd.DataFrame,
    month_filter: Optional[Sequence[int]],
    season_filter: Optional[Sequence[str]],
) -> Tuple[Optional[pd.Series], Dict[str, object]]:
    filters: Dict[str, object] = {}
    mask = pd.Series(True, index=df.index, dtype=bool)
    applied = False

    if month_filter:
        months = sorted({int(m) for m in month_filter})
        for month in months:
            if month < 1 or month > 12:
                raise ValueError(f"Month filter values must be between 1 and 12. Got {month}.")
        mask &= df["month"].isin(months)
        filters["months"] = months
        applied = True

    if season_filter:
        seasons, quarters, normalized = _normalize_season_filters(season_filter)
        if seasons:
            mask &= df["season"].isin(seasons)
        if quarters:
            mask &= df["quarter"].isin(list(quarters))
            filters["quarters"] = sorted(quarters)
        if normalized:
            filters["season_filters"] = normalized
        applied = True

    if not applied:
        return None, filters

    return mask, filters


def _metric_bias(metric: str, mean_value: float) -> str:
    if np.isnan(mean_value) or mean_value == 0:
        return "neutral"
    lower_metric = metric.lower()
    if lower_metric in {"buy_pressure", "quote_buy_pressure"}:
        return "buy" if mean_value > 0 else "sell"
    if lower_metric == "sell_pressure":
        return "sell" if mean_value > 0 else "buy"
    return "positive" if mean_value > 0 else "negative"


def _weekly_peak_pressure(
    bucket_df: pd.DataFrame,
    weekly_table: pd.DataFrame,
    metrics: Dict[str, pd.Series],
    ticker: str,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "metric",
        "weekday",
        "clock_time",
        "pressure_bias",
        "seasonality_mean",
        "seasonality_t_stat",
        "seasonality_q_value",
        "seasonality_n",
        "seasonality_sig_fdr_5pct",
        "intraday_mean",
        "intraday_median",
        "intraday_n",
        "exploratory",
    ]

    if weekly_table is None or weekly_table.empty:
        return pd.DataFrame(columns=columns)

    sig_mask = weekly_table.get("sig_fdr_5pct", pd.Series(dtype=bool)).fillna(False)
    if not sig_mask.any():
        return pd.DataFrame(columns=columns)

    records: List[Dict[str, object]] = []

    for _, row in weekly_table.loc[sig_mask].iterrows():
        metric = row["metric"]
        weekday = row["weekday"]
        if metric not in metrics:
            continue
        subset = bucket_df.loc[bucket_df["weekday"] == weekday, ["clock_time", metric]]
        subset = subset.dropna(subset=[metric])
        if subset.empty:
            continue
        grouped = subset.groupby("clock_time")[metric].agg(["mean", "median", "count"])
        if grouped.empty:
            continue

        weekly_mean = float(row.get("mean", np.nan))
        if np.isnan(weekly_mean):
            target_time = grouped["mean"].abs().idxmax()
        elif weekly_mean > 0:
            target_time = grouped["mean"].idxmax()
        else:
            target_time = grouped["mean"].idxmin()

        peak = grouped.loc[target_time]
        bias = _metric_bias(metric, weekly_mean if not np.isnan(weekly_mean) else float(peak["mean"]))

        records.append(
            {
                "ticker": ticker,
                "metric": metric,
                "weekday": weekday,
                "clock_time": target_time,
                "pressure_bias": bias,
                "seasonality_mean": weekly_mean,
                "seasonality_t_stat": float(row.get("t_stat", np.nan)),
                "seasonality_q_value": float(row.get("q_value", np.nan)),
                "seasonality_n": int(row.get("n", 0)) if not pd.isna(row.get("n")) else 0,
                "seasonality_sig_fdr_5pct": bool(row.get("sig_fdr_5pct", False)),
                "intraday_mean": float(peak["mean"]),
                "intraday_median": float(peak["median"]),
                "intraday_n": int(peak["count"]),
                "exploratory": bool(row.get("exploratory", False)),
            }
        )

    return pd.DataFrame(records, columns=columns)


class OrderflowScanner:
    """
    Orderflow scanner for tick data analysis.

    Analyzes volume-bucketed tick data for seasonality patterns, pressure metrics,
    and orderflow events. Stores results in HDF5 for persistence.

    Features:
        - Volume bucket analysis with auto-sizing
        - Buy/sell pressure metrics and VPIN calculation
        - Intraday seasonality detection (weekly, week-of-month)
        - Event detection with FDR-corrected significance
        - HDF5 storage with per-symbol organization
        - Multi-scan support: Run multiple configurations (like HistoricalScreener.run_screens)
    """

    def __init__(self,
                 params: Optional[OrderflowParams] = None,
                 hdf_path: Optional[Path | str] = None):
        """
        Initialize OrderflowScanner.

        Args:
            params: OrderflowParams configuration (optional if using run_scans)
            hdf_path: Path to HDF5 file for results storage (created if doesn't exist)
        """
        self.params = params
        self.hdf_path = Path(hdf_path) if hdf_path else None
        self.results: Dict[str, Dict[str, object]] = {}

        # Parse session times once (if params provided)
        if params:
            self.session_start = _parse_time(params.session_start)
            self.session_end = _parse_time(params.session_end)
        else:
            self.session_start = None
            self.session_end = None

    def scan(self, tick_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, object]]:
        """
        Scan tick data for all symbols.

        Args:
            tick_data: Dictionary mapping symbols to their tick DataFrames

        Returns:
            Dictionary mapping symbols to analysis results
        """
        self.results = {}

        for symbol, ticks in tick_data.items():
            try:
                if ticks is None or ticks.empty:
                    self.results[symbol] = {"error": "No tick data"}
                    continue

                # Normalize timestamp column
                ticks = _normalize_tick_timestamps(ticks)

            except Exception as exc:
                self.results[symbol] = {"error": str(exc)}
                continue

            # Process symbol
            result = self._process_symbol(symbol, ticks)
            self.results[symbol] = result

        return self.results

    def _process_symbol(self, symbol: str, ticks: pd.DataFrame) -> Dict[str, object]:
        """Process a single symbol's tick data."""
        # Filter to session window
        filtered = filter_session_ticks(ticks, self.params.tz, self.session_start, self.session_end)
        if filtered.empty:
            return {
                "error": "No ticks within session window",
                "metadata": {
                    "session_start": self.params.session_start,
                    "session_end": self.params.session_end,
                    "tz": self.params.tz,
                    "n_sessions": 0,
                    "n_buckets": 0,
                },
            }

        # Collapse and bucket
        collapsed = _collapse_ticks(filtered)
        bucket_param: Optional[int]
        if isinstance(self.params.bucket_size, str) and self.params.bucket_size.lower() == "auto":
            bucket_param = auto_bucket_size(
                collapsed,
                cadence_target=self.params.cadence_target,
                grid_multipliers=self.params.grid_multipliers,
            )
            selection_method = "auto"
        else:
            bucket_param = int(self.params.bucket_size)
            if bucket_param <= 0:
                raise ValueError("bucket_size must be positive")
            selection_method = "manual"

        bucket_df = ticks_to_volume_buckets(collapsed, bucket_param)
        if bucket_df.empty:
            return {
                "error": "No buckets generated",
                "metadata": {
                    "session_start": self.params.session_start,
                    "session_end": self.params.session_end,
                    "tz": self.params.tz,
                    "n_sessions": 0,
                    "n_buckets": 0,
                    "bucket_size": bucket_param,
                    "bucket_selection": selection_method,
                },
            }

        # Calculate metrics
        bucket_df = bucket_df.copy()
        bucket_df["ticker"] = symbol
        bucket_df["vpin"] = (
            bucket_df["Imbalance"].abs() / bucket_param
        ).rolling(window=self.params.vpin_window, min_periods=1).mean()

        # Buy pressure: AskShare > 0.5 means more buying (aggressors hitting ask)
        bucket_df["buy_pressure"] = bucket_df["AskShare"] - 0.5

        # Sell pressure: Calculate from bid side
        # First try BidShare if available, otherwise derive from AskShare
        if "BidShare" in bucket_df.columns:
            bucket_df["sell_pressure"] = bucket_df["BidShare"] - 0.5
        else:
            # BidShare = 1 - AskShare, so sell_pressure = (1 - AskShare) - 0.5 = 0.5 - AskShare
            bucket_df["sell_pressure"] = 0.5 - bucket_df["AskShare"]

        # Quote-based pressure (if quote data available)
        if "AskQuoteShare" in bucket_df.columns and bucket_df["AskQuoteShare"].notna().any():
            bucket_df["quote_buy_pressure"] = bucket_df["AskQuoteShare"] - 0.5

        local_ts_end = bucket_df["ts_end"].dt.tz_convert(self.params.tz)
        bucket_df["session_date"] = local_ts_end.dt.date
        bucket_df["weekday"] = local_ts_end.dt.day_name()
        bucket_df["weekday_number"] = local_ts_end.dt.dayofweek
        bucket_df["week_of_month"] = ((local_ts_end.dt.day - 1) // 7) + 1
        bucket_df["month"] = local_ts_end.dt.month
        bucket_df["quarter"] = local_ts_end.dt.quarter
        bucket_df["season"] = bucket_df["month"].map(_MONTH_TO_SEASON)
        bucket_df["clock_time"] = local_ts_end.dt.time

        pre_filter_sessions = int(bucket_df["session_date"].nunique())

        mask, filters_applied = _build_period_filter(
            bucket_df,
            self.params.month_filter,
            self.params.season_filter,
        )

        if mask is not None:
            bucket_df = bucket_df.loc[mask].copy()

        if bucket_df.empty:
            return {
                "error": "No buckets after applying seasonal filters",
                "metadata": {
                    "session_start": self.params.session_start,
                    "session_end": self.params.session_end,
                    "tz": self.params.tz,
                    "n_sessions": 0,
                    "n_buckets": 0,
                    "bucket_size": bucket_param,
                    "bucket_selection": selection_method,
                    "filters": filters_applied,
                    "days": 0,
                    "exploratory": True,
                    "n_sessions_pre_filter": pre_filter_sessions,
                },
            }

        # Pressure metrics - buy and sell pressure always available
        metrics = {
            "buy_pressure": bucket_df["buy_pressure"],
            "sell_pressure": bucket_df["sell_pressure"]
        }

        # Add quote-based pressure if available
        if "quote_buy_pressure" in bucket_df.columns:
            metrics["quote_buy_pressure"] = bucket_df["quote_buy_pressure"]

        # Intraday pressure table
        intraday = _intraday_pressure_table(bucket_df, metrics, symbol)

        # Event detection
        events_list: List[pd.DataFrame] = []
        for metric_name in metrics:
            z_col = f"{metric_name}_z"
            bucket_df[z_col] = _robust_daily_zscores(bucket_df, metric_name, "session_date")
            events_list.append(
                _event_runs(
                    bucket_df,
                    metric_name,
                    z_col,
                    self.params.threshold_z,
                    symbol,
                )
            )
        events = pd.concat(events_list, ignore_index=True) if events_list else pd.DataFrame()

        event_studies: Dict[str, Dict[str, object]] = {}
        if not bucket_df.empty and not events.empty:
            ts_end_index = pd.DatetimeIndex(bucket_df["ts_end"])
            ts_end_index = ts_end_index.tz_localize("UTC") if ts_end_index.tz is None else ts_end_index.tz_convert("UTC")

            for metric_name, series in metrics.items():
                metric_events = events[events["metric"] == metric_name]
                if metric_events.empty:
                    continue

                event_times = pd.to_datetime(metric_events["ts_end"])
                event_times = pd.DatetimeIndex(event_times)
                event_times = event_times.intersection(ts_end_index)
                if event_times.empty:
                    continue

                metric_frame = pd.DataFrame({"ret": series.astype(float).values}, index=ts_end_index)
                metric_frame = metric_frame.dropna()
                if metric_frame.empty:
                    continue

                spec = EventSpec(
                    name=f"{metric_name}_events",
                    event_fn=_event_flagger(event_times),
                    pre=int(self.params.event_pre_bars),
                    post=int(self.params.event_post_bars),
                )
                event_studies[metric_name] = run_event_study(metric_frame, spec, ret_col="ret")

        # Seasonality analysis
        n_sessions = int(bucket_df["session_date"].nunique())
        exploratory_flag = n_sessions < self.params.min_days

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
        df_weekly_peak = _weekly_peak_pressure(bucket_df, df_weekly, metrics, symbol)

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

        # Metadata
        metadata = {
            "session_start": self.params.session_start,
            "session_end": self.params.session_end,
            "tz": self.params.tz,
            "bucket_size": bucket_param,
            "bucket_selection": selection_method,
            "n_sessions": n_sessions,
            "days": n_sessions,
            "n_buckets": int(len(bucket_df)),
            "exploratory": n_sessions < self.params.min_days,
            "filters": filters_applied,
            "n_sessions_pre_filter": pre_filter_sessions,
        }

        # Select bucket columns
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
            "Imbalance",
            "ImbalanceFraction",
            "vpin",
            "buy_pressure",
            "sell_pressure",
        ]

        # Add optional columns if present
        if "AskQuoteShare" in bucket_df.columns:
            bucket_cols.append("AskQuoteShare")
        if "quote_buy_pressure" in bucket_df.columns:
            bucket_cols.append("quote_buy_pressure")
        if "AskQuoteVolume" in bucket_df.columns:
            bucket_cols.append("AskQuoteVolume")
        if "BidQuoteVolume" in bucket_df.columns:
            bucket_cols.append("BidQuoteVolume")
        if "BidShare" in bucket_df.columns:
            bucket_cols.append("BidShare")

        return {
            "df_buckets": bucket_df[bucket_cols].copy(),
            "df_intraday_pressure": intraday,
            "df_events": events,
            "event_study": event_studies,
            "df_weekly": df_weekly,
            "df_wom_weekday": df_wom,
            "df_weekly_peak_pressure": df_weekly_peak,
            "metadata": metadata,
        }

    def _process_symbol_from_buckets(
        self,
        symbol: str,
        bucket_df: pd.DataFrame,
        bucket_param: int,
        selection_method: str = "auto",
    ) -> Dict[str, object]:
        """
        Process a single symbol using pre-computed volume buckets.

        This method is used by run_scans() to reuse bucket data when multiple
        scans share the same session parameters.

        Args:
            symbol: Trading symbol
            bucket_df: Pre-computed bucket DataFrame
            bucket_param: Bucket size used

        Returns:
            Dictionary with analysis results
        """
        # Make a copy to avoid modifying shared data
        bucket_df = bucket_df.copy()

        # Calculate metrics
        bucket_df["ticker"] = symbol
        bucket_df["vpin"] = (
            bucket_df["Imbalance"].abs() / bucket_param
        ).rolling(window=self.params.vpin_window, min_periods=1).mean()

        # Buy pressure: AskShare > 0.5 means more buying (aggressors hitting ask)
        bucket_df["buy_pressure"] = bucket_df["AskShare"] - 0.5

        # Sell pressure: Calculate from bid side
        if "BidShare" in bucket_df.columns:
            bucket_df["sell_pressure"] = bucket_df["BidShare"] - 0.5
        else:
            bucket_df["sell_pressure"] = 0.5 - bucket_df["AskShare"]

        # Quote-based pressure (if quote data available)
        if "AskQuoteShare" in bucket_df.columns and bucket_df["AskQuoteShare"].notna().any():
            bucket_df["quote_buy_pressure"] = bucket_df["AskQuoteShare"] - 0.5

        # Add temporal columns
        local_ts_end = bucket_df["ts_end"].dt.tz_convert(self.params.tz)
        bucket_df["session_date"] = local_ts_end.dt.date
        bucket_df["weekday"] = local_ts_end.dt.day_name()
        bucket_df["weekday_number"] = local_ts_end.dt.dayofweek
        bucket_df["week_of_month"] = ((local_ts_end.dt.day - 1) // 7) + 1
        bucket_df["month"] = local_ts_end.dt.month
        bucket_df["quarter"] = local_ts_end.dt.quarter
        bucket_df["season"] = bucket_df["month"].map(_MONTH_TO_SEASON)
        bucket_df["clock_time"] = local_ts_end.dt.time

        pre_filter_sessions = int(bucket_df["session_date"].nunique())

        # Apply seasonal/temporal filters
        mask, filters_applied = _build_period_filter(
            bucket_df,
            self.params.month_filter,
            self.params.season_filter,
        )

        if mask is not None:
            bucket_df = bucket_df.loc[mask].copy()

        if bucket_df.empty:
            return {
                "error": "No buckets after applying seasonal filters",
                "metadata": {
                    "session_start": self.params.session_start,
                    "session_end": self.params.session_end,
                    "tz": self.params.tz,
                    "n_sessions": 0,
                    "n_buckets": 0,
                    "bucket_size": bucket_param,
                    "bucket_selection": selection_method,
                    "filters": filters_applied,
                    "days": 0,
                    "exploratory": True,
                    "n_sessions_pre_filter": pre_filter_sessions,
                },
            }

        # Pressure metrics
        metrics = {
            "buy_pressure": bucket_df["buy_pressure"],
            "sell_pressure": bucket_df["sell_pressure"]
        }

        if "quote_buy_pressure" in bucket_df.columns:
            metrics["quote_buy_pressure"] = bucket_df["quote_buy_pressure"]

        # Intraday pressure table
        intraday = _intraday_pressure_table(bucket_df, metrics, symbol)

        # Event detection
        events_list: List[pd.DataFrame] = []
        for metric_name in metrics:
            z_col = f"{metric_name}_z"
            bucket_df[z_col] = _robust_daily_zscores(bucket_df, metric_name, "session_date")
            events_list.append(
                _event_runs(
                    bucket_df,
                    metric_name,
                    z_col,
                    self.params.threshold_z,
                    symbol,
                )
            )
        events = pd.concat(events_list, ignore_index=True) if events_list else pd.DataFrame()

        # Seasonality analysis
        n_sessions = int(bucket_df["session_date"].nunique())
        exploratory_flag = n_sessions < self.params.min_days

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
        df_weekly_peak = _weekly_peak_pressure(bucket_df, df_weekly, metrics, symbol)

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

        # Metadata
        metadata = {
            "session_start": self.params.session_start,
            "session_end": self.params.session_end,
            "tz": self.params.tz,
            "bucket_size": bucket_param,
            "bucket_selection": selection_method,
            "n_sessions": n_sessions,
            "days": n_sessions,
            "n_buckets": int(len(bucket_df)),
            "exploratory": n_sessions < self.params.min_days,
            "filters": filters_applied,
            "n_sessions_pre_filter": pre_filter_sessions,
        }

        # Select bucket columns
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
            "Imbalance",
            "ImbalanceFraction",
            "vpin",
            "buy_pressure",
            "sell_pressure",
        ]

        # Add optional columns if present
        if "AskQuoteShare" in bucket_df.columns:
            bucket_cols.append("AskQuoteShare")
        if "quote_buy_pressure" in bucket_df.columns:
            bucket_cols.append("quote_buy_pressure")
        if "AskQuoteVolume" in bucket_df.columns:
            bucket_cols.append("AskQuoteVolume")
        if "BidQuoteVolume" in bucket_df.columns:
            bucket_cols.append("BidQuoteVolume")
        if "BidShare" in bucket_df.columns:
            bucket_cols.append("BidShare")

        return {
            "df_buckets": bucket_df[bucket_cols].copy(),
            "df_intraday_pressure": intraday,
            "df_events": events,
            "df_weekly": df_weekly,
            "df_wom_weekday": df_wom,
            "df_weekly_peak_pressure": df_weekly_peak,
            "metadata": metadata,
        }

    def write_to_hdf(self, hdf_path: Optional[Path | str] = None) -> Dict[str, str]:
        """
        Write scan results to HDF5 file.

        Creates organized HDF5 structure:
            /orderflow/{symbol}/buckets
            /orderflow/{symbol}/intraday_pressure
            /orderflow/{symbol}/events
            /orderflow/{symbol}/weekly
            /orderflow/{symbol}/wom_weekday
            /orderflow/{symbol}/metadata

        Args:
            hdf_path: Path to HDF5 file (uses self.hdf_path if not provided)

        Returns:
            Dictionary mapping symbols to write status

        Examples:
            scanner = OrderflowScanner(params, hdf_path='orderflow_results.h5')
            scanner.scan(tick_data)
            status = scanner.write_to_hdf()
        """
        if not self.results:
            raise ValueError("No results to write. Run scan() first.")

        target_path = Path(hdf_path) if hdf_path else self.hdf_path
        if target_path is None:
            raise ValueError("No HDF5 path provided")

        target_path = Path(target_path)

        # Create parent directory if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)

        write_status = {}

        with pd.HDFStore(target_path, mode='a') as store:
            for symbol, result in self.results.items():
                try:
                    if "error" in result:
                        write_status[symbol] = f"skipped: {result['error']}"
                        continue

                    base_key = f"orderflow/{symbol}"

                    # Write each DataFrame
                    for df_name in [
                        "df_buckets",
                        "df_intraday_pressure",
                        "df_events",
                        "df_weekly",
                        "df_wom_weekday",
                        "df_weekly_peak_pressure",
                    ]:
                        df = result.get(df_name)
                        if df is not None and not df.empty:
                            df = df.copy()

                            # Convert time columns to strings for HDF5 compatibility
                            for col in df.columns:
                                if hasattr(df[col], 'dtype') and df[col].dtype == 'object':
                                    if len(df[col]) > 0 and isinstance(df[col].iloc[0], time):
                                        df[col] = df[col].astype(str)

                            key = f"{base_key}/{df_name.replace('df_', '')}"
                            store.put(key, df, format='table', data_columns=True)

                    # Write metadata as series
                    if "metadata" in result:
                        metadata_series = pd.Series(result["metadata"])
                        store.put(f"{base_key}/metadata", metadata_series)

                    write_status[symbol] = "success"

                except Exception as e:
                    write_status[symbol] = f"error: {str(e)}"

        return write_status

    def read_from_hdf(self,
                     symbols: Optional[List[str]] = None,
                     hdf_path: Optional[Path | str] = None) -> Dict[str, Dict[str, object]]:
        """
        Read scan results from HDF5 file.

        Args:
            symbols: List of symbols to read (reads all if None)
            hdf_path: Path to HDF5 file (uses self.hdf_path if not provided)

        Returns:
            Dictionary mapping symbols to their results

        Examples:
            scanner = OrderflowScanner(params, hdf_path='orderflow_results.h5')
            results = scanner.read_from_hdf(symbols=['CL_F', 'NG_F'])
        """
        target_path = Path(hdf_path) if hdf_path else self.hdf_path
        if target_path is None:
            raise ValueError("No HDF5 path provided")

        target_path = Path(target_path)
        if not target_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {target_path}")

        results = {}

        with pd.HDFStore(target_path, mode='r') as store:
            # Get all symbols if not specified
            if symbols is None:
                all_keys = store.keys()
                symbols = list(set(k.split('/')[2] for k in all_keys if k.startswith('/orderflow/')))

            for symbol in symbols:
                try:
                    base_key = f"/orderflow/{symbol}"
                    result = {}

                    # Read DataFrames
                    df_names = {
                        "buckets": "df_buckets",
                        "intraday_pressure": "df_intraday_pressure",
                        "events": "df_events",
                        "weekly": "df_weekly",
                        "wom_weekday": "df_wom_weekday",
                        "weekly_peak_pressure": "df_weekly_peak_pressure",
                    }

                    for hdf_name, result_name in df_names.items():
                        key = f"{base_key}/{hdf_name}"
                        if key in store:
                            result[result_name] = store.get(key)

                    # Read metadata
                    metadata_key = f"{base_key}/metadata"
                    if metadata_key in store:
                        metadata_series = store.get(metadata_key)
                        result["metadata"] = metadata_series.to_dict()

                    results[symbol] = result

                except Exception as e:
                    results[symbol] = {"error": str(e)}

        return results

    def run_scans(
        self,
        scan_params: List[OrderflowParams],
        tick_data: Dict[str, pd.DataFrame],
        output_format: str = 'dict'
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        """
        Run multiple orderflow scans with different configurations.

        Similar to HistoricalScreener.run_screens(), this method allows you to run
        multiple scanning configurations (e.g., different sessions, seasons, or
        bucket sizes) and combine the results into a single output.

        **Optimization**: Volume buckets are pre-calculated per session window and
        reused across scans, even when running multiple sessions in parallel. This
        dramatically reduces the work required when experimenting with different
        seasonal filters or bucket configurations.

        Args:
            scan_params: List of OrderflowParams objects defining each scan to run
            tick_data: Dictionary mapping symbols to their tick DataFrames
            output_format: Output format ('dict' or 'dataframe')

        Returns:
            Dictionary mapping scan names to results:
            {scan_name: {symbol: results}}

        Examples:
            # Run scans for multiple sessions
            scanner = OrderflowScanner()
            scans = [
                OrderflowParams(
                    name='asia_session',
                    session_start='02:30',
                    session_end='08:30',
                    season_filter=['winter']
                ),
                OrderflowParams(
                    name='us_session',
                    session_start='09:30',
                    session_end='16:00',
                    season_filter=['winter']
                ),
                OrderflowParams(
                    name='full_day',
                    session_start='00:00',
                    session_end='23:59',
                    month_filter=[1, 2, 12]
                )
            ]
            results = scanner.run_scans(scans, tick_data)
            # Access: results['asia_session']['CL_F']

            # Run scans for different seasons (optimized - same session)
            scans = [
                OrderflowParams(
                    name='winter_orderflow',
                    session_start='09:30',
                    session_end='16:00',
                    season_filter=['winter'],
                    bucket_size='auto'
                ),
                OrderflowParams(
                    name='summer_orderflow',
                    session_start='09:30',
                    session_end='16:00',
                    season_filter=['summer'],
                    bucket_size='auto'
                )
            ]
            # Only computes volume buckets once since sessions are identical
            results = scanner.run_scans(scans, tick_data)

            # Auto-generate names if not provided
            scans = [
                OrderflowParams(session_start='02:30', session_end='08:30'),
                OrderflowParams(session_start='09:30', session_end='16:00')
            ]
            results = scanner.run_scans(scans, tick_data)
            # Names: 'scan_0230_0830', 'scan_0930_1600'
        """
        if not scan_params:
            raise ValueError("scan_params cannot be empty")

        session_groups: Dict[_SessionKey, List[Tuple[int, OrderflowParams]]] = defaultdict(list)
        session_order: List[_SessionKey] = []
        for index, params in enumerate(scan_params):
            session_key = _make_session_key(params)
            if session_key not in session_groups:
                session_order.append(session_key)
            session_groups[session_key].append((index, params))

        composite_results: Dict[str, Dict[str, Dict[str, object]]] = {}
        collapsed_cache: Dict[_SessionKey, Dict[str, pd.DataFrame]] = {}
        bucket_cache: Dict[Tuple[_SessionKey, _BucketKey], Dict[str, Tuple[pd.DataFrame, int, str]]] = {}

        for session_key in session_order:
            group = session_groups[session_key]
            collapsed = collapsed_cache.get(session_key)
            if collapsed is None:
                collapsed = _precompute_session_collapsed(tick_data, session_key)
                collapsed_cache[session_key] = collapsed

            for i, params in group:
                scan_name = params.name if params.name else self._generate_scan_name(params, i)
                temp_scanner = OrderflowScanner(params, hdf_path=self.hdf_path)
                bucket_key = _make_bucket_key(params)
                cache_key = (session_key, bucket_key)

                shared_bucket_data = bucket_cache.get(cache_key)
                if shared_bucket_data is None:
                    shared_bucket_data = _compute_bucket_data(collapsed, bucket_key)
                    bucket_cache[cache_key] = shared_bucket_data

                scan_result: Dict[str, Dict[str, object]] = {}
                for symbol, ticks in tick_data.items():
                    bucket_info = shared_bucket_data.get(symbol)
                    if bucket_info is not None:
                        bucket_df, bucket_param, selection_method = bucket_info
                        result = temp_scanner._process_symbol_from_buckets(
                            symbol,
                            bucket_df,
                            bucket_param,
                            selection_method,
                        )
                    else:
                        try:
                            if ticks is None or ticks.empty:
                                result = {"error": "No tick data"}
                            else:
                                normalized = _normalize_tick_timestamps(ticks)
                                result = temp_scanner._process_symbol(symbol, normalized)
                        except Exception as exc:
                            result = {"error": str(exc)}

                    scan_result[symbol] = result

                composite_results[scan_name] = scan_result

        # Store results in instance
        self.results = composite_results

        # Convert to requested format
        if output_format.lower() == 'dataframe':
            return self._composite_to_dataframe(composite_results)
        elif output_format.lower() == 'dict':
            return composite_results
        else:
            raise ValueError(f"output_format must be 'dict' or 'dataframe', got '{output_format}'")

    def _generate_scan_name(self, params: OrderflowParams, index: int) -> str:
        """Generate automatic scan name from parameters."""
        parts = []

        # Add session info
        start_str = params.session_start.replace(':', '')
        end_str = params.session_end.replace(':', '')
        parts.append(f"scan_{start_str}_{end_str}")

        # Add filter info
        if params.season_filter:
            # Get first season/quarter as identifier
            seasons, quarters, normalized = _normalize_season_filters(params.season_filter)
            if normalized:
                parts.append(normalized[0].lower())

        if params.month_filter:
            # Use first and last month as identifier
            months = sorted(params.month_filter)
            if len(months) == 1:
                parts.append(f"m{months[0]}")
            else:
                parts.append(f"m{months[0]}_m{months[-1]}")

        # Add bucket size if not auto
        if isinstance(params.bucket_size, int):
            parts.append(f"b{params.bucket_size}")

        # If we only have the scan prefix, add index
        if len(parts) == 1:
            parts.append(str(index))

        return '_'.join(parts)

    def _composite_to_dataframe(self, composite_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
        """
        Convert composite results dictionary to a flattened DataFrame.

        Args:
            composite_results: {scan_name: {symbol: results}}

        Returns:
            DataFrame with MultiIndex (scan_name, symbol)
        """
        records = []

        for scan_name, scan_results in composite_results.items():
            for symbol, result in scan_results.items():
                if 'error' in result:
                    record = {
                        'scan_name': scan_name,
                        'symbol': symbol,
                        'error': result['error']
                    }
                else:
                    # Extract key metadata
                    metadata = result.get('metadata', {})
                    record = {
                        'scan_name': scan_name,
                        'symbol': symbol,
                        'n_sessions': metadata.get('n_sessions', 0),
                        'n_buckets': metadata.get('n_buckets', 0),
                        'bucket_size': metadata.get('bucket_size'),
                        'bucket_selection': metadata.get('bucket_selection'),
                        'exploratory': metadata.get('exploratory', False),
                        'session_start': metadata.get('session_start'),
                        'session_end': metadata.get('session_end'),
                        'tz': metadata.get('tz'),
                    }

                    # Add filter info if present
                    if 'filters' in metadata:
                        record['filters'] = str(metadata['filters'])
                    if 'n_sessions_pre_filter' in metadata:
                        record['n_sessions_pre_filter'] = metadata['n_sessions_pre_filter']

                    # Count significant results
                    df_weekly = result.get('df_weekly')
                    if isinstance(df_weekly, pd.DataFrame) and not df_weekly.empty:
                        record['n_significant_weekly'] = df_weekly['sig_fdr_5pct'].sum()
                    else:
                        record['n_significant_weekly'] = 0

                    df_events = result.get('df_events')
                    if isinstance(df_events, pd.DataFrame):
                        record['n_events'] = len(df_events)
                    else:
                        record['n_events'] = 0

                records.append(record)

        df = pd.DataFrame(records)

        # Set MultiIndex
        if not df.empty:
            df = df.set_index(['scan_name', 'symbol'])

        return df


def _normalize_tick_timestamps(ticks: pd.DataFrame) -> pd.DataFrame:
    """Ensure tick data has a 'ts' column for timestamp and handle duplicates."""
    ticks = ticks.copy()

    # First, get or create the 'ts' column
    if "ts" not in ticks.columns:
        # Check for alternative timestamp column names
        if isinstance(ticks.index, pd.DatetimeIndex):
            ticks.insert(0, "ts", ticks.index)
        else:
            # Check common timestamp column names
            ts_candidates = ["timestamp", "datetime", "time", "date"]
            found = False
            for candidate in ts_candidates:
                if candidate in ticks.columns:
                    ticks.rename(columns={candidate: "ts"}, inplace=True)
                    found = True
                    break

            if not found:
                raise ValueError(
                    "Tick data must have a 'ts' column, DatetimeIndex, or one of: "
                    f"{', '.join(ts_candidates)}"
                )

    # Handle duplicate timestamps by aggregating
    if ticks["ts"].duplicated().any():
        # Group by timestamp and sum numeric columns (typical for tick data)
        numeric_cols = ticks.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Aggregate: sum volumes, keep first for non-numeric
            agg_dict = {col: 'sum' for col in numeric_cols}
            ticks = ticks.groupby("ts", as_index=False).agg(agg_dict)
        else:
            # If no numeric columns, just keep first occurrence
            ticks = ticks.drop_duplicates(subset=["ts"], keep="first")

    return ticks


def _determine_worker_count(task_count: int) -> int:
    max_workers = os.cpu_count() or 1
    if task_count <= 1 or max_workers <= 1:
        return 1
    return min(task_count, max_workers)


def _precompute_symbol_collapsed(args: Tuple[str, pd.DataFrame, _SessionKey]) -> Tuple[str, Optional[pd.DataFrame]]:
    symbol, ticks, session_key = args
    try:
        if ticks is None or ticks.empty:
            return symbol, None

        ticks = _normalize_tick_timestamps(ticks)
        filtered = filter_session_ticks(
            ticks,
            session_key.tz,
            session_key.session_start,
            session_key.session_end,
        )

        if filtered.empty:
            return symbol, None

        collapsed = _collapse_ticks(filtered)
        if collapsed.empty:
            return symbol, None

        return symbol, collapsed
    except Exception:
        return symbol, None


def _precompute_session_collapsed(
    tick_data: Dict[str, pd.DataFrame],
    session_key: _SessionKey,
) -> Dict[str, pd.DataFrame]:
    if not tick_data:
        return {}

    tasks = [
        (symbol, ticks, session_key)
        for symbol, ticks in tick_data.items()
    ]

    worker_count = _determine_worker_count(len(tasks))
    results: Dict[str, pd.DataFrame] = {}

    if worker_count == 1:
        for task in tasks:
            symbol, collapsed = _precompute_symbol_collapsed(task)
            if collapsed is not None:
                results[symbol] = collapsed
        return results

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_precompute_symbol_collapsed, task): task[0]
            for task in tasks
        }
        for future in as_completed(futures):
            symbol, collapsed = future.result()
            if collapsed is not None:
                results[symbol] = collapsed

    return results


def _compute_symbol_buckets(
    args: Tuple[str, pd.DataFrame, _BucketKey]
) -> Tuple[str, Optional[Tuple[pd.DataFrame, int, str]]]:
    symbol, collapsed, bucket_key = args
    try:
        if collapsed is None or collapsed.empty:
            return symbol, None

        if bucket_key.mode == "auto":
            bucket_param = auto_bucket_size(
                collapsed,
                cadence_target=bucket_key.cadence_target,
                grid_multipliers=bucket_key.grid_multipliers or (),
            )
            selection_method = "auto"
        else:
            bucket_param = bucket_key.bucket_size
            selection_method = "manual"

        if bucket_param is None:
            return symbol, None

        if int(bucket_param) <= 0:
            return symbol, None

        bucket_df = ticks_to_volume_buckets(collapsed, int(bucket_param))
        if bucket_df.empty:
            return symbol, None

        return symbol, (bucket_df, int(bucket_param), selection_method)
    except Exception:
        return symbol, None


def _compute_bucket_data(
    collapsed_data: Dict[str, pd.DataFrame],
    bucket_key: _BucketKey,
) -> Dict[str, Tuple[pd.DataFrame, int, str]]:
    if not collapsed_data:
        return {}

    tasks = [
        (symbol, collapsed, bucket_key)
        for symbol, collapsed in collapsed_data.items()
    ]

    worker_count = _determine_worker_count(len(tasks))
    results: Dict[str, Tuple[pd.DataFrame, int, str]] = {}

    if worker_count == 1:
        for task in tasks:
            symbol, bucket_info = _compute_symbol_buckets(task)
            if bucket_info is not None:
                results[symbol] = bucket_info
        return results

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_compute_symbol_buckets, task): task[0]
            for task in tasks
        }
        for future in as_completed(futures):
            symbol, bucket_info = future.result()
            if bucket_info is not None:
                results[symbol] = bucket_info

    return results


def orderflow_scan(
    tick_data: Dict[str, pd.DataFrame],
    params: OrderflowParams,
) -> Dict[str, Dict[str, object]]:
    """
    Scan orderflow patterns in tick data (convenience function).

    This is a wrapper around OrderflowScanner.scan() for backward compatibility.
    For more features (like HDF5 persistence), use OrderflowScanner directly.

    Args:
        tick_data: Dictionary mapping symbols to their tick DataFrames
        params: OrderflowParams configuration

    Returns:
        Dictionary mapping symbols to analysis results

    Examples:
        # Simple scan
        results = orderflow_scan(tick_data, params)

        # For HDF5 persistence, use the class:
        scanner = OrderflowScanner(params, hdf_path='results.h5')
        results = scanner.scan(tick_data)
        scanner.write_to_hdf()
    """
    scanner = OrderflowScanner(params)
    return scanner.scan(tick_data)


def _deprecated_orderflow_scan(
    tick_data: Dict[str, pd.DataFrame],
    params: OrderflowParams,
) -> Dict[str, Dict[str, object]]:
    """Old implementation - keeping for reference only."""
    results: Dict[str, Dict[str, object]] = {}
    session_start = _parse_time(params.session_start)
    session_end = _parse_time(params.session_end)

    for symbol, ticks in tick_data.items():
        try:
            if ticks is None or ticks.empty:
                results[symbol] = {"error": "No tick data"}
                continue

            # Normalize timestamp column
            ticks = _normalize_tick_timestamps(ticks)

        except Exception as exc:  # pragma: no cover - tick data normalization failure
            results[symbol] = {"error": str(exc)}
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


def _load_tick_data_dict(path_template: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load tick data for multiple symbols into a dictionary."""
    from pathlib import Path

    tick_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = Path(path_template.format(symbol=symbol))
        if not path.exists():
            print(f"Warning: No tick file found for {symbol} at {path}")
            continue
        try:
            if path.suffix.lower() == ".csv":
                tick_data[symbol] = pd.read_csv(path, index_col=0, parse_dates=True)
            elif path.suffix.lower() == ".parquet":
                tick_data[symbol] = pd.read_parquet(path)
            else:
                print(f"Warning: Unsupported tick format for {symbol}: {path.suffix}")
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
    return tick_data


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

    # Load tick data into dictionary
    tick_data = _load_tick_data_dict(args.ticks_path, args.symbols)
    if not tick_data:
        print("Error: No tick data could be loaded for any symbols")
        return

    scan_results = orderflow_scan(tick_data, params)
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
