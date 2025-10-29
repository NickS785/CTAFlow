from __future__ import annotations

import glob
import importlib
import importlib.util
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["SessionFirstHoursParams", "run_session_first_hours"]

logger = logging.getLogger("cta_session_first_hours")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

_SIERRAPY_SPEC = importlib.util.find_spec("SierraPy")
if _SIERRAPY_SPEC is not None:
    SierraPy = importlib.import_module("SierraPy")
else:  # pragma: no cover - optional dependency might be missing in tests
    SierraPy = None
_HAS_SIERRAPY = SierraPy is not None


@dataclass(slots=True)
class SessionFirstHoursParams:
    symbols: list[str]
    start_date: str | pd.Timestamp
    end_date: str | pd.Timestamp
    lookback_days: int = 20
    session_start_hhmm: str = "17:00"
    first_hours: int = 2
    bar_seconds: int = 60
    tz_display: str = "America/Chicago"
    scid_root: Optional[str] = None
    min_bars_in_window: int = 5


def _ensure_local_timestamp(value: str | pd.Timestamp, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def _load_intraday(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    bar_seconds: int,
    tz_display: str,
    scid_root: Optional[str],
) -> pd.DataFrame:
    """Load intraday bars using SierraPy when available or `.scid` files otherwise."""

    start_local = _ensure_local_timestamp(start, tz_display) - pd.Timedelta(days=1)
    end_local = _ensure_local_timestamp(end, tz_display) + pd.Timedelta(days=1)

    if _HAS_SIERRAPY:
        request = SierraPy.GraphDataRequest(  # type: ignore[attr-defined]
            symbol=symbol,
            start_datetime_utc=start_local.tz_convert("UTC").to_pydatetime(),
            end_datetime_utc=end_local.tz_convert("UTC").to_pydatetime(),
            bar_seconds=bar_seconds,
            include_bid_ask_vol=True,
            include_num_trades=True,
        )
        df = request.to_dataframe()
        df = df.rename(
            columns={
                "Datetime": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "NumberOfTrades": "num_trades",
                "BidVolume": "bid_vol",
                "AskVolume": "ask_vol",
            }
        )
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        df = df.loc[~df.index.duplicated(keep="last")]
        return df.tz_convert(tz_display)

    if scid_root is None:
        raise RuntimeError("SierraPy unavailable and scid_root not provided.")

    dtype = np.dtype(
        [
            ("Time", "<i8"),
            ("Open", "<f4"),
            ("High", "<f4"),
            ("Low", "<f4"),
            ("Close", "<f4"),
            ("NumTrades", "<i4"),
            ("Volume", "<i4"),
            ("BidVolume", "<i4"),
            ("AskVolume", "<i4"),
        ]
    )
    pattern = os.path.join(scid_root, f"{symbol}*.scid")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .scid files found for symbol {symbol!r}")

    arrays = []
    for path in paths:
        arr = np.fromfile(path, dtype=dtype)
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "num_trades", "bid_vol", "ask_vol"]
        )

    stacked = np.concatenate(arrays)
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(stacked["Time"], unit="s", utc=True),
            "open": stacked["Open"],
            "high": stacked["High"],
            "low": stacked["Low"],
            "close": stacked["Close"],
            "volume": stacked["Volume"].astype(float),
            "num_trades": stacked["NumTrades"].astype(float),
            "bid_vol": stacked["BidVolume"].astype(float),
            "ask_vol": stacked["AskVolume"].astype(float),
        }
    ).set_index("datetime").sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.tz_localize("UTC").tz_convert(tz_display)

    rule = f"{bar_seconds}s"
    ohlc = frame[["open", "high", "low", "close"]].resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    sums = frame[["volume", "num_trades", "bid_vol", "ask_vol"]].resample(rule).sum()
    out = pd.concat([ohlc, sums], axis=1).dropna(subset=["open", "close"])
    return out.loc[(out.index >= start_local) & (out.index <= end_local)]


def _session_labels(index: pd.DatetimeIndex, tz_display: str, session_start_hhmm: str) -> pd.DatetimeIndex:
    local = index.tz_convert(tz_display)
    hh, mm = map(int, session_start_hhmm.split(":"))
    session_open = local.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    session_dates = local.normalize()
    session_dates = session_dates.where(local >= session_open, session_dates - pd.Timedelta(days=1))
    return session_dates.tz_localize(None)


def _minute_since_open(local_ts: pd.DatetimeIndex, session_start_hhmm: str) -> np.ndarray:
    hh, mm = map(int, session_start_hhmm.split(":"))
    session_open = local_ts.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    delta = (local_ts - session_open) / pd.Timedelta(minutes=1)
    return delta.astype(int)


def _aggregate_first_hours(
    df: pd.DataFrame,
    tz_display: str,
    session_start_hhmm: str,
    first_hours: int,
    min_bars: int,
) -> tuple[pd.DataFrame, Dict[pd.Timestamp, np.ndarray]]:
    if df.empty:
        return pd.DataFrame(), {}

    df = df.sort_index()
    labels = _session_labels(df.index, tz_display, session_start_hhmm)
    local_index = df.index.tz_convert(tz_display)
    df = df.copy()
    df["__sess"] = labels.values
    df["__k"] = _minute_since_open(local_index, session_start_hhmm)

    window_limit = first_hours * 60
    rows: list[pd.Series] = []
    k_positions: Dict[pd.Timestamp, np.ndarray] = {}

    for session_label, group in df.groupby("__sess", sort=True):
        window = group.loc[(group["__k"] >= 0) & (group["__k"] < window_limit)]
        if len(window) < min_bars:
            continue

        log_close = np.log(window["close"].astype(float))
        returns = log_close.diff().dropna()
        realized_vol = float(math.sqrt(np.square(returns).sum()))

        row = pd.Series(
            {
                "open": float(window["open"].iloc[0]),
                "close": float(window["close"].iloc[-1]),
                "volume": float(window["volume"].sum()),
                "num_trades": float(window.get("num_trades", pd.Series(dtype=float)).sum()),
                "realized_vol": realized_vol,
            },
            name=pd.Timestamp(session_label),
        )
        rows.append(row)
        k_positions[pd.Timestamp(session_label)] = window["__k"].to_numpy(dtype=int)

    session_df = pd.DataFrame(rows).sort_index()
    if session_df.empty:
        return session_df, k_positions

    session_df["session_return"] = session_df["close"] / session_df["open"] - 1.0
    return session_df, k_positions


def _relative_volume_tod(
    intraday: pd.DataFrame,
    session_df: pd.DataFrame,
    k_positions: Dict[pd.Timestamp, np.ndarray],
    tz_display: str,
    session_start_hhmm: str,
    lookback_days: int,
) -> pd.Series:
    if session_df.empty:
        return pd.Series(dtype="float64")

    local_index = intraday.index.tz_convert(tz_display)
    intraday = intraday.copy()
    intraday["__sess"] = _session_labels(intraday.index, tz_display, session_start_hhmm).values
    intraday["__k"] = _minute_since_open(local_index, session_start_hhmm)
    intraday = intraday.loc[intraday["__k"] >= 0]

    grouped: Dict[pd.Timestamp, pd.Series] = {}
    for sess_label, group in intraday.groupby("__sess", sort=True):
        grouped[pd.Timestamp(sess_label)] = group.set_index("__k")["volume"].astype(float)

    if not grouped:
        return pd.Series(np.nan, index=session_df.index)

    ordered_sessions = sorted(grouped.keys())
    rel_values: Dict[pd.Timestamp, float] = {}
    lookback = max(1, lookback_days)

    for idx in session_df.index:
        ks = k_positions.get(idx)
        if ks is None or ks.size == 0:
            rel_values[idx] = np.nan
            continue

        hist_sessions = [s for s in ordered_sessions if s < idx][-lookback:]
        if not hist_sessions:
            rel_values[idx] = np.nan
            continue

        hist_frames = []
        for sess in hist_sessions:
            series = grouped.get(sess)
            if series is not None:
                hist_frames.append(series)
        if not hist_frames:
            rel_values[idx] = np.nan
            continue

        hist_df = pd.DataFrame(hist_frames).T
        baseline = hist_df.median(axis=1, skipna=True)
        if baseline.empty:
            rel_values[idx] = np.nan
            continue

        fallback = float(baseline.median(skipna=True)) if not baseline.dropna().empty else np.nan
        expected = baseline.reindex(ks).fillna(fallback).sum()

        actual_series = grouped.get(idx)
        actual = float(actual_series.reindex(ks).sum()) if actual_series is not None else 0.0
        if not expected or np.isnan(expected):
            rel_values[idx] = np.nan
        else:
            rel_values[idx] = actual / expected

    return pd.Series(rel_values, dtype="float64").reindex(session_df.index)


def run_session_first_hours(params: SessionFirstHoursParams) -> pd.DataFrame:
    """Run the session first-hours screener and return a wide DataFrame of metrics and ranks."""

    if not params.symbols:
        return pd.DataFrame()

    start_filter = pd.Timestamp(params.start_date).normalize().tz_localize(None)
    end_filter = pd.Timestamp(params.end_date).normalize().tz_localize(None)

    per_symbol = []

    for symbol in params.symbols:
        logger.info("[%s] loading intraday", symbol)
        intraday = _load_intraday(
            symbol,
            params.start_date,
            params.end_date,
            params.bar_seconds,
            params.tz_display,
            params.scid_root,
        )
        if intraday.empty:
            logger.warning("[%s] no intraday data loaded", symbol)
            continue

        logger.info("[%s] aggregating first %sh", symbol, params.first_hours)
        session_df, k_positions = _aggregate_first_hours(
            intraday,
            params.tz_display,
            params.session_start_hhmm,
            params.first_hours,
            params.min_bars_in_window,
        )
        if session_df.empty:
            logger.warning("[%s] no sessions after aggregation", symbol)
            continue

        session_df = session_df.loc[(session_df.index >= start_filter) & (session_df.index <= end_filter)]
        if session_df.empty:
            continue

        logger.info("[%s] computing vol-normalized return", symbol)
        window = max(1, params.lookback_days)
        min_periods = min(window, max(3, window // 2))
        rolling_vol = (
            session_df["session_return"].rolling(window, min_periods=min_periods).std().shift(1)
        )
        session_df["vol_norm_ret"] = session_df["session_return"] / rolling_vol.replace(0.0, np.nan)

        logger.info("[%s] computing relative volume vs time-of-day", symbol)
        filtered_positions = {idx: k_positions[idx] for idx in session_df.index if idx in k_positions}
        session_df["relative_volume_tod"] = _relative_volume_tod(
            intraday,
            session_df,
            filtered_positions,
            params.tz_display,
            params.session_start_hhmm,
            params.lookback_days,
        )

        metrics = session_df[["session_return", "realized_vol", "vol_norm_ret", "relative_volume_tod"]].copy()
        metrics.columns = pd.MultiIndex.from_product(
            [["momentum", "realized_vol", "vol_norm_ret", "relative_volume_tod"], [symbol]]
        )
        per_symbol.append(metrics)

    if not per_symbol:
        return pd.DataFrame()

    wide = pd.concat(per_symbol, axis=1).sort_index()
    for metric in ["momentum", "realized_vol", "vol_norm_ret", "relative_volume_tod"]:
        metric_slice = wide[metric]
        ranks = metric_slice.rank(axis=1, method="min", ascending=False)
        ranks.columns = pd.MultiIndex.from_product([[f"rank_{metric}"], metric_slice.columns])
        wide = pd.concat([wide, ranks], axis=1)

    return wide
