"""
Tools for analysing futures returns around scheduled data releases.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from zoneinfo import ZoneInfo

from .screener_types import SCREEN_EVENT
from .params import EventParams


@dataclass
class DataReleaseScanResult:
    """Container for event-screener outputs for a single symbol and event code."""

    events: pd.DataFrame
    summary: pd.DataFrame
    patterns: List[dict]


def _ensure_tz_aware(ts: pd.Series, tz: str) -> pd.Series:
    """Ensure a Series of timestamps is timezone-aware in the given tz."""

    if not len(ts):
        return ts
    if ts.dt.tz is None:
        return ts.dt.tz_localize(ZoneInfo(tz))
    return ts.dt.tz_convert(ZoneInfo(tz))


def _compute_log_return(p1: float, p0: float) -> float:
    if p1 <= 0 or p0 <= 0:
        return np.nan
    return float(np.log(p1 / p0))


def data_release_scan(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    params: EventParams,
    *,
    symbol: str,
    instrument_tz: str,
) -> DataReleaseScanResult:
    """
    Compute event-window and multi-horizon returns around scheduled data releases.
    """

    if "ts" in bars.columns:
        bar_ts = pd.to_datetime(bars["ts"])
    else:
        bar_ts = pd.to_datetime(bars.index)

    bar_ts = _ensure_tz_aware(bar_ts, instrument_tz)
    bars = bars.copy()
    bars["ts"] = bar_ts
    bars = bars.sort_values("ts").reset_index(drop=True)

    events = events.copy()
    events["release_ts"] = pd.to_datetime(events["release_ts"])
    events["release_ts"] = _ensure_tz_aware(events["release_ts"], instrument_tz)

    bars["session_date"] = bars["ts"].dt.date
    daily_close = (
        bars.sort_values("ts")
        .groupby("session_date")["close"]
        .last()
        .rename("close")
    )

    event_rows: List[Dict] = []

    pre_delta = timedelta(minutes=params.event_window_pre_minutes)
    post_delta = timedelta(minutes=params.event_window_post_minutes)

    for _, ev in events.iterrows():
        rel_ts = ev["release_ts"]

        pre_mask = bars["ts"] <= rel_ts - pre_delta
        post_mask = bars["ts"] <= rel_ts + post_delta
        if not pre_mask.any() or not post_mask.any():
            continue

        pre_bar = bars.loc[pre_mask].iloc[-1]
        post_bar = bars.loc[post_mask].iloc[-1]

        event_date = pre_bar["session_date"]
        if event_date not in daily_close.index:
            continue
        close_T0 = float(daily_close.loc[event_date])

        close_T1 = np.nan
        if params.include_t1_close:
            next_dates = [d for d in daily_close.index if d > event_date]
            if next_dates:
                close_T1 = float(daily_close.loc[next_dates[0]])

        extra_closes: Dict[int, float] = {}
        if params.extra_daily_horizons:
            sorted_dates = sorted(daily_close.index)
            idx = sorted_dates.index(event_date)
            for k in params.extra_daily_horizons:
                j = idx + k
                if 0 <= j < len(sorted_dates):
                    extra_closes[k] = float(daily_close.loc[sorted_dates[j]])
                else:
                    extra_closes[k] = np.nan

        pre_close = float(pre_bar["close"])
        post_close = float(post_bar["close"])
        r_event = _compute_log_return(post_close, pre_close)
        r_T0 = _compute_log_return(close_T0, post_close) if close_T0 and post_close else np.nan
        r_T1 = (
            _compute_log_return(close_T1, close_T0)
            if params.include_t1_close and close_T1 and close_T0
            else np.nan
        )

        row = {
            "symbol": symbol,
            "event_code": ev.get("event_code", params.event_code),
            "release_ts": rel_ts,
            "session_date": event_date,
            "pre_ts": pre_bar["ts"],
            "post_ts": post_bar["ts"],
            "pre_close": pre_close,
            "post_close": post_close,
            "close_T0": close_T0,
            "close_T1": close_T1,
            "r_event": r_event,
            "r_T0": r_T0,
            "r_T1": r_T1,
        }

        for k, c_k in extra_closes.items():
            row[f"close_T{k}"] = c_k
            if c_k and close_T0:
                row[f"r_T{k}"] = _compute_log_return(c_k, close_T0)
            else:
                row[f"r_T{k}"] = np.nan

        surprise = np.nan
        if (
            params.value_col
            and params.value_col in ev
            and params.consensus_col
            and params.consensus_col in ev
        ):
            actual = ev[params.value_col]
            cons = ev[params.consensus_col]
            raw = actual - cons
            if params.surprise_mode == "diff":
                surprise = raw
            elif params.surprise_mode == "pct":
                surprise = raw / abs(cons) if cons not in (0, None) else np.nan
            elif params.surprise_mode == "z":
                stdev = ev.get("stdev", np.nan)
                surprise = raw / stdev if stdev not in (0, None, np.nan) else np.nan
        row["surprise"] = surprise

        event_rows.append(row)

    events_df = pd.DataFrame(event_rows)
    if events_df.empty:
        return DataReleaseScanResult(events=events_df, summary=pd.DataFrame(), patterns=[])

    grp = events_df.groupby(["symbol", "event_code"])
    summary_rows: List[Dict] = []
    patterns: List[Dict] = []

    for (sym, code), g in grp:
        n = len(g)
        if n < params.min_events:
            continue

        def _mean_t(x: pd.Series) -> Tuple[float, float]:
            x = x.replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < 3:
                return np.nan, np.nan
            m = float(x.mean())
            s = float(x.std(ddof=1))
            if s == 0:
                return m, np.nan
            t = m / (s / np.sqrt(len(x)))
            p = 2 * (1 - stats.t.cdf(abs(t), df=len(x) - 1))
            return m, p

        mean_r_event, p_event = _mean_t(g["r_event"])
        mean_r_T0, p_T0 = _mean_t(g["r_T0"])
        mean_r_T1, p_T1 = _mean_t(g["r_T1"]) if params.include_t1_close else (np.nan, np.nan)

        def _corr(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
            df = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < 5:
                return np.nan, np.nan
            r, p = stats.pearsonr(df["a"], df["b"])
            return float(r), float(p)

        rho_e_T0, p_rho_e_T0 = _corr(g["r_event"], g["r_T0"])
        rho_e_T1, p_rho_e_T1 = _corr(g["r_event"], g["r_T1"]) if params.include_t1_close else (np.nan, np.nan)

        rho_s_event, p_s_event = (np.nan, np.nan)
        if "surprise" in g.columns and g["surprise"].notna().any():
            rho_s_event, p_s_event = _corr(g["surprise"], g["r_event"])

        summary_row = {
            "symbol": sym,
            "event_code": code,
            "n_events": n,
            "mean_r_event": mean_r_event,
            "p_event": p_event,
            "mean_r_T0": mean_r_T0,
            "p_T0": p_T0,
            "mean_r_T1": mean_r_T1,
            "p_T1": p_T1,
            "rho_event_T0": rho_e_T0,
            "p_rho_event_T0": p_rho_e_T0,
            "rho_event_T1": rho_e_T1,
            "p_rho_event_T1": p_rho_e_T1,
            "rho_surprise_event": rho_s_event,
            "p_rho_surprise_event": p_s_event,
        }
        summary_rows.append(summary_row)

        if not np.isnan(mean_r_event) and not np.isnan(mean_r_T0) and not np.isnan(rho_e_T0):
            if abs(rho_e_T0) >= params.corr_threshold:
                pattern_type = "event_trend_T0" if rho_e_T0 > 0 else "event_reversal_T0"
                gate_direction = np.sign(mean_r_event) if rho_e_T0 > 0 else -np.sign(mean_r_event)
                patterns.append(
                    {
                        "symbol": sym,
                        "screen_type": SCREEN_EVENT,
                        "pattern_type": pattern_type,
                        "event_code": code,
                        "gate_direction": int(gate_direction) if gate_direction != 0 else 0,
                        "strength": abs(rho_e_T0),
                        "description": f"{code} {pattern_type} from event-window to T0 close",
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    return DataReleaseScanResult(events=events_df, summary=summary_df, patterns=patterns)
