"""Volume bucket utilities for orderflow analysis."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = ["auto_bucket_size", "ticks_to_volume_buckets"]


_ASK_VOLUME_ALIASES: Tuple[str, ...] = (
    "AskVolume",
    "askVolume",
    "ask_volume",
    "buy_vol",
    "BuyVolume",
)
_BID_VOLUME_ALIASES: Tuple[str, ...] = (
    "BidVolume",
    "bidVolume",
    "bid_volume",
    "sell_vol",
    "SellVolume",
)


def _resolve_volume_column(df: pd.DataFrame, aliases: Sequence[str], label: str) -> str:
    for candidate in aliases:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing required column for {label}: expected one of {aliases}")


def _standardize_tick_volume_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise KeyError("Missing required tick column: 'ts'")

    ask_col = _resolve_volume_column(df, _ASK_VOLUME_ALIASES, "AskVolume")
    bid_col = _resolve_volume_column(df, _BID_VOLUME_ALIASES, "BidVolume")

    ticks = df.copy()
    ticks["AskVolume"] = ticks[ask_col].astype(float)
    ticks["BidVolume"] = ticks[bid_col].astype(float)
    return ticks


def auto_bucket_size(
    ticks_session: pd.DataFrame,
    cadence_target: int = 50,
    grid_multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
) -> int:
    """Select an optimal bucket size for the provided session ticks."""

    if ticks_session.empty:
        raise ValueError("Cannot determine bucket size on empty tick data")

    ticks = _standardize_tick_volume_columns(ticks_session)
    timestamps = pd.to_datetime(ticks["ts"])
    if timestamps.dt.tz is None:
        ticks["session_date"] = timestamps.dt.date
    else:
        ticks["session_date"] = timestamps.dt.tz_convert(None).dt.date
    ticks["total_vol"] = ticks["AskVolume"] + ticks["BidVolume"]

    daily_volume = ticks.groupby("session_date")["total_vol"].sum()
    median_volume = float(daily_volume.median())
    if median_volume <= 0:
        raise ValueError("Median daily volume must be positive for bucket tuning")

    base = median_volume / max(cadence_target, 1)
    candidates = sorted({max(int(round(base * mult)), 1) for mult in grid_multipliers})

    if not candidates:
        raise ValueError("No valid bucket size candidates were generated")

    target = float(cadence_target)
    best_bucket = candidates[0]
    best_score = float("inf")
    best_variance = float("inf")

    for candidate in candidates:
        buckets_per_day = np.ceil(daily_volume / candidate)
        median_cadence = float(np.median(buckets_per_day))
        score = abs(median_cadence - target)
        variance = float(np.var(buckets_per_day))
        if (score < best_score) or (np.isclose(score, best_score) and variance < best_variance):
            best_score = score
            best_variance = variance
            best_bucket = int(candidate)

    return best_bucket


def ticks_to_volume_buckets(ticks_session: pd.DataFrame, bucket_size: int) -> pd.DataFrame:
    """Aggregate ticks into volume buckets of size ``bucket_size``."""

    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    if ticks_session.empty:
        columns = [
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
        ]
        return pd.DataFrame(columns=columns)

    ticks = _standardize_tick_volume_columns(ticks_session)
    ticks = ticks.sort_values("ts").reset_index(drop=True)
    ticks["ts"] = pd.to_datetime(ticks["ts"])

    total_volume = ticks["AskVolume"] + ticks["BidVolume"]
    ticks["cum_vol"] = total_volume.cumsum()
    ticks["bucket"] = np.floor((ticks["cum_vol"] - 1e-9) / bucket_size).astype(int)

    grouped = ticks.groupby("bucket")
    ts_start = grouped["ts"].min()
    ts_end = grouped["ts"].max()
    ask_traded = grouped["AskVolume"].sum()
    bid_traded = grouped["BidVolume"].sum()
    n_ticks = grouped.size()

    aggregated = pd.DataFrame({
        "bucket": ts_start.index,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "AskVolume": ask_traded,
        "BidVolume": bid_traded,
        "n_ticks": n_ticks,
    }).reset_index(drop=True)

    if "ask_vol" in ticks.columns:
        aggregated["AskQuoteVolume"] = grouped["ask_vol"].sum()
    if "bid_vol" in ticks.columns:
        aggregated["BidQuoteVolume"] = grouped["bid_vol"].sum()

    aggregated = aggregated.sort_values("bucket").reset_index(drop=True)
    aggregated["TotalVolume"] = aggregated["AskVolume"] + aggregated["BidVolume"]
    aggregated["AskShare"] = np.where(
        aggregated["TotalVolume"] > 0,
        aggregated["AskVolume"] / aggregated["TotalVolume"],
        np.nan,
    )

    if {"AskQuoteVolume", "BidQuoteVolume"}.issubset(aggregated.columns):
        quote_total = aggregated["AskQuoteVolume"] + aggregated["BidQuoteVolume"]
        aggregated["AskQuoteShare"] = np.where(
            quote_total > 0, aggregated["AskQuoteVolume"] / quote_total, np.nan
        )
    elif "AskQuoteVolume" in aggregated.columns:
        aggregated["AskQuoteShare"] = np.nan
    elif "BidQuoteVolume" in aggregated.columns:
        aggregated["AskQuoteShare"] = np.nan
    else:
        aggregated["AskQuoteShare"] = np.nan

    aggregated["Imbalance"] = aggregated["AskVolume"] - aggregated["BidVolume"]
    aggregated["ImbalanceFraction"] = np.where(
        aggregated["TotalVolume"] > 0,
        aggregated["Imbalance"] / aggregated["TotalVolume"],
        np.nan,
    )

    return aggregated[
        [
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
        ]
        + (
            ["AskQuoteVolume"]
            if "AskQuoteVolume" in aggregated.columns
            else []
        )
        + (
            ["BidQuoteVolume"]
            if "BidQuoteVolume" in aggregated.columns
            else []
        )
    ]
