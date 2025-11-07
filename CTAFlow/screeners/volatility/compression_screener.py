#!/usr/bin/env python3
"""
Vol Compression → Breakout Screener
-----------------------------------
Screens OHLCV time series for short-term vs long-term volatility compression
(ATR and close-close standard deviation) followed by either price breakouts
or volatility expansion.

Two modes:
- historical: emit all trigger dates per symbol (for backtesting)
- current: emit only most recent triggers within a lookback window

Input data:
- CSV files per symbol in a folder OR a single CSV with a `symbol` column.
  Expected columns (case-insensitive ok): Date, Open, High, Low, Close, Volume
  Date is parsed to datetime and sorted ascending.

Example usage:
  python vol_screener.py --data-dir ./data/daily --symbols CL,NG,GC \
    --mode historical --short 5 --long 30 --hl-lookback 20 \
    --comp-atr-max 0.60 --comp-sd-max 0.60 --min-comp-days 5 \
    --exp-atr-min 1.25 --exp-sd-min 1.25 --post-window 10 \
    --out screener_results.csv

Author: Quant Commodities Engineer (ACSIL + Python)
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------- Utility / IO ----------------------------------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c.lower(): c for c in df.columns}
    # Try to map common variants to standard names
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("date",):
            rename[c] = "Date"
        elif cl in ("open",):
            rename[c] = "Open"
        elif cl in ("high",):
            rename[c] = "High"
        elif cl in ("low",):
            rename[c] = "Low"
        elif cl in ("close", "adjclose", "adj_close", "settle", "settlement"):
            rename[c] = "Close"
        elif cl in ("volume", "vol"):
            rename[c] = "Volume"
        elif cl in ("symbol", "ticker", "root"):
            rename[c] = "symbol"
    out = df.rename(columns=rename)
    return out

def _read_csv_any(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_cols(df)
    if "Date" not in df.columns:
        raise ValueError(f"{path} missing Date column")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# ------------------------ Volatility Calculations ----------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Wilder True Range using (High, Low, prev Close).
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int, method: str = "sma") -> pd.Series:
    """
    Average True Range over `length`. method: 'sma' or 'ema' (Wilder-like).
    """
    tr = true_range(df)
    if method == "ema":
        # Wilder's ATR uses an EMA with alpha=1/length
        return tr.ewm(alpha=1/length, adjust=False).mean()
    else:
        return tr.rolling(length, min_periods=length).mean()

def realized_sd(df: pd.DataFrame, length: int, log: bool = True) -> pd.Series:
    """
    Rolling standard deviation of returns over `length`. Close-to-close.
    (No annualization inside the screener; keep ratios comparable.)
    """
    close = df["Close"].astype(float)
    if log:
        ret = np.log(close).diff()
    else:
        ret = close.pct_change()
    return ret.rolling(length, min_periods=length).std()

def rolling_high(df: pd.DataFrame, length: int) -> pd.Series:
    return df["High"].rolling(length, min_periods=length).max()

def rolling_low(df: pd.DataFrame, length: int) -> pd.Series:
    return df["Low"].rolling(length, min_periods=length).min()

# ----------------------------- Parameters ------------------------------------

@dataclass
class Params:
    short: int = 5
    long: int = 30
    atr_method: str = "sma"  # 'sma' or 'ema'
    hl_lookback: int = 20    # Donchian window for breakouts
    comp_atr_max: float = 0.60
    comp_sd_max: float = 0.60
    min_comp_days: int = 5
    exp_atr_min: float = 1.25
    exp_sd_min: float = 1.25
    post_window: int = 10    # max days after compression end to accept a trigger
    current_window: int = 5  # only for mode=current: show triggers within last N days
    min_long_vol: float = 1e-8  # guard vs divide-by-zero for degenerate data

# ----------------------------- Core Logic ------------------------------------
def _ensure_datetime_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Missing '{col}' column")
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def compute_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:

    out = df.copy()
    out = _ensure_datetime_col(out, "Date")
    out["ATR_S"] = atr(out, p.short, p.atr_method)
    out["ATR_L"] = atr(out, p.long, p.atr_method)
    out["SD_S"] = realized_sd(out, p.short)
    out["SD_L"] = realized_sd(out, p.long)
    # Ratios (guard against tiny long values)
    out["ATR_ratio"] = out["ATR_S"] / out["ATR_L"].clip(lower=p.min_long_vol)
    out["SD_ratio"]  = out["SD_S"] / out["SD_L"].clip(lower=p.min_long_vol)

    # Compression condition
    out["compress"] = (out["ATR_ratio"] <= p.comp_atr_max) & (out["SD_ratio"] <= p.comp_sd_max)

    # Label contiguous compression runs & their lengths
    grp_id = (out["compress"] != out["compress"].shift(1, fill_value=False)).cumsum()
    # Count number of True in each group; False groups get 0
    out["_grp"] = grp_id
    out["_grp_true_count"] = out["compress"].astype(int).groupby(out["_grp"]).transform("sum")
    out["comp_run_len"] = np.where(out["compress"], out["_grp_true_count"], 0)

    # Compression end marker (last day of a True run)
    out["comp_end"] = out["compress"] & (~out["compress"].shift(-1, fill_value=False))
    out["valid_comp_end"] = out["comp_end"] & (out["comp_run_len"] >= p.min_comp_days)

    # Track "days since last valid compression end"
    last_valid_end_pos = np.where(out["valid_comp_end"], np.arange(len(out)), np.nan)
    out["_last_end_pos"] = pd.Series(last_valid_end_pos).ffill().to_numpy()
    out["days_since_comp_end"] = (np.arange(len(out)) - out["_last_end_pos"]).astype("float")
    out["has_prior_comp"] = ~np.isnan(out["_last_end_pos"])
    out["within_post_window"] = out["has_prior_comp"] & (out["days_since_comp_end"] >= 0) & (out["days_since_comp_end"] <= p.post_window)

    # Breakouts (Donchian)
    out["HH"] = rolling_high(out, p.hl_lookback)
    out["LL"] = rolling_low(out, p.hl_lookback)
    out["breakout_up"] = out["Close"] > out["HH"].shift(1)  # confirm vs prior window
    out["breakout_dn"] = out["Close"] < out["LL"].shift(1)

    # Vol expansion trigger
    out["vol_expand"] = (out["ATR_ratio"] >= p.exp_atr_min) | (out["SD_ratio"] >= p.exp_sd_min)

    # Trigger = (breakout OR vol_expand) within post-window after a valid compression
    out["trigger"] = out["within_post_window"] & (out["breakout_up"] | out["breakout_dn"] | out["vol_expand"])

    # Keep only the first trigger per compression window
    # Identify compression windows by the position of the last valid end; first trigger is the minimum index per window
    out["_window_id"] = out["_last_end_pos"]
    # Rank triggers within the same window
    out["_first_in_window"] = False
    wids = out.loc[out["trigger"], "_window_id"].dropna().unique()
    for wid in wids:
        mask = (out["trigger"]) & (out["_window_id"] == wid)
        if mask.any():
            first_idx = out.index[mask][0]
            out.loc[first_idx, "_first_in_window"] = True
    out["trigger_first"] = out["trigger"] & out["_first_in_window"]

    # Clean temp columns
    out.drop(columns=["_grp", "_grp_true_count", "_last_end_pos", "_window_id", "_first_in_window"], inplace=True)

    return out

def summarize_triggers(symbol: str, ind: pd.DataFrame, p: Params) -> pd.DataFrame:
    cols = ["Date","Close","ATR_S","ATR_L","ATR_ratio","SD_S","SD_L","SD_ratio",
            "comp_run_len","days_since_comp_end","breakout_up","breakout_dn","vol_expand","trigger_first"]
    tmp = ind.loc[ind["trigger_first"], cols].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["symbol"] + cols)
    tmp.insert(0, "symbol", symbol)
    # Direction label
    direction = np.where(tmp["breakout_up"], "UP",
                 np.where(tmp["breakout_dn"], "DOWN",
                 np.where(tmp["vol_expand"], "VOL_EXPAND", "UNKNOWN")))
    tmp["direction"] = direction
    return tmp

def run_historical(p: Params, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = []
    for sym, df in data.items():
        ind = compute_indicators(df, p)
        res = summarize_triggers(sym, ind, p)
        if not res.empty:
            results.append(res)
    if results:
        out = pd.concat(results, ignore_index=True)
        return out.sort_values(["Date","symbol"]).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=[
            "symbol","Date","Close","ATR_S","ATR_L","ATR_ratio","SD_S","SD_L","SD_ratio",
            "comp_run_len","days_since_comp_end","breakout_up","breakout_dn","vol_expand","trigger_first","direction"
        ])

def run_current(p: Params, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return last p.current_window days of triggers (if any) per symbol.
    """
    rows = []
    for sym, df in data.items():
        ind = compute_indicators(df, p)
        if ind.empty:
            continue
        last_date = ind["Date"].iloc[-1]
        cutoff = last_date - pd.Timedelta(days=p.current_window*2)  # buffer for weekends/holidays
        mask = ind["trigger_first"] & (ind["Date"] >= cutoff)
        cols = ["Date","Close","ATR_S","ATR_L","ATR_ratio","SD_S","SD_L","SD_ratio",
                "comp_run_len","days_since_comp_end","breakout_up","breakout_dn","vol_expand","trigger_first"]
        tmp = ind.loc[mask, cols].copy()
        if tmp.empty:
            continue
        tmp.insert(0, "symbol", sym)
        direction = np.where(tmp["breakout_up"], "UP",
                     np.where(tmp["breakout_dn"], "DOWN",
                     np.where(tmp["vol_expand"], "VOL_EXPAND", "UNKNOWN")))
        tmp["direction"] = direction
        rows.append(tmp)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        # Keep the most recent row per symbol if many
        out = out.sort_values(["symbol","Date"]).groupby("symbol").tail(1).reset_index(drop=True)
        return out.sort_values(["Date","symbol"]).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=[
            "symbol","Date","Close","ATR_S","ATR_L","ATR_ratio","SD_S","SD_L","SD_ratio",
            "comp_run_len","days_since_comp_end","breakout_up","breakout_dn","vol_expand","trigger_first","direction"
        ])
