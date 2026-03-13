"""
Event Trading Prep — feature builder and dataset for TFT-based event trading.

Aligns 5min OHLCV bars with event features (from EventExtractor), technical
indicators, temporal features, and macro features.  Produces PyTorch
DataLoaders with explicit TFT feature categorization (static, known_future,
past_observed).

Two modes:
  - "post_only": Only bars on event days, after the release
  - "continuous": All active-session bars, event features forward-filled

Usage
-----
    from CTAFlow.models.prep.event_trading import EventTradingPrep

    prep = EventTradingPrep(tickers=["CL"], mode="continuous")
    prep.load_ticker("CL", root_dir="/data", event_stats=stats_df)
    train_loader, val_loader = prep.get_loaders(
        val_cutoff_date=date(2024, 1, 1),
        batch_size=32,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from CTAFlow.data.datasets.tft import build_ticker_registry
from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df
from CTAFlow.screeners.event_presets import (
    ALL_EVENT_DEFS,
    TICKER_EVENT_MAP,
    EventDefinition,
    event_release_dt_for_date,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Event type -> integer ID for embedding (0 = no event / padding)
_EVENT_TYPE_IDS: Dict[str, int] = {code: i + 1 for i, code in enumerate(sorted(ALL_EVENT_DEFS.keys()))}
_EVENT_TYPE_IDS["none"] = 0
N_EVENT_TYPES = len(_EVENT_TYPE_IDS)

# Feature lists (populated after engineering)
_TECH_FEATURES = [
    "macd_hist", "rsi_scaled", "dist_sma_20", "dist_sma_50", "dist_sma_200",
    "atr_norm", "rv_1d", "rv_5d", "rv_20d",
]
_EVENT_FEATURES = [
    "event_surprise", "event_post_max_return", "event_post_max_dd",
    "event_post_volume", "event_vol_lowest_q", "event_vol_highest_q",
]
_TEMPORAL_KNOWN = [
    "month", "dow", "doy_sin", "doy_cos", "tod_sin", "tod_cos",
    "days_until_next_event", "next_event_type_id",
    "is_event_day", "bars_since_release",
]
_INTRADAY_FEATURES = ["log_ret", "vol_scaled_ret", "vwap_dist"]


# ---------------------------------------------------------------------------
# Technical indicator helpers (stationary, no lookahead)
# ---------------------------------------------------------------------------

def _add_technicals(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add daily technical indicators to an intraday DataFrame.

    All daily features are shift(1) then forward-filled to intraday bars
    to prevent lookahead.
    """
    close = df[price_col].astype(float)

    # Daily close: last value per date
    df["_date"] = df.index.normalize()
    daily_close = close.groupby(df["_date"]).last()

    # --- MACD ---
    ema_12 = daily_close.ewm(span=12, adjust=False).mean()
    ema_26 = daily_close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # --- RSI 14 ---
    delta = daily_close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # --- SMA distances ---
    sma_20 = daily_close.rolling(20).mean()
    sma_50 = daily_close.rolling(50).mean()
    sma_200 = daily_close.rolling(200).mean()
    dist_sma_20 = (daily_close - sma_20) / daily_close
    dist_sma_50 = (daily_close - sma_50) / daily_close
    dist_sma_200 = (daily_close - sma_200) / daily_close

    # --- ATR normalized ---
    # Approximate from daily close (no High/Low at daily level here)
    daily_ret = daily_close.pct_change().abs()
    atr_20 = daily_ret.rolling(20).mean()

    # --- Realized volatility ---
    log_ret_daily = np.log(daily_close / daily_close.shift(1))
    rv_1d = log_ret_daily.abs()
    rv_5d = log_ret_daily.rolling(5).std()
    rv_20d = log_ret_daily.rolling(20).std()

    # Build daily feature df, shift(1) to avoid lookahead
    daily_feats = pd.DataFrame({
        "macd_hist": macd_hist,
        "rsi_scaled": (rsi - 50.0) / 10.0,  # centered, ~[-5, 5]
        "dist_sma_20": dist_sma_20,
        "dist_sma_50": dist_sma_50,
        "dist_sma_200": dist_sma_200,
        "atr_norm": atr_20,
        "rv_1d": rv_1d,
        "rv_5d": rv_5d,
        "rv_20d": rv_20d,
    }, index=daily_close.index).shift(1)

    # Map to intraday via date
    for col in daily_feats.columns:
        mapped = df["_date"].map(daily_feats[col])
        df[col] = mapped.values

    df.drop(columns=["_date"], inplace=True)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical temporal features (all known future)."""
    idx = df.index

    df["month"] = idx.month
    df["dow"] = idx.dayofweek

    # Day of year cyclical
    doy = idx.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Time of day cyclical
    minutes = idx.hour * 60 + idx.minute
    total_minutes = 24 * 60
    df["tod_sin"] = np.sin(2 * np.pi * minutes / total_minutes)
    df["tod_cos"] = np.cos(2 * np.pi * minutes / total_minutes)

    return df


def _add_intraday_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add intraday return and VWAP features."""
    close = df[price_col].astype(float)

    # Log return
    df["log_ret"] = np.log(close / close.shift(1))

    # Vol-scaled return
    ewm_vol = df["log_ret"].ewm(span=63).std()
    df["vol_scaled_ret"] = df["log_ret"] / ewm_vol.clip(lower=1e-8)

    # VWAP distance (rolling 60-bar ~= 5h at 5min)
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
    elif "TotalVolume" in df.columns:
        vol = df["TotalVolume"].astype(float)
    else:
        vol = pd.Series(1.0, index=df.index)

    tp = close  # simplified: just close
    vwap = (tp * vol).rolling(60, min_periods=1).sum() / vol.rolling(60, min_periods=1).sum().replace(0, np.nan)
    df["vwap_dist"] = (close - vwap) / vwap

    return df


# ---------------------------------------------------------------------------
# Event alignment helpers
# ---------------------------------------------------------------------------

def _align_event_features(
    df: pd.DataFrame,
    event_stats: pd.DataFrame,
    event_dates_by_code: Dict[str, List[date]],
    tz: str,
    mode: str,
) -> pd.DataFrame:
    """Align event_stats features to intraday bars.

    - On event days: features become available at release time
    - In continuous mode: features forward-fill until next event
    - Before release on event day: features are 0
    """
    # Initialize event columns
    for col in _EVENT_FEATURES:
        df[col] = 0.0
    df["is_event_day"] = 0
    df["bars_since_release"] = -1
    df["days_until_next_event"] = 999
    df["next_event_type_id"] = 0
    df["days_since_last_event"] = 999
    df["last_event_type_id"] = 0

    if event_stats.empty:
        return df

    # Build sorted list of all events with their release times
    all_events: List[Tuple[pd.Timestamp, str, dict]] = []
    for event_code, dates in event_dates_by_code.items():
        event_def = ALL_EVENT_DEFS.get(event_code)
        if event_def is None:
            continue
        for dt in dates:
            release_dt = event_release_dt_for_date(event_def, dt, tz)
            release_ts = pd.Timestamp(release_dt)
            if release_ts.tz is None:
                release_ts = release_ts.tz_localize(tz)

            # Look up stats for this event
            try:
                row = event_stats.loc[(pd.Timestamp(dt), event_code)]
                stats = row.to_dict() if hasattr(row, "to_dict") else {}
            except (KeyError, TypeError):
                stats = {}

            all_events.append((release_ts, event_code, stats))

    all_events.sort(key=lambda x: x[0])

    if not all_events:
        return df

    # Ensure df index is tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)

    # For each event, mark bars
    for release_ts, event_code, stats in all_events:
        event_date = release_ts.normalize()
        day_mask = df.index.normalize() == event_date
        df.loc[day_mask, "is_event_day"] = 1

        # Bars since release (on event day)
        post_mask = day_mask & (df.index >= release_ts)
        if post_mask.any():
            bar_indices = np.where(post_mask)[0]
            df.iloc[bar_indices, df.columns.get_loc("bars_since_release")] = np.arange(len(bar_indices))

        # Event feature columns (available only after release)
        stat_map = {
            "event_surprise": stats.get("surprise", 0.0) or 0.0,
            "event_post_max_return": stats.get("post_max_return", 0.0),
            "event_post_max_dd": stats.get("post_max_dd", 0.0),
            "event_post_volume": stats.get("post_total_volume", 0.0),
            "event_vol_lowest_q": stats.get("post_vol_lowest_quintile", 0.0),
            "event_vol_highest_q": stats.get("post_vol_highest_quintile", 0.0),
        }
        for col, val in stat_map.items():
            if col in df.columns:
                df.loc[post_mask, col] = float(val) if np.isfinite(float(val or 0)) else 0.0

    # Forward-fill event features in continuous mode
    if mode == "continuous":
        for col in _EVENT_FEATURES:
            # Only forward-fill non-zero values (zeros are pre-release)
            s = df[col].replace(0.0, np.nan)
            df[col] = s.ffill().fillna(0.0)

    # Compute days_until_next_event and days_since_last_event
    event_timestamps = sorted([ts for ts, _, _ in all_events])
    event_codes = {ts: code for ts, code, _ in all_events}

    if event_timestamps:
        ts_array = np.array([t.value for t in event_timestamps])
        bar_values = df.index.values.astype("int64")

        # Days until next event
        next_idx = np.searchsorted(ts_array, bar_values, side="left")
        for i in range(len(df)):
            idx = next_idx[i]
            if idx < len(event_timestamps):
                delta = (event_timestamps[idx] - df.index[i]).total_seconds() / 86400.0
                df.iloc[i, df.columns.get_loc("days_until_next_event")] = max(0.0, delta)
                df.iloc[i, df.columns.get_loc("next_event_type_id")] = _EVENT_TYPE_IDS.get(
                    event_codes[event_timestamps[idx]], 0
                )

            # Days since last event
            prev_idx = idx - 1
            if prev_idx >= 0:
                delta = (df.index[i] - event_timestamps[prev_idx]).total_seconds() / 86400.0
                df.iloc[i, df.columns.get_loc("days_since_last_event")] = max(0.0, delta)
                df.iloc[i, df.columns.get_loc("last_event_type_id")] = _EVENT_TYPE_IDS.get(
                    event_codes[event_timestamps[prev_idx]], 0
                )

    return df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def _scale_features(df: pd.DataFrame, clip_val: float = 5.0) -> pd.DataFrame:
    """Scale all features to ~[-5, +5] following CTAFlow conventions."""
    out = df.copy()

    # Distance/return features: * 100 to bps
    for col in ["dist_sma_20", "dist_sma_50", "dist_sma_200", "vwap_dist"]:
        if col in out.columns:
            out[col] = (out[col] * 100.0).clip(-clip_val, clip_val)

    # ATR and vol: * 100
    for col in ["atr_norm", "rv_1d", "rv_5d", "rv_20d"]:
        if col in out.columns:
            out[col] = (out[col] * 100.0).clip(-clip_val, clip_val)

    # MACD hist: rolling z-score
    if "macd_hist" in out.columns:
        rm = out["macd_hist"].rolling(252, min_periods=20).mean()
        rs = out["macd_hist"].rolling(252, min_periods=20).std().clip(lower=1e-8)
        out["macd_hist"] = ((out["macd_hist"] - rm) / rs * 2.0).clip(-clip_val, clip_val)

    # RSI already scaled in _add_technicals: (rsi - 50) / 10

    # Log return and vol_scaled_ret: * 100, clip
    for col in ["log_ret"]:
        if col in out.columns:
            out[col] = (out[col] * 100.0).clip(-clip_val, clip_val)
    if "vol_scaled_ret" in out.columns:
        out["vol_scaled_ret"] = out["vol_scaled_ret"].clip(-clip_val, clip_val)

    # Event max_return / max_dd already in bps, just clip
    for col in ["event_post_max_return", "event_post_max_dd"]:
        if col in out.columns:
            out[col] = out[col].clip(-clip_val, clip_val)

    # Event volume: log1p, center
    for col in ["event_post_volume", "event_vol_lowest_q", "event_vol_highest_q"]:
        if col in out.columns:
            nonzero = out[col] > 0
            if nonzero.any():
                out.loc[nonzero, col] = np.log1p(out.loc[nonzero, col])
                median_val = out.loc[nonzero, col].median()
                out.loc[nonzero, col] = (out.loc[nonzero, col] - median_val).clip(-clip_val, clip_val)

    # Surprise: rolling z-score
    if "event_surprise" in out.columns:
        s = out["event_surprise"]
        nonzero = s != 0
        if nonzero.sum() > 20:
            rm = s.rolling(252, min_periods=10).mean()
            rs = s.rolling(252, min_periods=10).std().clip(lower=1e-8)
            out["event_surprise"] = ((s - rm) / rs * 2.0).clip(-clip_val, clip_val)

    # days_until_next_event: cap and normalize
    if "days_until_next_event" in out.columns:
        out["days_until_next_event"] = out["days_until_next_event"].clip(0, 30) / 6.0  # ~[0, 5]

    # bars_since_release: cap and scale
    if "bars_since_release" in out.columns:
        out["bars_since_release"] = out["bars_since_release"].clip(-1, 100) / 20.0  # ~[-0.05, 5]

    # Fill any remaining NaN
    out = out.ffill().bfill().fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# EventTradingPrep
# ---------------------------------------------------------------------------

class EventTradingPrep:
    """Prepare aligned features for TFT-based event trading.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to process.
    mode : str
        "continuous" (all bars, events forward-filled) or "post_only"
        (only bars on event days after release).
    bar_minutes : int
        Bar size in minutes (default 5).
    session_start, session_end : str
        Session boundaries as "HH:MM".
    target_horizon_bars : int
        Forward return horizon in bars (default 6 = 30min at 5min bars).
    lookback_days : int
        Number of lookback days for the TFT encoder window.
    tz : str
        Instrument timezone.
    """

    def __init__(
        self,
        tickers: List[str],
        mode: str = "continuous",
        bar_minutes: int = 5,
        session_start: str = "08:30",
        session_end: str = "16:00",
        target_horizon_bars: int = 6,
        lookback_days: int = 20,
        tz: str = "America/Chicago",
    ):
        if mode not in ("continuous", "post_only"):
            raise ValueError(f"mode must be 'continuous' or 'post_only', got '{mode}'")

        self.tickers = sorted(tickers)
        self.mode = mode
        self.bar_minutes = bar_minutes
        self.session_start = session_start
        self.session_end = session_end
        self.target_horizon_bars = target_horizon_bars
        self.lookback_days = lookback_days
        self.tz = tz

        self.registry = build_ticker_registry(self.tickers)

        # Per-ticker storage
        self._dfs: Dict[str, pd.DataFrame] = {}
        self._event_dates: Dict[str, Dict[str, List[date]]] = {}
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_ticker(
        self,
        ticker: str,
        root_dir: str,
        event_stats: pd.DataFrame,
        event_dates: Optional[Dict[str, List[date]]] = None,
        event_surprise: Optional[pd.Series] = None,
        intraday_file: str = "intraday.csv",
    ) -> int:
        """Load and prepare all features for one ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        root_dir : str
            Root directory containing ``{ticker}/{intraday_file}``.
        event_stats : pd.DataFrame
            From ``EventExtractor.extract_all()["event_stats"]``.
            Indexed by (date, event_code).
        event_dates : dict, optional
            ``{event_code: [date, ...]}`` for event alignment.
            If None, inferred from event_stats index.
        event_surprise : pd.Series, optional
            Surprise values indexed by (date, event_code).
        intraday_file : str
            Filename for 5min OHLCV data.

        Returns
        -------
        int
            Number of valid sample bars.
        """
        from pathlib import Path

        # 1. Load intraday OHLCV
        intra_path = Path(root_dir) / ticker / intraday_file
        df = read_exported_df(str(intra_path))

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Standardize close column
        if "Close" not in df.columns:
            for c in df.columns:
                if c.lower() == "close" or c.lower() == "last":
                    df = df.rename(columns={c: "Close"})
                    break

        # 2. Infer event_dates from event_stats if not provided
        if event_dates is None and not event_stats.empty:
            event_dates = {}
            for (dt, code) in event_stats.index:
                d = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
                event_dates.setdefault(code, []).append(d)
        elif event_dates is None:
            event_dates = {}

        self._event_dates[ticker] = event_dates

        # 3. Merge surprise into event_stats if provided
        if event_surprise is not None and not event_stats.empty:
            if "surprise" not in event_stats.columns:
                event_stats = event_stats.copy()
                event_stats["surprise"] = event_surprise

        # 4. Add technical features
        df = _add_technicals(df)

        # 5. Add temporal features
        df = _add_temporal_features(df)

        # 6. Add intraday features
        df = _add_intraday_features(df)

        # 7. Align event features
        df = _align_event_features(df, event_stats, event_dates, self.tz, self.mode)

        # 8. Compute target: forward log return, vol-normalized
        log_close = np.log(df["Close"].astype(float).clip(lower=1e-8))
        fwd_ret = log_close.shift(-self.target_horizon_bars) - log_close
        ewm_vol = fwd_ret.ewm(span=63).std().clip(lower=1e-8)
        df["target"] = (fwd_ret / ewm_vol).clip(-5.0, 5.0)

        # 9. Filter to session hours
        session_start_time = time(
            int(self.session_start.split(":")[0]),
            int(self.session_start.split(":")[1]),
        )
        session_end_time = time(
            int(self.session_end.split(":")[0]),
            int(self.session_end.split(":")[1]),
        )
        # Use tz-naive times for filtering
        idx_time = df.index.tz_localize(None).time if df.index.tz is not None else df.index.time
        session_mask = pd.Series(
            [(session_start_time <= t <= session_end_time) for t in idx_time],
            index=df.index,
        )
        df = df[session_mask]

        # 10. Post-only mode: further filter to event days after release
        if self.mode == "post_only":
            df = df[df["bars_since_release"] >= 0]

        # 11. Scale features
        df = _scale_features(df)

        # 12. Add static features
        meta = self.registry.get(ticker)
        if meta is not None:
            df["ticker_id"] = meta.ticker_id
            df["asset_class_id"] = meta.asset_class_id
            df["asset_subclass_id"] = meta.asset_subclass_id
        else:
            df["ticker_id"] = 0
            df["asset_class_id"] = 0
            df["asset_subclass_id"] = 0

        # Store
        self._dfs[ticker] = df

        # Build feature column list
        self._feature_cols = (
            _TECH_FEATURES + _EVENT_FEATURES + _INTRADAY_FEATURES
        )
        # Add continuous-mode extras
        if self.mode == "continuous":
            self._feature_cols = self._feature_cols + ["days_since_last_event", "last_event_type_id"]

        n_valid = int(df["target"].notna().sum())
        logger.info(f"[{ticker}] {n_valid} valid bars, mode={self.mode}")
        return n_valid

    # ------------------------------------------------------------------
    # Convenience loader
    # ------------------------------------------------------------------

    @classmethod
    def from_directories(
        cls,
        root_dir: str,
        tickers: List[str],
        event_stats: pd.DataFrame,
        event_dates: Optional[Dict[str, List[date]]] = None,
        mode: str = "continuous",
        **kwargs,
    ) -> "EventTradingPrep":
        """Load all tickers from standard directory layout."""
        prep = cls(tickers=tickers, mode=mode, **kwargs)
        for ticker in prep.tickers:
            n = prep.load_ticker(ticker, root_dir, event_stats, event_dates)
            print(f"  [{ticker}] {n} valid bars")
        return prep

    # ------------------------------------------------------------------
    # TFT feature config
    # ------------------------------------------------------------------

    def get_tft_feature_config(self) -> dict:
        """Return feature categorization for TFT model construction."""
        past_observed = list(self._feature_cols)
        return {
            "static": ["ticker_id", "asset_class_id", "asset_subclass_id"],
            "known_future": list(_TEMPORAL_KNOWN),
            "past_observed": past_observed,
            "n_static": 3,
            "n_known": len(_TEMPORAL_KNOWN),
            "n_observed": len(past_observed),
            "n_event_types": N_EVENT_TYPES,
        }

    def get_dims(self) -> dict:
        """Return feature dimensions for model construction."""
        cfg = self.get_tft_feature_config()
        return {
            "n_static": cfg["n_static"],
            "n_known_future": cfg["n_known"],
            "n_past_observed": cfg["n_observed"],
            "n_event_types": cfg["n_event_types"],
            "lookback_days": self.lookback_days,
            "target_horizon_bars": self.target_horizon_bars,
        }

    # ------------------------------------------------------------------
    # Sample building
    # ------------------------------------------------------------------

    def build_samples(
        self,
        lookback_bars: Optional[int] = None,
    ) -> List[dict]:
        """Build aligned samples for all loaded tickers.

        Each sample is a dict with:
            static: np.ndarray (n_static,)
            known_future: np.ndarray (lookback_bars, n_known)
            past_observed: np.ndarray (lookback_bars, n_observed)
            target: float
            date: date  (for splitting)

        Parameters
        ----------
        lookback_bars : int, optional
            Number of lookback bars per sample.  Defaults to
            ``lookback_days * bars_per_day`` where bars_per_day is estimated
            from session length.
        """
        if lookback_bars is None:
            # Estimate bars per day from session
            h_start, m_start = map(int, self.session_start.split(":"))
            h_end, m_end = map(int, self.session_end.split(":"))
            session_minutes = (h_end * 60 + m_end) - (h_start * 60 + m_start)
            bars_per_day = session_minutes // self.bar_minutes
            lookback_bars = self.lookback_days * bars_per_day

        cfg = self.get_tft_feature_config()
        static_cols = cfg["static"]
        known_cols = cfg["known_future"]
        past_cols = cfg["past_observed"]

        samples: List[dict] = []

        for ticker, df in self._dfs.items():
            # Ensure all columns exist
            for col in known_cols + past_cols + static_cols:
                if col not in df.columns:
                    df[col] = 0.0

            valid_mask = df["target"].notna()
            valid_indices = np.where(valid_mask.values)[0]

            static_vals = df[static_cols].iloc[0].values.astype(np.float32)

            known_arr = df[known_cols].values.astype(np.float32)
            past_arr = df[past_cols].values.astype(np.float32)
            target_arr = df["target"].values.astype(np.float32)

            for idx in valid_indices:
                if idx < lookback_bars:
                    continue

                start = idx - lookback_bars
                end = idx  # current bar (inclusive in window)

                sample = {
                    "static": static_vals,
                    "known_future": known_arr[start:end + 1],       # (lookback+1, n_known)
                    "past_observed": past_arr[start:end + 1],       # (lookback+1, n_observed)
                    "target": target_arr[idx],
                    "date": df.index[idx].date() if hasattr(df.index[idx], "date") else df.index[idx],
                }
                samples.append(sample)

        logger.info(f"Built {len(samples)} samples across {len(self._dfs)} tickers")
        return samples

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def get_loaders(
        self,
        val_cutoff_date: Union[date, str],
        batch_size: int = 32,
        lookback_bars: Optional[int] = None,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """Build train/val DataLoaders split by date.

        Parameters
        ----------
        val_cutoff_date : date or str
            Samples on or after this date go to validation.
        batch_size : int
            Batch size for both loaders.
        lookback_bars : int, optional
            Lookback window in bars.
        num_workers : int
            DataLoader workers.

        Returns
        -------
        (train_loader, val_loader)
        """
        if isinstance(val_cutoff_date, str):
            val_cutoff_date = pd.Timestamp(val_cutoff_date).date()

        samples = self.build_samples(lookback_bars=lookback_bars)

        train_samples = [s for s in samples if s["date"] < val_cutoff_date]
        val_samples = [s for s in samples if s["date"] >= val_cutoff_date]

        logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

        train_ds = EventTradingDataset(train_samples)
        val_ds = EventTradingDataset(val_samples)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

        return train_loader, val_loader


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class EventTradingDataset(Dataset):
    """PyTorch Dataset wrapping event trading samples for TFT.

    Each sample is a dict of tensors:
        static: (n_static,)
        known_future: (lookback+1, n_known)
        past_observed: (lookback+1, n_observed)
        target: scalar
    """

    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "static": torch.tensor(s["static"], dtype=torch.float32),
            "known_future": torch.tensor(s["known_future"], dtype=torch.float32),
            "past_observed": torch.tensor(s["past_observed"], dtype=torch.float32),
            "target": torch.tensor(s["target"], dtype=torch.float32),
        }
