"""
Continuous intraday feature and target builder for high-frequency data.

This module provides bar-by-bar feature engineering for intraday trading models,
with proper deseasonalization using the CTAFlow diurnal seasonality utilities.

Features:
    - Multi-step forward returns for target prediction
    - Rolling VWAP with distance features
    - Deseasonalized volatility (FFF-based, rolling)
    - Deseasonalized volume (FFF-based, rolling)
    - Vol-scaled returns for standardization
    - Overnight returns (previous session gap)
    - Safe daily features (momentum/SMA/RV with proper lagging)

Anti-leakage:
    - Daily features use shift(1) before forward-filling
    - Deseasonalization uses rolling windows (no look-ahead)
    - Targets computed on full timeline before session filtering
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type, time, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Import CTAFlow deseasonalization utilities (aliased to avoid shadowing by params)
try:
    from CTAFlow.features.cyclical.seasonality.diurnal_seasonality import (
        deseasonalize_volatility as _deseas_vol_func,
        deseasonalize_volume as _deseas_volume_func,
    )
    _HAS_DESEAS = True
except ImportError:
    _HAS_DESEAS = False


@dataclass
class SessionSpec:
    """Session time specification."""
    name: str
    start: str  # "HH:MM"
    end: str    # "HH:MM"


class ContinuousIntradayPrep:
    """
    Continuous (bar-by-bar) feature + target builder for intraday data.

    Supports multiple session definitions (e.g., London/US overlap) and
    uses CTAFlow's deseasonalization utilities for proper treatment of
    intraday patterns.

    Parameters
    ----------
    sessions : Sequence[SessionSpec], optional
        Session definitions. Default: London (02:00-11:00) and USA (08:30-16:00)
    bar_minutes : int, default 5
        Bar frequency in minutes
    eps : float, default 1e-8
        Small constant for numerical stability

    Examples
    --------
    >>> prep = ContinuousIntradayPrep(bar_minutes=5)
    >>> df_out, mask, target_cols = prep.prepare(df, steps_60m=12)
    >>> train_df = df_out.loc[mask]
    """

    @staticmethod
    def get_feature_cols(
        steps_60m: int = 12,
        bar_minutes: int = 5,
        momentum_lookbacks: Tuple[int, ...] = (5, 10, 20),
        sma_windows: Tuple[int, ...] = (50, 200),
        rv_lookbacks: Tuple[int, ...] = (1, 5, 20),
        add_daily: bool = True,
        add_overnight: bool = True,
        add_deseas: bool = True,
        add_time_features: bool = True,
        add_resample_precalc: bool = True,
        resample_rules: Tuple[str, ...] = ("15min", "30min", "60min"),
        add_bid_ask: bool = True,
        add_event_markers: bool = True,
    ) -> List[str]:
        """Generate feature column names based on prepare() parameters.

        Returns the list of feature columns that prepare() will create,
        excluding OHLCV, targets, and metadata columns.
        """
        cols = ["tod_slot", "session_code", "is_active", "is_london", "is_usa", "is_session_overlap"]

        if add_time_features:
            cols.extend([
                "tod_sin",
                "tod_cos",
                "dow_sin",
                "dow_cos",
                "doy_sin",
                "doy_cos",
                "is_london_session",
                "is_usa_session",
                "is_overlap_session",
            ])

        # VWAP distance feature
        vwap_suffix = f"{steps_60m * bar_minutes}m"
        cols.append(f"vwap_{vwap_suffix}_dist")

        # Overnight return
        if add_overnight:
            cols.append("overnight_return")

        # Daily features
        if add_daily:
            for lb in momentum_lookbacks:
                cols.append(f"mom_{lb}d")
            for w in sma_windows:
                cols.append(f"dist_sma_{w}d")
            cols.append("rv_1d")
            for w in rv_lookbacks:
                if w != 1:
                    cols.append(f"rv_{w}d_mean")
            cols.extend(["macd_norm", "macd_signal_norm", "macd_hist_norm", "rsi_14"])

        # Deseasonalized features
        if add_deseas:
            cols.extend([
                "log_ret",
                "vol_scaled_ret",
                "deseasonalized_vol",
                "ret_deseasonalized",
                "deseasonalized_volume",
            ])

        # Pre-calculated multi-resolution context features
        if add_resample_precalc:
            for rule in resample_rules:
                suffix = (
                    rule.lower()
                    .replace("minutes", "min")
                    .replace("minute", "min")
                    .replace("hours", "h")
                    .replace("hour", "h")
                    .replace(" ", "")
                )
                suffix = suffix.replace("min", "m")
                cols.extend(
                    [
                        f"rs_{suffix}_logret",
                        f"rs_{suffix}_range",
                        f"rs_{suffix}_vwap_dist",
                        f"rs_{suffix}_logvol_z",
                        f"rs_{suffix}_ret_deseas",
                    ]
                )

        # Bid/ask volume features
        if add_bid_ask:
            cols.extend(["ba_imbalance", "ba_imbalance_ema"])

        # Known-future event markers
        if add_event_markers:
            cols.extend(["evt_is_release_day", "evt_bars_to_release", "evt_is_pre_release"])

        return cols

    def __init__(
        self,
        sessions: Optional[Sequence[SessionSpec]] = None,
        bar_minutes: int = 5,
        eps: float = 1e-8,
    ):
        self.sessions = sessions or [
            SessionSpec("LONDON", "02:00", "11:00"),
            SessionSpec("USA", "08:30", "16:00"),
        ]
        self.bar_minutes = int(bar_minutes)
        self.eps = eps

    # -------------------------
    # Column handling / helpers
    # -------------------------
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to OHLCV format."""
        df = df.copy()
        colmap = {}
        for c in df.columns:
            cl = c.strip().lower().replace(" ", "_")
            if cl == "open":
                colmap[c] = "Open"
            elif cl == "high":
                colmap[c] = "High"
            elif cl == "low":
                colmap[c] = "Low"
            elif cl in ("close", "last"):
                colmap[c] = "Close"
            elif cl in ("volume", "vol"):
                colmap[c] = "Volume"
            elif cl in ("bid_volume", "bidvolume", "bidvol", "bid_vol"):
                colmap[c] = "BidVolume"
            elif cl in ("ask_volume", "askvolume", "askvol", "ask_vol"):
                colmap[c] = "AskVolume"
        df = df.rename(columns=colmap)

        req = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    @staticmethod
    def _parse_time(hhmm: str) -> time:
        """Parse HH:MM string to time object."""
        parts = hhmm.split(":")
        return time(int(parts[0]), int(parts[1]))

    def _time_mask(self, idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
        """Create boolean mask for time range."""
        s = self._parse_time(start)
        e = self._parse_time(end)
        t = idx.time
        return (t >= s) & (t <= e)

    def _slot_index(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Compute time-of-day slot index."""
        mins = idx.hour * 60 + idx.minute
        slot = (mins // self.bar_minutes).astype(int)
        return pd.Series(slot, index=idx, name="tod_slot")

    # -------------------------
    # Sessions
    # -------------------------
    def add_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session indicator columns."""
        df = df.copy()

        # Support up to 2 sessions with overlap detection
        if len(self.sessions) >= 1:
            is_s1 = self._time_mask(df.index, self.sessions[0].start, self.sessions[0].end)
            df[f"is_{self.sessions[0].name.lower()}"] = is_s1.astype(np.int8)
        else:
            is_s1 = np.zeros(len(df), dtype=bool)

        if len(self.sessions) >= 2:
            is_s2 = self._time_mask(df.index, self.sessions[1].start, self.sessions[1].end)
            df[f"is_{self.sessions[1].name.lower()}"] = is_s2.astype(np.int8)
            # Session code: 0=none, 1=s1 only, 2=s2 only, 3=overlap
            df["session_code"] = (is_s1.astype(np.int8) + 2 * is_s2.astype(np.int8))
        else:
            is_s2 = np.zeros(len(df), dtype=bool)
            df["session_code"] = is_s1.astype(np.int8)

        df["is_active"] = ((is_s1) | (is_s2)).astype(np.int8)
        # Compatibility aliases for common 2-session setup
        if len(self.sessions) >= 1:
            s1_col = f"is_{self.sessions[0].name.lower()}"
            if s1_col in df.columns:
                df["is_london"] = df[s1_col].astype(np.int8)
        else:
            df["is_london"] = 0

        if len(self.sessions) >= 2:
            s2_col = f"is_{self.sessions[1].name.lower()}"
            if s2_col in df.columns:
                df["is_usa"] = df[s2_col].astype(np.int8)
        else:
            df["is_usa"] = 0

        df["is_session_overlap"] = ((df["is_london"] > 0) & (df["is_usa"] > 0)).astype(np.int8)
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical clock/calendar features and session one-hot indicators.

        Designed to stay stable across resampling frequencies.
        """
        df = df.copy()
        idx = df.index

        secs = idx.hour * 3600 + idx.minute * 60 + idx.second
        tod = secs / (24 * 3600.0)
        dow = idx.dayofweek / 7.0
        doy = (idx.dayofyear - 1) / 365.25

        df["tod_sin"] = np.sin(2.0 * np.pi * tod).astype(np.float32)
        df["tod_cos"] = np.cos(2.0 * np.pi * tod).astype(np.float32)
        df["dow_sin"] = np.sin(2.0 * np.pi * dow).astype(np.float32)
        df["dow_cos"] = np.cos(2.0 * np.pi * dow).astype(np.float32)
        df["doy_sin"] = np.sin(2.0 * np.pi * doy).astype(np.float32)
        df["doy_cos"] = np.cos(2.0 * np.pi * doy).astype(np.float32)

        df["is_london_session"] = df.get("is_london", pd.Series(0, index=df.index)).astype(np.float32)
        df["is_usa_session"] = df.get("is_usa", pd.Series(0, index=df.index)).astype(np.float32)
        df["is_overlap_session"] = df.get("is_session_overlap", pd.Series(0, index=df.index)).astype(np.float32)
        return df

    def add_resample_precalcs(
        self,
        df: pd.DataFrame,
        rules: Tuple[str, ...] = ("15min", "30min", "60min"),
        vwap_window_bars: int = 4,
        vol_z_window: int = 64,
    ) -> pd.DataFrame:
        """
        Pre-calculate multi-resolution features and align them back to base index.

        Uses forward-fill alignment to avoid dropping rows due to sparse coarser bars.
        """
        df = df.copy()
        base_idx = df.index

        for rule in rules:
            suffix = (
                rule.lower()
                .replace("minutes", "min")
                .replace("minute", "min")
                .replace("hours", "h")
                .replace("hour", "h")
                .replace(" ", "")
            ).replace("min", "m")

            agg = {}
            if "Open" in df.columns:
                agg["Open"] = "first"
            if "High" in df.columns:
                agg["High"] = "max"
            if "Low" in df.columns:
                agg["Low"] = "min"
            if "Close" in df.columns:
                agg["Close"] = "last"
            if "Volume" in df.columns:
                agg["Volume"] = "sum"

            if not agg:
                continue

            rs = df.resample(rule).agg(agg).dropna(how="all")
            if rs.empty or "Close" not in rs.columns:
                continue

            rs_close = rs["Close"].astype(float).clip(lower=self.eps)
            rs_logret = np.log(rs_close).diff()
            rs_range = ((rs["High"] - rs["Low"]) / (rs_close + self.eps)).astype(float) if {"High", "Low"}.issubset(rs.columns) else pd.Series(np.nan, index=rs.index)

            if "Volume" in rs.columns and "High" in rs.columns and "Low" in rs.columns:
                tp = (rs["High"] + rs["Low"] + rs_close) / 3.0
                vol = rs["Volume"].astype(float)
                pv = (tp * vol).rolling(vwap_window_bars, min_periods=1).sum()
                vv = vol.rolling(vwap_window_bars, min_periods=1).sum().replace(0.0, np.nan)
                rs_vwap = pv / vv
                rs_vwap_dist = (rs_close - rs_vwap) / (rs_vwap + self.eps)
            else:
                rs_vwap_dist = pd.Series(np.nan, index=rs.index)

            if "Volume" in rs.columns:
                rs_logvol = np.log1p(rs["Volume"].astype(float))
                lv_mu = rs_logvol.rolling(vol_z_window, min_periods=max(8, vol_z_window // 4)).mean()
                lv_sd = rs_logvol.rolling(vol_z_window, min_periods=max(8, vol_z_window // 4)).std()
                rs_logvol_z = (rs_logvol - lv_mu) / (lv_sd + self.eps)
            else:
                rs_logvol_z = pd.Series(np.nan, index=rs.index)

            # Deseasonalized return: logret / rolling abs-return mean
            # Uses rolling window at the resampled frequency to avoid lookahead
            abs_ret = rs_logret.abs()
            rolling_vol = abs_ret.rolling(
                vol_z_window, min_periods=max(8, vol_z_window // 4),
            ).mean()
            rs_ret_deseas = rs_logret / rolling_vol.clip(lower=self.eps)

            rs_feats = pd.DataFrame(
                {
                    f"rs_{suffix}_logret": rs_logret,
                    f"rs_{suffix}_range": rs_range,
                    f"rs_{suffix}_vwap_dist": rs_vwap_dist,
                    f"rs_{suffix}_logvol_z": rs_logvol_z,
                    f"rs_{suffix}_ret_deseas": rs_ret_deseas,
                },
                index=rs.index,
            )
            rs_feats = rs_feats.reindex(base_idx, method="ffill")
            rs_feats = rs_feats.ffill().bfill()

            for c in rs_feats.columns:
                df[c] = rs_feats[c].astype(np.float32)

        return df

    # -------------------------
    # Rolling VWAP
    # -------------------------
    def add_vwap(
        self,
        df: pd.DataFrame,
        window_bars: int = 12,
        suffix: str = "60m",
    ) -> pd.DataFrame:
        """Add rolling VWAP and distance features.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
        window_bars : int, default 12
            Rolling window in bars (12 * 5min = 60min)
        suffix : str, default "60m"
            Suffix for column names
        """
        df = df.copy()
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = df["Volume"].astype(float)

        pv = (tp * vol).rolling(window_bars, min_periods=window_bars).sum()
        vv = vol.rolling(window_bars, min_periods=window_bars).sum().replace(0.0, np.nan)
        vwap = pv / vv

        df[f"vwap_{suffix}"] = vwap
        df[f"vwap_{suffix}_dist"] = (df["Close"] - vwap) / (vwap + self.eps)
        return df

    # -------------------------
    # Overnight returns
    # -------------------------
    def add_overnight_returns(
        self,
        df: pd.DataFrame,
        session_open: Optional[str] = None,
        session_end: Optional[str] = None,
        lookback_hours: int = 8,
    ) -> pd.DataFrame:
        """Add overnight return feature (gap from previous session close).

        Computes the return from the previous session's close to the current
        session's open. If exact session times aren't available, uses a
        lookback window approach.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex
        session_open : str, optional
            Session open time as "HH:MM". If None, uses first session's start.
        session_end : str, optional
            Session end time as "HH:MM". If None, uses last session's end.
        lookback_hours : int, default 8
            Fallback: hours to look back for previous close if session times
            don't yield exact matches

        Returns
        -------
        pd.DataFrame
            DataFrame with 'overnight_return' column added
        """
        df = df.copy()

        # Default to session times
        if session_open is None:
            session_open = self.sessions[0].start
        if session_end is None:
            session_end = self.sessions[-1].end

        open_time = self._parse_time(session_open)
        end_time = self._parse_time(session_end)

        close = df["Close"].astype(float)

        # Get session closes (bars at session_end time)
        close_mask = df.index.time == end_time
        session_closes = close[close_mask].copy()
        session_closes.index = session_closes.index.normalize()

        # Get session opens (bars at session_open time)
        open_mask = df.index.time == open_time
        session_opens = close[open_mask].copy()
        session_opens.index = session_opens.index.normalize()

        # If exact matches are sparse, use daily first/last
        if len(session_closes) < len(df.index.normalize().unique()) * 0.5:
            # Fallback: use daily last price as close
            session_closes = close.groupby(close.index.normalize()).last()
        if len(session_opens) < len(df.index.normalize().unique()) * 0.5:
            # Fallback: use daily first price as open
            session_opens = close.groupby(close.index.normalize()).first()

        # Align indices
        common_dates = session_opens.index.intersection(session_closes.index)

        # Overnight return: (open[t] - close[t-1]) / close[t-1]
        prev_closes = session_closes.shift(1)
        overnight_ret = (session_opens - prev_closes) / (prev_closes + self.eps)
        overnight_ret = overnight_ret.reindex(common_dates)

        # Map back to intraday index (forward fill within each day)
        df["overnight_return"] = overnight_ret.reindex(df.index.normalize()).values

        return df

    # -------------------------
    # Bid/Ask volume features
    # -------------------------
    def add_bid_ask_volume(
        self,
        df: pd.DataFrame,
        imbalance_ema_bars: int = 20,
    ) -> pd.DataFrame:
        """Add bid/ask volume imbalance features.

        Requires ``BidVolume`` and ``AskVolume`` columns (mapped
        automatically from Sierra Chart style ``Bid Volume`` /
        ``Ask Volume`` by ``_standardize_columns``).

        Features added:
        - ``ba_imbalance``: (ask - bid) / total, clipped to [-1, 1]
        - ``ba_imbalance_ema``: EMA-smoothed imbalance

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with BidVolume and AskVolume columns.
        imbalance_ema_bars : int, default 20
            EMA span for smoothed imbalance.
        """
        df = df.copy()
        bid = df["BidVolume"].astype(float)
        ask = df["AskVolume"].astype(float)
        total = bid + ask
        imb = (ask - bid) / total.replace(0.0, np.nan)
        df["ba_imbalance"] = imb.clip(-1, 1)
        df["ba_imbalance_ema"] = (
            df["ba_imbalance"]
            .ewm(span=imbalance_ema_bars, min_periods=1)
            .mean()
        )
        return df

    # -------------------------
    # Known-future event markers
    # -------------------------
    def add_event_markers(
        self,
        df: pd.DataFrame,
        ticker: str,
        pre_release_bars: int = 12,
    ) -> pd.DataFrame:
        """Add known-future event release markers for a ticker.

        Uses ``event_presets`` to identify recurring release days (CPI, NFP,
        EIA, etc.) based on weekday/week-of-month rules and adds:

        - ``evt_is_release_day``: 1.0 on days matching any relevant event
        - ``evt_bars_to_release``: normalised countdown [0, 1] on release days
          (1 = far from release, 0 = at release time), -1 on non-event days
        - ``evt_is_pre_release``: 1.0 within *pre_release_bars* bars before
          release time on release days

        These are **known-future** features — they depend only on the calendar
        and do not leak any realised market data.

        Parameters
        ----------
        df : pd.DataFrame
            Intraday DataFrame with DatetimeIndex.
        ticker : str
            Ticker symbol used to look up relevant events.
        pre_release_bars : int, default 12
            Number of bars before release to flag as pre-release.
        """
        from CTAFlow.screeners.event_presets import (
            get_events_for_ticker,
            is_matching_event_slot,
            EventDefinition,
        )

        df = df.copy()
        events = get_events_for_ticker(ticker)
        if not events:
            df["evt_is_release_day"] = np.float32(0)
            df["evt_bars_to_release"] = np.float32(-1)
            df["evt_is_pre_release"] = np.float32(0)
            return df

        idx = df.index
        dates = idx.date
        unique_dates = np.unique(dates)

        # Build set of (date, release_time_as_minutes) for matching days
        release_info: Dict[date_type, int] = {}  # date -> earliest release minute
        for d in unique_dates:
            for ev in events:
                if is_matching_event_slot(d, ev):
                    release_min = ev.release_time_local.hour * 60 + ev.release_time_local.minute
                    if d not in release_info or release_min < release_info[d]:
                        release_info[d] = release_min

        is_release_day = np.zeros(len(df), dtype=np.float32)
        bars_to_release = np.full(len(df), -1.0, dtype=np.float32)
        is_pre_release = np.zeros(len(df), dtype=np.float32)

        bar_minutes_total = idx.hour * 60 + idx.minute

        for d, rel_min in release_info.items():
            day_mask = dates == d
            is_release_day[day_mask] = 1.0

            # Bars to release: normalised by session length
            day_bar_mins = bar_minutes_total[day_mask]
            diff = (rel_min - day_bar_mins).astype(np.float32)
            # Normalise: 1 far away, 0 at release, keep negative for post-release
            session_len = max(day_bar_mins.max() - day_bar_mins.min(), 1)
            normalised = (diff / session_len).clip(-1, 1)
            bars_to_release[day_mask] = normalised

            # Pre-release window: bars within pre_release_bars * bar_minutes before release
            pre_window_mins = pre_release_bars * self.bar_minutes
            pre_mask = day_mask & (bar_minutes_total >= rel_min - pre_window_mins) & (bar_minutes_total < rel_min)
            is_pre_release[pre_mask] = 1.0

        df["evt_is_release_day"] = is_release_day
        df["evt_bars_to_release"] = bars_to_release
        df["evt_is_pre_release"] = is_pre_release
        return df

    # -------------------------
    # Safe daily features (shift(1) + ffill)
    # -------------------------
    def add_daily_features(
        self,
        df: pd.DataFrame,
        momentum_lookbacks: Tuple[int, ...] = (5, 10, 20),
        sma_windows: Tuple[int, ...] = (50, 200),
        rv_lookbacks: Tuple[int, ...] = (1, 5, 20),
    ) -> pd.DataFrame:
        """Add daily features with proper lagging to avoid lookahead.

        All daily features are computed on daily close, shifted by 1 day,
        then forward-filled to intraday frequency.
        """
        df = df.copy()

        close = df["Close"].astype(float)
        logp = np.log(close.clip(lower=self.eps))
        r = logp.diff()

        # Daily close
        daily_close = close.resample("1D").last().dropna()
        daily_close.index = daily_close.index.normalize()

        # Daily RV from intraday returns: sqrt(sum r^2)
        daily_rv = r.groupby(df.index.normalize()).apply(
            lambda x: np.sqrt(np.nansum(x.values ** 2))
        )
        daily_rv.index = pd.to_datetime(daily_rv.index).normalize()

        # Momentum (shift 1 day to avoid lookahead)
        for lb in momentum_lookbacks:
            mom = np.log(daily_close / daily_close.shift(lb))
            mom = mom.replace([np.inf, -np.inf], np.nan).shift(1)
            df[f"mom_{lb}d"] = mom.reindex(df.index.normalize()).ffill().values

        # SMA (shift 1 day)
        for w in sma_windows:
            sma = daily_close.rolling(w, min_periods=max(5, w // 5)).mean().shift(1)
            df[f"sma_{w}d"] = sma.reindex(df.index.normalize()).ffill().values
            df[f"dist_sma_{w}d"] = df["Close"] / (df[f"sma_{w}d"] + self.eps) - 1.0

        # RV (shift 1 day)
        df["rv_1d"] = daily_rv.shift(1).reindex(df.index.normalize()).ffill().values
        for w in rv_lookbacks:
            if w == 1:
                continue  # Already added rv_1d
            rvw = daily_rv.rolling(w, min_periods=max(2, w // 3)).mean().shift(1)
            df[f"rv_{w}d_mean"] = rvw.reindex(df.index.normalize()).ffill().values

        # MACD (12/26/9) — normalized by close to make scale-invariant
        ema12 = daily_close.ewm(span=12, min_periods=12).mean()
        ema26 = daily_close.ewm(span=26, min_periods=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, min_periods=9).mean()
        macd_hist = macd_line - signal_line
        # Normalize by close so values are comparable across assets
        macd_norm = (macd_line / (daily_close + self.eps)).shift(1)
        signal_norm = (signal_line / (daily_close + self.eps)).shift(1)
        hist_norm = (macd_hist / (daily_close + self.eps)).shift(1)
        df["macd_norm"] = macd_norm.reindex(df.index.normalize()).ffill().values
        df["macd_signal_norm"] = signal_norm.reindex(df.index.normalize()).ffill().values
        df["macd_hist_norm"] = hist_norm.reindex(df.index.normalize()).ffill().values

        # RSI-14 — already bounded [0, 100], rescale to [-1, 1]
        daily_ret = daily_close.diff()
        gain = daily_ret.clip(lower=0)
        loss = (-daily_ret).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + self.eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Rescale [0, 100] → [-1, 1]
        rsi_scaled = (rsi / 50.0 - 1.0).shift(1)
        df["rsi_14"] = rsi_scaled.reindex(df.index.normalize()).ffill().values

        return df

    # -------------------------
    # Deseasonalized features (using CTAFlow utilities)
    # -------------------------
    def add_deseasonalized_features(
        self,
        df: pd.DataFrame,
        rolling_days: int = 252,
        refit_interval: int = 10,
        deseasonalize_vol: bool = True,
        deseasonalize_volume: bool = True,
        deseasonalize_returns: bool = True,
        vol_scale_returns: bool = True,
        order: int = 3,
    ) -> pd.DataFrame:
        """Add deseasonalized volatility, volume, and returns.

        Uses CTAFlow's FFF-based deseasonalization with rolling windows
        to avoid look-ahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex
        rolling_days : int, default 252
            Rolling window for deseasonalization model
        refit_interval : int, default 10
            Refit model every N days (higher = faster, less accurate)
        deseasonalize_vol : bool, default True
            Deseasonalize absolute returns (volatility proxy)
        deseasonalize_volume : bool, default True
            Deseasonalize volume
        deseasonalize_returns : bool, default True
            Scale returns by seasonal volatility factor
        vol_scale_returns : bool, default True
            Additionally scale returns by lagged daily RV (if available)
        order : int, default 3
            FFF order (number of harmonic pairs)

        Returns
        -------
        pd.DataFrame
            DataFrame with deseasonalized features added
        """
        if not _HAS_DESEAS:
            raise ImportError(
                "CTAFlow deseasonalization utilities not available. "
                "Install CTAFlow or use add_deseasonalized_features_legacy()."
            )

        df = df.copy()

        close = df["Close"].astype(float)
        logp = np.log(close.clip(lower=self.eps))
        df["log_ret"] = logp.diff()

        # Build intraday index for deseasonalization
        intraday_idx = df.groupby(df.index.normalize()).cumcount()
        intraday_idx = pd.Series(intraday_idx.values, index=df.index)

        # Estimate bins per day
        bins_per_day = int(intraday_idx.groupby(df.index.normalize()).max().median()) + 1

        # Vol-scale returns first if requested (using lagged daily RV)
        if vol_scale_returns and "rv_1d" in df.columns:
            rv_scale = df["rv_1d"].astype(float).clip(lower=self.eps)
            df["vol_scaled_ret"] = df["log_ret"] / rv_scale
        else:
            df["vol_scaled_ret"] = df["log_ret"]

        # Deseasonalize volatility (absolute returns)
        if deseasonalize_vol:
            volatility = df["vol_scaled_ret"].abs()
            # Filter valid values
            valid_mask = volatility.notna() & (volatility > 0)

            if valid_mask.sum() > bins_per_day * rolling_days:
                result = _deseas_vol_func(
                    volatility[valid_mask],
                    intraday_idx=intraday_idx[valid_mask],
                    bins_per_day=bins_per_day,
                    order=order,
                    use_log=True,
                    rolling_days=rolling_days,
                    refit_interval=refit_interval,
                )
                # Map results back positionally (avoids reindex failures
                # when the datetime index has duplicate timestamps).
                df["deseasonalized_vol"] = np.nan
                df.loc[valid_mask, "deseasonalized_vol"] = result["adjusted"].values
                df["seasonal_vol_factor"] = np.nan
                df.loc[valid_mask, "seasonal_vol_factor"] = result["seasonal"].values

                # Scale returns by seasonal factor
                if deseasonalize_returns:
                    seasonal = df["seasonal_vol_factor"].fillna(1.0)
                    df["ret_deseasonalized"] = df["vol_scaled_ret"] / seasonal.clip(lower=self.eps)
            else:
                df["deseasonalized_vol"] = volatility
                df["seasonal_vol_factor"] = 1.0
                df["ret_deseasonalized"] = df["vol_scaled_ret"]

        # Deseasonalize volume
        if deseasonalize_volume:
            volume = df["Volume"].astype(float)
            valid_mask = volume.notna() & (volume > 0)

            if valid_mask.sum() > bins_per_day * rolling_days:
                result = _deseas_volume_func(
                    volume[valid_mask],
                    intraday_idx=intraday_idx[valid_mask],
                    bins_per_day=bins_per_day,
                    order=order + 1,  # Volume often needs higher order
                    rolling_days=rolling_days,
                    refit_interval=refit_interval,
                )
                df["deseasonalized_volume"] = np.nan
                df.loc[valid_mask, "deseasonalized_volume"] = result["adjusted"].values
                df["seasonal_volume_factor"] = np.nan
                df.loc[valid_mask, "seasonal_volume_factor"] = result["seasonal"].values
            else:
                df["deseasonalized_volume"] = volume
                df["seasonal_volume_factor"] = 1.0

        return df

    def add_deseasonalized_features_legacy(
        self,
        df: pd.DataFrame,
        rolling_days: int = 252,
        refit_interval: int = 10,
    ) -> pd.DataFrame:
        """Legacy deseasonalization using simple rolling mean by time slot.

        Use this if CTAFlow utilities are not available.
        """
        df = df.copy()

        close = df["Close"].astype(float)
        logp = np.log(close.clip(lower=self.eps))
        df["log_ret"] = logp.diff()

        slot = df["tod_slot"] if "tod_slot" in df.columns else self._slot_index(df.index)

        # Vol scaling
        if "rv_1d" in df.columns:
            scale = df["rv_1d"].astype(float).clip(lower=self.eps)
        else:
            scale = 1.0
        df["vol_scaled_ret"] = df["log_ret"] / scale

        # Deseasonalize by slot (simple approach)
        mag = df["vol_scaled_ret"].abs()

        # Group by slot and compute rolling mean
        tmp = pd.DataFrame({
            "mag": mag.values,
            "vol": df["Volume"].values,
            "date": df.index.normalize(),
            "slot": slot.values,
        }, index=df.index)

        # Compute daily slot aggregates
        daily_slot = tmp.groupby(["date", "slot"]).agg({
            "mag": "mean",
            "vol": "sum",
        })

        # Rolling mean by slot
        seasonal_mag = daily_slot.groupby(level=1)["mag"].transform(
            lambda x: x.rolling(rolling_days, min_periods=max(10, rolling_days // 10)).mean()
        )
        seasonal_vol = daily_slot.groupby(level=1)["vol"].transform(
            lambda x: x.rolling(rolling_days, min_periods=max(10, rollig_days // 10)).mean()
        )

        # Map back to original index
        join_key = pd.MultiIndex.from_arrays(
            [tmp["date"].values, tmp["slot"].values],
            names=["date", "slot"]
        )

        df["seasonal_vol_factor"] = seasonal_mag.reindex(join_key).values
        df["deseasonalized_vol"] = mag / df["seasonal_vol_factor"].clip(lower=self.eps)
        df["ret_deseasonalized"] = df["vol_scaled_ret"] / df["seasonal_vol_factor"].clip(lower=self.eps)

        df["seasonal_volume_factor"] = seasonal_vol.reindex(join_key).values
        df["deseasonalized_volume"] = df["Volume"] / df["seasonal_volume_factor"].clip(lower=self.eps)

        return df

    # -------------------------
    # Feature Scaling (similar to DeepIDMomentum)
    # -------------------------
    def scale_features(
        self,
        df: pd.DataFrame,
        scale_to_basis_points: bool = True,
        clip_outliers: bool = True,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Scale features to comparable ranges for neural network training.

        Applies feature-specific scaling based on column patterns to ensure
        all features are on comparable scales (~[-5, +5]) without lookahead bias.
        Uses fixed scaling factors where possible.

        Scaling Rules by Feature Type
        -----------------------------
        **Distance features (already fractional):**
        - vwap_*_dist, dist_sma_*: * 100 (to basis points)

        **Absolute price features:**
        - sma_*d, vwap_* (no _dist): normalize relative to Close

        **Return features:**
        - mom_*d: * 100 (to basis points)
        - overnight_return: * 100 (to basis points)
        - vol_scaled_ret, ret_deseasonalized: * 100 (to basis points)

        **Volatility features:**
        - rv_1d, rv_*d_mean: * 100 (convert to percentage), then clip
        - deseasonalized_vol: clip outliers only (already z-scored)

        **Volume features:**
        - deseasonalized_volume: clip outliers only (already z-scored)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features from prepare()
        scale_to_basis_points : bool, default True
            If True, multiply relevant features by 100 to convert to basis points
        clip_outliers : bool, default True
            If True, clip features to [-10, 10] to handle extreme values
        inplace : bool, default False
            If True, modify DataFrame in place

        Returns
        -------
        pd.DataFrame
            Scaled DataFrame

        Notes
        -----
        - All scaled features target approximate range [-5, +5]
        - Fixed scaling factors avoid lookahead bias entirely
        - Call this after prepare() but before creating datasets

        Examples
        --------
        >>> prep = ContinuousIntradayPrep()
        >>> df, mask, targets = prep.prepare(raw_df)
        >>> df_scaled = prep.scale_features(df)
        """
        if not inplace:
            df = df.copy()

        scale_factor = 100.0 if scale_to_basis_points else 1.0
        scaled_cols = set()

        # --- DISTANCE FEATURES (fractional -> basis points) ---
        # These are already (value - reference) / reference format
        dist_patterns = ['_dist', 'dist_']
        for col in df.columns:
            if any(p in col.lower() for p in dist_patterns):
                if clip_outliers:
                    df[col] = df[col].clip(-0.1, 0.1) * scale_factor
                else:
                    df[col] = df[col] * scale_factor
                scaled_cols.add(col)

        # --- ABSOLUTE PRICE FEATURES (normalize to Close) ---
        # sma_*d without dist_, vwap_* without _dist
        close = df["Close"].astype(float) if "Close" in df.columns else None
        if close is not None:
            for col in df.columns:
                # SMA columns (not distance)
                if col.startswith("sma_") and col.endswith("d") and col not in scaled_cols:
                    df[col] = ((df[col] - close) / (close + self.eps)) * scale_factor
                    if clip_outliers:
                        df[col] = df[col].clip(-10, 10)
                    scaled_cols.add(col)

                # VWAP columns (not distance)
                if col.startswith("vwap_") and "_dist" not in col and col not in scaled_cols:
                    df[col] = ((df[col] - close) / (close + self.eps)) * scale_factor
                    if clip_outliers:
                        df[col] = df[col].clip(-10, 10)
                    scaled_cols.add(col)

        # --- RETURN FEATURES (fractional -> basis points) ---
        return_cols = ['mom_', 'overnight_return', 'vol_scaled_ret', 'ret_deseasonalized', 'log_ret']
        for col in df.columns:
            if any(col.startswith(p) or col == p for p in return_cols) and col not in scaled_cols:
                if clip_outliers:
                    # Clip before scaling (returns > 10% are extreme)
                    df[col] = df[col].clip(-0.1, 0.1) * scale_factor
                else:
                    df[col] = df[col] * scale_factor
                scaled_cols.add(col)

        # --- MACD (normalized by close — small fractional, scale to bps) ---
        for col in ("macd_norm", "macd_signal_norm", "macd_hist_norm"):
            if col in df.columns and col not in scaled_cols:
                if clip_outliers:
                    df[col] = df[col].clip(-0.1, 0.1) * scale_factor
                else:
                    df[col] = df[col] * scale_factor
                scaled_cols.add(col)

        # --- RSI (already [-1, 1], no scaling needed) ---
        if "rsi_14" in df.columns and "rsi_14" not in scaled_cols:
            scaled_cols.add("rsi_14")

        # --- VOLATILITY FEATURES ---
        # RV features: small values, scale up
        for col in df.columns:
            if (col.startswith("rv_") or "rv_" in col) and col not in scaled_cols:
                # RV is typically 0.001-0.05, * 100 gives 0.1-5
                df[col] = df[col] * scale_factor
                if clip_outliers:
                    df[col] = df[col].clip(0, 10)
                scaled_cols.add(col)

        # Deseasonalized vol: already z-scored, just clip
        if "deseasonalized_vol" in df.columns and "deseasonalized_vol" not in scaled_cols:
            if clip_outliers:
                df["deseasonalized_vol"] = df["deseasonalized_vol"].clip(-5, 5)
            scaled_cols.add("deseasonalized_vol")

        # --- VOLUME FEATURES ---
        if "deseasonalized_volume" in df.columns and "deseasonalized_volume" not in scaled_cols:
            # Already z-scored by seasonal, just clip
            if clip_outliers:
                df["deseasonalized_volume"] = df["deseasonalized_volume"].clip(-5, 5)
            scaled_cols.add("deseasonalized_volume")

        # --- DESEASONALIZED RESAMPLED RETURNS ---
        for col in df.columns:
            if col.endswith("_ret_deseas") and col not in scaled_cols:
                # Already z-scored (logret / rolling_vol), just clip
                if clip_outliers:
                    df[col] = df[col].clip(-5, 5)
                scaled_cols.add(col)

        # --- BID/ASK IMBALANCE ---
        for col in ("ba_imbalance", "ba_imbalance_ema"):
            if col in df.columns and col not in scaled_cols:
                # Already in [-1, 1], no scaling needed
                scaled_cols.add(col)

        # --- EVENT MARKERS ---
        for col in ("evt_is_release_day", "evt_bars_to_release", "evt_is_pre_release"):
            if col in df.columns and col not in scaled_cols:
                # Already in [-1, 1], no scaling needed
                scaled_cols.add(col)

        # --- FINAL CLIP for any remaining numeric columns ---
        if clip_outliers:
            for col in df.columns:
                if col not in scaled_cols and df[col].dtype in [np.float32, np.float64]:
                    # Skip target columns and metadata
                    if col.startswith("y_fwd") or col in ["Open", "High", "Low", "Close", "Volume"]:
                        continue
                    if col in ["session_code", "is_active", "tod_slot"]:
                        continue
                    df[col] = df[col].clip(-10, 10)

        return df

    # -------------------------
    # Targets: multi-step forward returns
    # -------------------------
    def add_targets_multistep(
        self,
        df: pd.DataFrame,
        steps: int = 12,
        prefix: str = "y_fwd",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create multi-step forward return targets.

        Parameters
        ----------
        df : pd.DataFrame
            Data with Close prices
        steps : int, default 12
            Number of forward steps (12 * 5min = 60min)
        prefix : str, default "y_fwd"
            Prefix for target column names

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            DataFrame with targets and list of target column names
        """
        df = df.copy()
        logp = np.log(df["Close"].astype(float).clip(lower=self.eps))
        target_cols = []

        for h in range(1, steps + 1):
            c = f"{prefix}_{h}"
            df[c] = logp.shift(-h) - logp
            target_cols.append(c)

        return df, target_cols

    # -------------------------
    # Main prepare method
    # -------------------------
    def prepare(
        self,
        df: pd.DataFrame,
        steps_60m: int = 12,
        keep_only_active: bool = False,
        add_daily: bool = True,
        add_overnight: bool = True,
        add_deseas: bool = True,
        add_time_features: bool = True,
        add_resample_precalc: bool = True,
        resample_rules: Tuple[str, ...] = ("15min", "30min", "60min"),
        rolling_days_deseas: int = 252,
        refit_interval: int = 10,
        use_legacy_deseas: bool = False,
        apply_scaling: bool = False,
        scale_to_basis_points: bool = True,
        add_bid_ask: bool = True,
        ticker: Optional[str] = None,
        add_event_markers: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare full feature set and targets.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data with DatetimeIndex
        steps_60m : int, default 12
            Number of forward steps for targets (12 * 5min = 60min)
        keep_only_active : bool, default False
            If True, filter to active session rows only
        add_daily : bool, default True
            Add daily momentum/SMA/RV features
        add_overnight : bool, default True
            Add overnight return feature
        add_deseas : bool, default True
            Add deseasonalized volatility/volume features
        add_time_features : bool, default True
            Add cyclical clock/calendar and session indicator features
        add_resample_precalc : bool, default True
            Add aligned multi-resolution pre-calculated features
        resample_rules : tuple[str], default ("15min","30min","60min")
            Frequencies used for multi-resolution pre-calculations
        rolling_days_deseas : int, default 252
            Rolling window for deseasonalization
        refit_interval : int, default 10
            Refit deseasonalization model every N days
        use_legacy_deseas : bool, default False
            Use legacy (simple) deseasonalization instead of FFF
        apply_scaling : bool, default False
            If True, apply feature scaling for neural network training.
            Scales features to comparable ranges (~[-5, +5]).
        scale_to_basis_points : bool, default True
            If apply_scaling=True, multiply relevant features by 100.
            Only used when apply_scaling=True.
        add_bid_ask : bool, default True
            Add bid/ask volume imbalance features (requires BidVolume/AskVolume
            columns in raw data).
        ticker : str, optional
            Ticker symbol for ticker-specific features (event markers).
        add_event_markers : bool, default True
            Add known-future event release markers (requires ticker).

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, List[str]]
            - df_out: Processed DataFrame
            - train_mask: Boolean mask for valid training rows
            - target_cols: List of target column names
        """
        df = self._standardize_columns(df)

        # Time-of-day slot
        df["tod_slot"] = self._slot_index(df.index).astype(int).values

        # Sessions
        df = self.add_sessions(df)

        # Time features
        if add_time_features:
            df = self.add_time_features(df)

        # Rolling VWAP
        vwap_bars = steps_60m  # Match target horizon
        df = self.add_vwap(df, window_bars=vwap_bars, suffix=f"{vwap_bars * self.bar_minutes}m")

        # Multi-resolution pre-calculations (aligned via ffill)
        if add_resample_precalc:
            df = self.add_resample_precalcs(df, rules=resample_rules)

        # Daily features (safe - uses shift(1))
        if add_daily:
            df = self.add_daily_features(df)

        # Overnight returns
        if add_overnight:
            df = self.add_overnight_returns(df)

        # Deseasonalization
        if add_deseas:
            if use_legacy_deseas or not _HAS_DESEAS:
                df = self.add_deseasonalized_features_legacy(
                    df,
                    rolling_days=rolling_days_deseas,
                    refit_interval=refit_interval,
                )
            else:
                df = self.add_deseasonalized_features(
                    df,
                    rolling_days=rolling_days_deseas,
                    refit_interval=refit_interval,
                    deseasonalize_vol=True,
                    deseasonalize_volume=True,
                    deseasonalize_returns=True,
                    vol_scale_returns=True,
                )

        # Bid/ask volume imbalance
        if add_bid_ask and "BidVolume" in df.columns and "AskVolume" in df.columns:
            df = self.add_bid_ask_volume(df)

        # Known-future event markers
        if add_event_markers and ticker is not None:
            df = self.add_event_markers(df, ticker=ticker)

        # Targets (continuous - do NOT filter before computing)
        df, target_cols = self.add_targets_multistep(df, steps=steps_60m)

        # Training mask: in-session and all targets exist
        train_mask = df["is_active"].astype(bool)
        for c in target_cols:
            train_mask &= df[c].notna()

        if keep_only_active:
            df = df.loc[train_mask].copy()

        # Drop rows with missing OHLCV
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        # Optional feature scaling for neural network training
        if apply_scaling:
            df = self.scale_features(
                df,
                scale_to_basis_points=scale_to_basis_points,
                clip_outliers=True,
                inplace=False,
            )

        return df, train_mask, target_cols


# Backwards compatibility alias
ContinuousIntradayPrepShort = ContinuousIntradayPrep


class FinancialsIntradayPrep(ContinuousIntradayPrep):
    """
    Financials-focused continuous intraday prep with macro market-state features.

    Extends ContinuousIntradayPrep for FinMambaCMD training by:
    - Defaulting to USA session (08:30-16:00)
    - Accepting a daily macro context DataFrame (rates, indices, sectors)
    - Processing macro data through MacroFeaturePrep into engineered features
    - Forward-filling daily macro features onto the intraday bar index
    - Providing get_loaders() to build FinMambaContinuousDataset DataLoaders

    Parameters
    ----------
    sessions : Sequence[SessionSpec], optional
        Session definitions. Default: USA only (08:30-16:00).
    bar_minutes : int, default 5
        Bar frequency in minutes.
    eps : float, default 1e-8
        Small constant for numerical stability.
    macro_df : pd.DataFrame, optional
        Raw daily macro context (e.g. from MacroClient.get_macro_context()).
        Columns like SPX, VIX, YIELD_10Y, sector ETFs, etc.
    macro_prep_kwargs : dict, optional
        Keyword arguments forwarded to MacroFeaturePrep constructor.

    Examples
    --------
    >>> from CTAFlow.data.ext.macro_client import MacroClient
    >>> macro_df = MacroClient().get_macro_context(start_date="2015-01-01")
    >>> prep = FinancialsIntradayPrep(macro_df=macro_df)
    >>> df_out, mask, target_cols = prep.prepare(raw_ohlcv_df, steps_60m=12)
    >>> train_loader, val_loader = prep.get_loaders(
    ...     df=raw_ohlcv_df, raster_npz_path="GC_rasterized.npz",
    ...     val_split=True, batch_size=32,
    ... )
    """

    def __init__(
        self,
        sessions: Optional[Sequence[SessionSpec]] = None,
        bar_minutes: int = 5,
        eps: float = 1e-8,
        macro_df: Optional[pd.DataFrame] = None,
        macro_prep_kwargs: Optional[Dict[str, object]] = None,
        exclude_tickers: Optional[Sequence[str]] = None,
    ):
        if sessions is None:
            sessions = [SessionSpec("USA", "08:30", "16:00")]
        super().__init__(sessions=sessions, bar_minutes=bar_minutes, eps=eps)

        from CTAFlow.features.macro_prep import MacroFeaturePrep

        self._macro_prep = MacroFeaturePrep(**(macro_prep_kwargs or {}))
        self._macro_features: Optional[pd.DataFrame] = None
        self._macro_feature_cols: List[str] = []
        self._exclude_tickers: set = set(exclude_tickers or [])

        if macro_df is not None:
            self.set_macro_data(macro_df)

    # ------------------------------------------------------------------
    # Macro data handling
    # ------------------------------------------------------------------
    def set_macro_data(self, macro_df: pd.DataFrame) -> None:
        """Process raw daily macro context and store engineered features.

        If ``exclude_tickers`` was set at init (e.g. ``["SPX"]`` when
        training ES), the raw columns are dropped **before**
        MacroFeaturePrep so that no derived features (returns, relative
        strength) are generated from the excluded ticker.
        """
        if self._exclude_tickers:
            drop = [c for c in macro_df.columns if c in self._exclude_tickers]
            if drop:
                macro_df = macro_df.drop(columns=drop)
        processed = self._macro_prep.process(macro_df)
        self._macro_features = processed
        self._macro_feature_cols = list(processed.columns)

    @property
    def macro_features(self) -> Optional[pd.DataFrame]:
        """Engineered macro feature DataFrame (daily frequency)."""
        return self._macro_features

    @property
    def macro_feature_cols(self) -> List[str]:
        """Column names of the macro feature set."""
        return list(self._macro_feature_cols)

    def _align_macro_to_intraday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align daily macro features onto the intraday index with 1-day lag.

        Yahoo Finance / FRED data for day T represents the close/settlement
        at 16:00 ET on day T.  Using that data on intraday bars within day T
        would be forward-looking.  We shift by 1 business day so that bars on
        day T only see macro features computed from day T-1's closes.
        """
        if self._macro_features is None or self._macro_features.empty:
            return df

        df = df.copy()
        macro = self._macro_features.copy()
        macro.index = pd.to_datetime(macro.index).normalize()

        # Shift by 1 day: day T bars see day T-1 macro features
        macro = macro.shift(1)

        # Map daily values by date then forward-fill within days
        intraday_dates = df.index.normalize()
        macro_aligned = macro.reindex(intraday_dates)
        macro_aligned.index = df.index
        macro_aligned = macro_aligned.ffill().bfill()

        for col in macro_aligned.columns:
            df[col] = macro_aligned[col].astype(np.float32)

        return df

    @staticmethod
    def _scale_macro_features(
        df: pd.DataFrame,
        macro_cols: List[str],
        clip_range: float = 10.0,
    ) -> pd.DataFrame:
        """Scale macro features to ~[-5, +5] range matching asset feature scale.

        Handles both MarketFeatureEngine columns (returns, VIX) and
        MacroFeaturePrep lean columns (real rates, spreads, econ levels).
        """
        df = df.copy()
        for col in macro_cols:
            if col not in df.columns:
                continue
            cl = col.lower()
            if "_chg" in cl:
                df[col] = (df[col] * 10.0).clip(-clip_range, clip_range)
            elif "_ret_" in cl or "_rel_" in cl:
                df[col] = (df[col] * 100.0).clip(-clip_range, clip_range)
            elif cl == "vix":
                df[col] = (df[col] / 10.0).clip(0, clip_range)
            elif cl == "umcsent":
                df[col] = (df[col] / 20.0).clip(0, clip_range)
            elif cl in ("unrate", "core_inflation", "rgdp_yoy"):
                df[col] = (df[col] / 2.0).clip(-clip_range, clip_range)
            else:
                df[col] = df[col].clip(-clip_range, clip_range)
        return df

    # ------------------------------------------------------------------
    # Override prepare to inject macro features
    # ------------------------------------------------------------------
    def prepare(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare asset features then align and scale macro features.

        Accepts all keyword arguments of ContinuousIntradayPrep.prepare().
        Macro features are scaled when ``apply_scaling=True``.
        """
        apply_scaling = kwargs.get("apply_scaling", False)
        df_out, train_mask, target_cols = super().prepare(df, **kwargs)
        df_out = self._align_macro_to_intraday(df_out)
        if apply_scaling and self._macro_feature_cols:
            df_out = self._scale_macro_features(df_out, self._macro_feature_cols)
        return df_out, train_mask, target_cols

    # ------------------------------------------------------------------
    # get_loaders – build FinMambaContinuousDataset DataLoaders
    # ------------------------------------------------------------------
    def get_loaders(
        self,
        df: pd.DataFrame,
        raster_npz_path: str,
        *,
        # Feature columns (auto-detected if None)
        feature_cols: Optional[List[str]] = None,
        market_feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        # Dataset geometry
        long_lookback: int = 256,
        short_len_range: Tuple[int, int] = (32, 128),
        resample_rule: str = "15min",
        session_start: Optional[str] = None,
        session_end: Optional[str] = None,
        sample_mode: str = "usa",
        sample_stride: int = 1,
        # Classification
        classification: bool = False,
        classification_target: Optional[Union[str, int]] = None,
        classification_thresholds: Sequence[float] = (-0.001, 0.001),
        # Train / val split
        val_split: bool = True,
        val_ratio: float = 0.2,
        val_cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
        # DataLoader
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 0,
        # prepare() passthrough — target horizon
        target_steps: Optional[int] = None,
        target_horizon_minutes: Optional[int] = None,
        # prepare() passthrough — feature toggles
        add_daily: bool = True,
        add_overnight: bool = True,
        add_deseas: bool = True,
        add_time_features: bool = True,
        add_resample_precalc: bool = True,
        resample_rules: Tuple[str, ...] = ("15min", "30min", "60min"),
        apply_scaling: bool = True,
        verbose: bool = False,
    ):
        """Build train (and optionally val) DataLoaders for FinMambaCMD.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data with DatetimeIndex.
        raster_npz_path : str
            Path to rasterized .npz file (keys: 'data', 'idx').
        feature_cols : list[str], optional
            Asset feature columns. Auto-detected from get_feature_cols() if None.
        market_feature_cols : list[str], optional
            Macro/market feature columns. Uses self.macro_feature_cols if None.
        target_cols : list[str], optional
            Target columns. Uses prepare() output if None.
        long_lookback : int, default 256
            Lookback window for the long (asset feature) stream.
        short_len_range : tuple[int, int], default (32, 128)
            Min/max raster sequence length per sample.
        resample_rule : str, default "15min"
            Resampling frequency for dataset alignment.
        session_start, session_end : str, optional
            Session window for raster alignment. Defaults to first session's
            start and last session's end.
        sample_mode : str, default "usa"
            Session filter for eligible samples.
        sample_stride : int, default 1
            Take every N-th sample per day to reduce autocorrelation.
        classification : bool, default False
            Build classification targets via dataset (discretises the
            selected target column using ``classification_thresholds``).
        classification_target : str or int, optional
            Which target column to discretize for classification.
        classification_thresholds : sequence of float
            Threshold cuts for classification labels.
        val_split : bool, default True
            Whether to return separate train and val loaders.
        val_ratio : float, default 0.2
            Fraction of calendar days for validation (ignored if
            val_cutoff_date is set).
        val_cutoff_date : str or Timestamp, optional
            Explicit date cutoff. Dates < cutoff = train, >= cutoff = val.
        batch_size : int, default 32
        shuffle_train : bool, default True
        num_workers : int, default 0
        target_steps : int, optional
            Number of forward bars for multi-step return targets.
            For 15-min bars: 4 steps = 60 min horizon.
            Exactly one of ``target_steps`` or ``target_horizon_minutes``
            must be provided.
        target_horizon_minutes : int, optional
            Forward horizon in minutes. Converted to steps via
            ``target_horizon_minutes // bar_minutes``.
        apply_scaling : bool, default True
            Scale asset and macro features to comparable ranges.
        verbose : bool, default False
            Print alignment statistics.

        Returns
        -------
        DataLoader or Tuple[DataLoader, DataLoader]
            Single loader when val_split=False, else (train_loader, val_loader).
        """
        from CTAFlow.data.datasets.continuous import (
            FinMambaContinuousDataset,
            collate_finmamba_continuous,
        )
        from torch.utils.data import DataLoader

        if self._macro_features is None:
            raise ValueError(
                "No macro data set. Call set_macro_data(macro_df) or pass "
                "macro_df= to the constructor before get_loaders()."
            )

        # --- Resolve target horizon ---
        if target_steps is not None and target_horizon_minutes is not None:
            raise ValueError("Specify only one of target_steps or target_horizon_minutes.")
        if target_horizon_minutes is not None:
            steps = int(target_horizon_minutes) // self.bar_minutes
            if steps < 1:
                raise ValueError(
                    f"target_horizon_minutes={target_horizon_minutes} is less "
                    f"than one bar ({self.bar_minutes} min)."
                )
        elif target_steps is not None:
            steps = int(target_steps)
        else:
            # Default: 60 minutes worth of bars
            steps = 60 // self.bar_minutes

        # --- 1. Prepare asset + macro features ---
        df_out, train_mask, tgt_cols = self.prepare(
            df,
            steps_60m=steps,
            add_daily=add_daily,
            add_overnight=add_overnight,
            add_deseas=add_deseas,
            add_time_features=add_time_features,
            add_resample_precalc=add_resample_precalc,
            resample_rules=resample_rules,
            apply_scaling=apply_scaling,
        )

        if target_cols is None:
            target_cols = tgt_cols

        # --- 2. Resolve feature columns ---
        if feature_cols is None:
            feature_cols = self.get_feature_cols(
                steps_60m=steps,
                bar_minutes=self.bar_minutes,
                add_daily=add_daily,
                add_overnight=add_overnight,
                add_deseas=add_deseas,
                add_time_features=add_time_features,
                add_resample_precalc=add_resample_precalc,
                resample_rules=resample_rules,
            )
            feature_cols = [c for c in feature_cols if c in df_out.columns]

        if market_feature_cols is None:
            market_feature_cols = self.macro_feature_cols
        missing = [c for c in market_feature_cols if c not in df_out.columns]
        if missing:
            raise KeyError(
                f"market_feature_cols not found in prepared DataFrame: {missing}. "
                f"Ensure macro_df covers the date range of df."
            )

        # --- 3. Session window defaults ---
        if session_start is None:
            session_start = self.sessions[0].start
        if session_end is None:
            session_end = self.sessions[-1].end

        # --- 4. Date-based train / val split ---
        if val_split:
            if val_cutoff_date is not None:
                cutoff = pd.Timestamp(val_cutoff_date)
            else:
                dates = sorted(df_out.index.normalize().unique())
                n_val = max(1, int(len(dates) * val_ratio))
                cutoff = dates[-n_val]
            train_df = df_out[df_out.index < cutoff]
            val_df = df_out[df_out.index >= cutoff]
            if verbose:
                print(f"Split: cutoff={cutoff.date()}, "
                      f"train_days={train_df.index.normalize().nunique()}, "
                      f"val_days={val_df.index.normalize().nunique()}")
        else:
            train_df = df_out
            val_df = None

        # --- 5. Shared dataset kwargs ---
        ds_kwargs: Dict[str, object] = dict(
            raster_npz_path=raster_npz_path,
            feature_cols=feature_cols,
            target_cols=target_cols,
            market_feature_cols=market_feature_cols,
            long_lookback=long_lookback,
            short_len_range=short_len_range,
            resample_rule=resample_rule,
            session_start=session_start,
            session_end=session_end,
            sample_mode=sample_mode,
            sample_stride=sample_stride,
            classification=classification,
            classification_target=classification_target,
            classification_thresholds=classification_thresholds,
        )

        # --- 6. Build datasets and loaders ---
        train_ds = FinMambaContinuousDataset(train_df, **ds_kwargs)
        if verbose:
            s = train_ds.alignment_stats
            print(f"Train: {len(train_ds)} samples, "
                  f"{s['num_days_kept']}/{s['num_days_in_common']} days, "
                  f"{len(feature_cols)} asset feats, "
                  f"{len(market_feature_cols)} market feats")

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_finmamba_continuous,
            num_workers=num_workers,
        )

        if not val_split:
            return train_loader

        val_ds = FinMambaContinuousDataset(val_df, **ds_kwargs)
        if verbose:
            s = val_ds.alignment_stats
            print(f"Val:   {len(val_ds)} samples, "
                  f"{s['num_days_kept']}/{s['num_days_in_common']} days")

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_finmamba_continuous,
            num_workers=num_workers,
        )

        return train_loader, val_loader
