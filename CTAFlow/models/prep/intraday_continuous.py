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
from datetime import time, timedelta
from typing import List, Optional, Sequence, Tuple, Union

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
                    ]
                )

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
            cl = c.lower()
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

            rs_feats = pd.DataFrame(
                {
                    f"rs_{suffix}_logret": rs_logret,
                    f"rs_{suffix}_range": rs_range,
                    f"rs_{suffix}_vwap_dist": rs_vwap_dist,
                    f"rs_{suffix}_logvol_z": rs_logvol_z,
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
                df["deseasonalized_vol"] = result["adjusted"].reindex(df.index)
                df["seasonal_vol_factor"] = result["seasonal"].reindex(df.index)

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
                df["deseasonalized_volume"] = result["adjusted"].reindex(df.index)
                df["seasonal_volume_factor"] = result["seasonal"].reindex(df.index)
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
            lambda x: x.rolling(rolling_days, min_periods=max(10, rolling_days // 10)).mean()
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


# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    # df = your 5-min OHLCV dataframe with DatetimeIndex
    prep = ContinuousIntradayPrep(
        sessions=[
            SessionSpec("LONDON", "02:00", "11:00"),
            SessionSpec("USA", "08:30", "16:00"),
        ],
        bar_minutes=5,
    )

    # Example with dummy data
    print("ContinuousIntradayPrep initialized")
    print(f"Sessions: {[s.name for s in prep.sessions]}")
    print(f"Bar minutes: {prep.bar_minutes}")

    # To use:
    # df_out, mask, target_cols = prep.prepare(
    #     df,
    #     steps_60m=12,
    #     keep_only_active=False,
    #     add_daily=True,
    #     add_overnight=True,
    #     add_deseas=True,
    # )
    # train_df = df_out.loc[mask]
