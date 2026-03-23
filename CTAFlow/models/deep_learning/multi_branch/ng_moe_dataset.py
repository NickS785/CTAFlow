"""
NGMoEDataset -- Data preparation for NatGasMoE training
=======================================================

Two aligned feature streams built from price + EIA storage:

  x_seq    (seq_len, n_feat)   -- price technicals, momentum, vol
  ae_input (ae_window, 12)     -- regime features driven by storage state

Target: **1-day forward log return** (price_return) or storage_change.
EIA storage data feeds regime features and daily storage inputs, not the target.

Regime features (f_ae = 12):
  Returns & Vol:   ret_1d, ret_5d, ret_21d, rv_5d, rv_21d
  Storage state:   pct_in_5y_band, dev_from_5y_mean_zscore, band_width_pct
  Forecast:        forecast_vs_seasonal_zscore, change_vs_seasonal_zscore
  Seasonal:        is_injection_season, dev_x_season
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NGMoEDataConfig:
    """Settings for feature construction and windowing."""

    # Windowing
    seq_len: int = 20          # lookback window for x_seq (business days)
    ae_window: int = 21        # lookback window for regime encoder

    # Target
    target: str = "price_return"  # storage_change | surprise | price_return
    target_horizon: int = 1       # forward days for price_return target (1=next day, 5=week)

    # Classification mode
    n_classes: int = 0            # 0 = regression, 4 = quartile classification
    class_boundaries: Optional[List[float]] = None  # custom quantile boundaries

    # Technical indicator windows
    vol_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    momentum_windows: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    lookback_lags: int = 5

    # Volatility scaling
    vol_span: int = 63         # EWM span for target vol-scaling


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

REGIME_COLS = [
    "regime_ret_1d", "regime_ret_5d", "regime_ret_21d",
    "regime_rv_5d", "regime_rv_21d",
    "regime_pct_in_5y_band", "regime_dev_5y_zscore", "regime_band_width_pct",
    "regime_fc_vs_seasonal_z", "regime_chg_vs_seasonal_z",
    "regime_is_injection", "regime_dev_x_season",
]


class NGMoEDataBuilder:
    """Build a daily DataFrame with technical + regime + target columns.

    Usage::

        builder = NGMoEDataBuilder()
        daily = builder.build(price_df, storage_wkly)
        ds = NGMoEWindowDataset(daily, builder.feature_cols, cfg=builder.config)
    """

    def __init__(self, config: NGMoEDataConfig | None = None):
        self.config = config or NGMoEDataConfig()
        self.feature_cols: List[str] = []
        self.regime_cols: List[str] = list(REGIME_COLS)
        self.feature_groups: Dict[str, List[str]] = {}
        self._current_group: Optional[str] = None

    # ------------------------------------------------------------------ build
    def build(
        self,
        price_df: pd.DataFrame,
        storage_wkly: pd.DataFrame,
        sarimax_features: pd.DataFrame | None = None,
        daily_weather: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return a daily DataFrame ready for windowed sampling.

        Parameters
        ----------
        price_df : OHLCV with DatetimeIndex (yfinance or similar).
        storage_wkly : Weekly EIA storage with ``storage_level``,
            ``storage_change``, and optionally ``consensus_est``, ``surprise``.
        sarimax_features : Output of ``NatGasStorageForecaster.build_features``
            (optional).  Used for forecast-vs-seasonal regime features.
        daily_weather : Output of ``PopulationWeatherGrid.get_weighted_daily()``
            (optional).  Adds degree-day features with spline HDD basis.
        """
        df = price_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        # Base return
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        # --- Technical features (x_seq) ---
        self.feature_cols = []
        self.feature_groups = {}

        self._begin_group("returns")
        self._add_return_features(df)
        self._end_group("returns")

        self._begin_group("volatility")
        self._add_volatility_features(df)
        self._end_group("volatility")

        self._begin_group("momentum")
        self._add_momentum_features(df)
        self._end_group("momentum")

        self._begin_group("microstructure")
        self._add_microstructure_features(df)
        self._end_group("microstructure")

        self._begin_group("temporal")
        self._add_temporal_features(df)
        self._end_group("temporal")

        self._begin_group("storage")
        self._add_storage_daily_features(df, storage_wkly)
        self._end_group("storage")

        # --- Weather features (degree days + spline HDD) ---
        if daily_weather is not None:
            self._begin_group("weather")
            self._add_weather_features(df, daily_weather)
            self._end_group("weather")

        # --- Regime features (ae_input, storage-driven) ---
        self._add_regime_features(df, storage_wkly, sarimax_features)

        # --- Target ---
        self._add_target(df, storage_wkly)

        # Drop warm-up NaNs
        keep = self.feature_cols + self.regime_cols + ["target", "target_std"]
        if "target_class" in df.columns:
            keep.append("target_class")
        keep = [c for c in keep if c in df.columns]
        df = df.dropna(subset=keep)

        return df

    # ============================================================ Returns
    def _add_return_features(self, df: pd.DataFrame) -> None:
        r = df["log_ret"]

        # Lagged returns
        for lag in range(1, self.config.lookback_lags + 1):
            col = f"ret_lag_{lag}"
            df[col] = r.shift(lag)
            self.feature_cols.append(col)

        # Cumulative returns
        for w in [5, 10, 20]:
            col = f"cum_ret_{w}d"
            df[col] = r.rolling(w).sum()
            self.feature_cols.append(col)

        # Higher moments
        for w in [20, 60]:
            sk = f"ret_skew_{w}d"
            ku = f"ret_kurt_{w}d"
            df[sk] = r.rolling(w).skew()
            df[ku] = r.rolling(w).kurt()
            self.feature_cols.extend([sk, ku])

    # ============================================================ Volatility
    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        r = df["log_ret"]
        sq = r ** 2

        # Realized vol at multiple horizons
        for w in self.config.vol_windows:
            col = f"rv_{w}d"
            df[col] = np.sqrt(sq.rolling(w).mean()) * math.sqrt(252)
            self.feature_cols.append(col)

        # Vol ratio (short / long regime signal)
        if 5 in self.config.vol_windows and 60 in self.config.vol_windows:
            df["vol_ratio"] = df["rv_5d"] / df["rv_60d"].clip(lower=1e-8)
            self.feature_cols.append("vol_ratio")

        # Vol-of-vol
        df["vol_of_vol"] = df["rv_5d"].rolling(20).std()
        self.feature_cols.append("vol_of_vol")

        # Parkinson estimator
        if {"high", "low"}.issubset(df.columns):
            hl = np.log(df["high"] / df["low"])
            df["parkinson_vol"] = (
                hl.pow(2).rolling(20).mean()
                / (4 * math.log(2))
            ).apply(np.sqrt) * math.sqrt(252)
            self.feature_cols.append("parkinson_vol")

        # Garman-Klass
        if {"open", "high", "low", "close"}.issubset(df.columns):
            hl2 = np.log(df["high"] / df["low"]) ** 2
            co2 = np.log(df["close"] / df["open"]) ** 2
            df["gk_vol"] = np.sqrt(
                (0.5 * hl2 - (2 * math.log(2) - 1) * co2).rolling(20).mean()
            ) * math.sqrt(252)
            self.feature_cols.append("gk_vol")

        # ATR (14-period)
        if {"high", "low", "close"}.issubset(df.columns):
            prev_c = df["close"].shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - prev_c).abs(),
                (df["low"] - prev_c).abs(),
            ], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14).mean()
            # Normalise by close for scale-invariance
            df["atr_14_pct"] = df["atr_14"] / df["close"]
            self.feature_cols.append("atr_14_pct")

    # ============================================================ Momentum
    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        c = df["close"]
        r = df["log_ret"]

        # RSI-14
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss.clip(lower=1e-10)
        df["rsi_14"] = 100 - 100 / (1 + rs)
        self.feature_cols.append("rsi_14")

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (macd_line - macd_signal) / c  # normalised
        self.feature_cols.append("macd_hist")

        # Bollinger %B and width
        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        df["bb_pctb"] = (c - lower) / (upper - lower).clip(lower=1e-8)
        df["bb_width"] = (upper - lower) / ma20
        self.feature_cols.extend(["bb_pctb", "bb_width"])

        # Rate of change
        for w in self.config.momentum_windows:
            col = f"roc_{w}d"
            df[col] = c.pct_change(w)
            self.feature_cols.append(col)

        # Z-score vs moving average
        for w in [20, 60]:
            col = f"zscore_ma_{w}d"
            ma = c.rolling(w).mean()
            sd = c.rolling(w).std()
            df[col] = (c - ma) / sd.clip(lower=1e-8)
            self.feature_cols.append(col)

    # ============================================================ Micro
    def _add_microstructure_features(self, df: pd.DataFrame) -> None:
        if "volume" not in df.columns:
            return
        v = df["volume"].replace(0, np.nan)
        df["vol_ratio_20"] = v / v.rolling(20).mean().clip(lower=1)
        df["vol_zscore_60"] = (
            (v - v.rolling(60).mean()) / v.rolling(60).std().clip(lower=1)
        )
        self.feature_cols.extend(["vol_ratio_20", "vol_zscore_60"])

    # ============================================================ Temporal
    def _add_temporal_features(self, df: pd.DataFrame) -> None:
        idx = df.index
        doy = idx.dayofyear
        dow = idx.dayofweek

        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
        df["woy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["woy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        df["is_thursday"] = (dow == 3).astype(np.float32)

        self.feature_cols.extend([
            "month_sin", "month_cos", "dow_sin", "dow_cos",
            "woy_sin", "woy_cos", "is_thursday",
        ])

    # ============================================================ Storage daily
    def _add_storage_daily_features(
        self, df: pd.DataFrame, storage_wkly: pd.DataFrame,
    ) -> None:
        """Forward-fill weekly storage columns to the daily grid."""
        sw = storage_wkly.copy()

        # 4-week rolling features
        sl = sw["storage_level"]
        sw["sl_4wk_mean"] = sl.rolling(4, min_periods=2).mean()
        sw["sl_4wk_max"] = sl.rolling(4, min_periods=2).max()
        sw["sl_4wk_min"] = sl.rolling(4, min_periods=2).min()
        sw["sl_change_4wk_mean"] = sw["storage_change"].rolling(4, min_periods=2).mean()

        # Forward-fill to daily
        cols_to_ffill = [
            "storage_level", "storage_change",
            "sl_4wk_mean", "sl_4wk_max", "sl_4wk_min", "sl_change_4wk_mean",
        ]
        for extra in ["consensus_est", "surprise"]:
            if extra in sw.columns:
                cols_to_ffill.append(extra)

        daily_storage = sw[cols_to_ffill].reindex(
            sw.index.union(df.index)
        ).sort_index().ffill().reindex(df.index)

        for col in daily_storage.columns:
            df[col] = daily_storage[col].values
            self.feature_cols.append(col)

    # ============================================================ Weather
    def _add_weather_features(
        self, df: pd.DataFrame, daily_weather: pd.DataFrame,
    ) -> None:
        """Add population-weighted degree-day features with spline HDD basis.

        Parameters
        ----------
        daily_weather : Output of ``PopulationWeatherGrid.get_weighted_daily()``
            with at least ``wtd_TAVG`` column.  Index is DatetimeIndex.
        """
        from MacrOSINT.models.energy.natgas_storage_forecast import (
            compute_degree_days,
            compute_spline_hdd_basis,
        )

        # Compute daily degree days
        dd = compute_degree_days(daily_weather)  # HDD, CDD, HDD_mild, HDD_extreme
        dd = dd.reindex(df.index)

        # Daily HDD and CDD
        df["dd_hdd"] = dd["HDD"].values
        df["dd_cdd"] = dd["CDD"].values
        self.feature_cols.extend(["dd_hdd", "dd_cdd"])

        # Rolling 7-day sums (weekly demand signal aligned to daily grid)
        df["dd_hdd_7d"] = dd["HDD"].rolling(7, min_periods=3).sum().values
        df["dd_cdd_7d"] = dd["CDD"].rolling(7, min_periods=3).sum().values
        self.feature_cols.extend(["dd_hdd_7d", "dd_cdd_7d"])

        # Spline HDD basis on rolling 7-day HDD (captures non-linear cold response)
        hdd_7d = pd.Series(df["dd_hdd_7d"].values, index=df.index).fillna(0)
        try:
            spline_df, self._spline_transformer = compute_spline_hdd_basis(
                hdd_7d,
                n_knots=4,
                transformer=getattr(self, "_spline_transformer", None),
            )
            for col in spline_df.columns:
                df[col] = spline_df[col].values
                self.feature_cols.append(col)
        except Exception:
            pass  # sklearn not available -- skip spline

        # HDD/CDD momentum (weekly change)
        df["dd_hdd_7d_chg"] = df["dd_hdd_7d"] - pd.Series(
            df["dd_hdd_7d"].values, index=df.index
        ).shift(7).values
        df["dd_cdd_7d_chg"] = df["dd_cdd_7d"] - pd.Series(
            df["dd_cdd_7d"].values, index=df.index
        ).shift(7).values
        self.feature_cols.extend(["dd_hdd_7d_chg", "dd_cdd_7d_chg"])

        # Population-weighted temperature (raw signal)
        if "wtd_TAVG" in daily_weather.columns:
            tavg = daily_weather["wtd_TAVG"].reindex(df.index)
            df["wtd_tavg"] = tavg.values
            df["wtd_tavg_7d_ma"] = tavg.rolling(7, min_periods=3).mean().values
            self.feature_cols.extend(["wtd_tavg", "wtd_tavg_7d_ma"])

    # ============================================================ Regime
    def _add_regime_features(
        self,
        df: pd.DataFrame,
        storage_wkly: pd.DataFrame,
        sarimax_features: pd.DataFrame | None = None,
    ) -> None:
        """Build the 12 regime features that inform the VAE encoder.

        Storage state drives the regime — returns/vol provide market context
        but the storage position within the 5-year band, the forecast error,
        and the seasonal phase are the primary regime discriminators.
        """
        r = df["log_ret"]

        # --- Returns & vol context (5 features) ---
        df["regime_ret_1d"] = r
        df["regime_ret_5d"] = r.rolling(5).sum()
        df["regime_ret_21d"] = r.rolling(21).sum()
        df["regime_rv_5d"] = np.sqrt((r ** 2).rolling(5).mean()) * math.sqrt(252)
        df["regime_rv_21d"] = np.sqrt((r ** 2).rolling(21).mean()) * math.sqrt(252)

        # --- Storage state (3 features) ---
        sl = storage_wkly["storage_level"]
        wk_idx = sl.index.isocalendar().week.values

        # 5-year band per ISO week (causal: expanding over prior years)
        hi_5y = pd.Series(np.nan, index=sl.index)
        lo_5y = pd.Series(np.nan, index=sl.index)
        mean_5y = pd.Series(np.nan, index=sl.index)

        for w in range(1, 54):
            mask = wk_idx == w
            if mask.sum() < 2:
                continue
            idx_pos = np.where(mask)[0]
            vals = sl.iloc[idx_pos]
            hi_5y.iloc[idx_pos] = vals.expanding().max().shift(1).values
            lo_5y.iloc[idx_pos] = vals.expanding().min().shift(1).values
            mean_5y.iloc[idx_pos] = vals.expanding().mean().shift(1).values

        # Fill first occurrences
        hi_5y = hi_5y.ffill().bfill()
        lo_5y = lo_5y.ffill().bfill()
        mean_5y = mean_5y.ffill().bfill()

        band_w = (hi_5y - lo_5y).clip(lower=1)
        pct_band = (sl - lo_5y) / band_w
        dev_zscore = (sl - mean_5y) / band_w

        # Forward-fill weekly regime storage features to daily
        regime_wkly = pd.DataFrame({
            "regime_pct_in_5y_band": pct_band.values,
            "regime_dev_5y_zscore": dev_zscore.values,
            "regime_band_width_pct": (band_w / mean_5y.clip(lower=1)).values,
        }, index=sl.index)

        regime_daily = regime_wkly.reindex(
            regime_wkly.index.union(df.index)
        ).sort_index().ffill().reindex(df.index)

        for col in regime_daily.columns:
            df[col] = regime_daily[col].values

        # --- Forecast vs seasonal (2 features) ---
        sc = storage_wkly["storage_change"]
        # Seasonal mean change per ISO week (expanding, causal)
        sea_chg = pd.Series(np.nan, index=sc.index)
        for w in range(1, 54):
            mask = wk_idx == w
            if mask.sum() < 2:
                continue
            idx_pos = np.where(mask)[0]
            sea_chg.iloc[idx_pos] = (
                sc.iloc[idx_pos].expanding().mean().shift(1).values
            )
        sea_chg = sea_chg.ffill().bfill()
        sea_std = (sc - sea_chg).expanding().std().clip(lower=1)

        chg_vs_sea = (sc - sea_chg) / sea_std

        # If SARIMAX forecast available, use it; else use consensus or seasonal
        if sarimax_features is not None and "forecast" in sarimax_features.columns:
            fc = sarimax_features["forecast"].reindex(sc.index).ffill()
        elif "consensus_est" in storage_wkly.columns:
            fc = storage_wkly["consensus_est"]
        else:
            fc = sea_chg
        fc_vs_sea = (fc - sea_chg) / sea_std

        fc_regime = pd.DataFrame({
            "regime_fc_vs_seasonal_z": fc_vs_sea.values,
            "regime_chg_vs_seasonal_z": chg_vs_sea.values,
        }, index=sc.index)

        fc_daily = fc_regime.reindex(
            fc_regime.index.union(df.index)
        ).sort_index().ffill().reindex(df.index)
        for col in fc_daily.columns:
            df[col] = fc_daily[col].values

        # --- Seasonal flags (2 features) ---
        month = df.index.month
        # Injection season: April (4) through October (10)
        df["regime_is_injection"] = ((month >= 4) & (month <= 10)).astype(np.float32)
        # Interaction: deviation x season sign (+1 injection, -1 withdrawal)
        season_sign = np.where(df["regime_is_injection"].values > 0.5, 1.0, -1.0)
        df["regime_dev_x_season"] = df["regime_dev_5y_zscore"] * season_sign

    # ============================================================ Target
    def _add_target(
        self, df: pd.DataFrame, storage_wkly: pd.DataFrame,
    ) -> None:
        """Align target to each daily row.

        For ``storage_change`` target: the next Thursday EIA report value,
        forward-filled to daily so every day in a given week has the same
        target (the upcoming report).

        For ``price_return`` target: forward log return over ``target_horizon`` days.

        If ``n_classes > 0``, also computes ``target_class`` using expanding
        quantile boundaries (causal -- no lookahead).
        """
        cfg = self.config

        if cfg.target in ("storage_change", "surprise"):
            col = cfg.target if cfg.target in storage_wkly.columns else "storage_change"
            tgt_wkly = storage_wkly[col].shift(-1)
            tgt_daily = tgt_wkly.reindex(
                tgt_wkly.index.union(df.index)
            ).sort_index().ffill().reindex(df.index)
            df["target"] = tgt_daily.values

            sc = storage_wkly["storage_change"]
            sc_std = sc.rolling(8, min_periods=4).std()
            std_daily = sc_std.reindex(
                sc_std.index.union(df.index)
            ).sort_index().ffill().reindex(df.index)
            df["target_std"] = std_daily.values
        else:
            h = cfg.target_horizon
            if h == 1:
                df["target"] = df["log_ret"].shift(-1)
            else:
                df["target"] = df["log_ret"].rolling(h).sum().shift(-h)

            daily_vol = df["log_ret"].rolling(21, min_periods=10).std()
            df["target_std"] = daily_vol * np.sqrt(h)

        # --- Classification target (expanding quantile, causal) ---
        if cfg.n_classes > 0:
            df["target_class"] = self._compute_target_classes(
                df["target"], cfg.n_classes, cfg.class_boundaries,
            )

    @staticmethod
    def _compute_target_classes(
        target: pd.Series,
        n_classes: int,
        fixed_boundaries: Optional[List[float]] = None,
    ) -> pd.Series:
        """Assign each return to a class via expanding quantile boundaries.

        Default 4 classes: <25th, 25-50th, 50-75th, >75th percentile.
        Uses expanding window so boundaries are causal (no lookahead).
        """
        classes = pd.Series(np.nan, index=target.index, dtype=np.float32)
        if fixed_boundaries is not None:
            boundaries = fixed_boundaries
            for i, val in enumerate(target):
                if np.isnan(val):
                    continue
                cls = n_classes - 1
                for j, b in enumerate(boundaries):
                    if val < b:
                        cls = j
                        break
                classes.iloc[i] = cls
        else:
            quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]  # e.g. [0.25, 0.5, 0.75]
            min_obs = max(50, n_classes * 10)
            vals = target.values.astype(np.float64)
            for i in range(len(vals)):
                if np.isnan(vals[i]):
                    continue
                history = vals[:i]
                history = history[np.isfinite(history)]
                if len(history) < min_obs:
                    continue
                boundaries = np.quantile(history, quantiles)
                cls = n_classes - 1
                for j, b in enumerate(boundaries):
                    if vals[i] < b:
                        cls = j
                        break
                classes.iloc[i] = cls
        return classes

    # ============================================================ Group tracking
    def _begin_group(self, name: str) -> None:
        """Mark the start of a feature group for variable selection."""
        self._current_group = name
        self._group_start_idx = len(self.feature_cols)

    def _end_group(self, name: str) -> None:
        """Record which feature_cols belong to this group."""
        cols = self.feature_cols[self._group_start_idx:]
        if cols:
            self.feature_groups[name] = list(cols)
        self._current_group = None

    # ============================================================ Helpers
    def get_monday_mask(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask for Monday rows (forecast origin)."""
        return df.index.dayofweek == 0


# ---------------------------------------------------------------------------
# Windowed PyTorch Dataset
# ---------------------------------------------------------------------------

class NGMoEWindowDataset(Dataset):
    """Sliding-window dataset that yields (x_seq, ae_input, y_ret, y_std)
    or (x_seq, ae_input, y_ret, y_std, y_class) when classification is enabled.

    Parameters
    ----------
    daily_df : Output of ``NGMoEDataBuilder.build()``.
    feature_cols : Technical feature column names (for x_seq).
    regime_cols : Regime feature column names (for ae_input, default REGIME_COLS).
    cfg : NGMoEDataConfig for window sizes.
    monday_only : If True, only sample windows ending on a Monday.
    """

    def __init__(
        self,
        daily_df: pd.DataFrame,
        feature_cols: List[str],
        regime_cols: List[str] | None = None,
        cfg: NGMoEDataConfig | None = None,
        monday_only: bool = True,
    ):
        cfg = cfg or NGMoEDataConfig()
        regime_cols = regime_cols or list(REGIME_COLS)

        self.seq_len = cfg.seq_len
        self.ae_window = cfg.ae_window
        self.monday_only = monday_only
        self.n_classes = cfg.n_classes
        warmup = max(self.seq_len, self.ae_window)

        # Convert to arrays
        self.X = daily_df[feature_cols].values.astype(np.float32)
        self.R = daily_df[regime_cols].values.astype(np.float32)
        self.y_ret = daily_df["target"].values.astype(np.float32)
        self.y_std = daily_df["target_std"].values.astype(np.float32)
        self.has_classes = "target_class" in daily_df.columns and cfg.n_classes > 0
        if self.has_classes:
            self.y_class = daily_df["target_class"].values.astype(np.int64)
        self.dates = daily_df.index

        # Valid indices (enough lookback + non-NaN target)
        valid = np.arange(warmup, len(daily_df))
        if monday_only:
            is_mon = daily_df.index.dayofweek.values == 0
            valid = valid[is_mon[valid]]

        # Remove any with NaN target
        ok = np.isfinite(self.y_ret[valid]) & np.isfinite(self.y_std[valid])
        if self.has_classes:
            ok &= np.isfinite(self.y_class[valid].astype(np.float64))
        self.indices = valid[ok]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t = self.indices[idx]

        x_seq = torch.from_numpy(
            self.X[t - self.seq_len: t]          # (seq_len, n_feat)
        )
        ae_input = torch.from_numpy(
            self.R[t - self.ae_window: t]        # (ae_window, 12)
        )
        y_ret = torch.tensor(self.y_ret[t])
        y_std = torch.tensor(self.y_std[t])

        if self.has_classes:
            y_class = torch.tensor(self.y_class[t], dtype=torch.long)
            return x_seq, ae_input, y_ret, y_std, y_class

        return x_seq, ae_input, y_ret, y_std

    def get_date(self, idx: int) -> pd.Timestamp:
        return self.dates[self.indices[idx]]


# ---------------------------------------------------------------------------
# Convenience: split into train / val / test
# ---------------------------------------------------------------------------

def build_datasets(
    price_df: pd.DataFrame,
    storage_wkly: pd.DataFrame,
    sarimax_features: pd.DataFrame | None = None,
    daily_weather: pd.DataFrame | None = None,
    config: NGMoEDataConfig | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    monday_only: bool = True,
) -> Tuple[NGMoEWindowDataset, NGMoEWindowDataset, NGMoEWindowDataset, Dict]:
    """End-to-end: raw data -> three Datasets + metadata dict.

    Returns
    -------
    train_ds, val_ds, test_ds, meta

    ``meta`` contains:
      - ``feature_cols``: list of technical feature names
      - ``regime_cols``: list of regime feature names
      - ``n_features``: int, input dim for MoEConfig
      - ``daily_df``: the full daily DataFrame (for inspection)
      - ``builder``: the NGMoEDataBuilder instance
    """
    config = config or NGMoEDataConfig()
    builder = NGMoEDataBuilder(config)
    daily = builder.build(price_df, storage_wkly, sarimax_features, daily_weather)

    # Chronological split on the full daily df
    n = len(daily)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)

    train_df = daily.iloc[:n_tr]
    val_df = daily.iloc[n_tr: n_tr + n_va]
    test_df = daily.iloc[n_tr + n_va:]

    kw = dict(
        feature_cols=builder.feature_cols,
        regime_cols=builder.regime_cols,
        cfg=config,
        monday_only=monday_only,
    )
    train_ds = NGMoEWindowDataset(train_df, **kw)
    val_ds = NGMoEWindowDataset(val_df, **kw)
    test_ds = NGMoEWindowDataset(test_df, **kw)

    meta = {
        "feature_cols": builder.feature_cols,
        "regime_cols": builder.regime_cols,
        "feature_groups": builder.feature_groups,
        "n_features": len(builder.feature_cols),
        "daily_df": daily,
        "builder": builder,
        "splits": {
            "train": (train_df.index[0], train_df.index[-1], len(train_ds)),
            "val": (val_df.index[0], val_df.index[-1], len(val_ds)),
            "test": (test_df.index[0], test_df.index[-1], len(test_ds)),
        },
    }
    return train_ds, val_ds, test_ds, meta
