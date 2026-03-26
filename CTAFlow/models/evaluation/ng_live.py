"""NatGas-specific live inference interface.

Extends :class:`LiveV3FeatureInterface` with daily EIA storage,
weather (HDD/CDD), and VAE regime features required by the
:class:`~CTAFlow.models.deep_learning.multi_branch.ng_moe.HybridMixtureNetwork`
and :class:`~CTAFlow.models.deep_learning.multi_branch.ng_moe.NatGasMoE` models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from CTAFlow.models.evaluation.live import (
    LiveEvaluationConfig,
    LiveFeatureSnapshot,
    LiveV3FeatureInterface,
    SessionSpec,
    TickDataSource,
    TimestampLike,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default config tuned for NG 30-min inference
# ---------------------------------------------------------------------------

def ng_default_config(**overrides: Any) -> LiveEvaluationConfig:
    """Return a :class:`LiveEvaluationConfig` pre-tuned for NG futures."""
    defaults = dict(
        ticker="NG",
        tick_size=0.001,
        tz="America/Chicago",
        bar_minutes=30,
        target_horizon_minutes=60,
        refresh_interval="30min",
        history_lookback_days=400,
        sessions=[SessionSpec("USA", "08:30", "16:00")],
        vpin_bucket_volume=150,
        vpin_window=60,
        vpin_start_time="08:30",
        vpin_end_time="16:00",
        profile_start_time="02:00",
        profile_end_time="09:30",
        ae_window=21,
        ae_target_time="10:00",
    )
    defaults.update(overrides)
    return LiveEvaluationConfig(**defaults)


# ---------------------------------------------------------------------------
# Daily context cache specification
# ---------------------------------------------------------------------------

@dataclass
class DailyContextPaths:
    """Paths to cached daily context files (refreshed by external cron)."""

    eia_storage_path: Optional[str] = None
    weather_path: Optional[str] = None
    daily_features_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Regime feature spec (12 features expected by VAERegimeEncoder)
# ---------------------------------------------------------------------------

REGIME_FEATURES: List[str] = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "rv_5d",
    "rv_21d",
    "pct_in_5y_band",
    "dev_from_5y_mean_zscore",
    "band_width_pct",
    "forecast_vs_seasonal_zscore",
    "change_vs_seasonal_zscore",
    "is_injection_season",
    "dev_x_season",
]


# ---------------------------------------------------------------------------
# NatGas Live Interface
# ---------------------------------------------------------------------------

class NatGasLiveInterface(LiveV3FeatureInterface):
    """Live feature interface with NG-specific daily context injection.

    Wraps the standard :class:`LiveV3FeatureInterface` refresh cycle and
    appends daily EIA storage, weather, and VAE regime features to the
    snapshot before inference.

    Parameters
    ----------
    config : LiveEvaluationConfig
        Pipeline configuration (use :func:`ng_default_config` for defaults).
    data_source : TickDataSource
        Tick/bar provider (e.g. :class:`IBKRTickDataSource`).
    daily_paths : DailyContextPaths, optional
        Paths to pre-cached daily feature files.
    ae_window : int
        Lookback window for VAE regime encoder input (default 21).
    """

    def __init__(
        self,
        config: LiveEvaluationConfig,
        data_source: TickDataSource,
        daily_paths: Optional[DailyContextPaths] = None,
        ae_window: int = 21,
    ) -> None:
        super().__init__(config, data_source)
        self.daily_paths = daily_paths or DailyContextPaths()
        self.ae_window = ae_window

        # Caches for daily data (loaded lazily or refreshed externally)
        self._eia_cache: Optional[pd.DataFrame] = None
        self._weather_cache: Optional[pd.DataFrame] = None
        self._daily_features_cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_inference_snapshot(
        self,
        now: Optional[TimestampLike] = None,
    ) -> LiveFeatureSnapshot:
        """Build a complete feature snapshot for NG model inference.

        1. Refreshes live tick data and builds technical/orderflow features
           via the parent :meth:`refresh`.
        2. Loads and appends daily EIA/weather context.
        3. Computes the VAE regime encoder input tensor.

        Returns
        -------
        LiveFeatureSnapshot
            Snapshot with ``sample["ae_input"]`` and enriched
            ``sample["tech_features"]`` containing daily context columns.
        """
        snapshot = self.refresh(now)
        anchor_ts = snapshot.anchor_ts

        # Refresh daily caches if date changed
        self._maybe_refresh_daily_caches(anchor_ts)

        # Inject daily context into tech features
        daily_context = self._get_daily_context(anchor_ts)
        if daily_context is not None:
            snapshot.sample["daily_context"] = daily_context

        # Build regime encoder input
        regime_input = self._compute_regime_input(anchor_ts, snapshot.bars)
        if regime_input is not None:
            snapshot.sample["ae_input"] = regime_input

        return snapshot

    # ------------------------------------------------------------------
    # Daily cache management
    # ------------------------------------------------------------------

    def _maybe_refresh_daily_caches(self, anchor_ts: pd.Timestamp) -> None:
        """Reload daily caches if the trading date has changed."""
        current_date = anchor_ts.normalize()
        if self._cache_date is not None and self._cache_date == current_date:
            return

        self._cache_date = current_date

        if self.daily_paths.eia_storage_path:
            self._eia_cache = self._load_eia(self.daily_paths.eia_storage_path)

        if self.daily_paths.weather_path:
            self._weather_cache = self._load_weather(self.daily_paths.weather_path)

        if self.daily_paths.daily_features_path:
            self._daily_features_cache = self._load_daily_features(
                self.daily_paths.daily_features_path
            )

    def reload_daily_caches(self) -> None:
        """Force-reload all daily caches (call after cron refresh)."""
        self._cache_date = None
        if self.last_snapshot is not None:
            self._maybe_refresh_daily_caches(self.last_snapshot.anchor_ts)

    # ------------------------------------------------------------------
    # Daily feature loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_eia(path: str) -> Optional[pd.DataFrame]:
        """Load EIA natural gas storage data from cache file."""
        p = Path(path)
        if not p.exists():
            logger.warning("EIA cache not found: %s", path)
            return None
        try:
            if p.suffix == ".h5":
                df = pd.read_hdf(path)
            elif p.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
            logger.info("Loaded EIA cache: %d rows from %s", len(df), path)
            return df
        except Exception as exc:
            logger.error("Failed to load EIA cache: %s", exc)
            return None

    @staticmethod
    def _load_weather(path: str) -> Optional[pd.DataFrame]:
        """Load HDD/CDD weather data from cache file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Weather cache not found: %s", path)
            return None
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
            logger.info("Loaded weather cache: %d rows from %s", len(df), path)
            return df
        except Exception as exc:
            logger.error("Failed to load weather cache: %s", exc)
            return None

    @staticmethod
    def _load_daily_features(path: str) -> Optional[pd.DataFrame]:
        """Load pre-computed daily features from cache."""
        p = Path(path)
        if not p.exists():
            logger.warning("Daily features cache not found: %s", path)
            return None
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
            logger.info("Loaded daily features: %d rows from %s", len(df), path)
            return df
        except Exception as exc:
            logger.error("Failed to load daily features: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _get_daily_context(self, anchor_ts: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Extract daily EIA/weather context for the anchor date.

        Uses forward-fill logic: if today's data hasn't been released yet,
        the most recent available value is used.

        Returns
        -------
        dict or None
            Dictionary of daily feature name -> value pairs.
        """
        context: Dict[str, float] = {}
        anchor_date = anchor_ts.normalize()

        if self._eia_cache is not None and not self._eia_cache.empty:
            eia = self._eia_cache
            if not isinstance(eia.index, pd.DatetimeIndex):
                eia.index = pd.to_datetime(eia.index)
            # Forward-fill: take most recent row <= anchor_date
            mask = eia.index <= anchor_date
            if mask.any():
                latest = eia.loc[mask].iloc[-1]
                for col in eia.columns:
                    context[f"eia_{col}"] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        if self._weather_cache is not None and not self._weather_cache.empty:
            wx = self._weather_cache
            if not isinstance(wx.index, pd.DatetimeIndex):
                wx.index = pd.to_datetime(wx.index)
            mask = wx.index <= anchor_date
            if mask.any():
                latest = wx.loc[mask].iloc[-1]
                for col in wx.columns:
                    context[f"wx_{col}"] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        if self._daily_features_cache is not None and not self._daily_features_cache.empty:
            df = self._daily_features_cache
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            mask = df.index <= anchor_date
            if mask.any():
                latest = df.loc[mask].iloc[-1]
                for col in df.columns:
                    context[col] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        return context if context else None

    def _compute_regime_input(
        self,
        anchor_ts: pd.Timestamp,
        bars: pd.DataFrame,
    ) -> Optional[torch.Tensor]:
        """Build the VAE regime encoder input: ``(1, ae_window, 12)``.

        Computes the 12 regime features from available bar data and
        daily context caches. Returns ``None`` if insufficient data.
        """
        if bars is None or bars.empty:
            logger.warning("No bar data available for regime computation")
            return None

        close = bars["Close"].dropna()
        if len(close) < self.ae_window + 21:
            logger.warning(
                "Insufficient bar data for regime features: %d bars, need %d",
                len(close),
                self.ae_window + 21,
            )
            return None

        # Resample to daily close prices
        daily_close = close.resample("1D").last().dropna()
        if len(daily_close) < self.ae_window + 21:
            logger.warning("Insufficient daily closes for regime: %d", len(daily_close))
            return None

        # Compute return and volatility features
        log_ret = np.log(daily_close / daily_close.shift(1)).dropna()

        ret_1d = log_ret
        ret_5d = log_ret.rolling(5).sum()
        ret_21d = log_ret.rolling(21).sum()
        rv_5d = log_ret.rolling(5).std() * np.sqrt(252)
        rv_21d = log_ret.rolling(21).std() * np.sqrt(252)

        # Storage-state features from EIA cache
        storage_features = self._get_storage_regime_features(anchor_ts)

        # Seasonal features
        anchor_month = anchor_ts.month
        is_injection = 1.0 if 4 <= anchor_month <= 10 else 0.0

        # Build feature matrix for the last ae_window days
        n = len(daily_close)
        regime_rows = []
        for i in range(max(0, n - self.ae_window), n):
            idx = daily_close.index[i]
            row = np.zeros(12, dtype=np.float32)
            row[0] = ret_1d.get(idx, 0.0)
            row[1] = ret_5d.get(idx, 0.0)
            row[2] = ret_21d.get(idx, 0.0)
            row[3] = rv_5d.get(idx, 0.0)
            row[4] = rv_21d.get(idx, 0.0)

            # Storage features (from daily cache)
            if storage_features is not None:
                row[5] = storage_features.get("pct_in_5y_band", 0.0)
                row[6] = storage_features.get("dev_from_5y_mean_zscore", 0.0)
                row[7] = storage_features.get("band_width_pct", 0.0)
                row[8] = storage_features.get("forecast_vs_seasonal_zscore", 0.0)
                row[9] = storage_features.get("change_vs_seasonal_zscore", 0.0)

            row[10] = is_injection
            # dev_x_season: interaction of deviation and season
            row[11] = row[6] * (1.0 if is_injection else -1.0)

            regime_rows.append(row)

        regime_arr = np.array(regime_rows, dtype=np.float32)

        # Pad if we have fewer than ae_window rows
        if len(regime_arr) < self.ae_window:
            pad = np.zeros(
                (self.ae_window - len(regime_arr), 12), dtype=np.float32
            )
            regime_arr = np.concatenate([pad, regime_arr], axis=0)

        # Replace any NaN/inf with 0
        regime_arr = np.nan_to_num(regime_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(regime_arr).unsqueeze(0)  # (1, ae_window, 12)

    def _get_storage_regime_features(
        self, anchor_ts: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """Extract EIA storage-based regime features for the anchor date."""
        if self._eia_cache is None or self._eia_cache.empty:
            return None

        eia = self._eia_cache
        if not isinstance(eia.index, pd.DatetimeIndex):
            eia.index = pd.to_datetime(eia.index)

        anchor_date = anchor_ts.normalize()
        mask = eia.index <= anchor_date
        if not mask.any():
            return None

        features: Dict[str, float] = {}
        for col in [
            "pct_in_5y_band",
            "dev_from_5y_mean_zscore",
            "band_width_pct",
            "forecast_vs_seasonal_zscore",
            "change_vs_seasonal_zscore",
        ]:
            if col in eia.columns:
                val = eia.loc[mask, col].iloc[-1]
                features[col] = float(val) if pd.notna(val) else 0.0

        return features if features else None
