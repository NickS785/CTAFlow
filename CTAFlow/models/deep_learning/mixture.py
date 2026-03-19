"""
Mixture Density Network (MDN) for Commodity Return Forecasting.

Architecture:
    Input → Shared Trunk (FC + BN + SiLU + Dropout) → MDN Head → {π, μ, σ} for K Gaussians

The network outputs a full conditional density p(r_t | X_t) as a Gaussian mixture,
enabling regime-aware probabilistic forecasting.

Feature engineering uses deseasonalized returns (monthly mean subtraction via
CTAFlow.utils.seasonal.deseasonalize_monthly) and extends the volatility set with
Garman-Klass and Yang-Zhang estimators beyond the standard Parkinson estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from CTAFlow.utils.seasonal import deseasonalize_monthly


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MDNConfig:
    """Full configuration for the MDN pipeline."""
    # Architecture
    n_components: int = 5
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    use_batch_norm: bool = True
    activation: str = "silu"           # silu | relu | gelu

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    lr_scheduler: str = "cosine"       # cosine | plateau | step
    grad_clip: float = 1.0
    min_sigma: float = 1e-4
    sigma_bias_init: float = 0.5
    entropy_weight: float = 0.1        # Weight for entropy regularization term

    # Feature engineering
    lookback_lags: int = 5             # Number of individual lagged returns
    vol_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    momentum_windows: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    include_fundamentals: bool = True
    include_cross_asset: bool = True
    deseasonalize_returns: bool = True  # Apply monthly deseasonalization

    # Walk-forward
    train_window: int = 504
    val_window: int = 63
    test_window: int = 21

    # Target horizon
    target_horizon: int = 1         # 1 = next-day, 5 = weekly, 21 = monthly forward return

    # Signal generation
    signal_confidence: float = 0.6

    seed: int = 42


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class NGFeatureEngine:
    """
    Daily feature engine for commodity return forecasting.

    Expects a DataFrame with at minimum:
      - 'close': settlement price
      - 'high', 'low' (optional): for range-based vol estimators
      - 'open' (optional): for Yang-Zhang estimator
      - 'volume', 'open_interest' (optional)

    Fundamental columns (optional):
      - 'storage_actual', 'storage_expected': EIA weekly storage
      - 'hdd', 'cdd': degree-day weather proxies

    Cross-asset columns (optional):
      - 'crude_close', 'usd_index'

    Deseasonalization:
      Monthly mean subtraction is applied to log returns via
      CTAFlow.utils.seasonal.deseasonalize_monthly, removing the
      dominant seasonal drift in the return target and lagged features.
    """

    def __init__(self, config: MDNConfig):
        self.config = config
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features from raw OHLCV data.

        Returns a DataFrame with feature columns + 'target' (next-period
        deseasonalized log return). Rows with NaN are dropped.
        """
        df = df.copy().sort_index()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Monthly deseasonalization on the return series
        if self.config.deseasonalize_returns and len(df) > 24:
            ret_arr = df["log_return"].values.reshape(-1, 1).astype(float)
            deseas = deseasonalize_monthly(ret_arr, pd.DatetimeIndex(df.index))
            df["log_return_ds"] = deseas[:, 0]
        else:
            df["log_return_ds"] = df["log_return"]

        features = pd.DataFrame(index=df.index)
        features = self._add_return_features(df, features)
        features = self._add_volatility_features(df, features)
        features = self._add_momentum_features(df, features)
        features = self._add_microstructure_features(df, features)
        features = self._add_temporal_features(df, features)

        if self.config.include_fundamentals:
            features = self._add_fundamental_features(df, features)
        if self.config.include_cross_asset:
            features = self._add_cross_asset_features(df, features)

        # Target: forward return over target_horizon days (log-additive, deseasonalized)
        h = self.config.target_horizon
        if h == 1:
            fwd_ret = df["log_return_ds"].shift(-1)
        else:
            # rolling(h).sum() at t = sum of [t-h+1 .. t]; shift(-h) maps t → [t+1 .. t+h]
            fwd_ret = df["log_return_ds"].rolling(h).sum().shift(-h)
        features["target"] = fwd_ret

        self._feature_names = [c for c in features.columns if c != "target"]
        return features.dropna()

    # --- Return features ---

    def _add_return_features(self, df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        ret = df["log_return_ds"]

        for lag in range(1, self.config.lookback_lags + 1):
            feat[f"ret_lag_{lag}"] = ret.shift(lag)

        for w in [5, 10, 20]:
            feat[f"cum_ret_{w}d"] = ret.rolling(w).sum()

        for w in [20, 60]:
            feat[f"ret_skew_{w}d"] = ret.rolling(w).skew()
            feat[f"ret_kurt_{w}d"] = ret.rolling(w).kurt()

        return feat

    # --- Volatility features ---

    def _add_volatility_features(self, df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        ret = df["log_return"]  # Use raw returns for vol estimators

        # Realized vol at multiple horizons (annualized)
        for w in self.config.vol_windows:
            feat[f"rvol_{w}d"] = ret.rolling(w).std() * math.sqrt(252)

        # Vol ratio: short / long regime signal
        if len(self.config.vol_windows) >= 2:
            sw, lw = self.config.vol_windows[0], self.config.vol_windows[-1]
            feat["vol_ratio"] = (
                ret.rolling(sw).std() / ret.rolling(lw).std().replace(0, np.nan)
            )

        # Vol-of-vol: 20d rolling std of 5d vol
        rv5 = ret.rolling(5).std()
        feat["vol_of_vol"] = rv5.rolling(20).std()

        has_hl = "high" in df.columns and "low" in df.columns
        has_open = "open" in df.columns

        if has_hl:
            # Parkinson estimator (high/low range)
            hl = np.log(df["high"] / df["low"])
            parkinson_sq = hl ** 2 / (4 * math.log(2))
            feat["parkinson_vol"] = (
                parkinson_sq.rolling(20).mean().apply(np.sqrt) * math.sqrt(252)
            )

            # Garman-Klass estimator
            co = np.log(df["close"] / df["close"].shift(1))
            feat["gk_vol"] = self._garman_klass(df, co, window=20)

            # Yang-Zhang estimator (requires open)
            if has_open:
                feat["yz_vol"] = self._yang_zhang(df, window=20)

        return feat

    @staticmethod
    def _garman_klass(df: pd.DataFrame, co: pd.Series, window: int) -> pd.Series:
        """Garman-Klass realized volatility (annualized)."""
        hl = np.log(df["high"] / df["low"])
        gk = 0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2
        return gk.rolling(window).mean().apply(np.sqrt) * math.sqrt(252)

    @staticmethod
    def _yang_zhang(df: pd.DataFrame, window: int, k: float = 0.34) -> pd.Series:
        """Yang-Zhang realized volatility (annualized, requires OHLC)."""
        log_oc = np.log(df["open"] / df["close"].shift(1))   # overnight return
        log_co = np.log(df["close"] / df["open"])             # open-to-close return
        log_ho = np.log(df["high"] / df["open"])
        log_lo = np.log(df["low"] / df["open"])

        var_oc = log_oc.rolling(window).var()
        var_co = log_co.rolling(window).var()
        rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()

        # YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·RS
        yz = var_oc + k * var_co + (1 - k) * rs
        return yz.apply(np.sqrt) * math.sqrt(252)

    # --- Momentum features ---

    def _add_momentum_features(self, df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        ret = df["log_return"]

        # RSI (14-period)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        # Z-score of price vs MA (mean-reversion)
        for w in [20, 60]:
            ma = close.rolling(w).mean()
            sd = close.rolling(w).std().replace(0, np.nan)
            feat[f"zscore_{w}d"] = (close - ma) / sd

        for w in self.config.momentum_windows:
            feat[f"roc_{w}d"] = close.pct_change(w)

        return feat

    # --- Microstructure features ---

    def _add_microstructure_features(self, df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        if "volume" in df.columns:
            vol = df["volume"]
            feat["volume_ma_ratio"] = vol / vol.rolling(20).mean()
            feat["volume_zscore"] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()

        if "open_interest" in df.columns:
            oi = df["open_interest"]
            feat["oi_change"] = oi.pct_change()
            feat["oi_ma_ratio"] = oi / oi.rolling(20).mean()

        return feat

    # --- Temporal features ---

    def _add_temporal_features(self, df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        idx = pd.DatetimeIndex(df.index)

        feat["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
        feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)

        woy = idx.isocalendar().week.values.astype(float)
        feat["woy_sin"] = np.sin(2 * np.pi * woy / 52)
        feat["woy_cos"] = np.cos(2 * np.pi * woy / 52)

        feat["is_thursday"] = (idx.dayofweek == 3).astype(float)  # EIA report day

        return feat

    # --- Fundamental features ---

    def _add_fundamental_features(self, df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        # --- Storage surprise ---
        # Preferred: 'storage_surprise' pre-computed upstream (e.g. via ConsensusForecast)
        # Fallback:  derive from 'storage_actual' - 'storage_expected' columns
        if "storage_surprise" in df.columns:
            surprise = df["storage_surprise"]
            feat["storage_surprise"] = surprise.ffill()
            roll_mean = surprise.rolling(52).mean()
            roll_std = surprise.rolling(52).std().replace(0, np.nan)
            feat["storage_surprise_zscore"] = ((surprise - roll_mean) / roll_std).ffill()
        elif "storage_actual" in df.columns and "storage_expected" in df.columns:
            surprise = df["storage_actual"] - df["storage_expected"]
            feat["storage_surprise"] = surprise.ffill()
            roll_mean = surprise.rolling(52).mean()
            roll_std = surprise.rolling(52).std().replace(0, np.nan)
            feat["storage_surprise_zscore"] = ((surprise - roll_mean) / roll_std).ffill()

        # --- Storage level (BCF) and 5yr deviation ---
        # 'storage_level' takes priority; 'storage_actual' used as fallback
        level_col = "storage_level" if "storage_level" in df.columns else "storage_actual"
        if level_col in df.columns:
            level = df[level_col].ffill()
            feat["storage_level"] = level
            feat["storage_change_ma4"] = df.get("storage_change", level.diff()).rolling(4).mean().ffill()
            # Seasonal norm: 5yr rolling mean for same week-of-year (simple z-score proxy)
            feat["storage_5yr_dev"] = (
                df.get("storage_5yr_dev", (level - level.rolling(260, min_periods=52).mean()) / level)
            ).ffill()

        # EIA report day flag for the storage release (Thursday)
        # (also added in temporal features, but explicit here for fundamentals path)

        if "hdd" in df.columns:
            hdd = df["hdd"]
            feat["hdd"] = hdd.ffill()
            feat["hdd_zscore"] = (
                (hdd - hdd.rolling(52).mean()) / hdd.rolling(52).std().replace(0, np.nan)
            ).ffill()

        if "cdd" in df.columns:
            feat["cdd"] = df["cdd"].ffill()

        return feat

    # --- Cross-asset features ---

    def _add_cross_asset_features(self, df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        if "crude_close" in df.columns:
            crude_ret = np.log(df["crude_close"] / df["crude_close"].shift(1))
            feat["crude_ret_1d"] = crude_ret
            feat["crude_ret_5d"] = crude_ret.rolling(5).sum()
            ratio = df["close"] / df["crude_close"]
            feat["ng_cl_ratio_zscore"] = (
                (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std().replace(0, np.nan)
            )

        if "usd_index" in df.columns:
            usd_ret = np.log(df["usd_index"] / df["usd_index"].shift(1))
            feat["usd_ret_1d"] = usd_ret
            feat["usd_ret_5d"] = usd_ret.rolling(5).sum()

        return feat


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MixtureDensityHead(nn.Module):
    """
    MDN output head: trunk → (π, μ, σ) for K Gaussian components.

    π: softmax-normalized mixing weights (sum to 1)
    μ: component means (unconstrained)
    σ: component std devs (softplus + floor for numerical stability)
    """

    def __init__(self, in_features: int, n_components: int,
                 min_sigma: float = 1e-4, sigma_bias_init: float = 0.5):
        super().__init__()
        self.n_components = n_components
        self.min_sigma = min_sigma

        self.pi_head = nn.Linear(in_features, n_components)
        self.mu_head = nn.Linear(in_features, n_components)
        self.sigma_head = nn.Linear(in_features, n_components)

        # Initialize sigma bias so initial σ ≈ sigma_bias_init
        inv_sp = math.log(math.exp(sigma_bias_init) - 1)
        nn.init.constant_(self.sigma_head.bias, inv_sp)

        # Pi: near-uniform initialization
        nn.init.zeros_(self.pi_head.bias)
        nn.init.normal_(self.pi_head.weight, std=0.01)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, in_features)
        Returns:
            pi:    (B, K) — mixing weights summing to 1
            mu:    (B, K) — component means
            sigma: (B, K) — component std devs > min_sigma
        """
        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        sigma = nn.functional.softplus(self.sigma_head(h)) + self.min_sigma
        return pi, mu, sigma


class MDNNetwork(nn.Module):
    """
    Mixture Density Network.

    Architecture:
        Input → [Linear → (BatchNorm) → Activation → Dropout] × L → MDN Head
    """

    _ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}

    def __init__(self, config: MDNConfig, n_features: int):
        super().__init__()
        self.config = config

        act_cls = self._ACTIVATIONS[config.activation]
        dims = [n_features] + config.hidden_dims
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_cls())
            layers.append(nn.Dropout(config.dropout))

        self.trunk = nn.Sequential(*layers)
        self.mdn_head = MixtureDensityHead(
            in_features=config.hidden_dims[-1],
            n_components=config.n_components,
            min_sigma=config.min_sigma,
            sigma_bias_init=config.sigma_bias_init,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_features)
        Returns:
            pi, mu, sigma: each (B, K)
        """
        return self.mdn_head(self.trunk(x))

    def predict_density(self, x: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        """
        Evaluate mixture density at grid points.

        Args:
            x:      (B, n_features)
            y_grid: (G,) return values to evaluate at

        Returns:
            density: (B, G) — p(y | x) at each grid point
        """
        pi, mu, sigma = self.forward(x)              # (B, K) each
        pi_e = pi.unsqueeze(-1)                      # (B, K, 1)
        mu_e = mu.unsqueeze(-1)
        sigma_e = sigma.unsqueeze(-1)
        y_e = y_grid.unsqueeze(0).unsqueeze(0)       # (1, 1, G)

        norm_const = 1.0 / (sigma_e * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((y_e - mu_e) / sigma_e) ** 2
        component_pdf = norm_const * torch.exp(exponent)  # (B, K, G)

        return (pi_e * component_pdf).sum(dim=1)     # (B, G)

    def sample(self, x: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """
        Draw Monte Carlo samples from the predicted mixture.

        Args:
            x:         (B, n_features)
            n_samples: samples per observation

        Returns:
            samples: (B, n_samples)
        """
        pi, mu, sigma = self.forward(x)
        comp_idx = torch.multinomial(pi, n_samples, replacement=True)  # (B, n_samples)
        mu_s = torch.gather(mu, 1, comp_idx)
        sigma_s = torch.gather(sigma, 1, comp_idx)
        return mu_s + sigma_s * torch.randn_like(mu_s)


# ---------------------------------------------------------------------------
# EIA Storage utilities
# ---------------------------------------------------------------------------

def interpolate_weekly_storage(
    weekly_df: pd.DataFrame,
    daily_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Forward-fill a weekly-frequency storage DataFrame to daily business-day frequency.

    This is a generic interpolator — all column construction (surprise, z-score,
    5yr deviation, etc.) should be done upstream before calling this function.
    The recommended upstream pipeline is:

        from MacrOSINT.data.sources.eia.api_tools import NatGasHelper
        from MacrOSINT.models.energy.natgas_storage_forecast import (
            fetch_storage_data, ConsensusForecast,
        )

        ng  = NatGasHelper()
        st  = fetch_storage_data(ng, start='2010-01', end='2026-03')
        cf  = ConsensusForecast()
        cf.fit(st['storage_change'])
        surprise_df = cf.transform()   # actual, consensus_est, surprise, sea_*

        weekly = st.join(surprise_df[['consensus_est', 'surprise']], how='left')
        daily  = interpolate_weekly_storage(weekly, daily_index=price_df.index)

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly-frequency DataFrame with DatetimeIndex (or 'period' column).
        All numeric columns are forward-filled to the daily index.
    daily_index : pd.DatetimeIndex, optional
        Target business-day index. Defaults to bdate_range over weekly_df span.

    Returns
    -------
    pd.DataFrame aligned to daily_index with all columns forward-filled.
    """
    df = weekly_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "period" in df.columns:
            df = df.set_index("period")
        else:
            raise ValueError("weekly_df must have a DatetimeIndex or 'period' column")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if daily_index is None:
        daily_index = pd.bdate_range(df.index[0], df.index[-1])

    daily = (
        df.reindex(df.index.union(daily_index))
        .sort_index()
        .ffill()
        .reindex(daily_index)
    )
    return daily
