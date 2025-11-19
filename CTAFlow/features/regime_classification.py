"""Discrete regime classification helpers used by the HistoricalScreener."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "BaseRegimeClassifier",
    "TrendRegimeClassifier",
    "VolatilityRegimeClassifier",
    "CrowdingRegimeClassifier",
    "RegimeSpecification",
    "RegimeSpecificationLike",
    "build_regime_classifier",
]


class BaseRegimeClassifier:
    """Base interface for discrete regime classifiers."""

    name: str = "base"

    def classify(self, data: pd.DataFrame) -> pd.Series:  # pragma: no cover - interface only
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.name}

    def cache_key(self) -> str:
        payload = json.dumps(self.describe(), sort_keys=True, default=str)
        return payload


@dataclass(frozen=True)
class RegimeSpecification:
    """Serializable configuration describing a classifier type and its kwargs."""

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def normalized_kind(self) -> str:
        return str(self.kind).strip().lower()


RegimeSpecificationLike = Union[
    BaseRegimeClassifier,
    RegimeSpecification,
    Mapping[str, Any],
]


class TrendRegimeClassifier(BaseRegimeClassifier):
    """Classify regimes based on moving-average differentials."""

    name = "trend"

    def __init__(
        self,
        *,
        price_col: str = "close",
        fast_window: int = 10,
        slow_window: int = 50,
        method: str = "ema",
        neutral_band: float = 0.0,
        band_pct: float = 0.0,
        resample_rule: str = "1D",
        resample_offset: str | pd.Timedelta | None = "-9H",
    ) -> None:
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("Moving-average windows must be positive")
        if method not in {"ema", "sma"}:
            raise ValueError("method must be either 'ema' or 'sma'")
        self.price_col = price_col
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.method = method
        self.neutral_band = float(neutral_band)
        self.band_pct = float(band_pct)
        self.resample_rule = resample_rule
        self.resample_offset = resample_offset

    def _moving_average(self, series: pd.Series, window: int) -> pd.Series:
        if self.method == "ema":
            return series.ewm(span=window, adjust=False, min_periods=1).mean()
        return series.rolling(window=window, min_periods=window).mean()

    def _resample_price(self, series: pd.Series) -> pd.Series:
        """Aggregate the price series to the configured daily frequency."""

        cleaned = series.dropna()
        if cleaned.empty or self.resample_rule is None:
            return cleaned

        if not isinstance(cleaned.index, pd.DatetimeIndex):
            try:
                datetime_index = pd.to_datetime(cleaned.index, errors="raise")
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise TypeError(
                    "TrendRegimeClassifier requires a DatetimeIndex to resample prices"
                ) from exc
            cleaned = cleaned.set_axis(datetime_index)

        resampled = cleaned.resample(
            self.resample_rule,
            offset=self.resample_offset,
        ).last()
        return resampled.dropna()

    def classify(self, data: pd.DataFrame) -> pd.Series:
        if self.price_col not in data.columns:
            raise KeyError(self.price_col)
        price = pd.to_numeric(data[self.price_col], errors="coerce")
        original_index = price.index
        daily_price = self._resample_price(price)
        if daily_price.empty:
            return pd.Series(pd.NA, index=original_index, dtype="Int64")
        fast = self._moving_average(daily_price, self.fast_window)
        slow = self._moving_average(daily_price, self.slow_window)
        diff = fast - slow
        threshold = pd.Series(abs(self.neutral_band), index=diff.index, dtype=float)
        if self.band_pct > 0:
            dynamic = np.abs(slow) * abs(self.band_pct)
            threshold = threshold.combine(dynamic, lambda base, dyn: np.nanmax([base, dyn]))
        regimes = pd.Series(pd.NA, index=diff.index, dtype="Int64")
        regimes.loc[diff > threshold] = 1
        regimes.loc[diff < -threshold] = -1
        mid_mask = diff.abs() <= threshold
        regimes.loc[mid_mask & diff.notna()] = 0
        regimes = regimes.reindex(original_index, method="ffill")
        return regimes.astype("Int64")

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.name,
            "price_col": self.price_col,
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "method": self.method,
            "neutral_band": self.neutral_band,
            "band_pct": self.band_pct,
            "resample_rule": self.resample_rule,
            "resample_offset": self.resample_offset,
        }


class VolatilityRegimeClassifier(BaseRegimeClassifier):
    """Classify regimes by realised volatility buckets."""

    name = "volatility"

    def __init__(
            self,
            *,
            returns_col: str = "returns",
            price_col: Optional[str] = None,
            window: int = 20,
            method: str = "ewm",
            low_quantile: float = 0.33,
            high_quantile: float = 0.66,
            resample_rule: str = "1D",
            resample_offset: str | pd.Timedelta | None = "-9H",
    ) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        if method not in {"ewm", "sma"}:
            raise ValueError("method must be either 'ewm' or 'sma'")
        if not (0 < low_quantile < high_quantile < 1):
            raise ValueError("quantiles must satisfy 0 < low < high < 1")
        self.returns_col = returns_col
        self.price_col = price_col
        self.window = int(window)
        self.method = method
        self.low_quantile = float(low_quantile)
        self.high_quantile = float(high_quantile)
        self.resample_rule = resample_rule
        self.resample_offset = resample_offset

    def _resolve_returns(self, data: pd.DataFrame) -> pd.Series:
        if self.returns_col in data.columns:
            return pd.to_numeric(data[self.returns_col], errors="coerce")
        if self.price_col and self.price_col in data.columns:
            price = pd.to_numeric(data[self.price_col], errors="coerce")
            return np.log(price).diff()
        raise KeyError(self.returns_col)

    def _ensure_dt_index(self, s: pd.Series) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex):
            return s
        try:
            idx = pd.to_datetime(s.index, errors="raise")
        except Exception as exc:
            raise TypeError("VolatilityRegimeClassifier requires a DatetimeIndex") from exc
        return s.set_axis(idx)

    def _daily_returns(self, data: pd.DataFrame) -> pd.Series:
        """
        Returns a DAILY series of log returns using last price per day
        (aligned to resample_rule/offset). Used by 'ewm' and 'sma' vol.
        """
        if self.price_col and self.price_col in data.columns:
            price = pd.to_numeric(data[self.price_col], errors="coerce").dropna()
            price = self._ensure_dt_index(price)
            daily_price = price.resample(self.resample_rule, offset=self.resample_offset).last().dropna()
            return np.log(daily_price).diff()
        # fallback from returns_col: resample by *sum* of intraday returns for the day
        r = self._resolve_returns(data).dropna()
        r = self._ensure_dt_index(r)
        daily_r = r.resample(self.resample_rule, offset=self.resample_offset).sum().dropna()
        return daily_r

    def _daily_rv(self, data: pd.DataFrame) -> pd.Series:
        """
        Realized volatility proxy: sum of squared intraday log returns per day.
        """
        r = self._resolve_returns(data).dropna()
        r = self._ensure_dt_index(r)
        rv_daily = (r ** 2).resample(self.resample_rule, offset=self.resample_offset).sum().dropna()
        return rv_daily

    def classify(self, data: pd.DataFrame) -> pd.Series:
        # Compute a daily volatility series per 'method'
        if self.method == "rv":
            vol = self._daily_rv(data)
        else:
            # daily returns then a daily std estimator
            daily_r = self._daily_returns(data)
            if self.method == "ewm":
                vol = daily_r.ewm(span=self.window, adjust=False, min_periods=self.window).std(bias=False)
            elif self.method == "sma":
                vol = daily_r.rolling(window=self.window, min_periods=self.window).std(ddof=0)
            else:
                raise ValueError("method must be one of {'ewm','sma','rv'}")

        # Rank in-sample (daily)
        rank = vol.rank(pct=True, method="first")
        regimes_daily = pd.Series(pd.NA, index=vol.index, dtype="Int64")
        regimes_daily.loc[rank >= self.high_quantile] = 1
        regimes_daily.loc[rank <= self.low_quantile] = -1
        mid_mask = rank.between(self.low_quantile, self.high_quantile, inclusive="neither")
        regimes_daily.loc[mid_mask] = 0

        # Reindex back to input index (ffill), to keep API consistent
        original_index = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index,
                                                                                                    errors="coerce")
        regimes = regimes_daily.reindex(original_index, method="ffill")
        return regimes.astype("Int64")

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.name,
            "returns_col": self.returns_col,
            "price_col": self.price_col,
            "window": self.window,
            "method": self.method,
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
            "resample_rule": self.resample_rule,
            "resample_offset": self.resample_offset,
        }

class CrowdingRegimeClassifier(BaseRegimeClassifier):
    """Classify regimes using COT net-position z-scores."""

    name = "crowding"

    def __init__(
        self,
        *,
        net_position_col: str = "net_positions",
        window: int = 52,
        neutral_band: float = 0.5,
    ) -> None:
        if window <= 1:
            raise ValueError("window must be greater than 1")
        self.net_position_col = net_position_col
        self.window = int(window)
        self.neutral_band = float(neutral_band)

    def classify(self, data: pd.DataFrame) -> pd.Series:
        if self.net_position_col not in data.columns:
            raise KeyError(self.net_position_col)
        positions = pd.to_numeric(data[self.net_position_col], errors="coerce")
        rolling_mean = positions.rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = positions.rolling(window=self.window, min_periods=self.window).std(ddof=0)
        zscore = (positions - rolling_mean) / rolling_std
        regimes = pd.Series(pd.NA, index=zscore.index, dtype="Int64")
        regimes.loc[zscore >= abs(self.neutral_band)] = 1
        regimes.loc[zscore <= -abs(self.neutral_band)] = -1
        neutral_mask = zscore.abs() < abs(self.neutral_band)
        regimes.loc[neutral_mask & zscore.notna()] = 0
        return regimes

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.name,
            "net_position_col": self.net_position_col,
            "window": self.window,
            "neutral_band": self.neutral_band,
        }


def build_regime_classifier(
    settings: Optional[RegimeSpecificationLike],
) -> Optional[BaseRegimeClassifier]:
    """Normalise ``settings`` into a concrete classifier instance."""

    if settings is None:
        return None

    if isinstance(settings, BaseRegimeClassifier):
        return settings

    kind: Optional[str]
    params: Dict[str, Any]

    if isinstance(settings, RegimeSpecification):
        kind = settings.normalized_kind()
        params = dict(settings.params)
    elif isinstance(settings, Mapping):
        mapping = dict(settings)
        kind = str(mapping.pop("kind", mapping.pop("type", ""))).strip().lower()
        params = mapping
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported regime settings type: {type(settings)!r}")

    if not kind:
        raise ValueError("Regime specification must include a 'kind' field")

    if kind == "trend":
        return TrendRegimeClassifier(**params)
    if kind == "volatility":
        return VolatilityRegimeClassifier(**params)
    if kind == "crowding":
        return CrowdingRegimeClassifier(**params)
    raise ValueError(f"Unsupported regime kind: {kind}")
