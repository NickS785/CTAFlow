from __future__ import annotations

from datetime import time, timedelta
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .base_models import CTALight
from .volatility import RVForecast


class IntradayMomentumLight(CTALight):
    """LightGBM wrapper focused on intraday momentum and volatility features.

    The model is tailored to predict session-end behaviour (default 15:00 CST)
    and exposes convenience helpers for constructing intraday-aware training
    matrices.
    """

    def __init__(
        self,
        intraday_data: pd.DataFrame,
        session_end: time = time(15, 0),
        session_open: time = time(8, 30),
        closing_length: timedelta = timedelta(minutes=60),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.intraday_data = intraday_data
        self.session_end = session_end
        self.session_open = session_open
        self.closing_length = closing_length

    @staticmethod
    def _coerce_price(df: pd.DataFrame, price_col: str) -> pd.Series:
        if price_col not in df.columns:
            raise KeyError(f"Price column '{price_col}' not found")
        return df[price_col].copy()

    def opening_range_volatility(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        price_col: str = "Close",
        period_length: Optional[timedelta] = None,
    ) -> pd.DataFrame:
        """Compute opening range realised volatility and semivariance.

        Parameters
        ----------
        intraday_df : pd.DataFrame
            Intraday OHLCV data with a DatetimeIndex.
        price_col : str, default "Close"
            Column containing the price series used for returns.
        period_length : Optional[timedelta]
            Optional override for the opening window. Defaults to
            ``closing_length`` if not provided.
        """

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for opening range calculations")

        window = period_length or self.closing_length
        prices = self._coerce_price(data, price_col)
        returns = prices.pct_change().dropna()

        grouped = returns.groupby(pd.Grouper(freq="1D"))
        records = []
        index_vals = []
        session_open_offset = pd.Timedelta(
            hours=self.session_open.hour,
            minutes=self.session_open.minute,
            seconds=self.session_open.second,
        )

        for day, day_series in grouped:
            if day_series.empty:
                continue

            session_start = day.normalize() + session_open_offset
            session_end = session_start + window
            opening_slice = day_series[(day_series.index >= session_start) & (day_series.index < session_end)]
            if opening_slice.empty:
                continue

            rv = float(np.sqrt((opening_slice ** 2).sum()))
            pos = opening_slice[opening_slice > 0]
            neg = opening_slice[opening_slice < 0]
            rsv_pos = float(np.sqrt((pos ** 2).sum())) if not pos.empty else np.nan
            rsv_neg = float(np.sqrt((neg ** 2).sum())) if not neg.empty else np.nan
            records.append(
                {
                    "rv_open": rv,
                    "rsv_pos_open": rsv_pos,
                    "rsv_neg_open": rsv_neg,
                }
            )
            index_vals.append(day.normalize())

        feature_df = pd.DataFrame(records, index=index_vals)
        return feature_df

    def har_volatility_features(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        horizons: Sequence[int] = (1, 5, 22),
    ) -> pd.DataFrame:
        """Generate HAR-style realised volatility features for 1d ahead forecasts."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        forecaster = RVForecast(intraday_df=data)
        return forecaster.har_features(horizons=horizons)

    def target_time_returns(
        self,
        target_time: time,
        intraday_df: Optional[pd.DataFrame] = None,
        period_length: Optional[timedelta] = None,
        price_col: str = "Close",
    ) -> pd.Series:
        """Return series for a specific target time window."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for target time returns")

        window = period_length or self.closing_length
        prices = self._coerce_price(data, price_col)
        ret = prices.pct_change().dropna()

        mask = ret.index.time == target_time
        if window:
            bars = max(int(window.total_seconds() // 60 // 5), 1)
            ret = ret.rolling(bars).sum()
        return ret[mask]

    def market_structure_features(
        self,
        daily_df: pd.DataFrame,
        lookbacks: Iterable[int] = (5, 10, 20),
        price_col: str = "Close",
    ) -> pd.DataFrame:
        """Build multi-horizon market structure context (high/low/VAH/VAL)."""

        prices = self._coerce_price(daily_df, price_col)
        feats: Dict[str, pd.Series] = {}
        for lb in lookbacks:
            feats[f"hh_{lb}"] = prices.rolling(lb).max()
            feats[f"ll_{lb}"] = prices.rolling(lb).min()
            feats[f"range_{lb}"] = feats[f"hh_{lb}"] - feats[f"ll_{lb}"]

        if {"High", "Low"}.issubset(daily_df.columns):
            vah = daily_df["High"].rolling(1).max()
            val = daily_df["Low"].rolling(1).min()
            feats["vah_prev"] = vah.shift(1)
            feats["val_prev"] = val.shift(1)

        return pd.DataFrame(feats)

    def microstructure_features(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        price_col: str = "Close",
    ) -> pd.DataFrame:
        """Add microstructure proxies such as imbalance and realised range."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        if data is None or data.empty:
            raise ValueError("Intraday data is required for microstructure features")

        prices = self._coerce_price(data, price_col)
        diffs = prices.diff().fillna(0)

        features = pd.DataFrame(index=data.index)
        features["signed_volume"] = np.sign(diffs) * data.get("Volume", 0)
        features["price_range"] = data.get("High", prices).fillna(prices) - data.get("Low", prices).fillna(prices)
        features["abs_ret"] = diffs.abs()
        return features

    def assemble_training_frame(
        self,
        daily_df: pd.DataFrame,
        target_times: Sequence[time],
        intraday_df: Optional[pd.DataFrame] = None,
        price_col: str = "Close",
    ) -> pd.DataFrame:
        """Convenience constructor for a full training feature matrix."""

        data = intraday_df if intraday_df is not None else self.intraday_data
        har = self.har_volatility_features(data)
        market_struct = self.market_structure_features(daily_df, price_col=price_col)

        intraday_features = []
        for t in target_times:
            intraday_returns = self.target_time_returns(
                data,
                target_time=t,
                price_col=price_col,
            )
            intraday_features.append(intraday_returns.rename(f"ret_{t.strftime('%H%M')}") )

        intraday_frame = pd.concat(intraday_features, axis=1)
        combined = pd.concat([har, market_struct, intraday_frame], axis=1)
        return combined.dropna()
