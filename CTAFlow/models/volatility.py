from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .base_models import CTALinear


class RVForecast(CTALinear):
    """Lightweight realised-volatility forecaster using HAR-style features."""

    def __init__(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        price_col: str = "Close",
        model_type: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_type=model_type,
            alpha=alpha,
            l1_ratio=l1_ratio,
            normalize=normalize,
            **kwargs,
        )
        self.intraday_df = intraday_df
        self.price_col = price_col

    def _require_intraday(self, intraday_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        data = intraday_df if intraday_df is not None else self.intraday_df
        if data is None or data.empty:
            raise ValueError("Intraday data is required to compute realised volatility")
        return data

    def realized_volatility(self, intraday_df: Optional[pd.DataFrame] = None) -> pd.Series:
        data = self._require_intraday(intraday_df)
        prices = data[self.price_col]
        returns = prices.pct_change().dropna()
        rv = returns.pow(2).groupby(pd.Grouper(freq="1D")).sum()
        return rv

    def har_features(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        horizons: Sequence[int] = (1, 5, 22),
    ) -> pd.DataFrame:
        rv = self.realized_volatility(intraday_df)
        features = {}
        for horizon in horizons:
            features[f"rv_lag_{horizon}d"] = rv.rolling(horizon).mean().shift(1)
        return pd.DataFrame(features)

    def build_training_frame(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        horizons: Sequence[int] = (1, 5, 22),
        target_horizon: int = 1,
    ) -> pd.DataFrame:
        if target_horizon < 1:
            raise ValueError("target_horizon must be at least 1 day")

        har = self.har_features(intraday_df, horizons=horizons)
        rv = self.realized_volatility(intraday_df)
        target = rv.rolling(target_horizon).mean().shift(-target_horizon)

        frame = har.copy()
        frame["target_rv"] = target
        return frame.dropna()

    def fit_volatility(
        self,
        intraday_df: Optional[pd.DataFrame] = None,
        horizons: Sequence[int] = (1, 5, 22),
        target_horizon: int = 1,
        **fit_kwargs,
    ) -> "RVForecast":
        training = self.build_training_frame(
            intraday_df=intraday_df,
            horizons=horizons,
            target_horizon=target_horizon,
        )
        X = training.drop(columns=["target_rv"])
        y = training["target_rv"]
        self.fit(X, y, **fit_kwargs)
        return self
