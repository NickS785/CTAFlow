"""Synthetic spread liquidity helpers.

This module provides utilities that operate on :class:`IntradayLeg`
instances defined in :mod:`CTAFlow.data.raw_formatting.synthetic`. The
functions focus on building synthetic spread time series and computing
basic liquidity proxies that can be reused by downstream analytics.

The helpers now live in ``CTAFlow.utils`` so that downstream code can
share a single set of primitives without touching the core
``SyntheticSymbol`` and ``IntradayLeg`` implementations.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from CTAFlow.data.raw_formatting.synthetic import IntradayLeg, SyntheticSymbol
except Exception:  # pragma: no cover - optional dependency fallback for tests
    class IntradayLeg:  # type: ignore[override]
        """Lightweight fallback IntradayLeg used when core dependency is absent."""

        def __init__(
            self,
            symbol: str,
            data: pd.DataFrame,
            base_weight: float = 1.0,
            unit: Optional[str] = None,
            notional_multiplier: float = 1.0,
            fx_rate: Union[float, pd.Series] = 1.0,
        ) -> None:
            self.symbol = symbol
            self.data = data
            self.base_weight = base_weight
            self.unit = unit or ""
            self.notional_multiplier = notional_multiplier
            self.fx_rate = fx_rate

        def get_effective_weight(self, date: Optional[pd.Timestamp] = None) -> float:
            weight = self.base_weight * self.notional_multiplier
            if isinstance(self.fx_rate, pd.Series) and date is not None:
                if date in self.fx_rate.index:
                    weight *= float(self.fx_rate.loc[date])
            else:
                weight *= float(self.fx_rate)
            return weight

        def convert_price_to_unit(self, price: float, target_unit: str) -> float:
            return float(price)

    class _StubEngine:
        def __init__(self, legs: Iterable[IntradayLeg]):
            self.legs = list(legs)

        def build_spread_series(self, return_ohlc: bool = False) -> Union[pd.Series, pd.DataFrame]:
            if return_ohlc:
                raise ValueError("OHLC output not supported in stub SyntheticSymbol")

            merged = merge_intraday_legs(self.legs, max_ffill=None)
            spread = pd.Series(0.0, index=merged.index)
            for leg in self.legs:
                price_series = merged.get((leg.symbol, "price"))
                if price_series is None:
                    continue
                spread = spread.add(leg.get_effective_weight() * price_series, fill_value=0.0)
            return spread

    class SyntheticSymbol:  # type: ignore[override]
        def __init__(self, legs: Iterable[IntradayLeg], **_: object) -> None:
            self.legs = list(legs)
            self.data_engine = _StubEngine(self.legs)
            self.price = self.data_engine.build_spread_series(return_ohlc=False)


FieldKey = Tuple[str, str]


def _coerce_utc_index(df: pd.DataFrame, ts_col: Optional[str]) -> pd.DatetimeIndex:
    """Return a UTC ``DatetimeIndex`` for ``df``.

    Parameters
    ----------
    df:
        Source DataFrame.
    ts_col:
        Optional column containing timestamps.  When provided, the column is
        used and dropped from the copy that callers generally pass around.
    """

    if ts_col and ts_col in df.columns:
        idx = pd.to_datetime(df[ts_col])
    else:
        idx = pd.to_datetime(df.index)

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    return pd.DatetimeIndex(idx)


def _extract_series(
    df: pd.DataFrame,
    symbol: str,
    index: pd.DatetimeIndex,
    column: str,
    alias: str,
    max_ffill: Optional[pd.Timedelta],
) -> pd.Series:
    """Extract and align a single column for a leg.

    ``max_ffill`` limits forward filling via ``Series.reindex`` with
    ``tolerance`` so that stale legs are automatically flagged as ``NaN``.
    """

    if column not in df.columns:
        return pd.Series(index=index, dtype="float64")

    series = df[column].astype(float)
    series = series.sort_index()

    if max_ffill is None:
        aligned = series.reindex(index)
    else:
        tolerance = pd.Timedelta(max_ffill)
        aligned = series.reindex(index, method="pad", tolerance=tolerance)

    aligned.name = (symbol, alias)
    return aligned


def merge_intraday_legs(
    legs: Iterable[IntradayLeg],
    *,
    price_col: str = "Close",
    volume_col: str = "Volume",
    ts_col: Optional[str] = None,
    max_ffill: Optional[Union[str, pd.Timedelta]] = pd.Timedelta("2min"),
) -> pd.DataFrame:
    """Merge multiple :class:`IntradayLeg` DataFrames on a common UTC index.

    Parameters
    ----------
    legs:
        Iterable of legs to merge.
    price_col / volume_col:
        Column names that correspond to price and volume fields.  ``Close`` and
        ``Volume`` are used by default which matches the intraday loader.
    ts_col:
        Optional timestamp column name.  When omitted, the index from the
        underlying leg ``data`` is used.
    max_ffill:
        Maximum staleness allowed when forward filling.  Values older than the
        tolerance are dropped.  ``None`` disables forward filling altogether.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a ``MultiIndex`` on the columns ``(symbol, field)``
        containing price and volume information for every requested leg.
    """

    legs = list(legs)
    if not legs:
        return pd.DataFrame()

    # Collect UTC indices so we can build an aligned union index.
    indices: List[pd.DatetimeIndex] = []
    leg_frames: List[Tuple[FieldKey, pd.Series]] = []

    tolerance: Optional[pd.Timedelta]
    if max_ffill is None:
        tolerance = None
    elif isinstance(max_ffill, pd.Timedelta):
        tolerance = max_ffill
    else:
        tolerance = pd.Timedelta(max_ffill)

    for leg in legs:
        raw_df = leg.data.copy()
        utc_index = _coerce_utc_index(raw_df, ts_col)
        raw_df.index = utc_index
        raw_df = raw_df[~raw_df.index.duplicated(keep="last")]

        indices.append(raw_df.index)

        leg_frames.append(
            ((leg.symbol, "price"), _extract_series(raw_df, leg.symbol, raw_df.index, price_col, "price", tolerance))
        )
        leg_frames.append(
            ((leg.symbol, "volume"), _extract_series(raw_df, leg.symbol, raw_df.index, volume_col, "volume", tolerance))
        )

    union_index = indices[0]
    for idx in indices[1:]:
        union_index = union_index.union(idx)
    union_index = union_index.sort_values()

    data: Dict[FieldKey, pd.Series] = {}
    for (symbol, field), series in leg_frames:
        if series.empty and len(union_index) == 0:
            aligned = series
        elif tolerance is None:
            aligned = series.reindex(union_index)
        else:
            aligned = series.reindex(union_index, method="pad", tolerance=tolerance)
        data[(symbol, field)] = aligned

    merged = pd.DataFrame(data)
    merged.columns = pd.MultiIndex.from_tuples(merged.columns, names=["symbol", "field"])
    merged = merged.sort_index()

    # Drop rows where every price column is NaN to keep the panel compact.
    price_cols = [col for col in merged.columns if col[1] == "price"]
    if price_cols:
        merged = merged.loc[~merged[price_cols].isna().all(axis=1)]

    return merged


def synthetic_price(
    legs: Iterable[IntradayLeg],
    *,
    return_ohlc: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Compute the synthetic spread price using :class:`SyntheticSymbol`.

    Parameters
    ----------
    legs:
        Iterable of legs making up the spread.
    return_ohlc:
        When ``True`` the full OHLC DataFrame is returned (requires legs with
        OHLC fields).  Otherwise the close-only spread series is produced.
    """

    legs = list(legs)
    if not legs:
        return pd.Series(dtype="float64")

    symbol = SyntheticSymbol(legs=legs)
    if return_ohlc:
        return symbol.data_engine.build_spread_series(return_ohlc=True)
    if isinstance(symbol.price, pd.DataFrame):
        return symbol.price["Close"] if "Close" in symbol.price.columns else symbol.price.squeeze()
    return symbol.price


def synthetic_returns(syn_price: pd.Series) -> pd.Series:
    """Compute log returns for a synthetic price series."""

    syn_price = syn_price.astype(float)
    return np.log(syn_price).diff()


def _get_leg_series(
    merged: pd.DataFrame,
    symbol: str,
    field: str,
) -> Optional[pd.Series]:
    if (symbol, field) in merged.columns:
        return merged[(symbol, field)]
    if symbol in merged.columns:
        maybe = merged[symbol]
        if isinstance(maybe, pd.DataFrame) and field in maybe.columns:
            return maybe[field]
    column_name = f"{symbol}_{field}"
    if column_name in merged.columns:
        return merged[column_name]
    return None


def syn_vol_min(
    legs: Iterable[IntradayLeg],
    merged: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """Compute the minimum executable volume across legs per bar."""

    legs = list(legs)
    if merged.empty or not legs:
        return pd.Series(index=merged.index, dtype="float64")

    per_leg = []
    for leg in legs:
        weight = weights.get(leg.symbol, leg.base_weight)
        if weight == 0:
            continue
        vol = _get_leg_series(merged, leg.symbol, "volume")
        if vol is None:
            continue
        per_leg.append(vol.abs() / abs(weight))

    if not per_leg:
        return pd.Series(index=merged.index, dtype="float64")

    stacked = pd.concat(per_leg, axis=1)
    return stacked.min(axis=1)


def syn_dollar_turnover(
    legs: Iterable[IntradayLeg],
    merged: pd.DataFrame,
    weights: Dict[str, float],
    multipliers: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Dollar turnover proxy for a synthetic spread."""

    legs = list(legs)
    if merged.empty or not legs:
        return pd.Series(index=merged.index, dtype="float64")

    multipliers = multipliers or {}
    contributions = []

    for leg in legs:
        price = _get_leg_series(merged, leg.symbol, "price")
        volume = _get_leg_series(merged, leg.symbol, "volume")
        if price is None or volume is None:
            continue
        weight = abs(weights.get(leg.symbol, leg.base_weight))
        multiplier = multipliers.get(leg.symbol, getattr(leg, "notional_multiplier", 1.0))
        contributions.append(weight * multiplier * price.abs() * volume.abs())

    if not contributions:
        return pd.Series(index=merged.index, dtype="float64")

    return pd.concat(contributions, axis=1).sum(axis=1)


def syn_capacity_impact(
    legs: Iterable[IntradayLeg],
    merged: pd.DataFrame,
    lambdas: Dict[str, Optional[float]],
) -> pd.Series:
    """Impact-adjusted capacity proxy using Kyle's lambda estimates."""

    legs = list(legs)
    if merged.empty or not legs:
        return pd.Series(index=merged.index, dtype="float64")

    per_leg = []
    for leg in legs:
        lam = lambdas.get(leg.symbol)
        if lam is None or not np.isfinite(lam) or lam <= 0:
            continue
        vol = _get_leg_series(merged, leg.symbol, "volume")
        if vol is None:
            continue
        per_leg.append(np.sqrt(vol.clip(lower=0) / lam))

    if not per_leg:
        return pd.Series(index=merged.index, dtype="float64")

    stacked = pd.concat(per_leg, axis=1)
    return stacked.min(axis=1)


__all__ = [
    "merge_intraday_legs",
    "synthetic_price",
    "synthetic_returns",
    "syn_vol_min",
    "syn_dollar_turnover",
    "syn_capacity_impact",
]

