"""Utilities for organising and analysing screener pattern output."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from CTAFlow.screeners.historical_screener import HistoricalScreener, ScreenParams
from CTAFlow.utils.seasonal import aggregate_window, log_returns, monthly_returns, tod_mask
from CTAFlow.utils.session import filter_session_bars


@dataclass
class PatternSummary:
    """Container describing a statistically significant pattern."""

    key: str
    symbol: str
    source_screen: str
    screen_params: Optional[ScreenParams]
    pattern_type: str
    description: str
    strength: Optional[float]
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation suitable for JSON/export."""

        params_dict: Optional[Dict[str, Any]]
        if self.screen_params is None:
            params_dict = None
        else:
            params_dict = {
                field_name: getattr(self.screen_params, field_name)
                for field_name in self.screen_params.__dataclass_fields__  # type: ignore[attr-defined]
            }

        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "strength": self.strength,
            "source_screen": self.source_screen,
            "screen_parameters": params_dict,
            "metadata": self.metadata,
            "pattern_payload": self.payload,
        }


class PatternExtractor:
    """Restructure screener output and expose analysis helpers."""

    def __init__(
        self,
        screener: HistoricalScreener,
        results: Mapping[str, Mapping[str, Dict[str, Any]]],
        screen_params: Optional[Sequence[ScreenParams]] = None,
    ) -> None:
        self._screener = screener
        self._results = results
        self._screen_params: Dict[str, ScreenParams] = (
            {params.name: params for params in screen_params}
            if screen_params is not None
            else {}
        )

        self._pattern_index: Dict[str, Dict[str, PatternSummary]] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def patterns(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return patterns grouped by symbol and keyed by pattern identifier."""

        return {
            symbol: {key: summary.as_dict() for key, summary in entries.items()}
            for symbol, entries in self._pattern_index.items()
        }

    def get_pattern_keys(self, symbol: str) -> List[str]:
        """Return all pattern identifiers for ``symbol``."""

        return list(self._pattern_index.get(symbol, {}).keys())

    def get_pattern_summary(self, symbol: str, pattern_key: str) -> PatternSummary:
        """Retrieve the :class:`PatternSummary` for the requested pattern."""

        try:
            return self._pattern_index[symbol][pattern_key]
        except KeyError as exc:  # pragma: no cover - user error surface
            raise KeyError(f"Unknown pattern '{pattern_key}' for symbol '{symbol}'") from exc

    def get_pattern_series(self, symbol: str, pattern_key: str) -> pd.Series:
        """Return the time-series associated with a pattern."""

        summary = self.get_pattern_summary(symbol, pattern_key)
        screen_type = summary.metadata.get("screen_type")

        if summary.pattern_type in {"weekday_mean", "weekday_returns"}:
            return self._extract_weekday_series(symbol, summary)

        if summary.pattern_type in {"time_predictive_nextday", "time_predictive_nextweek"}:
            return self._extract_time_of_day_series(symbol, summary)

        if summary.pattern_type in {
            "abnormal_month",
            "high_sharpe_month",
            "annual_seasonality",
            "lag_month_momentum",
            "prewindow_predictor",
        }:
            return self._extract_monthly_series(symbol, summary)

        raise NotImplementedError(
            f"Pattern type '{summary.pattern_type}' from screen '{screen_type}' is not supported yet"
        )

    def get_pattern_dates(self, symbol: str, pattern_key: str) -> List[pd.Timestamp]:
        """Return the unique dates where the pattern occurs."""

        series = self.get_pattern_series(symbol, pattern_key)
        return list(pd.Index(series.index).unique())

    def rebase_price_series(
        self,
        symbol: str,
        pattern_key: str,
        price_col: str = "Close",
    ) -> pd.DataFrame:
        """Construct a rebased price series using pattern occurrences."""

        pattern_series = self.get_pattern_series(symbol, pattern_key)
        if pattern_series.empty:
            return pd.DataFrame(columns=["pattern_return", "close", "rebased"])

        price_series = self._get_price_series(symbol, price_col)
        daily_close = price_series.groupby(price_series.index.normalize()).last()
        aligned_close = daily_close.reindex(pattern_series.index, method="ffill")

        rebased = (pattern_series.fillna(0).add(1.0)).cumprod()

        return pd.DataFrame(
            {
                "pattern_return": pattern_series,
                "close": aligned_close,
                "rebased": rebased,
            }
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        for screen_name, ticker_results in self._results.items():
            params = self._screen_params.get(screen_name)
            screen_type = getattr(params, "screen_type", None)

            for symbol, result in ticker_results.items():
                if not isinstance(result, Mapping) or "error" in result:
                    continue

                symbol_patterns = self._pattern_index.setdefault(symbol, {})

                for summary in self._iter_pattern_summaries(
                    symbol=symbol,
                    screen_name=screen_name,
                    screen_type=screen_type,
                    ticker_result=result,
                    params=params,
                ):
                    symbol_patterns[summary.key] = summary

    def _iter_pattern_summaries(
        self,
        symbol: str,
        screen_name: str,
        screen_type: Optional[str],
        ticker_result: Mapping[str, Any],
        params: Optional[ScreenParams],
    ) -> Iterable[PatternSummary]:
        strongest: Iterable[Dict[str, Any]] = ticker_result.get("strongest_patterns", [])  # type: ignore[assignment]
        for pattern in strongest:
            yield self._build_summary(
                symbol,
                screen_name,
                screen_type,
                params,
                pattern,
                origin="strongest_patterns",
            )

    def _build_summary(
        self,
        symbol: str,
        screen_name: str,
        screen_type: Optional[str],
        params: Optional[ScreenParams],
        pattern: Mapping[str, Any],
        origin: str,
    ) -> PatternSummary:
        pattern_type = self._infer_pattern_type(pattern)
        strength = self._extract_strength(pattern)
        description = str(pattern.get("description", pattern_type))

        qualifiers: List[str] = [pattern_type]
        for key in ("day", "time", "lag", "month"):
            if key in pattern and pattern[key] is not None:
                qualifiers.append(str(pattern[key]))
        key = "|".join([screen_name, *qualifiers])

        metadata = {
            "pattern_origin": origin,
            "screen_type": screen_type,
        }
        if screen_type == "seasonality" and isinstance(params, ScreenParams):
            metadata.update(
                {
                    "target_time": pattern.get("time"),
                    "most_prevalent_day": pattern.get("most_prevalent_day"),
                }
            )

        return PatternSummary(
            key=key,
            symbol=symbol,
            source_screen=screen_name,
            screen_params=params,
            pattern_type=pattern_type,
            description=description,
            strength=strength,
            payload=dict(pattern),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Series reconstruction helpers
    # ------------------------------------------------------------------
    def _extract_weekday_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        params = summary.screen_params
        if params is None or params.screen_type != "seasonality":
            raise ValueError("Weekday patterns require seasonality screen parameters")

        session_data, price_col, is_synthetic = self._get_seasonality_session_data(symbol, params)
        returns = self._compute_intraday_returns(session_data, price_col, is_synthetic)
        if returns.empty:
            return pd.Series(dtype=float)

        daily_returns = returns.groupby(returns.index.normalize()).sum()
        target_day = summary.payload.get("day")
        if target_day is None:
            return daily_returns

        weekday_number = self._weekday_name_to_number(target_day)
        mask = daily_returns.index.dayofweek == weekday_number
        return daily_returns[mask]

    def _extract_time_of_day_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        params = summary.screen_params
        if params is None or params.screen_type != "seasonality":
            raise ValueError("Time-of-day patterns require seasonality screen parameters")

        target_time_raw = summary.payload.get("time")
        if target_time_raw is None:
            raise ValueError("Pattern payload missing target time")

        if isinstance(target_time_raw, str):
            target_time_obj = pd.to_datetime(target_time_raw).time()
        elif isinstance(target_time_raw, time):
            target_time_obj = target_time_raw
        else:
            target_time_obj = pd.to_datetime(target_time_raw).time()

        session_data, price_col, is_synthetic = self._get_seasonality_session_data(symbol, params)
        if session_data.empty:
            return pd.Series(dtype=float)

        period_length = params.period_length
        if isinstance(period_length, (int, float)):
            period_length = pd.Timedelta(minutes=period_length)

        returns = self._compute_time_of_day_returns(
            session_data,
            price_col,
            is_synthetic,
            target_time_obj,
            period_length,
        )
        return returns

    def _extract_monthly_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        price_df = self._get_price_series(symbol, price_col="Close")
        if price_df.empty:
            return pd.Series(dtype=float)

        price_col = price_df.columns[0]
        use_log_returns = not self._screener.synthetic_tickers.get(symbol, False)
        monthly_ret = monthly_returns(
            price_df,
            price_col=price_col,
            use_log_returns=use_log_returns,
        )

        series = monthly_ret[price_col]
        if summary.pattern_type == "abnormal_month":
            month_name = summary.payload.get("month")
            if month_name:
                mask = series.index.month_name() == month_name
                series = series[mask]
        return series

    # ------------------------------------------------------------------
    # Low-level utilities
    # ------------------------------------------------------------------
    def _get_price_series(self, symbol: str, price_col: str) -> pd.DataFrame:
        data = self._screener.data[symbol]
        if data.empty:
            return pd.DataFrame(columns=[price_col])
        if price_col not in data.columns:
            price_col = data.columns[0]
        return data[[price_col]].copy()

    def _get_seasonality_session_data(
        self,
        symbol: str,
        params: ScreenParams,
    ) -> Tuple[pd.DataFrame, str, bool]:
        data = self._screener.data[symbol]
        if data.empty:
            return pd.DataFrame(), "Close", False

        session_start = params.seasonality_session_start or "00:00"
        session_end = params.seasonality_session_end or "23:59:59"

        start_time = self._screener._convert_times([session_start])[0]
        end_time = self._screener._convert_times([session_end])[0]
        tz = params.tz or "America/Chicago"

        session_data = filter_session_bars(
            data,
            tz,
            start_time,
            end_time,
        )

        months = self._screener._parse_season_months(params.months, params.season)
        if months:
            session_data = self._screener._filter_by_months(session_data, months)

        if session_data.empty:
            return session_data, "Close", self._screener.synthetic_tickers.get(symbol, False)

        price_col = "Close" if "Close" in session_data.columns else session_data.columns[0]
        is_synthetic = self._screener.synthetic_tickers.get(symbol, False)
        return session_data, price_col, is_synthetic

    @staticmethod
    def _compute_intraday_returns(
        session_data: pd.DataFrame,
        price_col: str,
        is_synthetic: bool,
    ) -> pd.Series:
        if session_data.empty:
            return pd.Series(dtype=float)

        if is_synthetic:
            returns = session_data[price_col].diff()
        else:
            valid_prices = session_data[price_col] > 0
            prices = session_data.loc[valid_prices, price_col]
            returns = pd.Series(index=session_data.index, dtype=float)
            returns.loc[valid_prices] = log_returns(prices)
        return returns.dropna()

    @staticmethod
    def _compute_time_of_day_returns(
        session_data: pd.DataFrame,
        price_col: str,
        is_synthetic: bool,
        target_time: time,
        period_length: Optional[pd.Timedelta],
    ) -> pd.Series:
        returns = PatternExtractor._compute_intraday_returns(session_data, price_col, is_synthetic)
        if returns.empty:
            return pd.Series(dtype=float)

        mask = tod_mask(returns.index, target_time.strftime("%H:%M"))
        if period_length:
            window_bars = max(int(period_length.total_seconds() / 60 / 5), 1)
            daily_returns = aggregate_window(returns, mask, window_bars)
        else:
            daily_returns = returns[mask]

        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            daily_returns.index = pd.to_datetime(daily_returns.index)
        daily_returns = daily_returns.groupby(daily_returns.index.date).sum()
        daily_returns.index = pd.to_datetime(daily_returns.index)
        return daily_returns

    @staticmethod
    def _infer_pattern_type(pattern: Mapping[str, Any]) -> str:
        for key in ("type", "pattern_type", "momentum_type"):
            value = pattern.get(key)
            if value:
                return str(value)
        return pattern.get("description", "pattern").replace(" ", "_")

    @staticmethod
    def _extract_strength(pattern: Mapping[str, Any]) -> Optional[float]:
        for key in ("strength", "f_stat", "correlation", "sharpe", "t_stat"):
            value = pattern.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _weekday_name_to_number(name: str) -> int:
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        try:
            return weekdays[name.lower()]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown weekday '{name}'") from exc


__all__ = ["PatternExtractor", "PatternSummary"]