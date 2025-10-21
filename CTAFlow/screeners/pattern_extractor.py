"""Utilities for organising and analysing screener pattern output."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

try:  # Optional dependency during lightweight usage
    from .historical_screener import HistoricalScreener, ScreenParams
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    HistoricalScreener = Any  # type: ignore[assignment]
    ScreenParams = Any  # type: ignore[assignment]
from .orderflow_scan import OrderflowParams
from ..utils.seasonal import aggregate_window, log_returns, monthly_returns, tod_mask
from ..utils.session import filter_session_bars


ScreenParamLike = Union[ScreenParams, OrderflowParams]


@dataclass
class PatternSummary:
    """Container describing a statistically significant pattern."""

    key: str
    symbol: str
    source_screen: str
    screen_params: Optional[ScreenParamLike]
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
        screen_params: Optional[Sequence[ScreenParamLike]] = None,
    ) -> None:
        self._screener = screener
        self._results = results
        self._screen_params: Dict[str, ScreenParamLike] = (
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

        if summary.metadata.get("screen_type") == "orderflow" or summary.pattern_type.startswith("orderflow_"):
            return self._extract_orderflow_series(symbol, summary)

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

                if self._is_orderflow_result(result):
                    for summary in self._iter_orderflow_summaries(
                        symbol=symbol,
                        screen_name=screen_name,
                        ticker_result=result,
                        params=params,
                    ):
                        symbol_patterns[summary.key] = summary
                    continue

                for summary in self._iter_historical_summaries(
                    symbol=symbol,
                    screen_name=screen_name,
                    screen_type=screen_type,
                    ticker_result=result,
                    params=params,
                ):
                    symbol_patterns[summary.key] = summary

    def _iter_historical_summaries(
        self,
        symbol: str,
        screen_name: str,
        screen_type: Optional[str],
        ticker_result: Mapping[str, Any],
        params: Optional[ScreenParamLike],
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

    def _iter_orderflow_summaries(
        self,
        symbol: str,
        screen_name: str,
        ticker_result: Mapping[str, Any],
        params: Optional[ScreenParamLike],
    ) -> Iterable[PatternSummary]:
        weekly = ticker_result.get("df_weekly")
        if isinstance(weekly, pd.DataFrame) and not weekly.empty:
            mask_series = weekly.get("sig_fdr_5pct")
            if mask_series is not None:
                mask_series = mask_series.fillna(False).astype(bool)
            else:
                mask_series = pd.Series(True, index=weekly.index)
            for row in weekly.loc[mask_series].itertuples(index=False):
                payload = row._asdict()
                metric = str(payload.get("metric", ""))
                weekday = str(payload.get("weekday", ""))
                bias = self._orderflow_bias(metric, payload.get("mean"))
                metadata = {
                    "pattern_origin": "orderflow_weekly",
                    "screen_type": "orderflow",
                    "orderflow_bias": bias,
                }
                yield PatternSummary(
                    key="|".join(filter(None, [screen_name, "orderflow_weekly", metric, weekday])),
                    symbol=symbol,
                    source_screen=screen_name,
                    screen_params=params,
                    pattern_type="orderflow_weekly",
                    description=self._format_orderflow_weekly_description(metric, weekday, payload, bias),
                    strength=self._orderflow_strength(payload.get("t_stat"), payload.get("mean")),
                    payload=dict(payload),
                    metadata=metadata,
                )

        wom = ticker_result.get("df_wom_weekday")
        if isinstance(wom, pd.DataFrame) and not wom.empty:
            mask_series = wom.get("sig_fdr_5pct")
            if mask_series is not None:
                mask_series = mask_series.fillna(False).astype(bool)
            else:
                mask_series = pd.Series(True, index=wom.index)
            for row in wom.loc[mask_series].itertuples(index=False):
                payload = row._asdict()
                metric = str(payload.get("metric", ""))
                weekday = str(payload.get("weekday", ""))
                wom_value = payload.get("week_of_month")
                bias = self._orderflow_bias(metric, payload.get("mean"))
                parts = [screen_name, "orderflow_week_of_month", metric]
                if wom_value is not None and not pd.isna(wom_value):
                    parts.append(f"w{int(wom_value)}")
                parts.append(weekday)
                metadata = {
                    "pattern_origin": "orderflow_week_of_month",
                    "screen_type": "orderflow",
                    "orderflow_bias": bias,
                }
                yield PatternSummary(
                    key="|".join(filter(None, parts)),
                    symbol=symbol,
                    source_screen=screen_name,
                    screen_params=params,
                    pattern_type="orderflow_week_of_month",
                    description=self._format_orderflow_wom_description(metric, weekday, wom_value, payload, bias),
                    strength=self._orderflow_strength(payload.get("t_stat"), payload.get("mean")),
                    payload=dict(payload),
                    metadata=metadata,
                )

        peak = ticker_result.get("df_weekly_peak_pressure")
        if isinstance(peak, pd.DataFrame) and not peak.empty:
            for row in peak.itertuples(index=False):
                payload = row._asdict()
                metric = str(payload.get("metric", ""))
                weekday = str(payload.get("weekday", ""))
                clock_time = payload.get("clock_time")
                bias = str(payload.get("pressure_bias", "neutral"))
                parts = [screen_name, "orderflow_peak_pressure", metric, weekday]
                if clock_time is not None:
                    parts.append(str(clock_time))
                metadata = {
                    "pattern_origin": "orderflow_peak_pressure",
                    "screen_type": "orderflow",
                    "orderflow_bias": bias,
                }
                yield PatternSummary(
                    key="|".join(filter(None, parts)),
                    symbol=symbol,
                    source_screen=screen_name,
                    screen_params=params,
                    pattern_type="orderflow_peak_pressure",
                    description=self._format_orderflow_peak_description(metric, weekday, clock_time, payload, bias),
                    strength=self._orderflow_strength(payload.get("seasonality_t_stat"), payload.get("intraday_mean")),
                    payload=dict(payload),
                    metadata=metadata,
                )

        events = ticker_result.get("df_events")
        if isinstance(events, pd.DataFrame) and not events.empty:
            if "max_abs_z" in events.columns:
                ranked = events.nlargest(25, "max_abs_z")
            else:
                ranked = events.head(25)
            for row in ranked.itertuples(index=False):
                payload = row._asdict()
                metric = str(payload.get("metric", ""))
                direction = str(payload.get("direction", ""))
                ts_start = payload.get("ts_start")
                parts = [screen_name, "orderflow_event_run", metric, direction]
                if ts_start is not None:
                    parts.append(str(ts_start))
                metadata = {
                    "pattern_origin": "orderflow_events",
                    "screen_type": "orderflow",
                }
                yield PatternSummary(
                    key="|".join(filter(None, parts)),
                    symbol=symbol,
                    source_screen=screen_name,
                    screen_params=params,
                    pattern_type="orderflow_event_run",
                    description=self._format_orderflow_event_description(metric, direction, payload),
                    strength=self._orderflow_strength(payload.get("max_abs_z"), None),
                    payload=dict(payload),
                    metadata=metadata,
                )

    def _build_summary(
        self,
        symbol: str,
        screen_name: str,
        screen_type: Optional[str],
        params: Optional[ScreenParamLike],
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

    # ------------------------------------------------------------------
    # Orderflow helpers
    # ------------------------------------------------------------------
    def _is_orderflow_result(self, ticker_result: Mapping[str, Any]) -> bool:
        keys = set(ticker_result.keys())
        if {"df_weekly", "df_events", "df_buckets"}.intersection(keys):
            metadata = ticker_result.get("metadata")
            return isinstance(metadata, Mapping)
        return False

    def _get_orderflow_result(self, screen_name: str, symbol: str) -> Optional[Mapping[str, Any]]:
        screen_results = self._results.get(screen_name)
        if not isinstance(screen_results, Mapping):
            return None
        result = screen_results.get(symbol)
        if not isinstance(result, Mapping):
            return None
        if not self._is_orderflow_result(result):
            return None
        return result

    def _extract_orderflow_series(self, symbol: str, summary: PatternSummary) -> pd.Series:
        result = self._get_orderflow_result(summary.source_screen, symbol)
        if result is None:
            return pd.Series(dtype=float)

        bucket_df = result.get("df_buckets")
        if not isinstance(bucket_df, pd.DataFrame) or bucket_df.empty:
            return pd.Series(dtype=float)

        metric = summary.payload.get("metric")
        if metric is None or metric not in bucket_df.columns:
            return pd.Series(dtype=float)

        df = bucket_df.copy()
        df["ts_end"] = pd.to_datetime(df["ts_end"])
        df = df.dropna(subset=["ts_end", metric])

        pattern_type = summary.pattern_type

        if pattern_type == "orderflow_weekly":
            weekday = summary.payload.get("weekday")
            if weekday is None or "weekday" not in df.columns:
                return pd.Series(dtype=float)
            mask = df["weekday"].astype(str) == str(weekday)
        elif pattern_type == "orderflow_week_of_month":
            mask = pd.Series(True, index=df.index)
            weekday = summary.payload.get("weekday")
            if weekday is not None and "weekday" in df.columns:
                mask &= df["weekday"].astype(str) == str(weekday)
            week_of_month = summary.payload.get("week_of_month")
            if week_of_month is not None and "week_of_month" in df.columns:
                try:
                    week_value = int(week_of_month)
                except (TypeError, ValueError):
                    week_value = None
                if week_value is not None:
                    mask &= df["week_of_month"].astype(int) == week_value
        elif pattern_type == "orderflow_peak_pressure":
            mask = pd.Series(True, index=df.index)
            weekday = summary.payload.get("weekday")
            if weekday is not None and "weekday" in df.columns:
                mask &= df["weekday"].astype(str) == str(weekday)
            clock_time = summary.payload.get("clock_time")
            if clock_time is not None and "clock_time" in df.columns:
                normalized = self._normalize_time_value(clock_time)
                mask &= df["clock_time"].apply(self._normalize_time_value) == normalized
        elif pattern_type == "orderflow_event_run":
            ts_start = summary.payload.get("ts_start")
            ts_end = summary.payload.get("ts_end")
            if ts_start is None or ts_end is None:
                return pd.Series(dtype=float)
            start_ts = pd.to_datetime(ts_start)
            end_ts = pd.to_datetime(ts_end)
            mask = (df["ts_end"] >= start_ts) & (df["ts_end"] <= end_ts)
        else:
            return pd.Series(dtype=float)

        if mask.empty or not mask.any():
            return pd.Series(dtype=float)

        series = df.loc[mask, ["ts_end", metric]].set_index("ts_end")[metric]
        series.name = metric
        return series.sort_index()

    @staticmethod
    def _normalize_time_value(value: Any) -> Optional[time]:
        if value is None:
            return None
        if isinstance(value, time):
            return value
        try:
            return pd.to_datetime(value).time()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _orderflow_bias(metric: str, mean_value: Any) -> str:
        try:
            mean = float(mean_value)
        except (TypeError, ValueError):
            return "neutral"
        if pd.isna(mean) or mean == 0:
            return "neutral"
        lower_metric = str(metric).lower()
        if lower_metric in {"buy_pressure", "quote_buy_pressure"}:
            return "buy" if mean > 0 else "sell"
        if lower_metric == "sell_pressure":
            return "sell" if mean > 0 else "buy"
        return "positive" if mean > 0 else "negative"

    @staticmethod
    def _orderflow_strength(primary: Any, fallback: Any) -> Optional[float]:
        for value in (primary, fallback):
            if value is None:
                continue
            try:
                return float(abs(value))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _format_orderflow_weekly_description(
        metric: str,
        weekday: str,
        payload: Dict[str, Any],
        bias: str,
    ) -> str:
        mean_value = payload.get("mean")
        t_stat = payload.get("t_stat")
        q_value = payload.get("q_value")
        parts = [f"{metric} bias {bias} on {weekday}"]
        if pd.notna(mean_value):
            parts.append(f"mean={float(mean_value):.4f}")
        if pd.notna(t_stat):
            parts.append(f"t={float(t_stat):.2f}")
        if pd.notna(q_value):
            parts.append(f"q={float(q_value):.3g}")
        if len(parts) == 1:
            return parts[0]
        return parts[0] + " (" + ", ".join(parts[1:]) + ")"

    @staticmethod
    def _format_orderflow_wom_description(
        metric: str,
        weekday: str,
        wom_value: Any,
        payload: Dict[str, Any],
        bias: str,
    ) -> str:
        prefix = f"{metric} week-of-month"
        if wom_value is not None and not pd.isna(wom_value):
            try:
                prefix += f" {int(wom_value)}"
            except (TypeError, ValueError):
                pass
        prefix += f" {weekday} bias {bias}"
        details: List[str] = []
        mean_value = payload.get("mean")
        t_stat = payload.get("t_stat")
        q_value = payload.get("q_value")
        if pd.notna(mean_value):
            details.append(f"mean={float(mean_value):.4f}")
        if pd.notna(t_stat):
            details.append(f"t={float(t_stat):.2f}")
        if pd.notna(q_value):
            details.append(f"q={float(q_value):.3g}")
        if details:
            prefix += " (" + ", ".join(details) + ")"
        return prefix

    @staticmethod
    def _format_orderflow_peak_description(
        metric: str,
        weekday: str,
        clock_time: Any,
        payload: Dict[str, Any],
        bias: str,
    ) -> str:
        time_display = str(clock_time) if clock_time is not None else "?"
        details: List[str] = []
        seasonality_mean = payload.get("seasonality_mean")
        intraday_mean = payload.get("intraday_mean")
        if pd.notna(seasonality_mean):
            details.append(f"seasonal_mean={float(seasonality_mean):.4f}")
        if pd.notna(intraday_mean):
            details.append(f"intraday_mean={float(intraday_mean):.4f}")
        detail_str = ""
        if details:
            detail_str = " (" + ", ".join(details) + ")"
        return f"{metric} peak {bias} on {weekday} at {time_display}{detail_str}"

    @staticmethod
    def _format_orderflow_event_description(
        metric: str,
        direction: str,
        payload: Dict[str, Any],
    ) -> str:
        parts = [f"{metric} {direction} run"]
        max_abs_z = payload.get("max_abs_z")
        if pd.notna(max_abs_z):
            parts.append(f"|z|={float(max_abs_z):.2f}")
        run_len = payload.get("run_len")
        if run_len:
            try:
                parts.append(f"len={int(run_len)}")
            except (TypeError, ValueError):
                pass
        ts_start = payload.get("ts_start")
        ts_end = payload.get("ts_end")
        if ts_start is not None and ts_end is not None:
            parts.append(f"{ts_start}→{ts_end}")
        return " ".join(parts)


__all__ = ["PatternExtractor", "PatternSummary"]
