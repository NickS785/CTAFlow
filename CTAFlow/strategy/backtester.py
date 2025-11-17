from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .prediction_to_position import PredictionToPosition


@dataclass
class BacktestSummary:
    """Container summarising threshold backtest results."""

    total_return: float
    mean_return: float
    hit_rate: float
    sharpe: float
    max_drawdown: float
    trades: int


class ScreenerBacktester:
    """Lightweight backtester operating on ``ScreenerPipeline.build_xy`` outputs."""

    def __init__(self, *, annualisation: int = 252, risk_free_rate: float = 0.0) -> None:
        self.annualisation = annualisation
        self.risk_free_rate = risk_free_rate
        self._collision_resolver = PredictionToPosition()

    def threshold(
        self,
        xy: pd.DataFrame,
        *,
        threshold: float = 0.0,
        use_side_hint: bool = True,
        group_field: Optional[str] = None,
        prediction_resolver: Optional["PredictionToPosition"] = None,
    ) -> Dict[str, Any]:
        if xy.empty:
            empty = pd.Series(dtype=float)
            return {
                "pnl": empty,
                "positions": empty,
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                "monthly": pd.Series(dtype=float),
                "cumulative": empty,
            }

        required = {"returns_x", "returns_y"}
        missing = required.difference(xy.columns)
        if missing:
            raise KeyError(f"XY frame missing required columns: {sorted(missing)}")

        frame = xy.dropna(subset=["returns_x", "returns_y"]).copy()
        if prediction_resolver is not None and not frame.empty:
            frame = prediction_resolver.aggregate(frame)
            if frame.empty:
                return {
                    "pnl": pd.Series(dtype=float),
                    "positions": pd.Series(dtype=float),
                    "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                    "monthly": pd.Series(dtype=float),
                    "cumulative": pd.Series(dtype=float),
                }
        if frame.empty:
            return {
                "pnl": pd.Series(dtype=float),
                "positions": pd.Series(dtype=float),
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                "monthly": pd.Series(dtype=float),
                "cumulative": pd.Series(dtype=float),
            }

        frame = self._collision_resolver.resolve(frame, group_field=group_field)

        direction = np.sign(frame["returns_x"])
        if use_side_hint and "side_hint" in frame.columns:
            hinted = frame["side_hint"].replace(0, np.nan)
            direction = hinted.fillna(direction)
        if prediction_resolver is not None and "prediction_position" in frame.columns:
            hinted = frame["prediction_position"].replace(0, np.nan)
            direction = hinted.fillna(direction)

        signal_mask = frame["returns_x"].abs() >= float(threshold)
        positions = direction.where(signal_mask, 0.0)

        pnl = positions * frame["returns_y"].astype(float)
        cumulative = pnl.cumsum()
        rolling_max = cumulative.cummax()
        drawdown = cumulative - rolling_max
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        trades = int(signal_mask.sum())
        mean_return = float(pnl.mean()) if not pnl.empty else 0.0
        total_return = float(pnl.sum())
        hit_rate = float((pnl > 0).mean()) if not pnl.empty else np.nan
        std_return = float(pnl.std(ddof=0))
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(self.annualisation)
        else:
            sharpe = np.nan

        if "ts_decision" in frame.columns:
            ts = pd.to_datetime(frame["ts_decision"])
            monthly = pnl.groupby(ts.dt.to_period("M")).sum().astype(float)
        else:
            monthly = pd.Series(dtype=float)

        summary = BacktestSummary(
            total_return=total_return,
            mean_return=mean_return,
            hit_rate=hit_rate,
            sharpe=float(sharpe) if np.isfinite(sharpe) else np.nan,
            max_drawdown=max_drawdown,
            trades=trades,
        )

        result: Dict[str, Any] = {
            "pnl": pnl,
            "positions": positions,
            "summary": summary,
            "monthly": monthly,
            "cumulative": cumulative,
        }

        if group_field and group_field in frame.columns:
            grouped_frame = frame.loc[signal_mask].copy()
            grouped_results: Dict[Any, Dict[str, float]] = {}
            if not grouped_frame.empty:
                for idx, row in grouped_frame.iterrows():
                    members = row.get("_group_members") if "_group_members" in row else None
                    if not members:
                        value = row.get(group_field)
                        members = [] if pd.isna(value) else [value]
                    if not members:
                        continue
                    share = 1.0 / len(members)
                    row_position = positions.loc[idx]
                    row_pnl = pnl.loc[idx]
                    for member in members:
                        stats = grouped_results.setdefault(
                            member,
                            {"trades": 0, "total_return": 0.0, "_pnl_count": 0.0},
                        )
                        stats["trades"] += int(row_position != 0)
                        stats["total_return"] += float(row_pnl * share)
                        stats["_pnl_count"] += share
            formatted: Dict[Any, Dict[str, float]] = {}
            for member, stats in grouped_results.items():
                count = stats.get("_pnl_count", 1.0) or 1.0
                formatted[member] = {
                    "trades": stats.get("trades", 0),
                    "total_return": stats.get("total_return", 0.0),
                    "mean_return": stats.get("total_return", 0.0) / count,
                }
            result["group_breakdown"] = formatted
            result["group_field"] = group_field

        return result


class MomentumBacktester:
    """Utility for running light backtests on historical momentum screen outputs."""

    def __init__(
        self,
        *,
        annualisation: int = 252,
        risk_free_rate: float = 0.0,
        backtester: Optional[ScreenerBacktester] = None,
    ) -> None:
        self._backtester = backtester or ScreenerBacktester(
            annualisation=annualisation,
            risk_free_rate=risk_free_rate,
        )

    @staticmethod
    def _resolve_screen_results(
        results: Mapping[str, Any], screen_name: Optional[str]
    ) -> Mapping[str, Any]:
        if screen_name is None:
            return results
        try:
            screen_results = results[screen_name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Screen '{screen_name}' not found in results") from exc
        if not isinstance(screen_results, Mapping):  # pragma: no cover - defensive
            raise TypeError(
                f"Screen '{screen_name}' results must be a mapping, got {type(screen_results)!r}"
            )
        return screen_results

    def build_xy(
        self,
        results: Mapping[str, Any],
        *,
        screen_name: Optional[str] = None,
        session_key: str = "session_0",
        predictor: str = "closing_returns",
        target: str = "rest_of_session_returns",
        tickers: Optional[Sequence[str]] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Convert a nested momentum screen result dictionary into an ``XY`` frame.

        Parameters
        ----------
        results
            Mapping produced by :meth:`HistoricalScreener.intraday_momentum_screen`.
        screen_name
            Optional name of the screen to extract. If ``None`` the top-level
            mapping is assumed to already be ticker keyed.
        session_key
            Session identifier such as ``"session_0"``.
        predictor
            Key inside ``return_series`` used for ``returns_x``.
        target
            Key inside ``return_series`` used for ``returns_y``.
        tickers
            Optional subset of tickers to include.
        dropna
            Whether to drop rows missing either ``returns_x`` or ``returns_y``.
        """

        if not isinstance(results, Mapping):  # pragma: no cover - defensive
            raise TypeError(f"results must be a mapping, got {type(results)!r}")

        screen_results = self._resolve_screen_results(results, screen_name)
        columns = [
            "returns_x",
            "returns_y",
            "ticker",
            "session_key",
            "screen_name",
            "ts_decision",
        ]
        frames: List[pd.DataFrame] = []
        iterable = tickers if tickers is not None else screen_results.keys()
        for ticker in iterable:
            ticker_data = screen_results.get(ticker)
            if not isinstance(ticker_data, Mapping):
                continue
            session_data = ticker_data.get(session_key)
            if not isinstance(session_data, Mapping):
                continue
            return_series = session_data.get("return_series")
            if not isinstance(return_series, Mapping):
                continue
            predictor_series = return_series.get(predictor)
            target_series = return_series.get(target)
            if not isinstance(predictor_series, pd.Series) or not isinstance(
                target_series, pd.Series
            ):
                continue
            predictor_clean = pd.to_numeric(predictor_series, errors="coerce")
            target_clean = pd.to_numeric(target_series, errors="coerce")
            combined = pd.concat(
                [
                    predictor_clean.rename("returns_x"),
                    target_clean.rename("returns_y"),
                ],
                axis=1,
                join="inner",
            )
            if dropna:
                combined = combined.dropna(subset=["returns_x", "returns_y"])
            if combined.empty:
                continue
            decision_ts = pd.to_datetime(combined.index)
            combined = combined.assign(
                ticker=ticker,
                session_key=session_key,
                screen_name=screen_name,
                ts_decision=decision_ts,
            )
            frames.append(combined)

        if not frames:
            return pd.DataFrame(columns=columns)

        return pd.concat(frames).reset_index(drop=True)

    def threshold_backtest(
        self,
        results: Mapping[str, Any],
        *,
        screen_name: Optional[str] = None,
        session_key: str = "session_0",
        predictor: str = "closing_returns",
        target: str = "rest_of_session_returns",
        tickers: Optional[Sequence[str]] = None,
        threshold: float = 0.0,
        use_side_hint: bool = False,
        group_field: str = "ticker",
        prediction_resolver: Optional["PredictionToPosition"] = None,
    ) -> Dict[str, Any]:
        """Run a quick threshold backtest over the extracted return pairs."""

        xy = self.build_xy(
            results,
            screen_name=screen_name,
            session_key=session_key,
            predictor=predictor,
            target=target,
            tickers=tickers,
        )
        return self._backtester.threshold(
            xy,
            threshold=threshold,
            use_side_hint=use_side_hint,
            group_field=group_field,
            prediction_resolver=prediction_resolver,
        )
