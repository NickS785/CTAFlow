from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
                "daily_pnl": pd.Series(dtype=float),
            }

        required = {"returns_x", "returns_y"}
        missing = required.difference(xy.columns)
        if missing:
            raise KeyError(f"XY frame missing required columns: {sorted(missing)}")

        frame = xy.dropna(subset=["returns_x", "returns_y"]).copy()
        frame = self._drop_duplicate_rows(frame)
        if prediction_resolver is not None and not frame.empty:
            frame = prediction_resolver.aggregate(frame)
            frame = self._drop_duplicate_rows(frame)
            if frame.empty:
                return {
                    "pnl": pd.Series(dtype=float),
                    "positions": pd.Series(dtype=float),
                    "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                    "monthly": pd.Series(dtype=float),
                    "cumulative": pd.Series(dtype=float),
                    "daily_pnl": pd.Series(dtype=float),
                }
        if frame.empty:
            return {
                "pnl": pd.Series(dtype=float),
                "positions": pd.Series(dtype=float),
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                "monthly": pd.Series(dtype=float),
                "cumulative": pd.Series(dtype=float),
                "daily_pnl": pd.Series(dtype=float),
            }

        frame = self._collision_resolver.resolve(frame, group_field=group_field)
        frame = self._drop_duplicate_rows(frame)

        direction = np.sign(frame["returns_x"])
        if use_side_hint and "side_hint" in frame.columns:
            hinted = frame["side_hint"].replace(0, np.nan)
            direction = hinted.fillna(direction)
        if prediction_resolver is not None and "prediction_position" in frame.columns:
            hinted = frame["prediction_position"].replace(0, np.nan)
            direction = hinted.fillna(direction)

        signal_mask = frame["returns_x"].abs() >= float(threshold)
        raw_positions = direction.where(signal_mask, 0.0)

        trade_rows = frame.loc[signal_mask].copy()
        if not trade_rows.empty and "ts_decision" in trade_rows.columns:
            trade_rows["_trade_day"] = pd.to_datetime(
                trade_rows["ts_decision"], errors="coerce"
            ).dt.normalize()
        else:
            trade_rows["_trade_day"] = pd.NaT

        if "ts_decision" in frame.columns:
            trade_index = pd.to_datetime(frame["ts_decision"], errors="coerce")
            trade_index.name = "ts_decision"
        else:
            trade_index = pd.Index(frame.index, name=frame.index.name or "row")

        raw_pnl = raw_positions * frame["returns_y"].astype(float)
        raw_cumulative = raw_pnl.cumsum()
        rolling_max = raw_cumulative.cummax()
        drawdown = raw_cumulative - rolling_max
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        def _assign_trade_index(series: pd.Series) -> pd.Series:
            aligned = series.copy()
            aligned.index = trade_index
            return aligned

        positions = _assign_trade_index(raw_positions)
        pnl = _assign_trade_index(raw_pnl)
        cumulative = _assign_trade_index(raw_cumulative)

        trades = int(signal_mask.sum())
        if not trade_rows.empty:
            trade_days = trade_rows["_trade_day"].dropna()
            if not trade_days.empty:
                if group_field and group_field in trade_rows.columns:
                    combos = trade_rows.loc[trade_days.index, ["_trade_day", group_field]]
                    valid = combos.dropna(subset=[group_field])
                    trades = int(len(valid.drop_duplicates()))
                    missing = combos[combos[group_field].isna()]
                    if not missing.empty:
                        trades += int(missing["_trade_day"].nunique())
                else:
                    trades = int(trade_days.nunique())
        mean_return = float(pnl.mean()) if not pnl.empty else 0.0
        total_return = float(pnl.sum())
        hit_rate = float((pnl > 0).mean()) if not pnl.empty else np.nan
        std_return = float(pnl.std(ddof=0))
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(self.annualisation)
        else:
            sharpe = np.nan

        if isinstance(trade_index, pd.DatetimeIndex):
            valid = trade_index.notna()
            if valid.any():
                monthly = (
                    pnl.iloc[valid]
                    .groupby(trade_index[valid].to_period("M"))
                    .sum()
                    .astype(float)
                )
                daily_pnl = (
                    pnl.iloc[valid]
                    .groupby(trade_index[valid].normalize())
                    .sum()
                    .astype(float)
                )
            else:
                monthly = pd.Series(dtype=float)
                daily_pnl = pd.Series(dtype=float)
        else:
            monthly = pd.Series(dtype=float)
            daily_pnl = pd.Series(dtype=float)

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
            "daily_pnl": daily_pnl,
        }

        if group_field and group_field in frame.columns:
            grouped_frame = trade_rows.copy()
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
                    row_position = raw_positions.loc[idx]
                    row_pnl = raw_pnl.loc[idx]
                    trade_day = row.get("_trade_day")
                    for member in members:
                        stats = grouped_results.setdefault(
                            member,
                            {
                                "total_return": 0.0,
                                "_pnl_count": 0.0,
                                "_trade_days": set(),
                                "_fallback_trade_count": 0,
                            },
                        )
                        stats["total_return"] += float(row_pnl * share)
                        stats["_pnl_count"] += share
                        if row_position != 0:
                            if pd.notna(trade_day):
                                stats["_trade_days"].add(trade_day)
                            else:
                                stats["_fallback_trade_count"] += 1
            formatted: Dict[Any, Dict[str, float]] = {}
            for member, stats in grouped_results.items():
                count = stats.get("_pnl_count", 1.0) or 1.0
                trade_count = len(stats.get("_trade_days", set()))
                trade_count += stats.get("_fallback_trade_count", 0)
                formatted[member] = {
                    "trades": trade_count,
                    "total_return": stats.get("total_return", 0.0),
                    "mean_return": stats.get("total_return", 0.0) / count,
                }
            result["group_breakdown"] = formatted
            result["group_field"] = group_field

        return result

    @staticmethod
    def _drop_duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        subset = [
            col
            for col in [
                "ts_decision",
                "gate",
                "pattern_type",
                "returns_x",
                "returns_y",
                "side_hint",
            ]
            if col in frame.columns
        ]
        if subset:
            return frame.drop_duplicates(subset=subset).copy()
        return frame.drop_duplicates().copy()


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
        if tickers is not None:
            iterable = list(dict.fromkeys(tickers))
        else:
            iterable = screen_results.keys()
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

        combined = pd.concat(frames).reset_index(drop=True)
        return combined.drop_duplicates().reset_index(drop=True)

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


def diagnose_alignment(a: pd.Series, b: pd.Series) -> dict:
    def _info(s: pd.Series):
        idx = s.index
        tzs = str(getattr(idx, "tz", None))
        return {
            "first": str(idx.min()) if len(idx) else "NA",
            "last":  str(idx.max()) if len(idx) else "NA",
            "len":   int(len(idx)),
            "na":    int(s.isna().sum()),
            "tz":    tzs,
        }
    info_a = _info(a); info_b = _info(b)
    overlap_ts = len(a.index.intersection(b.index))
    return {"a": info_a, "b": info_b, "overlap_exact_timestamps": overlap_ts}

def _ensure_dt_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    return s.dropna().sort_index()

def _coerce_tz(s: pd.Series, tz: Optional[str]) -> pd.Series:
    s = _ensure_dt_index(s.copy())
    if tz is not None:
        if s.index.tz is None:
            s.index = s.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        else:
            s.index = s.index.tz_convert(tz)
        s.index = s.index.tz_localize(None)  # make naive for exact equality
    return s

def _align_on_date(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    a = _ensure_dt_index(a); b = _ensure_dt_index(b)
    # normalize to midnight (date) and keep the last value per day (close)
    a = a.groupby(a.index.normalize()).last()
    b = b.groupby(b.index.normalize()).last()
    common = a.index.intersection(b.index)
    return a.reindex(common), b.reindex(common)

def build_overlay_frame(
    cumulative: pd.Series,
    price: Optional[pd.Series] = None,
    *,
    scale: Literal["minmax","rebase100","zscore"] = "minmax",
    align: Literal["date","timestamp"] = "date",
    tz: Optional[str] = "America/Chicago",
    dropna: bool = True,
    require_overlap: bool = True,
) -> pd.DataFrame:
    """
    Returns:
      DataFrame with columns:
        - cum_pnl
        - price_scaled (if price is provided)
    Defaults:
      - align='date' (daily close alignment)
      - tz='America/Chicago' (both series normalized then made tz-naive)
    """
    if cumulative is None or len(cumulative) == 0:
        return pd.DataFrame(columns=["cum_pnl","price_scaled"])

    cum = pd.to_numeric(pd.Series(cumulative.squeeze()), errors="coerce").dropna()
    cum = _coerce_tz(cum, tz)

    if price is None or len(price) == 0:
        return pd.DataFrame({"cum_pnl": cum})

    px = pd.to_numeric(pd.Series(price.squeeze()), errors="coerce").dropna()
    px = _coerce_tz(px, tz)

    if align == "date":
        cum_al, px_al = _align_on_date(cum, px)
    else:
        idx = cum.index.intersection(px.index)
        cum_al = cum.reindex(idx)
        px_al  = px.reindex(idx)

    if dropna:
        m = cum_al.notna() & px_al.notna()
        cum_al, px_al = cum_al[m], px_al[m]

    if require_overlap and (len(cum_al) == 0 or len(px_al) == 0):
        diag = diagnose_alignment(cum, px)
        raise ValueError(
            "No overlap after alignment. "
            f"cum[{diag['a']['first']} → {diag['a']['last']} tz={diag['a']['tz']}], "
            f"price[{diag['b']['first']} → {diag['b']['last']} tz={diag['b']['tz']}]."
        )

    # scale price to the cum range
    lo, hi = float(cum_al.min()), float(cum_al.max())
    span = max(hi - lo, 1e-12)

    if scale == "minmax":
        plo, phi = float(px_al.min()), float(px_al.max())
        pspan = max(phi - plo, 1e-12)
        px_scaled = (px_al - plo) / pspan
    elif scale == "rebase100":
        base = px_al.iloc[0]
        px_scaled = (px_al / base)
        px_scaled = (px_scaled - px_scaled.min()) / max(px_scaled.max() - px_scaled.min(), 1e-12)
    elif scale == "zscore":
        mu, sd = float(px_al.mean()), float(px_al.std(ddof=0)) or 1.0
        px_scaled = (px_al - mu) / sd
        px_scaled = (px_scaled - px_scaled.min()) / max(px_scaled.max() - px_scaled.min(), 1e-12)
    else:
        raise ValueError(scale)

    return pd.DataFrame({"cum_pnl": cum_al, "price_scaled": lo + px_scaled * span})

def plot_backtest_results(
    cumulative: pd.Series,
    price: Optional[pd.Series] = None,
    *,
    scale: Literal["minmax", "rebase100", "zscore"] = "minmax",
    ax=None,
):
    """
    Convenience wrapper. Returns the matplotlib Axes.
    (Leave styling to caller; this function only plots two aligned lines.)
    """
    frame = build_overlay_frame(cumulative, price, scale=scale)
    if frame.empty:
        return None
    if ax is None:
        fig, ax = plt.subplots()
    frame["cum_pnl"].plot(ax=ax, label="Cumulative PnL")
    if "price_scaled" in frame:
        frame["price_scaled"].plot(ax=ax, label="Price (scaled)")
    ax.legend()
    ax.set_title("Cumulative PnL with optional price overlay")
    return ax
