from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


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

    def threshold(
        self,
        xy: pd.DataFrame,
        *,
        threshold: float = 0.0,
        use_side_hint: bool = True,
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
        if frame.empty:
            return {
                "pnl": pd.Series(dtype=float),
                "positions": pd.Series(dtype=float),
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                "monthly": pd.Series(dtype=float),
                "cumulative": pd.Series(dtype=float),
            }

        direction = np.sign(frame["returns_x"])
        if use_side_hint and "side_hint" in frame.columns:
            hinted = frame["side_hint"].replace(0, np.nan)
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

        return {
            "pnl": pnl,
            "positions": positions,
            "summary": summary,
            "monthly": monthly,
            "cumulative": cumulative,
        }
