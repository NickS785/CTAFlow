from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class TradingMode(str, Enum):
    """Trading strategy mode for cost calculation.

    CONTINUOUS: Position held across periods, costs on position changes only.
                Use for swing trading or when positions roll over.

    ROUND_TRIP: Each non-flat prediction is a complete trade (entry + exit).
                Use for intraday/end-of-day momentum where you enter at
                prediction time and exit at target time each period.
    """
    CONTINUOUS = "continuous"
    ROUND_TRIP = "round_trip"


@dataclass
class BacktestAttributionResult:
    """Results from backtesting predictions.

    Attributes
    ----------
    summary : Dict[str, float]
        Aggregate performance metrics
    frame : pd.DataFrame
        Per-period breakdown with positions, returns, costs, PnL
    pnl_by_side : pd.DataFrame
        PnL aggregated by position direction (long/short/flat)
    pnl_by_meta : Dict[str, pd.DataFrame]
        PnL aggregated by each meta field provided
    trade_stats : Dict[str, float]
        Trade-level statistics (win rate, avg win/loss, profit factor)
    """
    summary: Dict[str, float]
    frame: pd.DataFrame
    pnl_by_side: pd.DataFrame
    pnl_by_meta: Dict[str, pd.DataFrame]
    trade_stats: Dict[str, float] = field(default_factory=dict)


def predictions_to_positions(
    predictions: np.ndarray,
    *,
    task: str = "classification",
    long_class: int = 2,
    short_class: int = 0,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Convert model outputs to position vector in {-1, 0, +1}.

    Parameters
    ----------
    predictions : np.ndarray
        Model outputs. For classification, either logits/probabilities (2D)
        or class labels (1D). For regression, continuous values.
    task : str
        'classification' or 'regression'
    long_class : int
        Class index for long position (classification only)
    short_class : int
        Class index for short position (classification only)
    threshold : float
        Threshold for regression: values > threshold -> long,
        values < -threshold -> short

    Returns
    -------
    np.ndarray
        Position vector with values in {-1, 0, +1}
    """
    preds = np.asarray(predictions)
    if task == "classification":
        if preds.ndim == 2:
            labels = preds.argmax(axis=1)
        else:
            labels = preds.astype(int)
        pos = np.zeros(len(labels), dtype=np.float32)
        pos[labels == long_class] = 1.0
        pos[labels == short_class] = -1.0
        return pos

    # regression-like outputs
    vals = preds.reshape(-1).astype(np.float32)
    pos = np.zeros_like(vals)
    pos[vals > threshold] = 1.0
    pos[vals < -threshold] = -1.0
    return pos


def _meta_col(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr[:, -1]
    arr = arr.reshape(-1)
    if len(arr) != n:
        raise ValueError(f"Meta field length mismatch: expected {n}, got {len(arr)}")
    return arr


def _compute_trade_stats(positions: np.ndarray, net_pnl: np.ndarray) -> Dict[str, float]:
    """Compute trade-level statistics for non-flat positions."""
    # Only consider actual trades (non-flat positions)
    trade_mask = positions != 0
    trade_pnl = net_pnl[trade_mask]

    if len(trade_pnl) == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_trade_pnl": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]

    n_trades = len(trade_pnl)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else float('inf') if gross_profit > 0 else 0.0

    # Expectancy: avg win * win_rate - avg loss * loss_rate
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss  # avg_loss is negative

    return {
        "n_trades": float(n_trades),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "avg_trade_pnl": float(trade_pnl.mean()),
        "best_trade": float(trade_pnl.max()),
        "worst_trade": float(trade_pnl.min()),
    }


def backtest_from_predictions(
    predictions: np.ndarray,
    returns: Sequence[float],
    *,
    dates: Optional[Sequence[Any]] = None,
    task: str = "classification",
    long_class: int = 2,
    short_class: int = 0,
    threshold: float = 0.0,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    mode: TradingMode = TradingMode.ROUND_TRIP,
    meta: Optional[Mapping[str, Any]] = None,
) -> BacktestAttributionResult:
    """
    Backtest predictions and return PnL attribution tables.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (logits, probabilities, or regression values)
    returns : Sequence[float]
        Realized returns for each prediction period
    dates : Sequence, optional
        Dates/timestamps for each observation
    task : str
        'classification' or 'regression'
    long_class : int
        Class index for long position (default 2)
    short_class : int
        Class index for short position (default 0)
    threshold : float
        Threshold for regression predictions
    transaction_cost_bps : float
        Transaction cost in basis points (per leg)
    slippage_bps : float
        Slippage in basis points (per leg)
    mode : TradingMode
        ROUND_TRIP: Each prediction = entry + exit, costs charged for both legs
                    on any non-flat position. Use for intraday momentum.
        CONTINUOUS: Costs charged only when position changes. Use for swing trading.
    meta : Mapping, optional
        Additional metadata for PnL attribution (e.g., ticker_id, day_of_week)

    Returns
    -------
    BacktestAttributionResult
        Backtest results with summary stats, per-period frame, and attributions

    Examples
    --------
    >>> # End-of-day momentum strategy (round-trip per prediction)
    >>> result = backtest_from_predictions(
    ...     predictions, returns,
    ...     mode=TradingMode.ROUND_TRIP,
    ...     transaction_cost_bps=2.0,  # 2 bps per leg = 4 bps round-trip
    ...     slippage_bps=1.0,          # 1 bps per leg = 2 bps round-trip
    ... )

    >>> # Swing trading (position held across periods)
    >>> result = backtest_from_predictions(
    ...     predictions, returns,
    ...     mode=TradingMode.CONTINUOUS,
    ...     transaction_cost_bps=2.0,
    ... )
    """
    rets = np.asarray(returns, dtype=np.float32).reshape(-1)
    pos = predictions_to_positions(
        predictions,
        task=task,
        long_class=long_class,
        short_class=short_class,
        threshold=threshold,
    )
    if len(pos) != len(rets):
        raise ValueError(f"Length mismatch: positions={len(pos)} returns={len(rets)}")

    # Cost per leg in decimal form
    cost_per_leg = (transaction_cost_bps + slippage_bps) / 10000.0

    if mode == TradingMode.ROUND_TRIP:
        # Round-trip: each non-flat position incurs entry + exit costs
        # Entry and exit each have transaction + slippage costs
        is_trade = np.abs(pos) > 0
        costs = is_trade.astype(np.float32) * cost_per_leg * 2.0  # 2 legs
        turnover = is_trade.astype(np.float32) * 2.0  # Entry + exit
    else:
        # Continuous: costs only when position changes
        turn = np.abs(np.diff(np.r_[0.0, pos]))
        costs = turn * cost_per_leg
        turnover = turn

    gross = pos * rets
    net = gross - costs
    cum = np.cumsum(net)

    idx = pd.RangeIndex(len(net)) if dates is None else pd.to_datetime(dates)
    frame = pd.DataFrame(
        {
            "position": pos,
            "returns": rets,
            "turnover": turnover,
            "costs": costs,
            "gross_pnl": gross,
            "net_pnl": net,
            "cum_pnl": cum,
        },
        index=idx,
    )

    side = np.where(pos > 0, "long", np.where(pos < 0, "short", "flat"))
    pnl_by_side = frame.assign(side=side).groupby("side", dropna=False)["net_pnl"].agg(["sum", "mean", "count"])

    pnl_by_meta: Dict[str, pd.DataFrame] = {}
    if meta:
        n = len(frame)
        for key, vals in meta.items():
            col = _meta_col(vals, n)
            g = frame.assign(_k=col).groupby("_k", dropna=False)["net_pnl"].agg(["sum", "mean", "count"])
            pnl_by_meta[key] = g.sort_values("sum", ascending=False)

    avg = float(np.mean(net)) if len(net) else 0.0
    std = float(np.std(net)) if len(net) else 0.0
    sharpe = (avg / std * np.sqrt(252.0)) if std > 1e-12 else 0.0
    drawdown = cum - np.maximum.accumulate(cum) if len(cum) else np.array([0.0], dtype=np.float32)

    # Compute trade-level statistics
    trade_stats = _compute_trade_stats(pos, net)

    summary = {
        "n_obs": float(len(frame)),
        "gross_pnl_sum": float(np.sum(gross)),
        "net_pnl_sum": float(np.sum(net)),
        "avg_net_pnl": avg,
        "net_pnl_vol": std,
        "sharpe_like": float(sharpe),
        "max_drawdown": float(np.min(drawdown)),
        "turnover_sum": float(np.sum(turnover)),
        "cost_sum": float(np.sum(costs)),
        "hit_rate": float(np.mean(net > 0.0)) if len(net) else 0.0,
        "trading_mode": mode.value,
    }

    return BacktestAttributionResult(
        summary=summary,
        frame=frame,
        pnl_by_side=pnl_by_side,
        pnl_by_meta=pnl_by_meta,
        trade_stats=trade_stats,
    )


def backtest_eod_momentum(
    predictions: np.ndarray,
    returns: Sequence[float],
    *,
    dates: Optional[Sequence[Any]] = None,
    task: str = "classification",
    long_class: int = 2,
    short_class: int = 0,
    threshold: float = 0.0,
    entry_cost_bps: float = 1.0,
    exit_cost_bps: float = 1.0,
    entry_slippage_bps: float = 0.5,
    exit_slippage_bps: float = 0.5,
    meta: Optional[Mapping[str, Any]] = None,
) -> BacktestAttributionResult:
    """
    Backtest end-of-day momentum strategy with asymmetric entry/exit costs.

    This is a convenience wrapper for strategies where:
    - Each prediction triggers a trade (entry) at prediction time
    - The trade is closed (exit) at target time within the same period
    - Entry and exit may have different costs (e.g., market vs limit orders)

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    returns : Sequence[float]
        Realized returns for the target period
    dates : Sequence, optional
        Dates for each observation
    task : str
        'classification' or 'regression'
    long_class : int
        Class index for long position
    short_class : int
        Class index for short position
    threshold : float
        Threshold for regression predictions
    entry_cost_bps : float
        Transaction cost for entry (basis points)
    exit_cost_bps : float
        Transaction cost for exit (basis points)
    entry_slippage_bps : float
        Slippage for entry (basis points)
    exit_slippage_bps : float
        Slippage for exit (basis points)
    meta : Mapping, optional
        Additional metadata for attribution

    Returns
    -------
    BacktestAttributionResult

    Examples
    --------
    >>> # Livestock EOD momentum: enter at 8:30 AM, exit at 1:00 PM
    >>> result = backtest_eod_momentum(
    ...     predictions, returns,
    ...     entry_cost_bps=1.5,   # Market order entry
    ...     exit_cost_bps=1.0,    # Limit order exit
    ...     entry_slippage_bps=1.0,
    ...     exit_slippage_bps=0.5,
    ... )
    """
    rets = np.asarray(returns, dtype=np.float32).reshape(-1)
    pos = predictions_to_positions(
        predictions,
        task=task,
        long_class=long_class,
        short_class=short_class,
        threshold=threshold,
    )
    if len(pos) != len(rets):
        raise ValueError(f"Length mismatch: positions={len(pos)} returns={len(rets)}")

    # Costs in decimal form
    entry_total = (entry_cost_bps + entry_slippage_bps) / 10000.0
    exit_total = (exit_cost_bps + exit_slippage_bps) / 10000.0
    round_trip_cost = entry_total + exit_total

    # Each non-flat position incurs full round-trip cost
    is_trade = np.abs(pos) > 0
    costs = is_trade.astype(np.float32) * round_trip_cost
    turnover = is_trade.astype(np.float32) * 2.0  # 2 legs per trade

    gross = pos * rets
    net = gross - costs
    cum = np.cumsum(net)

    idx = pd.RangeIndex(len(net)) if dates is None else pd.to_datetime(dates)
    frame = pd.DataFrame(
        {
            "position": pos,
            "returns": rets,
            "turnover": turnover,
            "entry_cost": is_trade.astype(np.float32) * entry_total,
            "exit_cost": is_trade.astype(np.float32) * exit_total,
            "costs": costs,
            "gross_pnl": gross,
            "net_pnl": net,
            "cum_pnl": cum,
        },
        index=idx,
    )

    side = np.where(pos > 0, "long", np.where(pos < 0, "short", "flat"))
    pnl_by_side = frame.assign(side=side).groupby("side", dropna=False)["net_pnl"].agg(["sum", "mean", "count"])

    pnl_by_meta: Dict[str, pd.DataFrame] = {}
    if meta:
        n = len(frame)
        for key, vals in meta.items():
            col = _meta_col(vals, n)
            g = frame.assign(_k=col).groupby("_k", dropna=False)["net_pnl"].agg(["sum", "mean", "count"])
            pnl_by_meta[key] = g.sort_values("sum", ascending=False)

    avg = float(np.mean(net)) if len(net) else 0.0
    std = float(np.std(net)) if len(net) else 0.0
    sharpe = (avg / std * np.sqrt(252.0)) if std > 1e-12 else 0.0
    drawdown = cum - np.maximum.accumulate(cum) if len(cum) else np.array([0.0], dtype=np.float32)

    trade_stats = _compute_trade_stats(pos, net)

    summary = {
        "n_obs": float(len(frame)),
        "gross_pnl_sum": float(np.sum(gross)),
        "net_pnl_sum": float(np.sum(net)),
        "avg_net_pnl": avg,
        "net_pnl_vol": std,
        "sharpe_like": float(sharpe),
        "max_drawdown": float(np.min(drawdown)),
        "turnover_sum": float(np.sum(turnover)),
        "cost_sum": float(np.sum(costs)),
        "entry_cost_sum": float(np.sum(frame["entry_cost"])),
        "exit_cost_sum": float(np.sum(frame["exit_cost"])),
        "hit_rate": float(np.mean(net > 0.0)) if len(net) else 0.0,
        "trading_mode": "eod_momentum",
    }

    return BacktestAttributionResult(
        summary=summary,
        frame=frame,
        pnl_by_side=pnl_by_side,
        pnl_by_meta=pnl_by_meta,
        trade_stats=trade_stats,
    )
