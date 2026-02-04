from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class BacktestAttributionResult:
    summary: Dict[str, float]
    frame: pd.DataFrame
    pnl_by_side: pd.DataFrame
    pnl_by_meta: Dict[str, pd.DataFrame]


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
    meta: Optional[Mapping[str, Any]] = None,
) -> BacktestAttributionResult:
    """
    Backtest predictions and return PnL attribution tables.
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

    turn = np.abs(np.diff(np.r_[0.0, pos]))
    total_cost = (transaction_cost_bps + slippage_bps) / 10000.0
    costs = turn * total_cost
    gross = pos * rets
    net = gross - costs
    cum = np.cumsum(net)

    idx = pd.RangeIndex(len(net)) if dates is None else pd.to_datetime(dates)
    frame = pd.DataFrame(
        {
            "position": pos,
            "returns": rets,
            "turnover": turn,
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

    summary = {
        "n_obs": float(len(frame)),
        "gross_pnl_sum": float(np.sum(gross)),
        "net_pnl_sum": float(np.sum(net)),
        "avg_net_pnl": avg,
        "net_pnl_vol": std,
        "sharpe_like": float(sharpe),
        "max_drawdown": float(np.min(drawdown)),
        "turnover_sum": float(np.sum(turn)),
        "cost_sum": float(np.sum(costs)),
        "hit_rate": float(np.mean(net > 0.0)) if len(net) else 0.0,
    }

    return BacktestAttributionResult(
        summary=summary,
        frame=frame,
        pnl_by_side=pnl_by_side,
        pnl_by_meta=pnl_by_meta,
    )
