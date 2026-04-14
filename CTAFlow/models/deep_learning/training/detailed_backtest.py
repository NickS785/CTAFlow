from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .backtest import TradingMode


@dataclass
class DetailedBacktestResult:
    """Structured output for detailed model attribution backtests."""

    summary: Dict[str, float]
    frame: pd.DataFrame
    batch_tracker_frame: pd.DataFrame
    attention_frame: pd.DataFrame
    feature_importance_frame: pd.DataFrame
    branch_importance_frame: pd.DataFrame
    component_importance_frame: pd.DataFrame
    feature_summary: pd.DataFrame
    branch_summary: pd.DataFrame
    component_summary: pd.DataFrame


def _flatten_scalar_tracker(
    value: Any,
    prefix: str = "",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, inner in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_scalar_tracker(inner, prefix=child))
        return out
    if torch.is_tensor(value):
        if value.ndim == 0:
            return {prefix: float(value.detach().cpu().item())}
        return out
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return {prefix: float(value.item())}
        return out
    if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return out


def _iter_tracker_arrays(
    value: Any,
    prefix: str = "",
) -> Iterable[tuple[str, np.ndarray]]:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tracker_arrays(inner, prefix=child)
        return
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        return
    if arr.ndim > 0:
        yield prefix, np.asarray(arr)


def _safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _compute_summary(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {
            "n_obs": 0.0,
            "gross_pnl_sum": 0.0,
            "net_pnl_sum": 0.0,
            "cost_sum": 0.0,
            "mean_net_pnl": 0.0,
            "sharpe_like": 0.0,
            "sortino_like": 0.0,
            "win_rate": 0.0,
            "avg_abs_position": 0.0,
            "avg_turnover": 0.0,
            "max_drawdown": 0.0,
        }

    net = frame["net_pnl"].to_numpy(dtype=float)
    gross = frame["gross_pnl"].to_numpy(dtype=float)
    costs = frame["costs"].to_numpy(dtype=float)
    turnover = frame["turnover"].to_numpy(dtype=float)
    positions = frame["position"].to_numpy(dtype=float)

    mean_net = float(net.mean()) if len(net) else 0.0
    vol = _safe_std(net)
    neg = net[net < 0.0]
    downside_vol = _safe_std(neg) if len(neg) > 1 else 0.0
    sharpe = 0.0 if vol <= 1e-12 else mean_net / vol * math.sqrt(252.0 * 26.0)
    sortino = 0.0 if downside_vol <= 1e-12 else mean_net / downside_vol * math.sqrt(252.0 * 26.0)

    equity = np.cumsum(net)
    running_max = np.maximum.accumulate(equity) if len(equity) else np.zeros((0,), dtype=float)
    max_drawdown = float((running_max - equity).max()) if len(equity) else 0.0

    return {
        "n_obs": float(len(frame)),
        "gross_pnl_sum": float(gross.sum()),
        "net_pnl_sum": float(net.sum()),
        "cost_sum": float(costs.sum()),
        "mean_net_pnl": mean_net,
        "sharpe_like": float(sharpe),
        "sortino_like": float(sortino),
        "win_rate": float((net > 0.0).mean()) if len(net) else 0.0,
        "avg_abs_position": float(np.abs(positions).mean()) if len(positions) else 0.0,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "max_drawdown": max_drawdown,
    }


def _parse_model_output(output: Any) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    tracker: Dict[str, Any] = {}
    if isinstance(output, Mapping):
        position = output.get("position")
        logits = output.get("logits")
        tracker = dict(output.get("tracker", {}))
        return position, logits, tracker

    if isinstance(output, (tuple, list)):
        items = list(output)
        if items and isinstance(items[-1], Mapping):
            tracker = dict(items.pop(-1))
        tensors = [item for item in items if torch.is_tensor(item)]
        position = tensors[0] if tensors else None
        logits = tensors[1] if len(tensors) > 1 else None
        return position, logits, tracker

    if torch.is_tensor(output):
        return output, None, tracker
    return None, None, tracker


def _collect_model_tracker(
    model: torch.nn.Module,
    explicit_tracker: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    tracker: Dict[str, Any] = {}
    if explicit_tracker:
        tracker.update(dict(explicit_tracker))

    get_last = getattr(model, "get_last_tracker", None)
    if callable(get_last):
        try:
            model_tracker = get_last()
            if isinstance(model_tracker, Mapping):
                tracker.setdefault("model", dict(model_tracker))
        except Exception:
            pass

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        base_get_last = getattr(base_model, "get_last_tracker", None)
        if callable(base_get_last):
            try:
                base_tracker = base_get_last()
                if isinstance(base_tracker, Mapping):
                    tracker.setdefault("base_model", dict(base_tracker))
            except Exception:
                pass
        elif hasattr(base_model, "last_tracker"):
            base_tracker = getattr(base_model, "last_tracker")
            if isinstance(base_tracker, Mapping):
                tracker.setdefault("base_model", dict(base_tracker))

    return tracker


def _as_iso_list(values: Any, batch_size: int) -> list[Optional[str]]:
    if values is None:
        return [None] * batch_size
    if isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        seq = [values] * batch_size
    if len(seq) != batch_size:
        seq = (seq + [None] * batch_size)[:batch_size]
    out: list[Optional[str]] = []
    for value in seq:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out.append(None)
            continue
        try:
            ts = pd.Timestamp(value)
            out.append(ts.isoformat())
        except Exception:
            out.append(str(value))
    return out


def _melt_matrix(
    values: np.ndarray,
    anchor_ts: Sequence[Optional[str]],
    source: str,
    label_prefix: str,
    value_col: str = "value",
) -> pd.DataFrame:
    if values.ndim != 2 or values.shape[0] != len(anchor_ts):
        return pd.DataFrame()
    frame = pd.DataFrame(values, columns=[f"{label_prefix}_{idx}" for idx in range(values.shape[1])])
    frame.insert(0, "anchor_ts", anchor_ts)
    melted = frame.melt(id_vars="anchor_ts", var_name="component", value_name=value_col)
    melted.insert(1, "source", source)
    return melted


def _summarize_long_frame(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str = "importance",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "mean_importance", "std_importance", "count"])
    summary = (
        frame.groupby(list(group_cols), dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_importance", "std": "std_importance"})
        .sort_values("mean_importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    return summary


def run_detailed_backtest(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    batch_to_device_fn: Callable[[Mapping[str, Any], torch.device], tuple[Dict[str, torch.Tensor], torch.Tensor]],
    feature_names: Optional[Sequence[str]] = None,
    gradient_keys: Optional[Sequence[str]] = ("features",),
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    mode: TradingMode = TradingMode.CONTINUOUS,
) -> DetailedBacktestResult:
    """Run a model backtest and collect time-varying attribution signals.

    The collector is intentionally conservative: it only reports signals that are
    directly measurable from model outputs, tracker payloads, or input gradients.
    No synthetic SHAP-like approximation is introduced here.
    """

    model.eval()
    grad_key_set = set(gradient_keys or [])
    cost_per_leg = (float(transaction_cost_bps) + float(slippage_bps)) / 10000.0

    frame_rows: list[Dict[str, Any]] = []
    batch_tracker_rows: list[Dict[str, Any]] = []
    attention_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    branch_frames: list[pd.DataFrame] = []
    component_frames: list[pd.DataFrame] = []

    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch_to_device_fn(batch, device)
        prepared_inputs: Dict[str, torch.Tensor] = {}
        grad_inputs: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and torch.is_floating_point(value) and key in grad_key_set:
                grad_value = value.detach().clone().requires_grad_(True)
                prepared_inputs[key] = grad_value
                grad_inputs[key] = grad_value
            else:
                prepared_inputs[key] = value

        output = model(return_tracker=True, **prepared_inputs)
        position_t, logits_t, explicit_tracker = _parse_model_output(output)
        if position_t is None:
            raise ValueError("Unable to parse model output into position tensor.")

        tracker = _collect_model_tracker(model, explicit_tracker)
        batch_size = int(position_t.shape[0])
        anchor_ts = _as_iso_list(batch.get("_anchor_ts"), batch_size)
        target_end_ts = _as_iso_list(batch.get("_target_end_ts"), batch_size)

        pos_np = position_t.detach().cpu().reshape(-1).numpy().astype(np.float32)
        tgt_np = targets.detach().cpu().reshape(-1).numpy().astype(np.float32)
        logits_np = None
        pred_class = np.full(batch_size, -1, dtype=np.int64)
        if logits_t is not None and torch.is_tensor(logits_t):
            logits_np = logits_t.detach().cpu().numpy()
            if logits_np.ndim >= 2:
                pred_class = logits_np.argmax(axis=-1).astype(np.int64)

        grad_arrays: Dict[str, np.ndarray] = {}
        if grad_inputs:
            grad_targets = torch.autograd.grad(
                position_t.reshape(-1).sum(),
                list(grad_inputs.values()),
                retain_graph=False,
                allow_unused=True,
            )
            for name, grad_value, grad_tensor in zip(grad_inputs.keys(), grad_inputs.values(), grad_targets):
                if grad_tensor is None:
                    continue
                grad_arrays[name] = (
                    (grad_tensor.detach().cpu().numpy() * grad_value.detach().cpu().numpy()).astype(np.float32)
                )

        for sample_idx in range(batch_size):
            row = {
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "anchor_ts": anchor_ts[sample_idx],
                "target_end_ts": target_end_ts[sample_idx],
                "position": float(pos_np[sample_idx]),
                "forward_return": float(tgt_np[sample_idx]),
                "pred_class": int(pred_class[sample_idx]),
            }
            if logits_np is not None and logits_np.ndim >= 2 and logits_np.shape[1] > 0:
                probs = torch.softmax(torch.tensor(logits_np[sample_idx], dtype=torch.float32), dim=-1).numpy()
                row["pred_confidence"] = float(np.max(probs))
                for class_idx in range(logits_np.shape[1]):
                    row[f"logit_{class_idx}"] = float(logits_np[sample_idx, class_idx])
                    row[f"prob_{class_idx}"] = float(probs[class_idx])
            frame_rows.append(row)

        batch_row = {
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "batch_start_ts": anchor_ts[0] if anchor_ts else None,
            "batch_end_ts": anchor_ts[-1] if anchor_ts else None,
        }
        batch_row.update(_flatten_scalar_tracker(tracker))
        batch_tracker_rows.append(batch_row)

        for tracker_key, arr in _iter_tracker_arrays(tracker):
            if arr.ndim == 2 and arr.shape[0] == batch_size:
                if "attn" in tracker_key or "pool" in tracker_key:
                    attention_frames.append(
                        _melt_matrix(arr.astype(np.float32), anchor_ts, tracker_key, label_prefix="lag", value_col="weight")
                    )
                elif "weight" in tracker_key:
                    component_frames.append(
                        _melt_matrix(arr.astype(np.float32), anchor_ts, tracker_key, label_prefix="component", value_col="importance")
                    )
            elif arr.ndim == 3 and arr.shape[0] == batch_size:
                mean_over_time = arr.mean(axis=1).astype(np.float32)
                mean_over_components = arr.mean(axis=2).astype(np.float32)
                component_frames.append(
                    _melt_matrix(mean_over_time, anchor_ts, f"{tracker_key}.mean_over_time", label_prefix="component", value_col="importance")
                )
                attention_frames.append(
                    _melt_matrix(mean_over_components, anchor_ts, f"{tracker_key}.mean_over_components", label_prefix="lag", value_col="weight")
                )

        for grad_key, grad_arr in grad_arrays.items():
            grad_abs = np.abs(grad_arr)
            branch_values = grad_abs.reshape(batch_size, -1).mean(axis=1)
            branch_frames.append(
                pd.DataFrame(
                    {
                        "anchor_ts": anchor_ts,
                        "source": "gradient_branch",
                        "branch": grad_key,
                        "importance": branch_values.astype(np.float32),
                    }
                )
            )

            if grad_abs.ndim == 3:
                temporal_values = grad_abs.mean(axis=2).astype(np.float32)
                attention_frames.append(
                    _melt_matrix(
                        temporal_values,
                        anchor_ts,
                        f"gradient_temporal.{grad_key}",
                        label_prefix="lag",
                        value_col="weight",
                    )
                )
                if grad_key == "features":
                    feature_values = grad_abs.mean(axis=1).astype(np.float32)
                    feature_labels = list(feature_names) if feature_names is not None else [
                        f"feature_{idx}" for idx in range(feature_values.shape[1])
                    ]
                    feat_frame = pd.DataFrame(feature_values, columns=feature_labels)
                    feat_frame.insert(0, "anchor_ts", anchor_ts)
                    feature_frames.append(
                        feat_frame.melt(id_vars="anchor_ts", var_name="feature", value_name="importance")
                    )
            elif grad_abs.ndim >= 2:
                component_values = grad_abs.reshape(batch_size, -1).astype(np.float32)
                component_frames.append(
                    _melt_matrix(
                        component_values,
                        anchor_ts,
                        f"gradient_component.{grad_key}",
                        label_prefix="component",
                        value_col="importance",
                    )
                )

    frame = pd.DataFrame(frame_rows)
    if not frame.empty and "anchor_ts" in frame.columns:
        frame["anchor_ts"] = pd.to_datetime(frame["anchor_ts"])
        frame["target_end_ts"] = pd.to_datetime(frame["target_end_ts"])
        frame = frame.sort_values("anchor_ts", kind="stable").reset_index(drop=True)

    if frame.empty:
        gross = np.zeros((0,), dtype=np.float32)
        turnover = np.zeros((0,), dtype=np.float32)
        costs = np.zeros((0,), dtype=np.float32)
    else:
        positions = frame["position"].to_numpy(dtype=np.float32)
        returns = frame["forward_return"].to_numpy(dtype=np.float32)
        gross = positions * returns
        if mode == TradingMode.ROUND_TRIP:
            turnover = np.abs(positions) * 2.0
        else:
            turnover = np.abs(np.diff(np.r_[0.0, positions]))
        costs = turnover * cost_per_leg

    if not frame.empty:
        frame["gross_pnl"] = gross
        frame["turnover"] = turnover
        frame["costs"] = costs
        frame["net_pnl"] = frame["gross_pnl"] - frame["costs"]
        frame["cum_net_pnl"] = frame["net_pnl"].cumsum()

    batch_tracker_frame = pd.DataFrame(batch_tracker_rows)
    attention_frame = pd.concat(attention_frames, ignore_index=True) if attention_frames else pd.DataFrame()
    feature_importance_frame = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(
        columns=["anchor_ts", "feature", "importance"]
    )
    branch_importance_frame = pd.concat(branch_frames, ignore_index=True) if branch_frames else pd.DataFrame(
        columns=["anchor_ts", "source", "branch", "importance"]
    )
    component_importance_frame = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()

    feature_summary = _summarize_long_frame(feature_importance_frame, group_cols=("feature",))
    branch_summary = _summarize_long_frame(branch_importance_frame, group_cols=("source", "branch"))
    component_summary = _summarize_long_frame(component_importance_frame, group_cols=("source", "component"))
    summary = _compute_summary(frame)

    return DetailedBacktestResult(
        summary=summary,
        frame=frame,
        batch_tracker_frame=batch_tracker_frame,
        attention_frame=attention_frame,
        feature_importance_frame=feature_importance_frame,
        branch_importance_frame=branch_importance_frame,
        component_importance_frame=component_importance_frame,
        feature_summary=feature_summary,
        branch_summary=branch_summary,
        component_summary=component_summary,
    )


def save_detailed_backtest_artifacts(
    result: DetailedBacktestResult,
    output_dir: str | Path,
    *,
    prefix: str = "detailed_backtest",
) -> Dict[str, Path]:
    """Persist detailed backtest outputs as CSV/JSON artifacts."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": out_dir / f"{prefix}_summary.json",
        "frame": out_dir / f"{prefix}_frame.csv",
        "batch_trackers": out_dir / f"{prefix}_batch_trackers.csv",
        "attention": out_dir / f"{prefix}_attention.csv",
        "feature_importance": out_dir / f"{prefix}_feature_importance.csv",
        "branch_importance": out_dir / f"{prefix}_branch_importance.csv",
        "component_importance": out_dir / f"{prefix}_component_importance.csv",
        "feature_summary": out_dir / f"{prefix}_feature_summary.csv",
        "branch_summary": out_dir / f"{prefix}_branch_summary.csv",
        "component_summary": out_dir / f"{prefix}_component_summary.csv",
    }

    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2, default=float)
    result.frame.to_csv(paths["frame"], index=False)
    result.batch_tracker_frame.to_csv(paths["batch_trackers"], index=False)
    result.attention_frame.to_csv(paths["attention"], index=False)
    result.feature_importance_frame.to_csv(paths["feature_importance"], index=False)
    result.branch_importance_frame.to_csv(paths["branch_importance"], index=False)
    result.component_importance_frame.to_csv(paths["component_importance"], index=False)
    result.feature_summary.to_csv(paths["feature_summary"], index=False)
    result.branch_summary.to_csv(paths["branch_summary"], index=False)
    result.component_summary.to_csv(paths["component_summary"], index=False)
    return paths
