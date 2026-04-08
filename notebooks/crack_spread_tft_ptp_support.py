from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from CTAFlow.data.datasets.crack_spread_continuous import (
    build_crack_spread_samples,
    crack_spread_collate_fn,
)
from CTAFlow.models.deep_learning.multi_branch.tft.c_mmtft import (
    PTPLoss,
    PredictionToPosition,
    returns_to_classes,
)
from CTAFlow.models.deep_learning.multi_branch.tft.crack_spread_tft import (
    CrackSpreadTFT,
)
from CTAFlow.models.deep_learning.training.loss.clf import SharpeScheduler
from CTAFlow.models.prep.crack_spread_continuous import (
    DEFAULT_CRACK_TICKERS,
    CrackSpreadContinuousPrep,
)


def infer_spread_feature_cols(
    df_out: pd.DataFrame,
    target_col: str,
    known_temporal_cols: Sequence[str],
) -> List[str]:
    exclude = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "BidVolume",
        "AskVolume",
        target_col,
    }
    known_set = set(known_temporal_cols)
    cols: List[str] = []
    for col in df_out.columns:
        if col in exclude or col in known_set or col.startswith("y_fwd_"):
            continue
        if pd.api.types.is_bool_dtype(df_out[col]) or pd.api.types.is_numeric_dtype(df_out[col]):
            cols.append(col)
    if not cols:
        raise ValueError("Unable to infer spread feature columns from df_out.")
    return cols


def hybrid_selection_score(
    metrics: Dict[str, float],
    n_samples: int,
    sharpe_weight: float = 0.20,
    sortino_weight: float = 0.35,
    pf_weight: float = 0.25,
    return_weight: float = 0.20,
    total_return_scale: float = 100.0,
) -> float:
    sharpe = float(metrics.get("sharpe", 0.0))
    sortino = float(metrics.get("sortino", 0.0))
    profit_factor = float(metrics.get("profit_factor", 1e-8))
    mean_strategy_ret = float(metrics.get("mean_strategy_ret", 0.0))

    total_return = mean_strategy_ret * float(max(n_samples, 1))
    total_return_score = math.copysign(
        math.log1p(abs(total_return) * total_return_scale),
        total_return,
    )
    log_pf = math.log(max(profit_factor, 1e-8))
    return (
        sharpe_weight * sharpe
        + sortino_weight * sortino
        + pf_weight * log_pf
        + return_weight * total_return_score
    )


def attach_known_future_sequences(
    samples: Sequence[Dict],
    df_out: pd.DataFrame,
    known_temporal_cols: Sequence[str],
    encoder_steps: int,
    decoder_steps: int,
) -> List[Dict]:
    known_df = df_out.loc[:, list(known_temporal_cols)].copy()
    known_df = known_df.ffill().bfill().fillna(0.0)
    ts_to_pos = {pd.Timestamp(ts): idx for idx, ts in enumerate(known_df.index)}

    enriched: List[Dict] = []
    for sample in samples:
        anchor_ts = pd.Timestamp(sample["anchor_ts"])
        anchor_pos = ts_to_pos.get(anchor_ts)
        if anchor_pos is None:
            continue
        start = anchor_pos - encoder_steps
        stop = anchor_pos + decoder_steps
        if start < 0 or stop > len(known_df):
            continue
        sample_out = dict(sample)
        sample_out["known_future"] = known_df.iloc[start:stop].values.astype(np.float32)
        enriched.append(sample_out)
    return enriched


def build_windowed_crack_samples(
    df_out: pd.DataFrame,
    orderflow_frames: Dict[str, pd.DataFrame],
    target_col: str,
    known_temporal_cols: Sequence[str],
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    encoder_steps: int = 48,
    decoder_steps: int = 4,
    session_only: bool = True,
    sample_session: Optional[str] = None,
    stride: int = 1,
    require_all_assets: bool = True,
) -> List[Dict]:
    base_samples = build_crack_spread_samples(
        df_out=df_out,
        orderflow_frames=orderflow_frames,
        target_col=target_col,
        tickers=tickers,
        spread_lookback=encoder_steps,
        orderflow_lookback=encoder_steps,
        session_only=session_only,
        sample_session=sample_session,
        stride=stride,
        require_all_assets=require_all_assets,
        include_known_temporal_features=True,
    )
    return attach_known_future_sequences(
        base_samples,
        df_out=df_out,
        known_temporal_cols=known_temporal_cols,
        encoder_steps=encoder_steps,
        decoder_steps=decoder_steps,
    )


def split_crack_samples(
    samples: Sequence[Dict],
    val_start: str | pd.Timestamp,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    cutoff = pd.Timestamp(val_start)
    train_samples = [dict(s) for s in samples if pd.Timestamp(s["anchor_ts"]) < cutoff]
    val_samples = [dict(s) for s in samples if pd.Timestamp(s["anchor_ts"]) >= cutoff]

    if max_train_samples is not None and len(train_samples) > max_train_samples:
        train_samples = train_samples[-int(max_train_samples):]
    if max_val_samples is not None and len(val_samples) > max_val_samples:
        val_samples = val_samples[:int(max_val_samples)]
    return train_samples, val_samples


class CrackSpreadPTPDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict],
        tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
        return_metadata: bool = False,
    ):
        self.samples = list(samples)
        self.tickers = tuple(t.upper() for t in tickers)
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        out: Dict[str, object] = {
            "spread_time_features": torch.tensor(sample["spread_time_features"], dtype=torch.float32),
            "spread_time_lens": torch.tensor(sample["spread_time_len"], dtype=torch.long),
            "known_future": torch.tensor(sample["known_future"], dtype=torch.float32),
            "target": torch.tensor(sample["target"], dtype=torch.float32),
        }
        for ticker in self.tickers:
            out[f"orderflow_{ticker}"] = torch.tensor(sample[f"orderflow_{ticker}"], dtype=torch.float32)
            out[f"orderflow_{ticker}_len"] = torch.tensor(sample[f"orderflow_{ticker}_len"], dtype=torch.long)
        if self.return_metadata:
            out["_anchor_ts"] = sample["anchor_ts"]
            out["_date"] = sample["date"]
            out["_target_end_ts"] = sample["target_end_ts"]
        return out


def crack_ptp_collate_fn(
    batch: List[Dict[str, object]],
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    orderflow_lookback: int = 48,
) -> Dict[str, object]:
    out = crack_spread_collate_fn(
        batch,
        tickers=tickers,
        orderflow_lookback=orderflow_lookback,
    )
    out["known_future"] = torch.stack([sample["known_future"] for sample in batch])
    return out


def make_crack_loaders(
    train_samples: Sequence[Dict],
    val_samples: Sequence[Dict],
    batch_size: int = 64,
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    orderflow_lookback: int = 48,
    num_workers: int = 0,
    return_metadata: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    tickers = tuple(t.upper() for t in tickers)
    collate = partial(
        crack_ptp_collate_fn,
        tickers=tickers,
        orderflow_lookback=orderflow_lookback,
    )
    train_ds = CrackSpreadPTPDataset(train_samples, tickers=tickers, return_metadata=return_metadata)
    val_ds = CrackSpreadPTPDataset(val_samples, tickers=tickers, return_metadata=return_metadata)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    return train_loader, val_loader


def batch_to_device(
    batch: Dict[str, object],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
        "spread_time_features": batch["spread_time_features"].to(device),
        "known_future": batch["known_future"].to(device),
        "orderflow_cube": batch["orderflow_cube"].to(device),
        "orderflow_mask": batch["orderflow_mask"].to(device),
    }
    targets = batch["target"].to(device)
    return inputs, targets


class CrackSpreadPTP(nn.Module):
    def __init__(
        self,
        base_model: CrackSpreadTFT,
        ptp_temperature: float = 1.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.refiner = nn.Sequential(
            nn.Linear(self.base_model.num_classes, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, self.base_model.num_classes),
        )
        self.ptp = PredictionToPosition(temperature=ptp_temperature)

    def forward(
        self,
        spread_time_features: torch.Tensor,
        known_future: torch.Tensor,
        orderflow_cube: torch.Tensor,
        orderflow_mask: Optional[torch.Tensor] = None,
        return_tracker: bool = False,
    ):
        out = self.base_model(
            past_observed=spread_time_features,
            known_future=known_future,
            orderflow=orderflow_cube,
            orderflow_mask=orderflow_mask,
            return_dict=True,
        )
        logits_seq = out["logits"]
        final_logits = logits_seq[:, -1, :]
        refined_logits = final_logits + self.refiner(final_logits)
        position, logits = self.ptp(refined_logits)
        if return_tracker:
            tracker = dict(out.get("tracker", {}))
            tracker.update({f"ptp_{k}": v for k, v in self.ptp.get_last_stats().items()})
            return position, logits, logits_seq, tracker
        return position, logits, logits_seq

    def get_aux_loss(self) -> torch.Tensor:
        return self.base_model.get_aux_loss()


def build_crack_ptp_param_groups(
    model: CrackSpreadPTP,
    base_lr: float,
    weight_decay: float = 0.0,
    ptp_lr_scale: float = 1.0,
) -> List[Dict[str, object]]:
    ptp_params = list(model.ptp.parameters()) + list(model.refiner.parameters())
    ptp_ids = {id(p) for p in ptp_params}
    trunk_params = [p for p in model.parameters() if p.requires_grad and id(p) not in ptp_ids]

    groups: List[Dict[str, object]] = []
    if trunk_params:
        groups.append({
            "params": trunk_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "group_name": "trunk",
        })
    if ptp_params:
        groups.append({
            "params": ptp_params,
            "lr": base_lr * ptp_lr_scale,
            "weight_decay": weight_decay,
            "group_name": "ptp_head",
        })
    return groups


def _safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def compute_trading_metrics(
    positions: np.ndarray,
    forward_returns: np.ndarray,
    logits: Optional[np.ndarray] = None,
    outer_threshold: float = 1.0,
    bars_per_year: int = 252 * 26,
) -> Dict[str, float]:
    pos = np.asarray(positions, dtype=float).reshape(-1)
    fwd = np.asarray(forward_returns, dtype=float).reshape(-1)
    strategy = pos * fwd

    mean_ret = float(strategy.mean()) if strategy.size else 0.0
    vol = _safe_std(strategy)
    neg = strategy[strategy < 0.0]
    downside_vol = _safe_std(neg) if neg.size > 1 else 0.0

    sharpe = 0.0 if vol <= 1e-12 else mean_ret / vol * math.sqrt(bars_per_year)
    sortino = 0.0 if downside_vol <= 1e-12 else mean_ret / downside_vol * math.sqrt(bars_per_year)

    gross_pos = float(strategy[strategy > 0.0].sum())
    gross_neg = float(-strategy[strategy < 0.0].sum())
    profit_factor = gross_pos / gross_neg if gross_neg > 1e-12 else (gross_pos if gross_pos > 0 else 0.0)
    win_rate = float((strategy > 0.0).mean()) if strategy.size else 0.0
    dir_accuracy = float((np.sign(pos) == np.sign(fwd)).mean()) if strategy.size else 0.0
    avg_exposure = float(np.abs(pos).mean()) if pos.size else 0.0

    equity = np.cumsum(strategy)
    if equity.size:
        running_max = np.maximum.accumulate(equity)
        max_drawdown = float((running_max - equity).max())
    else:
        max_drawdown = 0.0

    metrics = {
        "mean_strategy_ret": mean_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "dir_accuracy": dir_accuracy,
        "avg_exposure": avg_exposure,
        "max_drawdown": max_drawdown,
        "downside_vol": downside_vol,
    }

    if logits is not None and logits.size:
        logits_t = torch.tensor(logits, dtype=torch.float32)
        targets_t = returns_to_classes(torch.tensor(fwd, dtype=torch.float32), outer=outer_threshold)
        pred = logits_t.argmax(dim=-1)
        metrics["cls_accuracy"] = float((pred == targets_t).float().mean().item() * 100.0)
        pred_dir = (pred >= 2).long() - (pred < 2).long()
        true_dir = (targets_t >= 2).long() - (targets_t < 2).long()
        metrics["cls_dir_accuracy"] = float((pred_dir == true_dir).float().mean().item() * 100.0)
    else:
        metrics["cls_accuracy"] = 0.0
        metrics["cls_dir_accuracy"] = 0.0
    return metrics


def _merge_mean_metrics(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({k for row in metric_rows for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in metric_rows if key in row]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def train_epoch_crack_ptp(
    model: CrackSpreadPTP,
    loader: DataLoader,
    loss_fn: PTPLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
    aux_weight: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_count = 0
    batch_metrics: List[Dict[str, float]] = []

    for batch in loader:
        inputs, targets = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                position, logits, _, tracker = model(return_tracker=True, **inputs)
                loss, metrics = loss_fn(position, logits, targets)
                aux = model.get_aux_loss()
                total = loss + aux_weight * aux
            scaler.scale(total).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            position, logits, _, tracker = model(return_tracker=True, **inputs)
            loss, metrics = loss_fn(position, logits, targets)
            aux = model.get_aux_loss()
            total = loss + aux_weight * aux
            total.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        count = int(targets.numel())
        total_loss += float(total.detach().item()) * count
        total_count += count
        row = dict(metrics)
        row["aux_loss"] = float(aux.detach().item())
        row.update({k: float(v) for k, v in tracker.items()})
        batch_metrics.append(row)

    mean_loss = total_loss / max(total_count, 1)
    return mean_loss, _merge_mean_metrics(batch_metrics)


def evaluate_crack_ptp(
    model: CrackSpreadPTP,
    loader: DataLoader,
    loss_fn: PTPLoss,
    device: torch.device,
    aux_weight: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    loss_rows: List[Dict[str, float]] = []
    all_positions: List[np.ndarray] = []
    all_returns: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch_to_device(batch, device)
            position, logits, _, tracker = model(return_tracker=True, **inputs)
            loss, metrics = loss_fn(position, logits, targets)
            aux = model.get_aux_loss()
            total = loss + aux_weight * aux

            count = int(targets.numel())
            total_loss += float(total.detach().item()) * count
            total_count += count

            row = dict(metrics)
            row["aux_loss"] = float(aux.detach().item())
            row.update({k: float(v) for k, v in tracker.items()})
            loss_rows.append(row)

            all_positions.append(position.detach().cpu().numpy())
            all_returns.append(targets.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    metrics = _merge_mean_metrics(loss_rows)
    pos_np = np.concatenate(all_positions, axis=0) if all_positions else np.zeros((0, 1), dtype=float)
    ret_np = np.concatenate(all_returns, axis=0) if all_returns else np.zeros((0,), dtype=float)
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 4), dtype=float)
    trading = compute_trading_metrics(
        positions=pos_np,
        forward_returns=ret_np,
        logits=logits_np,
        outer_threshold=loss_fn.outer_threshold,
    )
    metrics.update(trading)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


@dataclass
class FinalTrainingResult:
    model: CrackSpreadPTP
    best_state: Optional[Dict[str, torch.Tensor]]
    history: Dict[str, List[float]]
    best_score: float
    best_metrics: Dict[str, float]


def fit_final_crack_ptp(
    model: CrackSpreadPTP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: PTPLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    sharpe_scheduler: Optional[SharpeScheduler],
    device: torch.device,
    num_epochs: int = 20,
    warmup_epochs: int = 3,
    max_norm: float = 1.0,
    aux_weight: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> FinalTrainingResult:
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_score": [],
        "val_sharpe": [],
        "val_sortino": [],
        "val_pf": [],
        "val_cls_acc": [],
        "lr": [],
    }

    best_score = -1e9
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Dict[str, float] = {}

    for epoch in range(num_epochs):
        if sharpe_scheduler is not None:
            sharpe_scheduler.step(epoch, loss_fn.trading_loss)

        train_loss, _ = train_epoch_crack_ptp(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            max_norm=max_norm,
            aux_weight=aux_weight,
            scaler=scaler,
        )
        val_metrics = evaluate_crack_ptp(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            aux_weight=aux_weight,
        )
        if scheduler is not None:
            scheduler.step()

        score = hybrid_selection_score(val_metrics, len(val_loader.dataset))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_score"].append(float(score))
        history["val_sharpe"].append(float(val_metrics["sharpe"]))
        history["val_sortino"].append(float(val_metrics["sortino"]))
        history["val_pf"].append(float(val_metrics["profit_factor"]))
        history["val_cls_acc"].append(float(val_metrics.get("cls_accuracy", 0.0)))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch >= warmup_epochs and score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_state = deepcopy(model.state_dict())

    return FinalTrainingResult(
        model=model,
        best_state=best_state,
        history=history,
        best_score=best_score,
        best_metrics=best_metrics,
    )


def prepare_crack_data(
    data_root: str | bytes | "os.PathLike[str]",
    tickers: Sequence[str] = DEFAULT_CRACK_TICKERS,
    bar_minutes: int = 15,
    target_mode: str = "logret",
    steps_60m: int = 4,
    orderflow_columns: Optional[Sequence[str]] = None,
) -> Tuple[CrackSpreadContinuousPrep, pd.DataFrame, pd.Series, List[str], Dict[str, pd.DataFrame], List[str]]:
    prep = CrackSpreadContinuousPrep(
        root_features_dir=data_root,
        tickers=tuple(t.upper() for t in tickers),
        bar_minutes=bar_minutes,
        target_mode=target_mode,
    )
    df_out, train_mask, target_cols = prep.prepare_from_root(
        steps_60m=steps_60m,
        keep_only_active=False,
        add_daily=True,
        add_overnight=True,
        add_deseas=True,
        add_time_features=True,
        add_resample_precalc=True,
        resample_rules=("15min", "30min", "60min"),
        apply_scaling=False,
        add_bid_ask=True,
    )
    orderflow_frames, orderflow_cols = prep.load_orderflow_frames(orderflow_columns=orderflow_columns)
    return prep, df_out, train_mask, target_cols, orderflow_frames, orderflow_cols
