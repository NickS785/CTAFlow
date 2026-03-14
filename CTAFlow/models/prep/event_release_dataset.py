from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_EVENT_RELEASE_QUANTILES: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
DEFAULT_EVENT_RELEASE_TECHNICAL_COLS: Tuple[str, ...] = (
    "tech_dist_sma_200d",
    "tech_macd_12_26",
    "tech_macd_hist_12_26",
    "tech_rsi_14d",
    "tech_ret_30m_deseas",
    "tech_ret_1h_deseas",
    "tech_dist_ema_12",
    "tech_dist_ema_24",
    "tech_ema_12_24_spread",
)


def _validate_quantiles(quantiles: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(q) for q in quantiles)
    if not values:
        raise ValueError("quantiles must not be empty")
    if any(q <= 0.0 or q >= 1.0 for q in values):
        raise ValueError(f"quantiles must be strictly between 0 and 1, got {values}")
    if tuple(sorted(values)) != values:
        raise ValueError(f"quantiles must be sorted ascending, got {values}")
    return values


def _prepare_time_index(
    df: pd.DataFrame,
    *,
    timestamp_col: Optional[str] = None,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    dt_series: Optional[pd.Series] = None
    if timestamp_col is not None:
        if timestamp_col not in out.columns:
            raise KeyError(f"timestamp_col '{timestamp_col}' not found in columns: {list(out.columns)}")
        dt_series = pd.to_datetime(out[timestamp_col], errors="coerce")
        out = out.drop(columns=[timestamp_col])
    elif {"Date", "Time"}.issubset(out.columns):
        dt_series = pd.to_datetime(
            out["Date"].astype(str).str.strip() + " " + out["Time"].astype(str).str.strip(),
            errors="coerce",
        )
        out = out.drop(columns=["Date", "Time"])
    else:
        candidates = [
            "Datetime",
            "datetime",
            "Timestamp",
            "timestamp",
            "DateTime",
            "date",
            "Date",
            "time",
            "Time",
        ]
        for candidate in candidates:
            if candidate in out.columns:
                parsed = pd.to_datetime(out[candidate], errors="coerce")
                if parsed.notna().any():
                    dt_series = parsed
                    out = out.drop(columns=[candidate])
                    break

    if dt_series is None:
        if isinstance(out.index, pd.DatetimeIndex):
            dt_index = out.index
        else:
            parsed_index = pd.to_datetime(out.index, errors="coerce")
            if parsed_index.notna().any():
                dt_index = pd.DatetimeIndex(parsed_index)
            else:
                raise ValueError("Could not infer datetime index from the provided frame")
    else:
        if dt_series.isna().all():
            raise ValueError("Failed to parse any timestamps from the provided frame")
        dt_index = pd.DatetimeIndex(dt_series)

    out.index = dt_index
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]

    if timezone is not None:
        if out.index.tz is None:
            out.index = out.index.tz_localize(timezone)
        else:
            out.index = out.index.tz_convert(timezone)

    return out


def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower().replace(" ", "")
        if key == "open":
            rename_map[column] = "Open"
        elif key == "high":
            rename_map[column] = "High"
        elif key == "low":
            rename_map[column] = "Low"
        elif key in {"close", "last", "settle"}:
            rename_map[column] = "Close"
        elif key in {"volume", "vol", "totalvolume"}:
            rename_map[column] = "Volume"

    out = df.rename(columns=rename_map).copy()
    required = {"Open", "High", "Low", "Close"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise KeyError(f"Missing OHLC columns after standardization: {missing}")
    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def read_ohlcv_csv(
    path: Union[str, Path],
    *,
    timestamp_col: Optional[str] = None,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """Read a user-supplied 5-minute OHLCV CSV into a standardized frame."""
    raw = pd.read_csv(path)
    indexed = _prepare_time_index(raw, timestamp_col=timestamp_col, timezone=timezone)
    return _standardize_ohlcv_columns(indexed)


def _build_price_context(ohlcv: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    close = ohlcv["Close"].astype(float).clip(lower=eps)
    open_ = ohlcv["Open"].astype(float).clip(lower=eps)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    volume = ohlcv["Volume"].astype(float).clip(lower=0.0)

    log_ret = np.log(close).diff()
    log_volume = np.log1p(volume)
    volume_mean = log_volume.rolling(64, min_periods=8).mean()
    volume_std = log_volume.rolling(64, min_periods=8).std().replace(0.0, np.nan)

    minutes = ohlcv.index.hour * 60 + ohlcv.index.minute
    tod_frac = minutes / (24.0 * 60.0)

    return pd.DataFrame(
        {
            "ctx_log_ret": log_ret,
            "ctx_bar_range": (high - low) / close,
            "ctx_bar_body": (close - open_) / open_,
            "ctx_log_volume": log_volume,
            "ctx_volume_z": (log_volume - volume_mean) / volume_std,
            "ctx_tod_sin": np.sin(2.0 * np.pi * tod_frac),
            "ctx_tod_cos": np.cos(2.0 * np.pi * tod_frac),
        },
        index=ohlcv.index,
    )


def _expanding_slot_zscore(
    series: pd.Series,
    slots: pd.Series,
    min_history: int = 20,
) -> pd.Series:
    grouped = series.groupby(slots)
    mean = (
        grouped.expanding(min_periods=min_history)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    std = (
        grouped.expanding(min_periods=min_history)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
        .replace(0.0, np.nan)
    )
    return (series - mean) / std


def _build_technical_branch_features(
    ohlcv: pd.DataFrame,
    eps: float = 1e-8,
) -> pd.DataFrame:
    close = ohlcv["Close"].astype(float).clip(lower=eps)
    date_index = ohlcv.index.normalize()

    daily_close = close.groupby(date_index).last()
    daily_delta = daily_close.diff()
    gain = daily_delta.where(daily_delta > 0.0, 0.0).rolling(14, min_periods=14).mean()
    loss = (-daily_delta.where(daily_delta < 0.0, 0.0)).rolling(14, min_periods=14).mean().replace(0.0, np.nan)
    rs = gain / loss
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    sma_200 = daily_close.rolling(200, min_periods=50).mean()
    ema_12d = daily_close.ewm(span=12, adjust=False).mean()
    ema_26d = daily_close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12d - ema_26d
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    daily_feats = pd.DataFrame(
        {
            "tech_sma_200d": sma_200,
            "tech_macd_12_26": macd_line,
            "tech_macd_hist_12_26": macd_hist,
            "tech_rsi_14d": (rsi_14 - 50.0) / 10.0,
        },
        index=daily_close.index,
    ).shift(1)

    tech = pd.DataFrame(index=ohlcv.index)
    mapped_sma = date_index.map(daily_feats["tech_sma_200d"])
    tech["tech_dist_sma_200d"] = (close - mapped_sma.values) / np.where(np.abs(mapped_sma.values) < eps, np.nan, mapped_sma.values)
    tech["tech_macd_12_26"] = date_index.map(daily_feats["tech_macd_12_26"]).values
    tech["tech_macd_hist_12_26"] = date_index.map(daily_feats["tech_macd_hist_12_26"]).values
    tech["tech_rsi_14d"] = date_index.map(daily_feats["tech_rsi_14d"]).values

    log_close = np.log(close)
    ret_30m = log_close.diff(6)
    ret_1h = log_close.diff(12)
    slots = pd.Series(ohlcv.index.hour * 60 + ohlcv.index.minute, index=ohlcv.index)
    tech["tech_ret_30m_deseas"] = _expanding_slot_zscore(ret_30m, slots, min_history=20)
    tech["tech_ret_1h_deseas"] = _expanding_slot_zscore(ret_1h, slots, min_history=20)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_24 = close.ewm(span=24, adjust=False).mean()
    tech["tech_dist_ema_12"] = (close - ema_12) / ema_12.replace(0.0, np.nan)
    tech["tech_dist_ema_24"] = (close - ema_24) / ema_24.replace(0.0, np.nan)
    tech["tech_ema_12_24_spread"] = (ema_12 - ema_24) / close

    return tech.replace([np.inf, -np.inf], np.nan)


def _compute_forward_returns(
    close: pd.Series,
    *,
    horizon_bars: int,
    prefix: str = "target_ret",
    eps: float = 1e-8,
) -> pd.DataFrame:
    log_close = np.log(close.astype(float).clip(lower=eps))
    out = pd.DataFrame(index=close.index)
    for step in range(1, horizon_bars + 1):
        out[f"{prefix}_{step}"] = log_close.shift(-step) - log_close
    return out


def _add_quantile_class_targets(
    frame: pd.DataFrame,
    *,
    target_return_cols: Sequence[str],
    quantiles: Sequence[float],
    min_history: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = frame.copy()
    class_cols: List[str] = []
    threshold_cols: List[str] = []
    q_values = _validate_quantiles(quantiles)

    for step, target_col in enumerate(target_return_cols, start=1):
        thresholds: List[pd.Series] = []
        for q in q_values:
            pct_label = int(round(q * 100))
            threshold_col = f"target_q_{step}_{pct_label}"
            out[threshold_col] = out[target_col].expanding(min_periods=min_history).quantile(q).shift(1)
            thresholds.append(out[threshold_col])
            threshold_cols.append(threshold_col)

        valid = out[target_col].notna()
        for threshold in thresholds:
            valid &= threshold.notna()

        label_values = np.zeros(len(out), dtype=np.float32)
        target_values = out[target_col].to_numpy(dtype=np.float64, copy=False)
        for threshold in thresholds:
            threshold_values = threshold.to_numpy(dtype=np.float64, copy=False)
            label_values += (target_values > threshold_values).astype(np.float32)

        class_col = f"target_cls_{step}"
        out[class_col] = np.where(valid, label_values, np.nan)
        class_cols.append(class_col)

    return out, class_cols, threshold_cols


@dataclass
class EventReleaseDatasetConfig:
    lookback_bars: int = 72
    horizon_bars: int = 6
    quantiles: Tuple[float, ...] = DEFAULT_EVENT_RELEASE_QUANTILES
    min_quantile_history: int = 40
    add_price_context: bool = True
    anchor_col: Optional[str] = None
    group_col: Optional[str] = None
    timezone: Optional[str] = None
    timestamp_col: Optional[str] = None
    technical_min_history: int = 20
    enable_single_point_classification: bool = False
    single_point_target_step: Optional[int] = None
    single_point_class_quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75)

    @property
    def num_classes(self) -> int:
        return len(self.quantiles) + 1

    @property
    def single_point_num_classes(self) -> int:
        return len(self.single_point_class_quantiles) + 1


class EventReleaseQuantilePrep:
    """
    Build a multi-horizon event-release dataset from orderflow features + 5-minute OHLCV.

    The supplied ``feature_frame`` is expected to contain orderflow/event features already
    aligned to the same 5-minute timestamp grid as the OHLCV series. Targets are always
    computed from the raw OHLCV path so the model can use the full future close path even
    if the feature frame only contains release windows.
    """

    def __init__(
        self,
        *,
        lookback_bars: int = 72,
        horizon_bars: int = 6,
        quantiles: Sequence[float] = DEFAULT_EVENT_RELEASE_QUANTILES,
        min_quantile_history: int = 40,
        past_observed_cols: Optional[Sequence[str]] = None,
        known_future_cols: Optional[Sequence[str]] = None,
        static_cols: Optional[Sequence[str]] = None,
        technical_cols: Optional[Sequence[str]] = None,
        add_price_context: bool = True,
        anchor_col: Optional[str] = None,
        group_col: Optional[str] = None,
        timezone: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        enable_single_point_classification: bool = False,
        single_point_target_step: Optional[int] = None,
        single_point_class_quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    ):
        single_point_q = _validate_quantiles(single_point_class_quantiles)
        self.config = EventReleaseDatasetConfig(
            lookback_bars=int(lookback_bars),
            horizon_bars=int(horizon_bars),
            quantiles=_validate_quantiles(quantiles),
            min_quantile_history=int(min_quantile_history),
            add_price_context=bool(add_price_context),
            anchor_col=anchor_col,
            group_col=group_col,
            timezone=timezone,
            timestamp_col=timestamp_col,
            enable_single_point_classification=bool(enable_single_point_classification),
            single_point_target_step=single_point_target_step,
            single_point_class_quantiles=single_point_q,
        )
        if self.config.lookback_bars < 1:
            raise ValueError("lookback_bars must be >= 1")
        if self.config.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.config.single_point_target_step is None:
            self.config.single_point_target_step = self.config.horizon_bars
        if not (1 <= int(self.config.single_point_target_step) <= self.config.horizon_bars):
            raise ValueError("single_point_target_step must be between 1 and horizon_bars")

        self.past_observed_cols = list(past_observed_cols or [])
        self.known_future_cols = list(known_future_cols or [])
        self.static_cols = list(static_cols or [])
        self.technical_cols = list(technical_cols or DEFAULT_EVENT_RELEASE_TECHNICAL_COLS)

        self.frame: Optional[pd.DataFrame] = None
        self.ohlcv: Optional[pd.DataFrame] = None
        self.target_return_cols: List[str] = []
        self.target_class_cols: List[str] = []
        self.threshold_cols: List[str] = []
        self.single_point_threshold_cols: List[str] = []
        self.single_point_target_col: Optional[str] = None
        self.single_point_class_col: Optional[str] = None

    def _add_single_point_classification_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        target_step = int(self.config.single_point_target_step or self.config.horizon_bars)
        target_col = f"target_ret_{target_step}"
        out = frame.copy()
        thresholds: List[str] = []
        threshold_series = []
        for q in self.config.single_point_class_quantiles:
            pct_label = int(round(q * 100))
            col = f"single_point_q_{target_step}_{pct_label}"
            out[col] = out[target_col].expanding(min_periods=self.config.min_quantile_history).quantile(q).shift(1)
            thresholds.append(col)
            threshold_series.append(out[col])

        valid = out[target_col].notna()
        for series in threshold_series:
            valid &= series.notna()

        labels = np.zeros(len(out), dtype=np.float32)
        target_values = out[target_col].to_numpy(dtype=np.float64, copy=False)
        for series in threshold_series:
            threshold_values = series.to_numpy(dtype=np.float64, copy=False)
            labels += (target_values > threshold_values).astype(np.float32)

        class_col = f"single_point_cls_{target_step}"
        out[class_col] = np.where(valid, labels, np.nan)
        self.single_point_target_col = target_col
        self.single_point_class_col = class_col
        self.single_point_threshold_cols = thresholds
        return out

    def load_data(
        self,
        *,
        ohlcv_csv_path: Union[str, Path],
        feature_frame: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load OHLCV, align features, and compute multi-step quantile targets."""
        self.ohlcv = read_ohlcv_csv(
            ohlcv_csv_path,
            timestamp_col=self.config.timestamp_col,
            timezone=self.config.timezone,
        )

        targets = _compute_forward_returns(
            self.ohlcv["Close"],
            horizon_bars=self.config.horizon_bars,
        )
        self.target_return_cols = list(targets.columns)

        price_context = _build_price_context(self.ohlcv) if self.config.add_price_context else pd.DataFrame(index=self.ohlcv.index)
        technical_context = _build_technical_branch_features(self.ohlcv)

        if feature_frame is None:
            base = price_context.copy()
        else:
            features = _prepare_time_index(
                feature_frame,
                timezone=self.config.timezone,
            )
            base = features.copy()
            if self.config.add_price_context:
                base = base.join(price_context, how="left")

        frame = base.join(technical_context, how="left")
        frame = frame.join(targets, how="left").sort_index()
        frame, self.target_class_cols, self.threshold_cols = _add_quantile_class_targets(
            frame,
            target_return_cols=self.target_return_cols,
            quantiles=self.config.quantiles,
            min_history=self.config.min_quantile_history,
        )
        if self.config.enable_single_point_classification:
            frame = self._add_single_point_classification_targets(frame)

        if not self.past_observed_cols:
            excluded = set(self.target_return_cols + self.target_class_cols + self.threshold_cols)
            excluded.update(self.single_point_threshold_cols)
            if self.single_point_class_col is not None:
                excluded.add(self.single_point_class_col)
            excluded.update(self.known_future_cols)
            excluded.update(self.static_cols)
            if self.config.anchor_col is not None:
                excluded.add(self.config.anchor_col)
            if self.config.group_col is not None:
                excluded.add(self.config.group_col)

            numeric_cols = [
                col for col in frame.columns
                if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
            ]
            self.past_observed_cols = numeric_cols

        self.frame = frame
        return frame

    def get_dims(self) -> Dict[str, int]:
        return {
            "n_past_observed": len(self.past_observed_cols),
            "n_known_future": len(self.known_future_cols),
            "n_static": len(self.static_cols),
            "n_technical": len(self.technical_cols),
            "horizon_bars": self.config.horizon_bars,
            "n_classes": self.config.num_classes,
            "n_quantiles": len(self.config.quantiles),
            "single_point_num_classes": (
                self.config.single_point_num_classes
                if self.config.enable_single_point_classification
                else 0
            ),
        }

    def build_samples(
        self,
        *,
        lookback_bars: Optional[int] = None,
    ) -> List[dict]:
        if self.frame is None:
            raise ValueError("No prepared frame loaded. Call load_data() first.")

        frame = self.frame.copy()
        lookback = int(lookback_bars or self.config.lookback_bars)
        if lookback < 1:
            raise ValueError("lookback_bars must be >= 1")

        for col in self.past_observed_cols + self.known_future_cols + self.static_cols + self.technical_cols:
            if col not in frame.columns:
                raise KeyError(f"Configured column '{col}' is missing from the prepared frame")

        valid = frame[self.target_class_cols].notna().all(axis=1)
        valid &= frame[self.target_return_cols].notna().all(axis=1)
        if self.config.enable_single_point_classification:
            if self.single_point_class_col is None:
                raise ValueError("Single-point classification enabled but no class column was built")
            valid &= frame[self.single_point_class_col].notna()
        if self.config.anchor_col is not None:
            if self.config.anchor_col not in frame.columns:
                raise KeyError(f"anchor_col '{self.config.anchor_col}' is missing from the prepared frame")
            valid &= frame[self.config.anchor_col].astype(bool)

        past_arr = frame[self.past_observed_cols].fillna(0.0).to_numpy(dtype=np.float32)
        if self.known_future_cols:
            known_arr = frame[self.known_future_cols].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            known_arr = np.zeros((len(frame), 0), dtype=np.float32)
        if self.static_cols:
            static_arr = frame[self.static_cols].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            static_arr = np.zeros((len(frame), 0), dtype=np.float32)
        if self.technical_cols:
            technical_arr = frame[self.technical_cols].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            technical_arr = np.zeros((len(frame), 0), dtype=np.float32)

        target_returns = frame[self.target_return_cols].to_numpy(dtype=np.float32)
        target_classes = frame[self.target_class_cols].to_numpy(dtype=np.int64)

        thresholds_by_step: List[np.ndarray] = []
        for step in range(1, self.config.horizon_bars + 1):
            step_cols = [f"target_q_{step}_{int(round(q * 100))}" for q in self.config.quantiles]
            thresholds_by_step.append(frame[step_cols].to_numpy(dtype=np.float32))
        threshold_arr = np.stack(thresholds_by_step, axis=1) if thresholds_by_step else np.empty((len(frame), 0, 0), dtype=np.float32)

        group_values = frame[self.config.group_col].to_numpy(copy=False) if self.config.group_col else None

        samples: List[dict] = []
        valid_indices = np.flatnonzero(valid.to_numpy())
        for idx in valid_indices:
            if idx < lookback:
                continue
            future_end = idx + self.config.horizon_bars
            if future_end >= len(frame):
                continue

            if group_values is not None:
                window_group = group_values[idx - lookback: future_end + 1]
                if not np.all(window_group == group_values[idx]):
                    continue

            sample = {
                "past_observed": past_arr[idx - lookback: idx + 1],
                "known_future": known_arr[idx - lookback: future_end + 1],
                "static": static_arr[idx],
                "technical_features": technical_arr[idx - lookback: idx + 1],
                "target_returns": target_returns[idx],
                "target_classes": target_classes[idx],
                "quantile_thresholds": threshold_arr[idx],
                "anchor_ts": frame.index[idx],
                "date": frame.index[idx].date() if hasattr(frame.index[idx], "date") else frame.index[idx],
            }
            if self.config.enable_single_point_classification and self.single_point_class_col is not None:
                sample["single_point_target"] = int(frame.iloc[idx][self.single_point_class_col])
            samples.append(sample)

        return samples

    def get_loaders(
        self,
        *,
        val_cutoff_date: Union[str, date],
        batch_size: int = 32,
        lookback_bars: Optional[int] = None,
        num_workers: int = 0,
        return_metadata: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        if isinstance(val_cutoff_date, str):
            val_cutoff_date = pd.Timestamp(val_cutoff_date).date()

        samples = self.build_samples(lookback_bars=lookback_bars)
        train_samples = [sample for sample in samples if sample["date"] < val_cutoff_date]
        val_samples = [sample for sample in samples if sample["date"] >= val_cutoff_date]

        train_ds = EventReleaseQuantileDataset(train_samples, return_metadata=return_metadata)
        val_ds = EventReleaseQuantileDataset(val_samples, return_metadata=return_metadata)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader


class EventReleaseQuantileDataset(Dataset):
    """PyTorch dataset for multi-step event-release quantile classification."""

    def __init__(self, samples: List[dict], return_metadata: bool = False):
        self.samples = samples
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        out = {
            "past_observed": torch.tensor(sample["past_observed"], dtype=torch.float32),
            "known_future": torch.tensor(sample["known_future"], dtype=torch.float32),
            "static": torch.tensor(sample["static"], dtype=torch.float32),
            "technical_features": torch.tensor(sample["technical_features"], dtype=torch.float32),
            "target_returns": torch.tensor(sample["target_returns"], dtype=torch.float32),
            "target_classes": torch.tensor(sample["target_classes"], dtype=torch.long),
            "quantile_thresholds": torch.tensor(sample["quantile_thresholds"], dtype=torch.float32),
        }
        if "single_point_target" in sample:
            out["single_point_target"] = torch.tensor(sample["single_point_target"], dtype=torch.long)
        if self.return_metadata:
            out["anchor_ts"] = sample["anchor_ts"]
            out["date"] = sample["date"]
        return out


# =====================================================================
# Per-Bar Samples for ClassificationEventTFT
# =====================================================================


def build_session_samples(
    prep: "EventReleaseQuantilePrep",
    *,
    event_dates: Sequence[date],
    post_end_times: Dict[date, pd.Timestamp],
    phase_orderflow: Dict[date, np.ndarray],
    session_close_time: str = "17:00",
    stride: int = 6,
    horizon_bars: int = 6,
    price_context_cols: Optional[Sequence[str]] = None,
    include_next_day_rth: bool = False,
    next_day_rth_start: str = "08:30",
    next_day_rth_end: str = "13:00",
) -> List[dict]:
    """Build one sample per eligible bar in the event session window.

    Each sample carries the event day's shared phase orderflow as context
    and the bar's own technical features + rolling forward return target.

    Parameters
    ----------
    prep : EventReleaseQuantilePrep
        Must have ``load_data()`` already called.
    event_dates : sequence of date
        Actual event release dates.
    post_end_times : dict
        ``{date: pd.Timestamp}`` — post-window end time per event day.
    phase_orderflow : dict
        ``{date: np.ndarray (P, 128, F)}`` — phase orderflow per event day
        from the orderflow parquet.  P=3 (pre, event, post).
    session_close_time : str
        Session close ``"HH:MM"``.  Last valid bar leaves room for the
        forward target to complete before close.
    stride : int
        Take every *stride*-th eligible bar.  stride=6 with 5-min bars
        gives non-overlapping 30-min windows.
    horizon_bars : int
        Forward bars for the rolling target (6 bars = 30 min).
    price_context_cols : list of str, optional
        Extra columns from the frame to append to bar_features.
        Default: ctx_* columns from ``_build_price_context``.
    include_next_day_rth : bool
        If True, extend the sample window into the next calendar day's
        US RTH session.
    next_day_rth_start : str
        Next-day RTH start time in ``"HH:MM"`` format.
    next_day_rth_end : str
        Next-day RTH end time in ``"HH:MM"`` format.

    Returns
    -------
    list of dict, one per bar:
        phase_features : np.ndarray (P, 128, F) — shared event context
        bar_features : np.ndarray (F_bar,) — technical + price context
        target_return : float — rolling 30-min forward log return
        target_class : int — ordinal class label
        is_last_bar : bool — True if this is the last eligible bar of the day
        date : date
    """
    if prep.frame is None:
        raise ValueError("Call prep.load_data() before building session samples")

    frame = prep.frame
    close_h, close_m = (int(x) for x in session_close_time.split(":"))
    next_rth_start_h, next_rth_start_m = (int(x) for x in next_day_rth_start.split(":"))
    next_rth_end_h, next_rth_end_m = (int(x) for x in next_day_rth_end.split(":"))

    target_col = f"target_ret_{horizon_bars}"
    if target_col not in frame.columns:
        raise KeyError(
            f"Target column '{target_col}' not found.  "
            f"Ensure horizon_bars={horizon_bars} matches prep config."
        )

    if prep.single_point_class_col and prep.single_point_class_col in frame.columns:
        class_col = prep.single_point_class_col
    elif prep.target_class_cols:
        class_col = prep.target_class_cols[-1]
    else:
        raise ValueError("No classification target columns found")

    tech_cols = list(prep.technical_cols)
    if price_context_cols is None:
        price_context_cols = [c for c in frame.columns if c.startswith("ctx_")]
    bar_feature_cols = tech_cols + list(price_context_cols)

    bar_arr = frame[bar_feature_cols].fillna(0.0).to_numpy(dtype=np.float32) if bar_feature_cols else None
    target_arr = frame[target_col].to_numpy(dtype=np.float32)
    class_arr = frame[class_col].to_numpy(dtype=np.float64)
    timestamps = frame.index

    def _time_tuple(ts: pd.Timestamp) -> Tuple[int, int]:
        return int(ts.hour), int(ts.minute)

    def _strip_tz(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_localize(None) if getattr(ts, "tzinfo", None) is not None else ts

    def _is_event_day_bar(
        ts: pd.Timestamp,
        future_ts: pd.Timestamp,
        *,
        sample_date: date,
        window_start: pd.Timestamp,
    ) -> bool:
        if ts.date() != sample_date or future_ts.date() != sample_date:
            return False
        if _strip_tz(ts) < _strip_tz(window_start):
            return False
        return _time_tuple(ts) <= (close_h, close_m) and _time_tuple(future_ts) <= (close_h, close_m)

    def _is_next_day_rth_bar(
        ts: pd.Timestamp,
        future_ts: pd.Timestamp,
        *,
        sample_date: date,
    ) -> bool:
        if ts.date() != sample_date or future_ts.date() != sample_date:
            return False
        ts_time = _time_tuple(ts)
        future_time = _time_tuple(future_ts)
        return (
            (next_rth_start_h, next_rth_start_m) <= ts_time <= (next_rth_end_h, next_rth_end_m)
            and future_time <= (next_rth_end_h, next_rth_end_m)
        )

    samples: List[dict] = []
    for ev_date in sorted(event_dates):
        post_end = post_end_times.get(ev_date)
        if post_end is None:
            continue
        phase_of = phase_orderflow.get(ev_date)
        if phase_of is None:
            continue

        eligible = []
        next_day = ev_date + pd.Timedelta(days=1)
        for idx in range(len(timestamps)):
            ts = timestamps[idx]
            future_idx = idx + horizon_bars
            if future_idx >= len(frame):
                continue
            future_ts = timestamps[future_idx]
            if not np.isfinite(target_arr[idx]):
                continue
            if np.isnan(class_arr[idx]):
                continue
            in_event_window = _is_event_day_bar(
                ts,
                future_ts,
                sample_date=ev_date,
                window_start=post_end,
            )
            in_next_day_window = include_next_day_rth and _is_next_day_rth_bar(
                ts,
                future_ts,
                sample_date=next_day,
            )
            if not (in_event_window or in_next_day_window):
                continue
            eligible.append(idx)

        if not eligible:
            continue

        if stride > 1:
            eligible = eligible[::stride]

        for bar_i, idx in enumerate(eligible):
            is_last = bar_i == len(eligible) - 1
            samples.append(
                {
                    "phase_features": phase_of,
                    "bar_features": bar_arr[idx] if bar_arr is not None else np.zeros(0, dtype=np.float32),
                    "target_return": float(target_arr[idx]),
                    "target_class": int(class_arr[idx]),
                    "is_last_bar": is_last,
                    "date": timestamps[idx].date(),
                    "event_date": ev_date,
                }
            )

    return samples


class EventSessionDataset(Dataset):
    """Per-bar dataset for ClassificationEventTFT.

    Each item is one bar with its event day's phase orderflow context.
    Standard ``DataLoader`` batching works (fixed-size samples).
    """

    def __init__(self, samples: List[dict], return_metadata: bool = False):
        self.samples = samples
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        out = {
            "phase_features": torch.from_numpy(
                np.asarray(s["phase_features"])
            ).float(),
            "bar_features": torch.from_numpy(
                np.asarray(s["bar_features"])
            ).float(),
            "target_return": torch.tensor(s["target_return"], dtype=torch.float32),
            "target_class": torch.tensor(s["target_class"], dtype=torch.long),
            "is_last_bar": torch.tensor(s["is_last_bar"], dtype=torch.bool),
        }
        if self.return_metadata:
            out["date"] = s["date"]
            out["event_date"] = s.get("event_date", s["date"])
        return out


def event_session_collate_fn(
    batch: List[dict],
) -> Dict[str, torch.Tensor]:
    """Default collate for per-bar event samples."""
    collated: Dict[str, torch.Tensor] = {
        "phase_features": torch.stack([b["phase_features"] for b in batch]),
        "bar_features": torch.stack([b["bar_features"] for b in batch]),
        "target_return": torch.stack([b["target_return"] for b in batch]),
        "target_class": torch.stack([b["target_class"] for b in batch]),
        "is_last_bar": torch.stack([b["is_last_bar"] for b in batch]),
    }
    if "date" in batch[0]:
        collated["_dates"] = [b["date"] for b in batch]
    if "event_date" in batch[0]:
        collated["_event_dates"] = [b["event_date"] for b in batch]
    return collated
