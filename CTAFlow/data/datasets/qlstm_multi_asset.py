from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from CTAFlow.features.daily_engine import AssetFeatureEngine, MarketFeatureEngine
from CTAFlow.models.deep_learning.multi_asset_qlstm import QLSTMConfig


def compute_group_ewma_volatilities(
    asset_log_returns: Dict[str, pd.Series],
    asset_class_map: Dict[str, int],
    ewma_lambda: float = 0.94,
) -> Dict[str, pd.Series]:
    """
    Compute class-group EWMA volatility series for each asset.

    Per-asset:
      var_t = lambda * var_{t-1} + (1-lambda) * r_{t-1}^2
    Per-group:
      sigma_group(asset_t) = average sigma_t across assets in same class.
    """
    groups: Dict[int, List[str]] = {}
    for ticker, cls_id in asset_class_map.items():
        groups.setdefault(int(cls_id), []).append(ticker)

    per_asset_vol: Dict[str, pd.Series] = {}
    for ticker, log_ret in asset_log_returns.items():
        var = pd.Series(0.0, index=log_ret.index, dtype=np.float64)
        r2 = log_ret.fillna(0.0).astype(np.float64) ** 2
        for i in range(1, len(var)):
            var.iloc[i] = ewma_lambda * var.iloc[i - 1] + (1.0 - ewma_lambda) * r2.iloc[i - 1]
        per_asset_vol[ticker] = np.sqrt(var + 1e-10)

    out: Dict[str, pd.Series] = {}
    for ticker, log_ret in asset_log_returns.items():
        cls_id = int(asset_class_map.get(ticker, 0))
        members = [m for m in groups.get(cls_id, [ticker]) if m in per_asset_vol]
        if not members:
            out[ticker] = pd.Series(0.01, index=log_ret.index, dtype=np.float64)
            continue
        vols = [per_asset_vol[m].reindex(log_ret.index).ffill().bfill().fillna(0.01) for m in members]
        out[ticker] = pd.concat(vols, axis=1).mean(axis=1)
    return out


class MultiAssetDistributionDataset(Dataset):
    """
    Variable-length sequence dataset for qLSTM distribution modeling.

    The model predicts the conditional PDF of the **next single-day return**
    r_{t+1}, following the paper methodology.  Each sample therefore produces
    scalar target values rather than a multi-step horizon vector.

    Output keys:
      - asset_features   : [seq_len, F_asset]
      - market_features  : [seq_len, F_market]
      - asset_class_id   : scalar long
      - target_return    : scalar float — r_{t+1} (next-day raw log return)
      - target_norm      : scalar float — r_{t+1} / σ̄_t (normalised by today's
                           group EWMA vol, known at prediction time)
      - group_vol        : scalar float — σ̄_t (for denormalising predictions)
      - lookback_returns : [seq_len]
      - seq_length       : scalar long
    """

    def __init__(
        self,
        asset_features: Dict[str, pd.DataFrame],
        market_features: pd.DataFrame,
        asset_log_returns: Dict[str, pd.Series],
        group_volatilities: Dict[str, pd.Series],
        asset_class_map: Dict[str, int],
        date_range: Tuple[str, str],
        seq_range: Tuple[int, int] = (15, 30),
    ):
        self.asset_class_map = dict(asset_class_map)
        self.seq_range = (int(seq_range[0]), int(seq_range[1]))

        self.date_start = pd.Timestamp(date_range[0])
        self.date_end = pd.Timestamp(date_range[1])

        self.market_dates = pd.to_datetime(market_features.index)
        split_mask = (self.market_dates >= self.date_start) & (self.market_dates <= self.date_end)
        self.split_dates = self.market_dates[split_mask]
        self.market_np = market_features.loc[self.split_dates].values.astype(np.float32)

        self.asset_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.samples: List[Tuple[str, int]] = []

        max_seq = self.seq_range[1]
        t_total = len(self.split_dates)
        # Need at least max_seq days for lookback + 1 day for the target
        if t_total < (max_seq + 2):
            return

        for ticker, feat_df in asset_features.items():
            if ticker not in asset_log_returns:
                continue
            if ticker not in group_volatilities:
                continue

            feat_aligned = feat_df.reindex(self.split_dates).ffill().bfill().fillna(0.0)
            ret_aligned = asset_log_returns[ticker].reindex(self.split_dates).fillna(0.0)
            vol_aligned = group_volatilities[ticker].reindex(self.split_dates).ffill().bfill().fillna(0.01)

            self.asset_data[ticker] = {
                "features": feat_aligned.values.astype(np.float32),
                "returns": ret_aligned.values.astype(np.float32),
                "group_vol": vol_aligned.values.astype(np.float32),
            }

            # t is the prediction point: we observe data up to t and predict
            # the return at t (i.e. r_{t+1} in calendar terms).  We need at
            # least one future return, so t can go up to t_total - 1.
            for t in range(max_seq, t_total - 1):
                self.samples.append((ticker, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ticker, t = self.samples[idx]
        seq_len = int(np.random.randint(self.seq_range[0], self.seq_range[1] + 1))
        start = t - seq_len

        data = self.asset_data[ticker]
        asset_feat = data["features"][start:t]
        market_feat = self.market_np[start:t]
        lookback_ret = data["returns"][start:t]

        # Scalar target: next-day return r_{t+1}
        raw_ret = data["returns"][t]
        # Normalise by TODAY's group EWMA vol (known at prediction time, no
        # look-ahead bias).
        gvol = data["group_vol"][t]
        norm_ret = raw_ret / (gvol + 1e-10)

        class_id = int(self.asset_class_map.get(ticker, 0))
        return {
            "asset_features": torch.from_numpy(asset_feat),
            "market_features": torch.from_numpy(market_feat),
            "asset_class_id": torch.tensor(class_id, dtype=torch.long),
            "target_return": torch.tensor(raw_ret, dtype=torch.float32),
            "target_norm": torch.tensor(norm_ret, dtype=torch.float32),
            "group_vol": torch.tensor(gvol, dtype=torch.float32),
            "lookback_returns": torch.from_numpy(lookback_ret),
            "seq_length": torch.tensor(seq_len, dtype=torch.long),
        }


def collate_variable_seq(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_seq = max(int(item["seq_length"].item()) for item in batch)
    f_asset = int(batch[0]["asset_features"].shape[-1])
    f_market = int(batch[0]["market_features"].shape[-1])

    asset_features = torch.zeros(batch_size, max_seq, f_asset)
    market_features = torch.zeros(batch_size, max_seq, f_market)
    lookback_returns = torch.zeros(batch_size, max_seq)
    target_return = torch.zeros(batch_size)
    target_norm = torch.zeros(batch_size)
    group_vol = torch.zeros(batch_size)
    asset_class_ids = torch.zeros(batch_size, dtype=torch.long)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        sl = int(item["seq_length"].item())
        asset_features[i, :sl] = item["asset_features"]
        market_features[i, :sl] = item["market_features"]
        lookback_returns[i, :sl] = item["lookback_returns"]
        target_return[i] = item["target_return"]
        target_norm[i] = item["target_norm"]
        group_vol[i] = item["group_vol"]
        asset_class_ids[i] = item["asset_class_id"]
        seq_lengths[i] = item["seq_length"]

    return {
        "asset_features": asset_features,
        "market_features": market_features,
        "asset_class_ids": asset_class_ids,
        "target_return": target_return,
        "target_norm": target_norm,
        "group_vol": group_vol,
        "lookback_returns": lookback_returns,
        "seq_lengths": seq_lengths,
    }


def build_full_pipeline(
    start: str = "2000-01-01",
    end: str = "2024-01-01",
    zscore_window: int = 219,
    ewma_lambda: float = 0.94,
    seq_range: Tuple[int, int] = (15, 30),
    batch_size: int = 64,
    num_workers: int = 0,
    cache_dir: str = "./data_cache",
    market_tickers: Optional[Sequence[str]] = None,
    market_download_all: bool = True,
    market_raw_data: Optional[Dict[str, pd.DataFrame]] = None,
    asset_tickers: Optional[Sequence[str]] = None,
    asset_download_map: Optional[Dict[str, str]] = None,
    asset_raw_data: Optional[Dict[str, pd.DataFrame]] = None,
    split_dates: Optional[Dict[str, Tuple[str, str]]] = None,
    classification_group: str = "category",
) -> Tuple[DataLoader, DataLoader, DataLoader, QLSTMConfig]:
    """
    End-to-end qLSTM data pipeline.
    """
    market_engine = MarketFeatureEngine(
        start=start,
        end=end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        market_tickers=market_tickers,
        download_all=market_download_all,
        raw_data=market_raw_data,
    )
    market_engine.download()
    market_df = market_engine.build_features()

    asset_engine = AssetFeatureEngine(
        start=start,
        end=end,
        zscore_window=zscore_window,
        ewma_lambda=ewma_lambda,
        cache_dir=cache_dir,
        asset_tickers=asset_tickers,
        ticker_download_map=asset_download_map,
        raw_data=asset_raw_data,
        classification_group=classification_group,
    )
    asset_engine.download()
    asset_features = asset_engine.build_features()

    asset_class_map = {t: c for t, c in asset_engine.asset_class_map.items() if t in asset_features}
    group_vols = compute_group_ewma_volatilities(
        asset_log_returns=asset_engine.asset_log_returns,
        asset_class_map=asset_class_map,
        ewma_lambda=ewma_lambda,
    )

    splits = split_dates or {
        "train": (start, "2017-12-31"),
        "val": ("2018-01-01", "2019-12-31"),
        "test": ("2020-01-01", end),
    }

    datasets = {
        split: MultiAssetDistributionDataset(
            asset_features=asset_features,
            market_features=market_df,
            asset_log_returns=asset_engine.asset_log_returns,
            group_volatilities=group_vols,
            asset_class_map=asset_class_map,
            date_range=date_range,
            seq_range=seq_range,
        )
        for split, date_range in splits.items()
    }

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_seq,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(datasets["train"]) >= batch_size,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_seq,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_seq,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = max(asset_class_map.values()) + 1 if asset_class_map else 1
    config = QLSTMConfig(
        asset_input_dim=asset_engine.num_features,
        market_input_dim=market_engine.num_features,
        num_asset_classes=num_classes,
    )
    return train_loader, val_loader, test_loader, config
