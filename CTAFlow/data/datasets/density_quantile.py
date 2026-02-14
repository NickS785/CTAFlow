from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from CTAFlow.data.datasets.qlstm_multi_asset import compute_group_ewma_volatilities
from CTAFlow.features.daily_engine import AssetFeatureEngine, MarketFeatureEngine


@dataclass
class DensityQuantilePipelineConfig:
    asset_dim: int
    market_dim: int
    num_classes: int
    class_names: Dict[int, str]
    target_step: int


class DensityQuantileDataset(Dataset):
    """
    Variable-length sequence dataset for dense quantile / quantile LSTM models.

    Output keys (aligned with train_compare.py):
      - asset_features:   [seq_len, F_asset]
      - market_features:  [seq_len, F_market]
      - asset_class_ids:  scalar long
      - seq_lengths:      scalar long
      - target_return:    scalar float
      - target_norm:      scalar float
      - group_vol:        scalar float
      - lookback_returns: [seq_len]
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
        target_step: int = 1,
    ):
        self.asset_class_map = dict(asset_class_map)
        self.seq_range = (int(seq_range[0]), int(seq_range[1]))
        self.target_step = int(target_step)
        if self.target_step < 1:
            raise ValueError("target_step must be >= 1")

        self.date_start = pd.Timestamp(date_range[0])
        self.date_end = pd.Timestamp(date_range[1])

        market_dates = pd.to_datetime(market_features.index)
        split_mask = (market_dates >= self.date_start) & (market_dates <= self.date_end)
        self.split_dates = market_dates[split_mask]
        self.market_np = market_features.loc[self.split_dates].values.astype(np.float32)

        self.asset_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.samples: List[Tuple[str, int]] = []

        max_seq = self.seq_range[1]
        step_idx = self.target_step - 1
        t_total = len(self.split_dates)
        if t_total < (max_seq + self.target_step + 1):
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

            for t in range(max_seq, t_total - step_idx):
                # We access target at index t + step_idx
                if t + step_idx >= t_total:
                    break
                self.samples.append((ticker, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ticker, t = self.samples[idx]
        seq_len = int(np.random.randint(self.seq_range[0], self.seq_range[1] + 1))
        start = t - seq_len
        step_idx = self.target_step - 1

        ad = self.asset_data[ticker]
        asset_feat = ad["features"][start:t]  # [seq_len, F_asset]
        market_feat = self.market_np[start:t]  # [seq_len, F_market]
        lookback_ret = ad["returns"][start:t]  # [seq_len]

        target_return = float(ad["returns"][t + step_idx])
        group_vol = float(ad["group_vol"][t + step_idx])
        target_norm = target_return / (group_vol + 1e-10)

        class_id = int(self.asset_class_map.get(ticker, 0))
        return {
            "asset_features": torch.from_numpy(asset_feat),
            "market_features": torch.from_numpy(market_feat),
            "asset_class_ids": torch.tensor(class_id, dtype=torch.long),
            "seq_lengths": torch.tensor(seq_len, dtype=torch.long),
            "target_return": torch.tensor(target_return, dtype=torch.float32),
            "target_norm": torch.tensor(target_norm, dtype=torch.float32),
            "group_vol": torch.tensor(group_vol, dtype=torch.float32),
            "lookback_returns": torch.from_numpy(lookback_ret),
        }


def collate_density_quantile(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_seq = max(int(item["seq_lengths"].item()) for item in batch)
    f_asset = int(batch[0]["asset_features"].shape[-1])
    f_market = int(batch[0]["market_features"].shape[-1])

    asset_features = torch.zeros(batch_size, max_seq, f_asset, dtype=torch.float32)
    market_features = torch.zeros(batch_size, max_seq, f_market, dtype=torch.float32)
    lookback_returns = torch.zeros(batch_size, max_seq, dtype=torch.float32)
    asset_class_ids = torch.zeros(batch_size, dtype=torch.long)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_return = torch.zeros(batch_size, dtype=torch.float32)
    target_norm = torch.zeros(batch_size, dtype=torch.float32)
    group_vol = torch.zeros(batch_size, dtype=torch.float32)

    for i, item in enumerate(batch):
        sl = int(item["seq_lengths"].item())
        asset_features[i, :sl] = item["asset_features"]
        market_features[i, :sl] = item["market_features"]
        lookback_returns[i, :sl] = item["lookback_returns"]
        asset_class_ids[i] = item["asset_class_ids"]
        seq_lengths[i] = item["seq_lengths"]
        target_return[i] = item["target_return"]
        target_norm[i] = item["target_norm"]
        group_vol[i] = item["group_vol"]

    return {
        "asset_features": asset_features,
        "market_features": market_features,
        "asset_class_ids": asset_class_ids,
        "seq_lengths": seq_lengths,
        "target_return": target_return,
        "target_norm": target_norm,
        "group_vol": group_vol,
        "lookback_returns": lookback_returns,
    }


def build_density_quantile_pipeline(
    start: str = "2000-01-01",
    end: str = "2024-01-01",
    zscore_window: int = 219,
    ewma_lambda: float = 0.94,
    seq_range: Tuple[int, int] = (15, 30),
    target_step: int = 1,
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
) -> Tuple[DataLoader, DataLoader, DataLoader, DensityQuantilePipelineConfig]:
    """
    End-to-end pipeline for dense quantile / quantile LSTM models.
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
        split: DensityQuantileDataset(
            asset_features=asset_features,
            market_features=market_df,
            asset_log_returns=asset_engine.asset_log_returns,
            group_volatilities=group_vols,
            asset_class_map=asset_class_map,
            date_range=date_range,
            seq_range=seq_range,
            target_step=target_step,
        )
        for split, date_range in splits.items()
    }

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_density_quantile,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(datasets["train"]) >= batch_size,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_density_quantile,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_density_quantile,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = max(asset_class_map.values()) + 1 if asset_class_map else 1
    cfg = DensityQuantilePipelineConfig(
        asset_dim=asset_engine.num_features,
        market_dim=market_engine.num_features,
        num_classes=num_classes,
        class_names=asset_engine.class_names,
        target_step=target_step,
    )
    return train_loader, val_loader, test_loader, cfg

