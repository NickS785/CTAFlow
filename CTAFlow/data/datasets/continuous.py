from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict


def _build_sample_mask(
    df: pd.DataFrame,
    target_cols: List[str],
    lookback: int,
    sample_mode: str = "active",
    allow_overlap: bool = True,
) -> np.ndarray:
    """
    Returns boolean mask of eligible rows by index position.
    Requires: df has 'session_code' (0 none, 1 london, 2 usa, 3 overlap) and/or 'is_active'.
    """
    n = len(df)
    ok = np.ones(n, dtype=bool)

    # must have targets
    for c in target_cols:
        ok &= df[c].notna().to_numpy()

    # must have enough history
    ok[:lookback - 1] = False

    # session filtering
    if sample_mode == "all":
        return ok

    if sample_mode == "active":
        # London or USA including overlap
        if "is_active" in df.columns:
            ok &= df["is_active"].astype(bool).to_numpy()
        else:
            ok &= (df["session_code"].to_numpy() > 0)

    elif sample_mode == "overlap":
        ok &= (df["session_code"].to_numpy() == 3)

    elif sample_mode == "london":
        sc = df["session_code"].to_numpy()
        if allow_overlap:
            ok &= (sc == 1) | (sc == 3)
        else:
            ok &= (sc == 1)

    elif sample_mode == "usa":
        sc = df["session_code"].to_numpy()
        if allow_overlap:
            ok &= (sc == 2) | (sc == 3)
        else:
            ok &= (sc == 2)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    return ok


def _contiguous_5m_check(idx: pd.DatetimeIndex, lookback: int, bar_minutes: int = 5) -> np.ndarray:
    """
    Precompute whether each row i has a fully contiguous lookback window [i-lookback+1..i]
    at bar_minutes frequency.
    """
    n = len(idx)
    ok = np.ones(n, dtype=bool)
    ok[:lookback - 1] = False

    # diff in minutes between consecutive rows
    diffs = idx.to_series().diff().dt.total_seconds().div(60.0).to_numpy()
    diffs[0] = bar_minutes  # dummy

    # a window is contiguous if all diffs inside window are == bar_minutes
    # i is valid if diffs[i-lookback+1 .. i] are all bar_minutes (except the first element which corresponds to gap into window)
    good_step = (diffs == bar_minutes)
    # rolling min over good_step
    # for i, need all good_step in [i-lookback+1..i] True
    # easiest: cumulative sum of bad steps
    bad = (~good_step).astype(np.int32)
    cbad = np.cumsum(bad)
    for i in range(lookback - 1, n):
        left = i - lookback + 1
        bad_in_window = cbad[i] - (cbad[left - 1] if left > 0 else 0)
        ok[i] = (bad_in_window == 0)
    return ok


class ContinuousWindowDataset(Dataset):
    """
    Takes a continuous dataframe and produces:
      x: [lookback, n_features]
      y: [n_targets]  (e.g. 12 horizons for next 60m)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        lookback: int = 256,
        sample_mode: str = "active",        # "all"|"active"|"london"|"usa"|"overlap"
        allow_overlap: bool = True,
        enforce_contiguous: bool = True,
        bar_minutes: int = 5,
        return_meta: bool = False,
    ):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.lookback = int(lookback)
        self.return_meta = return_meta

        base_mask = _build_sample_mask(
            df=df,
            target_cols=target_cols,
            lookback=self.lookback,
            sample_mode=sample_mode,
            allow_overlap=allow_overlap,
        )

        if enforce_contiguous:
            cont = _contiguous_5m_check(df.index, lookback=self.lookback, bar_minutes=bar_minutes)
            base_mask &= cont

        self.sample_pos = np.flatnonzero(base_mask)
        if len(self.sample_pos) == 0:
            raise ValueError("No eligible samples found. Check your sample_mode / targets / lookback / contiguity.")

        # cache arrays for speed
        self.X = df[feature_cols].to_numpy(dtype=np.float32)
        self.Y = df[target_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sample_pos)

    def __getitem__(self, idx: int):
        i = self.sample_pos[idx]
        j0 = i - self.lookback + 1
        x = torch.from_numpy(self.X[j0:i + 1])     # [L, F]
        y = torch.from_numpy(self.Y[i])            # [T]

        if not self.return_meta:
            return x, y

        ts = self.df.index[i]
        sess = int(self.df["session_code"].iloc[i]) if "session_code" in self.df.columns else -1
        return x, y, {"timestamp": ts, "session_code": sess}
