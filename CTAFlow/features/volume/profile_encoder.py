from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd


def _vw_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """
    Volume-weighted quantile of x at q in [0, 1].
    """
    if x.size == 0:
        return np.nan
    q = float(np.clip(q, 0.0, 1.0))
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx].astype(np.float64)
    wsum = ws.sum()
    if wsum <= 0:
        return float(np.nan)
    cdf = np.cumsum(ws) / (wsum + 1e-12)
    return float(np.interp(q, cdf, xs))


@dataclass
class VolumeProfileEncoderConfig:
    num_bins: int = 128

    # Range selection (per-day VW quantiles, then take robust global quantiles across days)
    alpha: float = 0.01      # per-day VW quantile tail cut (1% -> ~98% volume coverage)
    global_low_q: float = 0.05
    global_high_q: float = 0.95
    pad_frac: float = 0.08   # expand global range by this fraction of width

    # Coordinate system
    anchor: str = "vwap"     # currently supports "vwap" or "median"
    use_log_coord: bool = True  # x = log(price / anchor)

    # Histogram behavior
    clip_to_range: bool = True   # if True, clip x into [L,U] so no volume is dropped
    smooth: bool = False         # light smoothing of hist after binning

    # Channels
    include_shape_total: bool = True
    include_imbalance: bool = True       # (ask - bid) / (ask + bid)
    include_magnitude: bool = True       # repeated log1p(total volume) across bins
    include_shape_bidask: bool = False   # if True, adds separate normalized bid + ask shapes
    include_grad_shape: bool = False     # gradient of total shape


class VolumeProfileEncoder:
    """
    Fit global bin edges on training days, then transform daily volume profiles
    into fixed-size CNN-ready arrays with shape-first normalization.

    Output is (C, B) float32 by default.
    """

    def __init__(self, cfg: VolumeProfileEncoderConfig):
        self.cfg = cfg
        self.bin_edges_: Optional[np.ndarray] = None
        self.range_: Optional[Tuple[float, float]] = None

    # --------------------------
    # Fitting (train-set only)
    # --------------------------
    def fit(self, train_days: Sequence[pd.DataFrame]) -> "VolumeProfileEncoder":
        """
        Fit global [L,U] range + bin edges using robust aggregation of per-day
        volume-weighted quantile ranges in the chosen coordinate system.
        """
        lows, highs = [], []

        for df in train_days:
            x, w = self._day_xw(df, col="TotalVolume")
            if x is None:
                continue
            lo = _vw_quantile(x, w, self.cfg.alpha)
            hi = _vw_quantile(x, w, 1.0 - self.cfg.alpha)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                lows.append(lo)
                highs.append(hi)

        if len(lows) < 10:
            raise ValueError(f"Not enough valid training days to fit range (got {len(lows)}).")

        L = float(np.quantile(lows, self.cfg.global_low_q))
        U = float(np.quantile(highs, self.cfg.global_high_q))

        if not np.isfinite(L) or not np.isfinite(U) or U <= L:
            raise ValueError(f"Invalid fitted range: L={L}, U={U}")

        # Pad range
        width = U - L
        pad = self.cfg.pad_frac * width
        Lp, Up = L - pad, U + pad

        self.range_ = (Lp, Up)
        self.bin_edges_ = np.linspace(Lp, Up, self.cfg.num_bins + 1, dtype=np.float64)
        return self

    # --------------------------
    # Transform
    # --------------------------
    def transform(self, day_df: pd.DataFrame) -> np.ndarray:
        """
        Transform one day into (C, B) array.
        """
        if self.bin_edges_ is None or self.range_ is None:
            raise RuntimeError("Encoder not fit. Call fit(train_days) first.")

        B = self.cfg.num_bins
        L, U = self.range_
        edges = self.bin_edges_

        # Build histograms for total/bid/ask
        h_total = self._hist(day_df, "TotalVolume", edges, L, U)
        h_bid = self._hist(day_df, "BidVolume", edges, L, U) if "BidVolume" in day_df.columns else None
        h_ask = self._hist(day_df, "AskVolume", edges, L, U) if "AskVolume" in day_df.columns else None

        if self.cfg.smooth:
            h_total = self._smooth1d(h_total)
            if h_bid is not None: h_bid = self._smooth1d(h_bid)
            if h_ask is not None: h_ask = self._smooth1d(h_ask)

        channels = []

        # Channel: total SHAPE (normalized)
        if self.cfg.include_shape_total:
            s_total = self._normalize_shape(h_total)
            channels.append(s_total)

        # Channel(s): bid/ask SHAPES (optional)
        if self.cfg.include_shape_bidask:
            if h_bid is None or h_ask is None:
                raise ValueError("include_shape_bidask=True requires BidVolume and AskVolume columns.")
            channels.append(self._normalize_shape(h_bid))
            channels.append(self._normalize_shape(h_ask))

        # Channel: imbalance per bin in [-1, 1]
        if self.cfg.include_imbalance:
            if h_bid is None or h_ask is None:
                # If you don't have bid/ask, you can disable this channel
                raise ValueError("include_imbalance=True requires BidVolume and AskVolume columns.")
            imb = (h_ask - h_bid) / (h_ask + h_bid + 1e-12)
            channels.append(imb.astype(np.float32))

        # Channel: magnitude as a repeated scalar (shape-first, magnitude available)
        if self.cfg.include_magnitude:
            total_vol = float(np.sum(h_total))
            m = np.log1p(total_vol).astype(np.float32)
            channels.append(np.full((B,), m, dtype=np.float32))

        # Channel: gradient of total shape (optional)
        if self.cfg.include_grad_shape:
            s_total = self._normalize_shape(h_total)
            grad = np.gradient(s_total).astype(np.float32)
            channels.append(grad)

        X = np.stack(channels, axis=0).astype(np.float32)  # (C, B)
        return X

    def transform_many(self, days: Sequence[pd.DataFrame]) -> np.ndarray:
        """
        Transform multiple days -> (N, C, B).
        """
        Xs = [self.transform(df) for df in days]
        return np.stack(Xs, axis=0)

    # --------------------------
    # Internals
    # --------------------------
    def _day_anchor(self, df: pd.DataFrame) -> float:
        prices = df.index.to_numpy(dtype=np.float64)
        if self.cfg.anchor == "vwap":
            w = df["TotalVolume"].to_numpy(dtype=np.float64)
            wsum = w.sum()
            if wsum <= 0:
                return float(np.nan)
            return float((prices * w).sum() / (wsum + 1e-12))
        elif self.cfg.anchor == "median":
            return float(np.median(prices))
        else:
            raise ValueError(f"Unknown anchor: {self.cfg.anchor}")

    def _day_xw(self, df: pd.DataFrame, col: str = "TotalVolume") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if df is None or df.empty or col not in df.columns:
            return None, None

        prices = df.index.to_numpy(dtype=np.float64)
        w = df[col].to_numpy(dtype=np.float64)

        # Drop non-positive weights and non-finite prices
        mask = np.isfinite(prices) & np.isfinite(w) & (w > 0)
        prices = prices[mask]
        w = w[mask]
        if prices.size < 2:
            return None, None

        anchor = self._day_anchor(df)
        if not np.isfinite(anchor) or anchor <= 0:
            return None, None

        if self.cfg.use_log_coord:
            x = np.log(prices / anchor)
        else:
            x = prices - anchor

        return x.astype(np.float64), w.astype(np.float64)

    def _hist(self, df: pd.DataFrame, col: str, edges: np.ndarray, L: float, U: float) -> np.ndarray:
        x, w = self._day_xw(df, col=col)
        if x is None:
            return np.zeros((self.cfg.num_bins,), dtype=np.float32)

        if self.cfg.clip_to_range:
            x = np.clip(x, L, U)

        h, _ = np.histogram(x, bins=edges, weights=w)
        return h.astype(np.float32)

    @staticmethod
    def _normalize_shape(h: np.ndarray) -> np.ndarray:
        # Shape-first: probability mass over bins
        s = h / (np.sum(h) + 1e-12)
        return s.astype(np.float32)

    @staticmethod
    def _smooth1d(h: np.ndarray) -> np.ndarray:
        # light 3-tap smoothing: [0.25, 0.5, 0.25]
        k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        hp = np.pad(h.astype(np.float32), (1, 1), mode="edge")
        return np.convolve(hp, k, mode="valid").astype(np.float32)

def save_profiles_npz(path, dates, X, **extra):
    """
    dates: array-like of datetime/date/str, len N
    X: np.ndarray shape (N, C, B)
    """
    dates = pd.to_datetime(pd.Index(dates)).tz_localize(None)
    dates64 = dates.values.astype("datetime64[ns]")

    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        dates=dates64,
        **extra
    )

    return
