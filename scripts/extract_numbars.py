"""Extract NumberBars for CL, GC, ES using 1h candles + 1h rolling VWAP.

Produces {TICKER}_numbars.npz with:
  data : (N, 4, 32)  — [VolumeShape, Imbalance%, BarReturn, PriceOffset]
  idx  : (N,)         — datetime64[ns] timestamps per bar

Usage:
    python scripts/extract_numbars.py --data-dir D:/SierraChart/Data
    python scripts/extract_numbars.py --data-dir D:/SierraChart/Data --tickers GC ES
    python scripts/extract_numbars.py --data-dir D:/SierraChart/Data --out-dir F:/Upload/s3/model_data
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CTAFlow.features.volume.profile import NumberBarsExtractor

logger = logging.getLogger(__name__)

# ── Ticker configs ────────────────────────────────────────────────────────
TICKER_CFG = {
    "CL": {"tick_size": 0.01, "tz": "America/Chicago"},
    "GC": {"tick_size": 0.10, "tz": "America/Chicago"},
    "ES": {"tick_size": 0.25, "tz": "America/Chicago"},
    "HG": {"tick_size": 0.0005, "tz": "America/Chicago"},
    "NG": {"tick_size": 0.001, "tz": "America/Chicago"},
    "RB": {"tick_size": 0.0001, "tz": "America/Chicago"},
    "HO": {"tick_size": 0.0001, "tz": "America/Chicago"},
    "HE": {"tick_size": 0.00025, "tz": "America/Chicago", "session_start": "08:30", "session_end": "13:00"},
    "LE": {"tick_size": 0.00025, "tz": "America/Chicago", "session_start": "08:30", "session_end": "13:00"},
}

# ── Extraction settings ──────────────────────────────────────────────────
INTERVAL = "1h"
VWAP_WINDOW = "1h"
NUM_LEVELS = 50          # 101 raw bins — enough headroom for 1h bars
TARGET_BINS = 32         # final spatial resolution
YEARS_BACK = 16
NORMALIZE = True
SESSION_START = "00:00"
SESSION_END = "23:59"


def rasterize_to_fixed_bins(
    tensor: np.ndarray,
    target_bins: int = 32,
) -> np.ndarray:
    """Downsample (N, raw_bins, 4) → (N, 4, target_bins) via adaptive avg pool.

    For channels that represent densities / shapes (ch 0, 1) we average-pool.
    For the scalar channels (ch 2 = bar return, ch 3 = price offset) we
    also average-pool which preserves the monotone structure of price offset
    and the constant bar-return channel.
    """
    if tensor.ndim != 3 or tensor.shape[0] == 0:
        return np.empty((0, 4, target_bins), dtype=np.float32)

    N, raw_bins, C = tensor.shape

    if raw_bins == target_bins:
        return tensor.transpose(0, 2, 1).astype(np.float32)

    # Transpose to (N, C, raw_bins) for spatial resampling
    x = tensor.transpose(0, 2, 1)  # (N, 4, raw_bins)

    # Adaptive average pool along the bins axis
    out = np.zeros((N, C, target_bins), dtype=np.float32)
    for b in range(target_bins):
        lo = int(round(b * raw_bins / target_bins))
        hi = int(round((b + 1) * raw_bins / target_bins))
        hi = max(hi, lo + 1)  # at least one source bin
        out[:, :, b] = x[:, :, lo:hi].mean(axis=2)

    return out


def extract_ticker(
    data_dir: str,
    ticker: str,
    out_dir: str,
    start_date: datetime,
    end_date: datetime,
    chunk_months: int = 3,
):
    """Extract number bars for one ticker in monthly chunks."""
    cfg = TICKER_CFG[ticker]
    sess_start = cfg.get("session_start", SESSION_START)
    sess_end = cfg.get("session_end", SESSION_END)

    extractor = NumberBarsExtractor(
        data_dir=data_dir,
        ticker=ticker,
        tz=cfg["tz"],
        tick_size=cfg["tick_size"],
        interval=INTERVAL,
        vwap_window=VWAP_WINDOW,
        num_levels=NUM_LEVELS,
        centering_method="rolling_vwap",
    )

    all_data = []
    all_ts = []
    cursor = start_date

    while cursor < end_date:
        chunk_end = min(
            cursor + timedelta(days=30 * chunk_months),
            end_date,
        )

        s_str = cursor.strftime("%Y-%m-%d") + f" {sess_start}"
        e_str = chunk_end.strftime("%Y-%m-%d") + f" {sess_end}"

        try:
            tensor, meta = extractor.get_number_bars(
                start_time=s_str,
                end_time=e_str,
                normalize=NORMALIZE,
            )
        except Exception as e:
            logger.warning(f"[{ticker}] chunk {cursor.date()}→{chunk_end.date()} failed: {e}")
            cursor = chunk_end
            continue

        if tensor.size == 0:
            logger.info(f"[{ticker}] no data {cursor.date()}→{chunk_end.date()}")
            cursor = chunk_end
            continue

        # Rasterize raw bins → 32
        raster = rasterize_to_fixed_bins(tensor, TARGET_BINS)
        timestamps = pd.to_datetime(meta["Time"].values)

        all_data.append(raster)
        all_ts.append(timestamps.values.astype("datetime64[ns]"))

        bars = raster.shape[0]
        logger.info(
            f"[{ticker}] {cursor.date()}→{chunk_end.date()}: "
            f"{bars} bars, raw_bins={tensor.shape[1]}→{TARGET_BINS}"
        )

        cursor = chunk_end

    if not all_data:
        logger.error(f"[{ticker}] no data extracted — check SCID files in {data_dir}")
        return

    data = np.concatenate(all_data, axis=0).astype(np.float32)
    idx = np.concatenate(all_ts)

    # Deduplicate (overlapping chunk boundaries)
    _, uniq_mask = np.unique(idx, return_index=True)
    uniq_mask = np.sort(uniq_mask)
    data = data[uniq_mask]
    idx = idx[uniq_mask]

    out_path = Path(out_dir) / ticker / f"{ticker}_rasterized.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), data=data, idx=idx)

    logger.info(
        f"[{ticker}] saved {out_path}: "
        f"data={data.shape}, idx={idx.shape}, "
        f"range={pd.Timestamp(idx[0]).date()}→{pd.Timestamp(idx[-1]).date()}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract NumberBars for CL/GC/ES (1h bars, 1h rolling VWAP, 32 bins)"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to SierraChart SCID data directory",
    )
    parser.add_argument(
        "--out-dir", default="F:/Upload/s3/model_data",
        help="Output root directory (default: F:/Upload/s3/model_data)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["CL", "GC", "ES"],
        help="Tickers to extract (default: CL GC ES)",
    )
    parser.add_argument(
        "--years", type=int, default=YEARS_BACK,
        help=f"Years of history (default: {YEARS_BACK})",
    )
    parser.add_argument(
        "--chunk-months", type=int, default=3,
        help="Months per extraction chunk (default: 3)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)

    for ticker in args.tickers:
        if ticker not in TICKER_CFG:
            logger.error(f"Unknown ticker {ticker}, skipping (known: {list(TICKER_CFG.keys())})")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"Extracting {ticker}: {start_date.date()} → {end_date.date()}")
        logger.info(f"  interval={INTERVAL}, vwap={VWAP_WINDOW}, "
                     f"num_levels={NUM_LEVELS}→{TARGET_BINS} bins")
        logger.info(f"{'='*60}")

        extract_ticker(
            data_dir=args.data_dir,
            ticker=ticker,
            out_dir=args.out_dir,
            start_date=start_date,
            end_date=end_date,
            chunk_months=args.chunk_months,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
