"""Extract full-day VPIN + rasterized VPIN for HG, NG, HE, LE.

Produces per ticker:
  {out_dir}/{TICKER}/vpin.parquet     — sequential VPIN features
  {out_dir}/{TICKER}/rasterized.npz   — date-keyed rasterized VPIN grids

Usage:
    python scripts/extract_vpin_fullday.py --tickers HG NG HE LE
    python scripts/extract_vpin_fullday.py --tickers HE LE --start 2011-01-01
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from CTAFlow.features.tick_extractor import (
    FeatureExtractorConfig,
    MultiFeatureExtraction,
)
from CTAFlow.features.base_extractor import SmartScidManager
from CTAFlow.config import DLY_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Ticker configs ────────────────────────────────────────────────────────
# session_start/end = VPIN extraction window (CT)
# profile_start/end = US RTH session (used for ps_* causal profile levels)
# tick_size from CONTRACT_SPECS_RAW
TICKER_CFG = {
    "CL": {"tick_size": 0.01},
    "GC": {"tick_size": 0.10},
    "HG": {"tick_size": 0.05},
    "ES": {"tick_size": 0.25},
    "NG": {"tick_size": 0.001},
    "RB": {"tick_size": 0.0001},
    "HO": {"tick_size": 0.0001},
    "HE": {"tick_size": 0.025},
    "LE": {"tick_size": 0.025},
}

# Shared session window for all tickers
SESSION_START = "02:00"
SESSION_END = "16:00"
# US RTH session for previous-session profile (ps_poc/val/vah)
PROFILE_START = "08:30"
PROFILE_END = "15:00"
# 14h session = 28 × 30min raster bars
RASTER_NUM_BARS = 28
RASTER_INTERVAL_MINS = 30

# ── Shared extraction settings ───────────────────────────────────────────
VPIN_WINDOW = 20
AUTO_BUCKET_CADENCE = 400
RASTER_BINS = 128
RASTER_SPAN_PCT = 0.035
RASTER_VOL_SCALE = 10.0
RASTER_PRICE_SCALE = 100.0


def extract_ticker(
    ticker: str,
    dates: list,
    out_dir: str,
    data_dir: str,
    cache_path: str = None,
    include_prev_24h_profile: bool = True,
):
    """Extract VPIN + rasterized VPIN for one ticker."""
    cfg = TICKER_CFG[ticker]
    ticker_dir = Path(out_dir) / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    config = FeatureExtractorConfig(
        ticker=ticker,
        data_dir=data_dir,
        tz="America/Chicago",
        contract_cache=cache_path,
        # VPIN
        vpin_bucket_size=None,
        auto_bucket_cadence=AUTO_BUCKET_CADENCE,
        vpin_window=VPIN_WINDOW,
        vpin_start_time=SESSION_START,
        vpin_end_time=SESSION_END,
        auto_bucket=True,
        # Profile = US RTH session (for ps_* causal profile levels)
        include_profile=True,
        include_prev_24h_profile=include_prev_24h_profile,
        profile_tick_size=cfg["tick_size"],
        profile_start_time=PROFILE_START,
        profile_end_time=PROFILE_END,
        # Number bars disabled
        include_number_bars=False,
        # Rasterization
        include_rasterized=True,
        raster_interval_mins=RASTER_INTERVAL_MINS,
        raster_num_bars=RASTER_NUM_BARS,
        raster_bins=RASTER_BINS,
        raster_span_pct=RASTER_SPAN_PCT,
        raster_vol_scale=RASTER_VOL_SCALE,
        raster_price_scale=RASTER_PRICE_SCALE,
        # Sequence features for LSTM
        include_sequence_features=True,
        include_ib=False,
        include_pre_summary=False,
    )

    logger.info(f"[{ticker}] vpin={SESSION_START}-{SESSION_END}, "
                f"profile(ps)={PROFILE_START}-{PROFILE_END}, "
                f"raster={RASTER_NUM_BARS}×{RASTER_INTERVAL_MINS}min, "
                f"tick_size={cfg['tick_size']}")

    extractor = MultiFeatureExtraction(config, dates)
    all_results = extractor.extract_all(verbose=True, n_jobs=1)

    if not all_results:
        logger.error(f"[{ticker}] no results extracted")
        return

    # ── Save VPIN parquet ────────────────────────────────────────────────
    vpin_frames = []
    for r in all_results:
        if "vpin" in r and not r["vpin"].empty:
            vpin = r["vpin"].copy()
            vpin["date"] = r["date"].date()
            vpin_frames.append(vpin)

    if vpin_frames:
        vpin_df = pd.concat(vpin_frames, axis=0)
        vpin_path = ticker_dir / "vpin.parquet"
        vpin_df.to_parquet(vpin_path)
        logger.info(f"[{ticker}] saved {vpin_path}: {len(vpin_df)} rows, "
                     f"{len(vpin_frames)} dates")
    else:
        logger.warning(f"[{ticker}] no VPIN data to save")

    # ── Save rasterized NPZ ─────────────────────────────────────────────
    def _size(arr):
        if arr is None:
            return 0
        return arr.numel() if hasattr(arr, "numel") else arr.size

    valid_raster = [
        (r["date"], r["rasterized"])
        for r in all_results
        if "rasterized" in r and _size(r["rasterized"]) > 0
    ]

    if valid_raster:
        raster_dict = {}
        for dt, tensor in valid_raster:
            arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor
            raster_dict[dt.strftime("%Y-%m-%d")] = arr.astype(np.float32)

        raster_path = ticker_dir / "rasterized.npz"
        np.savez_compressed(str(raster_path), **raster_dict)

        sample_shape = list(raster_dict.values())[0].shape
        logger.info(f"[{ticker}] saved {raster_path}: {len(raster_dict)} dates, "
                     f"shape={sample_shape}")
    else:
        logger.warning(f"[{ticker}] no rasterized data to save")


def main():
    parser = argparse.ArgumentParser(
        description="Extract full-day VPIN + rasterized VPIN"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["GC", "CL", "HG", "ES", "NG", "RB", "HO"],
    )
    parser.add_argument(
        "--out-dir", default="F:/Upload/s3/model_data",
    )
    parser.add_argument(
        "--data-dir", default=DLY_DATA_PATH,
    )
    parser.add_argument(
        "--start", default="2012-01-01",
    )
    parser.add_argument(
        "--end", default="2026-02-01",
    )
    parser.add_argument(
        "--cache", default=None,
        help="Path to contract map cache (.pkl). Auto-created if missing.",
    )
    parser.add_argument(
        "--no-pd", action="store_true",
        help="Skip pd_* (previous 24h) profile levels, keep only ps_* (previous session).",
    )
    args = parser.parse_args()

    # Build / load contract map cache
    cache_path = args.cache
    if cache_path is None:
        cache_path = str(Path(args.data_dir) / ".contract_cache.pkl")
    if not Path(cache_path).exists():
        logger.info("Building contract map cache (one-time)...")
        mgr = SmartScidManager(args.data_dir)
        mgr.save_contract_cache(cache_path)
    else:
        logger.info(f"Using cached contract map: {cache_path}")

    dates = pd.date_range(args.start, args.end, freq="B").tolist()
    logger.info(f"Processing {len(dates)} business days: {args.start} → {args.end}")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Output: {args.out_dir}")
    logger.info(f"Data: {args.data_dir}")

    for ticker in args.tickers:
        if ticker not in TICKER_CFG:
            logger.error(f"Unknown ticker {ticker} (known: {list(TICKER_CFG.keys())})")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting {ticker}")
        logger.info(f"{'='*60}")
        extract_ticker(ticker, dates, args.out_dir, args.data_dir,
                       cache_path=cache_path,
                       include_prev_24h_profile=not args.no_pd)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
