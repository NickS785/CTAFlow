"""
Daily Session Feature Extraction Script

Extracts VPIN, Number Bars, and Volume Profile features for energy tickers (RB, CL, HO)
during the 8:30-9:30 AM CT opening session.

Features extracted:
- VPIN: 8:30-9:30 (with pre-session 2:00-8:30 summary and VWAP)
- Number Bars: 15-minute intervals for 8:30-9:30
- Profile: 2:00-9:30 AM
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from CTAFlow.features.tick_extractor import (
    FeatureExtractorConfig,
    MultiFeatureExtraction,
    get_default_tick_size
)
from CTAFlow.config import DLY_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_session_features(
    ticker: str,
    dates: List[pd.Timestamp],
    output_dir: Optional[str] = None,
    data_dir: str = DLY_DATA_PATH,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Extract daily session features for a ticker.

    Time windows (Central Time):
    - VPIN: 8:30-9:30 AM
    - Number Bars: 8:30-9:30 AM with 15-minute intervals
    - Profile: 2:00-9:30 AM
    - Pre-summary: 2:00-8:30 AM (VWAP, volume, delta, high, low)

    Args:
        ticker: Contract symbol (e.g., 'CL', 'RB', 'HO')
        dates: List of dates to process
        output_dir: Output directory (default: F:/Upload/{ticker})
        data_dir: Path to SCID data directory
        verbose: Print progress

    Returns:
        Dict with saved file paths
    """
    ticker = ticker.upper()
    output_dir = Path(output_dir) if output_dir else Path("F:/Upload") / ticker.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get default tick size from contract specs
    tick_size = get_default_tick_size(ticker)

    # Configure extractor
    config = FeatureExtractorConfig(
        ticker=ticker,
        data_dir=data_dir,
        tz="America/Chicago",
        # VPIN settings
        vpin_bucket_size=None,  # Use auto_bucket_size
        auto_bucket_cadence=250,
        vpin_window=20,
        vpin_start_time="08:30",
        vpin_end_time="12:00",
        auto_bucket=True,
        # Profile settings
        include_profile=True,
        profile_tick_size=tick_size,
        profile_start_time="08:30",
        profile_end_time="12:00",
        # Number Bars settings
        include_number_bars=False,
        num_bars_start_time="08:30",
        num_bars_end_time="12:00",
        num_bars_interval="15min",
        num_bars_levels=128,
        num_bars_centering="rolling_vwap",
        num_bars_vwap_window="2h",
        # Pre-summary settings
        include_pre_summary=True,
        pre_summary_start_time="02:00",
        # Other settings
        include_sequence_features=True,
        include_ib=True,
        ib_minutes=60,
        value_area_pct=0.7
        include_rasterized=True,
        raster_interval_mins = 10
        raster_num_bars = 24,
        raster_span_pct = 0.03,
        raster_vol_scale= 10.0,
        raster_price_scale= 100.0
    )

    logger.info(f"Extracting features for {ticker}")
    logger.info(f"  Tick size: {tick_size}")
    logger.info(f"  Dates: {len(dates)}")
    logger.info(f"  Output: {output_dir}")

    # Create extractor and run
    extractor = MultiFeatureExtraction(config, dates)
    all_results = extractor.extract_all(verbose=verbose, n_jobs=1)

    # Save results
    prefix = f"{ticker}_session_0830_0930"
    paths = {}

    # 1. Save VPIN data with pre-summary
    vpin_frames = []
    for r in all_results:
        if not r['vpin'].empty:
            vpin = r['vpin'].copy()
            vpin['date'] = r['date'].date()
            vpin_frames.append(vpin)

    if vpin_frames:
        vpin_df = pd.concat(vpin_frames, axis=0)
        vpin_path = output_dir / f"{prefix}_vpin.parquet"
        vpin_df.to_parquet(vpin_path)
        paths['vpin'] = str(vpin_path)
        logger.info(f"Saved VPIN: {vpin_path}")

    # 2. Save Number Bars as NPZ with date-based keys
    valid_bars = [(r['date'], r['number_bars'], r['number_bars_meta'])
                  for r in all_results if r['number_bars'].size > 0]
    if valid_bars:
        # Use date strings as keys (e.g., '2024-01-15')
        nb_path = output_dir / f"{prefix}_number_bars.npz"
        np.savez_compressed(
            nb_path,
            **{v[0].strftime('%Y-%m-%d'): v[1] for v in valid_bars}
        )
        paths['number_bars'] = str(nb_path)
        logger.info(f"Saved number bars: {nb_path}")

        # Save metadata as well
        meta_frames = []
        for i, v in enumerate(valid_bars):
            if not v[2].empty:
                meta = v[2].copy()
                meta['date'] = v[0].date()
                meta_frames.append(meta)

        if meta_frames:
            meta_df = pd.concat(meta_frames, axis=0)
            meta_path = output_dir / f"{prefix}_number_bars_meta.parquet"
            meta_df.to_parquet(meta_path)
            paths['number_bars_meta'] = str(meta_path)

    # 3. Save Profile summaries
    profile_summaries = []
    for r in all_results:
        if not r['profile'].empty and 'TotalVolume' in r['profile'].columns:
            profile = r['profile']
            total_vol = profile['TotalVolume'].sum()
            poc = profile['TotalVolume'].idxmax()

            sorted_profile = profile.sort_values('TotalVolume', ascending=False)
            cumvol = sorted_profile['TotalVolume'].cumsum()
            va_levels = sorted_profile[cumvol <= total_vol * 0.7].index

            summary = {
                'date': r['date'],
                'poc': poc,
                'val': va_levels.min() if len(va_levels) > 0 else poc,
                'vah': va_levels.max() if len(va_levels) > 0 else poc,
                'total_volume': total_vol
            }

            if 'Delta' in profile.columns:
                summary['total_delta'] = profile['Delta'].sum()

            profile_summaries.append(summary)

    if profile_summaries:
        summary_df = pd.DataFrame(profile_summaries).set_index('date')
        summary_path = output_dir / f"{prefix}_profile_summary.parquet"
        summary_df.to_parquet(summary_path)
        paths['profile_summary'] = str(summary_path)
        logger.info(f"Saved profile summary: {summary_path}")

    # 4. Save encoded profiles using VolumeProfileEncoder
    from CTAFlow.features.volume.profile_encoder import (
        VolumeProfileEncoder, VolumeProfileEncoderConfig, save_profiles_npz
    )

    valid_profiles = [(r['date'], r['profile']) for r in all_results if not r['profile'].empty]
    if valid_profiles:
        profiles = [v[1] for v in valid_profiles]
        profile_dates = [v[0] for v in valid_profiles]

        # Fit encoder on first 80% of data
        n_train = int(len(profiles) * 0.8)
        enc_cfg = VolumeProfileEncoderConfig(num_bins=96, include_imbalance=True)
        encoder = VolumeProfileEncoder(enc_cfg)
        encoder.fit(profiles[:n_train])

        # Transform all profiles
        encoded = encoder.transform_many(profiles)

        profile_path = output_dir / f"{prefix}_profiles.npz"
        save_profiles_npz(profile_path, profile_dates, encoded)
        paths['profiles'] = str(profile_path)
        logger.info(f"Saved encoded profiles: {profile_path}")

    return paths


def run_session_extraction(
    tickers: List[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-31",
    output_base: str = "F:/Upload"
):
    """
    Run session feature extraction for specified tickers.

    Args:
        tickers: List of tickers (default: ['RB', 'CL', 'HO'])
        start_date: Start date string
        end_date: End date string
        output_base: Base output directory
    """
    tickers = tickers or ['CL', 'RB', 'HO', 'GC', 'NG']

    # Generate business days
    dates = pd.date_range(start_date, end_date, freq='B').tolist()
    logger.info(f"Processing {len(dates)} business days from {start_date} to {end_date}")

    all_paths = {}

    for ticker in tickers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {ticker}")
        logger.info(f"{'='*50}")

        output_dir = Path(output_base) / ticker.lower()

        paths = extract_session_features(
            ticker=ticker,
            dates=dates,
            output_dir=str(output_dir),
            verbose=True
        )

        all_paths[ticker] = paths

        logger.info(f"\nSaved files for {ticker}:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")

    return all_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract daily session features")
    parser.add_argument("--tickers", nargs="+", default=["RB", "GC"],
                        help="Tickers to process")
    parser.add_argument("--start", default="2012-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-12",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="F:/Upload",
                        help="Output directory")

    args = parser.parse_args()

    run_session_extraction(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        output_base=args.output
    )
