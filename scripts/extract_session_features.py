"""
Daily Session Feature Extraction Script

Extracts VPIN, Volume Profile, and Rasterized features for livestock tickers (LE, GF, HE)
during the 8:30-12:00 CT session.

Features extracted:
- Profile: 8:30-12:00 CT (volume profile → POC/VAL/VAH/VWAP metrics)
- VPIN: 8:30-12:00 CT (variable-length orderflow sequences with profile metrics)
- Rasterized: 8:30-12:00 CT (VPIN sequences → fixed spatial grids)
- Number Bars: Disabled by default
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
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Extract daily session features for a ticker.

    Time windows (Central Time):
    All features use the same 8:30-12:00 PM session window:
    - Profile: 8:30-12:00 PM → Volume profile (POC/VAL/VAH/VWAP)
    - VPIN: 8:30-12:00 PM → Variable-length orderflow sequences
    - Rasterized: 8:30-12:00 PM → Fixed spatial grids from VPIN
    - Number Bars: Disabled by default

    The profile metrics (POC/VAL/VAH/VWAP) are calculated from the full session
    and added to the VPIN sequential data for model training.

    Outputs:
    - vpin: Parquet with sequential VPIN features + POC/VAL/VAH/VWAP
    - profiles: NPZ with (C, bins) encoded volume profiles per date
    - rasterized: NPZ with (num_bars, channels, bins) spatial grids per date
    - profile_summary: Parquet with POC, VAL, VAH metrics per date

    Args:
        ticker: Contract symbol (e.g., 'LE', 'GF', 'HE')
        dates: List of dates to process
        output_dir: Output directory (default: F:/Upload/{ticker})
        data_dir: Path to SCID data directory
        verbose: Print progress
        bucket_cache: Path to bucket cache CSV file (optional)
        save_bucket_cache: If True, save bucket cache after extraction (default: True)

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
        auto_bucket_cadence=400,
        vpin_window=20,
        vpin_start_time="06:30",
        vpin_end_time="09:30",
        auto_bucket=True,
        # Profile settings - SAME WINDOW AS VPIN
        include_profile=True,
        profile_tick_size=tick_size,
        profile_start_time="06:30",  # Profile during full session
        profile_end_time="09:30",     # Same as VPIN end
        # Number Bars settings
        include_number_bars=False,
        num_bars_start_time="06:30",
        num_bars_end_time="09:30",
        num_bars_interval="15min",
        num_bars_levels=64,
        num_bars_centering="rolling_vwap",
        num_bars_vwap_window="2h",
        # Rasterization settings
        include_rasterized=True,
        raster_interval_mins=15,      # 15-minute bars
        raster_num_bars=12,             # 14 bars for full session
        raster_bins=128,                # 128 vertical price levels
        raster_span_pct=0.035,          # ±3.5% from VWAP
        raster_vol_scale=10.0,
        raster_price_scale=100.0,
        # Pre-summary settings - DISABLED (calculate at end instead)
        include_pre_summary=False,
        pre_summary_start_time=None,
        # Other settings
        include_sequence_features=True,
        include_ib=True,
        ib_minutes=60,
        value_area_pct=0.7
    )

    logger.info(f"Extracting features for {ticker}")
    logger.info(f"  Tick size: {tick_size}")
    logger.info(f"  Dates: {len(dates)}")
    logger.info(f"  Output: {output_dir}")


    # Create extractor and run
    extractor = MultiFeatureExtraction(
        config, dates,

    )
    all_results = extractor.extract_all(verbose=verbose, n_jobs=1)

    # Diagnostic: Check what was extracted
    logger.info(f"\nExtraction Summary for {ticker}:")
    logger.info(f"  Total results: {len(all_results)}")
    if all_results:
        sample = all_results[0]
        logger.info(f"  Available keys: {list(sample.keys())}")

        # Count valid results for each feature type
        logger.info(f"\n  First sample details:")
        for key in sample.keys():
            if key != 'date':
                val = sample[key]
                if hasattr(val, 'empty'):
                    logger.info(f"    {key}: {'empty' if val.empty else f'{len(val)} rows'}")
                elif hasattr(val, 'shape'):
                    logger.info(f"    {key}: shape {val.shape}")
                elif hasattr(val, 'numel'):
                    logger.info(f"    {key}: {val.numel()} elements, shape {val.shape}")
                else:
                    logger.info(f"    {key}: {type(val)}")

        # Count how many valid results across all dates
        logger.info(f"\n  Valid results across all {len(all_results)} dates:")
        vpin_count = sum(1 for r in all_results if 'vpin' in r and hasattr(r['vpin'], 'empty') and not r['vpin'].empty)
        profile_count = sum(1 for r in all_results if 'profile' in r and hasattr(r['profile'], 'shape') and r['profile'].shape[0] > 0)
        nb_count = sum(1 for r in all_results if 'number_bars' in r and hasattr(r['number_bars'], 'shape') and (r['number_bars'].numel() if hasattr(r['number_bars'], 'numel') else r['number_bars'].size) > 0)
        raster_count = sum(1 for r in all_results if 'rasterized' in r and hasattr(r['rasterized'], 'shape') and (r['rasterized'].numel() if hasattr(r['rasterized'], 'numel') else r['rasterized'].size) > 0)

        logger.info(f"    vpin: {vpin_count} dates with data")
        logger.info(f"    profile: {profile_count} dates with data")
        logger.info(f"    number_bars: {nb_count} dates with data")
        logger.info(f"    rasterized: {raster_count} dates with data")

        # Show a few sample dates with data (if any)
        if profile_count > 0:
            logger.info(f"\n  First 5 dates with profile data:")
            count = 0
            for r in all_results:
                if 'profile' in r and hasattr(r['profile'], 'shape') and r['profile'].shape[0] > 0:
                    logger.info(f"    {r['date']}: profile shape {r['profile'].shape}")
                    count += 1
                    if count >= 5:
                        break
        else:
            logger.warning(f"\n  NO PROFILE DATA EXTRACTED! Checking first 5 results for errors...")
            for i, r in enumerate(all_results[:5]):
                logger.info(f"    Date {r['date']}: keys={list(r.keys())}, profile={'profile' in r}")

    # Save results
    prefix = f"{ticker}_0700_1000_2"
    paths = {}

    # 1. Save VPIN data with pre-summary
    try:
        vpin_frames = []
        for r in all_results:
            if 'vpin' in r and not r['vpin'].empty:
                vpin = r['vpin'].copy()
                vpin['date'] = r['date'].date()
                vpin_frames.append(vpin)

        if vpin_frames:
            vpin_df = pd.concat(vpin_frames, axis=0)
            vpin_path = output_dir / f"{prefix}_vpin.parquet"
            vpin_df.to_parquet(vpin_path)
            paths['vpin'] = str(vpin_path)
            logger.info(f"Saved VPIN: {vpin_path}")
    except Exception as e:
        logger.error(f"Failed to save VPIN: {e}")

    # Helper function to get size for both torch tensors and numpy arrays
    def _get_size(arr):
        """Get size for both torch tensors and numpy arrays."""
        if arr is None:
            return 0
        if hasattr(arr, 'numel'):
            return arr.numel()
        if hasattr(arr, 'size'):
            return arr.size
        return 0

    # 2. Save Number Bars as NPZ with date-based keys
    try:
        valid_bars = [(r['date'], r['number_bars'], r['number_bars_meta'])
                      for r in all_results if 'number_bars' in r and _get_size(r['number_bars']) > 0]
        if valid_bars:
            # Use date strings as keys (e.g., '2024-01-15')
            nb_path = output_dir / f"{prefix}_number_bars.npz"
            # Convert torch tensors to numpy before saving
            np.savez_compressed(
                nb_path,
                **{v[0].strftime('%Y-%m-%d'): (v[1].cpu().numpy() if hasattr(v[1], 'cpu') else v[1])
                   for v in valid_bars}
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
    except Exception as e:
        logger.error(f"Failed to save number bars: {e}")

    # 3. Save Rasterized VPIN sequences as NPZ with date-based keys
    try:
        valid_raster = [(r['date'], r['rasterized'])
                        for r in all_results if 'rasterized' in r and _get_size(r['rasterized']) > 0]
        if valid_raster:
            # Use date strings as keys (e.g., '2024-01-15')
            raster_path = output_dir / f"{prefix}_rasterized_vpin.npz"
            # Convert torch tensors to numpy before saving
            np.savez_compressed(
                raster_path,
                **{v[0].strftime('%Y-%m-%d'): (v[1].cpu().numpy() if hasattr(v[1], 'cpu') else v[1])
                   for v in valid_raster}
            )
            paths['rasterized'] = str(raster_path)
            logger.info(f"Saved rasterized VPIN: {raster_path}")
            first_raster = valid_raster[0][1].cpu().numpy() if hasattr(valid_raster[0][1], 'cpu') else valid_raster[0][1]
            logger.info(f"  Shape per date: {first_raster.shape} (num_bars, channels, bins)")
            logger.info(f"  Channels: [Density, LogVolume, Imbalance, Returns]")
    except Exception as e:
        logger.error(f"Failed to save rasterized VPIN: {e}")

    # 4. Save Profile summaries
    try:
        profile_summaries = []
        for r in all_results:
            if 'profile' in r and not r['profile'].empty and 'TotalVolume' in r['profile'].columns:
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

        logger.info(f"Created {len(profile_summaries)} profile summaries")

        if profile_summaries:
            summary_df = pd.DataFrame(profile_summaries).set_index('date')
            summary_path = output_dir / f"{prefix}_profile_summary.parquet"
            summary_df.to_parquet(summary_path)
            paths['profile_summary'] = str(summary_path)
            logger.info(f"Saved profile summary: {summary_path}")
        else:
            logger.warning(f"No profile summaries to save!")
    except Exception as e:
        logger.error(f"Failed to save profile summary: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 5. Save encoded profiles using VolumeProfileEncoder
    try:
        from CTAFlow.features.volume.profile_encoder import (
            VolumeProfileEncoder, VolumeProfileEncoderConfig, save_profiles_npz
        )

        valid_profiles = [(r['date'], r['profile']) for r in all_results
                          if 'profile' in r and not r['profile'].empty]

        logger.info(f"Found {len(valid_profiles)} valid profiles out of {len(all_results)} results")

        if valid_profiles:
            profiles = [v[1] for v in valid_profiles]
            profile_dates = [v[0] for v in valid_profiles]

            # Fit encoder on first 80% of data (at least 1 sample)
            n_train = max(1, int(len(profiles) * 0.8))
            logger.info(f"Fitting encoder on {n_train} profiles, encoding {len(profiles)} total")

            enc_cfg = VolumeProfileEncoderConfig(num_bins=96, include_imbalance=True)
            encoder = VolumeProfileEncoder(enc_cfg)
            encoder.fit(profiles[:n_train])

            # Transform all profiles
            encoded = encoder.transform_many(profiles)
            logger.info(f"Encoded profiles shape: {encoded.shape}")

            profile_path = output_dir / f"{prefix}_profiles.npz"
            save_profiles_npz(profile_path, profile_dates, encoded)
            paths['profiles'] = str(profile_path)
            logger.info(f"Saved encoded profiles: {profile_path}")
        else:
            logger.warning(f"No valid profiles found to save!")
            logger.warning(f"Profile check details:")
            for i, r in enumerate(all_results[:5]):  # Check first 5
                has_profile = 'profile' in r
                is_empty = r['profile'].empty if has_profile else True
                logger.warning(f"  Result {i}: has_profile={has_profile}, empty={is_empty}")
    except Exception as e:
        logger.error(f"Failed to save encoded profiles: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return paths


def _process_single_ticker(ticker: str, dates: List[pd.Timestamp], output_base: str,
                          bucket_cache: Optional[str] = None, save_bucket_cache: bool = True) -> tuple:
    """
    Process a single ticker (used for parallel processing).

    Args:
        ticker: Contract symbol
        dates: List of dates to process
        output_base: Base output directory
        bucket_cache: Path to bucket cache CSV (optional)
        save_bucket_cache: If True, save bucket cache after extraction

    Returns:
        Tuple of (ticker, paths_dict)
    """
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {ticker}")
        logger.info(f"{'='*50}")

        output_dir = Path(output_base) / ticker.lower()

        paths = extract_session_features(
            ticker=ticker,
            dates=dates,
            output_dir=str(output_dir),
            verbose=True,




        )

        logger.info(f"\nSaved files for {ticker}:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")

        return (ticker, paths)
    except Exception as e:
        logger.error(f"Failed to process {ticker}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return (ticker, {})


def run_session_extraction(
    tickers: List[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-31",
    output_base: str = "F:/Upload",
    n_jobs: int = 1,
    bucket_cache: Optional[str] = None,
    save_bucket_cache: bool = True
):
    """
    Run session feature extraction for specified tickers.

    Args:
        tickers: List of tickers (default: ['RB', 'CL', 'HO'])
        start_date: Start date string
        end_date: End date string
        output_base: Base output directory
        n_jobs: Number of parallel workers (default: 1 for sequential)
                Set to -1 to use all CPUs, or specify number of workers
        bucket_cache: Path to bucket cache CSV (optional, auto-uses ticker-specific cache if available)
        save_bucket_cache: If True, save bucket cache after extraction (default: True)
    """
    tickers = tickers or ['CL', 'RB', 'HO', 'GC', 'NG']

    # Generate business days
    dates = pd.date_range(start_date, end_date, freq='B').tolist()
    logger.info(f"Processing {len(dates)} business days from {start_date} to {end_date}")
    logger.info(f"Processing {len(tickers)} tickers with {n_jobs} worker(s)")

    all_paths = {}

    if n_jobs == 1:
        # Sequential processing
        for ticker in tickers:
            ticker_result, paths = _process_single_ticker(ticker, dates, output_base)
            all_paths[ticker_result] = paths
    else:
        # Parallel processing
        from joblib import Parallel, delayed

        logger.info(f"Starting parallel processing with {n_jobs} workers...")

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_single_ticker)(ticker, dates, output_base)
            for ticker in tickers
        )

        # Collect results
        for ticker, paths in results:
            all_paths[ticker] = paths

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXTRACTION COMPLETE - SUMMARY")
    logger.info("="*60)
    for ticker, paths in all_paths.items():
        logger.info(f"\n{ticker}:")
        if paths:
            for key, path in paths.items():
                logger.info(f"  ✓ {key}: {path}")
        else:
            logger.info(f"  ✗ Failed or no data saved")
    logger.info("="*60)

    return all_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract daily session features")
    parser.add_argument("--tickers", nargs="+", default=["GC", "CL", "RB", "HO", "NG"],
                        help="Tickers to process")
    parser.add_argument("--start", default="2011-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-20",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="F:/Upload",
                        help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers (1=sequential, -1=all CPUs)")

    args = parser.parse_args()

    run_session_extraction(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        output_base=args.output,
    )
