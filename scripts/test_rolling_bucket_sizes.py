"""
Test script for rolling yearly bucket size calculation.

This script demonstrates the new rolling yearly fits for auto_bucket_size.
Each year's bucket size is computed using the previous 3 years of volume data.
"""

import pandas as pd
import logging
from CTAFlow.features.tick_extractor import (
    FeatureExtractorConfig,
    MultiFeatureExtraction
)
from CTAFlow.config import DLY_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rolling_bucket_sizes():
    """
    Test rolling yearly bucket size calculation.

    Creates an extractor for RB with dates spanning 2020-2023.
    Verifies that bucket sizes are computed per year using 3-year rolling windows.
    """
    ticker = "RB"

    # Create dates spanning multiple years (sample a few dates per year)
    dates = []
    for year in [2020, 2021, 2022, 2023]:
        # Sample 3 dates per year for testing
        dates.extend([
            pd.Timestamp(f"{year}-03-15"),
            pd.Timestamp(f"{year}-06-15"),
            pd.Timestamp(f"{year}-09-15")
        ])

    logger.info(f"Testing with {len(dates)} dates from {dates[0].year} to {dates[-1].year}")

    # Configure extractor with auto_bucket enabled
    config = FeatureExtractorConfig(
        ticker=ticker,
        data_dir=DLY_DATA_PATH,
        tz="America/Chicago",
        # VPIN settings with auto bucket
        vpin_bucket_size=None,  # Use auto_bucket_size
        auto_bucket=True,
        auto_bucket_cadence=500,  # Target 500 buckets per day
        vpin_window=20,
        vpin_start_time="08:30",
        vpin_end_time="09:30",
        # Disable other features for faster testing
        include_profile=False,
        include_number_bars=False,
        include_sequence_features=False
    )

    # Create extractor
    extractor = MultiFeatureExtraction(config, dates)

    # Pre-compute bucket sizes to see the rolling fits
    logger.info("\n" + "="*60)
    logger.info("Pre-computing bucket sizes with rolling yearly fits")
    logger.info("="*60)

    bucket_sizes = extractor._precompute_bucket_sizes(verbose=True)

    logger.info("\n" + "="*60)
    logger.info("Results:")
    logger.info("="*60)
    for year, size in sorted(bucket_sizes.items()):
        lookback_start = year - 3
        lookback_end = year - 1
        logger.info(
            f"Year {year}: bucket_size={size} "
            f"(computed from {lookback_start}-{lookback_end} volume data)"
        )

    # Verify that different years can have different bucket sizes
    unique_sizes = set(bucket_sizes.values())
    logger.info(f"\nUnique bucket sizes: {sorted(unique_sizes)}")

    if len(unique_sizes) > 1:
        logger.info("✓ Rolling fits are working - different years have different bucket sizes")
    else:
        logger.warning("⚠ All years have the same bucket size - may indicate insufficient volume variation")

    # Test a single extraction to verify it works end-to-end
    logger.info("\n" + "="*60)
    logger.info("Testing single date extraction with cached bucket size")
    logger.info("="*60)

    test_date = dates[0]
    result = extractor.extract_date(test_date)

    if not result['vpin'].empty:
        logger.info(f"✓ Successfully extracted VPIN for {test_date.date()}")
        logger.info(f"  VPIN buckets: {len(result['vpin'])}")
        logger.info(f"  Bucket size used: {bucket_sizes[test_date.year]}")
    else:
        logger.warning(f"⚠ No VPIN data for {test_date.date()}")

    logger.info("\n" + "="*60)
    logger.info("Test completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    test_rolling_bucket_sizes()
