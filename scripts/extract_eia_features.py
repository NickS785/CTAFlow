"""
EIA Release Feature Extraction Script

Extracts VPIN, Number Bars, and Volume Profile features around EIA Weekly
Petroleum Status Report releases for energy tickers (RB, CL, HO).

EIA Weekly Release Schedule:
- Standard: Wednesdays 10:30 AM ET (9:30 AM CT)
- Holiday-adjusted: Thursdays, various times (typically 12:00 PM ET)

Features extracted:
- VPIN: 1h before event, 2h during/after
- Number Bars: 15-minute intervals for the event period
- Profile: 2AM to event start
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

from CTAFlow.features.tick_extractor import (
    FeatureExtractorConfig,
    MultiFeatureExtraction,
    get_default_tick_size
)
from CTAFlow.features.volume.profile import NumberBarsExtractor, MarketProfileExtractor
from CTAFlow.features.volume.vpin import VPINExtractor
from CTAFlow.features.base_extractor import ScidBaseExtractor
from CTAFlow.config import DLY_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EIAReleaseEvent:
    """Represents an EIA Weekly Petroleum Status Report release."""
    date: pd.Timestamp
    release_time_ct: str  # Central Time
    is_holiday_adjusted: bool = False


def get_eia_weekly_releases(start_year: int = 2020, end_year: int = 2025) -> List[EIAReleaseEvent]:
    """
    Generate EIA Weekly Petroleum Status Report release dates.

    Standard releases: Wednesdays at 9:30 AM CT (10:30 AM ET)
    Holiday releases: Thursdays, typically later in the day

    Args:
        start_year: Start year for date range
        end_year: End year for date range

    Returns:
        List of EIAReleaseEvent objects
    """
    releases = []

    # Generate all Wednesdays in the date range
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")

    # Get all Wednesdays (weekday=2)
    wednesdays = pd.date_range(start, end, freq='W-WED')

    # US Federal Holidays that cause Thursday releases
    # (simplified - actual schedule varies by year)
    holiday_weeks = set()
    for year in range(start_year, end_year + 1):
        # MLK Day (3rd Monday of January)
        mlk = pd.Timestamp(f"{year}-01-01") + pd.DateOffset(weekday=0) + pd.DateOffset(weeks=2)
        holiday_weeks.add(mlk.isocalendar()[1])

        # Presidents Day (3rd Monday of February)
        pres = pd.Timestamp(f"{year}-02-01") + pd.DateOffset(weekday=0) + pd.DateOffset(weeks=2)
        holiday_weeks.add(pres.isocalendar()[1])

        # Memorial Day (last Monday of May)
        memorial = pd.Timestamp(f"{year}-05-31") - pd.DateOffset(weekday=0)
        holiday_weeks.add(memorial.isocalendar()[1])

        # July 4th week
        july4 = pd.Timestamp(f"{year}-07-04")
        holiday_weeks.add(july4.isocalendar()[1])

        # Labor Day (1st Monday of September)
        labor = pd.Timestamp(f"{year}-09-01") + pd.DateOffset(weekday=0)
        holiday_weeks.add(labor.isocalendar()[1])

        # Thanksgiving (4th Thursday of November)
        thanksgiving = pd.Timestamp(f"{year}-11-01") + pd.DateOffset(weekday=3) + pd.DateOffset(weeks=3)
        holiday_weeks.add(thanksgiving.isocalendar()[1])

        # Christmas week
        christmas = pd.Timestamp(f"{year}-12-25")
        holiday_weeks.add(christmas.isocalendar()[1])

    for wed in wednesdays:
        week_num = wed.isocalendar()[1]
        year = wed.year

        # Check if this is a holiday week (release on Thursday)
        if week_num in holiday_weeks:
            thursday = wed + pd.Timedelta(days=1)
            releases.append(EIAReleaseEvent(
                date=thursday,
                release_time_ct="11:00",  # Typically noon ET = 11 CT
                is_holiday_adjusted=True
            ))
        else:
            releases.append(EIAReleaseEvent(
                date=wed,
                release_time_ct="09:30",  # 10:30 ET = 9:30 CT
                is_holiday_adjusted=False
            ))

    return releases


class EIAFeatureExtractor:
    """
    Extracts features around EIA release events for energy tickers.

    Time windows (all in Central Time):
    - Pre-event VPIN: 1 hour before release (e.g., 8:30-9:30 for standard)
    - Event VPIN: 2 hours during/after release (e.g., 9:30-11:30 for standard)
    - Number Bars: 15-minute intervals during event window
    - Profile: 2:00 AM to event start
    """

    def __init__(
        self,
        ticker: str,
        data_dir: str = DLY_DATA_PATH,
        output_dir: Optional[str] = None,
        tz: str = "America/Chicago"
    ):
        self.ticker = ticker.upper()
        self.data_dir = data_dir
        self.output_dir = Path(output_dir) if output_dir else Path("F:/Upload") / ticker.lower()
        self.tz = tz

        # Get default tick size from contract specs
        self.tick_size = get_default_tick_size(self.ticker) or 0.0001

        # Initialize extractors
        self.base_extractor = ScidBaseExtractor(data_dir, ticker, tz)
        self.vpin_extractor = VPINExtractor(data_dir, ticker, tz)
        self.profile_extractor = MarketProfileExtractor(data_dir, ticker, tz, tick_size=self.tick_size)
        self.number_bars_extractor = NumberBarsExtractor(
            data_dir, ticker, tz,
            tick_size=self.tick_size,
            interval="15min",
            num_levels=100,
            centering_method="rolling_vwap"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_for_event(self, event: EIAReleaseEvent) -> Dict:
        """
        Extract all features for a single EIA release event.

        Args:
            event: EIAReleaseEvent with date and release time

        Returns:
            Dict with 'pre_vpin', 'event_vpin', 'number_bars', 'number_bars_meta', 'profile'
        """
        date_str = event.date.strftime('%Y-%m-%d')
        release_time = pd.Timestamp(f"{date_str} {event.release_time_ct}")

        # Define time windows
        pre_vpin_start = release_time - pd.Timedelta(hours=1)
        pre_vpin_end = release_time
        event_vpin_start = release_time
        event_vpin_end = release_time + pd.Timedelta(hours=2)
        profile_start = pd.Timestamp(f"{date_str} 02:00")
        profile_end = release_time

        results = {
            'date': event.date,
            'release_time': release_time,
            'is_holiday_adjusted': event.is_holiday_adjusted,
            'pre_vpin': pd.DataFrame(),
            'event_vpin': pd.DataFrame(),
            'number_bars': np.array([]),
            'number_bars_meta': pd.DataFrame(),
            'profile': pd.DataFrame(),
            'profile_summary': {}
        }

        try:
            # Fetch all raw data needed
            df_raw = self.base_extractor.get_stitched_data(
                start_time=profile_start,
                end_time=event_vpin_end,
                columns=["Close", "TotalVolume", "BidVolume", "AskVolume"]
            )
        except Exception as e:
            logger.warning(f"Failed to fetch data for {date_str}: {e}")
            return results

        if df_raw.empty:
            logger.warning(f"No data for {date_str}")
            return results

        # 1. Profile extraction (2AM to release time)
        try:
            profile_data = df_raw.between_time(profile_start.time(), profile_end.time(), inclusive='left')
            if not profile_data.empty:
                profile = self.profile_extractor.calculate_volume_profile(
                    profile_data, tick_size=self.tick_size
                )
                results['profile'] = profile

                # Calculate summary stats
                if not profile.empty and 'TotalVolume' in profile.columns:
                    total_vol = profile['TotalVolume'].sum()
                    poc = profile['TotalVolume'].idxmax()
                    sorted_profile = profile.sort_values('TotalVolume', ascending=False)
                    cumvol = sorted_profile['TotalVolume'].cumsum()
                    va_levels = sorted_profile[cumvol <= total_vol * 0.7].index

                    results['profile_summary'] = {
                        'poc': poc,
                        'val': va_levels.min() if len(va_levels) > 0 else poc,
                        'vah': va_levels.max() if len(va_levels) > 0 else poc,
                        'total_volume': total_vol,
                        'vwap': (profile_data['Close'] * profile_data['TotalVolume']).sum() / total_vol if total_vol > 0 else np.nan
                    }
        except Exception as e:
            logger.warning(f"Profile extraction failed for {date_str}: {e}")

        # 2. Pre-event VPIN (1h before release)
        try:
            pre_data = df_raw.between_time(pre_vpin_start.time(), pre_vpin_end.time(), inclusive='left')
            if not pre_data.empty:
                pre_vpin = self.vpin_extractor.calculate_vpin(
                    pre_data, bucket_volume=50, window=20, include_sequence_features=True
                )
                # Add profile summary to VPIN
                for key, value in results['profile_summary'].items():
                    pre_vpin[key] = value
                results['pre_vpin'] = pre_vpin
        except Exception as e:
            logger.warning(f"Pre-VPIN extraction failed for {date_str}: {e}")

        # 3. Event VPIN (2h during/after release)
        try:
            event_data = df_raw.between_time(event_vpin_start.time(), event_vpin_end.time(), inclusive='both')
            if not event_data.empty:
                event_vpin = self.vpin_extractor.calculate_vpin(
                    event_data, bucket_volume=50, window=20, include_sequence_features=True
                )
                for key, value in results['profile_summary'].items():
                    event_vpin[key] = value
                results['event_vpin'] = event_vpin
        except Exception as e:
            logger.warning(f"Event VPIN extraction failed for {date_str}: {e}")

        # 4. Number Bars (15-minute intervals during event window)
        # Use calculate_number_bars with pre-fetched data to avoid redundant extraction
        try:
            # Filter data to event window (with VWAP lookback for centering)
            vwap_lookback = pd.Timedelta("1h") * 2
            nb_data = df_raw.between_time(
                (event_vpin_start - vwap_lookback).time(),
                event_vpin_end.time(),
                inclusive='both'
            )
            if not nb_data.empty:
                tensor, meta = self.number_bars_extractor.calculate_number_bars(
                    df=nb_data,
                    start_time=event_vpin_start,
                    interval="15min",
                    tick_size=self.tick_size,
                    vwap_window="1h"
                )
                results['number_bars'] = tensor
                results['number_bars_meta'] = meta
        except Exception as e:
            logger.warning(f"Number bars extraction failed for {date_str}: {e}")

        return results

    def extract_all_events(
        self,
        events: List[EIAReleaseEvent],
        verbose: bool = True
    ) -> List[Dict]:
        """Extract features for all events."""
        all_results = []

        for i, event in enumerate(events):
            if verbose:
                logger.info(f"Processing {i+1}/{len(events)}: {event.date.date()}")

            result = self.extract_for_event(event)
            all_results.append(result)

        return all_results

    def save_results(self, results: List[Dict], prefix: str = None) -> Dict[str, str]:
        """
        Save extraction results to files.

        Args:
            results: List of extraction results
            prefix: File prefix (default: {ticker}_EIA_release)

        Returns:
            Dict with saved file paths
        """
        prefix = prefix or f"{self.ticker}_EIA_release"
        paths = {}

        # Collect VPIN data
        pre_vpin_frames = []
        event_vpin_frames = []

        for r in results:
            if not r['pre_vpin'].empty:
                vpin = r['pre_vpin'].copy()
                vpin['date'] = r['date'].date()
                vpin['release_time'] = r['release_time']
                pre_vpin_frames.append(vpin)

            if not r['event_vpin'].empty:
                vpin = r['event_vpin'].copy()
                vpin['date'] = r['date'].date()
                vpin['release_time'] = r['release_time']
                event_vpin_frames.append(vpin)

        # Save pre-event VPIN
        if pre_vpin_frames:
            pre_df = pd.concat(pre_vpin_frames, axis=0)
            pre_path = self.output_dir / f"{prefix}_pre_vpin.parquet"
            pre_df.to_parquet(pre_path)
            paths['pre_vpin'] = str(pre_path)
            logger.info(f"Saved pre-VPIN: {pre_path}")

        # Save event VPIN
        if event_vpin_frames:
            event_df = pd.concat(event_vpin_frames, axis=0)
            event_path = self.output_dir / f"{prefix}_event_vpin.parquet"
            event_df.to_parquet(event_path)
            paths['event_vpin'] = str(event_path)
            logger.info(f"Saved event VPIN: {event_path}")

        # Save Number Bars as NPZ with date-based keys
        valid_bars = [(r['date'], r['number_bars'], r['number_bars_meta'])
                      for r in results if r['number_bars'].size > 0]
        if valid_bars:
            # Use date strings as keys (e.g., '2024-01-15')
            nb_path = self.output_dir / f"{prefix}_number_bars.npz"
            np.savez_compressed(
                nb_path,
                **{v[0].strftime('%Y-%m-%d'): v[1] for v in valid_bars}
            )
            paths['number_bars'] = str(nb_path)
            logger.info(f"Saved number bars: {nb_path}")

        # Save profile summaries
        summaries = [{'date': r['date'], **r['profile_summary']}
                     for r in results if r['profile_summary']]
        if summaries:
            summary_df = pd.DataFrame(summaries).set_index('date')
            summary_path = self.output_dir / f"{prefix}_profile_summary.parquet"
            summary_df.to_parquet(summary_path)
            paths['profile_summary'] = str(summary_path)
            logger.info(f"Saved profile summary: {summary_path}")

        return paths


def run_eia_extraction(
    tickers: List[str] = None,
    start_year: int = 2020,
    end_year: int = 2025,
    output_base: str = "F:/Upload"
):
    """
    Run EIA feature extraction for specified tickers.

    Args:
        tickers: List of tickers (default: ['RB', 'CL', 'HO'])
        start_year: Start year for EIA releases
        end_year: End year for EIA releases
        output_base: Base output directory
    """
    tickers = tickers or ['RB', 'CL', 'HO']

    # Get EIA release events
    events = get_eia_weekly_releases(start_year, end_year)
    logger.info(f"Found {len(events)} EIA release events from {start_year} to {end_year}")

    for ticker in tickers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {ticker}")
        logger.info(f"{'='*50}")

        output_dir = Path(output_base) / ticker.lower()
        extractor = EIAFeatureExtractor(ticker, output_dir=str(output_dir))

        # Extract features
        results = extractor.extract_all_events(events)

        # Save results
        paths = extractor.save_results(results)

        logger.info(f"Saved files for {ticker}:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features around EIA releases")
    parser.add_argument("--tickers", nargs="+", default=["RB", "CL", "HO"],
                        help="Tickers to process")
    parser.add_argument("--start-year", type=int, default=2012,
                        help="Start year")
    parser.add_argument("--end-year", type=int, default=2026,
                        help="End year")
    parser.add_argument("--output", default="F:/Upload",
                        help="Output directory")

    args = parser.parse_args()

    run_eia_extraction(
        tickers=args.tickers,
        start_year=args.start_year,
        end_year=args.end_year,
        output_base=args.output
    )
