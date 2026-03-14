"""
Extract EIA Natural Gas Storage event features for NG.

Output: F:/Upload/s3/model_data/NG/ng_release_*
  - ng_release_stats.csv
  - ng_release_orderflow/<date>_<code>_<side>.parquet
  - ng_release_surprise.csv (if surprise data provided)
"""
import logging
from datetime import date

from CTAFlow.features.event_extractor import (
    EventExtractor,
    EventExtractorConfig,
    generate_event_dates_from_presets,
)
from CTAFlow.screeners.event_presets import EIA_NG_STORAGE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# --- Config ---
config = EventExtractorConfig(
    ticker="NG",
    data_dir="F:/SierraChart/Data",
    tz="America/Chicago",
    pre_minutes=60,
    post_minutes=60,
    bucket_volume=250,
    vpin_window=20,
)

# --- Generate synthetic event dates from preset rules ---
# EIA NG Storage: every Thursday (weekday=3), 9:30 AM CT
event_dates = generate_event_dates_from_presets(
    events=[EIA_NG_STORAGE],
    start_date=date(2011, 1, 1),
    end_date=date(2025, 12, 31),
)
print(f"Generated {len(event_dates['EIA_NG_STORAGE'])} candidate dates for EIA_NG_STORAGE")

# --- Extract ---
extractor = EventExtractor(
    config=config,
    events=[EIA_NG_STORAGE],
    event_dates=event_dates,
)

results = extractor.extract_all(verbose=True)

# --- Save ---
written = extractor.save_results(
    results,
    output_dir="F:/Upload/s3/model_data/NG",
    prefix="ng_release",
)

print(f"\nDone. {len(written)} files written:")
for f in written:
    print(f"  {f}")

# Quick summary
stats = results["event_stats"]
if stats is not None and not stats.empty:
    print(f"\nStats shape: {stats.shape}")
    print(stats.describe().round(3))
