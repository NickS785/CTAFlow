#!/usr/bin/env python
"""
Regenerate the population-weighted weather HDF cache from NCEI GHCND data.

Uses census-epoch grids (2020, 2024) so population weighting updates every
4 years. Fetches from the earliest intraday data date through today.

Usage (RunPod):
    python scripts/regenerate_weather_cache.py \
        --intraday /workspace/model_data/NG/intraday_2.csv \
        --output   /workspace/model_data/new_weather.hdf \
        --config-dir /workspace/results/ng_hybrid_intraday/weather_configs

Requires NCEI_TOKEN in environment (via .env or export).
"""
import argparse
import os
import sys
from datetime import date
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Regenerate weather HDF cache")
    parser.add_argument("--intraday", required=True, help="Path to intraday CSV")
    parser.add_argument("--output", required=True, help="Output HDF path")
    parser.add_argument("--config-dir", default=None,
                        help="Dir to cache grid configs (avoids re-running station search)")
    parser.add_argument("--env-file", default=None,
                        help="Path to .env file with NCEI_TOKEN")
    parser.add_argument("--start", default=None,
                        help="Override start date (YYYY-MM-DD); default: first bar in CSV")
    parser.add_argument("--end", default=None,
                        help="Override end date (YYYY-MM-DD); default: last bar in CSV")
    args = parser.parse_args()

    # Load environment
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)

    token = os.getenv("NCEI_TOKEN", "")
    if not token:
        print("ERROR: NCEI_TOKEN not set. Provide via --env-file or environment.")
        sys.exit(1)
    print(f"NCEI_TOKEN: {'*' * (len(token) - 4)}{token[-4:]}")

    import pandas as pd
    from MacrOSINT.models.energy.natgas_storage_forecast import NatGasStorageForecaster
    from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df

    # Determine date range from intraday CSV
    print(f"Reading intraday CSV: {args.intraday}")
    raw = read_exported_df(args.intraday)
    start_dt = pd.Timestamp(args.start).date() if args.start else raw.index[0].date()
    end_dt = pd.Timestamp(args.end).date() if args.end else raw.index[-1].date()
    print(f"Date range: {start_dt} to {end_dt}")

    # Fetch weather by epoch
    config_dir = args.config_dir
    if config_dir:
        Path(config_dir).mkdir(parents=True, exist_ok=True)

    forecaster = NatGasStorageForecaster(
        ncei_token=token,
        config_dir=config_dir,
    )

    print(f"\nFetching weather (epoch-aware)...")
    daily_weather = forecaster._fetch_weather_by_epoch(start_dt, end_dt)
    print(f"Fetched: {daily_weather.shape} rows, "
          f"{daily_weather.index[0].date()} to {daily_weather.index[-1].date()}")
    print(f"Columns: {daily_weather.columns.tolist()}")

    # Save
    output_path = Path(args.output)
    # Remove old file to avoid appending to stale data
    if output_path.exists():
        output_path.unlink()
        print(f"Removed old cache: {output_path}")

    NatGasStorageForecaster.save_weather_hdf(daily_weather, hdf_path=str(output_path))
    print(f"\nDone. Saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
