"""
Generate continuous forward DataFrames for HE and LE using SpreadEngine.
Patches gaps using SCID data from DLY_DATA_PATH.
"""

import sys
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\CTAFlow")

import pandas as pd
from pathlib import Path
from CTAFlow.data.multi_expiry import SpreadEngine
from CTAFlow.config import DLY_DATA_PATH

# Configuration
TICKERS = ["HE", "LE"]
DATA_DIR_TEMPLATE = r"F:\Data\intraday\{ticker}\monthly"
OUTPUT_DIR = r"F:\Upload"
START_DATE = "2012-01-01"
END_DATE = "2025-12-31"
FREQ = "5min"
MAX_MONTHS = 6
EXCHANGE = "CME"  # Livestock futures are on CME


def process_ticker(ticker: str):
    """Process a single ticker to generate continuous forward DataFrame."""
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print(f"{'='*60}")

    data_dir = DATA_DIR_TEMPLATE.format(ticker=ticker)
    output_path = Path(OUTPUT_DIR) / ticker.lower() / f"{ticker}_continuous_fwd.parquet"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"SCID directory: {DLY_DATA_PATH}")
    print(f"Output path: {output_path}")

    # Initialize SpreadEngine
    print(f"\nInitializing SpreadEngine...")
    engine = SpreadEngine(data_dir, ticker)
    print(f"Indexed {len(engine.lazy_pool)} contract slices")

    # Generate continuous forward prices
    print(f"\nGenerating sequential prices ({START_DATE} to {END_DATE}, freq={FREQ})...")
    chunks = []
    for i, chunk in enumerate(engine.get_sequential_prices(
        START_DATE,
        END_DATE,
        freq=FREQ,
        max_months=MAX_MONTHS,
        return_expiries=False
    )):
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} periods, {sum(len(c) for c in chunks)} total rows...")

    if not chunks:
        print(f"ERROR: No data generated for {ticker}")
        return None

    # Concatenate all chunks
    df = pd.concat(chunks).sort_index()
    print(f"\nRaw DataFrame: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")

    # Count gaps before patching
    gaps_before = df.isna().sum()
    print(f"\nGaps before patching:")
    for col in df.columns:
        if gaps_before[col] > 0:
            print(f"  {col}: {gaps_before[col]} ({gaps_before[col]/len(df)*100:.1f}%)")

    # Build contract map for accurate gap patching
    print(f"\nBuilding contract map...")
    contract_map = engine.build_contract_map(
        str(df.index.min().date()),
        str(df.index.max().date()),
        max_months=MAX_MONTHS,
        freq='D'
    )

    # Patch gaps using SCID files
    print(f"\nPatching gaps from SCID files ({DLY_DATA_PATH})...")
    df_patched = engine.patch_gaps_from_scid(
        df,
        scid_directory=str(DLY_DATA_PATH),
        exchange=EXCHANGE,
        resample_rule=FREQ,
        target_tz='America/Chicago',
        contract_map=contract_map
    )

    # Count gaps after patching
    gaps_after = df_patched.isna().sum()
    print(f"\nGaps after patching:")
    for col in df_patched.columns:
        if gaps_after[col] > 0:
            print(f"  {col}: {gaps_after[col]} ({gaps_after[col]/len(df_patched)*100:.1f}%)")

    # Save to parquet
    print(f"\nSaving to {output_path}...")
    df_patched.to_parquet(output_path)
    print(f"Saved {len(df_patched)} rows")

    # Also save contract map for reference
    contract_map_path = output_path.parent / f"{ticker}_contract_map.csv"
    contract_map.to_csv(contract_map_path)
    print(f"Saved contract map to {contract_map_path}")

    return df_patched


def main():
    results = {}
    for ticker in TICKERS:
        try:
            df = process_ticker(ticker)
            if df is not None:
                results[ticker] = df
                print(f"\n{ticker} SUCCESS: {df.shape}")
        except Exception as e:
            print(f"\n{ticker} ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ticker, df in results.items():
        print(f"{ticker}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Total gaps: {df.isna().sum().sum()}")


if __name__ == "__main__":
    main()
