import pandas as pd
import numpy as np
import os
import re
import logging
from typing import List, Dict, Optional
from . import  read_exported_df
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("FuturesBuilder")

# Map letter codes to integer months
MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}


class ContractSlice:
    """
    Represents a specific vintage (e.g., Jan 2024) extracted from a multi-year file.
    """

    def __init__(self, ticker: str, month_code: str, delivery_year: int, data: pd.DataFrame):
        self.ticker = ticker
        self.month_code = month_code
        self.month_int = MONTH_MAP[month_code]
        self.delivery_year = delivery_year
        self.data = data
        self.expiry_date = self._estimate_expiry()

    def _estimate_expiry(self) -> pd.Timestamp:
        # Simplistic expiry: 3rd Friday of the delivery month
        try:
            first_day = pd.Timestamp(year=self.delivery_year, month=self.month_int, day=1)
            # 0=Mon, 4=Fri. Find offset to first Friday.
            days_to_fri = (4 - first_day.dayofweek + 7) % 7
            # 3rd Friday = 1st Friday + 14 days
            expiry = first_day + pd.Timedelta(days=days_to_fri + 14)
            return expiry
        except ValueError:
            # Handle potential calendar errors
            return pd.Timestamp.max

    def __repr__(self):
        return f"{self.ticker}-{self.month_code}{self.delivery_year}"


class MonthlyFileLoader:
    """
    Reads 12 CSVs (one per month letter) and splits them into individual contract years.
    """

    def __init__(self, data_dir: str, ticker: str):
        self.data_dir = data_dir
        self.ticker = ticker
        self.pool: List[ContractSlice] = []

    def load_all(self):
        """Scans folder for {ticker}_{Code}.csv and parses them."""
        for code, month_int in MONTH_MAP.items():
            # Adjust filename pattern to match your data
            # Assuming format: "CL_F.csv" or "CL-F.csv"
            filename = f"{self.ticker}_{code}.csv"
            filepath = os.path.join(self.data_dir, filename)

            if not os.path.exists(filepath):
                # Try hyphen format
                filepath = os.path.join(self.data_dir, f"{self.ticker}-{code}.csv")
                if not os.path.exists(filepath):
                    continue

            logger.info(f"Processing {filename}...")
            self._process_file(filepath, code, month_int)

        # Sort pool by delivery date
        self.pool.sort(key=lambda x: (x.delivery_year, x.month_int))
        logger.info(f"Total extracted contracts: {len(self.pool)}")

    def _process_file(self, filepath: str, month_code: str, month_int: int):
        try:
            df = read_exported_df(filepath)

            # Standardize Date Index
            if 'datetime' in df.columns:
                df['ts'] = pd.to_datetime(df['datetime'])
            elif 'Date' in df.columns and 'Time' in df.columns:
                df['ts'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            else:
                # Fallback: look for first column with 'date' in name
                date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                if date_col:
                    df['ts'] = pd.to_datetime(df[date_col])
                else:
                    return

            df = df.set_index('ts').sort_index()

            # Ensure Clean Prices
            cols_map = {c: c.lower() for c in df.columns}
            df = df.rename(columns=cols_map)
            # Keep only OHLC
            df = df[['close']].copy()  # We mainly need Close for spreads

            # --- SPLIT LOGIC ---
            # We calculate 'Delivery Year' for every row
            # Logic: If trading in Dec (12) for Jan (1) contract, year is +1
            # Using vectorized operations for speed

            years = df.index.year.values
            months = df.index.month.values

            # Default: Delivery Year = Calendar Year
            delivery_years = years.copy()

            # Adjustment: If Trade Month > Contract Month, add 1 to year
            # Example: Trading F (Jan/1) in Dec (12) -> 12 > 1 -> Add 1 year
            mask_next_year = months > month_int
            delivery_years[mask_next_year] += 1

            df['delivery_year'] = delivery_years

            # Group by computed delivery year to create ContractSlice objects
            for year, group_df in df.groupby('delivery_year'):
                # Heuristic: Filter out tiny fragments (bad data)
                if len(group_df) > 50:
                    contract = ContractSlice(self.ticker, month_code, year, group_df[['close']])
                    self.pool.append(contract)

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")


class SpreadEngine:
    def __init__(self, contract_pool: List[ContractSlice]):
        self.pool = contract_pool

    def get_active_chain(self, date: pd.Timestamp) -> List[ContractSlice]:
        """Returns M1, M2, M3, M4 active contracts for a given date."""
        # Filter for contracts that expire AFTER current date
        # Note: We buffer expiry slightly (e.g., roll a few days before actual expiry)
        # to ensure we don't hold into delivery.
        active = [c for c in self.pool if c.expiry_date > date]
        return active[:4]

    def run(self, start_date: str, end_date: str, freq: str = '1h') -> pd.DataFrame:
        timeline = pd.date_range(start_date, end_date, freq=freq)
        results = []

        logger.info("Generating spreads...")

        # Optimization: We iterate by DAY, determine contracts, then reindex data
        # Iterating tick-by-tick is too slow.

        daily_timeline = pd.date_range(start_date, end_date, freq='D')

        for day in daily_timeline:
            chain = self.get_active_chain(day)
            if len(chain) < 4:
                continue

            # Get M1..M4 data for THIS day
            day_slice_start = day
            day_slice_end = day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

            daily_data = {}
            valid_day = True

            for i, contract in enumerate(chain):
                label = f"M{i + 1}"
                # Slice the dataframe for exactly this day
                # We use the pre-loaded data in contract.data
                try:
                    # Slicing in pandas can be slow if index is not sorted (we sorted in loader)
                    # We utilize the fact that contract.data is sorted
                    subset = contract.data.loc[day_slice_start:day_slice_end]

                    if subset.empty:
                        # If M1 is missing data for a whole day, the day is invalid
                        valid_day = False
                        break

                    # Resample to desired frequency (e.g., Hourly)
                    resampled = subset['close'].resample(freq).last().ffill()
                    daily_data[label] = resampled
                except KeyError:
                    valid_day = False
                    break

            if valid_day:
                # Align columns
                df_day = pd.DataFrame(daily_data)
                results.append(df_day)

        if not results:
            return pd.DataFrame()

        full_df = pd.concat(results)

        # --- CALCULATE FEATURES ---
        # 1. Front Spread (M1 - M2)
        full_df['Spread_Front'] = full_df['M1'] - full_df['M2']

        # 2. Back Spread (M3 - M4)
        full_df['Spread_Back'] = full_df['M2'] - full_df['M3']

        # 3. Butterfly / Curvature (Front Spread - Back Spread)
        # This shows if the curve is steepening at the front relative to back
        full_df['Butterfly_Spread'] = full_df['Spread_Front'] - full_df['Spread_Back']

        return full_df.dropna()


# --- EXECUTION ---
if __name__ == "__main__":
    # Setup
    ticker = "CL"
    data_folder = f"F:\\monthly_contracts\\{ticker}"  # Folder containing CL_F.csv, CL_G.csv...


    # 1. Load and Slice
    loader = MonthlyFileLoader(data_folder, ticker)
    loader.load_all()

    # 2. Process Spreads
    engine = SpreadEngine(loader.pool)

    # Generate for specific window
    df_features = engine.run(
        start_date="2009-01-01",
        end_date="2023-12-31",
        freq="30min"  # Change to '5min' or '1D' as needed
    )

    if not df_features.empty:
        print(df_features.head())
        print(df_features.describe())
        df_features.to_csv("CL_TermStructure_Features.csv")
    else:
        print("No overlapping data found for the term structure.")