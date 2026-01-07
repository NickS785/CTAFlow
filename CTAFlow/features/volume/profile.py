import pandas as pd
import numpy as np
from datetime import datetime, date, time
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
import logging
from dataclasses import dataclass
# Import from your provided files
from sierrapy.parser.scid_parse import FastScidReader, ScidTickerFileManager, ScidContractInfo
from ...data.contract_expiry_rules import calculate_expiry, get_roll_buffer_days
from ..base_extractor import ScidBaseExtractor
logger = logging.getLogger(__name__)

@dataclass
class MarketProfileConfig:
    data_dir: str
    ticker: str
    tz : str = "America/Chicago"
    tick_size = 0.01




class MarketProfileExtractor(ScidBaseExtractor):
    def __init__(self, data_dir: str, ticker: Optional[str] = None, tz: str = "America/Chicago", tick_size=None):
        """
        Initialize MarketProfileExtractor.

        Args:
            data_dir: Path to SCID data directory
            ticker: Default ticker symbol (can be overridden per-call)
            tz: Timezone for time interpretation (default: America/Chicago)
        """
        super().__init__(data_dir, ticker, tz)
        self.tick_size = tick_size if tick_size else None
        return

    def calculate_volume_profile(self, df: pd.DataFrame, tick_size: Optional[float] = None) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        data = df.copy()

        # Convert volume columns to float64 to prevent uint32 overflow on subtraction
        for col in ['TotalVolume', 'BidVolume', 'AskVolume']:
            if col in data.columns:
                data[col] = data[col].astype(np.float64)

        # Binning logic
        if tick_size is not None and tick_size > 0:
            data['PriceBin'] = (np.round(data['Close'] / tick_size) * tick_size).astype(float)
            data['PriceBin'] = data['PriceBin'].round(decimals=6)
        else:
            data['PriceBin'] = data['Close']

        agg_dict = {'TotalVolume': 'sum'}
        for col in ['BidVolume', 'AskVolume', 'NumTrades']:
            if col in data.columns:
                agg_dict[col] = 'sum'

        profile = data.groupby('PriceBin').agg(agg_dict)
        profile.index.name = 'Price'
        profile.sort_index(ascending=True, inplace=True)

        if 'BidVolume' in profile.columns and 'AskVolume' in profile.columns:
            # Ensure float64 after aggregation to prevent overflow
            profile['Delta'] = profile['AskVolume'].astype(np.float64) - profile['BidVolume'].astype(np.float64)

        return profile

    def get_profile(self, start_time: str, end_time: str, tick_size: float = None) -> pd.DataFrame:
        """
        Main entry point for getting a profile over a period.
        """
        # Fetch underlying data
        df_raw = self.get_stitched_data(start_time, end_time)

        # Transform to profile
        return self.calculate_volume_profile(df_raw, tick_size=tick_size)

    def __getitem__(self, item: slice):
        """
        Slice syntax: extractor['2023-01-01':'2023-01-02':0.01]
        """
        if not isinstance(item, slice):
            raise TypeError("Expected slice object")

        start_str = self._to_time_string(item.start)
        end_str = self._to_time_string(item.stop)
        tick_size = float(item.step) if isinstance(item.step, (int, float)) else None

        return self.get_profile(start_str, end_str, tick_size=tick_size)