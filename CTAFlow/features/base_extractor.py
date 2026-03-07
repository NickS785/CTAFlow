import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Union
import logging

# Adjust imports based on your project structure
from sierrapy.parser.scid_parse import FastScidReader, ScidTickerFileManager, ScidContractInfo
from ..data.contract_expiry_rules import calculate_expiry, get_roll_buffer_days

logger = logging.getLogger(__name__)

# Default cache location for contract maps
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ctaflow"


class SmartScidManager(ScidTickerFileManager):
    """
    Extends SierraPy's file manager to use CTAFlow's precise expiry rules.

    Supports caching the contract map to disk to avoid slow filesystem
    rescans on every instantiation.  Use ``save_contract_cache()`` after the
    first run, then pass ``cache_path`` on subsequent runs.
    """

    def __init__(self, folder: str, *, service: str = "sierra",
                 cache_path: Optional[Union[str, Path]] = None):
        if cache_path is not None and Path(cache_path).exists():
            # Fast path: load cached contract map instead of scanning
            self.folder = Path(folder)
            self.service = service.lower()
            self._ticker_files, self._ticker_contracts = {}, {}
            self._load_contract_cache(cache_path)
            logger.info(f"Loaded contract cache from {cache_path} "
                        f"({sum(len(v) for v in self._ticker_contracts.values())} contracts)")
        else:
            # Normal path: scan filesystem (slow)
            super().__init__(folder, service=service)

    def _calculate_contract_expiry(self, contract: ScidContractInfo) -> pd.Timestamp:
        try:
            month_map = {
                'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
            }
            month_int = month_map.get(contract.month.upper())
            ticker_fmt = f"{contract.ticker}_F"
            expiry = calculate_expiry(ticker_fmt, contract.year, month_int)
            # End of day buffer
            return expiry + pd.Timedelta(hours=23, minutes=59, seconds=59)
        except Exception as e:
            logger.warning(f"CTAFlow expiry calc failed for {contract.ticker}: {e}")
            return super()._calculate_contract_expiry(contract)

    def save_contract_cache(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the contract map to a pickle file for fast reloading."""
        if path is None:
            _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = _DEFAULT_CACHE_DIR / f"contract_map_{self.folder.name}.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "folder": str(self.folder),
            "service": self.service,
            "ticker_files": {k: [str(p) for p in v] for k, v in self._ticker_files.items()},
            "ticker_contracts": {
                k: [(c.ticker, c.month, c.year, c.exchange, str(c.file_path), c.service)
                    for c in v]
                for k, v in self._ticker_contracts.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved contract cache to {path} "
                    f"({sum(len(v) for v in self._ticker_contracts.values())} contracts)")
        return path

    def _load_contract_cache(self, path: Union[str, Path]) -> None:
        """Load contract map from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._ticker_files = {
            k: [Path(p) for p in v] for k, v in data["ticker_files"].items()
        }
        self._ticker_contracts = {
            k: [ScidContractInfo(ticker=t[0], month=t[1], year=t[2],
                                 exchange=t[3], file_path=Path(t[4]), service=t[5])
                for t in v]
            for k, v in data["ticker_contracts"].items()
        }


class ScidBaseExtractor:
    """
    Base class for extracting continuous, stitched data from SCID files.
    Handles timezone conversion, contract rolling, and file reading.
    """

    def __init__(self, data_dir: str, ticker: Optional[str] = None, tz: str = "America/Chicago",
                 contract_cache: Optional[Union[str, Path]] = None):
        self.manager = SmartScidManager(data_dir, cache_path=contract_cache)
        self.ticker = ticker
        self.tz = tz

    def _normalize_time(self, t: Union[str, pd.Timestamp]) -> pd.Timestamp:
        """Converts input time (assumed self.tz if naive) to UTC Timestamp."""
        ts = pd.Timestamp(t)
        if ts.tz is None:
            ts = ts.tz_localize(self.tz)
        return ts.tz_convert('UTC')

    def _to_time_string(self, t: Union[str, pd.Timestamp, datetime, date]) -> str:
        """Helper to standardize time inputs for downstream methods."""
        if isinstance(t, str): return t
        if isinstance(t, pd.Timestamp):
            return t.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S") if t.tz else t.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(t, (datetime, date)):
            return t.strftime("%Y-%m-%d %H:%M:%S")
        raise TypeError(f"Cannot convert {type(t)} to time string")

    def get_stitched_data(self,
                          start_time: Union[str, pd.Timestamp],
                          end_time: Union[str, pd.Timestamp],
                          columns: List[str] = ["Close", "TotalVolume", "BidVolume", "AskVolume"],
                          ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Core method: Extracts and stitches data across contract expiries.
        """
        target_ticker = ticker or self.ticker
        if not target_ticker:
            raise ValueError("No ticker provided via init or method call.")

        start_utc = self._normalize_time(start_time)
        end_utc = self._normalize_time(end_time)

        contracts = self.manager.get_contracts_for_ticker(target_ticker)
        if not contracts:
            raise ValueError(f"No .scid files found for {target_ticker}")

        # Determine Roll Buffer
        try:
            roll_days = get_roll_buffer_days(f"{target_ticker}_F")
        except:
            roll_days = 0
        buffer = pd.Timedelta(days=roll_days)

        # Build Schedule
        schedule = []
        for i, contract in enumerate(contracts):
            expiry_naive = self.manager._calculate_contract_expiry(contract)
            expiry_utc = expiry_naive.tz_localize('UTC')
            roll_date = expiry_utc - buffer

            if i == 0:
                period_start = pd.Timestamp.min.replace(tzinfo=roll_date.tzinfo)
            else:
                prev_expiry = self.manager._calculate_contract_expiry(contracts[i - 1])
                period_start = prev_expiry.tz_localize('UTC') - buffer

            period_end = roll_date

            if period_end < start_utc: continue
            if period_start > end_utc: break

            schedule.append({
                'contract': contract,
                'start': period_start,
                'end': period_end,
                'path': contract.file_path
            })

        # Extract Data
        chunks = []
        for item in schedule:
            req_start = max(start_utc, item['start'])
            req_end = min(end_utc, item['end'])

            if req_start < req_end:
                logger.info(f"Reading {item['contract'].contract_id} [{req_start} -> {req_end} UTC]")
                start_ms = int(req_start.timestamp() * 1000)
                end_ms = int(req_end.timestamp() * 1000)

                try:
                    with FastScidReader(str(item['path'])) as reader:
                        df = reader.to_pandas(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            columns=columns,
                            drop_invalid_rows=True
                        )
                        if not df.empty:
                            chunks.append(df)
                except Exception as e:
                    logger.error(f"Error reading {item['path']}: {e}")

        if not chunks:
            return pd.DataFrame()

        result = pd.concat(chunks)
        result = result[~result.index.duplicated(keep='last')]
        result.sort_index(inplace=True)

        if self.tz:
            result.index = result.index.tz_convert(self.tz)

        return result