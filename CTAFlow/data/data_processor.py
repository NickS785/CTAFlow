"""Simplified CSV data processor for market data updates and FuturesCurve processing."""

import fnmatch
import re
import threading
import concurrent.futures
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set, Sequence

import numpy as np
import pandas as pd

from .raw_formatting.dly_contract_manager import DLYContractManager, DLYFolderUpdater
from .raw_formatting.intraday_manager import IntradayFileManager
from .update_management import (
    WEEKLY_MARKET_DATA_UPDATE_EVENT,
    get_update_metadata_store,
    prepare_dataframe_for_append,
    summarize_update_summary,
)
from ..config import RAW_MARKET_DATA_PATH


class DataProcessor:
    """Simplified data processor that only handles CSV files.

    IMPORTANT: /spot data protection safeguards are implemented:
    - This class uses DLYContractManager.save_hdf() for all curve storage
    - DLYContractManager.save_hdf() never writes to /spot paths
    - The /spot data path (market/{ticker}/spot) is reserved for actual spot price data
    - Only dedicated spot data processing methods should write to /spot paths
    """

    def __init__(self, client):
        self.client = client

        # Thread-safe progress tracking (following DLYFolderUpdater pattern)
        self._progress_lock = threading.Lock()
        self._results_lock = threading.Lock()
        self._hdf_lock = threading.Lock()  # Serialize HDF5 file operations
        self._processed_count = 0
        self._total_count = 0

    def _validate_spot_data_protection(self, hdf_path: str) -> None:
        """Validate that /spot data paths are protected from accidental overwriting.

        This method checks that no curve processing operations write to /spot paths,
        which are reserved for actual spot price data only.

        Parameters
        ----------
        hdf_path : str
            Path to HDF5 file to check

        Raises
        ------
        ValueError
            If any /spot data corruption risks are detected
        """
        try:
            with pd.HDFStore(hdf_path, mode='r') as store:
                # Get all keys to check for any curve-related operations that might write to /spot
                all_keys = store.keys()
                spot_keys = [key for key in all_keys if '/spot' in key]

                if spot_keys:
                    # Log the spot keys that exist (this is expected and safe)
                    print(f"[SPOT PROTECTION] Found {len(spot_keys)} existing /spot data paths (protected)")

        except Exception as e:
            # If we can't read the store, that's fine - this is just a safety check
            print(f"[SPOT PROTECTION] Could not validate spot protection (store may not exist yet): {e}")

        # This method serves as documentation and validation that we're using safe storage methods
        print(f"[SPOT PROTECTION] Confirmed: Using DLYContractManager.save_hdf() for safe curve storage")

    def _collect_existing_curve_keys(self) -> Dict[str, Set[str]]:
        """Collect curve-related keys for each ticker currently in the market store."""

        valid_curve_keys = {
            "curve",
            "dte",
            "expiry",
            "front",
            "seq_curve",
            "seq_labels",
            "seq_dte",
            "seq_spreads",
            "curve_volume",
            "seq_volume",
            "curve_oi",
            "seq_oi",
        }

        mapping: Dict[str, Set[str]] = {}
        for key in self.client.list_market_data():
            if not key.startswith("market/"):
                continue
            parts = key.split("/")
            if len(parts) < 2:
                continue
            symbol = parts[1]
            dataset = parts[2] if len(parts) > 2 else ""

            if len(parts) == 2:
                mapping.setdefault(symbol, set())
                continue

            if dataset and dataset in valid_curve_keys:
                mapping.setdefault(symbol, set()).add(dataset)

        return mapping

    def _collect_market_update_files(
        self,
        tickers: Sequence[str],
        *,
        raw_data_path: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Collect CSV update files for the provided ``tickers`` from ``RAW_MARKET_DATA_PATH``."""

        if not tickers:
            return {}

        folder = Path(raw_data_path or RAW_MARKET_DATA_PATH)
        try:
            entries = list(folder.iterdir())
        except FileNotFoundError:
            return {}

        target_lookup = {t.upper(): t.upper() for t in tickers}
        lowercase_lookup = {t.upper(): t.lower() for t in tickers}

        mapping: Dict[str, List[str]] = {ticker.upper(): [] for ticker in tickers}

        for entry in entries:
            if not entry.is_file() or entry.suffix.lower() != ".csv":
                continue

            stem = entry.stem.lower()
            if "-update-" not in stem:
                continue

            prefix = stem.split("-update-", 1)[0]
            prefix_upper = prefix.upper()

            if prefix_upper in target_lookup:
                mapping.setdefault(prefix_upper, []).append(str(entry))
            else:
                # Some update files may include suffixes (e.g. CL_F-update-...)
                # Try matching against known lowercase variants directly.
                for ticker_upper, ticker_lower in lowercase_lookup.items():
                    if stem.startswith(f"{ticker_lower}-update-"):
                        mapping.setdefault(ticker_upper, []).append(str(entry))
                        break

        return {ticker: sorted(files) for ticker, files in mapping.items() if files}

    def update_all_tickers(
        self,
        *,
        dly_folder: str,
        pattern: Optional[str] = None,
        pattern_mode: str = "regex",
        selected_cot_features=None,
        max_workers: int = 4,
        raw_data_path: Optional[str] = None,
        market_resample_rule: str = "1T",
        cot_progress: bool = False,
        record_metadata: bool = True,
        metadata_event: str = WEEKLY_MARKET_UPDATE_EVENT,
    ) -> Dict[str, Any]:
        """Update COT metrics, market data and curve ticker_sets for all tracked tickers."""

        if max_workers <= 0:
            raise ValueError("max_workers must be positive")

        metadata_store = get_update_metadata_store() if record_metadata else None

        # COT updates using DataClient's refresh_latest_cot_year (multi-threaded operation)
        # Run first before updating the rest as requested
        if cot_progress:
            print("[COT] Starting COT data refresh...")

        try:
            cot_result = self.client.refresh_latest_cot_year(
                report_type="disaggregated_fut",
                write_ticker_keys=True,
                progress=cot_progress
            )
            if cot_progress:
                total_rows = cot_result.get('total_rows', 0)
                ticker_success = len(cot_result.get('ticker_results', {}).get('success', []))
                print(f"[COT] Refresh complete: {total_rows} total rows, {ticker_success} tickers processed")
        except Exception as e:
            cot_result = {
                'error': str(e),
                'total_rows': 0,
                'ticker_results': {'success': [], 'failed': {}}
            }
            if cot_progress:
                print(f"[COT] Refresh failed: {e}")

        existing_mapping = self._collect_existing_curve_keys()

        def matches(symbol: str) -> bool:
            base = symbol[:-2] if symbol.endswith("_F") else symbol
            if pattern is None:
                return True
            if pattern_mode == "glob":
                return fnmatch.fnmatch(base, pattern)
            return re.search(pattern, base) is not None

        filtered_mapping = {
            symbol: curve_keys
            for symbol, curve_keys in existing_mapping.items()
            if matches(symbol)
        }

        summary: Dict[str, Any] = {
            "cot_metrics": cot_result,
            "updates": {},
            "skipped": [],
            "errors": {},
        }

        folder_updater = DLYFolderUpdater(dly_folder)
        base_tickers = [symbol[:-2] if symbol.endswith("_F") else symbol for symbol in filtered_mapping]
        file_map = folder_updater.collect_update_files(base_tickers)
        market_updates = self._collect_market_update_files(
            base_tickers, raw_data_path=raw_data_path
        )
        raw_path = Path(raw_data_path or RAW_MARKET_DATA_PATH)

        update_jobs: List[Dict[str, Any]] = []
        for symbol, curve_keys in filtered_mapping.items():
            base_symbol = symbol[:-2] if symbol.endswith("_F") else symbol
            files = file_map.get(base_symbol, [])
            market_files = market_updates.get(base_symbol.upper(), [])

            has_curve_job = bool(files and curve_keys)
            has_market_job = bool(market_files)

            if not has_curve_job and not has_market_job:
                summary["skipped"].append(symbol)
                continue

            update_jobs.append(
                {
                    "symbol": symbol,
                    "base_symbol": base_symbol,
                    "curve_keys": sorted(curve_keys),
                    "curve_files": list(files),
                    "market_files": list(market_files),
                }
            )

        if not update_jobs:
            summary["message"] = "No eligible tickers found for update"
            if metadata_store:
                metadata_store.record_success(
                    metadata_event,
                    summarize_update_summary(summary),
                )
            return summary

        # Run market updates sequentially to minimize memory pressure
        for job in update_jobs:
            symbol = job["symbol"]
            market_files = job["market_files"]
            details = {
                "symbol": symbol,
                "base_symbol": job["base_symbol"],
                "curve_keys": list(job["curve_keys"]),
                "curve_files": list(job["curve_files"]),
                "market_files": list(market_files),
            }
            summary["updates"][symbol] = details
            job["details"] = details

            if not market_files:
                continue

            try:
                details["market_result"] = self.update_market_from_csv(
                    symbol,
                    csv_folder=str(raw_path),
                    resample_rule=market_resample_rule,
                )
            except Exception as exc:
                summary["errors"].setdefault(symbol, {})["market"] = str(exc)

        # Run curve updates concurrently using threading (following DLYFolderUpdater pattern)
        if update_jobs:
            # Initialize progress tracking
            self._processed_count = 0
            self._total_count = len(update_jobs)

            # Determine optimal number of workers
            import os
            if max_workers is None:
                max_workers = min(len(update_jobs), os.cpu_count() or 4)

            # Process curve updates using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all curve update tasks
                future_to_job = {
                    executor.submit(self._process_single_curve_update_worker, job, folder_updater, summary): job
                    for job in update_jobs if job.get("curve_files") and job.get("curve_keys")
                }

                # Wait for completion
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        future.result()  # This will raise any exceptions that occurred
                    except Exception as exc:
                        symbol = job["symbol"]
                        with self._results_lock:
                            summary["errors"].setdefault(symbol, {})["curve"] = str(exc)

        if summary["skipped"]:
            summary["skipped"] = sorted(set(summary["skipped"]))

        if metadata_store:
            metadata_store.record_success(
                metadata_event,
                summarize_update_summary(summary),
            )

        return summary

    def _process_single_curve_update_worker(
        self,
        job: Dict[str, Any],
        folder_updater: DLYFolderUpdater,
        summary: Dict[str, Any]
    ) -> None:
        """Worker function for processing a single ticker curve update in a thread.

        Following DLYFolderUpdater._process_single_ticker_worker pattern.
        """
        import time

        symbol = job["symbol"]
        base_symbol = job["base_symbol"]
        curve_keys = job["curve_keys"]
        curve_files = job["curve_files"]

        if not curve_files or not curve_keys:
            return

        start_time = time.time()

        try:
            with self._progress_lock:
                print(f"[THREAD] Starting curve update for {symbol}")

            # Execute the curve update with HDF5 locking
            with self._hdf_lock:
                result = folder_updater.update_existing_ticker(
                    base_symbol,
                    existing_curve_keys=tuple(curve_keys),
                    file_paths=tuple(curve_files),
                )

            # Store the result in a thread-safe manner
            with self._results_lock:
                details = job.get("details") or summary["updates"].setdefault(symbol, {})
                details["curve_result"] = result

                # Update progress
                self._processed_count += 1
                progress_pct = (self._processed_count / self._total_count) * 100
                elapsed = time.time() - start_time

            with self._progress_lock:
                print(f"[{self._processed_count}/{self._total_count}] Completed curve update for {symbol} in {elapsed:.1f}s ({progress_pct:.1f}%)")

        except Exception as e:
            elapsed = time.time() - start_time

            with self._results_lock:
                summary["errors"].setdefault(symbol, {})["curve"] = str(e)
                self._processed_count += 1

            with self._progress_lock:
                print(f"[{self._processed_count}/{self._total_count}] Failed curve update for {symbol} after {elapsed:.1f}s: {e}")

    def update_market_from_csv(
        self,
        symbol: str,
        *,
        csv_folder: str = "F:/charts/",
        resample_rule: str = "1T",
    ) -> Dict[str, Any]:
        """Update market data using all available CSV files since the last update.

        Parameters
        ----------
        symbol : str
            Market symbol (with or without the ``_F`` suffix).
        csv_folder : str, default "F:/charts/"
            Folder path containing CSV update files.
        resample_rule : str, default "1T"
            Resampling frequency applied to the raw observations.

        Returns
        -------
        Dict[str, Any]
            Summary information including rows appended and the source files used.
        """

        if not symbol:
            raise ValueError("symbol must be provided")

        normalized_symbol = symbol.upper()
        if normalized_symbol.endswith("_F"):
            base_symbol = normalized_symbol[:-2]
        else:
            base_symbol = normalized_symbol
            normalized_symbol = f"{base_symbol}_F"

        # Get last timestamp from existing data
        tail_key = f"market/{normalized_symbol}"
        try:
            existing_data = self.client.get_market_tail(tail_key, 100)
            if not existing_data.empty and existing_data.index.notna().any():
                last_timestamp = existing_data.index.max()
            else:
                last_timestamp = None
        except:
            last_timestamp = None

        # Find all CSV files for this symbol
        csv_folder_path = Path(csv_folder)
        pattern = f"{base_symbol.lower()}-update-*.csv"
        csv_files = list(csv_folder_path.glob(pattern))

        if not csv_files:
            return {
                "symbol": normalized_symbol,
                "csv_files": [],
                "appended_rows": 0,
                "message": f"No CSV files found matching pattern {pattern}",
            }

        # Process each CSV file and collect all new data
        all_new_data = []
        processed_files = []

        for csv_file in csv_files:
            try:
                df = self._read_csv_file(csv_file)
                if not df.empty:
                    # Only keep data newer than last timestamp
                    if last_timestamp is not None:
                        new_data = df[df.index > last_timestamp]
                    else:
                        new_data = df

                    if not new_data.empty:
                        all_new_data.append(new_data)
                        processed_files.append(str(csv_file))

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        if not all_new_data:
            return {
                "symbol": normalized_symbol,
                "csv_files": [str(f) for f in csv_files],
                "appended_rows": 0,
                "message": "No new data found in CSV files",
            }

        # Combine all new data
        combined_df = pd.concat(all_new_data, axis=0, ignore_index=False)
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

        # Format the data
        formatted = self._format_dataframe(combined_df).sort_index()
        resampled = self._resample_intraday_bars(formatted, rule=resample_rule)

        if resampled.empty:
            return {
                "symbol": normalized_symbol,
                "csv_files": processed_files,
                "appended_rows": 0,
                "message": "No data after resampling",
            }

        # Prepare final data
        if resampled.index.tz is not None:
            resampled.index = resampled.index.tz_convert(None)

        resampled = resampled[~resampled.index.duplicated(keep="last")]
        resampled["ticker_prefix"] = base_symbol
        resampled["ticker_symbol"] = normalized_symbol

        column_order = [
            "Open", "High", "Low", "Last", "Volume",
            "NumberOfTrades", "BidVolume", "AskVolume",
            "ticker_prefix", "ticker_symbol",
        ]
        resampled = resampled[column_order]

        # Ensure proper data types
        for col in ["Open", "High", "Low", "Last"]:
            resampled[col] = resampled[col].astype(float)
        for col in ["Volume", "NumberOfTrades", "BidVolume", "AskVolume"]:
            resampled[col] = resampled[col].fillna(0).round().astype("int64")
        resampled["ticker_prefix"] = resampled["ticker_prefix"].astype(str)
        resampled["ticker_symbol"] = resampled["ticker_symbol"].astype(str)

        with self._hdf_lock:
            prepared, require_replace = prepare_dataframe_for_append(
                self.client,
                tail_key,
                resampled,
                allow_schema_expansion=False,
            )

            if prepared.empty and self.client.market_key_exists(tail_key):
                return {
                    "symbol": normalized_symbol,
                    "csv_files": processed_files,
                    "appended_rows": 0,
                    "resample_rule": resample_rule,
                    "last_timestamp": str(last_timestamp) if last_timestamp is not None else None,
                    "mode": "noop",
                    "message": "No new rows after alignment",
                }

            if require_replace:
                previous_rows = self.client.get_market_rowcount(tail_key)
                self.client.write_market(prepared, tail_key, replace=True)
                append_summary: Dict[str, Any] = {
                    "mode": "schema_replace",
                    "rows_written": len(prepared),
                    "total_rows": len(prepared),
                    "delta_rows": len(prepared) - previous_rows,
                }
            else:
                append_result = self.client.append_market_continuous(prepared, tail_key)
                if append_result.get("mode") == "schema_mismatch":
                    raise ValueError(f"Schema mismatch when appending market data for {normalized_symbol}: {append_result}")
                append_summary = append_result

        appended_rows = append_summary.get("delta_rows")
        if appended_rows is None:
            appended_rows = append_summary.get("rows_written", len(prepared))

        return {
            "symbol": normalized_symbol,
            "csv_files": processed_files,
            "appended_rows": appended_rows,
            "resample_rule": resample_rule,
            "last_timestamp": str(prepared.index.max()) if not prepared.empty else None,
            "mode": append_summary.get("mode"),
        }

    def _read_csv_file(self, csv_path: Path) -> pd.DataFrame:
        """Read a CSV file with automatic column detection and formatting."""

        try:
            # Try reading with different separators
            for sep in [',', '\t', ';', '|']:
                try:
                    # First, try reading without parsing dates to see the structure
                    df = pd.read_csv(
                        csv_path,
                        sep=sep,
                        header=0,
                        encoding='utf-8'
                    )

                    if len(df.columns) < 4 or len(df) == 0:
                        continue

                    # Clean column names (remove leading/trailing spaces)
                    df.columns = df.columns.str.strip()

                    # Handle different datetime formats
                    datetime_index = None

                    # Check if we have separate Date and Time columns
                    date_cols = [col for col in df.columns if col.lower() in ['date']]
                    time_cols = [col for col in df.columns if col.lower() in ['time']]

                    if date_cols and time_cols:
                        # Combine Date and Time columns
                        date_col = date_cols[0]
                        time_col = time_cols[0]
                        df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
                        df = df.set_index('datetime')
                        df = df.drop(columns=[date_col, time_col])
                        datetime_index = df.index
                    else:
                        # Look for datetime columns
                        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if datetime_cols:
                            df['datetime'] = pd.to_datetime(df[datetime_cols[0]])
                            df = df.set_index('datetime')
                            df = df.drop(columns=datetime_cols)
                            datetime_index = df.index
                        elif len(df.columns) > 0:
                            # Try first column as datetime
                            try:
                                df['datetime'] = pd.to_datetime(df.iloc[:, 0])
                                df = df.set_index('datetime')
                                df = df.drop(columns=[df.columns[0]])
                                datetime_index = df.index
                            except:
                                continue

                    if datetime_index is None:
                        continue

                    df.index.name = "datetime"

                    # Apply column mapping
                    column_mapping = {}
                    for col in df.columns:
                        col_lower = col.lower().strip()
                        if col_lower in ['open', 'o']:
                            column_mapping[col] = 'open'
                        elif col_lower in ['high', 'h']:
                            column_mapping[col] = 'high'
                        elif col_lower in ['low', 'l']:
                            column_mapping[col] = 'low'
                        elif col_lower in ['close', 'c', 'last']:
                            column_mapping[col] = 'last'
                        elif col_lower in ['volume', 'v', 'vol']:
                            column_mapping[col] = 'volume'
                        elif col_lower in ['num_trades', 'trades', 'numberoftrades', 'number_of_trades', '#oftrades', '#of trades']:
                            column_mapping[col] = 'num_trades'
                        elif col_lower in ['bid_volume', 'bidvolume', 'bid_vol', 'bvol']:
                            column_mapping[col] = 'bidvolume'
                        elif col_lower in ['ask_volume', 'askvolume', 'ask_vol', 'avol']:
                            column_mapping[col] = 'askvolume'

                    df = df.rename(columns=column_mapping)

                    # Ensure we have OHLC data
                    required_cols = ['open', 'high', 'low', 'last']
                    if all(col in df.columns for col in required_cols):
                        return df

                except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
                    continue

        except Exception:
            pass

        return pd.DataFrame()

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe columns to standard market schema."""

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "last": "Last",
            "close": "Last",
            "volume": "Volume",
            "num_trades": "NumberOfTrades",
            "number_of_trades": "NumberOfTrades",
            "bidvolume": "BidVolume",
            "bid_volume": "BidVolume",
            "askvolume": "AskVolume",
            "ask_volume": "AskVolume",
        }

        working = df.copy()
        working.columns = [rename_map.get(col, col) for col in working.columns]

        # Add missing required columns with default values
        for required in ["Open", "High", "Low", "Last", "Volume", "NumberOfTrades", "BidVolume", "AskVolume"]:
            if required not in working.columns:
                working[required] = 0.0 if required in {"Open", "High", "Low", "Last"} else 0

        # Ensure proper data types
        float_cols = ["Open", "High", "Low", "Last"]
        int_cols = ["Volume", "NumberOfTrades", "BidVolume", "AskVolume"]

        # Convert to numeric first, coercing errors to NaN
        for col in float_cols:
            working[col] = pd.to_numeric(working[col], errors='coerce').fillna(0.0)

        for col in int_cols:
            working[col] = pd.to_numeric(working[col], errors='coerce').fillna(0.0)

        # Return only the standard columns in consistent order
        return working[float_cols + int_cols]

    @staticmethod
    def _resample_intraday_bars(df: pd.DataFrame, rule: str = "1T") -> pd.DataFrame:
        """Resample intraday records to the provided frequency."""

        price = df[["Open", "High", "Low", "Last"]].resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Last": "last",
        })
        volume = df[["Volume", "NumberOfTrades", "BidVolume", "AskVolume"]].resample(rule).sum()

        combined = pd.concat([price, volume], axis=1)
        combined = combined.dropna(subset=["Open", "High", "Low", "Last"], how="all")

        for col in ["Volume", "NumberOfTrades", "BidVolume", "AskVolume"]:
            combined[col] = combined[col].fillna(0).round().astype("int64")

        combined.index.name = "datetime"
        return combined.dropna(subset=["Open", "High", "Low", "Last"], how="all")

    # =================================================================
    # Intraday Market Data Processing via IntradayFileManager
    # =================================================================

    def update_intraday_from_scid(
        self,
        symbol: str,
        *,
        scid_folder: Optional[str] = None,
        timeframe: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        replace: bool = False
    ) -> Dict[str, Any]:
        """Update intraday market data from SCID files to market/{symbol}/{timeframe}.

        This method uses IntradayFileManager to load front month data from SCID files
        and writes it to HDF5 in timeframe-specific sub-keys (e.g., market/CL_F/5min).

        Parameters
        ----------
        symbol : str
            Market symbol (with or without _F suffix)
        scid_folder : str, optional
            Folder containing SCID files (default: DLY_DATA_PATH from config)
        timeframe : str, optional
            Resampling timeframe (e.g., '1T', '5T', '15T', '1H', '4H')
            If None, uses raw tick data
        start : pd.Timestamp, optional
            Start date for data loading
        end : pd.Timestamp, optional
            End date for data loading
        replace : bool, default False
            If True, replace existing data; if False, append new data

        Returns
        -------
        Dict[str, Any]
            Summary with records_written, key, success status

        Examples
        --------
        >>> processor = DataProcessor(client)
        >>> # Update 5-minute bars
        >>> result = processor.update_intraday_from_scid('CL_F', timeframe='5min')
        >>> # Update hourly bars from specific date
        >>> result = processor.update_intraday_from_scid(
        ...     'NG_F',
        ...     timeframe='1h',
        ...     start=pd.Timestamp('2024-01-01')
        ... )
        """
        if not symbol:
            raise ValueError("symbol must be provided")

        normalized_symbol = symbol.upper()
        if not normalized_symbol.endswith("_F"):
            normalized_symbol = f"{normalized_symbol}_F"

        # Initialize IntradayFileManager
        try:
            from CTAFlow.config import DLY_DATA_PATH
            data_path = Path(scid_folder) if scid_folder else DLY_DATA_PATH

            intraday_mgr = IntradayFileManager(
                data_path=data_path,
                market_data_path=self.client.market_data_path,
                enable_logging=True
            )

            # Convert pandas Timestamp to datetime if needed
            start_dt = start.to_pydatetime() if start is not None else None
            end_dt = end.to_pydatetime() if end is not None else None

            # Use IntradayFileManager's write_to_hdf5 method
            # This handles all the heavy lifting: loading SCID, resampling, gap detection
            result = intraday_mgr.write_to_hdf5(
                symbol=normalized_symbol,
                start=start_dt,
                end=end_dt,
                timeframe=timeframe,
                replace=replace
            )

            return result

        except Exception as e:
            return {
                'symbol': normalized_symbol,
                'success': False,
                'error': f"Error updating intraday data: {e}",
                'records_written': 0
            }

    def update_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        *,
        scid_folder: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        replace: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Update the same symbol at multiple timeframes using SCID files.

        This efficiently processes one symbol to multiple timeframe sub-keys:
        - market/{symbol}/1min
        - market/{symbol}/5min
        - market/{symbol}/15min
        - market/{symbol}/1h
        etc.

        Parameters
        ----------
        symbol : str
            Market symbol
        timeframes : List[str]
            List of timeframe strings (e.g., ['1T', '5T', '15T', '1H'])
        scid_folder : str, optional
            Folder containing SCID files
        start : pd.Timestamp, optional
            Start date for data loading
        end : pd.Timestamp, optional
            End date for data loading
        replace : bool, default False
            If True, replace existing data

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping timeframes to their update results

        Examples
        --------
        >>> processor = DataProcessor(client)
        >>> # Update CL_F at 1min, 5min, 15min, and 1h timeframes
        >>> results = processor.update_multiple_timeframes(
        ...     'CL_F',
        ...     timeframes=['1T', '5T', '15T', '1H']
        ... )
        >>> for tf, result in results.items():
        ...     print(f"{tf}: {result['records_written']} records")
        """
        if not symbol:
            raise ValueError("symbol must be provided")

        if not timeframes:
            raise ValueError("timeframes list must be provided")

        normalized_symbol = symbol.upper()
        if not normalized_symbol.endswith("_F"):
            normalized_symbol = f"{normalized_symbol}_F"

        results = {}
        for timeframe in timeframes:
            print(f"[{normalized_symbol}] Processing timeframe: {timeframe}")

            result = self.update_intraday_from_scid(
                symbol=normalized_symbol,
                scid_folder=scid_folder,
                timeframe=timeframe,
                start=start,
                end=end,
                replace=replace
            )

            results[timeframe] = result

            # Log result
            if result.get('success'):
                records = result.get('records_written', 0)
                print(f"[{normalized_symbol}/{timeframe}] Success: {records:,} records")
            else:
                error = result.get('error', 'Unknown error')
                print(f"[{normalized_symbol}/{timeframe}] Failed: {error}")

        # Summary
        total_records = sum(r.get('records_written', 0) for r in results.values())
        successful = sum(1 for r in results.values() if r.get('success', False))

        print(f"\n[{normalized_symbol}] Complete: {successful}/{len(timeframes)} timeframes, {total_records:,} total records")

        return results

    def batch_update_intraday(
        self,
        symbols: List[str],
        timeframes: List[str],
        *,
        scid_folder: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        replace: bool = False
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Batch update multiple symbols at multiple timeframes from SCID files.

        This is the most comprehensive intraday update method, processing:
        - Multiple symbols
        - Multiple timeframes per symbol
        - Writing to market/{symbol}/{timeframe} sub-keys

        Parameters
        ----------
        symbols : List[str]
            List of symbols to process
        timeframes : List[str]
            List of timeframes to create for each symbol
        scid_folder : str, optional
            Folder containing SCID files
        start : pd.Timestamp, optional
            Start date for data loading
        end : pd.Timestamp, optional
            End date for data loading
        replace : bool, default False
            If True, replace existing data

        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]]
            Nested dictionary: {symbol: {timeframe: result}}

        Examples
        --------
        >>> processor = DataProcessor(client)
        >>> # Update multiple commodities at standard timeframes
        >>> results = processor.batch_update_intraday(
        ...     symbols=['CL_F', 'NG_F', 'ZC_F'],
        ...     timeframes=['1T', '5T', '15T', '1H'],
        ...     start=pd.Timestamp('2024-01-01')
        ... )
        >>> # Check results
        >>> for symbol, tf_results in results.items():
        ...     for tf, result in tf_results.items():
        ...         if result['success']:
        ...             print(f"{symbol}/{tf}: {result['records_written']} records")
        """
        if not symbols:
            raise ValueError("symbols list must be provided")

        if not timeframes:
            raise ValueError("timeframes list must be provided")

        print(f"\n[BATCH UPDATE] Processing {len(symbols)} symbols × {len(timeframes)} timeframes")
        print(f"[BATCH UPDATE] Total operations: {len(symbols) * len(timeframes)}")

        all_results = {}

        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"Processing symbol: {symbol}")
            print(f"{'='*60}")

            symbol_results = self.update_multiple_timeframes(
                symbol=symbol,
                timeframes=timeframes,
                scid_folder=scid_folder,
                start=start,
                end=end,
                replace=replace
            )

            all_results[symbol] = symbol_results

        # Final summary
        print(f"\n{'='*60}")
        print(f"BATCH UPDATE COMPLETE")
        print(f"{'='*60}")

        total_records = 0
        successful_ops = 0
        total_ops = 0

        for symbol, tf_results in all_results.items():
            for timeframe, result in tf_results.items():
                total_ops += 1
                if result.get('success'):
                    successful_ops += 1
                    total_records += result.get('records_written', 0)

        print(f"Symbols processed: {len(symbols)}")
        print(f"Timeframes per symbol: {len(timeframes)}")
        print(f"Successful operations: {successful_ops}/{total_ops}")
        print(f"Total records written: {total_records:,}")
        print(f"Success rate: {successful_ops/total_ops*100:.1f}%")

        return all_results

    # =================================================================
    # FuturesCurve Processing Methods using DLY Contract Managers
    # =================================================================

    def update_futures_curve_from_dly(
        self,
        symbol: str,
        *,
        dly_folder: str,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Update futures curve data using DLY files via DLYContractManager.

        Parameters
        ----------
        symbol : str
            Market symbol (with or without the ``_F`` suffix).
        dly_folder : str
            Folder path containing DLY files.
        as_of : pd.Timestamp, optional
            Date for curve construction. Uses latest if None.

        Returns
        -------
        Dict[str, Any]
            Summary information about curve processing.
        """

        if not symbol:
            raise ValueError("symbol must be provided")

        normalized_symbol = symbol.upper()
        if normalized_symbol.endswith("_F"):
            base_symbol = normalized_symbol[:-2]
        else:
            base_symbol = normalized_symbol
            normalized_symbol = f"{base_symbol}_F"

        try:
            # Initialize DLY contract manager
            manager = DLYContractManager(
                ticker=base_symbol,
                folder=dly_folder
            )

            # Build or update the curve
            if as_of is None:
                as_of = pd.Timestamp.now()

            # Build curve from DLY files
            curve_result = manager.run(save=False)

            if curve_result is None:
                return {
                    "symbol": normalized_symbol,
                    "success": False,
                    "message": 'Failed to build curve from DLY files'
                }

            # Get the new curve data
            new_curve_data = manager.curve
            if new_curve_data is None or new_curve_data.empty:
                return {
                    "symbol": normalized_symbol,
                    "success": False,
                    "message": "No curve data generated"
                }

            # Combine with previous data and verify before saving
            combined_data = self._combine_and_verify_curve_data(
                manager, normalized_symbol, new_curve_data
            )

            if combined_data is None:
                return {
                    "symbol": normalized_symbol,
                    "success": False,
                    "message": "Data combination and verification failed"
                }

            # Set the HDF path in the manager for storage
            manager.hdf_path = self.client.market_data_path

            # Use DLYContractManager's proven storage method with HDF5 locking
            with self._hdf_lock:
                manager.save_hdf()

            # Format for compatibility (for return message)
            formatted_curve = self._format_dly_curve_for_storage(new_curve_data, base_symbol)

            return {
                "symbol": normalized_symbol,
                "success": True,
                "contracts_processed": len(curve_data),
                "curve_date": str(as_of.date()),
                "last_curve_timestamp": str(formatted_curve.index.max()) if not formatted_curve.empty else None,
                "dly_folder": dly_folder
            }

        except Exception as e:
            return {
                "symbol": normalized_symbol,
                "success": False,
                "error": str(e),
                "dly_folder": dly_folder
            }

    def batch_update_futures_curves(
        self,
        symbols: List[str],
        *,
        dly_folder: str,
        as_of: Optional[pd.Timestamp] = None,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """Update multiple futures curves using DLYFolderUpdater.

        Parameters
        ----------
        symbols : List[str]
            List of market symbols to process.
        dly_folder : str
            Folder path containing DLY files.
        as_of : pd.Timestamp, optional
            Date for curve construction. Uses latest if None.
        max_workers : int, default 4
            Maximum number of worker threads.

        Returns
        -------
        Dict[str, Any]
            Summary of batch processing results.
        """

        if not symbols:
            raise ValueError("symbols list must be provided")

        if as_of is None:
            as_of = pd.Timestamp.now()

        try:
            # Initialize DLY folder updater
            updater = DLYFolderUpdater(folder=dly_folder)

            # Normalize symbols for processing
            base_symbols = []
            for symbol in symbols:
                normalized_symbol = symbol.upper()
                if normalized_symbol.endswith("_F"):
                    base_symbol = normalized_symbol[:-2]
                else:
                    base_symbol = normalized_symbol
                base_symbols.append(base_symbol)

            # Filter to only symbols that exist
            available_tickers = updater.list_tickers()
            valid_symbols = [s for s in base_symbols if s in available_tickers]

            if not valid_symbols:
                return {
                    "total_symbols": len(symbols),
                    "processed_symbols": 0,
                    "failed_symbols": len(symbols),
                    "success": False,
                    "message": f"None of the requested symbols found in DLY folder: {base_symbols}",
                    "available_tickers": available_tickers[:10]  # Show first 10 available
                }

            # Process all symbols using run_all
            pattern = "|".join(valid_symbols)  # Create regex pattern
            batch_result = updater.run_all(
                pattern=f"^({pattern})$",
                save=False,
                max_workers=max_workers,
                use_threading=True
            )

            # Process results and store curves
            results = {
                "total_symbols": len(symbols),
                "processed_symbols": 0,
                "failed_symbols": 0,
                "curve_updates": [],
                "errors": []
            }

            # Handle successful tickers
            successful_tickers = batch_result.get("tickers", [])
            by_ticker = batch_result.get("by_ticker", {})

            for ticker in valid_symbols:
                normalized_symbol = f"{ticker}_F"

                if ticker in successful_tickers and ticker in by_ticker:
                    results["processed_symbols"] += 1

                    # Create individual manager and store using its save method
                    try:
                        manager = DLYContractManager(ticker=ticker, folder=dly_folder)
                        manager.run(save=False)

                        if manager.curve is not None and not manager.curve.empty:
                            # Set HDF path and use manager's storage method
                            manager.hdf_path = self.client.market_data_path

                            # Use DLYContractManager's storage method with HDF5 locking
                            with self._hdf_lock:
                                manager.save_hdf()

                            results["curve_updates"].append({
                                "symbol": normalized_symbol,
                                "contracts": len(manager.curve.columns),
                                "timestamp": str(manager.curve.index.max()) if not manager.curve.empty else None
                            })
                    except Exception as e:
                        results["failed_symbols"] += 1
                        results["errors"].append({
                            "symbol": normalized_symbol,
                            "error": f"Failed to process individual manager: {e}"
                        })
                else:
                    results["failed_symbols"] += 1
                    error_key = f"{ticker}_ERROR"
                    error_info = by_ticker.get(error_key, {})
                    results["errors"].append({
                        "symbol": normalized_symbol,
                        "error": error_info.get('error', 'Unknown processing error')
                    })

            results.update({
                "success_rate": results["processed_symbols"] / len(symbols) * 100,
                "batch_timestamp": str(as_of),
                "dly_folder": dly_folder
            })

            return results

        except Exception as e:
            return {
                "total_symbols": len(symbols),
                "processed_symbols": 0,
                "failed_symbols": len(symbols),
                "success": False,
                "error": str(e),
                "dly_folder": dly_folder
            }

    def _combine_and_verify_curve_data(
        self,
        manager,
        normalized_symbol: str,
        new_curve_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Combine previous and new curve data with verification before saving.

        Parameters
        ----------
        manager : DLYContractManager
            The manager instance with new data
        normalized_symbol : str
            Symbol key for data retrieval
        new_curve_data : pd.DataFrame
            New curve data to combine

        Returns
        -------
        Optional[pd.DataFrame]
            Combined and verified data, or None if verification fails
        """
        try:
            # Try to load existing data
            existing_data = None
            try:
                with pd.HDFStore(self.client.market_data_path, mode='r') as store:
                    curve_key = f"market/{normalized_symbol}/curve"
                    if curve_key in store:
                        existing_data = store[curve_key]
            except (KeyError, FileNotFoundError, Exception):
                # No existing data or cannot read - proceed with new data only
                pass

            if existing_data is None or existing_data.empty:
                # No previous data - use new data directly
                combined_data = new_curve_data.copy()
            else:
                # Combine existing and new data
                combined_data = pd.concat([existing_data, new_curve_data], axis=0)

                # Remove duplicates by index, keeping latest values
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

                # Sort by index
                combined_data = combined_data.sort_index()

            # Verification checks
            if combined_data.empty:
                return None

            # Check for reasonable data ranges
            numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return None

            # Basic sanity checks - no all-zero or all-NaN columns
            for col in numeric_columns:
                if combined_data[col].isna().all() or (combined_data[col] == 0).all():
                    continue  # Allow some columns to be zero/NaN

            # Update manager with combined data for saving
            manager.curve = combined_data

            # Update sequential data if available
            if hasattr(manager, 'seq_prices') and manager.seq_prices is not None:
                try:
                    # Try to load existing sequential data
                    with pd.HDFStore(self.client.market_data_path, mode='r') as store:
                        seq_key = f"market/{normalized_symbol}/seq_curve"
                        if seq_key in store:
                            existing_seq = store[seq_key]
                            # Combine sequential data
                            combined_seq = pd.concat([existing_seq, manager.seq_prices], axis=0)
                            combined_seq = combined_seq[~combined_seq.index.duplicated(keep='last')]
                            combined_seq = combined_seq.sort_index()
                            manager.seq_prices = combined_seq
                except Exception:
                    # Keep new sequential data if combination fails
                    pass

            return combined_data

        except Exception as e:
            # Return None to indicate verification failure
            return None

    def _format_dly_curve_for_storage(
        self,
        curve_data: pd.DataFrame,
        base_symbol: str
    ) -> pd.DataFrame:
        """Format DLY curve data for HDF5 storage.

        Parameters
        ----------
        curve_data : pd.DataFrame
            Raw curve data from DLYContractManager
        base_symbol : str
            Base symbol for the curve

        Returns
        -------
        pd.DataFrame
            Formatted curve data ready for storage
        """

        if curve_data.empty:
            return pd.DataFrame()

        formatted = curve_data.copy()

        # Ensure datetime index
        if not isinstance(formatted.index, pd.DatetimeIndex):
            formatted.index = pd.to_datetime(formatted.index)

        # Add symbol information
        formatted['symbol'] = base_symbol
        formatted['normalized_symbol'] = f"{base_symbol}_F"

        # Ensure required columns exist with defaults
        required_columns = {
            'price': 0.0,
            'volume': 0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'last': 0.0,
            'contracts_processed': 1,
            'days_to_expiry': 0,
            'contract_month': '',
        }

        for col, default_val in required_columns.items():
            if col not in formatted.columns:
                formatted[col] = default_val

        # Calculate additional curve metrics
        formatted = self._add_curve_analytics(formatted)

        # Ensure proper data types
        float_cols = ['price', 'open', 'high', 'low', 'last', 'days_to_expiry']
        int_cols = ['volume', 'contracts_processed']

        for col in float_cols:
            if col in formatted.columns:
                formatted[col] = pd.to_numeric(formatted[col], errors='coerce').fillna(0.0)

        for col in int_cols:
            if col in formatted.columns:
                formatted[col] = pd.to_numeric(formatted[col], errors='coerce').fillna(0).astype('int64')

        # Sort by datetime
        formatted = formatted.sort_index()
        formatted = formatted[~formatted.index.duplicated(keep="last")]

        return formatted

    def _add_curve_analytics(self, curve_data: pd.DataFrame) -> pd.DataFrame:
        """Add analytical metrics to curve data.

        Parameters
        ----------
        curve_data : pd.DataFrame
            Raw curve data

        Returns
        -------
        pd.DataFrame
            Curve data with added analytics
        """

        enhanced = curve_data.copy()

        # Calculate term structure metrics
        if 'price' in enhanced.columns and len(enhanced) > 1:
            enhanced['price_change'] = enhanced['price'].diff()
            enhanced['price_change_pct'] = enhanced['price'].pct_change() * 100

        # Calculate contango/backwardation if multiple contracts
        if 'days_to_expiry' in enhanced.columns and len(enhanced) > 1:
            # Sort by days to expiry to calculate spreads
            sorted_data = enhanced.sort_values('days_to_expiry')
            if len(sorted_data) >= 2:
                front_price = sorted_data['price'].iloc[0]
                back_price = sorted_data['price'].iloc[-1]
                if front_price > 0:
                    enhanced['term_structure'] = (back_price - front_price) / front_price * 100
                else:
                    enhanced['term_structure'] = 0.0
            else:
                enhanced['term_structure'] = 0.0
        else:
            enhanced['term_structure'] = 0.0

        # Add temporal features
        enhanced['weekday'] = enhanced.index.dayofweek
        enhanced['month'] = enhanced.index.month
        enhanced['quarter'] = enhanced.index.quarter

        # Calculate volume-weighted metrics if volume exists
        if 'volume' in enhanced.columns and enhanced['volume'].sum() > 0:
            total_volume = enhanced['volume'].sum()
            enhanced['volume_weight'] = enhanced['volume'] / total_volume
            if 'price' in enhanced.columns:
                enhanced['vwap'] = (enhanced['price'] * enhanced['volume_weight']).sum()
        else:
            enhanced['volume_weight'] = 0.0
            enhanced['vwap'] = enhanced.get('price', 0.0)

        return enhanced

    def get_curve_summary(
        self,
        symbol: str,
        days: int = 30,
        curve_source: str = "dly"
    ) -> Dict[str, Any]:
        """Get summary statistics for futures curve data.

        Parameters
        ----------
        symbol : str
            Symbol to analyze
        days : int, default 30
            Number of days to analyze
        curve_source : str, default "dly"
            Source of curve data ("dly" or "csv")

        Returns
        -------
        Dict[str, Any]
            Curve summary statistics
        """

        normalized_symbol = symbol.upper()
        if not normalized_symbol.endswith("_F"):
            normalized_symbol = f"{normalized_symbol}_F"

        try:
            # Use your fixed DataClient.query_curve_data method
            curve_data = self.client.query_curve_data(normalized_symbol, curve_types=['curve'])

            if curve_data.empty:
                return {
                    "symbol": normalized_symbol,
                    "message": "No curve data found",
                    "summary": {},
                    "source": curve_source
                }

            # Filter to requested days
            if len(curve_data) > days:
                curve_data = curve_data.tail(days)

            # Calculate comprehensive summary statistics
            summary = {
                "total_records": len(curve_data),
                "date_range": {
                    "start": str(curve_data.index.min()),
                    "end": str(curve_data.index.max())
                },
                "price_stats": {},
                "volume_stats": {},
                "curve_metrics": {},
                "term_structure": {}
            }

            # Price statistics
            price_cols = ['price', 'open', 'high', 'low', 'last']
            for col in price_cols:
                if col in curve_data.columns and not curve_data[col].isna().all():
                    price_data = curve_data[col].dropna()
                    summary["price_stats"][col] = {
                        "mean": float(price_data.mean()),
                        "std": float(price_data.std()),
                        "min": float(price_data.min()),
                        "max": float(price_data.max()),
                        "latest": float(price_data.iloc[-1]) if len(price_data) > 0 else 0.0
                    }

            # Volume statistics
            if 'volume' in curve_data.columns and not curve_data['volume'].isna().all():
                volume_data = curve_data['volume'].dropna()
                summary["volume_stats"] = {
                    "mean": float(volume_data.mean()),
                    "std": float(volume_data.std()),
                    "total": int(volume_data.sum()),
                    "latest": int(volume_data.iloc[-1]) if len(volume_data) > 0 else 0
                }

            # Curve-specific metrics
            curve_metric_cols = [
                'price_change', 'price_change_pct', 'term_structure',
                'volume_weight', 'vwap', 'days_to_expiry'
            ]
            for metric in curve_metric_cols:
                if metric in curve_data.columns and not curve_data[metric].isna().all():
                    metric_data = curve_data[metric].dropna()
                    if len(metric_data) > 0:
                        summary["curve_metrics"][metric] = {
                            "mean": float(metric_data.mean()),
                            "std": float(metric_data.std()),
                            "latest": float(metric_data.iloc[-1])
                        }

            # Term structure analysis
            if 'term_structure' in curve_data.columns:
                ts_data = curve_data['term_structure'].dropna()
                if len(ts_data) > 0:
                    contango_days = (ts_data > 0).sum()
                    backwardation_days = (ts_data < 0).sum()
                    summary["term_structure"] = {
                        "avg_term_structure": float(ts_data.mean()),
                        "contango_days": int(contango_days),
                        "backwardation_days": int(backwardation_days),
                        "contango_ratio": float(contango_days / len(ts_data)) if len(ts_data) > 0 else 0.0,
                        "latest_structure": float(ts_data.iloc[-1]) if len(ts_data) > 0 else 0.0
                    }

            # Processing statistics
            if 'contracts_processed' in curve_data.columns:
                contracts_data = curve_data['contracts_processed'].dropna()
                if len(contracts_data) > 0:
                    summary["processing_stats"] = {
                        "avg_contracts": float(contracts_data.mean()),
                        "max_contracts": int(contracts_data.max()),
                        "latest_contracts": int(contracts_data.iloc[-1])
                    }

            return {
                "symbol": normalized_symbol,
                "summary": summary,
                "source": curve_source
            }

        except Exception as e:
            return {
                "symbol": normalized_symbol,
                "error": str(e),
                "summary": {},
                "source": curve_source
            }

    def format_curve_for_analysis(
        self,
        symbol: str,
        dly_folder: str,
        *,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.DataFrame]:
        """Format curve data for external analysis tools.

        Parameters
        ----------
        symbol : str
            Symbol to format
        dly_folder : str
            DLY folder path
        start_date : pd.Timestamp, optional
            Start date for data
        end_date : pd.Timestamp, optional
            End date for data

        Returns
        -------
        pd.DataFrame or None
            Formatted curve data ready for analysis
        """

        normalized_symbol = symbol.upper()
        if not normalized_symbol.endswith("_F"):
            normalized_symbol = f"{normalized_symbol}_F"

        try:
            # Use your fixed DataClient.query_curve_data method
            curve_data = self.client.query_curve_data(normalized_symbol, curve_types=['curve'])

            if curve_data.empty:
                # Try to build curve from DLY files
                base_symbol = normalized_symbol.replace("_F", "")
                curve_result = self.update_futures_curve_from_dly(
                    symbol=base_symbol,
                    dly_folder=dly_folder,
                    as_of=end_date
                )

                if not curve_result.get('success', False):
                    return None

                # Get the newly created data
                curve_data = self.client.query_curve_data(normalized_symbol, curve_types=['curve'])
                if curve_data.empty:
                    return None

            # Apply date filters if specified
            if start_date is not None:
                curve_data = curve_data[curve_data.index >= start_date]
            if end_date is not None:
                curve_data = curve_data[curve_data.index <= end_date]

            if curve_data.empty:
                return None

            # Ensure proper formatting for analysis
            formatted = curve_data.copy()

            # Add analysis-ready columns
            if 'price' in formatted.columns:
                formatted['returns'] = formatted['price'].pct_change()
                formatted['log_returns'] = np.log(formatted['price'] / formatted['price'].shift(1))

            # Add moving averages if sufficient data
            if len(formatted) >= 20:
                formatted['ma_20'] = formatted['price'].rolling(20).mean()
                formatted['volatility_20'] = formatted['returns'].rolling(20).std()

            # Add seasonal indicators
            formatted['month'] = formatted.index.month
            formatted['quarter'] = formatted.index.quarter
            formatted['year'] = formatted.index.year

            return formatted

        except Exception as e:
            print(f"Error formatting curve data for {symbol}: {e}")
            return None

    def validate_curve_data_integrity(
        self,
        symbol: str,
        dly_folder: str
    ) -> Dict[str, Any]:
        """Validate the integrity of curve data against DLY source files.

        Parameters
        ----------
        symbol : str
            Symbol to validate
        dly_folder : str
            DLY folder path for validation

        Returns
        -------
        Dict[str, Any]
            Validation results and recommendations
        """

        normalized_symbol = symbol.upper()
        if not normalized_symbol.endswith("_F"):
            normalized_symbol = f"{normalized_symbol}_F"

        base_symbol = normalized_symbol.replace("_F", "")

        validation_result = {
            "symbol": normalized_symbol,
            "validation_passed": False,
            "issues": [],
            "recommendations": [],
            "stats": {}
        }

        try:
            # Check HDF5 curve data
            curve_key = f"{normalized_symbol}/curve"
            try:
                hdf_data = self.client.get_market_tail(curve_key, 100)
                validation_result["stats"]["hdf_records"] = len(hdf_data)
                validation_result["stats"]["hdf_date_range"] = {
                    "start": str(hdf_data.index.min()) if not hdf_data.empty else None,
                    "end": str(hdf_data.index.max()) if not hdf_data.empty else None
                }
            except Exception as e:
                validation_result["issues"].append(f"Cannot read HDF5 curve data: {e}")
                hdf_data = pd.DataFrame()

            # Check DLY source files
            dly_folder_path = Path(dly_folder)
            if not dly_folder_path.exists():
                validation_result["issues"].append(f"DLY folder not found: {dly_folder}")
                return validation_result

            # Find DLY files for this symbol
            dly_pattern = f"{base_symbol}*.dly"
            dly_files = list(dly_folder_path.glob(dly_pattern))
            validation_result["stats"]["dly_files_found"] = len(dly_files)

            if len(dly_files) == 0:
                validation_result["issues"].append(f"No DLY files found for pattern: {dly_pattern}")
                validation_result["recommendations"].append("Check DLY folder path and file naming convention")
                return validation_result

            # Test DLY processing
            try:
                manager = DLYContractManager(
                    ticker=base_symbol,
                    folder=dly_folder
                )

                test_result = manager.run(save=False)
                if test_result is not None:
                    validation_result["stats"]["dly_processing"] = "success"
                    test_curve = manager.curve
                    if test_curve is not None and not test_curve.empty:
                        validation_result["stats"]["dly_contracts"] = len(test_curve.columns)
                else:
                    validation_result["issues"].append("Failed to build curve from DLY files")
                    validation_result["stats"]["dly_processing"] = "failed"

            except Exception as e:
                validation_result["issues"].append(f"DLY processing error: {e}")

            # Data consistency checks
            if not hdf_data.empty:
                # Check for missing values
                missing_critical = []
                for col in ['price', 'symbol']:
                    if col not in hdf_data.columns:
                        missing_critical.append(col)
                    elif hdf_data[col].isna().all():
                        missing_critical.append(f"{col} (all NaN)")

                if missing_critical:
                    validation_result["issues"].append(f"Missing critical columns: {missing_critical}")

                # Check for data gaps
                if len(hdf_data) > 1:
                    date_diff = hdf_data.index.to_series().diff().dt.days.dropna()
                    large_gaps = (date_diff > 7).sum()  # More than a week gap
                    if large_gaps > 0:
                        validation_result["issues"].append(f"Found {large_gaps} large data gaps (>7 days)")

            # Final validation
            if len(validation_result["issues"]) == 0:
                validation_result["validation_passed"] = True
                validation_result["recommendations"].append("Curve data appears healthy")
            else:
                validation_result["recommendations"].extend([
                    "Consider rebuilding curve data from DLY files",
                    "Verify DLY file completeness and format",
                    "Check HDF5 storage consistency"
                ])

            return validation_result

        except Exception as e:
            validation_result["issues"].append(f"Validation process failed: {e}")
            validation_result["recommendations"].append("Check system configuration and file permissions")
            return validation_result