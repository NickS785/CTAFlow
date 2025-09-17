"""Simplified CSV data processor for market data updates and FuturesCurve processing."""

import asyncio
import fnmatch
import re
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set, Sequence

import numpy as np
import pandas as pd

from .contract_handling.dly_contract_manager import DLYContractManager, DLYFolderUpdater
from CTAFlow.config import RAW_MARKET_DATA_PATH


class SimpleDataProcessor:
    """Simplified data processor that only handles CSV files."""

    def __init__(self, client):
        self.client = client

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

    async def update_all_tickers(
        self,
        *,
        dly_folder: str,
        pattern: Optional[str] = None,
        pattern_mode: str = "regex",
        selected_cot_features=None,
        max_concurrency: int = 4,
        raw_data_path: Optional[str] = None,
        market_resample_rule: str = "1T",
        cot_progress: bool = False,
    ) -> Dict[str, Any]:
        """Update COT metrics, market data and curve datasets for all tracked tickers."""

        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        cot_result = await self.client.write_all_metrics(
            selected_cot_features, progress=cot_progress
        )

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

        # Run curve updates concurrently (if any)
        semaphore = asyncio.Semaphore(max_concurrency)
        loop = asyncio.get_running_loop()

        async def run_curve_job(job: Dict[str, Any]) -> None:
            symbol = job["symbol"]
            curve_files = job["curve_files"]
            curve_keys = job["curve_keys"]

            if not curve_files or not curve_keys:
                return

            async with semaphore:
                try:
                    task = partial(
                        self._execute_curve_update,
                        symbol=symbol,
                        base_symbol=job["base_symbol"],
                        curve_keys=tuple(curve_keys),
                        curve_files=tuple(curve_files),
                        folder_updater=folder_updater,
                    )
                    result = await loop.run_in_executor(None, task)
                    details = job.get("details") or summary["updates"].setdefault(symbol, {})
                    details["curve_result"] = result
                except Exception as exc:
                    summary["errors"].setdefault(symbol, {})["curve"] = str(exc)

        await asyncio.gather(*(run_curve_job(job) for job in update_jobs))

        if summary["skipped"]:
            summary["skipped"] = sorted(set(summary["skipped"]))

        return summary

    def _execute_curve_update(
        self,
        *,
        symbol: str,
        base_symbol: str,
        curve_keys: Sequence[str],
        curve_files: Sequence[str],
        folder_updater: DLYFolderUpdater,
    ) -> Dict[str, Any]:
        """Run curve updates for a single ticker synchronously."""

        if not curve_files or not curve_keys:
            return {}

        return folder_updater.update_existing_ticker(
            base_symbol,
            existing_curve_keys=tuple(curve_keys),
            file_paths=tuple(curve_files),
        )

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

        # Write to HDF5
        self.client.write_market(resampled, tail_key, replace=False)

        return {
            "symbol": normalized_symbol,
            "csv_files": processed_files,
            "appended_rows": len(resampled),
            "resample_rule": resample_rule,
            "last_timestamp": str(resampled.index.max()) if not resampled.empty else None,
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
                        elif col_lower in ['num_trades', 'trades', 'numberoftrades', 'number_of_trades', '#oftrades']:
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

        working[float_cols] = working[float_cols].astype(float)
        working[int_cols] = working[int_cols].astype(float)

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

            # Get the curve data
            curve_data = manager.curve
            if curve_data is None or curve_data.empty:
                return {
                    "symbol": normalized_symbol,
                    "success": False,
                    "message": "No curve data generated"
                }

            # Format for HDF5 storage
            formatted_curve = self._format_dly_curve_for_storage(curve_data, base_symbol)

            # Store curve data
            curve_key = f"market/{normalized_symbol}/curve"
            self.client.write_market(formatted_curve, curve_key, replace=False)

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

                    # Create individual manager to get curve data
                    try:
                        manager = DLYContractManager(ticker=ticker, folder=dly_folder)
                        manager.run(save=False)

                        if manager.curve is not None and not manager.curve.empty:
                            formatted_curve = self._format_dly_curve_for_storage(manager.curve, ticker)
                            curve_key = f"market/{normalized_symbol}/curve"
                            self.client.write_market(formatted_curve, curve_key, replace=False)

                            results["curve_updates"].append({
                                "symbol": normalized_symbol,
                                "contracts": len(manager.curve.columns),
                                "timestamp": str(formatted_curve.index.max()) if not formatted_curve.empty else None
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

        curve_key = f"{normalized_symbol}/curve"

        try:
            # Get recent curve data from HDF5
            curve_data = self.client.get_market_tail(curve_key, days * 10)  # Get more data to ensure we have enough

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
            # Get curve data from HDF5
            curve_data = self.client.query_market_data(normalized_symbol)

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
                curve_data = self.client.query_market_data(normalized_symbol)
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