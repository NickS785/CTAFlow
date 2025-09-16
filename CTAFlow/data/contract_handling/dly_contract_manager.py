import os
import re
import fnmatch
import time
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from CTAFlow.config import MARKET_DATA_PATH, RAW_MARKET_DATA_PATH

# Default location for DLY files
DLY_DATA_PATH = RAW_MARKET_DATA_PATH / "daily"

MONTH_CODE_MAP = {
    "F": 1,  # Jan
    "G": 2,  # Feb
    "H": 3,  # Mar
    "J": 4,  # Apr
    "K": 5,  # May
    "M": 6,  # Jun
    "N": 7,  # Jul
    "Q": 8,  # Aug
    "U": 9,  # Sep
    "V": 10, # Oct
    "X": 11, # Nov
    "Z": 12, # Dec
}


@dataclass(frozen=True)
class ContractInfo:
    ticker: str
    month: str
    year: int
    exchange: str

    @property
    def contract_id(self) -> str:
        return f"{self.month}{str(self.year)[-2:]}"


def parse_contract_filename(fn: str, pivot: int = 1970) -> Optional[ContractInfo]:
    """Parse filenames like CLH25-NYM.dly."""
    m = re.match(r"([A-Za-z]+)([FGHJKMNQUVXZ])(\d{2})-([A-Za-z]+)\.dly", fn, re.IGNORECASE)
    if not m:
        return None
    ticker, mcode, yy, exch = m.groups()
    yy = int(yy)
    year = 2000 + yy if yy <= pivot % 100 else 1900 + yy
    return ContractInfo(ticker.upper(), mcode.upper(), year, exch.upper())


def discover_tickers(folder: str) -> Dict[str, List[str]]:
    """
    Return mapping: ticker -> [file paths] for all .dly files that parse.
    """
    out: Dict[str, List[str]] = {}
    folder_str = str(folder)  # Ensure we have a string path

    if not os.path.isdir(folder_str):
        return out

    for fn in os.listdir(folder_str):
        if not fn.endswith(".dly"):
            continue
        info = parse_contract_filename(fn)
        if info is None:
            continue
        out.setdefault(info.ticker, []).append(os.path.join(folder_str, fn))

    for t, files in out.items():
        files.sort()
    return out


def filter_tickers(tickers: List[str], pattern: Optional[str], mode: str = "regex") -> List[str]:
    if pattern is None:
        return tickers

    if mode == "glob":
        return [t for t in tickers if fnmatch.fnmatch(t, pattern)]
    else:
        regex = re.compile(pattern)
        return [t for t in tickers if regex.search(t)]


def _read_dly(path: str) -> pd.DataFrame:
    """Read a .dly file with columns date, close and optionally volume/OI."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None,
                         names=["date", "close", "volume", "open_interest"],
                         parse_dates=[0])

    # Strip spaces from column names first
    df.columns = [str(c).strip() for c in df.columns]

    # Find date column with case-insensitive search
    date_col = None
    for col in df.columns:
        if str(col).lower().strip() in ['date', 'datetime', 'timestamp']:
            date_col = col
            break

    if date_col is None:
        # If no date column found, assume first column is date
        date_col = df.columns[0]

    # Ensure we have the expected column name
    if date_col != 'date':
        df = df.rename(columns={date_col: 'date'})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


class DLYContractManager:
    """
    Single-ticker contract manager for .dly files.
    Provides price, volume and open interest curves with sequential alignment.
    """

    def __init__(self, ticker: str, folder: str = None, hdf_path: Optional[str] = None, yy_pivot: int = 1970):
        self.folder = str(folder) or str(DLY_DATA_PATH)
        self.base_ticker = ticker.upper()  # Store the base ticker for file matching
        self.ticker = f"{ticker.upper()}_F"  # Store the display/save ticker with _F
        self.hdf_path = hdf_path or MARKET_DATA_PATH
        self.yy_pivot = yy_pivot

        self.contracts: Dict[str, pd.DataFrame] = {}
        self.expiries: Dict[str, pd.Timestamp] = {}
        self.label_base_to_full: Dict[str, List[str]] = {}

        self._file_cache: Optional[List[Tuple[ContractInfo, str]]] = None
        self._folder_mtime: Optional[float] = None

        self.curve: Optional[pd.DataFrame] = None
        self.dte: Optional[pd.DataFrame] = None
        self.expiry_series: Optional[pd.Series] = None
        self.front: Optional[pd.Series] = None
        self.seq_prices: Optional[pd.DataFrame] = None
        self.seq_labels: Optional[pd.DataFrame] = None
        self.seq_dte: Optional[pd.DataFrame] = None
        self.curve_volume: Optional[pd.DataFrame] = None
        self.seq_volume: Optional[pd.DataFrame] = None
        self.curve_oi: Optional[pd.DataFrame] = None
        self.seq_oi: Optional[pd.DataFrame] = None

    # ---------- ingestion ----------
    def collect_files(self) -> List[Tuple[ContractInfo, str]]:
        folder_str = str(self.folder)
        if not os.path.exists(folder_str):
            raise FileNotFoundError(f"Folder does not exist: {folder_str}")

        current_mtime = os.path.getmtime(folder_str)
        if (
            self._file_cache is not None and self._folder_mtime is not None and current_mtime == self._folder_mtime
        ):
            return self._file_cache

        pairs: List[Tuple[ContractInfo, str]] = []
        with os.scandir(folder_str) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".dly"):
                    info = parse_contract_filename(entry.name, pivot=self.yy_pivot)
                    if info and info.ticker == self.base_ticker:
                        pairs.append((info, entry.path))

        pairs.sort(key=lambda p: (p[0].year, MONTH_CODE_MAP[p[0].month]))
        self._file_cache = pairs
        self._folder_mtime = current_mtime
        return pairs

    def _unique_label_for(self, info: ContractInfo) -> str:
        base = info.contract_id
        label = base
        if label in self.contracts:
            label = f"{base}~{info.exchange}"
            k = 2
            while label in self.contracts:
                label = f"{base}~{info.exchange}~{k}"
                k += 1
        return label

    def load(self) -> None:
        self.contracts.clear()
        self.expiries.clear()
        self.label_base_to_full.clear()
        for info, path in self.collect_files():
            df = _read_dly(path)
            if df.empty:
                continue
            expiry = df.index.max()
            unique_label = self._unique_label_for(info)
            self.label_base_to_full.setdefault(info.contract_id, []).append(unique_label)

            # Find price column (case insensitive)
            price_col = None
            for col in df.columns:
                if str(col).lower().strip() in ['last', 'close', 'price', 'settle', 'settlement']:
                    price_col = col
                    break

            if price_col is None:
                print(f"Warning: No price column found in {path}")
                print(f"Available columns: {list(df.columns)}")
                continue

            contract_data = df[[price_col]].rename(columns={price_col: unique_label})

            # Find volume column (case insensitive)
            volume_col = None
            for col in df.columns:
                if str(col).lower().strip() in ['volume', 'vol']:
                    volume_col = col
                    break
            if volume_col is not None:
                contract_data[f"{unique_label}_volume"] = df[volume_col]

            # Find open interest column (case insensitive)
            oi_col = None
            for col in df.columns:
                if str(col).lower().strip() in ['open_interest', 'oi', 'openinterest', 'open interest']:
                    oi_col = col
                    break
            if oi_col is not None:
                contract_data[f"{unique_label}_oi"] = df[oi_col]

            self.contracts[unique_label] = contract_data
            self.expiries[unique_label] = pd.Timestamp(expiry)

    # ---------- curve & dte ----------
    def build_curve(self) -> pd.DataFrame:
        if not self.contracts:
            self.load()
        if not self.contracts:
            raise RuntimeError(f"No contracts found for {self.ticker} in {self.folder}")

        price_dfs = []
        volume_dfs = []
        oi_dfs = []
        for label, df in self.contracts.items():
            price_dfs.append(df[[label]])
            vol_col = f"{label}_volume"
            if vol_col in df.columns:
                volume_dfs.append(df[[vol_col]].rename(columns={vol_col: label}))
            oi_col = f"{label}_oi"
            if oi_col in df.columns:
                oi_dfs.append(df[[oi_col]].rename(columns={oi_col: label}))

        self.curve = pd.concat(price_dfs, axis=1).sort_index()
        self.curve.index = pd.to_datetime(self.curve.index, errors="coerce")
        self.curve = self.curve[self.curve.index.notna()]

        if volume_dfs:
            self.curve_volume = pd.concat(volume_dfs, axis=1).sort_index().reindex(self.curve.index)
        if oi_dfs:
            self.curve_oi = pd.concat(oi_dfs, axis=1).sort_index().reindex(self.curve.index)
        return self.curve

    def build_dte(self) -> pd.DataFrame:
        if self.curve is None:
            self.build_curve()
        dte = pd.DataFrame(index=self.curve.index, columns=self.curve.columns, dtype="float")
        for label in self.curve.columns:
            exp = self.expiries.get(label)
            if exp is not None:
                dte[label] = (pd.to_datetime(exp) - dte.index.normalize()).days
        self.dte = dte
        return dte

    def build_expiry_series(self) -> pd.Series:
        """
        Create a Series mapping contract labels to their expiry dates.
        This is needed for TenorInterpolator which requires expiry_data as a Series.

        Returns:
        --------
        pd.Series
            Series with contract labels as index and expiry dates as values
        """
        if not self.expiries:
            if not self.contracts:
                self.load()
            if not self.expiries:
                raise RuntimeError("No expiry data available")

        # Create Series from expiries dictionary
        expiry_series = pd.Series(self.expiries, name='expiry_date')
        expiry_series.index.name = 'contract_id'
        self.expiry_series = expiry_series
        return expiry_series

    # ---------- front & sequential ----------
    def compute_front(self) -> pd.Series:
        if self.curve is None:
            self.build_curve()
        if self.dte is None:
            self.build_dte()

        valid_dte = self.dte.where(self.dte >= 0)
        mask_all = valid_dte.isna().all(axis=1)
        front = valid_dte.fillna(np.inf).idxmin(axis=1)
        front[mask_all] = None
        self.front = front.rename("front")
        return self.front

    def sequentialize(self, max_buckets: Optional[int] = None):
        if self.curve is None:
            self.build_curve()
        if self.dte is None:
            self.build_dte()
        if self.front is None:
            self.compute_front()

        if max_buckets is None:
            max_buckets = 48

        valid_dte = self.dte.where(self.dte >= 0)
        dte_arr = valid_dte.values
        px_arr = self.curve.values
        vol_arr = self.curve_volume.values if self.curve_volume is not None else None
        oi_arr = self.curve_oi.values if self.curve_oi is not None else None
        cols = self.curve.columns.to_numpy()

        sort_keys = np.where(np.isnan(dte_arr), np.inf, dte_arr)
        order = np.argsort(sort_keys, axis=1)
        valid_counts = np.sum(~np.isnan(dte_arr), axis=1)

        rows = dte_arr.shape[0]
        max_buckets = min(max_buckets, dte_arr.shape[1])

        seq_px = np.full((rows, max_buckets), np.nan)
        seq_lb = np.empty((rows, max_buckets), dtype=object)
        seq_te = np.full((rows, max_buckets), np.nan)
        seq_vol = np.full((rows, max_buckets), np.nan) if vol_arr is not None else None
        seq_oi_arr = np.full((rows, max_buckets), np.nan) if oi_arr is not None else None

        for i in range(max_buckets):
            mask = valid_counts > i
            r_idx = np.where(mask)[0]
            c_idx = order[r_idx, i]
            seq_px[r_idx, i] = px_arr[r_idx, c_idx]
            seq_lb[r_idx, i] = cols[c_idx]
            seq_te[r_idx, i] = dte_arr[r_idx, c_idx]
            if seq_vol is not None:
                seq_vol[r_idx, i] = vol_arr[r_idx, c_idx]
            if seq_oi_arr is not None:
                seq_oi_arr[r_idx, i] = oi_arr[r_idx, c_idx]

        time_cols = [f"M{i}" for i in range(max_buckets)]
        self.seq_prices = pd.DataFrame(seq_px, index=self.curve.index, columns=time_cols)
        self.seq_labels = pd.DataFrame(seq_lb, index=self.curve.index, columns=time_cols)
        self.seq_dte = pd.DataFrame(seq_te, index=self.curve.index, columns=time_cols)
        if seq_vol is not None:
            self.seq_volume = pd.DataFrame(seq_vol, index=self.curve.index, columns=time_cols)
        if seq_oi_arr is not None:
            self.seq_oi = pd.DataFrame(seq_oi_arr, index=self.curve.index, columns=time_cols)

        # Drop columns that are entirely NaN
        non_empty = ~np.isnan(seq_px).all(axis=0)
        self.seq_prices = self.seq_prices.loc[:, non_empty]
        self.seq_labels = self.seq_labels.loc[:, non_empty]
        self.seq_dte = self.seq_dte.loc[:, non_empty]
        if self.seq_volume is not None:
            self.seq_volume = self.seq_volume.loc[:, non_empty]
        if self.seq_oi is not None:
            self.seq_oi = self.seq_oi.loc[:, non_empty]

        return self.seq_prices, self.seq_labels, self.seq_dte

    # ---------- convenience & persistence ----------
    def spreads_vs_front(self) -> Optional[pd.DataFrame]:
        if self.seq_prices is None:
            return None
        spreads = self.seq_prices.subtract(self.seq_prices["M0"], axis=0)
        return spreads

    def to_arrays(self) -> Dict[str, Any]:
        if self.curve is None:
            self.build_curve()
        if self.dte is None:
            self.build_dte()
        if self.seq_prices is None or self.seq_labels is None or self.seq_dte is None:
            self.sequentialize()
        spreads = self.spreads_vs_front()

        result = {
            "curve": self.curve.values,
            "labels": [str(c) for c in self.curve.columns],
            "dates": self.curve.index.values.astype("datetime64[ns]"),
            "curve_seq": None if self.seq_prices is None else self.seq_prices.values,
            "seq_labels": None if self.seq_labels is None else self.seq_labels.values.astype(object),
            "dte": None if self.dte is None else self.dte.values,
            "dte_seq": None if self.seq_dte is None else self.seq_dte.values,
            "spreads_seq": None if spreads is None else spreads.values,
            "front": None if self.front is None else self.front.values.astype(object),
            "label_groups": self.label_base_to_full,
        }

        if self.curve_volume is not None:
            result["curve_volume"] = self.curve_volume.values
            result["volume_labels"] = [str(c) for c in self.curve_volume.columns]
        if self.seq_volume is not None:
            result["seq_volume"] = self.seq_volume.values
        if self.curve_oi is not None:
            result["curve_oi"] = self.curve_oi.values
            result["oi_labels"] = [str(c) for c in self.curve_oi.columns]
        if self.seq_oi is not None:
            result["seq_oi"] = self.seq_oi.values
        return result

    def save_hdf(self) -> None:
        if self.hdf_path is None:
            return
        with pd.HDFStore(self.hdf_path) as store:
            if self.curve is not None:
                store.put(f"market/{self.ticker}/curve", self.curve, format="table", data_columns=True)
            if self.dte is not None:
                store.put(f"market/{self.ticker}/dte", self.dte, format="table", data_columns=True)
            if self.expiry_series is not None:
                store.put(f"market/{self.ticker}/expiry", self.expiry_series.to_frame(), format="table", data_columns=True)
            if self.front is not None:
                store.put(f"market/{self.ticker}/front", self.front.to_frame(), format="table", data_columns=True)
            if self.seq_prices is not None:
                store.put(f"market/{self.ticker}/seq_curve", self.seq_prices, format="table", data_columns=True)
            if self.seq_labels is not None:
                store.put(f"market/{self.ticker}/seq_labels", self.seq_labels, format="table", data_columns=True)
            if self.seq_dte is not None:
                store.put(f"market/{self.ticker}/seq_dte", self.seq_dte, format="table", data_columns=True)
            spreads = self.spreads_vs_front()
            if spreads is not None:
                store.put(f"market/{self.ticker}/seq_spreads", spreads, format="table", data_columns=True)
            if self.curve_volume is not None:
                store.put(f"market/{self.ticker}/curve_volume", self.curve_volume, format="table", data_columns=True)
            if self.seq_volume is not None:
                store.put(f"market/{self.ticker}/seq_volume", self.seq_volume, format="table", data_columns=True)
            if self.curve_oi is not None:
                store.put(f"market/{self.ticker}/curve_oi", self.curve_oi, format="table", data_columns=True)
            if self.seq_oi is not None:
                store.put(f"market/{self.ticker}/seq_oi", self.seq_oi, format="table", data_columns=True)

    def run(self, save: bool = False) -> Dict[str, str]:
        self.load()
        self.build_curve()
        self.build_dte()
        self.build_expiry_series()
        self.compute_front()
        self.sequentialize()
        if save:
            self.save_hdf()

        result = {
            "curve": f"market/{self.ticker}/curve",
            "dte": f"market/{self.ticker}/dte",
            "expiry": f"market/{self.ticker}/expiry",
            "front": f"market/{self.ticker}/front",
            "seq_curve": f"market/{self.ticker}/seq_curve",
            "seq_labels": f"market/{self.ticker}/seq_labels",
            "seq_dte": f"market/{self.ticker}/seq_dte",
            "seq_spreads": f"market/{self.ticker}/seq_spreads",
        }
        if self.curve_volume is not None:
            result["curve_volume"] = f"market/{self.ticker}/curve_volume"
        if self.seq_volume is not None:
            result["seq_volume"] = f"market/{self.ticker}/seq_volume"
        if self.curve_oi is not None:
            result["curve_oi"] = f"market/{self.ticker}/curve_oi"
        if self.seq_oi is not None:
            result["seq_oi"] = f"market/{self.ticker}/seq_oi"
        return result


class DLYFolderUpdater:
    """
    Batch processor for ALL tickers in a folder, with optional pattern filtering.
      - pattern can be regex (default) or glob ('*', '?', etc).
      - Save into one combined HDF (/market/{ticker}/...) or per-ticker HDF via template.
      - Now supports multi-threading with progress tracking for improved performance.
    """

    def __init__(self, folder: str, yy_pivot: int = 1970):
        self.folder = str(folder)  # Ensure folder is a string
        self.yy_pivot = yy_pivot

        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self._results_lock = threading.Lock()
        self._processed_count = 0
        self._total_count = 0

    def list_tickers(self, pattern: Optional[str] = None, mode: str = "regex", min_contracts: int = 10) -> List[str]:
        mapping = discover_tickers(self.folder)
        all_tickers = sorted(mapping.keys())
        candidate_tickers = filter_tickers(all_tickers, pattern, mode=mode)

        # Filter by minimum contract count
        return [ticker for ticker in candidate_tickers
                if len(mapping.get(ticker, [])) >= min_contracts]

    def _process_single_ticker_worker(self, ticker: str, hdf_path: Optional[str],
                                    per_ticker_hdf_tpl: Optional[str],
                                    results: Dict[str, Any]) -> None:
        """
        Worker function for processing a single ticker in a thread.

        Parameters:
        -----------
        ticker : str
            Ticker symbol to process
        hdf_path : str, optional
            Path to shared HDF file
        per_ticker_hdf_tpl : str, optional
            Template for per-ticker HDF files
        results : Dict[str, Any]
            Shared results dictionary (thread-safe access required)
        """
        start_time = time.time()

        try:
            with self._progress_lock:
                print(f"[THREAD] Starting {ticker}")

            # Create manager and process - UPDATED TO USE NEW CONSTRUCTOR SYNTAX
            if hdf_path:
                # Shared HDF file - will be saved later in thread-safe manner
                mgr = DLYContractManager(ticker, self.folder)
                mgr.run(save=False)
            elif per_ticker_hdf_tpl:
                # Per-ticker HDF file
                path = per_ticker_hdf_tpl.format(ticker=ticker)
                mgr = DLYContractManager(ticker, self.folder)
                mgr.run(save=True)
            else:
                # No saving
                mgr = DLYContractManager(ticker, self.folder)
                mgr.run(save=False)

            # Store the manager result for later HDF saving (if needed)
            with self._results_lock:
                if hdf_path:
                    results["managers"][ticker] = mgr
                    results["by_ticker"][ticker] = {"saved_to": hdf_path}
                elif per_ticker_hdf_tpl:
                    path = per_ticker_hdf_tpl.format(ticker=ticker)
                    results["by_ticker"][ticker] = {"saved_to": path}
                else:
                    results["by_ticker"][ticker] = {"saved_to": None}

                # Update progress
                self._processed_count += 1
                progress_pct = (self._processed_count / self._total_count) * 100
                elapsed = time.time() - start_time

            with self._progress_lock:
                print(f"[{self._processed_count}/{self._total_count}] Completed {ticker} in {elapsed:.1f}s ({progress_pct:.1f}%)")

        except Exception as e:
            elapsed = time.time() - start_time
            error_key = f"{ticker}_ERROR"

            with self._results_lock:
                results["by_ticker"][error_key] = {"error": str(e)}
                self._processed_count += 1

            with self._progress_lock:
                print(f"[{self._processed_count}/{self._total_count}] Failed {ticker} after {elapsed:.1f}s: {e}")

    def run_all(self, hdf_path: Optional[str] = None, per_ticker_hdf_tpl: Optional[str] = None,
                pattern: Optional[str] = None, mode: str = "regex", save: bool = True,
                max_workers: Optional[int] = None, use_threading: bool = True) -> Dict[str, Any]:
        """
        Process all tickers in the folder matching pattern with optional multi-threading.

        Parameters:
        -----------
        hdf_path : str, optional
            If provided, all results go into that single HDF under /market/{ticker}/...
        per_ticker_hdf_tpl : str, optional
            If provided, format with {ticker}, e.g. '/out/{ticker}.h5'
        pattern : str, optional
            Pattern to filter tickers
        mode : str, default 'regex'
            Pattern matching mode ('regex' or 'glob')
        save : bool, default True
            Whether to save results
        max_workers : int, optional
            Maximum number of worker threads. Defaults to CPU count.
        use_threading : bool, default True
            Whether to use multi-threading for processing

        Returns:
        --------
        Dict[str, Any]
            Results with {'tickers': [...], 'by_ticker': {T: {'saved_to': PATH}}}
        """
        mapping = discover_tickers(self.folder)
        all_tickers = sorted(mapping.keys())
        candidate_tickers = filter_tickers(all_tickers, pattern, mode=mode)

        # Filter out tickers with fewer than 10 contracts
        target_tickers = []
        for ticker in candidate_tickers:
            contract_count = len(mapping.get(ticker, []))
            if contract_count >= 10:
                target_tickers.append(ticker)
            else:
                print(f"Skipping {ticker}: only {contract_count} contracts (minimum 10 required)")

        if not target_tickers:
            print("No tickers found to process!")
            return {"tickers": [], "by_ticker": {}}

        # Initialize progress tracking
        self._processed_count = 0
        self._total_count = len(target_tickers)

        results: Dict[str, Any] = {
            "tickers": target_tickers,
            "by_ticker": {},
            "managers": {}  # Store managers for shared HDF saving
        }

        # Determine optimal number of workers
        if use_threading and max_workers is None:
            import os
            max_workers = min(len(target_tickers), os.cpu_count() or 4)

        start_time = time.time()

        if use_threading and len(target_tickers) > 1:
            print(f"Processing {len(target_tickers)} tickers using {max_workers} threads:")
            for ticker in target_tickers:
                contract_count = len(mapping.get(ticker, []))
                print(f"  {ticker} ({contract_count} contracts)")

            # Process tickers using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(self._process_single_ticker_worker,
                                  ticker, hdf_path, per_ticker_hdf_tpl, results): ticker
                    for ticker in target_tickers
                }

                # Wait for completion
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        future.result()  # This will raise any exceptions that occurred
                    except Exception as e:
                        print(f"Thread execution failed for {ticker}: {e}")
        else:
            # Sequential processing
            print(f"Processing {len(target_tickers)} tickers sequentially:")
            for ticker in target_tickers:
                contract_count = len(mapping.get(ticker, []))
                print(f"  {ticker} ({contract_count} contracts)")
                self._process_single_ticker_worker(ticker, hdf_path, per_ticker_hdf_tpl, results)

        # Save to shared HDF file if specified (thread-safe)
        if hdf_path and save:
            print(f"\nSaving all results to shared HDF: {hdf_path}")
            self._save_to_shared_hdf(hdf_path, results)

        # Generate summary
        total_time = time.time() - start_time
        successful = {k: v for k, v in results["by_ticker"].items() if not k.endswith('_ERROR')}
        failed = {k: v for k, v in results["by_ticker"].items() if k.endswith('_ERROR')}

        print(f"\nDLY processing completed in {total_time:.1f} seconds")
        print(f"Successfully processed: {len(successful)} tickers")
        print(f"Failed to process: {len(failed)} tickers")
        if use_threading:
            print(f"Average processing speed: {len(successful) / total_time:.2f} tickers/second")

        if failed:
            print(f"\nFailed tickers:")
            for error_key, error_info in failed.items():
                ticker = error_key.replace('_ERROR', '')
                print(f"  {ticker}: {error_info.get('error', 'Unknown error')}")

        # Clean up managers from results
        results.pop("managers", None)

        return results

    def _save_to_shared_hdf(self, hdf_path: str, results: Dict[str, Any]) -> None:
        """
        Save all processed managers to a shared HDF file in a thread-safe manner.

        Parameters:
        -----------
        hdf_path : str
            Path to the shared HDF file
        results : Dict[str, Any]
            Results dictionary containing processed managers
        """
        from pathlib import Path

        # Create directory if it doesn't exist
        hdf_path_obj = Path(hdf_path)
        hdf_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save all manager results to shared HDF
        with pd.HDFStore(hdf_path, mode='a') as store:
            managers = results.get("managers", {})
            for ticker, mgr in managers.items():
                try:
                    # Save all data types including volume and OI
                    if mgr.curve is not None:
                        store.put(f"market/{ticker}/curve", mgr.curve, format="table", data_columns=True)
                    if mgr.dte is not None:
                        store.put(f"market/{ticker}/dte", mgr.dte, format="table", data_columns=True)
                    if mgr.expiry_series is not None:
                        store.put(f"market/{ticker}/expiry", mgr.expiry_series.to_frame(), format="table", data_columns=True)
                    if mgr.front is not None:
                        store.put(f"market/{ticker}/front", mgr.front.to_frame(), format="table", data_columns=True)
                    if mgr.seq_prices is not None:
                        store.put(f"market/{ticker}/seq_curve", mgr.seq_prices, format="table", data_columns=True)
                    if mgr.seq_labels is not None:
                        store.put(f"market/{ticker}/seq_labels", mgr.seq_labels, format="table", data_columns=True)
                    if mgr.seq_dte is not None:
                        store.put(f"market/{ticker}/seq_dte", mgr.seq_dte, format="table", data_columns=True)

                    # Save volume curves if available
                    if mgr.curve_volume is not None:
                        store.put(f"market/{ticker}/curve_volume", mgr.curve_volume, format="table", data_columns=True)
                    if mgr.seq_volume is not None:
                        store.put(f"market/{ticker}/seq_volume", mgr.seq_volume, format="table", data_columns=True)

                    # Save OI curves if available
                    if mgr.curve_oi is not None:
                        store.put(f"market/{ticker}/curve_oi", mgr.curve_oi, format="table", data_columns=True)
                    if mgr.seq_oi is not None:
                        store.put(f"market/{ticker}/seq_oi", mgr.seq_oi, format="table", data_columns=True)

                    # Save spreads
                    spreads = mgr.spreads_vs_front()
                    if spreads is not None:
                        store.put(f"market/{ticker}/seq_spreads", spreads, format="table", data_columns=True)

                    print(f"Saved {ticker} to HDF")

                except Exception as e:
                    print(f"Error saving {ticker} to HDF: {e}")

    def run_all_fast(self, hdf_path: Optional[str] = None, pattern: Optional[str] = None,
                     max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Fast multi-threaded processing with maximum performance settings.

        Parameters:
        -----------
        hdf_path : str, optional
            Path to shared HDF file for saving results
        pattern : str, optional
            Pattern to filter tickers
        max_workers : int, optional
            Maximum number of worker threads. Defaults to CPU count.

        Returns:
        --------
        Dict[str, Any]
            Processing results
        """
        if max_workers is None:
            import os
            max_workers = os.cpu_count() or 4

        print(f"FAST MODE: Using {max_workers} threads for maximum DLY processing performance")
        return self.run_all(
            hdf_path=hdf_path,
            pattern=pattern,
            save=True,
            max_workers=max_workers,
            use_threading=True
        )