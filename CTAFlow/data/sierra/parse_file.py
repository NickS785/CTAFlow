# scid_reader.py
# A fast, zero-copy .scid reader with pandas-friendly conversions.
# Requires: numpy, pandas

from __future__ import annotations
import io, os, mmap
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Optional, Tuple, List
import numpy as np
import pandas as pd

# -------------------------
# Constants / dtypes
# -------------------------
_SC_EPOCH_UTC = pd.Timestamp("1899-12-30T00:00:00Z")  # Sierra SCDateTime epoch (UTC)
_SC_EPOCH_NAIVE = pd.Timestamp("1899-12-30")  # naive, localize later

HEADER_DTYPE = np.dtype([
    ("FileTypeUniqueHeaderID", "S4"),   # b"SCID"
    ("HeaderSize", "<u4"),
    ("RecordSize", "<u4"),
    ("Version",    "<u2"),              # not the app version; struct version
    ("Unused1",    "<u2"),
    ("UTCStartIndex", "<u4"),
    ("Reserve",    "S36"),
])

# Variant A: modern (since Sierra build ~2151): int64 microseconds
RECORD_DTYPE_INT = np.dtype([
    ("DateTime",   "<i8"),   # int64 microseconds since 1899-12-30 (UTC)
    ("Open",       "<f4"),
    ("High",       "<f4"),
    ("Low",        "<f4"),
    ("Close",      "<f4"),
    ("NumTrades",  "<u4"),
    ("TotalVolume","<u4"),
    ("BidVolume",  "<u4"),
    ("AskVolume",  "<u4"),
])

# Variant B: legacy: double (8 bytes) = days since 1899-12-30
RECORD_DTYPE_DBL = np.dtype([
    ("DateTime",   "<f8"),   # float64 "days since 1899-12-30"
    ("Open",       "<f4"),
    ("High",       "<f4"),
    ("Low",        "<f4"),
    ("Close",      "<f4"),
    ("NumTrades",  "<u4"),
    ("TotalVolume","<u4"),
    ("BidVolume",  "<u4"),
    ("AskVolume",  "<u4"),
])

# Special markers (per doc) for tick-with-bid/ask and CME unbundling flags.
SINGLE_TRADE_WITH_BID_ASK = np.float32(0.0)
FIRST_SUB_TRADE_OF_UNBUNDLED_TRADE = np.float32(-1.99900095e+37)
LAST_SUB_TRADE_OF_UNBUNDLED_TRADE  = np.float32(-1.99900197e+37)


def _plausible_header_size(n: int) -> bool:
    """Check if header size is reasonable."""
    return 40 <= n <= 4096 and (n % 4 == 0)


@dataclass
class ScidHeader:
    filetype: bytes
    header_size: int
    record_size: int
    version: int
    utc_start_index: int

    @classmethod
    def from_bytes(cls, b: bytes) -> "ScidHeader":
        arr = np.frombuffer(b, dtype=HEADER_DTYPE, count=1)[0]
        return cls(
            filetype=arr["FileTypeUniqueHeaderID"],
            header_size=int(arr["HeaderSize"]),
            record_size=int(arr["RecordSize"]),
            version=int(arr["Version"]),
            utc_start_index=int(arr["UTCStartIndex"]),
        )


class ScidReader:
    """
    Memory-mapped .scid reader.

    - Auto-detects DateTime encoding: 'int64_us' (modern) vs 'double_days' (legacy).
    - Streams records without copying; converts to pandas on demand.
    - Assumes little-endian platform (Sierra files are LE).

    Use as context manager for automatic cleanup:
        with ScidReader(path) as reader:
            df = reader.to_dataframe()
    """

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._fd = open(self.path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)

        # Parse header
        header_bytes = self._mm[:HEADER_DTYPE.itemsize]
        hdr = ScidHeader.from_bytes(header_bytes)
        self.header = hdr

        if hdr.filetype != b"SCID":
            raise ValueError(f"{self.path}: not a SCID file (magic={hdr.filetype!r})")

        # Trust but verify; fall back to 56 if header reports nonsense
        self._header_size = hdr.header_size if _plausible_header_size(hdr.header_size) else HEADER_DTYPE.itemsize
        self._record_size = hdr.record_size if hdr.record_size in (RECORD_DTYPE_INT.itemsize, RECORD_DTYPE_DBL.itemsize) else 40

        # Map the record region as a raw byte view
        self._rec_base = self._header_size
        self._nbytes = max(0, len(self._mm) - self._rec_base)
        self._nrecs = self._nbytes // self._record_size
        self._nbytes = self._nrecs * self._record_size

        self._raw = np.frombuffer(self._mm, dtype=np.uint8, count=self._nbytes, offset=self._rec_base)

        # Decide DateTime encoding and scale
        self._variant, self._scale = self._auto_variant_and_scale()

    def close(self):
        # Clear references to numpy arrays that might hold buffer references
        if hasattr(self, '_raw'):
            del self._raw

        try:
            if hasattr(self, '_mm') and self._mm:
                self._mm.close()
        except BufferError:
            # Force garbage collection to clear any lingering references
            import gc
            gc.collect()
            try:
                self._mm.close()
            except BufferError:
                # If still failing, just close the file descriptor
                pass
        finally:
            if hasattr(self, '_fd') and self._fd:
                self._fd.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def peek(self, n: int = 3) -> dict:
        """Return a small dict with header/offsets and first few raw DateTimes under both interpretations."""
        if self._nrecs == 0:
            return {
                "header_size": self._header_size,
                "record_size": self._record_size,
                "n_records": self._nrecs,
                "int64_us_first": [],
                "double_days_first": [],
            }

        n = min(n, self._nrecs)
        view_i = self._raw.view(RECORD_DTYPE_INT)[:n]
        view_d = self._raw.view(RECORD_DTYPE_DBL)[:n]
        return {
            "header_size": self._header_size,
            "record_size": self._record_size,
            "n_records": self._nrecs,
            "int64_us_first": view_i["DateTime"].astype(np.int64).tolist(),
            "double_days_first": view_d["DateTime"].astype(float).tolist(),
        }

    # ------------ Variant detection ------------
    def _view_int(self) -> np.ndarray:
        return self._raw.view(RECORD_DTYPE_INT)

    def _view_dbl(self) -> np.ndarray:
        return self._raw.view(RECORD_DTYPE_DBL)

    def _auto_variant_and_scale(self):
        """
        Decide between int64 microseconds vs double days, and detect a mis-scaled int64
        (us / ms / s) by order of magnitude. Returns (variant, scale_string).
        """
        if self._nrecs == 0:
            return "int64_us", "us"

        n = min(self._nrecs, 16)
        di = self._raw.view(RECORD_DTYPE_INT)["DateTime"][:n].astype(np.int64, copy=False)
        dd = self._raw.view(RECORD_DTYPE_DBL)["DateTime"][:n].astype(np.float64, copy=False)

        def plaus_us(x: np.ndarray) -> bool:
            # microseconds since 1899 for modern data ~ 3e15 to 6e15; monotone nondecreasing usually holds
            if x.size == 0:
                return False
            xmin, xmax = int(x.min()), int(x.max())
            return (10**14 <= xmin <= 10**17) and (xmin <= xmax)

        def plaus_days(x: np.ndarray) -> bool:
            # days since 1899 for 1980–2100 ~ [29400, 73000]
            if x.size == 0:
                return False
            xmin, xmax = float(x.min()), float(x.max())
            return (2.0e4 <= xmin <= 1.0e5) and (xmin <= xmax)

        if plaus_us(di):
            # Also detect if provider accidentally wrote ms or s instead of us
            xmin = int(di.min())
            # thresholds (~1899->2025): us ~4e15, ms ~4e12, s ~4e9
            # But be more conservative about ms detection to avoid overflow
            if xmin < 10**11:  # Very small values suggest seconds
                return "int64_us", "s"
            elif 10**11 <= xmin < 10**14:  # Medium values suggest milliseconds
                return "int64_us", "ms"
            else:  # Large values are likely microseconds (standard)
                return "int64_us", "us"
        elif plaus_days(dd):
            return "double_days", "days"
        else:
            # Fallback: prefer int64_us and let consumer inspect peek()
            return "int64_us", "us"

    # ------------ Public API ------------
    @property
    def n_records(self) -> int:
        return self._nrecs

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def scale(self) -> str:
        """Return the detected time scale (us/ms/s/days)."""
        return getattr(self, "_scale", "us")

    def _to_datetime_from_int(self, di: np.ndarray) -> pd.DatetimeIndex:
        """Convert int64 DateTime values to pandas DatetimeIndex using detected scale."""
        scale = getattr(self, "_scale", "us")

        # Convert all to microseconds first, then to datetime
        # This avoids overflow issues with large millisecond values
        if scale == "us":
            di_us = di.astype("int64", copy=False)
        elif scale == "ms":
            # Check for potential overflow before multiplication
            max_safe_ms = 9223372036854775 // 1000  # Max int64 / 1000
            if np.any(di > max_safe_ms):
                # Values too large for ms, likely actually microseconds
                di_us = di.astype("int64", copy=False)  # Treat as microseconds
            else:
                di_us = di.astype("int64", copy=False) * 1000  # ms to us
        elif scale == "s":
            # Check for potential overflow before multiplication
            max_safe_s = 9223372036854775 // 1_000_000  # Max int64 / 1M
            if np.any(di > max_safe_s):
                # Values too large for seconds, likely microseconds
                di_us = di.astype("int64", copy=False)  # Treat as microseconds
            else:
                di_us = di.astype("int64", copy=False) * 1_000_000  # s to us
        else:
            di_us = di.astype("int64", copy=False)

        try:
            # Convert microseconds to timedelta
            td = pd.to_timedelta(di_us, unit="us")
            return (_SC_EPOCH_NAIVE + td).tz_localize("UTC")
        except (OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta):
            # Last resort: use numpy datetime64 directly
            # Convert to seconds to avoid overflow
            seconds = di_us.astype("float64") / 1_000_000.0

            # Calculate epoch offset in seconds
            epoch_seconds = (_SC_EPOCH_NAIVE - pd.Timestamp("1970-01-01")).total_seconds()
            unix_seconds = epoch_seconds + seconds

            # Convert to datetime64[s] then to nanoseconds
            dt_s = unix_seconds.astype("datetime64[s]")
            dt_ns = dt_s.astype("datetime64[ns]")
            return pd.DatetimeIndex(dt_ns).tz_localize("UTC")

    def _to_datetime_from_double_days(self, dd: np.ndarray) -> pd.DatetimeIndex:
        """Convert double DateTime values (days since epoch) to pandas DatetimeIndex."""
        td = pd.to_timedelta(dd.astype("float64", copy=False), unit="D")
        return (_SC_EPOCH_NAIVE + td).tz_localize("UTC")

    def iter_chunks(self, chunk_size: int = 1_000_000) -> Iterator[np.ndarray]:
        """
        Yield record chunks as numpy structured arrays (no copies).
        """
        if self._variant == "int64_us":
            view = self._view_int()
        else:
            view = self._view_dbl()
        for start in range(0, self._nrecs, chunk_size):
            stop = min(self._nrecs, start + chunk_size)
            yield view[start:stop]

    def to_dataframe(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        fields: Optional[Tuple[str, ...]] = None,
        price_scale: Optional[float] = None,   # e.g., 100 for prices stored as integer pennies
        decode_tick_bidask: bool = True,
    ) -> pd.DataFrame:
        """
        Load all records (filtered to [start, end]) into a pandas DataFrame.

        fields: subset of {"Open","High","Low","Close","NumTrades","TotalVolume","BidVolume","AskVolume"}
        price_scale: optional divisor if your feed stores integer-like prices in the float slots.
        decode_tick_bidask: when Open==0.0, expose Ask=High, Bid=Low, Trade=Close, and 'aggressor' flag.
        """
        # Concatenate in chunks (keeps memory reasonable for very large files)
        parts = []
        for rec in self.iter_chunks():
            if self._variant == "int64_us":
                ts = self._to_datetime_from_int(rec["DateTime"])
            else:
                ts = self._to_datetime_from_double_days(rec["DateTime"])

            df = pd.DataFrame({
                "DateTime": ts,
                "Open": rec["Open"].astype(np.float64, copy=True),
                "High": rec["High"].astype(np.float64, copy=True),
                "Low":  rec["Low"].astype(np.float64, copy=True),
                "Close": rec["Close"].astype(np.float64, copy=True),
                "NumTrades": rec["NumTrades"].astype(np.uint64, copy=True),
                "TotalVolume": rec["TotalVolume"].astype(np.uint64, copy=True),
                "BidVolume": rec["BidVolume"].astype(np.uint64, copy=True),
                "AskVolume": rec["AskVolume"].astype(np.uint64, copy=True),
            })

            parts.append(df)

        if not parts:
            return pd.DataFrame(columns=["DateTime"])

        out = pd.concat(parts, axis=0, ignore_index=True)
        # UTC index
        out = out.set_index("DateTime").sort_index()

        # Time filter
        if start is not None:
            out = out.loc[pd.Timestamp(start, tz="UTC"):]
        if end is not None:
            out = out.loc[:pd.Timestamp(end, tz="UTC")]

        # Optional price scaling
        if price_scale and price_scale != 1.0:
            for c in ("Open","High","Low","Close"):
                out[c] = out[c] / price_scale

        # Optional field selection
        if fields:
            keep = set(fields) | set()  # ensure set
            # Always keep index; nothing to add

            cols = [c for c in out.columns if c in keep]
            out = out[cols]

        # Optional tick-with-bid/ask decode (Open == 0 per spec)
        if decode_tick_bidask:
            is_tick_ba = (out.get("Open") is not None) and (out["Open"] == float(SINGLE_TRADE_WITH_BID_ASK))
            if "High" in out and "Low" in out and "Close" in out:
                out.loc[is_tick_ba, "Ask"] = out.loc[is_tick_ba, "High"]
                out.loc[is_tick_ba, "Bid"] = out.loc[is_tick_ba, "Low"]
                out.loc[is_tick_ba, "Trade"] = out.loc[is_tick_ba, "Close"]

            # Aggressor from BidVolume/AskVolume when NumTrades==1 (per docs)
            if {"BidVolume","AskVolume","NumTrades"}.issubset(out.columns):
                one_trade = out["NumTrades"] == 1
                buy = one_trade & (out["AskVolume"] > 0)
                sell = one_trade & (out["BidVolume"] > 0)
                out.loc[buy, "Aggressor"] = "BUY"
                out.loc[sell, "Aggressor"] = "SELL"

        return out

    def resample_ohlcv(self, df: pd.DataFrame, rule: str = "1T") -> pd.DataFrame:
        """
        Resample a tick/second stream into OHLCV using pandas (UTC index required).
        """
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        o = df["Open"].resample(rule).first()
        h = df["High"].resample(rule).max()
        l = df["Low"].resample(rule).min()
        c = df["Close"].resample(rule).last()
        v = df["TotalVolume"].resample(rule).sum()
        nt = df["NumTrades"].resample(rule).sum() if "NumTrades" in df else None
        out = pd.concat({"Open":o,"High":h,"Low":l,"Close":c,"TotalVolume":v}, axis=1)
        if nt is not None:
            out["NumTrades"] = nt
        return out.dropna(how="all")


# -------------------------
# Convenience functions
# -------------------------
def read_scid(
    path: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    fields: Optional[Tuple[str, ...]] = None,
    price_scale: Optional[float] = None,
    decode_tick_bidask: bool = True
) -> pd.DataFrame:
    """
    One-shot helper: open, read, close using context manager for safe cleanup.
    """
    with ScidReader(path) as reader:
        return reader.to_dataframe(start=start, end=end, fields=fields,
                                   price_scale=price_scale,
                                   decode_tick_bidask=decode_tick_bidask)


