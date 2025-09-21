# fast_scid_reader.py
"""
Fast, zero-copy Python reader for Sierra Chart Intraday (.scid) files.

Key features
------------
- Memory-mapped I/O (mmap) for O(1) open on multi‑GB files
- Auto-detects common fixed-record variants (40B v10, 44B v8)
- Vectorized decoding with NumPy structured dtypes
- Binary search by time; fast slicing to [start,end]
- Optional chunked iteration for low‑RAM hosts
- Clean API: records(), to_numpy(), to_pandas(), iter_chunks(), export_csv()

Assumptions & notes
-------------------
- .scid files may have a header (typically 56 bytes). Auto-detects and skips header.
- 40‑byte variant uses 32‑bit **UNIX seconds** (fits in int32). 44‑byte variant uses
  **SCDateTime days** (Excel-like days since 1899‑12‑30) in float64.
- Little‑endian layout (Windows default). If you have big‑endian data, set `force_variant`
  and adjust dtypes accordingly.

Dependencies: numpy (required), pandas (optional for to_pandas).
"""
from __future__ import annotations

import io
import os
import mmap
import struct
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Sequence, Tuple, Dict, Any

import numpy as np

try:  # optional
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # optional
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore

Variant = Literal["V10_40B", "V8_44B"]

_EPOCH_DAYS_OFFSET = 25569.0  # 1970-01-01 relative to 1899-12-30
_SECONDS_PER_DAY = 86400.0
_MAX_REASONABLE_UNIX = 4102444800  # 2100-01-01
_MIN_REASONABLE_UNIX = 315532800   # 1980-01-01

# NumPy structured dtypes (little-endian) - Compatible with Sierra Chart SCID format
DTYPE_V10_40B = np.dtype([
    ("DateTime", "<i8"),      # int64 microseconds since 1899-12-30 (modern format)
    ("Open", "<f4"),
    ("High", "<f4"),
    ("Low", "<f4"),
    ("Close", "<f4"),
    ("NumTrades", "<u4"),
    ("TotalVolume", "<u4"),
    ("BidVolume", "<u4"),
    ("AskVolume", "<u4"),
])  # 40 bytes

DTYPE_V8_44B = np.dtype([
    ("DateTime", "<f8"),      # float64 days since 1899-12-30 (legacy format)
    ("Open", "<f4"),
    ("High", "<f4"),
    ("Low", "<f4"),
    ("Close", "<f4"),
    ("NumTrades", "<u4"),
    ("TotalVolume", "<u4"),
    ("BidVolume", "<u4"),
    ("AskVolume", "<u4"),
])  # 40 bytes (not 44 as originally thought)


@dataclass(frozen=True)
class Schema:
    variant: Variant
    dtype: np.dtype
    record_size: int
    has_unix_seconds: bool  # True for V10_40B


SCHEMAS = {
    "V10_40B": Schema("V10_40B", DTYPE_V10_40B, 40, False),  # Uses int64 microseconds, not UNIX seconds
    "V8_44B": Schema("V8_44B", DTYPE_V8_44B, 40, False),    # Legacy format, also 40 bytes
}


def _sc_microseconds_to_epoch_ms(microseconds: np.ndarray | int) -> np.ndarray | int:
    """Convert Sierra Chart int64 microseconds (since 1899-12-30) to UNIX epoch milliseconds."""
    # Sierra Chart epoch: 1899-12-30 00:00:00 UTC
    # Unix epoch: 1970-01-01 00:00:00 UTC
    # Difference: 25569 days * 86400 seconds * 1000000 microseconds
    sc_epoch_offset_us = 25569 * 86400 * 1000000

    if isinstance(microseconds, np.ndarray):
        # Convert to epoch microseconds, then to milliseconds
        epoch_us = microseconds - sc_epoch_offset_us
        return (epoch_us // 1000).astype(np.int64)
    else:
        # Single value
        epoch_us = microseconds - sc_epoch_offset_us
        return epoch_us // 1000

def _sc_days_to_epoch_ms(days: np.ndarray | float) -> np.ndarray | int:
    """Convert SCDateTime days (since 1899-12-30) to UNIX epoch milliseconds."""
    if isinstance(days, np.ndarray):
        return ((days - _EPOCH_DAYS_OFFSET) * _SECONDS_PER_DAY * 1000.0).astype(np.int64)
    return int(round((days - _EPOCH_DAYS_OFFSET) * _SECONDS_PER_DAY * 1000.0))


def _validate_unix_seconds(x: int) -> bool:
    return _MIN_REASONABLE_UNIX <= x <= _MAX_REASONABLE_UNIX

def _validate_sc_microseconds(x: int) -> bool:
    # Validate int64 microseconds since 1899-12-30
    # Modern data: ~3e15 to 6e15 microseconds (roughly 1980-2100)
    return 10**14 <= x <= 10**17

def _validate_days(x: float) -> bool:
    # Accept roughly 1930..2100 to be generous
    # 1930-01-01 in SC days ~ 10957, 2100-01-01 ~ 73050
    return 9000.0 <= x <= 80000.0


class FastScidReader:
    """
    High-throughput reader for Sierra Chart .scid files using memory-mapped I/O.

    Parameters
    ----------
    path : str
        Path to .scid file.
    force_variant : Optional[Variant]
        If provided, skip auto-detection and use this schema.
    read_only : bool
        Map the file as read-only (recommended). If False, uses ACCESS_COPY on
        platforms that support it.

    Example
    -------
    >>> rdr = FastScidReader("/path/Data/ESU25-CME.scid").open()
    >>> len(rdr)  # number of records
    12345678
    >>> # Slice by time
    >>> df = rdr.to_pandas(start_ms=1726000000000, end_ms=1726086400000)
    """

    def __init__(self, path: str, *, force_variant: Optional[Variant] = None, read_only: bool = True):
        self.path = os.fspath(path)
        self.force_variant: Optional[Variant] = force_variant
        self.read_only = read_only

        self._fh: Optional[io.BufferedReader] = None
        self._mm: Optional[mmap.mmap] = None
        self._schema: Optional[Schema] = None
        self._np_view: Optional[np.ndarray] = None  # structured array view
        self._header_size: int = 0  # Size of header to skip

    # ------------------------------ context manager ------------------------------
    def __enter__(self) -> "FastScidReader":
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --------------------------------- core API ---------------------------------
    def open(self) -> "FastScidReader":
        """Memory-map the file and prepare a zero-copy NumPy view."""
        if self._mm is not None:
            return self

        size = os.path.getsize(self.path)
        if size == 0:
            raise ValueError("Empty file")

        self._fh = open(self.path, "rb", buffering=0)
        access = mmap.ACCESS_READ if self.read_only else mmap.ACCESS_COPY
        self._mm = mmap.mmap(self._fh.fileno(), length=0, access=access)

        # Detect header and schema
        self._header_size = self._detect_header_size()
        data_size = size - self._header_size

        schema = self._detect_schema(data_size) if self.force_variant is None else SCHEMAS[self.force_variant]
        self._schema = schema

        if data_size % schema.record_size != 0:
            raise ValueError(f"Data size {data_size} (after {self._header_size}-byte header) is not a multiple of record size {schema.record_size}")

        # Zero-copy structured array view over the data portion (skip header)
        self._np_view = np.frombuffer(self._mm, dtype=schema.dtype, offset=self._header_size)
        return self

    def close(self) -> None:
        # Clear references to numpy arrays that might hold buffer references
        if hasattr(self, '_np_view'):
            self._np_view = None

        # Close memory map
        if self._mm is not None:
            try:
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
                self._mm = None

        # Close file handle
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

        # Clear other references
        self._schema = None
        self._header_size = 0

    # --------------------------------- helpers ----------------------------------
    def _detect_header_size(self) -> int:
        """Detect if file has a SCID header and return its size."""
        if not self._fh:
            return 0

        # Check for SCID header signature
        self._fh.seek(0)
        first_4_bytes = self._fh.read(4)

        if first_4_bytes == b'SCID':
            # Read header size from offset 4 (little-endian uint32)
            header_size_bytes = self._fh.read(4)
            if len(header_size_bytes) == 4:
                header_size = struct.unpack('<I', header_size_bytes)[0]
                # Validate header size is reasonable (typically 56 bytes)
                if 40 <= header_size <= 4096 and header_size % 4 == 0:
                    return header_size
            # Fallback to typical header size
            return 56

        # No header detected
        return 0

    def _detect_schema(self, data_size: int) -> Schema:
        """Heuristically choose between int64 microseconds (modern) and float64 days (legacy)."""
        # Both formats use 40-byte records
        if data_size % 40 != 0:
            raise ValueError(f"Data size {data_size} is not a multiple of 40-byte records")

        # Try int64 microseconds (modern format)
        microseconds_plausible = False
        microseconds_first = None
        microseconds_last = None

        # Read first and last int64 timestamps (accounting for header offset)
        first_bytes = self._peek(self._header_size, 8)
        last_bytes = self._peek(self._header_size + data_size - 40, 8)
        if first_bytes and last_bytes:
            microseconds_first = struct.unpack("<q", first_bytes)[0]  # int64
            microseconds_last = struct.unpack("<q", last_bytes)[0]
            microseconds_plausible = (_validate_sc_microseconds(microseconds_first) and
                                    _validate_sc_microseconds(microseconds_last) and
                                    microseconds_last >= microseconds_first)

        # Try float64 days (legacy format)
        days_plausible = False
        days_first = None
        days_last = None

        if first_bytes and last_bytes:
            days_first = struct.unpack("<d", first_bytes)[0]  # float64
            days_last = struct.unpack("<d", last_bytes)[0]
            days_plausible = (_validate_days(days_first) and
                            _validate_days(days_last) and
                            days_last >= days_first)

        # Decide between formats
        if microseconds_plausible and not days_plausible:
            return SCHEMAS["V10_40B"]  # Modern int64 microseconds
        if days_plausible and not microseconds_plausible:
            return SCHEMAS["V8_44B"]   # Legacy float64 days
        if microseconds_plausible and days_plausible:
            # Both look plausible - prefer modern format
            return SCHEMAS["V10_40B"]

        # Neither format looks plausible - default to modern
        return SCHEMAS["V10_40B"]

    def _peek(self, offset: int, n: int) -> bytes:
        assert self._fh is not None
        self._fh.seek(offset)
        return self._fh.read(n)

    # -------------------------------- query API ---------------------------------
    @property
    def schema(self) -> Schema:
        if self._schema is None:
            raise RuntimeError("Reader not opened")
        return self._schema

    def __len__(self) -> int:
        if self._schema is None:
            return 0
        size = os.path.getsize(self.path)
        data_size = size - self._header_size
        return data_size // self._schema.record_size

    @property
    def view(self) -> np.ndarray:
        """Structured array view over the file (zero-copy)."""
        if self._np_view is None:
            raise RuntimeError("Reader not opened")
        return self._np_view

    def columns(self) -> Tuple[str, ...]:
        return tuple(self.view.dtype.names or ())

    # ------------------------------ time handling -------------------------------
    def times_epoch_ms(self) -> np.ndarray:
        """Return vector of timestamps in UNIX epoch milliseconds (NumPy int64).
        This allocates once (derived from the on-disk column).
        """
        v = self.view
        if self.schema.variant == "V10_40B":
            # Modern format: int64 microseconds since 1899-12-30
            microseconds = v["DateTime"].astype(np.int64, copy=True)  # Force copy
            return _sc_microseconds_to_epoch_ms(microseconds)
        else:
            # Legacy format: float64 days since 1899-12-30
            days = v["DateTime"].astype(np.float64, copy=True)  # Force copy
            return _sc_days_to_epoch_ms(days)  # int64 array

    def searchsorted(self, ts_ms: int, *, side: Literal["left", "right"] = "left") -> int:
        """Binary search index for a timestamp in epoch milliseconds.
        Assumes data are non-decreasing in time (typical for .scid).
        """
        t = self.times_epoch_ms()
        return int(np.searchsorted(t, ts_ms, side=side))

    def slice_by_time(self, *, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> slice:
        t0 = 0 if start_ms is None else self.searchsorted(start_ms, side="left")
        t1 = len(self) if end_ms is None else self.searchsorted(end_ms, side="right")
        return slice(t0, t1)

    # ------------------------------ data extractors -----------------------------
    def to_numpy(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        copy: bool = True,  # Default to True for safety
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (data, times_ms) where `data` is a 2D float64 array (n x k) for requested columns.
        If `columns` is None, defaults to OHLCV (Open, High, Low, Close, TotalVolume).
        Set `copy=False` to attempt zero-copy (views) when possible for numeric types.
        WARNING: copy=False may cause BufferError on file close if references are held.
        """
        if columns is None:
            columns = ("Open", "High", "Low", "Close", "TotalVolume")
        sl = self.slice_by_time(start_ms=start_ms, end_ms=end_ms)
        v = self.view[sl]
        # Build a 2D array efficiently
        mats = []
        for c in columns:
            col = v[c]
            mats.append(col.astype(np.float64, copy=copy))
        data = np.stack(mats, axis=1) if mats else np.empty((len(v), 0), dtype=np.float64)
        times = self.times_epoch_ms()[sl]
        return data, times

    def to_pandas(
        self,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        tz: Optional[str] = None,
    ):
        if pd is None:
            raise RuntimeError("pandas is not installed; run `pip install pandas`")
        sl = self.slice_by_time(start_ms=start_ms, end_ms=end_ms)
        v = self.view[sl]
        t = self.times_epoch_ms()[sl]
        idx = pd.to_datetime(t, unit="ms", utc=True)
        if tz:
            idx = idx.tz_convert(tz)

        # Force copies to avoid holding references to memory-mapped buffer
        data_dict = {}
        for name in (columns or ["Open", "High", "Low", "Close", "NumTrades", "TotalVolume", "BidVolume", "AskVolume"]):
            if name in v.dtype.names:
                data_dict[name] = v[name].copy()  # Force copy

        frame = pd.DataFrame(data_dict, index=idx)
        frame.index.name = "DateTime"
        return frame

    # ------------------------------ iteration utils -----------------------------
    def iter_chunks(self, *, chunk_records: int = 1_000_000, copy: bool = False) -> Iterator[np.ndarray]:
        """Yield structured-array chunks (zero-copy slices) of at most `chunk_records` length.

        Args:
            chunk_records: Maximum records per chunk
            copy: If True, yield copies instead of views (safer for concurrent usage)
        """
        n = len(self)
        v = self.view
        for start in range(0, n, chunk_records):
            end = min(start + chunk_records, n)
            chunk = v[start:end]
            yield chunk.copy() if copy else chunk

    # ------------------------------- export utils -------------------------------
    def export_csv(
        self,
        out_path: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        include_columns: Sequence[str] = ("Open", "High", "Low", "Close", "NumTrades", "TotalVolume", "BidVolume", "AskVolume"),
        include_time: bool = True,
        chunk_records: int = 5_000_000,
    ) -> None:
        """High-speed CSV export using chunked writes. Suitable for multi-GB files.
        Writes header. Time column is ISO-8601 in UTC.
        """
        if pd is None:
            # pandas-free path: format with NumPy and Python I/O
            self._export_csv_numpy(out_path, start_ms=start_ms, end_ms=end_ms,
                                   include_columns=include_columns, include_time=include_time,
                                   chunk_records=chunk_records)
            return
        sl = self.slice_by_time(start_ms=start_ms, end_ms=end_ms)
        cols = list(include_columns)
        with open(out_path, "w", newline="") as f:
            # header
            header = (["DateTime"] if include_time else []) + cols
            f.write(",".join(header) + "\n")
            # chunks
            start = sl.start or 0
            stop = sl.stop or len(self)
            for chunk_start in range(start, stop, chunk_records):
                chunk_end = min(chunk_start + chunk_records, stop)
                c = self.view[chunk_start:chunk_end]
                if c.size == 0:
                    continue
                t_ms = (self.times_epoch_ms())[chunk_start:chunk_end]
                df = pd.DataFrame({name: c[name] for name in cols})
                if include_time:
                    dt = pd.to_datetime(t_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S.%fZ")
                    df.insert(0, "DateTime", dt.values)
                df.to_csv(f, header=False, index=False, lineterminator="\n")

    def _export_csv_numpy(
        self,
        out_path: str,
        *,
        start_ms: Optional[int],
        end_ms: Optional[int],
        include_columns: Sequence[str],
        include_time: bool,
        chunk_records: int,
    ) -> None:
        sl = self.slice_by_time(start_ms=start_ms, end_ms=end_ms)
        cols = list(include_columns)
        with open(out_path, "w", newline="") as f:
            header = (["DateTime"] if include_time else []) + cols
            f.write(",".join(header) + "\n")
            n = len(self)
            start = sl.start or 0
            stop = sl.stop or n
            for chunk_start in range(start, stop, chunk_records):
                chunk_end = min(chunk_start + chunk_records, stop)
                block = self.view[chunk_start:chunk_end]
                if block.size == 0:
                    continue
                arrs = [block[name] for name in cols]
                mat = np.column_stack(arrs)
                if include_time:
                    t_ms = self.times_epoch_ms()[chunk_start:chunk_end]
                    # Use numpy datetime64 for speed then format as ISO Z
                    dt64 = t_ms.astype("datetime64[ms]")
                    # numpy prints ISO without 'Z'; append 'Z'
                    date_str = np.array([str(x).replace(" ", "T") + "Z" for x in dt64], dtype=object)
                    out = np.column_stack((date_str, mat))
                else:
                    out = mat
                for row in out:
                    f.write(",".join(map(str, row)))
                    f.write("\n")

    def export_to_parquet(
        self,
        out_path: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        include_columns: Sequence[str] = ("Open", "High", "Low", "Close", "NumTrades", "TotalVolume", "BidVolume", "AskVolume"),
        chunk_records: int = 1_000_000,
        compression: str = "snappy",
        include_time: bool = True,
    ) -> None:
        """
        Export data to Parquet format using chunked processing for memory efficiency.

        Suitable for multi-GB files without loading entire dataset into memory.
        Uses PyArrow for efficient columnar storage and compression.

        Args:
            out_path: Output Parquet file path
            start_ms: Start timestamp in epoch milliseconds (None = from beginning)
            end_ms: End timestamp in epoch milliseconds (None = to end)
            include_columns: Columns to include in export
            chunk_records: Records per chunk (controls memory usage)
            compression: Parquet compression ('snappy', 'gzip', 'lz4', 'brotli', 'zstd')
            include_time: Include DateTime column

        Raises:
            RuntimeError: If PyArrow is not installed
        """
        if pa is None or pq is None:
            raise RuntimeError("PyArrow is required for Parquet export. Install with: pip install pyarrow")

        sl = self.slice_by_time(start_ms=start_ms, end_ms=end_ms)
        cols = list(include_columns)

        # Build Arrow schema
        schema_fields = []
        if include_time:
            schema_fields.append(pa.field("DateTime", pa.timestamp("ms", tz="UTC")))

        for col in cols:
            if col in ["Open", "High", "Low", "Close"]:
                schema_fields.append(pa.field(col, pa.float32()))
            elif col in ["NumTrades", "TotalVolume", "BidVolume", "AskVolume"]:
                schema_fields.append(pa.field(col, pa.uint32()))
            else:
                # Default to float32 for unknown columns
                schema_fields.append(pa.field(col, pa.float32()))

        schema = pa.schema(schema_fields)

        # Initialize Parquet writer
        start = sl.start or 0
        stop = sl.stop or len(self)

        with pq.ParquetWriter(out_path, schema, compression=compression) as writer:
            # Process in chunks to control memory usage
            for chunk_start in range(start, stop, chunk_records):
                chunk_end = min(chunk_start + chunk_records, stop)
                chunk_view = self.view[chunk_start:chunk_end]

                if chunk_view.size == 0:
                    continue

                # Build Arrow arrays for this chunk
                arrays = []

                if include_time:
                    # Get timestamps for this chunk
                    chunk_times = self.times_epoch_ms()[chunk_start:chunk_end]
                    # Convert to Arrow timestamp array
                    time_array = pa.array(chunk_times, type=pa.timestamp("ms", tz="UTC"))
                    arrays.append(time_array)

                # Add data columns
                for col in cols:
                    if col in chunk_view.dtype.names:
                        col_data = chunk_view[col]

                        # Convert to appropriate Arrow type
                        if col in ["Open", "High", "Low", "Close"]:
                            arrow_array = pa.array(col_data.astype(np.float32))
                        elif col in ["NumTrades", "TotalVolume", "BidVolume", "AskVolume"]:
                            arrow_array = pa.array(col_data.astype(np.uint32))
                        else:
                            arrow_array = pa.array(col_data.astype(np.float32))

                        arrays.append(arrow_array)
                    else:
                        # Column not found, create null array
                        null_array = pa.nulls(len(chunk_view), type=schema.field(col).type)
                        arrays.append(null_array)

                # Create record batch and write to file
                batch = pa.record_batch(arrays, schema=schema)
                writer.write_batch(batch)

    def export_to_parquet_optimized(
        self,
        out_path: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        include_columns: Sequence[str] = ("Open", "High", "Low", "Close", "NumTrades", "TotalVolume", "BidVolume", "AskVolume"),
        chunk_records: int = 2_000_000,
        compression: str = "zstd",
        include_time: bool = True,
        use_dictionary: bool = False,
    ) -> Dict[str, Any]:
        """
        Optimized Parquet export with advanced features and statistics.

        Returns metadata about the export process including compression ratios,
        processing time, and file statistics.

        Args:
            out_path: Output Parquet file path
            start_ms: Start timestamp in epoch milliseconds
            end_ms: End timestamp in epoch milliseconds
            include_columns: Columns to include
            chunk_records: Records per chunk (larger = better compression, more memory)
            compression: Compression algorithm ('zstd', 'snappy', 'gzip', 'lz4', 'brotli')
            include_time: Include DateTime column
            use_dictionary: Use dictionary encoding for string-like data

        Returns:
            Dictionary with export statistics
        """
        if pa is None or pq is None:
            raise RuntimeError("PyArrow is required for Parquet export. Install with: pip install pyarrow")

        import time
        start_time = time.perf_counter()

        sl = self.slice_by_time(start_ms=start_ms, end_ms=end_ms)
        cols = list(include_columns)

        # Build optimized Arrow schema with metadata
        schema_fields = []
        if include_time:
            schema_fields.append(pa.field("DateTime", pa.timestamp("us", tz="UTC")))  # Microsecond precision

        for col in cols:
            if col in ["Open", "High", "Low", "Close"]:
                field_type = pa.float32()
            elif col in ["NumTrades", "TotalVolume", "BidVolume", "AskVolume"]:
                field_type = pa.uint32()
            else:
                field_type = pa.float32()

            # Add dictionary encoding if requested
            if use_dictionary and col in ["NumTrades"]:  # Example: encode trade counts
                field_type = pa.dictionary(pa.int16(), field_type)

            schema_fields.append(pa.field(col, field_type))

        # Add metadata to schema
        metadata = {
            "source": "FastScidReader",
            "variant": self.schema.variant,
            "export_time": str(datetime.now()),
            "compression": compression
        }
        schema = pa.schema(schema_fields, metadata=metadata)

        # Export statistics
        stats = {
            "total_records": 0,
            "chunks_processed": 0,
            "bytes_processed": 0,
            "compression_ratio": 0.0,
            "export_time": 0.0
        }

        start_idx = sl.start or 0
        stop_idx = sl.stop or len(self)
        total_records = stop_idx - start_idx

        # Parquet writer with optimized settings
        writer_kwargs = {
            "compression": compression,
            "use_dictionary": use_dictionary,
            "row_group_size": chunk_records,  # Optimize row group size
            "data_page_size": 1024 * 1024,   # 1MB pages
        }

        with pq.ParquetWriter(out_path, schema, **writer_kwargs) as writer:
            for chunk_start in range(start_idx, stop_idx, chunk_records):
                chunk_end = min(chunk_start + chunk_records, stop_idx)
                chunk_view = self.view[chunk_start:chunk_end]

                if chunk_view.size == 0:
                    continue

                arrays = []

                if include_time:
                    # High-precision timestamps
                    chunk_times = self.times_epoch_ms()[chunk_start:chunk_end]
                    # Convert to microseconds for higher precision
                    time_us = chunk_times * 1000
                    time_array = pa.array(time_us, type=pa.timestamp("us", tz="UTC"))
                    arrays.append(time_array)

                # Process data columns
                for col in cols:
                    if col in chunk_view.dtype.names:
                        col_data = chunk_view[col]

                        if col in ["Open", "High", "Low", "Close"]:
                            arrow_array = pa.array(col_data.astype(np.float32))
                        elif col in ["NumTrades", "TotalVolume", "BidVolume", "AskVolume"]:
                            arrow_array = pa.array(col_data.astype(np.uint32))
                        else:
                            arrow_array = pa.array(col_data.astype(np.float32))

                        arrays.append(arrow_array)

                # Write batch
                batch = pa.record_batch(arrays, schema=schema)
                writer.write_batch(batch)

                # Update statistics
                stats["chunks_processed"] += 1
                stats["total_records"] += len(chunk_view)
                stats["bytes_processed"] += chunk_view.nbytes

        # Calculate final statistics
        export_time = time.perf_counter() - start_time
        stats["export_time"] = export_time

        # Get file size for compression ratio
        output_size = os.path.getsize(out_path)
        if stats["bytes_processed"] > 0:
            stats["compression_ratio"] = stats["bytes_processed"] / output_size

        stats["output_size_mb"] = output_size / (1024 * 1024)
        stats["records_per_second"] = total_records / export_time if export_time > 0 else 0

        return stats


# ------------------------------ small CLI helper ------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fast .scid reader")
    ap.add_argument("path", help="Path to .scid file")
    ap.add_argument("--info", action="store_true", help="Print detected schema and record count")
    ap.add_argument("--export", metavar="CSV", help="Export to CSV path")
    ap.add_argument("--start", type=str, default=None, help="Start ISO (UTC) e.g. 2025-09-19T00:00:00Z")
    ap.add_argument("--end", type=str, default=None, help="End ISO (UTC)")
    args = ap.parse_args()

    def parse_iso_to_ms(s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        s = s.rstrip("Z")
        ts = np.datetime64(s)
        return int(ts.astype("datetime64[ms]").astype(np.int64))

    rdr = FastScidReader(args.path).open()
    if args.info:
        print(f"Variant: {rdr.schema.variant}; records: {len(rdr)}; columns: {rdr.columns()}")

    if args.export:
        start_ms = parse_iso_to_ms(args.start)
        end_ms = parse_iso_to_ms(args.end)
        rdr.export_csv(args.export, start_ms=start_ms, end_ms=end_ms)
        print(f"Exported to {args.export}")
