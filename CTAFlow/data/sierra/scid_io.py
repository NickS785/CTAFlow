"""Async SCID IO: read & write concurrently across many .scid files.

This module adds **asynchronous orchestration** around the high performance
:class:`~CTAFlow.data.sierra.fast_parse.FastScidReader` so multiple SCID files
can be read and written concurrently without blocking the asyncio event loop.

It mirrors the API described in the product brief: build :class:`PipelineSpec`
instances, then call :func:`run_pipelines` to execute bounded producer/consumer
pipelines backed by a shared :class:`~concurrent.futures.ThreadPoolExecutor`.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:  # Optional dependencies
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # parquet optional
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore

from .fast_parse import FastScidReader


# ------------------------------- Models ------------------------------------
@dataclass
class ReadSpec:
    path: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    columns: Optional[Sequence[str]] = None
    chunk_records: int = 2_000_000
    copy_chunks: bool = True  # copy-out buffers before writer sees them


@dataclass
class WriteSpec:
    out_path: str
    format: str = "parquet"  # or "csv"
    include_time: bool = True
    parquet_compression: str = "zstd"
    csv_separator: str = ","
    csv_shard_rows: Optional[int] = None
    csv_header: bool = True


@dataclass
class PipelineSpec:
    read: ReadSpec
    write: WriteSpec
    queue_maxsize: int = 4


# Optional mapping if .view uses different field names
DEFAULT_COLUMNS = (
    "Open", "High", "Low", "Close",
    "NumTrades", "TotalVolume", "BidVolume", "AskVolume",
)


# ---------------------- Async Reader (producer) -----------------------------
class AsyncScidReader:
    """Async adapter around blocking :class:`FastScidReader`.

    The reader performs mmap/NumPy work inside a thread pool and pushes chunks
    into an :class:`asyncio.Queue`. Each chunk is a dict containing ``data``
    (``np.ndarray``), ``times`` (epoch ms) and metadata fields.
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None,
                 executor: Optional[concurrent.futures.Executor] = None) -> None:
        if FastScidReader is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "FastScidReader import failed; ensure fast_parse.py is on PYTHONPATH."
            )
        self.loop = loop or asyncio.get_event_loop()
        self.executor = executor

    async def _open(self, path: str):
        def _do():
            rdr = FastScidReader(path).open()
            return rdr
        return await self.loop.run_in_executor(self.executor, _do)

    async def _close(self, rdr):
        def _do():
            rdr.close()
        await self.loop.run_in_executor(self.executor, _do)

    async def produce(self, q: asyncio.Queue, spec: ReadSpec) -> None:
        rdr = None
        try:
            rdr = await self._open(spec.path)

            def _slice():
                return rdr.slice_by_time(start_ms=spec.start_ms, end_ms=spec.end_ms)

            sl: slice = await self.loop.run_in_executor(self.executor, _slice)

            n_total = len(rdr)
            start = sl.start or 0
            stop = sl.stop if sl.stop is not None else n_total
            step = max(1, int(spec.chunk_records))
            cols = tuple(spec.columns or DEFAULT_COLUMNS)

            def _times():
                return rdr.times_epoch_ms()

            times_all: np.ndarray = await self.loop.run_in_executor(self.executor, _times)

            for s in range(start, stop, step):
                e = min(s + step, stop)

                def _pull():
                    view = rdr.view[s:e]
                    mats = []
                    for c in cols:
                        if c in view.dtype.names:
                            mats.append(view[c].astype(np.float64, copy=spec.copy_chunks))
                    data = (
                        np.stack(mats, axis=1)
                        if mats else np.empty((e - s, 0), dtype=np.float64)
                    )
                    times = times_all[s:e].copy() if spec.copy_chunks else times_all[s:e]
                    return data, times

                data, times = await self.loop.run_in_executor(self.executor, _pull)

                await q.put({
                    "data": data,
                    "times": times,
                    "columns": cols,
                    "src_path": spec.path,
                    "start": s,
                    "end": e,
                })
        except Exception as exc:  # pragma: no cover - propagated downstream
            await q.put({"error": True, "exc": repr(exc), "src_path": spec.path})
        finally:
            await q.put(None)
            if rdr is not None:
                with contextlib.suppress(Exception):
                    await self._close(rdr)


# ---------------------- Async Writer (consumer) -----------------------------
class AsyncChunkWriter:
    """Consumes items from a queue and writes to Parquet or CSV."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None,
                 executor: Optional[concurrent.futures.Executor] = None) -> None:
        self.loop = loop or asyncio.get_event_loop()
        self.executor = executor

    class _ParquetStream:
        def __init__(self, out_path: str, compression: str) -> None:
            if pa is None or pq is None:  # pragma: no cover - optional dependency
                raise RuntimeError("pyarrow is required for Parquet output")
            self.out_path = out_path
            self.compression = compression
            self._writer: Optional[pq.ParquetWriter] = None

        def write(self, table: "pa.Table") -> None:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            if self._writer is None:
                self._writer = pq.ParquetWriter(
                    self.out_path,
                    table.schema,
                    compression=self.compression,
                )
            self._writer.write_table(table)

        def close(self) -> None:
            if self._writer is not None:
                self._writer.close()
                self._writer = None

    class _CSVStream:
        def __init__(self, spec: WriteSpec) -> None:
            if pd is None:  # pragma: no cover - optional dependency
                raise RuntimeError("pandas is required for CSV output")
            self.spec = spec
            self._shard_index = 0
            self._rows_in_shard = 0
            self._current_handle: Optional[object] = None
            self._columns: Optional[List[str]] = None
            self._header_written = False

        def _ensure_dir(self) -> None:
            os.makedirs(os.path.dirname(self.spec.out_path) or ".", exist_ok=True)

        def _next_shard_path(self) -> str:
            base = Path(self.spec.out_path)
            name = f"{base.stem}.part{self._shard_index:04d}{base.suffix or '.csv'}"
            self._shard_index += 1
            return str(base.with_name(name))

        def _open_shard(self) -> None:
            path = self._next_shard_path()
            self._current_handle = open(path, "w", newline="", encoding="utf-8")
            self._rows_in_shard = 0
            self._header_written = False

        def write(self, frame: "pd.DataFrame") -> None:
            self._ensure_dir()
            if self._columns is None:
                self._columns = list(frame.columns)

            shard_rows = self.spec.csv_shard_rows
            if shard_rows and shard_rows > 0:
                remaining = frame
                while not remaining.empty:
                    if self._current_handle is None:
                        self._open_shard()
                        if self.spec.csv_header and self._columns:
                            self._current_handle.write(
                                self.spec.csv_separator.join(self._columns) + "\n"
                            )
                            self._header_written = True
                    room = shard_rows - self._rows_in_shard
                    take = min(len(remaining), room)
                    chunk = remaining.iloc[:take]
                    chunk.to_csv(
                        self._current_handle,
                        index=False,
                        header=False,
                        sep=self.spec.csv_separator,
                    )
                    self._rows_in_shard += take
                    remaining = remaining.iloc[take:]
                    if self._rows_in_shard >= shard_rows:
                        self._current_handle.close()
                        self._current_handle = None
                return

            mode = "a" if self._header_written else "w"
            frame.to_csv(
                self.spec.out_path,
                mode=mode,
                index=False,
                header=self.spec.csv_header and not self._header_written,
                sep=self.spec.csv_separator,
            )
            self._header_written = True

        def close(self) -> None:
            if self._current_handle is not None:
                self._current_handle.close()
                self._current_handle = None

    async def consume(self, q: asyncio.Queue, spec: WriteSpec, src_label: str = "") -> None:
        async def _iter_chunks():
            while True:
                item = await q.get()
                if item is None:
                    break
                if item.get("error"):
                    print(f"[writer] upstream error from {item.get('src_path')}: {item.get('exc')}")
                    continue
                yield item

        if spec.format.lower() == "parquet":
            if pa is None or pq is None:
                raise RuntimeError("pyarrow not available; cannot write parquet")
            stream = self._ParquetStream(spec.out_path, spec.parquet_compression)

            async def _write(table: "pa.Table") -> None:
                await self.loop.run_in_executor(self.executor, stream.write, table)

            try:
                async for item in _iter_chunks():
                    arrays = []
                    names = []
                    if spec.include_time:
                        arrays.append(pa.array(item["times"], type=pa.timestamp("ms", tz="UTC")))
                        names.append("DateTime")
                    for idx, name in enumerate(item["columns"]):
                        arrays.append(pa.array(item["data"][:, idx]))
                        names.append(name)
                    table = pa.table(arrays, names=names)
                    await _write(table)
            finally:
                await self.loop.run_in_executor(self.executor, stream.close)

        elif spec.format.lower() == "csv":
            if pd is None:
                raise RuntimeError("pandas not available; cannot write csv")
            stream = self._CSVStream(spec)

            async def _write(frame: "pd.DataFrame") -> None:
                await self.loop.run_in_executor(self.executor, stream.write, frame)

            try:
                async for item in _iter_chunks():
                    frame = pd.DataFrame(item["data"], columns=list(item["columns"]))
                    if spec.include_time:
                        frame.insert(0, "DateTime", pd.to_datetime(item["times"], unit="ms", utc=True))
                    await _write(frame)
            finally:
                await self.loop.run_in_executor(self.executor, stream.close)

        else:
            raise ValueError(f"Unknown output format: {spec.format}")


# -------------------------- Orchestration -----------------------------------
async def _pipe_one(spec: PipelineSpec,
                    reader: AsyncScidReader,
                    writer: AsyncChunkWriter) -> None:
    q: asyncio.Queue = asyncio.Queue(maxsize=max(1, spec.queue_maxsize))

    async def _producer():
        await reader.produce(q, spec.read)

    async def _consumer():
        await writer.consume(q, spec.write, src_label=spec.read.path)

    prod = asyncio.create_task(_producer(), name=f"prod:{Path(spec.read.path).name}")
    cons = asyncio.create_task(_consumer(), name=f"cons:{Path(spec.write.out_path).name}")

    try:
        await asyncio.gather(prod, cons)
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            if not cons.done():
                cons.cancel()
            if not prod.done():
                prod.cancel()


def _thread_pool_size(max_reader_tasks: int, max_writer_tasks: int) -> int:
    return max(2, max_reader_tasks + max_writer_tasks)


async def run_pipelines(specs: Sequence[PipelineSpec],
                        max_reader_tasks: int = 2,
                        max_writer_tasks: int = 2) -> None:
    loop = asyncio.get_event_loop()
    pool_size = _thread_pool_size(max_reader_tasks, max_writer_tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size,
                                               thread_name_prefix="scid-io") as ex:
        reader = AsyncScidReader(loop=loop, executor=ex)
        writer = AsyncChunkWriter(loop=loop, executor=ex)

        async def _staggered(idx: int, ps: PipelineSpec):
            await asyncio.sleep(0.05 * idx)
            await _pipe_one(ps, reader, writer)

        tasks = [asyncio.create_task(_staggered(i, ps), name=f"pipe:{i}")
                 for i, ps in enumerate(specs)]

        stop = asyncio.Event()

        def _on_signal(*_):
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _on_signal)

        async def _watch_stop():
            await stop.wait()
            for t in tasks:
                t.cancel()

        await asyncio.wait([*tasks, asyncio.create_task(_watch_stop())],
                           return_when=asyncio.ALL_COMPLETED)


# ------------------------------ CLI ----------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async SCID IO: concurrent read+write")
    parser.add_argument("--inputs", nargs="+", help="Input .scid files")
    parser.add_argument("--out-dir", required=True, help="Directory for outputs")
    parser.add_argument("--fmt", default="parquet", choices=["parquet", "csv"],
                        help="Output format")
    parser.add_argument("--chunk", type=int, default=2_000_000,
                        help="Records per chunk")
    parser.add_argument("--start-ms", type=int, default=None,
                        help="Start time (ms since epoch)")
    parser.add_argument("--end-ms", type=int, default=None,
                        help="End time (ms since epoch)")
    parser.add_argument("--cols", type=str, default=",".join(DEFAULT_COLUMNS),
                        help="Comma separated list of columns to export")
    parser.add_argument("--readers", type=int, default=2, help="Reader thread hint")
    parser.add_argument("--writers", type=int, default=2, help="Writer thread hint")
    parser.add_argument("--queue", type=int, default=4, help="Per pipeline queue maxsize")
    parser.add_argument("--csv-sep", type=str, default=",", help="CSV separator")
    parser.add_argument("--csv-shard-rows", type=int, default=None,
                        help="Shard CSV every N rows (optional)")
    parser.add_argument("--no-time", action="store_true", help="Do not include DateTime column")
    parser.add_argument("--parquet-zstd", type=str, default="zstd",
                        help="Parquet compression codec")
    return parser.parse_args(argv)


def _build_specs(ns: argparse.Namespace) -> List[PipelineSpec]:
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ns.cols.split(",") if c]

    specs: List[PipelineSpec] = []
    for inp in ns.inputs:
        input_path = Path(inp)
        suffix = ".parquet" if ns.fmt == "parquet" else ".csv"
        out_path = out_dir / input_path.with_suffix(suffix).name
        specs.append(
            PipelineSpec(
                read=ReadSpec(
                    path=str(input_path),
                    start_ms=ns.start_ms,
                    end_ms=ns.end_ms,
                    columns=cols,
                    chunk_records=ns.chunk,
                ),
                write=WriteSpec(
                    out_path=str(out_path),
                    format=ns.fmt,
                    include_time=not ns.no_time,
                    parquet_compression=ns.parquet_zstd,
                    csv_separator=ns.csv_sep,
                    csv_shard_rows=ns.csv_shard_rows,
                    csv_header=True,
                ),
                queue_maxsize=ns.queue,
            )
        )
    return specs


async def _amain(ns: argparse.Namespace) -> None:
    specs = _build_specs(ns)
    await run_pipelines(specs,
                        max_reader_tasks=max(1, ns.readers),
                        max_writer_tasks=max(1, ns.writers))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ns = _parse_args(argv)
    try:
        asyncio.run(_amain(ns))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
