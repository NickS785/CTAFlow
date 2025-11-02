"""Stub parser providing minimal interfaces for tests."""


class ScidReader:  # pragma: no cover - simple stub
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter(())


class AsyncScidReader(ScidReader):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def export_scid_files_to_parquet(self, **kwargs):  # pragma: no cover - simple stub
        """Minimal stub for async export helper.

        The real implementation in SierraPy performs asynchronous SCID to
        Parquet exports.  Tests in this repository only need a predictable
        return payload, so we echo back the arguments.
        """

        return {
            "success": True,
            "arguments": kwargs,
            "records_written": kwargs.get("records_written", 0),
        }


def bucket_by_volume(*args, **kwargs):
    return []


def resample_ohlcv(*args, **kwargs):
    return []
