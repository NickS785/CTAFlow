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


def bucket_by_volume(*args, **kwargs):
    return []


def resample_ohlcv(*args, **kwargs):
    return []
