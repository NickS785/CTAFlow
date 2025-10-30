"""Asyncio convenience helpers."""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar


T = TypeVar("T")


def run(awaitable: Awaitable[T]) -> T:
    """Run ``awaitable`` to completion and return its result.

    The helper primarily proxies :func:`asyncio.run` while gracefully handling
    contexts where an event loop is already running (e.g. notebooks).  In those
    situations we spin up a dedicated loop for the awaitable, ensuring callers
    can synchronously execute asynchronous entry-points without worrying about
    loop management.
    """

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(awaitable)
    finally:
        new_loop.close()


__all__ = ["run"]

