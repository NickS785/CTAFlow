"""Compatibility wrapper forwarding to the event screener implementation."""
from .event_screener import *  # noqa: F401,F403
from .event_screener import run_event_screener as data_release_scan  # noqa: F401
