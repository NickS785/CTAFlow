"""Callback handlers for the CTAFlow dashboard."""

from __future__ import annotations

from dash import Dash, Input, Output, html

from . import tab_callbacks


def register_callbacks(app: Dash) -> None:
    """Register all callbacks with the Dash application.

    Parameters
    ----------
    app : Dash
        The Dash application instance.
    """
    tab_callbacks.register(app)
