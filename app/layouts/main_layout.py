"""Main dashboard layout definition."""

from __future__ import annotations

from dash import html, dcc


def get_layout() -> html.Div:
    """Build the main dashboard layout.

    Returns
    -------
    html.Div
        Root layout component containing navigation and content areas.
    """
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1("CTAFlow Dashboard", className="header-title"),
                    html.P(
                        "CTA Positioning and Orderflow Analysis",
                        className="header-subtitle",
                    ),
                ],
                className="header",
            ),
            # Navigation tabs
            dcc.Tabs(
                id="main-tabs",
                value="overview",
                children=[
                    dcc.Tab(label="Overview", value="overview"),
                    dcc.Tab(label="Screeners", value="screeners"),
                    dcc.Tab(label="Models", value="models"),
                    dcc.Tab(label="Results", value="results"),
                ],
                className="main-tabs",
            ),
            # Content area
            html.Div(id="tab-content", className="content"),
            # Data stores
            dcc.Store(id="selected-ticker-store"),
            dcc.Store(id="date-range-store"),
            dcc.Store(id="results-store"),
        ],
        className="app-container",
    )
