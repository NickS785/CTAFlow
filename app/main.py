"""Main Dash application entry point.

Usage
-----
Run the app with::

    python -m app.main

Or import and run programmatically::

    from app import create_app
    app = create_app()
    app.run(debug=True)
"""

from __future__ import annotations

from dash import Dash, html, dcc
from .layouts import main_layout
from .callbacks import register_callbacks


def create_app(debug: bool = False) -> Dash:
    """Create and configure the Dash application.

    Parameters
    ----------
    debug : bool
        Whether to run in debug mode with hot reloading.

    Returns
    -------
    Dash
        Configured Dash application instance.
    """
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="CTAFlow Dashboard",
    )

    app.layout = main_layout.get_layout()
    register_callbacks(app)

    return app


def main() -> None:
    """Run the dashboard application."""
    app = create_app(debug=True)
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
