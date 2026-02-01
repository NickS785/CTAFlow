"""Tab navigation callbacks."""

from __future__ import annotations

from dash import Dash, Input, Output, html


def register(app: Dash) -> None:
    """Register tab navigation callbacks.

    Parameters
    ----------
    app : Dash
        The Dash application instance.
    """

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab_content(tab: str) -> html.Div:
        """Render content based on selected tab.

        Parameters
        ----------
        tab : str
            The selected tab value.

        Returns
        -------
        html.Div
            Content component for the selected tab.
        """
        if tab == "overview":
            return html.Div(
                [
                    html.H2("Overview"),
                    html.P("Select a ticker to view analysis."),
                ],
                className="tab-content",
            )
        elif tab == "screeners":
            return html.Div(
                [
                    html.H2("Screeners"),
                    html.P("Pattern and orderflow screeners."),
                ],
                className="tab-content",
            )
        elif tab == "models":
            return html.Div(
                [
                    html.H2("Models"),
                    html.P("Forecasting models and training."),
                ],
                className="tab-content",
            )
        elif tab == "results":
            return html.Div(
                [
                    html.H2("Results"),
                    html.P("Model results and analysis."),
                ],
                className="tab-content",
            )
        return html.Div("Select a tab")
