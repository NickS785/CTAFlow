"""
Main Dash Application for CTAFlow Dashboard

This dashboard provides visualization and analysis of:
- Multi-asset WSPR model performance
- Real-time predictions
- Backtesting results
- Trading metrics and PnL analysis
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from pathlib import Path

# Import components (will be created)
# from .components.header import create_header
# from .components.model_selector import create_model_selector
# from .components.performance_charts import create_performance_charts

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="CTAFlow Dashboard",
)

server = app.server

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("CTAFlow Trading Dashboard", className="text-center my-4"),
            html.Hr(),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H3("Model Selection"),
            dcc.Dropdown(
                id='model-selector',
                options=[],  # Will be populated from S3
                placeholder="Select a trained model...",
            ),
        ], width=6),
        dbc.Col([
            html.H3("Ticker Selection"),
            dcc.Dropdown(
                id='ticker-selector',
                options=[],  # Will be populated based on model
                multi=True,
                placeholder="Select tickers...",
            ),
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H3("Performance Overview"),
            dcc.Graph(id='performance-chart'),
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H3("Predictions"),
            dcc.Graph(id='predictions-chart'),
        ], width=6),
        dbc.Col([
            html.H3("Trading Metrics"),
            html.Div(id='metrics-display'),
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H3("Recent Activity"),
            html.Div(id='recent-activity'),
        ])
    ]),

    # Interval component for auto-refresh (optional)
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every 60 seconds
        n_intervals=0,
        disabled=True,  # Disabled by default
    ),

], fluid=True, className="px-4")


# Placeholder callbacks - will be implemented
# @app.callback(
#     Output('model-selector', 'options'),
#     Input('interval-component', 'n_intervals')
# )
# def update_model_list(n):
#     """Load available models from S3/local storage."""
#     pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
