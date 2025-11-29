"""
Canonical screener type constants used by HistoricalScreenerV2.
"""

SCREEN_SEASONALITY = "seasonality"
SCREEN_MOMENTUM = "momentum"
SCREEN_ORDERFLOW = "orderflow"
SCREEN_EVENT = "event"

VALID_SCREEN_TYPES = {
    SCREEN_SEASONALITY,
    SCREEN_MOMENTUM,
    SCREEN_ORDERFLOW,
    SCREEN_EVENT,
}
