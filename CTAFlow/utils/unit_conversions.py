"""Commodity unit conversion helpers used throughout CTA analytics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

GALLONS_PER_BARREL = 42.0
POUNDS_TO_KG = 0.45359237

_BUSHEL_WEIGHT_LBS: Dict[str, float] = {
    "corn": 56.0,
    "soybean": 60.0,
    "wheat": 60.0,
    "oat": 32.0,
    "barley": 48.0,
    "canola": 50.0,
    "sorghum": 56.0,
    "rice": 45.0,
}

_COMMODITY_ALIASES: Dict[str, str] = {
    "soybeans": "soybean",
    "soya": "soybean",
    "bean": "soybean",
    "beans": "soybean",
    "oats": "oat",
    "barleys": "barley",
    "corns": "corn",
    "rices": "rice",
    "sorghums": "sorghum",
    "canolas": "canola",
}


def _apply_conversion(value: Any, factor: float):
    """Apply a multiplicative conversion factor while preserving the input structure."""
    if isinstance(value, pd.Series):
        return value.astype(float) * factor
    if isinstance(value, pd.DataFrame):
        return value.astype(float) * factor
    if np.isscalar(value):
        return float(value) * factor
    array = np.asarray(value, dtype=float)
    return array * factor


def gallons_to_barrels(gallons: Any):
    """Convert gallons to barrels (42 gallons per barrel)."""
    return _apply_conversion(gallons, 1.0 / GALLONS_PER_BARREL)


def barrels_to_gallons(barrels: Any):
    """Convert barrels to gallons."""
    return _apply_conversion(barrels, GALLONS_PER_BARREL)


def _normalize_commodity(commodity: str) -> str:
    if not isinstance(commodity, str):
        raise TypeError("Commodity name must be provided as a string")

    key = commodity.strip().lower().replace(" ", "").replace("-", "")
    key = _COMMODITY_ALIASES.get(key, key)

    if key not in _BUSHEL_WEIGHT_LBS and key.endswith("s"):
        singular = key[:-1]
        if singular in _BUSHEL_WEIGHT_LBS:
            key = singular

    if key not in _BUSHEL_WEIGHT_LBS:
        raise KeyError(f"Unsupported commodity for bushel conversion: {commodity}")

    return key


def bushels_to_kilograms(bushels: Any, commodity: str):
    """Convert bushels of a commodity to kilograms."""
    key = _normalize_commodity(commodity)
    kilograms_per_bushel = _BUSHEL_WEIGHT_LBS[key] * POUNDS_TO_KG
    return _apply_conversion(bushels, kilograms_per_bushel)


def bushels_to_metric_tons(bushels: Any, commodity: str):
    """Convert bushels of a commodity to metric tons."""
    kilograms = bushels_to_kilograms(bushels, commodity)
    return kilograms / 1000.0


__all__ = [
    "gallons_to_barrels",
    "barrels_to_gallons",
    "bushels_to_kilograms",
    "bushels_to_metric_tons",
]
