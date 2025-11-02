"""
CrossProductEngine: Build cross-product spreads (CPS) with flexible ratios and calendar structures.

This module implements the design from cps.md, providing:
- Front-month CPS (k=0)
- Back-month CPS (k>0 by index or months-to-expiry)
- Calendar of CPS (near-far tenor spreads)
- Flexible ratio weighting (notional, volatility, margin-based)
- Automatic unit conversion to common reference unit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from datetime import datetime
from CTAFlow.config import DLY_DATA_PATH, INTRADAY_DATA_PATH

import numpy as np
import pandas as pd
from ...data import DataClient
from .spread_manager import SpreadData, FuturesCurve
from sierrapy.parser import ScidReader

try:
    from ...utils.unit_conversions import (
        GALLONS_PER_BARREL,
        barrels_to_gallons,
        gallons_to_barrels,
        bushels_to_metric_tons
    )

    HAS_CONVERSIONS = True
except ImportError:
    GALLONS_PER_BARREL = 42.0
    HAS_CONVERSIONS = False

# CME contract specifications - maps symbol to (unit, commodity_name)
CME_CONTRACT_SPECS = {
    # Energy - NYMEX
    'CL': ('$/bbl', 'crude_oil'),  # WTI Crude Oil
    'HO': ('$/gal', 'heating_oil'),  # Heating Oil
    'RB': ('$/gal', 'gasoline'),  # RBOB Gasoline
    'NG': ('$/mmbtu', 'natural_gas'),  # Natural Gas
    'BZ': ('$/bbl', 'brent_crude'),  # Brent Crude

    # Grains - CBOT
    'ZC': ('cents/bu', 'corn'),  # Corn
    'ZS': ('cents/bu', 'soybean'),  # Soybeans
    'ZW': ('cents/bu', 'wheat'),  # Wheat
    'ZM': ('$/ton', 'soybean_meal'),  # Soybean Meal
    'ZL': ('cents/lb', 'soybean_oil'),  # Soybean Oil
    'ZO': ('cents/bu', 'oat'),  # Oats

    # Softs
    'KC': ('cents/lb', 'coffee'),  # Coffee
    'SB': ('cents/lb', 'sugar'),  # Sugar
    'CC': ('$/mt', 'cocoa'),  # Cocoa
    'CT': ('cents/lb', 'cotton'),  # Cotton

    # Livestock
    'LE': ('cents/lb', 'live_cattle'),  # Live Cattle
    'GF': ('cents/lb', 'feeder_cattle'),  # Feeder Cattle
    'HE': ('cents/lb', 'lean_hogs'),  # Lean Hogs

    # Metals - COMEX
    'GC': ('$/oz', 'gold'),  # Gold
    'SI': ('$/oz', 'silver'),  # Silver
    'HG': ('cents/lb', 'copper'),  # Copper
    'PL': ('$/oz', 'platinum'),  # Platinum
    'PA': ('$/oz', 'palladium'),  # Palladium
}


def detect_contract_unit(symbol: str) -> tuple[str, str]:
    """
    Auto-detect unit and commodity from symbol.

    Parameters:
    -----------
    symbol : str
        Contract symbol (e.g., 'CL_F', 'ZC', 'HO_F')

    Returns:
    --------
    tuple[str, str]
        (unit, commodity_name)
    """
    # Strip _F suffix if present
    base_symbol = symbol.replace('_F', '').upper()

    # Extract first 2 characters (standard CME code)
    if len(base_symbol) >= 2:
        code = base_symbol[:2]
        if code in CME_CONTRACT_SPECS:
            return CME_CONTRACT_SPECS[code]

    # Try single character codes
    if base_symbol and base_symbol[0] in CME_CONTRACT_SPECS:
        return CME_CONTRACT_SPECS[base_symbol[0]]

    # Default fallback
    return ('$/unit', 'unknown')


# Unit hierarchy for determining reference unit (largest/most standard first)
UNIT_HIERARCHY = [
    '$/bbl',  # Barrels (energy - most common reference)
    '$/mt',  # Metric tons (large bulk)
    '$/ton',  # Short tons (bulk)
    '$/mmbtu',  # Natural gas (energy equivalent)
    '$/bu',  # Bushels (agricultural)
    '$/oz',  # Ounces (precious metals)
    '$/kg',  # Kilograms
    '$/lb',  # Pounds
    '$/gal',  # Gallons (refined products)
    'cents/bu',  # Cents per bushel (smaller denomination)
    'cents/lb',  # Cents per pound (smaller denomination)
    'cents/gal',  # Cents per gallon (smaller denomination)
    'cents/bbl',  # Cents per barrel (smaller denomination)
    '$/unit',  # Generic fallback
]


def detect_reference_unit(symbols: List[str]) -> str:
    """
    Auto-detect reference unit based on the largest/most standard unit among symbols.

    Parameters:
    -----------
    symbols : List[str]
        List of contract symbols (e.g., ['CL_F', 'HO_F', 'RB_F'])

    Returns:
    --------
    str
        Reference unit for the spread (e.g., '$/bbl', '$/bu', '$/oz')

    Examples:
    ---------
    >>> detect_reference_unit(['CL_F', 'HO_F', 'RB_F'])
    '$/bbl'
    >>> detect_reference_unit(['ZC_F', 'ZS_F'])
    'cents/bu'
    >>> detect_reference_unit(['GC_F', 'SI_F'])
    '$/oz'
    """
    # Detect units for all symbols
    units = []
    for symbol in symbols:
        unit, _ = detect_contract_unit(symbol)
        units.append(unit.lower().strip())

    # Find highest priority unit in the hierarchy
    for ref_unit in UNIT_HIERARCHY:
        if ref_unit.lower() in units:
            return ref_unit

    # Fallback to first unit if none match hierarchy
    return units[0] if units else '$/unit'


# Unit conversion registry
UNIT_CONVERSIONS = {
    # Energy products
    ('$/gal', '$/bbl'): GALLONS_PER_BARREL,
    ('$/bbl', '$/gal'): 1.0 / GALLONS_PER_BARREL,
    ('cents/gal', 'cents/bbl'): GALLONS_PER_BARREL,
    ('cents/bbl', 'cents/gal'): 1.0 / GALLONS_PER_BARREL,

    # Agricultural (will need symbol-specific conversions)
    ('$/bu', '$/mt'): None,  # Requires symbol type
    ('$/mt', '$/bu'): None,  # Requires symbol type
    ('cents/bu', '$/mt'): None,  # Requires symbol type
    ('cents/bu', '$/ton'): None,  # Requires symbol type (bushel weight varies by commodity)

    # Cents to dollar conversions
    ('cents/bu', '$/bu'): 0.01,
    ('$/bu', 'cents/bu'): 100.0,
    ('cents/lb', '$/lb'): 0.01,
    ('$/lb', 'cents/lb'): 100.0,

    # Weight conversions (lb to ton)
    # 1 short ton = 2000 lbs
    ('cents/lb', '$/ton'): 0.01 * 2000.0,  # cents/lb to $/ton: (cents/lb) * 0.01 * 2000 lbs/ton
    ('$/ton', 'cents/lb'): 1.0 / (0.01 * 2000.0),  # $/ton to cents/lb
    ('$/lb', '$/ton'): 2000.0,  # $/lb to $/ton
    ('$/ton', '$/lb'): 1.0 / 2000.0,  # $/ton to $/lb
    ('cents/lb', 'cents/ton'): 2000.0,  # cents/lb to cents/ton
    ('cents/ton', 'cents/lb'): 1.0 / 2000.0,  # cents/ton to cents/lb

    # Metals (troy oz conversions if needed)
    ('$/oz', '$/kg'): 32.1507,  # troy oz to kg
    ('$/kg', '$/oz'): 1.0 / 32.1507,

    # Identity conversions
    ('$/bbl', '$/bbl'): 1.0,
    ('$/gal', '$/gal'): 1.0,
    ('$/mt', '$/mt'): 1.0,
    ('$/bu', '$/bu'): 1.0,
    ('$/oz', '$/oz'): 1.0,
    ('cents/lb', 'cents/lb'): 1.0,
    ('cents/bu', 'cents/bu'): 1.0,
    ('$/ton', '$/ton'): 1.0,
    ('$/mmbtu', '$/mmbtu'): 1.0,
    ('$/lb', '$/lb'): 1.0,
}


def convert_units(value: float, from_unit: str, to_unit: str, symbol: Optional[str] = None) -> float:
    """
    Convert price from one unit to another.

    Parameters:
    -----------
    value : float
        Price value to convert
    from_unit : str
        Source unit (e.g., '$/gal', '$/bbl', '$/bu', '$/mt')
    to_unit : str
        Target unit
    symbol : str, optional
        Symbol/commodity name for unit conversions (e.g., 'corn', 'soybean', 'CL_F')

    Returns:
    --------
    float
        Converted value
    """
    # Normalize unit strings
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    # No conversion needed
    if from_unit == to_unit:
        return value

    # Check registry
    key = (from_unit, to_unit)
    if key in UNIT_CONVERSIONS:
        factor = UNIT_CONVERSIONS[key]
        if factor is None:
            if symbol is None:
                raise ValueError(f"Conversion {from_unit} -> {to_unit} requires symbol specification")

            # Handle bushel conversions (requires symbol-specific logic)
            if not HAS_CONVERSIONS:
                raise NotImplementedError(
                    f"Bushel conversions require CTAFlow.utils.unit_conversions module"
                )

            # Extract commodity name from symbol
            _, commodity = detect_contract_unit(symbol)

            # Handle cents/bu -> $/ton conversion
            if key == ('cents/bu', '$/ton'):
                # Convert cents to dollars first, then bushels to metric tons, then to short tons
                # 1 metric ton = 1.10231 short tons
                dollars_per_bu = value * 0.01  # cents to dollars
                metric_tons = bushels_to_metric_tons(1.0, commodity)  # Get MT per bushel
                short_tons_per_bushel = metric_tons * 1.10231  # Convert MT to short tons
                return dollars_per_bu / short_tons_per_bushel  # $/bu / (short tons/bu) = $/ton

            # Handle $/bu -> $/mt conversion
            elif key == ('$/bu', '$/mt'):
                metric_tons_per_bushel = bushels_to_metric_tons(1.0, commodity)
                return value / metric_tons_per_bushel

            # Handle $/mt -> $/bu conversion
            elif key == ('$/mt', '$/bu'):
                metric_tons_per_bushel = bushels_to_metric_tons(1.0, commodity)
                return value * metric_tons_per_bushel

            # Handle cents/bu -> $/mt conversion
            elif key == ('cents/bu', '$/mt'):
                dollars_per_bu = value * 0.01
                metric_tons_per_bushel = bushels_to_metric_tons(1.0, commodity)
                return dollars_per_bu / metric_tons_per_bushel
            else:
                raise NotImplementedError(f"Bushel conversion {from_unit} -> {to_unit} not implemented")
        return value * factor

    raise ValueError(f"Unknown unit conversion: {from_unit} -> {to_unit}")


def dte_to_months(dte_value: Optional[float]) -> Optional[float]:
    """Convert days-to-expiry to approximate months."""
    if dte_value is None:
        return None
    try:
        if np.isnan(dte_value):
            return None
    except TypeError:
        pass
    return float(dte_value) / 30.4375


def align_tenors(
        seq_labels_A: Sequence[str],
        seq_dte_A: Sequence[float],
        seq_labels_B: Sequence[str],
        seq_dte_B: Sequence[float],
        method: str = "index",
        k: int = 0,
        target_months: Optional[float] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (contractA, contractB) for the chosen tenor.

    Parameters:
    -----------
    seq_labels_A : Sequence[str]
        Sequential contract labels for product A
    seq_dte_A : Sequence[float]
        Days to expiry for product A contracts
    seq_labels_B : Sequence[str]
        Sequential contract labels for product B
    seq_dte_B : Sequence[float]
        Days to expiry for product B contracts
    method : str
        "index" to pick M{k} for both, "months" to pick closest-to target_months
    k : int
        Sequential index (M0, M1, M2, ...) when method="index"
    target_months : float, optional
        Target months to expiry when method="months"

    Returns:
    --------
    Tuple[Optional[str], Optional[str]]
        (contract_A, contract_B) labels aligned by tenor
    """
    if method == "index":
        ca = seq_labels_A[k] if k < len(seq_labels_A) else None
        cb = seq_labels_B[k] if k < len(seq_labels_B) else None
        return ca, cb

    # months-to-expiry based: pick closest by DTE in months
    def pick_closest(labels: Sequence[str], dte_row: Sequence[float], target: float) -> Optional[str]:
        best = None
        best_err = 1e9
        for j, lab in enumerate(labels):
            if j >= len(dte_row):
                continue
            dte = float(dte_row[j])
            if not np.isfinite(dte):
                continue
            m = dte_to_months(dte)
            if m is None:
                continue
            err = abs(m - target)
            if err < best_err:
                best, best_err = lab, err
        return best

    if target_months is None:
        raise ValueError("target_months required when method='months'")

    ca = pick_closest(seq_labels_A, seq_dte_A, target_months)
    cb = pick_closest(seq_labels_B, seq_dte_B, target_months)
    return ca, cb


@dataclass
class CrossSpreadLeg:
    """
    Single leg of a cross-product spread with flexible weighting and unit conversion.

    Attributes:
    -----------
    data : SpreadData
        Underlying spread data for this leg
    base_weight : float
        Base weight (±1 for long/short)
    symbol : str, optional
        Contract symbol (e.g., 'CL_F', 'HO_F', 'ZC_F'). Auto-detects unit if provided.
    unit : str, optional
        Price unit for this leg (e.g., '$/bbl', '$/gal', '$/bu', '$/mt').
        Auto-detected from symbol if not provided.
    contract_ratios : Dict[str, float], optional
        Per-contract weight overrides (e.g., {"M0": 1.0, "M1": 0.8})
    notional_multiplier : float
        Contract multiplier for notional scaling
    fx_rate : float or pd.Series
        FX conversion rate (can be time-varying)
    vol_scale : Optional[pd.Series]
        Per-tenor volatility for risk parity scaling
    """
    data: SpreadData
    base_weight: float = 1.0
    symbol: Optional[str] = None
    unit: Optional[str] = None
    contract_ratios: Optional[Dict[str, float]] = None
    notional_multiplier: float = 1.0
    fx_rate: Union[float, pd.Series] = 1.0
    vol_scale: Optional[pd.Series] = None

    def __post_init__(self):
        """Auto-detect unit and commodity from symbol if not provided."""
        # Try to get symbol from data.symbol if not provided
        if self.symbol is None and hasattr(self.data, 'symbol'):
            self.symbol = self.data.symbol

        # Auto-detect unit if symbol is provided and unit is not
        if self.symbol is not None and self.unit is None:
            detected_unit, _ = detect_contract_unit(self.symbol)
            self.unit = detected_unit

        # Set default unit if still None
        if self.unit is None:
            self.unit = "$/unit"

    def convert_price_to_unit(self, price: float, target_unit: str) -> float:
        """
        Convert price from this leg's unit to target unit.

        Parameters:
        -----------
        price : float
            Price in this leg's unit
        target_unit : str
            Target unit for conversion

        Returns:
        --------
        float
            Converted price
        """
        return convert_units(price, self.unit, target_unit, symbol=self.symbol)

    def weight_for_contract(self, contract_label: str, date: Optional[datetime] = None) -> float:
        """
        Calculate effective weight for a specific contract at a given date.

        Parameters:
        -----------
        contract_label : str
            Contract label (e.g., "M0", "H", "K")
        date : datetime, optional
            Date for time-varying FX rates

        Returns:
        --------
        float
            Effective weight incorporating all scaling factors
        """
        # Start with per-contract ratio override if available
        ratio = self.contract_ratios.get(contract_label, 1.0) if self.contract_ratios else 1.0

        # Apply base weight
        w = self.base_weight * ratio

        # Apply notional scaling
        w *= self.notional_multiplier

        # Apply FX scaling
        if isinstance(self.fx_rate, pd.Series) and date is not None:
            fx = self.fx_rate.loc[date] if date in self.fx_rate.index else 1.0
            w *= fx
        else:
            w *= float(self.fx_rate)

        # Apply volatility scaling if available
        if self.vol_scale is not None and contract_label in self.vol_scale.index:
            vol = self.vol_scale.loc[contract_label]
            if np.isfinite(vol) and vol > 0:
                w /= vol

        return w

    @classmethod
    def load_from_dclient(
            cls,
            symbol: str,
            base_weight: float = 1.0,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            **kwargs
    ) -> 'CrossSpreadLeg':
        """
        Load CrossSpreadLeg from DataClient with auto-detected units.

        Parameters:
        -----------
        symbol : str
            Contract symbol (e.g., 'CL_F', 'HO_F', 'ZC_F')
        base_weight : float
            Base weight for this leg (default 1.0)
        start_date : str, optional
            Start date for filtering
        end_date : str, optional
            End date for filtering
        daily : bool
            Use daily resampled data
        **kwargs
            Additional arguments passed to CrossSpreadLeg constructor

        Returns:
        --------
        CrossSpreadLeg
            Leg with loaded data and auto-detected units
        """
        from ...data import DataClient

        client = DataClient()

        # Load SpreadData from client
        spread_data = SpreadData()
        spread_data = spread_data.load_from_client_filtered(
            symbol,
            start_date=start_date,
            end_date=end_date,
        )

        # Create leg with auto-detection via __post_init__
        return cls(
            data=spread_data,
            base_weight=base_weight,
            symbol=symbol,  # This will trigger auto-detection in __post_init__
            **kwargs
        )


@dataclass
class CrossProductSpreadData:
    """
    Container for cross-product spread data with multiple legs.

    Attributes:
    -----------
    legs : List[CrossSpreadLeg]
        Component legs of the cross spread
    name : str
        Identifier for this cross spread
    timestamps : pd.DatetimeIndex
        Time index for the spread series
    values : pd.Series
        Calculated spread values over time
    metadata : Dict[str, Any]
        Additional information about spread construction
    """
    legs: List[CrossSpreadLeg]
    name: str
    timestamps: Optional[pd.DatetimeIndex] = None
    values: Optional[pd.Series] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossProductEngine:
    """
    Build cross-product spreads (CPS) with flexible ratios and tenor alignment.

    This engine creates:
    1. Front-month CPS (k=0 on each product)
    2. Back-month CPS (k>0 by index or months-to-expiry)
    3. Calendar of CPS ((CPS near) - (CPS far))

    Supports multiple weighting schemes:
    - Notional scaling (contract multiplier × FX)
    - Volatility targeting (risk parity)
    - Per-contract ratio maps
    - Automatic unit conversion to common reference unit
    - Multi-leg spreads (2, 3, 4+ legs with flexible long/short positions)
    """

    def __init__(
            self,
            base: Optional[SpreadData] = None,
            hedge: Optional[SpreadData] = None,
            base_leg: Optional[CrossSpreadLeg] = None,
            hedge_leg: Optional[CrossSpreadLeg] = None,
            legs: Optional[List[CrossSpreadLeg]] = None,
            reference_unit: Optional[str] = None
    ):
        """
        Initialize CrossProductEngine with flexible leg configuration.

        Parameters:
        -----------
        base : SpreadData, optional
            Primary product (for backward compatibility, creates leg with weight=1.0)
        hedge : SpreadData, optional
            Hedge product (for backward compatibility, creates leg with weight=-1.0)
        base_leg : CrossSpreadLeg, optional
            Pre-configured base leg (backward compatibility)
        hedge_leg : CrossSpreadLeg, optional
            Pre-configured hedge leg (backward compatibility)
        legs : List[CrossSpreadLeg], optional
            List of all legs for multi-leg spreads (preferred for 3+ legs)
        reference_unit : str, optional
            Common unit to convert all prices to. If None, auto-detects from symbols.

        Notes:
        ------
        - If `legs` is provided, it takes precedence over base/hedge parameters
        - If base/hedge provided without legs, creates 2-leg spread (backward compatible)
        - Use `from_legs()` classmethod for cleaner multi-leg construction
        - If reference_unit not provided, auto-detects based on largest unit among legs
        """
        # Multi-leg mode: legs parameter takes precedence
        if legs is not None:
            self.legs = legs
        else:
            # Backward compatibility: construct from base/hedge
            if base is None and hedge is None:
                raise ValueError("Must provide either 'legs' or 'base'/'hedge' parameters")

            self.legs = []
            if base is not None:
                self.legs.append(base_leg or CrossSpreadLeg(data=base, base_weight=1.0))
            if hedge is not None:
                self.legs.append(hedge_leg or CrossSpreadLeg(data=hedge, base_weight=-1.0))

        # Auto-detect reference unit if not provided
        if reference_unit is None:
            symbols = [getattr(leg.data, 'symbol', None) for leg in self.legs]
            symbols = [s for s in symbols if s is not None]
            if symbols:
                reference_unit = detect_reference_unit(symbols)
            else:
                reference_unit = '$/bbl'  # Fallback default

        self.reference_unit = reference_unit

        # Store first two legs as base/hedge for backward compatibility
        self.base_leg = self.legs[0] if len(self.legs) > 0 else None
        self.hedge_leg = self.legs[1] if len(self.legs) > 1 else None
        self.base = self.base_leg.data if self.base_leg else None
        self.hedge = self.hedge_leg.data if self.hedge_leg else None

        # Cache for tenor alignments (now supports N legs)
        self._alignment_cache: Dict[str, Dict[datetime, List[str]]] = {}

        # Set price attribute to k=0 (front month) spread series
        self.price = None
        try:
            self.price = self.build_cps_series(method="index", k=0)
        except Exception:
            # If building k=0 fails, leave price as None
            pass

    @classmethod
    def from_legs(
            cls,
            legs: List[CrossSpreadLeg],
            reference_unit: Optional[str] = None
    ) -> 'CrossProductEngine':
        """
        Create CrossProductEngine from a list of legs (cleaner multi-leg construction).

        Parameters:
        -----------
        legs : List[CrossSpreadLeg]
            List of 2+ CrossSpreadLeg instances with configured weights
        reference_unit : str, optional
            Common unit to convert all prices to. If None, auto-detects from symbols.

        Returns:
        --------
        CrossProductEngine
            Engine with multi-leg configuration

        Examples:
        ---------
        # Crack spread: CL (long) - 2*HO (short) - RB (short) - auto-detects $/bbl
        >>> cl_leg = CrossSpreadLeg.load_from_dclient('CL_F', base_weight=1.0)
        >>> ho_leg = CrossSpreadLeg.load_from_dclient('HO_F', base_weight=-2.0)
        >>> rb_leg = CrossSpreadLeg.load_from_dclient('RB_F', base_weight=-1.0)
        >>> engine = CrossProductEngine.from_legs([cl_leg, ho_leg, rb_leg])

        # Grain spread - auto-detects cents/bu
        >>> zc_leg = CrossSpreadLeg.load_from_dclient('ZC_F', base_weight=1.0)
        >>> zs_leg = CrossSpreadLeg.load_from_dclient('ZS_F', base_weight=-1.0)
        >>> engine = CrossProductEngine.from_legs([zc_leg, zs_leg])
        """
        if not legs or len(legs) < 2:
            raise ValueError("Must provide at least 2 legs")

        return cls(legs=legs, reference_unit=reference_unit)

    def cross_value_on_date(
            self,
            date: datetime,
            label_map: Dict[str, str],
            prices_by_symbol: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calculate cross-product value at a specific date with automatic unit conversion.

        Parameters:
        -----------
        date : datetime
            Date for valuation
        label_map : Dict[str, str]
            Maps symbol to contract label (e.g., {"CL": "M0", "HO": "M1", "RB": "M0"})
        prices_by_symbol : Dict[str, Dict[str, float]]
            Maps symbol -> {contract_label -> price in leg's native unit}

        Returns:
        --------
        float
            Cross-product spread value in reference unit
        """
        v = 0.0

        for leg in self.legs:
            sym = getattr(leg, "symbol", None)
            if sym is None:
                continue

            c = label_map.get(sym)
            if c is None:
                continue

            # Get effective weight for this contract
            w = leg.weight_for_contract(c, date=date)

            # Get price in leg's native unit
            if sym not in prices_by_symbol:
                continue
            p = prices_by_symbol[sym].get(c, np.nan)

            if np.isfinite(p):
                # Convert price to reference unit
                p_converted = leg.convert_price_to_unit(p, self.reference_unit)
                v += w * p_converted

        return v

    def build_cps_series(
            self,
            method: str = "index",
            k: int = 0,
            target_months: Optional[float] = None,
            align_index: str = "intersection"
    ) -> pd.Series:
        """
        Build cross-product spread time series for chosen tenor rule.

        Parameters:
        -----------
        method : str
            "index" for M{k} alignment, "months" for months-to-expiry
        k : int
            Sequential index when method="index"
        target_months : float, optional
            Target months to expiry when method="months"
        align_index : str
            "intersection", "union", "left", or "right" for date alignment

        Returns:
        --------
        pd.Series
            CPS values indexed by datetime
        """
        # Get sequential data from all legs
        leg_data = []
        for leg in self.legs:
            if leg.data.seq_prices is None:
                raise ValueError(f"Leg with symbol {getattr(leg.data, 'symbol', 'unknown')} missing seq_prices")
            leg_data.append(leg.data)

        # Align date indices across all legs
        timestamps_list = [ld.seq_prices.index for ld in leg_data]
        if align_index == "intersection":
            idx = timestamps_list[0]
            for ts in timestamps_list[1:]:
                idx = idx.intersection(ts)
        elif align_index == "union":
            idx = timestamps_list[0]
            for ts in timestamps_list[1:]:
                idx = idx.union(ts)
        elif align_index == "left":
            idx = timestamps_list[0]
        elif align_index == "right":
            idx = timestamps_list[-1]
        else:
            raise ValueError(f"Unknown align_index: {align_index}")

        out = pd.Series(index=idx, dtype=float, name=f"CPS_{method}_{k if method == 'index' else target_months}")

        # Build cache key for this tenor rule
        cache_key = f"{method}_{k}_{target_months}"
        if cache_key not in self._alignment_cache:
            self._alignment_cache[cache_key] = {}

        # Iterate dates
        for t in idx:
            # Check cache first
            if t in self._alignment_cache[cache_key]:
                contract_labels = self._alignment_cache[cache_key][t]
            else:
                # Extract labels + DTE for this date for all legs
                contract_labels = []
                for i, data in enumerate(leg_data):
                    labels = list(data.seq_labels.loc[t]) if t in data.seq_labels.index else []
                    dte = list(data.seq_dte.loc[t]) if data.seq_dte is not None and t in data.seq_dte.index else []

                    # For first leg, determine alignment
                    if i == 0:
                        # Simple index-based or months-based selection
                        if method == "index":
                            selected = labels[k] if k < len(labels) else None
                        else:
                            # months-to-expiry based
                            selected = None
                            if target_months is not None:
                                best_err = 1e9
                                for j, lab in enumerate(labels):
                                    if j >= len(dte):
                                        continue
                                    dte_val = float(dte[j])
                                    if not np.isfinite(dte_val):
                                        continue
                                    m = dte_to_months(dte_val)
                                    if m is None:
                                        continue
                                    err = abs(m - target_months)
                                    if err < best_err:
                                        selected, best_err = lab, err
                        contract_labels.append(selected)
                    else:
                        # Align subsequent legs to first leg's tenor
                        first_labels = list(leg_data[0].seq_labels.loc[t]) if t in leg_data[0].seq_labels.index else []
                        first_dte = list(leg_data[0].seq_dte.loc[t]) if leg_data[0].seq_dte is not None and t in \
                                                                        leg_data[0].seq_dte.index else []

                        c1, c2 = align_tenors(
                            first_labels, first_dte, labels, dte,
                            method=method, k=k, target_months=target_months
                        )
                        contract_labels.append(c2)

                # Cache the alignment
                self._alignment_cache[cache_key][t] = contract_labels

            # Check if all contracts are available
            if any(c is None for c in contract_labels):
                continue

            # Build label_map and prices_by_symbol for all legs
            label_map = {}
            prices_by_symbol = {}

            for i, (leg, data, contract_label) in enumerate(zip(self.legs, leg_data, contract_labels)):
                sym = getattr(leg.data, "symbol", f"LEG{i}")

                if t not in data.seq_prices.index or t not in data.seq_labels.index:
                    continue

                labels = list(data.seq_labels.loc[t])
                prices_row = list(data.seq_prices.loc[t])

                label_map[sym] = contract_label
                prices_by_symbol[sym] = dict(zip(labels, prices_row))

            # Calculate cross value
            v = self.cross_value_on_date(
                date=t,
                label_map=label_map,
                prices_by_symbol=prices_by_symbol
            )

            out.at[t] = v

        return out.dropna()

    def build_cps_calendar(
            self,
            near_rule: Dict[str, Any],
            far_rule: Dict[str, Any],
            align_index: str = "intersection"
    ) -> pd.Series:
        """
        Build calendar of CPS: (CPS near) - (CPS far).

        Parameters:
        -----------
        near_rule : Dict[str, Any]
            Tenor rule for near leg (e.g., {"method": "index", "k": 0})
        far_rule : Dict[str, Any]
            Tenor rule for far leg (e.g., {"method": "index", "k": 1})
        align_index : str
            Date alignment method

        Returns:
        --------
        pd.Series
            Calendar spread of CPS

        Examples:
        ---------
        # (M0 - M1) of the cross
        >>> cps_cal = engine.build_cps_calendar(
        ...     near_rule={"method": "index", "k": 0},
        ...     far_rule={"method": "index", "k": 1}
        ... )

        # (3m - 12m) of the cross
        >>> cps_cal = engine.build_cps_calendar(
        ...     near_rule={"method": "months", "target_months": 3},
        ...     far_rule={"method": "months", "target_months": 12}
        ... )
        """
        cps_near = self.build_cps_series(align_index=align_index, **near_rule)
        cps_far = self.build_cps_series(align_index=align_index, **far_rule)

        calendar = (cps_near - cps_far).dropna()
        calendar.name = f"CPS_CAL_{near_rule.get('k', near_rule.get('target_months'))}_" \
                        f"{far_rule.get('k', far_rule.get('target_months'))}"

        return calendar

    def build_cps_family(
            self,
            k_values: Optional[List[int]] = None,
            month_targets: Optional[List[float]] = None,
            align_index: str = "intersection"
    ) -> Dict[str, pd.Series]:
        """
        Build family of CPS for multiple tenors.

        Parameters:
        -----------
        k_values : List[int], optional
            Sequential indices to build (e.g., [0, 1, 2] for M0, M1, M2)
        month_targets : List[float], optional
            Months-to-expiry targets (e.g., [3, 6, 12, 24])
        align_index : str
            Date alignment method

        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary of CPS series keyed by tenor identifier
        """
        results = {}

        # Build index-based CPS
        if k_values:
            for k in k_values:
                key = f"M{k}"
                results[key] = self.build_cps_series(
                    method="index",
                    k=k,
                    align_index=align_index
                )

        # Build months-based CPS
        if month_targets:
            for months in month_targets:
                key = f"{months}m"
                results[key] = self.build_cps_series(
                    method="months",
                    target_months=months,
                    align_index=align_index
                )

        return results

    def get_alignment_diagnostics(
            self,
            method: str = "index",
            k: int = 0,
            target_months: Optional[float] = None,
            n_samples: int = 10
    ) -> pd.DataFrame:
        """
        Get diagnostic report showing contract alignments across all legs per date.

        Parameters:
        -----------
        method : str
            Tenor alignment method
        k : int
            Sequential index if method="index"
        target_months : float, optional
            Target months if method="months"
        n_samples : int
            Number of sample dates to show

        Returns:
        --------
        pd.DataFrame
            Diagnostic report with contract alignments for all legs
        """
        cache_key = f"{method}_{k}_{target_months}"

        # Build series to populate cache if needed
        if cache_key not in self._alignment_cache or not self._alignment_cache[cache_key]:
            _ = self.build_cps_series(method=method, k=k, target_months=target_months)

        alignments = self._alignment_cache.get(cache_key, {})

        # Sample dates
        dates = list(alignments.keys())
        if len(dates) > n_samples:
            step = len(dates) // n_samples
            dates = dates[::step][:n_samples]

        records = []
        for date in dates:
            contract_labels = alignments[date]
            record = {
                'date': date,
                'method': method,
                'k': k if method == 'index' else None,
                'target_months': target_months if method == 'months' else None
            }
            # Add contract for each leg
            for i, (leg, label) in enumerate(zip(self.legs, contract_labels)):
                sym = getattr(leg.data, 'symbol', f'LEG{i}')
                record[f'{sym}_contract'] = label
                record[f'{sym}_weight'] = leg.base_weight
            records.append(record)

        return pd.DataFrame(records)

    def clear_cache(self):
        """Clear the tenor alignment cache."""
        self._alignment_cache.clear()

    def get_leg_weights(self) -> Dict[str, float]:
        """
        Return the base weight for each leg.

        Returns:
        --------
        Dict[str, float]
            Maps leg symbol to its base weight
        """
        weights = {}
        for i, leg in enumerate(self.legs):
            sym = getattr(leg.data, 'symbol', f'LEG{i}')
            weights[sym] = leg.base_weight
        return weights

    def get_leg_contributions(
            self,
            method: str = "index",
            k: int = 0,
            target_months: Optional[float] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate the price contribution of each leg to the cross-product spread.

        Parameters:
        -----------
        method : str
            "index" for M{k} alignment, "months" for months-to-expiry
        k : int
            Sequential index when method="index"
        target_months : float, optional
            Target months to expiry when method="months"

        Returns:
        --------
        Dict[str, pd.Series]
            Maps leg symbol to its contribution series (weighted prices in reference unit)
        """
        # Build the main CPS series to populate cache
        cps_series = self.build_cps_series(method=method, k=k, target_months=target_months)

        # Build individual contributions for each leg
        contributions = {}

        for i, leg in enumerate(self.legs):
            sym = getattr(leg.data, 'symbol', f'LEG{i}')
            leg_values = []

            for t in cps_series.index:
                # Get the aligned contract for this leg at this date
                cache_key = f"{method}_{k}_{target_months}"
                if cache_key in self._alignment_cache and t in self._alignment_cache[cache_key]:
                    contract_labels = self._alignment_cache[cache_key][t]
                    if i < len(contract_labels):
                        contract_label = contract_labels[i]
                    else:
                        leg_values.append(np.nan)
                        continue
                else:
                    leg_values.append(np.nan)
                    continue

                # Get the price for this contract
                if t not in leg.data.seq_prices.index or t not in leg.data.seq_labels.index:
                    leg_values.append(np.nan)
                    continue

                labels = list(leg.data.seq_labels.loc[t])
                prices_row = list(leg.data.seq_prices.loc[t])

                if contract_label not in labels:
                    leg_values.append(np.nan)
                    continue

                price_idx = labels.index(contract_label)
                if price_idx >= len(prices_row):
                    leg_values.append(np.nan)
                    continue

                price = prices_row[price_idx]

                # Apply weight and unit conversion
                weight = leg.weight_for_contract(contract_label, date=t)
                price_converted = leg.convert_price_to_unit(price, self.reference_unit)
                contribution = weight * price_converted

                leg_values.append(contribution)

            contributions[sym] = pd.Series(leg_values, index=cps_series.index, name=f"{sym}_contribution")

        return contributions


def create_simple_cps(
        base_symbol: str,
        hedge_symbol: str,
        base_weight: float = 1.0,
        hedge_weight: float = -1.0,
        k: int = 0
) -> pd.Series:
    """
    Convenience function to create simple front-month CPS.

    Parameters:
    -----------
    base_symbol : str
        Base product symbol (e.g., "CL_F")
    hedge_symbol : str
        Hedge product symbol (e.g., "HO_F")
    base_weight : float
        Weight for base leg
    hedge_weight : float
        Weight for hedge leg
    k : int
        Sequential index (0=front month)

    Returns:
    --------
    pd.Series
        Simple CPS time series
    """
    base_data = SpreadData(base_symbol)
    hedge_data = SpreadData(hedge_symbol)

    base_leg = CrossSpreadLeg(data=base_data, base_weight=base_weight)
    hedge_leg = CrossSpreadLeg(data=hedge_data, base_weight=hedge_weight)

    engine = CrossProductEngine(base_data, hedge_data, base_leg, hedge_leg)

    return engine.build_cps_series(method="index", k=k)


# ============================================================================
# Intraday Spread Engine
# ============================================================================


@dataclass
class IntradayLeg:
    """
    Single leg of an intraday spread using market close prices.

    Attributes:
    -----------
    symbol : str
        Contract symbol (e.g., 'CL_F', 'HO_F', 'ZC_F')
    data : pd.DataFrame
        Market OHLCV data from DataClient
    base_weight : float
        Base weight (±1 for long/short, can be any ratio)
    unit : str, optional
        Price unit for this leg (auto-detected from symbol if not provided)
    notional_multiplier : float
        Contract multiplier for notional scaling
    fx_rate : Union[float, pd.Series]
        FX conversion rate (can be time-varying)
    """
    symbol: str
    data: pd.DataFrame
    base_weight: float = 1.0
    unit: Optional[str] = None
    notional_multiplier: float = 1.0
    fx_rate: Union[float, pd.Series] = 1.0

    def __post_init__(self):
        """Auto-detect unit from symbol if not provided."""
        if self.unit is None:
            detected_unit, _ = detect_contract_unit(self.symbol)
            self.unit = detected_unit

    def convert_price_to_unit(self, price: float, target_unit: str) -> float:
        """
        Convert price from this leg's unit to target unit.

        Parameters:
        -----------
        price : float
            Price in this leg's unit
        target_unit : str
            Target unit for conversion

        Returns:
        --------
        float
            Converted price
        """
        return convert_units(price, self.unit, target_unit, symbol=self.symbol)

    def get_effective_weight(self, date: Optional[datetime] = None) -> float:
        """
        Calculate effective weight incorporating notional and FX scaling.

        Parameters:
        -----------
        date : datetime, optional
            Date for time-varying FX rates

        Returns:
        --------
        float
            Effective weight
        """
        w = self.base_weight * self.notional_multiplier

        # Apply FX scaling
        if isinstance(self.fx_rate, pd.Series) and date is not None:
            fx = self.fx_rate.loc[date] if date in self.fx_rate.index else 1.0
            w *= fx
        else:
            w *= float(self.fx_rate)

        return w

    @classmethod
    def load_from_scid(cls, symbol,
                       path: str = None,
                       base_weight: float = 1.0,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       resample_rule='1T',
                       **kwargs):
        if not path:
            path = str(DLY_DATA_PATH)
        rdr = ScidReader(path)
        data = rdr.load_front_month_series(ticker=symbol, start=start_date, end=end_date, resample_rule=resample_rule)
        return cls(symbol, data, base_weight)

    @classmethod
    def load_from_dclient(
            cls,
            symbol: str,
            base_weight: float = 1.0,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            daily: bool = False,
            **kwargs
    ) -> 'IntradayLeg':
        """
        Load IntradayLeg from DataClient market data.

        Parameters:
        -----------
        symbol : str
            Contract symbol (e.g., 'CL_F', 'HO_F')
        base_weight : float
            Base weight for this leg (default 1.0)
        start_date : str, optional
            Start date for filtering (ISO format)
        end_date : str, optional
            End date for filtering (ISO format)
        daily : bool
            Use daily resampled data (default False for intraday)
        **kwargs
            Additional arguments passed to IntradayLeg constructor

        Returns:
        --------
        IntradayLeg
            Leg with loaded market data and auto-detected units
        """
        from ...data import DataClient

        client = DataClient()

        # Query market data
        market_data = client.query_market_data(
            tickers=[symbol],
            start_date=start_date,
            end_date=end_date,
            daily=daily
        )

        if symbol not in market_data:
            raise ValueError(f"No market data found for symbol {symbol}")

        df = market_data[symbol]

        # Ensure we have Close column
        if 'Close' not in df.columns:
            raise ValueError(f"Market data for {symbol} missing 'Close' column")

        return cls(
            symbol=symbol,
            data=df,
            base_weight=base_weight,
            **kwargs
        )


class IntradaySpreadEngine:
    """
    Build intraday cross-product spreads using market close prices.

    Similar to CrossProductEngine but simplified for intraday data:
    - Uses Close prices from market data (no tenor alignment)
    - Supports multi-leg spreads (2, 3, 4+ legs)
    - Automatic unit conversion to common reference unit
    - No calendar spread functionality (use CrossProductEngine for that)

    Examples:
    ---------
    # Simple 2-leg crack spread
    >>> cl_leg = IntradayLeg.load_from_dclient('CL_F', base_weight=1.0)
    >>> ho_leg = IntradayLeg.load_from_dclient('HO_F', base_weight=-1.0)
    >>> engine = IntradaySpreadEngine.from_legs([cl_leg, ho_leg])
    >>> spread_series = engine.build_spread_series()

    # 3-leg crack spread with custom weights
    >>> cl_leg = IntradayLeg.load_from_dclient('CL_F', base_weight=3.0)
    >>> ho_leg = IntradayLeg.load_from_dclient('HO_F', base_weight=-2.0)
    >>> rb_leg = IntradayLeg.load_from_dclient('RB_F', base_weight=-1.0)
    >>> engine = IntradaySpreadEngine.from_legs([cl_leg, ho_leg, rb_leg], reference_unit='$/bbl')
    >>> spread = engine.build_spread_series()
    """

    def __init__(
            self,
            legs: List[IntradayLeg],
            reference_unit: Optional[str] = None
    ):
        """
        Initialize IntradaySpreadEngine with flexible leg configuration.

        Parameters:
        -----------
        legs : List[IntradayLeg]
            List of 2+ IntradayLeg instances with configured weights
        reference_unit : str, optional
            Common unit to convert all prices to. If None, auto-detects from symbols.
        """
        if not legs or len(legs) < 2:
            raise ValueError("Must provide at least 2 legs")

        self.legs = legs

        # Auto-detect reference unit if not provided
        if reference_unit is None:
            symbols = [leg.symbol for leg in self.legs]
            reference_unit = detect_reference_unit(symbols)

        self.reference_unit = reference_unit

    @classmethod
    def from_legs(
            cls,
            legs: List[IntradayLeg],
            reference_unit: Optional[str] = None
    ) -> 'IntradaySpreadEngine':
        """
        Create IntradaySpreadEngine from a list of legs.

        Parameters:
        -----------
        legs : List[IntradayLeg]
            List of 2+ IntradayLeg instances
        reference_unit : str, optional
            Common unit to convert all prices to. If None, auto-detects from symbols.

        Returns:
        --------
        IntradaySpreadEngine
            Engine with multi-leg configuration

        Examples:
        ---------
        # 3-2-1 crack spread - auto-detects $/bbl
        >>> cl = IntradayLeg.load_from_dclient('CL_F', base_weight=3.0)
        >>> rb = IntradayLeg.load_from_dclient('RB_F', base_weight=-2.0)
        >>> ho = IntradayLeg.load_from_dclient('HO_F', base_weight=-1.0)
        >>> engine = IntradaySpreadEngine.from_legs([cl, rb, ho])

        # Grain spread - auto-detects cents/bu
        >>> zc = IntradayLeg.load_from_dclient('ZC_F', base_weight=1.0)
        >>> zs = IntradayLeg.load_from_dclient('ZS_F', base_weight=-1.0)
        >>> engine = IntradaySpreadEngine.from_legs([zc, zs])
        """
        return cls(legs=legs, reference_unit=reference_unit)

    def spread_value_on_date(
            self,
            date: datetime,
            prices_by_symbol: Dict[str, float]
    ) -> float:
        """
        Calculate spread value at a specific date with automatic unit conversion.

        Parameters:
        -----------
        date : datetime
            Date/timestamp for valuation
        prices_by_symbol : Dict[str, float]
            Maps symbol -> close price in leg's native unit

        Returns:
        --------
        float
            Spread value in reference unit (NaN if ANY leg has invalid price)
        """
        v = 0.0

        for leg in self.legs:
            if leg.symbol not in prices_by_symbol:
                return np.nan

            p = prices_by_symbol[leg.symbol]

            if not np.isfinite(p):
                return np.nan

            # Get effective weight
            w = leg.get_effective_weight(date=date)

            # Convert price to reference unit
            p_converted = leg.convert_price_to_unit(p, self.reference_unit)
            v += w * p_converted

        return v

    def build_spread_series(
            self,
            align_index: str = "intersection",
            return_ohlc: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Build intraday spread time series using synchronized OHLC or Close prices.
        Uses DataFrame broadcasting with intersected indices for efficient computation.

        Parameters:
        -----------
        align_index : str
            "intersection", "union", "left", or "right" for timestamp alignment
        return_ohlc : bool
            If True, returns DataFrame with OHLC columns calculated from leg OHLC data
            If False, returns Series with Close prices only (default)

        Returns:
        --------
        pd.Series or pd.DataFrame
            If return_ohlc=False: pd.Series of spread close prices
            If return_ohlc=True: pd.DataFrame with Open, High, Low, Close columns
        """
        # Check if all legs have OHLC data
        has_ohlc = all(
            all(col in leg.data.columns for col in ['Open', 'High', 'Low', 'Close'])
            for leg in self.legs
        )

        if return_ohlc and not has_ohlc:
            raise ValueError("Cannot return OHLC: one or more legs missing OHLC columns")

        # Build DataFrames for each leg with proper index alignment
        if return_ohlc:
            # Create OHLC DataFrames for each leg
            leg_ohlc_dfs = []
            for leg in self.legs:
                df = leg.data[['Open', 'High', 'Low', 'Close']].copy()

                # Apply weight and unit conversion
                weight = leg.get_effective_weight()

                # Convert prices to reference unit
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col].apply(lambda p: leg.convert_price_to_unit(p, self.reference_unit) if np.isfinite(p) else np.nan)

                # Apply weight
                df = df * weight
                df.columns = pd.MultiIndex.from_product([[leg.symbol], df.columns])
                leg_ohlc_dfs.append(df)

            # Combine all leg DataFrames
            combined_df = pd.concat(leg_ohlc_dfs, axis=1)

            # Align index based on method
            if align_index == "intersection":
                combined_df = combined_df.dropna()
            elif align_index == "union":
                pass  # Keep all timestamps with NaNs
            elif align_index == "left":
                combined_df = combined_df.loc[leg_ohlc_dfs[0].index]
            elif align_index == "right":
                combined_df = combined_df.loc[leg_ohlc_dfs[-1].index]
            else:
                raise ValueError(f"Unknown align_index: {align_index}")

            # Broadcast sum across all legs for each OHLC column
            result_df = pd.DataFrame(index=combined_df.index)
            for ohlc_col in ['Open', 'High', 'Low', 'Close']:
                # Sum across all legs for this OHLC column
                col_sum = combined_df.xs(ohlc_col, level=1, axis=1).sum(axis=1)
                result_df[ohlc_col] = col_sum

            symbols_str = "_".join([leg.symbol.replace('_F', '') for leg in self.legs])
            result_df.columns.name = f"Spread_{symbols_str}"

            return result_df.dropna()

        else:
            # Close-only series using DataFrame broadcasting
            close_dfs = []
            for leg in self.legs:
                # Get Close prices
                close_series = leg.data['Close'].copy()

                # Apply weight and unit conversion
                weight = leg.get_effective_weight()

                # Convert to reference unit and apply weight
                close_series = close_series.apply(
                    lambda p: weight * leg.convert_price_to_unit(p, self.reference_unit) if np.isfinite(p) else np.nan
                )
                close_series.name = leg.symbol
                close_dfs.append(close_series)

            # Combine into DataFrame
            combined_df = pd.concat(close_dfs, axis=1)

            # Align index based on method
            if align_index == "intersection":
                combined_df = combined_df.dropna()
            elif align_index == "union":
                pass  # Keep all timestamps with NaNs
            elif align_index == "left":
                combined_df = combined_df.loc[close_dfs[0].index]
            elif align_index == "right":
                combined_df = combined_df.loc[close_dfs[-1].index]
            else:
                raise ValueError(f"Unknown align_index: {align_index}")

            # Broadcast sum across all legs
            spread_series = combined_df.sum(axis=1)

            # Name the series
            symbols_str = "_".join([leg.symbol.replace('_F', '') for leg in self.legs])
            weights_str = "_".join([f"{leg.base_weight:+.1f}".replace('.', 'p') for leg in self.legs])
            spread_series.name = f"Spread_{symbols_str}_{weights_str}"

            return spread_series.dropna()

    def get_diagnostics(self, n_samples: int = 10) -> pd.DataFrame:
        """
        Get diagnostic report showing leg prices and spread calculation.

        Parameters:
        -----------
        n_samples : int
            Number of sample timestamps to show

        Returns:
        --------
        pd.DataFrame
            Diagnostic report with leg prices, weights, and spread values
        """
        spread_series = self.build_spread_series()

        # Sample timestamps
        timestamps = list(spread_series.index)
        if len(timestamps) > n_samples:
            step = len(timestamps) // n_samples
            timestamps = timestamps[::step][:n_samples]

        records = []
        for t in timestamps:
            record = {'timestamp': t}

            # Add each leg's info
            for leg in self.legs:
                if t in leg.data.index:
                    price = leg.data.loc[t, 'Close']
                    weight = leg.get_effective_weight(date=t)
                    price_converted = leg.convert_price_to_unit(price, self.reference_unit)

                    record[f'{leg.symbol}_price'] = price
                    record[f'{leg.symbol}_unit'] = leg.unit
                    record[f'{leg.symbol}_weight'] = weight
                    record[f'{leg.symbol}_converted'] = price_converted
                    record[f'{leg.symbol}_contribution'] = weight * price_converted

            record['spread_value'] = spread_series.loc[t] if t in spread_series.index else np.nan
            records.append(record)

        return pd.DataFrame(records)


@dataclass
class SyntheticSymbol:
    legs: List[Union[CrossSpreadLeg, IntradayLeg]]
    type: str = "product"
    ticker: str = "CS"
    full_name: str = "Custom Spread"
    intraday: bool = False

    def __post_init__(self):
        # Auto-detect intraday mode if not explicitly set
        if not self.intraday:
            # Check if any leg is IntradayLeg type
            self.intraday = any(isinstance(leg, IntradayLeg) for leg in self.legs)

        # Validate legs are compatible
        has_intraday = any(isinstance(leg, IntradayLeg) for leg in self.legs)
        has_cross = any(isinstance(leg, CrossSpreadLeg) for leg in self.legs)

        if has_intraday and has_cross:
            raise ValueError(
                "Cannot mix IntradayLeg and CrossSpreadLeg in same SyntheticSymbol. "
                "All legs must be the same type."
            )

        # Create appropriate engine
        if self.intraday or isinstance(self.legs[0], IntradayLeg):
            self.data_engine = IntradaySpreadEngine(legs=self.legs)
        else:
            self.data_engine = CrossProductEngine.from_legs(legs=self.legs)

        # Generate individual leg price data
        self.legs_df = self._generate_legs_data()

        # Calculate combined spread price
        self.price = self._calculate_legs_price()

    @classmethod
    def from_weights_dict(
            cls,
            weights: Dict[str, float],
            path: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            resample_rule: str = '1T',
            ticker: Optional[str] = None,
            full_name: Optional[str] = None,
            spread_type: str = "product",
            reference_unit: Optional[str] = None,
            **kwargs
    ) -> 'SyntheticSymbol':
        """
        Create SyntheticSymbol from a dictionary mapping tickers to hedge ratios.
        Loads data from SCID files using IntradayLeg.load_from_scid().

        Parameters:
        -----------
        weights : Dict[str, float]
            Dictionary mapping ticker symbols to weights/hedge ratios
            Example: {'CL_F': 1.0, 'HO_F': -2.0, 'RB_F': -1.0}
        path : str, optional
            Path to SCID files directory (defaults to DLY_DATA_PATH if None)
        start_date : str, optional
            Start date for data filtering (ISO format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for data filtering (ISO format: 'YYYY-MM-DD')
        resample_rule : str
            Resampling rule for tick data (default: '1T' for 1-minute bars)
            Examples: '30S' (30-second), '1T' (1-minute), '5T' (5-minute), '1H' (hourly)
        ticker : str, optional
            Custom ticker symbol for the spread (auto-generated if None)
        full_name : str, optional
            Custom full name for the spread (auto-generated if None)
        spread_type : str
            Type of spread: 'product', 'calendar', etc. (default: 'product')
        reference_unit : str, optional
            Common unit to convert all prices to (auto-detects if None)
        **kwargs
            Additional arguments passed to IntradayLeg.load_from_scid()

        Returns:
        --------
        SyntheticSymbol
            Initialized synthetic symbol with loaded legs

        Examples:
        ---------
        # 3-2-1 crack spread from weights dict
        >>> weights = {'CL_F': 3.0, 'HO_F': -2.0, 'RB_F': -1.0}
        >>> crack_spread = SyntheticSymbol.from_weights_dict(
        ...     weights,
        ...     path='/path/to/scid/files',
        ...     start_date='2020-01-01',
        ...     resample_rule='1T',
        ...     ticker='CRACK_321',
        ...     full_name='3-2-1 Crack Spread'
        ... )

        # Simple 2-leg corn-soybean spread
        >>> weights = {'ZC_F': 1.0, 'ZS_F': -1.0}
        >>> grain_spread = SyntheticSymbol.from_weights_dict(
        ...     weights,
        ...     start_date='2020-01-01',
        ...     ticker='ZC_ZS',
        ...     full_name='Corn-Soybean Spread'
        ... )

        # Gold-silver ratio spread with 15-minute bars
        >>> weights = {'GC_F': 1.0, 'SI_F': -1.0}
        >>> metal_spread = SyntheticSymbol.from_weights_dict(
        ...     weights,
        ...     resample_rule='15T',
        ...     ticker='GC_SI_RATIO'
        ... )
        """
        if not weights or len(weights) < 2:
            raise ValueError("Must provide at least 2 tickers with weights")

        # Load legs using IntradayLeg.load_from_scid()
        legs = []
        symbols = []

        for symbol, weight in weights.items():
            symbols.append(symbol)

            # Load IntradayLeg from SCID files
            leg = IntradayLeg.load_from_scid(
                symbol=symbol,
                path=path,
                base_weight=weight,
                start_date=start_date,
                end_date=end_date,
                resample_rule=resample_rule,
                **kwargs
            )

            legs.append(leg)

        # Auto-generate ticker if not provided
        if ticker is None:
            # Create ticker from symbols: CL_HO_RB or CL_HO depending on length
            symbol_codes = [s.replace('_F', '') for s in symbols]
            ticker = '_'.join(symbol_codes)
            if len(ticker) > 15:  # Limit ticker length
                ticker = ticker[:15]

        # Auto-generate full name if not provided
        if full_name is None:
            # Create descriptive name showing weights
            leg_strs = [f"{w:+.1f}*{s.replace('_F', '')}" for s, w in weights.items()]
            full_name = ' '.join(leg_strs) + ' Spread'

        # Create and return SyntheticSymbol
        return cls(
            legs=legs,
            type=spread_type,
            ticker=ticker,
            full_name=full_name,
            intraday=True  # Always True since we're loading from SCID files
        )

    @classmethod
    def load_from_parquets(cls, weights_map:Dict[str, float], path:str, start_date:Optional[str] = None, resample='1s'):
        from CTAFlow.data import AsyncParquetWriter
        symbols = [*weights_map.keys()]
        pq_reader = AsyncParquetWriter()
        price_data = pq_reader.batch_read_to_dict_sync(symbols)
        legs = []
        for s in symbols:
            legs.append(
            IntradayLeg(s, price_data[s], weights_map[s])

            )

        return cls()

    def _generate_legs_data(self) -> pd.DataFrame:
        """
        Generate price data (OHLC if available, Close otherwise) for each leg.

        Returns:
        --------
        pd.DataFrame
            Multi-column DataFrame with legs as level 0 columns and OHLC/Close as level 1
            Index is datetime timestamps
        """
        if self.intraday:
            return self._generate_intraday_legs_data()
        else:
            return self._generate_daily_legs_data()

    def _generate_intraday_legs_data(self) -> pd.DataFrame:
        """Generate granular OHLC data for intraday legs with full market data."""
        leg_data_dict = {}

        for leg in self.legs:
            symbol = leg.symbol
            df = leg.data

            # Check if OHLC columns are all valid
            required_ohlc = ['Open', 'High', 'Low', 'Close']
            has_valid_ohlc = all(
                col in df.columns and df[col].notna().sum() > 0
                for col in required_ohlc
            )

            if has_valid_ohlc:
                # Use full OHLC data
                leg_data_dict[(symbol, 'Open')] = df['Open']
                leg_data_dict[(symbol, 'High')] = df['High']
                leg_data_dict[(symbol, 'Low')] = df['Low']
                leg_data_dict[(symbol, 'Close')] = df['Close']

                # Add volume if available
                if 'Volume' in df.columns:
                    leg_data_dict[(symbol, 'Volume')] = df['Volume']
                if 'BidVolume' in df.columns and 'AskVolume' in df.columns:
                    leg_data_dict[(symbol, 'BidVolume')] = df['BidVolume']
                    leg_data_dict[(symbol, 'AskVolume')] = df['AskVolume']
            else:
                # Fallback to Close only
                if 'Close' in df.columns:
                    leg_data_dict[(symbol, 'Close')] = df['Close']

        # Create MultiIndex DataFrame
        result = pd.DataFrame(leg_data_dict)
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=['symbol', 'field'])

        return result

    def _generate_daily_legs_data(self) -> pd.DataFrame:
        """Generate price data for daily/sequentialized legs (CrossProductEngine)."""
        leg_data_dict = {}

        for leg in self.legs:
            # Get symbol - handle both CrossSpreadLeg and IntradayLeg
            if hasattr(leg, 'symbol'):
                symbol = leg.symbol
            elif hasattr(leg.data, 'symbol'):
                symbol = leg.data.symbol
            else:
                symbol = f'LEG_{id(leg)}'

            # For CrossSpreadLeg, use seq_prices for M0 (front month)
            if hasattr(leg.data, 'seq_prices') and leg.data.seq_prices is not None:
                seq_prices = leg.data.seq_prices
                seq_labels = leg.data.seq_labels

                # Extract M0 (front month) prices
                m0_prices = []
                timestamps = []

                for ts in seq_prices.index:
                    if ts in seq_labels.index:
                        labels = list(seq_labels.loc[ts])
                        prices = list(seq_prices.loc[ts])

                        # Get M0 (first contract)
                        if len(prices) > 0 and len(labels) > 0:
                            m0_price = prices[0]
                            if np.isfinite(m0_price):
                                m0_prices.append(m0_price)
                                timestamps.append(ts)

                if m0_prices:
                    # Create series for M0 prices
                    price_series = pd.Series(m0_prices, index=timestamps, name=symbol)
                    leg_data_dict[(symbol, 'Close')] = price_series
            # Handle IntradayLeg that might have been used with CrossProductEngine
            elif hasattr(leg, 'data') and isinstance(leg.data, pd.DataFrame):
                if 'Close' in leg.data.columns:
                    leg_data_dict[(symbol, 'Close')] = leg.data['Close']

        # Create MultiIndex DataFrame
        if leg_data_dict:
            result = pd.DataFrame(leg_data_dict)
            result.columns = pd.MultiIndex.from_tuples(result.columns, names=['symbol', 'field'])
        else:
            # Empty DataFrame with proper structure
            result = pd.DataFrame()

        return result

    def _calculate_legs_price(
        self,
        columns: str = "close",
        resample_rule: Optional[str] = None,
        spread_rules: Optional[Dict[str, Any]] = None,
        resample_kwargs: Optional[Dict[str, Any]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate spread price from legs with optional resampling.

        Parameters:
        -----------
        columns : str
            "close" for last price only (default)
            "ohlc" for full OHLC resampled data (only valid for intraday mode)
        resample_rule : str, optional
            Pandas resample rule (e.g., "5min", "1H", "1D")
            If None, returns raw spread series without resampling
        spread_rules : Dict[str, Any], optional
            Rules for calendar spread construction (only for CrossProductEngine)
        resample_kwargs : Dict[str, Any], optional
            Additional kwargs passed to pandas resample

        Returns:
        --------
        pd.Series or pd.DataFrame
            If columns="close": Returns pd.Series of spread prices
            If columns="ohlc" and intraday=True: Returns pd.DataFrame with OHLC columns
            If columns="ohlc" and intraday=False: Raises ValueError (cannot create OHLC from daily data)

        Raises:
        -------
        ValueError
            If OHLC is requested for non-intraday data (only Close prices available)
        """
        if self.intraday:
            # Intraday engine with OHLC support - has actual tick/bar data
            return_ohlc = columns.lower() == "ohlc"

            # Build spread series (OHLC or Close)
            spread_data = self.data_engine.build_spread_series(return_ohlc=return_ohlc)

            # No resampling requested - return raw data
            if resample_rule is None:
                return spread_data

            # Set default resample kwargs if not provided
            if resample_kwargs is None:
                resample_kwargs = {}

            # Resample based on data type
            if return_ohlc:
                # OHLC DataFrame - resample each column appropriately
                resampler = spread_data.resample(resample_rule, **resample_kwargs)
                result = pd.DataFrame({
                    'Open': resampler['Open'].first(),
                    'High': resampler['High'].max(),
                    'Low': resampler['Low'].min(),
                    'Close': resampler['Close'].last()
                })
                return result.dropna()
            else:
                # Close series - resample with last value
                return spread_data.resample(resample_rule, **resample_kwargs).last()

        else:
            # Cross-product engine uses sequentialized contracts (daily/Close only)
            # Cannot create valid OHLC from daily Close prices
            if columns.lower() == "ohlc":
                raise ValueError(
                    "OHLC data cannot be calculated for non-intraday spreads. "
                    "Daily/CrossProductEngine mode only provides Close prices from sequentialized contracts. "
                    "Use columns='close' or set intraday=True with IntradayLeg data."
                )

            # Handle calendar spreads
            if "calendar" in self.type.lower():
                if isinstance(self.data_engine, CrossProductEngine):
                    if spread_rules is None:
                        spread_rules = {"k_values": [0, 3, 6, 11], "month_targets": [1, 4, 6, 12]}
                    self.seq_curve = self.data_engine.build_cps_family(**spread_rules)

            # Build front month spread series (Close only)
            spread_series = self.data_engine.build_cps_series(method="index", k=0)

            # No resampling requested - return raw series
            if resample_rule is None:
                return spread_series

            # Set default resample kwargs if not provided
            if resample_kwargs is None:
                resample_kwargs = {}

            return spread_series.resample(resample_rule, **resample_kwargs).last()

    def save_to_hdf(
        self,
        ticker: Optional[str] = None,
        client: Optional[DataClient] = None
    ) -> None:
        """
        Save this SyntheticSymbol to HDF5 storage via DataClient.

        Uses DataClient.save_synthetic_symbol() under the hood with proper
        storage structure: market/synthetic/{ticker}

        Parameters:
        -----------
        ticker : str, optional
            Ticker symbol for storage. If None, uses self.ticker
        client : DataClient, optional
            DataClient instance to use. If None, creates new instance.

        Examples:
        ---------
        >>> weights = {'CL_F': 3.0, 'HO_F': -2.0, 'RB_F': -1.0}
        >>> crack = SyntheticSymbol.from_weights_dict(weights, ticker='CRACK_321')
        >>> crack.save_to_hdf()  # Saves to market/synthetic/CRACK_321
        >>> crack.save_to_hdf(ticker='CRACK_CUSTOM')  # Override ticker
        """
        if client is None:
            client = DataClient()

        client.save_synthetic_symbol(self, ticker=ticker)

    @classmethod
    def load_from_dclient(
        cls,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        reconstruct: bool = True,
        client: Optional[DataClient] = None
    ) -> 'SyntheticSymbol':
        """
        Load SyntheticSymbol from DataClient storage.

        Reconstructs the full object by reloading legs from market data
        if reconstruct=True (default).

        Parameters:
        -----------
        ticker : str
            Ticker symbol of stored synthetic spread
        start_date : str, optional
            Start date for filtering (ISO format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for filtering (ISO format: 'YYYY-MM-DD')
        reconstruct : bool
            If True (default), reconstructs full SyntheticSymbol by reloading legs.
            If False, returns faster but with limited functionality.
        client : DataClient, optional
            DataClient instance to use. If None, creates new instance.

        Returns:
        --------
        SyntheticSymbol
            Loaded and reconstructed synthetic symbol

        Examples:
        ---------
        >>> # Load full synthetic symbol with all legs reconstructed
        >>> crack = SyntheticSymbol.load_from_dclient('CRACK_321')
        >>> print(crack.price.head())

        >>> # Load with date filtering
        >>> crack_2024 = SyntheticSymbol.load_from_dclient(
        ...     'CRACK_321',
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31'
        ... )
        """
        if client is None:
            client = DataClient()

        return client.load_synthetic_symbol(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            reconstruct=reconstruct
        )
