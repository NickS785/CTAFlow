"""
Contract Expiry Rules for Different Futures Types

This module provides accurate expiry date calculations for different futures contracts.
Expiry rules vary significantly by contract type and exchange.
"""
import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExpiryRule(Enum):
    """Types of expiry rules"""
    THIRD_FRIDAY = "third_friday"  # Most financial futures
    ENERGY_NYMEX = "energy_nymex"  # NYMEX energy (CL, HO, RB, NG)
    AGRICULTURAL = "agricultural"  # Most ag futures
    METALS = "metals"  # Metals futures
    FX = "fx"  # Currency futures


# Mapping of tickers to expiry rules
TICKER_EXPIRY_RULES: Dict[str, ExpiryRule] = {
    # Energy (NYMEX) - expire 3-4 business days before end of month prior to delivery
    'CL_F': ExpiryRule.ENERGY_NYMEX,  # Crude Oil
    'HO_F': ExpiryRule.ENERGY_NYMEX,  # Heating Oil
    'RB_F': ExpiryRule.ENERGY_NYMEX,  # RBOB Gasoline
    'NG_F': ExpiryRule.ENERGY_NYMEX,  # Natural Gas

    # Financial Index Futures - 3rd Friday
    'ES_F': ExpiryRule.THIRD_FRIDAY,  # E-mini S&P 500
    'NQ_F': ExpiryRule.THIRD_FRIDAY,  # E-mini NASDAQ
    'YM_F': ExpiryRule.THIRD_FRIDAY,  # E-mini Dow
    'RTY_F': ExpiryRule.THIRD_FRIDAY,  # E-mini Russell 2000

    # Treasury Futures - Last business day of month
    'ZB_F': ExpiryRule.THIRD_FRIDAY,  # 30-Year T-Bond
    'ZN_F': ExpiryRule.THIRD_FRIDAY,  # 10-Year T-Note
    'ZF_F': ExpiryRule.THIRD_FRIDAY,  # 5-Year T-Note
    'ZT_F': ExpiryRule.THIRD_FRIDAY,  # 2-Year T-Note

    # Agricultural - varies, but most around mid-month
    'ZC_F': ExpiryRule.AGRICULTURAL,  # Corn
    'ZS_F': ExpiryRule.AGRICULTURAL,  # Soybeans
    'ZW_F': ExpiryRule.AGRICULTURAL,  # Wheat
    'ZL_F': ExpiryRule.AGRICULTURAL,  # Soybean Oil
    'ZM_F': ExpiryRule.AGRICULTURAL,  # Soybean Meal
    'KC_F': ExpiryRule.AGRICULTURAL,  # Coffee
    'CT_F': ExpiryRule.AGRICULTURAL,  # Cotton
    'SB_F': ExpiryRule.AGRICULTURAL,  # Sugar

    # Metals
    'GC_F': ExpiryRule.METALS,  # Gold
    'SI_F': ExpiryRule.METALS,  # Silver
    'HG_F': ExpiryRule.METALS,  # Copper
    'PL_F': ExpiryRule.METALS,  # Platinum
    'PA_F': ExpiryRule.METALS,  # Palladium

    # Livestock
    'LE_F': ExpiryRule.AGRICULTURAL,  # Live Cattle
    'HE_F': ExpiryRule.AGRICULTURAL,  # Lean Hogs
    'GF_F': ExpiryRule.AGRICULTURAL,  # Feeder Cattle

    # Currency (CME) - 2 business days before 3rd Wednesday
    'EC_F': ExpiryRule.FX,  # Euro FX
    'BP_F': ExpiryRule.FX,  # British Pound
    'JY_F': ExpiryRule.FX,  # Japanese Yen
    'AD_F': ExpiryRule.FX,  # Australian Dollar
    'CD_F': ExpiryRule.FX,  # Canadian Dollar
    'SF_F': ExpiryRule.FX,  # Swiss Franc
}


def get_nth_business_day(year: int, month: int, n: int, reverse: bool = False) -> pd.Timestamp:
    """
    Get the nth business day of a month.

    Parameters:
    -----------
    year : int
    month : int
    n : int
        Which business day (1-indexed)
    reverse : bool
        If True, count from end of month backwards

    Returns:
    --------
    pd.Timestamp of the nth business day
    """
    # Get all days in the month
    if month == 12:
        end_date = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(days=1)
    else:
        end_date = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)

    start_date = pd.Timestamp(year, month, 1)

    # Generate business days
    business_days = pd.bdate_range(start_date, end_date)

    if len(business_days) < n:
        logger.warning(f"Month {year}-{month} has only {len(business_days)} business days, requested {n}")
        return business_days[-1] if business_days.any() else end_date

    if reverse:
        return business_days[-n]
    else:
        return business_days[n - 1]


def get_nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """
    Get the nth occurrence of a weekday in a month.

    Parameters:
    -----------
    year : int
    month : int
    weekday : int
        0=Monday, 1=Tuesday, ..., 4=Friday
    n : int
        Which occurrence (1=first, 2=second, 3=third, etc.)

    Returns:
    --------
    pd.Timestamp of the nth weekday
    """
    first_day = pd.Timestamp(year, month, 1)

    # Find first occurrence of the weekday
    days_ahead = (weekday - first_day.dayofweek) % 7
    first_occurrence = first_day + pd.Timedelta(days=days_ahead)

    # Add weeks to get nth occurrence
    nth_occurrence = first_occurrence + pd.Timedelta(weeks=n - 1)

    # Make sure it's still in the same month
    if nth_occurrence.month != month:
        logger.warning(f"nth weekday {n} of weekday {weekday} doesn't exist in {year}-{month}")
        return first_occurrence + pd.Timedelta(weeks=n - 2)

    return nth_occurrence


def expiry_third_friday(year: int, month: int) -> pd.Timestamp:
    """
    Calculate 3rd Friday of the month.
    Used for most financial index futures (ES, NQ, etc.)
    """
    return get_nth_weekday(year, month, 4, 3)  # 4 = Friday, 3 = third


def expiry_energy_nymex(year: int, month: int) -> pd.Timestamp:
    """
    Calculate NYMEX energy contract expiry.

    For CL, HO, RB:
    - Trading ceases 3 business days before the 25th calendar day
      of the month PRIOR to the delivery month

    For example:
    - September contract expires around August 22nd
      (3 business days before August 25th)

    Parameters:
    -----------
    year : int
        Delivery year
    month : int
        Delivery month

    Returns:
    --------
    pd.Timestamp of expiry date
    """
    # Get month PRIOR to delivery month
    if month == 1:
        prior_year = year - 1
        prior_month = 12
    else:
        prior_year = year
        prior_month = month - 1

    # 25th of the prior month
    try:
        target_date = pd.Timestamp(prior_year, prior_month, 25)
    except ValueError:
        # If 25th doesn't exist (shouldn't happen), use last day
        if prior_month == 12:
            target_date = pd.Timestamp(prior_year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            target_date = pd.Timestamp(prior_year, prior_month + 1, 1) - pd.Timedelta(days=1)

    # Go back 3 business days
    expiry = target_date
    business_days_back = 0
    while business_days_back < 3:
        expiry = expiry - pd.Timedelta(days=1)
        # Check if it's a business day (Monday=0, Sunday=6)
        if expiry.dayofweek < 5:  # Weekday
            business_days_back += 1

    return expiry


def expiry_agricultural(year: int, month: int) -> pd.Timestamp:
    """
    Agricultural futures expiry (approximate).

    Most ag futures expire around the 15th business day of the month
    prior to delivery month. This is an approximation - actual rules
    vary by contract.
    """
    # Use month prior to delivery for most ag contracts
    if month == 1:
        prior_year = year - 1
        prior_month = 12
    else:
        prior_year = year
        prior_month = month - 1

    # Around 15th business day
    return get_nth_business_day(prior_year, prior_month, 15)


def expiry_metals(year: int, month: int) -> pd.Timestamp:
    """
    Metals futures expiry.

    Most metals expire around the end of the month prior to delivery.
    """
    if month == 1:
        prior_year = year - 1
        prior_month = 12
    else:
        prior_year = year
        prior_month = month - 1

    # Last business day of prior month
    return get_nth_business_day(prior_year, prior_month, 1, reverse=True)


def expiry_fx(year: int, month: int) -> pd.Timestamp:
    """
    FX futures expiry.

    CME FX futures expire 2 business days before the 3rd Wednesday
    of the contract month.
    """
    # 3rd Wednesday
    third_wednesday = get_nth_weekday(year, month, 2, 3)  # 2 = Wednesday

    # Go back 2 business days
    expiry = third_wednesday
    business_days_back = 0
    while business_days_back < 2:
        expiry = expiry - pd.Timedelta(days=1)
        if expiry.dayofweek < 5:  # Weekday
            business_days_back += 1

    return expiry


# Map expiry rules to functions
EXPIRY_FUNCTIONS: Dict[ExpiryRule, Callable] = {
    ExpiryRule.THIRD_FRIDAY: expiry_third_friday,
    ExpiryRule.ENERGY_NYMEX: expiry_energy_nymex,
    ExpiryRule.AGRICULTURAL: expiry_agricultural,
    ExpiryRule.METALS: expiry_metals,
    ExpiryRule.FX: expiry_fx,
}


def calculate_expiry(ticker: str, year: int, month: int) -> pd.Timestamp:
    """
    Calculate expiry date for a given ticker and delivery month.

    Parameters:
    -----------
    ticker : str
        Ticker symbol (e.g., 'CL_F', 'ES_F')
    year : int
        Delivery year
    month : int
        Delivery month (1-12)

    Returns:
    --------
    pd.Timestamp of the expiry date

    Example:
    --------
    >>> calculate_expiry('CL_F', 2024, 9)  # September 2024 CL
    Timestamp('2024-08-22 00:00:00')  # Expires ~Aug 22nd
    """
    # Get expiry rule for this ticker
    expiry_rule = TICKER_EXPIRY_RULES.get(ticker)

    if expiry_rule is None:
        # Default to 3rd Friday if unknown
        logger.warning(f"Unknown ticker {ticker}, using 3rd Friday rule")
        expiry_rule = ExpiryRule.THIRD_FRIDAY

    # Get corresponding function
    expiry_func = EXPIRY_FUNCTIONS[expiry_rule]

    # Calculate expiry
    try:
        expiry = expiry_func(year, month)
        return expiry
    except Exception as e:
        logger.error(f"Error calculating expiry for {ticker} {year}-{month}: {e}")
        # Fallback to simple 3rd Friday
        return expiry_third_friday(year, month)


def get_roll_buffer_days(ticker: str) -> int:
    """
    Get recommended number of days before expiry to roll.

    For continuous contracts, we typically roll at or after expiry:
    - Negative values mean roll AFTER expiry
    - Zero means roll AT expiry
    - Positive values mean roll BEFORE expiry

    For energy futures (CL, HO, RB, NG), we roll 1 day AFTER expiry
    since new contracts start trading in the evening session the day after expiry.

    Parameters:
    -----------
    ticker : str
        Ticker symbol

    Returns:
    --------
    int : Number of days before expiry to roll (negative = after expiry)
    """
    expiry_rule = TICKER_EXPIRY_RULES.get(ticker, ExpiryRule.THIRD_FRIDAY)

    # Buffers for continuous contract rolling
    # Negative = roll AFTER expiry, Positive = roll BEFORE expiry
    buffers = {
        ExpiryRule.THIRD_FRIDAY: 0,     # Financial futures - roll at expiry
        ExpiryRule.ENERGY_NYMEX: -1,    # Energy - roll 1 day AFTER expiry
        ExpiryRule.AGRICULTURAL: 0,     # Ag - roll at expiry
        ExpiryRule.METALS: 0,           # Metals - roll at expiry
        ExpiryRule.FX: 0,               # FX - roll at expiry
    }

    return buffers.get(expiry_rule, 0)


if __name__ == "__main__":
    # Test expiry calculations
    print("="*70)
    print("EXPIRY DATE CALCULATIONS TEST")
    print("="*70)

    test_cases = [
        ('CL_F', 2024, 9),   # Sep 2024 Crude Oil
        ('ES_F', 2024, 9),   # Sep 2024 E-mini S&P
        ('NG_F', 2024, 10),  # Oct 2024 Natural Gas
        ('ZC_F', 2024, 12),  # Dec 2024 Corn
        ('GC_F', 2024, 12),  # Dec 2024 Gold
    ]

    for ticker, year, month in test_cases:
        expiry = calculate_expiry(ticker, year, month)
        buffer = get_roll_buffer_days(ticker)
        roll_date = expiry - pd.Timedelta(days=buffer)

        print(f"\n{ticker} {pd.Timestamp(year, month, 1).strftime('%B %Y')} contract:")
        print(f"  Expiry: {expiry.strftime('%Y-%m-%d %A')}")
        print(f"  Roll date (with {buffer}d buffer): {roll_date.strftime('%Y-%m-%d %A')}")
