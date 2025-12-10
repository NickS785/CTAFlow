"""Generic screen parameter generators for seasonality, momentum, and orderflow screens.

This module provides factory functions for creating screen parameters instead of
module-level constants, reducing import overhead and improving flexibility.
"""

from ..data import SyntheticSymbol, IntradayLeg, IntradayFileManager, DataClient
from ..features import VolatilityRegimeClassifier
from . import HistoricalScreener, ScreenParams, OrderflowParams
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional

from ..config import DLY_DATA_PATH


# ============================================================================
# Constants and shared settings
# ============================================================================

def _get_vol_classifier():
    """Get default volatility regime classifier."""
    return VolatilityRegimeClassifier(method='rv', window=10)


def _default_period_length():
    """Default period length for time-of-day screens."""
    return timedelta(hours=1, minutes=30)


def _season_months():
    """Return standard seasonal month mappings."""
    return {
        'winter': [12, 1, 2, 3],
        'spring': [3, 4, 5],
        'summer': [5, 6, 7, 8],
        'fall': [9, 10, 11],
        'q1': [1, 2, 3],
        'q2': [4, 5, 6],
        'q3': [7, 8, 9],
        'q4': [10, 11, 12],
    }


# ============================================================================
# Session definitions
# ============================================================================

def _session_times():
    """Return session time definitions for major trading regions."""
    return {
        'london': {
            'start': time(hour=2, minute=30),
            'end': time(hour=11, minute=30),
            'target_times': ["03:00", "03:30", "02:30", "07:30", "09:30"],
        },
        'usa': {
            'start': time(hour=8, minute=30),
            'end': time(hour=15, minute=30),
            'target_times': ["08:30", "10:30", "9:00", "13:30"],
        },
        'asia': {
            'start': time(hour=18, minute=0),  # 18:00 CST = Asia morning
            'end': time(hour=2, minute=0),     # 02:00 CST = Asia close
            'target_times': ["18:30", "19:00", "20:00", "23:00", "01:00"],
        },
        'livestock': {
            'start': time(hour=8, minute=30),
            'end': time(hour=13, minute=0),
            'target_times': ["9:00", "9:30", "11:00", "12:00"],
        },
        'grain': {
            'start': time(hour=7, minute=0),
            'end': time(hour=13, minute=0),
            'target_times': ["07:30", "09:30", "11:00", "12:00"],
        },
    }


# ============================================================================
# Factory functions for seasonality screens
# ============================================================================

def make_seasonality_screen(
    name: str,
    session: str = 'usa',
    months: Optional[List[int]] = None,
    target_times: Optional[List[str]] = None,
    period_length: Optional[timedelta] = None,
    **kwargs
) -> ScreenParams:
    """Create a seasonality screen with standardized parameters.

    Parameters
    ----------
    name : str
        Screen name
    session : str
        Session type: 'london', 'usa', 'asia', 'livestock', 'grain'
    months : Optional[List[int]]
        Month filter (1-12), None for all months
    target_times : Optional[List[str]]
        Override default target times for session
    period_length : Optional[timedelta]
        Override default period length
    **kwargs
        Additional parameters passed to ScreenParams

    Returns
    -------
    ScreenParams
        Configured seasonality screen parameters
    """
    sessions = _session_times()
    session_info = sessions.get(session, sessions['usa'])

    if target_times is None:
        target_times = session_info['target_times']

    if period_length is None:
        period_length = _default_period_length()

    return ScreenParams(
        screen_type="seasonality",
        name=name,
        months=months,
        target_times=target_times,
        period_length=period_length,
        seasonality_session_start=session_info['start'],
        seasonality_session_end=session_info['end'],
        **kwargs
    )


def make_momentum_screen(
    name: str,
    session: str = 'usa',
    months: Optional[List[int]] = None,
    sess_start_hrs: int = 1,
    sess_start_minutes: int = 30,
    st_momentum_days: int = 5,
    use_regime: bool = True,
    target_regimes: Optional[List[int]] = None,
    **kwargs
) -> ScreenParams:
    """Create a momentum screen with standardized parameters.

    Parameters
    ----------
    name : str
        Screen name
    session : str
        Session type: 'london', 'usa', 'asia', 'livestock', 'grain'
    months : Optional[List[int]]
        Month filter (1-12), None for all months
    sess_start_hrs : int
        Hours from session start for momentum calculation
    sess_start_minutes : int
        Minutes from session start for momentum calculation
    st_momentum_days : int
        Days for short-term momentum calculation
    use_regime : bool
        Whether to use volatility regime filtering
    target_regimes : Optional[List[int]]
        Target volatility regimes (default: [0, 1])
    **kwargs
        Additional parameters passed to ScreenParams

    Returns
    -------
    ScreenParams
        Configured momentum screen parameters
    """
    sessions = _session_times()
    session_info = sessions.get(session, sessions['usa'])

    # For momentum screens, we typically use multiple session starts/ends
    # Default to London + USA sessions for multi-session support
    if 'session_starts' not in kwargs:
        if session == 'asia':
            kwargs['session_starts'] = ["18:00", "02:00"]
            kwargs['session_ends'] = ["02:00", "08:00"]
        elif session == 'livestock' or session == 'grain':
            kwargs['session_starts'] = [session_info['start'].strftime("%H:%M")]
            kwargs['session_ends'] = [session_info['end'].strftime("%H:%M")]
        else:
            kwargs['session_starts'] = ["02:30", "08:30"]
            kwargs['session_ends'] = ["10:30", "15:00"]

    params = {
        'screen_type': "momentum",
        'name': name,
        'months': months,
        'sess_start_hrs': sess_start_hrs,
        'sess_start_minutes': sess_start_minutes,
        'st_momentum_days': st_momentum_days,
    }

    if use_regime:
        params['regime_settings'] = _get_vol_classifier()
        params['target_regimes'] = target_regimes or [0, 1]

    params.update(kwargs)
    return ScreenParams(**params)


def make_orderflow_screen(
    name: str,
    session: str = 'usa',
    months: Optional[List[int]] = None,
    vpin_window: int = 25,
    **kwargs
) -> OrderflowParams:
    """Create an orderflow screen with standardized parameters.

    Parameters
    ----------
    name : str
        Screen name
    session : str
        Session type: 'london', 'usa', 'asia'
    months : Optional[List[int]]
        Month filter (1-12), None for all months
    vpin_window : int
        VPIN calculation window
    **kwargs
        Additional parameters passed to OrderflowParams

    Returns
    -------
    OrderflowParams
        Configured orderflow screen parameters
    """
    sessions = _session_times()
    session_info = sessions.get(session, sessions['usa'])

    return OrderflowParams(
        name=name,
        session_start=session_info['start'].strftime("%H:%M"),
        session_end=session_info['end'].strftime("%H:%M"),
        month_filter=months,
        vpin_window=vpin_window,
        **kwargs
    )


# ============================================================================
# Screen generators by region
# ============================================================================

def usa_seasonality_screens(quarterly: bool = True) -> List[ScreenParams]:
    """Generate USA seasonality screens.

    Parameters
    ----------
    quarterly : bool
        If True, use quarterly splits (Q1-Q4). If False, use seasonal splits (winter/spring/summer/fall)

    Returns
    -------
    List[ScreenParams]
        List of USA seasonality screen parameters
    """
    seasons = _season_months()

    if quarterly:
        periods = ['q1', 'q2', 'q3', 'q4']
    else:
        periods = ['winter', 'spring', 'summer', 'fall']

    screens = []
    for period in periods:
        screens.append(make_seasonality_screen(
            name=f"usa_{period}",
            session='usa',
            months=seasons[period]
        ))

    # Add all-months screen
    screens.append(make_seasonality_screen(name="usa_all", session='usa'))

    return screens


def london_seasonality_screens(quarterly: bool = True) -> List[ScreenParams]:
    """Generate London seasonality screens.

    Parameters
    ----------
    quarterly : bool
        If True, use quarterly splits. If False, use seasonal splits

    Returns
    -------
    List[ScreenParams]
        List of London seasonality screen parameters
    """
    seasons = _season_months()

    if quarterly:
        periods = ['q1', 'q2', 'q3', 'q4']
    else:
        periods = ['winter', 'spring', 'summer', 'fall']

    screens = []
    for period in periods:
        screens.append(make_seasonality_screen(
            name=f"london_{period}",
            session='london',
            months=seasons[period]
        ))

    # Add all-months screen
    screens.append(make_seasonality_screen(name="london_all", session='london'))

    return screens


def asia_seasonality_screens(quarterly: bool = True) -> List[ScreenParams]:
    """Generate Asia session seasonality screens.

    Asia session runs 18:00 CST - 02:00 CST, covering Tokyo, Hong Kong, Singapore markets.

    Parameters
    ----------
    quarterly : bool
        If True, use quarterly splits. If False, use seasonal splits

    Returns
    -------
    List[ScreenParams]
        List of Asia seasonality screen parameters
    """
    seasons = _season_months()

    if quarterly:
        periods = ['q1', 'q2', 'q3', 'q4']
    else:
        periods = ['winter', 'spring', 'summer', 'fall']

    screens = []
    for period in periods:
        screens.append(make_seasonality_screen(
            name=f"asia_{period}",
            session='asia',
            months=seasons[period]
        ))

    # Add all-months screen
    screens.append(make_seasonality_screen(name="asia_all", session='asia'))

    return screens


def usa_momentum_screens(seasonal: bool = True) -> List[ScreenParams]:
    """Generate USA momentum screens.

    Parameters
    ----------
    seasonal : bool
        If True, create seasonal splits. If False, only create all-months screen

    Returns
    -------
    List[ScreenParams]
        List of USA momentum screen parameters
    """
    screens = []
    seasons = _season_months()

    if seasonal:
        for season_name in ['winter', 'spring', 'summer', 'fall']:
            screens.append(make_momentum_screen(
                name=f"usa_{season_name}_momentum",
                session='usa',
                months=seasons[season_name]
            ))

    # Add all-months screen
    screens.append(make_momentum_screen(name="momentum_generic", session='usa'))

    return screens


def asia_momentum_screens(seasonal: bool = True) -> List[ScreenParams]:
    """Generate Asia session momentum screens.

    Parameters
    ----------
    seasonal : bool
        If True, create seasonal splits. If False, only create all-months screen

    Returns
    -------
    List[ScreenParams]
        List of Asia momentum screen parameters
    """
    screens = []
    seasons = _season_months()

    if seasonal:
        for season_name in ['winter', 'spring', 'summer', 'fall']:
            screens.append(make_momentum_screen(
                name=f"asia_{season_name}_momentum",
                session='asia',
                months=seasons[season_name]
            ))

    # Add all-months screen
    screens.append(make_momentum_screen(name="asia_momentum_generic", session='asia'))

    return screens


def livestock_screens() -> Dict[str, List[ScreenParams]]:
    """Generate livestock-specific screens (seasonal and momentum).

    Returns
    -------
    Dict[str, List[ScreenParams]]
        Dictionary with 'seasonal' and 'momentum' keys containing screen lists
    """
    seasons = _season_months()

    seasonal_screens = []
    momentum_screens = []

    # Seasonal screens
    for season_name in ['winter', 'spring', 'summer', 'fall']:
        seasonal_screens.append(make_seasonality_screen(
            name=f"livestock_seasonal_{season_name}",
            session='livestock',
            months=seasons[season_name]
        ))
    seasonal_screens.append(make_seasonality_screen(
        name="livestock_seasonal",
        session='livestock'
    ))

    # Momentum screens
    for season_name in ['winter', 'spring', 'summer', 'fall']:
        momentum_screens.append(make_momentum_screen(
            name=f"livestock_{season_name}_momo",
            session='livestock',
            months=seasons[season_name]
        ))
    momentum_screens.append(make_momentum_screen(
        name="livestock_momentum",
        session='livestock'
    ))

    return {
        'seasonal': seasonal_screens,
        'momentum': momentum_screens
    }


def grain_screens(seasonal: bool = True) -> List[ScreenParams]:
    """Generate grain-specific screens.

    Parameters
    ----------
    seasonal : bool
        If True, create seasonal splits. If False, only create all-months screen

    Returns
    -------
    List[ScreenParams]
        List of grain screen parameters
    """
    screens = []
    seasons = _season_months()

    if seasonal:
        for season_name in ['winter', 'spring', 'summer', 'fall']:
            screens.append(ScreenParams(
                screen_type="seasonality",
                name=f"grain_{season_name}",
                months=seasons[season_name],
                session_starts=["18:00", "07:00"],
                session_ends=["13:00", "13:00"],
                period_length=_default_period_length(),
                target_times=["07:30", "09:30", "11:00", "12:00"],
                seasonality_session_start=time(hour=7, minute=0),
                seasonality_session_end=time(hour=13, minute=0)
            ))

    # Add all-months screen
    screens.append(ScreenParams(
        screen_type="seasonality",
        name="grain_all",
        session_starts=["18:00", "07:00"],
        session_ends=["13:00", "13:00"],
        period_length=_default_period_length(),
        target_times=["07:30", "09:30", "11:00", "12:00"],
        seasonality_session_start=time(hour=7, minute=0),
        seasonality_session_end=time(hour=13, minute=0)
    ))

    return screens


def usa_orderflow_screens(seasonal: bool = True) -> List[OrderflowParams]:
    """Generate USA orderflow screens.

    Parameters
    ----------
    seasonal : bool
        If True, create seasonal splits. If False, only create all-months screen

    Returns
    -------
    List[OrderflowParams]
        List of USA orderflow screen parameters
    """
    screens = []
    seasons = _season_months()

    if seasonal:
        for season_name in ['winter', 'spring', 'summer', 'fall']:
            screens.append(make_orderflow_screen(
                name=f"us_of_{season_name}",
                session='usa',
                months=seasons[season_name]
            ))

    # Add all-months screen
    screens.append(make_orderflow_screen(name="us_of", session='usa'))

    return screens


def london_orderflow_screens(seasonal: bool = True) -> List[OrderflowParams]:
    """Generate London orderflow screens.

    Parameters
    ----------
    seasonal : bool
        If True, create seasonal splits. If False, only create all-months screen

    Returns
    -------
    List[OrderflowParams]
        List of London orderflow screen parameters
    """
    screens = []
    seasons = _season_months()

    if seasonal:
        for season_name in ['winter', 'spring', 'summer', 'fall']:
            screens.append(make_orderflow_screen(
                name=f"london_of_{season_name}",
                session='london',
                months=seasons[season_name],
                vpin_window=30 if season_name == 'winter' else 25
            ))

    # Add all-months screen
    screens.append(make_orderflow_screen(
        name="london_of",
        session='london',
        vpin_window=30
    ))

    return screens


def asia_orderflow_screens(seasonal: bool = True) -> List[OrderflowParams]:
    """Generate Asia session orderflow screens.

    Parameters
    ----------
    seasonal : bool
        If True, create seasonal splits. If False, only create all-months screen

    Returns
    -------
    List[OrderflowParams]
        List of Asia orderflow screen parameters
    """
    screens = []
    seasons = _season_months()

    if seasonal:
        for season_name in ['winter', 'spring', 'summer', 'fall']:
            screens.append(make_orderflow_screen(
                name=f"asia_of_{season_name}",
                session='asia',
                months=seasons[season_name]
            ))

    # Add all-months screen
    screens.append(make_orderflow_screen(name="asia_of", session='asia'))

    return screens


# ============================================================================
# Convenience getters for common configurations
# ============================================================================

def get_all_seasonality_screens(regions: Optional[List[str]] = None) -> List[ScreenParams]:
    """Get all seasonality screens for specified regions.

    Parameters
    ----------
    regions : Optional[List[str]]
        List of regions: 'usa', 'london', 'asia', 'livestock', 'grain'
        If None, returns all regions

    Returns
    -------
    List[ScreenParams]
        Combined list of seasonality screens
    """
    if regions is None:
        regions = ['usa', 'london', 'asia', 'livestock', 'grain']

    all_screens = []
    for region in regions:
        if region == 'usa':
            all_screens.extend(usa_seasonality_screens())
        elif region == 'london':
            all_screens.extend(london_seasonality_screens())
        elif region == 'asia':
            all_screens.extend(asia_seasonality_screens())
        elif region == 'livestock':
            all_screens.extend(livestock_screens()['seasonal'])
        elif region == 'grain':
            all_screens.extend(grain_screens())

    return all_screens


def get_all_momentum_screens(regions: Optional[List[str]] = None) -> List[ScreenParams]:
    """Get all momentum screens for specified regions.

    Parameters
    ----------
    regions : Optional[List[str]]
        List of regions: 'usa', 'asia', 'livestock'
        If None, returns all regions

    Returns
    -------
    List[ScreenParams]
        Combined list of momentum screens
    """
    if regions is None:
        regions = ['usa', 'asia', 'livestock']

    all_screens = []
    for region in regions:
        if region == 'usa':
            all_screens.extend(usa_momentum_screens())
        elif region == 'asia':
            all_screens.extend(asia_momentum_screens())
        elif region == 'livestock':
            all_screens.extend(livestock_screens()['momentum'])

    return all_screens


def get_all_orderflow_screens(regions: Optional[List[str]] = None) -> List[OrderflowParams]:
    """Get all orderflow screens for specified regions.

    Parameters
    ----------
    regions : Optional[List[str]]
        List of regions: 'usa', 'london', 'asia'
        If None, returns all regions

    Returns
    -------
    List[OrderflowParams]
        Combined list of orderflow screens
    """
    if regions is None:
        regions = ['usa', 'london', 'asia']

    all_screens = []
    for region in regions:
        if region == 'usa':
            all_screens.extend(usa_orderflow_screens())
        elif region == 'london':
            all_screens.extend(london_orderflow_screens())
        elif region == 'asia':
            all_screens.extend(asia_orderflow_screens())

    return all_screens


# ============================================================================
# Backward compatibility - lazy instantiation
# ============================================================================

# Only create instances when explicitly requested to maintain backward compatibility
# without incurring the cost of creating all objects at import time

def get_usa_quarterly_params() -> List[ScreenParams]:
    """Backward compatibility: USA quarterly seasonality screens."""
    return usa_seasonality_screens(quarterly=True)


def get_london_quarterly_params() -> List[ScreenParams]:
    """Backward compatibility: London quarterly seasonality screens."""
    return london_seasonality_screens(quarterly=True)


def get_usa_seasonals() -> List[ScreenParams]:
    """Backward compatibility: USA seasonal screens."""
    return usa_seasonality_screens(quarterly=False)


def get_london_seasonals() -> List[ScreenParams]:
    """Backward compatibility: London seasonal screens."""
    return london_seasonality_screens(quarterly=False)


def get_momentums() -> List[ScreenParams]:
    """Backward compatibility: USA momentum screens."""
    return usa_momentum_screens(seasonal=True)


def get_us_of_all() -> List[OrderflowParams]:
    """Backward compatibility: USA orderflow screens."""
    return usa_orderflow_screens(seasonal=True)


def get_london_of_all() -> List[OrderflowParams]:
    """Backward compatibility: London orderflow screens."""
    return london_orderflow_screens(seasonal=True)
