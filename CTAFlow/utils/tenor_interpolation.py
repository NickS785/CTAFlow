"""
Fixed tenor interpolation utilities for constant-maturity PCA analysis.

This module provides functions to interpolate futures prices onto fixed tenor grids
using log-linear interpolation as specified in the PCA model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Tuple, List,Any
from datetime import datetime, timedelta
import warnings


def log_interp_to_tau(
    df_prices: pd.DataFrame,
    expiries: pd.Series,
    taus: np.ndarray,
    method: str = 'log_linear'
) -> Dict[float, pd.Series]:
    """
    Interpolate futures prices to fixed tenors using log-linear interpolation.
    
    This is the core function from the PCA model that creates constant-maturity
    time series by interpolating log prices onto fixed tenor grid.
    
    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame with index=date, columns=contract_id, values=settle price
    expiries : pd.Series
        Series mapping contract_id -> expiry Timestamp
    taus : np.ndarray
        Target constant maturities in years (e.g., months/12)
    method : str, default 'log_linear'
        Interpolation method ('log_linear', 'cubic_hermite')
        
    Returns
    -------
    Dict[float, pd.Series]
        Dictionary mapping tau -> Series of log F(t,tau)
    """
    # Validate and clean taus array
    if not isinstance(taus, (np.ndarray, list)):
        taus = np.array(taus)
    
    # Ensure all taus are numeric and finite
    try:
        taus_clean = []
        for tau in taus:
            tau_float = float(tau)
            if np.isfinite(tau_float) and tau_float > 0:
                taus_clean.append(tau_float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid tau value in taus array. All values must be numeric and positive. Error: {e}")
    
    if len(taus_clean) == 0:
        raise ValueError("No valid tau values found. All values must be positive and finite.")
    
    taus = np.array(taus_clean)
    out = {tau: pd.Series(index=df_prices.index, dtype=float) for tau in taus}
    
    # Pre-compute time-to-maturity matrix in years per day
    for t, row in df_prices.iterrows():
        avail = row.dropna()
        if avail.empty:
            continue

        expiry_values = pd.to_datetime(expiries.loc[avail.index])
        T = (expiry_values - pd.Timestamp(t)).dt.days / 365.25
        lnF = np.log(avail.values.astype(float))
        
        # Sort by time to maturity
        T_array = T.to_numpy()
        order = np.argsort(T_array)
        T = T_array[order]
        lnF = np.asarray(lnF)[order]
        
        # Interpolate for each target tau
        for tau in taus:
            if len(T) < 2:
                # Only one contract available
                out[tau].at[t] = lnF[0] if len(T) == 1 else np.nan
                continue
                
            if tau <= T[0]:
                # Extrapolate before first contract
                y = lnF[0] + (lnF[1] - lnF[0]) * (tau - T[0]) / (T[1] - T[0])
            elif tau >= T[-1]:
                # Extrapolate after last contract
                y = lnF[-2] + (lnF[-1] - lnF[-2]) * (tau - T[-2]) / (T[-1] - T[-2])
            else:
                # Interpolate between contracts
                j = np.searchsorted(T, tau)
                t0, t1 = T[j-1], T[j]
                y0, y1 = lnF[j-1], lnF[j]
                y = y0 + (y1 - y0) * (tau - t0) / (t1 - t0)
            
            out[tau].at[t] = y
    
    # Return only non-empty series
    return {tau: s.dropna() for tau, s in out.items() if not s.dropna().empty}


def create_tenor_grid(
    min_tau: float = 1/12,
    max_tau: float = 3.0,
    tenor_type: str = 'monthly',
    custom_tenors: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create standardized tenor grid for constant-maturity analysis.
    
    Parameters
    ----------
    min_tau : float, default 1/12
        Minimum tenor in years (1 month)
    max_tau : float, default 3.0
        Maximum tenor in years (36 months)
    tenor_type : str, default 'monthly'
        Grid type: 'monthly', 'quarterly', 'mixed', 'custom'
    custom_tenors : np.ndarray, optional
        Custom tenor array (used if tenor_type='custom')
        
    Returns
    -------
    np.ndarray
        Array of tenor values in years
    """
    # Validate input parameters
    try:
        min_tau = float(min_tau)
        max_tau = float(max_tau)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid type {type(min_tau) if 'min_tau' in str(e) else type(max_tau)}. Must be int or float.")
    
    if not np.isfinite(min_tau) or not np.isfinite(max_tau):
        raise ValueError("min_tau and max_tau must be finite values")
    
    if min_tau <= 0 or max_tau <= 0:
        raise ValueError("min_tau and max_tau must be positive")
        
    if min_tau >= max_tau:
        raise ValueError("min_tau must be less than max_tau")
    
    if tenor_type == 'custom' and custom_tenors is not None:
        return np.array(custom_tenors, dtype=float)
    
    if tenor_type == 'monthly':
        # Monthly tenors from min_tau to max_tau
        months = np.arange(
            max(1, int(min_tau * 12)), 
            int(max_tau * 12) + 1
        )
        return months / 12.0
    
    elif tenor_type == 'quarterly':
        # Quarterly tenors
        quarters = np.arange(1, int(max_tau * 4) + 1)
        return quarters / 4.0
    
    elif tenor_type == 'mixed':
        # Mixed grid: monthly for first year, quarterly after
        monthly = np.arange(1, 13) / 12.0  # 1-12 months
        quarterly = np.arange(5, int(max_tau * 4) + 1) / 4.0  # 15+ months
        return np.concatenate([monthly, quarterly])
    
    else:
        raise ValueError(f"Unknown tenor_type: {tenor_type}")


def cubic_hermite_interp(
    T: np.ndarray,
    lnF: np.ndarray,
    tau: float
) -> float:
    """
    Cubic Hermite interpolation for dense contracts.
    
    Parameters
    ----------
    T : np.ndarray
        Time to maturity array
    lnF : np.ndarray
        Log futures prices
    tau : float
        Target tenor
        
    Returns
    -------
    float
        Interpolated log price
    """
    if len(T) < 4:
        # Fall back to linear interpolation
        j = np.searchsorted(T, tau)
        if j == 0:
            j = 1
        elif j >= len(T):
            j = len(T) - 1
        
        t0, t1 = T[j-1], T[j]
        y0, y1 = lnF[j-1], lnF[j]
        return y0 + (y1 - y0) * (tau - t0) / (t1 - t0)
    
    # Cubic Hermite implementation
    j = np.searchsorted(T, tau)
    if j == 0:
        j = 1
    elif j >= len(T):
        j = len(T) - 1
    
    # Use 4-point stencil around target
    i0 = max(0, j - 2)
    i1 = min(len(T), j + 2)
    
    T_local = T[i0:i1]
    lnF_local = lnF[i0:i1]
    
    # Simple cubic spline through 4 points
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(T_local, lnF_local, bc_type='natural')
    return float(cs(tau))


def interpolate_curve_snapshot(
    prices: pd.Series,
    expiries: pd.Series,
    taus: np.ndarray,
    trade_date: pd.Timestamp,
    method: str = 'log_linear'
) -> pd.Series:
    """
    Interpolate a single curve snapshot to fixed tenors.
    
    Parameters
    ----------
    prices : pd.Series
        Contract prices (contract_id -> price)
    expiries : pd.Series
        Contract expiries (contract_id -> expiry)
    taus : np.ndarray
        Target tenors in years
    trade_date : pd.Timestamp
        Trade date for TTM calculation
    method : str, default 'log_linear'
        Interpolation method
        
    Returns
    -------
    pd.Series
        Interpolated log prices at fixed tenors
    """
    # Calculate time to maturity
    T = (expiries.loc[prices.index].values - trade_date).days / 365.25
    lnF = np.log(prices.values.astype(float))
    
    # Sort by TTM
    order = np.argsorted(T)
    T = T[order]
    lnF = lnF[order]
    
    results = {}
    for tau in taus:
        if len(T) < 2:
            results[tau] = lnF[0] if len(T) == 1 else np.nan
            continue
        
        if method == 'cubic_hermite' and len(T) >= 4:
            y = cubic_hermite_interp(T, lnF, tau)
        else:
            # Log-linear interpolation
            if tau <= T[0]:
                y = lnF[0] + (lnF[1] - lnF[0]) * (tau - T[0]) / (T[1] - T[0])
            elif tau >= T[-1]:
                y = lnF[-2] + (lnF[-1] - lnF[-2]) * (tau - T[-2]) / (T[-1] - T[-2])
            else:
                j = np.searchsorted(T, tau)
                t0, t1 = T[j-1], T[j]
                y0, y1 = lnF[j-1], lnF[j]
                y = y0 + (y1 - y0) * (tau - t0) / (t1 - t0)
        
        results[tau] = y
    
    return pd.Series(results, index=taus)


def build_constant_maturity_panel(
    price_data: pd.DataFrame,
    expiry_data: pd.Series,
    tenor_grid: np.ndarray,
    interpolation_method: str = 'log_linear',
    min_contracts: int = 2
) -> pd.DataFrame:
    """
    Build complete constant-maturity panel for PCA analysis.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price data (dates x contracts)
    expiry_data : pd.Series
        Contract expiry mapping (contract_id -> expiry)
    tenor_grid : np.ndarray
        Fixed tenor grid in years
    interpolation_method : str, default 'log_linear'
        Interpolation method
    min_contracts : int, default 2
        Minimum contracts required per date
        
    Returns
    -------
    pd.DataFrame
        Constant-maturity log prices (dates x tenors)
    """
    # Get interpolated series for each tenor
    tenor_dict = log_interp_to_tau(
        price_data, 
        expiry_data, 
        tenor_grid, 
        method=interpolation_method
    )
    
    # Combine into DataFrame
    panel = pd.DataFrame(tenor_dict)
    
    # Filter dates with insufficient contracts
    valid_dates = []
    for date in panel.index:
        n_valid = panel.loc[date].notna().sum()
        if n_valid >= min_contracts:
            valid_dates.append(date)
    
    panel = panel.loc[valid_dates]
    
    return panel.sort_index()


class TenorInterpolator:
    """
    Class for managing tenor interpolation with caching and validation.
    """
    
    def __init__(
        self, 
        tenor_grid: np.ndarray,
        method: str = 'log_linear',
        min_contracts: int = 2,
        cache_results: bool = True
    ):
        """
        Initialize tenor interpolator.
        
        Parameters
        ----------
        tenor_grid : np.ndarray
            Fixed tenor grid in years
        method : str, default 'log_linear'
            Interpolation method
        min_contracts : int, default 2
            Minimum contracts per date
        cache_results : bool, default True
            Whether to cache interpolation results
        """
        self.tenor_grid = np.array(tenor_grid)
        self.method = method
        self.min_contracts = min_contracts
        self.cache_results = cache_results
        self._cache = {}
    
    def interpolate(
        self,
        price_data: pd.DataFrame,
        expiry_data: pd.Series
    ) -> pd.DataFrame:
        """
        Interpolate prices to fixed tenor grid.
        
        Parameters
        ----------
        price_data : pd.DataFrame
            Price data to interpolate
        expiry_data : pd.Series
            Expiry mapping
            
        Returns
        -------
        pd.DataFrame
            Constant-maturity panel
        """
        cache_key = f"{id(price_data)}_{id(expiry_data)}_{self.method}"
        
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]
        
        result = build_constant_maturity_panel(
            price_data,
            expiry_data,
            self.tenor_grid,
            self.method,
            self.min_contracts
        )
        
        if self.cache_results:
            self._cache[cache_key] = result
        
        return result
    
    def validate_interpolation(
        self,
        original_data: pd.DataFrame,
        interpolated_data: pd.DataFrame,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Validate interpolation quality.
        
        Parameters
        ----------
        original_data : pd.DataFrame
            Original price data
        interpolated_data : pd.DataFrame
            Interpolated constant-maturity data
        tolerance : float, default 0.01
            Maximum acceptable error
            
        Returns
        -------
        dict
            Validation metrics
        """
        metrics = {
            'dates_retained': len(interpolated_data) / len(original_data),
            'tenor_coverage': len(self.tenor_grid),
            'avg_contracts_per_date': original_data.notna().sum(axis=1).mean(),
            'interpolation_gaps': interpolated_data.isna().sum().sum(),
        }
        
        return metrics
    
    def clear_cache(self):
        """Clear interpolation cache."""
        self._cache.clear()


def validate_tenor_continuity(
    tenor_panel: pd.DataFrame,
    max_jump: float = 0.1,
    window: int = 5
) -> pd.DataFrame:
    """
    Validate continuity of tenor time series.
    
    Parameters
    ----------
    tenor_panel : pd.DataFrame
        Constant-maturity panel
    max_jump : float, default 0.1
        Maximum acceptable log price jump
    window : int, default 5
        Rolling window for smoothness check
        
    Returns
    -------
    pd.DataFrame
        Boolean mask of valid observations
    """
    # Calculate returns
    returns = tenor_panel.diff()
    
    # Flag large jumps
    jump_flags = np.abs(returns) > max_jump
    
    # Flag observations with too many recent jumps
    rolling_jumps = jump_flags.rolling(window=window, min_periods=1).sum()
    smooth_flags = rolling_jumps <= (window * 0.2)  # Max 20% jumps in window
    
    return smooth_flags & ~jump_flags