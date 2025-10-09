"""
Volatility-weighted returns and EWMA volatility scaling utilities for PCA model.

This module provides functions to calculate vol-adjusted returns and EWMA volatility 
estimates as required by the PCA forward curve model.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any


def ewma_volatility(
    returns: Union[pd.Series, pd.DataFrame], 
    halflife: int = 60,
    min_periods: int = 10
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate exponentially weighted moving average volatility.
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Log returns or price changes
    halflife : int, default 60
        EWMA halflife in periods (trading days)
    min_periods : int, default 10
        Minimum periods required for calculation
        
    Returns
    -------
    pd.Series or pd.DataFrame
        EWMA volatility estimates
    """
    if isinstance(returns, pd.Series):
        if returns.empty:
            return pd.Series(index=returns.index, dtype=float)
        return returns.ewm(halflife=halflife, min_periods=min_periods, adjust=False).std()
    
    # DataFrame case
    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        if not returns[col].dropna().empty:
            result[col] = returns[col].ewm(
                halflife=halflife, min_periods=min_periods, adjust=False
            ).std()
    return result

def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series).diff()

def vol_weighted_returns(
    returns: Union[pd.Series, pd.DataFrame],
    vol_halflife: int = 60,
    min_vol: float = 1e-6,
    forward_fill_vol: bool = True
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate volatility-weighted (vol-scaled) returns for PCA analysis.
    
    This function computes vol-adjusted returns as:
    vol_weighted_return = return / ewma_volatility
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Raw log returns
    vol_halflife : int, default 60
        EWMA halflife for volatility calculation
    min_vol : float, default 1e-6
        Minimum volatility floor to prevent division by zero
    forward_fill_vol : bool, default True
        Whether to forward fill volatility estimates
        
    Returns
    -------
    pd.Series or pd.DataFrame
        Volatility-weighted returns
    """
    # Calculate EWMA volatility
    vol_estimates = ewma_volatility(returns, halflife=vol_halflife)
    
    # Forward fill volatility if requested
    if forward_fill_vol:
        vol_estimates = vol_estimates.fillna(method='ffill')
    
    # Apply minimum volatility floor
    if isinstance(vol_estimates, pd.Series):
        vol_estimates = vol_estimates.clip(lower=min_vol)
    else:
        vol_estimates = vol_estimates.clip(lower=min_vol)
    
    # Calculate vol-weighted returns
    vol_weighted = returns / vol_estimates
    
    # Replace infinite values with NaN
    vol_weighted = vol_weighted.replace([np.inf, -np.inf], np.nan)
    
    return vol_weighted


def rolling_volatility_stats(
    returns: pd.DataFrame,
    window: int = 252,
    vol_halflife: int = 60
) -> Dict[str, pd.DataFrame]:
    """
    Calculate rolling volatility statistics for multiple series.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix (dates x tenors/contracts)
    window : int, default 252
        Rolling window size
    vol_halflife : int, default 60
        EWMA halflife for volatility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'vol_estimates': Rolling EWMA volatilities
        - 'vol_weighted': Volatility-weighted returns
        - 'correlation': Rolling correlation matrix of vol-weighted returns
    """
    # Calculate volatility estimates
    vol_estimates = ewma_volatility(returns, halflife=vol_halflife)
    
    # Calculate vol-weighted returns
    vol_weighted = vol_weighted_returns(returns, vol_halflife=vol_halflife)
    
    # Calculate rolling correlation matrix
    correlations = {}
    for i, date in enumerate(vol_weighted.index[window:], start=window):
        window_data = vol_weighted.iloc[i-window:i].dropna(axis=1, how='all')
        if len(window_data.columns) > 1:
            corr_matrix = window_data.corr()
            correlations[date] = corr_matrix
    
    return {
        'vol_estimates': vol_estimates,
        'vol_weighted': vol_weighted,
        'correlations': correlations
    }


def dynamic_vol_scaling(
    returns: pd.DataFrame,
    vol_window: int = 60,
    scaling_factor: float = 1.0,
    target_vol: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply dynamic volatility scaling with optional vol targeting.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix
    vol_window : int, default 60
        Window for volatility calculation
    scaling_factor : float, default 1.0
        Global scaling factor
    target_vol : float, optional
        Target volatility level (annualized)
        
    Returns
    -------
    pd.DataFrame
        Dynamically scaled returns
    """
    vol_estimates = ewma_volatility(returns, halflife=vol_window)
    
    if target_vol is not None:
        # Scale to target volatility
        current_vol = vol_estimates.mean(axis=1, skipna=True)
        scaling = target_vol / current_vol.clip(lower=1e-6)
        scaled_returns = returns.multiply(scaling, axis=0) * scaling_factor
    else:
        # Standard vol weighting
        scaled_returns = vol_weighted_returns(returns, vol_halflife=vol_window) * scaling_factor
    
    return scaled_returns.replace([np.inf, -np.inf], np.nan)


class VolatilityProcessor:
    """
    Class for advanced volatility processing and vol-weighted return analysis.
    """
    
    def __init__(
        self, 
        vol_halflife: int = 60,
        min_vol: float = 1e-6,
        scaling_factor: float = 1.0
    ):
        """
        Initialize volatility processor.
        
        Parameters
        ----------
        vol_halflife : int, default 60
            EWMA halflife for volatility
        min_vol : float, default 1e-6
            Minimum volatility floor
        scaling_factor : float, default 1.0
            Global scaling factor
        """
        self.vol_halflife = vol_halflife
        self.min_vol = min_vol
        self.scaling_factor = scaling_factor
        self._vol_cache = {}
    
    def process_returns(
        self, 
        returns: pd.DataFrame,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process returns with volatility weighting and caching.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Raw returns matrix
        use_cache : bool, default True
            Whether to use cached volatility estimates
            
        Returns
        -------
        dict
            Processed returns and volatility metrics
        """
        cache_key = f"{id(returns)}_{self.vol_halflife}"
        
        if use_cache and cache_key in self._vol_cache:
            vol_estimates = self._vol_cache[cache_key]
        else:
            vol_estimates = ewma_volatility(returns, halflife=self.vol_halflife)
            if use_cache:
                self._vol_cache[cache_key] = vol_estimates
        
        vol_weighted = vol_weighted_returns(
            returns, 
            vol_halflife=self.vol_halflife,
            min_vol=self.min_vol
        ) * self.scaling_factor
        
        return {
            'raw_returns': returns,
            'vol_estimates': vol_estimates,
            'vol_weighted': vol_weighted,
            'scaling_factor': self.scaling_factor
        }
    
    def clear_cache(self):
        """Clear volatility cache."""
        self._vol_cache.clear()


def calculate_vol_surface(
    returns_matrix: pd.DataFrame,
    tenors: np.ndarray,
    vol_halflife: int = 60
) -> pd.DataFrame:
    """
    Calculate volatility surface across tenors.
    
    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Returns matrix with tenor columns
    tenors : np.ndarray
        Array of tenor values (in years)
    vol_halflife : int, default 60
        EWMA halflife
        
    Returns
    -------
    pd.DataFrame
        Volatility surface (dates x tenors)
    """
    if len(tenors) != len(returns_matrix.columns):
        raise ValueError("Number of tenors must match number of return columns")
    
    vol_surface = ewma_volatility(returns_matrix, halflife=vol_halflife)
    vol_surface.columns = tenors
    
    return vol_surface.fillna(method='ffill')