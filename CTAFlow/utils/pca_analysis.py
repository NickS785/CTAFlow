"""
Rolling PCA analysis with shrinkage covariance for forward curve modeling.

This module implements the core PCA methodology from the PCA forward curve model,
including shrinkage covariance estimation, rolling PCA, and HJM factor mapping.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Tuple, List, Any, Callable
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
from sklearn.decomposition import PCA
from scipy.optimize import minimize, curve_fit
import warnings


def rolling_pca(
    deseas_logF: Dict[float, pd.Series],
    taus: np.ndarray,
    window: int = 252,
    max_k: int = 5,
    expl_var: float = 0.98,
    vol_halflife: int = 60,
    weight: Optional[np.ndarray] = None,
    shrinkage_method: str = 'ledoit_wolf'
) -> List[Dict[str, Any]]:
    """
    Rolling PCA analysis with shrinkage covariance estimation.
    
    This is the core function from the PCA model that performs rolling PCA
    on vol-weighted, deseasonalized log returns.
    
    Parameters
    ----------
    deseas_logF : dict
        Dictionary of deseasonalized log price series by tau
    taus : np.ndarray
        Array of tenor values
    window : int, default 252
        Rolling window size (trading days)
    max_k : int, default 5
        Maximum number of principal components
    expl_var : float, default 0.98
        Minimum explained variance threshold
    vol_halflife : int, default 60
        EWMA halflife for volatility weighting
    weight : np.ndarray, optional
        Weights for tenors (e.g., based on OI/volume)
    shrinkage_method : str, default 'ledoit_wolf'
        Covariance shrinkage method: 'ledoit_wolf', 'oas', 'empirical'
        
    Returns
    -------
    List[Dict]
        List of PCA results for each date containing:
        - date: Analysis date
        - taus: Available tenors
        - V: Principal component loadings (DataFrame)
        - eig: Eigenvalues (Series)
        - cumvar: Cumulative explained variance
        - explained_var: Individual explained variances
        - factor_scores: Factor scores (if available)
    """
    # Build aligned panel of returns
    df = pd.concat({tau: deseas_logF[tau] for tau in taus}, axis=1).dropna()
    df.columns = pd.Index(taus, name='tau')
    
    if len(df) < window + 1:
        warnings.warn(f"Insufficient data: {len(df)} < {window + 1}")
        return []
    
    ret = df.diff().dropna()  # Log returns at constant tau
    
    # Vol-scale returns
    vols = ret.apply(lambda col: col.ewm(halflife=vol_halflife, adjust=False).std())
    Z = ret / vols.replace(0, np.nan)
    Z = Z.fillna(0)  # Replace any remaining NaNs with 0
    
    # Set up weights
    if weight is None:
        w = pd.Series(1.0, index=taus)
    else:
        w = pd.Series(weight, index=taus)
    
    # Rolling PCA analysis
    results = []
    dates = Z.index[window:]
    
    for i, dt in enumerate(dates, start=window):
        Zwin = Z.iloc[i-window:i].dropna(axis=1, how='any')
        taus_in = Zwin.columns.values
        
        if len(taus_in) < 3:
            continue
        
        X = Zwin.values
        
        # Apply tenor weights
        w_subset = w.reindex(taus_in).fillna(1.0)
        W = np.diag(np.sqrt(w_subset.values))
        X_weighted = X @ W
        
        try:
            # Shrinkage covariance estimation
            if shrinkage_method == 'ledoit_wolf':
                lw = LedoitWolf().fit(X_weighted)
                Sigma = lw.covariance_
            elif shrinkage_method == 'oas':
                oas = OAS().fit(X_weighted)
                Sigma = oas.covariance_
            else:  # empirical
                emp = EmpiricalCovariance().fit(X_weighted)
                Sigma = emp.covariance_
            
            # Eigendecomposition
            evals, evecs = np.linalg.eigh(Sigma)
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
            
            # Ensure positive eigenvalues
            evals = np.maximum(evals, 1e-10)
            
            # Determine number of components
            total_var = np.sum(evals)
            cum_var = np.cumsum(evals) / total_var
            K = min(max_k, 1 + np.searchsorted(cum_var, expl_var))
            
            # Create results
            V = pd.DataFrame(
                evecs[:, :K], 
                index=taus_in, 
                columns=[f'PC{k+1}' for k in range(K)]
            )
            Lam = pd.Series(evals[:K], index=V.columns, name='eigenvalue')
            expl_vars = evals[:K] / total_var
            
            # Calculate factor scores for this window
            factor_scores = pd.DataFrame(
                X_weighted @ evecs[:, :K],
                index=Zwin.index,
                columns=V.columns
            )
            
            result = {
                'date': dt,
                'taus': taus_in,
                'V': V,
                'eig': Lam,
                'cumvar': cum_var[K-1],
                'explained_var': expl_vars,
                'factor_scores': factor_scores,
                'shrinkage_method': shrinkage_method,
                'n_components': K
            }
            
            results.append(result)
            
        except np.linalg.LinAlgError as e:
            warnings.warn(f"PCA failed at {dt}: {e}")
            continue
        except Exception as e:
            warnings.warn(f"Unexpected error at {dt}: {e}")
            continue
    
    return results


def fit_hjm_loadings(
    pca_loadings: pd.DataFrame,
    taus: np.ndarray,
    factor_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fit PCA loadings to HJM-style parametric forms.
    
    Maps empirical PCA loadings to HJM factorization:
    - Level: q1(τ) ≈ 1
    - Slope: q2(τ) ≈ 1 - exp(-β2*τ)  
    - Curvature: q3(τ) ≈ τ * exp(-β3*τ)
    
    Parameters
    ----------
    pca_loadings : pd.DataFrame
        PCA loadings matrix (taus x factors)
    taus : np.ndarray
        Tenor values in years
    factor_names : List[str], optional
        Names for factors (default: Level, Slope, Curvature, ...)
        
    Returns
    -------
    dict
        HJM fitting results with parameters and fitted loadings
    """
    n_factors = pca_loadings.shape[1]
    
    if factor_names is None:
        factor_names = ['Level', 'Slope', 'Curvature', 'Factor4', 'Factor5'][:n_factors]
    
    # Define HJM loading functions
    def level_func(tau, a):
        return a * np.ones_like(tau)
    
    def slope_func(tau, a, beta):
        return a * (1 - np.exp(-beta * tau))
    
    def curvature_func(tau, a, beta):
        return a * tau * np.exp(-beta * tau)
    
    def higher_order_func(tau, a, beta, gamma):
        return a * (tau**gamma) * np.exp(-beta * tau)
    
    fitting_results = {}
    fitted_loadings = pd.DataFrame(index=pca_loadings.index, columns=pca_loadings.columns)
    
    for i, factor in enumerate(pca_loadings.columns):
        loadings = pca_loadings[factor].values
        factor_name = factor_names[i] if i < len(factor_names) else f'Factor{i+1}'
        
        try:
            if i == 0:  # Level factor
                popt, pcov = curve_fit(level_func, taus, loadings, p0=[1.0])
                fitted = level_func(taus, *popt)
                params = {'a': popt[0]}
                
            elif i == 1:  # Slope factor
                popt, pcov = curve_fit(slope_func, taus, loadings, 
                                     p0=[1.0, 0.5], bounds=([-np.inf, 0.01], [np.inf, 10]))
                fitted = slope_func(taus, *popt)
                params = {'a': popt[0], 'beta': popt[1]}
                
            elif i == 2:  # Curvature factor
                popt, pcov = curve_fit(curvature_func, taus, loadings,
                                     p0=[1.0, 1.0], bounds=([-np.inf, 0.01], [np.inf, 10]))
                fitted = curvature_func(taus, *popt)
                params = {'a': popt[0], 'beta': popt[1]}
                
            else:  # Higher order factors
                popt, pcov = curve_fit(higher_order_func, taus, loadings,
                                     p0=[1.0, 1.0, 1.0], 
                                     bounds=([-np.inf, 0.01, 0.1], [np.inf, 10, 5]))
                fitted = higher_order_func(taus, *popt)
                params = {'a': popt[0], 'beta': popt[1], 'gamma': popt[2]}
            
            # Calculate fit quality
            r_squared = 1 - np.sum((loadings - fitted)**2) / np.sum((loadings - np.mean(loadings))**2)
            rmse = np.sqrt(np.mean((loadings - fitted)**2))
            
            fitting_results[factor_name] = {
                'params': params,
                'r_squared': r_squared,
                'rmse': rmse,
                'param_std_errors': np.sqrt(np.diag(pcov)),
                'original_loadings': loadings,
                'fitted_loadings': fitted
            }
            
            fitted_loadings[factor] = fitted
            
        except Exception as e:
            warnings.warn(f"HJM fitting failed for factor {factor}: {e}")
            fitted_loadings[factor] = loadings  # Use original loadings
            fitting_results[factor_name] = {
                'params': {},
                'r_squared': 0,
                'rmse': np.inf,
                'error': str(e)
            }
    
    return {
        'factor_names': factor_names,
        'fitting_results': fitting_results,
        'fitted_loadings': fitted_loadings,
        'original_loadings': pca_loadings
    }


def cross_commodity_pca(
    factor_scores_by_commodity: Dict[str, pd.DataFrame],
    window: int = 252,
    max_global_factors: int = 3
) -> Dict[str, Any]:
    """
    Cross-commodity PCA analysis on individual commodity factor scores.
    
    Step 1: Extract factor scores from individual commodity PCAs
    Step 2: Run PCA on cross-section of factor scores to get global factors
    
    Parameters
    ----------
    factor_scores_by_commodity : dict
        Dictionary mapping commodity -> factor scores DataFrame
    window : int, default 252
        Rolling window for cross-commodity analysis
    max_global_factors : int, default 3
        Maximum number of global factors
        
    Returns
    -------
    dict
        Cross-commodity PCA results
    """
    # Align factor scores across commodities
    all_scores = []
    commodity_names = []
    
    for commodity, scores in factor_scores_by_commodity.items():
        if scores.empty:
            continue
        # Flatten factor scores (date x factor combinations)
        for factor in scores.columns:
            series_name = f"{commodity}_{factor}"
            all_scores.append(scores[factor])
            commodity_names.append(series_name)
    
    if len(all_scores) < 2:
        return {'error': 'Insufficient data for cross-commodity analysis'}
    
    # Create combined DataFrame
    combined_scores = pd.concat(all_scores, axis=1, keys=commodity_names)
    combined_scores = combined_scores.dropna()
    
    if len(combined_scores) < window:
        return {'error': f'Insufficient data: {len(combined_scores)} < {window}'}
    
    # Rolling cross-commodity PCA
    results = []
    for i in range(window, len(combined_scores)):
        window_data = combined_scores.iloc[i-window:i]
        
        try:
            # Standardize data
            standardized = (window_data - window_data.mean()) / window_data.std()
            standardized = standardized.fillna(0)
            
            # PCA
            pca = PCA(n_components=max_global_factors)
            pca.fit(standardized)
            
            # Global factor loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                index=combined_scores.columns,
                columns=[f'Global_PC{i+1}' for i in range(pca.n_components_)]
            )
            
            # Transform current data
            current_scores = pca.transform(standardized.iloc[-1:].values)
            
            result = {
                'date': combined_scores.index[i],
                'global_loadings': loadings,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'global_factor_scores': pd.Series(current_scores[0], 
                                                 index=loadings.columns),
                'n_commodities': len(factor_scores_by_commodity),
                'n_individual_factors': len(combined_scores.columns)
            }
            
            results.append(result)
            
        except Exception as e:
            warnings.warn(f"Cross-commodity PCA failed at date {combined_scores.index[i]}: {e}")
    
    return {
        'rolling_results': results,
        'commodity_list': list(factor_scores_by_commodity.keys()),
        'total_individual_factors': len(combined_scores.columns)
    }


class PCAAnalyzer:
    """
    Advanced PCA analyzer with caching and validation.
    """
    
    def __init__(
        self,
        window: int = 252,
        vol_halflife: int = 60,
        max_components: int = 5,
        explained_var_threshold: float = 0.98,
        shrinkage_method: str = 'ledoit_wolf'
    ):
        """
        Initialize PCA analyzer.
        
        Parameters
        ----------
        window : int, default 252
            Rolling window size
        vol_halflife : int, default 60
            Volatility halflife for weighting
        max_components : int, default 5
            Maximum principal components
        explained_var_threshold : float, default 0.98
            Explained variance threshold
        shrinkage_method : str, default 'ledoit_wolf'
            Covariance shrinkage method
        """
        self.window = window
        self.vol_halflife = vol_halflife
        self.max_components = max_components
        self.explained_var_threshold = explained_var_threshold
        self.shrinkage_method = shrinkage_method
        self._cache = {}
    
    def analyze_curve_factors(
        self,
        logF_by_tau: Dict[float, pd.Series],
        taus: np.ndarray,
        weights: Optional[np.ndarray] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Complete curve factor analysis pipeline.
        
        Parameters
        ----------
        logF_by_tau : dict
            Dictionary of log price series by tau
        taus : np.ndarray
            Tenor array
        weights : np.ndarray, optional
            Tenor weights
        use_cache : bool, default True
            Whether to use caching
            
        Returns
        -------
        List[Dict]
            Rolling PCA results
        """
        cache_key = f"{id(logF_by_tau)}_{self.window}_{self.shrinkage_method}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Run rolling PCA
        pca_results = rolling_pca(
            logF_by_tau,
            taus,
            window=self.window,
            max_k=self.max_components,
            expl_var=self.explained_var_threshold,
            vol_halflife=self.vol_halflife,
            weight=weights,
            shrinkage_method=self.shrinkage_method
        )
        
        # Add HJM fitting to each result
        for result in pca_results:
            if 'V' in result and not result['V'].empty:
                hjm_fit = fit_hjm_loadings(result['V'], result['taus'])
                result['hjm_fit'] = hjm_fit
        
        if use_cache:
            self._cache[cache_key] = pca_results
        
        return pca_results
    
    def validate_pca_stability(
        self,
        pca_results: List[Dict[str, Any]],
        min_correlation: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate stability of PCA factors over time.
        
        Parameters
        ----------
        pca_results : List[Dict]
            Rolling PCA results
        min_correlation : float, default 0.7
            Minimum correlation for factor stability
            
        Returns
        -------
        dict
            Stability validation metrics
        """
        if len(pca_results) < 2:
            return {'error': 'Insufficient results for stability analysis'}
        
        stability_metrics = {}
        
        # Check loading stability
        for i in range(len(pca_results) - 1):
            current = pca_results[i]['V']
            next_result = pca_results[i + 1]['V']
            
            # Find common tenors
            common_taus = current.index.intersection(next_result.index)
            if len(common_taus) < 3:
                continue
            
            # Calculate correlations between corresponding factors
            correlations = {}
            for j, factor in enumerate(current.columns):
                if factor in next_result.columns:
                    corr = current.loc[common_taus, factor].corr(
                        next_result.loc[common_taus, factor]
                    )
                    correlations[factor] = corr
            
            date_key = pca_results[i]['date']
            stability_metrics[date_key] = correlations
        
        # Aggregate stability metrics
        factor_stability = {}
        for factor in pca_results[0]['V'].columns:
            factor_corrs = []
            for metrics in stability_metrics.values():
                if factor in metrics and not np.isnan(metrics[factor]):
                    factor_corrs.append(metrics[factor])
            
            if factor_corrs:
                factor_stability[factor] = {
                    'mean_correlation': np.mean(factor_corrs),
                    'min_correlation': np.min(factor_corrs),
                    'stability_ratio': np.mean([c > min_correlation for c in factor_corrs])
                }
        
        return {
            'factor_stability': factor_stability,
            'rolling_correlations': stability_metrics,
            'overall_stability': np.mean([
                metrics['mean_correlation'] for metrics in factor_stability.values()
                if 'mean_correlation' in metrics
            ]) if factor_stability else 0
        }
    
    def clear_cache(self):
        """Clear PCA analysis cache."""
        self._cache.clear()


def reconstruct_curves(
    pca_result: Dict[str, Any],
    factor_values: Optional[np.ndarray] = None,
    use_hjm_fit: bool = True
) -> pd.Series:
    """
    Reconstruct futures curves from PCA factors.
    
    Parameters
    ----------
    pca_result : dict
        Single PCA result dictionary
    factor_values : np.ndarray, optional
        Factor values to use (default: use latest from factor_scores)
    use_hjm_fit : bool, default True
        Whether to use HJM fitted loadings
        
    Returns
    -------
    pd.Series
        Reconstructed curve (tau -> log_price)
    """
    if use_hjm_fit and 'hjm_fit' in pca_result:
        loadings = pca_result['hjm_fit']['fitted_loadings']
    else:
        loadings = pca_result['V']
    
    if factor_values is None:
        if 'factor_scores' in pca_result and not pca_result['factor_scores'].empty:
            factor_values = pca_result['factor_scores'].iloc[-1].values
        else:
            # Use zero factor values (mean curve)
            factor_values = np.zeros(loadings.shape[1])
    
    # Reconstruct: curve = loadings @ factor_values
    reconstructed = loadings.values @ factor_values
    
    return pd.Series(reconstructed, index=loadings.index)