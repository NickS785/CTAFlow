"""
Advanced Futures Curve Analysis Framework
Incorporates FuturesCurve, SpreadData, and Lévy Area/Path Signature features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import skew, kurtosis
import plotly.io as pio
from numpy.linalg import svd
from ...data.raw_formatting.spread_manager import FuturesCurve, SpreadData, SpreadFeature
from ...utils.seasonal import deseasonalize_monthly
pio.renderers.default = "browser"
# Import data client and utilities if available
try:
    from ..data.data_client import DataClient
    from ..data.raw_formatting.spread_manager import MONTH_CODE_MAP, _is_empty
except ImportError:
    # Fallback definitions for standalone use
    MONTH_CODE_MAP = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    DataClient = None
    
    def _is_empty(data):
        """Fallback function to check if data is empty"""
        if data is None:
            return True
        try:
            return len(data) == 0
        except:
            return True

# Additional imports for advanced analysis
from numba import jit, prange
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import tenor interpolation utilities
try:
    from ...utils.tenor_interpolation import TenorInterpolator, create_tenor_grid
except ImportError:
    TenorInterpolator = None
    create_tenor_grid = None
    warnings.warn("TenorInterpolator not available - constant maturity analysis disabled")

MONTH_CODE_ORDER = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']


def _as_2d_df(arr: np.ndarray, index: pd.DatetimeIndex, cols: Optional[list] = None) -> pd.DataFrame:
    """Utility: shape check + DF wrap."""
    df = pd.DataFrame(arr, index=index)
    if cols is not None and len(cols) == df.shape[1]:
        df.columns = cols
    return df

class CurveShapeAnalyzer:
    """
    Analyzes curve shapes and extracts features from SpreadData
    """

    def __init__(self, spread_data: SpreadData):
        """Initialize analyzer with data validation"""
        if not isinstance(spread_data, SpreadData):
            raise TypeError(f"Expected SpreadData, got {type(spread_data)}")
        
        self.sf = spread_data
        self._current_prices: Optional[Dict[str, float]] = None
        self._current_prices_array: Optional[np.ndarray] = None  # Store as array for consistency
        
        # Validate spread data has required attributes
        if not hasattr(spread_data, 'curve') or spread_data.curve is None:
            warnings.warn("SpreadData lacks curve data - shape analysis may be limited")
            
        self._process_latest_curve()

    def _process_latest_curve(self):
        """Process the latest curve data using pre-loaded sequential labels when available"""
        if not _is_empty(self.sf.curve):
            latest_idx = self.sf.curve.index[-1]
            self._current_prices = {}

            # First try to use pre-loaded sequential labels to avoid redundant calculation
            seq_labels = None
            if not _is_empty(self.sf.seq_labels) and latest_idx in self.sf.seq_labels.index:
                labels_row = self.sf.seq_labels.loc[latest_idx]
                if 'labels' in labels_row:
                    actual_labels = labels_row['labels']
                    if isinstance(actual_labels, list) and len(actual_labels) > 0:
                        seq_labels = actual_labels

            if seq_labels is not None:
                # Use pre-loaded sequence order
                for month_code in seq_labels:
                    if month_code in self.sf.curve.columns:
                        value = self.sf.curve.loc[latest_idx, month_code]
                        if pd.notna(value):
                            self._current_prices[month_code] = value
            else:
                # Fallback to standard calendar order
                for month_code in MONTH_CODE_ORDER:
                    if month_code in self.sf.curve.columns:
                        value = self.sf.curve.loc[latest_idx, month_code]
                        if pd.notna(value):
                            self._current_prices[month_code] = value
            
            # Convert to numpy array for consistent data access
            if self._current_prices:
                self._current_prices_array = np.array(list(self._current_prices.values()))
            else:
                self._current_prices_array = np.array([])

    def get_shape_features(self) -> Dict[str, float]:
        """Extract comprehensive shape features using consistent numpy arrays"""
        if self._current_prices_array is None or len(self._current_prices_array) == 0:
            return {}
        
        # Type validation
        if not isinstance(self._current_prices_array, np.ndarray):
            raise TypeError("Price data must be numpy array for shape analysis")

        # Use pre-stored numpy array for consistency
        prices = self._current_prices_array
        features = {}

        # Basic statistics
        features['mean_price'] = np.mean(prices)
        features['std_price'] = np.std(prices)
        features['min_price'] = np.min(prices)
        features['max_price'] = np.max(prices)
        features['price_range'] = features['max_price'] - features['min_price']

        # Shape metrics
        features['skewness'] = skew(prices)
        features['kurtosis'] = kurtosis(prices)

        # Slope features
        if len(prices) >= 2:
            features['overall_slope'] = (prices[-1] - prices[0]) / len(prices)

            front_contracts = min(3, len(prices))
            features['front_slope'] = (prices[front_contracts - 1] - prices[0]) / front_contracts

            back_contracts = min(3, len(prices))
            features['back_slope'] = (prices[-1] - prices[-back_contracts]) / back_contracts

            if len(prices) >= 3:
                first_diff = np.diff(prices)
                second_diff = np.diff(first_diff)
                features['mean_curvature'] = np.mean(second_diff)
                features['max_curvature'] = np.max(np.abs(second_diff))

        # Contango/Backwardation
        features['contango_ratio'] = self._calculate_contango_ratio()
        features['backwardation_depth'] = self._calculate_backwardation_depth()

        # Spread features
        if not _is_empty(self.sf.seq_spreads):
            latest_spreads = self.sf.seq_spreads.iloc[-1].dropna()

            if len(latest_spreads) > 0:
                features['mean_spread'] = np.mean(latest_spreads)
                features['std_spread'] = np.std(latest_spreads)
                features['max_spread'] = np.max(latest_spreads)
                features['min_spread'] = np.min(latest_spreads)

                # Roll yield (annualized)
                if hasattr(self.sf, 'expiry_trackers') and len(getattr(self.sf, 'expiry_trackers', {})) >= 2:
                    features['roll_yield'] = self._calculate_roll_yield()

        # Term structure complexity
        features['term_complexity'] = self._calculate_term_complexity()

        # Market structure features
        features['volume_concentration'] = self._calculate_volume_concentration()
        features['oi_concentration'] = self._calculate_oi_concentration()

        # Seasonality strength
        features['seasonality_strength'] = self._calculate_seasonality_strength()

        return features

    def _calculate_contango_ratio(self) -> float:
        """Ratio of spreads in contango"""
        if not _is_empty(self.sf.seq_spreads):
            latest_spreads = self.sf.seq_spreads.iloc[-1].dropna()
            if len(latest_spreads) > 0:
                contango_count = np.sum(latest_spreads > 0)
                return contango_count / len(latest_spreads)
        return 0.0

    def _calculate_backwardation_depth(self) -> float:
        """Average depth of backwardation spreads"""
        if not _is_empty(self.sf.seq_spreads):
            latest_spreads = self.sf.seq_spreads.iloc[-1].dropna()
            back_spreads = latest_spreads[latest_spreads < 0]
            if len(back_spreads) > 0:
                return np.mean(back_spreads)
        return 0.0

    def _calculate_roll_yield(self) -> float:
        """Calculate annualized roll yield using numpy arrays"""
        if self._current_prices_array is None or len(self._current_prices_array) < 2:
            return 0.0

        # Get front two contracts from array
        front_price = self._current_prices_array[0]
        next_price = self._current_prices_array[1]
        
        # Get month codes for expiry calculation (still need keys for this)
        month_codes = list(self._current_prices.keys())[:2] if self._current_prices else []

        # Get days between contracts (simplified calculation)
        if hasattr(self.sf, 'expiry_trackers') and month_codes[0] in getattr(self.sf, 'expiry_trackers', {}) and month_codes[1] in getattr(self.sf, 'expiry_trackers', {}):
            front_expiry = self.sf.expiry_trackers[month_codes[0]].expiry_date
            next_expiry = self.sf.expiry_trackers[month_codes[1]].expiry_date
            days_diff = (next_expiry - front_expiry).days

            if days_diff > 0:
                return ((next_price / front_price) - 1) * (365 / days_diff)
        else:
            # Fallback: assume standard monthly contracts (~30 days apart)
            days_diff = 30
            if days_diff > 0:
                return ((next_price / front_price) - 1) * (365 / days_diff)

        return 0.0

    def _calculate_term_complexity(self) -> float:
        """Calculate term structure complexity using entropy"""
        if self._current_prices_array is None or len(self._current_prices_array) < 3:
            return 0.0

        prices = self._current_prices_array
        changes = np.diff(prices)
        abs_changes = np.abs(changes)

        if np.sum(abs_changes) == 0:
            return 0.0

        probs = abs_changes / np.sum(abs_changes)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def _calculate_volume_concentration(self) -> float:
        """Herfindahl index for volume"""
        if _is_empty(self.sf.volume_curve):
            return 0.0

        latest_volumes = self.sf.volume_curve.iloc[-1].dropna()
        if len(latest_volumes) == 0 or latest_volumes.sum() == 0:
            return 0.0

        shares = latest_volumes / latest_volumes.sum()
        return np.sum(shares ** 2)

    def _calculate_oi_concentration(self) -> float:
        """Herfindahl index for OI"""
        if _is_empty(self.sf.oi_curve):
            return 0.0

        latest_oi = self.sf.oi_curve.iloc[-1].dropna()
        if len(latest_oi) == 0 or latest_oi.sum() == 0:
            return 0.0

        shares = latest_oi / latest_oi.sum()
        return np.sum(shares ** 2)

    def _calculate_seasonality_strength(self) -> float:
        """Measure seasonality in price patterns"""
        if len(self._current_prices) < 4:
            return 0.0

        prices = np.array(list(self._current_prices.values()))
        changes = np.diff(prices)

        if len(changes) >= 3:
            # Autocorrelation at different lags
            autocorr_1 = np.corrcoef(changes[:-1], changes[1:])[0, 1]
            autocorr_2 = np.corrcoef(changes[:-2], changes[2:])[0, 1] if len(changes) >= 4 else 0

            return abs(autocorr_1) + abs(autocorr_2)

        return 0.0

    # ==================================================================================
    # EVOLUTION TRACKING METHODS (Migrated from CurveEvolution)
    # ==================================================================================

    def calculate_curve_changes(self, lookback: int = 1) -> Dict[str, float]:
        """Calculate curve shape changes over time (migrated from CurveEvolution)"""
        # This would track changes in the current curve vs historical curves
        # For now, return basic shape metrics that can be compared over time
        current_features = self.get_shape_features()

        # Could be extended to store historical features and calculate changes
        changes = {}
        for key, value in current_features.items():
            changes[f'{key}_current'] = value

        return changes

    def analyze_structural_changes(self) -> Dict[str, float]:
        """Analyze structural changes in curve shape (migrated from CurveEvolution)"""
        if self._current_prices_array is None or len(self._current_prices_array) < 3:
            return {}

        prices = self._current_prices_array
        changes = {}

        # Calculate structural curve metrics
        if len(prices) >= 2:
            # Parallel shift proxy (average price level)
            changes['price_level'] = np.mean(prices)

            # Twist proxy (front vs back)
            if len(prices) >= 4:
                front_avg = np.mean(prices[:2])
                back_avg = np.mean(prices[-2:])
                changes['twist'] = back_avg - front_avg

            # Butterfly proxy (middle vs edges)
            if len(prices) >= 3:
                mid_idx = len(prices) // 2
                mid_price = prices[mid_idx]
                edge_avg = np.mean([prices[0], prices[-1]])
                changes['butterfly'] = mid_price - edge_avg

        return changes

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution summary combining shape and structural analysis"""
        shape_features = self.get_shape_features()
        structural_changes = self.analyze_structural_changes()

        return {
            'shape_features': shape_features,
            'structural_metrics': structural_changes,
            'symbol': getattr(self.sf, 'symbol', None),
            'analysis_date': datetime.now()
        }


# Additional advanced features based on mathematical insights
def calculate_seasonal_decomposition(spread_data: SpreadData,
                                     period: int = 252) -> Dict[str, pd.DataFrame]:
    """
    Decompose curve into trend, seasonal, and residual components
    Using STL or X-13ARIMA-SEATS methodology
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    results = {}
    seq_curve = spread_data.get_sequentialized_curve()

    for col in seq_curve.columns:
        if seq_curve[col].notna().sum() > period * 2:
            decomposition = seasonal_decompose(
                seq_curve[col].dropna(),
                model='additive',
                period=period
            )

            results[f'{col}_trend'] = decomposition.trend
            results[f'{col}_seasonal'] = decomposition.seasonal
            results[f'{col}_residual'] = decomposition.resid

    return results


def calculate_information_flow(spread_data: SpreadData,
                               method: str = 'transfer_entropy') -> pd.DataFrame:
    """
    Calculate information flow between contracts using:
    - Transfer entropy
    - Granger causality
    - Directed information

    This quantifies which contracts drive information to others
    """
    results = []
    curve = spread_data.curve

    for col1 in curve.columns:
        for col2 in curve.columns:
            if col1 != col2:
                series1 = curve[col1].dropna()
                series2 = curve[col2].dropna()

                # Align series
                aligned = pd.DataFrame({'s1': series1, 's2': series2}).dropna()

                if len(aligned) < 50:
                    continue

                if method == 'granger':
                    from statsmodels.tsa.stattools import grangercausalitytests

                    try:
                        # Test if col1 Granger-causes col2
                        result = grangercausalitytests(
                            aligned[['s2', 's1']],
                            maxlag=5,
                            verbose=False
                        )

                        # Get minimum p-value across lags
                        p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, 6)]
                        min_p = min(p_values)

                        results.append({
                            'from': col1,
                            'to': col2,
                            'p_value': min_p,
                            'significant': min_p < 0.05
                        })
                    except:
                        pass

    return pd.DataFrame(results)


def calculate_microstructure_features(spread_data: SpreadData) -> Dict[str, float]:
    """
    Calculate market microstructure features:
    - Roll measure (effective spread proxy)
    - Amihud illiquidity
    - Kyle's lambda
    - Volume-synchronized probability of informed trading (VPIN)
    """
    features = {}

    if hasattr(spread_data, 'curve') and hasattr(spread_data, 'volume'):
        curve = spread_data.curve
        volume = spread_data.volume

        # Roll measure (Hasbrouck 2009)
        for col in curve.columns:
            if col in volume.columns:
                price_changes = curve[col].diff()

                # Serial covariance of price changes
                cov = price_changes.cov(price_changes.shift(1))

                if cov < 0:
                    # Effective spread estimate
                    features[f'roll_spread_{col}'] = 2 * np.sqrt(-cov)

        # Amihud illiquidity
        for col in curve.columns:
            if col in volume.columns:
                returns = curve[col].pct_change()

                # |return| / volume
                illiq = (returns.abs() / (volume[col] + 1)).mean()
                features[f'amihud_illiq_{col}'] = illiq

        # Kyle's lambda (price impact)
        for col in curve.columns:
            if col in volume.columns:
                price_changes = curve[col].diff()
                volume_signed = volume[col] * np.sign(price_changes)

                # Regression of price change on signed volume
                valid_data = pd.DataFrame({
                    'price_change': price_changes,
                    'signed_volume': volume_signed
                }).dropna()

                if len(valid_data) > 20:
                    slope, _, r_value, _, _ = stats.linregress(
                        valid_data['signed_volume'],
                        valid_data['price_change']
                    )
                    features[f'kyle_lambda_{col}'] = slope
                    features[f'kyle_r2_{col}'] = r_value ** 2

    return features


# ====================================================================================
# JIT-COMPILED UTILITY FUNCTIONS
# ====================================================================================

@jit(nopython=True)
def _calculate_log_levy_areas_numba(log_prices: np.ndarray, window: int) -> np.ndarray:
    """
    JIT-optimized Lévy area calculation on log prices

    Computes Lévy areas between consecutive contract months using log prices
    to detect fundamental drivers of curve evolution.

    Proper formula: A = 0.5 * sum(X_mid * dY - Y_mid * dX)
    where X_mid = (X[i] + X[i-1])/2 and dY = Y[i] - Y[i-1]

    Interpretation: A > 0 => front contract leads (drives) the back contract
                    A < 0 => back contract leads (drives) the front contract
    """

    n_dates, n_contracts = log_prices.shape
    levy_areas = np.full((n_dates, n_contracts - 1), np.nan)

    for contract_pair in prange(n_contracts - 1):
        front_log = log_prices[:, contract_pair]
        back_log = log_prices[:, contract_pair + 1]

        for i in range(window, n_dates):
            # Extract windows
            front_window = front_log[i-window:i]
            back_window = back_log[i-window:i]

            # Check for valid data
            has_nan_front = False
            has_nan_back = False

            for j in range(len(front_window)):
                if np.isnan(front_window[j]):
                    has_nan_front = True
                    break

            for j in range(len(back_window)):
                if np.isnan(back_window[j]):
                    has_nan_back = True
                    break

            if has_nan_front or has_nan_back:
                continue

            # Calculate Lévy area using proper midpoint formula
            # A = 0.5 * sum(X_mid * dY - Y_mid * dX)
            area = 0.0

            for k in range(1, len(front_window)):
                # Current and previous values
                front_curr = front_window[k]
                front_prev = front_window[k - 1]
                back_curr = back_window[k]
                back_prev = back_window[k - 1]

                # Increments
                dX = front_curr - front_prev
                dY = back_curr - back_prev

                # Midpoints
                X_mid = (front_curr + front_prev) * 0.5
                Y_mid = (back_curr + back_prev) * 0.5

                # Lévy area increment
                area += X_mid * dY - Y_mid * dX

            levy_areas[i, contract_pair] = 0.5 * area

    return levy_areas


# ====================================================================================
# UNIFIED CURVE EVOLUTION ANALYZER
# Merges CurveEvolution and SpreadAnalyzer with advanced path signature analysis
# ====================================================================================

class CurveEvolutionAnalyzer:
    """
    Unified analyzer combining CurveEvolution and SpreadAnalyzer capabilities
    
    Provides comprehensive analysis of futures curve evolution using:
    - Path signatures and Lévy areas on log prices
    - Regime change detection via curve driver analysis
    - Seasonal patterns and market structure evolution
    - Performance-optimized calculations with caching
    
    Key Features:
    - Log price Lévy areas for detecting fundamental curve drivers
    - Path signature analysis for regime identification
    - Broadcast operations for efficient time series analysis
    - Integration with SpreadData.get_seq_curves() pipeline
    """
    
    def __init__(self,
                 curves_data: Optional[Union[pd.Series, SpreadData, List[FuturesCurve]]] = None,
                 symbol: Optional[str] = None,
                 back_month_years: float = 1.0):
        """
        Initialize CurveEvolutionAnalyzer

        Parameters:
        -----------
        curves_data : pd.Series, SpreadData, or List[FuturesCurve]
            Input curve data. Can be:
            - pd.Series from SpreadData.get_seq_curves()
            - SpreadData object (will extract curves automatically)
            - List of FuturesCurve objects
        symbol : str, optional
            Symbol identifier for the curves
        back_month_years : float, default 1.0
            Number of years for back month contracts (1.0 = 1 year, 0.5 = 6 months, etc.)
            Controls which contracts are considered "back-end" in driver analysis
        """

        self.symbol = symbol
        self.back_month_years = back_month_years
        self.spread_data = None  # Reference to original SpreadData for spot data access
        self.curves = self._process_input_data(curves_data)

        # Core analysis components
        self.path_signatures = None
        self.regime_analysis = None
        self.seasonal_patterns = None

        # Performance caches
        self._cache = {}
        self._log_price_cache = {}
        self._levy_cache = {}

        # Analysis parameters
        self.default_window = 63
        self.regime_threshold = 2.0

        # Constant maturity support
        self.constant_maturity_data = None
        self.expiry_data = None
        self.tenor_interpolator = None
        self.min_regime_length = 5
        
        # Validate initialization
        if self.curves is not None and len(self.curves) > 0:
            self._validate_curve_data()
    
    def _process_input_data(self, data) -> Optional[pd.Series]:
        """Process different input data types into standardized pd.Series format"""

        if data is None:
            return None

        elif isinstance(data, pd.Series):
            # pd.Series already in correct format (from SpreadData.get_seq_curves() or slicing)
            return data

        elif hasattr(data, 'get_seq_curves'):
            # SpreadData object - extract curves with proper datetime index
            self.symbol = getattr(data, 'symbol', self.symbol)
            self.spread_data = data  # Store reference for spot data access
            return data.get_seq_curves()  # Already returns pd.Series with DatetimeIndex

        elif isinstance(data, list):
            # List of FuturesCurve objects - use existing timestamps/ref_dates
            if not data:
                return None

            # Validate all are FuturesCurve objects
            if not all(hasattr(curve, '__class__') and 'FuturesCurve' in str(curve.__class__) for curve in data):
                raise TypeError("All list elements must be FuturesCurve objects")

            # Extract existing timestamps from curves
            dates = []
            for curve in data:
                if hasattr(curve, 'timestamp') and curve.timestamp is not None:
                    dates.append(curve.timestamp)
                elif hasattr(curve, 'ref_date') and curve.ref_date is not None:
                    dates.append(curve.ref_date)
                else:
                    raise ValueError("FuturesCurve objects must have timestamp or ref_date")

            return pd.Series(data, index=pd.DatetimeIndex(dates), name='curves')

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _validate_curve_data(self):
        """Validate curve data consistency"""
        if self.curves is None or len(self.curves) == 0:
            raise ValueError("No curve data available")

        # Check for consistent curve structure
        non_null_curves = self.curves.dropna()
        if len(non_null_curves) == 0:
            warnings.warn("All curves are None/NaN - analysis may be limited")
            return

        sample_curve = next(iter(non_null_curves))
        if not isinstance(sample_curve, FuturesCurve):
            raise TypeError("All curves must be FuturesCurve objects")

        # Validate minimum data requirements
        if len(non_null_curves) < self.default_window:
            warnings.warn(f"Limited data: {len(non_null_curves)} valid curves < {self.default_window} window")
    
    def _extract_front_month_broadcast(self) -> pd.Series:
        """
        Broadcast extraction of front month (M0) prices from all curves
        
        Returns:
        --------
        pd.Series with datetime index and front month prices
        """
        if self.curves is None:
            return pd.Series(dtype=float)
        
        # Vectorized approach using pandas apply
        def extract_m0_price(curve):
            if curve is None:
                return np.nan
            try:
                # Use new __getitem__ method for M0 (front month)
                return curve['M0']
            except (KeyError, IndexError, AttributeError):
                # Fallback to direct access
                try:
                    if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                        return curve.seq_prices[0] if len(curve.seq_prices) > 0 else np.nan
                    elif hasattr(curve, 'prices') and curve.prices is not None:
                        return curve.prices[0] if len(curve.prices) > 0 else np.nan
                except (IndexError, AttributeError):
                    return np.nan
            return np.nan
        
        # Apply vectorized extraction
        front_month_series = self.curves.apply(extract_m0_price)
        
        # Filter out NaN values
        return front_month_series.dropna()
    
    @classmethod
    def from_spread_data(cls,
                        spread_data: SpreadData,
                        date_range: Optional[Union[slice, pd.DatetimeIndex]] = None,
                        step: int = 1,
                        back_month_years: float = 1.0) -> 'CurveEvolutionAnalyzer':
        """
        Factory method to create analyzer from SpreadData

        Parameters:
        -----------
        spread_data : SpreadData
            Source data container
        date_range : slice or DatetimeIndex, optional
            Date range for analysis
        step : int
            Sampling step for dates
        back_month_years : float, default 1.0
            Number of years for back month contracts (1.0 = 1 year, 0.5 = 6 months, etc.)

        Returns:
        --------
        CurveEvolutionAnalyzer
        """

        curves = spread_data.get_seq_curves(date_range=date_range, step=step)
        analyzer = cls(curves, symbol=spread_data.symbol, back_month_years=back_month_years)
        analyzer.spread_data = spread_data  # Store reference for spot data access
        return analyzer

    def setup_constant_maturity(self,
                               tenor_grid: Optional[np.ndarray] = None,
                               method: str = 'log_linear',
                               cache_results: bool = True) -> bool:
        """
        Setup constant maturity interpolation for the curve analysis.

        Parameters:
        -----------
        tenor_grid : np.ndarray, optional
            Target tenors in years. If None, creates monthly grid (1m to 36m)
        method : str, default 'log_linear'
            Interpolation method ('log_linear', 'cubic_hermite')
        cache_results : bool, default True
            Cache interpolation results

        Returns:
        --------
        bool
            True if setup successful, False if TenorInterpolator unavailable
        """
        if TenorInterpolator is None or create_tenor_grid is None:
            warnings.warn("TenorInterpolator not available - constant maturity disabled")
            return False

        if self.spread_data is None and self.symbol is None:
            warnings.warn("No SpreadData or symbol available - cannot setup constant maturity")
            return False

        try:
            # Import necessary functions
            from ..data.raw_formatting.dly_contract_manager import calculate_contract_expiry
            import pandas as pd
            import numpy as np
            import re

            # Determine symbol and get price data
            symbol = self.symbol or getattr(self.spread_data, 'symbol', None)
            if not symbol:
                warnings.warn("No symbol available for expiry calculation")
                return False

            # Get price data from SpreadData if available, otherwise use DataClient
            if self.spread_data is not None and hasattr(self.spread_data, 'curve') and self.spread_data.curve is not None:
                curve_data = self.spread_data.curve
            else:
                # Fall back to DataClient
                from ..data.data_client import DataClient
                client = DataClient()
                curve_data = client.query_curve_data(symbol, curve_types=['curve'])

            if curve_data.empty:
                warnings.warn(f"No curve data available for {symbol}")
                return False

            # Extract contract IDs from curve column names and calculate expiry dates
            contract_ids = curve_data.columns.tolist()
            expiry_dict = {}

            for contract_id in contract_ids:
                try:
                    # Parse contract ID (e.g., "H25" format)
                    contract_str = str(contract_id).strip()
                    match = re.match(r"([FGHJKMNQUVXZ])(\d{2})", contract_str, re.IGNORECASE)

                    if match:
                        month_code, yy = match.groups()
                        yy = int(yy)
                        # Convert 2-digit year to 4-digit year
                        year = 2000 + yy if yy <= 50 else 1900 + yy

                        # Calculate expiry date
                        expiry_date = calculate_contract_expiry(month_code.upper(), year, symbol.upper())
                        expiry_dict[contract_id] = expiry_date

                except Exception as e:
                    warnings.warn(f"Could not calculate expiry for {contract_id}: {e}")

            if not expiry_dict:
                warnings.warn(f"No valid expiry dates calculated for {symbol}")
                return False

            # Create expiry data Series
            import pandas as pd
            self.expiry_data = pd.Series(expiry_dict, name='expiry_date')
            self.expiry_data.index.name = 'contract_id'
            self.expiry_data = pd.to_datetime(self.expiry_data)

            # Create tenor grid if not provided
            if tenor_grid is None:
                tenor_grid = create_tenor_grid(
                    min_tau=1/12,  # 1 month
                    max_tau=3.0,   # 3 years
                    tenor_type='monthly'
                )

            # Initialize interpolator
            self.tenor_interpolator = TenorInterpolator(
                tenor_grid=tenor_grid,
                method=method,
                cache_results=cache_results
            )

            # Perform initial interpolation
            self.constant_maturity_data = self.tenor_interpolator.interpolate(
                curve_data,
                self.expiry_data
            )

            return True

        except Exception as e:
            warnings.warn(f"Failed to setup constant maturity: {e}")
            import traceback
            print(f"DEBUG: Constant maturity setup failed with exception: {e}")
            print("DEBUG: Traceback:")
            traceback.print_exc()
            return False
    
    def get_log_prices_matrix(self,
                              cache: bool = True,
                              deseasonalize: bool = True) -> np.ndarray:
        """Extract log prices matrix for all curves and contracts.

        Parameters
        ----------
        cache : bool, default True
            Use cached result if available.
        deseasonalize : bool, default True
            Remove simple monthly seasonal pattern from the log prices before
            returning the matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_dates, n_contracts)`` with log prices.
        """

        cache_key = 'log_prices_matrix_deseasonalized' if deseasonalize else 'log_prices_matrix_raw'
        if cache and cache_key in self._log_price_cache:
            return self._log_price_cache[cache_key]
        
        if self.curves is None:
            raise ValueError("No curve data available")
        
        # Get dimensions from first valid curve
        sample_curve = next(iter(self.curves.dropna()))
        if hasattr(sample_curve, 'seq_prices') and sample_curve.seq_prices is not None:
            n_contracts = len(sample_curve.seq_prices)
        elif hasattr(sample_curve, 'prices') and sample_curve.prices is not None:
            n_contracts = len(sample_curve.prices)
        else:
            raise ValueError("No price data found in sample curve")
            
        n_dates = len(self.curves)
        
        # Initialize log price matrix
        log_prices = np.full((n_dates, n_contracts), np.nan)
        
        # Fill matrix with log prices
        for i, (date, curve) in enumerate(self.curves.items()):
            if curve is not None:
                # Prefer seq_prices over regular prices
                if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                    prices = np.array(curve.seq_prices)
                elif hasattr(curve, 'prices') and curve.prices is not None:
                    prices = np.array(curve.prices)
                else:
                    continue
                    
                # Handle zero/negative prices and dimension alignment
                valid_prices = prices > 0
                if np.any(valid_prices):
                    # Ensure we don't exceed matrix dimensions
                    max_len = min(len(prices), log_prices.shape[1])
                    prices_subset = prices[:max_len]
                    valid_subset = valid_prices[:max_len]
                    
                    if np.any(valid_subset):
                        log_prices[i, :max_len][valid_subset] = np.log(prices_subset[valid_subset])
        
        if deseasonalize:
            log_prices = deseasonalize_monthly(log_prices, self.curves.index)

        if cache:
            self._log_price_cache[cache_key] = log_prices

        return log_prices

    def _deseasonalize_log_prices(self, log_prices: np.ndarray) -> np.ndarray:
        """Remove monthly seasonal pattern from log price matrix.

        A simple approach that subtracts the average log price for each
        calendar month from the corresponding observations.

        Parameters
        ----------
        log_prices : np.ndarray
            Matrix of log prices with shape ``(n_dates, n_contracts)``.

        Returns
        -------
        np.ndarray
            Deseasonalized log price matrix.
        """

        if self.curves is None or len(self.curves) == 0:
            return log_prices

        dates = self.curves.index
        df = pd.DataFrame(log_prices, index=dates)
        monthly_means = df.groupby(df.index.month).transform('mean')
        deseasonalized = df - monthly_means

        return deseasonalized.to_numpy()
    
    def _calculate_log_levy_areas_jit(self, 
                                     log_prices: np.ndarray,
                                     window: int) -> np.ndarray:
        """
        Wrapper for JIT-optimized Lévy area calculation on log prices
        
        Computes Lévy areas between consecutive contract months using log prices
        to detect fundamental drivers of curve evolution
        """
        return _calculate_log_levy_areas_numba(log_prices, window)






    def pca_deseasonalized(
            self,
            constant_maturity: bool = False,
            n_components: int = 3,
            standardize: bool = True,
            drop_na: str = "row"  # {"row","col","none"}
    ) -> Dict[str, Any]:
        """
        Run PCA on deseasonalized log-returns across tenors.
        - Uses monthly deseasonalization in log space (leverages deseasonalize_monthly).
        - Returns components, loadings, explained variance, and diagnostics.
        """

        # (A) Build log price matrix (sequential by default; constant maturity if provided)
        if constant_maturity and getattr(self, "constant_maturity_data", None) is not None:
            log_prices = np.log(self.constant_maturity_data.values.astype(float))
            tenor_labels = list(self.constant_maturity_data.columns)
            dates = self.constant_maturity_data.index
        else:
            # Use the helper you already have: log_prices_matrix(deseasonalize=True)
            log_prices = self.get_log_prices_matrix(deseasonalize=True, cache=True)  # uses deseasonalize_monthly
            # labels: attempt from the most recent curve
            last_curve = self.spread_data.seq_prices.dropna().iloc[-1]
            tenor_labels = (
                list(self.spread_data.seq_labels.iloc[-1])
                if hasattr(last_curve, "seq_labels") and isinstance(last_curve.seq_labels,
                                                                    dict) and "labels" in last_curve.seq_labels
                else [f"M{i + 1}" for i in range(log_prices.shape[1])]
            )
            dates = self.spread_data.index

        # (B) Time-difference to returns (Δlog)
        ret = np.diff(log_prices, axis=0)
        ret_dates = dates[1:]

        # (C) NA handling
        R = pd.DataFrame(ret, index=ret_dates, columns=tenor_labels)
        if drop_na == "row":
            R = R.dropna(axis=0, how="any")
        elif drop_na == "col":
            R = R.dropna(axis=1, how="any")
        else:
            R = R.fillna(0.0)
        if R.empty:
            raise ValueError("No valid returns after NA handling")

        # (D) Column standardization (optional)
        X = R.values.copy()
        mu = X.mean(axis=0, keepdims=True)
        X -= mu
        if standardize:
            std = X.std(axis=0, ddof=1, keepdims=True)
            std[std == 0] = 1.0
            X /= std

        # (E) PCA via SVD on T x N (time x tenor)
        U, S, Vt = svd(X, full_matrices=False)
        # Variance explained
        sing2 = S ** 2
        evr = sing2 / sing2.sum()
        k = min(n_components, Vt.shape[0])

        loadings = Vt[:k, :]  # k x N  (tenor loadings)
        scores = U[:, :k] * S[:k]  # T x k  (time series of factors)

        out = {
            "dates": R.index,
            "tenors": list(R.columns),
            "loadings": pd.DataFrame(loadings, index=[f"PC{i + 1}" for i in range(k)], columns=R.columns),
            "scores": pd.DataFrame(scores, index=R.index, columns=[f"PC{i + 1}" for i in range(k)]),
            "explained_variance_ratio": pd.Series(evr[:k], index=[f"PC{i + 1}" for i in range(k)]),
            "mean_by_tenor": pd.Series(mu.ravel(), index=R.columns),
            "std_by_tenor": pd.Series((std if standardize else np.ones_like(mu)).ravel(), index=R.columns),
            "config": {
                "constant_maturity": constant_maturity,
                "standardize": standardize,
                "deseasonalized": True,
                "drop_na": drop_na
            }
        }
        return out

    # ---- 2) Cash & Carry / Convenience Yield sheet ----
    def carry_sheet(
            self,
            spot: Optional[pd.Series] = None,  # optional true spot; if None, uses front future as proxy
            rate: Optional[pd.Series] = None,  # risk-free annualized (e.g., SOFR), daily freq
            storage_annual: float = 0.0,  # simple flat storage assumption; can be a Series if you prefer
            use_calendar_next: bool = True  # use F1 and F2 for roll stats
    ) -> pd.DataFrame:
        """
        Build daily carry metrics:
        - implied_carry_tau: (ln F(T) - ln S) / tau
        - implied_convenience = r + storage_annual - implied_carry_tau
        - roll_yield_annualized: ((F2/F1)-1) * 365/days_between
        """
        # Extract F1 (and F2 if available)
        F1, F2, T1, T2, idx = [], [], [], [], []
        for dt, curve in self.history.items():
            if curve is None or not hasattr(curve, "prices") or curve.prices is None or len(curve.prices) == 0:
                continue
            f1 = float(curve.prices[0]) if not np.isnan(curve.prices[0]) else np.nan
            f2 = float(curve.prices[1]) if (len(curve.prices) > 1 and not np.isnan(curve.prices[1])) else np.nan
            # days to expiry if available
            dte = getattr(curve, "days_to_expiry", None)
            t1 = float(dte[0]) if (dte is not None and len(dte) > 0 and not np.isnan(dte[0])) else np.nan
            t2 = float(dte[1]) if (dte is not None and len(dte) > 1 and not np.isnan(dte[1])) else np.nan
            F1.append(f1);
            F2.append(f2);
            T1.append(t1);
            T2.append(t2);
            idx.append(dt)
        df = pd.DataFrame({"F1": F1, "F2": F2, "DTE1": T1, "DTE2": T2}, index=pd.DatetimeIndex(idx)).sort_index()

        # Spot proxy if not provided
        if spot is None:
            S = df["F1"].copy()
        else:
            S = spot.reindex(df.index).ffill()

        # Rate alignment (annualized, continuously compounded or simple; we’ll treat as simple annual)
        r = (rate.reindex(df.index).ffill() if isinstance(rate, pd.Series) else pd.Series(0.0, index=df.index))

        # Storage: allow scalar or Series
        if isinstance(storage_annual, pd.Series):
            stor = storage_annual.reindex(df.index).ffill()
        else:
            stor = pd.Series(storage_annual, index=df.index)

        # Implied carry to a chosen back tenor: prefer F2, else F1 (if you want longer tenors, swap here)
        target = "F2" if use_calendar_next and df["F2"].notna().any() else "F1"
        tau_days = np.where(target == "F2", df["DTE2"].values, df["DTE1"].values)
        tau = pd.Series(np.maximum(tau_days, 1.0), index=df.index) / 365.0  # years, avoid zero

        # Core carry metrics
        with np.errstate(invalid="ignore", divide="ignore"):
            implied_carry = (np.log(df[target]) - np.log(S)) / tau  # (r + s - c)
        implied_convenience = r + stor - implied_carry  # c = r + s - implied
        df_out = pd.DataFrame({
            "spot_used": S,
            "forward_used": df[target],
            "tau_years": tau,
            "rate": r,
            "storage_annual": stor,
            "implied_carry": implied_carry,
            "implied_convenience": implied_convenience
        }, index=df.index)

        # Roll yield (front→next), annualized
        valid_gap_days = (df["DTE2"] - df["DTE1"]).where((df["DTE2"] > 0) & (df["DTE1"] > 0)).fillna(30.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            roll_yield_ann = ((df["F2"] / df["F1"]) - 1.0) * (365.0 / valid_gap_days)
        df_out["roll_yield_annualized"] = roll_yield_ann  # same as your feature, formalized

        # Simple cash-and-carry edge (ignoring fees) → positive suggests carry > (r+storage)
        df_out["carry_edge_no_fees"] = implied_carry - (r + stor)

        return df_out

    def calculate_path_signatures(self,
                                window: int = None,
                                constant_maturity: bool = False) -> Dict[str, Any]:
        """
        Calculate Lévy areas focusing on front vs back month leadership

        Parameters:
        -----------
        window : int
            Rolling window for calculations
        constant_maturity : bool, default False
            Use constant maturity tenors instead of sequential contracts

        Returns:
        --------
        Dict[str, Any]
            Lévy area analysis results with:
            - front_levy_area: Aggregated front-end Lévy area (who leads front months)
            - back_levy_area: Aggregated back-end Lévy area (who leads back months)
            - front_back_leadership: Overall front vs back leadership signal
            - regime_changes: Regime change detection based on leadership shifts
        """

        if window is None:
            window = self.default_window

        cache_key = f'levy_leadership_{window}_cm_{constant_maturity}'
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get log prices matrix - either constant maturity or sequential
        if constant_maturity and self.constant_maturity_data is not None:
            log_prices = np.log(self.constant_maturity_data.values)
        else:
            log_prices = self.get_log_prices_matrix()

        # Calculate front-end and back-end leadership using Lévy areas
        leadership_results = self._calculate_front_back_leadership(log_prices, window)

        # Detect regime changes from leadership shifts
        regime_changes = self._detect_regime_changes_from_levy(leadership_results['front_back_leadership'])

        # Create result dictionary
        path_sig = {
            'front_levy_area': leadership_results['front_levy_area'],
            'back_levy_area': leadership_results['back_levy_area'],
            'front_back_leadership': leadership_results['front_back_leadership'],
            'regime_changes': regime_changes,
            'front_leader_pct': leadership_results['front_leader_pct'],
            'back_leader_pct': leadership_results['back_leader_pct']
        }

        # Cache results
        self._cache[cache_key] = path_sig
        self.path_signatures = path_sig

        return path_sig

    def _calculate_front_back_leadership(self, log_prices: np.ndarray, window: int) -> Dict[str, np.ndarray]:
        """
        Calculate front vs back month leadership using Lévy areas.

        For each pair of consecutive contracts, calculates Lévy area to determine
        which contract leads. Then aggregates into front-end and back-end zones.

        Interpretation:
        - Positive Lévy area: Front contract of the pair leads
        - Negative Lévy area: Back contract of the pair leads

        Parameters:
        -----------
        log_prices : np.ndarray
            Matrix of log prices (n_dates, n_contracts)
        window : int
            Rolling window size

        Returns:
        --------
        Dict[str, np.ndarray]
            - front_levy_area: Average Lévy area for front-end pairs
            - back_levy_area: Average Lévy area for back-end pairs
            - front_back_leadership: Difference (front - back) showing overall leadership
            - front_leader_pct: Percentage of time front leads back
            - back_leader_pct: Percentage of time back leads front
        """
        n_dates = log_prices.shape[0]

        # Calculate Lévy areas between consecutive contracts
        all_levy_areas = self._calculate_log_levy_areas_jit(log_prices, window)

        # Determine split between front and back
        n_contract_pairs = all_levy_areas.shape[1]
        back_split_index = self._determine_back_month_split(
            self.back_month_years * 365,
            n_contract_pairs + 1  # +1 because pairs is n_contracts - 1
        )

        # Aggregate front-end Lévy areas (average across front pairs)
        if back_split_index > 0:
            front_levy_area = np.nanmean(all_levy_areas[:, :back_split_index], axis=1)
        else:
            front_levy_area = np.full(n_dates, np.nan)

        # Aggregate back-end Lévy areas (average across back pairs)
        if back_split_index < n_contract_pairs:
            back_levy_area = np.nanmean(all_levy_areas[:, back_split_index:], axis=1)
        else:
            back_levy_area = np.full(n_dates, np.nan)

        # Overall front vs back leadership signal
        # Positive: front-end leading overall curve
        # Negative: back-end leading overall curve
        front_back_leadership = front_levy_area - back_levy_area

        # Calculate percentage of time each side leads
        valid_mask = ~np.isnan(front_back_leadership)
        if np.sum(valid_mask) > 0:
            front_leader_pct = np.sum(front_back_leadership[valid_mask] > 0) / np.sum(valid_mask)
            back_leader_pct = np.sum(front_back_leadership[valid_mask] < 0) / np.sum(valid_mask)
        else:
            front_leader_pct = 0.0
            back_leader_pct = 0.0

        return {
            'front_levy_area': front_levy_area,
            'back_levy_area': back_levy_area,
            'front_back_leadership': front_back_leadership,
            'front_leader_pct': front_leader_pct,
            'back_leader_pct': back_leader_pct
        }

    def _calculate_path_variations(self,
                                  log_prices: np.ndarray,
                                  window: int) -> np.ndarray:
        """Calculate path variation (total variation of the path)"""
        
        n_dates, n_contracts = log_prices.shape
        variations = np.full((n_dates, n_contracts - 1), np.nan)
        
        for contract_pair in range(n_contracts - 1):
            for i in range(window, n_dates):
                front_window = log_prices[i-window:i, contract_pair]
                back_window = log_prices[i-window:i, contract_pair + 1]
                
                if not (np.any(np.isnan(front_window)) or np.any(np.isnan(back_window))):
                    # Calculate path length (total variation)
                    front_diffs = np.abs(np.diff(front_window))
                    back_diffs = np.abs(np.diff(back_window))
                    total_variation = np.sum(np.sqrt(front_diffs**2 + back_diffs**2))
                    variations[i, contract_pair] = total_variation
        
        return variations
    
    def _calculate_signature_levels(self, 
                                   log_prices: np.ndarray,
                                   window: int,
                                   max_level: int) -> Dict[int, np.ndarray]:
        """Calculate higher-order path signature terms"""
        
        signature_levels = {}
        n_dates, n_contracts = log_prices.shape
        
        for level in range(1, max_level + 1):
            if level == 1:
                # Level 1: Just the increments
                sig = np.full((n_dates, n_contracts - 1), np.nan)
                for i in range(window, n_dates):
                    for j in range(n_contracts - 1):
                        window_data = log_prices[i-window:i, j:j+2]
                        if not np.any(np.isnan(window_data)):
                            increments = np.diff(window_data, axis=0)
                            sig[i, j] = np.sum(increments[:, 1] - increments[:, 0])
                signature_levels[level] = sig
                
            elif level == 2:
                # Level 2: Lévy areas (already calculated)
                signature_levels[level] = self._cache.get('log_levy_areas', np.array([]))
                
            else:
                # Higher levels: Simplified approximations
                prev_sig = signature_levels[level - 1]
                sig = np.full_like(prev_sig, np.nan)
                
                for i in range(1, prev_sig.shape[0]):
                    sig[i] = prev_sig[i] * prev_sig[i-1]  # Simplified interaction
                
                signature_levels[level] = sig
        
        return signature_levels
    
    def _detect_curve_drivers(self, log_levy_areas: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect essential drivers of curve evolution from log Lévy areas

        Focuses on 4 core drivers:
        1. Front-end changes (near-term contract dynamics)
        2. Back-end changes (long-term contract dynamics)
        3. Seasonal deviations (deviations from typical seasonal patterns)
        4. Momentum (rate of change in curve evolution)

        Parameters:
        -----------
        log_levy_areas : np.ndarray
            Log Lévy areas for analysis
        """
        
        drivers = {}

        if log_levy_areas.size == 0:
            return drivers

        n_contracts = log_levy_areas.shape[1] + 1

        # Calculate target days for back months based on instance parameter
        target_back_days = self.back_month_years * 365

        # Determine split point based on DTE if available, otherwise use midpoint fallback
        back_split_index = self._determine_back_month_split(target_back_days, n_contracts)

        # 1. Front-End Driver - changes in near-term contracts
        if back_split_index > 0:
            front_end_levy = np.nanmean(log_levy_areas[:, :back_split_index], axis=1)
            drivers['front_end_changes'] = front_end_levy
        else:
            # Fallback for limited contracts
            drivers['front_end_changes'] = np.nanmean(log_levy_areas[:, :1], axis=1)

        # 2. Back-End Driver - changes in long-term contracts
        if back_split_index < n_contracts - 1:
            back_end_levy = np.nanmean(log_levy_areas[:, back_split_index:], axis=1)
            drivers['back_end_changes'] = back_end_levy
        else:
            # Fallback for limited contracts
            drivers['back_end_changes'] = np.nanmean(log_levy_areas[:, -1:], axis=1)
        
        # 3. Seasonal Driver - deviations from seasonal patterns
        seasonal_driver = self._calculate_seasonal_driver(log_levy_areas)
        drivers['seasonal_deviations'] = seasonal_driver
        
        # 4. Momentum Driver - rate of change in curve evolution
        momentum_window = min(10, log_levy_areas.shape[0] // 6)
        momentum_driver = np.full(log_levy_areas.shape[0], np.nan)
        
        for i in range(momentum_window, log_levy_areas.shape[0]):
            current_levy = np.nanmean(log_levy_areas[i-5:i, :])
            past_levy = np.nanmean(log_levy_areas[i-momentum_window:i-5, :])
            if not (np.isnan(current_levy) or np.isnan(past_levy)):
                momentum_driver[i] = current_levy - past_levy
        
        drivers['momentum'] = momentum_driver
        
        return drivers

    def _determine_back_month_split(self, target_back_days: float, n_contracts: int) -> int:
        """
        Determine the split index between front-end and back-end contracts.

        Parameters:
        -----------
        target_back_days : float
            Target days to expiry for back month contracts
        n_contracts : int
            Total number of contracts

        Returns:
        --------
        int
            Index where back-end contracts start (0-based)
        """

        # Try to use DTE data if available
        if hasattr(self, 'curves') and self.curves is not None and len(self.curves) > 0:
            # Use a representative curve to determine split
            sample_curves = []

            # Handle different curve data types (self.curves should be pd.Series)
            if isinstance(self.curves, pd.Series):
                # pd.Series case - iterate over values
                curve_iterator = self.curves.values
            else:
                # Direct iterator for other types (fallback)
                curve_iterator = self.curves

            for curve in curve_iterator:
                if curve is not None and hasattr(curve, 'days_to_expiry') and curve.days_to_expiry is not None:
                    sample_curves.append(curve)
                if len(sample_curves) >= 5:  # Use up to 5 sample curves
                    break

            if sample_curves:
                # Calculate average split index across sample curves
                split_indices = []
                for curve in sample_curves:
                    try:
                        dte_array = np.array(curve.days_to_expiry)
                        # Find first contract that meets or exceeds target days
                        valid_dte = dte_array[~np.isnan(dte_array)]
                        if len(valid_dte) > 0:
                            back_mask = valid_dte >= target_back_days
                            if np.any(back_mask):
                                split_idx = np.argmax(back_mask)
                                split_indices.append(split_idx)
                    except (AttributeError, ValueError):
                        continue

                if split_indices:
                    # Use median split index to be robust to outliers
                    avg_split = int(np.median(split_indices))
                    # Ensure reasonable bounds
                    return max(1, min(avg_split, n_contracts - 2))

        # Fallback: use contract position-based logic
        if n_contracts >= 4:
            # For monthly contracts, approximate DTE conversion
            # Assume ~30 days between contracts
            target_contract_position = max(1, int(target_back_days / 30))
            return min(target_contract_position, n_contracts - 2)
        else:
            # For very limited contracts, use simple midpoint
            return max(1, n_contracts // 2)

    def _calculate_3m_12m_spread_driver(self) -> np.ndarray:
        """
        Calculate 3m-12m spread driver using actual contract expirations
        
        Uses days-to-expiry data to find contracts closest to 3 months (~90 days) 
        and 12 months (~365 days) and calculates the spread between them.
        """
        
        if self.curves is None or len(self.curves) == 0:
            return np.array([])
        
        n_dates = len(self.curves)
        spread_driver = np.full(n_dates, np.nan)
        
        for i, (date, curve) in enumerate(self.curves.items()):
            if curve is None:
                continue
            
            try:
                # Get days to expiry for this curve
                if hasattr(curve, 'days_to_expiry') and curve.days_to_expiry is not None:
                    dte_array = np.array(curve.days_to_expiry)
                    prices_array = np.array(curve.seq_prices) if curve.seq_prices is not None else np.array(curve.prices)
                    
                    if len(dte_array) != len(prices_array):
                        continue
                    
                    # Filter for valid contracts - prioritize contracts with sufficient DTE
                    # Allow recently expired contracts to maintain curve structure
                    preferred_mask = (dte_array >= -30) & ~np.isnan(prices_array)
                    fallback_mask = (dte_array > -90) & ~np.isnan(prices_array)

                    # Use preferred mask if we have enough contracts, otherwise use fallback
                    if np.sum(preferred_mask) >= 2:
                        valid_mask = preferred_mask
                    else:
                        valid_mask = fallback_mask

                    if np.sum(valid_mask) < 2:
                        continue

                    valid_dte = dte_array[valid_mask]
                    valid_prices = prices_array[valid_mask]
                    
                    # Find contract closest to 3 months (90 days)
                    target_3m = 90
                    dte_diff_3m = np.abs(valid_dte - target_3m)
                    idx_3m = np.argmin(dte_diff_3m)
                    contract_3m_dte = valid_dte[idx_3m]
                    contract_3m_price = valid_prices[idx_3m]
                    
                    # Find contract closest to 12 months (365 days)
                    target_12m = 365
                    dte_diff_12m = np.abs(valid_dte - target_12m)
                    idx_12m = np.argmin(dte_diff_12m)
                    contract_12m_dte = valid_dte[idx_12m]
                    contract_12m_price = valid_prices[idx_12m]
                    
                    # Only calculate spread if we have reasonable approximations
                    # More flexible tolerances to maintain spread calculation as contracts mature
                    # Allow wider tolerances but prioritize maintaining the spread structure
                    tolerance_3m = max(60, 0.5 * target_3m)  # At least 60 days or 50% of target
                    tolerance_12m = max(120, 0.3 * target_12m)  # At least 120 days or 30% of target

                    if (abs(contract_3m_dte - target_3m) <= tolerance_3m and
                        abs(contract_12m_dte - target_12m) <= tolerance_12m and
                        contract_3m_dte < contract_12m_dte):  # Ensure proper ordering
                        
                        # Calculate spread (12m - 3m) normalized by 3m price
                        spread_3m_12m = (contract_12m_price - contract_3m_price) / contract_3m_price
                        spread_driver[i] = spread_3m_12m
                
                else:
                    # Fallback: estimate using contract positions if no DTE available
                    if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                        prices = np.array(curve.seq_prices)
                        valid_prices = prices[~np.isnan(prices)]
                        
                        if len(valid_prices) >= 4:
                            # Approximate: assume contracts are ~1 month apart
                            # 3m ≈ M2-M3, 12m ≈ M11-M12
                            idx_3m = min(2, len(valid_prices) - 1)
                            idx_12m = min(11, len(valid_prices) - 1)
                            
                            if idx_3m < idx_12m:
                                spread_3m_12m = (valid_prices[idx_12m] - valid_prices[idx_3m]) / valid_prices[idx_3m]
                                spread_driver[i] = spread_3m_12m
                                
            except (AttributeError, IndexError, ValueError):
                # Skip problematic dates
                continue
        
        # Apply smoothing to reduce noise (5-day rolling average)
        smoothed_driver = np.full_like(spread_driver, np.nan)
        window = 5
        
        for i in range(len(spread_driver)):
            if i >= window - 1:
                window_data = spread_driver[max(0, i-window+1):i+1]
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) > 0:
                    smoothed_driver[i] = np.mean(valid_data)
            else:
                smoothed_driver[i] = spread_driver[i]
        
        return smoothed_driver
    
    def _calculate_seasonal_driver(self, log_levy_areas: np.ndarray) -> np.ndarray:
        """
        Calculate seasonal driver that detects deviations from typical seasonal patterns
        
        This driver identifies when curve evolution significantly deviates from
        historical seasonal patterns, indicating seasonal supply/demand disruptions.
        
        IMPORTANT: Uses only historical data to avoid lookahead bias. Seasonal baselines
        are calculated using expanding windows of past data only.
        
        Parameters:
        -----------
        log_levy_areas : np.ndarray
            Log price Lévy areas matrix
            
        Returns:
        --------
        np.ndarray
            Seasonal deviation scores (higher = more deviation from seasonal norm)
            Values are 0.0 until sufficient historical data is available (2+ years)
        """
        
        if log_levy_areas.size == 0:
            return np.array([])
        
        n_dates = log_levy_areas.shape[0]

        # Need at least 1 year of data for meaningful seasonal analysis
        min_seasonal_window = min(252, n_dates // 2)  # 1 year or half the dataset

        if n_dates < min_seasonal_window:
            # Not enough data for seasonal analysis - return zeros
            return np.zeros(n_dates)

        levy_magnitudes = np.sqrt(np.nansum(log_levy_areas**2, axis=1))

        if self.curves is not None and len(self.curves) == n_dates:
            dates = self.curves.index
        else:
            dates = pd.date_range(start='2020-01-01', periods=n_dates, freq='D')

        month_codes = dates.month.to_numpy()
        levy_series = pd.Series(levy_magnitudes, index=dates)
        levy_values = levy_series.to_numpy()

        min_seasonal_lookback = min(504, n_dates)

        # Precompute cumulative statistics for each month (12 iterations at most)
        month_mean = np.full(n_dates, np.nan, dtype=float)
        month_std = np.full(n_dates, np.nan, dtype=float)
        month_count = np.zeros(n_dates, dtype=int)

        for month in np.unique(month_codes):
            month_idx = np.where(month_codes == month)[0]
            if month_idx.size == 0:
                continue

            month_vals = levy_values[month_idx]
            valid = ~np.isnan(month_vals)
            cleaned = np.where(valid, month_vals, 0.0)
            cumsum = np.cumsum(cleaned)
            cumsum_sq = np.cumsum(np.where(valid, month_vals ** 2, 0.0))
            counts = np.cumsum(valid.astype(int))

            sum_prev = np.concatenate(([0.0], cumsum[:-1]))
            sum_sq_prev = np.concatenate(([0.0], cumsum_sq[:-1]))
            count_prev = np.concatenate(([0], counts[:-1]))

            with np.errstate(invalid='ignore', divide='ignore'):
                mean_prev = np.where(count_prev > 0, sum_prev / count_prev, np.nan)
                var_prev = np.where(
                    count_prev > 1,
                    (sum_sq_prev - (sum_prev ** 2) / count_prev) / (count_prev - 1),
                    np.nan,
                )

            std_prev = np.sqrt(var_prev)
            month_mean[month_idx] = mean_prev
            month_std[month_idx] = std_prev
            month_count[month_idx] = count_prev

        # Cumulative statistics for overall history as fallback
        valid_all = ~np.isnan(levy_values)
        cleaned_all = np.where(valid_all, levy_values, 0.0)
        cumsum_all = np.cumsum(cleaned_all)
        cumsum_sq_all = np.cumsum(np.where(valid_all, levy_values ** 2, 0.0))
        counts_all = np.cumsum(valid_all.astype(int))

        sum_prev_all = np.concatenate(([0.0], cumsum_all[:-1]))
        sum_sq_prev_all = np.concatenate(([0.0], cumsum_sq_all[:-1]))
        count_prev_all = np.concatenate(([0], counts_all[:-1]))

        with np.errstate(invalid='ignore', divide='ignore'):
            overall_mean = np.where(count_prev_all > 0, sum_prev_all / count_prev_all, np.nan)
            overall_var = np.where(
                count_prev_all > 1,
                (sum_sq_prev_all - (sum_prev_all ** 2) / count_prev_all) / (count_prev_all - 1),
                np.nan,
            )

        overall_std = np.sqrt(overall_var)

        seasonal_driver = np.zeros(n_dates, dtype=float)
        eligible = (
            (np.arange(n_dates) >= min_seasonal_lookback)
            & (~np.isnan(levy_values))
        )

        month_mask = (
            eligible
            & (month_count >= 10)
            & np.isfinite(month_std)
            & (month_std > 0)
        )

        with np.errstate(invalid='ignore', divide='ignore'):
            month_z = np.abs(levy_values - month_mean) / month_std

        seasonal_driver[month_mask] = month_z[month_mask]

        fallback_mask = (
            eligible
            & ~month_mask
            & (count_prev_all >= 50)
            & np.isfinite(overall_std)
            & (overall_std > 0)
        )

        with np.errstate(invalid='ignore', divide='ignore'):
            overall_z = np.abs(levy_values - overall_mean) / overall_std

        seasonal_driver[fallback_mask] = overall_z[fallback_mask]

        seasonal_driver[~np.isfinite(seasonal_driver)] = 0.0

        window_size = min(5, n_dates // 10)
        if window_size > 1:
            seasonal_driver = (
                pd.Series(seasonal_driver, index=dates)
                .rolling(window=window_size, min_periods=1)
                .mean()
                .to_numpy()
            )

        return seasonal_driver
        
        return seasonal_driver
    
    def _detect_regime_changes_from_levy(self, leadership_signal: np.ndarray, confirmation_days: int = 3) -> np.ndarray:
        """
        Detect regime changes with time-based confirmation period.

        Regime changes occur when the leadership signal changes sign (crosses zero) and
        maintains the new sign for a confirmation period (default 2-3 days).

        Parameters:
        -----------
        leadership_signal : np.ndarray
            1D array of front_back_leadership values (positive = front leads, negative = back leads)
        confirmation_days : int, default 3
            Number of consecutive days the new regime must persist to confirm the change

        Returns:
        --------
        np.ndarray
            Indices where confirmed regime changes occur
        """

        if leadership_signal.size == 0:
            return np.array([])

        # Remove NaN values for sign detection
        valid_mask = ~np.isnan(leadership_signal)
        if np.sum(valid_mask) < confirmation_days:
            return np.array([])

        # Get sign of leadership signal
        sign_signal = np.sign(leadership_signal)

        # Detect potential regime changes (zero crossings)
        sign_changes = np.diff(sign_signal)
        potential_changes = np.where(np.abs(sign_changes) > 0)[0] + 1

        # Confirm regime changes by checking if new regime persists
        confirmed_changes = []

        for change_idx in potential_changes:
            # Check if we have enough data after the change
            if change_idx + confirmation_days > len(sign_signal):
                continue

            # Get the new regime sign
            new_regime_sign = sign_signal[change_idx]

            # Skip if new regime is neutral (zero)
            if new_regime_sign == 0:
                continue

            # Check if the new sign persists for confirmation_days
            confirmation_window = sign_signal[change_idx:change_idx + confirmation_days]

            # Regime is confirmed if all values in confirmation window have same sign as new regime
            # (allowing for zeros which are treated as continuation)
            confirmed = np.all((confirmation_window == new_regime_sign) | (confirmation_window == 0))

            if confirmed:
                confirmed_changes.append(change_idx)

        # Create regime change array
        regime_changes = np.zeros(len(leadership_signal), dtype=int)
        if len(confirmed_changes) > 0:
            regime_changes[confirmed_changes] = 1

        return regime_changes
    
    def analyze_curve_evolution_drivers(self,
                                       window: int = None,
                                       constant_maturity: bool = False) -> Dict[str, Any]:
        """
        Analyze front vs back month leadership dynamics

        Parameters:
        -----------
        window : int, optional
            Analysis window size
        constant_maturity : bool, default False
            Use constant maturity tenors instead of sequential contracts

        Returns:
        --------
        Dict[str, Any]
            Leadership analysis results including:
            - front_levy_area: Front-end leadership signal
            - back_levy_area: Back-end leadership signal
            - front_back_leadership: Overall front vs back leadership
            - regime_changes: Detected leadership regime changes
            - leadership_statistics: Summary statistics

        Notes:
        ------
        Back-end contract definition is controlled by the `back_month_years` parameter
        set during analyzer initialization (default 1.0 year).
        """

        if window is None:
            window = self.default_window

        # Setup constant maturity if requested
        if constant_maturity and self.constant_maturity_data is None:
            success = self.setup_constant_maturity()
            if not success:
                warnings.warn("Constant maturity setup failed, falling back to sequential contracts")
                constant_maturity = False

        # Calculate Lévy area leadership
        levy_results = self.calculate_path_signatures(window, constant_maturity=constant_maturity)

        # Regime transition analysis
        regime_analysis = self._analyze_regime_transitions(levy_results['regime_changes'])

        # Calculate leadership statistics
        leadership_stats = self._analyze_leadership_statistics(levy_results)

        return {
            'front_levy_area': levy_results['front_levy_area'],
            'back_levy_area': levy_results['back_levy_area'],
            'front_back_leadership': levy_results['front_back_leadership'],
            'regime_changes': levy_results['regime_changes'],
            'regime_analysis': regime_analysis,
            'leadership_statistics': leadership_stats,
            'summary_statistics': {
                'n_curves': len(self.curves) if self.curves is not None else 0,
                'date_range': (self.curves.index.min(), self.curves.index.max()) if self.curves is not None else None,
                'n_regime_changes': int(np.sum(levy_results['regime_changes'])) if levy_results['regime_changes'].size > 0 else 0,
                'front_leader_pct': levy_results['front_leader_pct'],
                'back_leader_pct': levy_results['back_leader_pct'],
                'current_leader': 'front' if levy_results['front_back_leadership'][-1] > 0 else 'back' if levy_results['front_back_leadership'][-1] < 0 else 'neutral'
            }
        }

    def _analyze_leadership_statistics(self, levy_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate detailed leadership statistics

        Parameters:
        -----------
        levy_results : Dict[str, np.ndarray]
            Results from calculate_path_signatures

        Returns:
        --------
        Dict[str, float]
            Leadership statistics
        """
        stats = {}

        front_levy = levy_results['front_levy_area']
        back_levy = levy_results['back_levy_area']
        leadership = levy_results['front_back_leadership']

        # Remove NaN values for statistics
        valid_mask = ~np.isnan(leadership)
        if np.sum(valid_mask) > 0:
            leadership_valid = leadership[valid_mask]

            # Basic statistics
            stats['mean_leadership'] = float(np.mean(leadership_valid))
            stats['std_leadership'] = float(np.std(leadership_valid))
            stats['median_leadership'] = float(np.median(leadership_valid))

            # Leadership strength
            stats['avg_front_strength'] = float(np.mean(front_levy[~np.isnan(front_levy)]))
            stats['avg_back_strength'] = float(np.mean(back_levy[~np.isnan(back_levy)]))

            # Persistence (autocorrelation at lag 1)
            if len(leadership_valid) > 1:
                stats['leadership_persistence'] = float(np.corrcoef(leadership_valid[:-1], leadership_valid[1:])[0, 1])
            else:
                stats['leadership_persistence'] = 0.0

            # Volatility of leadership
            stats['leadership_volatility'] = float(np.std(np.diff(leadership_valid)))

        else:
            stats = {
                'mean_leadership': 0.0,
                'std_leadership': 0.0,
                'median_leadership': 0.0,
                'avg_front_strength': 0.0,
                'avg_back_strength': 0.0,
                'leadership_persistence': 0.0,
                'leadership_volatility': 0.0
            }

        return stats

    def _analyze_driver_importance(self, drivers: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze the importance of each curve driver (legacy method - deprecated)"""
        
        importance = {}
        
        for driver_name, driver_values in drivers.items():
            if driver_values.size > 0:
                # Use variance as a measure of importance
                valid_values = driver_values[~np.isnan(driver_values)]
                if len(valid_values) > 0:
                    importance[driver_name] = np.var(valid_values)
                else:
                    importance[driver_name] = 0.0
            else:
                importance[driver_name] = 0.0
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def _calculate_driver_correlations(self, drivers: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate correlations between curve drivers"""
        
        driver_names = list(drivers.keys())
        n_drivers = len(driver_names)
        
        if n_drivers == 0:
            return pd.DataFrame()
        
        # Create correlation matrix
        correlations = np.full((n_drivers, n_drivers), np.nan)
        
        for i, driver1 in enumerate(driver_names):
            for j, driver2 in enumerate(driver_names):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    values1 = drivers[driver1]
                    values2 = drivers[driver2]
                    
                    # Find common valid indices
                    valid_mask = ~(np.isnan(values1) | np.isnan(values2))
                    
                    if np.sum(valid_mask) > 10:  # Minimum data points
                        correlations[i, j] = np.corrcoef(
                            values1[valid_mask], 
                            values2[valid_mask]
                        )[0, 1]
        
        return pd.DataFrame(
            correlations, 
            index=driver_names, 
            columns=driver_names
        )
    
    def _analyze_regime_transitions(self, regime_changes: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transition patterns"""
        
        if regime_changes.size == 0:
            return {}
        
        # Find regime change points
        change_points = np.where(regime_changes > 0)[0]
        
        if len(change_points) == 0:
            return {'n_regimes': 1, 'average_regime_length': len(regime_changes)}
        
        # Calculate regime lengths
        if len(change_points) == 1:
            regime_lengths = [len(regime_changes)]
        else:
            regime_lengths = np.diff(np.concatenate(([0], change_points, [len(regime_changes)])))
        
        return {
            'n_regimes': len(change_points) + 1,
            'regime_change_points': change_points.tolist(),
            'regime_lengths': regime_lengths.tolist(),
            'average_regime_length': np.mean(regime_lengths),
            'regime_length_std': np.std(regime_lengths),
            'regime_stability': 1.0 / (1.0 + len(change_points) / len(regime_changes))
        }
    
    def detect_curve_flips(self, flip_threshold: float = 2.0, min_flip_duration: int = 5) -> pd.DataFrame:
        """
        Detect significant curve flips during evolution (contango/backwardation changes)
        
        Parameters:
        -----------
        flip_threshold : float
            Z-score threshold for detecting significant curve shape changes
        min_flip_duration : int
            Minimum number of periods for a flip to be considered significant
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with flip events including:
            - flip_start: Start date of flip
            - flip_end: End date of flip 
            - flip_type: 'contango_to_backwardation' or 'backwardation_to_contango'
            - flip_magnitude: Magnitude of the flip
            - flip_duration: Duration in periods
        """
        
        if self.curves is None or len(self.curves) < 10:
            return pd.DataFrame()
        
        # Calculate curve slopes over time using broadcast method
        slopes = self._calculate_curve_slopes_broadcast()
        
        if slopes.empty:
            return pd.DataFrame()
        
        slope_changes = slopes.diff().abs()
        min_window = 60  # Need at least 60 observations for meaningful statistics

        expanding_mean = slope_changes.expanding(min_periods=min_window).mean()
        expanding_std = slope_changes.expanding(min_periods=min_window).std()

        with np.errstate(invalid='ignore', divide='ignore'):
            slope_zscore = (slope_changes - expanding_mean) / expanding_std

        slope_zscore = slope_zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        slope_zscore[expanding_std <= 0] = 0.0

        flip_mask = slope_zscore > flip_threshold
        if not flip_mask.any():
            return pd.DataFrame()

        start_mask = flip_mask & ~flip_mask.shift(fill_value=False)
        end_mask = ~flip_mask & flip_mask.shift(fill_value=False)

        start_positions = np.flatnonzero(start_mask.to_numpy())
        end_positions = np.flatnonzero(end_mask.to_numpy())

        prev_slope = slopes.ffill().shift(1).fillna(0.0)

        flip_events = []
        end_idx = 0

        for start_pos in start_positions:
            while end_idx < len(end_positions) and end_positions[end_idx] <= start_pos:
                end_idx += 1

            if end_idx >= len(end_positions):
                break

            end_pos = end_positions[end_idx]
            start_date = slope_zscore.index[start_pos]
            end_date = slope_zscore.index[end_pos]

            duration_days = int((end_date - start_date) / np.timedelta64(1, 'D'))
            if duration_days < min_flip_duration:
                continue

            flip_magnitude = slope_changes.iloc[start_pos:end_pos].max()
            current_slope = slopes.iloc[start_pos]
            previous_slope = prev_slope.iloc[start_pos]

            flip_type = (
                'backwardation_to_contango'
                if current_slope > previous_slope
                else 'contango_to_backwardation'
            )

            flip_events.append({
                'flip_start': start_date,
                'flip_end': end_date,
                'flip_type': flip_type,
                'flip_magnitude': flip_magnitude,
                'flip_duration': duration_days
            })

            end_idx += 1

        return pd.DataFrame(flip_events)
    
    def _calculate_curve_slopes_broadcast(self) -> pd.Series:
        """
        Calculate curve slopes for all dates using broadcasting
        
        Returns:
        --------
        pd.Series with datetime index and slope values
        """
        
        def extract_slope(curve):
            if curve is None:
                return np.nan
            try:
                # Use the curve's slope calculation method
                slope_data = curve.get_granular_slope(method='linear_regression')
                return slope_data.get('slope', np.nan)
            except (AttributeError, ValueError):
                # Fallback: simple slope calculation
                try:
                    if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                        prices = curve.seq_prices
                    else:
                        prices = curve.prices
                    
                    valid_prices = [p for p in prices if not np.isnan(p)]
                    if len(valid_prices) >= 2:
                        x = np.arange(len(valid_prices))
                        return np.polyfit(x, valid_prices, 1)[0]  # Linear slope
                except (AttributeError, ValueError):
                    pass
            return np.nan
        
        # Apply slope extraction to all curves
        slopes = self.curves.apply(extract_slope)
        return slopes.dropna()
    
    def normalize_drivers_for_seasonality(self, drivers: Dict[str, np.ndarray], 
                                        seasonal_window: int = 252) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Normalize curve drivers for seasonal patterns and provide seasonal baselines
        
        Parameters:
        -----------
        drivers : Dict[str, np.ndarray]
            Raw driver data
        seasonal_window : int
            Window for seasonal pattern calculation (default 252 days = 1 year)
            
        Returns:
        --------
        Dict[str, Dict[str, np.ndarray]]
            For each driver, returns:
            - 'raw': Original values
            - 'seasonal_normalized': Seasonally adjusted values
            - 'seasonal_baseline': What's normal for each season
            - 'seasonal_deviation': How much current values deviate from seasonal norm
        """
        
        normalized_data = {}
        
        for driver_name, driver_values in drivers.items():
            if len(driver_values) < seasonal_window:
                # Not enough data for seasonal normalization
                normalized_data[driver_name] = {
                    'raw': driver_values,
                    'seasonal_normalized': driver_values,
                    'seasonal_baseline': np.full_like(driver_values, np.nan),
                    'seasonal_deviation': np.zeros_like(driver_values)
                }
                continue
            
            # Calculate seasonal patterns
            seasonal_results = self._calculate_detailed_seasonal_pattern(driver_values, seasonal_window)
            
            normalized_data[driver_name] = seasonal_results
        
        return normalized_data
    
    def _calculate_detailed_seasonal_pattern(self, values: np.ndarray, window: int) -> Dict[str, np.ndarray]:
        """
        Calculate detailed seasonal patterns with baselines and deviations
        
        Parameters:
        -----------
        values : np.ndarray
            Time series values
        window : int
            Seasonal window (typically 252 for yearly patterns)
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing raw, normalized, baseline, and deviation data
        """
        
        try:
            # Create date index if we have curve data
            if self.curves is not None and len(self.curves) == len(values):
                date_index = self.curves.index
            else:
                # Fallback to generic date range
                date_index = pd.date_range(start='2020-01-01', periods=len(values), freq='D')
            
            series = pd.Series(values, index=date_index[:len(values)])
            
            # Calculate expanding monthly seasonal averages (avoiding lookahead bias)
            seasonal_baseline = np.full(len(series), np.nan)
            min_history = 252  # Need at least 1 year of data
            
            for i in range(len(series)):
                current_month = series.index[i].month
                
                if i >= min_history:
                    # Use only historical data up to current point
                    historical_series = series.iloc[:i]
                    same_month_data = historical_series[historical_series.index.month == current_month]
                    
                    if len(same_month_data) >= 5:  # Need at least 5 historical observations for the month
                        seasonal_baseline[i] = same_month_data.mean()
                    else:
                        # Fallback to expanding mean of all historical data
                        seasonal_baseline[i] = historical_series.mean()
                else:
                    # Insufficient history - use expanding mean
                    if i > 0:
                        seasonal_baseline[i] = series.iloc[:i].mean()
                    else:
                        seasonal_baseline[i] = series.iloc[i]
            
            # Calculate deviations from seasonal norm
            seasonal_deviation = values - seasonal_baseline
            
            # Normalize by seasonal pattern (remove seasonal component)
            seasonal_normalized = seasonal_deviation
            
            return {
                'raw': values,
                'seasonal_normalized': seasonal_normalized,
                'seasonal_baseline': seasonal_baseline,
                'seasonal_deviation': seasonal_deviation
            }
            
        except Exception as e:
            # Fallback to simple pattern
            seasonal_pattern = self._calculate_seasonal_pattern(values, window)
            return {
                'raw': values,
                'seasonal_normalized': values - seasonal_pattern if seasonal_pattern is not None else values,
                'seasonal_baseline': seasonal_pattern if seasonal_pattern is not None else np.full_like(values, np.nan),
                'seasonal_deviation': values - seasonal_pattern if seasonal_pattern is not None else np.zeros_like(values)
            }
    
    def _calculate_seasonal_pattern(self, values: np.ndarray, window: int) -> Optional[np.ndarray]:
        """
        Calculate seasonal pattern using rolling statistics (fallback method)
        
        Parameters:
        -----------
        values : np.ndarray
            Time series values
        window : int
            Seasonal window
            
        Returns:
        --------
        np.ndarray or None
            Seasonal pattern or None if calculation fails
        """
        
        try:
            # Use DataFrame for convenient rolling operations
            series = pd.Series(values)
            
            # Calculate rolling seasonal mean
            seasonal_mean = series.rolling(window=window, center=True).mean()
            
            # Forward fill and backward fill to handle edges
            seasonal_pattern = seasonal_mean.fillna(method='bfill').fillna(method='ffill')
            
            return seasonal_pattern.values
            
        except Exception:
            return None
    
    def get_contango_backwardation_analysis(self, seasonal_adjust: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of contango vs backwardation patterns with seasonal context
        
        Parameters:
        -----------
        seasonal_adjust : bool
            Whether to provide seasonal analysis
            
        Returns:
        --------
        Dict[str, Any]
            Complete contango/backwardation analysis including:
            - current_state: 'contango' or 'backwardation'
            - curve_slopes: Time series of curve slopes
            - flip_events: DataFrame of significant structure changes
            - seasonal_patterns: What's normal for each month (if seasonal_adjust=True)
            - seasonal_deviations: How current structure compares to seasonal norms
        """
        
        if self.curves is None:
            return {}
        
        # Get curve slopes over time
        slopes = self._calculate_curve_slopes_broadcast()
        
        if slopes.empty:
            return {}
        
        # Detect flip events
        flip_events = self.detect_curve_flips()
        
        # Determine current state
        current_slope = slopes.iloc[-1] if len(slopes) > 0 else 0
        current_state = 'contango' if current_slope > 0 else 'backwardation'
        
        result = {
            'current_state': current_state,
            'current_slope': current_slope,
            'curve_slopes': slopes,
            'flip_events': flip_events,
            'slope_statistics': {
                'mean': slopes.mean(),
                'std': slopes.std(),
                'contango_days': (slopes > 0).sum(),
                'backwardation_days': (slopes < 0).sum(),
                'contango_percentage': (slopes > 0).mean() * 100
            }
        }
        
        if seasonal_adjust and len(slopes) >= 252:  # Need at least 1 year of data
            # Analyze seasonal patterns in curve structure
            seasonal_data = self.normalize_drivers_for_seasonality({'curve_slope': slopes.values})
            
            if 'curve_slope' in seasonal_data:
                slope_seasonal = seasonal_data['curve_slope']
                
                # Calculate monthly contango/backwardation tendencies
                monthly_tendencies = self._calculate_monthly_curve_tendencies(slopes)
                
                result['seasonal_analysis'] = {
                    'seasonal_baseline': slope_seasonal['seasonal_baseline'],
                    'seasonal_deviation': slope_seasonal['seasonal_deviation'],
                    'monthly_tendencies': monthly_tendencies,
                    'most_contango_months': [month for month, tendency in monthly_tendencies.items() 
                                           if tendency > 0.6],
                    'most_backwardation_months': [month for month, tendency in monthly_tendencies.items() 
                                                if tendency < 0.4]
                }
        
        return result
    
    def _calculate_monthly_curve_tendencies(self, slopes: pd.Series) -> Dict[str, float]:
        """
        Calculate the tendency for each month to be in contango vs backwardation
        
        Returns:
        --------
        Dict[str, float]
            For each month (name), the percentage of time in contango (0-1)
        """
        
        monthly_contango = slopes.groupby(slopes.index.month).apply(lambda x: (x > 0).mean())
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return {month_names[i-1]: monthly_contango.get(i, 0.5) for i in range(1, 13)}
    
    def clear_all_caches(self):
        """
        Clear all cached calculations to force recalculation with updated formulas
        
        Use this after the Lévy area calculation fix to ensure all results use
        the corrected scaling-free formula.
        """
        self._cache.clear()
        self._log_price_cache.clear()
        self._levy_cache.clear()
        
        # Reset analysis components to None to force recalculation
        self.path_signatures = None
        self.regime_analysis = None
        self.seasonal_patterns = None
        
        print("All caches cleared. Next analysis will use corrected Lévy area calculations.")
    
    def plot_curve_evolution_analysis(self,
                                     show_drivers: bool = True,
                                     show_seasonal_analysis: bool = True,
                                     show_3m_12m_spread: bool = True,
                                     constant_maturity: bool = False,
                                     regime_confirmation_days: int = 3,
                                     height: int = 1200) -> go.Figure:
        """
        Focused visualization showing front month, back/front spread, drivers, seasonal analysis, and 3m-12m spread

        Parameters:
        -----------
        show_drivers : bool
            Show curve driver analysis
        show_seasonal_analysis : bool
            Show seasonal analysis of curve patterns
        show_3m_12m_spread : bool
            Show 3m-12m spread as separate bottom chart
        constant_maturity : bool, default False
            Use constant maturity tenors instead of sequential contracts for analysis
        regime_confirmation_days : int, default 3
            Number of consecutive days required to confirm a regime change (2-3 day confirmation period)
        height : int
            Plot height

        Returns:
        --------
        plotly.graph_objects.Figure
        """

        if self.path_signatures is None or constant_maturity:
            self.calculate_path_signatures(constant_maturity=constant_maturity)
        
        # Fixed number of subplots: Front Month, Back/Front Spread, Drivers, Seasonal Analysis, 3m-12m Spread
        n_plots = 2  # Always show front month and back/front spread
        if show_drivers:
            n_plots += 1
        if show_seasonal_analysis:
            n_plots += 1
        if show_3m_12m_spread:
            n_plots += 1
        
        # Create subplot titles
        subplot_titles = ['Front Month, Synthetic Spot & Spot Prices', 'Back/Front Spread']
        if show_drivers:
            subplot_titles.append('Front vs Back Leadership (Lévy Areas)')
        if show_seasonal_analysis:
            subplot_titles.append('Seasonal Deviations')
        if show_3m_12m_spread:
            subplot_titles.append('3m-12m Spread')
        
        fig = make_subplots(
            rows=n_plots, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Get dates for x-axis - use constant maturity dates if available
        if constant_maturity and hasattr(self, 'constant_maturity_data') and self.constant_maturity_data is not None:
            dates = self.constant_maturity_data.index
        elif self.curves is not None:
            dates = self.curves.index
        else:
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        current_row = 1

        # 1. Front month price
        self._add_front_month_plot(fig, current_row, dates, constant_maturity=constant_maturity, confirmation_days=regime_confirmation_days)
        current_row += 1
        
        # 2. Back/Front spread
        self._add_back_front_spread_plot(fig, current_row, dates, constant_maturity=constant_maturity)
        current_row += 1
        
        # 3. Leadership analysis (replaces old curve drivers)
        if show_drivers:
            self._add_leadership_analysis_plot(fig, current_row, dates, confirmation_days=regime_confirmation_days)
            current_row += 1
        
        # 4. Seasonal analysis
        if show_seasonal_analysis:
            self._add_seasonal_analysis_plot(fig, current_row, dates)
            current_row += 1
        
        # 5. 3m-12m spread
        if show_3m_12m_spread:
            self._add_3m_12m_spread_plot(fig, current_row, dates, constant_maturity=constant_maturity)
        
        # Update layout
        title = f'Curve Evolution Analysis'
        if self.symbol:
            title += f' - {self.symbol}'
            
        fig.update_layout(
            title=title,
            height=height,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def _add_curve_evolution_plot(self, fig, row, dates):
        """Add main curve evolution plot"""
        
        if self.curves is None:
            return
        
        # Sample a few dates for visualization
        sample_dates = dates[::max(1, len(dates)//20)]  # Sample ~20 curves
        
        for i, date in enumerate(sample_dates):
            if date in self.curves.index:
                curve = self.curves[date]
                if curve is not None:
                    # Get prices from curve
                    if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                        prices = curve.seq_prices
                    elif hasattr(curve, 'prices') and curve.prices is not None:
                        prices = curve.prices
                    else:
                        continue
                        
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(prices))),
                            y=prices,
                            mode='lines',
                            name=f'{date.strftime("%Y-%m-%d")}',
                            opacity=0.7,
                            showlegend=(i < 5)  # Only show legend for first few
                        ),
                        row=row, col=1
                    )
    
    def _add_leadership_analysis_plot(self, fig, row, dates, confirmation_days: int = 3):
        """Add front vs back leadership analysis plot"""

        # Get leadership data from path_signatures
        front_levy = self.path_signatures['front_levy_area']
        back_levy = self.path_signatures['back_levy_area']
        leadership = self.path_signatures['front_back_leadership']

        # Recalculate regime changes with the specified confirmation period
        regime_changes = self._detect_regime_changes_from_levy(leadership, confirmation_days=confirmation_days)

        # Align dates with data
        aligned_length = min(len(dates), len(leadership))
        plot_dates = dates[:aligned_length]

        # Plot front-end Lévy area
        valid_front = ~np.isnan(front_levy[:aligned_length])
        if np.sum(valid_front) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_dates[valid_front],
                    y=front_levy[:aligned_length][valid_front],
                    mode='lines',
                    name='Front-End Leadership',
                    line=dict(color='blue', width=2)
                ),
                row=row, col=1
            )

        # Plot back-end Lévy area
        valid_back = ~np.isnan(back_levy[:aligned_length])
        if np.sum(valid_back) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_dates[valid_back],
                    y=back_levy[:aligned_length][valid_back],
                    mode='lines',
                    name='Back-End Leadership',
                    line=dict(color='red', width=2)
                ),
                row=row, col=1
            )

        # Plot overall leadership signal
        valid_leadership = ~np.isnan(leadership[:aligned_length])
        if np.sum(valid_leadership) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_dates[valid_leadership],
                    y=leadership[:aligned_length][valid_leadership],
                    mode='lines',
                    name='Front-Back Difference',
                    line=dict(color='green', width=3, dash='dash')
                ),
                row=row, col=1
            )

        # Add zero line (regime boundary)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row, col=1,
                     annotation_text=f"{confirmation_days}-day confirmation",
                     annotation_position="top right")

        # Highlight regime changes if available
        if regime_changes.size > 0 and len(regime_changes) >= aligned_length:
            regime_points = np.where(regime_changes[:aligned_length] > 0)[0]
            if len(regime_points) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=plot_dates[regime_points],
                        y=leadership[:aligned_length][regime_points],
                        mode='markers',
                        name='Regime Changes',
                        marker=dict(color='purple', size=10, symbol='x')
                    ),
                    row=row, col=1
                )
    
    def _add_regime_plot(self, fig, row, dates):
        """Add regime changes plot"""
        
        regime_changes = self.path_signatures['regime_changes']
        
        if regime_changes.size > 0:
            # Trim dates to match regime_changes length
            plot_dates = dates[:len(regime_changes)]
            
            fig.add_trace(
                go.Scatter(
                    x=plot_dates,
                    y=regime_changes,
                    mode='lines+markers',
                    name='Regime Changes',
                    line=dict(color='purple'),
                    fill='tonexty',
                    fillcolor='rgba(128, 0, 128, 0.2)'
                ),
                row=row, col=1
            )
    
    def _add_levy_areas_plot(self, fig, row, dates):
        """Add Lévy areas plot"""
        
        log_levy_areas = self.path_signatures['log_levy_areas']
        
        if log_levy_areas.size > 0:
            # Plot mean Lévy area across all contract pairs
            mean_levy = np.nanmean(log_levy_areas, axis=1)
            valid_mask = ~np.isnan(mean_levy)
            
            if np.sum(valid_mask) > 0:
                # Ensure we align dates and levy values correctly
                aligned_length = min(len(dates), len(mean_levy))
                aligned_dates = dates[:aligned_length]
                aligned_mean_levy = mean_levy[:aligned_length]
                aligned_valid_mask = valid_mask[:aligned_length]

                plot_dates = aligned_dates[aligned_valid_mask]
                plot_values = aligned_mean_levy[aligned_valid_mask]
                
                fig.add_trace(
                    go.Scatter(
                        x=plot_dates,
                        y=plot_values,
                        mode='lines',
                        name='Mean Log Lévy Area',
                        line=dict(color='orange'),
                        fill='tonexty',
                        fillcolor='rgba(255, 165, 0, 0.1)'
                    ),
                    row=row, col=1
                )
    
    def _add_front_month_plot(self, fig, row, dates, constant_maturity=False, confirmation_days=3):
        """Add front month (F0) price plot with synthetic spot and spot data using broadcasting"""

        valid_dates = None
        front_month_prices = None

        if constant_maturity and hasattr(self, 'constant_maturity_data') and self.constant_maturity_data is not None:
            # Use constant maturity data for front month plot
            cm_data = self.constant_maturity_data

            # Try to get 1M (front month equivalent) from constant maturity data
            if '1M' in cm_data.columns:
                front_month_series = cm_data['1M']
            elif '3M' in cm_data.columns:
                # Fallback to 3M if 1M not available
                front_month_series = cm_data['3M']
            else:
                # Use first available tenor
                front_month_series = cm_data.iloc[:, 0]

            valid_mask = ~np.isnan(front_month_series)
            if np.sum(valid_mask) > 0:
                valid_dates = front_month_series.index[valid_mask]
                front_month_prices = front_month_series.values[valid_mask]

                # Plot front month constant maturity price series
                fig.add_trace(
                    go.Scatter(
                        x=valid_dates,
                        y=front_month_prices,
                        mode='lines',
                        name='Front Month (Constant Maturity)',
                        line=dict(color='blue', width=2),
                        hovertemplate='Date: %{x}<br>Front Month Price: %{y:.2f}<extra></extra>'
                    ),
                    row=row, col=1
                )

        if self.curves is not None:
            # Use broadcast method to extract F0 prices
            front_month_series = self._extract_front_month_broadcast()
            
            if len(front_month_series) > 0:
                valid_dates = front_month_series.index
                front_month_prices = front_month_series.values
            
            if len(front_month_prices) > 0:
                # Plot front month price series
                fig.add_trace(
                    go.Scatter(
                        x=valid_dates,
                        y=front_month_prices,
                        mode='lines',
                        name='Front Month (F0)',
                        line=dict(color='blue', width=2),
                        hovertemplate='Date: %{x}<br>F0 Price: %{y:.2f}<extra></extra>'
                    ),
                    row=row, col=1
                )
                
                # Add synthetic spot data if available
                if self.spread_data is not None:
                    # Get synthetic spot prices
                    synthetic_spot = self.spread_data.get_spot_prices(prefer_real=False)
                    if synthetic_spot is not None and len(synthetic_spot) > 0:
                        # Handle both Series and ndarray returns
                        if isinstance(synthetic_spot, pd.Series):
                            x_data, y_data = synthetic_spot.index, synthetic_spot.values
                        else:
                            # If it's an ndarray, use the full spread_data index
                            x_data = self.spread_data.index
                            y_data = synthetic_spot
                            
                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_data,
                                mode='lines',
                                name='Synthetic Spot',
                                line=dict(color='orange', width=2, dash='dot'),
                                hovertemplate='Date: %{x}<br>Synthetic Spot: %{y:.2f}<extra></extra>'
                            ),
                            row=row, col=1
                        )
                    
                    # Add real spot data if available
                    real_spot = self.spread_data.get_spot_prices(prefer_real=True)
                    if (real_spot is not None and len(real_spot) > 0 and 
                        hasattr(self.spread_data, 'spot_prices') and 
                        self.spread_data.spot_prices is not None):
                        # Handle both Series and ndarray returns  
                        if isinstance(real_spot, pd.Series):
                            x_data, y_data = real_spot.index, real_spot.values
                        else:
                            # If it's an ndarray, use the full spread_data index
                            x_data = self.spread_data.index
                            y_data = real_spot
                            
                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_data,
                                mode='lines',
                                name='Real Spot',
                                line=dict(color='green', width=2),
                                hovertemplate='Date: %{x}<br>Real Spot: %{y:.2f}<extra></extra>'
                            ),
                            row=row, col=1
                        )
                
                # Add trend line if enough data points
                if len(front_month_prices) >= 10:
                    x_numeric = np.arange(len(front_month_prices))
                    coeffs = np.polyfit(x_numeric, front_month_prices, 1)
                    trend_line = coeffs[0] * x_numeric + coeffs[1]

                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=trend_line,
                            mode='lines',
                            name='F0 Trend',
                            line=dict(color='red', width=1, dash='dash'),
                            hovertemplate='Date: %{x}<br>Trend: %{y:.2f}<extra></extra>'
                        ),
                        row=row, col=1
                    )

        # Add regime change markers if path_signatures available and we have price data
        if (valid_dates is not None and front_month_prices is not None and
            self.path_signatures is not None and 'front_back_leadership' in self.path_signatures):

            leadership = self.path_signatures['front_back_leadership']
            regime_changes = self._detect_regime_changes_from_levy(leadership, confirmation_days=confirmation_days)

            # Align regime changes with front month prices
            aligned_length = min(len(valid_dates), len(regime_changes))
            regime_points = np.where(regime_changes[:aligned_length] > 0)[0]

            if len(regime_points) > 0:
                # Get prices at regime change points
                regime_dates = valid_dates[regime_points]
                regime_prices = front_month_prices[regime_points]

                fig.add_trace(
                    go.Scatter(
                        x=regime_dates,
                        y=regime_prices,
                        mode='markers',
                        name='Regime Changes (Price)',
                        marker=dict(
                            color='purple',
                            size=12,
                            symbol='star',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate='Regime Change<br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
                    ),
                    row=row, col=1
                )

        # Update y-axis label for this subplot
        fig.update_yaxes(title_text="Price", row=row, col=1)

    def _select_contract_by_dte(self,
                                 curve: FuturesCurve,
                                 target_days: int,
                                 tolerance: Optional[int] = 120,
                                 allow_fallback: bool = True) -> Tuple[Optional[float], Optional[float]]:
        """Select contract price closest to target days-to-expiry."""
        if curve is None:
            return None, None

        dte_source = getattr(curve, 'days_to_expiry', None)
        if dte_source is None and hasattr(curve, 'seq_dte'):
            dte_source = getattr(curve, 'seq_dte')

        if dte_source is None:
            return None, None

        try:
            dte_array = np.asarray(dte_source, dtype=float)
        except Exception:
            return None, None

        price_source = None
        if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
            try:
                price_source = np.asarray(curve.seq_prices, dtype=float)
            except Exception:
                price_source = None

        if price_source is None:
            try:
                price_source = np.asarray(curve.prices, dtype=float)
            except Exception:
                return None, None

        if price_source.size == 0 or dte_array.size == 0:
            return None, None

        min_len = min(len(price_source), len(dte_array))
        price_array = price_source[:min_len]
        dte_array = dte_array[:min_len]

        # Prioritize contracts with 360+ DTE, but allow shorter if absolutely necessary
        preferred_mask = (~np.isnan(price_array)) & (~np.isnan(dte_array)) & (dte_array >= 360)
        fallback_mask = (~np.isnan(price_array)) & (~np.isnan(dte_array)) & (dte_array > 0)

        # Use preferred (360+ DTE) contracts if available
        if np.any(preferred_mask):
            valid_mask = preferred_mask
        else:
            valid_mask = fallback_mask

        if not np.any(valid_mask):
            return None, None

        valid_prices = price_array[valid_mask]
        valid_dte = dte_array[valid_mask]

        if valid_prices.size == 0:
            return None, None

        idx = int(np.argmin(np.abs(valid_dte - target_days)))
        if tolerance is not None and abs(valid_dte[idx] - target_days) > tolerance:
            if allow_fallback:
                # Prefer the longest available DTE, but ensure it's at least 360 if possible
                if np.any(preferred_mask):
                    preferred_prices = price_array[preferred_mask]
                    preferred_dte = dte_array[preferred_mask]
                    idx = int(np.argmax(preferred_dte))
                    return float(preferred_prices[idx]), float(preferred_dte[idx])
                else:
                    idx = int(np.argmax(valid_dte))
            else:
                return None, None

        return float(valid_prices[idx]), float(valid_dte[idx])

    def _add_back_front_spread_plot(self, fig, row, dates, constant_maturity=False):
        """Add back/front spread plot using new __getitem__ method or constant maturity data"""

        if constant_maturity and hasattr(self, 'constant_maturity_data') and self.constant_maturity_data is not None:
            # Use constant maturity data for spread calculation
            cm_data = self.constant_maturity_data

            # Get front month (1 month) and back month (24 months) from constant maturity data
            if '1M' in cm_data.columns and '24M' in cm_data.columns:
                front_prices = cm_data['1M']
                back_prices = cm_data['24M']

                # Calculate spread
                spread_values = back_prices - front_prices

                # Remove NaN values
                valid_mask = ~(np.isnan(spread_values))
                if np.sum(valid_mask) > 0:
                    valid_dates = cm_data.index[valid_mask]
                    valid_spreads = spread_values[valid_mask]

                    # Plot constant maturity back/front spread
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=valid_spreads,
                            mode='lines',
                            name='24M-1M Spread (Constant Maturity)',
                            line=dict(color='purple', width=2),
                            hovertemplate='Date: %{x}<br>24M-1M Spread: %{y:.3f}<br>Positive = Contango<extra></extra>'
                        ),
                        row=row, col=1
                    )

                    # Add zero line for reference
                    fig.add_hline(y=0, line_dash="solid", line_color="gray",
                                 annotation_text="Zero Spread", row=row, col=1)
                    return

            # Fallback: try 3M and 18M if 1M and 24M not available
            elif '3M' in cm_data.columns and '18M' in cm_data.columns:
                front_prices = cm_data['3M']
                back_prices = cm_data['18M']

                # Calculate spread
                spread_values = back_prices - front_prices

                # Remove NaN values
                valid_mask = ~(np.isnan(spread_values))
                if np.sum(valid_mask) > 0:
                    valid_dates = cm_data.index[valid_mask]
                    valid_spreads = spread_values[valid_mask]

                    # Plot constant maturity back/front spread
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=valid_spreads,
                            mode='lines',
                            name='18M-3M Spread (Constant Maturity)',
                            line=dict(color='purple', width=2),
                            hovertemplate='Date: %{x}<br>18M-3M Spread: %{y:.3f}<br>Positive = Contango<extra></extra>'
                        ),
                        row=row, col=1
                    )

                    # Add zero line for reference
                    fig.add_hline(y=0, line_dash="solid", line_color="gray",
                                 annotation_text="Zero Spread", row=row, col=1)
                    return

        if self.curves is not None:
            # Extract back/front spread using broadcasting
            spread_data: List[float] = []
            valid_dates: List[pd.Timestamp] = []
            back_dte_info: List[Optional[float]] = []

            for date, curve in self.curves.dropna().items():
                front_price = None
                back_price = None
                back_dte = None

                try:
                    front_price = curve['M0']
                except (KeyError, IndexError, AttributeError):
                    if hasattr(curve, 'seq_prices') and curve.seq_prices is not None and len(curve.seq_prices) > 0:
                        front_price = curve.seq_prices[0]
                    elif hasattr(curve, 'prices') and len(curve.prices) > 0:
                        front_price = curve.prices[0]

                # Target longer contracts (2 years) but ensure we maintain at least 360+ DTE
                back_price, back_dte = self._select_contract_by_dte(curve, target_days=730, tolerance=200)
                if back_price is None:
                    # Fallback: try for at least 1 year contract with wider tolerance
                    back_price, back_dte = self._select_contract_by_dte(curve, target_days=365, tolerance=None)
                if back_price is None:
                    # Final fallback: get the longest available contract (should be 360+ due to new logic)
                    back_price, back_dte = self._select_contract_by_dte(curve, target_days=730, tolerance=None)

                if back_price is None:
                    try:
                        back_price = curve[-1]
                    except (KeyError, IndexError, AttributeError):
                        back_price = None

                if back_price is None and hasattr(curve, 'prices') and len(curve.prices) > 0:
                    price_array = np.asarray(curve.prices, dtype=float)
                    valid_idx = np.where(~np.isnan(price_array))[0]
                    if len(valid_idx) > 0:
                        back_price = price_array[valid_idx[-1]]
                        dte_source = getattr(curve, 'days_to_expiry', None)
                        if dte_source is not None:
                            try:
                                dte_array = np.asarray(dte_source, dtype=float)
                                if len(dte_array) > valid_idx[-1] and not np.isnan(dte_array[valid_idx[-1]]):
                                    back_dte = float(dte_array[valid_idx[-1]])
                            except Exception:
                                back_dte = None

                try:
                    front_price = float(front_price)
                except (TypeError, ValueError):
                    front_price = None

                try:
                    back_price = float(back_price)
                except (TypeError, ValueError):
                    back_price = None

                if front_price is None or back_price is None:
                    continue

                if np.isnan(front_price) or np.isnan(back_price):
                    continue

                spread = back_price - front_price
                spread_data.append(spread)
                valid_dates.append(date)
                back_dte_info.append(back_dte)

            if len(spread_data) > 0:
                spread_state = ['Contango' if s > 0 else 'Backwardation' for s in spread_data]
                hover_text = []
                for state, dte in zip(spread_state, back_dte_info):
                    if dte is not None and not np.isnan(dte):
                        hover_text.append(f"State: {state}<br>Back DTE: {int(round(dte))} days")
                    else:
                        hover_text.append(f"State: {state}")

                # Plot back/front spread
                fig.add_trace(
                    go.Scatter(
                        x=valid_dates,
                        y=spread_data,
                        mode='lines',
                        name='Back/Front Spread',
                        line=dict(color='green', width=2),
                        hovertemplate='Date: %{x}<br>Spread: %{y:.2f}<br>%{text}<extra></extra>',
                        text=hover_text
                    ),
                    row=row, col=1
                )

                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                             annotation_text="Contango ↑ / Backwardation ↓", row=row, col=1)
                
                # Add moving average if enough data
                if len(spread_data) >= 20:
                    spread_series = pd.Series(spread_data, index=valid_dates)
                    ma_20 = spread_series.rolling(20).mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=ma_20.values,
                            mode='lines',
                            name='20-Day MA',
                            line=dict(color='red', width=1, dash='dot'),
                            hovertemplate='Date: %{x}<br>20-Day MA: %{y:.2f}<extra></extra>'
                        ),
                        row=row, col=1
                    )
                
                # Update y-axis label
                fig.update_yaxes(title_text="Spread (Back - Front)", row=row, col=1)
    
    def _add_seasonal_analysis_plot(self, fig, row, dates):
        """Add seasonal deviations plot - shows seasonal_deviations driver from curve evolution"""
        
        # Get seasonal deviations driver from path signatures
        if 'curve_drivers' in self.path_signatures:
            drivers = self.path_signatures['curve_drivers']
            
            if 'seasonal_deviations' in drivers and drivers['seasonal_deviations'].size > 0:
                seasonal_values = drivers['seasonal_deviations']
                valid_mask = ~np.isnan(seasonal_values)
                
                if np.sum(valid_mask) > 0:
                    # Ensure we align dates and seasonal values correctly
                    aligned_length = min(len(dates), len(seasonal_values))
                    aligned_dates = dates[:aligned_length]
                    aligned_seasonal_values = seasonal_values[:aligned_length]
                    aligned_valid_mask = valid_mask[:aligned_length]

                    valid_dates = aligned_dates[aligned_valid_mask]
                    valid_values = aligned_seasonal_values[aligned_valid_mask]
                    
                    # Plot seasonal deviations time series
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=valid_values,
                            mode='lines',
                            name='Seasonal Deviations',
                            line=dict(color='purple', width=2),
                            fill='tonexty',
                            fillcolor='rgba(128, 0, 128, 0.1)',
                            hovertemplate='Date: %{x}<br>Seasonal Deviation: %{y:.3f}<br>Higher = More Unusual<extra></extra>'
                        ),
                        row=row, col=1
                    )
                    
                    # Add threshold lines for significant deviations
                    significant_threshold = 2.0  # 2-sigma threshold
                    fig.add_hline(y=significant_threshold, line_dash="dash", line_color="red", 
                                 annotation_text="High Seasonal Deviation", row=row, col=1)
                    fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                                 annotation_text="Normal Seasonal Pattern", row=row, col=1)
                    
                    # Color regions with high seasonal deviation
                    high_deviation_mask = valid_values > significant_threshold
                    if np.any(high_deviation_mask):
                        high_deviation_dates = valid_dates[high_deviation_mask]
                        high_deviation_values = valid_values[high_deviation_mask]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=high_deviation_dates,
                                y=high_deviation_values,
                                mode='markers',
                                name='High Seasonal Deviation Events',
                                marker=dict(color='red', size=8, symbol='circle'),
                                hovertemplate='Date: %{x}<br>High Deviation: %{y:.3f}<br>Seasonal Disruption Event<extra></extra>'
                            ),
                            row=row, col=1
                        )
                else:
                    # No valid data - add placeholder
                    fig.add_trace(
                        go.Scatter(
                            x=[dates[0], dates[-1]] if len(dates) > 1 else [dates[0]],
                            y=[0, 0] if len(dates) > 1 else [0],
                            mode='lines',
                            name='No Seasonal Data',
                            line=dict(color='gray', dash='dot'),
                            hovertemplate='No seasonal deviation data available<extra></extra>'
                        ),
                        row=row, col=1
                    )
            else:
                # Seasonal deviations not available - add placeholder
                fig.add_trace(
                    go.Scatter(
                        x=[dates[0], dates[-1]] if len(dates) > 1 else [dates[0]],
                        y=[0, 0] if len(dates) > 1 else [0],
                        mode='lines',
                        name='Seasonal Analysis Unavailable',
                        line=dict(color='gray', dash='dot'),
                        hovertemplate='Seasonal analysis requires more data<extra></extra>'
                    ),
                    row=row, col=1
                )
        
        # Update y-axis label
        fig.update_yaxes(title_text="Seasonal Deviations (Z-Score)", row=row, col=1)
    
    def _add_3m_12m_spread_plot(self, fig, row, dates, constant_maturity=False):
        """Add 3m-12m spread plot as a separate bottom chart"""

        if constant_maturity and hasattr(self, 'constant_maturity_data') and self.constant_maturity_data is not None:
            # Use constant maturity data for 3m-12m spread calculation
            cm_data = self.constant_maturity_data

            # Try to get 3M and 12M from constant maturity data
            if '3M' in cm_data.columns and '12M' in cm_data.columns:
                spread_3m_12m = cm_data['12M'] - cm_data['3M']
                plot_dates = cm_data.index

                # Remove NaN values
                valid_mask = ~np.isnan(spread_3m_12m)
                if np.sum(valid_mask) > 0:
                    valid_dates = plot_dates[valid_mask]
                    valid_spreads = spread_3m_12m[valid_mask]

                    # Plot 3m-12m spread
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=valid_spreads,
                            mode='lines',
                            name='12M-3M Spread (Constant Maturity)',
                            line=dict(color='#1f77b4', width=2),
                            fill='tonexty',
                            fillcolor='rgba(31, 119, 180, 0.1)',
                            hovertemplate='Date: %{x}<br>12M-3M Spread: %{y:.3f}<br>Positive = Contango<extra></extra>'
                        ),
                        row=row, col=1
                    )

                    # Add zero line for reference
                    fig.add_hline(y=0, line_dash="solid", line_color="gray",
                                 annotation_text="Zero Spread", row=row, col=1)

                    # Add percentile lines for context
                    if len(valid_spreads) > 10:
                        spread_75th = np.percentile(valid_spreads, 75)
                        spread_25th = np.percentile(valid_spreads, 25)

                        fig.add_hline(y=spread_75th, line_dash="dash", line_color="green",
                                     annotation_text="75th Percentile", row=row, col=1)
                        fig.add_hline(y=spread_25th, line_dash="dash", line_color="red",
                                     annotation_text="25th Percentile", row=row, col=1)

                    # Update y-axis label
                    fig.update_yaxes(title_text="12M-3M Spread (Constant Maturity)", row=row, col=1)
                    return

        # Calculate 3m-12m spread using the existing method
        spread_3m_12m = self._calculate_3m_12m_spread_driver()
        
        if spread_3m_12m.size > 0:
            # Align dates with spread data
            n_spread_points = len(spread_3m_12m)
            plot_dates = dates[:n_spread_points] if len(dates) >= n_spread_points else dates
            
            # Only use valid (non-NaN) data points
            valid_mask = ~np.isnan(spread_3m_12m)
            
            if np.sum(valid_mask) > 0:
                valid_dates = plot_dates[valid_mask]
                valid_spreads = spread_3m_12m[valid_mask]
                
                # Plot 3m-12m spread
                fig.add_trace(
                    go.Scatter(
                        x=valid_dates,
                        y=valid_spreads,
                        mode='lines',
                        name='3m-12m Spread',
                        line=dict(color='#1f77b4', width=2),
                        fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.1)',
                        hovertemplate='Date: %{x}<br>3m-12m Spread: %{y:.3f}<br>Positive = Contango<extra></extra>'
                    ),
                    row=row, col=1
                )
                
                # Add zero line for reference
                fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                             annotation_text="Zero Spread", row=row, col=1)
                
                # Add percentile lines for context
                if len(valid_spreads) > 10:
                    spread_75th = np.percentile(valid_spreads, 75)
                    spread_25th = np.percentile(valid_spreads, 25)
                    
                    fig.add_hline(y=spread_75th, line_dash="dash", line_color="green", 
                                 annotation_text="75th Percentile", row=row, col=1)
                    fig.add_hline(y=spread_25th, line_dash="dash", line_color="red", 
                                 annotation_text="25th Percentile", row=row, col=1)
                
                # Highlight extreme spreads
                if len(valid_spreads) > 20:
                    spread_std = np.std(valid_spreads)
                    spread_mean = np.mean(valid_spreads)
                    extreme_threshold = 2 * spread_std
                    
                    extreme_mask = np.abs(valid_spreads - spread_mean) > extreme_threshold
                    if np.any(extreme_mask):
                        extreme_dates = valid_dates[extreme_mask]
                        extreme_spreads = valid_spreads[extreme_mask]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=extreme_dates,
                                y=extreme_spreads,
                                mode='markers',
                                name='Extreme Spread Events',
                                marker=dict(color='orange', size=8, symbol='diamond'),
                                hovertemplate='Date: %{x}<br>Extreme Spread: %{y:.3f}<br>2+ Std Dev Event<extra></extra>'
                            ),
                            row=row, col=1
                        )
            else:
                # No valid data - add placeholder
                fig.add_trace(
                    go.Scatter(
                        x=[dates[0], dates[-1]] if len(dates) > 1 else [dates[0]],
                        y=[0, 0] if len(dates) > 1 else [0],
                        mode='lines',
                        name='No 3m-12m Spread Data',
                        line=dict(color='gray', dash='dot'),
                        hovertemplate='No 3m-12m spread data available<extra></extra>'
                    ),
                    row=row, col=1
                )
        else:
            # No spread data available - add placeholder
            fig.add_trace(
                go.Scatter(
                    x=[dates[0], dates[-1]] if len(dates) > 1 else [dates[0]],
                    y=[0, 0] if len(dates) > 1 else [0],
                    mode='lines',
                    name='3m-12m Spread Unavailable',
                    line=dict(color='gray', dash='dot'),
                    hovertemplate='3m-12m spread requires sufficient contract data<extra></extra>'
                ),
                row=row, col=1
            )
        
        # Update y-axis label
        fig.update_yaxes(title_text="3m-12m Spread (Price Difference)", row=row, col=1)
    
    def analyze_levy_curve_returns_relationship(self, 
                                              window: int = None,
                                              return_window: int = 1) -> Dict[str, Any]:
        """
        Analyze relationship between curve Lévy areas and F0 returns
        
        Parameters:
        -----------
        window : int, optional
            Window for Lévy area calculation
        return_window : int
            Window for return calculation (1 = daily returns)
            
        Returns:
        --------
        Dict containing analysis results with correlations, regressions, and predictive power
        """
        if window is None:
            window = self.default_window
        
        if self.curves is None:
            raise ValueError("No curve data available")
        
        # Calculate path signatures if not done
        if self.path_signatures is None:
            self.calculate_path_signatures(window=window)
        
        # Extract F0 prices for return calculation
        f0_prices = []
        dates = []
        
        for date, curve in self.curves.items():
            if curve is not None and hasattr(curve, 'prices') and curve.prices is not None:
                if len(curve.prices) > 0 and not np.isnan(curve.prices[0]):
                    f0_prices.append(curve.prices[0])
                    dates.append(date)
        
        if len(f0_prices) < window + return_window:
            raise ValueError(f"Insufficient data: need at least {window + return_window} points")
        
        f0_prices = np.array(f0_prices)
        
        # Calculate F0 returns
        if return_window == 1:
            f0_returns = np.diff(np.log(f0_prices))
        else:
            f0_returns = np.log(f0_prices[return_window:]) - np.log(f0_prices[:-return_window])
        
        # Get Lévy areas from path signatures
        log_levy_areas = self.path_signatures['log_levy_areas']
        curve_drivers = self.path_signatures.get('curve_drivers', {})
        
        if log_levy_areas.size == 0:
            return {'error': 'No Lévy area data available'}
        
        # Calculate mean Lévy area across contract pairs
        mean_levy = np.nanmean(log_levy_areas, axis=1)
        
        # Align data lengths for analysis
        min_len = min(len(f0_returns), len(mean_levy) - 1)  # -1 for returns shift
        
        if min_len < 10:
            return {'error': f'Insufficient aligned data: {min_len} points'}
        
        returns_aligned = f0_returns[:min_len]
        levy_aligned = mean_levy[:-1][:min_len]  # Lag Lévy areas by 1 period
        
        # Remove any remaining NaNs
        valid_mask = ~(np.isnan(returns_aligned) | np.isnan(levy_aligned))
        returns_clean = returns_aligned[valid_mask]
        levy_clean = levy_aligned[valid_mask]
        
        if len(returns_clean) < 5:
            return {'error': 'Too few valid data points after cleaning'}
        
        # Core analysis
        correlation = np.corrcoef(levy_clean, returns_clean)[0, 1]
        
        # Simple linear regression: returns ~ levy_areas
        X = np.column_stack([np.ones(len(levy_clean)), levy_clean])
        try:
            coeffs = np.linalg.lstsq(X, returns_clean, rcond=None)[0]
            intercept, slope = coeffs[0], coeffs[1]
            
            # R-squared
            y_pred = intercept + slope * levy_clean
            ss_res = np.sum((returns_clean - y_pred) ** 2)
            ss_tot = np.sum((returns_clean - np.mean(returns_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
        except np.linalg.LinAlgError:
            slope, intercept, r_squared = np.nan, np.nan, np.nan
        
        # Analyze individual drivers if available
        driver_correlations = {}
        if curve_drivers:
            for driver_name, driver_values in curve_drivers.items():
                if len(driver_values) >= min_len and not np.all(np.isnan(driver_values)):
                    driver_aligned = driver_values[:-1][:min_len][valid_mask]
                    if len(driver_aligned) == len(returns_clean):
                        driver_corr = np.corrcoef(driver_aligned, returns_clean)[0, 1]
                        if not np.isnan(driver_corr):
                            driver_correlations[driver_name] = driver_corr
        
        # Directional accuracy (sign prediction)
        levy_signs = np.sign(levy_clean)
        return_signs = np.sign(returns_clean)
        directional_accuracy = np.mean(levy_signs == return_signs)
        
        return {
            'correlation': correlation,
            'regression': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared
            },
            'directional_accuracy': directional_accuracy,
            'driver_correlations': driver_correlations,
            'data_summary': {
                'n_observations': len(returns_clean),
                'return_std': np.std(returns_clean),
                'levy_std': np.std(levy_clean),
                'return_mean': np.mean(returns_clean),
                'levy_mean': np.mean(levy_clean)
            },
            'strongest_driver': max(driver_correlations.items(), key=lambda x: abs(x[1])) if driver_correlations else None
        }
    
    def plot_regime_changes_f0(self,
                              regime_window: int = 63,
                              regime_threshold: float = 2.0,
                              min_regime_length: int = 10,
                              show_seasonal: bool = True,
                              constant_maturity: bool = False,
                              height: int = 800,
                              title: Optional[str] = None) -> go.Figure:
        """
        Plot regime changes along the F0 (front month) contract prices

        Parameters:
        -----------
        regime_window : int
            Window for regime detection calculations
        regime_threshold : float
            Threshold for regime change sensitivity
        min_regime_length : int
            Minimum length of a regime to avoid noise
        show_seasonal : bool
            Whether to overlay seasonal patterns
        constant_maturity : bool, default False
            Use constant maturity tenor (1-month) instead of sequential F0 contract
        height : int
            Plot height in pixels
        title : str, optional
            Custom plot title

        Returns:
        --------
        plotly.graph_objects.Figure
        """

        if constant_maturity and self.constant_maturity_data is not None:
            # Use 1-month constant maturity tenor
            if '1M' in self.constant_maturity_data.columns:
                f0_series = self.constant_maturity_data['1M'].dropna()
            elif 0.083333 in self.constant_maturity_data.columns:  # 1/12 ≈ 1 month
                f0_series = self.constant_maturity_data[0.083333].dropna()
            else:
                # Fall back to shortest tenor available
                f0_series = self.constant_maturity_data.iloc[:, 0].dropna()
        else:
            # Use sequential F0 contract
            if self.curves is None:
                raise ValueError("No curve data available for regime analysis")
            f0_series = self._extract_front_month_broadcast()
        
        if len(f0_series) < regime_window * 2:
            raise ValueError(f"Insufficient data for regime analysis. Need at least {regime_window * 2} points")
        
        # Detect regime changes (simplified version)
        regime_data = self._detect_regime_changes_simple(f0_series, regime_window, regime_threshold)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=['F0 Price with Regime Changes', 'Regime Intensity'],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Main price chart
        fig.add_trace(
            go.Scatter(
                x=regime_data.index,
                y=regime_data['price'],
                mode='lines',
                name='F0 Price',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Add regime change markers
        regime_points = regime_data[regime_data['regime_change'] > 0]
        if not regime_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=regime_points.index,
                    y=regime_points['price'],
                    mode='markers',
                    name='Regime Changes',
                    marker=dict(color='red', size=8, symbol='diamond')
                ),
                row=1, col=1
            )
        
        # Regime intensity
        fig.add_trace(
            go.Scatter(
                x=regime_data.index,
                y=regime_data['regime_change'],
                mode='lines',
                name='Regime Intensity',
                line=dict(color='purple'),
                fill='tonexty',
                fillcolor='rgba(128, 0, 128, 0.2)'
            ),
            row=2, col=1
        )
        
        # Update layout
        symbol_name = self.symbol or 'Unknown'
        default_title = f'Regime Analysis: {symbol_name} F0 Contract'
        
        fig.update_layout(
            title=title or default_title,
            height=height,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def _detect_regime_changes_simple(self, price_series: pd.Series, window: int, threshold: float) -> pd.DataFrame:
        """Simplified regime change detection for F0 prices"""
        
        # Calculate rolling statistics
        rolling_mean = price_series.rolling(window, min_periods=window//2).mean()
        rolling_std = price_series.rolling(window, min_periods=window//2).std()
        
        # Z-score for outlier detection
        z_score = (price_series - rolling_mean) / rolling_std
        
        # Simple regime change detection
        regime_changes = (np.abs(z_score) > threshold).astype(float)
        
        # Create result DataFrame
        regime_data = pd.DataFrame({
            'price': price_series,
            'rolling_mean': rolling_mean,
            'z_score': z_score,
            'regime_change': regime_changes
        })
        
        return regime_data

    def plot_contango_backwardation_analysis(self, 
                                           show_seasonal: bool = True,
                                           show_flips: bool = True,
                                           height: int = 1000,
                                           title: Optional[str] = None) -> go.Figure:
        """
        Plot comprehensive contango/backwardation analysis with seasonal patterns
        
        Parameters:
        -----------
        show_seasonal : bool
            Show seasonal baseline and deviations
        show_flips : bool
            Mark significant curve structure flips
        height : int
            Plot height in pixels
        title : str, optional
            Custom plot title
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        # Get comprehensive analysis
        analysis = self.get_contango_backwardation_analysis(seasonal_adjust=show_seasonal)
        
        if not analysis or 'curve_slopes' not in analysis:
            raise ValueError("No curve slope data available for analysis")
        
        slopes = analysis['curve_slopes']
        flip_events = analysis.get('flip_events', pd.DataFrame())
        
        # Determine number of subplots
        n_plots = 2  # Always show slopes and monthly patterns
        
        subplot_titles = ['Curve Structure (Contango/Backwardation)', 'Monthly Seasonal Patterns']
        
        fig = make_subplots(
            rows=n_plots, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        current_row = 1
        
        # 1. Main curve structure plot
        fig.add_trace(
            go.Scatter(
                x=slopes.index,
                y=slopes.values,
                mode='lines',
                name='Curve Slope',
                line=dict(color='blue', width=1),
                hovertemplate='Date: %{x}<br>Slope: %{y:.4f}<br>State: %{text}<extra></extra>',
                text=['Contango' if s > 0 else 'Backwardation' for s in slopes.values]
            ),
            row=current_row, col=1
        )
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Contango ↑ / Backwardation ↓", row=current_row, col=1)
        
        # Mark flip events if requested
        if show_flips and not flip_events.empty:
            for _, flip in flip_events.iterrows():
                color = 'red' if flip['flip_type'] == 'contango_to_backwardation' else 'green'
                fig.add_vrect(
                    x0=flip['flip_start'], x1=flip['flip_end'],
                    fillcolor=color, opacity=0.2,
                    layer="below", line_width=0,
                    row=current_row, col=1
                )
                
                # Add flip markers
                fig.add_trace(
                    go.Scatter(
                        x=[flip['flip_start']],
                        y=[slopes.loc[flip['flip_start']] if flip['flip_start'] in slopes.index else 0],
                        mode='markers',
                        marker=dict(color=color, size=10, symbol='diamond'),
                        name=flip['flip_type'].replace('_', ' ').title(),
                        showlegend=True,
                        hovertemplate=f"Flip Start: {flip['flip_start']}<br>Type: {flip['flip_type']}<br>Duration: {flip['flip_duration']} days<extra></extra>"
                    ),
                    row=current_row, col=1
                )
        
        current_row += 1
        
        # 2. Monthly seasonal patterns
        if 'seasonal_analysis' in analysis:
            monthly_tendencies = analysis['seasonal_analysis']['monthly_tendencies']
            months = list(monthly_tendencies.keys())
            tendencies = list(monthly_tendencies.values())
            
            colors = ['red' if t < 0.5 else 'green' for t in tendencies]
            
            fig.add_trace(
                go.Bar(
                    x=months,
                    y=[(t - 0.5) * 100 for t in tendencies],  # Center around 0
                    marker_color=colors,
                    name='Monthly Tendency',
                    hovertemplate='Month: %{x}<br>Contango Tendency: %{text}%<extra></extra>',
                    text=[f"{t*100:.1f}" for t in tendencies]
                ),
                row=current_row, col=1
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="50% Contango Line", row=current_row, col=1)
            
            current_row += 1
            
        
        # Update layout
        symbol_name = self.symbol if self.symbol else "Futures"
        current_state = analysis['current_state'].title()
        contango_pct = analysis['slope_statistics']['contango_percentage']
        
        default_title = f'{symbol_name} Contango/Backwardation Analysis<br><sub>Current: {current_state} | Historical: {contango_pct:.1f}% Contango</sub>'
        
        fig.update_layout(
            title=title or default_title,
            height=height,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Customize y-axis labels
        fig.update_yaxes(title_text="Curve Slope", row=1, col=1)
        if 'seasonal_analysis' in analysis:
            fig.update_yaxes(title_text="Tendency (%)", row=2, col=1)
        
        return fig
    
    def create_3d_surface_plot(self,
                              data_type: str = 'spread',
                              title: Optional[str] = None,
                              colorscale: str = 'RdYlBu',
                              height: int = 800,
                              constant_maturity: bool = False) -> go.Figure:
        """
        Create 3D surface plot with date (x), days-to-expiry/tenor (y), and spread/price (z)

        Parameters:
        -----------
        data_type : str
            Type of data to plot: 'spread', 'price', or 'log_price'
        title : str, optional
            Custom plot title
        colorscale : str
            Plotly colorscale name
        height : int
            Plot height in pixels
        constant_maturity : bool, default False
            Use constant maturity data instead of sequential contracts

        Returns:
        --------
        plotly.graph_objects.Figure
            3D surface plot
        """
        
        if self.curves is None or len(self.curves) == 0:
            raise ValueError("No curve data available for 3D surface plot")

        # Setup constant maturity if requested
        if constant_maturity:
            if self.constant_maturity_data is None:
                success = self.setup_constant_maturity()
                if not success:
                    raise ValueError("Constant maturity setup failed and no existing data available")
            return self._create_constant_maturity_surface_plot(data_type, title, colorscale, height)

        # Extract data matrices for sequential contracts
        dates = []
        dte_matrix = []
        data_matrix = []
        
        # Determine maximum number of contracts for consistent dimensions
        max_contracts = 0
        valid_curves = []
        
        for date, curve in self.curves.items():
            if curve is not None:
                # Check if we have the required data
                has_dte = hasattr(curve, 'days_to_expiry') and curve.days_to_expiry is not None
                has_seq_dte = hasattr(curve, 'seq_dte') and curve.seq_dte is not None
                has_prices = (hasattr(curve, 'seq_prices') and curve.seq_prices is not None) or \
                           (hasattr(curve, 'prices') and curve.prices is not None)
                
                if has_prices and (has_dte or has_seq_dte):
                    valid_curves.append((date, curve))
                    
                    # Determine contract count for dimension consistency
                    if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                        n_contracts = len(curve.seq_prices)
                    elif hasattr(curve, 'prices') and curve.prices is not None:
                        n_contracts = len(curve.prices)
                    else:
                        continue
                    
                    max_contracts = max(max_contracts, n_contracts)
        
        if not valid_curves:
            raise ValueError("No valid curves with both price and DTE data found")
        
        if max_contracts == 0:
            raise ValueError("No valid contract data found")
        
        # Initialize matrices
        n_dates = len(valid_curves)
        data_matrix = np.full((n_dates, max_contracts), np.nan)
        dte_matrix = np.full((n_dates, max_contracts), np.nan)
        
        # Fill matrices
        for i, (date, curve) in enumerate(valid_curves):
            dates.append(date)
            
            # Extract price/spread data
            if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                prices = np.array(curve.seq_prices)
            elif hasattr(curve, 'prices') and curve.prices is not None:
                prices = np.array(curve.prices)
            else:
                continue
            
            # Extract DTE data
            if hasattr(curve, 'seq_dte') and curve.seq_dte is not None:
                dte_data = np.array(curve.seq_dte)
            elif hasattr(curve, 'days_to_expiry') and curve.days_to_expiry is not None:
                dte_data = np.array(curve.days_to_expiry)
            else:
                # Fallback: create estimated DTE based on contract position
                dte_data = np.arange(30, 30 + len(prices) * 30, 30)  # 30-day increments
            
            # Ensure consistent dimensions
            n_current = min(len(prices), len(dte_data), max_contracts)
            
            if data_type == 'spread':
                # Calculate spreads between consecutive contracts
                if n_current > 1:
                    spreads = np.diff(prices[:n_current])
                    data_matrix[i, :len(spreads)] = spreads
                    dte_matrix[i, :len(spreads)] = dte_data[:len(spreads)]
            elif data_type == 'price':
                data_matrix[i, :n_current] = prices[:n_current]
                dte_matrix[i, :n_current] = dte_data[:n_current]
            elif data_type == 'log_price':
                valid_prices = prices[:n_current] > 0
                if np.any(valid_prices):
                    log_prices = np.full(n_current, np.nan)
                    log_prices[valid_prices] = np.log(prices[:n_current][valid_prices])
                    data_matrix[i, :n_current] = log_prices
                    dte_matrix[i, :n_current] = dte_data[:n_current]
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")

        # Create date arrays for surface plot
        dates_array = np.array(dates)

        # Convert to numeric timestamps (handle both pandas Timestamp and numpy datetime64)
        dates_numeric = []
        for d in dates_array:
            if hasattr(d, 'timestamp'):
                # pandas Timestamp
                dates_numeric.append(d.timestamp())
            else:
                # numpy datetime64 - convert to pandas first
                pd_date = pd.Timestamp(d)
                dates_numeric.append(pd_date.timestamp())
        dates_numeric = np.array(dates_numeric)

        # Create meshgrid for surface
        # For surface plots, we need consistent DTE values across all dates
        # Use the median DTE values as the Y-axis
        median_dte = np.nanmedian(dte_matrix, axis=0)
        valid_dte_mask = ~np.isnan(median_dte)
        
        if not np.any(valid_dte_mask):
            raise ValueError("No valid DTE data found for surface plot")
        
        # Filter to valid contracts only
        valid_dte = median_dte[valid_dte_mask]
        valid_data = data_matrix[:, valid_dte_mask]
        
        # Create meshgrid
        X, Y = np.meshgrid(dates_numeric, valid_dte, indexing='ij')
        Z = valid_data
        
        # Create 3D surface plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
                name=f'{data_type.title()} Surface',
                hovertemplate='Date: %{x|%Y-%m-%d}<br>DTE: %{y:.0f} days<br>' + 
                             f'{data_type.title()}: %{{z:.3f}}<extra></extra>',
                showscale=True,
                opacity=0.8
            )
        )
        
        # Add contour projection on Z plane
        fig.add_trace(
            go.Contour(
                x=dates_numeric,
                y=valid_dte,
                z=Z.T,  # Transpose for contour
                colorscale=colorscale,
                showscale=False,
                opacity=0.3,
                contours=dict(
                    showlines=True,
                    coloring='lines'
                ),
                name='Contour Projection',
                hovertemplate='Date: %{x|%Y-%m-%d}<br>DTE: %{y:.0f} days<br>' +
                             f'{data_type.title()}: %{{z:.3f}}<extra></extra>'
            )
        )
        
        # Configure layout
        data_label = {
            'spread': 'Spread',
            'price': 'Price',
            'log_price': 'Log Price'
        }.get(data_type, data_type.title())
        
        symbol_name = self.symbol if self.symbol else "Futures"
        default_title = f'{symbol_name} {data_label} Surface: Date vs Days-to-Expiry'
        
        fig.update_layout(
            title=title or default_title,
            height=height,
            scene=dict(
                xaxis_title='Date',
                yaxis_title='Days to Expiry',
                zaxis_title=data_label,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='cube'
            ),
            showlegend=True
        )

        # Convert X axis labels to dates for better readability
        step = max(1, len(dates_array)//10)
        sample_dates = dates_array[::step]

        # Handle both pandas Timestamp and numpy datetime64 for strftime
        date_labels = []
        for d in sample_dates:
            if hasattr(d, 'strftime'):
                # pandas Timestamp
                date_labels.append(d.strftime('%Y-%m-%d'))
            else:
                # numpy datetime64 - convert to pandas first
                pd_date = pd.Timestamp(d)
                date_labels.append(pd_date.strftime('%Y-%m-%d'))

        fig.update_scenes(
            xaxis=dict(
                tickmode='array',
                tickvals=dates_numeric[::step],  # Show ~10 tick marks
                ticktext=date_labels
            )
        )

        return fig

    def _create_constant_maturity_surface_plot(self,
                                              data_type: str,
                                              title: Optional[str],
                                              colorscale: str,
                                              height: int) -> go.Figure:
        """
        Create 3D surface plot using constant maturity data

        Parameters:
        -----------
        data_type : str
            Type of data to plot: 'spread', 'price', or 'log_price'
        title : str, optional
            Custom plot title
        colorscale : str
            Plotly colorscale name
        height : int
            Plot height in pixels

        Returns:
        --------
        plotly.graph_objects.Figure
            3D surface plot with constant maturity data
        """

        if self.constant_maturity_data is None:
            raise ValueError("Constant maturity data not available")

        cm_data = self.constant_maturity_data
        dates = cm_data.index
        tenor_columns = cm_data.columns

        # Convert tenor labels to numeric values (years)
        tenor_values = []
        for col in tenor_columns:
            if isinstance(col, str):
                if col.endswith('M'):
                    # Monthly tenors (e.g., '1M', '3M', '12M')
                    months = float(col[:-1])
                    tenor_values.append(months / 12.0)  # Convert to years
                elif col.endswith('Y'):
                    # Yearly tenors (e.g., '1Y', '2Y')
                    years = float(col[:-1])
                    tenor_values.append(years)
                else:
                    try:
                        # Numeric tenor (assume years)
                        tenor_values.append(float(col))
                    except:
                        tenor_values.append(len(tenor_values) * 0.25)  # 3-month increments as fallback
            else:
                # Numeric tenor (assume years)
                tenor_values.append(float(col))

        tenor_values = np.array(tenor_values)

        # Prepare data matrices
        if data_type == 'spread':
            # Calculate spreads vs front month (shortest tenor)
            if len(tenor_columns) < 2:
                raise ValueError("Need at least 2 tenors to calculate spreads")

            # Get front month data (shortest tenor)
            front_month_data = cm_data.iloc[:, 0].values  # First column (shortest tenor)

            # Calculate spreads vs front month for all other tenors
            spread_data = cm_data.iloc[:, 1:].values - front_month_data[:, np.newaxis]
            spread_tenors = tenor_values[1:]  # Exclude front month tenor
            tenor_days = spread_tenors * 365  # Convert to days for consistency

            Z = spread_data
            Y_values = tenor_days

        elif data_type == 'price':
            Z = cm_data.values
            Y_values = tenor_values * 365  # Convert years to days

        elif data_type == 'log_price':
            # Handle potential negative or zero values
            positive_mask = cm_data.values > 0
            log_data = np.full_like(cm_data.values, np.nan)
            log_data[positive_mask] = np.log(cm_data.values[positive_mask])

            Z = log_data
            Y_values = tenor_values * 365  # Convert years to days

        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        # Create date arrays for surface plot
        dates_array = np.array(dates)

        # Convert to numeric timestamps (handle both pandas Timestamp and numpy datetime64)
        dates_numeric = []
        for d in dates_array:
            if hasattr(d, 'timestamp'):
                # pandas Timestamp
                dates_numeric.append(d.timestamp())
            else:
                # numpy datetime64 - convert to pandas first
                pd_date = pd.Timestamp(d)
                dates_numeric.append(pd_date.timestamp())
        dates_numeric = np.array(dates_numeric)

        # Create meshgrid
        X, Y = np.meshgrid(dates_numeric, Y_values, indexing='ij')

        # Create surface plot
        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
                name=f'{data_type.title()} Surface',
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Tenor: %{y:.0f} days<br>' +
                             f'{data_type.title()}: %{{z:.3f}}<extra></extra>',
                showscale=True,
                opacity=0.8
            )
        )

        # Add contour projection
        fig.add_trace(
            go.Contour(
                x=dates_numeric,
                y=Y_values,
                z=Z.T,  # Transpose for contour
                colorscale=colorscale,
                showscale=False,
                opacity=0.3,
                contours=dict(
                    showlines=True,
                    coloring='lines'
                ),
                name='Contour Projection',
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Tenor: %{y:.0f} days<br>' +
                             f'{data_type.title()}: %{{z:.3f}}<extra></extra>'
            )
        )

        # Configure layout
        data_label = {
            'spread': 'Spread',
            'price': 'Price',
            'log_price': 'Log Price'
        }[data_type]

        plot_title = title or f'3D {data_label} Surface (Constant Maturity) - {getattr(self, "symbol", "Unknown")}'

        fig.update_layout(
            title=plot_title,
            height=height,
            scene=dict(
                xaxis_title='Date',
                yaxis_title='Tenor (Days)',
                zaxis_title=data_label,
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Convert X axis labels to dates for better readability
        step = max(1, len(dates_array)//10)
        sample_dates = dates_array[::step]

        # Handle both pandas Timestamp and numpy datetime64 for strftime
        date_labels = []
        for d in sample_dates:
            if hasattr(d, 'strftime'):
                # pandas Timestamp
                date_labels.append(d.strftime('%Y-%m-%d'))
            else:
                # numpy datetime64 - convert to pandas first
                pd_date = pd.Timestamp(d)
                date_labels.append(pd_date.strftime('%Y-%m-%d'))

        fig.update_scenes(
            xaxis=dict(
                tickmode='array',
                tickvals=dates_numeric[::step],  # Show ~10 tick marks
                ticktext=date_labels
            )
        )

        return fig

    def get_curve_slopes_broadcast(self) -> pd.DataFrame:
        """
        Broadcasted method to collect slopes from all FuturesCurves in the array
        
        Efficiently extracts curve slopes using vectorized operations to analyze
        the term structure slope (contango/backwardation) across all dates.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with datetime index and slope metrics:
            - overall_slope: Linear slope across all contracts
            - front_slope: Slope of front portion of curve (first 3 contracts)
            - back_slope: Slope of back portion of curve (last 3 contracts)
            - slope_curvature: Second derivative (change in slope)
            - slope_std: Standard deviation of contract-to-contract slopes
            - contango_ratio: Fraction of positive spreads (contango)
            - max_slope_segment: Maximum slope between any two consecutive contracts
            - min_slope_segment: Minimum slope between any two consecutive contracts
        """
        
        if self.curves is None or len(self.curves) == 0:
            raise ValueError("No curve data available for slope analysis")
        
        # Initialize results dictionary
        slope_metrics = {
            'overall_slope': [],
            'front_slope': [],
            'back_slope': [],
            'slope_curvature': [],
            'slope_std': [],
            'contango_ratio': [],
            'max_slope_segment': [],
            'min_slope_segment': [],
            'price_range': [],
            'curve_convexity': []
        }
        
        valid_dates = []
        
        # Process each curve
        for date, curve in self.curves.items():
            if curve is None:
                continue
            
            # Extract prices (prefer seq_prices for consistency)
            if hasattr(curve, 'seq_prices') and curve.seq_prices is not None:
                prices = np.array(curve.seq_prices)
            elif hasattr(curve, 'prices') and curve.prices is not None:
                prices = np.array(curve.prices)
            else:
                continue
            
            # Filter out NaN values and ensure we have enough data
            valid_mask = ~np.isnan(prices)
            if np.sum(valid_mask) < 2:
                continue
            
            valid_prices = prices[valid_mask]
            n_contracts = len(valid_prices)
            
            if n_contracts < 2:
                continue
            
            valid_dates.append(date)
            
            # 1. Overall slope (linear fit across all contracts)
            x_indices = np.arange(n_contracts)
            if n_contracts >= 2:
                overall_slope = np.polyfit(x_indices, valid_prices, 1)[0]
            else:
                overall_slope = np.nan
            slope_metrics['overall_slope'].append(overall_slope)
            
            # 2. Front slope (first 3 contracts or available)
            front_contracts = min(3, n_contracts)
            if front_contracts >= 2:
                front_x = np.arange(front_contracts)
                front_slope = np.polyfit(front_x, valid_prices[:front_contracts], 1)[0]
            else:
                front_slope = np.nan
            slope_metrics['front_slope'].append(front_slope)
            
            # 3. Back slope (last 3 contracts or available)
            back_contracts = min(3, n_contracts)
            if back_contracts >= 2 and n_contracts >= back_contracts:
                back_x = np.arange(back_contracts)
                back_prices = valid_prices[-back_contracts:]
                back_slope = np.polyfit(back_x, back_prices, 1)[0]
            else:
                back_slope = np.nan
            slope_metrics['back_slope'].append(back_slope)
            
            # 4. Slope curvature (second derivative approximation)
            if n_contracts >= 3:
                # Calculate first differences (slopes between adjacent contracts)
                first_diffs = np.diff(valid_prices)
                # Calculate second differences (change in slope)
                second_diffs = np.diff(first_diffs)
                slope_curvature = np.mean(second_diffs)
            else:
                slope_curvature = np.nan
            slope_metrics['slope_curvature'].append(slope_curvature)
            
            # 5. Slope standard deviation (variability in local slopes)
            if n_contracts >= 3:
                local_slopes = np.diff(valid_prices)
                slope_std = np.std(local_slopes)
            else:
                slope_std = np.nan
            slope_metrics['slope_std'].append(slope_std)
            
            # 6. Contango ratio (fraction of positive spreads)
            if n_contracts >= 2:
                spreads = np.diff(valid_prices)
                contango_ratio = np.sum(spreads > 0) / len(spreads)
            else:
                contango_ratio = np.nan
            slope_metrics['contango_ratio'].append(contango_ratio)
            
            # 7. Max and min slope segments
            if n_contracts >= 2:
                segment_slopes = np.diff(valid_prices)
                max_slope_segment = np.max(segment_slopes)
                min_slope_segment = np.min(segment_slopes)
            else:
                max_slope_segment = np.nan
                min_slope_segment = np.nan
            slope_metrics['max_slope_segment'].append(max_slope_segment)
            slope_metrics['min_slope_segment'].append(min_slope_segment)
            
            # 8. Price range (additional metric)
            price_range = np.max(valid_prices) - np.min(valid_prices)
            slope_metrics['price_range'].append(price_range)
            
            # 9. Curve convexity (measure of curve bending)
            if n_contracts >= 4:
                # Fit quadratic and extract curvature coefficient
                try:
                    quad_coeffs = np.polyfit(x_indices, valid_prices, 2)
                    curve_convexity = quad_coeffs[0]  # a in ax^2 + bx + c
                except np.linalg.LinAlgError:
                    curve_convexity = np.nan
            else:
                curve_convexity = np.nan
            slope_metrics['curve_convexity'].append(curve_convexity)
        
        if not valid_dates:
            raise ValueError("No valid curves found for slope analysis")
        
        # Create DataFrame with results
        slopes_df = pd.DataFrame(slope_metrics, index=pd.DatetimeIndex(valid_dates))
        
        # Add additional computed metrics
        slopes_df['front_back_slope_diff'] = slopes_df['back_slope'] - slopes_df['front_slope']
        slopes_df['slope_trend'] = slopes_df['overall_slope'].rolling(window=5, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
        )
        
        # Classify curve shape
        def classify_curve_shape(row):
            overall = row['overall_slope']
            curvature = row['slope_curvature']
            convexity = row['curve_convexity']
            
            if pd.isna(overall):
                return 'unknown'
            elif overall > 0.01:
                if pd.notna(curvature) and curvature > 0:
                    return 'steep_contango'
                else:
                    return 'contango'
            elif overall < -0.01:
                if pd.notna(curvature) and curvature < 0:
                    return 'steep_backwardation'
                else:
                    return 'backwardation'
            else:
                return 'flat'
        
        slopes_df['curve_shape'] = slopes_df.apply(classify_curve_shape, axis=1)
        
        # Sort by date for consistency
        slopes_df = slopes_df.sort_index()
        
        return slopes_df

    # ==================================================================================
    # SEASONAL ANALYSIS METHODS (Migrated from SpreadAnalyzer)
    # ==================================================================================

    def calculate_seasonal_statistics_advanced(self,
                                     data_type: str = 'spreads',
                                     groupby: str = 'month',
                                     rolling_window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate seasonal statistics with optional rolling window
        (Migrated from SpreadAnalyzer for comprehensive seasonal analysis)

        Parameters:
        - data_type: 'spreads', 'returns', 'prices', or 'levy_areas'
        - groupby: 'month', 'week_of_year', 'week_of_month', 'day_of_week'
        - rolling_window: If specified, calculate rolling seasonal stats
        """
        # Use direct DataFrame access (simplified from original seq_data patterns)
        if data_type == 'spreads':
            if hasattr(self.spread_data, 'seq_spreads') and self.spread_data.seq_spreads is not None:
                data = self.spread_data.seq_spreads
            else:
                return pd.DataFrame()
        elif data_type == 'returns':
            if hasattr(self.spread_data, 'seq_prices') and self.spread_data.seq_prices is not None:
                data = self.spread_data.seq_prices.pct_change()
            else:
                return pd.DataFrame()
        elif data_type == 'prices':
            if hasattr(self.spread_data, 'seq_prices') and self.spread_data.seq_prices is not None:
                data = self.spread_data.seq_prices
            else:
                return pd.DataFrame()
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        if data.empty:
            return pd.DataFrame()

        # Add time grouping columns
        data = data.copy()
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['week_of_year'] = data.index.isocalendar().week
        data['day_of_week'] = data.index.dayofweek
        data['week_of_month'] = (data.index.day - 1) // 7 + 1

        # Select grouping
        group_cols = {
            'month': ['month'],
            'week_of_year': ['week_of_year'],
            'week_of_month': ['month', 'week_of_month'],
            'day_of_week': ['day_of_week']
        }

        if groupby not in group_cols:
            raise ValueError(f"Unknown groupby: {groupby}")

        group_by_cols = group_cols[groupby]

        # Calculate statistics
        if rolling_window is None:
            stats = self._calculate_static_seasonal_stats_advanced(data, group_by_cols)
        else:
            stats = self._calculate_rolling_seasonal_stats_advanced(data, group_by_cols, rolling_window)

        return stats

    def _calculate_static_seasonal_stats_advanced(self, data: pd.DataFrame, group_by_cols: List[str]) -> pd.DataFrame:
        """Calculate static seasonal statistics"""
        value_cols = [col for col in data.columns
                     if col not in ['year', 'month', 'week_of_year', 'day_of_week', 'week_of_month']]

        stats_list = []

        for col in value_cols:
            if data[col].notna().sum() < 10:
                continue

            grouped = data.groupby(group_by_cols)[col]

            stats_df = pd.DataFrame({
                f'{col}_mean': grouped.mean(),
                f'{col}_median': grouped.median(),
                f'{col}_high': grouped.max(),
                f'{col}_low': grouped.min(),
                f'{col}_std': grouped.std(),
                f'{col}_25pct': grouped.quantile(0.25),
                f'{col}_75pct': grouped.quantile(0.75),
                f'{col}_count': grouped.count()
            })

            stats_list.append(stats_df)

        if stats_list:
            return pd.concat(stats_list, axis=1)
        else:
            return pd.DataFrame()

    def _calculate_rolling_seasonal_stats_advanced(self, data: pd.DataFrame,
                                         group_by_cols: List[str],
                                         window: int) -> pd.DataFrame:
        """Calculate rolling seasonal statistics"""
        rolling_stats = pd.DataFrame(index=data.index)

        value_cols = [col for col in data.columns
                     if col not in ['year', 'month', 'week_of_year', 'day_of_week', 'week_of_month']]

        for col in value_cols:
            if data[col].notna().sum() < window:
                continue

            for idx in data.index[window:]:
                hist_data = data.loc[:idx].tail(window * 10)

                current_groups = {g: data.loc[idx, g] for g in group_by_cols}

                mask = pd.Series([True] * len(hist_data), index=hist_data.index)
                for g, v in current_groups.items():
                    mask &= (hist_data[g] == v)

                seasonal_data = hist_data.loc[mask, col].tail(window)

                if len(seasonal_data) >= min(5, window // 2):
                    rolling_stats.loc[idx, f'{col}_roll_mean'] = seasonal_data.mean()
                    rolling_stats.loc[idx, f'{col}_roll_median'] = seasonal_data.median()
                    rolling_stats.loc[idx, f'{col}_roll_high'] = seasonal_data.max()
                    rolling_stats.loc[idx, f'{col}_roll_low'] = seasonal_data.min()
                    rolling_stats.loc[idx, f'{col}_roll_std'] = seasonal_data.std()

                    if seasonal_data.std() > 0:
                        current_value = data.loc[idx, col]
                        z_score = (current_value - seasonal_data.mean()) / seasonal_data.std()
                        rolling_stats.loc[idx, f'{col}_seasonal_zscore'] = z_score

        return rolling_stats

    def calculate_seasonal_strength_advanced(self, data_type: str = 'spreads') -> pd.Series:
        """Calculate strength of seasonality for each series (migrated from SpreadAnalyzer)"""
        if data_type == 'spreads':
            if hasattr(self.spread_data, 'seq_spreads') and self.spread_data.seq_spreads is not None:
                data = self.spread_data.seq_spreads
            else:
                return pd.Series(dtype=float)
        elif data_type == 'returns':
            if hasattr(self.spread_data, 'seq_prices') and self.spread_data.seq_prices is not None:
                data = self.spread_data.seq_prices.pct_change()
            else:
                return pd.Series(dtype=float)
        else:
            if hasattr(self.spread_data, 'seq_prices') and self.spread_data.seq_prices is not None:
                data = self.spread_data.seq_prices
            else:
                return pd.Series(dtype=float)

        if data.empty:
            return pd.Series(dtype=float)

        strength = {}

        for col in data.columns:
            series = data[col].dropna()

            if len(series) < 252:
                continue

            # Autocorrelation at seasonal lags
            seasonal_lags = [21, 63, 126, 252]
            acf_values = []

            for lag in seasonal_lags:
                if len(series) > lag:
                    acf = series.autocorr(lag)
                    if not np.isnan(acf):
                        acf_values.append(abs(acf))

            strength[col] = np.mean(acf_values) if acf_values else 0

        return pd.Series(strength)


