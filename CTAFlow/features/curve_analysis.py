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

from .. import FuturesCurve, SpreadData

# Import data client if available
try:
    from ..data.data_client import DataClient
    from CTAFlow.data.contract_handling.futures_curve_manager import MONTH_CODE_MAP, _is_empty
except ImportError:
    # Fallback definitions for standalone use
    MONTH_CODE_MAP = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    DataClient = None

MONTH_CODE_ORDER = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']


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


class CurveEvolution:
    """Track and analyze curve evolution over time using FuturesCurve snapshots"""

    def __init__(self):
        """Initialize evolution tracker with validated data structures"""
        self.history: List[FuturesCurve] = []
        self.shape_history: List[Dict[str, float]] = []
        self.timestamps: List[datetime] = []


    def add_snapshot(self, futures_curve: FuturesCurve) -> None:
        """Add a FuturesCurve snapshot to history with validation"""
        if not isinstance(futures_curve, FuturesCurve):
            raise TypeError(f"Expected FuturesCurve, got {type(futures_curve)}")
        self.history.append(futures_curve)
        self.timestamps.append(futures_curve.ref_date)
        
        # Calculate features from the futures curve
        features = self._extract_curve_features(futures_curve)
        self.shape_history.append(features)
    
    def add_snapshot_from_spread_data(self, spread_data: SpreadData, date: Optional[datetime] = None):
        """Add a snapshot from SpreadData by creating a FuturesCurve"""
        if date is None:
            date = datetime.now()
        
        futures_curve = spread_data.create_futures_curve(date)
        self.add_snapshot(futures_curve)
    
    def _extract_curve_features(self, curve: FuturesCurve) -> Dict[str, float]:
        """Extract features from a FuturesCurve"""
        features = {}
        
        # Use seq_prices if available, otherwise regular prices - ensure array consistency
        if curve.seq_prices is not None:
            prices = np.array(curve.seq_prices) if not isinstance(curve.seq_prices, np.ndarray) else curve.seq_prices
        else:
            prices = np.array(curve.prices) if not isinstance(curve.prices, np.ndarray) else curve.prices
        
        if len(prices) >= 2:
            features['mean_price'] = np.mean(prices)
            features['std_price'] = np.std(prices)
            features['min_price'] = np.min(prices)
            features['max_price'] = np.max(prices)
            features['price_range'] = features['max_price'] - features['min_price']
            features['overall_slope'] = (prices[-1] - prices[0]) / len(prices)
            
            # Shape metrics
            features['skewness'] = skew(prices)
            features['kurtosis'] = kurtosis(prices)
            
            # Contango/backwardation
            if len(prices) > 1:
                price_diffs = np.diff(prices)
                features['contango_ratio'] = np.sum(price_diffs > 0) / len(price_diffs)
                features['avg_calendar_spread'] = np.mean(price_diffs)
                
                # Curvature if enough points
                if len(prices) >= 3:
                    second_diffs = np.diff(price_diffs)
                    features['convexity'] = np.mean(second_diffs)
                    features['max_convexity'] = np.max(np.abs(second_diffs))
        
        # Volume concentration if available - use numpy operations
        if curve.volumes is not None and len(curve.volumes) > 0:
            volumes = np.array(curve.volumes) if not isinstance(curve.volumes, np.ndarray) else curve.volumes
            valid_vols = volumes[~np.isnan(volumes)]
            if len(valid_vols) > 0 and np.sum(valid_vols) > 0:
                vol_shares = valid_vols / np.sum(valid_vols)
                features['volume_concentration'] = np.sum(vol_shares ** 2)
        
        # OI concentration if available - use numpy operations
        if curve.open_interest is not None and len(curve.open_interest) > 0:
            oi = np.array(curve.open_interest) if not isinstance(curve.open_interest, np.ndarray) else curve.open_interest
            valid_oi = oi[~np.isnan(oi)]
            if len(valid_oi) > 0 and np.sum(valid_oi) > 0:
                oi_shares = valid_oi / np.sum(valid_oi)
                features['oi_concentration'] = np.sum(oi_shares ** 2)
                
        return features

    def calculate_changes(self, lookback: int = 1) -> Dict[str, float]:
        """Calculate changes over lookback periods"""
        if len(self.history) < lookback + 1:
            return {}

        current = self.shape_history[-1]
        previous = self.shape_history[-lookback - 1]

        changes = {}
        for key in current:
            if key in previous:
                changes[f'{key}_change'] = current[key] - previous[key]
                if previous[key] != 0:
                    changes[f'{key}_pct_change'] = ((current[key] - previous[key]) / previous[key]) * 100

        changes.update(self._calculate_structural_changes(lookback))

        return changes

    def _calculate_structural_changes(self, lookback: int = 1) -> Dict[str, float]:
        """Calculate structural curve changes using FuturesCurve objects"""
        if len(self.history) < lookback + 1:
            return {}

        current_curve = self.history[-1]
        previous_curve = self.history[-lookback - 1]

        changes = {}

        # Get prices from FuturesCurve objects (prefer seq_prices if available)
        current_prices = np.array(current_curve.seq_prices if current_curve.seq_prices is not None else current_curve.prices)
        previous_prices = np.array(previous_curve.seq_prices if previous_curve.seq_prices is not None else previous_curve.prices)

        # Align by length
        min_len = min(len(current_prices), len(previous_prices))

        if min_len > 0:
            current_prices = current_prices[:min_len]
            previous_prices = previous_prices[:min_len]

            mask = ~(np.isnan(current_prices) | np.isnan(previous_prices))
            if np.any(mask):
                current_prices = current_prices[mask]
                previous_prices = previous_prices[mask]

                # Parallel shift
                changes['parallel_shift'] = np.mean(current_prices - previous_prices)

                # Twist
                if len(current_prices) >= 4:
                    front_change = np.mean(current_prices[:2] - previous_prices[:2])
                    back_change = np.mean(current_prices[-2:] - previous_prices[-2:])
                    changes['curve_twist'] = back_change - front_change

                # Butterfly
                if len(current_prices) >= 3:
                    mid_idx = len(current_prices) // 2
                    mid_change = current_prices[mid_idx] - previous_prices[mid_idx]
                    edge_change = np.mean([
                        current_prices[0] - previous_prices[0],
                        current_prices[-1] - previous_prices[-1]
                    ])
                    changes['curve_butterfly'] = mid_change - edge_change

                # RMSE
                changes['rmse'] = np.sqrt(np.mean((current_prices - previous_prices) ** 2))

        return changes

    def get_feature_timeseries(self, feature_name: str) -> pd.Series:
        """Get time series of a specific feature"""
        if not self.shape_history:
            return pd.Series()

        values = [shapes.get(feature_name, np.nan) for shapes in self.shape_history]
        return pd.Series(values, index=self.timestamps, name=feature_name)
    
    def generate_dataframe(self, include_prices: bool = True, include_metadata: bool = True) -> pd.DataFrame:
        """
        Generate a DataFrame aggregating all history data
        
        Parameters:
        - include_prices: Whether to include individual price series
        - include_metadata: Whether to include ref_date, labels, etc.
        
        Returns:
        - DataFrame with timestamps as index and features/prices as columns
        """
        if not self.history:
            return pd.DataFrame()
        
        data_rows = []
        
        for i, (curve, features, timestamp) in enumerate(zip(self.history, self.shape_history, self.timestamps)):
            row = {'timestamp': timestamp}
            
            # Add metadata if requested
            if include_metadata:
                row['ref_date'] = curve.ref_date
                if curve.seq_labels:
                    row['seq_labels'] = '|'.join(curve.seq_labels)  # Join with delimiter
                if curve.curve_month_labels:
                    row['month_labels'] = '|'.join(curve.curve_month_labels)
            
            # Add prices if requested
            if include_prices:
                # Sequential prices (preferred)
                if curve.seq_prices is not None:
                    for j, price in enumerate(curve.seq_prices):
                        row[f'seq_price_M{j+1}'] = price
                
                # Original calendar month prices
                for j, (label, price) in enumerate(zip(curve.curve_month_labels, curve.prices)):
                    row[f'price_{label}'] = price
                
                # Volumes if available
                if curve.volumes is not None:
                    for j, (label, vol) in enumerate(zip(curve.curve_month_labels, curve.volumes)):
                        if not np.isnan(vol):
                            row[f'volume_{label}'] = vol
                
                # Open interest if available
                if curve.open_interest is not None:
                    for j, (label, oi) in enumerate(zip(curve.curve_month_labels, curve.open_interest)):
                        if not np.isnan(oi):
                            row[f'oi_{label}'] = oi
                
                # Days to expiry if available
                if curve.days_to_expiry is not None:
                    for j, (label, dte) in enumerate(zip(curve.curve_month_labels, curve.days_to_expiry)):
                        row[f'dte_{label}'] = dte
            
            # Add all calculated features
            row.update(features)
            
            data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Set timestamp as index if it exists
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        return df


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



class SpreadAnalyzer:
    """
    Analyzer for SpreadFeatures - handles spreads, seasonality, and advanced features
    """

    def __init__(self, spread_features: SpreadData):
        self.sf = spread_features
        self.levy_areas = None  # Store as numpy array for consistency
        self.signatures = None  # Store as numpy array for consistency
        self.seasonal_cache = {}

    def calculate_levy_areas(self, window: int = 20) -> np.ndarray:
        """Calculate Lévy areas between contract pairs using sequentialized data"""
        if not hasattr(self.sf, 'seq_data') or self.sf.seq_data is None:
            return np.array([])
        
        # Work with sequentialized data, not raw curve data
        if (self.sf.seq_data.seq_prices is None or 
            self.sf.seq_data.seq_prices.data is None or
            len(self.sf.seq_data.seq_prices.data.shape) != 2):
            return np.array([])
            
        seq_prices = self.sf.seq_data.seq_prices.data
        n_times, n_contracts = seq_prices.shape
        
        # Calculate Lévy areas between adjacent contracts
        levy_areas = np.full((n_times, n_contracts - 1), np.nan)
        
        for contract_pair in range(n_contracts - 1):
            front_prices = seq_prices[:, contract_pair]
            back_prices = seq_prices[:, contract_pair + 1]
            
            levy_area = self._compute_levy_area(front_prices, back_prices, window)
            levy_areas[:, contract_pair] = levy_area
        
        self.levy_areas = levy_areas
        return levy_areas

    def _compute_levy_area(self, X: np.ndarray, Y: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling Lévy area between two time series"""
        n = len(X)
        levy_areas = np.full(n, np.nan)

        for i in range(window, n):
            x_window = X[i-window:i]
            y_window = Y[i-window:i]

            if np.any(np.isnan(x_window)) or np.any(np.isnan(y_window)):
                continue

            area = 0.0
            for j in range(len(x_window) - 1):
                area += x_window[j] * (y_window[j+1] - y_window[j])
                area -= y_window[j] * (x_window[j+1] - x_window[j])

            levy_areas[i] = area / 2.0

        return levy_areas

    def calculate_path_signatures(self, depth: int = 3) -> np.ndarray:
        """Calculate path signatures up to specified depth using sequentialized data"""
        if not hasattr(self.sf, 'seq_data') or self.sf.seq_data is None:
            return np.array([])
        
        # Work with sequentialized data directly
        if (self.sf.seq_data.seq_prices is None or 
            self.sf.seq_data.seq_prices.data is None or
            len(self.sf.seq_data.seq_prices.data.shape) != 2):
            return np.array([])
            
        seq_prices = self.sf.seq_data.seq_prices.data
        n_times, n_contracts = seq_prices.shape
        
        if n_contracts < 2:
            return np.array([])
        
        # Calculate signatures using numpy operations
        signatures_list = []
        
        # First-order: increments
        increments = np.diff(seq_prices, axis=0)  # (n_times-1, n_contracts)
        signatures_list.append(increments)
        
        # Second-order: Lévy areas (if already calculated)
        if self.levy_areas is not None:
            # Pad to match dimensions if needed
            levy_padded = np.pad(self.levy_areas, ((1, 0), (0, 0)), mode='constant', constant_values=np.nan)
            signatures_list.append(levy_padded[:n_times-1])
        
        # Third-order: Triple products (if requested and sufficient contracts)
        if depth >= 3 and n_contracts >= 3:
            triple_products = increments[:, :-2] * increments[:, 1:-1] * increments[:, 2:]
            # Pad to maintain consistent shape
            padded_triple = np.pad(triple_products, ((0, 0), (0, n_contracts-triple_products.shape[1])), 
                                 mode='constant', constant_values=np.nan)
            signatures_list.append(padded_triple)
        
        # Combine all signature levels
        if signatures_list:
            self.signatures = np.concatenate(signatures_list, axis=1)
        else:
            self.signatures = np.array([])
            
        return self.signatures

    def calculate_seasonal_statistics(self,
                                     data_type: str = 'spreads',
                                     groupby: str = 'month',
                                     rolling_window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate seasonal statistics with optional rolling window

        Parameters:
        - data_type: 'spreads', 'returns', 'prices', or 'levy_areas'
        - groupby: 'month', 'week_of_year', 'week_of_month', 'day_of_week'
        - rolling_window: If specified, calculate rolling seasonal stats
        """
        # Select data source - convert to DataFrame for seasonal analysis
        if data_type == 'spreads':
            if hasattr(self.sf, 'seq_data') and self.sf.seq_data and self.sf.seq_data.seq_spreads:
                data = pd.DataFrame(self.sf.seq_data.seq_spreads.data, index=self.sf.seq_data.timestamps)
            else:
                return pd.DataFrame()
        elif data_type == 'returns':
            # Use sequentialized price data for returns
            if hasattr(self.sf, 'seq_data') and self.sf.seq_data and self.sf.seq_data.seq_prices:
                prices_df = pd.DataFrame(self.sf.seq_data.seq_prices.data, index=self.sf.seq_data.timestamps)
                data = prices_df.pct_change()
            else:
                return pd.DataFrame()
        elif data_type == 'prices':
            if hasattr(self.sf, 'seq_data') and self.sf.seq_data and self.sf.seq_data.seq_prices:
                data = pd.DataFrame(self.sf.seq_data.seq_prices.data, index=self.sf.seq_data.timestamps)
            else:
                return pd.DataFrame()
        elif data_type == 'levy_areas':
            if self.levy_areas is None or self.levy_areas.size == 0:
                self.calculate_levy_areas()
            if self.levy_areas is not None and self.levy_areas.size > 0:
                data = pd.DataFrame(self.levy_areas, index=self.sf.seq_data.timestamps[:len(self.levy_areas)])
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
            stats = self._calculate_static_seasonal_stats(data, group_by_cols)
        else:
            stats = self._calculate_rolling_seasonal_stats(data, group_by_cols, rolling_window)

        # Cache results
        cache_key = f"{data_type}_{groupby}_roll{rolling_window}"
        self.seasonal_cache[cache_key] = stats

        return stats

    def _calculate_static_seasonal_stats(self, data: pd.DataFrame, group_by_cols: List[str]) -> pd.DataFrame:
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

    def _calculate_rolling_seasonal_stats(self, data: pd.DataFrame,
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

    def calculate_seasonal_strength(self, data_type: str = 'spreads') -> pd.Series:
        """Calculate strength of seasonality for each series"""
        # Convert numpy arrays to DataFrame for seasonal analysis
        if data_type == 'spreads':
            if hasattr(self.sf, 'seq_data') and self.sf.seq_data and self.sf.seq_data.seq_spreads:
                data = pd.DataFrame(self.sf.seq_data.seq_spreads.data, index=self.sf.seq_data.timestamps)
            else:
                return pd.Series(dtype=float)
        elif data_type == 'returns':
            if hasattr(self.sf, 'seq_data') and self.sf.seq_data and self.sf.seq_data.seq_prices:
                prices_df = pd.DataFrame(self.sf.seq_data.seq_prices.data, index=self.sf.seq_data.timestamps)
                data = prices_df.pct_change()
            else:
                return pd.Series(dtype=float)
        else:
            if hasattr(self.sf, 'seq_data') and self.sf.seq_data and self.sf.seq_data.seq_prices:
                data = pd.DataFrame(self.sf.seq_data.seq_prices.data, index=self.sf.seq_data.timestamps)
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

    def detect_regime_changes(self, threshold: float = 2.0) -> pd.DataFrame:
        """Detect regime changes using Lévy areas"""
        if self.levy_areas is None or self.levy_areas.size == 0:
            self.calculate_levy_areas()

        if self.levy_areas is None or self.levy_areas.size == 0:
            return pd.DataFrame()

        # Convert numpy array to DataFrame for regime analysis
        levy_df = pd.DataFrame(self.levy_areas, index=self.sf.seq_data.timestamps[:len(self.levy_areas)])
        
        regime_changes = []

        for col in levy_df.columns:
            levy_series = levy_df[col].dropna()

            if len(levy_series) < 20:
                continue

            rolling_mean = levy_series.rolling(20).mean()
            rolling_std = levy_series.rolling(20).std()

            z_scores = (levy_series - rolling_mean) / (rolling_std + 1e-10)

            sign_changes = np.diff(np.sign(levy_series))
            sign_change_points = np.where(np.abs(sign_changes) > 0)[0]

            outliers = np.abs(z_scores) > threshold

            for idx in levy_series.index[outliers]:
                regime_changes.append({
                    'date': idx,
                    'contract_pair': col.replace('levy_', ''),
                    'levy_area': levy_series[idx],
                    'z_score': z_scores[idx] if idx in z_scores.index else np.nan,
                    'type': 'outlier'
                })

            for point in sign_change_points:
                if point < len(levy_series):
                    idx = levy_series.index[point]
                    regime_changes.append({
                        'date': idx,
                        'contract_pair': col.replace('levy_', ''),
                        'levy_area': levy_series.iloc[point],
                        'z_score': z_scores.iloc[point] if point < len(z_scores) else np.nan,
                        'type': 'sign_change'
                    })

        return pd.DataFrame(regime_changes)


