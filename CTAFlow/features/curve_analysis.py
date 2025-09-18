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

from ..data.contract_handling.curve_manager import FuturesCurve, SpreadData, SpreadFeature
from ..utils.seasonal import deseasonalize_monthly

# Import data client and utilities if available
try:
    from ..data.data_client import DataClient
    from CTAFlow.data.contract_handling.curve_manager import MONTH_CODE_MAP, _is_empty
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
import plotly.express as px

# Import tenor interpolation utilities
try:
    from ..utils.tenor_interpolation import TenorInterpolator, create_tenor_grid
except ImportError:
    TenorInterpolator = None
    create_tenor_grid = None
    warnings.warn("TenorInterpolator not available - constant maturity analysis disabled")

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

    def __init__(self, 
                 curves_data: Optional[Union[pd.Series, List[FuturesCurve], np.ndarray]] = None,
                 dates: Optional[Union[pd.DatetimeIndex, List[datetime], np.ndarray]] = None):
        """
        Initialize evolution tracker with validated data structures
        
        Parameters:
        - curves_data: Optional bulk data input (pd.Series, list, or array of FuturesCurves)
        - dates: Optional datetime index (required if curves_data is list/array)
        """
        self.history: pd.Series = pd.Series(dtype=object, name='futures_curves')
        self.shape_history: pd.DataFrame = pd.DataFrame()
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        
        # Handle bulk data input during initialization
        if curves_data is not None:
            self.load_curves_bulk(curves_data, dates)

    def load_curves_bulk(self, 
                        curves_data: Union[pd.Series, List[FuturesCurve], np.ndarray],
                        dates: Optional[Union[pd.DatetimeIndex, List[datetime], np.ndarray]] = None):
        """
        Load multiple curves at once for fast pipeline creation
        
        Parameters:
        - curves_data: Series, list, or array of FuturesCurve objects
        - dates: Datetime index (required if curves_data is list/array)
        """
        if isinstance(curves_data, pd.Series):
            # Direct Series input with datetime index
            if not pd.api.types.is_datetime64_any_dtype(curves_data.index):
                raise ValueError("Series index must be datetime-like")
            
            self.history = curves_data.copy()
            
        elif isinstance(curves_data, (list, np.ndarray)):
            # List or array input - dates parameter required
            if dates is None:
                raise ValueError("dates parameter required when curves_data is list or array")
            
            # Convert dates to DatetimeIndex if needed
            if not isinstance(dates, pd.DatetimeIndex):
                dates = pd.DatetimeIndex(dates)
            
            if len(curves_data) != len(dates):
                raise ValueError(f"Length mismatch: curves_data ({len(curves_data)}) vs dates ({len(dates)})")
            
            # Create Series
            self.history = pd.Series(curves_data, index=dates, name='futures_curves')
            
        else:
            raise TypeError(f"curves_data must be pd.Series, list, or np.ndarray, got {type(curves_data)}")
        
        # Validate all entries are FuturesCurve objects or None
        for idx, curve in self.history.items():
            if curve is not None and not isinstance(curve, FuturesCurve):
                raise TypeError(f"All curves must be FuturesCurve objects or None, got {type(curve)} at {idx}")
        
        # Calculate features for all curves using broadcast method
        self._calculate_all_features_bulk()
        
        print(f"Loaded {len(self.history)} curves covering {self.history.index.min()} to {self.history.index.max()}")

    def _calculate_all_features_bulk(self):
        """Calculate features for all curves in bulk using vectorized operations"""
        if self.history.empty:
            return
        
        # Extract all curves and organize data
        valid_entries = []
        timestamps = []
        
        for timestamp, curve in self.history.items():
            if curve is not None:
                try:
                    # Extract features using existing method
                    features = self._extract_curve_features(curve)
                    valid_entries.append((timestamp, features))
                    timestamps.append(timestamp)
                except Exception as e:
                    # Skip problematic curves but log the issue
                    warnings.warn(f"Failed to extract features for curve at {timestamp}: {e}")
                    continue
        
        if not valid_entries:
            self.shape_history = pd.DataFrame()
            return
        
        # Create DataFrame from all features at once
        features_list = [entry[1] for entry in valid_entries]
        self.shape_history = pd.DataFrame(features_list, index=timestamps)
        
        # Clear cache since we have new data
        self._feature_cache.clear()

    @classmethod
    def from_spread_data(cls, 
                        spread_data: 'SpreadData',
                        date_range: Optional[Union[slice, pd.DatetimeIndex, List[datetime]]] = None) -> 'CurveEvolution':
        """
        Create CurveEvolution directly from SpreadData for fast pipeline
        
        Parameters:
        - spread_data: SpreadData object with seq_data
        - date_range: Optional date range to extract (slice, DatetimeIndex, or list)
        
        Returns:
        - CurveEvolution instance with curves pre-loaded
        """
        if not hasattr(spread_data, 'seq_data') or spread_data.seq_data is None:
            raise ValueError("SpreadData must have seq_data for bulk curve creation")
        
        # Determine date range
        if date_range is None:
            dates_to_use = spread_data.index
        elif isinstance(date_range, slice):
            dates_to_use = spread_data.index[date_range]
        elif isinstance(date_range, (pd.DatetimeIndex, list, np.ndarray)):
            # Filter to dates that exist in spread_data
            available_dates = set(spread_data.index)
            dates_to_use = [d for d in date_range if d in available_dates]
            if not dates_to_use:
                raise ValueError("No valid dates found in date_range")
            dates_to_use = pd.DatetimeIndex(dates_to_use)
        else:
            raise TypeError("date_range must be slice, DatetimeIndex, list, or None")
        
        # Create curves for all dates using vectorized approach
        curves_list = []
        valid_dates = []
        
        # Use batch processing for efficiency
        for date in dates_to_use:
            try:
                curve = spread_data.create_futures_curve(date)
                curves_list.append(curve)
                valid_dates.append(date)
            except Exception as e:
                warnings.warn(f"Failed to create curve for {date}: {e}")
                curves_list.append(None)
                valid_dates.append(date)
        
        # Create instance with bulk data
        instance = cls(curves_list, valid_dates)
        
        # Store reference to source data for potential re-use
        instance._source_spread_data = spread_data
        
        return instance
    
    @classmethod 
    def from_series(cls, curves_series: pd.Series) -> 'CurveEvolution':
        """
        Create CurveEvolution from pd.Series of FuturesCurve objects
        
        Parameters:
        - curves_series: pd.Series with datetime index and FuturesCurve values
        """
        return cls(curves_series)
    
    def add_curves_bulk(self, 
                       curves_data: Union[pd.Series, List[FuturesCurve], np.ndarray],
                       dates: Optional[Union[pd.DatetimeIndex, List[datetime], np.ndarray]] = None):
        """
        Add multiple curves to existing CurveEvolution instance
        
        Parameters:
        - curves_data: Series, list, or array of FuturesCurve objects  
        - dates: Datetime index (required if curves_data is list/array)
        """
        # Create temporary instance to validate and process input
        temp_instance = CurveEvolution(curves_data, dates)
        
        # Merge with existing history
        if self.history.empty:
            self.history = temp_instance.history.copy()
            self.shape_history = temp_instance.shape_history.copy()
        else:
            # Concatenate and sort
            combined_history = pd.concat([self.history, temp_instance.history])
            combined_features = pd.concat([self.shape_history, temp_instance.shape_history])
            
            # Remove duplicates (keep last) and sort
            self.history = combined_history[~combined_history.index.duplicated(keep='last')].sort_index()
            self.shape_history = combined_features[~combined_features.index.duplicated(keep='last')].sort_index()
        
        # Clear cache
        self._feature_cache.clear()
        
        print(f"Added curves. Total: {len(self.history)} curves from {self.history.index.min()} to {self.history.index.max()}")

    def update_from_spread_data(self, 
                               spread_data: 'SpreadData',
                               date_range: Optional[Union[slice, pd.DatetimeIndex, List[datetime]]] = None):
        """
        Update existing CurveEvolution with new data from SpreadData
        
        Parameters:
        - spread_data: SpreadData object
        - date_range: Optional date range to update
        """
        # Create new instance from spread_data
        new_instance = self.from_spread_data(spread_data, date_range)
        
        # Add to existing data
        self.add_curves_bulk(new_instance.history)

    def add_snapshot(self, futures_curve: FuturesCurve) -> None:
        """Add a FuturesCurve snapshot to history with validation"""
        if not isinstance(futures_curve, FuturesCurve):
            raise TypeError(f"Expected FuturesCurve, got {type(futures_curve)}")
        
        # Add to history Series with datetime index
        self.history.loc[futures_curve.ref_date] = futures_curve
        
        # Calculate features from the futures curve
        features = self._extract_curve_features(futures_curve)
        
        # Add features to DataFrame
        if self.shape_history.empty:
            self.shape_history = pd.DataFrame([features], index=[futures_curve.ref_date])
        else:
            # Concatenate new features
            new_row = pd.DataFrame([features], index=[futures_curve.ref_date])
            self.shape_history = pd.concat([self.shape_history, new_row])
        
        # Clear cache when new data is added
        self._feature_cache.clear()
    
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
        """Calculate changes over lookback periods using DataFrame structure"""
        if len(self.history) < lookback + 1:
            return {}

        # Get current and previous rows from DataFrame
        current = self.shape_history.iloc[-1]
        previous = self.shape_history.iloc[-lookback - 1]

        changes = {}
        for key in current.index:
            if key in previous.index and pd.notna(current[key]) and pd.notna(previous[key]):
                changes[f'{key}_change'] = current[key] - previous[key]
                if previous[key] != 0:
                    changes[f'{key}_pct_change'] = ((current[key] - previous[key]) / previous[key]) * 100

        # Add structural changes
        structural_changes = self._calculate_structural_changes(lookback)
        changes.update(structural_changes)

        return changes

    def _calculate_structural_changes(self, lookback: int = 1) -> Dict[str, float]:
        """Calculate structural curve changes using FuturesCurve objects from Series"""
        if len(self.history) < lookback + 1:
            return {}

        # Get current and previous curves from Series
        current_curve = self.history.iloc[-1]
        previous_curve = self.history.iloc[-lookback - 1]

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
        """Get time series of a specific feature from DataFrame"""
        if self.shape_history.empty:
            return pd.Series()

        if feature_name not in self.shape_history.columns:
            return pd.Series(index=self.shape_history.index, name=feature_name, dtype=float)

        return self.shape_history[feature_name].copy()
    
    def generate_dataframe(self, include_prices: bool = True, include_metadata: bool = True) -> pd.DataFrame:
        """
        Generate a DataFrame aggregating all history data
        
        Parameters:
        - include_prices: Whether to include individual price series
        - include_metadata: Whether to include ref_date, labels, etc.
        
        Returns:
        - DataFrame with timestamps as index and features/prices as columns
        """
        if self.history.empty:
            return pd.DataFrame()
        
        # Start with shape_history as base DataFrame
        result_df = self.shape_history.copy()
        
        # Add price and metadata columns if requested
        if include_prices or include_metadata:
            for timestamp, curve in self.history.items():
                if curve is None:
                    continue
                
                # Add metadata if requested
                if include_metadata:
                    result_df.loc[timestamp, 'ref_date'] = curve.ref_date
                    if curve.seq_labels:
                        result_df.loc[timestamp, 'seq_labels'] = '|'.join(curve.seq_labels)
                    if curve.curve_month_labels:
                        result_df.loc[timestamp, 'month_labels'] = '|'.join(curve.curve_month_labels)
                
                # Add prices if requested
                if include_prices:
                    # Sequential prices (preferred)
                    if curve.seq_prices is not None:
                        for j, price in enumerate(curve.seq_prices):
                            result_df.loc[timestamp, f'seq_price_M{j+1}'] = price
                    
                    # Original calendar month prices
                    if curve.prices is not None and curve.curve_month_labels is not None:
                        for j, (label, price) in enumerate(zip(curve.curve_month_labels, curve.prices)):
                            result_df.loc[timestamp, f'price_{label}'] = price
                    
                    # Volumes if available
                    if curve.volumes is not None and curve.curve_month_labels is not None:
                        for j, (label, vol) in enumerate(zip(curve.curve_month_labels, curve.volumes)):
                            if not np.isnan(vol):
                                result_df.loc[timestamp, f'volume_{label}'] = vol
                    
                    # Open interest if available
                    if curve.open_interest is not None and curve.curve_month_labels is not None:
                        for j, (label, oi) in enumerate(zip(curve.curve_month_labels, curve.open_interest)):
                            if not np.isnan(oi):
                                result_df.loc[timestamp, f'oi_{label}'] = oi
                    
                    # Days to expiry if available
                    if curve.days_to_expiry is not None and curve.curve_month_labels is not None:
                        for j, (label, dte) in enumerate(zip(curve.curve_month_labels, curve.days_to_expiry)):
                            result_df.loc[timestamp, f'dte_{label}'] = dte
        
        return result_df
    
    def calculate_all_features_broadcast(self) -> pd.DataFrame:

        """
        Vectorized calculation of features across all curves simultaneously
        Avoids iteration and leverages pandas/numpy broadcast operations
        """
        if self.history.empty:
            return pd.DataFrame()
        
        # Check cache first
        if 'broadcast_features' in self._feature_cache:
            return self._feature_cache['broadcast_features'].copy()
        
        # Extract all price arrays and organize into 3D array
        max_contracts = 0
        valid_curves = []
        timestamps = []
        
        for timestamp, curve in self.history.items():
            if curve is not None:
                prices = np.array(curve.seq_prices if curve.seq_prices is not None else curve.prices)
                if len(prices) > 0:
                    valid_curves.append((timestamp, curve, prices))
                    max_contracts = max(max_contracts, len(prices))
                    timestamps.append(timestamp)
        
        if not valid_curves:
            return pd.DataFrame()
        
        # Create 3D array: (n_dates, n_contracts, n_features)
        n_dates = len(valid_curves)
        price_matrix = np.full((n_dates, max_contracts), np.nan)
        
        for i, (timestamp, curve, prices) in enumerate(valid_curves):
            price_matrix[i, :len(prices)] = prices
        
        # Broadcast calculations
        features_dict = {}
        
        # Basic statistics across contracts (axis=1)
        features_dict['mean_price'] = np.nanmean(price_matrix, axis=1)
        features_dict['std_price'] = np.nanstd(price_matrix, axis=1)
        features_dict['min_price'] = np.nanmin(price_matrix, axis=1)
        features_dict['max_price'] = np.nanmax(price_matrix, axis=1)
        features_dict['price_range'] = features_dict['max_price'] - features_dict['min_price']
        
        # Slope calculations (vectorized)
        slopes = np.full(n_dates, np.nan)
        for i in range(n_dates):
            valid_mask = ~np.isnan(price_matrix[i, :])
            if np.sum(valid_mask) >= 2:
                x_vals = np.arange(np.sum(valid_mask))
                y_vals = price_matrix[i, valid_mask]
                slopes[i] = np.polyfit(x_vals, y_vals, 1)[0]
        
        features_dict['overall_slope'] = slopes
        
        # Shape metrics using scipy functions on the matrix
        skewness = np.full(n_dates, np.nan)
        kurt = np.full(n_dates, np.nan)
        
        for i in range(n_dates):
            valid_prices = price_matrix[i, ~np.isnan(price_matrix[i, :])]
            if len(valid_prices) >= 3:
                skewness[i] = skew(valid_prices)
                kurt[i] = kurtosis(valid_prices)
        
        features_dict['skewness'] = skewness
        features_dict['kurtosis'] = kurt
        
        # Contango/Backwardation ratios
        contango_ratios = np.full(n_dates, np.nan)
        for i in range(n_dates):
            valid_prices = price_matrix[i, ~np.isnan(price_matrix[i, :])]
            if len(valid_prices) >= 2:
                diffs = np.diff(valid_prices)
                contango_ratios[i] = np.sum(diffs > 0) / len(diffs)
        
        features_dict['contango_ratio'] = contango_ratios
        
        # Create DataFrame
        features_df = pd.DataFrame(features_dict, index=timestamps)
        
        # Cache results
        self._feature_cache['broadcast_features'] = features_df.copy()
        
        return features_df
    
    def calculate_rolling_metrics_broadcast(self, 
                                          window: int = 20, 
                                          metrics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling metrics using vectorized operations
        
        Parameters:
        - window: Rolling window size
        - metrics: List of metrics to calculate ['volatility', 'correlation', 'slope_trend', 'regime_stability']
        """
        if metrics is None:
            metrics = ['volatility', 'slope_trend', 'regime_stability']
        
        # Get base features first
        base_features = self.calculate_all_features_broadcast()
        if base_features.empty:
            return pd.DataFrame()
        
        cache_key = f'rolling_{window}_{hash(tuple(metrics))}'
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key].copy()
        
        rolling_results = {}
        
        for metric in metrics:
            if metric == 'volatility':
                # Rolling volatility of slopes
                if 'overall_slope' in base_features.columns:
                    rolling_results['slope_volatility'] = base_features['overall_slope'].rolling(window).std()
                    rolling_results['price_volatility'] = base_features['mean_price'].rolling(window).std()
            
            elif metric == 'slope_trend':
                # Trend in slope over rolling window
                if 'overall_slope' in base_features.columns:
                    slope_trend = base_features['overall_slope'].rolling(window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                    )
                    rolling_results['slope_trend'] = slope_trend
            
            elif metric == 'regime_stability':
                # Stability of contango/backwardation regime
                if 'contango_ratio' in base_features.columns:
                    regime_std = base_features['contango_ratio'].rolling(window).std()
                    rolling_results['regime_stability'] = 1 / (1 + regime_std)  # Higher = more stable
            
            elif metric == 'correlation':
                # Rolling correlation between slope and price level
                if 'overall_slope' in base_features.columns and 'mean_price' in base_features.columns:
                    rolling_corr = base_features['overall_slope'].rolling(window).corr(
                        base_features['mean_price'].rolling(window)
                    )
                    rolling_results['slope_price_correlation'] = rolling_corr
        
        rolling_df = pd.DataFrame(rolling_results, index=base_features.index)
        
        # Cache results
        self._feature_cache[cache_key] = rolling_df.copy()
        
        return rolling_df
    
    def detect_regime_changes_broadcast(self, 
                                      threshold: float = 2.0,
                                      min_regime_length: int = 5) -> pd.DataFrame:
        """
        Detect regime changes using vectorized operations on slope data
        
        Parameters:
        - threshold: Z-score threshold for regime change detection
        - min_regime_length: Minimum length of a regime to be considered valid
        """
        cache_key = f'regime_changes_{threshold}_{min_regime_length}'
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key].copy()
        
        base_features = self.calculate_all_features_broadcast()
        if base_features.empty or 'overall_slope' not in base_features.columns:
            return pd.DataFrame()
        
        slopes = base_features['overall_slope'].dropna()
        if len(slopes) < min_regime_length * 2:
            return pd.DataFrame()
        
        # Calculate rolling statistics
        rolling_mean = slopes.rolling(20, center=True).mean()
        rolling_std = slopes.rolling(20, center=True).std()
        z_scores = (slopes - rolling_mean) / (rolling_std + 1e-10)
        
        # Detect regime changes using sign changes and outliers
        regime_changes = []
        
        # Sign-based regime detection
        contango_regime = (slopes > 0.01).astype(int)  # 1 for contango, 0 for backwardation/flat
        regime_changes_idx = np.where(np.abs(np.diff(contango_regime)) > 0)[0]
        
        for idx in regime_changes_idx:
            if idx < len(slopes):
                regime_changes.append({
                    'date': slopes.index[idx],
                    'type': 'regime_shift',
                    'slope': slopes.iloc[idx],
                    'z_score': z_scores.iloc[idx] if idx < len(z_scores) and pd.notna(z_scores.iloc[idx]) else np.nan,
                    'from_regime': 'Contango' if contango_regime.iloc[idx-1] == 1 else 'Backwardation',
                    'to_regime': 'Contango' if contango_regime.iloc[idx] == 1 else 'Backwardation'
                })
        
        # Outlier-based detection
        outliers = np.abs(z_scores) > threshold
        for idx in outliers.index[outliers]:
            regime_changes.append({
                'date': idx,
                'type': 'outlier',
                'slope': slopes[idx],
                'z_score': z_scores[idx],
                'from_regime': 'Normal',
                'to_regime': 'Outlier'
            })
        
        regime_df = pd.DataFrame(regime_changes).sort_values('date') if regime_changes else pd.DataFrame()
        
        # Filter by minimum regime length
        if not regime_df.empty and min_regime_length > 1:
            filtered_changes = []
            last_change_date = None
            
            for _, row in regime_df.iterrows():
                if last_change_date is None:
                    filtered_changes.append(row)
                    last_change_date = row['date']
                else:
                    days_diff = (row['date'] - last_change_date).days
                    if days_diff >= min_regime_length:
                        filtered_changes.append(row)
                        last_change_date = row['date']
            
            regime_df = pd.DataFrame(filtered_changes) if filtered_changes else pd.DataFrame()
        
        # Cache results
        self._feature_cache[cache_key] = regime_df.copy() if not regime_df.empty else pd.DataFrame()
        
        return regime_df
    
    def calculate_correlations_matrix(self, features: List[str] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between different curve features over time
        Uses vectorized operations for efficiency
        """
        if features is None:
            features = ['mean_price', 'overall_slope', 'std_price', 'contango_ratio', 'skewness']
        
        cache_key = f'correlations_{hash(tuple(features))}'
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key].copy()
        
        base_features = self.calculate_all_features_broadcast()
        if base_features.empty:
            return pd.DataFrame()
        
        # Select available features
        available_features = [f for f in features if f in base_features.columns]
        if not available_features:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        feature_data = base_features[available_features]
        corr_matrix = feature_data.corr()
        
        # Cache results
        self._feature_cache[cache_key] = corr_matrix.copy()
        
        return corr_matrix
    
    def clear_cache(self):
        """Clear all cached calculations"""
        self._feature_cache.clear()
    
    def resample_curves(self, 
                       freq: str = 'W-FRI',
                       method: str = 'last') -> 'CurveEvolution':
        """
        Resample curves to different frequency for analysis
        
        Parameters:
        - freq: Pandas frequency string (e.g., 'W-FRI', 'M', 'Q')
        - method: Resampling method ('last', 'first', 'mean')
        
        Returns:
        - New CurveEvolution instance with resampled data
        """
        if self.history.empty:
            return CurveEvolution()
        
        if method == 'last':
            resampled_history = self.history.resample(freq).last()
        elif method == 'first':
            resampled_history = self.history.resample(freq).first()
        elif method == 'mean':
            # For 'mean', we need custom logic since we can't average FuturesCurve objects
            # Use last for now, but could implement curve averaging in future
            resampled_history = self.history.resample(freq).last()
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'last', 'first', or 'mean'")
        
        # Remove NaN entries
        resampled_history = resampled_history.dropna()
        
        return CurveEvolution.from_series(resampled_history)
    
    def get_date_range_info(self) -> Dict[str, Any]:
        """Get information about the date range and data coverage"""
        if self.history.empty:
            return {'empty': True}
        
        non_null_curves = self.history.dropna()
        
        info = {
            'empty': False,
            'total_dates': len(self.history),
            'valid_curves': len(non_null_curves),
            'missing_curves': len(self.history) - len(non_null_curves),
            'date_range': {
                'start': self.history.index.min(),
                'end': self.history.index.max(),
                'span_days': (self.history.index.max() - self.history.index.min()).days
            },
            'frequency_analysis': {
                'avg_gap_days': np.mean(np.diff(self.history.index.values) / np.timedelta64(1, 'D')) if len(self.history) > 1 else 0,
                'max_gap_days': np.max(np.diff(self.history.index.values) / np.timedelta64(1, 'D')) if len(self.history) > 1 else 0,
                'inferred_freq': pd.infer_freq(self.history.index)
            }
        }
        
        return info
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and identify potential issues"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if self.history.empty:
            validation['valid'] = False
            validation['errors'].append("No data available")
            return validation
        
        # Check for missing curves
        null_count = self.history.isnull().sum()
        if null_count > 0:
            pct_missing = (null_count / len(self.history)) * 100
            validation['warnings'].append(f"{null_count} missing curves ({pct_missing:.1f}%)")
        
        # Check curve consistency
        contract_counts = []
        price_ranges = []
        
        for date, curve in self.history.items():
            if curve is not None:
                prices = curve.seq_prices if curve.seq_prices is not None else curve.prices
                if prices is not None:
                    contract_counts.append(len(prices))
                    price_ranges.append(np.max(prices) - np.min(prices) if len(prices) > 1 else 0)
        
        if contract_counts:
            unique_counts = set(contract_counts)
            if len(unique_counts) > 1:
                validation['warnings'].append(f"Inconsistent contract counts: {unique_counts}")
            
            validation['statistics'] = {
                'avg_contracts': np.mean(contract_counts),
                'avg_price_range': np.mean(price_ranges),
                'std_price_range': np.std(price_ranges)
            }
        
        # Check for data gaps
        if len(self.history) > 1:
            date_diffs = np.diff(self.history.index.values)
            max_gap = np.max(date_diffs) / np.timedelta64(1, 'D')
            if max_gap > 7:  # More than a week gap
                validation['warnings'].append(f"Large data gap detected: {max_gap:.1f} days")
        
        return validation
    
    def get_pipeline_summary(self) -> str:
        """Get a summary string describing the pipeline state"""
        if self.history.empty:
            return "CurveEvolution: Empty (no curves loaded)"
        
        info = self.get_date_range_info()
        validation = self.validate_data_quality()
        
        summary_parts = [
            f"CurveEvolution Summary:",
            f"  • {info['valid_curves']:,} curves from {info['date_range']['start'].strftime('%Y-%m-%d')} to {info['date_range']['end'].strftime('%Y-%m-%d')}",
            f"  • {info['date_range']['span_days']:,} days span, ~{info['frequency_analysis']['avg_gap_days']:.1f} day average frequency"
        ]
        
        if validation['warnings']:
            summary_parts.append(f"  • Warnings: {'; '.join(validation['warnings'])}")
        
        if hasattr(self, '_source_spread_data'):
            summary_parts.append(f"  • Source: SpreadData pipeline")
        
        feature_count = len(self.shape_history.columns) if not self.shape_history.empty else 0
        cache_count = len(self._feature_cache)
        summary_parts.append(f"  • Features: {feature_count} calculated, {cache_count} cached")
        
        return '\n'.join(summary_parts)
    
    def __repr__(self) -> str:
        """Enhanced repr with pipeline information"""
        return self.get_pipeline_summary()


# Factory functions for common pipeline patterns
def create_curve_evolution_pipeline(spread_data: 'SpreadData', 
                                   date_range: Optional[Union[slice, pd.DatetimeIndex]] = None,
                                   resample_freq: Optional[str] = None) -> CurveEvolution:
    """
    Factory function to create a complete CurveEvolution pipeline from SpreadData
    
    Parameters:
    - spread_data: Source SpreadData object
    - date_range: Optional date range filter
    - resample_freq: Optional resampling frequency ('W-FRI', 'M', etc.)
    
    Returns:
    - Configured CurveEvolution instance
    
    Example:
    >>> evolution = create_curve_evolution_pipeline(spread_data, 
    ...                                           date_range=slice('2023-01-01', '2023-12-31'),
    ...                                           resample_freq='W-FRI')
    """
    # Create base evolution
    evolution = CurveEvolution.from_spread_data(spread_data, date_range)
    
    # Apply resampling if requested
    if resample_freq is not None:
        evolution = evolution.resample_curves(resample_freq)
    
    print(f"\n{evolution.get_pipeline_summary()}")
    
    return evolution


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

    def calculate_path_signatures(self, depth: int = 2) -> np.ndarray:
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

        # Validate required data structures
        if not hasattr(self.sf, 'seq_data') or self.sf.seq_data is None:
            raise ValueError("seq_data is required for regime change detection")
        
        if self.sf.seq_data.timestamps is None:
            raise ValueError("timestamps are required for regime change detection")

        # Convert numpy array to DataFrame for regime analysis with proper column names
        n_contracts = self.levy_areas.shape[1] if len(self.levy_areas.shape) > 1 else 1
        column_names = [f'levy_pair_{i}' for i in range(n_contracts)]
        
        # Ensure we don't exceed available timestamps
        n_timestamps = min(len(self.levy_areas), len(self.sf.seq_data.timestamps))
        levy_df = pd.DataFrame(self.levy_areas[:n_timestamps], 
                              index=self.sf.seq_data.timestamps[:n_timestamps],
                              columns=column_names)
        
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

            # Get contract pair name
            contract_pair = col.replace('levy_', '') if 'levy_' in col else col

            for idx in levy_series.index[outliers]:
                regime_changes.append({
                    'date': idx,
                    'contract_pair': contract_pair,
                    'levy_area': levy_series[idx],
                    'z_score': z_scores[idx] if idx in z_scores.index else np.nan,
                    'type': 'outlier'
                })

            for point in sign_change_points:
                if point < len(levy_series):
                    idx = levy_series.index[point]
                    regime_changes.append({
                        'date': idx,
                        'contract_pair': contract_pair,
                        'levy_area': levy_series.iloc[point],
                        'z_score': z_scores.iloc[point] if point < len(z_scores) else np.nan,
                        'type': 'sign_change'
                    })

        return pd.DataFrame(regime_changes)


# ====================================================================================
# JIT-COMPILED UTILITY FUNCTIONS
# ====================================================================================

@jit(nopython=True)
def _calculate_log_levy_areas_numba(log_prices: np.ndarray, window: int) -> np.ndarray:
    """
    JIT-optimized Lévy area calculation on log prices
    
    Computes Lévy areas between consecutive contract months using log prices
    to detect fundamental drivers of curve evolution.
    
    Fixed scaling issue: Uses increments-only formula to avoid large outliers
    from mixing absolute log price levels with increments.
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
            
            # Calculate increments first
            front_increments = np.empty(len(front_window) - 1)
            back_increments = np.empty(len(back_window) - 1)
            
            for k in range(len(front_window) - 1):
                front_increments[k] = front_window[k + 1] - front_window[k]
                back_increments[k] = back_window[k + 1] - back_window[k]
            
            # Corrected Lévy area formula using increments only
            # This avoids the scaling issue by not mixing absolute levels with increments
            area = 0.0
            cumulative_front = 0.0
            cumulative_back = 0.0
            
            for k in range(len(front_increments)):
                # Build cumulative sums of increments (relative to window start)
                cumulative_front += front_increments[k]
                cumulative_back += back_increments[k]
                
                # Lévy area using cumulative increments instead of absolute levels
                area += cumulative_front * back_increments[k] - cumulative_back * front_increments[k]
            
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
                 symbol: Optional[str] = None):
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
        """
        
        self.symbol = symbol
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
            # Handle pandas Series (from SpreadData slicing)
            # Ensure it has a datetime index if possible
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data = data.copy()
                    data.index = pd.to_datetime(data.index)
                except Exception:
                    pass  # Keep original index if conversion fails
            return data
            
        elif hasattr(data, 'get_seq_curves'):
            # SpreadData object - extract curves and store reference
            try:
                self.symbol = getattr(data, 'symbol', self.symbol)
                self.spread_data = data  # Store reference for spot data access
                return data.get_seq_curves()
            except Exception as e:
                raise ValueError(f"Failed to extract curves from SpreadData: {e}")
                
        elif isinstance(data, list):
            # List of FuturesCurve objects
            if not data:
                return None
            
            # Validate all are FuturesCurve objects
            if not all(isinstance(curve, FuturesCurve) for curve in data):
                raise TypeError("All list elements must be FuturesCurve objects")
            
            # Create datetime index (use curve timestamps if available)
            dates = []
            for i, curve in enumerate(data):
                if hasattr(curve, 'timestamp') and curve.timestamp is not None:
                    dates.append(curve.timestamp)
                elif hasattr(curve, 'ref_date') and curve.ref_date is not None:
                    dates.append(curve.ref_date)
                else:
                    # Fallback to sequential dates
                    dates.append(pd.Timestamp.now() + pd.Timedelta(days=i))
            
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
                        step: int = 1) -> 'CurveEvolutionAnalyzer':
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
            
        Returns:
        --------
        CurveEvolutionAnalyzer
        """
        
        curves = spread_data.get_seq_curves(date_range=date_range, step=step)
        analyzer = cls(curves, symbol=spread_data.symbol)
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
            from ..data.contract_handling.dly_contract_manager import calculate_contract_expiry
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
        deseasonalized = np.full_like(log_prices, np.nan)

        for col in df.columns:
            series = df[col]
            if series.notna().any():
                monthly_means = series.groupby(series.index.month).transform('mean')
                deseasonalized[:, col] = (series - monthly_means).values

        return deseasonalized
    
    def _calculate_log_levy_areas_jit(self, 
                                     log_prices: np.ndarray,
                                     window: int) -> np.ndarray:
        """
        Wrapper for JIT-optimized Lévy area calculation on log prices
        
        Computes Lévy areas between consecutive contract months using log prices
        to detect fundamental drivers of curve evolution
        """
        return _calculate_log_levy_areas_numba(log_prices, window)
    
    def calculate_path_signatures(self,
                                window: int = None,
                                max_signature_level: int = 2,
                                constant_maturity: bool = False) -> Dict[str, Any]:
        """
        Calculate comprehensive path signatures for curve evolution analysis

        Parameters:
        -----------
        window : int
            Rolling window for calculations
        max_signature_level : int
            Maximum level of path signature to compute
        constant_maturity : bool, default False
            Use constant maturity tenors instead of sequential contracts

        Returns:
        --------
        Dict[str, Any]
            Complete path signature analysis results with:
            - levy_areas: Standard Lévy areas
            - log_levy_areas: Log price Lévy areas (key innovation)
            - path_variations: Path complexity measures
            - signature_levels: Multi-level signatures
            - curve_drivers: Identified drivers
            - regime_changes: Regime change detection
        """
        
        if window is None:
            window = self.default_window
        
        cache_key = f'path_signatures_{window}_{max_signature_level}_cm_{constant_maturity}'
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get log prices matrix - either constant maturity or sequential
        if constant_maturity and self.constant_maturity_data is not None:
            # Use constant maturity data
            log_prices = np.log(self.constant_maturity_data.values)
        else:
            # Use sequential contract data
            log_prices = self.get_log_prices_matrix()
        
        # Calculate log price Lévy areas (primary driver detection)
        log_levy_areas = self._calculate_log_levy_areas_jit(log_prices, window)
        
        # Calculate standard Lévy areas for comparison
        prices = np.exp(log_prices)  # Convert back to price level
        levy_areas = self._calculate_log_levy_areas_jit(prices, window)
        
        # Path variation analysis
        path_variations = self._calculate_path_variations(log_prices, window)
        
        # Higher-order signature terms
        signature_levels = self._calculate_signature_levels(
            log_prices, window, max_signature_level
        )
        
        # Detect curve drivers from log Lévy areas
        curve_drivers = self._detect_curve_drivers(log_levy_areas)
        
        # Regime change detection
        regime_changes = self._detect_regime_changes_from_levy(log_levy_areas)
        
        # Create result dictionary
        path_sig = {
            'levy_areas': levy_areas,
            'log_levy_areas': log_levy_areas,
            'path_variations': path_variations,
            'signature_levels': signature_levels,
            'curve_drivers': curve_drivers,
            'regime_changes': regime_changes
        }
        
        # Cache results
        self._cache[cache_key] = path_sig
        self.path_signatures = path_sig
        
        return path_sig
    
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
        """
        
        drivers = {}
        
        if log_levy_areas.size == 0:
            return drivers
        
        n_contracts = log_levy_areas.shape[1] + 1
        
        # 1. Front-End Driver - changes in near-term contracts
        if n_contracts >= 4:
            midpoint = n_contracts // 2
            front_end_levy = np.nanmean(log_levy_areas[:, :midpoint-1], axis=1)
            drivers['front_end_changes'] = front_end_levy
        else:
            # Fallback for limited contracts
            drivers['front_end_changes'] = np.nanmean(log_levy_areas[:, :1], axis=1)
        
        # 2. Back-End Driver - changes in long-term contracts
        if n_contracts >= 4:
            back_end_levy = np.nanmean(log_levy_areas[:, midpoint-1:], axis=1)
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
        seasonal_driver = np.full(n_dates, np.nan)
        
        # Need at least 1 year of data for meaningful seasonal analysis
        min_seasonal_window = min(252, n_dates // 2)  # 1 year or half the dataset
        
        if n_dates < min_seasonal_window:
            # Not enough data for seasonal analysis - return zeros
            return np.zeros(n_dates)
        
        # Calculate average Lévy area magnitude for each observation
        levy_magnitudes = np.sqrt(np.nansum(log_levy_areas**2, axis=1))
        
        # Get date information if available from curves
        if self.curves is not None and len(self.curves) == n_dates:
            dates = self.curves.index
        else:
            # Fallback: create artificial date sequence starting from a reference date
            dates = pd.date_range(start='2020-01-01', periods=n_dates, freq='D')
        
        # Create DataFrame for easier seasonal analysis
        seasonal_data = pd.DataFrame({
            'levy_magnitude': levy_magnitudes,
            'month': dates.month,
            'day_of_year': dates.dayofyear
        }, index=dates[:n_dates])
        
        # Minimum lookback for seasonal calculations (at least 2 years of monthly data)
        min_seasonal_lookback = min(504, n_dates)  # 2 years or available data
        
        # For each observation, calculate deviation from historical seasonal norm (avoiding lookahead)
        for i in range(n_dates):
            current_month = seasonal_data.iloc[i]['month']
            current_magnitude = seasonal_data.iloc[i]['levy_magnitude']
            
            if not np.isnan(current_magnitude):
                # Use only historical data up to current point for seasonal baseline
                # Need sufficient historical data for meaningful seasonal calculation
                if i >= min_seasonal_lookback:
                    # Get historical data for this month only (avoiding lookahead)
                    historical_data = seasonal_data.iloc[:i]  # Only past data
                    same_month_historical = historical_data[historical_data['month'] == current_month]['levy_magnitude']
                    
                    if len(same_month_historical) >= 10:  # Need at least 10 historical observations
                        # Calculate rolling seasonal baseline using only historical data
                        month_mean = same_month_historical.mean()
                        month_std = same_month_historical.std()
                        
                        if not np.isnan(month_mean) and month_std > 0:
                            # Calculate Z-score deviation from historical seasonal norm
                            seasonal_deviation = abs(current_magnitude - month_mean) / month_std
                            seasonal_driver[i] = seasonal_deviation
                        else:
                            seasonal_driver[i] = 0.0
                    else:
                        # Insufficient same-month historical data - use broader historical baseline
                        historical_magnitudes = historical_data['levy_magnitude'].dropna()
                        if len(historical_magnitudes) >= 50:  # Need reasonable historical sample
                            hist_mean = historical_magnitudes.mean()
                            hist_std = historical_magnitudes.std()
                            if hist_std > 0:
                                seasonal_driver[i] = abs(current_magnitude - hist_mean) / hist_std
                            else:
                                seasonal_driver[i] = 0.0
                        else:
                            seasonal_driver[i] = 0.0
                else:
                    # Insufficient data for seasonal analysis - set to 0
                    seasonal_driver[i] = 0.0
        
        # Apply causal smoothing to reduce noise while avoiding lookahead bias
        window_size = min(5, n_dates // 10)  # Small smoothing window
        if window_size > 1:
            # Backward-looking moving average smoothing (no lookahead)
            smoothed_driver = np.full_like(seasonal_driver, np.nan)
            for i in range(n_dates):
                if i >= window_size:
                    # Use only historical data (backward-looking window)
                    window_data = seasonal_driver[i-window_size:i+1]  # Include current but no future
                    valid_data = window_data[~np.isnan(window_data)]
                    if len(valid_data) > 0:
                        smoothed_driver[i] = np.mean(valid_data)
                    else:
                        smoothed_driver[i] = seasonal_driver[i]
                else:
                    # For early observations, use expanding window (still no lookahead)
                    window_data = seasonal_driver[:i+1]
                    valid_data = window_data[~np.isnan(window_data)]
                    if len(valid_data) > 0:
                        smoothed_driver[i] = np.mean(valid_data)
                    else:
                        smoothed_driver[i] = seasonal_driver[i]
            
            return smoothed_driver
        
        return seasonal_driver
    
    def _detect_regime_changes_from_levy(self, log_levy_areas: np.ndarray) -> np.ndarray:
        """
        Detect regime changes using log price Lévy areas
        
        Regime changes are identified as significant shifts in the distribution
        of Lévy areas, indicating fundamental changes in curve evolution dynamics
        """
        
        if log_levy_areas.size == 0:
            return np.array([])
        
        n_dates = log_levy_areas.shape[0]
        regime_changes = np.zeros(n_dates)
        
        # Use rolling statistics to detect regime shifts
        window = min(30, n_dates // 4)
        if window < 10:
            return regime_changes
        
        # Calculate rolling mean and std of Lévy area magnitudes
        levy_magnitudes = np.sqrt(np.nansum(log_levy_areas**2, axis=1))
        
        for i in range(window, n_dates - window):
            # Current window statistics
            current_window = levy_magnitudes[i:i+window]
            current_mean = np.nanmean(current_window)
            current_std = np.nanstd(current_window)
            
            # Past window statistics
            past_window = levy_magnitudes[i-window:i]
            past_mean = np.nanmean(past_window)
            past_std = np.nanstd(past_window)
            
            # Detect significant shifts
            if not (np.isnan(current_mean) or np.isnan(past_mean) or 
                   np.isnan(current_std) or np.isnan(past_std)):
                
                # Test for mean shift
                if past_std > 0:
                    z_score_mean = abs(current_mean - past_mean) / past_std
                    if z_score_mean > self.regime_threshold:
                        regime_changes[i] = 1
                
                # Test for volatility shift
                if past_std > 0 and current_std > 0:
                    vol_ratio = max(current_std / past_std, past_std / current_std)
                    if vol_ratio > 1.5:  # 50% volatility change
                        regime_changes[i] = max(regime_changes[i], 0.5)
        
        # Smooth regime changes
        smoothed_changes = np.convolve(
            regime_changes, 
            np.ones(self.min_regime_length) / self.min_regime_length, 
            mode='same'
        )
        
        return (smoothed_changes > 0.3).astype(float)
    
    def analyze_curve_evolution_drivers(self,
                                       window: int = None,
                                       constant_maturity: bool = False) -> Dict[str, Any]:
        """
        Comprehensive analysis of curve evolution drivers

        Parameters:
        -----------
        window : int, optional
            Analysis window size
        constant_maturity : bool, default False
            Use constant maturity tenors instead of sequential contracts

        Returns:
        --------
        Dict[str, Any]
            Complete analysis results including:
            - Path signatures and Lévy areas
            - Identified curve drivers
            - Regime change analysis
            - Driver importance rankings
        """

        if window is None:
            window = self.default_window

        # Setup constant maturity if requested
        if constant_maturity and self.constant_maturity_data is None:
            success = self.setup_constant_maturity()
            if not success:
                warnings.warn("Constant maturity setup failed, falling back to sequential contracts")
                constant_maturity = False

        # Calculate path signatures
        path_sig = self.calculate_path_signatures(window, constant_maturity=constant_maturity)
        
        # Analyze driver importance
        driver_importance = self._analyze_driver_importance(path_sig['curve_drivers'])
        
        # Cross-correlation analysis between drivers
        driver_correlations = self._calculate_driver_correlations(path_sig['curve_drivers'])
        
        # Regime transition analysis
        regime_analysis = self._analyze_regime_transitions(path_sig['regime_changes'])
        
        return {
            'path_signatures': path_sig,
            'driver_importance': driver_importance,
            'driver_correlations': driver_correlations,
            'regime_analysis': regime_analysis,
            'summary_statistics': {
                'n_curves': len(self.curves) if self.curves is not None else 0,
                'date_range': (self.curves.index.min(), self.curves.index.max()) if self.curves is not None else None,
                'n_regime_changes': np.sum(path_sig['regime_changes']) if path_sig['regime_changes'].size > 0 else 0,
                'primary_driver': max(driver_importance.items(), key=lambda x: x[1])[0] if driver_importance else None
            }
        }
    
    def _analyze_driver_importance(self, drivers: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze the importance of each curve driver"""
        
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
        
        # Detect regime changes in slope (contango/backwardation flips) - avoiding lookahead bias
        slope_changes = slopes.diff().abs()
        
        # Calculate expanding Z-scores to avoid lookahead bias
        slope_zscore = pd.Series(index=slope_changes.index, dtype=float)
        min_window = 60  # Need at least 60 observations for meaningful statistics
        
        for i in range(len(slope_changes)):
            if i >= min_window:
                # Use only historical data up to current point
                historical_changes = slope_changes.iloc[:i+1]
                hist_mean = historical_changes.mean()
                hist_std = historical_changes.std()
                
                if hist_std > 0:
                    slope_zscore.iloc[i] = (slope_changes.iloc[i] - hist_mean) / hist_std
                else:
                    slope_zscore.iloc[i] = 0.0
            else:
                slope_zscore.iloc[i] = 0.0
        
        # Find significant flips
        flip_points = slope_zscore > flip_threshold
        
        # Group consecutive flip periods
        flip_events = []
        in_flip = False
        flip_start = None
        flip_type = None
        
        for date, is_flip in flip_points.items():
            if is_flip and not in_flip:
                # Start of new flip
                in_flip = True
                flip_start = date
                # Determine flip type based on slope direction
                current_slope = slopes[date] if not np.isnan(slopes[date]) else 0
                previous_slope = slopes[:date].dropna().iloc[-2] if len(slopes[:date].dropna()) > 1 else 0
                
                if current_slope > previous_slope:
                    flip_type = 'backwardation_to_contango'
                else:
                    flip_type = 'contango_to_backwardation'
                    
            elif not is_flip and in_flip:
                # End of flip - check if duration is significant
                flip_duration = (date - flip_start).days
                
                if flip_duration >= min_flip_duration:
                    flip_magnitude = slope_changes[flip_start:date].max()
                    
                    flip_events.append({
                        'flip_start': flip_start,
                        'flip_end': date,
                        'flip_type': flip_type,
                        'flip_magnitude': flip_magnitude,
                        'flip_duration': flip_duration
                    })
                
                in_flip = False
        
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
            subplot_titles.append('Curve Drivers (4 Core)')
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
        self._add_front_month_plot(fig, current_row, dates, constant_maturity=constant_maturity)
        current_row += 1
        
        # 2. Back/Front spread
        self._add_back_front_spread_plot(fig, current_row, dates, constant_maturity=constant_maturity)
        current_row += 1
        
        # 3. Curve drivers
        if show_drivers and 'curve_drivers' in self.path_signatures:
            self._add_drivers_plot(fig, current_row, dates)
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
    
    def _add_drivers_plot(self, fig, row, dates):
        """Add curve drivers plot - shows 3 core curve drivers (excluding seasonal_deviations)"""
        
        drivers = self.path_signatures['curve_drivers']
        
        # Exclude seasonal_deviations from drivers plot (it goes to seasonal analysis)
        filtered_drivers = {k: v for k, v in drivers.items() if k != 'seasonal_deviations'}
        
        # Plot 3 core curve drivers (sorted by importance)
        driver_importance = self._analyze_driver_importance(filtered_drivers)
        core_drivers = sorted(driver_importance.items(), 
                           key=lambda x: x[1], reverse=True)  # Show core drivers only
        
        # Color palette for 3 core drivers
        colors = ['blue', 'red', 'green']
        
        for i, (driver_name, importance) in enumerate(core_drivers):
            if driver_name in drivers and drivers[driver_name].size > 0:
                driver_values = drivers[driver_name]
                valid_mask = ~np.isnan(driver_values)
                
                if np.sum(valid_mask) > 0:
                    # Ensure we align dates and driver values correctly
                    aligned_length = min(len(dates), len(driver_values))
                    aligned_dates = dates[:aligned_length]
                    aligned_driver_values = driver_values[:aligned_length]
                    aligned_valid_mask = valid_mask[:aligned_length]

                    valid_dates = aligned_dates[aligned_valid_mask]
                    valid_values = aligned_driver_values[aligned_valid_mask]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=valid_dates,
                            y=valid_values,
                            mode='lines',
                            name=f'{driver_name} ({importance:.3f})',
                            line=dict(color=colors[i % len(colors)])
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
    
    def _add_front_month_plot(self, fig, row, dates, constant_maturity=False):
        """Add front month (F0) price plot with synthetic spot and spot data using broadcasting"""

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
                return

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
                              height: int = 800) -> go.Figure:
        """
        Create 3D surface plot with date (x), days-to-expiry (y), and spread/price (z)
        
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
            3D surface plot
        """
        
        if self.curves is None or len(self.curves) == 0:
            raise ValueError("No curve data available for 3D surface plot")
        
        # Extract data matrices
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
        dates_numeric = np.array([d.timestamp() for d in dates_array])
        
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
        fig.update_scenes(
            xaxis=dict(
                tickmode='array',
                tickvals=dates_numeric[::max(1, len(dates_numeric)//10)],  # Show ~10 tick marks
                ticktext=[d.strftime('%Y-%m-%d') for d in dates_array[::max(1, len(dates_array)//10)]]
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


