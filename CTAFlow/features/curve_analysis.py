"""
Advanced Futures Curve Analysis Framework
Incorporates FuturesCurve, SpreadData, and Lévy Area/Path Signature features
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from datetime import datetime, date, timedelta
from scipy import interpolate, stats
from scipy.stats import skew, kurtosis
import warnings
import calendar

# Import data client if available
try:
    from ..data.data_client import DataClient
    from ..data.futures_curve_manager import MONTH_CODE_MAP
except ImportError:
    # Fallback definitions for standalone use
    MONTH_CODE_MAP = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    DataClient = None

MONTH_CODE_ORDER = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']


def _is_empty(data) -> bool:
    """Check if data structure is empty (works for both DataFrames and numpy arrays)"""
    if hasattr(data, 'empty'):
        return data.empty
    elif hasattr(data, 'size'):
        return data.size == 0
    elif hasattr(data, '__len__'):
        return len(data) == 0
    else:
        return data is None


def _safe_get_value(data, row_idx, col_key, default=np.nan):
    """Safely get value from data structure (DataFrame or array)"""
    try:
        if hasattr(data, 'index'):
            return data.loc[row_idx, col_key]
        elif hasattr(data, '__getitem__'):
            return data[row_idx, col_key] if hasattr(data, 'ndim') and data.ndim > 1 else data[row_idx]
        else:
            return default
    except (KeyError, IndexError, TypeError):
        return default


def _safe_get_columns(data):
    """Safely get columns from data structure"""
    if hasattr(data, 'columns'):
        return data.columns
    elif hasattr(data, 'shape') and len(data.shape) > 1:
        return list(range(data.shape[1]))
    else:
        return []


def _safe_get_index(data):
    """Safely get index from data structure"""
    if hasattr(data, 'index'):
        return data.index
    elif hasattr(data, 'shape'):
        return list(range(data.shape[0]))
    else:
        return []


def _safe_check_column(data, col_key) -> bool:
    """Check if column exists in data structure"""
    if hasattr(data, 'columns'):
        return col_key in data.columns
    elif hasattr(data, 'shape') and len(data.shape) > 1:
        return isinstance(col_key, int) and 0 <= col_key < data.shape[1]
    else:
        return False


@dataclass
class ExpiryTracker:
    """
    Tracks expiry dates and rolls for futures contracts
    """
    symbol: str
    month_code: str
    year: int
    expiry_date: Optional[datetime] = None
    days_to_expiry: Optional[int] = None
    is_active: bool = True
    roll_date: Optional[datetime] = None

    def __post_init__(self):
        if self.expiry_date is None:
            self.expiry_date = self._calculate_expiry_date()

        if self.days_to_expiry is None and self.expiry_date:
            self.days_to_expiry = (self.expiry_date - datetime.now()).days

    def _calculate_expiry_date(self) -> datetime:
        """Calculate expiry date based on standard rules"""
        month_num = MONTH_CODE_MAP.get(self.month_code, 1)

        # Default expiry rules (can be customized per commodity)
        expiry_rules = {
            'CL': {'day': 25, 'offset': -3},  # Crude: 3 business days before 25th
            'NG': {'day': -3, 'offset': 0},  # Natural Gas: 3 business days before last
            'C': {'day': 15, 'offset': -1},  # Corn: business day prior to 15th
            'S': {'day': 15, 'offset': -1},  # Soybeans
            'W': {'day': 15, 'offset': -1},  # Wheat
        }

        # Extract commodity code from symbol
        commodity = ''.join([c for c in self.symbol if c.isalpha()])[:2]

        if commodity in expiry_rules:
            rule = expiry_rules[commodity]
            if rule['day'] > 0:
                exp_date = datetime(self.year, month_num, rule['day'])
            else:
                last_day = calendar.monthrange(self.year, month_num)[1]
                exp_date = datetime(self.year, month_num, last_day) + timedelta(days=rule['day'])

            # Apply business day offset
            if rule['offset'] != 0:
                exp_date = self._add_business_days(exp_date, rule['offset'])

            return exp_date

        # Default: 15th of the month
        return datetime(self.year, month_num, 15)

    def _add_business_days(self, date: datetime, days: int) -> datetime:
        """Add business days to a date"""
        delta = abs(days)
        direction = 1 if days > 0 else -1

        while delta > 0:
            date += timedelta(days=direction)
            if date.weekday() < 5:  # Monday = 0, Friday = 4
                delta -= 1

        return date


@dataclass
class SpreadFeature:
    dtype: type = float
    sequential : bool = False
    price_data : np.array = None
    labels : List[str] = None
    direction = "vertical"
    index : pd.Index = None



class FuturesCurve(SpreadFeature):
    """
    Single snapshot of a futures curve at a specific point in time
    """
    ref_date: date
    curve_month_labels: Set[str]
    prices: np.array
    volumes: Optional[np.array] = None
    open_interest : Optional[np.array] = None
    sequential:bool = False
    days_to_expiration : np.array = None

    def __post_init__(self):
        """Validate and process curve data after initialization"""
        if len(self.curve_month_labels) != len(self.prices):
            raise ValueError("Month codes and prices must have same length")

        if self.volumes is not None and len(self.volumes) != len(self.prices):
            raise ValueError("Volumes must have same length as prices")

        if self.open_interest is not None and len(self.open_interest) != len(self.prices):
            raise ValueError("Open interest must have same length as prices")

        if not self.sequential:
            self.sequence_curve()
            self.sequential = True


        if self.days_to_expiry is None:
            self.days_to_expiry = self._calculate_days_to_expiry()

        self.direction = "horizontal"

    def _calculate_days_to_expiry(self) -> List[int]:
        """Calculate days to expiry for each contract"""
        dte_list = []
        for month_code in self.curve_month_labels:
            month_num = MONTH_CODE_MAP.get(month_code, 1)
            year = self.ref_date.year

            # Infer year based on month progression
            if month_num < self.ref_date.month:
                year += 1
            elif month_num == self.ref_date.month and self.ref_date.day > 15:
                year += 1

            expiry_date = datetime(year, month_num, 15)
            days_to_expiry = (expiry_date - self.ref_date).days
            dte_list.append(max(0, days_to_expiry))
        self.days_to_expiry = np.array(dte_list)
        return dte_list

    def sequence_curve(self, roll_on: str = 'volume') -> np.ndarray:
        """
        Get prices in sequential order based on roll criteria

        Parameters:
        - roll_on: 'volume', 'oi', or 'calendar' (default order)
        """
        # If seq_prices is provided, use it directly
        if self.seq_prices is not None:
            return np.array([p for p in self.seq_prices if not np.isnan(p)])

        if roll_on == 'calendar':
            # Return prices in calendar month order
            return np.array([p for p in self.prices if not np.isnan(p)])

        elif roll_on == 'volume' and self.volumes:
            # Sort by volume (highest first)
            paired = [(p, v, i) for i, (p, v) in enumerate(zip(self.prices, self.volumes))
                      if not np.isnan(p) and not np.isnan(v)]
            paired.sort(key=lambda x: x[1], reverse=True)
            return np.array([p for p, _, _ in paired])

        elif roll_on == 'oi' and self.open_interest:
            # Sort by open interest (highest first)
            paired = [(p, oi, i) for i, (p, oi) in enumerate(zip(self.prices, self.open_interest))
                      if not np.isnan(p) and not np.isnan(oi)]
            paired.sort(key=lambda x: x[1], reverse=True)
            return np.array([p for p, _, _ in paired])

        else:
            # Fallback to calendar order
            return self.sequence_curve('calendar')


    def get_granular_slope(self, method='linear_regression', interpolate=True) -> Dict[str, float]:
        """
        Calculate granular slope using various methods

        Enhanced from original to include path-based features
        """
        valid_prices = [(i, p, self.curve_month_labels[i]) for i, p in enumerate(self.prices)
                        if not np.isnan(p)]

        if len(valid_prices) < 2:
            return {'slope': 0.0, 'valid_points': len(valid_prices)}

        positions = np.array([v[0] for v in valid_prices])
        prices = np.array([v[1] for v in valid_prices])

        result = {'valid_points': len(valid_prices)}

        if method == 'linear_regression':
            slope, intercept, r_value, p_value, std_err = stats.linregress(positions, prices)
            result.update({
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err
            })

        elif method == 'polynomial':
            degree = min(3, len(prices) - 1)
            coeffs = np.polyfit(positions, prices, degree)
            poly = np.poly1d(coeffs)

            # Calculate curvature (second derivative)
            if degree >= 2:
                second_deriv = np.polyder(poly, 2)
                result['curvature'] = float(second_deriv(positions.mean()))

            result['slope'] = float(np.polyder(poly)(positions.mean()))
            result['polynomial_coeffs'] = coeffs.tolist()

        # Add shape complexity using entropy
        if len(prices) >= 3:
            price_changes = np.diff(prices)
            if price_changes.std() > 0:
                normalized_changes = (price_changes - price_changes.mean()) / price_changes.std()
                # Calculate approximate entropy
                result['shape_entropy'] = -np.sum(np.abs(normalized_changes) *
                                                  np.log(np.abs(normalized_changes) + 1e-10))

        return result

    def calculate_term_structure_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive term structure metrics
        """
        metrics = {}
        valid_prices = [p for p in self.prices if not np.isnan(p)]

        if len(valid_prices) < 2:
            return metrics

        # Basic shape metrics
        metrics['mean_level'] = np.mean(valid_prices)
        metrics['price_dispersion'] = np.std(valid_prices)
        metrics['skewness'] = skew(valid_prices)
        metrics['kurtosis'] = kurtosis(valid_prices)

        # Contango/Backwardation metrics
        price_diffs = np.diff(valid_prices)
        metrics['contango_ratio'] = np.sum(price_diffs > 0) / len(price_diffs) if len(price_diffs) > 0 else 0
        metrics['avg_calendar_spread'] = np.mean(price_diffs) if len(price_diffs) > 0 else 0

        # Term structure convexity
        if len(valid_prices) >= 3:
            second_diffs = np.diff(price_diffs)
            metrics['convexity'] = np.mean(second_diffs)
            metrics['max_convexity'] = np.max(np.abs(second_diffs))

        # Volume/OI concentration (Herfindahl index)
        if self.volumes:
            valid_vols = [v for v in self.volumes if not np.isnan(v)]
            if valid_vols and sum(valid_vols) > 0:
                vol_shares = np.array(valid_vols) / sum(valid_vols)
                metrics['volume_concentration'] = np.sum(vol_shares ** 2)

        if self.open_interest:
            valid_oi = [o for o in self.open_interest if not np.isnan(o)]
            if valid_oi and sum(valid_oi) > 0:
                oi_shares = np.array(valid_oi) / sum(valid_oi)
                metrics['oi_concentration'] = np.sum(oi_shares ** 2)

        return metrics


class Contract (SpreadFeature):

    symbol : str = ""
    price_data : np.array = []
    label : str = ""
    volume : np.array = []
    dte : np.array = []
    index : pd.Index = None
    continuous :bool = False
    is_front_month :bool = False
    early_roll_days = 45
    direction =  "vertical"

    tracker = ExpiryTracker(symbol, month_code=label, year=date.today().year, days_to_expiry=dte )
    expiration_date = tracker.expiry_date


    @property

    def _is_rolled(self):

        return self.dte < self.early_roll_days

    def _is_expired(self):

        return self.dte < 0

    def roll_over(self):

        self.expiry_date = self.expiry_date + timedelta(days=365)

        return


class SpreadData:
    """
    Data loading and management class with volume-based sequentialization
    """
    symbol:str = None
    curve: np.array = None
    seq_labels: np.array = None
    seq_prices: np.array = None
    seq_spreads : np.array = None

    # Volume and OI
    volume_curve: np.array = None
    oi_curve: np.array = None
    seq_volume: np.array = None
    seq_oi: np.array = None

    # Metadata
    commodity: str = ""
    index: pd.DatetimeIndex = None

    def __init__(self, symbol, **params):
        self.symbol = symbol
        self.commodity = symbol  # For compatibility
        self.contracts = {}

        if hasattr(self, 'index'):
            self.timestamps = self.index

        if _is_empty(self.index) and self.symbol:
            self.load_from_client()


        # Initialize data client
        self.CODE_TO_MONTH = MONTH_CODE_MAP
        self.NUMBER_TO_CODE = {v: k for k, v in MONTH_CODE_MAP.items()}

        # Initialize derived features (like SpreadFeatures.__post_init__)
        self._initialize_features()

    def _initialize_features(self):
        """Initialize and calculate derived features (from SpreadFeatures.__post_init__)"""
        if not _is_empty(self.curve):
            if _is_empty(self.seq_labels):
                self._generate_seq_labels()
            if _is_empty(self.seq_spreads):
                self._calculate_seq_spreads()
            # Calculate Lévy areas if we have time series
            if len(self.curve) > 1:
                self._calculate_levy_areas()
            # Calculate path signatures
            self.calculate_path_signatures()

    def create_futures_curve(self, date: Optional[datetime] = None) -> FuturesCurve:
        """
        Create a FuturesCurve object for a specific date
        """
        if date is None:
            date = datetime.now()

        if not hasattr(self, 'curve'):
            raise ValueError("No curve data available")

        # Get data for specific date
        if date in set(self.index.tolist()):
            price_row = self.curve[self.index == date]
        else:
            # Get nearest date
            time_diff = np.abs(self.curve.index - date)
            nearest_idx = time_diff.argmin()
            price_row = self.curve[nearest_idx]

        # Extract month labels and prices
        month_labels = []
        prices = []
        volumes = []
        ois = []

        for month_code in MONTH_CODE_ORDER:
            if month_code in price_row.index:
                price = price_row[month_code]
                if pd.notna(price):
                    month_labels.append(month_code)
                    prices.append(float(price))

                    # Add volume if available
                    if hasattr(self, 'volume') and month_code in self.volume.columns:
                        if date in self.volume.index:
                            vol = _safe_get_value(self.volume)
                        else:
                            vol = self.volume.iloc[nearest_idx][month_code]
                        volumes.append(float(vol) if pd.notna(vol) else np.nan)
                    else:
                        volumes.append(np.nan)

                    # Add OI if available
                    if hasattr(self, 'oi') and month_code in self.oi.columns:
                        if date in self.oi.index:
                            oi = self.oi.loc[date, month_code]
                        else:
                            oi = self.oi.iloc[nearest_idx][month_code]
                        ois.append(float(oi) if pd.notna(oi) else np.nan)
                    else:
                        ois.append(np.nan)

        return FuturesCurve(
            ref_date=date,
            curve_month_labels=month_labels,
            prices=prices,
            volumes=volumes if any(not np.isnan(v) for v in volumes) else None,
            open_interest=ois if any(not np.isnan(o) for o in ois) else None
        )

    def _generate_seq_labels(self):
        """Generate sequential labels from curve columns"""
        labels_data = []
        index_data = _safe_get_index(self.curve)
        
        for idx in index_data:
            row_labels = []
            for month_code in MONTH_CODE_ORDER:
                if _safe_check_column(self.curve, month_code):
                    if pd.notna(_safe_get_value(self.curve, idx, month_code)):
                        row_labels.append(month_code)
            labels_data.append(row_labels)

        self.seq_labels = pd.DataFrame({'labels': labels_data}, index=index_data)

    def _calculate_seq_spreads(self):
        """Calculate sequential calendar spreads using pre-loaded seq_labels when available"""
        spreads_list = []
        index_data = _safe_get_index(self.curve)

        for idx in index_data:
            row_spreads = {}
            available_months = []

            # First try to use pre-loaded sequential labels to avoid redundant calculation
            seq_labels = None
            if not _is_empty(self.seq_labels) and idx in self.seq_labels.index:
                labels_row = self.seq_labels.loc[idx]
                if 'labels' in labels_row:
                    actual_labels = labels_row['labels']
                    if isinstance(actual_labels, list) and len(actual_labels) > 0:
                        seq_labels = actual_labels

            if seq_labels is not None:
                # Use pre-loaded sequence order
                for month_code in seq_labels:
                    if _safe_check_column(self.curve, month_code):
                        if pd.notna(_safe_get_value(self.curve, idx, month_code)):
                            available_months.append(month_code)
            else:
                # Fallback to standard calendar order
                for month_code in MONTH_CODE_ORDER:
                    if _safe_check_column(self.curve, month_code):
                        if pd.notna(_safe_get_value(self.curve, idx, month_code)):
                            available_months.append(month_code)

            # Calculate spreads (back - front)
            for i in range(len(available_months) - 1):
                front = available_months[i]
                back = available_months[i + 1]
                spread_name = f"{front}{back}"
                front_val = _safe_get_value(self.curve, idx, front)
                back_val = _safe_get_value(self.curve, idx, back)
                row_spreads[spread_name] = back_val - front_val

            spreads_list.append(row_spreads)

        self.seq_spreads = pd.DataFrame(spreads_list, index=index_data)

        # Calculate spread volumes/OI
        if not _is_empty(self.volume_curve):
            self._calculate_spread_volume()
        if not _is_empty(self.oi_curve):
            self._calculate_spread_oi()

    def _calculate_spread_volume(self):
        """Calculate volume for spreads (minimum of legs)"""
        volume_list = []
        vol_index = _safe_get_index(self.volume_curve)

        for idx in vol_index:
            row_volume = {}
            spread_cols = _safe_get_columns(self.seq_spreads)
            for spread_col in spread_cols:
                if len(str(spread_col)) == 2:
                    front, back = str(spread_col)[0], str(spread_col)[1]
                    if _safe_check_column(self.volume_curve, front) and _safe_check_column(self.volume_curve, back):
                        front_vol = _safe_get_value(self.volume_curve, idx, front)
                        back_vol = _safe_get_value(self.volume_curve, idx, back)
                        if pd.notna(front_vol) and pd.notna(back_vol):
                            row_volume[spread_col] = min(front_vol, back_vol)
            volume_list.append(row_volume)

        self.seq_volume = pd.DataFrame(volume_list, index=vol_index)

    def _calculate_spread_oi(self):
        """Calculate OI for spreads"""
        oi_list = []
        oi_index = _safe_get_index(self.oi_curve)

        for idx in oi_index:
            row_oi = {}
            spread_cols = _safe_get_columns(self.seq_spreads)
            for spread_col in spread_cols:
                if len(str(spread_col)) == 2:
                    front, back = str(spread_col)[0], str(spread_col)[1]
                    if _safe_check_column(self.oi_curve, front) and _safe_check_column(self.oi_curve, back):
                        front_oi = _safe_get_value(self.oi_curve, idx, front)
                        back_oi = _safe_get_value(self.oi_curve, idx, back)
                        if pd.notna(front_oi) and pd.notna(back_oi):
                            row_oi[spread_col] = min(front_oi, back_oi)
            oi_list.append(row_oi)

        self.seq_oi = pd.DataFrame(oi_list, index=oi_index)

    def load_from_client(self):
        cli = DataClient()
        curve_data = cli.query_curve_data(self.symbol)
        for k, val in curve_data.items():
            if k == 'curve':
                contracts_data = curve_data[k]
                for i, col in enumerate(contracts_data.columns):
                    self.index = contracts_data.index
                    self.contracts.update({
                        col:Contract(self.symbol,
                                     price_data=contracts_data.loc[:, col].values,
                                     labels=col,
                                     index= contracts_data.index)
                    })
            else:
                if isinstance(val, pd.DataFrame):
                    self.__setattr__(k, val.values)

    def __getitem__(self, key) -> Union[FuturesCurve, Tuple[FuturesCurve, FuturesCurve]]:
        """
        Get FuturesCurve object(s) based on datetime indexing

        Parameters:
        - key: datetime, int, or slice
            - datetime: Returns FuturesCurve for that date
            - int: Returns FuturesCurve for that row index
            - slice: Returns tuple of (start FuturesCurve, end FuturesCurve)
        """
        if isinstance(key, datetime):
            # Direct datetime access
            return self.create_futures_curve(key)

        elif isinstance(key, int):
            # Integer index access
            if not hasattr(self, 'index') or len(self.index) == 0:
                raise IndexError("No index data available")

            if key < 0:
                key = len(self.index) + key  # Handle negative indexing

            if key >= len(self.index) or key < 0:
                raise IndexError(f"Index {key} out of range")

            date = self.index[key]
            return self.create_futures_curve(date)

        elif isinstance(key, slice):
            # Slice access - return start and end curves
            if not hasattr(self, 'index') or len(self.index) == 0:
                raise IndexError("No index data available")

            start_idx = key.start if key.start is not None else 0
            stop_idx = key.stop if key.stop is not None else len(self.index) - 1

            # Handle negative indexing
            if start_idx < 0:
                start_idx = len(self.index) + start_idx
            if stop_idx < 0:
                stop_idx = len(self.index) + stop_idx

            # Validate bounds
            start_idx = max(0, min(start_idx, len(self.index) - 1))
            stop_idx = max(0, min(stop_idx, len(self.index) - 1))

            start_date = self.index[start_idx]
            end_date = self.index[stop_idx]

            curves = []
            for date in self.index[start_date:end_date].tolist():
                curves.append(
                    self.cr
                )

            return (start_curve, end_curve)

        else:
            raise TypeError(f"Unsupported index type: {type(key)}")


class CurveShapeAnalyzer:
    """
    Analyzes curve shapes and extracts features from SpreadData
    """

    def __init__(self, spread_data: SpreadData):
        self.sf = spread_data
        self._current_prices = None
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

    def get_shape_features(self) -> Dict[str, float]:
        """Extract comprehensive shape features"""
        if not self._current_prices:
            return {}

        prices = np.array(list(self._current_prices.values()))
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
        """Calculate annualized roll yield"""
        if len(self._current_prices) < 2:
            return 0.0

        # Get front two contracts
        month_codes = list(self._current_prices.keys())[:2]
        front_price = self._current_prices[month_codes[0]]
        next_price = self._current_prices[month_codes[1]]

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
        if len(self._current_prices) < 3:
            return 0.0

        prices = np.array(list(self._current_prices.values()))
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
        self.history: List[FuturesCurve] = []
        self.shape_history: List[Dict] = []
        self.timestamps: List[datetime] = []


    def add_snapshot(self, futures_curve: FuturesCurve):
        """Add a FuturesCurve snapshot to history"""
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
        
        # Use seq_prices if available, otherwise regular prices
        prices = curve.seq_prices if curve.seq_prices is not None else curve.prices
        
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
        
        # Volume concentration if available
        if curve.volumes is not None and len(curve.volumes) > 0:
            valid_vols = [v for v in curve.volumes if not np.isnan(v)]
            if valid_vols and sum(valid_vols) > 0:
                vol_shares = np.array(valid_vols) / sum(valid_vols)
                features['volume_concentration'] = np.sum(vol_shares ** 2)
        
        # OI concentration if available
        if curve.open_interest is not None and len(curve.open_interest) > 0:
            valid_oi = [o for o in curve.open_interest if not np.isnan(o)]
            if valid_oi and sum(valid_oi) > 0:
                oi_shares = np.array(valid_oi) / sum(valid_oi)
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
        self.levy_areas = pd.DataFrame()
        self.signatures = pd.DataFrame()
        self.seasonal_cache = {}

    def calculate_levy_areas(self, window: int = 20) -> pd.DataFrame:
        """Calculate Lévy areas between contract pairs"""
        levy_dict = {}

        for spread_name in self.sf.seq_spreads.columns:
            if len(spread_name) == 2:
                front, back = spread_name[0], spread_name[1]

                if front in self.sf.curve.columns and back in self.sf.curve.columns:
                    front_prices = self.sf.curve[front].values
                    back_prices = self.sf.curve[back].values

                    levy_area = self._compute_levy_area(front_prices, back_prices, window)
                    levy_dict[f'levy_{spread_name}'] = levy_area

        self.levy_areas = pd.DataFrame(levy_dict, index=self.sf.curve.index)
        return self.levy_areas

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

    def calculate_path_signatures(self, depth: int = 3) -> pd.DataFrame:
        """Calculate path signatures up to specified depth"""
        signatures = {}
        seq_curve = self.sf.get_sequentialized_curve()

        if seq_curve.empty or len(seq_curve.columns) < 2:
            return pd.DataFrame()

        # First-order: increments
        for col in seq_curve.columns:
            increments = seq_curve[col].diff()
            signatures[f'sig1_{col}'] = increments

        # Second-order: Lévy areas
        for i in range(len(seq_curve.columns) - 1):
            col1, col2 = seq_curve.columns[i], seq_curve.columns[i+1]
            levy = self._compute_levy_area(
                seq_curve[col1].values,
                seq_curve[col2].values,
                window=20
            )
            signatures[f'sig2_{col1}_{col2}'] = levy

        # Third-order if requested
        if depth >= 3 and len(seq_curve.columns) >= 3:
            for i in range(len(seq_curve.columns) - 2):
                cols = seq_curve.columns[i:i+3]
                triple = (seq_curve[cols[0]].diff() *
                         seq_curve[cols[1]].diff() *
                         seq_curve[cols[2]].diff())
                signatures[f'sig3_{"_".join(cols)}'] = triple

        self.signatures = pd.DataFrame(signatures, index=seq_curve.index)
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
        # Select data source
        if data_type == 'spreads':
            data = self.sf.seq_spreads
        elif data_type == 'returns':
            data = self.sf.curve.pct_change()
        elif data_type == 'prices':
            data = self.sf.curve
        elif data_type == 'levy_areas':
            if self.levy_areas.empty:
                self.calculate_levy_areas()
            data = self.levy_areas
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
        if data_type == 'spreads':
            data = self.sf.seq_spreads
        elif data_type == 'returns':
            data = self.sf.curve.pct_change()
        else:
            data = self.sf.curve

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
        if self.levy_areas.empty:
            self.calculate_levy_areas()

        if self.levy_areas.empty:
            return pd.DataFrame()

        regime_changes = []

        for col in self.levy_areas.columns:
            levy_series = self.levy_areas[col].dropna()

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
