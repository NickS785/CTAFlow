"""
Combined Futures Curve Analysis Framework
FuturesCurve and SpreadData are dataclasses for data representation
Analysis functionality is in dedicated analyzer classes
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, date, timedelta
from scipy import interpolate, stats
from scipy.stats import skew, kurtosis
import warnings

# Month code mappings
MONTH_CODE_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}
MONTH_CODE_ORDER = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']


@dataclass
class FuturesCurve:
    """
    Simple dataclass representing a futures curve snapshot at a specific point in time
    """
    ref_date: datetime
    curve_month_labels: List[str]
    prices: List[float]
    volumes: Optional[List[float]] = None
    open_interest: Optional[List[float]] = None
    days_to_expiry: Optional[List[int]] = None

    def __post_init__(self):
        """Validate data consistency"""
        if len(self.curve_month_labels) != len(self.prices):
            raise ValueError("Month codes and prices must have same length")
        
        if self.volumes is not None and len(self.volumes) != len(self.prices):
            raise ValueError("Volumes must have same length as prices")
        
        if self.open_interest is not None and len(self.open_interest) != len(self.prices):
            raise ValueError("Open interest must have same length as prices")
        
        if self.days_to_expiry is None:
            self.days_to_expiry = self._calculate_days_to_expiry()
    
    def _calculate_days_to_expiry(self) -> List[int]:
        """Calculate days to expiry for each contract"""
        dte_list = []
        for month_code in self.curve_month_labels:
            month_num = MONTH_CODE_MAP.get(month_code, 1)
            year = self.ref_date.year
            
            if month_num < self.ref_date.month:
                year += 1
            elif month_num == self.ref_date.month and self.ref_date.day > 15:
                year += 1
            
            expiry_date = datetime(year, month_num, 15)
            days_to_expiry = (expiry_date - self.ref_date).days
            dte_list.append(max(0, days_to_expiry))
        
        return dte_list


@dataclass
class SpreadData:
    """
    Simple dataclass for loading and storing futures curve data
    """
    symbol: str
    curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    oi: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    summary: Optional[Dict] = None
    CODE_TO_MONTH: Dict[str, int] = field(default_factory=lambda: MONTH_CODE_MAP)
    NUMBER_TO_CODE: Dict[int, str] = field(default_factory=lambda: {v: k for k, v in MONTH_CODE_MAP.items()})


@dataclass
class SpreadFeatures:
    """
    Enhanced dataclass containing futures curve and spread data
    """
    # Core DataFrames
    curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    seq_labels: pd.DataFrame = field(default_factory=pd.DataFrame)
    seq_spreads: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Volume and OI
    volume_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    oi_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    seq_volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    seq_oi: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    commodity: str = ""
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize derived data"""
        if not self.curve.empty:
            if self.seq_labels.empty:
                self._generate_seq_labels()
            if self.seq_spreads.empty:
                self._calculate_seq_spreads()
    
    def _generate_seq_labels(self):
        """Generate sequential labels from curve columns"""
        labels_data = []
        for idx in self.curve.index:
            row_labels = []
            for month_code in MONTH_CODE_ORDER:
                if month_code in self.curve.columns:
                    if pd.notna(self.curve.loc[idx, month_code]):
                        row_labels.append(month_code)
            labels_data.append(row_labels)
        
        self.seq_labels = pd.DataFrame({'labels': labels_data}, index=self.curve.index)
    
    def _calculate_seq_spreads(self):
        """Calculate sequential calendar spreads"""
        spreads_list = []
        
        for idx in self.curve.index:
            row_spreads = {}
            available_months = []
            
            for month_code in MONTH_CODE_ORDER:
                if month_code in self.curve.columns:
                    if pd.notna(self.curve.loc[idx, month_code]):
                        available_months.append(month_code)
            
            for i in range(len(available_months) - 1):
                front = available_months[i]
                back = available_months[i + 1]
                spread_name = f"{front}{back}"
                row_spreads[spread_name] = self.curve.loc[idx, back] - self.curve.loc[idx, front]
            
            spreads_list.append(row_spreads)
        
        self.seq_spreads = pd.DataFrame(spreads_list, index=self.curve.index)
        
        if not self.volume_curve.empty:
            self._calculate_spread_volume()
        if not self.oi_curve.empty:
            self._calculate_spread_oi()
    
    def _calculate_spread_volume(self):
        """Calculate volume for spreads"""
        volume_list = []
        
        for idx in self.volume_curve.index:
            row_volume = {}
            for spread_col in self.seq_spreads.columns:
                if len(spread_col) == 2:
                    front, back = spread_col[0], spread_col[1]
                    if front in self.volume_curve.columns and back in self.volume_curve.columns:
                        front_vol = self.volume_curve.loc[idx, front]
                        back_vol = self.volume_curve.loc[idx, back]
                        if pd.notna(front_vol) and pd.notna(back_vol):
                            row_volume[spread_col] = min(front_vol, back_vol)
            volume_list.append(row_volume)
        
        self.seq_volume = pd.DataFrame(volume_list, index=self.volume_curve.index)
    
    def _calculate_spread_oi(self):
        """Calculate OI for spreads"""
        oi_list = []
        
        for idx in self.oi_curve.index:
            row_oi = {}
            for spread_col in self.seq_spreads.columns:
                if len(spread_col) == 2:
                    front, back = spread_col[0], spread_col[1]
                    if front in self.oi_curve.columns and back in self.oi_curve.columns:
                        front_oi = self.oi_curve.loc[idx, front]
                        back_oi = self.oi_curve.loc[idx, back]
                        if pd.notna(front_oi) and pd.notna(back_oi):
                            row_oi[spread_col] = min(front_oi, back_oi)
            oi_list.append(row_oi)
        
        self.seq_oi = pd.DataFrame(oi_list, index=self.oi_curve.index)
    
    def get_sequentialized_curve(self, roll_on: str = 'calendar') -> pd.DataFrame:
        """Get curve in sequential order based on roll criteria"""
        seq_data = []
        
        for idx in self.curve.index:
            if roll_on == 'volume' and not self.volume_curve.empty:
                row_data = []
                vol_pairs = []
                for month_code in self.curve.columns:
                    if pd.notna(self.curve.loc[idx, month_code]):
                        vol = self.volume_curve.loc[idx, month_code] if month_code in self.volume_curve.columns else 0
                        vol_pairs.append((self.curve.loc[idx, month_code], vol))
                vol_pairs.sort(key=lambda x: x[1], reverse=True)
                row_data = [p for p, _ in vol_pairs]
                
            elif roll_on == 'oi' and not self.oi_curve.empty:
                row_data = []
                oi_pairs = []
                for month_code in self.curve.columns:
                    if pd.notna(self.curve.loc[idx, month_code]):
                        oi = self.oi_curve.loc[idx, month_code] if month_code in self.oi_curve.columns else 0
                        oi_pairs.append((self.curve.loc[idx, month_code], oi))
                oi_pairs.sort(key=lambda x: x[1], reverse=True)
                row_data = [p for p, _ in oi_pairs]
                
            else:
                row_data = []
                for month_code in MONTH_CODE_ORDER:
                    if month_code in self.curve.columns:
                        value = self.curve.loc[idx, month_code]
                        if pd.notna(value):
                            row_data.append(value)
            
            seq_data.append(row_data)
        
        max_months = max(len(row) for row in seq_data) if seq_data else 0
        columns = [f'M{i+1}' for i in range(max_months)]
        
        seq_df = pd.DataFrame(seq_data, index=self.curve.index)
        if not seq_df.empty:
            seq_df.columns = columns[:len(seq_df.columns)]
        
        return seq_df


class CurveAnalyzer:
    """
    Analyzer for FuturesCurve objects - extracts shape features and metrics
    """
    
    def __init__(self, futures_curve: FuturesCurve):
        self.curve = futures_curve
        self.valid_prices = [(i, p) for i, p in enumerate(self.curve.prices) if not np.isnan(p)]
    
    def get_slope(self) -> float:
        """Calculate overall curve slope"""
        if len(self.valid_prices) < 2:
            return 0.0
        
        prices = [p for _, p in self.valid_prices]
        return (prices[-1] - prices[0]) / len(prices)
    
    def get_granular_slope(self, method='linear_regression', interpolate=True) -> Dict[str, float]:
        """Calculate granular slope using various methods"""
        if len(self.valid_prices) < 2:
            return {'slope': 0.0, 'valid_points': len(self.valid_prices)}
        
        positions = np.array([i for i, _ in self.valid_prices])
        prices = np.array([p for _, p in self.valid_prices])
        
        result = {'valid_points': len(self.valid_prices)}
        
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
            
            if degree >= 2:
                second_deriv = np.polyder(poly, 2)
                result['curvature'] = float(second_deriv(positions.mean()))
            
            result['slope'] = float(np.polyder(poly)(positions.mean()))
            result['polynomial_coeffs'] = coeffs.tolist()
        
        elif method == 'piecewise':
            mid_point = len(prices) // 2
            front_slope = (prices[mid_point] - prices[0]) / (positions[mid_point] - positions[0])
            back_slope = (prices[-1] - prices[mid_point]) / (positions[-1] - positions[mid_point])
            
            result.update({
                'front_slope': front_slope,
                'back_slope': back_slope,
                'slope_difference': back_slope - front_slope
            })
        
        return result
    
    def calculate_term_structure_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive term structure metrics"""
        metrics = {}
        prices = [p for _, p in self.valid_prices]
        
        if len(prices) < 2:
            return metrics
        
        # Basic shape metrics
        metrics['mean_level'] = np.mean(prices)
        metrics['price_dispersion'] = np.std(prices)
        metrics['skewness'] = skew(prices)
        metrics['kurtosis'] = kurtosis(prices)
        
        # Contango/Backwardation
        price_diffs = np.diff(prices)
        metrics['contango_ratio'] = np.sum(price_diffs > 0) / len(price_diffs) if len(price_diffs) > 0 else 0
        metrics['avg_calendar_spread'] = np.mean(price_diffs) if len(price_diffs) > 0 else 0
        
        # Convexity
        if len(prices) >= 3:
            second_diffs = np.diff(price_diffs)
            metrics['convexity'] = np.mean(second_diffs)
            metrics['max_convexity'] = np.max(np.abs(second_diffs))
        
        # Volume/OI concentration
        if self.curve.volumes:
            valid_vols = [v for v in self.curve.volumes if not np.isnan(v)]
            if valid_vols and sum(valid_vols) > 0:
                vol_shares = np.array(valid_vols) / sum(valid_vols)
                metrics['volume_concentration'] = np.sum(vol_shares ** 2)
        
        return metrics


class SpreadAnalyzer:
    """
    Analyzer for SpreadFeatures - handles spreads, seasonality, and advanced features
    """
    
    def __init__(self, spread_features: SpreadFeatures):
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


class IntradayFeatures:
    """
    Intraday features calculation
    """
    
    def __init__(self, data: pd.DataFrame, close_col="Last", bid_volume="BidVolume", 
                 ask_volume="AskVolume", volume="Volume"):
        self.data = data
        self.returns = np.log(data[close_col]) - np.log(data[close_col].shift(1))
        self.buy_vol = data[ask_volume] if ask_volume in data.columns else None
        self.sell_vol = data[bid_volume] if bid_volume in data.columns else None
        self.volume = data[volume] if volume in data.columns else None
    
    def historical_rv(self, window=21, average=True, annualize=False):
        """Calculate historical realized volatility"""
        returns = self.returns
        
        dr = pd.bdate_range(returns.index.date[0], returns.index.date[-1])
        rv = np.zeros(len(dr))
        rv[:] = np.nan
        
        denom = window if average else 1
        
        for idx in range(window, len(dr)):
            d_start = dr[idx - window]
            d_end = dr[idx]
            rv[idx] = np.sqrt(np.sum(returns.loc[d_start:d_end] ** 2)) / denom
        
        if annualize:
            rv *= np.sqrt(252)
        
        hrv = pd.Series(data=rv, index=dr, name=f'{window}_rv')
        return hrv
    
    def realized_semivariance(self, window=1, average=True):
        """Calculate realized semivariance"""
        returns = self.returns
        data = []
        dr = pd.bdate