"""
Roll Date Management System for Futures Curve Manager

Tracks, validates, and persists roll dates to prevent early rolls and missing back months.
Provides configurable roll timing based on volume, open interest, and calendar rules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import json
from pathlib import Path

from CTAFlow.config import MODEL_DATA_PATH


class RollDateManager:
    """
    Advanced roll date management system that tracks optimal roll timing
    and prevents early rolls that cause missing back months.
    """
    
    def __init__(self, symbol: str, data_path: Optional[str] = None):
        self.symbol = symbol.upper()
        self.data_path = Path(data_path or MODEL_DATA_PATH) / 'roll_dates'
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Roll date storage
        self.roll_dates_file = self.data_path / f"{self.symbol}_roll_dates.json"
        self.roll_history: Dict[str, Dict] = self.load_roll_history()
        
        # Configurable roll criteria
        self.roll_criteria = {
            'min_days_to_expiry': 5,        # Minimum days before expiry to roll
            'max_days_to_expiry': 12,       # Maximum days before expiry to roll
            'volume_threshold': 0.3,        # Minimum volume ratio (new/old) to trigger roll
            'oi_threshold': 0.4,            # Minimum OI ratio (new/old) to trigger roll
            'price_stability_days': 3,      # Days of stable pricing before roll
            'back_month_min_volume': 100,   # Minimum volume to keep contract as back month
        }
        
    def load_roll_history(self) -> Dict[str, Dict]:
        """Load historical roll dates from disk"""
        if self.roll_dates_file.exists():
            try:
                with open(self.roll_dates_file, 'r') as f:
                    data = json.load(f)
                    # Convert date strings back to datetime keys
                    return {k: {
                        'roll_date': pd.to_datetime(v['roll_date']),
                        'from_contract': v['from_contract'],
                        'to_contract': v['to_contract'],
                        'days_to_expiry': v['days_to_expiry'],
                        'volume_ratio': v.get('volume_ratio'),
                        'oi_ratio': v.get('oi_ratio'),
                        'reason': v.get('reason', 'Manual'),
                        'confidence': v.get('confidence', 1.0)
                    } for k, v in data.items()}
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load roll history for {self.symbol}: {e}")
                return {}
        return {}
    
    def save_roll_history(self):
        """Save roll history to disk"""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for k, v in self.roll_history.items():
                serializable_data[k] = {
                    'roll_date': v['roll_date'].strftime('%Y-%m-%d') if pd.notna(v['roll_date']) else None,
                    'from_contract': v['from_contract'],
                    'to_contract': v['to_contract'],
                    'days_to_expiry': v['days_to_expiry'],
                    'volume_ratio': v.get('volume_ratio'),
                    'oi_ratio': v.get('oi_ratio'),
                    'reason': v.get('reason', 'Manual'),
                    'confidence': v.get('confidence', 1.0)
                }
            
            with open(self.roll_dates_file, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Could not save roll history for {self.symbol}: {e}")
    
    def calculate_optimal_roll_date(
        self,
        curve: pd.DataFrame,
        volume_curve: pd.DataFrame = None,
        oi_curve: pd.DataFrame = None,
        dte_matrix: pd.DataFrame = None,
        current_front: str = None,
        next_front: str = None
    ) -> Tuple[Optional[datetime], Dict]:
        """
        Calculate optimal roll date based on liquidity, volume, and calendar rules.
        
        Returns:
            (roll_date, analysis_dict) where analysis_dict contains detailed metrics
        """
        if current_front is None or next_front is None:
            return None, {"error": "Must specify both current_front and next_front contracts"}
            
        if curve is None or curve.empty:
            return None, {"error": "No curve data available"}
            
        if current_front not in curve.columns or next_front not in curve.columns:
            return None, {"error": f"Contracts {current_front}/{next_front} not found in curve data"}
        
        # Get price, volume, and OI series for both contracts
        current_prices = curve[current_front].dropna()
        next_prices = curve[next_front].dropna()
        
        current_volume = volume_curve[current_front].dropna() if volume_curve is not None else pd.Series()
        next_volume = volume_curve[next_front].dropna() if volume_curve is not None else pd.Series()
        
        current_oi = oi_curve[current_front].dropna() if oi_curve is not None else pd.Series()
        next_oi = oi_curve[next_front].dropna() if oi_curve is not None else pd.Series()
        
        # Find common date range
        common_dates = current_prices.index.intersection(next_prices.index)
        if len(common_dates) < 5:
            return None, {"error": "Insufficient overlapping data between contracts"}
        
        # Calculate roll signals for each date
        roll_scores = []
        analysis_data = []
        
        for date in common_dates:
            # Get days to expiry for current front
            days_to_expiry = None
            if dte_matrix is not None and current_front in dte_matrix.columns:
                if date in dte_matrix.index:
                    days_to_expiry = dte_matrix.at[date, current_front]
            
            if days_to_expiry is None or days_to_expiry < self.roll_criteria['min_days_to_expiry']:
                continue  # Too close to expiry or expired
                
            if days_to_expiry > self.roll_criteria['max_days_to_expiry']:
                continue  # Too early to roll
            
            # Calculate volume and OI ratios
            vol_ratio = None
            oi_ratio = None
            
            if not current_volume.empty and not next_volume.empty and date in current_volume.index and date in next_volume.index:
                curr_vol = current_volume.at[date]
                next_vol = next_volume.at[date]
                if curr_vol > 0:
                    vol_ratio = next_vol / curr_vol
            
            if not current_oi.empty and not next_oi.empty and date in current_oi.index and date in next_oi.index:
                curr_oi = current_oi.at[date]
                next_oi = next_oi.at[date]
                if curr_oi > 0:
                    oi_ratio = next_oi / curr_oi
            
            # Calculate roll score (0-1, higher is better time to roll)
            score = 0.0
            reasons = []
            
            # Volume criterion
            if vol_ratio is not None and vol_ratio >= self.roll_criteria['volume_threshold']:
                score += 0.3
                reasons.append(f"Volume ratio {vol_ratio:.2f} >= {self.roll_criteria['volume_threshold']}")
            
            # Open interest criterion  
            if oi_ratio is not None and oi_ratio >= self.roll_criteria['oi_threshold']:
                score += 0.4
                reasons.append(f"OI ratio {oi_ratio:.2f} >= {self.roll_criteria['oi_threshold']}")
            
            # Days to expiry scoring (optimal around 7-10 days)
            if 7 <= days_to_expiry <= 10:
                score += 0.2
                reasons.append(f"Optimal DTE range: {days_to_expiry} days")
            elif 5 <= days_to_expiry <= 12:
                score += 0.1
                reasons.append(f"Acceptable DTE range: {days_to_expiry} days")
            
            # Price stability check (last 3 days)
            if len(common_dates) >= 3:
                recent_dates = [d for d in common_dates if d <= date][-3:]
                if len(recent_dates) == 3:
                    recent_spreads = []
                    for d in recent_dates:
                        if d in current_prices.index and d in next_prices.index:
                            spread = next_prices.at[d] - current_prices.at[d]
                            recent_spreads.append(spread)
                    
                    if len(recent_spreads) == 3:
                        spread_std = np.std(recent_spreads)
                        spread_mean = np.mean(recent_spreads)
                        if spread_std < abs(spread_mean) * 0.02:  # 2% coefficient of variation
                            score += 0.1
                            reasons.append(f"Stable spread: std {spread_std:.4f}")
            
            roll_scores.append((date, score, days_to_expiry, vol_ratio, oi_ratio, reasons))
            analysis_data.append({
                'date': date,
                'score': score,
                'days_to_expiry': days_to_expiry,
                'volume_ratio': vol_ratio,
                'oi_ratio': oi_ratio,
                'reasons': reasons
            })
        
        if not roll_scores:
            return None, {"error": "No valid roll dates found within criteria"}
        
        # Find the date with highest score
        best_score_date, best_score, best_dte, best_vol_ratio, best_oi_ratio, best_reasons = max(
            roll_scores, key=lambda x: x[1]
        )
        
        analysis = {
            'optimal_roll_date': best_score_date,
            'score': best_score,
            'days_to_expiry': best_dte,
            'volume_ratio': best_vol_ratio,
            'oi_ratio': best_oi_ratio,
            'reasons': best_reasons,
            'all_candidates': analysis_data,
            'criteria_used': self.roll_criteria.copy()
        }
        
        return best_score_date, analysis
    
    def record_roll_event(
        self,
        roll_date: datetime,
        from_contract: str,
        to_contract: str,
        days_to_expiry: int,
        volume_ratio: Optional[float] = None,
        oi_ratio: Optional[float] = None,
        reason: str = "Automated",
        confidence: float = 1.0
    ):
        """Record a roll event in the history"""
        key = f"{from_contract}_{to_contract}_{roll_date.strftime('%Y%m%d')}"
        
        self.roll_history[key] = {
            'roll_date': pd.to_datetime(roll_date),
            'from_contract': from_contract,
            'to_contract': to_contract,
            'days_to_expiry': days_to_expiry,
            'volume_ratio': volume_ratio,
            'oi_ratio': oi_ratio,
            'reason': reason,
            'confidence': confidence
        }
        
        self.save_roll_history()
    
    def get_historical_roll_pattern(self, from_contract: str, to_contract: str) -> Dict:
        """Analyze historical roll patterns for a specific contract transition"""
        pattern_rolls = []
        
        for key, roll_data in self.roll_history.items():
            if (roll_data['from_contract'] == from_contract and 
                roll_data['to_contract'] == to_contract):
                pattern_rolls.append(roll_data)
        
        if not pattern_rolls:
            return {"error": f"No historical data for {from_contract} -> {to_contract} rolls"}
        
        # Calculate statistics
        days_to_expiry = [r['days_to_expiry'] for r in pattern_rolls if r['days_to_expiry'] is not None]
        volume_ratios = [r['volume_ratio'] for r in pattern_rolls if r['volume_ratio'] is not None]
        oi_ratios = [r['oi_ratio'] for r in pattern_rolls if r['oi_ratio'] is not None]
        
        return {
            'count': len(pattern_rolls),
            'avg_days_to_expiry': np.mean(days_to_expiry) if days_to_expiry else None,
            'std_days_to_expiry': np.std(days_to_expiry) if days_to_expiry else None,
            'avg_volume_ratio': np.mean(volume_ratios) if volume_ratios else None,
            'avg_oi_ratio': np.mean(oi_ratios) if oi_ratios else None,
            'rolls': pattern_rolls
        }
    
    def suggest_roll_schedule(
        self,
        curve: pd.DataFrame,
        volume_curve: pd.DataFrame = None,
        oi_curve: pd.DataFrame = None,
        dte_matrix: pd.DataFrame = None,
        front_month_series: pd.Series = None,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Generate a comprehensive roll schedule for the next year based on 
        historical patterns and current market conditions.
        """
        if front_month_series is None or curve is None:
            raise ValueError("Must provide both curve and front_month_series data")
        
        # Get recent data
        end_date = curve.index[-1] if not curve.empty else datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        recent_curve = curve[curve.index >= start_date]
        recent_front = front_month_series[front_month_series.index >= start_date]
        
        # Identify potential roll transitions
        roll_schedule = []
        unique_contracts = recent_front.dropna().unique()
        
        for i, current_contract in enumerate(unique_contracts):
            next_contract = unique_contracts[(i + 1) % len(unique_contracts)]
            
            # Calculate optimal roll date
            optimal_date, analysis = self.calculate_optimal_roll_date(
                recent_curve, volume_curve, oi_curve, dte_matrix,
                current_contract, next_contract
            )
            
            if optimal_date is not None:
                # Get historical pattern
                historical = self.get_historical_roll_pattern(current_contract, next_contract)
                
                roll_schedule.append({
                    'from_contract': current_contract,
                    'to_contract': next_contract,
                    'suggested_roll_date': optimal_date,
                    'days_to_expiry': analysis['days_to_expiry'],
                    'confidence_score': analysis['score'],
                    'volume_ratio': analysis['volume_ratio'],
                    'oi_ratio': analysis['oi_ratio'],
                    'historical_avg_dte': historical.get('avg_days_to_expiry'),
                    'historical_count': historical.get('count', 0),
                    'reasons': '; '.join(analysis['reasons'])
                })
        
        return pd.DataFrame(roll_schedule)
    
    def validate_current_roll_timing(
        self,
        curve: pd.DataFrame,
        current_front: str,
        proposed_roll_date: datetime,
        volume_curve: pd.DataFrame = None,
        oi_curve: pd.DataFrame = None,
        dte_matrix: pd.DataFrame = None
    ) -> Dict:
        """
        Validate if a proposed roll date will avoid the missing back months issue.
        """
        if current_front not in curve.columns:
            return {"valid": False, "reason": f"Contract {current_front} not found in curve"}
        
        # Check if we'll have sufficient back month data after roll
        roll_date_idx = curve.index.get_indexer([proposed_roll_date], method='nearest')[0]
        post_roll_dates = curve.index[roll_date_idx:roll_date_idx + 10]  # Check 10 days after roll
        
        back_month_data_quality = []
        
        for date in post_roll_dates:
            if date in curve.index:
                # Check if old front month will have data as back month
                old_front_price = curve.at[date, current_front]
                has_price = not pd.isna(old_front_price)
                
                has_volume = True
                if volume_curve is not None and current_front in volume_curve.columns:
                    vol = volume_curve.at[date, current_front]
                    has_volume = not pd.isna(vol) and vol >= self.roll_criteria['back_month_min_volume']
                
                back_month_data_quality.append({
                    'date': date,
                    'has_price': has_price,
                    'has_adequate_volume': has_volume,
                    'price': old_front_price,
                    'volume': volume_curve.at[date, current_front] if volume_curve is not None and current_front in volume_curve.columns else None
                })
        
        # Calculate data retention score
        valid_days = sum(1 for d in back_month_data_quality if d['has_price'] and d['has_adequate_volume'])
        data_retention_score = valid_days / len(back_month_data_quality) if back_month_data_quality else 0
        
        is_valid = data_retention_score >= 0.7  # 70% data retention threshold
        
        return {
            "valid": is_valid,
            "data_retention_score": data_retention_score,
            "days_checked": len(back_month_data_quality),
            "valid_days": valid_days,
            "detailed_analysis": back_month_data_quality,
            "recommendation": "Roll timing looks good" if is_valid else "Roll may be too early - risk of missing back months"
        }


def create_enhanced_curve_manager_with_roll_tracking(symbol: str, **kwargs):
    """
    Factory function to create FuturesCurveManager with integrated roll date management.
    """
    from CTAFlow.data.contract_handling.curve_manager import FuturesCurveManager
    
    # Create roll date manager
    roll_manager = RollDateManager(symbol)
    
    # Create curve manager with conservative roll parameters
    curve_manager = FuturesCurveManager(symbol, **kwargs)
    
    # Monkey patch the curve manager with enhanced roll detection
    original_run = curve_manager.run
    
    def enhanced_run(
        prefer_front_series: bool = True,
        match_tol: float = 0.01,
        rel_jump_thresh: float = 0.01,
        robust_k: float = 4.0,
        lookback: int = 10,
        near_expiry_days: int = 7,  # FIXED: More conservative roll timing
        min_persistence_days: int = 3,  # FIXED: Require more persistence
        smooth_window: int = 3,
        enforce_calendar_order: bool = True,
        save: bool = True,
        debug: bool = False,
        validate_rolls: bool = True,  # NEW: Validate roll timing
        track_rolls: bool = True      # NEW: Track roll events
    ):
        """Enhanced run method with roll date validation and tracking"""
        
        # Build curve first
        curve_manager.build_curve()
        curve_manager.compute_dte_matrix(curve_manager.curve)
        
        # Original front month detection
        result = original_run(
            prefer_front_series=prefer_front_series,
            match_tol=match_tol,
            rel_jump_thresh=rel_jump_thresh,
            robust_k=robust_k,
            lookback=lookback,
            near_expiry_days=near_expiry_days,
            min_persistence_days=min_persistence_days,
            smooth_window=smooth_window,
            enforce_calendar_order=enforce_calendar_order,
            save=save,
            debug=debug
        )
        
        # Enhanced roll validation and tracking
        roll_series_data = {}  # Will store roll events as {date: roll_key}
        
        if validate_rolls and curve_manager.front is not None:
            front_series = curve_manager.front
            # Detect roll changes by comparing consecutive values (strings)
            roll_changes = front_series != front_series.shift(1)
            roll_changes.iloc[0] = False  # First value is not a roll
            roll_dates = front_series[roll_changes].index
            
            if debug:
                print(f"Detected {len(roll_dates)} potential roll events")
            
            for roll_date in roll_dates:
                if roll_date == front_series.index[0]:
                    continue  # Skip first date
                    
                prev_idx = front_series.index.get_loc(roll_date) - 1
                prev_date = front_series.index[prev_idx]
                
                from_contract = front_series.at[prev_date]
                to_contract = front_series.at[roll_date]
                
                if pd.isna(from_contract) or pd.isna(to_contract):
                    continue
                
                # Validate roll timing
                validation = roll_manager.validate_current_roll_timing(
                    curve_manager.curve,
                    from_contract,
                    roll_date,
                    curve_manager.volume_curve,
                    curve_manager.oi_curve,
                    curve_manager.dte
                )
                
                if debug:
                    print(f"Roll {from_contract}->{to_contract} on {roll_date}: {validation['recommendation']}")
                
                # Track roll event
                if track_rolls:
                    days_to_expiry = None
                    if (curve_manager.dte is not None and 
                        from_contract in curve_manager.dte.columns and 
                        roll_date in curve_manager.dte.index):
                        days_to_expiry = curve_manager.dte.at[roll_date, from_contract]
                    
                    roll_manager.record_roll_event(
                        roll_date=roll_date,
                        from_contract=from_contract,
                        to_contract=to_contract,
                        days_to_expiry=days_to_expiry,
                        reason="FuturesCurveManager automated detection",
                        confidence=validation['data_retention_score']
                    )
                    
                    # Build roll series entry
                    roll_key = f"{from_contract}_{to_contract}_{roll_date.strftime('%Y%m%d')}"
                    roll_series_data[roll_date] = roll_key
        
        # Build roll series from all tracked roll events
        if roll_manager.roll_history:
            # Add any existing roll history that wasn't just detected
            for roll_key, roll_data in roll_manager.roll_history.items():
                roll_date = pd.to_datetime(roll_data['roll_date'])
                if roll_date not in roll_series_data:  # Don't overwrite newly detected rolls
                    roll_series_data[roll_date] = roll_key
        
        # Create comprehensive roll DataFrame and save to HDF
        if roll_series_data or roll_manager.roll_history:
            # Build comprehensive roll DataFrame from roll history
            roll_records = []
            
            for roll_key, roll_data in roll_manager.roll_history.items():
                # Calculate interval from previous roll if possible
                interval_days = None
                
                # Find previous roll for the same to_contract (to calculate intervals)
                prev_roll_date = None
                for other_key, other_data in roll_manager.roll_history.items():
                    if (other_data['to_contract'] == roll_data['from_contract'] and 
                        pd.to_datetime(other_data['roll_date']) < pd.to_datetime(roll_data['roll_date'])):
                        if prev_roll_date is None or pd.to_datetime(other_data['roll_date']) > prev_roll_date:
                            prev_roll_date = pd.to_datetime(other_data['roll_date'])
                
                if prev_roll_date is not None:
                    interval_days = (pd.to_datetime(roll_data['roll_date']) - prev_roll_date).days
                
                roll_records.append({
                    'roll_date': pd.to_datetime(roll_data['roll_date']),
                    'from_contract_expiration_code': roll_data['from_contract'],
                    'to_contract_expiration_code': roll_data['to_contract'],
                    'days_to_expiry': roll_data['days_to_expiry'],
                    'volume_ratio': roll_data.get('volume_ratio'),
                    'oi_ratio': roll_data.get('oi_ratio'),
                    'reason': roll_data.get('reason', 'Unknown'),
                    'confidence': roll_data.get('confidence', 1.0),
                    'interval_days': interval_days,
                    'roll_event_key': roll_key
                })
            
            if roll_records:
                # Create DataFrame
                roll_df = pd.DataFrame(roll_records)
                roll_df = roll_df.set_index('roll_date').sort_index()
                
                # Save to HDF using DataClient
                try:
                    from ..data_client import DataClient
                    data_client = DataClient()
                    
                    symbol_key = f"{curve_manager.symbol}_F" if not curve_manager.symbol.endswith('_F') else curve_manager.symbol
                    roll_path = f"market/{symbol_key}/roll_dates"
                    
                    data_client.write_dataframe(roll_df, roll_path)
                    
                    if debug:
                        print(f"Saved {len(roll_df)} roll events to {roll_path}")
                        print("Roll DataFrame preview:")
                        print(roll_df[['from_contract_expiration_code', 'to_contract_expiration_code', 
                                      'days_to_expiry', 'reason', 'interval_days']].head())
                        
                except Exception as e:
                    if debug:
                        print(f"Warning: Could not save roll DataFrame to HDF: {e}")
                        
                # Store roll DataFrame in curve manager for access
                curve_manager.roll_dataframe = roll_df
                
                # Also create simple roll series for backward compatibility
                roll_series = pd.Series(
                    roll_df['roll_event_key'].values, 
                    index=roll_df.index, 
                    name='roll_event_key'
                )
                curve_manager.roll_series = roll_series
        
        # Add roll manager to curve manager for future access
        curve_manager.roll_manager = roll_manager
        
        # Update result to include roll_dates path
        if isinstance(result, dict) and (roll_series_data or roll_manager.roll_history):
            symbol_key = f"{curve_manager.symbol}_F" if not curve_manager.symbol.endswith('_F') else curve_manager.symbol
            result["roll_dates"] = f"market/{symbol_key}/roll_dates"
        
        return result
    
    # Replace the run method
    curve_manager.run = enhanced_run
    curve_manager.roll_manager = roll_manager
    
    return curve_manager