import datetime
import os.path
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..features.signals_processing import COTAnalyzer, TechnicalAnalysis
from ..data.retrieval import fetch_data_sync
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, ParameterGrid, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, log_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from xgboost import XGBRegressor
import warnings
import pickle
from ..config import MODEL_DATA_PATH
from ..features import IntradayFeatures
from .config import build_default_params, infer_task_from_target

warnings.filterwarnings('ignore')
data_load_func = fetch_data_sync


def _detect_cuda_available():
    """Detect if CUDA-capable GPU is available for XGBoost/LightGBM

    Returns:
        bool: True if CUDA GPU is available, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    try:
        # Try XGBoost GPU detection
        import xgboost as xgb
        # Check if gpu_hist tree method is available
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
        xgb.train(params, dtrain, num_boost_round=1)
        return True
    except Exception:
        pass

    return False

class CTAForecast:
    """Forecasting framework optimized for your COT-enhanced dataset"""

    def __init__(self, ticker_symbol, use_daily_data=True, **kwargs):
        self.data_processor = COTAnalyzer()
        self.technical_analyzer = TechnicalAnalysis()
        self.models = {}
        self.symbol = ticker_symbol
        self.saved_model_folder = MODEL_DATA_PATH / ticker_symbol
        # Fetch data and apply COT column renaming
        raw_data = data_load_func(ticker_symbol, daily=use_daily_data, **kwargs)

        if raw_data is not None:
            # Apply COT column renaming from COTAnalyzer
            self.data = self.data_processor.load_and_clean_data(raw_data)

        else:
            self.data = None
            
        self.target = None
        self.features = None
        self.cot_features = None
        self.tech_features = None
        self.intraday_features = None



    def prepare_features(self, include_technical=True,
                         selected_indicators=None,
                         normalize_momentum=False,
                         vol_return_periods=None,
                         include_intraday=True,
                         selected_intraday_features=None,
                         intraday_horizon=5,
                         include_cot=True,
                         selected_cot_features=None,
                         resample_before_calcs=False,
                         resample_after=False,
                         resample_day="Wednesday",
                         remove_weekends=True,
                         filter_valid_prices=True):
        """Create comprehensive feature set from your data structure

        Args:
            include_technical: Whether to include any technical indicators
            selected_indicators: List of specific technical indicator groups to include
                               ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence', 'vol_normalized']
            normalize_momentum: Whether to include volatility-normalized momentum
            vol_return_periods: List of periods for volatility-normalized returns
            include_cot: Whether to include COT features
            selected_cot_features: List of COT feature groups to include
                                 ['positioning', 'flows', 'extremes', 'market_structure', 'interactions', 'spreads']
            filter_valid_prices: Whether to filter to rows with valid price data when including technical features
            :param include_intraday:
            :param resample_before_calcs:
        """
        df = self.data.copy()
        if remove_weekends:
            df = df.loc[(df.index.dayofweek != 6) & (df.index.dayofweek != 5)]

        if resample_before_calcs:
            df = self.resample_weekly(df=df, day_of_week=resample_day)

        if vol_return_periods is None:
            vol_return_periods = [1, 5, 10, 20]


        # Filter to rows with valid price data if technical indicators are requested
        if include_technical and filter_valid_prices:
            # Check for valid price data (either 'Close' or 'Last')
            price_col = 'Close' if 'Close' in df.columns else 'Last'
            if price_col in df.columns:
                valid_price_mask = df[price_col].notna()
                print(f"Filtering to {valid_price_mask.sum()}/{len(df)} rows with valid {price_col} data")
                df = df[valid_price_mask]
            else:
                print("Warning: No valid price column found for technical analysis")

        features = pd.DataFrame(index=df.index)


        # 1. COT Features (Primary advantage) - Using new selective method
        if include_cot:
            cot_features = self.data_processor.calculate_enhanced_cot_features(df, selected_cot_features)
            for col in cot_features.columns:
                if f'cot_{col}' not in features.columns:
                    features[f'cot_{col}'] = cot_features[col]

                self.cot_features = cot_features.columns

        # 3. Technical Indicators Features (selective calculation)
        if include_technical:
            tech_indicators = self.technical_analyzer.calculate_enhanced_indicators(
                df,
                selected_indicators=selected_indicators,
                normalize_momentum=normalize_momentum,
                vol_return_periods=vol_return_periods
            )
            self.tech_features = tech_indicators.columns

            # Add all calculated technical indicators with 'tech_' prefix
            for feature in tech_indicators.columns:
                if f'tech_{feature}' not in features.columns:
                    features[f'tech_{feature}'] = tech_indicators[feature]

        if include_intraday:
            intraday_feature_df = self.calculate_intraday_features(
                                intraday_horizon,
                                selected_intraday_features=selected_intraday_features
                                )

            self.intraday_features = intraday_feature_df.columns


            for feature in self.intraday_features:
                if f'ind_{feature}' not in features.columns:
                    features[f'ind_{feature}'] = intraday_feature_df[feature]





        # 4. Seasonal Features
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter

        # Store and return features (drop rows with all NaN)
        self.features = features

        if resample_after:
            self.resample_existing_features(day_of_week=resample_day)

        return self.features


    def create_target_variable(self, forecast_horizon=10, target_type='return'):
        """Create target variable for forecasting"""
        df = self.data.copy()
        
        # Map 'Last' to 'Close' if 'Close' doesn't exist but 'Last' does
        if 'Close' not in df.columns and 'Last' in df.columns:
            df['Close'] = df['Last']

            
        if target_type == 'return' and df['Close'].notna().any():
            # Price return target
            target = df['Close'].pct_change(periods=forecast_horizon).shift(-forecast_horizon)

        elif target_type == 'positioning':
            # CTA positioning change target (useful for position forecasting)
            mm_net = df['money_manager_longs'] - df['money_manager_shorts']
            target = mm_net.pct_change(periods=forecast_horizon).shift(-forecast_horizon)

        else:
            # Default to COT index change
            mm_net = df['money_manager_longs'] - df['money_manager_shorts']
            cot_index = self.data_processor.calculate_cot_index(mm_net)
            target = cot_index.diff(periods=forecast_horizon).shift(-forecast_horizon)

        self.target = target.dropna()
        self._align_features_and_target()
        return self.target

    def _align_features_and_target(self):
        """Align self.features and self.target by their shared index."""
        if self.features is not None and self.target is not None:
            common_index = self.features.index.intersection(self.target.index)
            self.features = self.features.loc[common_index]
            self.target = self.target.loc[common_index]

    def _compute_permutation_importance(self, model, X_val, y_val):
        """Compute permutation importance on a validation slice using model-compatible data."""
        if X_val is None or y_val is None or len(X_val) == 0:
            return None

        X_array = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
        y_array = y_val.values if isinstance(y_val, pd.Series) else np.asarray(y_val)

        # Respect any preprocessing baked into the model wrapper
        if hasattr(model, 'scaler') and model.scaler is not None:
            X_array = model.scaler.transform(X_array)

        estimator = getattr(model, 'model', model)
        feature_names = getattr(model, 'feature_names', None) or [f'feature_{i}' for i in range(X_array.shape[1])]

        try:
            perm = permutation_importance(
                estimator,
                X_array,
                y_array,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )
            return pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
        except Exception as exc:
            warnings.warn(f"Could not compute permutation importance: {exc}", RuntimeWarning)
            return None

    def calculate_intraday_features(self, indicator_horizons=5, selected_intraday_features=None):
        if not selected_intraday_features:
            selected_intraday_features = ['rv', 'rsv', 'cum_delta']
        features = pd.DataFrame(index=self.data.index)
        ind = IntradayFeatures(self.symbol)

        if 'rv' in selected_intraday_features:
            features['rv'] = ind.historical_rv(indicator_horizons)

        if 'rsv' in selected_intraday_features:
            features[['rsv_pos', 'rsv_neg']] = ind.realized_semivariance(indicator_horizons, average=False)[["RS_pos", "RS_neg"]]

        if 'cum_delta' in selected_intraday_features:
            features['cum_delta'] = ind.cumulative_delta(indicator_horizons)

        return features
    
    def resample_weekly(self, df, day_of_week='Friday', price_agg='last', volume_agg='sum', cot_agg='last'):
        """Resample data to weekly frequency on specified day
        
        Args:
            df: DataFrame to resample
            day_of_week: Day to resample on ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
            price_agg: Aggregation method for OHLCV data ('last', 'first', etc.)
            volume_agg: Aggregation method for Volume ('sum', 'last', etc.)
            cot_agg: Aggregation method for COT data ('last', 'mean', etc.)
            
        Returns:
            Resampled DataFrame
        """
        # Map day names to pandas offset aliases
        day_mapping = {
            'Monday': 'W-MON',
            'Tuesday': 'W-TUE', 
            'Wednesday': 'W-WED',
            'Thursday': 'W-THU',
            'Friday': 'W-FRI',
            'Saturday': 'W-SAT',
            'Sunday': 'W-SUN'
        }
        
        if day_of_week not in day_mapping:
            raise ValueError(f"day_of_week must be one of {list(day_mapping.keys())}")
        
        freq = day_mapping[day_of_week]
        
        # Create resampling rules based on column types
        agg_rules = {}
        
        # OHLCV columns
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in df.columns:
            if col in ohlcv_cols:
                if col == 'Open':
                    agg_rules[col] = 'first'
                elif col == 'High':
                    agg_rules[col] = 'max'
                elif col == 'Low':
                    agg_rules[col] = 'min'
                elif col == 'Close':
                    agg_rules[col] = price_agg
                elif col == 'Volume':
                    agg_rules[col] = volume_agg
            
            # COT columns (money manager, commercial, etc.)
            elif any(cot_term in col.lower() for cot_term in ['money_manager', 'producer_merchant', 'swap_dealer', 'reportable', 'market_participation']):
                agg_rules[col] = cot_agg
            
            # Default to last value for other columns
            else:
                agg_rules[col] = 'last'
        
        # Perform resampling
        resampled = df.resample(freq).agg(agg_rules)
        
        # Drop any rows with all NaN values
        resampled = resampled.dropna(how='all')
        
        return resampled

    def resample_features(self):
        return
    
    def resample_features_weekly(self, df, day_of_week='Friday', remove_duplicates=True, 
                                 aggregation_method='last'):
        """
        Resample features to weekly frequency with duplicate removal and selective day-of-week closing.
        
        This method handles:
        1. Weekly resampling with configurable day-of-week anchor
        2. Duplicate removal (keeping the last observation when multiple exist)
        3. Proper handling of different data types (numeric vs categorical)
        4. NaN handling and forward-fill options
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with datetime index to resample
        day_of_week : str, default 'Friday'
            Day of week to anchor weekly resampling ('Monday' through 'Sunday')
        remove_duplicates : bool, default True
            Whether to remove duplicate index values before resampling
        aggregation_method : str, default 'last'
            Method to use for aggregation ('last', 'first', 'mean', 'median')
            
        Returns:
        --------
        pd.DataFrame
            Resampled weekly DataFrame
            
        Examples:
        ---------
        # Resample to weekly Friday closes
        weekly_features = models.resample_features_weekly(daily_features, 'Friday')
        
        # Resample to weekly Wednesday with mean aggregation
        weekly_features = models.resample_features_weekly(
            daily_features, 'Wednesday', aggregation_method='mean'
        )
        """
        
        if df is None or df.empty:
            return df
            
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex for weekly resampling")
        
        # Remove duplicates if requested
        if remove_duplicates:
            original_length = len(df)
            # Keep the last occurrence of duplicate index values
            df = df[~df.index.duplicated(keep='last')]
            duplicates_removed = original_length - len(df)
            if duplicates_removed > 0:
                print(f"Removed {duplicates_removed} duplicate index values")
        
        # Sort by index to ensure proper chronological order
        df = df.sort_index()
        
        # Day-of-week mapping for pandas resampling
        day_mapping = {
            'Monday': 'W-MON', 
            'Tuesday': 'W-TUE', 
            'Wednesday': 'W-WED',
            'Thursday': 'W-THU', 
            'Friday': 'W-FRI', 
            'Saturday': 'W-SAT', 
            'Sunday': 'W-SUN'
        }
        
        if day_of_week not in day_mapping:
            raise ValueError(f"day_of_week must be one of {list(day_mapping.keys())}")
        
        freq = day_mapping[day_of_week]
        
        print(f"Resampling to weekly frequency anchored on {day_of_week} ({freq})")
        print(f"Original data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Determine aggregation method for different column types
        if aggregation_method == 'smart':
            # Smart aggregation based on column names and data types
            agg_dict = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric columns - use appropriate aggregation
                    if any(term in col.lower() for term in ['price', 'close', 'last', 'open', 'high', 'low']):
                        agg_dict[col] = 'last'  # Price data - use last
                    elif any(term in col.lower() for term in ['volume', 'count', 'total']):
                        agg_dict[col] = 'sum'   # Volume data - use sum
                    elif any(term in col.lower() for term in ['average', 'mean', 'ratio']):
                        agg_dict[col] = 'mean'  # Average data - use mean
                    else:
                        agg_dict[col] = 'last'  # Default to last
                else:
                    agg_dict[col] = 'last'      # Non-numeric - use last
            
            resampled_df = df.resample(freq).agg(agg_dict)
            
        elif aggregation_method in ['last', 'first']:
            # Simple aggregation methods
            if aggregation_method == 'last':
                resampled_df = df.resample(freq).last()
            else:  # first
                resampled_df = df.resample(freq).first()
                
        elif aggregation_method in ['mean', 'median']:
            # Statistical aggregation methods (only for numeric columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            
            agg_dict = {}
            for col in numeric_cols:
                agg_dict[col] = aggregation_method
            for col in non_numeric_cols:
                agg_dict[col] = 'last'  # Use last for non-numeric
            
            resampled_df = df.resample(freq).agg(agg_dict)
            
        else:
            raise ValueError(f"aggregation_method must be one of: 'last', 'first', 'mean', 'median', 'smart'")
        
        # Handle NaN values
        # For weekly data, we typically want to drop rows that are completely NaN
        resampled_df = resampled_df.dropna(how='all')
        
        # Optional: Forward fill recent NaN values (up to 2 periods)
        if len(resampled_df) > 2:
            resampled_df = resampled_df.ffill(limit=2)
        
        print(f"Resampled data shape: {resampled_df.shape}")
        print(f"Resampled date range: {resampled_df.index.min()} to {resampled_df.index.max()}")
        
        # Quality checks
        if len(resampled_df) == 0:
            print("Warning: Resampled data is empty")
        
        # Check for remaining duplicates (shouldn't happen but good to verify)
        duplicate_count = resampled_df.index.duplicated().sum()
        if duplicate_count > 0:
            print(f"Warning: {duplicate_count} duplicate indices remain after resampling")
        
        return resampled_df
    

    def get_available_indicators(self):
        """Return list of available technical indicator groups"""
        return ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence', 'vol_normalized']
    
    def get_available_cot_features(self):
        """Return list of available COT feature groups"""
        return ['positioning', 'flows', 'extremes', 'market_structure', 'interactions', 'spreads']
    
    def get_available_weekly_days(self):
        """Return list of available days for weekly resampling"""
        return ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def resample_existing_features(self, day_of_week='Friday'):
        """Resample existing self.features to weekly frequency after calculations are done
        
        Args:
            day_of_week: Day to resample on (default 'Friday')
            
        Returns:
            Resampled features DataFrame (also updates self.features)
        """
        if self.features is None:
            raise ValueError("No features found. Run prepare_forecasting_features() or load_and_prepare_data() first.")


        day_mapping = {
            'Monday': 'W-MON', 'Tuesday': 'W-TUE', 'Wednesday': 'W-WED',
            'Thursday': 'W-THU', 'Friday': 'W-FRI', 'Saturday': 'W-SAT', 'Sunday': 'W-SUN'
        }

        freq = day_mapping[day_of_week]


        # Resample the features using the dedicated weekly resampling method
        self.features = self.resample_features_weekly(self.features, day_of_week)
        self.resampled_features = True
        self.resampled_day = day_of_week
        
        # Also resample target if it exists
        if self.target is not None:

            freq = day_mapping[day_of_week]
            self.target = self.target.resample(freq).last().dropna()

        self._align_features_and_target()

        return self.features
    
    def get_technical_features_only(self, df, selected_indicators=None, normalize_momentum=False):
        """Get only technical indicator features for separate analysis
        
        Args:
            df: Input dataframe
            selected_indicators: Optional list of specific indicator groups to include
            normalize_momentum: Whether to include normalized momentum
        """
        return self.technical_analyzer.calculate_enhanced_indicators(
            df, 
            selected_indicators=selected_indicators,
            normalize_momentum=normalize_momentum
        )
    
    def get_cot_features_only(self,data, selected_cot_features=None):
        """Get only COT-based features for separate analysis with selective calculation
        
        Args:
             :param selected_cot_features:  ['positioning', 'flows', 'extremes', 'market_structure', 'interactions', 'spreads']
             :param data:
        """
        if selected_cot_features is None:
            selected_cot_features = ['positioning', 'flows', 'extremes', 'market_structure', 'interactions']
        
        return self.data_processor.calculate_enhanced_cot_features(data,selected_cot_features)

    def prepare_data(self, resample_weekly_first=False, resample_weekly_after=False, weekly_day='Friday',
                     include_technical=True, selected_indicators=None,
                     normalize_momentum=False, vol_return_periods=None,
                     include_cot=True, selected_cot_features=None, filter_valid_prices=True):
        """Load data and prepare all feature sets with selective technical and COT indicators
        
        Args:
            resample_weekly_first: Whether to resample raw data to weekly frequency before feature calculation
            resample_weekly_after: Whether to resample calculated features to weekly frequency 
            weekly_day: Day of week for weekly resampling ('Monday' through 'Sunday', default 'Friday')
            include_technical: Whether to include technical indicators
            selected_indicators: List of specific technical groups to include
                               ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence', 'vol_normalized']
            normalize_momentum: Whether to include volatility-normalized momentum
            vol_return_periods: List of periods for volatility-normalized returns.
                                 Defaults to [1, 5, 10, 20] when None
            include_cot: Whether to include COT features
            selected_cot_features: List of COT feature groups to include
                                 ['positioning', 'flows', 'extremes', 'market_structure', 'interactions', 'spreads']
        """
        # Load data using the data processor
        raw_data = self.data
        
        if raw_data is None:
            raise ValueError(f"Could not load data for ticker {self.symbol}")

        
        # Store data for training/validation use
        self.data = raw_data.copy()
        
        # Determine default volatility periods if not provided
        if vol_return_periods is None:
            vol_return_periods = [1, 5, 10, 20]

        # Prepare different feature sets with selective technical and COT indicators
        all_features = self.prepare_features(
            include_technical=include_technical,
            selected_indicators=selected_indicators,
            normalize_momentum=normalize_momentum,
            vol_return_periods=vol_return_periods,
            include_cot=include_cot,
            selected_cot_features=selected_cot_features,
            filter_valid_prices=filter_valid_prices
        )
        # Apply weekly resampling to calculated features if requested

        return {
            'raw_data': self.data,
            'all_features': self.features,
            'technical_features': self.tech_features,
            'cot_features': self.cot_features
        }

    def _fit_model(self, X, y, model_type='ridge', test_size=0.2,
                   use_grid_search=False, param_grid=None,
                   grid_search_cv=5, grid_search_scoring='neg_mean_squared_error',
                   **model_params):
        """Internal utility for training/validation split and model fitting."""

        # Remove samples containing NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            raise ValueError("No valid samples after removing NaN values")

        # Time-based split to avoid lookahead bias
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Reserve a validation slice from the training data for hyperparameter choices
        X_val = None
        y_val = None
        val_size = max(1, int(len(X_train) * 0.2))
        if val_size >= len(X_train):
            val_size = 0
        if val_size > 0:
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]
        else:
            X_train_fit = X_train
            y_train_fit = y_train

        # Initialize model
        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            model = CTALinear(model_type=model_type, **model_params)
        elif model_type == 'lightgbm':
            model = CTALight(**model_params)
        elif model_type == 'xgboost':
            model = CTAXGBoost(**model_params)
        elif model_type == 'randomforest':
            model = CTARForest(**model_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        grid_search_results = None
        validation_metrics = None
        validation_importance = None
        eval_set = None
        if X_val is not None and len(X_val) > 0:
            eval_set = (X_val, y_val)

        if model_type in ['lightgbm', 'xgboost']:
            if use_grid_search:
                grid_results = model.fit_with_grid_search(
                    X_train_fit, y_train_fit,
                    param_grid=param_grid,
                    eval_set=eval_set,
                    cv_folds=grid_search_cv,
                    scoring=grid_search_scoring,
                    verbose=True
                )
                grid_search_results = grid_results['grid_search_results']
            else:
                model.fit(X_train_fit, y_train_fit, eval_set=eval_set)
        elif model_type == 'randomforest':
            # Random Forest doesn't need separate validation set for early stopping
            if use_grid_search:
                grid_results = model.fit_with_grid_search(
                    X_train_fit, y_train_fit,
                    param_grid=param_grid,
                    cv_folds=grid_search_cv,
                    scoring=grid_search_scoring,
                    verbose=True
                )
                grid_search_results = grid_results['grid_search_results']
            else:
                model.fit(X_train_fit, y_train_fit)
        else:
            if use_grid_search:
                print("Warning: Grid search not supported for linear models. Using standard fit.")
            model.fit(X_train_fit, y_train_fit)

        if X_val is not None and len(X_val) > 0:
            validation_metrics = model.evaluate(X_val, y_val)
            validation_importance = self._compute_permutation_importance(model, X_val, y_val)

        train_eval_X = X_train_fit if X_val is not None else X_train
        train_eval_y = y_train_fit if X_val is not None else y_train

        train_pred = model.predict(train_eval_X)
        test_pred = model.predict(X_test)

        result = {
            'model': model,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'train_metrics': model.evaluate(train_eval_X, train_eval_y),
            'test_metrics': model.evaluate(X_test, y_test),
            'feature_importance': model.get_feature_importance(),
            'validation_metrics': validation_metrics,
            'validation_feature_importance': validation_importance,
            'hyperparameter_selection': 'cross_validation' if grid_search_results is not None else (
                'validation_holdout' if validation_metrics is not None else 'train_only'
            ),
            'data_split_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_period': (X_train.index[0], X_train.index[-1]) if len(X_train) > 0 else (None, None),
                'test_period': (X_test.index[0], X_test.index[-1]) if len(X_test) > 0 else (None, None)
            }
        }

        if grid_search_results is not None:
            result['grid_search_results'] = grid_search_results

        return result
    
    def train_model(self, model_type='ridge', target_type='return', forecast_horizon=10,
                   test_size=0.2, use_grid_search=False, param_grid=None,
                   grid_search_cv=5, grid_search_scoring='neg_mean_squared_error', **model_params):
        """Convenience method to train and validate models

        Args:
            model_type: 'linear', 'ridge', 'lasso', 'elastic_net', 'lightgbm', 'xgboost', 'randomforest'
            target_type: 'return', 'cta_positioning', 'cot_index_change'
            forecast_horizon: Days ahead to predict
            test_size: Fraction of data for testing
            use_grid_search: Whether to use grid search for hyperparameter tuning (LightGBM/XGBoost/RandomForest only)
            param_grid: Dictionary of parameters for grid search (LightGBM/XGBoost/RandomForest only)
            grid_search_cv: Number of CV folds for grid search
            grid_search_scoring: Scoring metric for grid search
            **model_params: Parameters to pass to the model

        Returns:
            Dictionary with model, predictions, performance metrics, and grid search results (if used)
        """
        if self.features is None or self.data is None:
            raise ValueError("Must load and prepare data first using load_and_prepare_data()")

        target = self.create_target_variable(forecast_horizon, target_type)
        common_index = self.features.index.intersection(target.index)
        X = self.features.loc[common_index]
        y = target.loc[common_index]

        result = self._fit_model(
            X, y, model_type=model_type, test_size=test_size,
            use_grid_search=use_grid_search, param_grid=param_grid,
            grid_search_cv=grid_search_cv,
            grid_search_scoring=grid_search_scoring, **model_params
        )

        model_key = f"{model_type}_{target_type}_{forecast_horizon}d"
        self.models[model_key] = result['model']
        result['model_key'] = model_key
        return result

    
    def cross_validate_model(self, model_type='ridge', target_type='return',
                           forecast_horizon=10, cv_folds=5, **model_params):
        """Perform time series cross-validation
        
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'elastic_net', 'lightgbm', 'xgboost', 'randomforest'
            target_type: 'return', 'cta_positioning', 'cot_index_change'
            forecast_horizon: Days ahead to predict
            cv_folds: Number of CV folds
            **model_params: Parameters to pass to the model
            
        Returns:
            Dictionary with CV results
        """
        if self.features is None or self.data is None:
            raise ValueError("Must load and prepare data first using load_and_prepare_data()")

        # Create target variable
        target = self.create_target_variable(forecast_horizon, target_type)

        # Align features with target index
        common_index = self.features.index.intersection(target.index)
        X = self.features.loc[common_index]
        y = target.loc[common_index]

        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            raise ValueError("No valid samples after removing NaN values")
        
        # Initialize model
        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            model = CTALinear(model_type=model_type, **model_params)
            cv_results = model.cross_validate(X, y, cv_folds=cv_folds)
        elif model_type == 'lightgbm':
            model = CTALight(**model_params)
            cv_results = model.cross_validate(X, y, cv_folds=cv_folds)
        elif model_type == 'xgboost':
            model = CTAXGBoost(**model_params)
            cv_results = model.cross_validate(X, y, cv_folds=cv_folds)
        elif model_type == 'randomforest':
            model = CTARForest(**model_params)
            cv_results = model.cross_validate(X, y, cv_folds=cv_folds)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return cv_results

    def run_selected_features(self, selected_features=None, base_model=None, model_type='ridge',top_n=20,
                              use_grid_search=False,
                              param_grid=None,
                              grid_search_cv=3,
                              grid_search_scoring="neg_mean_squared_error",
                              test_size=0.2, **model_params):
        """Train a model on a selected set of features.

        This method can work in two ways:
        1. Use a directly provided list of feature names (selected_features)
        2. Use the feature_importance from a base_model to select top_n features

        Args:
            selected_features: List of feature column names to use directly
            base_model: Trained model instance or key in ``self.models``
                with a ``feature_importance`` attribute (used if selected_features is None).
            model_type: Type of model to train on the selected features
                ('linear', 'ridge', 'lasso', 'elastic_net', 'lightgbm', 'xgboost', 'randomforest').
            top_n: Number of top features to use from ``base_model`` (ignored if selected_features provided).
            test_size: Fraction of samples for the test split.
            **model_params: Additional parameters passed to the new model.

        Returns:
            Dictionary with trained model, predictions, metrics and the
            selected feature names.
        """

        if self.features is None or self.target is None:
            raise ValueError("Features and target must be prepared first")

        # Determine which features to use
        if selected_features is not None:
            # Use directly provided feature list
            if not isinstance(selected_features, list):
                raise ValueError("selected_features must be a list of feature names")
            
            # Validate that all selected features exist in self.features
            missing_features = [f for f in selected_features if f not in self.features.columns]
            if missing_features:
                raise ValueError(f"Features not found in dataset: {missing_features}")
            
            top_features = selected_features
        else:
            # Fall back to base_model approach
            if base_model is None:
                raise ValueError("Either selected_features or base_model must be provided")
            
            if isinstance(base_model, str):
                if base_model not in self.models:
                    raise ValueError(f"Model '{base_model}' not found in self.models")
                base_model = self.models[base_model]

            if not hasattr(base_model, 'feature_importance') or base_model.feature_importance is None:
                raise ValueError("Base model must have feature_importance after fitting")

            top_features = base_model.feature_importance.head(top_n).index.tolist()

        X = self.features[top_features]
        y = self.target

        result = self._fit_model(X, y, model_type=model_type, test_size=test_size,
                                 use_grid_search=use_grid_search,param_grid=param_grid,grid_search_cv=grid_search_cv, grid_search_scoring=grid_search_scoring, **model_params)

        if selected_features is not None:
            model_key = f"fs_{model_type}_{len(selected_features)}features"
        else:
            model_key = f"fs_{model_type}_top{top_n}"
        self.models[model_key] = result['model']

        result.update({
            'model_key': model_key,
            'selected_features': top_features,
        })
        return result
    
    def predict_next_period(self, model_key, periods_ahead=1):
        """Make predictions for future periods using a trained model
        
        Args:
            model_key: Key of the trained model in self.models
            periods_ahead: Number of periods to predict ahead
            
        Returns:
            Predictions array
        """
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]
        
        # Use the most recent features for prediction
        if self.features is None:
            raise ValueError("No features available. Run load_and_prepare_data() first.")
        
        # Get the last valid row of features
        last_features = self.features.iloc[-periods_ahead:].dropna()
        
        if len(last_features) == 0:
            raise ValueError("No valid features for prediction")
        
        predictions = model.predict(last_features)

        return predictions

    def visualize_forecast(self, y_true, predictions):
        """Visualize forecast results using Plotly."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly required for plotting. Install with: pip install plotly")
            return None

        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=y_true.index)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Predicted vs Actual", "Raw Data Comparison"))

        fig.add_trace(
            go.Scatter(x=y_true, y=predictions, mode="markers", name="Predicted vs Actual"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=y_true.index, y=y_true, mode="lines", name="Actual"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions, mode="lines", name="Predicted"),
            row=1, col=2
        )
        fig.update_layout(width=900, height=400)

        return fig

    def visualize_raw_data(self, columns=None):
        """Plot raw input data as interactive time series using Plotly.

        Parameters
        ----------
        columns : list[str], optional
            Specific column names from ``self.data`` to plot. Defaults to all
            numeric columns when ``None``.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure containing the plot, or ``None`` if plotly is not installed.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No raw data available to visualize")

        try:
            import plotly.express as px
        except ImportError:
            print("plotly required for plotting. Install with: pip install plotly")
            return None

        df = self.data
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        fig = px.line(df, x=df.index, y=columns, title="Raw Data")
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="Value")
        return fig

    def visualize_features(self, feature_names=None):
        """Plot engineered feature values over time using Plotly.

        Parameters
        ----------
        feature_names : list[str], optional
            Names of feature columns from ``self.features`` to plot. Defaults to
            all available features when ``None``.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with the resulting plot, or ``None`` if plotly is not
            installed.
        """
        if self.features is None or self.features.empty:
            raise ValueError("No features available to visualize")

        try:
            import plotly.express as px
        except ImportError:
            print("plotly required for plotting. Install with: pip install plotly")
            return None

        df = self.features
        if feature_names is None:
            feature_names = df.columns.tolist()
        fig = px.line(df, x=df.index, y=feature_names, title="Engineered Features")
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="Value")
        return fig
    
    def get_model_summary(self):
        """Get summary of all trained models
        
        Returns:
            DataFrame with model information
        """
        if not self.models:
            return pd.DataFrame(columns=['Model', 'Type', 'Target', 'Horizon', 'Fitted'])
        
        summary_data = []
        for key, model in self.models.items():
            parts = key.split('_')
            model_type = parts[0]
            target_type = '_'.join(parts[1:-1])
            horizon = parts[-1]
            
            summary_data.append({
                'Model': key,
                'Type': model_type,
                'Target': target_type,
                'Horizon': horizon,
                'Fitted': model.is_fitted,
                'Features': len(model.feature_names) if hasattr(model, 'feature_names') and model.feature_names else 'Unknown'
            })
        
        return pd.DataFrame(summary_data)

    def save_model(self, base_model, file_out_name=None):
        if not os.path.exists(self.saved_model_folder):
            os.mkdir(self.saved_model_folder)

        if isinstance(base_model, str):
            if base_model not in self.models:
                raise ValueError(f"Model '{base_model}' not found in self.models")
            else:
                model_key = base_model
                saved_model = self.models[base_model]
        elif isinstance(base_model, dict):
            if 'model' in base_model.keys() and 'model_key' in base_model.keys():
                model_key = base_model['model_key']
                saved_model = base_model['model']
            else:
                raise ValueError("No model/model_key pairs found in dict keys")
        else:
            if isinstance(base_model, (CTALinear, CTALight, CTAXGBoost, CTARForest)):
                model_key = f"{self.symbol}_{datetime.datetime.now().strftime('%m-%d.%H.%M')}"
                saved_model = base_model
                pass
            else:
                raise TypeError("Failed to detect model")
        if not file_out_name:
            file_out_name = model_key

        file_loc = self.saved_model_folder / f"{file_out_name}.pkl"
        with open(file_loc, 'wb') as save_file:
            pickle.dump(saved_model, save_file)

        print(f"Model saved to {file_loc.__str__()}")

        return

    def load_model(self, model_key:str, model_folder=None):

        if not model_folder:
            model_folder = self.saved_model_folder
        if not isinstance(model_folder, Path or os.PathLike):
            try:
                model_folder = Path(model_folder)
            except Exception as e:
                raise e

        file_location = model_folder / model_key
        with open(file_location, 'rb') as file_loc:
            self.models[model_key] = pickle.load(file_loc)

        return self.models[model_key]


class CTALinear:
    """Linear regression models optimized for CTA positioning prediction"""
    
    def __init__(self, model_type='ridge', alpha=1.0, l1_ratio=0.5, normalize=True):
        """
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'elastic_net'
            alpha: Regularization strength for Ridge/Lasso/ElasticNet
            l1_ratio: L1 ratio for ElasticNet (0=Ridge, 1=Lasso)
            normalize: Whether to standardize features
        """
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        
        # Initialize model based on type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, max_iter=2000)
        elif model_type == 'elastic_net':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
        else:
            raise ValueError(f"model_type must be one of ['linear', 'ridge', 'lasso', 'elastic_net'], got {model_type}")
        
        self.scaler = StandardScaler() if normalize else None
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance = None
        
    def fit(self, X, y):
        """Fit the linear model
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
        """
        # Convert to numpy arrays and handle missing values
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        if isinstance(y, pd.Series):
            y = y.values
            
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid samples after removing NaN values")
        
        # Normalize features if requested
        if self.scaler:
            X_clean = self.scaler.fit_transform(X_clean)
        
        # Fit the model
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        
        # Store feature importance (coefficients)
        if hasattr(self.model, 'coef_'):
            self.feature_importance = pd.Series(
                self.model.coef_, 
                index=self.feature_names
            ).abs().sort_values(ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions
        
        Args:
            X: Features DataFrame or array
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Apply scaling if used during training
        if self.scaler:
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
    
    def cross_validate(self, X, y, cv_folds=5, scoring='neg_mean_squared_error'):
        """Perform time series cross-validation
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            cv_folds: Number of CV folds
            scoring: Scoring method
            
        Returns:
            Dictionary with CV results
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_clean, y_clean, 
            cv=tscv, scoring=scoring, n_jobs=-1
        )
        
        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'all_scores': cv_scores,
            'scoring_method': scoring
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics
        """
        predictions = self.predict(X_test)
        
        # Convert to numpy arrays for consistent handling
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        # Remove NaN values for evaluation
        mask = ~(np.isnan(predictions) | np.isnan(y_test))
        pred_clean = predictions[mask]
        y_clean = y_test[mask]
        
        if len(pred_clean) == 0:
            raise ValueError("No valid predictions for evaluation")
        
        return {
            'mse': mean_squared_error(y_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_clean, pred_clean)),
            'mae': mean_absolute_error(y_clean, pred_clean),
            'r2': r2_score(y_clean, pred_clean),
            'directional_accuracy': np.mean(np.sign(pred_clean) == np.sign(y_clean)) if len(y_clean) > 0 else 0.0
        }
    
    def get_feature_importance(self, top_n=35):
        """Get feature importance (coefficient magnitudes)
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Series with top feature importances
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first")
            
        return self.feature_importance.head(top_n)


class CTALight:
    """LightGBM model optimized for CTA positioning prediction"""

    def __init__(
        self,
        *,
        task: str = 'regression',
        use_gpu: bool = False,
        config_path: Optional[Union[str, Path]] = None,
        **lgb_params,
    ):
        """
        Initialize LightGBM model with CTA-optimized defaults

        Args:
            task: Learning task type ('regression', 'binary_classification', or 'multiclass').
            use_gpu: Enable GPU acceleration when True.
            config_path: Optional JSON file containing base LightGBM parameters.
            **lgb_params: LightGBM parameters to override defaults.
        """
        task_normalized = (task or 'regression').lower()
        self.use_gpu = bool(use_gpu)
        self.config_path = config_path
        self._user_params = dict(lgb_params)
        self.requested_task = task_normalized

        resolved_task = infer_task_from_target(None, task_normalized)
        default_params = build_default_params(
            task=resolved_task,
            use_gpu=self.use_gpu,
            config_path=self.config_path,
        )

        # Update with user parameters
        self.params: Dict[str, Any] = {**default_params, **lgb_params}
        self.task = resolved_task

        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance = None
        self.train_history = None

    def _maybe_update_task_from_y(self, y: Union[pd.Series, np.ndarray, list, tuple]) -> np.ndarray:
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = np.asarray(y)

        resolved_task = infer_task_from_target(y_values, self.requested_task)
        if resolved_task != self.task:
            base_params = build_default_params(
                task=resolved_task,
                use_gpu=self.use_gpu,
                config_path=self.config_path,
            )
            self.params = {**base_params, **self._user_params}
            self.task = resolved_task

        if self.task == 'multiclass' and 'num_class' not in self.params:
            unique_classes = np.unique(y_values[~np.isnan(y_values)])
            self.params['num_class'] = int(len(unique_classes))

        return y_values
        
    def fit(self, X, y, eval_set=None, early_stopping_rounds=50, num_boost_round=1000):
        """Fit the LightGBM model
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            eval_set: Validation set as tuple (X_val, y_val)
            early_stopping_rounds: Early stopping patience
            num_boost_round: Maximum number of boosting rounds
        """
        # Convert to numpy arrays and handle missing values
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        y = self._maybe_update_task_from_y(y)

        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            raise ValueError("No valid samples after removing NaN values")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_clean, label=y_clean, feature_name=self.feature_names)
        
        # Prepare validation set if provided
        valid_sets = [train_data]
        valid_names = ['train']
        
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
                
            # Clean validation data
            val_mask = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
            X_val_clean = X_val[val_mask]
            y_val_clean = y_val[val_mask]
            
            if len(X_val_clean) > 0:
                valid_data = lgb.Dataset(X_val_clean, label=y_val_clean, 
                                       reference=train_data, feature_name=self.feature_names)
                valid_sets.append(valid_data)
                valid_names.append('valid')
        
        # Train the model
        callbacks = []
        eval_result = {}
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
            callbacks.append(lgb.record_evaluation(eval_result))

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.is_fitted = True
        
        # Store feature importance
        importance_vals = self.model.feature_importance(importance_type='gain')
        self.feature_importance = pd.Series(
            importance_vals,
            index=self.feature_names
        ).sort_values(ascending=False)

        # Store training history if validation was used
        if len(valid_sets) > 1:
            metric = self.params.get('metric', 'rmse')
            if isinstance(metric, list):
                metric = metric[0]
            self.train_history = pd.DataFrame({name: res[metric] for name, res in eval_result.items()})

        return self
    
    def predict(self, X, num_iteration=None):
        """Make predictions
        
        Args:
            X: Features DataFrame or array
            num_iteration: Number of iterations to use (None for all)
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X, num_iteration=num_iteration)
    
    def cross_validate(self, X, y, cv_folds=5, early_stopping_rounds=50):
        """Perform time series cross-validation
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            cv_folds: Number of CV folds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary with CV results
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        y = self._maybe_update_task_from_y(y)
            
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Use TimeSeriesSplit for proper temporal validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        cv_scores = []
        feature_importances = []

        for train_idx, val_idx in tscv.split(X_clean):
            X_train, X_val = X_clean[train_idx], X_clean[val_idx]
            y_train, y_val = y_clean[train_idx], y_clean[val_idx]
            
            # Create ticker_sets
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
            
            # Train model
            model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(early_stopping_rounds)],
                num_boost_round=1000
            )
            
            # Make predictions and calculate score
            pred = model.predict(X_val)
            if self.task in {'binary_classification', 'multiclass'}:
                try:
                    score_val = log_loss(y_val, pred)
                except ValueError:
                    score_val = mean_squared_error(y_val, pred)
            else:
                score_val = mean_squared_error(y_val, pred)

            cv_scores.append(score_val)
            
            # Store feature importance
            importance = pd.Series(
                model.feature_importance(importance_type='gain'),
                index=feature_names
            )
            feature_importances.append(importance)
        
        # Average feature importances across folds
        avg_importance = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        
        return {
            'mean_mse': np.mean(cv_scores),
            'std_mse': np.std(cv_scores),
            'all_scores': cv_scores,
            'feature_importance': avg_importance
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics
        """
        predictions = self.predict(X_test)
        
        # Convert to numpy arrays for consistent handling
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        # Remove NaN values for evaluation
        mask = ~(np.isnan(predictions) | np.isnan(y_test))
        pred_clean = predictions[mask]
        y_clean = y_test[mask]
        
        if len(pred_clean) == 0:
            raise ValueError("No valid predictions for evaluation")
        
        return {
            'mse': mean_squared_error(y_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_clean, pred_clean)),
            'mae': mean_absolute_error(y_clean, pred_clean),
            'r2': r2_score(y_clean, pred_clean),
            'directional_accuracy': np.mean(np.sign(pred_clean) == np.sign(y_clean)) if len(y_clean) > 0 else 0.0
        }
    
    def get_feature_importance(self, importance_type='gain', top_n=20):
        """Get feature importance
        
        Args:
            importance_type: 'gain', 'split', or 'weight'
            top_n: Number of top features to return
            
        Returns:
            Series with top feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if importance_type != 'gain':
            # Recalculate importance with different type
            importance_vals = self.model.feature_importance(importance_type=importance_type)
            importance = pd.Series(
                importance_vals, 
                index=self.feature_names
            ).sort_values(ascending=False)
        else:
            importance = self.feature_importance
            
        return importance.head(top_n)
    
    def grid_search(self, X, y, param_grid=None, cv_folds=5, scoring='neg_mean_squared_error',
                   early_stopping_rounds=50, num_boost_round=1000, verbose=True):
        """Perform grid search for hyperparameter optimization
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            param_grid: Dictionary of parameters to search over. If None, uses default grid.
            cv_folds: Number of CV folds for validation
            scoring: Scoring metric ('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2')
            early_stopping_rounds: Early stopping patience for each model
            num_boost_round: Maximum boosting rounds for each model
            verbose: Whether to print progress
            
        Returns:
            Dictionary with best parameters, best score, and all results
        """
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.02, 0.05, 0.1],
                'num_leaves': [31, 63],
                'feature_fraction': [0.8, 0.95],
                'bagging_fraction': [0.85, 1.0],
                'min_child_samples': [10, 25],
                'reg_alpha': [0.0, 0.2],
                'reg_lambda': [0.1, 0.5],
            }
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        y = self._maybe_update_task_from_y(y)

        if scoring is None:
            scoring = 'neg_log_loss' if self.task in {'binary_classification', 'multiclass'} else 'neg_mean_squared_error'
            
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        if verbose:
            print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        best_score = float('-inf')
        best_params = None
        all_results = []

        def _compute_score(y_true, preds):
            if scoring == 'neg_mean_squared_error':
                return -mean_squared_error(y_true, preds)
            if scoring == 'neg_mean_absolute_error':
                return -mean_absolute_error(y_true, preds)
            if scoring == 'r2':
                return r2_score(y_true, preds)
            if scoring == 'neg_log_loss':
                try:
                    return -log_loss(y_true, preds)
                except ValueError:
                    return -mean_squared_error(y_true, preds)
            raise ValueError(f"Unsupported scoring metric: {scoring}")
        
        for i, param_combo in enumerate(param_combinations):
            # Create parameter dictionary for this combination
            current_params = dict(zip(param_names, param_combo))
            
            # Update base parameters with current combination
            test_params = {**self.params, **current_params}
            
            # Perform cross-validation for this parameter set
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean[train_idx], X_clean[val_idx]
                y_train, y_val = y_clean[train_idx], y_clean[val_idx]
                
                # Create ticker_sets
                train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
                
                # Train model with current parameters
                model = lgb.train(
                    test_params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    callbacks=[lgb.early_stopping(early_stopping_rounds)],
                    num_boost_round=num_boost_round,
                )
                
                # Make predictions and calculate score
                pred = model.predict(X_val)
                
                score = _compute_score(y_val, pred)

                cv_scores.append(score)
            
            # Calculate mean CV score
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            # Store results
            result = {
                'params': current_params,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'cv_scores': cv_scores
            }
            all_results.append(result)
            
            # Check if this is the best score
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = current_params
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(param_combinations)} combinations. Best score so far: {best_score:.4f}")
        
        if verbose:
            print(f"\nGrid search completed!")
            print(f"Best parameters: {best_params}")
            print(f"Best {scoring}: {best_score:.4f}")

        # Update model parameters with best found
        if best_params:
            self._user_params.update(best_params)
        self.params.update(best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_params_full': {**self.params, **best_params},
            'all_results': all_results,
            'scoring_metric': scoring
        }
    
    def fit_with_grid_search(self, X, y, param_grid=None, eval_set=None, cv_folds=5, 
                           scoring='neg_mean_squared_error', early_stopping_rounds=50, 
                           num_boost_round=1000, verbose=True):
        """Perform grid search and then fit the model with best parameters
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            param_grid: Dictionary of parameters to search over
            eval_set: Validation set as tuple (X_val, y_val)
            cv_folds: Number of CV folds for grid search
            scoring: Scoring metric for grid search
            early_stopping_rounds: Early stopping patience
            num_boost_round: Maximum boosting rounds
            verbose: Whether to print progress
            
        Returns:
            Dictionary with grid search results and fitted model
        """
        if verbose:
            print("Starting grid search for hyperparameter optimization...")
        
        # Perform grid search
        grid_results = self.grid_search(
            X, y, param_grid=param_grid, cv_folds=cv_folds, 
            scoring=scoring, early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round, verbose=verbose
        )
        
        if verbose:
            print(f"\nFitting final model with best parameters...")
        
        # Fit model with best parameters
        self.fit(X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, 
                num_boost_round=num_boost_round)
        
        if verbose:
            print("Model fitting completed!")
        
        return {
            'grid_search_results': grid_results,
            'model': self,
            'best_params': grid_results['best_params'],
            'best_score': grid_results['best_score']
        }
    
    def plot_importance(self, max_num_features=20, importance_type='gain'):
        """Plot feature importance (requires matplotlib)
        
        Args:
            max_num_features: Maximum number of features to plot
            importance_type: Type of importance to plot
        """
        try:
            lgb.plot_importance(
                self.model, 
                max_num_features=max_num_features,
                importance_type=importance_type
            )
        except ImportError:
            print("matplotlib required for plotting. Install with: pip install matplotlib")
            return self.get_feature_importance(importance_type, max_num_features)


class CTAXGBoost:
    """XGBoost model optimized for CTA positioning prediction with GPU support"""

    def __init__(self, use_gpu: bool = False, task: str = 'regression', **xgb_params):
        """Initialize XGBoost model with CTA-optimized defaults

        Args:
            use_gpu: Enable GPU acceleration (CUDA) when True
            task: Learning task type ('regression', 'binary_classification', or 'multiclass')
            **xgb_params: XGBoost parameters to override defaults
        """
        self.use_gpu = bool(use_gpu)
        self.task = task.lower() if task else 'regression'

        # Base parameters
        default_params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 200,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }

        # Task-specific parameters
        if self.task == 'binary_classification' or self.task == 'binary':
            default_params['objective'] = 'binary:logistic'
            default_params['eval_metric'] = 'logloss'
        elif self.task == 'multiclass':
            default_params['objective'] = 'multi:softprob'
            default_params['eval_metric'] = 'mlogloss'
        else:  # regression
            default_params['objective'] = 'reg:squarederror'
            default_params['eval_metric'] = 'rmse'

        # GPU parameters
        if self.use_gpu:
            default_params['tree_method'] = 'gpu_hist'
            default_params['gpu_id'] = 0
            default_params['predictor'] = 'gpu_predictor'
            # Remove n_jobs for GPU (not needed/compatible)
            default_params.pop('n_jobs', None)
        else:
            default_params['tree_method'] = 'hist'  # Use CPU histogram

        self.default_grid = dict(
            learning_rate=[0.03, 0.1],
            max_depth=[4, 6, 8],
            subsample=[0.85, 1.0],
            n_estimators=[150, 300],
            colsample_bytree=[0.8, 1.0]
        )

        self.params = {**default_params, **xgb_params}
        self.model = XGBRegressor(**self.params)
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance = None

    def fit(self, X, y, eval_set=None,):
        """Fit the XGBoost model"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            raise ValueError("No valid samples after removing NaN values")

        eval_sets = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            val_mask = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
            X_val_clean = X_val[val_mask]
            y_val_clean = y_val[val_mask]
            if len(X_val_clean) > 0:
                eval_sets = [(X_val_clean, y_val_clean)]

        self.model = XGBRegressor(**self.params)
        self.model.fit(
            X_clean,
            y_clean,
            eval_set=eval_sets,
            verbose=False
        )

        self.is_fitted = True
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        return self

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def cross_validate(self, X, y, cv_folds=5, scoring='neg_mean_squared_error'):
        """Perform time series cross-validation with GPU support"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        model = XGBRegressor(**self.params)

        # Don't use n_jobs with GPU (GPU already parallelizes)
        n_jobs_cv = 1 if self.use_gpu else -1
        cv_scores = cross_val_score(
            model, X_clean, y_clean, cv=tscv, scoring=scoring, n_jobs=n_jobs_cv
        )

        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'all_scores': cv_scores,
            'scoring_method': scoring
        }

    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test set"""
        predictions = self.predict(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        mask = ~(np.isnan(predictions) | np.isnan(y_test))
        pred_clean = predictions[mask]
        y_clean = y_test[mask]

        if len(pred_clean) == 0:
            raise ValueError("No valid predictions for evaluation")

        return {
            'mse': mean_squared_error(y_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_clean, pred_clean)),
            'mae': mean_absolute_error(y_clean, pred_clean),
            'r2': r2_score(y_clean, pred_clean),
            'directional_accuracy': np.mean(np.sign(pred_clean) == np.sign(y_clean)) if len(y_clean) > 0 else 0.0
        }

    def get_feature_importance(self, top_n=35):
        """Get feature importance"""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first")

        return self.feature_importance.head(top_n)

    def grid_search(self, X, y, param_grid=None, cv_folds=5,
                    scoring='neg_mean_squared_error', verbose=True):
        """Perform grid search over hyperparameters with GPU support

        Args:
            X: Features DataFrame or array
            y: Target Series or array
            param_grid: Dictionary of parameters to search over. If None, uses default grid.
            cv_folds: Number of CV folds for validation
            scoring: Scoring metric ('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2')
            verbose: Whether to print progress

        Returns:
            Dictionary with best parameters, best score, and cv_results
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if param_grid is None:
            param_grid = {
                'learning_rate': [0.03, 0.1],
                'max_depth': [4, 6, 8],
                'subsample': [0.85, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'n_estimators': [150, 300]
            }

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        base_model = XGBRegressor(**self.params)

        # Don't use n_jobs with GPU (GPU already parallelizes)
        n_jobs_grid = 1 if self.use_gpu else -1

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs_grid,
            verbose=1 if verbose else 0
        )
        grid.fit(X_clean, y_clean)

        self.params.update(grid.best_params_)
        self.model = grid.best_estimator_
        self.is_fitted = True
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        return {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
            'best_params_full': self.params,
            'cv_results': grid.cv_results_,
            'scoring_metric': scoring
        }

    def fit_with_grid_search(self, X, y, param_grid=None, eval_set=None, cv_folds=5,
                             scoring='neg_mean_squared_error',
                             verbose=True):
        """Perform grid search then fit the model with best parameters"""
        if verbose:
            print("Starting grid search for hyperparameter optimization...")
        param_grid = param_grid or self.default_grid

        grid_results = self.grid_search(
            X, y, param_grid=param_grid, cv_folds=cv_folds,
            scoring=scoring, verbose=verbose
        )

        if verbose:
            print("\nFitting final model with best parameters...")

        self.fit(X, y, eval_set=eval_set)

        if verbose:
            print("Model fitting completed!")

        return {
            'grid_search_results': grid_results,
            'model': self,
            'best_params': grid_results['best_params'],
            'best_score': grid_results['best_score']
        }


class CTARForest:
    """Random Forest model optimized for CTA positioning prediction

    Note: Random Forest in scikit-learn does not support GPU acceleration.
    For GPU-accelerated Random Forest, consider using cuML's RandomForestRegressor.
    This class maximizes CPU parallelization with n_jobs=-1.
    """

    def __init__(self, use_gpu: bool = False, task: str = 'regression', **rf_params):
        """Initialize Random Forest model with CTA-optimized defaults

        Args:
            use_gpu: Not used for Random Forest (scikit-learn RF is CPU-only).
                     Kept for interface consistency with other models.
            task: Learning task type ('regression', 'binary_classification', or 'multiclass')
            **rf_params: Random Forest parameters to override defaults
        """
        self.use_gpu = bool(use_gpu)  # Stored but not used (RF is CPU-only)
        self.task = task.lower() if task else 'regression'

        if self.use_gpu:
            warnings.warn(
                "GPU acceleration is not available for scikit-learn RandomForest. "
                "Using CPU with n_jobs=-1 for parallelization. "
                "For GPU support, consider using cuML's RandomForestRegressor.",
                UserWarning
            )

        # CTA-optimized default parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,  # Use all CPU cores
            'random_state': 42
        }

        # Default parameter grid for grid search
        self.default_grid = {
            'n_estimators': [150, 300],
            'max_depth': [6, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        self.params = {**default_params, **rf_params}
        self.model = RandomForestRegressor(**self.params)
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance = None
        
    def fit(self, X, y, eval_set=None):
        """Fit the Random Forest model
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array  
            eval_set: Not used for Random Forest but kept for interface consistency
        """
        # Convert to numpy arrays and handle missing values
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        if isinstance(y, pd.Series):
            y = y.values
            
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid samples after removing NaN values")
        
        # Create and fit the model
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        
        # Store feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions
        
        Args:
            X: Features DataFrame or array
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def cross_validate(self, X, y, cv_folds=5, scoring='neg_mean_squared_error'):
        """Perform time series cross-validation
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            cv_folds: Number of CV folds
            scoring: Scoring method
            
        Returns:
            Dictionary with CV results
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        model = RandomForestRegressor(**self.params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_clean, y_clean, 
            cv=tscv, scoring=scoring, n_jobs=-1
        )
        
        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'all_scores': cv_scores,
            'scoring_method': scoring
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics
        """
        predictions = self.predict(X_test)
        
        # Convert to numpy arrays for consistent handling
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        # Remove NaN values for evaluation
        mask = ~(np.isnan(predictions) | np.isnan(y_test))
        pred_clean = predictions[mask]
        y_clean = y_test[mask]
        
        if len(pred_clean) == 0:
            raise ValueError("No valid predictions for evaluation")
        
        return {
            'mse': mean_squared_error(y_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_clean, pred_clean)),
            'mae': mean_absolute_error(y_clean, pred_clean),
            'r2': r2_score(y_clean, pred_clean),
            'directional_accuracy': np.mean(np.sign(pred_clean) == np.sign(y_clean)) if len(y_clean) > 0 else 0.0,
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        }
    
    def get_feature_importance(self, top_n=35):
        """Get feature importance
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Series with top feature importances
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first")
            
        return self.feature_importance.head(top_n)
    
    def grid_search(self, X, y, param_grid=None, cv_folds=5, 
                   scoring='neg_mean_squared_error', verbose=True):
        """Perform grid search for hyperparameter optimization
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            param_grid: Dictionary of parameters to search over. If None, uses default grid.
            cv_folds: Number of CV folds for validation
            scoring: Scoring metric ('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2')
            verbose: Whether to print progress
            
        Returns:
            Dictionary with best parameters, best score, and all results
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        if isinstance(y, pd.Series):
            y = y.values
            
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [150, 300],
                'max_depth': [6, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        
        if verbose:
            print(f"Testing Random Forest grid search with {len(ParameterGrid(param_grid))} parameter combinations...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        base_model = RandomForestRegressor(**self.params)
        
        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid.fit(X_clean, y_clean)
        
        if verbose:
            print(f"\nGrid search completed!")
            print(f"Best parameters: {grid.best_params_}")
            print(f"Best {scoring}: {grid.best_score_:.4f}")
        
        # Update model parameters with best found
        self.params.update(grid.best_params_)
        self.model = grid.best_estimator_
        self.is_fitted = True
        
        # Update feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
            'best_params_full': self.params,
            'cv_results': grid.cv_results_,
            'scoring_metric': scoring
        }
    
    def fit_with_grid_search(self, X, y, param_grid=None, eval_set=None, cv_folds=5,
                           scoring='neg_mean_squared_error', verbose=True):
        """Perform grid search and then fit the model with best parameters
        
        Args:
            X: Features DataFrame or array
            y: Target Series or array
            param_grid: Dictionary of parameters to search over
            eval_set: Not used for Random Forest but kept for interface consistency
            cv_folds: Number of CV folds for grid search
            scoring: Scoring metric for grid search
            verbose: Whether to print progress
            
        Returns:
            Dictionary with grid search results and fitted model
        """
        if verbose:
            print("Starting grid search for Random Forest hyperparameter optimization...")
        
        param_grid = param_grid or self.default_grid
        
        # Perform grid search
        grid_results = self.grid_search(
            X, y, param_grid=param_grid, cv_folds=cv_folds,
            scoring=scoring, verbose=verbose
        )
        
        if verbose:
            print(f"\nFitting final model with best parameters...")
        
        # Model is already fitted by grid_search, but we'll refit to be consistent
        self.fit(X, y, eval_set=eval_set)
        
        if verbose:
            print("Random Forest model fitting completed!")
        
        return {
            'grid_search_results': grid_results,
            'model': self,
            'best_params': grid_results['best_params'],
            'best_score': grid_results['best_score']
        }
    
    def plot_importance(self, max_num_features=20):
        """Plot feature importance (requires matplotlib)
        
        Args:
            max_num_features: Maximum number of features to plot
        """
        try:
            import matplotlib.pyplot as plt
            
            importance = self.get_feature_importance(max_num_features)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importance)), importance.values)
            plt.yticks(range(len(importance)), importance.index)
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib required for plotting. Install with: pip install matplotlib")
            return self.get_feature_importance(max_num_features)
