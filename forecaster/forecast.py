from data.signals_processing import COTProcessor, TechnicalAnalysis
from data.retrieval import fetch_data_sync
from data.data_client import DataClient as Client
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class CTAForecast:
    """Forecasting framework optimized for your COT-enhanced dataset"""

    def __init__(self, ticker_symbol, use_daily_data=True, **kwargs):
        self.data_processor = COTProcessor()
        self.technical_analyzer = TechnicalAnalysis()
        self.models = {}
        self.symbol = ticker_symbol
        
        # Fetch data and apply COT column renaming
        raw_data = fetch_data_sync(ticker_symbol, daily=use_daily_data, **kwargs)

        if raw_data is not None:
            # Apply COT column renaming from COTProcessor
            self.data = self.data_processor.load_and_clean_data(raw_data)

        else:
            self.data = None
            
        self.target = None
        self.features = None
        self.cot_features = None
        self.tech_features = None



    def prepare_features(self, include_technical=True,
                         selected_indicators=None,
                         normalize_momentum=False,
                         vol_return_periods=[1, 5, 10, 20],
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
        """
        df = self.data.copy()
        if remove_weekends:
            df = df.loc[(df.index.dayofweek != 6) & (df.index.dayofweek != 5)]

        if resample_before_calcs:
            df = self.resample_weekly(df=df, day_of_week=resample_day)


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
            cot_index = self.calculate_cot_index(mm_net)
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
        weekly_features = forecaster.resample_features_weekly(daily_features, 'Friday')
        
        # Resample to weekly Wednesday with mean aggregation
        weekly_features = forecaster.resample_features_weekly(
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
                     normalize_momentum=False, vol_return_periods=[1, 5, 10, 20],
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
            vol_return_periods: List of periods for volatility-normalized returns
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
        self.tech_features = self.get_technical_features_only(self.data, selected_indicators, normalize_momentum).columns.tolist()
        self.cot_features = self.get_cot_features_only(self.data, selected_cot_features).columns.tolist()
        
        # Apply weekly resampling to calculated features if requested

        return {
            'raw_data': self.data,
            'all_features': self.features,
            'technical_features': self.tech_features,
            'cot_features': self.cot_features
        }
    
    def train_model(self, model_type='ridge', target_type='return', forecast_horizon=10, 
                   test_size=0.2, **model_params):
        """Convenience method to train and validate models
        
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'elastic_net', 'lightgbm'
            target_type: 'return', 'cta_positioning', 'cot_index_change'
            forecast_horizon: Days ahead to predict
            test_size: Fraction of data for testing
            **model_params: Parameters to pass to the model
            
        Returns:
            Dictionary with model, predictions, and performance metrics
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
        
        # Time-based train/test split to avoid lookahead bias
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Initialize and train model
        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            model = CTALinear(model_type=model_type, **model_params)
        elif model_type == 'lightgbm':
            model = CTALight(**model_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Fit the model
        if model_type == 'lightgbm':
            # Use validation set for early stopping
            val_size = int(len(X_train) * 0.2)
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]
            
            model.fit(X_train_fit, y_train_fit, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Evaluate performance
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        # Store model in self.models
        model_key = f"{model_type}_{target_type}_{forecast_horizon}d"
        self.models[model_key] = model
        
        return {
            'model': model,
            'model_key': model_key,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': model.get_feature_importance(),
            'data_split_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_period': (X_train.index[0], X_train.index[-1]),
                'test_period': (X_test.index[0], X_test.index[-1])
            }
        }
    
    def cross_validate_model(self, model_type='ridge', target_type='return', 
                           forecast_horizon=10, cv_folds=5, **model_params):
        """Perform time series cross-validation
        
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'elastic_net', 'lightgbm'
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
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return cv_results

    def run_feature_selection(self, base_model, model_type='ridge', top_n=20,
                              test_size=0.2, **model_params):
        """Train a secondary model on top features from a base model.

        This method uses the ``feature_importance`` attribute of an
        already-fitted ``base_model`` to select the ``top_n`` most
        important features and trains another model using only those
        features.

        Args:
            base_model: Trained model instance or key in ``self.models``
                with a ``feature_importance`` attribute.
            model_type: Type of model to train on the selected features
                ('linear', 'ridge', 'lasso', 'elastic_net', 'lightgbm').
            top_n: Number of top features to use from ``base_model``.
            test_size: Fraction of samples for the test split.
            **model_params: Additional parameters passed to the new model.

        Returns:
            Dictionary with trained model, predictions, metrics and the
            selected feature names.
        """

        if isinstance(base_model, str):
            if base_model not in self.models:
                raise ValueError(f"Model '{base_model}' not found in self.models")
            base_model = self.models[base_model]

        if not hasattr(base_model, 'feature_importance') or base_model.feature_importance is None:
            raise ValueError("Base model must have feature_importance after fitting")

        if self.features is None or self.target is None:
            raise ValueError("Features and target must be prepared and base model trained first")

        top_features = base_model.feature_importance.head(top_n).index.tolist()
        X = self.features[top_features]
        y = self.target

        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            raise ValueError("No valid samples after removing NaN values")

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            model = CTALinear(model_type=model_type, **model_params)
        elif model_type == 'lightgbm':
            model = CTALight(**model_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if model_type == 'lightgbm':
            val_size = int(len(X_train) * 0.2)
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]
            model.fit(X_train_fit, y_train_fit, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)

        model_key = f"fs_{model_type}_top{top_n}"
        self.models[model_key] = model

        return {
            'model': model,
            'model_key': model_key,
            'selected_features': top_features,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': model.get_feature_importance(),
            'data_split_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_period': (X_train.index[0], X_train.index[-1]) if len(X_train) > 0 else (None, None),
                'test_period': (X_test.index[0], X_test.index[-1]) if len(X_test) > 0 else (None, None)
            }
        }
    
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
    
    def get_feature_importance(self, top_n=20):
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
    
    def __init__(self, **lgb_params):
        """
        Initialize LightGBM model with CTA-optimized defaults
        
        Args:
            **lgb_params: LightGBM parameters to override defaults
        """
        # CTA-optimized default parameters
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        # Update with user parameters
        self.params = {**default_params, **lgb_params}
        
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance = None
        self.train_history = None
        
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
            
        if isinstance(y, pd.Series):
            y = y.values
            
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
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
            
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
            self.train_history = pd.DataFrame(self.model.eval_train)
        
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
            
        if isinstance(y, pd.Series):
            y = y.values
            
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
            
            # Create datasets
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
            mse = mean_squared_error(y_val, pred)
            cv_scores.append(mse)
            
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




