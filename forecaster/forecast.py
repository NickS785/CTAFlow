from data.signals_processing import DataProcessor, TechnicalAnalysis
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
        self.data_processor = DataProcessor()
        self.technical_analyzer = TechnicalAnalysis()
        self.models = {}
        self.symbol = ticker_symbol
        
        # Fetch data and apply COT column renaming
        raw_data = fetch_data_sync(ticker_symbol, daily=use_daily_data, **kwargs)
        if raw_data is not None:
            # Apply COT column renaming from DataProcessor
            self.data = self._apply_cot_column_renaming(raw_data)

        else:
            self.data = None
            
        self.target = None
        self.features = None
        self.cot_features = None
        self.technical_features = None

    def _apply_cot_column_renaming(self, df):
        """Apply COT column renaming as done in DataProcessor"""
        # COT column mapping from DataProcessor
        cot_column_mapping = {
            'Open_Interest_All': 'market_participation',
            'Prod_Merc_Positions_Long_All': 'producer_merchant_processor_user_longs',
            'Prod_Merc_Positions_Short_All': 'producer_merchant_processor_user_shorts',
            'Swap_Positions_Long_All': 'swap_dealer_longs',
            'Swap__Positions_Short_All': 'swap_dealer_shorts',
            'Swap__Positions_Spread_All': 'swap_dealer_spreads',
            'M_Money_Positions_Long_All': 'money_manager_longs',
            'M_Money_Positions_Short_All': 'money_manager_shorts',
            'M_Money_Positions_Spread_All': 'money_manager_spreads',
            'Other_Rept_Positions_Long_All': 'other_reportable_longs',
            'Other_Rept_Positions_Short_All': 'other_reportable_shorts',
            'Tot_Rept_Positions_Long_All': 'total_reportable_longs',
            'Tot_Rept_Positions_Short_All': 'total_reportable_shorts',
            'NonRept_Positions_Long_All': 'non_reportable_longs',
            'NonRept_Positions_Short_All': 'non_reportable_shorts'
        }
        
        # Apply column renaming
        df_renamed = df.rename(columns=cot_column_mapping)
        
        # Clean numeric columns - convert to numeric, keep NaN for later dropping
        cot_columns = list(cot_column_mapping.values())
        for col in cot_columns:
            if col in df_renamed.columns:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
        
        return df_renamed

    def prepare_forecasting_features(self, include_technical=True, selected_indicators=None,
                                   normalize_momentum=False, vol_return_periods=[1, 5, 10, 20],
                                   include_cot=True, selected_cot_features=None):
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
        """
        df = self.data.copy()
        features = pd.DataFrame(index=df.index)

        # 1. COT Features (Primary advantage) - Using new selective method
        if include_cot:
            cot_features = self.calculate_enhanced_cot_features(selected_cot_features)
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

        return self.features


    def calculate_cot_index(self, series, window=52):
        """Calculate COT index (0-100 scale)"""
        rolling_min = series.rolling(window=window).min()
        rolling_max = series.rolling(window=window).max()
        cot_index = ((series - rolling_min) / (rolling_max - rolling_min + 1e-8)) * 100
        return cot_index.dropna()
    
    def _classify_positioning_regime(self, series, lookback=252):
        """Classify current positioning into regime (low/medium/high)
        
        Args:
            series: Series of positioning values
            lookback: Days to look back for percentile calculation
            
        Returns:
            Series with positioning regime classification (0=low, 1=medium, 2=high)
        """
        # Calculate rolling percentiles
        pos_25th = series.rolling(window=lookback).quantile(0.25)
        pos_75th = series.rolling(window=lookback).quantile(0.75)
        
        # Classify regime
        regime = pd.Series(1, index=series.index)  # Default to medium
        regime[series <= pos_25th] = 0  # Low positioning
        regime[series >= pos_75th] = 2  # High positioning
        
        return regime

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
        return self.target
    
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
    
    def resample_features_weekly(self, day_of_week='Friday'):
        """Resample calculated features to weekly frequency using 'last' aggregation
        
        Args:
            features_df: DataFrame with calculated features
            day_of_week: Day to resample on (default 'Friday')
            
        Returns:
            Resampled features DataFrame
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
        
        if self.features is None:
            raise ValueError("No features to resample. Run prepare_forecasting_features() first.")
            
        # Use 'last' aggregation for all feature columns
        self.features = self.features.resample(freq).last()
        
        # Drop any rows with all NaN values
        self.features = self.features.dropna(how='all')
        
        return self.features
    
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
        
        # Resample the features using 'last' aggregation
        self.resample_features_weekly(day_of_week)
        
        # Also resample target if it exists
        if self.target is not None:
            day_mapping = {
                'Monday': 'W-MON', 'Tuesday': 'W-TUE', 'Wednesday': 'W-WED',
                'Thursday': 'W-THU', 'Friday': 'W-FRI', 'Saturday': 'W-SAT', 'Sunday': 'W-SUN'
            }
            freq = day_mapping[day_of_week]
            self.target = self.target.resample(freq).last().dropna()
        
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
    
    def get_cot_features_only(self, selected_cot_features=None):
        """Get only COT-based features for separate analysis with selective calculation
        
        Args:
            selected_cot_features: List of COT feature groups to calculate
                                 ['positioning', 'flows', 'extremes', 'market_structure', 'interactions', 'spreads']
        """
        if selected_cot_features is None:
            selected_cot_features = ['positioning', 'flows', 'extremes', 'market_structure', 'interactions']
        
        return self.calculate_enhanced_cot_features(selected_cot_features)
    
    def calculate_enhanced_cot_features(self, selected_cot_features=None, flow_periods=None):
        """Calculate comprehensive COT features with selective computation
        
        Args:
            df: DataFrame with COT data (must have standardized column names)
            selected_cot_features: List of COT feature groups to calculate
                                 ['positioning', 'flows', 'extremes', 'market_structure', 'interactions', 'spreads']
        
        Returns:
            DataFrame with selected COT features
        """
        if selected_cot_features is None:
            selected_cot_features = ['positioning', 'flows', 'extremes', 'market_structure', 'interactions']

        df = self.data.copy()
        
        features = pd.DataFrame(index=df.index)
        
        # Check if we have required COT data
        has_cot_data = ('money_manager_longs' in df.columns and 
                       'money_manager_shorts' in df.columns and
                       'market_participation' in df.columns)
        
        if not has_cot_data:
            return features
        
        # 1. Core Positioning Features
        if 'positioning' in selected_cot_features:
            # Money Manager (CTA) positioning
            features['mm_net_position'] = df['money_manager_longs'] - df['money_manager_shorts']
            features['mm_gross_position'] = df['money_manager_longs'] + df['money_manager_shorts']
            features['mm_long_ratio'] = (
                df['money_manager_longs'] / 
                (df['money_manager_longs'] + df['money_manager_shorts'] + 1e-8)
            )
            
            # Commercial positioning (hedgers/producers)
            if 'producer_merchant_processor_user_longs' in df.columns:
                features['commercial_net'] = (
                    df['producer_merchant_processor_user_longs'] - 
                    df['producer_merchant_processor_user_shorts']
                )
                features['commercial_long_ratio'] = (
                    df['producer_merchant_processor_user_longs'] /
                    (df['producer_merchant_processor_user_longs'] + df['producer_merchant_processor_user_shorts'] + 1e-8)
                )
            
            # Swap dealer positioning
            if 'swap_dealer_longs' in df.columns:
                features['swap_net'] = df['swap_dealer_longs'] - df['swap_dealer_shorts']
                features['swap_long_ratio'] = (
                    df['swap_dealer_longs'] / 
                    (df['swap_dealer_longs'] + df['swap_dealer_shorts'] + 1e-8)
                )
            
            # Other reportables
            if 'other_reportable_longs' in df.columns:
                features['other_net'] = df['other_reportable_longs'] - df['other_reportable_shorts']
        
        # 2. Flow Features (Changes in positioning)
        if 'flows' in selected_cot_features and 'mm_net_position' in features:
            if flow_periods is None:
                flow_periods = [4, 13]
            for periods in flow_periods:  # 1w to 26w (6 months)
                features[f'mm_net_flow_{periods}w'] = features['mm_net_position'].diff(periods)

                if 'commercial_net' in features:
                    features[f'commercial_flow_{periods}w'] = features['commercial_net'].diff(periods)
                
                # Flow momentum (acceleration/deceleration)
                if periods >= 4:
                    flow_col = f'mm_net_flow_{periods}w'
                    if flow_col in features:
                        features[f'{flow_col}_momentum'] = features[flow_col].diff(periods//2)
        
        # 3. Positioning Extremes (COT Index methodology)
        if 'extremes' in selected_cot_features:
            positioning_cols = [col for col in features.columns if any(x in col for x in ['_net', '_long_ratio'])]
            
            for col in positioning_cols:
                if col in features and features[col].notna().any():
                    # COT Index (0-100 scale over 52-week window)
                    features[f'{col}_cot_index'] = self.calculate_cot_index(features[col], window=52)
                    
                    # Extreme positioning flags
                    features[f'{col}_extreme_long'] = (features[f'{col}_cot_index'] >= 85).astype(int)
                    features[f'{col}_extreme_short'] = (features[f'{col}_cot_index'] <= 15).astype(int)
                    features[f'{col}_extreme'] = (
                        (features[f'{col}_cot_index'] <= 15) | 
                        (features[f'{col}_cot_index'] >= 85)
                    ).astype(int)
                    
                    # Regime classification (low/medium/high based on percentiles)
                    features[f'{col}_regime'] = self._classify_positioning_regime(features[col])
        
        # 4. Market Structure Features
        if 'market_structure' in selected_cot_features:
            features['total_open_interest'] = df['market_participation']
            features['oi_change'] = features['total_open_interest'].pct_change()
            features['oi_momentum'] = features['total_open_interest'].pct_change(4)  # 4-week change
            
            # Market concentration
            if 'mm_gross_position' in features:
                features['mm_concentration'] = features['mm_gross_position'] / df['market_participation']
            
            if 'commercial_net' in features:
                commercial_gross = df.get('producer_merchant_processor_user_longs', 0) + df.get('producer_merchant_processor_user_shorts', 0)
                features['commercial_concentration'] = commercial_gross / df['market_participation']
            
            # Reportable vs Non-reportable
            if 'total_reportable_longs' in df.columns:
                features['reportable_ratio'] = (
                    (df['total_reportable_longs'] + df['total_reportable_shorts']) / 
                    (2 * df['market_participation'] + 1e-8)
                )
        
        # 5. Interaction Features (Key for CTA analysis)
        if 'interactions' in selected_cot_features:
            if 'mm_net_position' in features and 'commercial_net' in features:
                # Speculative vs Commercial positioning
                features['mm_vs_commercial'] = features['mm_net_position'] - features['commercial_net']
                
                # Positioning divergence (opposite directions)
                features['positioning_divergence'] = (
                    features['mm_net_position'] * features['commercial_net'] < 0
                ).astype(int)
                
                # Positioning alignment strength
                features['positioning_alignment'] = np.abs(
                    np.corrcoef(features['mm_net_position'].rolling(26).apply(lambda x: x.iloc[-1]), 
                               features['commercial_net'].rolling(26).apply(lambda x: x.iloc[-1]))[0,1]
                ) if len(features) > 26 else 0.5
                
            # Smart money vs dumb money (commercial vs speculative)
            if 'commercial_net' in features and 'mm_net_position' in features:
                features['smart_money_indicator'] = -features['commercial_net']  # Commercials are contrarian
                features['dumb_money_indicator'] = features['mm_net_position']   # Speculators tend to be wrong at extremes
        
        # 6. Spread Activity Features (Sophisticated positioning indicator)
        if 'spreads' in selected_cot_features and 'money_manager_spreads' in df.columns:
            features['mm_spread_activity'] = df['money_manager_spreads']
            
            if 'mm_gross_position' in features:
                features['spread_to_outright_ratio'] = (
                    df['money_manager_spreads'] / (features['mm_gross_position'] + 1e-8)
                )
            
            # Spread activity momentum
            features['spread_activity_flow'] = features['mm_spread_activity'].diff(4)  # 4-week change
            features['spread_ratio_change'] = features.get('spread_to_outright_ratio', pd.Series(0, index=df.index)).diff(4)
            
            # Complex positioning indicator (spreads suggest sophisticated strategies)
            features['sophisticated_positioning'] = (
                features['spread_to_outright_ratio'] > features['spread_to_outright_ratio'].rolling(52).quantile(0.75)
            ).astype(int) if 'spread_to_outright_ratio' in features else pd.Series(0, index=df.index)
        
        # Remove columns that are all NaN
        features = features.dropna(axis=1, how='all')
        
        return features
    
    def prepare_data(self, resample_weekly_first=False, resample_weekly_after=False, weekly_day='Friday',
                     include_technical=True, selected_indicators=None,
                     normalize_momentum=False, vol_return_periods=[1, 5, 10, 20],
                     include_cot=True, selected_cot_features=None):
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
        
        # Apply weekly resampling if requested
        if resample_weekly_first:
            raw_data = self.resample_weekly(raw_data, day_of_week=weekly_day)
        
        # Store data for training/validation use
        self.data = raw_data.copy()
        
        # Prepare different feature sets with selective technical and COT indicators
        all_features = self.prepare_forecasting_features(
            include_technical=include_technical,
            selected_indicators=selected_indicators,
            normalize_momentum=normalize_momentum,
            vol_return_periods=vol_return_periods,
            include_cot=include_cot,
            selected_cot_features=selected_cot_features
        )
        self.tech_features = self.get_technical_features_only(self.data, selected_indicators, normalize_momentum).columns.tolist()
        self.cot_features = self.get_cot_features_only(selected_cot_features).columns.tolist()
        
        # Apply weekly resampling to calculated features if requested
        if resample_weekly_after and self.features is not None:
            self.features = self.resample_features_weekly(day_of_week=weekly_day)
        
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
        
        # Remove NaN values and align features with targets
        valid_idx = ~(self.features.isna().any(axis=1) | target.isna())
        X = self.features[valid_idx]
        y = target[valid_idx]
        
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
        target = self.create_target_variable(self.data, forecast_horizon, target_type)
        
        # Remove NaN values and align features with targets
        valid_idx = ~(self.features.isna().any(axis=1) | target.isna())
        X = self.features[valid_idx]
        y = target[valid_idx]
        
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




