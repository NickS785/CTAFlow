from .retrieval import fetch_market_cot_data
import asyncio
import pandas as pd
import numpy as np


class DataProcessor:
    """Optimized processor for your specific data format"""

    def __init__(self):
        self.column_mapping = {
            # Standard OHLCV columns
            'date_col': '',  # First unnamed column
            'ohlcv_cols': ['Open', 'High', 'Low', 'Close', 'Volume'],

            # COT positioning columns
            'cot_cols': {
                'market_participation': 'market_participation',
                'commercial_longs': 'producer_merchant_processor_user_longs',
                'commercial_shorts': 'producer_merchant_processor_user_shorts',
                'swap_longs': 'swap_dealer_longs',
                'swap_shorts': 'swap_dealer_shorts',
                'swap_spreads': 'swap_dealer_spreads',
                'money_manager_longs': 'money_manager_longs',
                'money_manager_shorts': 'money_manager_shorts',
                'money_manager_spreads': 'money_manager_spreads',
                'other_longs': 'other_reportable_longs',
                'other_shorts': 'other_reportable_shorts',
                'total_reportable_longs': 'total_reportable_longs',
                'total_reportable_shorts': 'total_reportable_shorts',
                'non_reportable_longs': 'non_reportable_longs',
                'non_reportable_shorts': 'non_reportable_shorts'
            }
        }

    def load_and_clean_data(self, ticker):
        """Load and clean your specific data format"""
        # Load data
        df = asyncio.run(fetch_market_cot_data(ticker))

        # Clean numeric columns - replace None/NaN with 0 for COT data
        cot_columns = list(self.column_mapping['cot_cols'].values())
        for col in cot_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Handle OHLCV columns - forward fill if needed
        ohlcv_columns = self.column_mapping['ohlcv_cols']
        for col in ohlcv_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def calculate_cta_positioning_metrics(self, df):
        """Calculate CTA positioning metrics from your money manager data"""
        metrics = pd.DataFrame(index=df.index)

        # Primary CTA metrics (Money Manager = CTAs + Hedge Funds)
        metrics['cta_net_position'] = (
                df['money_manager_longs'] - df['money_manager_shorts']
        )

        metrics['cta_gross_position'] = (
                df['money_manager_longs'] + df['money_manager_shorts']
        )

        metrics['cta_long_ratio'] = (
                df['money_manager_longs'] /
                (df['money_manager_longs'] + df['money_manager_shorts'] + 1e-8)
        )

        # Commercial vs Speculative positioning
        metrics['commercial_net'] = (
                df['producer_merchant_processor_user_longs'] -
                df['producer_merchant_processor_user_shorts']
        )

        metrics['spec_vs_commercial'] = (
                metrics['cta_net_position'] - metrics['commercial_net']
        )

        # Market participation metrics
        metrics['cta_market_share'] = (
                metrics['cta_gross_position'] / df['market_participation']
        )

        metrics['total_open_interest'] = df['market_participation']

        # Positioning extremes (COT Index methodology)
        for col in ['cta_net_position', 'commercial_net', 'spec_vs_commercial']:
            metrics[f'{col}_cot_index'] = self.calculate_cot_index(metrics[col])

        return metrics

    def calculate_cot_index(self, series, window=52):
        """Calculate COT index (0-100 scale) for extreme positioning detection"""
        rolling_min = series.rolling(window=window).min()
        rolling_max = series.rolling(window=window).max()
        cot_index = ((series - rolling_min) / (rolling_max - rolling_min + 1e-8)) * 100
        return cot_index.fillna(50)  # Fill NaN with neutral 50

    def generate_positioning_signals(self, positioning_metrics):
        """Generate trading signals from positioning extremes"""
        signals = pd.DataFrame(index=positioning_metrics.index)

        # CTA positioning extreme signals
        signals['cta_extreme_long'] = (
                positioning_metrics['cta_net_position_cot_index'] >= 85
        ).astype(int)

        signals['cta_extreme_short'] = (
                positioning_metrics['cta_net_position_cot_index'] <= 15
        ).astype(int)

        # Commercial vs speculative divergence
        signals['commercial_divergence'] = (
                (positioning_metrics['commercial_net_cot_index'] <= 20) &
                (positioning_metrics['cta_net_position_cot_index'] >= 80)
        ).astype(int)

        # Momentum signals based on positioning changes
        signals['cta_position_momentum'] = (
                positioning_metrics['cta_net_position'].diff(periods=4) > 0
        ).astype(int)

        return signals


class TechnicalAnalysis:
    """Technical analysis enhanced with your COT positioning data"""

    def __init__(self):
        self.data_processor = DataProcessor()

    def calculate_enhanced_indicators(self, df, selected_indicators=None, normalize_momentum=False, vol_return_periods=[1, 5, 10, 20]):
        """Calculate only selected technical indicators
        
        Args:
            df: DataFrame with price and COT data
            selected_indicators: List of indicator groups to calculate 
                               ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence', 'vol_normalized']
            normalize_momentum: Whether to include volatility-normalized momentum
            vol_return_periods: Periods for volatility-normalized returns
        """
        indicators = pd.DataFrame(index=df.index)
        
        # If no selection, calculate all
        if selected_indicators is None:
            selected_indicators = ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence']
        
        # Check if we have price data
        has_price_data = 'Close' in df.columns and df['Close'].notna().any()
        
        if has_price_data:
            # Moving Averages
            if 'moving_averages' in selected_indicators:
                indicators['sma_20'] = df['Close'].rolling(20).mean()
                indicators['sma_50'] = df['Close'].rolling(50).mean()
                indicators['sma_200'] = df['Close'].rolling(200).mean()
                indicators['ema_12'] = df['Close'].ewm(span=12).mean()
                indicators['ema_26'] = df['Close'].ewm(span=26).mean()
                
                # Golden Cross / Death Cross signals (50/200 MA)
                indicators['golden_cross'] = (indicators['sma_50'] > indicators['sma_200']).astype(int)
                indicators['ma_50_200_ratio'] = indicators['sma_50'] / indicators['sma_200']
                
                # 20/50 MA crossover (intermediate term)
                indicators['ma_20_50_cross'] = (indicators['sma_20'] > indicators['sma_50']).astype(int)
                indicators['ma_20_50_ratio'] = indicators['sma_20'] / indicators['sma_50']
            
            # MACD
            if 'macd' in selected_indicators:
                if 'ema_12' not in indicators:
                    indicators['ema_12'] = df['Close'].ewm(span=12).mean()
                    indicators['ema_26'] = df['Close'].ewm(span=26).mean()
                
                indicators['macd_line'] = indicators['ema_12'] - indicators['ema_26']
                indicators['macd_signal'] = indicators['macd_line'].ewm(span=9).mean()
                indicators['macd_histogram'] = indicators['macd_line'] - indicators['macd_signal']
                indicators['macd_bullish'] = (indicators['macd_line'] > indicators['macd_signal']).astype(int)
            
            # RSI
            if 'rsi' in selected_indicators:
                indicators['rsi_14'] = self.calculate_rsi(df['Close'], 14)
                indicators['rsi_oversold'] = (indicators['rsi_14'] < 30).astype(int)
                indicators['rsi_overbought'] = (indicators['rsi_14'] > 70).astype(int)
            
            # ATR
            if 'atr' in selected_indicators:
                indicators['atr_20'] = self.calculate_atr(df, 20)
                indicators['atr_expansion'] = (indicators['atr_20'] > indicators['atr_20'].rolling(20).mean() * 1.5).astype(int)
            
            # Volume
            if 'volume' in selected_indicators and 'Volume' in df.columns and df['Volume'].notna().any():
                indicators['vwap'] = self.calculate_vwap(df)
                indicators['vwap_deviation'] = (df['Close'] - indicators['vwap']) / indicators['vwap']
                indicators['vwap_extreme'] = (np.abs(indicators['vwap_deviation']) > 0.015).astype(int)
                indicators['obv'] = self.calculate_obv(df)
            
            # Momentum
            if 'momentum' in selected_indicators:
                indicators['momentum_5'] = df['Close'].pct_change(5)
                indicators['momentum_10'] = df['Close'].pct_change(10)
                indicators['momentum_20'] = df['Close'].pct_change(20)
            
            # Volatility-Normalized Returns
            if 'vol_normalized' in selected_indicators:
                vol_features = self.calculate_volatility_normalized_returns(df, vol_return_periods)
                for col in vol_features.columns:
                    indicators[col] = vol_features[col]
            
            # Normalized Momentum (combines momentum with volatility normalization)
            if normalize_momentum and ('momentum' in selected_indicators or 'vol_normalized' in selected_indicators):
                vol_features = self.calculate_volatility_normalized_returns(df, [5, 10, 20])
                for col in vol_features.columns:
                    if 'vol_norm_return' in col:
                        period = col.split('_')[-1].replace('d', '')
                        indicators[f'normalized_momentum_{period}d'] = vol_features[col]
            
            # Confluence signals (require multiple indicators)
            if 'confluence' in selected_indicators:
                # Only calculate if we have the required base indicators
                if 'macd_bullish' in indicators and 'rsi_14' in indicators:
                    indicators['macd_rsi_bullish'] = (
                        (indicators['macd_bullish'] == 1) & (indicators['rsi_14'] < 30)
                    ).astype(int)
                    indicators['macd_rsi_bearish'] = (
                        (indicators['macd_bullish'] == 0) & (indicators['rsi_14'] > 70)
                    ).astype(int)

        # COT-based indicators (always calculate if data is available)
        if 'money_manager_longs' in df.columns and 'money_manager_shorts' in df.columns:
            cta_net = df['money_manager_longs'] - df['money_manager_shorts']

            # CTA Flow Momentum (rate of change in positioning)
            indicators['cta_flow_1w'] = cta_net.pct_change(periods=1)
            indicators['cta_flow_4w'] = cta_net.pct_change(periods=4)
            indicators['cta_flow_8w'] = cta_net.pct_change(periods=8)
            
            # CTA positioning momentum (32-week lookback per project goals)
            indicators['cta_momentum_32w'] = cta_net.pct_change(periods=32)

            # Commercial vs Speculative Pressure
            if 'producer_merchant_processor_user_longs' in df.columns:
                commercial_net = (df['producer_merchant_processor_user_longs'] -
                                  df['producer_merchant_processor_user_shorts'])
                indicators['commercial_pressure'] = commercial_net / df['market_participation']
                indicators['speculative_pressure'] = cta_net / df['market_participation']
                
                # Position Concentration
                indicators['position_concentration'] = (
                        (df['money_manager_longs'] + df['money_manager_shorts']) /
                        df['market_participation']
                )

                # Spread Activity (indicates sophisticated positioning)
                if 'money_manager_spreads' in df.columns:
                    indicators['spread_activity_ratio'] = (
                            df['money_manager_spreads'] /
                            (df['money_manager_longs'] + df['money_manager_shorts'] + 1e-8)
                    )

        return indicators.fillna(method='bfill').fillna(0)

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df, period=20):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['Volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def calculate_volatility_normalized_returns(self, df, return_periods=[1, 5, 10, 20], vol_window=63, vol_alpha=None):
        """Calculate volatility-normalized returns using exponentially weighted standard deviation
        
        Args:
            df: DataFrame with price data (must have 'Close' column)
            return_periods: List of periods for return calculation
            vol_window: Window for volatility calculation (default 63 trading days ~ 3 months)
            vol_alpha: Alpha for exponential weighting. If None, uses 2/(vol_window+1)
            
        Returns:
            DataFrame with volatility-normalized returns
        """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column for volatility normalization")
        
        normalized_returns = pd.DataFrame(index=df.index)
        
        # Calculate daily returns
        daily_returns = df['Close'].pct_change()
        
        # Set alpha for exponential weighting if not provided
        if vol_alpha is None:
            vol_alpha = 2.0 / (vol_window + 1)
        
        # Calculate exponentially weighted volatility (standard deviation)
        ewm_vol = daily_returns.ewm(span=vol_window, alpha=vol_alpha).std()
        
        # Calculate raw returns for different periods
        for period in return_periods:
            raw_return = df['Close'].pct_change(periods=period)
            
            # Normalize by volatility (using period-appropriate scaling)
            vol_scaling = np.sqrt(period)  # Scale volatility for multi-period returns
            scaled_vol = ewm_vol * vol_scaling
            
            # Avoid division by zero
            normalized_return = raw_return / (scaled_vol + 1e-8)
            
            normalized_returns[f'vol_norm_return_{period}d'] = normalized_return
        
        # Also add the volatility series itself as a feature
        normalized_returns['ewm_volatility_63d'] = ewm_vol
        normalized_returns['vol_regime'] = self._classify_volatility_regime(ewm_vol)
        
        return normalized_returns.fillna(0)
    
    def _classify_volatility_regime(self, volatility_series, lookback=252):
        """Classify current volatility into regime (low/medium/high)
        
        Args:
            volatility_series: Series of volatility values
            lookback: Days to look back for percentile calculation
            
        Returns:
            Series with volatility regime classification (0=low, 1=medium, 2=high)
        """
        # Calculate rolling percentiles
        vol_25th = volatility_series.rolling(window=lookback).quantile(0.25)
        vol_75th = volatility_series.rolling(window=lookback).quantile(0.75)
        
        # Classify regime
        regime = pd.Series(1, index=volatility_series.index)  # Default to medium
        regime[volatility_series <= vol_25th] = 0  # Low volatility
        regime[volatility_series >= vol_75th] = 2  # High volatility
        
        return regime
