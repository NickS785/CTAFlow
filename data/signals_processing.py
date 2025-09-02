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

            # COT positioning columns (standardized names)
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

        # Rename COT columns from actual format to standardized names
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

        # Apply column renaming for COT data
        df = df.rename(columns=cot_column_mapping)

        # Clean numeric columns - convert to numeric, keep NaN for later dropping
        cot_columns = list(self.column_mapping['cot_cols'].values())
        for col in cot_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle OHLCV columns - forward fill if needed
        ohlcv_columns = self.column_mapping['ohlcv_cols']
        for col in ohlcv_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


class TechnicalAnalysis:
    """Technical analysis enhanced with your COT positioning data"""

    def __init__(self):
        self.data_processor = DataProcessor()

    def calculate_enhanced_indicators(self, df, selected_indicators=None, normalize_momentum=False,
                                      vol_return_periods=[1, 5, 10, 20]):
        """Calculate only selected technical indicators
        
        Args:
            df: DataFrame with price and COT data
            selected_indicators: List of indicator groups to calculate 
                               ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence', 'vol_normalized']
            normalize_momentum: Whether to include volatility-normalized momentum
            vol_return_periods: Periods for volatility-normalized returns
        """
        indicators = pd.DataFrame(index=df.index)
        
        # Create working copy and map column names
        df = df.copy()
        
        # Map 'Last' to 'Close' if 'Close' doesn't exist but 'Last' does
        if 'Close' not in df.columns and 'Last' in df.columns:
            df['Close'] = df['Last']

        # If no selection, calculate all
        if selected_indicators is None:
            selected_indicators = ['moving_averages', 'macd', 'rsi', 'atr', 'volume', 'momentum', 'confluence']

        # Check if we have price data
        has_price_data = 'Close' in df.columns and df['Close'].notna().any()

        if has_price_data:
            # Moving Averages (normalized to be stationary)
            if 'moving_averages' in selected_indicators:
                sma_20 = df['Close'].rolling(20).mean()
                sma_50 = df['Close'].rolling(50).mean()
                sma_200 = df['Close'].rolling(200).mean()
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()

                # Normalize MAs relative to current price (stationary)
                indicators['sma_20_norm'] = (df['Close'] - sma_20) / df['Close']
                indicators['sma_50_norm'] = (df['Close'] - sma_50) / df['Close']
                indicators['sma_200_norm'] = (df['Close'] - sma_200) / df['Close']
                indicators['ema_12_norm'] = (df['Close'] - ema_12) / df['Close']
                indicators['ema_26_norm'] = (df['Close'] - ema_26) / df['Close']

                # Golden Cross / Death Cross signals (already stationary)
                indicators['golden_cross'] = (sma_50 > sma_200).astype(int)
                indicators['ma_50_200_ratio'] = sma_50 / sma_200

                # 20/50 MA crossover (intermediate term)
                indicators['ma_20_50_cross'] = (sma_20 > sma_50).astype(int)
                indicators['ma_20_50_ratio'] = sma_20 / sma_50

            # MACD (already stationary - difference of EMAs)
            if 'macd' in selected_indicators:
                # Calculate EMAs if not already calculated in moving_averages section
                if 'moving_averages' not in selected_indicators:
                    ema_12 = df['Close'].ewm(span=12).mean()
                    ema_26 = df['Close'].ewm(span=26).mean()

                    indicators['macd_line'] = ema_12 - ema_26
                    indicators['macd_signal'] = indicators['macd_line'].ewm(span=9).mean()
                    indicators['macd_histogram'] = indicators['macd_line'] - indicators['macd_signal']
                    indicators['macd_bullish'] = (indicators['macd_line'] > indicators['macd_signal']).astype(int)

            # RSI
            if 'rsi' in selected_indicators:
                indicators['rsi_14'] = self.calculate_rsi(df['Close'], 14)
                indicators['rsi_oversold'] = (indicators['rsi_14'] < 30).astype(int)
                indicators['rsi_overbought'] = (indicators['rsi_14'] > 70).astype(int)

            # ATR (normalized by price level to be stationary)
            if 'atr' in selected_indicators:
                atr_20 = self.calculate_atr(df, 20)
                indicators['atr_20_norm'] = atr_20 / df['Close']  # Normalize by price level
                indicators['atr_expansion'] = (atr_20 > atr_20.rolling(20).mean() * 1.5).astype(int)

            # Volume (stationary indicators only)
            if 'volume' in selected_indicators and 'Volume' in df.columns and df['Volume'].notna().any():
                vwap = self.calculate_vwap(df)
                # Only keep stationary VWAP deviation, not raw VWAP level
                indicators['vwap_deviation'] = (df['Close'] - vwap) / vwap
                indicators['vwap_extreme'] = (np.abs(indicators['vwap_deviation']) > 0.015).astype(int)
                
                # Use OBV momentum instead of absolute OBV levels
                obv = self.calculate_obv(df)
                indicators['obv_momentum_5'] = obv.pct_change(5)  # 5-period momentum
                indicators['obv_momentum_10'] = obv.pct_change(10)  # 10-period momentum

            # Momentum

            # Volatility-Normalized Returns
            if 'vol_normalized' in selected_indicators:
                vol_features = self.calculate_volatility_normalized_returns(df, vol_return_periods)
                for col in vol_features.columns:
                    indicators[col] = vol_features[col]

            if 'momentum' in selected_indicators:
                # Normalized Momentum (combines momentum with volatility normalization)
                if normalize_momentum or 'vol_normalized' in selected_indicators:
                    vol_features = self.calculate_volatility_normalized_returns(df, [5, 10, 20])
                    for col in vol_features.columns:
                        if 'vol_norm_return' in col:
                            period = col.split('_')[-1].replace('d', '')
                            indicators[f'normalized_momentum_{period}d'] = vol_features[col]

                else:
                    indicators['momentum_5'] = df['Close'].pct_change(5)
                    indicators['momentum_10'] = df['Close'].pct_change(10)
                    indicators['momentum_20'] = df['Close'].pct_change(20)

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


        return indicators.dropna(how='all')

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
            if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

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
        ewm_vol = daily_returns.ewm(span=vol_window).std()

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

        return normalized_returns.dropna(how='all')

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
