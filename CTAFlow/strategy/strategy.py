from ..forecaster.forecast import CTAForecast
import pandas as pd

class RegimeStrategy:
    """Trading strategy using your comprehensive COT dataset"""

    def __init__(self):
        self.forecaster = CTAForecast()

    def analyze_positioning_regime(self, df):
        """Determine current positioning regime from your data"""
        latest_data = df.iloc[-1]

        # Calculate current positioning metrics
        mm_net = latest_data['money_manager_longs'] - latest_data['money_manager_shorts']
        commercial_net = (latest_data['producer_merchant_processor_user_longs'] -
                          latest_data['producer_merchant_processor_user_shorts'])

        # Calculate COT indices for current positioning
        mm_net_series = df['money_manager_longs'] - df['money_manager_shorts']
        commercial_net_series = (df['producer_merchant_processor_user_longs'] -
                                 df['producer_merchant_processor_user_shorts'])

        mm_cot_index = self.forecaster.calculate_cot_index(mm_net_series).iloc[-1]
        commercial_cot_index = self.forecaster.calculate_cot_index(commercial_net_series).iloc[-1]

        # Determine regime
        if mm_cot_index >= 80 and commercial_cot_index <= 20:
            regime = "EXTREME_BULLISH_SPEC"
            signal_strength = min(mm_cot_index / 100, (100 - commercial_cot_index) / 100)

        elif mm_cot_index <= 20 and commercial_cot_index >= 80:
            regime = "EXTREME_BEARISH_SPEC"
            signal_strength = min((100 - mm_cot_index) / 100, commercial_cot_index / 100)

        elif 40 <= mm_cot_index <= 60 and 40 <= commercial_cot_index <= 60:
            regime = "NEUTRAL"
            signal_strength = 0.1

        else:
            regime = "TRANSITIONAL"
            signal_strength = 0.3

        return {
            'regime': regime,
            'mm_cot_index': mm_cot_index,
            'commercial_cot_index': commercial_cot_index,
            'signal_strength': signal_strength,
            'mm_net_position': mm_net,
            'commercial_net_position': commercial_net,
            'positioning_divergence': mm_net * commercial_net < 0
        }

    def generate_forecast_signals(self, df, forecast_horizon=10):
        """Generate 5-15 day forecast signals using your COT data"""

        # Prepare features using your data structure
        features = self.forecaster.prepare_features(df)

        # Current positioning analysis
        positioning_regime = self.analyze_positioning_regime(df)

        # Generate signals based on COT extremes and flows
        signals = []

        # Signal 1: COT Extreme Reversal
        if positioning_regime['regime'] in ['EXTREME_BULLISH_SPEC', 'EXTREME_BEARISH_SPEC']:
            direction = -1 if positioning_regime['regime'] == 'EXTREME_BULLISH_SPEC' else 1
            confidence = positioning_regime['signal_strength']

            signals.append({
                'type': 'COT_EXTREME_REVERSAL',
                'direction': direction,
                'confidence': confidence,
                'time_horizon': f"{forecast_horizon} days",
                'description': f"Extreme {positioning_regime['regime']} positioning suggests reversal"
            })

        # Signal 2: COT Flow Momentum
        mm_flow_4w = features['mm_net_flow_4w'].iloc[-1]
        if abs(mm_flow_4w) > features['mm_net_flow_4w'].rolling(52).std().iloc[-1]:
            direction = 1 if mm_flow_4w > 0 else -1

            signals.append({
                'type': 'COT_FLOW_MOMENTUM',
                'direction': direction,
                'confidence': 0.6,
                'time_horizon': f"{forecast_horizon} days",
                'description': f"Strong CTA flow momentum: {mm_flow_4w:,.0f} contracts"
            })

        # Signal 3: Commercial vs Speculative Divergence
        if positioning_regime['positioning_divergence']:
            # When commercials and speculators disagree, commercials usually win
            direction = 1 if positioning_regime['commercial_net_position'] > 0 else -1

            signals.append({
                'type': 'COMMERCIAL_DIVERGENCE',
                'direction': direction,
                'confidence': 0.7,
                'time_horizon': f"{forecast_horizon} days",
                'description': "Commercial-Speculative divergence favors commercial positioning"
            })

        return {
            'positioning_regime': positioning_regime,
            'forecast_signals': signals,
            'key_metrics': {
                'current_mm_net': positioning_regime['mm_net_position'],
                'current_commercial_net': positioning_regime['commercial_net_position'],
                'mm_cot_index': positioning_regime['mm_cot_index'],
                'commercial_cot_index': positioning_regime['commercial_cot_index'],
                'total_open_interest': df['market_participation'].iloc[-1]
            }
        }
