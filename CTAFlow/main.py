from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from .data.data_client import DataClient
from .forecaster.forecast import CTAForecast

# Pipeline Example
forecast = CTAForecast("CL_F")
forecast.prepare_features(resample_after=True,
                          resample_day="Wednesday",
                          include_cot=True,
                          include_intraday=True,
                          selected_intraday_features=['rsv', 'rv'],
                          selected_cot_features=['positioning', 'flows', 'interactions', 'spreads'],
                          selected_indicators=['moving_averages', 'macd', 'rsi', 'momentum'], normalize_momentum=True)

forecast.features.dropna(axis=0, inplace=True)
forecast.create_target_variable(6, 'positioning')
res = forecast.train_model(model_type='lasso', target_type='positioning',forecast_horizon=6, alpha=0.01)
lasso_features = set(res['feature_importance'][res['feature_importance'] > 0.0].index.tolist())
test_2 = forecast.train_model(model_type="ridge",  target_type='positioning', forecast_horizon=6, alpha=0.7, l1_ratio=0.3)
ridge_features = set(test_2['feature_importance'].head(20).index)
