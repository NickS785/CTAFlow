from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from data.data_client import DataClient
from forecaster.forecast import CTAForecast

forecast = CTAForecast("CL_F")
forecast.prepare_features(resample_after=True,
                          resample_day="Wednesday",
                          include_cot=True,
                          include_intraday=True,
                          selected_cot_features=['positioning', 'market_structure', 'flows'],
                          selected_indicators=['moving_averages', 'macd'], normalize_momentum=True)

forecast.features.dropna(axis=0, inplace=True)
lasso = forecast.train_model(model_type='lasso', target_type='positioning', forecast_horizon=7, alpha=0.05, l1_ratio=0.1)
ridge_light = forecast.train_model(model_type='ridge', target_type='positioning', forecast_horizon=6, alpha = 0.7)
