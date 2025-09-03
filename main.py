from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from data.data_client import DataClient
from forecaster.forecast import CTAForecast

forecast = CTAForecast("CL_F")
forecast.prepare_features(resample_after=True,
                          resample_day="Friday",
                          include_cot=True,
                          include_intraday=True,
                          selected_cot_features=['positioning', 'extremes', 'market_structure', 'flows'],
                          selected_indicators=['moving_averages', 'macd', 'momentum'])

forecast.features.dropna(axis=0, inplace=True)
res = forecast.train_model(model_type='ridge', target_type='positioning', forecast_horizon=7, alpha=0.7, l1_ratio=0.4)
new = forecast.run_selected_features(res['model'], model_type='ridge', alpha=0.8, l1_ratio=0.2, top_n=20)