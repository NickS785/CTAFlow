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
                          selected_cot_features=['positioning', 'extremes', 'flows', 'interactions', 'spreads'],
                          selected_indicators=['moving_averages', 'macd', 'momentum', 'rsi', 'volume'], normalize_momentum=True)

forecast.features.dropna(axis=0, inplace=True)

res = forecast.train_model(model_type='ridge', target_type='positioning',forecast_horizon=6, alpha=0.7)
ridge_light = forecast.run_selected_features(base_model="ridge_positioning_6d", top_n=30, model_type='xgboost', use_grid_search=True)
forecast.save_model(ridge_light)