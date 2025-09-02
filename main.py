from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from data.data_client import DataClient
from forecaster.forecast import CTAForecast


forecast = CTAForecast("NG_F")
forecast.prepare_data(resample_weekly_after=True,
                      weekly_day="Friday",
                      include_cot=True,
                      selected_cot_features=['positioning', 'extremes'],
                      selected_indicators= ['moving_averages', 'macd'], normalize_momentum=True)

forecast.features.dropna(axis=0, inplace=True)
res = forecast.train_model(model_type='ridge',target_type='positioning', forecast_horizon=11, alpha=0.7, l1_ratio=0.4)
