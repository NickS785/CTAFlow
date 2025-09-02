from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from data.data_client import DataClient
from forecaster.forecast import CTAForecast


forecast = CTAForecast("NG_F")
forecast.prepare_data(resample_weekly_after=False,
                      selected_cot_features=['positioning', 'flows','market_structure'],
                      selected_indicators=['momentum', 'vol_normalized', 'moving_averages', 'atr'])

forecast.features.dropna(axis=0, inplace=True)
forecast.train_model(target_type='positioning', forecast_horizon=5)