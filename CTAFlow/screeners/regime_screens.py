import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time, date, timedelta
from ..features.regime_classification import RegimeSpecification, TrendRegimeClassifier, VolatilityRegimeClassifier
from . import HistoricalScreener, PatternExtractor, ScreenParams
from ..data import read_exported_df, read_synthetic_csv
from ..config import INTRADAY_DATA_PATH

from ..strategy import ScreenerPipeline

fast_trend = {
    'price_col': "Close",
    "fast_window": 12,
    "slow_window": 26,
    "method":"ema",
    "neutral_band":0.02
}
slow_trend = {
    'price_col': "Close",
    "fast_window": 50,
    "slow_window": 200,
    "neutral_band":0.06,
    "method":"sma"
}

macd_classifier =  TrendRegimeClassifier(**fast_trend)
sma_clf = TrendRegimeClassifier(**slow_trend)
target_regimes = [-1,1]
tickers = ["GC","CL", "RB", "HG", "NG"]

base_regime_screen_params = ScreenParams(screen_type="momentum",
                                        name="ema_trend_m",
                                        session_starts=["02:30", "08:30"],
                                        session_ends=["10:30", "15:00"],
                                        st_momentum_days=5,
                                         sess_start_hrs=1,
                                         sess_start_minutes=30,
                                         regime_settings=macd_classifier,
                                         target_regimes=target_regimes)

lt_regime_screen_params = ScreenParams(screen_type="momentum",
                                        name="sma_trend_m",
                                        session_starts=["02:30", "08:30"],
                                        session_ends=["10:30", "15:00"],
                                        st_momentum_days=5,
                                       sess_start_hrs=1,
                                       sess_start_minutes=30, regime_settings=sma_clf, target_regimes=target_regimes)

seasonal_screen_params_macd = ScreenParams(screen_type="seasonality",name="ema_trend_s",
                                      seasonality_session_start=time(8, 30),
                                      seasonality_session_end=time(15, 0),
                                      target_regimes=[-1,1],
                                      period_length=timedelta(hours=1, minutes=30),
                                      target_times=["08:30", "9:30", "10:00", "13:30"],
                                      regime_settings=macd_classifier)
seasonal_screen_params_lt = ScreenParams(screen_type="seasonality",name="sma_trend_s",
                                      seasonality_session_start=time(8, 30),
                                      seasonality_session_end=time(15, 0),
                                      target_regimes=[-1,1],
                                      period_length=timedelta(hours=1, minutes=30),
                                      target_times=["08:30", "9:30", "10:00", "13:30"],
                                      regime_settings=sma_clf)
if __name__ == "__main__":
    data = {t:read_exported_df(INTRADAY_DATA_PATH / f"CSV/{t}_5min.csv") for t in tickers }
    data["CS"] = read_synthetic_csv(INTRADAY_DATA_PATH / f'synthetics/crack_spread.csv')
    data = {k:v.loc["2020":] for k,v in data.items()}
    hs = HistoricalScreener(data)
    result = hs.run_screens([seasonal_screen_params_lt, seasonal_screen_params_macd])
    pe = PatternExtractor(screener=hs, results=result)
    ranked = pe.rank_patterns()
    sp = ScreenerPipeline()
