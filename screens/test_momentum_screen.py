from CTAFlow.screeners import ScreenParams, HistoricalScreener
from CTAFlow.data import read_exported_df
from CTAFlow.config import INTRADAY_DATA_PATH
import pandas as pd
import numpy as np
momentum_screens = [ScreenParams(screen_type='momentum', months=[12,1,2,3], session_starts=['08:30'], session_ends=['15:00']), ScreenParams(screen_type="momentum", months=[4,5,6], session_starts=['08:30'], session_ends=["15:00"]), ScreenParams(screen_type="momentum", session_starts=["08:30", "02:30"], session_ends=["15:00", "11:30"])]

tickers = ["RB", "CL", "NG"]
data = {}
for t in tickers:
    data[t] = read_exported_df(INTRADAY_DATA_PATH / f"CSV/{t}_5min.csv")

hs = HistoricalScreener(data)
momentum_result = hs.run_screens(momentum_screens)
