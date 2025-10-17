from CTAFlow.data import SyntheticSymbol, IntradayLeg, IntradayFileManager, DataClient
from CTAFlow.screeners import HistoricalScreener, ScreenParams, OrderflowParams
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from CTAFlow.config import DLY_DATA_PATH

two_h = timedelta(hours=2)
energy_tickers = ["HO", "RB", "CL", "NG"]
one_h = timedelta(hours=1)
london_start, london_end = time(hour=2, minute=30), time(hour=11, minute=30)
usa_start, usa_end = time(hour=8, minute=30), time(hour=15, minute=30)

london_tgt_times = ["03:30", "05:00", "07:00", "09:30"]
usa_tgt_times = ["08:30", "10:30", "13:30"]

session_starts = ["02:30", "08:30"]

session_ends = ["10:30", "15:00"]

london_winter_seasonality = ScreenParams("seasonality",months=[12,1,2,3], target_times=london_tgt_times, period_length=timedelta(hours=2), seasonality_session_start=london_start, seasonality_session_end=london_end)

london_spring_seasonality = ScreenParams("seasonality", months=[3,4,5,], target_times=london_tgt_times, period_length=two_h, seasonality_session_start=london_start, seasonality_session_end=london_end)

london_fall_seasonality = ScreenParams("seasonality", months=[9,10,11], target_times=london_tgt_times, period_length=two_h,seasonality_session_start=london_start, seasonality_session_end=london_end)

london_summer_seasonality = ScreenParams("seasonality", months=[5,6,7,8], target_times=london_tgt_times, period_length=two_h,seasonality_session_start=london_start, seasonality_session_end=london_end )

usa_winter_seasonality = ScreenParams("seasonality", months=[12,1,2,3], target_times=usa_tgt_times, period_length=two_h, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

usa_spring_seasonality = ScreenParams("seasonality", months=[3,4,5], target_times=usa_tgt_times, period_length=two_h, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

usa_fall_seasonality = ScreenParams("seasonality", months=[9,10,11], target_times=usa_tgt_times, period_length=two_h,  seasonality_session_start=usa_start, seasonality_session_end=usa_end)

usa_summer_seasonality = ScreenParams("seasonality", months=[5,6,7,8], target_times=usa_tgt_times, period_length=two_h, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

london_seasonality = ScreenParams("seasonality", target_times=london_tgt_times, period_length=two_h, seasonality_session_end=london_end, seasonality_session_start=london_start)

usa_seasonality = ScreenParams("seasonality", target_times=usa_tgt_times, period_length=two_h, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

summer_momentum = ScreenParams("momentum", months=[5,6,7,8], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

spring_momentum = ScreenParams("momentum", months=[3,4,5], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

fall_momentum = ScreenParams("momentum", months=[9, 10,11],session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

winter_momentum = ScreenParams("momentum", months=[12,1,2,3], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

momentum_generic = ScreenParams("momentum", session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

usa_seasonals = [usa_winter_seasonality, usa_summer_seasonality, usa_spring_seasonality, usa_fall_seasonality, usa_seasonality]

london_seasonals = [london_fall_seasonality, london_winter_seasonality, london_spring_seasonality, london_summer_seasonality, london_seasonality]

momentums = [spring_momentum, summer_momentum, fall_momentum, winter_momentum, momentum_generic]

us_of_screen = OrderflowParams(session_start="08:30", session_end="15:30")
us_of_winter = OrderflowParams(session_start="08:30", session_end="15:30", month_filter=[11,12,1,2,3], name="us_winter")
us_of_summer = OrderflowParams(session_start="08:30", session_end="15:30", month_filter=[5,6,7,8], name="us_summer")
london_of_screen = OrderflowParams(session_start="02:30", session_end="11:30", vpin_window=30)
london_of_winter = OrderflowParams(session_start="02:30", session_end="11:30", month_filter=[11,12,1,2,3], name="london_winter")
london_of_summer = OrderflowParams(session_start="02:30", session_end="11:30", month_filter=[5,6,7,8], name="london_summer")