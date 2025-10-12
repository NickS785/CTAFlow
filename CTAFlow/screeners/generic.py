from CTAFlow.data import SyntheticSymbol, IntradayLeg, IntradayFileManager, DataClient
from CTAFlow.screeners import HistoricalScreener, ScreenParams
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from CTAFlow.config import DLY_DATA_PATH

two_h = timedelta(hours=2)

one_h = timedelta(hours=1)

london_tgt_times = ["03:30", "05:00", "07:00", "09:30"]

usa_tgt_times = ["08:30", "10:30", "13:30"]

session_starts = ["02:30", "08:30"]

session_ends = ["10:30", "15:00"]

london_winter_seasonality = ScreenParams("seasonality",months=[12,1,2,3], target_times=london_tgt_times, period_length=timedelta(hours=2), use_tick_data=True, scid_folder=DLY_DATA_PATH)

london_spring_seasonality = ScreenParams("seasonality", months=[3,4,5,], target_times=london_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

london_fall_seasonality = ScreenParams("seasonality", months=[9,10,11], target_times=london_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

london_summer_seasonality = ScreenParams("seasonality", months=[5,6,7,8], target_times=london_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

usa_winter_seasonality = ScreenParams("seasonality", months=[12,1,2,3], target_times=usa_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

usa_spring_seasonality = ScreenParams("seasonality", months=[3,4,5], target_times=usa_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

usa_fall_seasonality = ScreenParams("seasonality", months=[9,10,11], target_times=usa_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

usa_summer_seasonality = ScreenParams("seasonality", months=[5,6,7,8], target_times=usa_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

london_seasonality = ScreenParams("seasonality", target_times=london_tgt_times, period_length=two_h, use_tick_data=True, scid_folder=DLY_DATA_PATH)

usa_seasonality = ScreenParams("seasonality", target_times=usa_tgt_times, period_length=two_h, use_tick_data=True,scid_folder=DLY_DATA_PATH)

summer_momentum = ScreenParams("momentum", months=[5,6,7,8], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

spring_momentum = ScreenParams("momentum", months=[3,4,5], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

fall_momentum = ScreenParams("momentum", months=[9, 10,11],session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

winter_momentum = ScreenParams("momentum", months=[12,1,2,3], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

momentum_generic = ScreenParams("momentum", session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

usa_seasonals = [usa_winter_seasonality, usa_summer_seasonality, usa_spring_seasonality, usa_fall_seasonality, usa_seasonality]

london_seasonals = [london_fall_seasonality, london_winter_seasonality, london_spring_seasonality, london_summer_seasonality, london_seasonality]

momentums = [spring_momentum, summer_momentum, fall_momentum, winter_momentum, momentum_generic]



