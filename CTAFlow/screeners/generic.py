from CTAFlow.data import SyntheticSymbol, IntradayLeg, IntradayFileManager, DataClient
from CTAFlow.screeners import HistoricalScreener, ScreenParams, OrderflowParams
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from CTAFlow.config import DLY_DATA_PATH

two_h = timedelta(hours=2)
energy_tickers = ["HO", "RB", "CL", "NG"]
period_time = timedelta(hours=1, minutes=30)
london_start, london_end = time(hour=2, minute=30), time(hour=11, minute=30)
usa_start, usa_end = time(hour=8, minute=30), time(hour=15, minute=30)
winter, spring, summer, fall = {'months':[12,1,2,3]}, {'months':[3,4,5]}, {'months':[5,6,7,8]}, {'months':[9,10,11]}
livestock_session =dict(seasonality_session_start=time(hour=8, minute=30), seasonality_session_end=time(hour=13, minute=0), session_starts=["08:30"], session_ends=["13:00"])
livestock_tgt_times = ["9:00", "9:30", "11:00", "12:00"]
seasonal_livestock_params = dict(screen_type="seasonality", target_times=livestock_tgt_times, period_length=period_time)
momentum_livestock_params = dict(screen_type="momentum", sess_start_hrs=1, sess_start_minutes=30, st_momentum_days=5)
livestock_seasonal = ScreenParams( name="livestock_seasonal", **seasonal_livestock_params, **livestock_session)
livestock_winter = ScreenParams(name="livestock_seasonal_winter", **seasonal_livestock_params, **livestock_session, **winter)
livestock_summer = ScreenParams(name="livestock_seasonal_summer", **seasonal_livestock_params, **livestock_session, **summer)
livestock_fall = ScreenParams(name="livestock_seasonal_fall", **seasonal_livestock_params, **livestock_session, **fall)
livestock_spring = ScreenParams(name="livestock_seasonal_spring", **seasonal_livestock_params, **livestock_session, **spring)
livestock_momentum = ScreenParams(name="livestock_momentum", **momentum_livestock_params, **livestock_session)
livestock_winter_momentum = ScreenParams(name="livestock_winter_momo", **momentum_livestock_params, **livestock_session, **winter)
livestock_spring_momentum = ScreenParams(name="livestock_spring_momo", **momentum_livestock_params, **livestock_session, **spring)
livestock_fall_momentum = ScreenParams(name="livestock_fall_momo", **momentum_livestock_params, **livestock_session, **fall)
livestock_summer_momentum = ScreenParams(name="livestock_summer_momo", **momentum_livestock_params, **livestock_session, **summer)

livestock_seasonals = [livestock_seasonal, livestock_spring, livestock_fall, livestock_winter, livestock_summer]
livestock_momentums = [livestock_momentum, livestock_spring_momentum, livestock_fall_momentum, livestock_winter_momentum, livestock_summer_momentum]
london_tgt_times = ["3:00", "03:30", "02:30", "07:30", "09:30"]
usa_tgt_times = ["08:30", "10:30","9:00", "13:30"]

session_starts = ["02:30", "08:30"]

session_ends = ["10:30", "15:00"]
default_london_settings = {"screen_type":"seasonality","target_times":london_tgt_times, "period_length":period_time, "seasonality_session_start":london_start, "seasonality_session_end":london_end}
default_usa_settings = {"screen_type":"seasonality", "target_times":usa_tgt_times, "period_length":period_time, "seasonality_session_start":usa_start, "seasonality_session_end":usa_end}
london_winter_seasonality = ScreenParams("seasonality",name="london_winter", months=[12,1,2,3], target_times=london_tgt_times, period_length=period_time, seasonality_session_start=london_start, seasonality_session_end=london_end)

london_spring_seasonality = ScreenParams("seasonality",name="london_spring" ,months=[3,4,5,], target_times=london_tgt_times, period_length=period_time, seasonality_session_start=london_start, seasonality_session_end=london_end)

london_fall_seasonality = ScreenParams("seasonality", name="london_fall", months=[9,10,11], target_times=london_tgt_times, period_length=period_time, seasonality_session_start=london_start, seasonality_session_end=london_end)

london_summer_seasonality = ScreenParams("seasonality", name="london_summer", months=[5,6,7,8], target_times=london_tgt_times, period_length=period_time, seasonality_session_start=london_start, seasonality_session_end=london_end)

usa_winter_seasonality = ScreenParams("seasonality", name="usa_winter", months=[12,1,2,3], target_times=usa_tgt_times, period_length=period_time, seasonality_session_start=usa_start, seasonality_session_end=usa_end)
usa_q1 = ScreenParams(name="usa_q1", months=[1,2,3], **default_usa_settings)
usa_q2 = ScreenParams(name="usa_q2", months=[4,5,6], **default_usa_settings)
usa_q3 = ScreenParams(name="usa_q3", months=[7,8,9], **default_usa_settings)
usa_q4 = ScreenParams(name="usa_q4", months=[10,11,12], **default_usa_settings)
usa_all = ScreenParams(name="usa_all", **default_usa_settings)
usa_quarterly_params = [usa_q1, usa_q2, usa_q3, usa_q4, usa_all]

london_q1 = ScreenParams(**default_london_settings, name="london_q1", months=[1,2,3])
london_q2 = ScreenParams(**default_london_settings, name="london_q2", months=[4,5,6])
london_q3 = ScreenParams(**default_london_settings, name="london_q3", months=[7,8,9])
london_q4 = ScreenParams(**default_london_settings, name="london_q4", months=[10,11,12])
london_all = ScreenParams(**default_london_settings)
london_quarterly_params = [london_q1, london_q2, london_q3, london_q4, london_all]

usa_spring_seasonality = ScreenParams("seasonality",name="usa_spring", months=[3,4,5], target_times=usa_tgt_times, period_length=period_time, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

usa_fall_seasonality = ScreenParams("seasonality",name="usa_fall" ,months=[9,10,11], target_times=usa_tgt_times, period_length=period_time, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

usa_summer_seasonality = ScreenParams("seasonality",name="usa_summer" ,months=[5,6,7,8], target_times=usa_tgt_times, period_length=period_time, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

london_seasonality = ScreenParams("seasonality", name="london_all",target_times=london_tgt_times, period_length=period_time, seasonality_session_end=london_end, seasonality_session_start=london_start)

usa_seasonality = ScreenParams("seasonality",name="usa_all", target_times=usa_tgt_times, period_length=period_time, seasonality_session_start=usa_start, seasonality_session_end=usa_end)

summer_momentum = ScreenParams("momentum",name="usa_summer_momentum", months=[5,6,7,8], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

spring_momentum = ScreenParams("momentum",name="usa_spring_momentum", months=[3,4,5], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

fall_momentum = ScreenParams("momentum",name="usa_fall_momentum", months=[9, 10,11],session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

winter_momentum = ScreenParams("momentum",name="usa_winter_momentum", months=[12,1,2,3], session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

momentum_generic = ScreenParams("momentum",name="momentum_generic", session_starts=session_starts, session_ends=session_ends, sess_start_hrs=1, sess_start_minutes=30)

usa_seasonals = [usa_winter_seasonality, usa_summer_seasonality, usa_spring_seasonality, usa_fall_seasonality, usa_seasonality]

london_seasonals = [london_fall_seasonality, london_winter_seasonality, london_spring_seasonality, london_summer_seasonality, london_seasonality]

momentums = [spring_momentum, summer_momentum, fall_momentum, winter_momentum, momentum_generic]
us_session = {'session_start':"08:30", 'session_end':"15:30"}
london_session = {'session_start':"02:30", "session_end":"10:30"}

fall = {'month_filter':[9,10,11]}
spring = {'month_filter': [3,4,5]}
winter = {'month_filter': [12,1,2,3]}
summer = {'month_filter': [5,6,7,8]}
us_of_screen = OrderflowParams(session_start="08:30", session_end="15:30")
us_of_fall = OrderflowParams(**us_session, **fall, vpin_window=25    , name="us_of_fall")
us_of_spring = OrderflowParams(**us_session, **spring, vpin_window=25, name="us_of_spring")
us_of_winter = OrderflowParams(session_start="08:30", session_end="15:30", month_filter=[11,12,1,2,3], name="us_winter")
us_of_summer = OrderflowParams(session_start="08:30", session_end="15:30", month_filter=[5,6,7,8], name="us_summer")
london_of = OrderflowParams(session_start="02:30", session_end="10:30", vpin_window=30, name="london_of")
london_of_winter = OrderflowParams(session_start="02:30", session_end="10:30", month_filter=[11,12,1,2,3], name="london_winter")
london_of_summer = OrderflowParams(session_start="02:30", session_end="10:30", month_filter=[5,6,7,8], name="london_summer")
london_of_spring = OrderflowParams(**london_session, **spring, vpin_window=25, name="london_of_spring")
london_of_fall = OrderflowParams(**london_session, **fall, vpin_window=25, name="london_of_fall")
london_of_all = [london_of, london_of_winter, london_of_summer, london_of_spring, london_of_fall]
us_of_all = [us_of_screen, us_of_fall, us_of_summer, us_of_spring, us_of_winter]

