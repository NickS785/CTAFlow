from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from CTAFlow.utils import TenorInterpolator, PCAAnalyzer, create_tenor_grid, vol_weighted_returns, deseasonalize_monthly

from CTAFlow.features import CurveEvolutionAnalyzer
from CTAFlow.data import SpreadData
from CTAFlow.config import RAW_MARKET_DATA_PATH, DLY_DATA_PATH
import plotly.io as pio

pio.renderers.default = "browser"

const_maturity = pd.read_csv("C:\\Users\\nicho\PycharmProjects\CTAFlow\\NG_constant_maturity_interpolated.csv",)
const_maturity.values[:, 1:] = const_maturity.values[:, 1:].astype(float)
const_maturity.set_index(pd.to_datetime(const_maturity['date']), inplace=True, drop=True)
const_maturity.drop(columns='date', inplace=True)

returns = np.log(const_maturity/const_maturity.shift(1))

vol_weighted_ret = vol_weighted_returns(returns)
deseasonalized_ret = deseasonalize_monthly(vol_weighted_ret.values, vol_weighted_ret.index)