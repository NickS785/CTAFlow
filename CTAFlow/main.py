from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from CTAFlow.utils import create_tenor_grid, vol_weighted_returns, deseasonalize_monthly

from CTAFlow.features import CurveEvolutionAnalyzer
from CTAFlow.data import SpreadData
from CTAFlow.config import RAW_MARKET_DATA_PATH, DLY_DATA_PATH
import plotly.io as pio

pio.renderers.default = "browser"

cl_f = SpreadData("NG")
clanalyze = CurveEvolutionAnalyzer(cl_f)
taus = create_tenor_grid(1/4, 3.0, 'quarterly')
clanalyze.setup_constant_maturity(taus)
