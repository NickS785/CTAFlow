from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from CTAFlow.data import DataClient
from CTAFlow.data import SpreadData
from CTAFlow.features import CurveEvolutionAnalyzer
import plotly.io as pio
pio.renderers.default = "browser"



# 1. Extract curves using new method
spread_data = SpreadData('HO')
sd = spread_data[-100:-1]
cea = CurveEvolutionAnalyzer(spread_data)