from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from CTAFlow.data import DataClient, FuturesCurveManager as FCM
from CTAFlow.forecaster import CTAForecast
from CTAFlow.data import SpreadData
from CTAFlow.features import CurveEvolutionAnalyzer, CurveShapeAnalyzer
import plotly.io as pio
pio.renderers.default = "browser"



# 1. Extract curves using new method
spread_data = SpreadData('GF')
analyzer = CurveEvolutionAnalyzer(spread_data)
analyzer.analyze_curve_evolution_drivers(21)
analyzer.plot_curve_evolution_analysis()

