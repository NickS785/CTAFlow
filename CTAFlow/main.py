from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from CTAFlow.data import DataClient
from CTAFlow.forecaster import CTAForecast
from CTAFlow.data import SpreadData
from CTAFlow.features import SpreadAnalyzer,CurveEvolution, CurveShapeAnalyzer, curve_visualization
import plotly.io as pio
pio.renderers.default = "browser"


# Pipeline Example
symbol = "CL"
sd = SpreadData(symbol)
test_idx = slice(3700, -1, 5)

test_curves = sd[test_idx]

evol = CurveEvolution()
evol.load_curves_bulk(test_curves, sd.index[test_idx])
evol.plot_3d_curve_surface(sd.seq_spreads[test_idx])