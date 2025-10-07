from talib import MA, SMA, EMA, MOM
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from CTAFlow.data import IntradayLeg, IntradaySpreadEngine, DataClient

cl_leg = IntradayLeg.load_from_scid('CL', base_weight=3.0, start_date="2015-01-01")
ho_leg = IntradayLeg.load_from_scid('HO', base_weight=-2.0,start_date="2015-01-01")
rb_leg = IntradayLeg.load_from_scid('RB', base_weight=-1.0, start_date="2015-01-01")

engine = IntradaySpreadEngine.from_legs([cl_leg, ho_leg, rb_leg])

# Get OHLC DataFrame with synchronized timestamps
ohlc_spread = engine.build_spread_series(return_ohlc=True)
