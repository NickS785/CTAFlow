from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from CTAFlow.data import DataProcessor


dp = DataProcessor()
dp.update_market_from_scid("CL_F")