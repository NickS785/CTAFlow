from talib import MA, SMA, EMA, MOM
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from CTAFlow.utils import create_tenor_grid, vol_weighted_returns, deseasonalize_monthly
from sklearn.decomposition import PCA
from CTAFlow.features import CurveEvolutionAnalyzer
from CTAFlow.data import SpreadData, DataClient
from CTAFlow.config import RAW_MARKET_DATA_PATH, DLY_DATA_PATH
import plotly.io as pio

pio.renderers.default = "browser"

zs = SpreadData("NG")
