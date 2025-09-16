from talib import MA, SMA, EMA, MOM
import pandas as pd
import numpy as np
from CTAFlow.data import DataClient
from CTAFlow.data import SpreadData
from CTAFlow.features import CurveEvolutionAnalyzer
from CTAFlow.utils.tenor_interpolation import TenorInterpolator, create_tenor_grid
import plotly.io as pio
pio.renderers.default = "browser"

taus = create_tenor_grid(1, 3)
crude = SpreadData("ZC")

grid = TenorInterpolator(tenor_grid=taus, method='cubic_hermite')
expiry_data = {}
crude.dte = pd.DataFrame(crude.dte, columns=crude.curve_df.columns, index=crude.index)
for contract_idx, contract_label in enumerate(crude.curve_df.columns):
    # Get the contract identifier and approximate expiry
    last_date = crude.seq_dte.index[-1]
    dte = crude.dte.iloc[-1, contract_idx]
    if pd.notna(dte):
        expiry_date = last_date + pd.Timedelta(days=dte)
        expiry_data[contract_label] = expiry_date

expiry_series = pd.Series(expiry_data)

const_maturity = grid.interpolate(crude.curve_df, expiry_series)