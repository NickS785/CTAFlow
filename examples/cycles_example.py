from CTAFlow.data import read_exported_df, read_synthetic_csv
from CTAFlow.config import CSV_DIR
from CTAFlow.features.cyclical.CEEMDAN import CEEMDANCycleAnalyzer, CEEMDANCycleAnalysisConfig, RVConfig
import pandas as pd
import numpy as np
from pathlib import Path
cfg = CEEMDANCycleAnalysisConfig(
    trials=15, max_imfs=6
)
cl_f = read_exported_df(CSV_DIR / "CL_5min.csv")
cl_f = cl_f.loc["2015-01-01":]
analyzer = CEEMDANCycleAnalyzer()

analyzer.analyze(cl_f.Close)




