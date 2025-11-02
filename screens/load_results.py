from CTAFlow.data import ResultsClient, AsyncParquetWriter
from CTAFlow.screeners import PatternExtractor, HistoricalScreener, OrderflowScanner
import pandas as pd
import numpy as np

us_winter_results = PatternExtractor.load_summaries_from_results(results_client=ResultsClient(), scan_type='seasonality',scan_name="usa_winter", tickers=['RB', "PL", "ZS", "ZC", "HG"] )