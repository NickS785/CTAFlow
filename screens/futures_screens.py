from CTAFlow.screeners import( HistoricalScreener,
                               ScreenParams,
                               OrderflowScanner,
                               OrderflowParams,
                               PatternExtractor,
                               PatternSummary)

from CTAFlow.screeners.generic import (
    get_usa_seasonals,
    get_london_seasonals,
    get_momentums,
    usa_seasonality_screens,
    london_seasonality_screens,
    usa_momentum_screens,
    livestock_screens,
    grain_screens,
    usa_orderflow_screens,
    london_orderflow_screens,
    get_us_of_all,
    get_london_of_all
)

from CTAFlow.strategy import ScreenerPipeline
from CTAFlow.data import IntradayLeg, IntradayFileManager, SyntheticSymbol, DataClient, AsyncParquetWriter, read_synthetic_csv, read_exported_df
from CTAFlow.config import DLY_DATA_PATH, INTRADAY_DATA_PATH,  RESULTS_HDF_PATH
import pandas as pd
import numpy as np
from datetime import datetime
from sierrapy.parser import ScidReader
import asyncio as aio
from typing import List, Dict

cli = DataClient()
def grain_screen(start_date="2020-05-01", add_synthetics=False, ):
    result = {}
    tickers = ["ZC", "ZS"]
    base_data = gather_tickers(tickers, start_date, load_method="csv")

    if add_synthetics:
        base_data["ZLM"] = read_exported_df(INTRADAY_DATA_PATH / "synthetics/soy_crush_spread.csv")

    screener = HistoricalScreener(base_data)
    result["USA"] = screener.run_screens(grain_screens(seasonal=True))
    result["London"] = screener.run_screens(get_london_seasonals())
    result['momentum'] = screener.run_screens(grain_screens(seasonal=False))


    return result, screener

def energy_event():

    return

def livestock_screen(start_date="2020-05-01"):
    result = {}
    tickers = ["HE", "LE"]
    data = {t:read_exported_df(INTRADAY_DATA_PATH / f'CSV/{t}_5min.csv').loc[start_date:] for t in tickers}

    hs = HistoricalScreener(data)
    livestock_params = livestock_screens()
    result["seasonal"] = hs.run_screens(livestock_params['seasonal'])
    result["momentum"] = hs.run_screens(livestock_params['momentum'])


    return result, hs

def metals_screen(start_date="2019-01-01"):
    res = {}
    tickers = ["PL", "PA", "SI"]

    data = {t:read_exported_df(INTRADAY_DATA_PATH/f"CSV/{t}_5min.csv").loc[start_date:] for t in tickers}

    hs = HistoricalScreener(data)
    res["seasonal"] = hs.run_screens(get_usa_seasonals())
    res["seasonal_london"] = hs.run_screens(get_london_seasonals())
    res["momentum"] = hs.run_screens(get_momentums())

    return res, hs

def gather_tickers(tickers, start_date="2020-05-01", load_method="csv"):
    mgr = IntradayFileManager(DLY_DATA_PATH)
    base_data = {}

    if load_method == "sierra" or load_method == "sierrapy":
        reader = ScidReader(DLY_DATA_PATH)
        for t in tickers:
            if t.endswith("_F"):
                ticker = t[:2]
            else:
                ticker = t

            base_data[t] = reader.load_front_month_series(ticker, start=start_date)
    elif load_method == "dataclient" or load_method=="dc":
        reader = cli
        base_data = reader.query_market_data(tickers, start_date=start_date)
    elif load_method == 'parquet':
        for t in tickers:
            base_data[t] = pd.read_parquet(INTRADAY_DATA_PATH / f"{t}/{t}_5min.parquet")

    elif load_method == 'csv':

        for t in tickers:

            data = read_exported_df(INTRADAY_DATA_PATH / f"CSV/{t}_5min.csv")
            base_data[t] = data
    else:
        reader = mgr
        for t in tickers:
            if not t.endswith("_F"):
                ticker = f'{t}_F'
            else:
                ticker = t

            base_data[t] = reader.load_front_month_series(ticker, start=pd.to_datetime(start_date), detect_gaps=False)


    return base_data

def energy_screen(start_date="2020-05-01", add_synthetics=True):
    result = {}
    tickers = ["RB", "CL", "NG", "HO"]
    base_data = gather_tickers(tickers, start_date, load_method="csv")



    if add_synthetics:
        crack = read_synthetic_csv(INTRADAY_DATA_PATH / "synthetics/crack_spread.csv")
        base_data["CS"] = crack.loc[start_date:]

    screener = HistoricalScreener(base_data)
    result['momentum'] = screener.run_screens(get_momentums())

    result["USA"] = screener.run_screens(get_usa_seasonals())
    result["London"] = screener.run_screens(get_london_seasonals())

    return result, screener



def run_screeners(tickers,id_data_path=INTRADAY_DATA_PATH, hdf_path="F:\\Data\\intraday\\screener_results.hdf"):
    res = {}
    mgr = AsyncParquetWriter(id_data_path)
    data = mgr.batch_read_to_dict_sync(tickers, start=datetime(2020,4,1))
    data_5m = {k:cli._resample_ohlc_data(v, '5min') for k,v in data.items()}

    # Get orderflow screens
    usa_of_screens = usa_orderflow_screens(seasonal=True)
    london_of_screens = london_orderflow_screens(seasonal=True)
    sessions = {'USA': usa_of_screens,
                'London': london_of_screens
                }

    OF_radar = OrderflowScanner(hdf_path=hdf_path)
    HS_screener = HistoricalScreener(data_5m)
    res['OF'] = {k:OF_radar.run_scans(sessions[k], data) for k in sessions.keys()}
    res['Hist']  = HS_screener.run_screens(get_usa_seasonals())
    return res, OF_radar, HS_screener


if __name__ == "__main__":
    BACKTEST_PATTERN_TYPES = ["time_predictive_eod", "time_predictive_intraday", "momentum_sc", "momentum_oc", "time_predictive_nextweek", "time_predictive_nextday", "weekend_hedging", "weekday_mean", "calendar"]
    tickers = ["PL", "PA", "SI"]
    start_date = "2019-01-01"

    data = {t: read_exported_df(INTRADAY_DATA_PATH / f"CSV/{t}_5min.csv").loc[start_date:] for t in tickers}

    hs = HistoricalScreener(data)
    res = {}
    res["seasonal"] = hs.run_screens(get_usa_seasonals())
    res["seasonal_london"] = hs.run_screens(get_london_seasonals())
    res["momentum"] = hs.run_screens(get_momentums())
    print("\n" + "="*70)
    print("DEBUGGING CALENDAR PATTERN STRUCTURE")
    print("="*70)

    # Look at one ticker's calendar patterns
    for ticker in ['PA', 'PL', 'SI']:
        if ticker in res["seasonal"]:
            ticker_result = res["seasonal"][ticker]
            print(f"\n{ticker} result keys: {list(ticker_result.keys())}")

            # Check strongest_patterns
            if 'strongest_patterns' in ticker_result:
                patterns = ticker_result['strongest_patterns']
                calendar_pats = [p for p in patterns if p.get('pattern_type') == 'calendar']
                print(f"  {ticker} has {len(calendar_pats)} calendar patterns")

                if calendar_pats:
                    print(f"\n  First calendar pattern for {ticker}:")
                    first_pat = calendar_pats[0]
                    for key in ['pattern_type', 'calendar_pattern', 'event', 'horizon']:
                        print(f"    {key}: {first_pat.get(key, 'MISSING')}")

    print("\n" + "="*70)
    print("TESTING PATTERN EXTRACTION")
    print("="*70)

    pe_usa = PatternExtractor(hs, res["seasonal"])
    pe_london = PatternExtractor(hs, res["seasonal_london"])
    pe_mom = PatternExtractor(hs, res['momentum'])

    # Check how PatternExtractor organizes patterns
    print(f"\npe_usa.patterns keys: {list(pe_usa.patterns.keys())[:10]}")

    # Look at a specific ticker's patterns in PatternExtractor
    if 'PA' in pe_usa.patterns:
        pa_patterns = pe_usa.patterns['PA']
        print(f"\nPA patterns in pe_usa:")
        print(f"  Type: {type(pa_patterns)}")
        if isinstance(pa_patterns, dict):
            print(f"  Keys: {list(pa_patterns.keys())}")
            # Check calendar patterns



            for screen_name in ['usa_winter', 'usa_spring', 'usa_all']:
                if screen_name in pa_patterns:
                    screen_pats = pa_patterns[screen_name]
                    print(f"\n  {screen_name}:")
                    print(f"    Type: {type(screen_pats)}")
                    if isinstance(screen_pats, list):
                        cal_pats = [p for p in screen_pats if p.get('pattern_type') == 'calendar']
                        print(f"    Calendar patterns: {len(cal_pats)}")
                        if cal_pats:
                            print(f"    First calendar pattern:")
                            for key in ['pattern_type', 'calendar_pattern', 'event', 'horizon']:
                                print(f"      {key}: {cal_pats[0].get(key, 'MISSING')}")

    print("\n" + "="*70)
    print("TESTING BACKTESTING")
    print("="*70)

    sp = ScreenerPipeline()
    ranked_usa = pe_usa.rank_patterns()
    pe_final = pe_mom.concat(pe_usa)
    pe_final.concat(pe_london, inplace=True)
    ranked = pe_final.rank_patterns()
    ticker = "PA"
    ticker_data = hs.data[ticker]
    patterns = pe_final.filter_patterns(symbol=ticker, pattern_types=BACKTEST_PATTERN_TYPES, screen_names=["usa_winter", "momentum_generic", "london_all", "usa_all"])

    print(f"\nFiltered patterns type: {type(patterns)}")
    if isinstance(patterns, dict):
        print(f"Filtered patterns keys: {list(patterns.keys())}")

        for key in list(patterns.keys())[:5]:
            pat = patterns[key]
            if isinstance(pat, dict):
                print(f"\n  Pattern '{key}':")
                for field in ['pattern_type', 'calendar_pattern', 'event', 'horizon']:
                    print(f"    {field}: {pat.get(field, 'MISSING')}")

    print("\n" + "="*70)
    print("RUNNING BATCH BACKTEST")
    print("="*70)

    # Use batch_multi_ticker_backtest instead
    ticker_backtest_data = {ticker: (ticker_data, patterns)}
    results = sp.batch_multi_ticker_backtest(ticker_backtest_data, threshold=0.0)

    print(f"\nBacktest results: {len(results)} patterns")
    print("\nFirst 10 result keys:")
    for key in list(results.keys())[:10]:
        print(f"  {key}")






