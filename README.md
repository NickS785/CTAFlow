# CTAFlow Screeners

This repository provides research and screening utilities for intraday futures data. The
orderflow tooling introduces deterministic session handling, automated volume bucket
selection, and seasonality statistics designed for systematic workflows.

## Orderflow scan

The `orderflow_scan` helper ingests raw ticks, resamples them into volume buckets,
summarises intraday pressure, and runs seasonality tests with Benjamini–Hochberg false
discovery rate control.

```python
import pandas as pd
from screeners.orderflow_scan import OrderflowParams, orderflow_scan

# ticks must contain columns ['ts', 'AskVolume', 'BidVolume']
# optional quote volume columns ['ask_vol', 'bid_vol'] are supported for quote-share metrics

def tick_source(symbol: str) -> pd.DataFrame:
    return load_my_ticks(symbol)  # implement loader returning a DataFrame per symbol

params = OrderflowParams(
    session_start="08:30",
    session_end="13:20",
    tz="America/Chicago",
    bucket_size="auto",      # use per-asset auto-V bucket sizing
    vpin_window=50,
    cadence_target=50,
)
results = orderflow_scan(tick_source, ["ZC_F", "ZS_F"], params)
```

Each symbol entry returns:

- `df_buckets`: tidy bucket-level orderflow metrics (volume, shares, VPIN, pressure).
- `df_intraday_pressure`: mean/median aggressor (ask) & quote pressure by bucket end-time.
- `df_weekly`: weekday seasonality with t-stats, two-sided p-values, and BH-FDR q-values.
- `df_wom_weekday`: week-of-month × weekday seasonality table with the same statistics.
- `df_weekly_peak_pressure`: blended weekly/clock-time summary of the largest seasonal
  pressure biases.
- `metadata`: session window, timezone, bucket size, number of sessions, and bucket counts.

### Automatic bucket sizing

When `bucket_size="auto"`, the scan computes median daily volume inside the session,
constructs a candidate grid from the supplied multipliers, and chooses the bucket size that
produces a bucket cadence closest to the requested `cadence_target`. Ties fall back to the
candidate with the lowest variance of buckets per day.

Override the behaviour by passing an integer `bucket_size` when a specific contract or
trading style requires fixed volume buckets.

### Command line interface

Run the scanner directly from the command line by pointing it to a directory of CSV or
Parquet tick files (file names are resolved via `{symbol}` substitution):

```bash
python -m screeners.orderflow_scan --symbols ZC_F ZS_F \
    --session 08:30 13:20 --tz America/Chicago --bucket auto \
    --ticks-path "data/{symbol}.csv"
```

> **Note**: Historical builds exposed an `--z` flag to configure event detection. That
> functionality has been removed and the flag is now ignored (a deprecation warning is
> emitted when supplied).

The CLI prints per-symbol bucket counts, the selected bucket size, and a preview of
weekday signals that survive a 5% FDR cutoff.

### Reading the seasonality tables

- `mean`: average pressure during the grouping window.
- `t_stat` and `p_value`: two-sided t-test against zero with degrees of freedom `n-1`.
- `q_value`: Benjamini–Hochberg adjusted p-value computed within each symbol.
- `sig_fdr_5pct`: indicator for `q_value <= 0.05`.
- `exploratory`: True when the number of sessions is below the configured `min_days`.

These tables make it easy to surface persistent weekday or week-of-month patterns while
keeping multiple-testing error under control.

## Session first-N-hours screener

Use `run_session_first_hours` to aggregate the opening hours of each futures session,
compute momentum, realised volatility, and a time-of-day relative volume signal, and then
rank symbols on each metric.

```python
from screeners.session_first_hours import SessionFirstHoursParams, run_session_first_hours

params = SessionFirstHoursParams(
    symbols=["CL", "NG", "ZC"],
    start_date="2025-06-01",
    end_date="2025-10-15",
    lookback_days=20,
    session_start_hhmm="17:00",
    first_hours=2,
    bar_seconds=60,
)

wide = run_session_first_hours(params)
print(wide.tail())
```

The wide DataFrame uses a MultiIndex with metric names on the first level and symbols on
the second. Each session includes:

- **momentum** – close/open return for the opening window.
- **realized_vol** – square-root of summed log-return variance inside the window.
- **vol_norm_ret** – momentum scaled by the rolling standard deviation of past
  window returns.
- **relative_volume_tod** – actual window volume divided by the median volume observed
  at the same minutes over the lookback window (values above 1 indicate heavier than
  usual participation).

Command line usage mirrors the Python helper:

```bash
python -m screens.run_session_first_hours --symbols CL NG ZC \
    --start 2025-06-01 --end 2025-10-15 --lookback 20 \
    --session-start 17:00 --first-hours 2 --bar-seconds 60 \
    --tz America/Chicago --tail 10 --out outputs/session_first_hours.csv
```

The CLI prints the latest rows and optionally saves the entire result set to CSV or
Parquet when `--out` is provided.

## Pattern extraction and strategy pipelines

Screeners surface dozens of candidate patterns, so the repository bundles tooling to
normalise their payloads and wire them into downstream research workflows:

- `CTAFlow.screeners.PatternExtractor` restructures raw screener output into tidy
  `PatternSummary` frames, exposes helpers such as `concat_many`, `filter_patterns`,
  and `significance_score`, and persists the canonical column layout used across the
  notebooks.
- `CTAFlow.strategy.screener_pipeline.ScreenerPipeline` consumes `PatternExtractor`
  summaries and materialises sparse boolean "gate" columns alongside the bar data
  (each gate carries directional metadata, pattern strength, and provenance fields).
- `CTAFlow.strategy.screener_pipeline.HorizonMapper` selects realised-return targets
  that correspond to each gate, ensuring same-day, next-day, next-week, and intraday
  horizons all line up on timezone-aware timestamps.

```python
from CTAFlow.screeners import PatternExtractor
from CTAFlow.strategy.screener_pipeline import HorizonMapper, ScreenerPipeline

# Combine seasonal and orderflow screeners that have already produced pattern payloads
seasonal_extractor = PatternExtractor(seasonal_screener, seasonal_results, [seasonal_params])
orderflow_extractor = PatternExtractor(orderflow_screener, orderflow_results, [orderflow_params])
combined = PatternExtractor.concat_many([seasonal_extractor, orderflow_extractor])

cl_patterns = combined.filter_patterns("CL")
bars = load_minute_bars("CL")  # user-supplied helper returning a DataFrame

pipeline = ScreenerPipeline(tz="America/Chicago")
features = pipeline.build_features(bars, cl_patterns)

mapper = HorizonMapper(tz="America/Chicago")
decisions = mapper.build_xy(features, cl_patterns, predictor_minutes=5)
```

### Weekend hedging semantics

Weekend hedging patterns now emit deterministic metadata so downstream research can rely on
true Friday→Monday spans:

- The Friday gate column only flags the configured weekday once per session. When
  `gate_time_hhmm` is omitted the pipeline anchors the gate to the last regular-session bar
  (session close).
- The `{prefix}_weekend_hedging_friday_monday_weekday` column is an int8 mask set to 1 on
  **every** Monday bar inside the active months and 0 elsewhere. These binary flags are stored
  alongside the pattern metadata so the backtester can locate the exact exit bars.
- `HorizonMapper` reads the recorded gate/weekday columns and, by default, exits on the last
  flagged Monday bar (`weekend_exit_policy="last"`). If a Monday session is missing because of a
  holiday, the mapper automatically rolls the trade to the next available session close so the
  Friday entry still produces a realised PnL.

This behaviour keeps the features consistent with the screening output (one Friday gate per
week, Monday mask covering the entire session) and guarantees that ScreenerBacktester spans
the full weekend horizon instead of collapsing to same-day moves.

`PatternExtractor.load_summaries_from_results` and
`PatternExtractor.load_summaries_from_results_async` remain available for loading persisted
summary tables, while `ScreenerPipeline.extract_ticker_patterns` provides a convenience
wrapper for fetching only the generated gate columns when you already have a price/volume
DataFrame in memory.
