# CTAFlow Screeners

This repository provides research and screening utilities for intraday futures data. The
orderflow tooling introduces deterministic session handling, automated volume bucket
selection, and seasonality statistics designed for systematic workflows.

## Orderflow scan

The `orderflow_scan` helper ingests raw ticks, resamples them into volume buckets, flags
pressure events, and runs seasonality tests with Benjamini–Hochberg false discovery rate
control.

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
    threshold_z=2.0,
    cadence_target=50,
)
results = orderflow_scan(tick_source, ["ZC_F", "ZS_F"], params)
```

Each symbol entry returns:

- `df_buckets`: tidy bucket-level orderflow metrics (volume, shares, VPIN, pressure).
- `df_intraday_pressure`: mean/median aggressor (ask) & quote pressure by bucket end-time.
- `df_events`: merged runs where robust z-scores breach the configured threshold.
- `df_weekly`: weekday seasonality with t-stats, two-sided p-values, and BH-FDR q-values.
- `df_wom_weekday`: week-of-month × weekday seasonality table with the same statistics.
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
    --session 08:30 13:20 --tz America/Chicago --bucket auto --z 2.0 \
    --ticks-path "data/{symbol}.csv"
```

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
