# Batch Screening with GPU Acceleration - Implementation Plan

## Current State Analysis

### Existing Concurrent Processing

`HistoricalScreener.run_screens()` already implements concurrent processing (lines 1560-1624):

```python
# Current implementation:
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_single_screen, params): params
               for params in screen_params}
```

**Status:** ✓ Already parallelizes across **screens** (different ScreenParams)
**Limitation:** ✗ Does NOT batch across **tickers** within each screen

### Computational Bottlenecks

The screening operations use scipy.stats functions that are CPU-bound:

1. **Correlations:** `stats.pearsonr()` - Called for each ticker individually
   - Lines 2467-2469: Open/close correlations
   - Lines 3114-3117: Lag autocorrelations
   - Lines 3233, 3384: Weekday correlations

2. **T-tests:** `stats.t.sf()` and `stats.t.cdf()` - P-value calculations
   - Lines 2600, 2606: Pattern significance testing
   - Line 3485: Weekday pattern p-values

3. **Data per ticker:** Each ticker processes independently
   - No batching of statistical operations across tickers
   - Repeated overhead for each ticker

## Opportunity for GPU Batching

### Current Flow (Inefficient)
```
For each screen:
    For each ticker:
        Load data for ticker
        Compute correlations (scipy, CPU)
        Compute p-values (scipy, CPU)
        Return results
```

### Proposed Flow (Efficient)
```
For each screen:
    Align all ticker data to common dates
    Stack all ticker arrays
    Batch compute correlations (CuPy, GPU)
    Batch compute p-values (CuPy, GPU)
    Split results back to per-ticker
```

## Implementation Strategy

### Phase 1: Data Alignment for Screening

Similar to ticker alignment for backtesting, but at the screening level:

```python
def align_screening_data(
    tickers: List[str],
    data_client: DataClient,
    date_col: str = 'date',
    method: str = 'intersection'
) -> Dict[str, pd.DataFrame]:
    """
    Align all ticker data to common date range for batch screening.

    Returns:
        Dict mapping ticker -> aligned DataFrame
    """
    # Load all ticker data
    ticker_data = {ticker: data_client[ticker] for ticker in tickers}

    # Find common date range
    if method == 'intersection':
        common_dates = set.intersection(*[
            set(data[date_col]) for data in ticker_data.values()
        ])
    else:  # union
        common_dates = set.union(*[
            set(data[date_col]) for data in ticker_data.values()
        ])

    # Align each ticker
    aligned = {}
    for ticker, data in ticker_data.items():
        aligned[ticker] = data[data[date_col].isin(common_dates)]

    return aligned
```

### Phase 2: GPU-Batched Statistical Operations

Create GPU-accelerated versions of scipy operations:

```python
# File: CTAFlow/screeners/gpu_stats.py

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

def batch_pearson_correlation(
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson correlations for multiple array pairs simultaneously.

    Args:
        x_arrays: List of x arrays (one per ticker)
        y_arrays: List of y arrays (one per ticker)
        use_gpu: Use GPU if available

    Returns:
        correlations: Array of correlation coefficients
        p_values: Array of p-values
    """
    xp = cp if (use_gpu and GPU_AVAILABLE) else np

    # Stack arrays (requires same length!)
    x_stack = xp.array(x_arrays)  # Shape: (n_tickers, n_samples)
    y_stack = xp.array(y_arrays)

    # Batch compute correlations
    n_samples = x_stack.shape[1]

    # Standardize
    x_mean = x_stack.mean(axis=1, keepdims=True)
    y_mean = y_stack.mean(axis=1, keepdims=True)
    x_std = x_stack.std(axis=1, keepdims=True)
    y_std = y_stack.std(axis=1, keepdims=True)

    x_norm = (x_stack - x_mean) / x_std
    y_norm = (y_stack - y_mean) / y_std

    # Correlation = mean of product of normalized values
    correlations = (x_norm * y_norm).mean(axis=1)

    # Compute p-values (t-distribution)
    t_stat = correlations * xp.sqrt((n_samples - 2) / (1 - correlations**2))
    # Use CPU for scipy.stats if needed, or implement CDF on GPU
    if xp is cp:
        t_stat_cpu = cp.asnumpy(t_stat)
        from scipy import stats
        p_values = 2 * stats.t.sf(np.abs(t_stat_cpu), n_samples - 2)
        p_values = xp.array(p_values)
    else:
        from scipy import stats
        p_values = 2 * stats.t.sf(np.abs(t_stat), n_samples - 2)

    # Transfer back to CPU
    if xp is cp:
        correlations = cp.asnumpy(correlations)
        p_values = cp.asnumpy(p_values)

    return correlations, p_values
```

### Phase 3: Modify HistoricalScreener for Batching

Add batch mode to `run_screens()`:

```python
class HistoricalScreener:
    def __init__(self, ..., use_gpu_batching: bool = True):
        self.use_gpu_batching = use_gpu_batching and GPU_AVAILABLE

    def run_screens(
        self,
        screen_params: List[ScreenParams],
        *,
        batch_tickers: bool = True,  # NEW: Enable ticker batching
        align_method: str = 'intersection',  # NEW: Alignment method
        ...
    ):
        """Run multiple screens with optional GPU batching across tickers."""

        if batch_tickers and self.use_gpu_batching:
            return self._run_screens_batched(
                screen_params,
                align_method=align_method,
                ...
            )
        else:
            # Original implementation (current code)
            return self._run_screens_sequential(screen_params, ...)

    def _run_screens_batched(self, screen_params, align_method, ...):
        """GPU-batched screening across all tickers."""

        # 1. Align all ticker data to common dates
        aligned_data = self._align_all_tickers(
            list(self.tickers),
            method=align_method
        )

        # 2. For each screen, batch process all tickers
        results = {}
        for params in screen_params:
            if params.screen_type == 'momentum':
                screen_results = self._screen_momentum_batched(
                    aligned_data,
                    params
                )
            elif params.screen_type == 'seasonality':
                screen_results = self._screen_seasonality_batched(
                    aligned_data,
                    params
                )

            results[params.name] = screen_results

        return results
```

### Phase 4: Batched Momentum Screening

```python
def _screen_momentum_batched(
    self,
    aligned_data: Dict[str, pd.DataFrame],
    params: ScreenParams
) -> Dict[str, Dict[str, Any]]:
    """
    Run momentum screen on all tickers simultaneously using GPU batching.
    """
    from .gpu_stats import batch_pearson_correlation

    # Extract returns for all tickers (aligned to same dates!)
    tickers = list(aligned_data.keys())
    open_returns = [aligned_data[t]['open'].values for t in tickers]
    close_returns = [aligned_data[t]['close'].values for t in tickers]
    st_mom = [aligned_data[t]['st_mom'].values for t in tickers]

    # Batch compute correlations on GPU
    open_close_corr, open_close_pval = batch_pearson_correlation(
        open_returns,
        close_returns,
        use_gpu=self.use_gpu_batching
    )

    open_st_corr, _ = batch_pearson_correlation(
        open_returns,
        st_mom,
        use_gpu=self.use_gpu_batching
    )

    close_st_corr, _ = batch_pearson_correlation(
        close_returns,
        st_mom,
        use_gpu=self.use_gpu_batching
    )

    # Build results dict
    results = {}
    for idx, ticker in enumerate(tickers):
        results[ticker] = {
            'open_close_corr': open_close_corr[idx],
            'open_close_pval': open_close_pval[idx],
            'open_st_corr': open_st_corr[idx],
            'close_st_corr': close_st_corr[idx],
            # ... other metrics
        }

    return results
```

## Benefits of Batched Screening

### Performance Gains

**Current (Sequential per ticker):**
```
Winter screen:
  CL: Load data → Compute stats (CPU) → 50ms
  GC: Load data → Compute stats (CPU) → 50ms
  ES: Load data → Compute stats (CPU) → 50ms
  ... (30 tickers)
Total: 30 × 50ms = 1,500ms

Spring screen: Another 1,500ms
Summer screen: Another 1,500ms
Fall screen: Another 1,500ms

Total: 6,000ms for 4 screens × 30 tickers
```

**With Batching:**
```
Winter screen:
  Align all tickers: 100ms
  Batch compute 30 tickers (GPU): 150ms
  Total: 250ms

All 4 screens: 4 × 250ms = 1,000ms
Speedup: 6x faster!
```

### Additional Benefits

1. **Reduced data loading overhead:** Load each ticker once, reuse across screens
2. **GPU utilization:** Simultaneous computation across all tickers
3. **Memory efficiency:** Batch operations use GPU memory more efficiently
4. **Scalability:** Performance improves with more tickers (better GPU utilization)

## Implementation Checklist

### Files to Create

- [ ] `CTAFlow/screeners/gpu_stats.py` - GPU-accelerated statistical functions
  - `batch_pearson_correlation()`
  - `batch_t_test()`
  - `batch_autocorr()`

- [ ] `CTAFlow/screeners/screening_alignment.py` - Data alignment for screening
  - `align_screening_data()`
  - `check_screening_alignment()`

### Files to Modify

- [ ] `CTAFlow/screeners/historical_screener.py`
  - Add `batch_tickers` parameter to `run_screens()`
  - Add `_run_screens_batched()` method
  - Add `_screen_momentum_batched()` method
  - Add `_screen_seasonality_batched()` method
  - Add `_align_all_tickers()` helper

### Testing

- [ ] Create `test_batched_screening.py`
  - Test alignment works correctly
  - Test GPU stats match scipy results
  - Test batched screening produces same results as sequential
  - Benchmark performance gains

### Documentation

- [ ] `BATCHED_SCREENING_GUIDE.md` - User guide
- [ ] Update `historical_screener.py` docstrings
- [ ] Add examples to README

## Compatibility Considerations

### Backward Compatibility

The implementation should be **fully backward compatible**:

```python
# Old code (still works):
results = screener.run_screens(screens, use_concurrent=True)

# New code (batched):
results = screener.run_screens(
    screens,
    use_concurrent=True,
    batch_tickers=True,  # NEW: Enable ticker batching
    align_method='intersection'  # NEW: Alignment method
)
```

Default `batch_tickers=False` initially, then change to `True` after testing.

### Data Alignment Trade-offs

Same as backtesting:
- **Intersection:** Lose some edge dates, guaranteed real data
- **Union:** Maximum coverage, may forward-fill gaps

## Next Steps

1. **Start with gpu_stats.py:** Implement GPU-batched correlation function
2. **Test on small dataset:** Verify GPU stats match scipy
3. **Implement alignment:** Reuse ticker_alignment.py logic
4. **Add batched methods:** Start with momentum screening
5. **Benchmark:** Compare performance vs. current implementation
6. **Iterate:** Extend to seasonality screening

## Expected Timeline

- **Phase 1 (gpu_stats.py):** 2-3 hours
- **Phase 2 (alignment):** 1 hour (reuse existing code)
- **Phase 3 (batched methods):** 3-4 hours
- **Phase 4 (testing):** 2-3 hours
- **Total:** ~1 day of work

## Questions for User

1. **Alignment preference:** intersection (safer) or union (more data)?
2. **GPU availability:** Do you have CuPy installed? GPU device count?
3. **Data coverage:** Do all your tickers have similar date ranges?
4. **Priority:** Which screen type is most important (momentum vs seasonality)?

---

**Status:** Ready to implement
**Dependencies:** CuPy (optional, graceful fallback to CPU)
**Breaking changes:** None (fully backward compatible)
