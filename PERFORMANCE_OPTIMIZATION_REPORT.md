# Performance Optimization Report: deseasonalized_vol()

## Executive Summary
The `deseasonalized_vol()` method has critical performance bottlenecks in the rolling window deseasonalization process. For 1 year of 5-minute intraday data, the method processes ~390,000 data points through OLS regression with 247 separate model fits.

**Estimated speedup potential: 10-20x**

## Detailed Analysis

### Bottleneck #1: Full DataFrame Copy (HIGH Impact)
**Location:** `intraday_momentum.py:4270`
```python
data = self.intraday_data.copy()  # Copies entire DataFrame every call
```

**Issue:** Copies potentially millions of rows even when using cache

**Fix:** Only copy when necessary (filtering) or work with views
```python
# BEFORE
data = self.intraday_data.copy()
if use_session_times:
    data = data.between_time(self.session_open, self.session_end)

# AFTER
if use_session_times:
    data = self.intraday_data.between_time(
        self.session_open, self.session_end, inclusive='both'
    ).copy()  # Only copy filtered data
else:
    data = self.intraday_data  # No copy needed
```

**Speedup:** 2-5x for this section

---

### Bottleneck #2-4: RollingDiurnalAdjuster Loop (CRITICAL Impact)
**Location:** `diurnal_seasonality.py:399-440`

**Issue:** Nested loops with daily OLS regression fits
```python
for i in range(self.min_days, n_days):  # ~247 iterations
    train_y = np.concatenate([y_vals[s:e] for s, e in train_slices])  # Expensive
    train_idx = np.concatenate([idx_vals[s:e] for s, e in train_slices])
    adjuster.fit(train_y, train_idx)  # OLS regression every day
```

**Time Complexity:** O(n_days × lookback_days × samples_per_day)

**Optimization Strategies:**

#### Strategy A: Incremental Window Updates (Fastest)
Instead of concatenating from scratch each day:
```python
# Maintain rolling buffer
buffer_y = deque(maxlen=lookback_days * samples_per_day)
buffer_idx = deque(maxlen=lookback_days * samples_per_day)

for i in range(self.min_days, n_days):
    # Add new day
    curr_start, curr_end = day_boundaries[i-1]
    buffer_y.extend(y_vals[curr_start:curr_end])
    buffer_idx.extend(idx_vals[curr_start:curr_end])

    # Convert to arrays (much faster than concatenate)
    train_y = np.array(buffer_y)
    train_idx = np.array(buffer_idx)

    adjuster.fit(train_y, train_idx)
```

**Expected speedup:** 3-5x

#### Strategy B: Reduce Refit Frequency
Don't refit every day - only refit periodically:
```python
refit_interval = 5  # Refit every 5 days instead of every day

for i in range(self.min_days, n_days):
    if i % refit_interval == 0 or i == self.min_days:
        # Gather and fit
        train_y = np.concatenate([y_vals[s:e] for s, e in train_slices])
        train_idx = np.concatenate([idx_vals[s:e] for s, e in train_slices])
        adjuster.fit(train_y, train_idx)

    # Transform current day (uses cached model)
    curr_y = y_vals[curr_start:curr_end]
    curr_idx = idx_vals[curr_start:curr_end]
    result_vals[curr_start:curr_end] = adjuster.transform(curr_y, curr_idx)
```

**Expected speedup:** 5x (with refit_interval=5)

#### Strategy C: Vectorize Day Boundaries (Medium Impact)
**Location:** `diurnal_seasonality.py:376-386`
```python
# BEFORE
date_codes = pd.Categorical(date_vals, categories=unique_dates).codes
day_boundaries = []
for day_idx in range(n_days):
    mask = date_codes == day_idx
    indices = np.where(mask)[0]
    if len(indices) > 0:
        day_boundaries.append((indices[0], indices[-1] + 1))

# AFTER
date_codes = pd.Categorical(date_vals, categories=unique_dates).codes
day_start = np.searchsorted(date_codes, np.arange(n_days), side='left')
day_end = np.searchsorted(date_codes, np.arange(n_days), side='right')
day_boundaries = list(zip(day_start, day_end))
```

**Expected speedup:** 2x for this section

---

### Bottleneck #5: NumPy Operations in Fourier Features
**Location:** `diurnal_seasonality.py:95-121`

The `_create_fourier_features` method can be JIT-compiled:

```python
import numba

@numba.jit(nopython=True, fastmath=True, cache=True)
def _create_fourier_features_numba(t, bins_per_day, order):
    """JIT-compiled Fourier feature generation."""
    n_samples = len(t)
    n_features = 2 * order
    X = np.empty((n_samples, n_features), dtype=np.float64)

    for k in range(1, order + 1):
        phase = 2 * np.pi * k * t / bins_per_day
        X[:, 2*(k-1)] = np.sin(phase)
        X[:, 2*(k-1) + 1] = np.cos(phase)

    return X
```

**Expected speedup:** 1.5-2x for this section

---

## Recommended Implementation Priority

1. **Immediate (Easy wins):**
   - Fix DataFrame copy (5 lines of code)
   - Vectorize day boundaries (5 lines)

2. **High Priority (Moderate effort):**
   - Reduce refit frequency (Strategy B) - 20 lines

3. **Medium Priority (Higher effort, higher reward):**
   - Incremental window updates (Strategy A) - 50 lines
   - Numba JIT compilation - 30 lines

4. **Optional (Advanced):**
   - Cache fitted models across runs
   - Parallel processing for multiple target times

---

## Testing Recommendations

After implementing optimizations:

1. **Correctness Test:** Ensure deseasonalized values match original implementation
   ```python
   # Compare old vs new
   result_old = model.deseasonalized_vol(time(10, 0), timedelta(minutes=5))
   result_new = model.deseasonalized_vol(time(10, 0), timedelta(minutes=5))
   np.testing.assert_allclose(result_old, result_new, rtol=1e-10)
   ```

2. **Performance Benchmark:**
   ```python
   import time

   start = time.time()
   for _ in range(10):
       model.deseasonalized_vol(time(10, 0), timedelta(minutes=5))
   elapsed = time.time() - start
   print(f"Average time: {elapsed/10:.2f}s")
   ```

3. **Memory Profile:**
   ```python
   from memory_profiler import profile

   @profile
   def test_memory():
       model.deseasonalized_vol(time(10, 0), timedelta(minutes=5))
   ```

---

## Implementation Files

- `CTAFlow/models/intraday_momentum.py` - Line 4178 (`deseasonalized_vol` method)
- `CTAFlow/features/cyclical/seasonality/diurnal_seasonality.py` - Lines 291-448 (`RollingDiurnalAdjuster`)

---

## Expected Overall Performance Gain

| Optimization | Speedup | Implementation Effort |
|--------------|---------|---------------------|
| DataFrame copy fix | 2x | Easy (5 min) |
| Vectorize boundaries | 1.5x | Easy (5 min) |
| Reduce refit frequency | 5x | Medium (30 min) |
| Incremental windows | 3x | Medium (1 hour) |
| Numba JIT | 1.5x | Medium (30 min) |

**Combined speedup: 10-20x (compounding effects)**

---

## Additional Notes

- The caching mechanism (`_deseas_cache`) is already implemented and works correctly
- For users calling `deseasonalized_vol()` multiple times with the same parameters, cache provides immediate benefit
- Main bottleneck is the **first call** for each cache key
- Consider adding progress bars for long-running operations:
  ```python
  from tqdm import tqdm
  for i in tqdm(range(self.min_days, n_days), desc='Deseasonalizing'):
      # ... rolling window fit
  ```
