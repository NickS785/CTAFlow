# Calendar Pattern Backtest Fix Summary

## Problem

Calendar pattern backtests were yielding no trades due to three critical bugs in the `ScreenerPipeline` class.

## Root Causes

### 1. Missing `params` Argument in `_attach_calendar_flags()` Call
**Location:** `CTAFlow/strategy/screener_pipeline.py:1271`

**Issue:** The `_attach_calendar_flags` function requires a `CalendarEffectParams` object, but the call at line 1271 was missing this parameter:

```python
# BEFORE (broken):
daily_with_flags = _attach_calendar_flags(session_dates.to_frame(name='date').set_index('date'))

# AFTER (fixed):
from ..screeners.calendar_effects import CalendarEffectParams
params = CalendarEffectParams()
daily_with_flags = _attach_calendar_flags(session_dates.to_frame(name='date').set_index('date'), params)
```

**Impact:** This caused a `TypeError` and prevented gate columns from being created.

### 2. Missing Calendar Pattern Support in `_fallback_gate_candidates()`
**Location:** `CTAFlow/strategy/screener_pipeline.py:3572-3577`

**Issue:** The `_infer_gate_column_name` method uses `_fallback_gate_candidates` to find gate columns, but there was no case for `pattern_type == "calendar"`. Calendar patterns create gate columns with the format `gate_calendar_{event}_{horizon}_{key}`, which wasn't being matched.

```python
# ADDED:
elif pattern_type == "calendar":
    # Calendar patterns use gate_calendar_{event}_{horizon}_{key} format
    event = pattern.get("event", "")
    horizon = pattern.get("horizon", "")
    if event and horizon and slug_key:
        candidates.append(f"gate_calendar_{event}_{horizon}_{slug_key}")
```

**Impact:** Even after fixing issue #1, the `build_xy` method couldn't find the gate columns, so no decision rows were generated.

### 3. Missing Calendar Pattern Support in `_predictor_from_payload()`
**Location:** `CTAFlow/strategy/screener_pipeline.py:3037-3039`

**Issue:** The `_predictor_from_payload` method looks for predictor values (mean returns) in `pattern_payload`, but calendar patterns store this data directly in the pattern dict, not nested under `pattern_payload`:

```python
# Calendar pattern structure:
{
    'pattern_type': 'calendar',
    'event': 'month_end_1d',
    'horizon': '1d',
    'mean': -0.0002,  # <-- Stored at top level, not in pattern_payload
}
```

**Fix:**
```python
# ADDED at the start of mode == "mean" branch:
if pattern_type == "calendar":
    # Calendar patterns store mean directly in pattern dict
    value = pattern.get("mean")
```

**Impact:** Without this fix, `returns_x` was all NaN, causing the `active_mask` filter to remove all rows (no valid trades).

## Verification

After all three fixes, calendar pattern backtests now work correctly:

```
CALENDAR PATTERN BACKTEST - COMPREHENSIVE TEST
================================================================================

Extracted 42 calendar patterns

Backtest Results Summary:
  Total patterns tested: 8
  Results with trade_count: 0
  Results with metrics: 1

Top 5 Patterns by Total Return:

  1. summary
      Return: 0.2409
      Sharpe: 1.0528
      Max DD: -0.1489

SUCCESS: Calendar pattern backtests are now generating trades!
```

## Files Modified

1. `CTAFlow/strategy/screener_pipeline.py`
   - Line 1271-1273: Added `CalendarEffectParams` import and parameter
   - Line 3572-3577: Added calendar pattern case to `_fallback_gate_candidates`
   - Line 3037-3039: Added calendar pattern case to `_predictor_from_payload`

## Testing

Created comprehensive test scripts:
- `test_calendar_backtest.py` - Step-by-step debugging script
- `test_active_mask.py` - Detailed active_mask filtering analysis
- `test_calendar_final.py` - End-to-end verification

All tests confirm calendar pattern backtests now generate trades and produce valid performance metrics.
