"""Utilities for aligning ticker data to enable efficient GPU batching."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple
import pandas as pd


def align_ticker_data(
    ticker_data: Mapping[str, Tuple[pd.DataFrame, Any]],
    *,
    method: str = 'intersection',
    date_col: str = 'session_date',
    fill_method: str = 'ffill',
    verbose: bool = False,
) -> Dict[str, Tuple[pd.DataFrame, Any]]:
    """Align all tickers to the same date range for optimal GPU batching efficiency.

    When running batch_multi_ticker_backtest(), tickers with different lengths are
    processed in separate GPU batches. Aligning them to a common date range enables
    processing all tickers in a single batch, resulting in 2-3x speedup.

    Args:
        ticker_data: Dictionary mapping ticker symbols to (bars, patterns) tuples
        method: Alignment method
            - 'intersection': Use only dates common to ALL tickers (recommended)
            - 'union': Use all dates from ANY ticker (requires filling gaps)
        date_col: Column name containing dates (default: 'session_date')
        fill_method: How to fill missing data in union mode
            - 'ffill': Forward fill (use last known value)
            - 'bfill': Backward fill (use next known value)
        verbose: Print alignment statistics

    Returns:
        Aligned ticker_data where all tickers have identical date ranges and lengths

    Examples:
        >>> # Before: Different date ranges
        >>> ticker_data = {
        ...     'CL': (cl_bars, cl_patterns),  # 1,305 rows
        ...     'GC': (gc_bars, gc_patterns),  # 936 rows
        ...     'ES': (es_bars, es_patterns),  # 913 rows
        ... }
        >>>
        >>> # After: All aligned to common dates
        >>> aligned = align_ticker_data(ticker_data, method='intersection')
        >>> # All now have 544 rows (common date range)
        >>>
        >>> # Run batch backtest (single GPU batch!)
        >>> results = pipeline.batch_multi_ticker_backtest(aligned, ...)

    Notes:
        - 'intersection' method: Safest, only uses real data, may lose some rows
        - 'union' method: Maximum coverage, but forward-fills gaps (potential bias)
        - For maximum GPU efficiency, all tickers should have the same length
        - Single GPU batch = 2-3x faster than multiple batches
    """

    if not ticker_data:
        return {}

    if method == 'intersection':
        return _align_intersection(ticker_data, date_col, verbose)
    elif method == 'union':
        return _align_union(ticker_data, date_col, fill_method, verbose)
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'intersection' or 'union'.")


def _align_intersection(
    ticker_data: Mapping[str, Tuple[pd.DataFrame, Any]],
    date_col: str,
    verbose: bool,
) -> Dict[str, Tuple[pd.DataFrame, Any]]:
    """Align to dates common to all tickers (intersection)."""

    # Find common dates across all tickers
    common_dates = None
    for ticker, (bars, _) in ticker_data.items():
        if bars.empty:
            continue
        dates = set(pd.to_datetime(bars[date_col]))
        common_dates = dates if common_dates is None else common_dates.intersection(dates)

    if common_dates is None or not common_dates:
        if verbose:
            print("Warning: No common dates found across tickers")
        return {}

    common_dates_sorted = sorted(common_dates)
    start_date = common_dates_sorted[0]
    end_date = common_dates_sorted[-1]

    if verbose:
        print(f"Aligning to intersection:")
        print(f"  Common date range: {start_date.date()} to {end_date.date()}")
        print(f"  Total common dates: {len(common_dates)}")
        print(f"  Per-ticker changes:")

    # Filter each ticker to common dates
    aligned = {}
    for ticker, (bars, patterns) in ticker_data.items():
        if bars.empty:
            continue

        original_len = len(bars)
        mask = pd.to_datetime(bars[date_col]).isin(common_dates)
        aligned_bars = bars[mask].copy()

        if verbose:
            print(f"    {ticker}: {original_len} -> {len(aligned_bars)} rows")

        aligned[ticker] = (aligned_bars, patterns)

    return aligned


def _align_union(
    ticker_data: Mapping[str, Tuple[pd.DataFrame, Any]],
    date_col: str,
    fill_method: str,
    verbose: bool,
) -> Dict[str, Tuple[pd.DataFrame, Any]]:
    """Align to all dates from any ticker (union)."""

    # Get all unique dates
    all_dates = set()
    for ticker, (bars, _) in ticker_data.items():
        if bars.empty:
            continue
        all_dates.update(pd.to_datetime(bars[date_col]))

    if not all_dates:
        if verbose:
            print("Warning: No dates found in any ticker")
        return {}

    start_date = min(all_dates)
    end_date = max(all_dates)
    full_index = pd.DatetimeIndex(sorted(all_dates))

    if verbose:
        print(f"Aligning to union (fill_method={fill_method}):")
        print(f"  Full date range: {start_date.date()} to {end_date.date()}")
        print(f"  Total dates: {len(all_dates)}")
        print(f"  Per-ticker changes:")

    # Reindex and fill each ticker
    aligned = {}
    for ticker, (bars, patterns) in ticker_data.items():
        if bars.empty:
            continue

        original_len = len(bars)

        # Set index to dates for reindexing
        bars_indexed = bars.set_index(pd.to_datetime(bars[date_col]))

        # Reindex to full date range
        aligned_bars = bars_indexed.reindex(full_index)

        # Fill missing values
        if fill_method == 'ffill':
            aligned_bars = aligned_bars.ffill()
        elif fill_method == 'bfill':
            aligned_bars = aligned_bars.bfill()
        else:
            raise ValueError(f"Invalid fill_method '{fill_method}'. Use 'ffill' or 'bfill'.")

        # Reset index
        aligned_bars = aligned_bars.reset_index(drop=False)
        aligned_bars.rename(columns={'index': date_col}, inplace=True)

        if verbose:
            filled_count = len(aligned_bars) - original_len
            print(f"    {ticker}: {original_len} -> {len(aligned_bars)} rows ({filled_count} filled)")

        aligned[ticker] = (aligned_bars, patterns)

    return aligned


def check_alignment(
    ticker_data: Mapping[str, Tuple[pd.DataFrame, Any]],
    date_col: str = 'session_date',
) -> Dict[str, Any]:
    """Check if ticker data is already aligned and report batching efficiency.

    Returns:
        Dict with alignment statistics:
        - 'is_aligned': bool, True if all tickers have same length
        - 'lengths': Dict[ticker, length]
        - 'num_batches': Number of GPU batches that will be created
        - 'batch_groups': Dict[length, list of tickers]
    """

    lengths = {}
    for ticker, (bars, _) in ticker_data.items():
        lengths[ticker] = len(bars)

    unique_lengths = set(lengths.values())
    is_aligned = len(unique_lengths) == 1

    # Group tickers by length (simulates batch_patterns grouping)
    batch_groups = {}
    for ticker, length in lengths.items():
        if length not in batch_groups:
            batch_groups[length] = []
        batch_groups[length].append(ticker)

    return {
        'is_aligned': is_aligned,
        'lengths': lengths,
        'num_batches': len(batch_groups),
        'batch_groups': batch_groups,
    }


def print_alignment_report(
    ticker_data: Mapping[str, Tuple[pd.DataFrame, Any]],
    date_col: str = 'session_date',
) -> None:
    """Print a formatted report of ticker alignment status."""

    stats = check_alignment(ticker_data, date_col)

    print("=" * 70)
    print("TICKER ALIGNMENT REPORT")
    print("=" * 70)

    print(f"\nAlignment Status: {'ALIGNED' if stats['is_aligned'] else 'NOT ALIGNED'}")
    print(f"Number of GPU batches: {stats['num_batches']}")

    if stats['is_aligned']:
        common_length = list(stats['lengths'].values())[0]
        print(f"Common length: {common_length} rows")
        print(f"Tickers: {list(stats['lengths'].keys())}")
        print("\n✓ All tickers will be processed in a SINGLE GPU batch (optimal)")
    else:
        print(f"\nBatch breakdown:")
        for length, tickers in sorted(stats['batch_groups'].items(), reverse=True):
            print(f"  Batch size {length}: {tickers}")
        print("\n! Multiple batches = reduced GPU efficiency")
        print("  Recommendation: Use align_ticker_data() to align before backtesting")

    print("\nPer-ticker lengths:")
    for ticker, length in sorted(stats['lengths'].items()):
        print(f"  {ticker}: {length} rows")

    print("=" * 70)
