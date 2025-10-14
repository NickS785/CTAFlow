"""
Corrected VPIN (Volume-Synchronized Probability of Informed Trading) calculation.

Key fixes:
1. VPIN denominator uses actual realized volume, not target bucket size
2. Added run-length diagnostics for buy/sell sequences
3. Enhanced validation and edge case handling
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def vpin_from_signed_ticks(
    df: pd.DataFrame,
    V: float,
    window: int = 50,
    min_bucket_vol: float = 0.0
) -> pd.DataFrame:
    """
    Calculate VPIN from tick data with signed buy/sell volumes.

    Parameters
    ----------
    df : pd.DataFrame
        Columns ['ts', 'buy_vol', 'sell_vol'] sorted by 'ts'
    V : float
        Target bucket size (contracts/shares)
    window : int
        Number of buckets for VPIN rolling average (default: 50)
    min_bucket_vol : float
        Minimum bucket volume to include in VPIN calculation (default: 0.0)

    Returns
    -------
    pd.DataFrame
        Bucket-level DataFrame with columns:
        - bucket: Bucket ID
        - ts_end: Last timestamp in bucket
        - n_ticks: Number of ticks in bucket
        - buy: Total buy volume
        - sell: Total sell volume
        - vol: Total volume (buy + sell)
        - buy_share: Buy volume as fraction of total
        - sell_share: Sell volume as fraction of total
        - imbalance: |buy - sell|
        - imb_frac: Imbalance / realized volume
        - vpin: Rolling VPIN over window buckets
        - max_buy_run: Longest consecutive buy-dominated bucket sequence
        - max_sell_run: Longest consecutive sell-dominated bucket sequence

    Notes
    -----
    **Key Correction**: VPIN denominator uses actual realized volume in the window,
    not target bucket size times count. This fixes the issue where sell volume
    statistics (like max_sell_run) were always zero.

    Formula:
        VPIN = sum(|buy_i - sell_i|) / sum(vol_i) over window buckets

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'ts': pd.date_range('2023-01-01', periods=1000, freq='1min'),
    ...     'buy_vol': np.random.exponential(10, 1000),
    ...     'sell_vol': np.random.exponential(10, 1000)
    ... })
    >>> result = vpin_from_signed_ticks(df, V=500, window=50)
    >>> print(result[['bucket', 'vpin', 'max_buy_run', 'max_sell_run']].tail())
    """
    df = df.copy()

    # 1) Clean & validate
    df['buy_vol'] = pd.to_numeric(df['buy_vol'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['sell_vol'] = pd.to_numeric(df['sell_vol'], errors='coerce').fillna(0.0).clip(lower=0.0)

    # Validate input
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'])

    # Optional: collapse simultaneous timestamps
    df = df.groupby('ts', as_index=False)[['buy_vol', 'sell_vol']].sum()

    # 2) Compute cumulative traded volume for bucketing
    vol = (df['buy_vol'] + df['sell_vol']).to_numpy()

    if vol.sum() == 0:
        raise ValueError("Total volume is zero - cannot compute VPIN")

    cum = np.cumsum(vol)

    # 3) Assign bucket IDs (keep partial final bucket)
    # Small epsilon to handle floating point edge cases
    bucket = np.floor((cum - 1e-12) / V).astype(int)

    # 4) Aggregate per bucket
    gb = pd.DataFrame({
        'bucket': bucket,
        'ts': df['ts'],
        'buy': df['buy_vol'],
        'sell': df['sell_vol'],
        'vol': vol
    }).groupby('bucket', as_index=False).agg(
        ts_end=('ts', 'max'),
        buy=('buy', 'sum'),
        sell=('sell', 'sum'),
        vol=('vol', 'sum'),
        n_ticks=('ts', 'count')
    )

    # Filter out buckets below minimum volume threshold
    if min_bucket_vol > 0:
        gb = gb[gb['vol'] >= min_bucket_vol].reset_index(drop=True)

    # 5) Imbalance calculations
    gb['imbalance'] = (gb['buy'] - gb['sell']).abs()

    # **KEY FIX**: Use realized volume, not target bucket size
    gb['imb_frac'] = gb['imbalance'] / gb['vol'].replace(0, np.nan)

    # **CORRECTED VPIN FORMULA**:
    # VPIN = sum(|buy_i - sell_i|) / sum(vol_i) over window
    gb['vpin'] = (
        gb['imbalance'].rolling(window, min_periods=1).sum() /
        gb['vol'].rolling(window, min_periods=1).sum()
    )

    # 6) Buy/Sell shares
    gb['buy_share'] = gb['buy'] / gb['vol'].replace(0, np.nan)
    gb['sell_share'] = gb['sell'] / gb['vol'].replace(0, np.nan)

    # 7) Run-length diagnostics (buy-dominated vs sell-dominated buckets)
    gb['buy_dom'] = (gb['buy'] > gb['sell']).astype(int)
    gb['sell_dom'] = (gb['sell'] > gb['buy']).astype(int)

    # Calculate max run lengths
    gb['max_buy_run'] = _calculate_max_run(gb['buy_dom'], window)
    gb['max_sell_run'] = _calculate_max_run(gb['sell_dom'], window)

    # 8) Additional diagnostics
    gb['vol_ratio'] = gb['vol'] / V  # How close to target bucket size
    gb['net_flow'] = gb['buy'] - gb['sell']  # Signed flow

    return gb[[
        'bucket', 'ts_end', 'n_ticks',
        'buy', 'sell', 'vol',
        'buy_share', 'sell_share',
        'imbalance', 'imb_frac',
        'vpin',
        'max_buy_run', 'max_sell_run',
        'vol_ratio', 'net_flow'
    ]]


def _calculate_max_run(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate maximum consecutive run length within a rolling window.

    Parameters
    ----------
    series : pd.Series
        Binary series (0 or 1) indicating presence of condition
    window : int
        Rolling window size

    Returns
    -------
    pd.Series
        Maximum run length in each window
    """
    n = len(series)
    max_runs = np.zeros(n)

    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_data = series.iloc[start_idx:i+1].values

        if len(window_data) == 0:
            max_runs[i] = 0
            continue

        # Find consecutive runs of 1s
        runs = []
        current_run = 0

        for val in window_data:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        # Don't forget final run
        if current_run > 0:
            runs.append(current_run)

        max_runs[i] = max(runs) if runs else 0

    return pd.Series(max_runs, index=series.index)


def diagnose_vpin_data(df: pd.DataFrame) -> dict:
    """
    Diagnostic function to identify issues with VPIN input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input tick data with ['ts', 'buy_vol', 'sell_vol']

    Returns
    -------
    dict
        Diagnostic statistics including:
        - total_ticks: Number of ticks
        - total_volume: Sum of buy + sell volume
        - buy_volume_total: Total buy volume
        - sell_volume_total: Total sell volume
        - zero_volume_ticks: Number of ticks with zero total volume
        - buy_only_ticks: Ticks with buy volume but no sell
        - sell_only_ticks: Ticks with sell volume but no buy
        - neutral_ticks: Ticks with both buy and sell equal
        - buy_sell_ratio: Overall buy/sell ratio
    """
    df = df.copy()
    df['buy_vol'] = pd.to_numeric(df['buy_vol'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['sell_vol'] = pd.to_numeric(df['sell_vol'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['total_vol'] = df['buy_vol'] + df['sell_vol']

    diagnostics = {
        'total_ticks': len(df),
        'total_volume': df['total_vol'].sum(),
        'buy_volume_total': df['buy_vol'].sum(),
        'sell_volume_total': df['sell_vol'].sum(),
        'zero_volume_ticks': (df['total_vol'] == 0).sum(),
        'buy_only_ticks': ((df['buy_vol'] > 0) & (df['sell_vol'] == 0)).sum(),
        'sell_only_ticks': ((df['sell_vol'] > 0) & (df['buy_vol'] == 0)).sum(),
        'neutral_ticks': ((df['buy_vol'] > 0) & (df['buy_vol'] == df['sell_vol'])).sum(),
        'buy_sell_ratio': df['buy_vol'].sum() / df['sell_vol'].sum() if df['sell_vol'].sum() > 0 else np.inf,
        'avg_tick_volume': df['total_vol'].mean(),
        'max_tick_volume': df['total_vol'].max(),
        'min_nonzero_tick_volume': df.loc[df['total_vol'] > 0, 'total_vol'].min() if (df['total_vol'] > 0).any() else 0
    }

    return diagnostics


def vpin_with_diagnostics(
    df: pd.DataFrame,
    V: float,
    window: int = 50,
    print_diagnostics: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculate VPIN with comprehensive diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with ['ts', 'buy_vol', 'sell_vol']
    V : float
        Target bucket size
    window : int
        VPIN window size
    print_diagnostics : bool
        Whether to print diagnostic information

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (VPIN results DataFrame, diagnostics dictionary)
    """
    # Run diagnostics first
    diag = diagnose_vpin_data(df)

    if print_diagnostics:
        print("=" * 60)
        print("VPIN DATA DIAGNOSTICS")
        print("=" * 60)
        for key, value in diag.items():
            print(f"{key:.<40} {value:>15,.2f}" if isinstance(value, float) else f"{key:.<40} {value:>15,}")
        print("=" * 60)

    # Calculate VPIN
    result = vpin_from_signed_ticks(df, V=V, window=window)

    if print_diagnostics:
        print("\nVPIN BUCKET SUMMARY")
        print("=" * 60)
        print(f"Total buckets: {len(result)}")
        print(f"Avg VPIN: {result['vpin'].mean():.4f}")
        print(f"Max VPIN: {result['vpin'].max():.4f}")
        print(f"Avg buy share: {result['buy_share'].mean():.4f}")
        print(f"Avg sell share: {result['sell_share'].mean():.4f}")
        print(f"Max buy run: {result['max_buy_run'].max():.0f}")
        print(f"Max sell run: {result['max_sell_run'].max():.0f}")
        print("=" * 60)

    return result, diag


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    n = 10000

    # Simulate informed trading periods
    informed_periods = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    # Base volumes
    buy_base = np.random.exponential(10, n)
    sell_base = np.random.exponential(10, n)

    # Add informed trading bias
    buy_vol = np.where(informed_periods == 1, buy_base * 1.5, buy_base)
    sell_vol = np.where(informed_periods == 1, sell_base * 0.5, sell_base)

    test_df = pd.DataFrame({
        'ts': pd.date_range('2023-01-01', periods=n, freq='1s'),
        'buy_vol': buy_vol,
        'sell_vol': sell_vol
    })

    print("Testing VPIN calculation with synthetic data...\n")
    result, diag = vpin_with_diagnostics(test_df, V=500, window=50, print_diagnostics=True)

    print("\nSample output:")
    print(result.tail(10))

    # Test edge cases
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    # All buy volume (should show max_buy_run > 0, max_sell_run = 0)
    all_buy = pd.DataFrame({
        'ts': pd.date_range('2023-01-01', periods=100, freq='1s'),
        'buy_vol': np.random.exponential(10, 100),
        'sell_vol': np.zeros(100)
    })

    print("\nTest 1: All buy volume (no sells)")
    result1, _ = vpin_with_diagnostics(all_buy, V=50, window=10, print_diagnostics=True)

    # All sell volume (should show max_sell_run > 0, max_buy_run = 0)
    all_sell = pd.DataFrame({
        'ts': pd.date_range('2023-01-01', periods=100, freq='1s'),
        'buy_vol': np.zeros(100),
        'sell_vol': np.random.exponential(10, 100)
    })

    print("\nTest 2: All sell volume (no buys)")
    result2, _ = vpin_with_diagnostics(all_sell, V=50, window=10, print_diagnostics=True)
