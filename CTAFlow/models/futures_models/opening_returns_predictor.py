"""Example: Predict opening period returns using momentum and HAR volatility.

This model uses:
1. Short-term daily momentum features (1d, 5d, 10d, 20d)
2. HAR-style realized volatility features for 1-day ahead forecasts
3. Opening range volatility measures

Target: Returns during the opening period (first 60 minutes of session)

The model demonstrates how to:
- Load futures data using gather_tickers
- Create IntradayMomentumLight instances
- Generate momentum and volatility features
- Build training/test sets with proper temporal splits
- Train and evaluate predictions
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import time, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from CTAFlow.models.intraday_momentum import IntradayMomentumLight
from CTAFlow.config import INTRADAY_DATA_PATH

# Import gather_tickers from futures_screens
from screens.futures_screens import gather_tickers


def calculate_opening_returns(
    intraday_df: pd.DataFrame,
    session_open: time = time(8, 30),
    opening_window: timedelta = timedelta(minutes=60),
    price_col: str = "Close",
) -> pd.Series:
    """Calculate daily returns during the opening period.

    Parameters
    ----------
    intraday_df : pd.DataFrame
        Intraday OHLCV data with DatetimeIndex
    session_open : time
        Session start time (default 8:30 AM)
    opening_window : timedelta
        Length of opening period (default 60 minutes)
    price_col : str
        Price column name

    Returns
    -------
    pd.Series
        Daily opening period returns (log returns from open to open+window)
    """
    if not isinstance(intraday_df.index, pd.DatetimeIndex):
        intraday_df.index = pd.to_datetime(intraday_df.index)

    # Create working dataframe with date and session times
    work_df = pd.DataFrame({'price': intraday_df[price_col]})
    work_df['date'] = work_df.index.normalize()

    # Calculate session start and end times
    session_open_offset = pd.Timedelta(
        hours=session_open.hour,
        minutes=session_open.minute
    )
    work_df['session_start'] = work_df['date'] + session_open_offset
    work_df['session_end'] = work_df['session_start'] + opening_window

    # Get opening and closing prices for each day
    # Opening: first price at or after session_start
    opening_mask = work_df.index >= work_df['session_start']
    opening_data = work_df[opening_mask].groupby('date')['price'].first()

    # Closing: last price before session_end
    closing_mask = (work_df.index >= work_df['session_start']) & (work_df.index < work_df['session_end'])
    closing_data = work_df[closing_mask].groupby('date')['price'].last()

    # Calculate log returns
    opening_returns = np.log(closing_data / opening_data)

    return opening_returns


def prepare_daily_data(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Create daily OHLC data from intraday bars.

    Parameters
    ----------
    intraday_df : pd.DataFrame
        Intraday data with OHLCV

    Returns
    -------
    pd.DataFrame
        Daily OHLC data
    """
    if not isinstance(intraday_df.index, pd.DatetimeIndex):
        intraday_df.index = pd.to_datetime(intraday_df.index)

    daily = intraday_df.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return daily


def build_features_and_target(
    ticker: str,
    intraday_data: pd.DataFrame,
    momentum_lookbacks: Tuple[int, ...] = (1, 5, 10, 20),
    har_horizons: Tuple[int, ...] = (1, 5, 22),
) -> pd.DataFrame:
    """Build complete feature set for opening returns prediction.

    Parameters
    ----------
    ticker : str
        Ticker symbol for logging
    intraday_data : pd.DataFrame
        Intraday OHLCV data
    momentum_lookbacks : tuple
        Lookback periods for momentum features
    har_horizons : tuple
        Horizons for HAR volatility features

    Returns
    -------
    pd.DataFrame
        Combined features and target with columns:
        - momentum_Xd: daily momentum features (lagged)
        - rv_Xd: realized volatility HAR features
        - rv_open: opening range volatility
        - rsv_pos_open, rsv_neg_open: opening range semivariance
        - target_opening_return: target variable
    """
    print(f"\n{'='*70}")
    print(f"Building features for {ticker}")
    print(f"{'='*70}")

    # Initialize model
    model = IntradayMomentumLight(
        intraday_data=intraday_data,
        session_open=time(8, 30),
        session_end=time(15, 0),
        closing_length=timedelta(minutes=60),
        tz="America/Chicago",
        price_col="Close"
    )

    # Prepare daily data
    print(f"\nPreparing daily data...")
    daily_df = prepare_daily_data(intraday_data)
    print(f"  Daily data shape: {daily_df.shape}")
    print(f"  Date range: {daily_df.index[0].date()} to {daily_df.index[-1].date()}")

    # 1. Daily momentum features (lagged by 1 day)
    print(f"\nGenerating momentum features...")
    momentum_feats = model.add_daily_momentum_features(
        daily_df,
        lookbacks=momentum_lookbacks
    )
    print(f"  Momentum features: {list(momentum_feats.columns)}")
    print(f"  Shape: {momentum_feats.shape}")

    # 2. HAR volatility features
    print(f"\nGenerating HAR volatility features...")
    har_feats = model.har_volatility_features(
        intraday_df=intraday_data,
        horizons=har_horizons
    )
    print(f"  HAR features: {list(har_feats.columns)}")
    print(f"  Shape: {har_feats.shape}")

    # 3. Opening range volatility
    print(f"\nGenerating opening range volatility...")
    opening_vol = model.opening_range_volatility(
        intraday_df=intraday_data,
        period_length=timedelta(minutes=60)
    )
    print(f"  Opening vol features: {list(opening_vol.columns)}")
    print(f"  Shape: {opening_vol.shape}")

    # 4. Target: Opening period returns
    print(f"\nCalculating target (opening returns)...")
    target = calculate_opening_returns(
        intraday_data,
        session_open=time(8, 30),
        opening_window=timedelta(minutes=60)
    )
    print(f"  Target shape: {target.shape}")
    print(f"  Target mean: {target.mean():.6f}")
    print(f"  Target std: {target.std():.6f}")

    # Combine all features
    print(f"\nCombining features...")
    combined = pd.concat([
        momentum_feats,
        har_feats,
        opening_vol,
        target.rename('target_opening_return')
    ], axis=1)

    # Drop rows with missing data
    combined = combined.dropna()
    print(f"  Combined shape (after dropna): {combined.shape}")
    print(f"  Features: {[c for c in combined.columns if c != 'target_opening_return']}")

    return combined


def train_and_evaluate(
    data: pd.DataFrame,
    test_size: float = 0.2,
    n_splits: int = 3,
) -> Dict:
    """Train model and evaluate with time series cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        Combined features and target
    test_size : float
        Proportion of data for final test set
    n_splits : int
        Number of time series CV splits

    Returns
    -------
    dict
        Results including metrics and predictions
    """
    from sklearn.ensemble import RandomForestRegressor

    print(f"\n{'='*70}")
    print("TRAINING AND EVALUATION")
    print(f"{'='*70}")

    # Separate features and target
    feature_cols = [c for c in data.columns if c != 'target_opening_return']
    X = data[feature_cols].values
    y = data['target_opening_return'].values
    dates = data.index

    print(f"\nData summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")

    # Time-based train/test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    print(f"\nTrain/Test split:")
    print(f"  Train: {len(X_train)} samples ({dates_train[0].date()} to {dates_train[-1].date()})")
    print(f"  Test:  {len(X_test)} samples ({dates_test[0].date()} to {dates_test[-1].date()})")

    # Time series cross-validation
    print(f"\nTime series cross-validation ({n_splits} splits)...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            random_state=42
        )
        model.fit(X_fold_train, y_fold_train)

        y_pred_val = model.predict(X_fold_val)
        mse = mean_squared_error(y_fold_val, y_pred_val)
        r2 = r2_score(y_fold_val, y_pred_val)
        cv_scores.append({'fold': fold + 1, 'mse': mse, 'r2': r2})
        print(f"  Fold {fold + 1}: MSE={mse:.6f}, R²={r2:.4f}")

    # Train final model on full training set
    print(f"\nTraining final model...")
    final_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        random_state=42
    )
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    print(f"\nTest set evaluation:")
    y_pred_test = final_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"  MSE: {test_mse:.6f}")
    print(f"  RMSE: {np.sqrt(test_mse):.6f}")
    print(f"  R²: {test_r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 feature importances:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    return {
        'model': final_model,
        'cv_scores': cv_scores,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'feature_importance': feature_importance,
        'predictions': pd.DataFrame({
            'date': dates_test,
            'actual': y_test,
            'predicted': y_pred_test
        }).set_index('date')
    }


def main():
    """Run the complete opening returns prediction example."""

    print(f"\n{'='*70}")
    print("OPENING RETURNS PREDICTION WITH MOMENTUM & HAR VOLATILITY")
    print(f"{'='*70}")

    # Load data using gather_tickers from futures_screens.py
    print(f"\nLoading ticker data...")
    tickers = ["CL", "GC", "SI"]
    start_date = "2020-01-01"

    print(f"  Tickers: {tickers}")
    print(f"  Start date: {start_date}")
    print(f"  Method: CSV")

    data = gather_tickers(tickers, start_date=start_date, load_method='csv')

    print(f"\nLoaded {len(data)} tickers:")
    for ticker, df in data.items():
        print(f"  {ticker}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Build features for each ticker
    results = {}

    for ticker in tickers:
        try:
            ticker_data = data[ticker]

            # Build features and target
            combined_df = build_features_and_target(
                ticker=ticker,
                intraday_data=ticker_data,
                momentum_lookbacks=(1, 5, 10, 20),
                har_horizons=(1, 5, 22)
            )

            # Train and evaluate
            ticker_results = train_and_evaluate(
                data=combined_df,
                test_size=0.2,
                n_splits=3
            )

            results[ticker] = ticker_results

        except Exception as e:
            print(f"\nError processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    summary_df = pd.DataFrame([
        {
            'ticker': ticker,
            'test_r2': res['test_r2'],
            'test_rmse': np.sqrt(res['test_mse']),
            'cv_avg_r2': np.mean([s['r2'] for s in res['cv_scores']])
        }
        for ticker, res in results.items()
    ])

    print(f"\n{summary_df.to_string(index=False)}")

    print(f"\n{'='*70}")
    print("Example complete!")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    results = main()
