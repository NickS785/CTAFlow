"""Example: Predict opening period returns using IntradayMomentumLight.

This model uses:
1. Short-term daily momentum features (1d, 5d, 10d, 20d) - properly lagged
2. HAR-style realized volatility features for 1-day ahead forecasts
3. Opening range volatility measures

Target: Returns during the opening period (first 60 minutes of session)

The model demonstrates proper use of IntradayMomentumLight which inherits from CTALight:
- Build features using IntradayMomentumLight methods
- Features are automatically added via _add_feature() method
- Use built-in .fit() or .fit_with_grid_search() methods (LightGBM)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import time, timedelta
from typing import Dict

import numpy as np
import pandas as pd

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from CTAFlow.models.intraday_momentum import IntradayMomentumLight
from CTAFlow.config import INTRADAY_DATA_PATH


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

    # Create working dataframe
    work_df = pd.DataFrame({'price': intraday_df[price_col]})
    work_df['date'] = work_df.index.normalize()

    # Calculate session times
    session_open_offset = pd.Timedelta(hours=session_open.hour, minutes=session_open.minute)
    work_df['session_start'] = work_df['date'] + session_open_offset
    work_df['session_end'] = work_df['session_start'] + opening_window

    # Get opening and closing prices
    opening_mask = work_df.index >= work_df['session_start']
    opening_data = work_df[opening_mask].groupby('date')['price'].first()

    closing_mask = (work_df.index >= work_df['session_start']) & (work_df.index < work_df['session_end'])
    closing_data = work_df[closing_mask].groupby('date')['price'].last()

    # Calculate log returns
    opening_returns = np.log(closing_data / opening_data)

    return opening_returns


def prepare_daily_data(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Create daily OHLC data from intraday bars."""
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


def main():
    """Run the complete opening returns prediction example."""

    print(f"\n{'='*70}")
    print("OPENING RETURNS PREDICTION WITH IntradayMomentumLight")
    print(f"{'='*70}")

    # Load single ticker data directly from CSV
    print(f"\nLoading ticker data...")
    ticker = "CL"

    # Load intraday data from CSV (same path as gather_tickers uses)
    csv_path = INTRADAY_DATA_PATH / f"{ticker}_intraday.csv"
    print(f"  Loading {ticker} from {csv_path}")

    try:
        intraday_data = pd.read_csv(csv_path, parse_dates=['timestamp'])
        intraday_data.set_index('timestamp', inplace=True)
        intraday_data.sort_index(inplace=True)

        print(f"  Loaded {len(intraday_data)} bars ({intraday_data.index[0].date()} to {intraday_data.index[-1].date()})")

        print(f"\n{'='*70}")
        print(f"PROCESSING {ticker}")
        print(f"{'='*70}")

        # Initialize IntradayMomentumLight (inherits from CTALight)
        print(f"\n1. Initializing IntradayMomentumLight...")
        model = IntradayMomentumLight(
            intraday_data=intraday_data,
            session_open=time(8, 30),
            session_end=time(15, 0),
            closing_length=timedelta(minutes=60),
            tz="America/Chicago",
            price_col="Close"
        )
        print(f"   Model initialized (inherits LightGBM functionality from CTALight)")

        # Prepare daily data
        print(f"\n2. Preparing daily data...")
        daily_df = prepare_daily_data(intraday_data)
        print(f"   Daily data: {len(daily_df)} days")

        # Initialize training_data
        model.training_data = pd.DataFrame(index=daily_df.index)

        # Build features using IntradayMomentumLight methods
        print(f"\n3. Building feature set...")

        # Daily momentum features (lagged by 1 day) - uses _add_feature internally
        print(f"   - Adding daily momentum features...")
        momentum_feats = model.add_daily_momentum_features(
            daily_df,
            lookbacks=(1, 5, 10, 20)
        )
        print(f"     Features: {list(momentum_feats.columns)}")

        # HAR volatility features
        print(f"   - Adding HAR volatility features...")
        har_feats = model.har_volatility_features(
            intraday_df=intraday_data,
            horizons=(1, 5, 22)
        )
        print(f"     Features: {list(har_feats.columns)}")

        # Opening range volatility
        print(f"   - Adding opening range volatility...")
        opening_vol = model.opening_range_volatility(
            intraday_df=intraday_data,
            period_length=timedelta(minutes=60)
        )
        print(f"     Features: {list(opening_vol.columns)}")

        # Combine all features - this becomes our training_data
        print(f"\n4. Combining features...")
        model.training_data = pd.concat([momentum_feats, har_feats, opening_vol], axis=1).dropna()
        print(f"   Combined features shape: {model.training_data.shape}")
        print(f"   Feature names tracked: {model.feature_names}")

        # Calculate target variable
        print(f"\n5. Calculating target (opening returns)...")
        target = calculate_opening_returns(intraday_data)
        print(f"   Target shape: {len(target)}")
        print(f"   Target mean: {target.mean():.6f}, std: {target.std():.6f}")

        # Align features and target
        print(f"\n6. Aligning features and target...")
        common_index = model.training_data.index.intersection(target.index)
        X = model.training_data.loc[common_index]
        y = target.loc[common_index]
        print(f"   Final dataset: {len(X)} samples, {X.shape[1]} features")

        # Train/test split (temporal)
        print(f"\n7. Creating train/test split...")
        test_size = 0.2
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"   Train: {len(X_train)} samples ({X_train.index[0].date()} to {X_train.index[-1].date()})")
        print(f"   Test:  {len(X_test)} samples ({X_test.index[0].date()} to {X_test.index[-1].date()})")

        # Create validation set for early stopping
        val_size = int(len(X_train) * 0.2)
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        X_train_fit = X_train.iloc[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]

        # Fit the model using built-in .fit() method from CTALight
        print(f"\n8. Training LightGBM model (via CTALight.fit())...")
        model.fit(
            X_train_fit,
            y_train_fit,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            num_boost_round=1000
        )
        print(f"   Model trained successfully!")

        # Evaluate on test set using built-in evaluate() method
        print(f"\n9. Evaluating on test set...")
        test_metrics = model.evaluate(X_test, y_test)

        print(f"\n   Test Metrics:")
        print(f"   - MSE:  {test_metrics['mse']:.6f}")
        print(f"   - RMSE: {test_metrics['rmse']:.6f}")
        print(f"   - MAE:  {test_metrics['mae']:.6f}")
        print(f"   - R²:   {test_metrics['r2']:.4f}")
        print(f"   - Directional Accuracy: {test_metrics['directional_accuracy']:.2%}")

        # Feature importance using built-in method
        print(f"\n10. Top 10 feature importances:")
        top_features = model.get_feature_importance(importance_type='gain', top_n=10)
        for feat, importance in top_features.items():
            print(f"    {feat:20s}: {importance:.1f}")

    except Exception as e:
        print(f"\n[ERROR] Failed to process {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"\n{'='*70}")
    print("DEMONSTRATING GRID SEARCH")
    print(f"{'='*70}")

    # Example of using fit_with_grid_search() with same ticker
    print(f"\nRunning grid search example for {ticker}...")

    # Reinitialize model for grid search
    model_gs = IntradayMomentumLight(
        intraday_data=intraday_data,
        session_open=time(8, 30),
        session_end=time(15, 0),
        tz="America/Chicago"
    )

    # Rebuild features (reusing same code as above)
    model_gs.training_data = pd.DataFrame(index=daily_df.index)
    momentum_feats_gs = model_gs.add_daily_momentum_features(daily_df, lookbacks=(1, 5, 10, 20))
    har_feats_gs = model_gs.har_volatility_features(intraday_df=intraday_data, horizons=(1, 5, 22))
    opening_vol_gs = model_gs.opening_range_volatility(intraday_df=intraday_data, period_length=timedelta(minutes=60))
    model_gs.training_data = pd.concat([momentum_feats_gs, har_feats_gs, opening_vol_gs], axis=1).dropna()

    # Use same target
    common_index_gs = model_gs.training_data.index.intersection(target.index)
    X_gs = model_gs.training_data.loc[common_index_gs]
    y_gs = target.loc[common_index_gs]

    # Split (80/20 for grid search demo)
    split_idx = int(len(X_gs) * 0.8)
    X_train_gs, X_test_gs = X_gs.iloc[:split_idx], X_gs.iloc[split_idx:]
    y_train_gs, y_test_gs = y_gs.iloc[:split_idx], y_gs.iloc[split_idx:]
    val_size = int(len(X_train_gs) * 0.2)
    X_val_gs = X_train_gs.iloc[-val_size:]
    y_val_gs = y_train_gs.iloc[-val_size:]
    X_train_fit_gs = X_train_gs.iloc[:-val_size]
    y_train_fit_gs = y_train_gs.iloc[:-val_size]

    # Small param grid for demo
    param_grid = {
        'num_leaves': [31, 63],
        'learning_rate': [0.03, 0.07],
        'feature_fraction': [0.7, 0.9]
    }

    print(f"\nRunning .fit_with_grid_search()...")
    print(f"  Parameter grid: {param_grid}")

    grid_results = model_gs.fit_with_grid_search(
        X_train_fit_gs,
        y_train_fit_gs,
        param_grid=param_grid,
        eval_set=(X_val_gs, y_val_gs),
        cv_folds=3,
        scoring='neg_mean_squared_error',
        verbose=True
    )

    print(f"\n  Best parameters: {grid_results['best_params']}")
    print(f"  Best CV score: {grid_results['best_score']:.6f}")

    # Evaluate grid search model
    gs_metrics = model_gs.evaluate(X_test_gs, y_test_gs)
    print(f"\n  Grid Search Test R²: {gs_metrics['r2']:.4f}")

    print(f"\n{'='*70}")
    print("Example complete!")
    print(f"{'='*70}")

    return {
        'model': model,
        'model_grid_search': model_gs,
        'test_metrics': test_metrics,
        'grid_search_results': grid_results
    }


if __name__ == "__main__":
    results = main()
