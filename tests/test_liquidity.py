import numpy as np
import pandas as pd

from liquidity.volume import (
    compute_liquidity_intraday,
    compute_volume_effects,
    compute_volume_seasonality,
)


def _make_intraday_bars(num_days: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-03", periods=num_days, freq="B", tz="UTC")
    close_shares = np.clip(0.35 + 0.06 * rng.standard_normal(num_days), 0.25, 0.45)
    open_share = 0.2
    lunch_total = 0.05
    total_volume = 50_000
    daily_returns = np.zeros(num_days)
    mean_close = close_shares.mean()
    noise = 0.03 * rng.standard_normal(num_days)
    for i in range(1, num_days):
        daily_returns[i] = 0.35 * (close_shares[i - 1] - mean_close) + noise[i]
    rows = []
    current_close = 100.0
    for day_idx, day in enumerate(days):
        times = pd.date_range(day + pd.Timedelta(hours=9), periods=14, freq="30min", tz="UTC")
        close_share = close_shares[day_idx]
        lunch_idx = [6, 7]
        residual = 1.0 - open_share - lunch_total - close_share
        base_share = residual / (len(times) - (len(lunch_idx) + 2))
        shares = np.full(len(times), base_share)
        shares[0] = open_share
        shares[-1] = close_share
        for idx in lunch_idx:
            shares[idx] = lunch_total / len(lunch_idx)
        shares *= 1.0 / shares.sum()
        next_close = current_close * (1 + daily_returns[day_idx])
        closes = np.linspace(current_close, next_close, len(times))
        for ts, share, close_price in zip(times, shares, closes):
            rows.append(
                {
                    "timestamp": ts,
                    "volume": share * total_volume,
                    "close": close_price,
                    "ticker": "ZC",
                }
            )
        current_close = next_close
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def test_liquidity_intraday_detects_peak_trough():
    bars = _make_intraday_bars()
    intraday = compute_liquidity_intraday(bars, min_samples=10)
    assert not intraday.empty
    peak_time = intraday.loc[intraday["peak_flag"], "clock_time"].iloc[0]
    trough_time = intraday.loc[intraday["trough_flag"], "clock_time"].iloc[0]
    assert peak_time == "15:30"
    assert trough_time in {"12:00", "12:30"}


def test_volume_seasonality_flags_inflated_month():
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    volumes = 1000 + 100 * np.sin(np.linspace(0, 12, len(dates)))
    volumes += np.where(dates.month == 9, 200, 0)
    daily = pd.DataFrame({"volume": volumes, "ticker": "ZC"}, index=dates)
    seasonality = compute_volume_seasonality(daily)
    sept = seasonality[(seasonality["bucket"] == "month") & (seasonality["bucket_value"] == 9)]
    assert not sept.empty
    assert bool(sept["sig_fdr_5pct"].iloc[0])


def test_volume_effects_recovers_close_correlation():
    bars = _make_intraday_bars()
    effects = compute_volume_effects(bars)
    row = effects[(effects["feature"] == "vol_share_close30") & (effects["target"] == "return_next")]
    assert not row.empty
    assert row["r"].abs().iloc[0] > 0.25
    assert row["p_value"].iloc[0] < 0.05
