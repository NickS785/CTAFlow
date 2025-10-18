import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

import numpy as np
import pandas as pd

import tests.conftest  # noqa: F401,E402

_syn_path = ROOT / "CTAFlow" / "utils" / "synthetic_liquidity.py"
_syn_spec = importlib.util.spec_from_file_location("synthetic_liquidity", _syn_path)
synthetic_module = importlib.util.module_from_spec(_syn_spec)
assert _syn_spec.loader is not None
_syn_spec.loader.exec_module(synthetic_module)

IntradayLeg = synthetic_module.IntradayLeg
merge_intraday_legs = synthetic_module.merge_intraday_legs
syn_capacity_impact = synthetic_module.syn_capacity_impact
syn_dollar_turnover = synthetic_module.syn_dollar_turnover
syn_vol_min = synthetic_module.syn_vol_min
synthetic_price = synthetic_module.synthetic_price
synthetic_returns = synthetic_module.synthetic_returns


def _make_leg(symbol: str, prices: list[float], volumes: list[int], start: str, weight: float) -> IntradayLeg:
    idx = pd.date_range(start=start, periods=len(prices), freq="min")
    data = pd.DataFrame({"Close": prices, "Volume": volumes}, index=idx)
    return IntradayLeg(symbol=symbol, data=data, base_weight=weight)


def test_merge_intraday_legs_ffill_and_utc():
    leg_a = _make_leg("CL", [70, 71, 72], [100, 110, 120], "2024-01-01 09:30", 1.0)
    leg_b_times = pd.to_datetime(["2024-01-01 09:30"])
    leg_b_df = pd.DataFrame({"Close": [2], "Volume": [80]}, index=leg_b_times)
    leg_b = IntradayLeg(symbol="RB", data=leg_b_df, base_weight=-1.0)

    merged = merge_intraday_legs([leg_a, leg_b], max_ffill=pd.Timedelta("1min"))

    assert merged.index.tz is not None
    assert merged.index.tz.utcoffset(None) == pd.Timedelta(0)

    # RB volumes should forward fill for one minute but not beyond
    rb_volume = merged[("RB", "volume")]
    assert rb_volume.iloc[1] == 80
    assert pd.isna(rb_volume.iloc[2])


def test_liquidity_metrics_and_price():
    leg_a = _make_leg("CL", [10, 11, 12], [100, 110, 120], "2024-01-01 00:00", 3.0)
    leg_b = _make_leg("HO", [5, 5.5, 6], [60, 70, 80], "2024-01-01 00:00", -2.0)

    merged = merge_intraday_legs([leg_a, leg_b], max_ffill=None)
    weights = {"CL": 3.0, "HO": -2.0}

    vol_min = syn_vol_min([leg_a, leg_b], merged, weights)
    expected_min = pd.Series([min(100 / 3.0, 60 / 2.0), min(110 / 3.0, 70 / 2.0), min(120 / 3.0, 80 / 2.0)], index=merged.index)
    pd.testing.assert_series_equal(vol_min, expected_min)

    turnover = syn_dollar_turnover([leg_a, leg_b], merged, weights)
    expected_turnover = pd.Series(
        [3 * 10 * 100 + 2 * 5 * 60, 3 * 11 * 110 + 2 * 5.5 * 70, 3 * 12 * 120 + 2 * 6 * 80],
        index=merged.index,
    )
    pd.testing.assert_series_equal(turnover, expected_turnover)

    lambdas = {"CL": 0.5, "HO": None}
    capacity = syn_capacity_impact([leg_a, leg_b], merged, lambdas)
    expected_capacity = pd.Series((merged[("CL", "volume")].pow(0.5) / 0.5 ** 0.5), index=merged.index)
    pd.testing.assert_series_equal(capacity, expected_capacity, check_names=False)

    price = synthetic_price([leg_a, leg_b])
    expected_price = 3 * merged[("CL", "price")] + (-2) * merged[("HO", "price")]
    pd.testing.assert_series_equal(price.loc[expected_price.index], expected_price)

    returns = synthetic_returns(price)
    expected_returns = expected_price.apply(np.log).diff()
    pd.testing.assert_series_equal(returns.loc[expected_returns.index], expected_returns)

