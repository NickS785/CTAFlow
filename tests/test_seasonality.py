import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

import numpy as np
import pandas as pd

import tests.conftest  # noqa: F401,E402

_season_path = ROOT / "CTAFlow" / "utils" / "seasonality.py"
_season_spec = importlib.util.spec_from_file_location("seasonality_utils", _season_path)
season_module = importlib.util.module_from_spec(_season_spec)
assert _season_spec.loader is not None
_season_spec.loader.exec_module(season_module)

add_seasonal_keys = season_module.add_seasonal_keys
seasonality_regression = season_module.seasonality_regression
seasonality_summary = season_module.seasonality_summary


def test_add_seasonal_keys():
    idx = pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC")
    df = pd.DataFrame({"ret": [0.0, 0.1, -0.1]}, index=idx)
    enriched = add_seasonal_keys(df)
    assert {"hour", "dow", "wom", "month"}.issubset(enriched.columns)
    assert enriched.loc[idx[0], "hour"] == 0


def test_seasonality_summary_hour_effect():
    idx = pd.date_range("2024-01-01", periods=24 * 5, freq="H", tz="UTC")
    ret = np.zeros(len(idx))
    ret[idx.hour == 3] = 0.01
    df = pd.DataFrame({"ret": ret}, index=idx)

    summary = seasonality_summary({"SPREAD": df}, hac_lags=3)
    hour_rows = summary[summary["factor"] == "hour"]
    assert not hour_rows.empty
    row = hour_rows[hour_rows["level"] == "3"].iloc[0]
    assert row["coef"] > 0.009


def test_seasonality_regression_handles_multiple_symbols():
    idx = pd.date_range("2024-01-01", periods=24, freq="H", tz="UTC")
    df1 = pd.DataFrame({"ret": np.sin(np.arange(24))}, index=idx)
    df2 = pd.DataFrame({"ret": np.cos(np.arange(24))}, index=idx)

    summary = seasonality_summary({"A": df1, "B": df2}, hac_lags=1)
    assert set(summary["symbol"]) == {"A", "B"}

    enriched = add_seasonal_keys(df1)
    design = pd.get_dummies(enriched["hour"].astype(int), prefix="hour")
    design.index = enriched.index
    res = seasonality_regression(enriched["ret"], design, hac_lags=1)
    assert "params" in res and len(res["params"]) == design.shape[1]
