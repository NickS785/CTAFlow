import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "CTAFlow" / "utils" / "unit_conversions.py"
spec = importlib.util.spec_from_file_location("CTAFlow.utils.unit_conversions", MODULE_PATH)
unit_conversions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unit_conversions)

gallons_to_barrels = unit_conversions.gallons_to_barrels
barrels_to_gallons = unit_conversions.barrels_to_gallons
bushels_to_kilograms = unit_conversions.bushels_to_kilograms
bushels_to_metric_tons = unit_conversions.bushels_to_metric_tons


def test_gallons_barrels_conversion():
    assert gallons_to_barrels(42) == pytest.approx(1.0)
    assert barrels_to_gallons(1) == pytest.approx(42.0)

    array_result = gallons_to_barrels(np.array([0, 84]))
    np.testing.assert_allclose(array_result, np.array([0.0, 2.0]))

    series = pd.Series([42, 84], name="volume")
    barrels_series = gallons_to_barrels(series)
    expected_series = pd.Series([1.0, 2.0], name="volume")
    pd.testing.assert_series_equal(barrels_series, expected_series)


def test_bushels_conversions():
    corn_kg = bushels_to_kilograms(2, "corn")
    assert corn_kg == pytest.approx(2 * 56.0 * 0.45359237)

    soy_series = pd.Series([1, 2], name="soy")
    soy_tons = bushels_to_metric_tons(soy_series, "soybeans")
    expected_tons = soy_series.astype(float) * 60.0 * 0.45359237 / 1000.0
    pd.testing.assert_series_equal(soy_tons, expected_tons)

    with pytest.raises(KeyError):
        bushels_to_kilograms(1, "unknown")

    with pytest.raises(TypeError):
        bushels_to_metric_tons(1, 123)
