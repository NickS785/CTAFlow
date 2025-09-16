import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "CTAFlow"


@contextmanager
def load_curve_manager_with_stubs():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    stub_names = [
        "CTAFlow",
        "CTAFlow.config",
        "CTAFlow.data",
        "CTAFlow.data.data_client",
        "CTAFlow.data.contract_handling",
        "CTAFlow.data.contract_handling.curve_manager",
        "CTAFlow.utils.tenor_interpolation",
    ]

    original_modules = {name: sys.modules.get(name) for name in stub_names}

    try:
        cta_pkg = types.ModuleType("CTAFlow")
        cta_pkg.__path__ = [str(PACKAGE_ROOT)]
        sys.modules["CTAFlow"] = cta_pkg

        config_stub = types.ModuleType("CTAFlow.config")
        config_stub.RAW_MARKET_DATA_PATH = Path(".")
        config_stub.MARKET_DATA_PATH = Path("./market_data.h5")
        sys.modules["CTAFlow.config"] = config_stub

        data_pkg = types.ModuleType("CTAFlow.data")
        data_pkg.__path__ = []
        sys.modules["CTAFlow.data"] = data_pkg

        data_client_stub = types.ModuleType("CTAFlow.data.data_client")

        class _StubDataClient:  # pragma: no cover - simple stub
            def query_curve_data(self, *args, **kwargs):
                return {}

            def read_roll_dates(self, *args, **kwargs):
                raise KeyError("no roll data in stub")

        data_client_stub.DataClient = _StubDataClient
        sys.modules["CTAFlow.data.data_client"] = data_client_stub

        contract_pkg = types.ModuleType("CTAFlow.data.contract_handling")
        contract_pkg.__path__ = []
        sys.modules["CTAFlow.data.contract_handling"] = contract_pkg

        curve_manager_path = PACKAGE_ROOT / "data" / "contract_handling" / "curve_manager.py"
        spec = importlib.util.spec_from_file_location(
            "CTAFlow.data.contract_handling.curve_manager", curve_manager_path
        )
        curve_manager = importlib.util.module_from_spec(spec)
        sys.modules["CTAFlow.data.contract_handling.curve_manager"] = curve_manager
        spec.loader.exec_module(curve_manager)

        tenor_path = PACKAGE_ROOT / "utils" / "tenor_interpolation.py"
        tenor_spec = importlib.util.spec_from_file_location(
            "CTAFlow.utils.tenor_interpolation", tenor_path
        )
        tenor_module = importlib.util.module_from_spec(tenor_spec)
        sys.modules["CTAFlow.utils.tenor_interpolation"] = tenor_module
        tenor_spec.loader.exec_module(tenor_module)

        yield curve_manager, tenor_module

    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_spread_data_utils_interfaces():
    with load_curve_manager_with_stubs() as (curve_manager, tenor_module):
        SpreadData = curve_manager.SpreadData
        log_interp_to_tau = tenor_module.log_interp_to_tau

        np.random.seed(0)
        spread = SpreadData.example()

        seq_prices = spread.get_sequential_prices()
        assert isinstance(seq_prices, pd.DataFrame)
        assert not seq_prices.empty

        prices, expiries = spread.get_constant_maturity_inputs()
        assert isinstance(prices, pd.DataFrame)
        assert isinstance(expiries, pd.Series)
        assert not prices.empty
        assert not expiries.empty

        taus = np.array([1 / 12, 0.5, 1.0])
        interpolated = log_interp_to_tau(prices, expiries, taus)
        assert set(interpolated.keys()) == set(taus)
        for series in interpolated.values():
            assert isinstance(series, pd.Series)


def test_cross_product_spread_data_differences():
    with load_curve_manager_with_stubs() as (curve_manager, tenor_module):
        SpreadData = curve_manager.SpreadData
        CrossProductSpreadData = curve_manager.CrossProductSpreadData

        np.random.seed(1)
        base = SpreadData.example()
        np.random.seed(2)
        hedge = SpreadData.example()

        cross = CrossProductSpreadData(base, hedge)
        df = cross.to_dataframe()
        assert not df.empty
        assert list(df.columns) == cross.columns

        m0_spread = cross.get_contract_spread('M0')
        assert isinstance(m0_spread, pd.Series)
        assert len(m0_spread) == len(cross.index)

        front = cross.get_front_spread()
        assert front is not None

        summary = cross.describe()
        assert summary['pair'] == cross.pair_label
        assert 'mean_spread' in summary and 'std_spread' in summary
        assert len(summary['legs']) == 2

        leg_labels = cross.leg_labels
        assert leg_labels[0].startswith('EXAMPLE')
        assert leg_labels[1] != leg_labels[0]

        weights = cross.get_leg_weights()
        assert set(weights.keys()) == set(leg_labels)

        first_leg_label = leg_labels[0]
        second_leg_label = leg_labels[1]

        first_contribution = cross.get_leg_contribution(first_leg_label)
        second_contribution = cross.get_leg_contribution(summary['legs'][1]['source_label'])

        assert isinstance(first_contribution, pd.DataFrame)
        assert isinstance(second_contribution, pd.DataFrame)

        assert summary['legs'][0]['source_label'] == cross.leg_label_map[first_leg_label]
        assert summary['legs'][1]['source_label'] == cross.leg_label_map[second_leg_label]


def test_cross_product_spread_multi_leg_ratios():
    with load_curve_manager_with_stubs() as (curve_manager, _):
        SpreadData = curve_manager.SpreadData
        CrossProductSpreadData = curve_manager.CrossProductSpreadData
        CrossSpreadLeg = curve_manager.CrossSpreadLeg

        np.random.seed(3)
        oil = SpreadData.example()
        np.random.seed(4)
        gasoline = SpreadData.example()
        np.random.seed(5)
        heating_oil = SpreadData.example()

        crack = CrossProductSpreadData(
            base=oil,
            hedge=gasoline,
            base_ratio=3.0,
            hedge_ratio=-2.0,
            additional_legs=[CrossSpreadLeg(data=heating_oil, ratio=-1.0, label="HeatingOil")],
        )

        crack_df = crack.to_dataframe()
        assert not crack_df.empty

        base_prices = oil.get_sequential_prices(dropna=False).reindex(
            index=crack_df.index,
            columns=crack_df.columns,
        )
        gas_prices = gasoline.get_sequential_prices(dropna=False).reindex(
            index=crack_df.index,
            columns=crack_df.columns,
        )
        heat_prices = heating_oil.get_sequential_prices(dropna=False).reindex(
            index=crack_df.index,
            columns=crack_df.columns,
        )

        expected = (base_prices * 3.0) + (gas_prices * -2.0) + (heat_prices * -1.0)
        pd.testing.assert_frame_equal(crack_df, expected)

        summary = crack.describe()
        assert len(summary['legs']) == 3
        heating_leg = next(leg for leg in summary['legs'] if leg['source_label'] == 'HeatingOil')
        assert heating_leg['ratio'] == -1.0

        contributions = crack.get_leg_contribution('HeatingOil')
        pd.testing.assert_frame_equal(contributions, heat_prices * -1.0)


def test_cross_product_spread_contract_overrides():
    with load_curve_manager_with_stubs() as (curve_manager, _):
        SpreadData = curve_manager.SpreadData
        CrossProductSpreadData = curve_manager.CrossProductSpreadData

        np.random.seed(6)
        base = SpreadData.example()
        np.random.seed(7)
        hedge = SpreadData.example()

        override_spread = CrossProductSpreadData(
            base=base,
            hedge=hedge,
            hedge_contract_ratios={'M0': -2.0, 'M1': -0.5},
        )

        override_df = override_spread.to_dataframe()
        base_prices = base.get_sequential_prices(dropna=False).reindex(
            index=override_df.index,
            columns=override_df.columns,
        )
        hedge_prices = hedge.get_sequential_prices(dropna=False).reindex(
            index=override_df.index,
            columns=override_df.columns,
        )

        hedge_label = override_spread.leg_labels[1]
        hedge_weights = pd.Series(override_spread.get_leg_weights()[hedge_label])
        expected_override = base_prices + hedge_prices.multiply(hedge_weights, axis=1)

        pd.testing.assert_frame_equal(override_df, expected_override)
        assert hedge_weights['M0'] == -2.0
        assert hedge_weights['M1'] == -0.5
