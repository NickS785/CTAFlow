import numpy as np

from exporters.tidy import (
    export_all_tidy,
    export_correlations_tidy,
    export_momentum_tidy,
    export_seasonality_tidy,
    write_tidy,
)
from CTAFlow.stats_utils import fdr_bh


def _sample_seasonality():
    return {
        "meta": {"start_date": "2022-01-01", "end_date": "2022-12-31", "tz": "UTC"},
        "results": [
            {
                "region": "US",
                "slice": "months_9_10_11",
                "ticker": "ZC",
                "mean_return": 0.012,
                "median_return": 0.01,
                "t_stat": 2.8,
                "n": 50,
                "month_breakdown": [
                    {"month": 9, "mean": 0.015, "t_stat": 2.4, "n": 12},
                    {"month": 10, "mean": 0.014, "t_stat": 2.0, "n": 10},
                ],
                "strongest_patterns": ["Harvest strength"],
            },
            {
                "region": "US",
                "slice": "months_9_10_11",
                "ticker": "ZW",
                "mean_return": 0.001,
                "median_return": 0.0,
                "t_stat": 0.5,
                "p_value": 0.62,
                "n": 48,
            },
        ],
    }


def _sample_momentum():
    return {
        "meta": {"return_mode": "close"},
        "results": [
            {
                "ticker": "ZC",
                "session": "session_0",
                "mean_return": 0.004,
                "median_return": 0.003,
                "t_stat": 2.1,
                "n": 40,
                "filtered_months": [3, 1, 3, 5],
            },
            {
                "ticker": "ZC",
                "session": "session_1",
                "mean_return": -0.002,
                "median_return": -0.001,
                "p_value": 0.2,
                "n": 40,
            },
        ],
    }


def _sample_correlations():
    return {
        "results": [
            {"ticker": "ZC", "container": "returns", "x": "momentum", "y": "carry", "r": 0.45, "n": 80},
            {"ticker": "ZC", "container": "returns", "x": "rsi", "y": "carry", "p_value": 0.6, "r": 0.05, "n": 80},
        ]
    }


def test_export_seasonality_includes_fdr():
    df = export_seasonality_tidy(_sample_seasonality())
    assert not df.empty
    assert set(["q_value", "sig_fdr_5pct"]).issubset(df.columns)
    grouped = df[df["p_value"].notna()].groupby(["region", "slice"])
    for key, grp in grouped:
        res = fdr_bh(grp["p_value"], alpha=0.05)
        assert np.allclose(res.q_values, grp["q_value"], atol=1e-12)


def test_export_momentum_filtered_months():
    df = export_momentum_tidy(_sample_momentum())
    assert any(df["metric"] == "filtered_months")
    filt_values = df.loc[df["metric"] == "filtered_months", "value"].iloc[0]
    assert filt_values == "1,3,5"


def test_export_correlations_computes_p():
    df = export_correlations_tidy(_sample_correlations())
    corr_row = df[df["metric"] == "correlation"].iloc[0]
    assert corr_row["p_value"] < 0.01


def test_export_all_and_write(tmp_path):
    results = {
        "seasonality": _sample_seasonality(),
        "momentum": _sample_momentum(),
        "correlations": _sample_correlations(),
    }
    dfs = export_all_tidy(results)
    assert set(dfs) == {"seasonality", "momentum", "correlations"}
    write_tidy(dfs, tmp_path)
    csv_files = list(tmp_path.glob("*.csv"))
    assert csv_files
