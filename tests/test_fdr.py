import numpy as np

from stats_utils.fdr import fdr_bh


def test_fdr_bh_known_values():
    pvals = np.array([0.01, 0.04, 0.03, 0.002, 0.5])
    result = fdr_bh(pvals, alpha=0.05)
    expected_q = np.array([0.025, 0.05, 0.05, 0.01, 0.5])
    assert np.allclose(result.q_values, expected_q, atol=1e-12)
    assert result.rejected.tolist() == [True, True, True, True, False]


def test_fdr_by_matches_bh_scaled():
    pvals = np.array([0.01, 0.02, 0.03, 0.2])
    res_bh = fdr_bh(pvals, alpha=0.1, method="BH")
    res_by = fdr_bh(pvals, alpha=0.1, method="BY")
    harmonic = sum(1.0 / (i + 1) for i in range(len(pvals)))
    assert np.allclose(res_by.q_values, res_bh.q_values * harmonic, atol=1e-12)


def test_fdr_handles_nan_and_inf():
    pvals = np.array([0.01, np.nan, np.inf, 1.0, 0.0])
    result = fdr_bh(pvals, alpha=0.05)
    assert np.isnan(result.q_values[1])
    assert np.isnan(result.q_values[2])
    assert not bool(result.rejected[1])
    assert not bool(result.rejected[2])
    assert result.q_values[4] == 0.0


def test_fdr_empty_and_all_nan():
    empty = fdr_bh([], alpha=0.05)
    assert empty.q_values.size == 0
    all_nan = fdr_bh([np.nan, np.nan])
    assert np.isnan(all_nan.q_values).all()
    assert not all_nan.rejected.any()


def test_invalid_alpha_raises():
    try:
        fdr_bh([0.1], alpha=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for alpha <= 0")
