"""upper_limit_mu with a full background covariance (cov_modes) against the diagonal treatment.

The pyhf limit fits take tens of seconds each, so those tests are marked ``slow`` (the pre-commit hook skips them;
run ``pytest tests/hnl_mupi`` for all)."""
import numpy as np
import pytest

import cafpybara.analyses.hnl_mupi as ana
from cafpybara.analyses.hnl_mupi.sensitivity import cov_modes


def _dicts(b, s, mc, cov):
    b, s, mc = (np.asarray(x, float) for x in (b, s, mc))
    syst = np.sqrt(np.diag(cov))
    bkg = {"counts": b, "mcstat_err": mc, "syst_cov": cov, "syst_err": syst,
           "total_err": np.sqrt(mc ** 2 + syst ** 2)}
    sig = {"counts": s, "mcstat_err": 0.1 * s, "total_err": 0.1 * s}
    return sig, bkg


def test_cov_modes_rebuild_the_covariance_and_drop_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T
    m = cov_modes(cov)
    assert np.allclose(sum(np.outer(v, v) for v in m), cov)
    assert len(cov_modes(np.outer([1, 2, 3.0], [1, 2, 3.0]))) == 1


@pytest.mark.slow
def test_one_bin_full_cov_equals_the_diagonal_treatment():
    sig, bkg = _dicts([100.0], [5.0], [4.0], np.array([[36.0]]))
    full = ana.upper_limit_mu(sig, bkg)["median"]
    diag = ana.upper_limit_mu(sig, bkg, full_cov=False)["median"]
    assert np.isclose(full, diag, rtol=2e-2)


@pytest.mark.slow
def test_correlations_change_the_multibin_limit():
    b, s, mc = [100.0, 100.0, 100.0], [1.0, 3.0, 8.0], [3.0, 3.0, 3.0]
    v = np.array([10.0, 10.0, 10.0])
    sig, bkg = _dicts(b, s, mc, np.outer(v, v))               # fully correlated systematic
    full = ana.upper_limit_mu(sig, bkg)["median"]
    diag = ana.upper_limit_mu(sig, bkg, full_cov=False)["median"]
    assert full != diag and np.isfinite(full) and full > 0


@pytest.mark.slow
def test_empty_background_bins_do_not_break_the_fit():
    b, s, mc = [0.0, 2.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0, 1.0, 0.0], [0.0, 1.4, 0.0, 0.0, 1.0, 0.0]
    v = np.array([0.0, 0.3, 0.0, 0.0, 0.2, 0.0])
    sig, bkg = _dicts(b, s, mc, np.outer(v, v) + np.diag(v ** 2))
    for kw in ({}, {"full_cov": False}, {"mc_stat": False}):
        lim = ana.upper_limit_mu(sig, bkg, **kw)
        assert np.isfinite(lim["median"]) and lim["median"] > 0


@pytest.mark.slow
def test_sensitivity_table_survives_a_fit_that_cannot_converge():
    """Numbers from a 300-event smoke run: 3 weighted BNB slices, 100% bin errors, detvar-only bins."""
    import warnings
    b = np.array([2031.07, 0, 2031.07, 2031.07, 0, 0])
    cov = np.array([[2733.2, -194.0, 2060.9, 2205.6, 575.3, 0], [-194.0, 194.0, 97.0, 0, -194.0, 0],
                    [2060.9, 97.0, 2206.5, 2110.3, -97.1, 0], [2205.6, 0, 2110.3, 2110.3, 95.3, 0],
                    [575.3, -194.0, -97.1, 95.3, 575.4, 0], [0, 0, 0, 0, 0, 0]])
    sig, bkg = _dicts(b, [208.2, 43.8, 0, 0, 0, 0], b, cov)
    sig.update({"m": 300, "U": 1e-7, "mcstat_err": np.array([78.8, 43.8, 0, 0, 0, 0])})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = ana.sensitivity_table({"mcbnb": bkg, 300: sig})
    assert len(tab) == 2 and set(tab["mc_stat"]) == {True, False}
