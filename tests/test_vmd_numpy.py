"""decomp.vmd_numpy.vmd must reproduce vmdpy exactly (vmdpy returns iterate n-1 after n iterations)."""
import numpy as np
import pytest
from vmdpy import VMD as vmdpy_VMD

from datasets.synthetic import make_synthetic
from decomp.vmd_numpy import vmd


@pytest.mark.parametrize("tau,K,DC,init", [(0.0, 4, 1, 1), (0.1, 4, 1, 1), (0.0, 6, 0, 1), (0.0, 3, 0, 0)])
def test_matches_vmdpy(tau, K, DC, init):
    x = make_synthetic(3, 512, 0.1, seed=1)["x"]
    for f in x:
        u_ref, _, om_ref = vmdpy_VMD(f, 2000, tau, K, DC, init, 1e-7)
        n = om_ref.shape[0]
        r = vmd(f, 2000, tau, K, bool(DC), init, tol=-1, max_iter=n - 1)
        np.testing.assert_allclose(r["u"], u_ref, atol=1e-10)
        np.testing.assert_allclose(r["omega"], om_ref[-1], atol=1e-12)


def test_converges_same_count():
    f = make_synthetic(1, 512, 0.1, seed=2)["x"][0]
    _, _, om_ref = vmdpy_VMD(f, 2000, 0.0, 4, 1, 1, 1e-7)
    assert vmd(f, 2000, 0.0, 4, True, 1, tol=1e-7)["n_iter"] == om_ref.shape[0]
