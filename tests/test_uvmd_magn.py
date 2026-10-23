"""Upstream UVMD (needs torch). Untrained N-layer UVMD == VMD with N iterations (tau=0, alpha=2000,
DC mode, uniform init); trained-parameter forward == NumPy replica."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from datasets.synthetic import make_synthetic  # noqa: E402
from decomp import uvmd_magn  # noqa: E402
from decomp.vmd_numpy import uvmd_forward_numpy, vmd  # noqa: E402


@pytest.mark.parametrize("N", [1, 2, 3])
def test_untrained_equals_vmd_n(N):
    D = make_synthetic(2, 256, 0.1, seed=3)
    m = uvmd_magn.make_model(256, 4, N)
    for f in D["x"]:
        u, om = uvmd_magn.decompose(m, f)
        r = vmd(f.astype(np.float32).astype(np.float64), 2000.0, 0.0, 4, True, 1, tol=-1, max_iter=N)
        np.testing.assert_allclose(u, r["u"], atol=1e-3 * np.abs(r["u"]).max())
        np.testing.assert_allclose(om, r["omega"], atol=1e-4)


def test_trained_matches_numpy_replica():
    D = make_synthetic(12, 256, 0.1, seed=4)
    m = uvmd_magn.make_model(256, 4, 2)
    uvmd_magn.train_uvmd(m, D["x"][:8], D["x"][8:], epochs=3, log=lambda s: None)
    P = uvmd_magn.export_params(m)
    for f in D["x"][8:]:
        u, _ = uvmd_magn.decompose(m, f)
        r = uvmd_forward_numpy(f.astype(np.float32).astype(np.float64), P["alpha"], P["lambda"], 4)
        np.testing.assert_allclose(u, r["u"], atol=1e-3 * np.abs(r["u"]).max())


def test_param_count_matches_upstream_print():
    m = uvmd_magn.make_model(256, 13, 1)
    # upstream: K alphas + (layers+1) x 2T complex lambdas
    assert uvmd_magn.count_params(m)["numel"] == 13 + 2 * 512
