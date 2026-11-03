"""Batched differentiable UVMD (decomp/uvmd_torch.py) == the references already used in this repo.

* float64  vs decomp/vmd_numpy.py (tested against vmdpy) with random per-mode alpha and complex lambda schedule
* float32  vs the upstream ``Unfolding`` (third_party/MAGN/uvmd.py) with random learned parameters
* a batch gives the same result as one window at a time
* gradients reach alpha_raw and lam
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from decomp import uvmd_magn  # noqa: E402
from decomp.uvmd_torch import BatchedUVMD, inv_softplus  # noqa: E402
from decomp.vmd_numpy import vmd  # noqa: E402


def _signals(n, T, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X = []
    for _ in range(n):
        f1, f2 = rng.uniform(0.01, 0.05), rng.uniform(0.1, 0.3)
        x = np.sin(2 * np.pi * f1 * t + rng.uniform(0, 6)) + 0.5 * np.sin(2 * np.pi * f2 * t) \
            + 0.2 * rng.standard_normal(T)
        X.append((x - x.mean()) / x.std())
    return np.stack(X)


def _random_params(K, N, T2, seed=1):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(5.0, 400.0, K)
    lam = 3.0 * (rng.standard_normal((N, T2)) + 1j * rng.standard_normal((N, T2)))
    return alpha, lam


@pytest.mark.parametrize("K,N", [(4, 1), (4, 3), (5, 2)])
def test_float64_equals_numpy_vmd(K, N):
    T = 256
    X = _signals(3, T)
    alpha, lam = _random_params(K, N, 2 * T)
    m = BatchedUVMD(T, K, N, dtype=torch.float64).load_params(alpha, lam)
    with torch.no_grad():
        u, om, aux = m(torch.as_tensor(X))
    for i, f in enumerate(X):
        r = vmd(f, alpha=alpha, K=K, DC=True, init=1, lambda_schedule=lam)
        np.testing.assert_allclose(u[i].numpy(), r["u"], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(om[i].numpy(), r["omega"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(aux["rec_l1"][i].item(), r["spec_l1"], rtol=1e-9)


@pytest.mark.parametrize("K,N", [(4, 3), (5, 2)])
def test_float32_equals_upstream_unfolding(K, N):
    T = 256
    X = _signals(3, T, seed=5)
    up = uvmd_magn.make_model(T, K, N)
    alpha, lam = _random_params(K, N, 2 * T, seed=7)
    with torch.no_grad():
        up.Alpha.copy_(inv_softplus(alpha).float())
        up.lambda_hat.zero_()
        up.lambda_hat[:N] = torch.as_tensor(lam, dtype=torch.complex64)   # row N is unused upstream
    P = uvmd_magn.export_params(up)
    np.testing.assert_allclose(P["alpha"], alpha, rtol=1e-6)
    m = BatchedUVMD(T, K, N).load_params(P["alpha"], P["lambda"])
    with torch.no_grad():
        u, om, aux = m(torch.as_tensor(X, dtype=torch.float32))
    for i, f in enumerate(X):
        with torch.no_grad():
            u_up, omega_plus, loss_up = up(torch.as_tensor(f, dtype=torch.float32))
        u_up, om_up = u_up.numpy(), omega_plus[-1].numpy()
        scale = np.abs(u_up).max()
        np.testing.assert_allclose(u[i].numpy(), u_up, atol=2e-4 * scale)
        np.testing.assert_allclose(om[i].numpy(), om_up, atol=2e-5)
        np.testing.assert_allclose(aux["rec_l1"][i].item(), loss_up.item(), rtol=2e-4)


def test_batch_equals_single():
    T, K, N = 128, 4, 2
    X = torch.as_tensor(_signals(5, T), dtype=torch.float32)
    alpha, lam = _random_params(K, N, 2 * T)
    m = BatchedUVMD(T, K, N).load_params(alpha, lam)
    with torch.no_grad():
        ub, ob, ab = m(X)
        for i in range(len(X)):
            u1, o1, a1 = m(X[i:i + 1])
            np.testing.assert_allclose(ub[i].numpy(), u1[0].numpy(), atol=1e-5)
            np.testing.assert_allclose(ob[i].numpy(), o1[0].numpy(), atol=1e-6)
            np.testing.assert_allclose(ab["rec_l1"][i].item(), a1["rec_l1"][0].item(), rtol=1e-5)


def test_untrained_equals_vmd_n_iterations():
    """Default init (alpha 2000, lambda 0) is plain VMD stopped after N iterations."""
    T, K, N = 256, 4, 3
    X = _signals(2, T, seed=9)
    m = BatchedUVMD(T, K, N, dtype=torch.float64)
    with torch.no_grad():
        u, om, _ = m(torch.as_tensor(X))
    for i, f in enumerate(X):
        r = vmd(f, 2000.0, 0.0, K, True, 1, tol=-1, max_iter=N)
        np.testing.assert_allclose(u[i].numpy(), r["u"], rtol=1e-8, atol=1e-9)


def test_gradients_reach_alpha_and_lambda():
    T, K, N = 128, 4, 2
    X = torch.as_tensor(_signals(4, T), dtype=torch.float32)
    m = BatchedUVMD(T, K, N)
    modes, om, aux = m(X)
    (modes[:, 1:].pow(2).mean() + (aux["rec_l1"] / aux["f_l1"]).mean()).backward()
    for p in (m.alpha_raw, m.lam):
        g = p.grad
        assert g is not None and torch.isfinite(torch.view_as_real(g) if g.is_complex() else g).all()
        assert g.abs().max() > 0
    assert np.all(np.isfinite(modes.detach().numpy()))


def test_param_roundtrip_and_shape_checks():
    T, K, N = 128, 4, 2
    alpha, lam = _random_params(K, N, 2 * T)
    m = BatchedUVMD(T, K, N, dtype=torch.float64).load_params(alpha, lam)
    P = m.export_params()
    np.testing.assert_allclose(P["alpha"], alpha, rtol=1e-12)
    np.testing.assert_allclose(P["lambda"], lam, rtol=1e-12)
    assert m.n_real_params() == K + 2 * N * 2 * T
    with pytest.raises(ValueError):
        BatchedUVMD(T, K, N + 1).load_params(alpha, lam)          # wrong number of layers
    with pytest.raises(ValueError):
        BatchedUVMD(T, K + 1, N).load_params(alpha, lam)          # wrong K
    with pytest.raises(ValueError):
        BatchedUVMD(T, K, N)(torch.zeros(2, T + 2))               # wrong window length
    np.testing.assert_allclose(BatchedUVMD(T, K, N).alpha().detach().numpy(), 2000.0)
