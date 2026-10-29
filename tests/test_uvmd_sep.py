"""Separation-aware loss: it must leave the upstream model and its reconstruction loss untouched."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from decomp import uvmd_magn, uvmd_sep  # noqa: E402

T, K = 128, 4


def _signal(seed=0, T=T):
    t = np.arange(T)
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * 6 * t / T) + 0.5 * np.sin(2 * np.pi * 20 * t / T + 1) + 0.05 * rng.standard_normal(T)).astype(np.float32)


def _model(layers=1, seed=0):
    torch.manual_seed(seed)
    return uvmd_magn.make_model(T, K, layers, "cpu", 2000.0)


@pytest.mark.parametrize("layers", [1, 2])
def test_forward_spectra_matches_upstream_forward(layers):
    m = _model(layers)
    with torch.no_grad():
        m.lambda_hat.data += 0.01 * (torch.randn_like(m.lambda_hat.real) + 1j * torch.randn_like(m.lambda_hat.real))
        x = torch.as_tensor(_signal())
        u, om, loss = m(x)
        U, omega_plus, f_hat, freqs, TT = uvmd_sep.forward_spectra(m, x)
        assert torch.allclose(m.postprocessing(U), u, atol=1e-5)
        assert torch.allclose(omega_plus, om, atol=1e-7)
        assert torch.allclose(torch.sum(torch.abs(torch.sum(U, dim=1) - f_hat)), loss, rtol=1e-5)


def test_recon_only_is_the_upstream_loss_up_to_a_constant():
    m = _model()
    x = torch.as_tensor(_signal())
    total, terms = uvmd_sep.loss_terms(m, x, uvmd_sep.VARIANTS["recon"])
    _, _, loss = m(x)
    f_hat = m.preprocessing(x)
    assert set(terms) == {"rec"}
    assert float(total * torch.sum(torch.abs(f_hat))) == pytest.approx(float(loss), rel=1e-5)
    g_new = torch.autograd.grad(total, m.Alpha)[0]
    g_up = torch.autograd.grad(loss, m.Alpha)[0]
    assert torch.allclose(g_new * torch.sum(torch.abs(f_hat)), g_up, rtol=1e-4, atol=1e-6)


def test_all_terms_finite_and_reach_both_parameters():
    m = _model(2)
    with torch.no_grad():
        m.lambda_hat.data += 0.01 * (torch.randn_like(m.lambda_hat.real) + 1j * torch.randn_like(m.lambda_hat.real))
    x = torch.as_tensor(_signal())
    total, terms = uvmd_sep.loss_terms(m, x, uvmd_sep.VARIANTS["full"])
    assert set(terms) == {"rec", "bw", "ortho", "lam", "alpha"}
    assert all(torch.isfinite(v) for v in terms.values())
    for name in ("bw", "ortho", "lam"):        # each separation term must produce a gradient on the parameters
        g = torch.autograd.grad(terms[name], [m.Alpha, m.lambda_hat], retain_graph=True, allow_unused=True)
        assert any(gi is not None and torch.any(gi != 0) for gi in g), name
    assert float(terms["alpha"]) == 0.0         # alpha = 2000 is above the floor


def test_alpha_floor_hinge_activates_below_the_floor():
    m = _model()
    with torch.no_grad():
        m.Alpha.data[0] = -20.0                 # softplus ~ 0
    _, terms = uvmd_sep.loss_terms(m, torch.as_tensor(_signal()), uvmd_sep.LossWeights(nu=1.0))
    assert float(terms["alpha"]) == pytest.approx(1.0 / K, rel=1e-3)


def test_bandwidth_term_is_larger_for_a_broadband_mode():
    w = uvmd_sep.LossWeights(beta=1.0)
    t = np.arange(T)
    tone = np.sin(2 * np.pi * 10 * t / T).astype(np.float32)
    noise = np.random.default_rng(1).standard_normal(T).astype(np.float32)
    m = _model()
    bw = lambda x: float(uvmd_sep.loss_terms(m, torch.as_tensor(x), w)[1]["bw"])  # noqa: E731
    assert bw(noise) > 2 * bw(tone)


def test_a_few_steps_of_training_reduce_the_composite_loss():
    X = np.stack([_signal(s) for s in range(6)])
    m = _model()
    w = uvmd_sep.VARIANTS["bw+ortho"]
    h = uvmd_sep.train_sep(m, X[:4], X[4:], w, epochs=4, lr=0.1, patience=10, log=lambda *_: None)
    assert h["train_loss"][-1] < h["train_loss"][0]
    assert set(h["terms"][0]) == {"rec", "bw", "ortho"}
