"""End-to-end UVMD + 1D-CNN: shapes, gradient flow, training loop, labels, and a full smoke run on fake data."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from datasets.cwru import load_cwru  # noqa: E402
from decomp.uvmd_torch import BatchedUVMD  # noqa: E402
from e2e.cnn1d import Fault1DCNN  # noqa: E402
from e2e.train import E2E, fit, make_labels, predict, score, vmd_modes  # noqa: E402
from experiments import run_cwru_e2e  # noqa: E402
from tests.fake_cwru import write_fake_cwru  # noqa: E402


def _tone_task(n_per_class=40, T=128, seed=0):
    """3 classes = sinusoids at 3 frequencies + noise (separable); standardised windows."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X, y = [], []
    for c, f in enumerate([0.03, 0.09, 0.2]):
        for _ in range(n_per_class):
            x = np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + 0.3 * rng.standard_normal(T)
            X.append((x - x.mean()) / x.std())
            y.append(c)
    X, y = np.stack(X).astype(np.float32), np.array(y)
    p = rng.permutation(len(X))
    return X[p], y[p]


def test_cnn_shapes_and_same_architecture_for_every_input():
    for ch in (1, 4, 5):
        m = Fault1DCNN(ch, 10)
        assert m(torch.randn(3, ch, 256)).shape == (3, 10)
    # only the first conv depends on the number of input channels
    assert Fault1DCNN(4, 10).n_params() - Fault1DCNN(1, 10).n_params() == 3 * 32 * 15 + 3 * 2   # conv weights + BN


def test_e2e_loss_reaches_both_uvmd_parameters_and_cnn():
    T, K, N = 128, 4, 2
    X, y = _tone_task(T=T)
    m = E2E(BatchedUVMD(T, K, N), Fault1DCNN(K, 3))
    a0, l0 = m.uvmd.alpha_raw.detach().clone(), m.uvmd.lam.detach().clone()
    w0 = m.cnn.net[1].weight.detach().clone()
    opt = torch.optim.Adam([{"params": m.cnn.parameters(), "lr": 1e-3},
                            {"params": [m.uvmd.alpha_raw], "lr": 2.0}, {"params": [m.uvmd.lam], "lr": 0.05}])
    logits, rec = m(torch.as_tensor(X[:16]))
    loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[:16])) + 0.1 * rec.mean()
    loss.backward()
    opt.step()
    assert not torch.equal(a0, m.uvmd.alpha_raw) and not torch.equal(l0, m.uvmd.lam)
    assert not torch.equal(w0, m.cnn.net[1].weight)
    assert rec.shape == (16,) and 0 <= rec.min() and np.isfinite(rec.detach().numpy()).all()


def test_labels(tmp_path):
    write_fake_cwru(tmp_path, seconds=1.2)
    D = load_cwru(tmp_path, T=256, windows_per_file=2)
    y10, n10 = make_labels(D, "fault10")
    y4, n4 = make_labels(D, "fault4")
    assert len(n10) == 10 and len(set(y10)) == 10 and len(n4) == 4 and len(set(y4)) == 4
    assert n10[0] == "normal" and (y10 == 0).sum() == (D["cls"] == "normal").sum()


@pytest.mark.parametrize("e2e", [False, True])
def test_fit_learns_and_restores_best_weights(e2e):
    T, K, N = 128, 3, 2
    X, y = _tone_task(T=T)
    tr, va = slice(0, 90), slice(90, 120)
    if e2e:
        model, Xtr, Xva = E2E(BatchedUVMD(T, K, N), Fault1DCNN(K, 3)), X[tr], X[va]
    else:
        F_ = vmd_modes(X, K, n_iter=N)[0]
        model, Xtr, Xva = Fault1DCNN(K, 3), F_[tr], F_[va]
    h = fit(model, Xtr, y[tr], Xva, y[va], n_classes=3, device=torch.device("cpu"), seed=0, epochs=25, patience=25,
            batch=16, log=lambda s: None)
    logits, _ = predict(model, Xva, torch.device("cpu"))
    sc = score(y[va], logits, 3)
    assert h["best_val_acc"] > 0.9
    assert sc["acc"] == pytest.approx(h["best_val_acc"])          # model really is at its best-validation weights
    assert sc["ce"] == pytest.approx(h["best_val_ce"], rel=1e-4)
    if e2e:
        assert h["alpha"][-1] != h["alpha"][0] and h["train_rec"][0] is not None


def test_frozen_uvmd_is_not_updated_by_fit():
    T, K, N = 128, 3, 2
    X, y = _tone_task(T=T)
    m = E2E(BatchedUVMD(T, K, N), Fault1DCNN(K, 3))
    a0, l0 = m.uvmd.alpha_raw.detach().clone(), m.uvmd.lam.detach().clone()
    fit(m, X[:90], y[:90], X[90:], y[90:], n_classes=3, device=torch.device("cpu"), seed=0, epochs=2, batch=16,
        lr_alpha=0.0, lr_lam=0.0, log=lambda s: None)
    assert torch.equal(a0, m.uvmd.alpha_raw) and torch.equal(l0, m.uvmd.lam)


def test_full_run_on_fake_cwru(tmp_path):
    data, out = tmp_path / "cwru", tmp_path / "out"
    write_fake_cwru(data, seconds=1.2)
    base = ["--data", str(data), "--T", "256", "--quick", "--layers", "2", "--out", str(out), "--epochs", "4",
            "--early-epochs", "1"]
    runs = run_cwru_e2e.main(base)
    assert set(runs.pipeline) == set(run_cwru_e2e.ALL)
    assert runs.test_acc.between(0, 1).all() and np.isfinite(runs.test_ce).all()
    pred = pd.read_csv(out / "predictions.csv")
    assert len(pred) == len(run_cwru_e2e.ALL) * (pred.window.nunique())
    s = json.load(open(out / "summary.json"))
    assert s["summary"]["uvmd_e2e"]["uvmd_params"] == 4 + 2 * 2 * 512 and s["summary"]["vmd"]["uvmd_params"] == 0
    early = out / "early_uvmd_params_N2_K4.npz"
    e2e_p = out / "uvmd_e2e_params_N2_seed0.npz"
    assert early.exists() and e2e_p.exists()
    P = dict(np.load(e2e_p))                                         # upstream export format
    assert P["alpha"].shape == (4,) and P["lambda"].shape == (2, 512) and P["omega0"].shape == (4,)
    # reuse the saved early parameters explicitly, only the cheap pipelines
    out2 = tmp_path / "out2"
    run_cwru_e2e.main(base[:-6] + ["--out", str(out2), "--epochs", "2", "--pipelines", "uvmd_early", "uvmd_e2e",
                                   "--early-params", str(early), "--e2e-init", "early"])
    assert (out2 / "runs.csv").exists() and not list(out2.glob("early_uvmd_params*"))
    with pytest.raises(ValueError):                                  # parameters of another configuration are refused
        run_cwru_e2e.main(["--data", str(data), "--T", "256", "--quick", "--layers", "3", "--out", str(tmp_path / "o3"),
                           "--pipelines", "uvmd_early", "--early-params", str(early)])
