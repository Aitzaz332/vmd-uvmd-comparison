"""UVMD trained with a separation-aware loss (branch ``separation-loss``).

The model is the UNCHANGED upstream ``Unfolding`` (see ``decomp/uvmd_magn.py``); only the training loss differs.
Upstream trains on reconstruction alone: ``sum |sum_k u_hat_k - f_hat_plus|``. Reconstruction error does not measure
separation (Spearman +0.52 against the true mode SNR on the synthetic data), and nothing in that loss touches
the centre frequencies, so here the loss is

    L = L_rec + beta * L_bw + gamma * L_ortho + mu * L_lam + nu * L_alpha

  L_rec    upstream reconstruction loss divided by sum|f_hat_plus|. Adam is invariant to a constant loss scale, so with
           beta = gamma = mu = nu = 0 this trains exactly like upstream (checked by a test).
  L_bw     VMD's own bandwidth objective, which the unfolded model drops from training. Energy-weighted spectral
           standard deviation of the modes around their centre frequency, divided by the Nyquist frequency:
           sum_k (E_k / E) * sqrt( sum_w (w - omega_k)^2 |u_hat_k(w)|^2 / E_k ) / 0.5.   omega_k is the model's own
           centre frequency, computed from u_hat_k, so this term is the first one to give omega a gradient.
           (Latent Mode Decomposition, arXiv 2505.17797, keeps the same term in its loss.)
  L_ortho  content shared between modes, sum_{i<j} |Re sum_w u_hat_i conj(u_hat_j)| / E (cf. the orthogonality index).
  L_lam    ||lambda_hat||^2 / E on the used part of the spectrum: stops lambda from absorbing the residual.
  L_alpha  hinge on alpha_k below alpha_floor: stops a mode turning into an all-pass residual (CWRU: one alpha -> 0).

E = sum |f_hat_plus|^2 over the positive half spectrum. All terms are scale-free, so one set of weights can be used
for every dataset.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict

import numpy as np

NYQUIST = 0.5     # cycles/sample


@dataclass(frozen=True)
class LossWeights:
    beta: float = 0.0       # bandwidth
    gamma: float = 0.0      # orthogonality
    mu: float = 0.0         # lambda norm
    nu: float = 0.0         # alpha floor
    alpha_floor: float = 200.0   # 10 % of the initial alpha = 2000

    def dict(self):
        return asdict(self)


# the variants of the study: name -> weights. "recon" is the upstream loss (control).
VARIANTS = {
    "recon": LossWeights(),
    "bw": LossWeights(beta=1.0),
    "bw+ortho": LossWeights(beta=1.0, gamma=1.0),
    "full": LossWeights(beta=1.0, gamma=1.0, mu=1.0, nu=1.0),
}


def forward_spectra(model, x):
    """Same computation as the upstream ``Unfolding.forward``, but returns the final one-sided spectra.

    Returns (u_hat_final (2T, K) complex, omega_plus (layers, K), f_hat_plus (2T,), freqs (2T,), T).
    ``model(x)`` of the unchanged upstream class returns the time-domain modes and the upstream loss; the loop below is its
    forward loop, copied so the spectra stay available (test_uvmd_sep verifies that both give identical results)."""
    f_hat_plus = model.preprocessing(x)
    u_hat_plus = model.u_hat_plus
    omega_plus = model.omega_plus.clone()
    sum_uk = 0
    n = 0
    for module in model.modules_list:
        n_u_hat_plus, omega, sum_uk = module(u_hat_plus[n + 1], u_hat_plus[n], omega_plus[n + 1], omega_plus[n], sum_uk)
        omega_plus[n + 1] = omega.clone()
        u_hat_plus = u_hat_plus.clone()
        u_hat_plus[n + 1] = n_u_hat_plus
        n += 1
    return u_hat_plus[model.layers - 1], omega_plus, f_hat_plus, model.modules_list[0].freqs, model.T


def loss_terms(model, x, w: LossWeights):
    """Dict of the loss terms (torch scalars, graph attached) and the total, for one signal."""
    import torch
    import torch.nn.functional as F
    U, omega_plus, f_hat, freqs, T = forward_spectra(model, x)
    h = T // 2
    Up, Fp, wf = U[h:T], f_hat[h:T], freqs[h:T]
    E = torch.sum(torch.abs(Fp) ** 2) + 1e-30
    terms = {"rec": torch.sum(torch.abs(torch.sum(U, dim=1) - f_hat)) / (torch.sum(torch.abs(f_hat)) + 1e-30)}
    P = torch.abs(Up) ** 2                                              # (T/2, K)
    if w.beta:
        e = P.sum(0)
        c = omega_plus[model.layers - 1]                                # (K,) centre frequencies of the final layer
        var = (((wf[:, None] - c[None, :]) ** 2) * P).sum(0) / (e + 1e-30)
        terms["bw"] = ((e / E) * torch.sqrt(var / NYQUIST ** 2 + 1e-12)).sum()
    if w.gamma:
        G = torch.real(Up.conj().T @ Up)                                # (K, K) inner products of the one-sided spectra
        terms["ortho"] = torch.triu(torch.abs(G), diagonal=1).sum() / E
    if w.mu:
        lam = model.lambda_hat[: model.layers - 1][:, h:T]
        terms["lam"] = (torch.abs(lam) ** 2).sum(1).mean() / E
    if w.nu:
        a = F.softplus(model.Alpha)
        terms["alpha"] = torch.relu(1.0 - a / w.alpha_floor).mean()
    total = terms["rec"] + w.beta * terms.get("bw", 0.0) + w.gamma * terms.get("ortho", 0.0) \
        + w.mu * terms.get("lam", 0.0) + w.nu * terms.get("alpha", 0.0)
    return total, terms


def _epoch(model, X, optimizer, device, w, train, rng):
    import torch
    idx = np.arange(len(X))
    if train:
        rng.shuffle(idx)
        model.train()
    else:
        model.eval()
    tot, parts = 0.0, {}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in idx:
            x = torch.as_tensor(X[i], dtype=torch.float32, device=device)
            loss, terms = loss_terms(model, x, w)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot += loss.item()
            for k, v in terms.items():
                parts[k] = parts.get(k, 0.0) + float(v)
    return tot / len(X), {k: v / len(X) for k, v in parts.items()}


def train_sep(model, X_train, X_val, w: LossWeights, epochs=100, lr=0.1, patience=10, device="cpu", seed=0,
              restore_best=False, log=print):
    """The upstream loop (Adam, batch size 1, shuffle, early stopping on the validation loss) with the loss above.
    The validation loss is the same composite loss, so early stopping follows what is being optimised."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, counter, best_state = float("inf"), 0, None
    hist = {"train_loss": [], "val_loss": [], "epoch_time_s": [], "terms": []}
    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr, parts = _epoch(model, X_train, optimizer, device, w, True, rng)
        hist["epoch_time_s"].append(time.perf_counter() - t0)
        va, _ = _epoch(model, X_val, None, device, w, False, None)
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va)
        hist["terms"].append(parts)
        log(f"  epoch {epoch + 1:>3}/{epochs}  train {tr:.5f}  val {va:.5f}  " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()))
        if va < best_val:
            best_val, counter = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
        if counter >= patience:
            log("  early stopping triggered")
            break
    hist["epochs_run"] = len(hist["train_loss"])
    hist["best_val_loss"] = best_val
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return hist
