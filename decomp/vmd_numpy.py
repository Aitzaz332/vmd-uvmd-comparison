"""Classic VMD (Dragomiretskiy & Zosso 2014), ADMM solver, NumPy.

Same arithmetic as vmdpy (third_party/vmdpy/vmdpy.py, itself a port of Zosso's MATLAB code), with
three changes that matter for benchmarking:

1. Memory O(T*K) instead of O(Niter*T*K): vmdpy keeps every iterate (500 x 2T x K complex).
2. ``max_iter`` and ``tol`` are free, so VMD can be stopped after exactly n ADMM iterations. That is
   what one UVMD "layer" is, so VMD-n is the direct baseline for an n-layer UVMD.
3. The returned modes are the LAST iterate. vmdpy returns the second-to-last one
   (``u_hat_plus[Niter-1]`` with ``Niter = n``), which is harmless at convergence but returns all-zero
   modes for a 1-iteration run.

``tests/test_vmd_numpy.py`` checks this against vmdpy to machine precision.

Optional per-iteration multipliers (``lambda_schedule``) and per-mode ``alpha`` let the same routine
evaluate a trained UVMD in NumPy (``uvmd_forward_numpy``). In upstream UVMD the multiplier of each layer
is a learned parameter and there is no dual-ascent step.
"""
from __future__ import annotations

import numpy as np


def _prep(f):
    f = np.asarray(f, dtype=np.float64)
    if len(f) % 2:
        f = f[:-1]
    ltemp = len(f) // 2
    fMirr = np.concatenate([np.flip(f[:ltemp]), f, np.flip(f[-ltemp:])])
    T = len(fMirr)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1.0 / T
    f_hat = np.fft.fftshift(np.fft.fft(fMirr))
    f_hat_plus = f_hat.copy()
    f_hat_plus[: T // 2] = 0
    return f_hat_plus, freqs, T


def _post(u_hat_plus, T):
    K = u_hat_plus.shape[1]
    idxs = np.flip(np.arange(1, T // 2 + 1))
    u_hat = np.zeros((T, K), dtype=complex)
    u_hat[T // 2:T, :] = u_hat_plus[T // 2:T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[T // 2:T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])
    u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat, axes=0), axis=0)).T  # (K, T)
    return u[:, T // 4: 3 * T // 4]


def vmd(f, alpha=2000.0, tau=0.0, K=4, DC=True, init=1, tol=1e-7, max_iter=499,
        lambda_schedule=None, omega_init=None, rng=None, return_history=False):
    """Decompose 1-D signal ``f`` into K modes.

    Returns dict with ``u`` (K, len(f) even-truncated), ``omega`` (K,) final centre frequencies in
    cycles/sample, ``n_iter`` (ADMM iterations run), ``converged`` (bool), ``spec_l1`` (upstream UVMD
    loss: sum |sum_k u_hat_k - f_hat_plus| on the one-sided mirrored spectrum).
    ``tol < 0`` disables the convergence test (run exactly ``max_iter`` iterations).
    """
    f_hat_plus, freqs, T = _prep(f)
    Alpha = alpha * np.ones(K) if np.isscalar(alpha) else np.asarray(alpha, dtype=np.float64)
    if omega_init is not None:
        omega = np.asarray(omega_init, dtype=np.float64).copy()
    elif init == 1:
        omega = (0.5 / K) * np.arange(K)
    elif init == 2:
        rng = np.random.default_rng() if rng is None else rng
        fs = 1.0 / (T // 2)
        omega = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * rng.random(K)))
    else:
        omega = np.zeros(K)
    if DC:
        omega[0] = 0.0
    if lambda_schedule is not None:
        lambda_schedule = np.asarray(lambda_schedule)
        max_iter, tol = len(lambda_schedule), -1.0
    lam = np.zeros(T, dtype=complex)
    u_prev = np.zeros((T, K), dtype=complex)
    sum_uk = np.zeros(T, dtype=complex)
    half = T // 2
    pos = freqs[half:T]
    n, converged, hist = 0, False, []
    while n < max_iter:
        if lambda_schedule is not None:
            lam = lambda_schedule[n]
        u_new = np.empty_like(u_prev)
        omega_new = omega.copy()
        for k in range(K):
            newest = u_prev[:, K - 1] if k == 0 else u_new[:, k - 1]
            sum_uk = newest + sum_uk - u_prev[:, k]
            u_new[:, k] = (f_hat_plus - sum_uk - lam / 2) / (1.0 + Alpha[k] * (freqs - omega[k]) ** 2)
            if k > 0 or not DC:
                w = np.abs(u_new[half:T, k]) ** 2
                omega_new[k] = np.dot(pos, w) / np.sum(w)
        if lambda_schedule is None:
            lam = lam + tau * (u_new.sum(axis=1) - f_hat_plus)
        n += 1
        d = u_new - u_prev
        uDiff = np.abs(np.spacing(1) + (1.0 / T) * np.sum(d * np.conj(d)))
        u_prev, omega = u_new, omega_new
        if return_history:
            hist.append(uDiff)
        if tol >= 0 and uDiff <= tol:
            converged = True
            break
    out = {
        "u": _post(u_prev, T),
        "omega": omega,
        "n_iter": n,
        "converged": converged,
        "spec_l1": float(np.sum(np.abs(u_prev.sum(axis=1) - f_hat_plus))),
    }
    if return_history:
        out["uDiff"] = np.array(hist)
    return out


def uvmd_forward_numpy(f, alpha_k, lambda_layers, K):
    """NumPy replica of the upstream UVMD forward pass with given (learned) parameters.

    alpha_k: (K,) = softplus(Alpha); lambda_layers: (N, 2T) complex. Mode 0 is the DC mode, omega
    starts uniform (0.5/K * k), exactly as ``Unfolding``.
    """
    return vmd(f, alpha=alpha_k, K=K, DC=True, init=1, lambda_schedule=lambda_layers)
