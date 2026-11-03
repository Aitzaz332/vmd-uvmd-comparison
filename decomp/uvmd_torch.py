"""Batched, differentiable UVMD in PyTorch, for training UVMD end to end with a downstream network.

Why a second implementation? The upstream ``Unfolding`` (third_party/MAGN/uvmd.py, wrapped by
``decomp/uvmd_magn.py``) handles ONE signal per call and rebuilds its layer modules on every forward pass, which
is too slow to put in front of a CNN. This module computes exactly the same arithmetic for a whole batch
``x`` of shape (B, T) at once:

    one layer = one ADMM iteration (no dual ascent; the multiplier of each layer is a learnable vector)
    u_k  <- (f - sum_{i != k} u_i - lambda_n / 2) / (1 + softplus(alpha_raw_k) * (w - omega_k)^2)
    omega_k <- power-weighted mean frequency of u_k on the positive half spectrum (mode 0 = DC, omega_0 = 0)

Trainable: ``alpha_raw`` (K,) real and ``lam`` (N, 2T) complex, the same two parameter sets as upstream.
Parameter files use the upstream export format, so ``decomp/uvmd_magn.export_params`` output loads here
(``load_params``) and ``export_params`` output can be used by ``experiments/omega_shift_all.py`` and friends.

Equivalence is enforced by ``tests/test_uvmd_torch.py``: float64 output equals ``decomp/vmd_numpy.py`` (itself
tested against vmdpy) and float32 output equals the upstream ``Unfolding`` with random parameters.

Differences from upstream, none of which change the result:
  * ``lam`` has N rows (upstream allocates N+1 and never uses the last);
  * ``eps`` in the centre-frequency division (1e-12; below float32 resolution of any non-zero energy sum, it only
    prevents 0/0 when a mode has no energy);
  * the forward pass also returns the per-window spectral L1 residual and the L1 norm of the input spectrum, so a
    scale-free reconstruction term can be built (see ``e2e/train.py``).

As upstream, every step is a global FFT filter: the decomposition is non-causal (fine for classifying a whole
window offline, not for forecasting the samples that follow).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(a):
    """Inverse of softplus, in float64 (a > 0)."""
    a = torch.as_tensor(a, dtype=torch.float64).clamp_min(1e-12)
    return torch.where(a > 30, a, torch.log(torch.expm1(a)))


class BatchedUVMD(nn.Module):
    def __init__(self, signal_len, K=4, n_layers=3, alpha_init=2000.0, dtype=torch.float32, eps=1e-12):
        super().__init__()
        if signal_len % 2:
            raise ValueError("use an even signal length (the upstream code drops the last sample if odd)")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        self.T, self.T2 = signal_len, 2 * signal_len            # T2 = length of the mirrored signal
        self.K, self.n_layers, self.eps, self.rdtype = K, n_layers, eps, dtype
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.alpha_raw = nn.Parameter(torch.full((K,), float(alpha_init), dtype=dtype))   # softplus(2000) = 2000
        self.lam = nn.Parameter(torch.zeros(n_layers, self.T2, dtype=cdtype))
        t = torch.arange(1, self.T2 + 1, dtype=dtype) / self.T2
        self.register_buffer("freqs", t - 0.5 - 1.0 / self.T2, persistent=False)         # normalised frequency axis
        self.register_buffer("keep", (torch.arange(self.T2) >= self.T2 // 2).to(cdtype), persistent=False)
        self.register_buffer("omega0", (0.5 / K) * torch.arange(K, dtype=dtype), persistent=False)

    # ---------------------------------------------------------------------------------- helpers
    def alpha(self):
        """Bandwidth penalty per mode, softplus(alpha_raw): shape (K,)."""
        return F.softplus(self.alpha_raw)

    def _preprocess(self, x):                                   # x: (B, T) -> one-sided spectrum (B, T2)
        lt = self.T // 2
        fm = torch.cat([x[:, :lt].flip(-1), x, x[:, -lt:].flip(-1)], dim=-1)          # mirror both ends
        f_hat = torch.fft.fftshift(torch.fft.fft(fm, dim=-1), dim=-1)
        return f_hat * self.keep

    def _postprocess(self, u_hat_plus):                         # (B, T2, K) -> modes (B, K, T)
        pos = u_hat_plus[:, self.T2 // 2:, :]
        full = torch.cat([torch.conj(pos[:, -1:, :]), torch.conj(pos.flip(1)), pos[:, 1:, :]], dim=1)
        u = torch.fft.ifft(torch.fft.ifftshift(full, dim=1), dim=1).real
        return u[:, self.T2 // 4: 3 * self.T2 // 4, :].permute(0, 2, 1)               # drop the mirrored parts

    # ---------------------------------------------------------------------------------- forward
    def forward(self, x):
        """x: (B, T). Returns modes (B, K, T), final centre frequencies (B, K) in cycles/sample, and
        aux = {"rec_l1": (B,), "f_l1": (B,)}: the upstream training loss per window,
        sum |sum_k u_hat_k - f_hat_plus|, and sum |f_hat_plus|."""
        if x.dim() != 2 or x.shape[1] != self.T:
            raise ValueError(f"expected (B, {self.T}), got {tuple(x.shape)}")
        x = x.to(self.rdtype)
        B, half = x.shape[0], self.T2 // 2
        f_hat_plus = self._preprocess(x)
        pos_freqs = self.freqs[half:]
        alpha = self.alpha()

        u_prev = torch.zeros(B, self.T2, self.K, dtype=f_hat_plus.dtype, device=x.device)
        omega_prev = self.omega0.unsqueeze(0).expand(B, self.K)
        sum_uk = torch.zeros(B, self.T2, dtype=f_hat_plus.dtype, device=x.device)
        for n in range(self.n_layers):                          # one layer = one ADMM iteration
            lam_n = self.lam[n]
            new_modes, new_omegas = [], []
            for k in range(self.K):
                newest = u_prev[..., self.K - 1] if k == 0 else new_modes[k - 1]
                sum_uk = newest + sum_uk - u_prev[..., k]       # sum of all other modes (Gauss-Seidel order)
                denom = 1.0 + alpha[k] * (self.freqs - omega_prev[:, k:k + 1]) ** 2
                u_k = (f_hat_plus - sum_uk - lam_n / 2) / denom
                new_modes.append(u_k)
                if k == 0:
                    new_omegas.append(omega_prev[:, 0])         # DC mode: omega stays 0
                else:
                    w = u_k[:, half:].abs() ** 2
                    new_omegas.append((pos_freqs * w).sum(-1) / (w.sum(-1) + self.eps))
            u_prev = torch.stack(new_modes, dim=-1)
            omega_prev = torch.stack(new_omegas, dim=-1)
        aux = {"rec_l1": (u_prev.sum(-1) - f_hat_plus).abs().sum(-1), "f_l1": f_hat_plus.abs().sum(-1)}
        return self._postprocess(u_prev), omega_prev, aux

    # ---------------------------------------------------------------------------------- parameters I/O
    @torch.no_grad()
    def load_params(self, alpha, lam):
        """alpha: (K,) = softplus(alpha_raw); lam: (N, 2T) complex. Same content as upstream ``export_params``."""
        alpha = torch.as_tensor(np.asarray(alpha, dtype=np.float64))
        lam = torch.as_tensor(np.asarray(lam))
        if tuple(alpha.shape) != (self.K,):
            raise ValueError(f"alpha has shape {tuple(alpha.shape)}, this model has K={self.K}")
        if tuple(lam.shape) != (self.n_layers, self.T2):
            raise ValueError(f"lambda has shape {tuple(lam.shape)}, this model needs ({self.n_layers}, {self.T2}) "
                             f"(N={self.n_layers} layers, T={self.T})")
        self.alpha_raw.copy_(inv_softplus(alpha).to(self.rdtype))
        self.lam.copy_(lam.to(self.lam.dtype))
        return self

    @torch.no_grad()
    def export_params(self):
        """Upstream export format: alpha (K,) = softplus(alpha_raw), lambda (N, 2T) complex, omega0 (K,)."""
        return {"alpha": self.alpha().cpu().numpy().astype(np.float64),
                "lambda": self.lam.detach().cpu().numpy().astype(np.complex128),
                "omega0": self.omega0.cpu().numpy().astype(np.float64),
                "alpha_raw": self.alpha_raw.detach().cpu().numpy().astype(np.float64)}

    def n_real_params(self):
        return int(self.K + 2 * self.n_layers * self.T2)
