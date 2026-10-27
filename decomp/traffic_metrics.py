"""Metrics for decomposing traffic-flow signals (no ground-truth modes exist).

Reconstruction (decomp.metrics.evaluate): relative error, SNR, spectral L1 (the UVMD training loss), energy
ratio, orthogonality index.

Spectral structure of the modes (this module). With P_k(f) = |FFT(mode k)|^2 and p_k = P_k / sum_f P_k:
  mean_bandwidth_cpd   energy-weighted std of each mode's spectrum around its centroid, in cycles/day.
                       Smaller = narrower-band modes (what VMD is meant to produce).
  mode_overlap         mean over mode pairs of sum_f min(p_i, p_j), in [0, 1]. 0 = modes occupy different
                       frequencies, 1 = identical spectra (mode mixing). Modes with < 0.01% of the energy are ignored.
  n_active_modes       modes carrying more than 1% of the total mode energy.
  <line>_ownership     for the daily line (1/96 cycles/sample), its 2nd harmonic and the weekly line (1/672):
                       max_k P_k(line) / sum_k P_k(line). 1 = a single mode owns the line, 1/K = split evenly
                       across all modes (mode mixing).
  <line>_capture       sum_k P_k(line) / P_input(line). Near 1 = the modes together keep the line's power;
                       above 1 = overlapping modes double-count it; below 1 = power was lost.
A line is the +-1.5 FFT bins around its frequency.
"""
from __future__ import annotations

import numpy as np

LINES = {"daily": 1 / 96, "daily2": 2 / 96, "weekly": 1 / 672}


def spectral_metrics(x, u, steps_per_day=96, tol_bins=1.5, lines=None):
    lines = LINES if lines is None else lines
    u = np.asarray(u, dtype=np.float64)
    T = u.shape[1]
    x = np.asarray(x, dtype=np.float64)[:T]
    freqs = np.fft.rfftfreq(T)
    P = np.abs(np.fft.rfft(u, axis=1)) ** 2
    Px = np.abs(np.fft.rfft(x)) ** 2
    E = P.sum(axis=1)
    share = E / E.sum()
    p = P / np.maximum(E[:, None], 1e-300)
    centroid = (p * freqs).sum(axis=1)
    bw = np.sqrt((p * (freqs[None] - centroid[:, None]) ** 2).sum(axis=1))
    m = {"mean_bandwidth_cpd": float((share * bw).sum() * steps_per_day),
         "n_active_modes": int((share > 0.01).sum())}
    act = np.where(share > 1e-4)[0]
    ov = [np.minimum(p[i], p[j]).sum() for a, i in enumerate(act) for j in act[a + 1:]]
    m["mode_overlap"] = float(np.mean(ov)) if ov else 0.0
    df = 1.0 / T
    for name, f0 in lines.items():
        band = np.abs(freqs - f0) <= tol_bins * df
        lp = P[:, band].sum(axis=1)
        tot, tx = lp.sum(), Px[band].sum()
        m[f"{name}_ownership"] = float(lp.max() / tot) if tot > 0 else float("nan")
        m[f"{name}_capture"] = float(tot / tx) if tx > 0 else float("nan")
        m[f"{name}_owner_mode"] = int(lp.argmax())
    return m


def win_rate(df, method_a, method_b, metric, lower_is_better=True):
    """Share of nodes where method_a beats method_b on ``metric`` (paired by node), ties count half."""
    a = df[df.method == method_a].set_index("node")[metric]
    b = df[df.method == method_b].set_index("node")[metric]
    j = a.index.intersection(b.index)
    if len(j) == 0:
        return float("nan")
    d = (b[j] - a[j]) if lower_is_better else (a[j] - b[j])
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(j))
