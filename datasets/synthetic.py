"""Dataset 1: synthetic signals made of known modes.

Each signal is the sum of four ground-truth modes plus white Gaussian noise. Every signal draws its own
amplitudes, phases and frequencies from the ranges below, so a model trained on some signals (UVMD)
is tested on unseen ones. Frequencies are in cycles/sample, the same unit VMD uses for omega.

    mode 0  trend      slow sine pair, f in [0.001, 0.004]        (DC mode of VMD/UVMD)
    mode 1  tone       A cos(2 pi f t + phi),   f in [0.040, 0.050]
    mode 2  AM tone    (1 + m cos(2 pi fm t)) A cos(2 pi f t + phi),  f in [0.095, 0.110]
    mode 3  chirp      A cos(2 pi (f t + c t^2 / 2T) + phi),  f in [0.220, 0.240], sweep c in [0, 0.02]

The bands are at least ~2 bandwidths of the alpha=2000 Wiener filter (1/sqrt(2000) ~ 0.022) apart, so a
well-tuned VMD with K=4 can separate them. That makes the truth recoverable and the comparison fair.
"""
from __future__ import annotations

import numpy as np

MODE_NAMES = ["trend", "tone", "AM tone", "chirp"]
CENTRE_RANGES = [(0.001, 0.004), (0.040, 0.050), (0.095, 0.110), (0.220, 0.240)]


def make_synthetic(n_signals=200, T=1024, noise_std=0.1, seed=0):
    """Returns dict: x (n,T) noisy input, clean (n,T), modes (n,4,T), centre_freq (n,4), meta."""
    if T % 2:
        raise ValueError("T must be even")
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    modes = np.zeros((n_signals, 4, T))
    fc = np.zeros((n_signals, 4))
    for i in range(n_signals):
        # mode 0: trend (two very slow sines)
        f0 = rng.uniform(0.001, 0.004)
        modes[i, 0] = rng.uniform(0.8, 1.5) * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)) \
            + 0.3 * np.sin(2 * np.pi * 0.5 * f0 * t + rng.uniform(0, 2 * np.pi))
        fc[i, 0] = f0
        # mode 1: pure tone
        f1 = rng.uniform(0.040, 0.050)
        modes[i, 1] = rng.uniform(0.7, 1.2) * np.cos(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi))
        fc[i, 1] = f1
        # mode 2: amplitude-modulated tone
        f2, fm, m = rng.uniform(0.095, 0.110), rng.uniform(0.002, 0.005), rng.uniform(0.2, 0.5)
        modes[i, 2] = rng.uniform(0.5, 1.0) * (1 + m * np.cos(2 * np.pi * fm * t)) \
            * np.cos(2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi))
        fc[i, 2] = f2
        # mode 3: linear chirp
        f3, c = rng.uniform(0.220, 0.240), rng.uniform(0.0, 0.02)
        modes[i, 3] = rng.uniform(0.3, 0.7) * np.cos(2 * np.pi * (f3 * t + 0.5 * c * t ** 2 / T)
                                                      + rng.uniform(0, 2 * np.pi))
        fc[i, 3] = f3 + 0.5 * c
    clean = modes.sum(axis=1)
    noise = noise_std * rng.standard_normal((n_signals, T))
    return {
        "x": clean + noise,
        "clean": clean,
        "modes": modes,
        "centre_freq": fc,
        "meta": {"name": "synthetic", "n_signals": n_signals, "T": T, "noise_std": noise_std,
                 "seed": seed, "mode_names": MODE_NAMES, "n_true_modes": 4},
    }
