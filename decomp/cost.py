"""Computational-cost model and timing helpers.

Analytic real-FLOP count for one signal of length T (mirrored length T2 = 2T), K modes.
Counts follow the update equations as implemented (vmdpy / upstream UVMD), per spectral bin:

  per mode, per ADMM iteration / UVMD layer
    accumulator   sum_uk = newest + sum_uk - u_k        2 complex add          4
    numerator     f - sum_uk - lambda/2                 2 complex sub + scale  6
    denominator   1 + a (w - w_k)^2                     sub, sq, mul, add      4
    division      complex / real                                              2
    omega (k>0)   |u|^2, dot, sum on T2/2 bins          (3 + 2 + 1)/2          3
    -> 19 FLOP/bin/mode
  VMD only, per iteration
    dual ascent   lambda += tau (sum_k u_k - f)         (2K + 4) per bin (skipped when tau = 0)
    convergence   ||u^{n+1}-u^n||^2 for every mode      7K per bin
  once per signal
    FFT of mirrored signal (5 N log2 N, N = T2) and K inverse FFTs for the modes.
UVMD training step ~ 3x forward (backward through the same graph ~ 2x forward).
These are estimates for comparison, not measurements. Wall-clock times are measured separately.
"""
from __future__ import annotations

import time

import numpy as np


def fft_flops(n):
    return 5.0 * n * np.log2(n)


def flops_vmd(T, K, n_iter, tau=0.0):
    T2 = 2 * T
    per_iter = 19 * K * T2 + 7 * K * T2 + ((2 * K + 4) * T2 if tau else 0)
    return fft_flops(T2) + n_iter * per_iter + K * fft_flops(T2)


def flops_uvmd_forward(T, K, layers):
    T2 = 2 * T
    return fft_flops(T2) + layers * 19 * K * T2 + K * fft_flops(T2)


def flops_uvmd_train(T, K, layers, n_signals, epochs):
    return 3.0 * flops_uvmd_forward(T, K, layers) * n_signals * epochs


def time_per_call(fn, items, warmup=2, repeats=1):
    """Median and mean wall-clock seconds of fn(item) over items (after warm-up calls)."""
    for it in items[:warmup]:
        fn(it)
    ts = []
    for _ in range(repeats):
        for it in items:
            t0 = time.perf_counter()
            fn(it)
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"median_s": float(np.median(ts)), "mean_s": float(ts.mean()), "p90_s": float(np.quantile(ts, 0.9)),
            "n_calls": int(len(ts))}
