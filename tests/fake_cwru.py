"""Synthetic stand-in for the 40-file CWRU subset, for tests and smoke runs (NOT real bearing data).

Each file follows datasets.cwru.CATALOG (class, fault size, load, native sampling rate) and is written with the
variable names of the real .mat files. A fault is a train of decaying resonance bursts at the fault frequency of
that class and speed; class decides resonance frequency and burst rate, size decides burst strength, load decides
speed. Normal files are noise at 48 kHz (the loader resamples them to 12 kHz, like the real ones)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from datasets.cwru import CATALOG, FAULT_MULT, NOMINAL_RPM

RESONANCE = {"inner": 3200.0, "ball": 4200.0, "outer": 2300.0}


def write_fake_cwru(folder, seconds=2.2, seed=0):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for num, (cls, size, load, fs) in CATALOG.items():
        n = int(seconds * fs)
        t = np.arange(n) / fs
        x = 0.3 * rng.standard_normal(n)
        rpm = NOMINAL_RPM[load] + rng.uniform(-3, 3)
        if cls != "normal":
            f_fault = FAULT_MULT[cls] * rpm / 60.0
            amp = 4.0 * (size / 0.007)
            for t0 in np.arange(rng.uniform(0, 1 / f_fault), t[-1], 1 / f_fault):
                m = t >= t0
                x[m] += amp * np.exp(-900 * (t[m] - t0)) * np.sin(2 * np.pi * RESONANCE[cls] * (t[m] - t0))
        savemat(folder / f"{num}.mat", {f"X{num:03d}_DE_time": x[:, None], f"X{num:03d}RPM": np.array([[rpm]])})
    return folder
