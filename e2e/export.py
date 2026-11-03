"""Extract decomposition modes and save them with everything needed to compare them elsewhere.

File format (``np.savez_compressed``), n = number of windows, K modes, T samples:
    modes (n, K, T) float32   the modes; mode k of every UVMD / VMD starts from the same ordered frequency grid
    omega (n, K)   float32    final centre frequency of every mode, cycles/sample (x 12000 = Hz for CWRU)
    signal (n, T)  float32    the standardised input window that was decomposed (sum of modes ~ signal)
    window, recording, cls, size, load, rpm, start   per-window metadata of datasets.cwru.load_cwru
                                                 (recording = the CWRU file number, e.g. 105 for 105.mat)
    label (n,) int64, class_names                class index used by the CNN (fault10/fault4)
    subset (str)                                 train / val / test / all
    alpha (K,), lambda (N, 2T) complex           the UVMD parameters that produced the modes (UVMD files only)
"""
from __future__ import annotations

import numpy as np
import torch

from datasets.cwru import FS
from decomp.uvmd_torch import BatchedUVMD


@torch.no_grad()
def decompose_with(uvmd: BatchedUVMD, X, device, batch=64):
    """Modes (n, K, T) float32 and final centre frequencies (n, K) float32 of a BatchedUVMD."""
    uvmd.to(device).eval()
    U, OM = [], []
    for i in range(0, len(X), batch):
        u, om, _ = uvmd(torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device))
        U.append(u.cpu())
        OM.append(om.cpu())
    return torch.cat(U).numpy().astype(np.float32), torch.cat(OM).numpy().astype(np.float32)


def decompose_params(P, X, device, batch=64):
    """Same, from an upstream-format parameter dict {alpha (K,), lambda (N, 2T)} (e.g. uvmd_params_N*.npz)."""
    K, (N, T2) = len(P["alpha"]), P["lambda"].shape
    uv = BatchedUVMD(T2 // 2, K, N).load_params(P["alpha"], P["lambda"])
    return decompose_with(uv, X, device, batch)


def save_modes(path, modes, omega, D, idx, labels=None, class_names=None, subset="test", params=None):
    """Write the modes of windows ``idx`` of dataset ``D`` (output of load_cwru) with their metadata."""
    idx = np.asarray(idx)
    d = dict(modes=np.asarray(modes, dtype=np.float32), omega=np.asarray(omega, dtype=np.float32),
             signal=D["x"][idx].astype(np.float32), window=idx, recording=D["file"][idx], cls=D["cls"][idx],
             size=D["size"][idx], load=D["load"][idx], rpm=D["rpm"][idx], start=D["start"][idx],
             subset=np.array(subset), fs=np.array(FS))
    if labels is not None:
        d["label"] = np.asarray(labels)[idx]
        d["class_names"] = np.array(class_names)
    if params is not None:
        d["alpha"] = np.asarray(params["alpha"], dtype=np.float64)
        d["lambda"] = np.asarray(params["lambda"], dtype=np.complex128)
    np.savez_compressed(path, **d)
