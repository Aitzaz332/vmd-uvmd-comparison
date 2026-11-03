"""Extract the modes of a trained UVMD (or of classic VMD) for CWRU windows and save them with their metadata.

Works on ANY upstream-format parameter file, so the same command gives the modes of the end-to-end UVMD and of a UVMD
trained on the reconstruction loss only, on identical windows:

    # modes of the end-to-end UVMD (written by run_cwru_e2e.py)
    python experiments/extract_uvmd_modes.py --params results/cwru_e2e/uvmd_e2e_params_N5_seed0.npz \
        --data ~/data/CWRU --subset test --out modes_e2e_seed0.npz
    # modes of a UVMD trained only on reconstruction (uvmd_params_N*.npz from run_cwru.py; K and N are read from the file)
    python experiments/extract_uvmd_modes.py --params results/cwru/uvmd_params_N2.npz --data ~/data/CWRU --out modes_recon.npz
    # classic VMD run to convergence
    python experiments/extract_uvmd_modes.py --vmd --K 4 --alpha 500 --data ~/data/CWRU --out modes_vmd.npz

The window settings (--T, --windows-per-file, --split) must match the run that trained the model; the defaults match
run_cwru_e2e.py. K and N come from the parameter file. File format: see e2e/export.py (modes (n, K, T), omega (n, K),
the input window, per-window metadata, and for UVMD the alpha / lambda that produced them).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from datasets.cwru import load_cwru, split_indices  # noqa: E402
from e2e.export import decompose_params, save_modes  # noqa: E402
from e2e.train import make_labels, vmd_modes  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--params", help="upstream-format UVMD parameter file (alpha, lambda)")
    src.add_argument("--vmd", action="store_true", help="classic VMD to convergence instead of a UVMD")
    p.add_argument("--K", type=int, default=4, help="modes (only with --vmd; UVMD files carry their own K)")
    p.add_argument("--alpha", type=float, default=500.0, help="VMD bandwidth penalty (only with --vmd)")
    p.add_argument("--data", required=True, help="folder with the CWRU .mat files")
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--windows-per-file", type=int, default=30)
    p.add_argument("--split", choices=["load", "time"], default="load")
    p.add_argument("--subset", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--labels", choices=["fault10", "fault4"], default="fault10")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    D = load_cwru(a.data, a.T, a.windows_per_file, True)
    tr, va, te = split_indices(D, a.split)
    idx = {"train": tr, "val": va, "test": te, "all": np.arange(len(D["x"]))}[a.subset]
    y, names = make_labels(D, a.labels)
    X = D["x"][idx]
    if a.vmd:
        modes, its, omega = vmd_modes(X, a.K, a.alpha, None, return_omega=True)
        params = None
        what = f"classic VMD (K={a.K}, alpha={a.alpha:g}, mean {its:.0f} iterations)"
    else:
        params = dict(np.load(a.params))
        K, (N, T2) = len(params["alpha"]), params["lambda"].shape
        if T2 != 2 * a.T:
            raise ValueError(f"{a.params} was trained for windows of {T2 // 2} samples, but --T is {a.T}")
        modes, omega = decompose_params(params, X, torch.device(a.device))
        what = f"UVMD (K={K}, N={N}) from {a.params}"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_modes(a.out, modes, omega, D, idx, y, names, a.subset, params)
    print(f"{what}: {len(idx)} '{a.subset}' windows -> modes {modes.shape} saved to {a.out}")
    return modes, omega


if __name__ == "__main__":
    main()
