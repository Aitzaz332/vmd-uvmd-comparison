"""Does a UVMD mode at time t depend on samples AFTER t?   (Diagnostic only; it changes nothing in the replication.)

    python replication/scripts/leak_check.py --his replication/magn/data/sd/2019/his_raw.npz \
        --model replication/results/sd_N1/model.pth --layers 1

It loads the trained UVMD from the unchanged upstream uvmd.py (class source only, the script part is skipped), takes a
sensor's year, nudges ONE later sample x[t0+h] by delta and reports max_k |change of mode_k at t0| / delta.
A causal decomposition gives exactly 0 for every h > 0. The forecasting task asks ASTGCN to predict x[t+1..t+12]
from modes at t-11..t, so a non-zero value at h = 1..12 means the inputs contain information about the targets.
"""
import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_classes(uvmd_path: Path):
    src = uvmd_path.read_text()
    start = src.index("class VMD_innerloop")
    end = src.index("device = torch.device")
    ns = dict(torch=torch, nn=nn, F=F, np=np, logging=logging)
    exec(compile(src[start:end], str(uvmd_path), "exec"), ns)
    return ns["Unfolding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--his", required=True, help="original (not unfolded) his.npz")
    ap.add_argument("--model", required=True, help="model.pth written by uvmd.py")
    ap.add_argument("--uvmd", default=str(Path(__file__).resolve().parents[1] / "magn" / "uvmd.py"))
    ap.add_argument("--K", type=int, default=13)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--nodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--delta", type=float, default=1e-2)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 12, 48, 96, 1000])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="optional json output path")
    a = ap.parse_args()

    dev = torch.device(a.device)
    d = np.load(a.his)["data"]                       # (T, N, F), channel 0 = standardised flow
    T = d.shape[0] - (d.shape[0] % 2)
    Unfolding = load_classes(Path(a.uvmd))
    model = Unfolding(dev, K=a.K, layers=a.layers, freqs_len=2 * T).to(dev)
    model.load_state_dict(torch.load(a.model, map_location=dev))
    model.eval()

    result = {}
    with torch.no_grad():
        for n in a.nodes:
            x = torch.tensor(d[:T, n, 0], dtype=torch.float32, device=dev)
            base, _, _ = model(x)
            t0 = T // 2
            rec = float(torch.linalg.norm(base.sum(0) - x) / torch.linalg.norm(x))
            row = {"recon_rel_err": rec}
            for h in a.horizons:
                if t0 + h >= T:
                    continue
                x2 = x.clone(); x2[t0 + h] += a.delta
                pert, _, _ = model(x2)
                row[f"h={h}"] = float((pert[:, t0] - base[:, t0]).abs().max() / a.delta)
            result[f"node{n}"] = row
            print(f"node {n}: " + "  ".join(f"{k}={v:.3g}" for k, v in row.items()))
    print("\nAll zeros for h > 0 would mean causal. Non-zero values at h = 1..12 mean mode values at t carry information "
          "about the samples the forecaster is asked to predict.")
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
