"""Time the UVMD decomposition of every sensor (the paper's 'decomposition time', 98.63 s for all 6902 sensors).

    python replication/scripts/decomp_timing.py --his replication/magn/data/sd/2019/his_raw.npz \
        --model replication/results/sd_N1/model.pth --layers 1 --out replication/results/sd_N1/decomp_timing.json

It runs the same loop as the last block of upstream uvmd.py (model(signal) per sensor, no_grad, eval) with the trained
weights and reports seconds in total and per sensor. The paper's number is for GBA+GLA+SD together on an RTX 3080 Ti,
so compare per-sensor time and extrapolate with --extrapolate-to 6902.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from leak_check import load_classes, load_trained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--his", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--uvmd", default=str(Path(__file__).resolve().parents[1] / "magn" / "uvmd.py"))
    ap.add_argument("--K", type=int, default=13)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-nodes", type=int, default=0, help="0 = all sensors")
    ap.add_argument("--extrapolate-to", type=int, default=6902)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device(a.device)
    d = torch.tensor(np.load(a.his)["data"][..., 0], dtype=torch.float32)   # (T, N)
    T, N = d.shape
    N = min(N, a.max_nodes) if a.max_nodes else N
    model = load_classes(Path(a.uvmd))(dev, K=a.K, layers=a.layers, freqs_len=2 * T).to(dev)
    load_trained(model, a.model, dev)
    model.eval()
    with torch.no_grad():
        model(d[:, 0].to(dev))                                  # warm-up
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N):
            model(d[:, i].to(dev))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    res = {"device": str(dev), "sensors": N, "T": T, "K": a.K, "layers": a.layers, "seconds_total": dt,
           "seconds_per_sensor": dt / N, f"seconds_extrapolated_{a.extrapolate_to}_sensors": dt / N * a.extrapolate_to,
           "paper_seconds_6902_sensors": 98.63}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
