"""How far do the centre frequencies (omega_k) move?  VMD-n, converged VMD, and trained UVMD-N.

Question: with only 1-2 layers, does UVMD let the mode centre frequencies adapt, or do they stay near the
uniform starting grid (omega_k = 0.5/K * k cycles/sample)?  Reads the trained parameters that
``run_traffic.py`` saved (``uvmd_params_N*.npz``), rebuilds the models (no retraining), decomposes the same
test sensors and reports the shift  omega_final - omega_init  in cycles per day (cpd).

    python experiments/omega_shift.py --data ~/data/LargeST/sd_his_2019.h5 --region sd \
        --params-dir results/traffic_sd_bounded --device cuda:0

omega_k of the upstream UVMD is not a learned parameter: it is recomputed inside every layer as the
power-weighted mean frequency of mode k (mode 0 is the DC mode and never moves). UVMD can only change it
indirectly, through the learned alpha_k and lambda.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.traffic import STEPS_PER_DAY, load_traffic, split_nodes  # noqa: E402
from decomp import uvmd_magn  # noqa: E402
from decomp.vmd_numpy import vmd  # noqa: E402


def inv_softplus(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 20, a, np.log(np.expm1(np.minimum(a, 20))))


def rebuild(params_file, T, K, layers, device):
    import torch
    P = np.load(params_file)
    model = uvmd_magn.make_model(T, K, layers, device)
    with torch.no_grad():
        model.Alpha.data = torch.tensor(inv_softplus(P["alpha"]), dtype=torch.float32, device=device)
        lam = model.lambda_hat.data.clone()
        lam[:layers] = torch.tensor(P["lambda"][:layers], dtype=torch.cfloat, device=device)
        model.lambda_hat.data = lam
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--region", default="sd")
    p.add_argument("--params-dir", required=True, help="results folder holding uvmd_params_N*.npz")
    p.add_argument("--K", type=int, default=13)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--vmd-converged-nodes", type=int, default=10)
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    D = load_traffic(a.data, a.length)
    X = D["x"]
    T = X.shape[1]
    _, _, te = split_nodes(X.shape[0])
    te = te[:a.n_test]
    cpd = STEPS_PER_DAY
    out = Path(a.params_dir)
    rows = {}     # method -> (n_nodes, K) final omega in cpd
    rel = {}

    def rel_err(x, u):
        return float(np.linalg.norm(x - u.sum(0)) / np.linalg.norm(x))

    for N in a.layers:                                # VMD with N iterations
        om, re = [], []
        for i in te:
            r = vmd(X[i], alpha=2000.0, tau=0.0, K=a.K, DC=True, tol=-1, max_iter=N)
            om.append(r["omega"] * cpd); re.append(rel_err(X[i][:r["u"].shape[1]], r["u"]))
        rows[f"VMD-{N}"], rel[f"VMD-{N}"] = np.array(om), np.mean(re)
    om, re = [], []
    for i in te[:a.vmd_converged_nodes]:              # converged VMD (the slow one)
        r = vmd(X[i], alpha=2000.0, tau=0.0, K=a.K, DC=True, tol=1e-7, max_iter=499)
        om.append(r["omega"] * cpd); re.append(rel_err(X[i][:r["u"].shape[1]], r["u"]))
    rows["VMD converged"], rel["VMD converged"] = np.array(om), np.mean(re)

    for N in a.layers:                                # trained UVMD-N rebuilt from the saved parameters
        f = out / f"uvmd_params_N{N}.npz"
        if not f.exists():
            print(f"missing {f}, skipping UVMD-{N}")
            continue
        model = rebuild(f, T, a.K, N, a.device)
        om0 = model.omega_plus[0].cpu().numpy() * cpd
        om, re = [], []
        for i in te:
            u, w = uvmd_magn.decompose(model, X[i], a.device)
            om.append(w * cpd); re.append(rel_err(X[i], u))
        rows[f"UVMD-{N}"], rel[f"UVMD-{N}"] = np.array(om), np.mean(re)
    init = om0 if "om0" in dir() else np.arange(a.K) * 0.5 / a.K * cpd

    lines = [f"# omega shift ({a.region}, {len(te)} test sensors, K={a.K}); units: cycles/day (cpd)\n",
             f"Initial grid: omega_k = 0.5/K * k cycles/sample = {init[1]:.2f} cpd apart, "
             f"{init[0]:.1f} .. {init[-1]:.1f} cpd. Mode 0 is the DC mode and does not move.\n",
             "Check that the rebuilt UVMD models are the trained ones: mean relative reconstruction error "
             "(bounded run reported 0.0205 for UVMD-1 and 0.0372 for UVMD-2):\n"]
    lines += [f"* {m}: {v:.4f}" for m, v in rel.items()]
    tab = pd.DataFrame({"init": init})
    for m, w in rows.items():
        tab[m] = w.mean(0)
    lines += ["\n## Final omega_k, mean over sensors (cpd)\n", tab.round(2).to_markdown(floatfmt=".2f")]
    shift = {m: np.abs(w[:, 1:] - init[None, 1:]) for m, w in rows.items()}
    sp = init[1] - init[0]
    s = pd.DataFrame({m: {"mean abs shift (cpd)": v.mean(), "max abs shift (cpd)": v.max(),
                          "mean abs shift per grid spacing": v.mean() / sp,
                          "modes that moved > 0.1 cpd": float((v.mean(0) > 0.1).sum())}
                      for m, v in shift.items()}).T
    lines += ["\n## How far omega_k moved from the start (modes 1..K-1)\n", s.round(3).to_markdown(floatfmt=".3f")]
    (out / "omega_shift.md").write_text("\n".join(lines) + "\n")
    tab.to_csv(out / "omega_shift.csv", index_label="mode")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.2))
        k = np.arange(a.K)
        ax.plot(k, init, "k--", label="start (uniform grid)")
        for m in tab.columns[1:]:
            ax.plot(k, tab[m], marker="o", ms=3, label=m)
        ax.set_xlabel("mode k"); ax.set_ylabel("centre frequency omega_k (cycles/day)")
        ax.set_title("Where the centre frequencies end up"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "omega_shift.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
