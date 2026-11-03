"""How the learned alpha_k and lambda change over the training epochs of UVMD (uniform and zero start).

    python experiments/param_trajectory.py run synthetic --src results/synthetic_K5_alpha500_L5
    python experiments/param_trajectory.py run cwru    --src results/cwru_K4    --data ~/data/CWRU --device cuda:0
    python experiments/param_trajectory.py run traffic --src results/traffic_K6 --data ~/data/LargeST/sd_his_2019.h5 --device cuda:0
    python experiments/param_trajectory.py plot        # all datasets that have been run

``run`` retrains UVMD with the settings of the finished run in --src (same data split, epochs, lr, seed), once with
the uniform start and once with all centre frequencies starting at 0, and records after every epoch
  alpha_k = softplus(Alpha)  (K values)   and   mean |lambda| of every layer's multiplier row.
Result: results/param_trajectory/<dataset>_{uniform,zero}.npz.  ``plot`` writes <dataset>.png and summary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))
OUT = ROOT / "results" / "param_trajectory"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODE_C = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#d6336c", "#0b8a8a"]


def data_split(dataset, S, data):
    if dataset == "synthetic":
        from sklearn.model_selection import train_test_split
        from datasets.synthetic import make_synthetic
        D = make_synthetic(S["n_signals"], S["T"], S["noise"], S["seed"])
        tr, te = train_test_split(np.arange(S["n_signals"]), test_size=0.3, random_state=42, shuffle=True)
        va, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
        return D["x"], tr, va
    if dataset == "cwru":
        from datasets.cwru import load_cwru, split_indices
        D = load_cwru(data, S["T"], S["windows_per_file"], not S.get("no_standardize", False))
        tr, va, _ = split_indices(D, S["split"])
        return D["x"], tr, va
    from datasets.traffic import load_traffic, split_nodes
    D = load_traffic(data, S.get("length"))
    tr, va, _ = split_nodes(D["x"].shape[0])
    if S.get("max_train_nodes"):
        tr = tr[: S["max_train_nodes"]]
    if S.get("max_val_nodes"):
        va = va[: S["max_val_nodes"]]
    return D["x"].astype(np.float64), tr, va


def run(a):
    import torch
    import torch.nn.functional as F
    from decomp import uvmd_magn
    S = json.load(open(Path(a.src) / "summary.json"))["args"]
    K, alpha, N = S["K"], S["alpha"], S["layers"][-1]
    X, tr, va = data_split(a.dataset, S, a.data)
    T = X.shape[1]
    OUT.mkdir(parents=True, exist_ok=True)
    for start in ("uniform", "zero"):
        torch.manual_seed(S.get("seed", 0))
        model = uvmd_magn.make_model(T, K, N, a.device, alpha)
        if start == "zero":
            model.omega_plus[0, :] = 0.0

        def snap(m):
            return {"alpha": F.softplus(m.Alpha.detach()).cpu().numpy().tolist(),
                    "lam": m.lambda_hat.detach().abs().mean(dim=1)[: N - 1].cpu().numpy().tolist()}
        h = uvmd_magn.train_uvmd(model, X[tr], X[va], S["epochs"], S["lr"], S["patience"], a.device, S.get("seed", 0),
                                 False, log=lambda s: print(f"[{a.dataset} {start}] {s}", flush=True), epoch_eval=snap)
        c = h["curve"]
        np.savez(OUT / f"{a.dataset}_{start}.npz", epoch=[e["epoch"] for e in c], alpha=[e["alpha"] for e in c],
                 lam=[e["lam"] for e in c], train_loss=h["train_loss"], val_loss=h["val_loss"], K=K, N=N, alpha0=alpha)
        print(f"saved {a.dataset}_{start}: {h['epochs_run']} epochs", flush=True)


def _ax(ax):
    ax.grid(True, color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot():
    names = [p.stem.rsplit("_", 1)[0] for p in sorted(OUT.glob("*_uniform.npz")) if (OUT / (p.stem.rsplit("_", 1)[0] + "_zero.npz")).exists()]
    md = ["# alpha_k and lambda over the training epochs\n"]
    for ds in names:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
        md.append(f"\n## {ds}\n")
        for c, start in enumerate(("uniform", "zero")):
            z = np.load(OUT / f"{ds}_{start}.npz")
            ep, al, lam = z["epoch"], z["alpha"], z["lam"]
            for k in range(al.shape[1]):
                axes[0, c].plot(ep, np.maximum(al[:, k], 1e-5), color=MODE_C[k % 8], lw=1.8, label=f"mode {k + 1}" + (" (DC)" if k == 0 else ""))
            axes[0, c].set_yscale("log")
            axes[0, c].axhline(float(z["alpha0"]), color=MUTED, lw=0.8, ls="--")
            axes[0, c].set_title(f"{start} start: alpha_k", loc="left", fontsize=11, color=INK)
            for l in range(lam.shape[1]):
                axes[1, c].plot(ep, lam[:, l], color=MODE_C[(l + 1) % 8], lw=1.8, label=f"layer {l + 1}")
            axes[1, c].set_title(f"{start} start: mean |lambda| per layer", loc="left", fontsize=11, color=INK)
            axes[1, c].set_xlabel("training epoch", color=MUTED)
            for r in (0, 1):
                _ax(axes[r, c])
            md.append(f"- {start} start, epoch 0 -> {int(ep[-1])}: alpha_k "
                      f"{np.round(al[0], 1).tolist()} -> {np.round(al[-1], 4).tolist()}; "
                      f"mean |lambda| per layer {np.round(lam[0], 4).tolist()} -> {np.round(lam[-1], 4).tolist()}")
        axes[0, 0].legend(fontsize=8, frameon=False, ncol=2)
        axes[1, 0].legend(fontsize=8, frameon=False, ncol=2)
        axes[0, 0].set_ylabel("alpha_k (log scale; dashed = start)", color=MUTED)
        axes[1, 0].set_ylabel("mean |lambda|", color=MUTED)
        fig.suptitle(f"{ds}: learned alpha_k and lambda over the training epochs", x=0.005, ha="left", fontsize=13, color=INK, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"{ds}.png", dpi=130)
        plt.close(fig)
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("dataset", choices=["synthetic", "cwru", "traffic"])
    r.add_argument("--src", required=True)
    r.add_argument("--data", default=None)
    r.add_argument("--device", default="cpu")
    sp.add_parser("plot")
    a = p.parse_args()
    run(a) if a.cmd == "run" else plot()
