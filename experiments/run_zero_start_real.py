"""Zero start (all centre frequencies start at 0) on CWRU and traffic: VMD and UVMD vs the uniform start already run.

    python experiments/run_zero_start_real.py cwru    --src results/cwru_K4    --data ~/data/CWRU --device cuda:0
    python experiments/run_zero_start_real.py traffic --src results/traffic_K6 --data ~/data/LargeST/sd_his_2019.h5 --device cuda:0

Uses the settings of the finished uniform-start run in --src (K, alpha, layers, epochs, train/val/test split, same
test windows/nodes) and changes ONLY the start of the centre frequencies. The uniform-start numbers are recomputed
on the same test set from the saved parameters, so both starts are compared like for like.
Writes <src>_zeroinit/{zero_vs_uniform_start.csv/.md, omegas.json, uvmd_params_N5.npz, uvmd_history_N5.json, modes.png}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from decomp.metrics import evaluate  # noqa: E402
from decomp.vmd_numpy import uvmd_forward_numpy, vmd  # noqa: E402
from experiments.mode_figures import BOX, FAINT, INK, MUTED, VMD_C, UVMD_C, _style  # noqa: E402

GREEN = "#1baf7a"
COLS = [("VMD (converged)", VMD_C), ("UVMD uniform start", UVMD_C), ("UVMD zero start", GREEN)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["cwru", "traffic"])
    p.add_argument("--src", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    src = Path(a.src)
    S = json.load(open(src / "summary.json"))["args"]
    K, alpha, N = S["K"], S["alpha"], S["layers"][-1]
    out = Path(a.out or f"{a.src.rstrip('/')}_zeroinit")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from decomp import uvmd_magn

    if a.dataset == "cwru":
        from datasets.cwru import FS, fault_freqs, load_cwru, split_indices
        from decomp.fault_metrics import fault_eval
        D = load_cwru(a.data, S["T"], S["windows_per_file"], not S.get("no_standardize", False))
        tr, va, te = split_indices(D, S["split"])
        X, T, unit, scale = D["x"], S["T"], "Hz", FS
        FF = [fault_freqs(r) for r in D["rpm"]]
        ex_i = int(next((i for i in te if D["cls"][i] == "inner"), te[0]))
        zoom = 600
    else:
        from datasets.traffic import STEPS_PER_DAY, load_traffic, split_nodes
        from decomp.traffic_metrics import spectral_metrics
        D = load_traffic(a.data, S.get("length"), {"n_nodes": S["demo_nodes"], "days": S["demo_days"]} if a.data == "demo" else None)
        tr, va, te = split_nodes(D["x"].shape[0])
        tr, te = tr[:S["max_train_nodes"]], te[:S["max_test_nodes"]]
        X, T, unit, scale = D["x"].astype(np.float64), D["x"].shape[1], "cycles/day", STEPS_PER_DAY
        ex_i = int(te[0])
        zoom = 7 * STEPS_PER_DAY
    X = np.asarray(X)

    def ev(i, u, om):
        m = evaluate(X[i], u, omega=om)
        if a.dataset == "cwru":
            f = fault_eval(u, X[i], D["cls"][i], FF[i], FS, om, S.get("harmonics", 3))
            m.update(fault_snr_db=f["fault_snr_db"], envelope_hit=f["envelope_hit"])
        else:
            s = spectral_metrics(X[i], u, STEPS_PER_DAY)
            m.update(mode_overlap=s["mode_overlap"], daily_ownership=s["daily_ownership"])
        return m

    P_uni = {k: np.load(src / f"uvmd_params_N{N}.npz")[k] for k in ("alpha", "lambda")}
    rows, om_sorted, example = [], {}, {}

    def add(label, i, u, om):
        r = ev(i, u, om)
        r.update(method=label, item=int(i))
        rows.append(r)
        om_sorted.setdefault(label, []).append(np.sort(om))
        if i == ex_i:
            example[label] = (u, om)

    t0 = time.perf_counter()
    for i in te:                                   # classic VMD, run to convergence (uniform and zero start)
        for lab, init in (("VMD uniform", 1), ("VMD zero", 0)):
            r = vmd(X[i], alpha, S.get("tau", 0.0), K, True, init, S.get("tol", 1e-7), S.get("max_iter", 499))
            add(lab, i, r["u"], r["omega"])
        for lab, init in (("VMD-5 uniform", 1), ("VMD-5 zero", 0)):
            r = vmd(X[i], alpha, S.get("tau", 0.0), K, True, init, -1.0, N)
            add(lab, i, r["u"], r["omega"])
        r = uvmd_forward_numpy(X[i].astype(np.float32).astype(np.float64), P_uni["alpha"], P_uni["lambda"], K)
        add("UVMD uniform", i, r["u"], r["omega"])
    print(f"VMD + uniform UVMD on {len(te)} test items: {time.perf_counter() - t0:.0f} s", flush=True)

    # UVMD with every centre frequency starting at 0 (only this changes; same recipe as the uniform run)
    torch.manual_seed(S.get("seed", 0))
    model = uvmd_magn.make_model(T, K, N, a.device, alpha)
    model.omega_plus[0, :] = 0.0
    assert float(model.omega_plus.abs().sum()) == 0.0
    h = uvmd_magn.train_uvmd(model, X[tr], X[va], S["epochs"], S["lr"], S["patience"], a.device, S.get("seed", 0),
                             False, log=lambda s: print(s, flush=True),
                             epoch_eval=uvmd_magn.recon_eval_fn(X[te[:20 if a.dataset == "cwru" else 10]], a.device))
    json.dump(h, open(out / f"uvmd_history_N{N}.json", "w"))
    np.savez(out / f"uvmd_params_N{N}.npz", **uvmd_magn.export_params(model))
    for i in te:
        u, om = uvmd_magn.decompose(model, X[i], a.device)
        add("UVMD zero", i, u, om)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_item.csv", index=False)
    keys = [c for c in ("recon_rel_err", "ortho_index", "fault_snr_db", "envelope_hit", "mode_overlap", "daily_ownership") if c in df]
    T_ = df.groupby("method")[keys].mean().loc[["VMD uniform", "VMD zero", "VMD-5 uniform", "VMD-5 zero", "UVMD uniform", "UVMD zero"]]
    T_["recon_rel_err"] *= 100
    T_ = T_.rename(columns={"recon_rel_err": "reconstruction error %"})
    T_.to_csv(out / "zero_vs_uniform_start.csv")
    om_mean = {k: (np.mean(v, axis=0) * scale).tolist() for k, v in om_sorted.items()}
    json.dump({"unit": unit, "omegas": om_mean}, open(out / "omegas.json", "w"), indent=1)
    md = [f"# {a.dataset}: zero start vs uniform start (K={K}, alpha={alpha:g}, {N} layers, {len(te)} test items)\n",
          T_.to_markdown(floatfmt=".3g"), f"\nMean final centre frequencies ({unit}, modes sorted by frequency):\n"]
    md += [f"- {k}: {np.round(v, 2).tolist()}" for k, v in om_mean.items()]
    md += [f"\nLearned alpha, UVMD zero start: {np.round(np.sort(np.load(out / f'uvmd_params_N{N}.npz')['alpha']), 4).tolist()}"]
    (out / "zero_vs_uniform_start.md").write_text("\n".join(md))
    print("\n".join(md))

    # modes side by side for one example, modes ordered by centre frequency
    fig, axes = plt.subplots(K, 3, figsize=(15, 1.3 * K + 1.2), sharex=True)
    keymap = {"VMD (converged)": "VMD uniform", "UVMD uniform start": "UVMD uniform", "UVMD zero start": "UVMD zero"}
    for c, (title, col) in enumerate(COLS):
        u, om = example[keymap[title]]
        o = np.argsort(om)
        axes[0, c].set_title(title, loc="left", fontsize=12, color=col, fontweight="bold")
        lim = np.abs(X[ex_i]).max()
        for r_, k in enumerate(o):
            ax = axes[r_, c]
            ax.plot(u[k][:zoom], color=col, lw=1)
            ax.set_ylim(-lim, lim)
            ax.text(0.99, 0.92, f"{om[k] * scale:.1f} {unit}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK, bbox=BOX)
            _style(ax, bottom=(r_ == K - 1))
    fig.suptitle(f"{a.dataset.upper()} example, K = {K}, alpha = {alpha:g}: modes ordered by centre frequency (label = centre frequency)",
                 x=0.005, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "modes.png", dpi=130)
    print("written to", out)


if __name__ == "__main__":
    main()
