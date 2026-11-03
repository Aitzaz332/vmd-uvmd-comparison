"""Three-way mode comparison on CWRU: converged VMD  |  UVMD trained on reconstruction only  |  UVMD trained end to end.

    python experiments/plot_modes_three_way.py --run results/cwru_e2e_K4_N5 --recon-params results/cwru_K4/uvmd_params_N5.npz

--run is an output folder of run_cwru_e2e.py (it holds the VMD and end-to-end modes of the test windows and the input
windows); --recon-params is the upstream-format parameter file of a UVMD trained on the reconstruction loss only
(K, N and T must match). The recon-trained modes are computed on the very same windows. Nothing is retrained.
Writes <run>/three_way/:
  modes_three_way.png        the K modes of one test window: VMD | recon-trained UVMD | end-to-end UVMD
  centres_and_alpha.png      left: centre frequency of every mode; right: alpha_k of every mode (VMD: fixed)
  three_way_table.csv/.md    per mode: alpha, centre frequency; and the correlation between the three sets of modes
Mode k of every method is the same slot (all start from the same frequency grid). Mode 1 is the DC mode.
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
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e2e.export import decompose_params  # noqa: E402
from experiments.compare_uvmd_modes import similarity  # noqa: E402

FS = 12000
C_VMD, C_REC, C_E2E = "#2f6db5", "#3a9d5d", "#e8702a"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--recon-params", required=True)
    p.add_argument("--seed", type=int, default=0, help="end-to-end seed shown in the figure")
    p.add_argument("--cls", default="outer_0.007")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "three_way"
    out.mkdir(parents=True, exist_ok=True)
    V = dict(np.load(run / "vmd_modes_test.npz"))
    E = dict(np.load(run / f"uvmd_e2e_modes_test_seed{a.seed}.npz"))
    P = dict(np.load(a.recon_params))
    n, K, T = V["modes"].shape
    if P["lambda"].shape != (len(E["lambda"]), 2 * T) or len(P["alpha"]) != K:
        raise ValueError(f"recon-trained parameters {P['lambda'].shape}/K={len(P['alpha'])} do not match the run "
                         f"(N={len(E['lambda'])}, K={K}, T={T})")
    R_modes, R_om = decompose_params(P, V["signal"], torch.device("cpu"))
    S = json.load(open(run / "summary.json"))
    alpha0 = S["args"]["alpha"]

    names = list(map(str, V["class_names"]))
    win = int(np.where(np.asarray(V["label"]) == names.index(a.cls))[0][0]) if a.cls in names else 0
    cls = names[int(V["label"][win])]
    nt = min(512, T)
    t = np.arange(nt) / FS * 1000
    cols = [("VMD (converged)", V["modes"], V["omega"], C_VMD),
            ("UVMD, reconstruction only", R_modes, R_om, C_REC),
            ("UVMD, end to end", E["modes"], E["omega"], C_E2E)]
    fig, axs = plt.subplots(K + 1, 3, figsize=(16, 1.7 * (K + 1)), dpi=150, sharex=True)
    for j, (name, M, OM, c) in enumerate(cols):
        axs[0, j].plot(t, V["signal"][win][:nt], color="#222", lw=1)
        axs[0, j].set_title(name, fontsize=15, fontweight="bold", color=c, loc="left")
        axs[0, j].set_ylabel("signal", fontsize=11)
        for k in range(K):
            ax = axs[k + 1, j]
            ax.plot(t, M[win, k][:nt], color=c, lw=1)
            ax.set_ylabel(f"mode {k + 1}\n{OM[win, k] * FS:.0f} Hz", fontsize=11)
        axs[-1, j].set_xlabel("time (ms)", fontsize=13)
    for ax in axs.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=10)
    fig.suptitle(f"The {K} modes of one CWRU test window ({cls}); y-label = centre frequency of the mode", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "modes_three_way.png")
    plt.close(fig)

    # --- numbers
    alphas = {"VMD (converged)": np.full(K, alpha0), "UVMD, reconstruction only": np.asarray(P["alpha"], float),
              "UVMD, end to end": np.asarray(E["alpha"], float)}
    centres = {c[0]: (c[2].mean(0) * FS, c[2].std(0) * FS) for c in cols}
    rows = []
    for name, _, _, _ in cols:
        for k in range(K):
            rows.append(dict(method=name, mode=k + 1, alpha=alphas[name][k], centre_Hz=centres[name][0][k], centre_std_Hz=centres[name][1][k]))
    tab = pd.DataFrame(rows)
    pairs = [("end to end vs VMD", E["modes"], V["modes"]), ("reconstruction-only vs VMD", R_modes, V["modes"]),
             ("end to end vs reconstruction-only", E["modes"], R_modes)]
    corr = []
    for nm, A_, B_ in pairs:
        al, mt = similarity(A_, B_)
        row = {"pair": nm, "matched_mean_abs_corr": float(mt.mean())}
        row.update({f"mode_{k + 1}_corr": float(al[:, k].mean()) for k in range(K)})
        row["aligned_mean_all_modes"] = float(al.mean())
        corr.append(row)
    corr = pd.DataFrame(corr)
    tab.to_csv(out / "three_way_table.csv", index=False)
    corr.to_csv(out / "three_way_correlation.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    x = np.arange(K)
    w = 0.27
    for i, (name, _, _, c) in enumerate(cols):
        m, s = centres[name]
        a1.bar(x + (i - 1) * w, m, w, yerr=s, color=c, capsize=3, label=name)
        a2.bar(x + (i - 1) * w, np.maximum(alphas[name], 1e-3), w, color=c, label=name)
    a1.set_xticks(x, [f"mode {k + 1}" for k in range(K)])
    a2.set_xticks(x, [f"mode {k + 1}" for k in range(K)])
    a1.set_ylabel("centre frequency (Hz)", fontsize=13)
    a1.set_title("Where each mode sits", fontsize=15, fontweight="bold", loc="left")
    a2.set_ylabel(r"$\alpha_k$  (larger = narrower mode)", fontsize=13)
    a2.set_title("Bandwidth penalty of each mode", fontsize=15, fontweight="bold", loc="left")
    a2.set_yscale("log")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=13, ncol=3, loc="upper center")
    fig.text(0.01, 0.005, "left: mean ± std over the test windows; right: log scale (VMD fixed at the initial alpha)", fontsize=10, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out / "centres_and_alpha.png")
    plt.close(fig)
    md = [f"# Three-way mode comparison ({n} CWRU test windows, K={K}, N={len(E['lambda'])})", "",
          "Correlation between the modes of two methods, mode k with mode k, averaged over the test windows.", "",
          corr.round(3).to_markdown(index=False), "", tab.round(1).to_markdown(index=False)]
    (out / "three_way.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {out}")
    return corr, tab


if __name__ == "__main__":
    main()
