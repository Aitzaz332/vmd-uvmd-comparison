"""Clean, slide-ready figures of the modes: converged VMD vs the UVMD trained end to end, and alpha vs the loss.

    python experiments/plot_modes_clean.py --run results/cwru_e2e_K4_N5_lolo/test_3HP

Reads one run folder of run_cwru_e2e.py (vmd_modes_test.npz, uvmd_e2e_modes_test_seed*.npz, histories.json); nothing is
retrained. Writes <run>/modes_figures/ (each figure answers ONE question):
  1_modes_one_window.png      the K modes of one test window, VMD (left) next to end-to-end UVMD (right)
  2_mode_spectra.png          where each mode sits in frequency (average over the test windows of one fault class)
  3_centre_frequencies.png    centre frequency of every mode, VMD vs end-to-end UVMD (mean +- std over test windows)
  4_alpha_and_loss.png        top: training and validation loss; bottom: alpha_k of every mode over the epochs
  alpha_table.csv, centre_frequencies.csv     the numbers behind figures 3 and 4
Mode k of the two methods is the same slot: both start from the same frequency grid. Mode 1 is the low-frequency (DC)
mode. alpha_k is the bandwidth penalty: a larger alpha makes the mode narrower around its centre frequency.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VMD_C, UVMD_C = "#2f6db5", "#e8702a"
MODE_C = ["#4c4c4c", "#2f6db5", "#e8702a", "#3a9d5d", "#9b59b6", "#c0392b"]
FS = 12000


def style(ax, fs=13):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=fs - 2)
    ax.grid(alpha=0.25)


def fig_modes_one_window(V, U, out, seed, win, n=512):
    K = V["modes"].shape[1]
    n = min(n, V["modes"].shape[-1])
    t = np.arange(n) / FS * 1000
    fig, axs = plt.subplots(K + 1, 2, figsize=(13, 1.7 * (K + 1)), dpi=150, sharex=True)
    cls = str(V["class_names"][int(V["label"][win])])
    for j, (D, name, c) in enumerate(((V, "VMD (converged)", VMD_C), (U, f"UVMD end-to-end (seed {seed})", UVMD_C))):
        axs[0, j].plot(t, D["signal"][win][:n], color="#222", lw=1)
        axs[0, j].set_title(name, fontsize=15, fontweight="bold", color=c, loc="left")
        axs[0, j].set_ylabel("signal", fontsize=11)
        for k in range(K):
            ax = axs[k + 1, j]
            ax.plot(t, D["modes"][win, k][:n], color=c, lw=1)
            ax.set_ylabel(f"mode {k + 1}\n{D['omega'][win, k] * FS:.0f} Hz", fontsize=11)
        for ax in axs[:, j]:
            style(ax, 12)
    for j in range(2):
        axs[-1, j].set_xlabel("time (ms)", fontsize=13)
    fig.suptitle(f"The {K} modes of one test window ({cls}) - first {n} samples; y-label = centre frequency", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "1_modes_one_window.png")
    plt.close(fig)


def mean_spectrum(modes):
    sp = np.abs(np.fft.rfft(modes, axis=-1)) ** 2
    return sp.mean(0)                                           # (K, F)


def fig_spectra(V, U, out, seed, cls="outer_0.007"):
    names = list(map(str, V["class_names"]))
    sel = np.where(np.asarray(V["label"]) == names.index(cls))[0] if cls in names else np.arange(len(V["label"]))
    T = V["modes"].shape[-1]
    fr = np.fft.rfftfreq(T, 1 / FS)
    K = V["modes"].shape[1]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150, sharey=True)
    for ax, D, name in zip(axs, (V, U), ("VMD (converged)", f"UVMD end-to-end (seed {seed})")):
        S = mean_spectrum(D["modes"][sel])
        S = S / S.max()
        for k in range(K):
            ax.plot(fr, S[k], color=MODE_C[k], lw=2, label=f"mode {k + 1}")
        ax.set_title(name, fontsize=15, fontweight="bold", loc="left")
        ax.set_xlabel("frequency (Hz)", fontsize=13)
        ax.set_xlim(0, FS / 2)
        style(ax)
    axs[0].set_ylabel("relative power", fontsize=13)
    axs[0].legend(frameon=False, fontsize=12)
    fig.suptitle(f"Frequency content of each mode (average over the test windows of class '{cls}')", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "2_mode_spectra.png")
    plt.close(fig)


def fig_centres(V, Us, out):
    K = V["modes"].shape[1]
    rows = []
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    x = np.arange(K)
    w = 0.8 / (1 + len(Us))
    series = [("VMD (converged)", V["omega"] * FS, VMD_C)] + [(f"UVMD end-to-end (seed {s})", U["omega"] * FS, UVMD_C) for s, U in Us]
    for i, (name, om, c) in enumerate(series):
        m, sd = om.mean(0), om.std(0)
        alpha = 1.0 if i == 0 else 0.45 + 0.55 * (1 - (i - 1) / max(len(Us), 1))
        ax.bar(x + (i - (len(series) - 1) / 2) * w, m, w, yerr=sd, color=c, alpha=alpha, capsize=3, label=("VMD (converged)" if i == 0 else ("UVMD end-to-end (one bar per seed)" if i == 1 else None)))
        for k in range(K):
            rows.append((name, k + 1, m[k], sd[k]))
    ax.set_xticks(x, [f"mode {k + 1}" for k in range(K)])
    ax.set_ylabel("centre frequency (Hz)", fontsize=13)
    ax.set_title("Where each mode sits: VMD vs end-to-end UVMD", fontsize=15, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=12)
    style(ax)
    ax.text(0, -0.17, "bars = mean over the test windows, whiskers = std across windows; one bar per UVMD seed", transform=ax.transAxes, fontsize=10, color="#666")
    fig.tight_layout()
    fig.savefig(out / "3_centre_frequencies.png")
    plt.close(fig)
    pd.DataFrame(rows, columns=["method", "mode", "mean_Hz", "std_Hz"]).to_csv(out / "centre_frequencies.csv", index=False)


def fig_alpha_loss(H, out, alpha0):
    seeds = sorted(k for k in H if k.startswith("uvmd_e2e_seed"))
    if not seeds:
        return
    n = min(len(H[k]["alpha"]) for k in seeds)
    A = np.stack([np.asarray(H[k]["alpha"][:n]) for k in seeds])                  # (S, n, K)
    tr = np.stack([np.asarray(H[k].get("train_eval_ce") or H[k]["train_ce"])[:n] for k in seeds])
    va = np.stack([np.asarray(H[k]["val_ce"][:n]) for k in seeds])
    best = [H[k].get("best_epoch") for k in seeds]
    K = A.shape[2]
    x = np.arange(1, n + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 8), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    for M, nm, ls in ((tr, "training loss", "--"), (va, "validation loss", "-")):
        a1.plot(x, M.mean(0), ls, lw=2.5, color="#222", label=nm)
        a1.fill_between(x, M.min(0), M.max(0), color="#222", alpha=0.08, lw=0)
    a1.set_ylabel("cross-entropy", fontsize=13)
    a1.set_title("Loss and alpha during end-to-end training", fontsize=15, fontweight="bold", loc="left")
    a1.legend(frameon=False, fontsize=12)
    a1.set_ylim(bottom=0)
    style(a1)
    for k in range(K):
        a2.plot(x, A[:, :, k].mean(0), lw=2.5, color=MODE_C[k], label=f"mode {k + 1}")
        a2.fill_between(x, A[:, :, k].min(0), A[:, :, k].max(0), color=MODE_C[k], alpha=0.12, lw=0)
    a2.axhline(alpha0, color="#999", lw=1.2, ls=":")
    a2.text(n, alpha0, " start", va="bottom", ha="right", fontsize=10, color="#777")
    a2.set_ylabel(r"$\alpha_k$  (larger = narrower mode)", fontsize=13)
    a2.set_xlabel("epoch", fontsize=13)
    a2.legend(frameon=False, fontsize=12, ncol=K, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style(a2)
    for a in (a1, a2):
        for b in best:
            if b:
                a.axvline(b, color="#aaa", lw=1, alpha=0.6)
    a2.text(0.01, 0.03, "line = mean over seeds, band = min-max; grey verticals = epoch of the selected model (one per seed)",
            transform=a2.transAxes, fontsize=10, color="#666")
    fig.tight_layout()
    fig.savefig(out / "4_alpha_and_loss.png")
    plt.close(fig)
    rows = []
    for s, k_ in enumerate(seeds):
        b = (best[s] or n) - 1
        for k in range(K):
            rows.append((k_, k + 1, alpha0, A[s, 0, k], A[s, b, k], A[s, -1, k], tr[s, b], va[s, b]))
    pd.DataFrame(rows, columns=["run", "mode", "alpha_start", "alpha_after_epoch1", "alpha_selected_epoch", "alpha_last_epoch",
                                "train_ce_selected", "val_ce_selected"]).to_csv(out / "alpha_table.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=None, help="UVMD seed used in figures 1 and 2 (default: the first)")
    p.add_argument("--cls", default="outer_0.007", help="fault class shown in figures 1 and 2")
    a = p.parse_args(argv)
    run = Path(a.run)
    out = run / "modes_figures"
    out.mkdir(parents=True, exist_ok=True)
    S = json.load(open(run / "summary.json"))
    V = dict(np.load(run / "vmd_modes_test.npz", allow_pickle=False))
    Us = []
    for f in sorted(run.glob("uvmd_e2e_modes_test_seed*.npz")):
        Us.append((int(f.stem.split("seed")[1]), dict(np.load(f, allow_pickle=False))))
    if not Us:
        raise FileNotFoundError(f"no uvmd_e2e_modes_test_seed*.npz in {run}")
    seed, U = next(((s, u) for s, u in Us if s == a.seed), Us[0])
    names = list(map(str, V["class_names"]))
    win = int(np.where(np.asarray(V["label"]) == names.index(a.cls))[0][0]) if a.cls in names else 0
    fig_modes_one_window(V, U, out, seed, win)
    fig_spectra(V, U, out, seed, a.cls)
    fig_centres(V, Us, out)
    fig_alpha_loss(json.load(open(run / "histories.json")), out, S["args"]["alpha"])
    print(f"mode figures -> {out}")
    return out


if __name__ == "__main__":
    main()
