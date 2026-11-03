"""Presentation-ready training / validation curves: VMD -> CNN vs end-to-end UVMD -> CNN.

    python experiments/plot_training_curves.py --run results/cwru_e2e_K4_N5

Reads <run>/histories.json (nothing is retrained) and writes <run>/training_curves/:
  training_loss.png, validation_loss.png, training_accuracy.png, validation_accuracy.png   one big figure each (slides)
  overview_2x2.png                                                                          the four together
  curves.csv                                                                                the plotted numbers
Line = mean over seeds, band = min..max over seeds, dot = epoch of the selected (best-validation) model.
Training loss/accuracy are measured on the training windows after each epoch in evaluation mode (no dropout), the same
way the validation numbers are, so the two are directly comparable. Older runs without these fall back to the running
training loss (with dropout) and have no training-accuracy figure.
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

COLORS = {"vmd": "#2f6db5", "uvmd_e2e": "#e8702a", "vmd_n": "#7f7f7f", "uvmd_early": "#3a9d5d", "raw": "#8c564b"}
PANELS = [("train_loss", "Training loss", "cross-entropy (lower is better)", ("train_eval_ce", "train_ce")),
          ("val_loss", "Validation loss", "cross-entropy (lower is better)", ("val_ce",)),
          ("train_acc", "Training accuracy", "accuracy", ("train_eval_acc",)),
          ("val_acc", "Validation accuracy", "accuracy", ("val_acc",))]


def labels(S):
    N = S["args"]["layers"]
    return {"raw": "raw signal", "vmd": "VMD (converged) → CNN", "vmd_n": f"VMD-{N} → CNN",
            "uvmd_early": f"UVMD-{N} recon-trained → CNN", "uvmd_e2e": f"UVMD-{N} end-to-end → CNN"}


def collect(H, pipe, keys):
    runs = [(k, h) for k, h in H.items() if k.rsplit("_seed", 1)[0] == pipe]
    for key in keys:
        curves = [np.asarray(h[key], float) for _, h in runs if h.get(key)]
        if len(curves) == len(runs) and curves:
            n = min(len(c) for c in curves)
            return np.stack([c[:n] for c in curves]), [h.get("best_epoch") for _, h in runs], key
    return None, None, None


def draw(ax, H, pipes, lab, keys, title, ylabel, big=False, rows=None, tag=None):
    drawn = False
    for pipe in pipes:
        C, best, used = collect(H, pipe, keys)
        if C is None:
            continue
        drawn = True
        x = np.arange(1, C.shape[1] + 1)
        m = C.mean(0)
        ax.fill_between(x, C.min(0), C.max(0), color=COLORS[pipe], alpha=0.18, lw=0)
        ax.plot(x, m, color=COLORS[pipe], lw=3 if big else 2, label=lab[pipe])
        b = [e for e in best if e and e <= C.shape[1]]
        if b:
            ax.plot(b, [m[e - 1] for e in b], "o", color=COLORS[pipe],
                    mec="white", mew=1.5, ms=9 if big else 7, zorder=5)
        if rows is not None:
            for s, c in enumerate(C):
                for e, v in enumerate(c):
                    rows.append((tag, pipe, s, e + 1, v))
    if not drawn:
        ax.text(0.5, 0.5, "not recorded in this run", ha="center", va="center", transform=ax.transAxes, color="#777")
    fs = 17 if big else 12
    ax.set_title(title, fontsize=fs + 3, fontweight="bold", loc="left")
    ax.set_xlabel("epoch", fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.tick_params(labelsize=fs - 2)
    ax.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if "accuracy" in title:
        ax.set_ylim(0, 1.02)
    else:
        ax.set_ylim(bottom=0)
    if drawn:
        ax.legend(fontsize=fs - 1, frameon=False)
    return drawn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--pipelines", nargs="+", default=["vmd", "uvmd_e2e"])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    run = Path(a.run)
    H = json.load(open(run / "histories.json"))
    S = json.load(open(run / "summary.json"))
    lab = labels(S)
    pipes = [q for q in a.pipelines if any(k.rsplit("_seed", 1)[0] == q for k in H)]
    out = Path(a.out) if a.out else run / "training_curves"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, title, yl, keys in PANELS:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        draw(ax, H, pipes, lab, keys, title, yl, big=True, rows=rows, tag=tag)
        fig.text(0.01, 0.005, "line: mean over seeds · band: min–max · dot: epoch of the selected model (one per seed)", fontsize=10, color="#666")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out / f"{ {'train_loss': 'training_loss', 'val_loss': 'validation_loss', 'train_acc': 'training_accuracy', 'val_acc': 'validation_accuracy'}[tag] }.png")
        plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    for ax, (tag, title, yl, keys) in zip(axs.ravel(), PANELS):
        draw(ax, H, pipes, lab, keys, title, yl)
    fig.tight_layout()
    fig.savefig(out / "overview_2x2.png")
    plt.close(fig)
    pd.DataFrame(rows, columns=["panel", "pipeline", "seed_index", "epoch", "value"]).to_csv(out / "curves.csv", index=False)
    print(f"curves -> {out}")
    return out


if __name__ == "__main__":
    main()
