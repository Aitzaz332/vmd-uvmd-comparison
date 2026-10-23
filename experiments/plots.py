"""Figures for the benchmark (matplotlib, static PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical slots, fixed order: blue, orange, aqua, yellow, magenta
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_example(path, t, x, true_modes, mode_names, decomps: dict):
    """decomps: {label: modes (J,T) already matched to true order}"""
    J = len(true_modes)
    fig, axes = plt.subplots(J + 1, 1, figsize=(10, 1.9 * (J + 1)), sharex=True)
    axes[0].plot(t, x, color=MUTED, lw=1)
    axes[0].set_title("Input signal (noisy)", loc="left", fontsize=10, color=INK)
    for j in range(J):
        ax = axes[j + 1]
        ax.plot(t, true_modes[j], color=INK, lw=2, label="true")
        for i, (lab, m) in enumerate(decomps.items()):
            ax.plot(t, m[j], color=C[i], lw=1.4, label=lab, alpha=0.9)
        ax.set_title(f"Mode {j}: {mode_names[j]}", loc="left", fontsize=10, color=INK)
    for ax in axes:
        _style(ax)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False, ncol=len(decomps) + 1)
    axes[-1].set_xlabel("sample", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_vs_iterations(path, sweep, uvmd_points, metric, ylabel, logy=False):
    """sweep: list of dicts with n_iter + metric mean; uvmd_points: {label: (layers, value)}"""
    fig, ax = plt.subplots(figsize=(7, 4))
    n = [r["n_iter"] for r in sweep]
    v = [r[metric] for r in sweep]
    ax.plot(n, v, color=C[0], lw=2, marker="o", ms=5, label="VMD, n ADMM iterations")
    for i, (lab, (layers, val)) in enumerate(uvmd_points.items()):
        ax.plot([layers], [val], marker="D", ms=9, ls="none", color=C[1 + i], label=lab)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("iterations (VMD) / layers (UVMD)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_runtime(path, labels, ms):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.2))
    y = np.arange(len(labels))
    ax.barh(y, ms, color=C[0], height=0.6)
    for yi, v in zip(y, ms):
        ax.text(v, yi, f"  {v:.2f} ms", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("median wall-clock per signal (ms, log scale)", color=MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training(path, hists: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lab, h) in enumerate(hists.items()):
        e = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(e, h["train_loss"], color=C[1 + i], lw=2, label=f"{lab} train")
        ax.plot(e, h["val_loss"], color=C[1 + i], lw=1.2, ls="--", label=f"{lab} val")
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("spectral L1 loss (upstream)", color=MUTED)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
