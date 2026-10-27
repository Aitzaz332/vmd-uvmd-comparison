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


def plot_modes_envelope(path, x, fs, decomps: dict, fault_hz, title):
    """decomps: {label: (modes (K,T), sel_mode)}; one column per method: modes, then envelope spectrum."""
    from decomp.fault_metrics import envelope_spectrum
    labs = list(decomps)
    K = max(m.shape[0] for m, _ in decomps.values())
    fig, axes = plt.subplots(K + 2, len(labs), figsize=(4.2 * len(labs), 1.25 * (K + 2) + 0.6),
                             squeeze=False)
    t = np.arange(len(x)) / fs * 1e3
    for c, lab in enumerate(labs):
        u, sel = decomps[lab]
        axes[0, c].plot(t, x, color=MUTED, lw=0.6)
        axes[0, c].set_title(f"{lab}\ninput", loc="left", fontsize=9, color=INK)
        for k in range(K):
            ax = axes[k + 1, c]
            if k < len(u):
                ax.plot(t[:u.shape[1]], u[k], color=C[1] if k == sel else C[0], lw=0.6)
                ax.set_title(f"mode {k}" + ("  (max kurtosis)" if k == sel else ""), loc="left",
                             fontsize=8, color=INK)
            else:
                ax.axis("off")
        f, s = envelope_spectrum(u[sel], fs)
        ax = axes[K + 1, c]
        keep = f <= 600
        ax.plot(f[keep], s[keep], color=C[1], lw=1)
        for h in (1, 2, 3):
            ax.axvline(h * fault_hz, color=INK, lw=0.8, ls=":")
        ax.set_title("envelope spectrum of selected mode (dotted: fault harmonics)", loc="left", fontsize=8,
                     color=INK)
        ax.set_xlabel("Hz", color=MUTED, fontsize=8)
    for ax in axes.ravel():
        _style(ax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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


def plot_traffic_modes(path, x, decomps: dict, steps_per_day=96, days=7, max_cpd=6.0, title=""):
    """decomps: {label: modes (K,T)}. Per method: input vs sum of modes (first ``days`` days), each mode,
    and the power spectra of all modes with the daily / 2nd-harmonic / weekly lines marked."""
    labs = list(decomps)
    K = max(u.shape[0] for u in decomps.values())
    n = days * steps_per_day
    fig, axes = plt.subplots(K + 2, len(labs), figsize=(4.4 * len(labs), 0.95 * (K + 2) + 1.2), squeeze=False)
    t = np.arange(n) / steps_per_day
    for c, lab in enumerate(labs):
        u = decomps[lab]
        T = u.shape[1]
        axes[0, c].plot(t, x[:n], color=MUTED, lw=0.8, label="input")
        axes[0, c].plot(t, u.sum(axis=0)[:n], color=C[1], lw=0.8, ls="--", label="sum of modes")
        axes[0, c].set_title(f"{lab}: input vs sum of modes", loc="left", fontsize=8, color=INK)
        if c == 0:
            axes[0, c].legend(fontsize=7, frameon=False, loc="upper right")
        for k in range(K):
            ax = axes[k + 1, c]
            ax.plot(t, u[k, :n], color=C[0], lw=0.7)
            ax.set_title(f"mode {k}", loc="left", fontsize=7, color=INK, pad=1)
            ax.tick_params(labelsize=6)
        ax = axes[K + 1, c]
        f = np.fft.rfftfreq(T) * steps_per_day
        P = np.abs(np.fft.rfft(u, axis=1)) ** 2
        keep = (f <= max_cpd) & (f > 0)
        for k in range(len(u)):
            ax.plot(f[keep], P[k, keep] + 1e-12, lw=0.8, color=C[k % len(C)])
        for v in (1.0, 2.0, 1 / 7):
            ax.axvline(v, color=INK, lw=0.7, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("cycles / day  (dotted: weekly, daily, 2nd harmonic)", fontsize=7, color=MUTED)
        ax.set_title("mode power spectra", loc="left", fontsize=7, color=INK, pad=1)
    for ax in axes.ravel():
        _style(ax)
        ax.title.set_fontsize(7)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_alpha(path, alphas: dict):
    """Learned bandwidth constraint phi(alpha_k)=softplus(alpha_raw_k) versus mode index (paper Fig. 2/3 right)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lab, a) in enumerate(alphas.items()):
        ax.plot(np.arange(len(a)), a, marker="o", ms=4, lw=1.8, color=C[1 + i], label=lab)
    ax.axhline(2000, color=MUTED, lw=0.8, ls=":")
    ax.text(0.2, 2000, " initial alpha = 2000", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlabel("mode index k", color=MUTED)
    ax.set_ylabel("learned bandwidth constraint", color=MUTED)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
