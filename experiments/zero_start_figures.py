"""Centre-frequency movement and mode comparison, uniform start vs zero start (synthetic, K=5, alpha=500, 5 layers).

    python experiments/zero_start_figures.py
Reads results/synthetic_K5_alpha500_L5 (uniform start) and results/synthetic_K5_alpha500_zeroinit (zero start);
writes freq_movement.{md,csv,png} and modes_side_by_side.png to results/synthetic_K5_alpha500_zeroinit/readable.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.synthetic import make_synthetic
from decomp.metrics import match_modes
from decomp.vmd_numpy import vmd
from experiments.mode_figures import BOX, FAINT, INK, MUTED, TRUTH, UVMD_C, VMD_C, _matched_idx, _style

K, ALPHA, N = 5, 500.0, 5
GREEN = "#1baf7a"
UNI, ZERO = ROOT / "results/synthetic_K5_alpha500_L5", ROOT / "results/synthetic_K5_alpha500_zeroinit"
OUT = ZERO / "readable"
NAMES = ["trend", "tone", "AM tone", "chirp"]


def uvmd(x, P, init, layers=N):
    r = vmd(x, alpha=P["alpha"], K=K, DC=True, init=init, lambda_schedule=P["lambda"][:layers])
    return r["u"], r["omega"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = make_synthetic(200, 1024, 0.1, 0)
    _, te = train_test_split(np.arange(200), test_size=0.3, random_state=42, shuffle=True)
    _, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
    ld = lambda d: (lambda z: {"alpha": z["alpha"], "lambda": z["lambda"]})(np.load(d / f"uvmd_params_N{N}.npz"))
    Pu, Pz = ld(UNI), ld(ZERO)
    start = {"uniform": 0.5 / K * np.arange(K), "zero": np.zeros(K)}
    # centre frequencies after 0..N layers, mean over test signals (modes sorted by frequency)
    traj = {}
    for key, P, init in (("UVMD uniform start", Pu, 1), ("UVMD zero start", Pz, 0)):
        T = [start["uniform" if init else "zero"]]
        for n in range(1, N + 1):
            T.append(np.mean([np.sort(uvmd(D["x"][i], P, init, n)[1]) for i in te], axis=0))
        traj[key] = np.array(T)                       # (N+1, K)
    vm = {}
    for key, init in (("VMD converged, uniform start", 1), ("VMD converged, zero start", 0)):
        vm[key] = np.mean([np.sort(vmd(D["x"][i], ALPHA, 0.0, K, True, init, 1e-7, 499)["omega"]) for i in te], axis=0)
    truth_fc = D["centre_freq"][te].mean(0)           # trend, tone, AM, chirp (mean)

    rows = []
    for key, T in traj.items():
        rows.append({"method": key, **{f"mode {k+1}": f"{T[0,k]:.3f} -> {T[-1,k]:.3f}" for k in range(K)},
                     "total movement": f"{np.abs(T[-1]-T[0]).sum():.3f}"})
    for key, f in vm.items():
        s = start["uniform" if "uniform" in key else "zero"]
        rows.append({"method": key, **{f"mode {k+1}": f"{s[k]:.3f} -> {f[k]:.3f}" for k in range(K)},
                     "total movement": f"{np.abs(f-s).sum():.3f}"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "freq_movement.csv", index=False)
    (OUT / "freq_movement.md").write_text(
        "# Centre frequencies, start -> end (cycles/sample, mean over 30 test signals, modes sorted)\n\n"
        f"True bands (mean): " + ", ".join(f"{n} {f:.3f}" for n, f in zip(NAMES, truth_fc)) + "\n\n"
        + df.to_markdown(index=False) + "\n")
    print(df.to_string())
    print("true", truth_fc)

    # ---- figure: trajectory over layers
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (key, T), c in zip(axs, traj.items(), (UVMD_C, GREEN)):
        for k in range(K):
            ax.plot(range(N + 1), T[:, k], color=c, lw=2, marker="o", ms=4)
        for n, f in zip(NAMES, truth_fc):
            ax.axhline(f, color=TRUTH, lw=1, ls="--")
            ax.text(N + 0.08, f, n, fontsize=9, va="center", color=INK)
        ax.set_title(key, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("UVMD layer (0 = start)", color=MUTED)
        ax.set_xlim(-0.1, N + 1)
        _style(ax, bottom=True)
    axs[0].set_ylabel("centre frequency (cycles/sample)", color=MUTED)
    fig.suptitle("How far the centre frequencies move through the 5 layers (dashed = true bands)", x=0.005, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "freq_movement.png", dpi=140); plt.close(fig)

    # ---- side-by-side modes for the median-difficulty test signal (by UVMD-zero error)
    def err(u, i): return np.mean([np.sqrt(np.mean((t - m) ** 2)) / np.std(t) for t, m in zip(D["modes"][i], match_modes(u, D["modes"][i]))])
    dec = {i: {"VMD": vmd(D["x"][i], ALPHA, 0.0, K, True, 1, 1e-7, 499)["u"], "UVMD uniform": uvmd(D["x"][i], Pu, 1)[0],
               "UVMD zero": uvmd(D["x"][i], Pz, 0)[0]} for i in te}
    order = sorted(te, key=lambda i: err(dec[i]["UVMD zero"], i))
    i = order[len(order) // 2]
    truth, t = D["modes"][i], np.arange(1024)
    cols = [("Original", None, TRUTH), ("VMD (converged)", "VMD", VMD_C), ("UVMD, uniform start", "UVMD uniform", UVMD_C),
            ("UVMD, zero start", "UVMD zero", GREEN)]
    J = len(truth)
    fig, axes = plt.subplots(J + 1, 4, figsize=(17, 1.5 * (J + 1) + 1), sharex=True)
    for c, (title, key, col) in enumerate(cols):
        axes[0, c].set_title(title, loc="left", fontsize=12, color=col, fontweight="bold")
        if key:
            u = dec[i][key]; m = match_modes(u, truth); used = _matched_idx(u, truth)
        for j in range(J):
            ax = axes[j, c]; lim = 1.15 * np.abs(truth[j]).max()
            if key is None:
                ax.plot(t, truth[j], color=TRUTH, lw=1.3); ax.set_ylabel(NAMES[j], color=INK)
            else:
                ax.plot(t, truth[j], color=FAINT, lw=1.6); ax.plot(t, m[j], color=col, lw=1.1)
                e = np.sqrt(np.mean((truth[j] - m[j]) ** 2)) / np.std(truth[j])
                ax.text(0.99, 0.93, f"error {100*e:.0f} %", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK, bbox=BOX)
            ax.set_ylim(-lim, lim); _style(ax)
        ax = axes[J, c]
        if key is None:
            ax.axis("off"); ax.text(0.02, 0.5, "no true mode (K=5, 4 true)", transform=ax.transAxes, color=MUTED)
        else:
            ex = [k for k in range(K) if k not in used]
            for k in ex: ax.plot(t, u[k], color=col, lw=1)
            ax.set_ylim(-1.15 * np.abs(truth).max(), 1.15 * np.abs(truth).max())
            share = sum(np.sum(u[k] ** 2) for k in ex) / np.sum(u ** 2)
            ax.text(0.99, 0.93, f"extra mode: {100*share:.1f} % of energy", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK, bbox=BOX)
            _style(ax, bottom=True); ax.set_xlabel("sample", color=MUTED)
    fig.text(0.005, 0.005, "Grey line = true mode. Error = RMSE to the true mode as % of its std. One test signal (median of the zero-start errors).",
             fontsize=8.5, color=MUTED)
    fig.suptitle("Synthetic, K = 5, alpha = 500: original, VMD, UVMD (uniform start) and UVMD (zero start)", x=0.005, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97)); fig.savefig(OUT / "modes_side_by_side.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
