"""Reconstruction-error trend: VMD over its ADMM iterations, UVMD over its training epochs.

    python experiments/trend_figures.py synthetic --dir results/synthetic_K5_alpha500_L5
    python experiments/trend_figures.py cwru      --dir results/cwru_K4     --data ~/data/CWRU
    python experiments/trend_figures.py traffic   --dir results/traffic_K6  --data ~/data/LargeST/sd_his_2019.h5

Needs ``uvmd_history_N*.json`` (written by the run_* scripts: the error of the model on the test signals after every
training epoch). The VMD curve is computed here: classic VMD stopped after n = 1, 2, 3 ... 499 iterations.
Writes ``recon_error_trend.png`` (all datasets) and, on synthetic data, ``mode_error_trend.png`` (error against the
true modes) to ``<dir>/readable``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decomp.metrics import evaluate  # noqa: E402
from decomp.vmd_numpy import vmd  # noqa: E402
from experiments.mode_figures import BOX, GRID, INK, MUTED, UVMD_C, VMD_C, _style  # noqa: E402

ITERS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 499]


def eval_set(dataset, S, data):
    """The test signals used for the per-epoch curve (same selection as the run_* scripts)."""
    if dataset == "synthetic":
        from datasets.synthetic import make_synthetic
        D = make_synthetic(S["n_signals"], S["T"], S["noise"], S["seed"])
        _, te = train_test_split(np.arange(S["n_signals"]), test_size=0.3, random_state=42, shuffle=True)
        _, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
        return D["x"][te], (D["modes"][te], D["centre_freq"][te])
    if dataset == "cwru":
        from datasets.cwru import load_cwru, split_indices
        D = load_cwru(data or S["data"], S["T"], S["windows_per_file"], not S.get("no_standardize", False))
        _, _, te = split_indices(D, S["split"])
        return D["x"][te[:20]], None
    from datasets.traffic import load_traffic, split_nodes
    D = load_traffic(data or S["data"], S.get("length"))
    _, _, te = split_nodes(D["x"].shape[0])
    if S.get("max_test_nodes"):
        te = te[: S["max_test_nodes"]]
    return D["x"][te[:10]].astype(np.float64), None


def vmd_curve(X, truth, S, K):
    rel = np.zeros((len(ITERS), len(X)))
    mode = np.zeros_like(rel) if truth else None
    for a, n in enumerate(ITERS):
        for i, x in enumerate(X):
            r = vmd(x, S.get("alpha", 2000.0), S.get("tau", 0.0), K, True, 1, -1.0, n)
            xx = np.asarray(x, dtype=np.float64)[: r["u"].shape[1]]
            rel[a, i] = np.linalg.norm(xx - r["u"].sum(0)) / np.linalg.norm(xx)
            if truth:
                m = evaluate(x, r["u"], omega=r["omega"], true_modes=truth[0][i], true_fc=truth[1][i])
                mode[a, i] = m["mean_mode_nrmse"]
    return rel, mode


def uvmd_curve(hist, key):
    ep = [c["epoch"] for c in hist["curve"]]
    return np.array(ep), np.array([c[key] for c in hist["curve"]])


def _band(ax, x, Y, color, label):
    """Mean over the signals (line) with the middle 50 % of the signals as a light band. Y: (len(x), n_signals)."""
    Yp = 100 * Y
    ax.fill_between(x, np.percentile(Yp, 25, axis=1), np.percentile(Yp, 75, axis=1), color=color, alpha=0.18, lw=0)
    ax.plot(x, Yp.mean(axis=1), color=color, lw=2.2, marker="o", ms=3.5, label=label)


def trend_figure(path, vmd_y, ep, uvmd_y, N, title, ylabel, note, ref_label, logy=True):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    _band(a1, ITERS, vmd_y, VMD_C, "VMD")
    a1.set_xscale("log")
    a1.set_xlabel("ADMM iterations (classic VMD)", color=MUTED)
    a1.set_ylabel(ylabel, color=MUTED)
    a1.set_title("VMD: error against the number of iterations", loc="left", fontsize=11, color=INK)
    k = ITERS.index(N) if N in ITERS else None
    if k is not None:
        a1.plot([N], [100 * vmd_y[k].mean()], "o", ms=9, mfc="white", mec=VMD_C, mew=2)
        a1.annotate(f"{N} iterations\n= untrained UVMD-{N}", (N, 100 * vmd_y[k].mean()), xytext=(0.33, 0.25) if logy else (0.30, 0.82),
                    textcoords="axes fraction", fontsize=9, color=INK, bbox=BOX,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    last = 100 * vmd_y[-1].mean()
    a1.annotate(f"converged ({ITERS[-1]} it.): {last:.1f} %", (ITERS[-1], last), xytext=(-8, 14),
                textcoords="offset points", ha="right", fontsize=9, color=INK, bbox=BOX)
    _band(a2, ep, uvmd_y, UVMD_C, f"UVMD ({N} layers)")
    a2.set_xlabel("training epoch (UVMD)", color=MUTED)
    a2.set_title(f"UVMD ({N} layers): error against the training epoch", loc="left", fontsize=11, color=INK)
    a2.axhline(last, color=VMD_C, lw=1.2, ls="--")
    a2.text(ep[-1], last, f"  {ref_label} {last:.1f} %", color=INK, fontsize=9, va="bottom", ha="right", bbox=BOX)
    a2.annotate(f"epoch 0 = untrained\n{100 * uvmd_y[0].mean():.1f} %", (0, 100 * uvmd_y[0].mean()), xytext=(18, -72 if logy else -46),
                textcoords="offset points", fontsize=9, color=INK, bbox=BOX)
    a2.annotate(f"after training\n{100 * uvmd_y[-1].mean():.2f} %", (ep[-1], 100 * uvmd_y[-1].mean()),
                xytext=(-12, 22 if logy else -40), textcoords="offset points", ha="right", fontsize=9, color=INK, bbox=BOX)
    for ax in (a1, a2):
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(0, 100)
        _style(ax, bottom=True)
    fig.suptitle(title, x=0.005, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.005, 0.005, note, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["synthetic", "cwru", "traffic"])
    p.add_argument("--dir", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--layers", type=int, default=None, help="UVMD depth (default: the deepest with a history file)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    d = Path(a.dir)
    S = json.load(open(d / "summary.json"))["args"]
    K = S["K"]
    depths = sorted(int(f.stem.split("_N")[1]) for f in d.glob("uvmd_history_N*.json"))
    if not depths:
        raise FileNotFoundError(f"no uvmd_history_N*.json in {d}: run the benchmark with the current code first")
    N = a.layers if a.layers in depths else depths[-1]
    hist = json.load(open(d / f"uvmd_history_N{N}.json"))
    X, truth = eval_set(a.dataset, S, a.data)
    out = Path(a.out or d / "readable")
    out.mkdir(parents=True, exist_ok=True)
    rel, mode = vmd_curve(X, truth, S, K)
    ep, u_rel = uvmd_curve(hist, "recon_rel_err")
    name = {"synthetic": "Synthetic", "cwru": "CWRU", "traffic": "Traffic"}[a.dataset]
    sub = f"K = {K}, alpha = {S.get('alpha', 2000):g}, {len(X)} unseen test signals"
    trend_figure(out / "recon_error_trend.png", rel, ep, u_rel, N,
                 f"{name}: reconstruction error of VMD (over iterations) and UVMD (over training epochs), {sub}",
                 "reconstruction error (% of the signal, log scale)",
                 "Reconstruction error = ||signal - sum of modes|| / ||signal||. Line = mean over the test signals, band = "
                 "middle 50 % of the signals. Same vertical scale in both panels.", "VMD converged:")
    if mode is not None:
        _, u_mode = uvmd_curve(hist, "mode_nrmse")
        trend_figure(out / "mode_error_trend.png", mode, ep, u_mode, N,
                     f"{name}: error against the TRUE modes, VMD (over iterations) and UVMD (over epochs), {sub}",
                     "error vs true modes (% of mode size)",
                     "Mean over the four true modes (each recovered mode matched to the true mode it correlates with most). "
                     "Line = mean over the test signals, band = middle 50 %.", "VMD converged:", logy=False)
    print("written to", out)
    print(f"VMD recon error: {100 * rel[0].mean():.1f} % (1 it.) -> {100 * rel[-1].mean():.1f} % ({ITERS[-1]} it.)")
    print(f"UVMD-{N} recon error: {100 * u_rel[0].mean():.1f} % (epoch 0) -> {100 * u_rel[-1].mean():.2f} % (epoch {ep[-1]})")
    if mode is not None:
        print(f"VMD mode error: {100 * mode[-1].mean():.1f} % converged | UVMD-{N}: {100 * u_mode[0].mean():.1f} % -> "
              f"{100 * u_mode[-1].mean():.1f} %")


if __name__ == "__main__":
    main()
