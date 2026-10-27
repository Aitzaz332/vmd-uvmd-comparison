"""Centre-frequency movement across the three datasets (synthetic, CWRU, LargeST traffic).

Question: with only 1-2 layers, do the mode centre frequencies omega_k of UVMD move away from the uniform
starting grid (omega_k = 0.5/K * k cycles/sample), and how does that compare with VMD stopped after the same
number of iterations and with converged VMD?

For every dataset this script loads the SAME test signals as the benchmark run (same split), rebuilds the trained
UVMD-N models from the ``uvmd_params_N*.npz`` the run saved (no retraining), decomposes the test signals, and
measures how far the final omega_k are from the start grid. Distances are given in *grid spacings*
(0.5/K cycles/sample), so datasets with different signal lengths and frequency ranges are comparable.

omega_k of the upstream UVMD is not a learned parameter: every layer recomputes it as the power-weighted mean
frequency of mode k (mode 0 is the DC mode and never moves). UVMD can only change it indirectly, through the learned
alpha_k and lambda.

    python experiments/omega_shift_all.py                                  # all three, default result folders
    python experiments/omega_shift_all.py --datasets synthetic             # only one
    python experiments/omega_shift_all.py --traffic-data ~/data/LargeST/sd_his_2019.h5 --device cuda:0

Writes results/omega_shift_all/omega_shift_all.md, .csv and .png. A dataset whose data or parameters are not
found is skipped with a message.
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
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {  # key: (label, default result folder)
    "synthetic": ("Synthetic", "results/synthetic_K13"),
    "cwru": ("CWRU", "results/cwru_K13"),
    "traffic": ("LargeST traffic", "results/traffic_sd_bounded"),
}
BASE = "VMD (converged)"
MOVED = 0.25      # a mode counts as "moved" when its mean shift exceeds this many grid spacings


# ---------------------------------------------------------------------------------------------- statistics
def shift_stats(omega, K):
    """omega: (n_signals, K) final centre frequencies in cycles/sample -> movement from the uniform start grid,
    in grid spacings. Mode 0 is the DC mode and never moves, so it is left out."""
    omega = np.asarray(omega, dtype=np.float64)
    init = 0.5 / K * np.arange(K)
    sh = np.abs(omega[:, 1:] - init[None, 1:]) / (0.5 / K)
    per_mode = sh.mean(axis=0)
    return {"mean_shift": float(sh.mean()), "max_mode_shift": float(per_mode.max()),
            "modes_moved": int((per_mode > MOVED).sum()), "n_modes": int(K - 1), "n_signals": int(len(omega))}


def build_table(results):
    """results: {label: {"K", "omega": {method: (n,K)}}} -> DataFrame, one row per dataset and method."""
    rows = []
    for label, R in results.items():
        stats = {m: shift_stats(om, R["K"]) for m, om in R["omega"].items()}
        base = stats.get(BASE, {}).get("mean_shift")
        for m, s in stats.items():
            rows.append({"dataset": f"{label} (K={R['K']})", "method": m, **s,
                         "share_of_converged": (s["mean_shift"] / base) if base else np.nan})
    return pd.DataFrame(rows)


def markdown(df):
    head = ["Dataset", "Method", "Mean shift (grid spacings)", "Largest single-mode shift",
            f"Modes moved > {MOVED} spacing", "Movement ÷ converged VMD", "Signals"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        share = "-" if not np.isfinite(r["share_of_converged"]) else f"{r['share_of_converged']:.2f}×"
        md.append(f"| {r['dataset']} | {r['method']} | {r['mean_shift']:.3f} | {r['max_mode_shift']:.3f} | "
                  f"{r['modes_moved']} of {r['n_modes']} | {share} | {r['n_signals']} |")
    return "\n".join(md)


# ---------------------------------------------------------------------------------------------- data
def _summary_args(folder):
    return json.loads((folder / "summary.json").read_text())["args"]


def load_synthetic(folder, data_path):
    from datasets.synthetic import make_synthetic
    from sklearn.model_selection import train_test_split
    d = _summary_args(folder)
    D = make_synthetic(d["n_signals"], d["T"], d["noise"], d["seed"])
    tr, te = train_test_split(np.arange(d["n_signals"]), test_size=0.3, random_state=42, shuffle=True)
    va, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
    return D["x"], te, d


def load_cwru_set(folder, data_path):
    from datasets.cwru import load_cwru, split_indices
    d = _summary_args(folder)
    D = load_cwru(data_path or d["data"], d["T"], d["windows_per_file"], not d.get("no_standardize", False))
    tr, va, te = split_indices(D, d["split"])
    return D["x"], te, d


def load_traffic_set(folder, data_path):
    from datasets.traffic import load_traffic, split_nodes
    d = _summary_args(folder)
    D = load_traffic(data_path or d["data"], d.get("length"))
    tr, va, te = split_nodes(D["x"].shape[0])
    return D["x"], te, d


LOADERS = {"synthetic": load_synthetic, "cwru": load_cwru_set, "traffic": load_traffic_set}


# ---------------------------------------------------------------------------------------------- decompositions
def decompose_all(X, te, d, folder, n_test, n_converged, device):
    """final omega (n, K) in cycles/sample for VMD-N, converged VMD and the trained UVMD-N."""
    from decomp import uvmd_magn
    from decomp.vmd_numpy import vmd
    from experiments.omega_shift import rebuild

    K, T = d["K"], X.shape[1]
    alpha, tau = d.get("alpha", 2000.0), d.get("tau", 0.0)
    layers = d.get("layers", [1, 2])
    te = np.asarray(te)
    te = te[np.linspace(0, len(te) - 1, min(n_test, len(te))).astype(int)]     # evenly spread over the test set
    om = {}
    for N in layers:
        om[f"VMD-{N}"] = np.array([vmd(X[i], alpha=alpha, tau=tau, K=K, DC=True, tol=-1, max_iter=N)["omega"]
                                   for i in te])
    sub = te[:n_converged]
    om[BASE] = np.array([vmd(X[i], alpha=alpha, tau=tau, K=K, DC=True, tol=d.get("tol", 1e-7),
                             max_iter=d.get("max_iter", 499))["omega"] for i in sub])
    for N in layers:
        f = folder / f"uvmd_params_N{N}.npz"
        if not f.exists():
            print(f"  missing {f}, UVMD-{N} skipped")
            continue
        model = rebuild(f, T, K, N, device)
        om[f"UVMD-{N}"] = np.array([uvmd_magn.decompose(model, X[i], device)[1] for i in te])
    order = [BASE] + [m for N in layers for m in (f"VMD-{N}", f"UVMD-{N}")]
    return {m: om[m] for m in order if m in om}, K


# ---------------------------------------------------------------------------------------------- figure
def _color(m):
    return "#4d4d4d" if m == BASE else ("#b3b3b3" if m.startswith("VMD-") else "#d95f02")


def figure(results, df, path):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(max(5.2 * n, 9.0), 8.4), squeeze=False)
    for c, (label, R) in enumerate(results.items()):
        K = R["K"]
        k = np.arange(K)
        top, bot = axes[0][c], axes[1][c]
        top.plot(k, 0.5 / K * k, "k--", lw=1, label="start grid")
        for m, om in R["omega"].items():
            top.plot(k, om.mean(axis=0), marker="o", ms=3, lw=1.3, color=_color(m),
                     ls="--" if m.endswith("-2") else "-", label=m)
        top.set_xlabel("mode k")
        top.set_ylabel("final centre frequency ω_k (cycles/sample)")
        top.set_title(f"{label} (K={K})")
        top.grid(alpha=0.25)
        if c == 0:
            top.legend(fontsize=8)
        g = df[df.dataset == f"{label} (K={K})"]
        bot.bar(range(len(g)), g["mean_shift"], color=[_color(m) for m in g["method"]], edgecolor="k", linewidth=0.5)
        for i, v in enumerate(g["mean_shift"]):
            bot.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        bot.set_xticks(range(len(g)))
        bot.set_xticklabels(g["method"], rotation=30, ha="right", fontsize=8)
        bot.set_ylabel("mean shift from start grid (grid spacings)")
        bot.set_title("how far the centre frequencies moved", fontsize=10)
        bot.grid(alpha=0.25, axis="y")
    fig.suptitle("Centre-frequency movement\ntop: where ω_k ends up (dashed = start grid)   "
                 "bottom: mean distance from the start grid", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    for k, (lab, default) in DATASETS.items():
        ap.add_argument(f"--{k}", default=default, help=f"result folder of the {lab} run (default {default})")
    ap.add_argument("--cwru-data", default=None, help="CWRU .mat folder (default: the path stored in the run)")
    ap.add_argument("--traffic-data", default=None, help="region .h5 (default: the path stored in the run)")
    ap.add_argument("--n-test", type=int, default=30, help="test signals per dataset")
    ap.add_argument("--n-converged", type=int, default=10, help="signals for converged VMD (slow on traffic)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/omega_shift_all")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for k in [s.strip() for s in a.datasets.split(",") if s.strip()]:
        folder = Path(getattr(a, k))
        folder = folder if folder.is_absolute() else ROOT / folder
        print(f"== {DATASETS[k][0]}  ({folder})")
        if not (folder / "summary.json").exists():
            print("  no summary.json, skipped")
            continue
        data_path = {"cwru": a.cwru_data, "traffic": a.traffic_data}.get(k)
        try:
            X, te, d = LOADERS[k](folder, data_path)
        except (FileNotFoundError, OSError) as e:
            print(f"  data not found ({e}), skipped")
            continue
        om, K = decompose_all(X, te, d, folder, a.n_test, a.n_converged, a.device)
        results[DATASETS[k][0]] = {"K": K, "omega": om}

    if not results:
        sys.exit("nothing was analysed")
    df = build_table(results)
    ks = sorted({R["K"] for R in results.values()})
    text = ["# Centre-frequency movement across the datasets", ""]
    if len(ks) > 1:
        text += [f"**Warning: different K across datasets ({ks}).**", ""]
    text += ["Distance of the final centre frequencies ω_k from the uniform start grid, in grid spacings "
             "(0.5/K cycles/sample), mean over the test signals and modes 1..K-1 (mode 0 is the DC mode and does not move). "
             f"A mode counts as moved when its mean shift exceeds {MOVED} spacing. 'Movement ÷ converged VMD' = mean shift "
             "divided by that of converged VMD, which is computed on fewer signals. Moving more is not the same as moving "
             "to the right place: on the synthetic data compare with the true-frequency error in results/comparison. "
             "VMD-N is also UVMD-N before training.", "",
             markdown(df), "",
             "ω_k is not a learned parameter of UVMD: each layer recomputes it from the mode spectrum, so UVMD can move it "
             "only through the learned alpha_k and lambda.", ""]
    (out / "omega_shift_all.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    df.to_csv(out / "omega_shift_all.csv", index=False)
    rows = [{"dataset": lab, "method": m, "mode": j, "mean_omega_cycles_per_sample": float(v)}
            for lab, R in results.items() for m, om in R["omega"].items() for j, v in enumerate(om.mean(axis=0))]
    pd.DataFrame(rows).to_csv(out / "omega_per_mode.csv", index=False)
    figure(results, df, out / "omega_shift_all.png")
    print("\n".join(text))
    print(f"wrote {out}/omega_shift_all.md, omega_shift_all.csv, omega_per_mode.csv, omega_shift_all.png")


if __name__ == "__main__":
    main()
