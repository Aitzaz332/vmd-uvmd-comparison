"""Reconstruction and separation side by side for the three datasets, plus the synthetic ground truth.

Rebuilds the trained UVMD-N models from the ``uvmd_params_N*.npz`` of the benchmark runs (no retraining), decomposes
the SAME test signals as those runs with converged VMD, VMD-N and trained UVMD-N, and computes for every signal

  reconstruction   relative reconstruction error   ||x - sum_k u_k|| / ||x||                       lower = better
  separation       orthogonality index             sum_{i<j} |<u_i,u_j>| / ||x||^2                 lower = better
                   spectral overlap                mean over active mode pairs of sum_f min(p_i, p_j), p = power
                                                   spectrum normalised to sum 1 (0 = disjoint)      lower = better
                   relative bandwidth              energy-weighted spectral std of each mode, / Nyquist (cycles/sample / 0.5)
                                                                                                    lower = better
                   energy ratio                    sum_k ||u_k||^2 / ||x||^2                       closer to 1 = better
                   effective number of modes       1 / sum_k s_k^2, s_k = energy share of mode k   guard, no direction
                                                   (1 = all energy in one mode; K = spread evenly)

None of these needs ground truth, so they are comparable across datasets (use the same K everywhere). The synthetic
signals have true modes, so for them the script also reports the ground-truth criteria (grouped mode SNR, mode
correlation, centre-frequency error, spurious energy share) and how well each criterion above tracks them.

    python experiments/separation_comparison.py                       # all three datasets, default result folders
    python experiments/separation_comparison.py --datasets synthetic  # one dataset
    python experiments/separation_comparison.py --n-test 5            # quick smoke run

Writes results/separation_comparison/: separation_comparison.md (all tables), comparison_bars.png,
synthetic_proxy_vs_truth.png and CSV files. A dataset whose data or parameters are missing is skipped.
Runtime is dominated by converged VMD (about 0.4 s per synthetic signal or CWRU window, 13 s per traffic sensor).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from decomp.metrics import evaluate  # noqa: E402
from decomp.traffic_metrics import spectral_metrics  # noqa: E402
from experiments import omega_shift_all as osa  # noqa: E402

BASE = osa.BASE
DATASETS = osa.DATASETS
# key, label, direction ("low" / "one" / "none"), bar axis scale, kind
CRITERIA = [
    ("recon_rel_err", "Relative reconstruction error", "low", "log", "reconstruction"),
    ("ortho_index", "Orthogonality index", "low", "log", "separation"),
    ("mode_overlap", "Spectral overlap", "low", "log", "separation"),
    ("bandwidth_rel", "Relative bandwidth (÷ Nyquist)", "low", "log", "separation"),
    ("energy_ratio", "Energy ratio", "one", "linear", "separation"),
    ("n_eff_modes", "Effective number of modes", "none", "linear", "separation"),
]
UP, DOWN = "↑", "↓"
ARROW = {"low": DOWN, "one": "(→ 1)", "none": "(guard)"}
TRUTH = [  # key, label, higher is better
    ("grouped_mean_snr_db", "Grouped mode SNR (dB)", True),
    ("mean_mode_corr", "Mode correlation", True),
    ("mean_omega_err", "Centre-freq. error (cycles/sample)", False),
    ("spurious_energy_share", "Spurious energy share", False),
]


# ---------------------------------------------------------------------------------------------- per-signal criteria
def signal_criteria(x, u, omega=None, truth=None):
    """All criteria for one decomposition u (K,T) of one signal x (T,)."""
    u = np.asarray(u, dtype=np.float64)
    T = min(len(x), u.shape[1])
    x, u = np.asarray(x, dtype=np.float64)[:T], u[:, :T]
    kw = {}
    if truth is not None:
        kw = {"clean": truth["clean"], "true_modes": truth["modes"], "true_fc": truth["fc"]}
    m = evaluate(x, u, omega=omega, **kw)
    sm = spectral_metrics(x, u, steps_per_day=1, lines={})          # bandwidth in cycles/sample
    m["mode_overlap"] = sm["mode_overlap"]
    m["bandwidth_rel"] = sm["mean_bandwidth_cpd"] / 0.5
    m["n_active_modes"] = sm["n_active_modes"]
    e = (u ** 2).sum(axis=1)
    s = e / e.sum()
    m["n_eff_modes"] = float(1.0 / np.sum(s ** 2))
    return m


# ---------------------------------------------------------------------------------------------- running the methods
def load_dataset(key, folder, data_path, n_test):
    X, te, d = osa.LOADERS[key](folder, data_path)
    te = np.asarray(te)
    if key == "traffic" and d.get("max_test_nodes"):                  # the benchmark scored only these test sensors
        te = te[:d["max_test_nodes"]]
    if n_test:
        te = te[:n_test]
    truth = None
    if key == "synthetic":
        from datasets.synthetic import make_synthetic
        truth = make_synthetic(d["n_signals"], d["T"], d["noise"], d["seed"])
    return X, te, d, truth


def run_dataset(key, folder, data_path, a):
    from decomp import uvmd_magn
    from decomp.vmd_numpy import vmd
    from experiments.omega_shift import rebuild

    X, te, d, truth = load_dataset(key, folder, data_path, a.n_test)
    label = DATASETS[key][0]
    K, T = d["K"], X.shape[1]
    alpha, tau = d.get("alpha", 2000.0), d.get("tau", 0.0)
    layers = d.get("layers", [1, 2])
    conv = te[:a.n_converged] if a.n_converged else te
    rows = []

    def tr(i):
        return None if truth is None else {"clean": truth["clean"][i], "modes": truth["modes"][i],
                                           "fc": truth["centre_freq"][i]}

    def add(method, i, u, om):
        m = signal_criteria(X[i], u, om, tr(i))
        m.update(dataset=label, dataset_key=key, method=method, signal=int(i))
        rows.append(m)

    def progress(method, j, n, t0):
        if n >= 20 and (j + 1) % max(1, n // 10) == 0:
            print(f"    {method}: {j + 1}/{n}  ({time.perf_counter() - t0:.0f} s)", flush=True)

    print(f"== {label}  K={K}  T={T}  test signals {len(te)} (converged VMD on {len(conv)})", flush=True)
    t0 = time.perf_counter()
    for j, i in enumerate(conv):
        r = vmd(X[i], alpha=alpha, tau=tau, K=K, DC=True, init=1, tol=d.get("tol", 1e-7), max_iter=d.get("max_iter", 499))
        add(BASE, i, r["u"], r["omega"])
        progress(BASE, j, len(conv), t0)
    for N in layers:
        for i in te:
            r = vmd(X[i], alpha=alpha, tau=tau, K=K, DC=True, init=1, tol=-1.0, max_iter=N)
            add(f"VMD-{N}", i, r["u"], r["omega"])
    for N in layers:
        f = folder / f"uvmd_params_N{N}.npz"
        if not f.exists():
            print(f"  missing {f}, UVMD-{N} skipped")
            continue
        model = rebuild(f, T, K, N, a.device)
        for i in te:
            u, om = uvmd_magn.decompose(model, X[i], a.device)
            add(f"UVMD-{N}", i, u, om)
    print(f"  done in {time.perf_counter() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows), d


# ---------------------------------------------------------------------------------------------- aggregation + tables
def method_order(layers):
    return [BASE] + [m for N in layers for m in (f"VMD-{N}", f"UVMD-{N}")]


def aggregate(df, cols, datasets, methods):
    """mean, std and standard error per dataset and method, rows in the order of ``datasets`` x ``methods``"""
    g = df.groupby(["dataset", "method"], sort=False)
    res = pd.concat([g.size().rename("n"), g[cols].mean(), g[cols].std(ddof=1).add_suffix("_std")], axis=1).reset_index()
    for c in cols:
        res[c + "_sem"] = res[c + "_std"] / np.sqrt(res["n"])
    res["_d"] = res["dataset"].map({d: i for i, d in enumerate(datasets)})
    res["_m"] = res["method"].map({m: i for i, m in enumerate(methods)})
    return res.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"]).reset_index(drop=True)


def fmt(mean, std=None):
    if mean is None or not np.isfinite(mean):
        return "-"
    return f"{mean:.3g}" + (f" ± {std:.2g}" if std is not None and np.isfinite(std) else "")


def best_row(g, key, direction):
    if direction == "none" or g[key].isna().all():
        return None
    score = (g[key] - 1).abs() if direction == "one" else g[key]
    return g.loc[score.idxmin(), "method"]


def table_main(agg, datasets, note_k):
    head = ["Dataset", "Method", "Signals"] + [f"{lab} {ARROW[dr]}" for _, lab, dr, _, _ in CRITERIA]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for ds in datasets:
        g = agg[agg.dataset == ds].reset_index(drop=True)
        best = {k: best_row(g, k, dr) for k, _, dr, _, _ in CRITERIA}
        for _, r in g.iterrows():
            cells = []
            for k, _, _, _, _ in CRITERIA:
                s = fmt(r[k], r[k + "_std"])
                cells.append(f"**{s}**" if best[k] == r["method"] else s)
            md.append("| " + " | ".join([f"{ds} (K={note_k[ds]})", r["method"], str(int(r["n"]))] + cells) + " |")
    return "\n".join(md)


def _better(a, b, direction):
    """per-signal paired comparison: 1 if a better than b, 0.5 tie, 0 otherwise"""
    if direction == "one":
        a, b = (a - 1).abs(), (b - 1).abs()
    return ((a < b).astype(float) + 0.5 * (a == b)).mean()


def table_training_effect(df, datasets, layers):
    crit = [c for c in CRITERIA if c[2] != "none"]
    head = ["Dataset", "Layers"] + [f"{lab} {ARROW[dr]}" for _, lab, dr, _, _ in crit]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for ds in datasets:
        for N in layers:
            u = df[(df.dataset == ds) & (df.method == f"UVMD-{N}")].set_index("signal")
            v = df[(df.dataset == ds) & (df.method == f"VMD-{N}")].set_index("signal")
            j = u.index.intersection(v.index)
            if len(j) == 0:
                continue
            cells = []
            for k, _, dr, _, _ in crit:
                a_, b_ = u.loc[j, k], v.loc[j, k]
                if dr == "one":                      # compare the distance from 1, not the value
                    ratio = (a_ - 1).abs().mean() / (b_ - 1).abs().mean()
                else:
                    ratio = a_.mean() / b_.mean() if b_.mean() else np.nan
                cells.append(f"{ratio:.2f}× ({100 * _better(a_, b_, dr):.0f}%)")
            md.append("| " + " | ".join([ds, str(N)] + cells) + " |")
    return "\n".join(md)


def table_truth(df_syn, datasets, methods):
    truth = [t for t in TRUTH if t[0] in df_syn.columns and df_syn[t[0]].notna().any()]
    agg = aggregate(df_syn, [t[0] for t in truth] + [c[0] for c in CRITERIA], datasets, methods)
    head = ["Method"] + [f"{lab} {UP if hib else DOWN}" for _, lab, hib in truth] + \
           [f"{lab} {ARROW[dr]}" for _, lab, dr, _, _ in CRITERIA]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    best = {k: agg.loc[agg[k].idxmax() if hib else agg[k].idxmin(), "method"] for k, _, hib in truth}
    best.update({k: best_row(agg, k, dr) for k, _, dr, _, _ in CRITERIA})
    keys = [t[0] for t in truth] + [c[0] for c in CRITERIA]
    for _, r in agg.iterrows():
        cells = []
        for k in keys:
            s = fmt(r[k], r[k + "_std"])
            cells.append(f"**{s}**" if best.get(k) == r["method"] else s)
        md.append("| " + " | ".join([r["method"]] + cells) + " |")
    return "\n".join(md), agg


def spearman_table(df_syn):
    """How well does each criterion track the true separation quality (grouped mode SNR)? Pooled over signals x methods."""
    rows = []
    for k, lab, dr, _, _ in CRITERIA:
        v = df_syn[k].values.astype(float)
        y = df_syn["grouped_mean_snr_db"].values.astype(float)
        if dr == "one":
            v = np.abs(v - 1)
        ok = np.isfinite(v) & np.isfinite(y)
        v, y = v[ok], y[ok]
        rho = spearmanr(v, y).statistic
        exp = {"low": -1, "one": -1, "none": 0}[dr]
        if dr == "none":
            verdict = "guard (no direction)"
        elif abs(rho) < 0.2:
            verdict = "no relation"
        elif np.sign(rho) == exp:
            verdict = "tracks truth" if abs(rho) >= 0.5 else "weakly tracks truth"
        else:
            verdict = "misleading (opposite sign)"
        rows.append({"criterion": lab, "key": k, "spearman_rho": float(rho), "n_points": int(len(y)), "verdict": verdict})
    out = pd.DataFrame(rows)
    md = ["| Criterion | Spearman ρ with true mode SNR | Reads as |", "|---|---|---|"]
    md += [f"| {r.criterion} {ARROW[dict((c[0], c[2]) for c in CRITERIA)[r.key]]} | {r.spearman_rho:+.2f} | {r.verdict} |"
           for r in out.itertuples()]
    return "\n".join(md), out


# ---------------------------------------------------------------------------------------------- figures
def color(m):
    if m == BASE:
        return "#4d4d4d"
    n = int(m.rsplit("-", 1)[1])
    if m.startswith("VMD-"):
        return {1: "#b3b3b3", 2: "#dcdcdc"}.get(n, "#eeeeee")
    return {1: "#d95f02", 2: "#fdae6b"}.get(n, "#fee6ce")


def figure_bars(agg, datasets, methods, path, K_text):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    w = 0.82 / len(methods)
    for ax, (k, lab, dr, scale, kind) in zip(axes.ravel(), CRITERIA):
        vals = []
        for mi, m in enumerate(methods):
            xs, ys, es = [], [], []
            for di, ds in enumerate(datasets):
                r = agg[(agg.dataset == ds) & (agg.method == m)]
                if r.empty:
                    continue
                xs.append(di + (mi - (len(methods) - 1) / 2) * w)
                ys.append(float(r[k].iloc[0]))
                es.append(float(r[k + "_sem"].iloc[0]) if np.isfinite(r[k + "_sem"].iloc[0]) else 0.0)
            ys_a, es_a = np.array(ys), np.array(es)
            lo = np.minimum(es_a, 0.95 * ys_a) if scale == "log" else es_a
            ax.bar(xs, np.maximum(ys_a, 1e-12) if scale == "log" else ys_a, width=w * 0.95, color=color(m),
                   edgecolor="k", linewidth=0.5, yerr=[lo, es_a], error_kw={"elinewidth": 0.8, "capsize": 2})
            vals += [v for v in ys if v > 0]
        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylim(min(vals) / 2.5, max(vals) * 2.5)
        if dr == "one":
            ax.axhline(1, color="k", ls="--", lw=0.9)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets)
        ax.set_title(f"{kind.upper()}\n{lab} {ARROW[dr]}", fontsize=11,
                     color="#1f4e79" if kind == "reconstruction" else "black")
        if kind == "reconstruction":
            ax.set_facecolor("#eef4fb")
        ax.grid(alpha=0.25, axis="y", which="both")
    fig.legend([Patch(facecolor=color(m), edgecolor="k", linewidth=0.5) for m in methods], methods, loc="lower center",
               ncol=len(methods), frameon=False, fontsize=10)
    fig.suptitle(f"VMD vs UVMD, {K_text}: reconstruction (shaded) and separation. Bars = mean over test signals, "
                 "whiskers = standard error", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_truth(df_syn, spear, methods, path):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    rho = dict(zip(spear.key, spear.spearman_rho))
    for ax, (k, lab, dr, scale, kind) in zip(axes.ravel(), CRITERIA):
        for m in methods:
            g = df_syn[df_syn.method == m]
            ax.scatter(g[k], g["grouped_mean_snr_db"], s=26, color=color(m), edgecolor="k", linewidth=0.4, label=m,
                       alpha=0.9)
        if scale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_tick_params(which="minor", labelsize=7)
        ax.set_xlabel(f"{lab} {ARROW[dr]}")
        ax.set_ylabel("true separation: grouped mode SNR (dB) ↑")
        ax.set_title(f"ρ = {rho[k]:+.2f}", fontsize=11)
        ax.grid(alpha=0.25, which="both")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Synthetic data: does each criterion track the true separation quality? One dot = one test signal and method",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- consistency check
def check_against_benchmark(df, folder, label):
    """compare the mean relative reconstruction error with the one stored by the benchmark run"""
    try:
        summ = json.loads((folder / "summary.json").read_text())["summary"]
    except (OSError, ValueError, KeyError):
        return None
    worst, n_cmp = 0.0, 0
    for m, g in df[df.dataset == label].groupby("method"):
        ref = summ.get(m, {}).get("recon_rel_err")
        if ref:
            worst = max(worst, abs(g["recon_rel_err"].mean() - ref) / ref)
            n_cmp += 1
    return (worst, n_cmp) if n_cmp else None


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    for k, (lab, default) in DATASETS.items():
        ap.add_argument(f"--{k}", default=default, help=f"result folder of the {lab} run (default {default})")
    ap.add_argument("--cwru-data", default=None, help="CWRU .mat folder (default: the path stored in the run)")
    ap.add_argument("--traffic-data", default=None, help="region .h5 (default: the path stored in the run)")
    ap.add_argument("--n-test", type=int, default=0, help="limit the test signals per dataset (0 = all of the benchmark's)")
    ap.add_argument("--n-converged", type=int, default=0, help="limit the signals for converged VMD (0 = all)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/separation_comparison")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    frames, args_by = [], {}
    for k in [s.strip() for s in a.datasets.split(",") if s.strip()]:
        folder = Path(getattr(a, k))
        folder = folder if folder.is_absolute() else ROOT / folder
        if not (folder / "summary.json").exists():
            print(f"[skip] {DATASETS[k][0]}: no summary.json in {folder}")
            continue
        try:
            df, d = run_dataset(k, folder, {"cwru": a.cwru_data, "traffic": a.traffic_data}.get(k), a)
        except (FileNotFoundError, OSError) as e:
            print(f"[skip] {DATASETS[k][0]}: data not found ({e})")
            continue
        frames.append(df)
        args_by[k] = (folder, d)
    if not frames:
        sys.exit("nothing was analysed")

    df = pd.concat(frames, ignore_index=True)
    datasets = list(dict.fromkeys(df.dataset))
    layers = next(iter(args_by.values()))[1].get("layers", [1, 2])
    methods = [m for m in method_order(layers) if m in set(df.method)]
    note_k = {DATASETS[k][0]: args_by[k][1]["K"] for k in args_by}
    ks = sorted(set(note_k.values()))
    agg = aggregate(df, [c[0] for c in CRITERIA], datasets, methods)

    text = ["# Reconstruction and separation across the datasets", ""]
    if len(ks) > 1:
        text += [f"**Warning: different K across datasets ({ks}). The orthogonality index grows with the number of mode "
                 "pairs, so compare datasets only with care.**", ""]
    text += ["## Table A. All datasets", "",
             "Mean ± standard deviation over the test signals of each dataset (the same signals as in the benchmark runs). "
             "Bold = best of the rows of that dataset. VMD-N is also UVMD-N before training.", "",
             table_main(agg, datasets, note_k), "",
             "* **Reconstruction:** relative error of the sum of the modes against the input.",
             "* **Orthogonality index:** content shared between modes. **Spectral overlap:** share of spectrum two modes "
             "occupy together (0 = disjoint). **Relative bandwidth:** width of each mode's spectrum as a fraction of the Nyquist "
             "frequency (energy-weighted).",
             "* **Energy ratio:** total mode energy over input energy; above 1 means the modes double count, below 1 that "
             "energy is missing.",
             "* **Effective number of modes** is a guard, not a score: a low orthogonality index or overlap is meaningless if "
             "one mode holds almost all the energy (value near 1).", ""]
    effect = table_training_effect(df, datasets, layers)
    text += ["## Table A2. What training changed: trained UVMD-N relative to VMD-N", "",
             "Ratio of the means, UVMD-N ÷ VMD-N, and in brackets the share of signals on which trained UVMD is better. "
             "Below 1× is better for every column (for the energy ratio it is the distance from 1 that is compared). Same "
             "number of layers or iterations, so the only difference is training.", "", effect, ""]
    check = []
    for k, (folder, d) in args_by.items():
        c = check_against_benchmark(df, folder, DATASETS[k][0])
        if c is not None and not a.n_test and not a.n_converged:
            check.append(f"{DATASETS[k][0]} {100 * c[0]:.2g}%")
    if check:
        text += ["Check: the mean relative reconstruction error recomputed here reproduces the one stored in each benchmark run's "
                 "summary.json (largest relative difference over the methods: " + ", ".join(check) + ").", ""]

    syn = df[df.dataset_key == "synthetic"]
    spear = None
    if not syn.empty and "grouped_mean_snr_db" in syn.columns:
        tt, tagg = table_truth(syn, list(dict.fromkeys(syn.dataset)), methods)
        sp_md, spear = spearman_table(syn)
        text += ["## Table B. Synthetic data: ground-truth criteria next to the criteria of Table A", "",
                 "The synthetic signals are built from 4 known modes, so separation can be measured against the truth. With K > 4 "
                 "the grouped mode SNR is used: each estimated mode is assigned to the true mode it projects onto most, and the "
                 "assigned modes are summed. Mean ± standard deviation over the test signals.", "", tt, "",
                 "## Table C. Do the criteria of Table A track the true separation quality?", "",
                 f"Spearman rank correlation between each criterion and the grouped mode SNR, over all {len(syn)} points "
                 "(test signals × methods). For a criterion where lower is better, a good one has a strongly negative ρ. "
                 "Reads as: |ρ| ≥ 0.5 tracks truth, 0.2 to 0.5 weakly, below 0.2 no relation; the opposite sign is "
                 "misleading.", "", sp_md, ""]
        tagg.to_csv(out / "table_synthetic_truth.csv", index=False)
        spear.to_csv(out / "synthetic_spearman.csv", index=False)
        figure_truth(syn, spear, methods, out / "synthetic_proxy_vs_truth.png")
    (out / "separation_comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    agg.to_csv(out / "table_all_datasets.csv", index=False)
    df.to_csv(out / "per_signal.csv", index=False)
    figure_bars(agg, datasets, methods, out / "comparison_bars.png", "K=" + "/".join(map(str, ks)))
    print("\n".join(text))
    print(f"wrote {out}/separation_comparison.md, comparison_bars.png, synthetic_proxy_vs_truth.png and CSV files")


if __name__ == "__main__":
    main()
