"""UVMD trained with a separation-aware loss, on the three datasets, scored with the criteria of the earlier comparison.

Question: can a term in the TRAINING loss give UVMD better separation, not only better reconstruction?
Variants (decomp/uvmd_sep.py; weights in VARIANTS):
    recon      upstream loss (control; the stored UVMD-N of the benchmark runs is used instead unless asked for)
    bw         + bandwidth of the modes around their centre frequency (VMD's own objective)
    bw+ortho   + orthogonality between modes
    full       + penalty on the norm of lambda and a floor on alpha

The SAME train / validation / test signals as the benchmark runs are used (the split is rebuilt from each run's summary.json),
so the new rows can be put next to converged VMD, VMD-N and the stored UVMD-N. Nothing of upstream is changed.

    python experiments/run_sep_loss.py                                   # all three, device cpu
    python experiments/run_sep_loss.py --datasets synthetic --device cuda:0
    python experiments/run_sep_loss.py --quick                           # smoke run (3 epochs, 12 train signals)

Per dataset and variant the trained parameters and the loss history are saved, and a finished (dataset, variant, N) is
skipped on a rerun (use --force). Writes results/sep_loss/sep_loss.md, bar figures and CSV files.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from decomp import uvmd_magn, uvmd_sep  # noqa: E402
from experiments import omega_shift_all as osa  # noqa: E402
from experiments import separation_comparison as sc  # noqa: E402
from experiments.omega_shift import rebuild  # noqa: E402

BASE = sc.BASE
DATASETS = sc.DATASETS
COLORS = {"VMD": "#4d4d4d", "recon": "#d95f02", "bw": "#1b9e77", "bw+ortho": "#7570b3", "full": "#e7298a"}


def label(N, variant):
    return f"UVMD-{N} [{variant}]"


def split_of(m):
    """'UVMD-2 [bw]' -> (2, 'bw');  'VMD-1' -> (1, None);  BASE -> (None, None)"""
    if m == BASE:
        return None, None
    head, _, var = m.partition(" [")
    return int(head.rsplit("-", 1)[1]), (var[:-1] if var else None)


# ---------------------------------------------------------------------------------------------- data + splits
def get_splits(key, folder, data_path):
    """X, train / validation / test indices and the run's args, rebuilt exactly as the benchmark run did."""
    d = osa._summary_args(folder)
    truth = None
    if key == "synthetic":
        from datasets.synthetic import make_synthetic
        from sklearn.model_selection import train_test_split
        D = make_synthetic(d["n_signals"], d["T"], d["noise"], d["seed"])
        tr, te = train_test_split(np.arange(d["n_signals"]), test_size=0.3, random_state=42, shuffle=True)
        va, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
        X, truth = D["x"], D
    elif key == "cwru":
        from datasets.cwru import load_cwru, split_indices
        D = load_cwru(data_path or d["data"], d["T"], d["windows_per_file"], not d.get("no_standardize", False))
        tr, va, te = split_indices(D, d["split"])
        X = D["x"]
    else:
        from datasets.traffic import load_traffic, split_nodes
        D = load_traffic(data_path or d["data"], d.get("length"))
        X = D["x"]
        tr, va, te = split_nodes(X.shape[0])
        tr, va, te = (s[:c] if c else s for s, c in zip((tr, va, te), (d.get("max_train_nodes"), d.get("max_val_nodes"),
                                                                       d.get("max_test_nodes"))))
    return X, np.asarray(tr), np.asarray(va), np.asarray(te), d, truth


def truth_of(truth, i):
    return None if truth is None else {"clean": truth["clean"][i], "modes": truth["modes"][i], "fc": truth["centre_freq"][i]}


def score(model, X, te, truth, device, name, key, lab):
    """criteria of one model on the test signals -> rows (list of dicts) and final omegas (n, K)"""
    rows, om = [], []
    for i in te:
        u, o = uvmd_magn.decompose(model, X[i], device)
        m = sc.signal_criteria(X[i], u, o, truth_of(truth, i))
        m.update(dataset=lab, dataset_key=key, method=name, signal=int(i))
        rows.append(m)
        om.append(o)
    return rows, np.array(om)


# ---------------------------------------------------------------------------------------------- run one dataset
def run_dataset(key, folder, a):
    X, tr, va, te, d, truth = get_splits(key, folder, {"cwru": a.cwru_data, "traffic": a.traffic_data}.get(key))
    lab = sc.DATASETS[key][0]
    K, T = d["K"], X.shape[1]
    alpha = d.get("alpha", 2000.0)
    if a.quick:
        tr, va, te = tr[:12], va[:4], te[:4]
    elif a.n_test:
        te = te[:a.n_test]
    out = Path(a.out) / key
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {lab}  K={K}  T={T}  train/val/test = {len(tr)}/{len(va)}/{len(te)}  device {a.device}", flush=True)

    rows, omegas, params, hists = [], {}, {}, {}
    # reference rows: converged VMD and VMD-N from the earlier comparison (same signals)
    ref_file = Path(a.reference)
    ref_file = ref_file if ref_file.is_absolute() else ROOT / ref_file
    if ref_file.exists():
        ref = pd.read_csv(ref_file)
        ref = ref[(ref.dataset_key == key) & ref.signal.isin(te)]
        keep = ref[ref.method.isin([BASE] + [f"VMD-{N}" for N in a.layers])]
        rows += keep.to_dict("records")
    else:
        print(f"  reference {ref_file} not found: converged VMD and VMD-N rows are left out")

    for N in a.layers:                                       # stored UVMD-N (reconstruction-only loss), rebuilt, not retrained
        f = Path(folder) / f"uvmd_params_N{N}.npz"
        if f.exists() and "recon" not in a.variants:
            model = rebuild(f, T, K, N, a.device)
            r, om = score(model, X, te, truth, a.device, label(N, "recon"), key, lab)
            rows += r
            omegas[label(N, "recon")] = om
            P = np.load(f)
            params[label(N, "recon")] = P["alpha"]
        elif not f.exists():
            print(f"  missing {f}: the reconstruction-only baseline is skipped")

    for v in a.variants:
        w = uvmd_sep.VARIANTS[v]
        for N in a.layers:
            name = label(N, v)
            tag = out / f"{v.replace('+', '_')}_N{N}"
            rf = Path(str(tag) + "_rows.csv")
            if rf.exists() and not a.force and not a.quick:
                saved = np.load(str(tag) + "_omega.npy")
                rows += pd.read_csv(rf).to_dict("records")
                omegas[name] = saved
                params[name] = np.load(str(tag) + ".npz")["alpha"]
                hists[name] = json.loads(Path(str(tag) + "_history.json").read_text())
                print(f"  {name}: loaded from {rf.name}", flush=True)
                continue
            print(f"\n[{name}] weights {w.dict()}", flush=True)
            model = uvmd_magn.make_model(T, K, N, a.device, alpha)
            epochs = 3 if a.quick else a.epochs
            t0 = time.perf_counter()
            h = uvmd_sep.train_sep(model, X[tr], X[va], w, epochs, a.lr, a.patience, a.device, a.seed, a.restore_best,
                                   log=(lambda s: None) if a.quiet else print)
            h["train_time_s"] = time.perf_counter() - t0
            print(f"  trained {h['epochs_run']} epochs in {h['train_time_s']:.0f} s", flush=True)
            r, om = score(model, X, te, truth, a.device, name, key, lab)
            P = uvmd_magn.export_params(model)
            rows += r
            omegas[name], params[name], hists[name] = om, P["alpha"], h
            if not a.quick:
                np.savez(str(tag) + ".npz", **P)
                np.save(str(tag) + "_omega.npy", om)
                pd.DataFrame(r).to_csv(rf, index=False)
                Path(str(tag) + "_history.json").write_text(json.dumps(h, default=float))
    return pd.DataFrame(rows), {"K": K, "T": T, "omega": omegas, "alpha": params, "hist": hists, "folder": folder,
                                "n_test": len(te)}


# ---------------------------------------------------------------------------------------------- tables
def method_order(layers, variants, has_recon):
    ms = [BASE]
    for N in layers:
        ms += [f"VMD-{N}"] + ([label(N, "recon")] if has_recon else []) + [label(N, v) for v in variants if v != "recon"]
    return ms


def table_effect(df, datasets, layers, variants):
    """each new variant against the stored UVMD-N [recon] of the same N: ratio of means and share of signals better"""
    crit = [c for c in sc.CRITERIA if c[2] != "none"]
    head = ["Dataset", "Layers", "Variant"] + [f"{lab} {sc.ARROW[dr]}" for _, lab, dr, _, _ in crit]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for ds in datasets:
        for N in layers:
            b = df[(df.dataset == ds) & (df.method == label(N, "recon"))].set_index("signal")
            for v in variants:
                if v == "recon":
                    continue
                u = df[(df.dataset == ds) & (df.method == label(N, v))].set_index("signal")
                j = u.index.intersection(b.index)
                if len(j) == 0:
                    continue
                cells = []
                for k, _, dr, _, _ in crit:
                    x, y = u.loc[j, k], b.loc[j, k]
                    ratio = ((x - 1).abs().mean() / (y - 1).abs().mean()) if dr == "one" else (x.mean() / y.mean() if y.mean() else np.nan)
                    cells.append(f"{ratio:.2f}× ({100 * sc._better(x, y, dr):.0f}%)")
                md.append("| " + " | ".join([ds, str(N), v] + cells) + " |")
    return "\n".join(md)


def table_omega(results, layers, variants):
    rows = ["| Dataset | Method | Mean shift (grid spacings) | Modes moved |", "|---|---|---|---|"]
    for lab_, R in results.items():
        for m, om in R["omega"].items():
            s = osa.shift_stats(om, R["K"])
            rows.append(f"| {lab_} (K={R['K']}) | {m} | {s['mean_shift']:.3f} | {s['modes_moved']} of {s['n_modes']} |")
    return "\n".join(rows)


def table_alpha(results):
    rows = ["| Dataset | Method | alpha min | alpha median | alpha max | alphas below 5 |", "|---|---|---|---|---|---|"]
    for lab_, R in results.items():
        for m, al in R["alpha"].items():
            al = np.asarray(al)
            rows.append(f"| {lab_} | {m} | {al.min():.0f} | {np.median(al):.0f} | {al.max():.0f} | {int((al < 5).sum())} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------------- figures
def color(m):
    N, v = split_of(m)
    base = COLORS["VMD"] if (m == BASE or v is None) else COLORS.get(v, "#999999")
    if m == BASE:
        return base
    if v is None:                                           # VMD-N: greys
        return {1: "#b3b3b3", 2: "#dcdcdc"}.get(N, "#eeeeee")
    if N and N >= 2:                                        # lighter for 2 layers
        r, g, b = (int(base[i:i + 2], 16) for i in (1, 3, 5))
        r, g, b = (int(c + (255 - c) * 0.45) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"
    return base


def figure_bars(agg, datasets, methods, path):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    w = 0.86 / len(methods)
    for ax, (k, lab, dr, scale, kind) in zip(axes.ravel(), sc.CRITERIA):
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
                   edgecolor="k", linewidth=0.4, yerr=[lo, es_a], error_kw={"elinewidth": 0.7, "capsize": 1.5})
            vals += [v for v in ys if v > 0]
        if scale == "log" and vals:
            ax.set_yscale("log")
            ax.set_ylim(min(vals) / 2.5, max(vals) * 2.5)
        if dr == "one":
            ax.axhline(1, color="k", ls="--", lw=0.9)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets)
        ax.set_title(f"{kind.upper()}\n{lab} {sc.ARROW[dr]}", fontsize=11, color="#1f4e79" if kind == "reconstruction" else "black")
        if kind == "reconstruction":
            ax.set_facecolor("#eef4fb")
        ax.grid(alpha=0.25, axis="y", which="both")
    fig.legend([Patch(facecolor=color(m), edgecolor="k", linewidth=0.4) for m in methods], methods, loc="lower center",
               ncol=min(len(methods), 7), frameon=False, fontsize=9)
    fig.suptitle("Separation-aware training of UVMD: reconstruction (shaded) and separation. Bars = mean over test signals, "
                 "whiskers = standard error. [recon] = the upstream loss", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def figure_truth(agg_syn, methods, path):
    keys = [(k, lab, hib) for k, lab, hib in sc.TRUTH if k in agg_syn.columns and agg_syn[k].notna().any()]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 5.2), squeeze=False)
    for ax, (k, lab, hib) in zip(axes[0], keys):
        for mi, m in enumerate(methods):
            r = agg_syn[agg_syn.method == m]
            if r.empty:
                continue
            ax.bar(mi, float(r[k].iloc[0]), color=color(m), edgecolor="k", linewidth=0.4,
                   yerr=float(r[k + "_sem"].iloc[0]) if np.isfinite(r[k + "_sem"].iloc[0]) else 0)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=70, ha="right", fontsize=8)
        ax.set_title(f"{lab} {sc.UP if hib else sc.DOWN}", fontsize=11)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Synthetic data, true separation quality (known modes). Bars = mean, whiskers = standard error", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    for k, (lab, default) in DATASETS.items():
        ap.add_argument(f"--{k}", default=default, help=f"result folder of the {lab} benchmark run (default {default})")
    ap.add_argument("--cwru-data", default=None)
    ap.add_argument("--traffic-data", default=None)
    ap.add_argument("--variants", default="bw,bw+ortho,full", help="comma list from: " + ", ".join(uvmd_sep.VARIANTS))
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restore-best", action="store_true")
    ap.add_argument("--n-test", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reference", default="results/separation_comparison/per_signal.csv",
                    help="per-signal table of the earlier comparison (converged VMD and VMD-N rows are taken from it)")
    ap.add_argument("--out", default="results/sep_loss")
    ap.add_argument("--quiet", action="store_true", help="do not print every epoch")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    a.variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    for v in a.variants:
        if v not in uvmd_sep.VARIANTS:
            sys.exit(f"unknown variant {v}")
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    a.out = str(out)
    out.mkdir(parents=True, exist_ok=True)

    frames, results = [], {}
    for k in [s.strip() for s in a.datasets.split(",") if s.strip()]:
        folder = Path(getattr(a, k))
        folder = folder if folder.is_absolute() else ROOT / folder
        if not (folder / "summary.json").exists():
            print(f"[skip] {DATASETS[k][0]}: no summary.json in {folder}")
            continue
        try:
            df, R = run_dataset(k, folder, a)
        except (FileNotFoundError, OSError) as e:
            print(f"[skip] {DATASETS[k][0]}: data not found ({e})")
            continue
        frames.append(df)
        results[DATASETS[k][0]] = R
    if not frames:
        sys.exit("nothing was run")

    df = pd.concat(frames, ignore_index=True)
    datasets = list(dict.fromkeys(df.dataset))
    has_recon = any(m.endswith("[recon]") for m in df.method)
    methods = [m for m in method_order(a.layers, a.variants, has_recon) if m in set(df.method)]
    note_k = {d: next(R["K"] for lab_, R in results.items() if lab_ == d) for d in datasets}
    agg = sc.aggregate(df, [c[0] for c in sc.CRITERIA], datasets, methods)

    text = ["# UVMD trained with a separation-aware loss", "",
            "Same train / validation / test signals as the benchmark runs. `[recon]` = UVMD trained with the upstream "
            "reconstruction loss (the stored models of the benchmark), the other rows add the terms of "
            "`decomp/uvmd_sep.py`. Converged VMD and VMD-N are taken from `results/separation_comparison/per_signal.csv`. "
            f"Single training run (seed {a.seed}) per row; test signals: " + ", ".join(f"{lab_} {R['n_test']}" for lab_, R in results.items())
            + ".", "",
            "Variants: " + "; ".join(f"**{v}**: {uvmd_sep.VARIANTS[v].dict()}" for v in a.variants), ""]
    text += ["## Table 1. All datasets", "", "Mean ± standard deviation over the test signals. Bold = best of the rows of that dataset.", "",
             sc.table_main(agg, datasets, note_k), ""]
    text += ["## Table 2. Each variant against the reconstruction-only UVMD of the same depth", "",
             "Ratio of the means, variant ÷ `[recon]`, and in brackets the share of test signals on which the variant is better. "
             "Below 1× is better in every column (energy ratio: distance from 1).", "",
             table_effect(df, datasets, a.layers, a.variants), ""]
    syn = df[df.dataset_key == "synthetic"]
    tagg = None
    if not syn.empty and "grouped_mean_snr_db" in syn.columns:
        tt, tagg = sc.table_truth(syn, list(dict.fromkeys(syn.dataset)), [m for m in methods if m in set(syn.method)])
        text += ["## Table 3. Synthetic data: separation against the true modes", "",
                 "Grouped mode SNR: each estimated mode is assigned to the true mode it projects onto most, and the assigned modes "
                 "are summed. This is independent of the training loss (the true modes are never used in training).", "", tt, ""]
        tagg.to_csv(out / "table_synthetic_truth.csv", index=False)
    text += ["## Table 4. Centre-frequency movement of the UVMD models", "",
             "Distance of the final omega_k from the uniform start grid, in grid spacings. For reference, converged VMD and VMD-N "
             "are in `results/omega_shift_all/omega_shift_all.md`.", "", table_omega(results, a.layers, a.variants), ""]
    text += ["## Table 5. Learned alpha", "", table_alpha(results), ""]
    (out / "sep_loss.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    agg.to_csv(out / "table_all_datasets.csv", index=False)
    df.to_csv(out / "per_signal.csv", index=False)
    figure_bars(agg, datasets, methods, out / "comparison_bars.png")
    if tagg is not None:
        figure_truth(tagg, [m for m in methods if m in set(syn.method)], out / "synthetic_truth_bars.png")
    print("\n".join(text))
    print(f"wrote {out}/sep_loss.md, comparison_bars.png, synthetic_truth_bars.png and CSV files")


if __name__ == "__main__":
    main()
