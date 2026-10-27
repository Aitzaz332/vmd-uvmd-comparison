"""Cross-dataset comparison: ONE table and ONE figure from the three benchmark runs.

Reads the ``summary.json`` that run_synthetic.py, run_cwru.py and run_traffic.py write, and writes

    results/comparison/comparison_table.md   one table: dataset x method x metrics (+ learned alpha)
    results/comparison/comparison_table.csv  same numbers, machine readable (mean and std columns)
    results/comparison/comparison.png        top row: reconstruction vs separation trade-off per dataset,
                                             bottom row: each dataset's own separation metric

    python experiments/compare_datasets.py                       # default result folders (all K = 13)
    python experiments/compare_datasets.py --cwru results/cwru   # use another folder, e.g. the K=4 CWRU run

Metrics shared by all three datasets (they need no ground truth):
  recon_rel_err   ||x - sum_k u_k|| / ||x||                       reconstruction, lower = better
  ortho_index     sum_{i<j} |<u_i,u_j>| / ||x||^2                 shared content between modes, lower = better
  energy_ratio    sum_k ||u_k||^2 / ||x||^2                       above 1 = modes overlap / double count
Separation metric of each dataset
  synthetic   grouped mode SNR (dB, higher = better) and centre-frequency error     (true modes are known)
  CWRU        SNR of the fault line in the best mode (dB, higher = better) and envelope hit rate
  traffic     mode overlap and bandwidth of the modes (lower = better)               (no ground truth)

VMD-N is UVMD-N before training (a test checks they are identical), so the table has one row for it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {  # key: (label, default result folder)
    "synthetic": ("Synthetic", "results/synthetic_K13"),
    "cwru": ("CWRU", "results/cwru_K13"),
    "traffic": ("LargeST traffic", "results/traffic_sd_bounded"),
}
# key, column label, higher_is_better (None = no direction, never bolded)
COMMON = [("recon_rel_err", "Rel. recon. error", False),
          ("ortho_index", "Orthogonality index", False),
          ("energy_ratio", "Energy ratio", None)]
SPECIFIC = {
    "synthetic": [("grouped_mean_snr_db", "Mode SNR, grouped (dB)", True), ("mean_omega_err", "Centre-freq. error", False)],
    "cwru": [("oracle_fault_snr_db", "Best-mode fault SNR (dB)", True), ("envelope_hit", "Fault hit rate", True)],
    "traffic": [("mode_overlap", "Mode overlap", False), ("mean_bandwidth_cpd", "Bandwidth (cycles/day)", False)],
}
BASE = "VMD (converged)"


def load_summary(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    f = p / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def color(m):
    return "#4d4d4d" if m == BASE else ("#b3b3b3" if m.startswith("VMD-") else "#d95f02")


def fmt(mean, std=None):
    if mean is None or not np.isfinite(mean):
        return "-"
    s = f"{mean:.3g}"
    return s + (f" ± {std:.2g}" if std is not None and np.isfinite(std) else "")


def build_rows(runs, methods):
    """one dict per (dataset, method) with <metric> and <metric>_std"""
    rows = []
    for k, d in runs.items():
        label = f"{DATASETS[k][0]} (K={d['args'].get('K', '?')})"
        for m in methods:
            s = d["summary"].get(m)
            if s is None:
                print(f"[warn] {k}: no '{m}' in summary.json, row skipped")
                continue
            r = {"dataset_key": k, "dataset": label, "method": m}
            for key, _, _ in COMMON + [c for k2 in SPECIFIC for c in SPECIFIC[k2]]:
                if key in s:
                    r[key] = s[key]
                    r[key + "_std"] = s.get(key + "_std", np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def best_method(df_ds, key, higher):
    if higher is None or key not in df_ds or df_ds[key].isna().all():
        return None
    return df_ds.loc[df_ds[key].idxmax() if higher else df_ds[key].idxmin(), "method"]


def write_table(df, runs, out):
    cols = COMMON + [c for k in SPECIFIC for c in SPECIFIC[k]]
    cols = [c for c in cols if c[0] in df.columns]
    header = ["Dataset", "Method"] + [c[1] + ("" if c[2] is None else " ↑" if c[2] else " ↓") for c in cols]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for ds, g in df.groupby("dataset", sort=False):
        k = g["dataset_key"].iloc[0]
        best = {c[0]: best_method(g, c[0], c[2]) for c in cols}
        for _, r in g.iterrows():
            cells = []
            for key, _, _ in cols:
                v = r.get(key, np.nan)
                s = fmt(v, r.get(key + "_std", np.nan)) if key in r and np.isfinite(v) else "-"
                cells.append(f"**{s}**" if best[key] == r["method"] and s != "-" else s)
            md.append("| " + " | ".join([ds, r["method"]] + cells) + " |")
    ks = {d["args"].get("K") for d in runs.values()}
    text = ["# VMD vs UVMD across the datasets", ""]
    if len(ks) > 1:
        text += [f"**Warning: the datasets were run with different K ({sorted(map(str, ks))}); "
                 "compare them only with care.**", ""]
    text += ["Mean over test signals / windows / nodes, with the standard deviation where the run stored it. "
             "Bold = best of the rows of that dataset. VMD-N is also UVMD-N before training. "
             "A dash means the metric does not exist for that dataset.", ""] + md + [""]
    text += ["Columns shared by all datasets need no ground truth. The last columns are each dataset's own separation measure: "
             "mode SNR against the known true modes (synthetic), the fault-line SNR in the best mode (CWRU), "
             "spectral overlap and bandwidth of the modes (traffic, no ground truth exists).", ""]
    al = ["## Learned bandwidth penalty alpha_k of the trained UVMD (initial value 2000)", "",
          "| dataset | model | min | median | max | alphas below 5 |", "|---|---|---|---|---|---|"]
    n_al = 0
    for k, d in runs.items():
        for m in (m for m in d["summary"] if m.startswith("UVMD-") and "untrained" not in m):
            a = d["summary"][m].get("alpha_learned")
            if a:
                a = np.asarray(a, float)
                al.append(f"| {DATASETS[k][0]} (K={d['args'].get('K', '?')}) | {m} | {a.min():.0f} | {np.median(a):.0f} | "
                          f"{a.max():.0f} | {int((a < 5).sum())} |")
                n_al += 1
    if n_al:
        text += al + ["", "An alpha near 0 turns a mode into a pass-everything residual: the reconstruction error drops "
                      "but that mode no longer separates anything.", ""]
    (out / "comparison_table.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    df.to_csv(out / "comparison_table.csv", index=False)
    print("\n".join(text))


def figure(df, runs, methods, out):
    keys = list(DATASETS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), squeeze=False)
    for c, k in enumerate(keys):
        top, bot = axes[0][c], axes[1][c]
        title = f"{DATASETS[k][0]}" + (f" (K={runs[k]['args'].get('K', '?')})" if k in runs else "")
        if k not in runs or df[df.dataset_key == k].empty:
            for ax in (top, bot):
                ax.text(0.5, 0.5, "no results yet\n(run on node 7)", ha="center", va="center", transform=ax.transAxes, color="#888")
                ax.set_xticks([])
                ax.set_yticks([])
            top.set_title(title)
            continue
        g = df[df.dataset_key == k].set_index("method")
        base = g.loc[BASE]
        pts = []
        for m in methods[1:]:
            if m in g.index:
                x = g.loc[m, "recon_rel_err"] / max(base["recon_rel_err"], 1e-12)
                y = g.loc[m, "ortho_index"] / max(base["ortho_index"], 1e-12)
                pts.append((m, x, y))
        xs = [p[1] for p in pts] + [1.0]
        ys = [p[2] for p in pts] + [1.0]
        xl, xh = min(xs) / 1.8, max(xs) * 1.8
        yl, yh = min(ys) / 1.8, max(ys) * 1.8
        top.fill([xl, 1, 1, xl], [yl, yl, 1, 1], color="#1b9e77", alpha=0.08, zorder=0)
        top.axvline(1, color="k", ls="--", lw=0.8)
        top.axhline(1, color="k", ls="--", lw=0.8)
        top.scatter([1], [1], marker="*", s=220, color=color(BASE), zorder=4)
        top.annotate(BASE, (1, 1), textcoords="offset points", xytext=(7, 6), fontsize=8)
        for i, (m, x, y) in enumerate(pts):
            top.scatter([x], [y], s=75, color=color(m), marker="s" if m.startswith("VMD-") else "o", zorder=3,
                        edgecolor="k", linewidth=0.5)
            top.annotate(m, (x, y), textcoords="offset points", xytext=(7, 6 if i % 2 == 0 else -13), fontsize=8)
        top.set_xscale("log")
        top.set_yscale("log")
        for axis in (top.xaxis, top.yaxis):                       # plain numbers, a few minor labels, no overlap
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_tick_params(which="minor", labelsize=7)
        top.set_xlim(xl, xh)
        top.set_ylim(yl, yh)
        top.set_xlabel("reconstruction error ÷ converged VMD    (← better)")
        top.set_ylabel("orthogonality index ÷ converged VMD    (↓ better)")
        top.set_title(title)
        top.grid(alpha=0.25, which="both")

        key, lab, hib = SPECIFIC[k][0]
        vals = [(m, g.loc[m, key]) for m in methods if m in g.index and key in g.columns and np.isfinite(g.loc[m, key])]
        bot.bar(range(len(vals)), [v for _, v in vals], color=[color(m) for m, _ in vals], edgecolor="k", linewidth=0.5)
        for i, (_, v) in enumerate(vals):
            bot.annotate(f"{v:.3g}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        bot.set_xticks(range(len(vals)))
        bot.set_xticklabels([m for m, _ in vals], rotation=30, ha="right", fontsize=8)
        bot.set_title(f"{lab}  ({'higher' if hib else 'lower'} = better)", fontsize=10)
        bot.grid(alpha=0.25, axis="y")
    fig.suptitle("VMD vs UVMD   top: reconstruction vs separation (green zone = better than converged VMD on both)   "
                 "bottom: each dataset's own separation measure", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "comparison.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, (lab, default) in DATASETS.items():
        ap.add_argument(f"--{k}", default=default, help=f"result folder of the {lab} run (default {default})")
    ap.add_argument("--out", default="results/comparison")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for k in DATASETS:
        d = load_summary(getattr(a, k))
        if d is None:
            print(f"[missing] {k}: no summary.json in {getattr(a, k)}")
        else:
            runs[k] = d
    if not runs:
        sys.exit("no results found")
    layers = next(iter(runs.values()))["args"].get("layers", [1, 2])
    methods = [BASE] + [m for n in layers for m in (f"VMD-{n}", f"UVMD-{n}")]
    df = build_rows(runs, methods)
    write_table(df, runs, out)
    figure(df, runs, methods, out)
    print(f"\nwrote {out}/comparison_table.md, comparison_table.csv, comparison.png")


if __name__ == "__main__":
    main()
