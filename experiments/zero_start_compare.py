"""One comparison of uniform vs zero start across synthetic, CWRU and traffic.

    python experiments/zero_start_compare.py        # after run_zero_start_real.py has finished for cwru and traffic
Writes results/zero_start_comparison/{comparison.md, centre_frequencies.png}. Missing datasets are skipped.
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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decomp.vmd_numpy import vmd  # noqa: E402
from experiments.mode_figures import INK, MUTED, TRUTH, UVMD_C, VMD_C, _style  # noqa: E402

GREEN = "#1baf7a"
METH = [("VMD uniform", VMD_C, "o"), ("VMD zero", VMD_C, "s"), ("UVMD uniform", UVMD_C, "o"), ("UVMD zero", GREEN, "s")]


def synthetic():
    from datasets.synthetic import make_synthetic
    d = ROOT / "results/synthetic_K5_alpha500_zeroinit"
    D = make_synthetic(200, 1024, 0.1, 0)
    _, te = train_test_split(np.arange(200), test_size=0.3, random_state=42, shuffle=True)
    _, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
    ld = lambda f: np.load(f)
    Pu, Pz = ld(ROOT / "results/synthetic_K5_alpha500_L5/uvmd_params_N5.npz"), ld(d / "uvmd_params_N5.npz")
    om = {}
    for lab, fn in (("VMD uniform", lambda x: vmd(x, 500.0, 0.0, 5, True, 1, 1e-7, 499)),
                    ("VMD zero", lambda x: vmd(x, 500.0, 0.0, 5, True, 0, 1e-7, 499)),
                    ("UVMD uniform", lambda x: vmd(x, alpha=Pu["alpha"], K=5, DC=True, init=1, lambda_schedule=Pu["lambda"][:5])),
                    ("UVMD zero", lambda x: vmd(x, alpha=Pz["alpha"], K=5, DC=True, init=0, lambda_schedule=Pz["lambda"][:5]))):
        om[lab] = np.mean([np.sort(fn(D["x"][i])["omega"]) for i in te], axis=0)
    T = pd.read_csv(d / "zero_vs_uniform_start.csv")
    g = lambda m, s: float(T[(T.method == m) & (T.start == s)]["reconstruction error %"].iloc[0])
    rec = {"VMD uniform": g("VMD converged", "uniform"), "VMD zero": g("VMD converged", "zero"),
           "UVMD uniform": g("UVMD-5", "uniform"), "UVMD zero": g("UVMD-5", "zero")}
    err = lambda m, s: float(T[(T.method == m) & (T.start == s)]["error vs true modes % (mean of 4)"].iloc[0])
    extra = {"VMD uniform": err("VMD converged", "uniform"), "VMD zero": err("VMD converged", "zero"),
             "UVMD uniform": err("UVMD-5", "uniform"), "UVMD zero": err("UVMD-5", "zero")}
    return {"om": om, "rec": rec, "extra": extra, "truth": D["centre_freq"][te].mean(0), "extra_name": "error vs true modes %"}


def real(folder, to_cps, extra_col, extra_name):
    f = ROOT / "results" / folder
    if not (f / "omegas.json").exists():
        return None
    o = json.load(open(f / "omegas.json"))
    T = pd.read_csv(f / "zero_vs_uniform_start.csv", index_col=0)
    return {"om": {k: np.array(v) * to_cps for k, v in o["omegas"].items() if k in dict((m[0], 1) for m in METH)},
            "rec": {k: float(T.loc[k, "reconstruction error %"]) for k in T.index if k in dict((m[0], 1) for m in METH)},
            "extra": {k: float(T.loc[k, extra_col]) for k in T.index if k in dict((m[0], 1) for m in METH)},
            "truth": None, "extra_name": extra_name}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from datasets.cwru import FS
    from datasets.traffic import STEPS_PER_DAY
    sets = {"Synthetic (K=5)": synthetic(),
            "CWRU (K=4)": real("cwru_K4_zeroinit", 1.0 / FS, "ortho_index", "orthogonality index"),
            "Traffic (K=6)": real("traffic_K6_zeroinit", 1.0 / STEPS_PER_DAY, "mode_overlap", "mode overlap")}
    out = ROOT / "results" / "zero_start_comparison"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Uniform start vs zero start, alpha = 500, UVMD 5 layers (test items of each dataset)\n",
          "Reconstruction error = ||x - sum of modes|| / ||x||. Second column per dataset: synthetic = error against the TRUE modes; "
          "CWRU = orthogonality index; traffic = mode overlap (lower = more separated).\n"]
    n = len([s for s in sets.values() if s])
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, [(k, v) for k, v in sets.items() if v]):
        rows = [{"method": m, "reconstruction error %": s["rec"][m], s["extra_name"]: s["extra"][m],
                 "mode centre frequencies (cycles/sample)": np.round(s["om"][m], 3).tolist()} for m, _, _ in METH if m in s["rec"]]
        md += [f"\n## {name}\n", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g")]
        if s["truth"] is not None:
            md.append(f"\nActual frequencies: {np.round(s['truth'], 3).tolist()}")
            for f in s["truth"]:
                ax.axhline(f, color=TRUTH, lw=0.9, ls="--")
        for x, (m, col, mk) in enumerate(METH):
            if m in s["om"]:
                ax.plot([x] * len(s["om"][m]), s["om"][m], mk, color=col, ms=8, mfc=col if mk == "s" else "white", mew=2)
        ax.set_xticks(range(4))
        ax.set_xticklabels([m.replace(" ", "\n") for m, _, _ in METH], fontsize=9)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        _style(ax, bottom=True)
    axes[0].set_ylabel("final centre frequency (cycles/sample)", color=MUTED)
    fig.suptitle("Where the centre frequencies end (dashed = actual frequencies, synthetic only)", x=0.005, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "centre_frequencies.png", dpi=140)
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
