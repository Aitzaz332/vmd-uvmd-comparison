"""Leave-one-load-out comparison: VMD -> CNN vs end-to-end UVMD -> CNN, every motor load held out once.

    python experiments/run_cwru_lolo.py --data ~/data/CWRU --device cuda:0 --out results/cwru_lolo_K4_N5

Runs experiments/run_cwru_e2e.py once per fold (split 'lolo', test load = 0, 1, 2, 3 HP; see datasets/cwru.py LOLO_FOLDS),
with the same CNN, K, N, alpha, seeds and model selection in every fold; any other run_cwru_e2e.py option can be added
(e.g. --seeds 0 1 2 --epochs 60). For every fold it also writes the prediction comparison, the training curves and the
clean mode / alpha figures into <out>/test_<L>HP/, and then the summary over folds:
  <out>/lolo_summary.md, lolo_summary.csv       test accuracy / macro-F1 / loss per fold and averaged over the folds
  <out>/lolo_summary.png                        accuracy and test loss per fold, VMD vs end-to-end UVMD
Converged VMD does not depend on the split, so it is computed once and reused by all folds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from experiments import compare_predictions, plot_modes_clean, plot_training_curves, run_cwru_e2e  # noqa: E402

PIPES = [("vmd", "VMD (converged) → CNN", "#2f6db5"), ("uvmd_e2e", "UVMD end-to-end → CNN", "#e8702a")]


def summarize(out, folds):
    rows = []
    for f in folds:
        r = pd.read_csv(out / f"test_{f}HP" / "runs.csv")
        r = r[r.pipeline.isin([p for p, _, _ in PIPES])].assign(fold=f)
        rows.append(r[["fold", "pipeline", "seed", "test_acc", "test_macro_f1", "test_ce", "best_val_acc"]])
    R = pd.concat(rows, ignore_index=True)
    per_fold = R.groupby(["fold", "pipeline"]).agg(acc=("test_acc", "mean"), acc_sd=("test_acc", "std"),
                                                   f1=("test_macro_f1", "mean"), ce=("test_ce", "mean"),
                                                   ce_sd=("test_ce", "std")).reset_index()
    P = per_fold.pivot(index="fold", columns="pipeline")
    summ = pd.DataFrame({"test_load_HP": P.index})
    for p, name, _ in PIPES:
        summ[f"acc {p}"] = P[("acc", p)].values
    for p, name, _ in PIPES:
        summ[f"ce {p}"] = P[("ce", p)].values
    summ["acc diff (uvmd - vmd)"] = summ["acc uvmd_e2e"] - summ["acc vmd"]
    summ["ce diff (uvmd - vmd)"] = summ["ce uvmd_e2e"] - summ["ce vmd"]
    mean = summ.drop(columns="test_load_HP").mean()
    summ.to_csv(out / "lolo_summary.csv", index=False)
    R.to_csv(out / "lolo_runs_all.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    x = np.arange(len(folds) + 1)
    w = 0.38
    for ax, key, ylab, title in ((axs[0], "test_acc", "test accuracy", "Accuracy on the held-out load"),
                                 (axs[1], "test_ce", "test cross-entropy (lower is better)", "Loss on the held-out load")):
        for i, (p, name, c) in enumerate(PIPES):
            vals = [R[(R.fold == f) & (R.pipeline == p)][key].values for f in folds]
            m = [v.mean() for v in vals]
            allm = float(np.mean(m))
            xs = x + (i - 0.5) * w
            ax.bar(xs, m + [allm], w, color=c, label=name, alpha=0.9)
            for xx, v in zip(xs[:-1], vals):
                ax.plot(np.full(len(v), xx), v, "o", color="white", mec="#333", ms=4, zorder=5)
        ax.set_xticks(x, [f"{f} HP" for f in folds] + ["mean of folds"], fontsize=12)
        ax.set_xlabel("held-out (test) load", fontsize=13)
        ax.set_ylabel(ylab, fontsize=13)
        ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axs[0].set_ylim(0, 1.05)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=13, ncol=2, loc="upper center")
    fig.text(0.01, 0.005, "bar = mean over seeds, dots = individual seeds", fontsize=10, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out / "lolo_summary.png")
    plt.close(fig)

    md = ["# Leave-one-load-out: VMD → CNN vs end-to-end UVMD → CNN", "",
          "Every motor load (0-3 HP) is the test set once; train and validation use the other loads "
          "(see `datasets/cwru.py::LOLO_FOLDS`). Same CNN, K, N, alpha, seeds and model selection in every fold.", "",
          summ.round(4).to_markdown(index=False), "",
          f"Mean over folds: accuracy VMD {mean['acc vmd']:.4f}, UVMD end-to-end {mean['acc uvmd_e2e']:.4f}; "
          f"test CE VMD {mean['ce vmd']:.4f}, UVMD end-to-end {mean['ce uvmd_e2e']:.4f}.", "",
          f"UVMD end-to-end has the higher accuracy in {int((summ['acc diff (uvmd - vmd)'] > 0).sum())} of {len(folds)} folds "
          f"(tied in {int((summ['acc diff (uvmd - vmd)'] == 0).sum())}) and the lower test loss in "
          f"{int((summ['ce diff (uvmd - vmd)'] < 0).sum())} of {len(folds)}. With four folds, treat differences as indications, "
          "not as significance tests."]
    (out / "lolo_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return summ


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--resume", action="store_true", help="skip the training of folds whose summary.json already exists (figures are redrawn)")
    a, rest = p.parse_known_args(argv)
    for flag in ("--split", "--fold", "--pipelines"):
        if flag in rest:
            raise SystemExit(f"{flag} is set by this script")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.folds:
        d = out / f"test_{f}HP"
        print(f"\n=== fold: test load {f} HP ===", flush=True)
        if a.resume and (d / "summary.json").exists():
            print(f"fold {f}: results found in {d}, not retrained (--resume)", flush=True)
        else:
            run_cwru_e2e.main(rest + ["--split", "lolo", "--fold", str(f), "--pipelines", "vmd", "uvmd_e2e", "--out", str(d)])
        compare_predictions.main(["--run", str(d)])
        plot_training_curves.main(["--run", str(d)])
        plot_modes_clean.main(["--run", str(d)])
    return summarize(out, a.folds)


if __name__ == "__main__":
    main()
