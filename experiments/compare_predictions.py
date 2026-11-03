"""Head-to-head: what the CNN predicts from VMD modes vs from end-to-end UVMD modes.

    python experiments/compare_predictions.py --run results/cwru_e2e_K4_N5_a500
    python experiments/compare_predictions.py --run results/cwru_e2e --a vmd --b uvmd_e2e   # the defaults

Reads the output folder of experiments/run_cwru_e2e.py (runs.csv, predictions.csv, histories.json, summary.json);
nothing is retrained. A = the pipeline whose modes are not updated by any loss (default ``vmd``, converged VMD),
B = the end-to-end UVMD (default ``uvmd_e2e``). Both feed the same CNN and are scored on the same unseen test windows.

Writes <run>/prediction_comparison/:
  prediction_comparison.md   test loss (cross-entropy), accuracy, macro-F1, balanced accuracy per seed and averaged;
                             per-class precision / recall / F1; most frequent confusions; windows right/wrong in both
  metrics_per_seed.csv, per_class.csv, paired.csv
  figures/confusion_matrices.png   A, B (row-normalised, pooled over seeds) and B - A
  figures/loss_and_accuracy.png    test CE and accuracy per seed; training/validation curves
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pipeline_labels(S):
    N = S["args"]["layers"]
    return {"raw": "raw signal", "vmd": "VMD (converged)", "vmd_n": f"VMD-{N}",
            "uvmd_early": f"UVMD-{N} recon-trained (frozen)", "uvmd_e2e": f"UVMD-{N} end-to-end"}


def seed_metrics(pr, classes):
    """pr: predictions of ONE pipeline and ONE seed."""
    t = pr["true"].map(classes.index).values
    p = pr["pred"].map(classes.index).values
    return {"test_ce": float(pr["window_ce"].mean()), "test_acc": float((t == p).mean()),
            "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(t, p)), "mean_p_true": float(pr["p_true"].mean()),
            "n_wrong": int((t != p).sum()), "n_test": int(len(t))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of run_cwru_e2e.py")
    ap.add_argument("--a", default="vmd", help="pipeline A (modes not updated), default vmd")
    ap.add_argument("--b", default="uvmd_e2e", help="pipeline B (end-to-end), default uvmd_e2e")
    ap.add_argument("--out", default=None)
    ar = ap.parse_args(argv)
    run = Path(ar.run)
    S = json.load(open(run / "summary.json"))
    classes = S["classes"]
    PL = pipeline_labels(S)
    A, B = ar.a, ar.b
    pr = pd.read_csv(run / "predictions.csv")
    H = json.load(open(run / "histories.json"))
    for k in (A, B):
        if k not in set(pr.pipeline):
            raise ValueError(f"pipeline '{k}' is not in {run}/predictions.csv (found {sorted(set(pr.pipeline))}); "
                             f"run run_cwru_e2e.py with --pipelines including it")
    out = Path(ar.out) if ar.out else run / "prediction_comparison"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seeds = sorted(set(pr[pr.pipeline == A].seed) & set(pr[pr.pipeline == B].seed))
    C = len(classes)

    # ---- per-seed metrics
    rows = []
    for k in (A, B):
        for s in seeds:
            rows.append({"pipeline": k, "seed": s, **seed_metrics(pr[(pr.pipeline == k) & (pr.seed == s)], classes)})
    ms = pd.DataFrame(rows)
    runs = pd.read_csv(run / "runs.csv")
    ms = ms.merge(runs[["pipeline", "seed", "best_val_acc", "best_epoch", "epochs_run"]], on=["pipeline", "seed"])
    ms.to_csv(out / "metrics_per_seed.csv", index=False)
    cols = ["test_ce", "test_acc", "macro_f1", "balanced_acc", "mean_p_true", "n_wrong", "best_val_acc"]
    mean = ms.groupby("pipeline")[cols].mean().loc[[A, B]]
    std = ms.groupby("pipeline")[cols].std().fillna(0.0).loc[[A, B]]
    summ = pd.DataFrame({c: [f"{mean.loc[k, c]:.4f} ± {std.loc[k, c]:.4f}" if c != "n_wrong"
                              else f"{mean.loc[k, c]:.1f} ± {std.loc[k, c]:.1f}" for k in (A, B)] for c in cols},
                        index=[PL[A], PL[B]])
    diff = ms[ms.pipeline == B].set_index("seed")[cols] - ms[ms.pipeline == A].set_index("seed")[cols]
    dsum = pd.DataFrame({"B - A (mean over seeds)": diff.mean(), "std over seeds": diff.std().fillna(0.0),
                         "seeds where B is better": [int((diff[c] < 0).sum()) if c in ("test_ce", "n_wrong")
                                                     else int((diff[c] > 0).sum()) for c in cols]})

    # ---- pooled confusion matrices, per-class precision / recall / F1
    cm, per_cls = {}, []
    for k in (A, B):
        sub = pr[pr.pipeline == k]
        t, p = sub["true"].map(classes.index).values, sub["pred"].map(classes.index).values
        cm[k] = confusion_matrix(t, p, labels=list(range(C)))
        P_, R_, F_, Sup = precision_recall_fscore_support(t, p, labels=list(range(C)), zero_division=0)
        per_cls.append(pd.DataFrame({"class": classes, f"{k}_precision": P_, f"{k}_recall": R_, f"{k}_f1": F_,
                                     "support": Sup}))
        pd.DataFrame(cm[k], index=classes, columns=classes).to_csv(out / f"confusion_{k}_pooled.csv")
    pc = per_cls[0].merge(per_cls[1].drop(columns="support"), on="class")
    pc["f1_B_minus_A"] = pc[f"{B}_f1"] - pc[f"{A}_f1"]
    pc.to_csv(out / "per_class.csv", index=False)
    pcm = pc.rename(columns={f"{A}_precision": f"P {PL[A]}", f"{B}_precision": f"P {PL[B]}",
                             f"{A}_recall": f"R {PL[A]}", f"{B}_recall": f"R {PL[B]}",
                             f"{A}_f1": f"F1 {PL[A]}", f"{B}_f1": f"F1 {PL[B]}", "f1_B_minus_A": "F1 B-A"})

    def top_errors(c, n=6):
        off = [(c[i, j], classes[i], classes[j]) for i in range(C) for j in range(C) if i != j and c[i, j] > 0]
        return sorted(off, reverse=True)[:n]

    err_tab = pd.DataFrame({"": range(1, 7)})
    for k in (A, B):
        te_ = top_errors(cm[k])
        err_tab[PL[k]] = [f"{t} → {p}: {n}" for n, t, p in te_] + [""] * (6 - len(te_))
    err_tab = err_tab.drop(columns="")

    # ---- paired agreement per seed (same test windows)
    prow = []
    for s in seeds:
        a_ = pr[(pr.pipeline == A) & (pr.seed == s)].set_index("window")
        b_ = pr[(pr.pipeline == B) & (pr.seed == s)].set_index("window").loc[a_.index]
        ra, rb = (a_["true"] == a_["pred"]).values, (b_["true"] == b_["pred"]).values
        only_a, only_b = int((ra & ~rb).sum()), int((~ra & rb).sum())
        prow.append({"seed": s, "both right": int((ra & rb).sum()), "only A right": only_a, "only B right": only_b,
                     "both wrong": int((~ra & ~rb).sum()),
                     "McNemar exact p": binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0,
                     "mean CE A - CE B per window": float((a_["window_ce"] - b_["window_ce"]).mean())})
    paired = pd.DataFrame(prow)
    paired.to_csv(out / "paired.csv", index=False)

    md = [f"# Prediction comparison: {PL[A]} → CNN  vs  {PL[B]} → CNN\n",
          f"Same CNN, training loop and model selection; test = {int(ms.n_test.iloc[0])} unseen windows (split "
          f"'{S['args']['split']}', {C} classes, {S['args']['labels']}), seeds {seeds}. K={S['args']['K']}, "
          f"alpha={S['args']['alpha']:g}, N={S['args']['layers']} layers. A = {PL[A]} (modes not updated by any loss), "
          f"B = {PL[B]} (UVMD updated by CE + {S['args']['rec_weight']} x reconstruction term).\n",
          "## Loss and accuracy on the test windows (mean ± std over seeds)\n", summ.to_markdown(),
          "\nLower is better for `test_ce` and `n_wrong`; higher for the rest. `test_ce` is the mean cross-entropy of the "
          "test windows, `mean_p_true` the mean probability given to the true class.\n",
          "## B minus A, per seed\n", dsum.round(4).to_markdown(),
          "\n## Windows right or wrong in both, per seed\n", paired.round(4).to_markdown(index=False),
          "\nA window is 'right' when the predicted class equals the true class. The McNemar p-value looks only at the "
          "windows where exactly one pipeline is right; with a few hundred windows from a handful of recordings, treat it as "
          "a rough guide.\n",
          "## Per-class precision / recall / F1 (pooled over seeds)\n", pcm.round(3).to_markdown(index=False),
          "\n## Most frequent confusions (true → predicted: count, pooled over seeds)\n", err_tab.to_markdown(index=False),
          "\nConfusion matrices: `figures/confusion_matrices.png`, `confusion_<pipeline>_pooled.csv`. "
          "Curves: `figures/loss_and_accuracy.png`.\n"]
    (out / "prediction_comparison.md").write_text("\n".join(md))
    figures(out, classes, cm, A, B, PL, ms, H, seeds)
    print(f"written to {out}")
    return {"metrics": ms, "mean": mean, "per_class": pc, "paired": paired, "confusion": cm}


def figures(out, classes, cm, A, B, PL, ms, H, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments import plots
    C = len(classes)
    ra = cm[A] / np.maximum(cm[A].sum(1, keepdims=True), 1) * 100
    rb = cm[B] / np.maximum(cm[B].sum(1, keepdims=True), 1) * 100
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, M, title, cmap, vmin, vmax in ((axes[0], ra, f"A: {PL[A]}", "Blues", 0, 100),
                                           (axes[1], rb, f"B: {PL[B]}", "Blues", 0, 100),
                                           (axes[2], rb - ra, "B − A (percentage points)", "RdBu", -50, 50)):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=9, loc="left")
        for i in range(C):
            for j in range(C):
                v = M[i, j]
                if abs(v) >= 0.5:
                    dark = (abs(v) > (50 if cmap == "Blues" else 30))
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if dark else plots.INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Confusion matrices, row-normalised (% of each true class), pooled over seeds {seeds}", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    col = {A: plots.C[0], B: plots.C[1]}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.5))
    for ax, key, lab in ((axes[0], "test_ce", "test cross-entropy (lower is better)"),
                         (axes[1], "test_acc", "test accuracy")):
        for i, k in enumerate((A, B)):
            v = ms[ms.pipeline == k][key].values
            ax.bar(i, v.mean(), color=col[k], alpha=0.8)
            ax.scatter(np.full(len(v), i), v, color=plots.INK, s=14, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([PL[A].replace(" ", "\n", 1), PL[B].replace(" ", "\n", 1)], fontsize=7)
        ax.set_ylabel(lab)
        plots._style(ax)
    for ax, key, lab in ((axes[2], "train_ce", "cross-entropy per epoch (solid: validation, dashed: training)"),
                         (axes[3], "val_acc", "validation accuracy per epoch")):
        for k in (A, B):
            for j, s in enumerate(seeds):
                h = H[f"{k}_seed{s}"]
                ep = range(1, len(h["val_acc"]) + 1)
                if key == "train_ce":
                    ax.plot(ep, h["val_ce"], color=col[k], lw=1, alpha=0.8, label=PL[k] if j == 0 else None)
                    ax.plot(ep, h["train_ce"], color=col[k], lw=1, ls="--", alpha=0.6)
                else:
                    ax.plot(ep, h[key], color=col[k], lw=1, alpha=0.8, label=PL[k] if j == 0 else None)
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab, fontsize=8)
        ax.legend(fontsize=6)
        plots._style(ax)
    fig.tight_layout()
    fig.savefig(out / "figures" / "loss_and_accuracy.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
