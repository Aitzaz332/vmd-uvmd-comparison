"""CWRU fault classification with one 1D-CNN fed by five different representations.

    bash scripts/download_cwru.sh ~/data/CWRU
    python experiments/run_cwru_e2e.py --data ~/data/CWRU --device cuda:0   # K=4, alpha=500, N=5; VMD vs end-to-end UVMD
    python experiments/run_cwru_e2e.py --data ~/data/CWRU --pipelines raw vmd vmd_n uvmd_early uvmd_e2e   # + references
    python experiments/run_cwru_e2e.py --data ~/data/CWRU --quick                          # smoke run

Pipelines (the CNN, its training loop, the split and the model selection are identical; only the input differs)

  vmd         classic ADMM VMD run to convergence, modes computed once, NOT updated by any loss     (K channels)
  vmd_n       the same solver stopped after N iterations = an untrained N-layer UVMD, not updated    (K channels)
  uvmd_early  UVMD trained earlier with the reconstruction loss only (upstream recipe), frozen        (K channels)
  uvmd_e2e    UVMD trained together with the CNN: loss = CE + rec_weight * relative spectral L1 residual;
              alpha_k and lambda of every layer are updated by it                                     (K channels)
  raw         the standardised window itself, no decomposition                                        (1 channel)

``vmd`` is the "VMD then CNN, modes not updated" case and ``uvmd_e2e`` is the "UVMD trained through the CNN" case
that were asked for. ``vmd_n``, ``uvmd_early`` and ``raw`` are references that isolate what the end-to-end
training changes: ``uvmd_e2e`` and ``vmd_n`` start from the same modes; ``uvmd_e2e`` and ``uvmd_early`` differ
only in which loss trained the UVMD.

Data: the 40-file CWRU subset of ``datasets/cwru.py`` (windows of T samples, standardised). Default split 'load':
train on 0-1 HP, validate on 2 HP, test on 3 HP (no recording is shared between sets). Labels: fault10 (normal +
3 fault types x 3 sizes) or fault4.

The early UVMD is read from ``--early-params`` (an ``uvmd_params_N*.npz`` written by run_cwru.py, e.g.
results/cwru/uvmd_params_N2.npz for K=4, N=2) or, when none is given, trained here with the upstream recipe
(Adam lr 0.1, batch size 1, up to 100 epochs; slow) and saved to <out>/early_uvmd_params_N<N>_K<K>.npz for reuse.

Outputs in --out: summary.md/json, runs.csv, predictions.csv (per window: predicted class, probability of the true class,
CE), histories.json, confusion_<pipeline>.csv, uvmd_e2e_params_N<N>_seed<s>.npz (upstream export format),
uvmd_e2e_modes_test_seed<s>.npz and vmd_modes_test.npz (the modes of the test windows, see e2e/export.py), figures/.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from datasets.cwru import FS, load_cwru, split_indices  # noqa: E402
from decomp.uvmd_torch import BatchedUVMD  # noqa: E402
from decomp.vmd_numpy import uvmd_forward_numpy  # noqa: E402
from e2e.cnn1d import Fault1DCNN  # noqa: E402
from e2e.export import decompose_with, save_modes  # noqa: E402
from e2e.train import E2E, fit, make_labels, predict, score, uvmd_modes, vmd_modes  # noqa: E402

ALL = ["raw", "vmd", "vmd_n", "uvmd_early", "uvmd_e2e"]


def names(N):
    return {"raw": "raw signal -> CNN",
            "vmd": "VMD (converged) -> CNN",
            "vmd_n": f"VMD-{N} -> CNN (untrained UVMD)",
            "uvmd_early": f"UVMD-{N} recon-trained, frozen -> CNN",
            "uvmd_e2e": f"UVMD-{N} end-to-end with CNN"}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder with the CWRU .mat files")
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--windows-per-file", type=int, default=30)
    p.add_argument("--split", choices=["load", "time"], default="load")
    p.add_argument("--labels", choices=["fault10", "fault4"], default="fault10")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--layers", type=int, default=5, help="UVMD layers N (also the iteration count of vmd_n)")
    p.add_argument("--alpha", type=float, default=500.0, help="VMD bandwidth penalty = initial alpha of the UVMD")
    p.add_argument("--pipelines", nargs="+", choices=ALL, default=["vmd", "uvmd_e2e"],
                   help="default: converged VMD -> CNN and end-to-end UVMD -> CNN; the others are optional references")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr-cnn", type=float, default=1e-3)
    p.add_argument("--lr-alpha", type=float, default=2.0, help="lr of alpha_raw in uvmd_e2e (0 freezes it)")
    p.add_argument("--lr-lam", type=float, default=0.05, help="lr of lambda in uvmd_e2e (0 freezes it)")
    p.add_argument("--rec-weight", type=float, default=0.1, help="weight of the relative reconstruction term")
    p.add_argument("--e2e-init", choices=["untrained", "early"], default="untrained",
                   help="start uvmd_e2e from the untrained UVMD (= VMD-N) or from the recon-trained one")
    p.add_argument("--early-params", default=None, help="uvmd_params_N*.npz of the recon-trained UVMD")
    p.add_argument("--early-epochs", type=int, default=100)
    p.add_argument("--early-lr", type=float, default=0.1)
    p.add_argument("--early-patience", type=int, default=10)
    p.add_argument("--vmd-tol", type=float, default=1e-7)
    p.add_argument("--vmd-max-iter", type=int, default=499)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-export-modes", action="store_true", help="do not save the modes of the test windows")
    p.add_argument("--quick", action="store_true", help="6 windows/file, 3 epochs, 1 seed, 1 early epoch")
    p.add_argument("--out", default=str(ROOT / "results" / "cwru_e2e"))
    a = p.parse_args(argv)
    if a.quick:
        a.windows_per_file, a.epochs, a.patience, a.seeds, a.early_epochs = 6, 3, 3, [0], 1
    return a


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------ early UVMD
def get_early_params(a, X, tr, va, out, log):
    """Upstream-format dict {alpha (K,), lambda (N, 2T)} of the reconstruction-trained UVMD."""
    default = out / f"early_uvmd_params_N{a.layers}_K{a.K}.npz"
    path = Path(a.early_params) if a.early_params else default
    if path.exists():
        P = dict(np.load(path))
        log(f"early UVMD: loaded {path}")
    else:
        if a.early_params:
            raise FileNotFoundError(a.early_params)
        from decomp import uvmd_magn
        log(f"early UVMD: no parameter file, training the upstream recipe (N={a.layers}, K={a.K}, "
            f"up to {a.early_epochs} epochs, batch size 1) -> {default}")
        torch.manual_seed(0)
        model = uvmd_magn.make_model(a.T, a.K, a.layers, a.device, a.alpha)
        t0 = time.perf_counter()
        h = uvmd_magn.train_uvmd(model, X[tr], X[va], a.early_epochs, a.early_lr, a.early_patience, a.device, 0,
                                 False, log=lambda s: log("  " + s))
        log(f"early UVMD: trained {h['epochs_run']} epochs in {time.perf_counter() - t0:.0f} s")
        P = uvmd_magn.export_params(model)
        np.savez(default, **P)
    if P["alpha"].shape != (a.K,) or P["lambda"].shape != (a.layers, 2 * a.T):
        raise ValueError(f"early parameters in {path} do not fit this run: alpha {P['alpha'].shape}, lambda "
                         f"{P['lambda'].shape}, expected ({a.K},) and ({a.layers}, {2 * a.T}). Use matching "
                         f"--K/--layers/--T, or delete the file to retrain.")
    return P


def check_against_numpy(uvmd, P, X, K, device):
    """Runtime guard: the batched torch model must reproduce the repo's verified NumPy replica."""
    u = uvmd_modes(uvmd, X, device)
    for i, f in enumerate(X):
        r = uvmd_forward_numpy(f.astype(np.float32).astype(np.float64), P["alpha"], P["lambda"], K)
        err = np.abs(u[i] - r["u"]).max() / max(np.abs(r["u"]).max(), 1e-12)
        if err > 1e-3:
            raise RuntimeError(f"batched UVMD disagrees with the NumPy replica (relative error {err:.2e}); "
                               f"refusing to continue")
    return float(err)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    a = parse(argv)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    dev = torch.device(a.device)
    D = load_cwru(a.data, a.T, a.windows_per_file, True)
    tr, va, te = split_indices(D, a.split)
    X = D["x"]
    y, cls_names = make_labels(D, a.labels)
    C, K, N = len(cls_names), a.K, a.layers
    PN = names(N)
    log(f"CWRU: {D['meta']['n_files']} files, {len(X)} windows x {a.T} samples | split '{a.split}': "
        f"train/val/test = {len(tr)}/{len(va)}/{len(te)} | {C} classes ({a.labels}) | K={K}, N={N} layers | "
        f"seeds {a.seeds} | device {a.device}")
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        cnt = np.bincount(y[idx], minlength=C)
        log(f"  {nm:>5} windows per class: " + ", ".join(f"{c}:{k}" for c, k in zip(cls_names, cnt)))
    if D["meta"]["missing_files"]:
        log(f"  WARNING missing files: {D['meta']['missing_files']}")
    if len(set(y[tr])) < C:
        log(f"  WARNING: only {len(set(y[tr]))} of {C} classes occur in the training split")

    # ---- frozen features, computed once
    feats, info = {}, {}
    if "raw" in a.pipelines:
        feats["raw"] = X[:, None, :].astype(np.float32)
    if "vmd" in a.pipelines:
        t0 = time.perf_counter()
        feats["vmd"], it, om_vmd = vmd_modes(X, K, a.alpha, None, a.vmd_tol, a.vmd_max_iter, return_omega=True)
        info["vmd"] = {"mean_iters": it, "feature_time_s": time.perf_counter() - t0}
        log(f"[vmd] converged VMD (alpha {a.alpha:g}) on {len(X)} windows: mean {it:.1f} iterations, "
            f"{info['vmd']['feature_time_s']:.0f} s")
        if not a.no_export_modes:
            save_modes(out / "vmd_modes_test.npz", feats["vmd"][te], om_vmd[te], D, te, y, cls_names, "test")
    if "vmd_n" in a.pipelines:
        t0 = time.perf_counter()
        feats["vmd_n"], _ = vmd_modes(X, K, a.alpha, N)
        info["vmd_n"] = {"feature_time_s": time.perf_counter() - t0}
    early = None
    if {"uvmd_early"} & set(a.pipelines) or a.e2e_init == "early":
        early = get_early_params(a, X, tr, va, out, log)
        eu = BatchedUVMD(a.T, K, N, a.alpha).load_params(early["alpha"], early["lambda"]).to(dev)
        err = check_against_numpy(eu, early, X[te[:3]], K, dev)
        log(f"[uvmd_early] alpha {np.round(early['alpha'], 1).tolist()} | batched torch vs NumPy replica: "
            f"max relative error {err:.1e}")
        if "uvmd_early" in a.pipelines:
            feats["uvmd_early"] = uvmd_modes(eu, X, dev)

    # ---- train / evaluate every pipeline for every seed
    rows, preds, hists, conf = [], [], {}, {}
    te_meta = pd.DataFrame({"window": te, "file": D["file"][te], "load": D["load"][te]})
    for pipe in a.pipelines:
        for seed in a.seeds:
            torch.manual_seed(seed)
            log(f"\n[{PN[pipe]}] seed {seed}")
            if pipe == "uvmd_e2e":
                uv = BatchedUVMD(a.T, K, N, a.alpha)
                if a.e2e_init == "early":
                    uv.load_params(early["alpha"], early["lambda"])
                model = E2E(uv, Fault1DCNN(K, C))
                Xtr, Xva, Xte = X[tr], X[va], X[te]
            else:
                model = Fault1DCNN(feats[pipe].shape[1], C)
                Xtr, Xva, Xte = feats[pipe][tr], feats[pipe][va], feats[pipe][te]
            h = fit(model, Xtr, y[tr], Xva, y[va], n_classes=C, device=dev, seed=seed, epochs=a.epochs,
                    patience=a.patience, batch=a.batch, rec_weight=a.rec_weight, lr_cnn=a.lr_cnn,
                    lr_alpha=a.lr_alpha, lr_lam=a.lr_lam, log=log)
            logits, rec = predict(model, Xte, dev)
            sc = score(y[te], logits, C)
            cnn = model.cnn if pipe == "uvmd_e2e" else model
            row = {"pipeline": pipe, "seed": seed, "test_acc": sc["acc"], "test_macro_f1": sc["macro_f1"],
                   "test_ce": sc["ce"], "best_val_acc": h["best_val_acc"], "best_epoch": h["best_epoch"],
                   "epochs_run": h["epochs_run"], "train_time_s": h["train_time_s"], "cnn_params": cnn.n_params(),
                   "uvmd_trainable_params": model.uvmd.n_real_params() if pipe == "uvmd_e2e" else 0,
                   "test_rec_rel": float(rec.mean()) if rec is not None else np.nan}
            if pipe == "uvmd_e2e":
                P = model.uvmd.export_params()
                np.savez(out / f"uvmd_e2e_params_N{N}_seed{seed}.npz", **P)
                row["alpha"] = np.round(P["alpha"], 2).tolist()
            rows.append(row)
            hists[f"{pipe}_seed{seed}"] = {k: v for k, v in h.items() if k != "epoch_s"}
            conf[pipe] = conf.get(pipe, 0) + sc["confusion"]
            lp = torch.log_softmax(torch.as_tensor(logits), 1).numpy()
            wce = -lp[np.arange(len(te)), y[te]]
            for w, yt, yp, ce_w in zip(te, y[te], sc["pred"], wce):
                preds.append((pipe, seed, int(w), cls_names[yt], cls_names[yp], float(np.exp(-ce_w)), float(ce_w)))
            if pipe == "uvmd_e2e" and not a.no_export_modes:
                mo, om = decompose_with(model.uvmd, X[te], dev)
                save_modes(out / f"uvmd_e2e_modes_test_seed{seed}.npz", mo, om, D, te, y, cls_names, "test", P)
            log(f"  -> test acc {sc['acc']:.4f}  macro-F1 {sc['macro_f1']:.4f}  CE {sc['ce']:.4f}  "
                f"(best val acc {h['best_val_acc']:.4f} @ epoch {h['best_epoch']}, {h['train_time_s']:.0f} s)"
                + (f"  test recon rel. L1 {row['test_rec_rel']:.4f}  alpha {row['alpha']}" if pipe == "uvmd_e2e" else ""))

    # ---- save
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    pr = pd.DataFrame(preds, columns=["pipeline", "seed", "window", "true", "pred", "p_true", "window_ce"]).merge(te_meta, on="window")
    pr.to_csv(out / "predictions.csv", index=False)
    json.dump(hists, open(out / "histories.json", "w"), default=float)
    for pipe, cm in conf.items():
        pd.DataFrame(cm, index=cls_names, columns=cls_names).to_csv(out / f"confusion_{pipe}.csv")
    agg = runs.groupby("pipeline", sort=False).agg(
        test_acc=("test_acc", "mean"), test_acc_std=("test_acc", "std"), macro_f1=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"), test_ce=("test_ce", "mean"), val_acc=("best_val_acc", "mean"),
        train_time_s=("train_time_s", "mean"), cnn_params=("cnn_params", "first"),
        uvmd_params=("uvmd_trainable_params", "first"), rec_rel=("test_rec_rel", "mean")).fillna({"test_acc_std": 0.0,
                                                                                                    "macro_f1_std": 0.0})
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "commit": git_commit()}
    json.dump({"args": vars(a), "env": env, "data": D["meta"], "classes": cls_names, "features": info,
               "summary": agg.to_dict("index"),
               "test_size": int(len(te)), "train_size": int(len(tr)), "val_size": int(len(va))},
              open(out / "summary.json", "w"), indent=2, default=float)
    recall = pd.DataFrame({PN[p]: np.diag(c) / np.maximum(c.sum(1), 1) for p, c in conf.items()}, index=cls_names)
    fmt = agg.copy()
    fmt.index = [PN[p] for p in agg.index]
    tab = pd.DataFrame({"test acc (mean ± std over seeds)": [f"{r.test_acc:.4f} ± {r.test_acc_std:.4f}" for r in fmt.itertuples()],
                        "macro-F1": [f"{r.macro_f1:.4f} ± {r.macro_f1_std:.4f}" for r in fmt.itertuples()],
                        "test CE": fmt.test_ce.round(4), "best val acc": fmt.val_acc.round(4),
                        "recon rel. L1 (test)": fmt.rec_rel.round(4),
                        "CNN params": fmt.cnn_params, "UVMD params": fmt.uvmd_params,
                        "train s": fmt.train_time_s.round(0)}, index=fmt.index)
    md = [f"# CWRU: one 1D-CNN, five input representations\n",
          f"{len(X)} windows x {a.T} samples (12 kHz), split '{a.split}' (train/val/test = {len(tr)}/{len(va)}/{len(te)}), "
          f"{C} classes ({a.labels}), K={K}, N={N}, seeds {a.seeds}. Same CNN, training loop, split and "
          f"model selection (best validation accuracy) in every row; only the CNN input differs. "
          f"uvmd_e2e: loss = CE + {a.rec_weight} x relative spectral L1 residual, lr(alpha)={a.lr_alpha}, "
          f"lr(lambda)={a.lr_lam}, start from '{a.e2e_init}' UVMD.\n",
          "## Test results (3 HP recordings, unseen load)\n", tab.to_markdown(),
          "\n## Per-class recall (summed over seeds)\n", recall.round(3).to_markdown(),
          f"\nEnvironment: {env}\n", "\n## Log\n```\n" + "\n".join(lines) + "\n```\n"]
    (out / "summary.md").write_text("\n".join(md))
    figures(out, a, runs, hists, PN)
    log(f"\nresults written to {out}")
    return runs


def figures(out, a, runs, hists, PN):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments import plots
    pipes = list(dict.fromkeys(runs["pipeline"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, p in enumerate(pipes):
        v = runs[runs.pipeline == p].test_acc.values
        ax.bar(i, v.mean(), color=plots.C[i % len(plots.C)], alpha=0.8)
        ax.scatter(np.full(len(v), i), v, color=plots.INK, s=12, zorder=3)
    ax.set_xticks(range(len(pipes)))
    ax.set_xticklabels([PN[p].replace(" -> ", "\n-> ").replace(", ", ",\n") for p in pipes], fontsize=7)
    ax.set_ylabel("test accuracy (3 HP)")
    lo = runs.test_acc.min()
    ax.set_ylim(max(0.0, lo - 0.1), 1.0)
    plots._style(ax)
    fig.tight_layout()
    fig.savefig(out / "figures" / "test_accuracy.png", dpi=130)
    plt.close(fig)

    seed = a.seeds[0]
    e = hists.get(f"uvmd_e2e_seed{seed}")
    fig, axes = plt.subplots(1, 3 if e else 1, figsize=(13 if e else 5, 3.4), squeeze=False)
    ax = axes[0, 0]
    for i, p in enumerate(pipes):
        h = hists[f"{p}_seed{seed}"]
        ax.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], color=plots.C[i % len(plots.C)], label=PN[p], lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (2 HP)")
    ax.legend(fontsize=6)
    plots._style(ax)
    if e:
        ep = range(1, len(e["val_acc"]) + 1)
        ax = axes[0, 1]
        ax.plot(ep, e["train_ce"], label="train CE", color=plots.C[0], lw=1)
        ax.plot(ep, e["val_ce"], label="val CE", color=plots.C[1], lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax2 = ax.twinx()
        ax2.plot(ep, e["train_rec"], label="train recon (rel. L1, right axis)", color=plots.MUTED, lw=1, ls="--")
        ax2.set_ylabel("relative reconstruction error")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")
        plots._style(ax)
        ax = axes[0, 2]
        al = np.array(e["alpha"])
        for k in range(al.shape[1]):
            ax.plot(ep, al[:, k], label=f"mode {k}", lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("bandwidth penalty alpha_k (larger = narrower mode)")
        ax.legend(fontsize=6)
        plots._style(ax)
    fig.suptitle(f"seed {seed}", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figures" / "training_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
