"""Dataset 2 - CWRU bearing vibration: VMD vs UVMD signal decomposition.

    bash scripts/download_cwru.sh ~/data/CWRU                 # 40 .mat files, ~135 MB
    python experiments/run_cwru.py --data ~/data/CWRU --device cuda:0
    python experiments/run_cwru.py --data ~/data/CWRU --K 13 --out results/cwru_K13
    python experiments/run_cwru.py --data ~/data/CWRU --quick   # smoke run

Same methods as the synthetic benchmark (VMD converged, VMD-n, UVMD-N untrained/trained with the upstream
code and training recipe), plus the raw signal as a no-decomposition reference for the fault metrics.
There are no ground-truth modes, so evaluation is
  * reconstruction: recon RMSE / relative error / SNR, upstream spectral L1 loss, energy ratio,
    orthogonality index (decomp/metrics.py)
  * fault content: kurtosis-selected mode, envelope-spectrum harmonic SNR at BPFI/BPFO/BSF, envelope hit
    rate, oracle (best-mode) fault SNR (decomp/fault_metrics.py)
  * cost: time per window, iterations/layers, FLOPs, parameters, training time (decomp/cost.py)
Split (default --split load): train on 0-1 HP recordings, validate on 2 HP, test on 3 HP.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from datasets.cwru import FS, fault_freqs, load_cwru, split_indices  # noqa: E402
from decomp import cost  # noqa: E402
from decomp.fault_metrics import fault_eval, raw_eval  # noqa: E402
from decomp.metrics import evaluate  # noqa: E402
from decomp.vmd_numpy import uvmd_forward_numpy, vmd  # noqa: E402
from experiments import plots  # noqa: E402

REC = ["recon_rmse", "recon_rel_err", "recon_snr_db", "spec_l1", "energy_ratio", "ortho_index"]
FLT = ["fault_snr_db", "envelope_hit", "oracle_fault_snr_db", "sel_kurtosis", "normal_max_fault_snr_db"]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder with the CWRU .mat files")
    p.add_argument("--T", type=int, default=2048, help="window length in samples (12 kHz)")
    p.add_argument("--windows-per-file", type=int, default=30)
    p.add_argument("--split", choices=["load", "time"], default="load")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--alpha", type=float, default=2000.0)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--max-iter", type=int, default=499)
    p.add_argument("--vmd-iters", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50, 100, 200])
    p.add_argument("--sweep-windows", type=int, default=100, help="test windows used for the VMD-n sweep")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--restore-best", action="store_true")
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-uvmd", action="store_true")
    p.add_argument("--quick", action="store_true", help="4 windows/file, 3 epochs, short sweep")
    p.add_argument("--out", default=str(ROOT / "results" / "cwru"))
    a = p.parse_args()
    if a.quick:
        a.windows_per_file, a.epochs, a.vmd_iters, a.sweep_windows = 4, 3, [1, 2, 20], 20
    return a


def agg(rows, by_class=False):
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    cols = [c for c in REC + FLT if c in df]
    if not by_class:
        return {c: float(df[c].mean()) for c in cols}
    return df.groupby("cls")[cols].mean()


def main():
    a = parse()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    D = load_cwru(a.data, a.T, a.windows_per_file, not a.no_standardize)
    tr, va, te = split_indices(D, a.split)
    X = D["x"]
    FF = [fault_freqs(r) for r in D["rpm"]]
    log(f"CWRU: {D['meta']['n_files']} files, {len(X)} windows of {a.T} samples ({1e3 * a.T / FS:.0f} ms) "
        f"| split '{a.split}': train/val/test = {len(tr)}/{len(va)}/{len(te)} | K={a.K}")
    if D["meta"]["missing_files"]:
        log(f"  WARNING missing files: {D['meta']['missing_files']}")

    def ev(i, u, omega, spec_l1, method, n_iter, t):
        m = evaluate(X[i], u, omega=omega, spec_l1=spec_l1)
        m.update(fault_eval(u, X[i], D["cls"][i], FF[i], FS, omega, a.harmonics))
        m.update(method=method, window=int(i), file=int(D["file"][i]), cls=str(D["cls"][i]),
                 size=float(D["size"][i]), load=int(D["load"][i]), n_iter=n_iter, time_s=t)
        return m

    per, summ, costs, by_cls = [], {}, [], {}
    # example window: first test window of an inner-race 0.007" recording
    ex_i = int(next((i for i in te if D["cls"][i] == "inner"), te[0]))
    examples = {}

    # raw-signal reference
    rows = []
    for i in te:
        m = raw_eval(X[i], D["cls"][i], FF[i], FS, a.harmonics)
        m.update(method="raw signal", window=int(i), file=int(D["file"][i]), cls=str(D["cls"][i]),
                 size=float(D["size"][i]), load=int(D["load"][i]))
        rows.append(m)
    per += rows
    summ["raw signal"] = agg(rows)
    by_cls["raw signal"] = agg(rows, True)

    # VMD converged
    log("\n[VMD converged]")
    rows, its, ts = [], [], []
    for i in te:
        t0 = time.perf_counter()
        r = vmd(X[i], a.alpha, a.tau, a.K, True, 1, a.tol, a.max_iter)
        ts.append(time.perf_counter() - t0)
        its.append(r["n_iter"])
        rows.append(ev(i, r["u"], r["omega"], r["spec_l1"], "VMD (converged)", r["n_iter"], ts[-1]))
        if i == ex_i:
            examples["VMD (converged)"] = (r["u"], rows[-1]["sel_mode"])
    per += rows
    s = agg(rows)
    s.update(mean_iters=float(np.mean(its)), converged_frac=float(np.mean(np.array(its) < a.max_iter)),
             median_ms=1e3 * float(np.median(ts)))
    summ["VMD (converged)"] = s
    by_cls["VMD (converged)"] = agg(rows, True)
    costs.append({"method": "VMD (converged)", "iterations_or_layers": round(float(np.mean(its)), 1),
                  "trainable_params": 0, "flops_per_window": cost.flops_vmd(a.T, a.K, float(np.mean(its)), a.tau),
                  "median_ms_per_window": s["median_ms"], "train_time_s": 0.0, "train_flops": 0.0})
    log(f"  mean iters {np.mean(its):.1f} ({100 * s['converged_frac']:.0f}% converged), "
        f"rel err {s['recon_rel_err']:.4f}, fault SNR {s['fault_snr_db']:.2f} dB, "
        f"hit {s['envelope_hit']:.2f}, {s['median_ms']:.1f} ms")

    # VMD-n sweep (subset of test windows; full test set for n in --layers)
    log("\n[VMD-n sweep]")
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(te, size=min(a.sweep_windows, len(te)), replace=False))
    sweep = []
    for n in sorted(set(a.vmd_iters) | set(a.layers)):
        idx = te if n in a.layers else sub
        rows, ts = [], []
        for i in idx:
            t0 = time.perf_counter()
            r = vmd(X[i], a.alpha, a.tau, a.K, True, 1, -1.0, n)
            ts.append(time.perf_counter() - t0)
            rows.append(ev(i, r["u"], r["omega"], r["spec_l1"], f"VMD-{n}", n, ts[-1]))
        per += rows
        s = agg(rows)
        s.update(n_iter=n, median_ms=1e3 * float(np.median(ts)), n_windows=len(idx))
        summ[f"VMD-{n}"] = s
        sweep.append(s)
        if n in a.layers:
            by_cls[f"VMD-{n}"] = agg(rows, True)
        costs.append({"method": f"VMD-{n}", "iterations_or_layers": n, "trainable_params": 0,
                      "flops_per_window": cost.flops_vmd(a.T, a.K, n, a.tau), "median_ms_per_window": s["median_ms"],
                      "train_time_s": 0.0, "train_flops": 0.0})
        log(f"  n={n:>3} ({len(idx)} win): rel err {s['recon_rel_err']:.4f}  fault SNR {s['fault_snr_db']:6.2f} dB"
            f"  hit {s['envelope_hit']:.2f}  {s['median_ms']:.2f} ms")

    # UVMD (upstream)
    hists = {}
    if not a.skip_uvmd:
        import torch
        from decomp import uvmd_magn
        for N in a.layers:
            log(f"\n[UVMD N={N}] upstream Unfolding, K={a.K}, device {a.device}")
            torch.manual_seed(a.seed)
            model = uvmd_magn.make_model(a.T, a.K, N, a.device, a.alpha)
            npar = uvmd_magn.count_params(model)
            for tag in ("untrained", "trained"):
                train_s, h = 0.0, None
                if tag == "trained":
                    t0 = time.perf_counter()
                    h = uvmd_magn.train_uvmd(model, X[tr], X[va], a.epochs, a.lr, a.patience, a.device, a.seed,
                                             a.restore_best, log=log)
                    train_s = time.perf_counter() - t0
                    hists[f"UVMD-{N}"] = h
                    log(f"  training: {h['epochs_run']} epochs, {train_s:.1f} s")
                lab = f"UVMD-{N}" + (" untrained" if tag == "untrained" else "")
                P = uvmd_magn.export_params(model)
                rows, ts = [], []
                for i in te:
                    if a.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    u, om = uvmd_magn.decompose(model, X[i], a.device)
                    ts.append(time.perf_counter() - t0)
                    rn = uvmd_forward_numpy(X[i].astype(np.float32).astype(np.float64), P["alpha"], P["lambda"], a.K)
                    rows.append(ev(i, u, om, rn["spec_l1"], lab, N, ts[-1]))
                    if i == ex_i and tag == "trained":
                        examples[lab] = (u, rows[-1]["sel_mode"])
                per += rows
                s = agg(rows)
                s.update(median_ms=1e3 * float(np.median(ts)), params=npar, alpha_learned=P["alpha"].tolist())
                if h:
                    s.update(train_time_s=train_s, epochs_run=h["epochs_run"], best_val_loss=h["best_val_loss"])
                    np.savez(out / f"uvmd_params_N{N}.npz", **P)
                summ[lab] = s
                by_cls[lab] = agg(rows, True)
                costs.append({"method": lab, "iterations_or_layers": N, "trainable_params": npar["real_dof"],
                              "flops_per_window": cost.flops_uvmd_forward(a.T, a.K, N),
                              "median_ms_per_window": s["median_ms"], "train_time_s": train_s,
                              "train_flops": cost.flops_uvmd_train(a.T, a.K, N, len(tr), h["epochs_run"]) if h else 0.0})
                log(f"  {lab}: rel err {s['recon_rel_err']:.4f}  fault SNR {s['fault_snr_db']:.2f} dB  "
                    f"hit {s['envelope_hit']:.2f}  {s['median_ms']:.2f} ms  alpha {np.round(P['alpha']).tolist()}")

    # save
    pd.DataFrame(per).to_csv(out / "per_window.csv", index=False)
    cdf = pd.DataFrame(costs)
    cdf.to_csv(out / "cost.csv", index=False)
    cls_tab = pd.concat({k: v for k, v in by_cls.items()}, names=["method", "cls"])
    cls_tab.to_csv(out / "by_class.csv")
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    json.dump({"args": vars(a), "env": env, "data": D["meta"], "summary": summ}, open(out / "summary.json", "w"),
              indent=2, default=float)

    # figures
    title = f"CWRU window {ex_i} (file {D['file'][ex_i]}, {D['cls'][ex_i]} {D['size'][ex_i]}\", {D['load'][ex_i]} HP)"
    plots.plot_modes_envelope(out / "figures" / "example_decomposition.png", X[ex_i], FS, examples,
                              FF[ex_i][str(D["cls"][ex_i])] if D["cls"][ex_i] != "normal" else FF[ex_i]["shaft"],
                              title)
    pts = lambda m: {f"UVMD-{N} trained": (N, summ[f"UVMD-{N}"][m]) for N in a.layers if f"UVMD-{N}" in summ}  # noqa: E731
    plots.plot_vs_iterations(out / "figures" / "fault_snr_vs_iterations.png", sweep, pts("fault_snr_db"),
                             "fault_snr_db", "envelope SNR at true fault freq. (dB)")
    plots.plot_vs_iterations(out / "figures" / "recon_err_vs_iterations.png", sweep, pts("recon_rel_err"),
                             "recon_rel_err", "relative reconstruction error", logy=True)
    plots.plot_runtime(out / "figures" / "runtime.png", cdf["method"].tolist(), cdf["median_ms_per_window"].tolist())
    if hists:
        plots.plot_training(out / "figures" / "uvmd_training.png", hists)

    # markdown
    show = ["raw signal", "VMD (converged)"] + [f"VMD-{n}" for n in a.layers] + [k for k in summ if k.startswith("UVMD")]
    cols = REC + FLT
    md = ["# CWRU benchmark - VMD vs UVMD\n",
          f"{D['meta']['n_files']} recordings, {len(X)} windows x {a.T} samples at 12 kHz, standardised="
          f"{not a.no_standardize}, split '{a.split}' (test = {len(te)} windows), K={a.K}, alpha={a.alpha}, "
          f"tau={a.tau}.\n",
          "## Overall (test windows)\n", "| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in show:
        md.append(f"| {k} | " + " | ".join(f"{summ[k].get(c, float('nan')):.4g}" for c in cols) + " |")
    md += ["\n## Fault metrics by class\n",
           cls_tab[[c for c in ["fault_snr_db", "envelope_hit", "oracle_fault_snr_db", "sel_kurtosis",
                                "recon_rel_err"] if c in cls_tab]].to_markdown(floatfmt=".3g"),
           "\n## Cost (per window)\n", cdf.to_markdown(index=False, floatfmt=".4g"), f"\nEnvironment: {env}\n"]
    (out / "summary.md").write_text("\n".join(md) + "\n\n## Log\n```\n" + "\n".join(lines) + "\n```\n")
    log(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
