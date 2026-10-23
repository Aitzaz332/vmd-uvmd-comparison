"""Dataset 1 - synthetic signals with known modes: VMD vs UVMD.

    python experiments/run_synthetic.py                    # full run (needs torch for UVMD)
    python experiments/run_synthetic.py --quick            # small, fast smoke run
    python experiments/run_synthetic.py --skip-uvmd        # VMD part only (no torch needed)

Methods compared on the same held-out test signals:
    VMD (converged)   classic ADMM VMD, stops at uDiff <= tol (max 499 iterations, as vmdpy)
    VMD-n             the same solver stopped after exactly n iterations (n = 1, 2, 5, ... )
    UVMD-N untrained  upstream Unfolding with N layers at its initial parameters (= VMD-N with tau=0)
    UVMD-N            upstream Unfolding trained with the upstream procedure on the training signals
    vmdpy             the reference vmdpy package, timed only (outputs are identical to VMD, see tests)

Outputs go to --out: summary.md, summary.json, per_signal.csv, cost.csv, uvmd_params_N*.npz, figures.
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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from datasets.synthetic import make_synthetic  # noqa: E402
from decomp import cost  # noqa: E402
from decomp.metrics import evaluate, match_modes  # noqa: E402
from decomp.vmd_numpy import uvmd_forward_numpy, vmd  # noqa: E402
from experiments import plots  # noqa: E402

KEY_METRICS = ["recon_rmse", "recon_rel_err", "recon_snr_db", "spec_l1", "denoise_rmse",
               "mean_mode_nrmse", "mean_mode_corr", "mean_mode_snr_db", "grouped_mean_snr_db",
               "mean_omega_err", "ortho_index", "energy_ratio", "spurious_energy_share"]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-signals", type=int, default=200)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--noise", type=float, default=0.1, help="std of additive white noise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=4, help="modes for VMD and UVMD (paper uses 13 on traffic data)")
    p.add_argument("--alpha", type=float, default=2000.0, help="VMD alpha = UVMD initial alpha (paper: 2000)")
    p.add_argument("--tau", type=float, default=0.0, help="VMD dual-ascent step (0 = noise-slack)")
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--max-iter", type=int, default=499)
    p.add_argument("--vmd-iters", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50, 100, 200, 499])
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2], help="UVMD layers to test (paper: 1-2)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--restore-best", action="store_true",
                   help="use best-val UVMD weights (upstream keeps the last-epoch weights)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-uvmd", action="store_true")
    p.add_argument("--quick", action="store_true", help="40 signals, T=512, 5 epochs")
    p.add_argument("--out", default=str(ROOT / "results" / "synthetic"))
    a = p.parse_args()
    if a.quick:
        a.n_signals, a.T, a.epochs = 40, 512, 5
        a.vmd_iters = [1, 2, 5, 20, 100, 499]
    return a


def aggregate(rows):
    df = pd.DataFrame(rows)
    out = {}
    for k in KEY_METRICS:
        if k in df:
            out[k] = float(df[k].replace([np.inf, -np.inf], np.nan).mean())
            out[k + "_std"] = float(df[k].replace([np.inf, -np.inf], np.nan).std())
    return out


def main():
    a = parse()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    # ---------------- data + split (same split procedure as upstream uvmd.py) ----------------
    D = make_synthetic(a.n_signals, a.T, a.noise, a.seed)
    idx = np.arange(a.n_signals)
    tr, te = train_test_split(idx, test_size=0.3, random_state=42, shuffle=True)
    va, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
    X = D["x"]
    log(f"synthetic: {a.n_signals} signals, T={a.T}, noise std={a.noise}, K={a.K} "
        f"| train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    def ev(i, r_u, omega, spec_l1):
        return evaluate(X[i], r_u, omega=omega, spec_l1=spec_l1, clean=D["clean"][i],
                        true_modes=D["modes"][i], true_fc=D["centre_freq"][i])

    per_signal, summary, cost_rows, examples = [], {}, [], {}
    ex_i = int(te[0])

    # ---------------- VMD converged ----------------
    log("\n[VMD converged]")
    rows, iters, times = [], [], []
    for i in te:
        t0 = time.perf_counter()
        r = vmd(X[i], a.alpha, a.tau, a.K, True, 1, a.tol, a.max_iter)
        times.append(time.perf_counter() - t0)
        m = ev(i, r["u"], r["omega"], r["spec_l1"])
        m.update(method="VMD (converged)", signal=int(i), n_iter=r["n_iter"], time_s=times[-1])
        rows.append(m)
        iters.append(r["n_iter"])
        if i == ex_i:
            examples["VMD (converged)"] = match_modes(r["u"], D["modes"][i])
    per_signal += rows
    s = aggregate(rows)
    s.update(mean_iters=float(np.mean(iters)), converged_frac=float(np.mean(np.array(iters) < a.max_iter)))
    summary["VMD (converged)"] = s
    mean_it = float(np.mean(iters))
    cost_rows.append({"method": "VMD (converged)", "iterations_or_layers": round(mean_it, 1),
                      "trainable_params": 0, "flops_per_signal": cost.flops_vmd(a.T, a.K, mean_it, a.tau),
                      "median_ms_per_signal": 1e3 * float(np.median(times)), "train_time_s": 0.0,
                      "train_flops": 0.0})
    log(f"  mean iterations {mean_it:.1f}, recon_rel_err {s['recon_rel_err']:.4f}, "
        f"mode SNR {s['mean_mode_snr_db']:.2f} dB, median {1e3 * np.median(times):.2f} ms")

    # ---------------- vmdpy reference timing ----------------
    try:
        from vmdpy import VMD as vmdpy_VMD
        sub = [X[i] for i in te[:min(10, len(te))]]
        tm = cost.time_per_call(lambda f: vmdpy_VMD(f, a.alpha, a.tau, a.K, 1, 1, a.tol), sub, warmup=1)
        cost_rows.append({"method": "vmdpy package (converged)", "iterations_or_layers": round(mean_it, 1),
                          "trainable_params": 0, "flops_per_signal": cost.flops_vmd(a.T, a.K, mean_it, a.tau),
                          "median_ms_per_signal": 1e3 * tm["median_s"], "train_time_s": 0.0, "train_flops": 0.0})
        log(f"  vmdpy package median {1e3 * tm['median_s']:.2f} ms (stores all 500 iterates)")
    except Exception as e:  # pragma: no cover
        log(f"  vmdpy timing skipped: {e}")

    # ---------------- VMD fixed-iteration sweep ----------------
    log("\n[VMD-n sweep]")
    sweep = []
    for n in a.vmd_iters:
        rows, times = [], []
        for i in te:
            t0 = time.perf_counter()
            r = vmd(X[i], a.alpha, a.tau, a.K, True, 1, -1.0, n)
            times.append(time.perf_counter() - t0)
            m = ev(i, r["u"], r["omega"], r["spec_l1"])
            m.update(method=f"VMD-{n}", signal=int(i), n_iter=n, time_s=times[-1])
            rows.append(m)
            if n in a.layers and i == ex_i:
                examples[f"VMD-{n}"] = match_modes(r["u"], D["modes"][i])
        per_signal += rows
        s = aggregate(rows)
        s["n_iter"] = n
        s["median_ms"] = 1e3 * float(np.median(times))
        sweep.append(s)
        summary[f"VMD-{n}"] = s
        cost_rows.append({"method": f"VMD-{n}", "iterations_or_layers": n, "trainable_params": 0,
                          "flops_per_signal": cost.flops_vmd(a.T, a.K, n, a.tau),
                          "median_ms_per_signal": s["median_ms"], "train_time_s": 0.0, "train_flops": 0.0})
        log(f"  n={n:>3}: recon_rel_err {s['recon_rel_err']:.4f}  mode SNR {s['mean_mode_snr_db']:6.2f} dB"
            f"  omega err {s.get('mean_omega_err', float('nan')):.4f}  {s['median_ms']:.2f} ms")

    # ---------------- UVMD (upstream code) ----------------
    hists = {}
    if not a.skip_uvmd:
        import torch
        from decomp import uvmd_magn
        torch.set_num_threads(max(1, torch.get_num_threads()))
        for N in a.layers:
            log(f"\n[UVMD N={N}]  upstream Unfolding, K={a.K}")
            torch.manual_seed(a.seed)
            model = uvmd_magn.make_model(a.T, a.K, N, a.device, a.alpha)
            npar = uvmd_magn.count_params(model)
            log(f"  trainable params: {npar['numel']} entries ({npar['real_dof']} real DOF)")
            for tag in ("untrained", "trained"):
                if tag == "trained":
                    t0 = time.perf_counter()
                    h = uvmd_magn.train_uvmd(model, X[tr], X[va], a.epochs, a.lr, a.patience, a.device,
                                             a.seed, a.restore_best, log=log)
                    train_s = time.perf_counter() - t0
                    hists[f"UVMD-{N}"] = h
                    log(f"  training: {h['epochs_run']} epochs, {train_s:.1f} s")
                label = f"UVMD-{N}" + (" untrained" if tag == "untrained" else "")
                rows, times = [], []
                P = uvmd_magn.export_params(model)
                max_dev = 0.0
                for i in te:
                    t0 = time.perf_counter()
                    u, om = uvmd_magn.decompose(model, X[i], a.device)
                    times.append(time.perf_counter() - t0)
                    rn = uvmd_forward_numpy(X[i].astype(np.float32).astype(np.float64), P["alpha"],
                                            P["lambda"], a.K)  # NumPy replica, cross-check
                    max_dev = max(max_dev, float(np.abs(rn["u"] - u).max() / (np.abs(rn["u"]).max() + 1e-12)))
                    m = ev(i, u, om, rn["spec_l1"])
                    m.update(method=label, signal=int(i), n_iter=N, time_s=times[-1])
                    rows.append(m)
                    if i == ex_i and tag == "trained":
                        examples[label] = match_modes(u, D["modes"][i])
                per_signal += rows
                s = aggregate(rows)
                s.update(params=npar, median_ms=1e3 * float(np.median(times)),
                         numpy_replica_max_rel_dev=max_dev, alpha_learned=P["alpha"].tolist())
                if tag == "trained":
                    s.update(train_time_s=train_s, epochs_run=h["epochs_run"],
                             final_train_loss=h["train_loss"][-1], best_val_loss=h["best_val_loss"])
                    np.savez(out / f"uvmd_params_N{N}.npz", **P)
                summary[label] = s
                cost_rows.append({
                    "method": label, "iterations_or_layers": N, "trainable_params": npar["real_dof"],
                    "flops_per_signal": cost.flops_uvmd_forward(a.T, a.K, N),
                    "median_ms_per_signal": s["median_ms"],
                    "train_time_s": s.get("train_time_s", 0.0),
                    "train_flops": cost.flops_uvmd_train(a.T, a.K, N, len(tr), h["epochs_run"])
                    if tag == "trained" else 0.0})
                log(f"  {label}: recon_rel_err {s['recon_rel_err']:.4f}  mode SNR {s['mean_mode_snr_db']:.2f} dB"
                    f"  median {s['median_ms']:.2f} ms  (numpy replica dev {max_dev:.1e})")

    # ---------------- save ----------------
    pd.DataFrame(per_signal).to_csv(out / "per_signal.csv", index=False)
    cdf = pd.DataFrame(cost_rows)
    cdf.to_csv(out / "cost.csv", index=False)
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
           "processor": platform.processor()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    json.dump({"args": vars(a), "env": env, "summary": summary}, open(out / "summary.json", "w"), indent=2,
              default=float)

    # ---------------- figures ----------------
    t = np.arange(a.T)
    plots.plot_example(out / "figures" / "example_decomposition.png", t, X[ex_i], D["modes"][ex_i],
                       D["meta"]["mode_names"], examples)
    pts = lambda m: {f"UVMD-{N} trained": (N, summary[f"UVMD-{N}"][m]) for N in a.layers  # noqa: E731
                     if f"UVMD-{N}" in summary}
    plots.plot_vs_iterations(out / "figures" / "mode_snr_vs_iterations.png", sweep, pts("mean_mode_snr_db"),
                             "mean_mode_snr_db", "mean mode SNR vs ground truth (dB)")
    plots.plot_vs_iterations(out / "figures" / "recon_err_vs_iterations.png", sweep, pts("recon_rel_err"),
                             "recon_rel_err", "relative reconstruction error", logy=True)
    plots.plot_runtime(out / "figures" / "runtime.png", cdf["method"].tolist(),
                       cdf["median_ms_per_signal"].tolist())
    if hists:
        plots.plot_training(out / "figures" / "uvmd_training.png", hists)

    # ---------------- markdown summary ----------------
    show = ["VMD (converged)"] + [f"VMD-{n}" for n in a.layers if f"VMD-{n}" in summary] \
        + [k for k in summary if k.startswith("UVMD")]
    cols = ["recon_rmse", "recon_rel_err", "recon_snr_db", "spec_l1", "denoise_rmse", "mean_mode_nrmse",
            "mean_mode_corr", "mean_mode_snr_db", "grouped_mean_snr_db", "mean_omega_err", "ortho_index"]
    md = [f"# Synthetic benchmark - VMD vs UVMD\n",
          f"{a.n_signals} signals, T={a.T}, 4 true modes, noise std {a.noise}, K={a.K}, alpha={a.alpha}, "
          f"tau={a.tau}. Test set: {len(te)} unseen signals. Mean over test signals.\n",
          "## Accuracy\n", "| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in show:
        md.append(f"| {k} | " + " | ".join(f"{summary[k].get(c, float('nan')):.4g}" for c in cols) + " |")
    md += ["\n## Cost (per signal unless stated)\n", cdf.to_markdown(index=False, floatfmt=".4g"),
           f"\nEnvironment: {env}\n"]
    (out / "summary.md").write_text("\n".join(md) + "\n\n## Log\n```\n" + "\n".join(log_lines) + "\n```\n")
    log(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
