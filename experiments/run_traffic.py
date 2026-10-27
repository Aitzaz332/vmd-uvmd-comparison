"""Dataset 3 - LargeST traffic flow (the dataset of the UVMD paper): VMD vs UVMD signal decomposition.

Protocol of the paper (arXiv:2509.00703, Sec. IV) and of the authors' ``uvmd.py``:
  * one signal per sensor node = the whole year 2019 at 15 minutes (35040 samples), flow standardised globally
  * nodes split 70% / 15% / 15% (train / val / test), random_state=42
  * UVMD: K=13 modes, 1-2 layers, alpha initialised at 2000, lambda (H) at zeros, Adam lr 0.1, batch size 1,
    100 epochs, early stopping (patience 10), reconstruction loss only
  * VMD baseline: classic ADMM VMD, same K, alpha=2000, run to convergence (and stopped after n iterations)

    # 0) installation check, no data needed (generated stand-in, NOT real traffic)
    python experiments/run_traffic.py --data demo --quick
    # 1) real data (see docs/TRAFFIC.md for the download and preparation)
    python experiments/run_traffic.py --data ~/data/LargeST/sd_his_2019.h5 --region sd --device cuda:0 \\
        --max-train-nodes 100 --max-test-nodes 30           # bounded run first
    python experiments/run_traffic.py --data ~/data/LargeST/sd_his_2019.h5 --region sd --device cuda:0   # full

No ground-truth modes exist for traffic, so the evaluation is
  * reconstruction: relative error, SNR, spectral L1 (upstream loss) and L2 (paper's L_rec), energy ratio,
    orthogonality (decomp/metrics.py)
  * spectral structure of the modes: bandwidth, mode overlap, who owns the daily / weekly lines (decomp/traffic_metrics.py)
  * cost: decomposition time per node and extrapolated to the whole dataset, FLOPs, parameters, training time
  * the learned bandwidth constraints phi(alpha_k) and the loss per epoch (the paper's Fig. 2 and 3)
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

from datasets.traffic import STEPS_PER_DAY, load_traffic, split_nodes  # noqa: E402
from decomp import cost  # noqa: E402
from decomp.metrics import evaluate  # noqa: E402
from decomp.traffic_metrics import spectral_metrics, win_rate  # noqa: E402
from decomp.vmd_numpy import uvmd_forward_numpy, vmd  # noqa: E402
from experiments import plots  # noqa: E402

PAPER_SENSORS = 6902  # GBA 2352 + GLA 3834 + SD 716
REC = ["recon_rel_err", "recon_snr_db", "spec_l1", "spec_l2", "energy_ratio", "ortho_index"]
SPEC = ["mean_bandwidth_cpd", "mode_overlap", "n_active_modes", "daily_ownership", "daily_capture",
        "weekly_ownership", "weekly_capture"]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="region h5 (e.g. sd_his_2019.h5), authors' his.npz, or 'demo'")
    p.add_argument("--region", default="", help="label for titles (gba / gla / sd)")
    p.add_argument("--length", type=int, default=None,
                   help="use only the first L samples of each node (paper's window cases: 35040, 17520, 8760, 4380)")
    p.add_argument("--K", type=int, default=13, help="number of modes (paper: 13)")
    p.add_argument("--alpha", type=float, default=2000.0)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--max-iter", type=int, default=499)
    p.add_argument("--vmd-iters", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 200])
    p.add_argument("--sweep-nodes", type=int, default=10, help="test nodes used for the VMD-n sweep")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2], help="UVMD layers (paper: 1-2)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--restore-best", action="store_true")
    p.add_argument("--max-train-nodes", type=int, default=None)
    p.add_argument("--max-val-nodes", type=int, default=None)
    p.add_argument("--max-test-nodes", type=int, default=None)
    p.add_argument("--cpu-timing-nodes", type=int, default=3,
                   help="when --device is cuda, also time trained UVMD on the CPU for this many test nodes")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-uvmd", action="store_true")
    p.add_argument("--demo-nodes", type=int, default=60)
    p.add_argument("--demo-days", type=int, default=28)
    p.add_argument("--quick", action="store_true", help="tiny run: 3 epochs, few nodes")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.quick:
        a.epochs, a.vmd_iters, a.sweep_nodes = 3, [1, 2, 20], 3
        a.max_train_nodes, a.max_val_nodes, a.max_test_nodes = 6, 3, 4
    if a.out is None:
        a.out = str(ROOT / "results" / ("traffic_demo" if a.data == "demo" else f"traffic_{a.region or 'run'}"))
    return a


def agg(rows):
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    out = {}
    for c in REC + SPEC:
        if c in df:
            out[c] = float(df[c].mean())
            out[c + "_std"] = float(df[c].std())
    return out


def main():
    a = parse()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    D = load_traffic(a.data, a.length, {"n_nodes": a.demo_nodes, "days": a.demo_days} if a.data == "demo" else None)
    X = D["x"]
    N, T = X.shape
    tr, va, te = split_nodes(N)
    tr, va, te = (s[:c] if c else s for s, c in zip((tr, va, te), (a.max_train_nodes, a.max_val_nodes,
                                                                      a.max_test_nodes)))
    log(f"traffic [{D['meta']['source']}]: {N} nodes x {T} samples ({T / STEPS_PER_DAY:.0f} days) | node split "
        f"train/val/test = {len(tr)}/{len(va)}/{len(te)} | K={a.K}, alpha={a.alpha}, device {a.device}")
    if a.data == "demo":
        log("  NOTE: generated stand-in data, only for checking the installation. Results are NOT real traffic.")

    per, summ, costs, sweep, examples = [], {}, [], [], {}
    ex_i = int(te[0])

    def ev(i, u, omega, l1, l2, method, n_iter, t):
        m = evaluate(X[i], u, omega=omega, spec_l1=l1)
        m["spec_l2"] = l2
        m.update(spectral_metrics(X[i], u, STEPS_PER_DAY))
        m.update(method=method, node=int(i), n_iter=n_iter, time_s=t)
        return m

    def cost_row(method, n, params, flops, ts, train_s=0.0, train_flops=0.0, device="cpu"):
        costs.append({"method": method, "iterations_or_layers": n, "trainable_params": params,
                      "flops_per_node": flops, "mean_s_per_node": float(np.mean(ts)),
                      "median_s_per_node": float(np.median(ts)), "device": device,
                      "train_time_s": train_s, "train_flops": train_flops})

    # ---------------- VMD converged ----------------
    log("\n[VMD converged] (NumPy, CPU)")
    rows, its, ts = [], [], []
    for n_done, i in enumerate(te, 1):
        t0 = time.perf_counter()
        r = vmd(X[i], a.alpha, a.tau, a.K, True, 1, a.tol, a.max_iter)
        ts.append(time.perf_counter() - t0)
        its.append(r["n_iter"])
        rows.append(ev(i, r["u"], r["omega"], r["spec_l1"], r["spec_l2"], "VMD (converged)", r["n_iter"], ts[-1]))
        if i == ex_i:
            examples["VMD (converged)"] = r["u"]
        if n_done % 5 == 0 or n_done == len(te):
            log(f"  {n_done}/{len(te)} nodes, {np.mean(ts):.2f} s/node, mean {np.mean(its):.0f} iterations")
    per += rows
    s = agg(rows)
    s.update(mean_iters=float(np.mean(its)), converged_frac=float(np.mean(np.array(its) < a.max_iter)))
    summ["VMD (converged)"] = s
    cost_row("VMD (converged)", round(float(np.mean(its)), 1), 0, cost.flops_vmd(T, a.K, float(np.mean(its)), a.tau), ts)
    log(f"  rel err {s['recon_rel_err']:.4f}, mode overlap {s['mode_overlap']:.3f}, "
        f"daily-line ownership {s['daily_ownership']:.2f}, {100 * s['converged_frac']:.0f}% converged")

    # ---------------- VMD-n sweep ----------------
    log("\n[VMD-n sweep]")
    sub = te[:a.sweep_nodes]
    for n in sorted(set(a.vmd_iters) | set(a.layers)):
        idx = te if n in a.layers else sub
        rows, ts = [], []
        for i in idx:
            t0 = time.perf_counter()
            r = vmd(X[i], a.alpha, a.tau, a.K, True, 1, -1.0, n)
            ts.append(time.perf_counter() - t0)
            rows.append(ev(i, r["u"], r["omega"], r["spec_l1"], r["spec_l2"], f"VMD-{n}", n, ts[-1]))
        per += rows
        s = agg(rows)
        s.update(n_iter=n, n_nodes=len(idx))
        summ[f"VMD-{n}"] = s
        sweep.append(s)
        cost_row(f"VMD-{n}", n, 0, cost.flops_vmd(T, a.K, n, a.tau), ts)
        log(f"  n={n:>3} ({len(idx)} nodes): rel err {s['recon_rel_err']:.4f}  overlap {s['mode_overlap']:.3f}  "
            f"{np.mean(ts):.3f} s/node")

    # ---------------- UVMD (upstream code) ----------------
    hists, alphas = {}, {}
    if not a.skip_uvmd:
        import torch
        from decomp import uvmd_magn
        for N_l in a.layers:
            log(f"\n[UVMD N={N_l}] upstream Unfolding, K={a.K}, device {a.device}")
            torch.manual_seed(a.seed)
            model = uvmd_magn.make_model(T, a.K, N_l, a.device, a.alpha)
            npar = uvmd_magn.count_params(model)
            log(f"  trainable parameters: {npar['numel']} entries ({npar['real_dof']} real DOF)")
            for tag in ("untrained", "trained"):
                train_s, h = 0.0, None
                if tag == "trained":
                    t0 = time.perf_counter()
                    h = uvmd_magn.train_uvmd(model, X[tr], X[va], a.epochs, a.lr, a.patience, a.device, a.seed,
                                             a.restore_best, log=log)
                    train_s = time.perf_counter() - t0
                    hists[f"UVMD-{N_l}"] = h
                    log(f"  training: {h['epochs_run']} epochs, {train_s:.1f} s")
                lab = f"UVMD-{N_l}" + (" untrained" if tag == "untrained" else "")
                P = uvmd_magn.export_params(model)
                rows, ts = [], []
                for i in te:
                    if a.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    u, om = uvmd_magn.decompose(model, X[i], a.device)
                    ts.append(time.perf_counter() - t0)
                    rn = uvmd_forward_numpy(X[i].astype(np.float64), P["alpha"], P["lambda"], a.K)
                    rows.append(ev(i, u, om, rn["spec_l1"], rn["spec_l2"], lab, N_l, ts[-1]))
                    if i == ex_i and tag == "trained":
                        examples[lab] = u
                per += rows
                s = agg(rows)
                s.update(params=npar, alpha_learned=P["alpha"].tolist())
                if h:
                    s.update(train_time_s=train_s, epochs_run=h["epochs_run"], best_val_loss=h["best_val_loss"],
                             final_train_loss=h["train_loss"][-1])
                    alphas[f"UVMD-{N_l}"] = P["alpha"]
                    np.savez(out / f"uvmd_params_N{N_l}.npz", **P)
                summ[lab] = s
                cost_row(lab, N_l, npar["real_dof"], cost.flops_uvmd_forward(T, a.K, N_l), ts, train_s,
                         cost.flops_uvmd_train(T, a.K, N_l, len(tr), h["epochs_run"]) if h else 0.0, a.device)
                log(f"  {lab}: rel err {s['recon_rel_err']:.4f}  overlap {s['mode_overlap']:.3f}  "
                    f"daily-line ownership {s['daily_ownership']:.2f}  {np.mean(ts):.3f} s/node")
                if tag == "trained":
                    log(f"  learned alpha_k: {np.round(P['alpha']).astype(int).tolist()}")
                    if a.device.startswith("cuda") and a.cpu_timing_nodes:
                        cpu_model = uvmd_magn.clone_to_cpu(model, T, a.K, N_l, a.alpha)
                        cts = []
                        for i in te[:a.cpu_timing_nodes]:
                            t0 = time.perf_counter()
                            uvmd_magn.decompose(cpu_model, X[i], "cpu")
                            cts.append(time.perf_counter() - t0)
                        cost_row(f"{lab} (CPU)", N_l, npar["real_dof"], cost.flops_uvmd_forward(T, a.K, N_l), cts,
                                 device="cpu")
                        log(f"  same trained model on CPU: {np.mean(cts):.3f} s/node ({len(cts)} nodes)")

    # ---------------- save ----------------
    dfp = pd.DataFrame(per)
    dfp.to_csv(out / "per_node.csv", index=False)
    cdf = pd.DataFrame(costs)
    vmd_s = cdf.loc[cdf.method == "VMD (converged)", "mean_s_per_node"].iloc[0]
    cdf["est_min_region"] = cdf.mean_s_per_node * N / 60
    cdf["est_min_6902_sensors"] = cdf.mean_s_per_node * PAPER_SENSORS / 60
    cdf["speedup_vs_VMD_converged"] = vmd_s / cdf.mean_s_per_node
    cdf.to_csv(out / "cost.csv", index=False)
    wins = {}
    for n in a.layers:
        if f"UVMD-{n}" in summ:
            wins[f"UVMD-{n} vs VMD-{n}"] = {
                m: win_rate(dfp, f"UVMD-{n}", f"VMD-{n}", m, lower)
                for m, lower in (("recon_rel_err", True), ("mode_overlap", True), ("mean_bandwidth_cpd", True))}
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    json.dump({"args": vars(a), "env": env, "data": D["meta"], "summary": summ, "win_rates": wins},
              open(out / "summary.json", "w"), indent=2, default=float)

    # ---------------- figures ----------------
    title = f"traffic node {D['node_ids'][ex_i]} ({D['meta']['source']}), first 7 days of {T / STEPS_PER_DAY:.0f}"
    show_ex = {k: v for k, v in examples.items()}
    plots.plot_traffic_modes(out / "figures" / "example_decomposition.png", X[ex_i], show_ex, STEPS_PER_DAY, 7,
                             title=title)
    plots.plot_vs_iterations(out / "figures" / "recon_err_vs_iterations.png", sweep,
                             {f"UVMD-{n} trained": (n, summ[f"UVMD-{n}"]["recon_rel_err"]) for n in a.layers
                              if f"UVMD-{n}" in summ}, "recon_rel_err", "relative reconstruction error", logy=True)
    plots.plot_vs_iterations(out / "figures" / "mode_overlap_vs_iterations.png", sweep,
                             {f"UVMD-{n} trained": (n, summ[f"UVMD-{n}"]["mode_overlap"]) for n in a.layers
                              if f"UVMD-{n}" in summ}, "mode_overlap", "mean spectral overlap between modes")
    plots.plot_runtime(out / "figures" / "runtime.png", cdf["method"].tolist(), (1e3 * cdf["mean_s_per_node"]).tolist())
    if hists:
        plots.plot_training(out / "figures" / "uvmd_training.png", hists)
        plots.plot_alpha(out / "figures" / "learned_bandwidth.png", alphas)

    # ---------------- markdown ----------------
    show = ["VMD (converged)"] + [f"VMD-{n}" for n in a.layers] + [k for k in summ if k.startswith("UVMD")]
    cols = REC + SPEC
    md = [f"# Traffic benchmark - VMD vs UVMD ({a.region or D['meta']['source']})\n",
          f"{N} nodes x {T} samples ({T / STEPS_PER_DAY:.0f} days at 15 min), standardised flow. Node split train/val/test = "
          f"{len(tr)}/{len(va)}/{len(te)} (test nodes scored). K={a.K}, alpha={a.alpha}, tau={a.tau}, UVMD device {a.device}.\n"
          + ("**Generated stand-in data - not real traffic.**\n" if a.data == "demo" else ""),
          "## Accuracy and mode structure (mean over test nodes)\n", "| method | " + " | ".join(cols) + " |",
          "|---" * (len(cols) + 1) + "|"]
    for k in show:
        md.append(f"| {k} | " + " | ".join(f"{summ[k].get(c, float('nan')):.4g}" for c in cols) + " |")
    if wins:
        md += ["\n## Paired comparison, trained UVMD-N vs VMD-N (share of nodes where UVMD is better)\n",
               pd.DataFrame(wins).T.to_markdown(floatfmt=".2f")]
    md += ["\n## Cost\n", cdf.to_markdown(index=False, floatfmt=".4g"),
           "\n`est_min_*` = mean seconds per node x number of nodes (an extrapolation, not a measurement). "
           f"{PAPER_SENSORS} = GBA+GLA+SD sensors of the paper. VMD is NumPy on CPU; UVMD runs on the device above.\n",
           f"Environment: {env}\n"]
    (out / "summary.md").write_text("\n".join(md) + "\n\n## Log\n```\n" + "\n".join(lines) + "\n```\n")
    log(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
