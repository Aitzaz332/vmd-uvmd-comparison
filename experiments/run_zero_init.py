"""What if every centre frequency starts at zero? VMD and UVMD on the synthetic data (K=5, alpha=500, 5 layers).

    python experiments/run_zero_init.py

The default start is the uniform grid omega_k = (0.5 / K) * k. Here all omega_k start at 0 (mode 1 is the DC mode and
is always 0). Everything else is as in run_synthetic.py (same signals, same split, same upstream training). The result is
compared with the uniform start taken from ``results/synthetic_K5_alpha500_L5/per_signal.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.synthetic import make_synthetic  # noqa: E402
from decomp import uvmd_magn  # noqa: E402
from decomp.metrics import evaluate  # noqa: E402
from decomp.vmd_numpy import vmd  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--alpha", type=float, default=500.0)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--n-signals", type=int, default=200)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--uniform", default=str(ROOT / "results" / "synthetic_K5_alpha500_L5" / "per_signal.csv"))
    p.add_argument("--out", default=str(ROOT / "results" / "synthetic_K5_alpha500_zeroinit"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    D = make_synthetic(a.n_signals, a.T, a.noise, a.seed)
    idx = np.arange(a.n_signals)
    tr, te = train_test_split(idx, test_size=0.3, random_state=42, shuffle=True)
    va, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
    X, K = D["x"], a.K

    def ev(i, u, om):
        return evaluate(X[i], u, omega=om, clean=D["clean"][i], true_modes=D["modes"][i], true_fc=D["centre_freq"][i])

    rows, omegas = [], {}

    # ---- VMD with all centre frequencies starting at 0
    for label, n, tol in (("VMD converged", 499, 1e-7), ("VMD-5", 5, -1.0)):
        om_all = []
        for i in te:
            r = vmd(X[i], a.alpha, 0.0, K, True, 0, tol, n)
            m = ev(i, r["u"], r["omega"])
            rows.append({"method": label, "start": "zero", "signal": int(i), **{k: m[k] for k in m if k in (
                "recon_rel_err", "mean_mode_nrmse", "mode_nrmse_0", "mode_nrmse_1", "mode_nrmse_2", "mode_nrmse_3")}})
            om_all.append(np.sort(r["omega"]))
        omegas[label] = np.mean(om_all, axis=0)

    # ---- UVMD with all centre frequencies starting at 0 (upstream model, only the start changed)
    torch.manual_seed(a.seed)
    model = uvmd_magn.make_model(a.T, K, a.layers, "cpu", a.alpha)
    model.omega_plus[0, :] = 0.0
    assert float(model.omega_plus.abs().sum()) == 0.0

    def _modes(j, u, om):
        m = ev(te[j], u, om)
        return {"mode_nrmse": m["mean_mode_nrmse"]}

    t0 = time.perf_counter()
    h = uvmd_magn.train_uvmd(model, X[tr], X[va], a.epochs, a.lr, a.patience, "cpu", a.seed, False,
                             log=lambda s: print(s, flush=True),
                             epoch_eval=uvmd_magn.recon_eval_fn(X[te], "cpu", _modes))
    print(f"training {time.perf_counter() - t0:.0f} s, {h['epochs_run']} epochs")
    json.dump(h, open(out / f"uvmd_history_N{a.layers}.json", "w"))
    om_all = []
    for i in te:
        u, om = uvmd_magn.decompose(model, X[i], "cpu")
        m = ev(i, u, om)
        rows.append({"method": f"UVMD-{a.layers}", "start": "zero", "signal": int(i), **{k: m[k] for k in m if k in (
            "recon_rel_err", "mean_mode_nrmse", "mode_nrmse_0", "mode_nrmse_1", "mode_nrmse_2", "mode_nrmse_3")}})
        om_all.append(np.sort(om))
    omegas[f"UVMD-{a.layers}"] = np.mean(om_all, axis=0)
    P = uvmd_magn.export_params(model)
    np.savez(out / f"uvmd_params_N{a.layers}.npz", **P)

    # ---- uniform start (from the earlier run) for the same methods
    old = pd.read_csv(a.uniform)
    for label, src in (("VMD converged", "VMD (converged)"), ("VMD-5", "VMD-5"), (f"UVMD-{a.layers}", f"UVMD-{a.layers}")):
        s = old[old.method == src]
        for _, r in s.iterrows():
            rows.append({"method": label, "start": "uniform", "signal": int(r.signal), **{k: r[k] for k in (
                "recon_rel_err", "mean_mode_nrmse", "mode_nrmse_0", "mode_nrmse_1", "mode_nrmse_2", "mode_nrmse_3")}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_signal.csv", index=False)

    names = ["trend", "tone", "AM tone", "chirp"]
    tab = []
    for meth in ("VMD converged", "VMD-5", f"UVMD-{a.layers}"):
        for st in ("uniform", "zero"):
            s = df[(df.method == meth) & (df.start == st)]
            tab.append({"method": meth, "start": st,
                        "reconstruction error %": 100 * s.recon_rel_err.mean(),
                        "error vs true modes % (mean of 4)": 100 * s.mean_mode_nrmse.mean(),
                        **{f"{n} %": 100 * s[f"mode_nrmse_{j}"].mean() for j, n in enumerate(names)}})
    T = pd.DataFrame(tab)
    T.to_csv(out / "zero_vs_uniform_start.csv", index=False)
    md = ["# Starting all centre frequencies at zero (synthetic, K=%d, alpha=%g)\n" % (K, a.alpha),
          T.to_markdown(index=False, floatfmt=".3g"),
          "\nMean final centre frequencies (cycles/sample, modes sorted), zero start:\n"]
    md += [f"- {k}: {np.array2string(v, precision=3)}" for k, v in omegas.items()]
    md += [f"\nLearned alpha, UVMD zero start: {np.array2string(np.sort(P['alpha']), precision=4)}\n"]
    (out / "zero_vs_uniform_start.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
