"""Compare the modes of the end-to-end UVMD (trained through the CNN) with converged VMD and, when its parameter file
is available, with a UVMD trained on the reconstruction loss only, on the same unseen test windows.

    python experiments/compare_uvmd_modes.py --run results/cwru_e2e --data ~/data/CWRU --device cuda:0
    python experiments/compare_uvmd_modes.py --run results/cwru_e2e --data ~/data/CWRU --early-params results/cwru/uvmd_params_N2.npz

``--run`` is the output folder of experiments/run_cwru_e2e.py: it supplies the configuration (T, K, N, alpha, split, ...)
and one uvmd_e2e_params_N*_seed*.npz per seed. The recon-trained UVMD is optional: pass its uvmd_params_N*.npz with
--early-params (it must have the same K, N and T), or leave it out. Nothing is retrained.

What is measured, per test window, for every decomposition
  * parameters      alpha_k, lambda (norm, change relative to the early lambda), centre frequencies omega_k in Hz and their
                    distance from the uniform start grid (in grid spacings)
  * similarity      correlation of mode k of A with mode k of B (modes of every UVMD and VMD start from the same
                    ordered grid, so the index is meaningful) and the Hungarian-matched |correlation|; energy share of each mode
  * mode quality    relative reconstruction error, energy ratio, orthogonality index (decomp/metrics.py)
  * fault content   envelope-spectrum SNR at the true fault frequency in the max-kurtosis mode, hit rate, best-mode SNR
                    (decomp/fault_metrics.py)
For the end-to-end UVMD every seed is a separate model; tables give the mean over seeds (and the spread of the seed means).

Writes <run>/mode_comparison/: mode_comparison.md, parameters.csv, similarity.csv, quality.csv, per_window.csv, figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "vmdpy"))

from datasets.cwru import FS, fault_freqs, load_cwru, split_indices  # noqa: E402
from decomp.fault_metrics import fault_eval  # noqa: E402
from decomp.metrics import evaluate  # noqa: E402
from decomp.vmd_numpy import vmd  # noqa: E402
from e2e.export import decompose_params  # noqa: E402

EARLY, E2E_, VMDC = "UVMD recon-trained", "UVMD end-to-end", "VMD converged"


# ------------------------------------------------------------------------------------------------ decompositions
def decompose_uvmd(P, X, a, dev):
    modes, om = decompose_params(P, X, dev)
    return modes.astype(np.float64), om.astype(np.float64)


def decompose_vmd(X, a, n_iter=None):
    U, OM = [], []
    for f in X:
        r = vmd(f, a["alpha"], 0.0, a["K"], True, 1, -1.0 if n_iter else a.get("vmd_tol", 1e-7),
                n_iter if n_iter else a.get("vmd_max_iter", 499))
        U.append(r["u"])
        OM.append(r["omega"])
    return np.stack(U), np.stack(OM)


# ------------------------------------------------------------------------------------------------ statistics
def similarity(A, B):
    """A, B: (n, K, T). Returns aligned correlation (n, K) of mode k with mode k, and the Hungarian-matched mean |corr| (n,)."""
    a = A - A.mean(-1, keepdims=True)
    b = B - B.mean(-1, keepdims=True)
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    aligned = (an * bn).sum(-1)
    Cm = np.abs(np.einsum("nkt,njt->nkj", an, bn))
    matched = np.array([Cm[i][linear_sum_assignment(-Cm[i])].mean() for i in range(len(Cm))])
    return aligned, matched


def energy_share(U):
    e = (U ** 2).sum(-1)
    return e / e.sum(-1, keepdims=True)


def grid_shift(OM, K):
    """Distance of the final centre frequencies from the uniform start grid, in grid spacings (DC mode excluded)."""
    return np.abs(OM[:, 1:] - (0.5 / K) * np.arange(1, K)[None]) / (0.5 / K)


# ------------------------------------------------------------------------------------------------ main
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output folder of run_cwru_e2e.py")
    p.add_argument("--data", required=True, help="folder with the CWRU .mat files")
    p.add_argument("--early-params", default=None, help="uvmd_params_N*.npz of a UVMD trained on reconstruction only (optional)")
    p.add_argument("--n-windows", type=int, default=0, help="use only the first n test windows (0 = all)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None):
    ar = parse(argv)
    run = Path(ar.run)
    S = json.load(open(run / "summary.json"))
    a = {k: S["args"][k] for k in ("T", "windows_per_file", "split", "K", "layers", "alpha")}
    a.update(vmd_tol=S["args"]["vmd_tol"], vmd_max_iter=S["args"]["vmd_max_iter"])
    K, N = a["K"], a["layers"]
    out = Path(ar.out) if ar.out else run / "mode_comparison"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    dev = torch.device(ar.device)

    ep = ar.early_params or S["args"].get("early_params") or str(run / f"early_uvmd_params_N{N}_K{K}.npz")
    early = None
    if Path(ep).exists():
        early = dict(np.load(ep))
        if early["alpha"].shape != (K,) or early["lambda"].shape != (N, 2 * a["T"]):
            raise ValueError(f"{ep}: alpha {early['alpha'].shape}, lambda {early['lambda'].shape} do not fit this run "
                             f"(K={K}, N={N}, T={a['T']})")
    else:
        if ar.early_params:
            raise FileNotFoundError(ar.early_params)
        print("no recon-trained UVMD parameters found: comparing the end-to-end UVMD with converged VMD only", flush=True)
    e2e_files = sorted(run.glob(f"uvmd_e2e_params_N{N}_seed*.npz"))
    if not e2e_files:
        raise FileNotFoundError(f"no uvmd_e2e_params_N{N}_seed*.npz in {run}")
    e2e = {int(f.stem.split("seed")[1]): dict(np.load(f)) for f in e2e_files}

    D = load_cwru(ar.data, a["T"], a["windows_per_file"], True)
    _, _, te = split_indices(D, a["split"])
    if ar.n_windows:
        te = te[:ar.n_windows]
    X = D["x"][te]
    FF = [fault_freqs(r) for r in D["rpm"][te]]
    cls = D["cls"][te]
    print(f"{len(te)} test windows | K={K}, N={N}, alpha={a['alpha']:g} | recon-trained UVMD: "
          f"{ep if early else 'none'} | e2e seeds {sorted(e2e)}", flush=True)

    # ---- decompositions: label -> (family, seed, modes (n,K,T), omega (n,K))
    dec = {VMDC: (VMDC, -1, *decompose_vmd(X, a))}
    if early:
        dec[EARLY] = (EARLY, -1, *decompose_uvmd(early, X, a, dev))
    for s, P in e2e.items():
        dec[f"{E2E_} seed {s}"] = (E2E_, s, *decompose_uvmd(P, X, a, dev))
    # guard: an untrained batched UVMD must equal VMD stopped after N iterations
    chk = decompose_uvmd({"alpha": np.full(K, a["alpha"]), "lambda": np.zeros((N, 2 * a["T"]), complex)}, X[:3], a, dev)[0]
    ref = decompose_vmd(X[:3], a, N)[0]
    assert np.abs(chk - ref).max() < 2e-3 * np.abs(ref).max(), "untrained batched UVMD differs from VMD-N"

    # ---- per-window quality and fault content
    rows = []
    for lab, (fam, seed, U, OM) in dec.items():
        for j in range(len(te)):
            m = evaluate(X[j], U[j], omega=OM[j])
            m.update(fault_eval(U[j], X[j], cls[j], FF[j], FS, OM[j]))
            m.update(method=lab, family=fam, seed=seed, window=int(te[j]), cls=str(cls[j]))
            rows.append(m)
    pw = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    pw.to_csv(out / "per_window.csv", index=False)
    cols = ["recon_rel_err", "energy_ratio", "ortho_index", "fault_snr_db", "envelope_hit", "oracle_fault_snr_db"]
    per_model = pw.groupby(["family", "method"], sort=False)[cols].mean().reset_index()
    quality = per_model.groupby("family", sort=False)[cols].mean()
    quality_sd = per_model.groupby("family", sort=False)[cols].std().fillna(0.0)
    quality.to_csv(out / "quality.csv")

    # ---- parameters
    ptab = []
    lam_e = early["lambda"] if early else None
    if early:
        ptab.append({"model": EARLY, **{f"alpha_{k}": early["alpha"][k] for k in range(K)},
                     "lambda_norm": float(np.linalg.norm(lam_e)), "lambda_change_vs_recon_trained": 0.0})
    for s, P in e2e.items():
        rel = float(np.linalg.norm(P["lambda"] - lam_e) / max(np.linalg.norm(lam_e), 1e-12)) if early else np.nan
        ptab.append({"model": f"{E2E_} seed {s}", **{f"alpha_{k}": P["alpha"][k] for k in range(K)},
                     "lambda_norm": float(np.linalg.norm(P["lambda"])), "lambda_change_vs_recon_trained": rel})
    ptab.append({"model": "start (untrained UVMD)", **{f"alpha_{k}": a["alpha"] for k in range(K)},
                 "lambda_norm": 0.0, "lambda_change_vs_recon_trained": np.nan})
    ptab = pd.DataFrame(ptab)
    if not early:
        ptab = ptab.drop(columns="lambda_change_vs_recon_trained")
    om_rows = []
    for lab, (fam, seed, U, OM) in dec.items():
        sh = grid_shift(OM, K)
        om_rows.append({"model": lab, "family": fam, **{f"omega_{k}_Hz": float(OM[:, k].mean() * FS) for k in range(K)},
                        "mean_shift_grid": float(sh.mean())})
    otab = pd.DataFrame(om_rows)
    ptab.to_csv(out / "parameters.csv", index=False)
    otab.to_csv(out / "centre_frequencies.csv", index=False)

    # ---- similarity between decompositions
    pairs = [(E2E_, VMDC)] + ([(E2E_, EARLY), (EARLY, VMDC)] if early else [])
    srows = []
    for A_, B_ in pairs:
        per_seed = []
        for la in [l for l, v in dec.items() if v[0] == A_]:
            for lb in [l for l, v in dec.items() if v[0] == B_]:
                al, mt = similarity(dec[la][2], dec[lb][2])
                per_seed.append((al.mean(0), mt.mean()))
        al = np.mean([p[0] for p in per_seed], axis=0)
        mt = [p[1] for p in per_seed]
        srows.append({"A": A_, "B": B_, **{f"corr_mode_{k}": float(al[k]) for k in range(K)},
                      "aligned_mean": float(al.mean()), "matched_mean_abs": float(np.mean(mt)),
                      "matched_spread": float(np.std(mt))})
    stab = pd.DataFrame(srows)
    stab.to_csv(out / "similarity.csv", index=False)
    fams = [f for f in (VMDC, EARLY, E2E_) if any(v[0] == f for v in dec.values())]
    share = pd.DataFrame({fam: energy_share(np.concatenate([v[2] for v in dec.values() if v[0] == fam])).mean(0)
                          for fam in fams}, index=[f"mode {k}" for k in range(K)])

    # ---- by class
    bycls = (pw.groupby(["family", "cls"], sort=False)[["fault_snr_db", "envelope_hit", "oracle_fault_snr_db"]]
             .mean().reset_index().rename(columns={"family": "method"}))
    spread = quality_sd[quality_sd.index == E2E_] if len(e2e) > 1 else None

    md = [f"# Mode comparison: end-to-end UVMD vs VMD" + (" vs recon-trained UVMD" if early else "") + "\n",
          f"{len(te)} unseen test windows (split '{a['split']}'), T={a['T']}, K={K}, N={N} layers, alpha={a['alpha']:g}, "
          f"e2e seeds {sorted(e2e)}." + (f" Recon-trained UVMD: `{ep}`." if early else
                                         " No recon-trained UVMD was given (`--early-params`).") +
          " For the end-to-end UVMD, numbers are means over seeds.\n",
          "## Learned parameters\n", ptab.round(3).to_markdown(index=False),
          f"\n`alpha_k` is the bandwidth penalty of mode k: the filter that carves mode k out of the spectrum is "
          f"1 / (1 + alpha_k (f - omega_k)^2), so a larger alpha_k gives a narrower mode and alpha_k near 0 lets everything "
          f"through. Every model starts at alpha = {a['alpha']:g} and lambda = 0."
          + (" `lambda_change_vs_recon_trained` = ||lambda - lambda_recon||_F / ||lambda_recon||_F." if early else "") + "\n",
          "## Centre frequencies of the final layer (mean over test windows) and distance from the start grid\n",
          otab.round(3).to_markdown(index=False),
          "\n## Similarity of the modes (mode k vs mode k; Hungarian-matched |correlation|)\n",
          stab.round(3).to_markdown(index=False), "\n## Energy share per mode\n", share.round(3).to_markdown(),
          "\n## Mode quality and fault content (mean over test windows; end-to-end: mean over seeds)\n",
          quality.round(4).to_markdown()]
    if spread is not None:
        md += ["\nStandard deviation across the end-to-end seeds (of the per-seed means):\n", spread.round(4).to_markdown()]
    md += ["\n## Fault content by class (normal windows have no fault frequency, hence no entry)\n",
           bycls.dropna(subset=["fault_snr_db"]).round(3).to_markdown(index=False)]
    (out / "mode_comparison.md").write_text("\n".join(md) + "\n")

    figures(out, a, dec, X, cls, FF, te, pw, otab, stab, early is not None)
    print(f"written to {out}")
    return {"similarity": stab, "parameters": ptab, "quality": quality, "centre": otab, "dec": dec}


def figures(out, a, dec, X, cls, FF, te, pw, otab, stab, has_early):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments import plots
    K = a["K"]
    ex = int(next((j for j in range(len(te)) if cls[j] == "inner"), 0))
    seed0 = min(v[1] for v in dec.values() if v[0] == E2E_)
    show = ([EARLY] if has_early else []) + [f"{E2E_} seed {seed0}", VMDC]
    decomps = {}
    for lab in show:
        U = dec[lab][2][ex]
        sel = int(pw[(pw.method == lab) & (pw.window == te[ex])].sel_mode.iloc[0])
        decomps[lab] = (U, sel)
    ff = FF[ex]
    plots.plot_modes_envelope(out / "figures" / "example_modes.png", X[ex], FS, decomps,
                              ff[str(cls[ex])] if cls[ex] != "normal" else ff["shaft"],
                              f"CWRU test window {te[ex]} ({cls[ex]})")

    # mean magnitude spectrum of every mode
    freqs = np.fft.rfftfreq(a["T"], 1 / FS)
    fig, axes = plt.subplots(1, len(show), figsize=(3.8 * len(show), 3.2), sharey=True, squeeze=False)
    for ax, lab in zip(axes[0], show):
        S = np.abs(np.fft.rfft(dec[lab][2], axis=-1)).mean(0)          # (K, F)
        for k in range(K):
            ax.plot(freqs, S[k], lw=1, label=f"mode {k}")
        ax.set_title(lab, fontsize=8, loc="left")
        ax.set_xlabel("Hz")
        plots._style(ax)
    axes[0, 0].set_ylabel("mean |FFT| over test windows")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "figures" / "mode_spectra.png", dpi=130)
    plt.close(fig)

    # centre frequencies and similarity of the modes
    ks = np.arange(K)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    series = [(VMDC, plots.MUTED, -0.12, "s")] + ([(EARLY, plots.C[0], 0.0, "o")] if has_early else []) + \
             [(E2E_, plots.C[1], 0.12, "D")]
    for lab, col, off, mk in series:
        v = otab[otab.family == lab][[f"omega_{k}_Hz" for k in range(K)]].values
        ax.errorbar(ks + off, v.mean(0), yerr=v.std(0) if len(v) > 1 else None, fmt=mk, color=col, ms=6, capsize=3,
                    label=lab)
    ax.set_xticks(ks)
    ax.set_xlabel("mode k")
    ax.set_ylabel("mean centre frequency of mode k (Hz)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)
    plots._style(ax)
    ax = axes[1]
    short = {EARLY: "recon-trained", E2E_: "end-to-end", VMDC: "VMD converged"}
    w = 0.8 / len(stab)
    for i, r in stab.iterrows():
        ax.bar(ks - 0.4 + w / 2 + i * w, [r[f"corr_mode_{k}"] for k in range(K)], w, label=f"{short[r.A]} vs {short[r.B]}",
               color=plots.C[i % len(plots.C)])
    ax.set_xticks(ks)
    ax.set_xlabel("mode k")
    ax.set_ylabel("correlation of mode k with mode k")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)
    plots._style(ax)
    fig.tight_layout()
    fig.savefig(out / "figures" / "centre_frequencies_and_similarity.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
