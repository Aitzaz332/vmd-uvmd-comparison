"""Readable mode figures at a fixed K: VMD and UVMD side by side (and the true modes on synthetic data).

Needs only the parameters saved by the benchmark runners (``uvmd_params_N*.npz``) - no retraining, no torch:
the trained UVMD is evaluated with the NumPy replica (``decomp.vmd_numpy.uvmd_forward_numpy``).

    # synthetic (runs anywhere; truth known, so it also writes the error table against the true modes)
    python experiments/mode_figures.py synthetic --dir results/synthetic_K5

    # CWRU / traffic (data only on node 7)
    python experiments/mode_figures.py cwru    --dir results/cwru_K4 --data ~/data/CWRU
    python experiments/mode_figures.py traffic --dir results/traffic_K6 --data ~/data/LargeST/sd_his_2019.h5

Figures and CSV tables go to ``<dir>/readable`` (``--out`` to change). ``--layers`` picks the UVMD depth drawn in
the big figure (default 2); the tables always list both depths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decomp.metrics import evaluate, match_modes  # noqa: E402
from decomp.vmd_numpy import uvmd_forward_numpy, vmd  # noqa: E402

# one colour per method, fixed everywhere
TRUTH, VMD_C, UVMD_C, UVMD1_C = "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c3"
COLOR = {"Original": TRUTH, "VMD": VMD_C}
_UV = [UVMD_C, UVMD1_C, "#6250d6", "#eda100"]


DEPTHS: list = []     # set by main() from the saved parameter files; fixes one colour per UVMD depth


def color_of(m):
    if m in COLOR:
        return COLOR[m]
    N = int(m.split("-")[1])
    return _UV[DEPTHS.index(N) % len(_UV)] if N in DEPTHS else _UV[0]


def _style(ax, bottom=False):
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if not bottom:
        ax.tick_params(labelbottom=False)


# ------------------------------------------------------------------ decompositions
def find_depths(d: Path):
    """UVMD depths (number of layers) that have saved parameters in the results folder."""
    ns = sorted(int(f.stem.split("_N")[1]) for f in d.glob("uvmd_params_N*.npz"))
    if not ns:
        raise FileNotFoundError(f"no uvmd_params_N*.npz in {d}")
    return ns


def load_params(d: Path, N: int):
    f = d / f"uvmd_params_N{N}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run the benchmark for this K first")
    z = np.load(f)
    return {"alpha": z["alpha"], "lambda": z["lambda"]}


def run_vmd(x, K, args):
    r = vmd(x, args.get("alpha", 2000.0), args.get("tau", 0.0), K, True, 1, args.get("tol", 1e-7),
            args.get("max_iter", 499))
    return r["u"], r["omega"]


def run_uvmd(x, K, P):
    r = uvmd_forward_numpy(np.asarray(x, dtype=np.float32).astype(np.float64), P["alpha"], P["lambda"], K)
    return r["u"], r["omega"]


def decompose(x, K, args, Ps: dict):
    out = {"VMD": run_vmd(x, K, args)}
    for N, P in Ps.items():
        out[f"UVMD-{N}"] = run_uvmd(x, K, P)
    return out


# ------------------------------------------------------------------ synthetic
def synthetic(a):
    from datasets.synthetic import make_synthetic
    d = Path(a.dir)
    S = json.load(open(d / "summary.json"))["args"]
    K, T = S["K"], S["T"]
    D = make_synthetic(S["n_signals"], T, S["noise"], S["seed"])
    idx = np.arange(S["n_signals"])
    _, te = train_test_split(idx, test_size=0.3, random_state=42, shuffle=True)
    _, te = train_test_split(te, test_size=0.5, random_state=42, shuffle=True)
    depths = find_depths(d)
    DEPTHS[:] = depths
    Ps = {N: load_params(d, N) for N in depths}
    names = D["meta"]["mode_names"]
    J = len(names)
    rows = []
    cache = {}
    for i in te:
        dec = decompose(D["x"][i], K, S, Ps)
        cache[int(i)] = dec
        clean = D["clean"][i]
        for meth, (u, om) in dec.items():
            m = evaluate(D["x"][i], u, omega=om, clean=clean, true_modes=D["modes"][i], true_fc=D["centre_freq"][i])
            rec = u.sum(0)
            row = {"method": meth, "signal": int(i),
                   "sum_vs_clean_rel_err": float(np.linalg.norm(clean - rec) / np.linalg.norm(clean)),
                   "sum_vs_input_rel_err": m["recon_rel_err"], "extra_mode_energy_share": m["spurious_energy_share"]}
            for j in range(J):
                row[f"nrmse_{j}"] = m[f"mode_nrmse_{j}"]
                row[f"snr_{j}"] = m[f"mode_snr_db_{j}"]
                row[f"corr_{j}"] = m[f"mode_corr_{j}"]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(a.out or d / "readable")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "synthetic_per_signal.csv", index=False)

    order = ["VMD"] + [f"UVMD-{N}" for N in depths]
    # ---- table: per true mode, mean +- std over the test signals
    tab = []
    for meth in order:
        s = df[df.method == meth]
        for j in range(J):
            tab.append({"method": meth, "true mode": names[j],
                        "error (% of mode size)": 100 * s[f"nrmse_{j}"].mean(),
                        "error std": 100 * s[f"nrmse_{j}"].std(),
                        "SNR (dB)": s[f"snr_{j}"].mean(), "correlation": s[f"corr_{j}"].mean()})
        tab.append({"method": meth, "true mode": "all modes (mean)",
                    "error (% of mode size)": 100 * np.mean([s[f"nrmse_{j}"].mean() for j in range(J)]),
                    "error std": np.nan, "SNR (dB)": np.mean([s[f"snr_{j}"].mean() for j in range(J)]),
                    "correlation": np.mean([s[f"corr_{j}"].mean() for j in range(J)])})
        tab.append({"method": meth, "true mode": "sum of modes vs clean signal",
                    "error (% of mode size)": 100 * s.sum_vs_clean_rel_err.mean(),
                    "error std": 100 * s.sum_vs_clean_rel_err.std(), "SNR (dB)": np.nan, "correlation": np.nan})
        tab.append({"method": meth, "true mode": f"extra mode energy share (K={K} > {J})",
                    "error (% of mode size)": 100 * s.extra_mode_energy_share.mean(),
                    "error std": 100 * s.extra_mode_energy_share.std(), "SNR (dB)": np.nan, "correlation": np.nan})
    T1 = pd.DataFrame(tab)
    T1.to_csv(out / "synthetic_error_vs_true_modes.csv", index=False)
    (out / "synthetic_error_vs_true_modes.md").write_text(
        f"Synthetic, K={K}, {len(te)} unseen test signals. Error = RMSE between a recovered mode and the true "
        f"mode, as % of the true mode's standard deviation (lower is better). Each estimated mode is matched "
        f"one-to-one to the true mode it correlates with most; the remaining mode is the extra one.\n\n"
        + T1.to_markdown(index=False, floatfmt=".3g") + "\n")

    # ---- example signal: the test signal whose UVMD error is the median one (not hand-picked)
    u2 = df[df.method == f"UVMD-{a.layers}"].assign(m=lambda x: x[[f"nrmse_{j}" for j in range(J)]].mean(axis=1))
    ex = int(u2.iloc[(u2.m - u2.m.median()).abs().argmin()].signal)
    dec = cache[ex]
    side_by_side_synthetic(out / "synthetic_side_by_side.png", D, ex, dec, names, K, a.layers, S["alpha"])
    error_bars_synthetic(out / "synthetic_error_bars.png", df, names, order, K, S["alpha"])
    print(f"example signal (median UVMD error): {ex}")
    print(T1.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("written to", out)


def side_by_side_synthetic(path, D, i, dec, names, K, N, alpha):
    x, clean, truth = D["x"][i], D["clean"][i], D["modes"][i]
    t = np.arange(len(x))
    J = len(truth)
    uv = f"UVMD-{N}"
    cols = ["Original", "VMD", uv]
    matched = {m: match_modes(dec[m][0], truth) for m in ("VMD", uv)}
    used = {m: _matched_idx(dec[m][0], truth) for m in ("VMD", uv)}
    fig, axes = plt.subplots(J + 2, 3, figsize=(15, 1.55 * (J + 2) + 0.9), sharex=True, squeeze=False)
    # row 0: signal
    ax = axes[0, 0]
    ax.plot(t, x, color=FAINT, lw=0.8, label="input (noisy)")
    ax.plot(t, clean, color=TRUTH, lw=1.2, label="sum of true modes")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    for c, m in ((1, "VMD"), (2, uv)):
        u = dec[m][0]
        e = np.linalg.norm(clean - u.sum(0)) / np.linalg.norm(clean)
        ax = axes[0, c]
        ax.plot(t, clean, color=FAINT, lw=1.0)
        ax.plot(t, u.sum(0), color=color_of(m), lw=1.1)
        ax.text(0.99, 0.93, f"error vs sum of true modes {100 * e:.1f} %", transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=INK)
    axes[0, 0].set_title("Original", fontsize=12, color=INK, loc="left", fontweight="bold")
    axes[0, 1].set_title("VMD (classic ADMM)", fontsize=12, color=VMD_C, loc="left", fontweight="bold")
    axes[0, 2].set_title(f"UVMD ({N} layers, trained)", fontsize=12, color=UVMD_C, loc="left", fontweight="bold")
    # rows 1..J: modes
    for j in range(J):
        r = j + 1
        lim = 1.15 * np.abs(truth[j]).max()
        axes[r, 0].plot(t, truth[j], color=TRUTH, lw=1.3)
        axes[r, 0].set_ylabel(names[j], fontsize=10, color=INK)
        for c, m in ((1, "VMD"), (2, uv)):
            ax = axes[r, c]
            ax.plot(t, truth[j], color=FAINT, lw=1.6)
            ax.plot(t, matched[m][j], color=color_of(m), lw=1.1)
            nr = np.sqrt(np.mean((truth[j] - matched[m][j]) ** 2)) / np.std(truth[j])
            ax.text(0.99, 0.93, f"error {100 * nr:.0f} %", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK)
        for c in range(3):
            axes[r, c].set_ylim(-lim, lim)
    # last row: the extra mode
    r = J + 1
    axes[r, 0].axis("off")
    axes[r, 0].text(0.02, 0.5, f"no true mode\n(K = {K} but only {J} true modes)", fontsize=10, color=MUTED,
                    transform=axes[r, 0].transAxes, va="center")
    for c, m in ((1, "VMD"), (2, uv)):
        u = dec[m][0]
        extra = [k for k in range(len(u)) if k not in used[m]]
        ax = axes[r, c]
        for k in extra:
            ax.plot(t, u[k], color=color_of(m), lw=1.0)
        share = sum(np.sum(u[k] ** 2) for k in extra) / np.sum(u ** 2)
        ax.set_ylim(-1.15 * np.abs(truth).max(), 1.15 * np.abs(truth).max())
        ax.text(0.99, 0.93, f"extra mode: {100 * share:.1f} % of the energy", transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=INK)
    axes[r, 1].set_ylabel("extra", fontsize=10, color=INK)
    for rr in range(J + 2):
        for c in range(3):
            if axes[rr, c].axison:
                _style(axes[rr, c], bottom=(rr == J + 1))
    for c in range(3):
        axes[J + 1, c].set_xlabel("sample", color=MUTED)
    fig.text(0.005, 0.005, "Grey line behind each estimate = the true mode. Error = RMSE to the true mode, as % of "
             "its standard deviation. Same vertical scale in every row.", fontsize=8.5, color=MUTED)
    fig.suptitle(f"Synthetic signal, K = {K}, alpha = {alpha:g}: original modes, VMD and UVMD side by side", x=0.005, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _matched_idx(u, truth):
    from scipy.optimize import linear_sum_assignment
    from decomp.metrics import _corr
    C = np.array([[abs(_corr(u[k], tm)) for tm in truth] for k in range(len(u))])
    rows, _ = linear_sum_assignment(-C)
    return set(rows.tolist())


def error_bars_synthetic(path, df, names, order, K, alpha):
    J = len(names)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [3, 1.35]})
    w = 0.8 / len(order)
    for q, m in enumerate(order):
        s = df[df.method == m]
        mean = [100 * s[f"nrmse_{j}"].mean() for j in range(J)]
        a1.bar(np.arange(J) + (q - (len(order) - 1) / 2) * w, mean, w * 0.92, color=color_of(m), label=m)
        for j in range(J):
            a1.text(j + (q - (len(order) - 1) / 2) * w, mean[j] + 1.5, f"{mean[j]:.0f}", ha="center", fontsize=8.5, color=INK)
        mc = 100 * s.sum_vs_clean_rel_err.mean()
        a2.bar(q, mc, 0.7, color=color_of(m))
        a2.text(q, mc + 0.3, f"{mc:.1f}", ha="center", fontsize=9, color=INK)
    a1.set_xticks(range(J), names)
    a1.set_ylabel("error vs true mode (% of mode size)", color=MUTED)
    a1.set_title("Error of each recovered mode (lower is better)", loc="left", fontsize=11, color=INK)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a2.set_xticks(range(len(order)), order)
    a2.set_ylabel("error of summed modes vs clean signal (%)", color=MUTED)
    a2.set_title("Summed modes vs the clean signal", loc="left", fontsize=11, color=INK)
    for ax in (a1, a2):
        _style(ax, bottom=True)
    fig.suptitle(f"Synthetic data, K = {K}, alpha = {alpha:g}, mean over the unseen test signals", x=0.005, ha="left", fontsize=13,
                 color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ real data (no truth): VMD | UVMD
def two_column(path, x, dec, labels, unit_fn, xaxis, xlabel, title, zoom=None, note=""):
    """dec = {'VMD': (u, omega), 'UVMD-N': (u, omega)}; modes ordered by centre frequency."""
    meths = list(dec)
    K = len(dec[meths[0]][0])
    sl = slice(*zoom) if zoom else slice(None)
    fig, axes = plt.subplots(K + 1, len(meths), figsize=(7.2 * len(meths), 1.5 * (K + 1) + 0.9), sharex=True,
                             squeeze=False)
    for c, m in enumerate(meths):
        u, om = dec[m]
        order = np.argsort(om)
        err = np.linalg.norm(x - u.sum(0)) / np.linalg.norm(x)
        ax = axes[0, c]
        ax.plot(xaxis[sl], x[sl], color=FAINT, lw=1.3)
        ax.plot(xaxis[sl], u.sum(0)[sl], color=color_of(m), lw=0.9)
        ax.text(0.99, 0.93, f"input (grey) and sum of modes   |   reconstruction error {100 * err:.1f} %",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
        ax.set_title(labels[m], loc="left", fontsize=12, color=color_of(m), fontweight="bold")
        for r, k in enumerate(order):
            ax = axes[r + 1, c]
            ax.plot(xaxis[sl], u[k][sl], color=color_of(m), lw=0.9)
            ax.text(0.99, 0.93, unit_fn(om[k]), transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color=INK)
            ax.set_ylabel(f"mode {r + 1}", fontsize=9, color=INK)
    for r in range(K + 1):
        lo = min(axes[r, c].get_ylim()[0] for c in range(len(meths)))
        hi = max(axes[r, c].get_ylim()[1] for c in range(len(meths)))
        for c in range(len(meths)):
            axes[r, c].set_ylim(lo, hi)
            _style(axes[r, c], bottom=(r == K))
    for c in range(len(meths)):
        axes[K, c].set_xlabel(xlabel, color=MUTED)
    fig.suptitle(title, x=0.005, ha="left", fontsize=13, color=INK, fontweight="bold")
    if note:
        fig.text(0.005, 0.005, note, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cwru(a):
    from datasets.cwru import FS, load_cwru, split_indices
    d = Path(a.dir)
    S = json.load(open(d / "summary.json"))["args"]
    K = S["K"]
    D = load_cwru(a.data or S["data"], S["T"], S["windows_per_file"], not S.get("no_standardize", False))
    _, _, te = split_indices(D, S["split"])
    depths = find_depths(d)
    DEPTHS[:] = depths
    Ps = {N: load_params(d, N) for N in depths}
    out = Path(a.out or d / "readable")
    out.mkdir(parents=True, exist_ok=True)
    for cls in ("inner", "outer", "ball", "normal"):
        cand = [i for i in te if str(D["cls"][i]) == cls]
        if not cand:
            continue
        i = int(cand[0])
        x = D["x"][i]
        dec = decompose(x, K, S, Ps)
        dec = {m: dec[m] for m in ("VMD", f"UVMD-{a.layers}")}
        labels = {"VMD": "VMD (classic ADMM)", f"UVMD-{a.layers}": f"UVMD ({a.layers} layers, trained)"}
        two_column(out / f"cwru_{cls}_vmd_vs_uvmd.png", x, dec, labels,
                   lambda w: f"centre frequency {w * FS:,.0f} Hz", np.arange(len(x)) / FS * 1e3,
                   "time (ms)", f"CWRU bearing vibration ({cls}), K = {K}: VMD and UVMD modes side by side",
                   note="One test window (2048 samples at 12 kHz), standardised. Modes are sorted by centre "
                        "frequency; the vertical scale is the same in both columns.")
    print("written to", out)


def traffic(a):
    from datasets.traffic import STEPS_PER_DAY, load_traffic, split_nodes
    d = Path(a.dir)
    S = json.load(open(d / "summary.json"))["args"]
    K = S["K"]
    D = load_traffic(a.data or S["data"], S.get("length"))
    _, _, te = split_nodes(D["x"].shape[0])
    if S.get("max_test_nodes"):
        te = te[: S["max_test_nodes"]]
    depths = find_depths(d)
    DEPTHS[:] = depths
    Ps = {N: load_params(d, N) for N in depths}
    out = Path(a.out or d / "readable")
    out.mkdir(parents=True, exist_ok=True)
    i = int(te[0])
    x = D["x"][i].astype(np.float64)
    dec = decompose(x, K, S, Ps)
    dec = {m: dec[m] for m in ("VMD", f"UVMD-{a.layers}")}
    labels = {"VMD": "VMD (classic ADMM)", f"UVMD-{a.layers}": f"UVMD ({a.layers} layers, trained)"}

    def per(w):
        if w <= 0:
            return "centre frequency 0 (trend / daily average level)"
        hours = 0.25 / w
        return f"typical period {hours / 24:.1f} days" if hours >= 48 else f"typical period {hours:.1f} hours"

    days = np.arange(len(x)) / STEPS_PER_DAY
    two_column(out / "traffic_vmd_vs_uvmd.png", x, dec, labels, per, days, "day of the year (first 3 weeks shown)",
               f"LargeST traffic flow (one sensor, whole year), K = {K}: VMD and UVMD modes side by side",
               zoom=(0, 21 * STEPS_PER_DAY),
               note="One test sensor; the modes were computed on the whole year and the first 21 days are drawn. "
                    "Modes are sorted by centre frequency; same vertical scale in both columns.")
    print("written to", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["synthetic", "cwru", "traffic"])
    p.add_argument("--dir", required=True, help="results folder of the benchmark run (summary.json, uvmd_params_N*.npz)")
    p.add_argument("--data", default=None, help="data path (cwru folder / traffic h5); default: the one in summary.json")
    p.add_argument("--layers", type=int, default=3, help="UVMD depth drawn in the big figure (must be a trained depth)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    {"synthetic": synthetic, "cwru": cwru, "traffic": traffic}[a.dataset](a)


if __name__ == "__main__":
    main()
