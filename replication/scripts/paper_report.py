"""Build replication/results/PAPER_STYLE_REPORT.md: Table I and Table II in the paper's layout (paper row next to the
replicated row, parameter counts as 'x+y'), the data behind Fig. 2/3, and the decomposition timing.

    python replication/scripts/paper_report.py --results replication/results

Works on whatever result folders exist (<region>_N<N> for K=13, <region>_K<K>_N<N> otherwise) and marks the rest 'not run'.
Paper numbers are typed in from the PDF (arXiv:2509.00703v1, Tables I and II). Baselines other than ASTGCN are not rerun.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from compare_to_paper import PAPER, parse

HS = ["H3", "H6", "H12", "Avg"]
PARAM_PAPER = {"gba": "140.174K+22.38M", "gla": "140.173K+59.2M", "sd": "140.173K+2.17M"}

# Table II (SD). value = (param string, {H: (MAE, RMSE, MAPE%)})
def _row(p, a, b, c, d):
    return (p, dict(zip(HS, (a, b, c, d))))

T2 = {
    "case I   shared alpha (not in upstream code)": ("sd", None, None, _row("140.161K+2.17M", (1.61, 3.84, 1.53), (14.69, 23.54, 9.46), (28.28, 45.20, 19.75), (13.70, 22.70, 9.34))),
    "case II  1/2 signal 17520 (not in upstream code)": ("sd", None, None, _row("70.093K+2.17M", (0.88, 2.08, 0.96), (1.05, 2.85, 1.17), (5.05, 13.27, 4.79), (1.80, 4.90, 1.83))),
    "case II  1/4 signal 8760 (not in upstream code)": ("sd", None, None, _row("35.053K+2.17M", (0.87, 2.09, 0.98), (1.06, 2.91, 1.22), (6.37, 17.06, 5.56), (2.03, 5.54, 2.01))),
    "case II  1/8 signal 4380 (not in upstream code)": ("sd", None, None, _row("17.533K+2.17M", (0.94, 2.92, 1.01), (1.18, 4.48, 1.28), (7.01, 28.89, 5.78), (2.26, 9.38, 2.09))),
    "case III K=3,  N=1": ("sd", 3, 1, _row("140.163K+2.16M", (7.65, 14.51, 5.42), (11.80, 20.35, 8.25), (25.24, 69.03, 23.41), (13.56, 29.76, 11.04))),
    "case III K=6,  N=1": ("sd", 6, 1, _row("140.166K+2.16M", (0.97, 2.48, 0.94), (2.33, 7.69, 1.86), (10.08, 16.18, 7.82), (4.00, 8.72, 3.11))),
    "case III K=9,  N=1": ("sd", 9, 1, _row("140.169K+2.16M", (0.86, 1.94, 0.86), (0.94, 2.48, 0.93), (5.93, 12.74, 4.17), (1.89, 4.52, 1.59))),
    "case III K=13, N=1 (= Table I SD)": ("sd", 13, 1, _row("140.173K+2.17M", (0.84, 1.88, 0.84), (0.90, 2.23, 0.92), (3.60, 8.49, 2.97), (1.38, 3.30, 1.28))),
    "case III K=15, N=1": ("sd", 15, 1, _row("140.175K+2.17M", (0.85, 1.72, 0.92), (0.94, 1.80, 1.01), (2.47, 6.34, 2.18), (1.16, 2.60, 1.19))),
    "case III K=6,  N=2": ("sd", 6, 2, _row("210.246K+2.16M", (5.35, 8.84, 5.22), (9.71, 15.15, 8.93), (21.14, 32.36, 22.01), (11.19, 17.57, 10.92))),
    "case III K=13, N=2": ("sd", 13, 2, _row("210.253K+2.17M", (7.42, 12.47, 9.53), (13.27, 18.57, 14.87), (23.62, 33.30, 29.22), (13.49, 19.82, 16.65))),
}

HEADER = ("| row | Param | H3 MAE | RMSE | MAPE | H6 MAE | RMSE | MAPE | H12 MAE | RMSE | MAPE | Avg MAE | RMSE | MAPE |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def cells(v):
    return " | ".join(f"{m:.2f} | {r:.2f} | {p:.2f}%" for m, r, p in (v[h] for h in HS))


def tag(reg, k, n):
    return f"{reg}_N{n}" if k in (None, 13) else f"{reg}_K{k}_N{n}"


def uvmd_params(d: Path):
    f = d / "uvmd_stdout.log"
    if f.exists():
        m = re.search(r"model\.params \(M\):\s*([\d.eE+-]+)", f.read_text())
        if m:
            return f"{float(m.group(1)) * 1000:.3f}K"
    return "?"


def astgcn_params(d: Path):
    for f in sorted(d.glob("record_s*.log")):
        m = re.search(r"The number of parameters: (\d+)", f.read_text())
        if m:
            return f"{int(m.group(1)) / 1e6:.2f}M"
    return "?"


def load_runs(d: Path):
    logs = sorted(d.glob("record_s*.log"))
    runs = []
    for l in logs:
        try:
            runs.append(parse(l))
        except ValueError:
            pass
    if not runs:
        return None
    return {h: tuple(statistics.mean(r[h][i] for r in runs) for i in range(3)) for h in HS}, len(runs)


def table1(res: Path):
    out = ["## Table I: MAGN (UVMD K=13, N=1 + ASTGCN), paper vs replicated", "",
           "Units as in the paper: MAPE in %. ASTGCN* rows are the paper's numbers for the plain ASTGCN (not rerun).", ""]
    for reg in ("gba", "gla", "sd"):
        out += [f"### {reg.upper()}", "", HEADER]
        out.append(f"| ASTGCN* (paper, not rerun) | - | {cells(PAPER[reg]['ASTGCN'])} |")
        out.append(f"| MAGN (paper) | {PARAM_PAPER[reg]} | {cells(PAPER[reg]['MAGN'])} |")
        d = res / tag(reg, 13, 1)
        got = load_runs(d) if d.exists() else None
        if got:
            mine, ns = got
            out.append(f"| **MAGN (replicated, {ns} seed{'s' if ns > 1 else ''})** | {uvmd_params(d)}+{astgcn_params(d)} | {cells(mine)} |")
            rel = " | ".join(f"{(mine[h][i] - PAPER[reg]['MAGN'][h][i]) / PAPER[reg]['MAGN'][h][i]:+.0%}" for h in HS for i in range(3))
            out.append(f"| relative difference | | {rel} |")
        else:
            out.append("| MAGN (replicated) | not run yet | |")
        out.append("")
    return out


def table2(res: Path):
    out = ["## Table II: SD ablations, paper vs replicated", "", HEADER]
    for name, (reg, k, n, (pstr, pv)) in T2.items():
        out.append(f"| {name} (paper) | {pstr} | {cells(pv)} |")
        d = res / tag(reg, k, n) if k else None
        got = load_runs(d) if d is not None and d.exists() else None
        if got:
            out.append(f"| **{name} (replicated)** | {uvmd_params(d)}+{astgcn_params(d)} | {cells(got[0])} |")
        else:
            out.append(f"| {name} (replicated) | {'not supported by upstream code' if k is None else 'not run yet'} | |")
    out.append("")
    return out


def timing(res: Path):
    out = ["## Decomposition time", "",
           "Paper: 98.63 s for all 6902 sensors (UVMD, RTX 3080 Ti) against 267 min for iterative VMD (about 162x; the abstract says 250x).", ""]
    rows = []
    for d in sorted(res.glob("*_N*")):
        wall = None
        f = d / "uvmd_stdout.log"
        if f.exists():
            m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", f.read_text())
            wall = m.group(1) if m else None
        t = d / "decomp_timing.json"
        tj = json.loads(t.read_text()) if t.exists() else None
        if wall or tj:
            rows.append((d.name, wall or "-", f"{tj['seconds_per_sensor']:.4f}" if tj else "-",
                         f"{tj['seconds_extrapolated_6902_sensors']:.1f}" if tj else "-"))
    if rows:
        out += ["| run | uvmd.py wall time incl. training (h:mm:ss) | decomposition s/sensor | extrapolated s for 6902 sensors |", "|---|---|---|---|"]
        out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    else:
        out.append("No timings yet (`decomp_timing.py` writes `decomp_timing.json`).")
    out.append("")
    return out


def figures(res: Path):
    out = ["## Figures (data behind paper Fig. 2 and Fig. 3)", ""]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return out + ["matplotlib not installed, no figures.", ""]
    fig_dir = res / "figures"; fig_dir.mkdir(exist_ok=True)
    runs = sorted(d for d in res.glob("*_N*") if d.is_dir())
    # UVMD reconstruction loss per epoch
    fig, ax = plt.subplots(figsize=(5, 3.5)); n = 0
    for d in runs:
        f = d / "training.log"
        if f.exists():
            ls = [float(x) for x in re.findall(r"Train Loss: ([\d.eE+-]+)", f.read_text())]
            if ls:
                ax.plot(ls, label=d.name); n += 1
    if n:
        ax.set_xlabel("epoch"); ax.set_ylabel("UVMD reconstruction loss (train)"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(fig_dir / "uvmd_reconstruction_loss.png", dpi=150); out.append("![UVMD loss](figures/uvmd_reconstruction_loss.png)")
    plt.close(fig)
    # ASTGCN validation MAE per epoch
    fig, ax = plt.subplots(figsize=(5, 3.5)); n = 0
    for d in runs:
        for f in sorted(d.glob("record_s*.log"))[:1]:
            ls = [float(x) for x in re.findall(r"Valid Loss: ([\d.eE+-]+)", f.read_text())]
            if ls:
                ax.plot(ls, label=d.name); n += 1
    if n:
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("ASTGCN validation MAE"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(fig_dir / "astgcn_validation_mae.png", dpi=150); out.append("![ASTGCN val MAE](figures/astgcn_validation_mae.png)")
    plt.close(fig)
    # learned bandwidth per mode (needs torch)
    try:
        import torch
        import torch.nn.functional as F
        fig, ax = plt.subplots(figsize=(5, 3.5)); n = 0
        for d in runs:
            f = d / "model.pth"
            if f.exists():
                a = F.softplus(torch.load(f, map_location="cpu")["Alpha"]).numpy()
                ax.plot(range(len(a)), a, marker="o", ms=3, label=d.name); n += 1
        if n:
            ax.set_xlabel("mode k"); ax.set_ylabel("softplus(alpha_k)"); ax.legend(fontsize=7); fig.tight_layout()
            fig.savefig(fig_dir / "learned_bandwidth.png", dpi=150); out.append("![alpha_k](figures/learned_bandwidth.png)")
        plt.close(fig)
    except Exception as e:
        out.append(f"(learned alpha_k plot skipped: {e})")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="replication/results")
    a = ap.parse_args()
    res = Path(a.results)
    lines = ["# MAGN replication, paper-style report", "",
             "Generated by `replication/scripts/paper_report.py` from the logs in this folder. Paper numbers: arXiv:2509.00703v1.", ""]
    for part in (table1, table2, timing, figures):
        lines += part(res)
    (res / "PAPER_STYLE_REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
