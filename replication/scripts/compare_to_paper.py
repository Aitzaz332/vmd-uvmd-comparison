"""Compare replicated MAGN results with the paper (arXiv:2509.00703, Table I and Table II).

    python replication/scripts/compare_to_paper.py --results replication/results --regions gba gla sd --layers 1

Reads replication/results/<region>_N<layers>/record_s<seed>.log (the upstream logger's output). If several seeds
exist they are averaged. The upstream logs MAPE as a fraction; it is multiplied by 100 here to match the paper's %.
Paper numbers were typed in from Table I (MAGN rows and the plain ASTGCN rows) and Table II (SD, case I, K=13).
"""
import argparse
import re
import statistics
from pathlib import Path

H = ["H3", "H6", "H12", "Avg"]
M = ["MAE", "RMSE", "MAPE"]

# paper[region][model][horizon] = (MAE, RMSE, MAPE %)
PAPER = {
    "gba": {
        "MAGN": {"H3": (0.62, 0.93, 0.68), "H6": (0.68, 1.09, 0.74), "H12": (2.00, 4.71, 1.90), "Avg": (0.86, 1.63, 0.91)},
        "ASTGCN": {"H3": (21.40, 33.61, 17.65), "H6": (26.70, 40.75, 24.02), "H12": (33.64, 51.21, 31.15), "Avg": (26.15, 40.25, 23.29)},
    },
    "gla": {
        "MAGN": {"H3": (0.67, 3.16, 0.62), "H6": (0.74, 4.30, 0.67), "H12": (2.73, 17.71, 1.89), "Avg": (1.11, 6.53, 0.92)},
        "ASTGCN": {"H3": (21.11, 32.41, 11.82), "H6": (27.80, 44.67, 17.79), "H12": (39.39, 59.31, 28.03), "Avg": (28.12, 44.40, 18.62)},
    },
    "sd": {
        "MAGN": {"H3": (0.84, 1.88, 0.84), "H6": (0.90, 2.23, 0.92), "H12": (3.60, 8.49, 2.97), "Avg": (1.38, 3.30, 1.28)},
        "ASTGCN": {"H3": (19.68, 31.53, 12.20), "H6": (24.45, 38.89, 15.36), "H12": (31.52, 49.77, 22.15), "Avg": (26.07, 38.42, 15.63)},
    },
}

HOR = re.compile(r"Horizon\s+(\d+), Test MAE: ([\d.eE+-]+), Test RMSE: ([\d.eE+-]+), Test MAPE: ([\d.eE+-]+)")
AVG = re.compile(r"Average Test MAE: ([\d.eE+-]+), Test RMSE: ([\d.eE+-]+), Test MAPE: ([\d.eE+-]+)")


def parse(log: Path):
    """Last complete test block in the log -> {'H3': (mae, rmse, mape%), ...}"""
    txt = log.read_text()
    hor = {}
    for m in HOR.finditer(txt):               # later blocks overwrite earlier ones (train-end test, then --mode test)
        hor[int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
    avg = list(AVG.finditer(txt))
    if not all(h in hor for h in (3, 6, 12)) or not avg:
        raise ValueError(f"{log}: no complete test block")
    a = avg[-1]
    out = {f"H{h}": (hor[h][0], hor[h][1], hor[h][2] * 100) for h in (3, 6, 12)}
    out["Avg"] = (float(a.group(1)), float(a.group(2)), float(a.group(3)) * 100)
    return out


def fmt(x):
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="replication/results")
    ap.add_argument("--regions", nargs="+", default=["gba", "gla", "sd"])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.15, help="relative difference below which a metric counts as 'close'")
    a = ap.parse_args()

    for reg in a.regions:
        d = Path(a.results) / f"{reg}_N{a.layers}"
        logs = sorted(d.glob("record_s*.log"))
        print(f"\n### {reg.upper()}  (UVMD K=13, N={a.layers}; {len(logs)} seed(s))\n")
        if not logs:
            print("no logs yet\n"); continue
        runs = [parse(l) for l in logs]
        mine = {h: tuple(statistics.mean(r[h][i] for r in runs) for i in range(3)) for h in H}
        print("| horizon | metric | paper MAGN | replicated | rel. diff | paper ASTGCN (no UVMD) |")
        print("|---|---|---|---|---|---|")
        close = total = 0
        for h in H:
            for i, name in enumerate(M):
                p, r, b = PAPER[reg]["MAGN"][h][i], mine[h][i], PAPER[reg]["ASTGCN"][h][i]
                rel = (r - p) / p
                total += 1; close += abs(rel) <= a.tol
                unit = "%" if name == "MAPE" else ""
                print(f"| {h} | {name} | {fmt(p)}{unit} | {fmt(r)}{unit} | {rel:+.0%} | {fmt(b)}{unit} |")
        print(f"\n{close}/{total} metrics within ±{a.tol:.0%} of the paper's MAGN numbers.")
        if len(runs) > 1:
            sd = statistics.pstdev([r["Avg"][0] for r in runs])
            print(f"Seed spread of average MAE: {sd:.3f} over {len(runs)} seeds.")


if __name__ == "__main__":
    main()
