"""Context for the plain-ASTGCN number: errors of trivial forecasters on the SAME test windows and in the same units.
    python replication/scripts/naive_baselines.py --his replication/magn/data/sd/2019/his_raw.npz --idx replication/magn/data/sd/2019/idx_test.npy
last value : y[t+h] = x[t]
tod mean   : y[t+h] = mean of the TRAIN-period flow at the same time of day and day of week (needs idx_train)
Metrics: MAE/RMSE/MAPE at horizons 3, 6, 12 and average, labels == 0 masked as in upstream. CPU only, seconds."""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--his", required=True); ap.add_argument("--idx", required=True)
ap.add_argument("--idx-train", default=None)
a = ap.parse_args()
z = np.load(a.his); d = z["data"][..., 0].astype(np.float64); mean, std = float(z["mean"]), float(z["std"])
x = d * std + mean                                    # original units (vehicles per 15 min)
idx = np.load(a.idx); T = x.shape[0]
tr = np.load(a.idx_train) if a.idx_train else None

def report(name, pred_fn):
    err = {h: [] for h in range(1, 13)}
    for h in range(1, 13):
        p = pred_fn(idx, h); y = x[idx + h]
        m = y != 0
        e = (p - y)[m]
        err[h] = (np.abs(e).mean(), np.sqrt((e ** 2).mean()), (np.abs(e) / y[m]).mean() * 100)
    avg = tuple(np.mean([err[h][i] for h in range(1, 13)]) for i in range(3))
    row = lambda v: f"{v[0]:6.2f} {v[1]:6.2f} {v[2]:6.2f}%"
    print(f"{name:10s} H3 {row(err[3])} | H6 {row(err[6])} | H12 {row(err[12])} | Avg {row(avg)}")

report("last value", lambda ix, h: x[ix])
if tr is not None:
    slot = np.arange(T) % 672                          # 96 x 7 slots of a week; slot = position within the week, the weekday offset is the same for train and test
    prof = np.zeros((672, x.shape[1]))
    trn = np.unique(np.concatenate([tr + h for h in range(1, 13)])); trn = trn[trn < T]
    for s in range(672):
        r = trn[slot[trn] == s]
        prof[s] = x[r].mean(0) if len(r) else x[trn].mean(0)
    report("tod/dow mean", lambda ix, h: prof[slot[ix + h]])
