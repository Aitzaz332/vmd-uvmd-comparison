"""Dataset 2: Case Western Reserve University (CWRU) bearing data, drive-end accelerometer.

Files: https://engineering.case.edu/bearingdatacenter/download-data-file  (``scripts/download_cwru.sh``)
Default subset (40 files): normal baseline 97-100 + 12 kHz drive-end inner race / ball / outer race @6:00,
fault sizes 0.007", 0.014", 0.021", motor loads 0-3 HP.

Handled quirks of the raw files
  * normal baseline is sampled at 48 kHz -> resampled to 12 kHz (polyphase, anti-aliased), otherwise the
    sampling rate itself would separate "normal" from the faults;
  * 99.mat also contains the X098_* arrays -> the key matching the file number is used;
  * 98.mat has no RPM variable -> nominal speed of that load (CWRU table) is used.

Bearing: SKF 6205-2RS JEM (drive end). Fault frequencies as multiples of shaft speed (CWRU page):
  BPFI 5.4152, BPFO 3.5848, BSF 2.3570 (ball spin), FTF 0.39828.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

FS = 12_000
FAULT_MULT = {"inner": 5.4152, "outer": 3.5848, "ball": 2.3570}
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# file number -> (class, fault size in inches, load HP, native sampling rate)
CATALOG: dict[int, tuple[str, float, int, int]] = {}
for load, f in enumerate([97, 98, 99, 100]):
    CATALOG[f] = ("normal", 0.0, load, 48_000)
for size, (ir, b, orr) in {0.007: (105, 118, 130), 0.014: (169, 185, 197), 0.021: (209, 222, 234)}.items():
    for load in range(4):
        CATALOG[ir + load] = ("inner", size, load, FS)
        CATALOG[b + load] = ("ball", size, load, FS)
        CATALOG[orr + load] = ("outer", size, load, FS)


def read_file(path: Path):
    """Return (drive-end signal at 12 kHz, rpm, meta dict) for one CWRU .mat file."""
    num = int(re.match(r"(\d+)", path.stem).group(1))
    if num not in CATALOG:
        raise KeyError(f"{path.name}: not in the default CWRU subset")
    cls, size, load, fs = CATALOG[num]
    d = loadmat(path)
    key = f"X{num:03d}_DE_time"
    if key not in d:
        cands = [k for k in d if k.endswith("_DE_time")]
        if len(cands) != 1:
            raise KeyError(f"{path.name}: cannot find drive-end signal (keys {cands})")
        key = cands[0]
    x = np.asarray(d[key], dtype=np.float64).ravel()
    rpm_key = f"X{num:03d}RPM"
    rpm = float(np.ravel(d[rpm_key])[0]) if rpm_key in d else float(NOMINAL_RPM[load])
    if fs != FS:
        x = resample_poly(x, FS, fs)
    meta = {"file": num, "cls": cls, "size": size, "load": load, "rpm": rpm, "native_fs": fs}
    return x, rpm, meta


def fault_freqs(rpm: float) -> dict:
    fr = rpm / 60.0
    return {"shaft": fr, **{k: m * fr for k, m in FAULT_MULT.items()}}


def load_cwru(data_dir, T=2048, windows_per_file=30, standardize=True, files=None):
    """Cut every recording into non-overlapping windows of T samples (the first ``windows_per_file``).

    Returns dict with x (n,T), raw_std (n,), and per-window metadata arrays: file, cls, size, load, rpm.
    Windows are standardised (zero mean, unit std) when ``standardize``, as the upstream UVMD script works
    on standardised data; raw_std keeps the original scale.
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.mat"), key=lambda p: int(re.match(r"(\d+)", p.stem).group(1)))
    paths = [p for p in paths if int(re.match(r"(\d+)", p.stem).group(1)) in CATALOG]
    if files is not None:
        paths = [p for p in paths if int(p.stem) in set(files)]
    if not paths:
        raise FileNotFoundError(f"no CWRU .mat files of the default subset found in {data_dir}")
    missing = sorted(set(CATALOG) - {int(p.stem) for p in paths}) if files is None else []
    X, M = [], {k: [] for k in ("file", "cls", "size", "load", "rpm", "start")}
    for p in paths:
        x, rpm, meta = read_file(p)
        n = min(len(x) // T, windows_per_file if windows_per_file else 10 ** 9)
        for w in range(n):
            seg = x[w * T:(w + 1) * T]
            X.append(seg)
            for k in ("file", "cls", "size", "load", "rpm"):
                M[k].append(meta[k])
            M["start"].append(w * T)
    X = np.stack(X)
    raw_std = X.std(axis=1)
    if standardize:
        X = (X - X.mean(axis=1, keepdims=True)) / raw_std[:, None]
    out = {"x": X, "raw_std": raw_std, **{k: np.asarray(v) for k, v in M.items()},
           "meta": {"name": "cwru", "fs": FS, "T": T, "windows_per_file": windows_per_file,
                    "standardize": standardize, "n_files": len(paths), "missing_files": missing}}
    return out


def split_indices(D, mode="load", seed=42):
    """'load': train on 0-1 HP, validate on 2 HP, test on 3 HP (no recording shared between sets).
    'time': every recording split in time, first 70% / next 15% / last 15% of its windows."""
    n = len(D["x"])
    if mode == "load":
        tr = np.where(np.isin(D["load"], [0, 1]))[0]
        va = np.where(D["load"] == 2)[0]
        te = np.where(D["load"] == 3)[0]
    elif mode == "time":
        tr, va, te = [], [], []
        for f in np.unique(D["file"]):
            idx = np.where(D["file"] == f)[0]
            a, b = int(0.7 * len(idx)), int(0.85 * len(idx))
            tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
        tr, va, te = map(np.asarray, (tr, va, te))
    else:
        raise ValueError(mode)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(va) + len(te) == n
    return tr, va, te
