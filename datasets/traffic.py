"""Dataset 3: LargeST traffic flow (GBA / GLA / SD, year 2019, 15-minute), as used in the UVMD paper
(arXiv:2509.00703, code https://github.com/OsamaAhmad369/MAGN).

How the authors feed UVMD (their ``uvmd.py``): ``his.npz`` holds ``data`` of shape (T, nodes, 3) =
[flow (globally standardised), time-of-day, day-of-week]; UVMD takes ``data[:, :, 0].T``, i.e. ONE signal per
sensor node covering the whole year (T = 365 * 96 = 35040 samples), splits the NODES 70/15/15 with
``train_test_split(random_state=42)``, trains on nodes (batch size 1) and decomposes every node.
This module reproduces that protocol.

Raw data (LargeST, Liu et al., NeurIPS 2023): https://www.kaggle.com/datasets/liuxu77/largest
  ca_his_raw_2019.h5   5-minute flow, DataFrame (time x 8600 sensors, columns = sensor ID strings)
  ca_meta.csv          sensor metadata (columns include ID, District, Lat, Lng)
Sub-datasets (district filters taken from the MAGN / LargeST notebooks):
  GBA = district 4 (2352 sensors), GLA = districts 7, 8, 12 (3834), SD = district 11 (716)

Accepted inputs of ``load_traffic``:
  * ``<region>_his_<year>.h5``  15-minute DataFrame (T x N), as written by ``prepare_region`` or by the
    authors' notebooks. Standardised here exactly like the authors' ``generate_data_for_training.py``.
  * ``his.npz`` written by the authors' ``generate_data_for_training.py`` (flow already standardised).
  * ``demo``: a generated stand-in in the same format, for installation checks only (not real traffic).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REGION_DISTRICTS = {"gba": [4], "gla": [7, 8, 12], "sd": [11]}
STEPS_PER_DAY = 96          # 15-minute sampling
STEPS_PER_WEEK = 7 * STEPS_PER_DAY


# ------------------------------------------------------------------------------------------------
# preprocessing (authors' notebooks + generate_data_for_training.py, rewritten for current pandas)
# ------------------------------------------------------------------------------------------------
def prepare_region(raw_h5, meta_csv, region, out_h5, year="2019", resample="15min"):
    """Raw 5-minute California file -> region file at 15 minutes (the authors' notebooks):
    keep the sensors of the region's districts, average 5-min values into 15-min bins, round to integers,
    fill missing values with 0. Writes ``out_h5`` (key 't') and returns the DataFrame."""
    region = region.lower()
    meta = pd.read_csv(meta_csv)
    ids = meta.loc[meta.District.isin(REGION_DISTRICTS[region]), "ID"].astype(str).tolist()
    if not ids:
        raise ValueError(f"no sensors of districts {REGION_DISTRICTS[region]} in {meta_csv}")
    try:  # table-format files allow reading only the needed columns (saves RAM)
        df = pd.read_hdf(raw_h5, columns=ids)
    except Exception:
        df = pd.read_hdf(raw_h5)
        missing = [i for i in ids if i not in df.columns]
        if missing:
            raise KeyError(f"{len(missing)} sensor IDs of {region} not found in {raw_h5}")
        df = df[ids]
    df = df.resample(resample).mean().round(0).fillna(0)
    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(out_h5, key="t", mode="w")
    return df


def standardize_like_authors(values, seq_x=12, seq_y=12, train_frac=0.6, val_frac=0.2):
    """Global standardisation of ``generate_data_for_training.py``: mean/std over ALL sensors of the first
    (60% of the usable time indices) part of the series; the same mean/std is applied to the whole array.
    values: (T, N). Returns (standardised (T, N) float32, mean, std)."""
    T = values.shape[0]
    idx = np.arange(abs(-(seq_x - 1)), abs(T - seq_y))          # authors: min_t = 11, max_t = T - 12
    n = len(idx)
    num_train, num_val = round(n * train_frac), round(n * val_frac)
    idx_val = idx[num_train: num_train + num_val]
    x_train = values[: idx_val[0] - seq_x]
    mean, std = float(x_train.mean()), float(x_train.std())
    return ((values - mean) / std).astype(np.float32), mean, std


# ------------------------------------------------------------------------------------------------
# demo stand-in (same format as LargeST at 15 minutes; NOT real traffic)
# ------------------------------------------------------------------------------------------------
def make_demo_traffic(n_nodes=60, days=28, seed=0, start="2019-01-07"):
    """Daily double-peak (morning/evening) x weekday/weekend profile with node-specific scale, peak timing
    and noise, plus a few missing stretches stored as 0 like in LargeST. For smoke tests only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * STEPS_PER_DAY, freq="15min")
    hod = np.asarray(idx.hour + idx.minute / 60)[:, None]
    weekend = np.asarray(idx.dayofweek >= 5)[:, None]
    scale = rng.uniform(50, 400, n_nodes)[None]
    am, pm = rng.uniform(7, 9, n_nodes)[None], rng.uniform(16.5, 18.5, n_nodes)[None]
    wd = np.exp(-0.5 * ((hod - am) / 1.3) ** 2) + 1.2 * np.exp(-0.5 * ((hod - pm) / 1.8) ** 2)
    we = 0.7 * np.exp(-0.5 * ((hod - 13) / 3.5) ** 2)
    base = 0.15 + np.where(weekend, we, wd)
    flow = scale * base * (1 + 0.15 * rng.standard_normal((len(idx), n_nodes)))
    flow = np.round(np.clip(flow, 0, None))
    for j in rng.choice(n_nodes, max(1, n_nodes // 10), replace=False):   # missing stretches -> 0
        a = int(rng.integers(0, len(idx) - 50))
        flow[a: a + int(rng.integers(8, 48)), j] = 0
    return pd.DataFrame(flow, index=idx, columns=[str(100000 + i) for i in range(n_nodes)])


# ------------------------------------------------------------------------------------------------
# loading + node split
# ------------------------------------------------------------------------------------------------
def load_traffic(path, length=None, demo_kwargs=None):
    """Return dict: x (N, T) float32 standardised flow, one row per sensor; mean, std; node_ids; meta.
    ``length``: keep only the first ``length`` samples (rounded down to even). Default: whole series, which
    is the paper's protocol (35040 samples for one year)."""
    if str(path) == "demo":
        df = make_demo_traffic(**(demo_kwargs or {}))
        vals, mean, std = standardize_like_authors(df.values.astype(np.float64))
        ids, src = df.columns.astype(str).tolist(), "demo (generated stand-in)"
    else:
        path = Path(path)
        if path.suffix == ".npz":
            z = np.load(path)
            vals, mean, std = z["data"][:, :, 0].astype(np.float32), float(z["mean"]), float(z["std"])
            ids, src = [str(i) for i in range(vals.shape[1])], f"authors' his.npz ({path.name})"
        else:
            df = pd.read_hdf(path)
            vals, mean, std = standardize_like_authors(df.values.astype(np.float64))
            ids, src = df.columns.astype(str).tolist(), path.name
    T = vals.shape[0]
    if length:
        T = min(T, length)
    T -= T % 2                                  # upstream drops the last sample of odd-length signals
    X = np.ascontiguousarray(vals[:T].T)
    return {"x": X, "mean": mean, "std": std, "node_ids": ids,
            "meta": {"name": "traffic", "source": src, "n_nodes": X.shape[0], "T": T,
                     "steps_per_day": STEPS_PER_DAY, "days": T / STEPS_PER_DAY}}


def split_nodes(n_nodes, seed=42):
    """Authors' split of the NODES: 70% train / 15% val / 15% test, random_state=42."""
    idx = np.arange(n_nodes)
    tr, rest = train_test_split(idx, test_size=0.3, random_state=seed, shuffle=True)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, shuffle=True)
    return tr, va, te
