"""CWRU loader quirks and fault metrics, on small fake .mat files (no download needed)."""
import numpy as np
from scipy.io import savemat

from datasets.cwru import FS, fault_freqs, load_cwru, read_file, split_indices
from decomp.fault_metrics import fault_eval, raw_eval


def _impacts(f_fault, n, fs=FS, f_res=3000.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = 0.2 * rng.standard_normal(n)
    for t0 in np.arange(0, t[-1], 1 / f_fault):
        m = t >= t0
        x[m] += np.exp(-800 * (t[m] - t0)) * np.sin(2 * np.pi * f_res * (t[m] - t0))
    return x


def test_loader_quirks(tmp_path):
    n = 12_000
    savemat(tmp_path / "99.mat", {"X098_DE_time": np.zeros((4 * n, 1)), "X099_DE_time": np.ones((4 * n, 1))})
    savemat(tmp_path / "98.mat", {"X098_DE_time": np.ones((4 * n, 1))})               # no RPM variable
    savemat(tmp_path / "105.mat", {"X105_DE_time": np.ones((n, 1)), "X105RPM": np.array([[1797]])})
    x, rpm, meta = read_file(tmp_path / "99.mat")
    assert len(x) == n and np.allclose(x[100:-100], 1.0)          # X099 chosen, 48k -> 12k
    assert read_file(tmp_path / "98.mat")[1] == 1772.0              # nominal rpm for 1 HP
    D = load_cwru(tmp_path, T=2048, windows_per_file=3, standardize=False)
    assert D["x"].shape == (9, 2048) and set(D["cls"]) == {"normal", "inner"}


def test_split_has_no_shared_recordings(tmp_path):
    for f in (97, 98, 99, 105, 106, 107, 108):
        savemat(tmp_path / f"{f}.mat", {f"X{f:03d}_DE_time": np.random.randn(48_000 if f < 100 else 12_000, 1)})
    D = load_cwru(tmp_path, T=1024, windows_per_file=5)
    tr, va, te = split_indices(D, "load")
    assert not set(D["file"][tr]) & set(D["file"][te])
    assert set(D["load"][te]) == {3}


def test_envelope_metric_finds_fault():
    ff = fault_freqs(1797)
    x = _impacts(ff["inner"], 4096)
    r = raw_eval(x, "inner", ff, FS)
    assert r["envelope_hit"] == 1.0 and r["fault_snr_db"] > 10
    u = np.stack([0.01 * np.random.randn(4096), x])
    m = fault_eval(u, x, "inner", ff, FS)
    assert m["sel_mode"] == 1 and m["envelope_hit"] == 1.0
