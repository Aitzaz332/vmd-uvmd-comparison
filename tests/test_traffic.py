"""Traffic (LargeST) pipeline on small stand-in files that have the exact LargeST format."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tables")

from datasets.traffic import (STEPS_PER_DAY, REGION_DISTRICTS, load_traffic, make_demo_traffic,  # noqa: E402
                              prepare_region, split_nodes, standardize_like_authors)
from decomp.traffic_metrics import spectral_metrics  # noqa: E402


def _fake_raw(tmp_path, days=14):
    """5-minute 'California' file + metadata with sensors in several districts (like ca_his_raw_2019.h5)."""
    idx = pd.date_range("2019-01-01", periods=days * 288, freq="5min")
    rng = np.random.default_rng(0)
    ids = [str(1000 + i) for i in range(12)]
    raw = pd.DataFrame(rng.integers(0, 300, (len(idx), 12)).astype(float), index=idx, columns=ids)
    raw.iloc[10:20, 0] = np.nan                                  # missing values
    raw.to_hdf(tmp_path / "ca_his_raw_2019.h5", key="t", mode="w")
    meta = pd.DataFrame({"ID": ids, "District": [4, 4, 4, 7, 8, 12, 11, 11, 3, 5, 4, 7]})
    meta.to_csv(tmp_path / "ca_meta.csv", index=False)
    return raw, meta


def test_prepare_region_matches_authors_notebooks(tmp_path):
    raw, meta = _fake_raw(tmp_path)
    for region, n in (("gba", 4), ("gla", 4), ("sd", 2)):
        df = prepare_region(tmp_path / "ca_his_raw_2019.h5", tmp_path / "ca_meta.csv", region,
                            tmp_path / f"{region}_his_2019.h5")
        assert df.shape == (14 * STEPS_PER_DAY, n)
        assert set(meta.loc[meta.District.isin(REGION_DISTRICTS[region]), "ID"]) == set(df.columns)
        assert not df.isnull().any().any()                       # missing -> 0
        assert (df.values == np.round(df.values)).all()          # rounded
    # 15-min value = rounded mean of its three 5-min values (the authors' resample('15T').mean().round(0))
    gba = pd.read_hdf(tmp_path / "gba_his_2019.h5")
    expect = round(raw["1001"].iloc[:3].mean())
    assert gba["1001"].iloc[0] == expect


def test_standardisation_follows_generate_data_for_training():
    vals = make_demo_traffic(8, 28, seed=1).values.astype(float)
    z, mean, std = standardize_like_authors(vals)
    n = vals.shape[0] - 23                                       # idx = arange(11, T - 12)
    cut = (11 + round(n * 0.6)) - 12                             # idx_val[0] - seq_x
    assert np.isclose(mean, vals[:cut].mean()) and np.isclose(std, vals[:cut].std())
    assert np.allclose(z * std + mean, vals, atol=1e-3 * np.abs(vals).max())


def test_loader_inputs_and_length(tmp_path):
    raw, meta = _fake_raw(tmp_path)
    prepare_region(tmp_path / "ca_his_raw_2019.h5", tmp_path / "ca_meta.csv", "gba", tmp_path / "gba_his_2019.h5")
    D = load_traffic(tmp_path / "gba_his_2019.h5")
    assert D["x"].shape == (4, 14 * STEPS_PER_DAY) and D["x"].dtype == np.float32
    assert load_traffic(tmp_path / "gba_his_2019.h5", length=501)["x"].shape[1] == 500    # even length
    # authors' his.npz: (T, N, 3) with the flow already standardised
    df = pd.read_hdf(tmp_path / "gba_his_2019.h5")
    z, mean, std = standardize_like_authors(df.values.astype(float))
    npz = np.zeros((len(df), 4, 3), np.float32)
    npz[:, :, 0] = z
    np.savez(tmp_path / "his.npz", data=npz, mean=mean, std=std)
    np.testing.assert_allclose(load_traffic(tmp_path / "his.npz")["x"], D["x"], atol=1e-6)


def test_node_split_is_the_authors():
    tr, va, te = split_nodes(716)
    assert (len(tr), len(va), len(te)) == (501, 107, 108)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)


def test_line_ownership_metric():
    T = 4 * 672
    t = np.arange(T)
    daily = np.cos(2 * np.pi * t / 96)
    weekly = np.cos(2 * np.pi * t / 672)
    x = daily + weekly
    clean = spectral_metrics(x, np.stack([daily, weekly]))              # each line in its own mode
    assert clean["daily_ownership"] > 0.99 and clean["weekly_ownership"] > 0.99
    assert abs(clean["daily_capture"] - 1) < 0.05 and clean["mode_overlap"] < 0.01
    mixed = spectral_metrics(x, np.stack([0.5 * x, 0.5 * x]))           # both lines split across two modes
    assert 0.45 < mixed["daily_ownership"] < 0.55 and mixed["mode_overlap"] > 0.99


def test_runner_smoke_vmd_only(tmp_path):
    import sys
    from experiments import run_traffic
    argv = sys.argv
    sys.argv = ["run_traffic.py", "--data", "demo", "--quick", "--skip-uvmd", "--demo-nodes", "30",
                "--demo-days", "14", "--K", "5", "--out", str(tmp_path)]
    try:
        run_traffic.main()
    finally:
        sys.argv = argv
    for f in ("summary.md", "summary.json", "per_node.csv", "cost.csv", "figures/example_decomposition.png"):
        assert (tmp_path / f).exists(), f
