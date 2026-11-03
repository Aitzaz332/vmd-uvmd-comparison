"""Leave-one-load-out split, the fold driver and the clean mode figures, on fake CWRU data."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from datasets.cwru import LOLO_FOLDS, load_cwru, split_indices  # noqa: E402
from experiments import run_cwru_lolo  # noqa: E402
from tests.fake_cwru import write_fake_cwru  # noqa: E402


def test_lolo_split_properties(tmp_path):
    write_fake_cwru(tmp_path, seconds=1.2)
    D = load_cwru(tmp_path, T=256, windows_per_file=3)
    for f in range(4):
        tr, va, te = split_indices(D, "lolo", fold=f)
        loads_tr, load_va = LOLO_FOLDS[f]
        assert set(D["load"][te]) == {f} and set(D["load"][va]) == {load_va} and set(D["load"][tr]) == set(loads_tr)
        assert f not in loads_tr and load_va != f
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(D["x"]) - (D["load"] == (set(range(4)) - set(loads_tr) - {load_va, f}).pop()).sum() \
            if len(set(loads_tr) | {load_va, f}) < 4 else len(tr) + len(va) + len(te) == len(D["x"])
    assert all(np.array_equal(a, b) for a, b in zip(split_indices(D, "lolo", fold=3), split_indices(D, "load")))
    assert sorted(f for f in LOLO_FOLDS) == [0, 1, 2, 3]


def test_lolo_driver_and_figures(tmp_path):
    data, out = tmp_path / "cwru", tmp_path / "lolo"
    write_fake_cwru(data, seconds=1.2)
    summ = run_cwru_lolo.main(["--data", str(data), "--T", "256", "--quick", "--layers", "2", "--epochs", "2",
                               "--folds", "0", "3", "--out", str(out)])
    assert list(summ.test_load_HP) == [0, 3]
    for f in (0, 3):
        d = out / f"test_{f}HP"
        for name in ("runs.csv", "prediction_comparison/prediction_comparison.md", "training_curves/overview_2x2.png",
                     "modes_figures/1_modes_one_window.png", "modes_figures/2_mode_spectra.png",
                     "modes_figures/3_centre_frequencies.png", "modes_figures/4_alpha_and_loss.png",
                     "modes_figures/alpha_table.csv", "vmd_modes_test.npz", "uvmd_e2e_modes_test_seed0.npz"):
            assert (d / name).exists(), (f, name)
        m = np.load(d / "uvmd_e2e_modes_test_seed0.npz")
        assert set(m["load"]) == {f}                                     # the exported modes are the held-out load
    assert (out / "lolo_summary.png").stat().st_size > 5000 and (out / "lolo_summary.md").exists()
    a = pd.read_csv(out / "test_0HP" / "modes_figures" / "alpha_table.csv")
    assert {"alpha_start", "alpha_selected_epoch", "alpha_last_epoch"} <= set(a.columns) and (a.alpha_start == 500.0).all()
