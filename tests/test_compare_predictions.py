"""Head-to-head prediction report (compare_predictions.py) and the standalone mode extraction, on fake CWRU data."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from experiments import compare_predictions as cp  # noqa: E402
from experiments import extract_uvmd_modes as ex  # noqa: E402
from experiments import run_cwru_e2e  # noqa: E402
from tests.fake_cwru import write_fake_cwru  # noqa: E402


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    data, run = tmp / "cwru", tmp / "run"
    write_fake_cwru(data, seconds=1.2)
    run_cwru_e2e.main(["--data", str(data), "--T", "256", "--quick", "--layers", "2", "--epochs", "3", "--out", str(run),
                       "--seeds", "0", "1"])
    return data, run


def test_prediction_comparison_numbers_match_the_run(fake_run):
    data, run = fake_run
    res = cp.main(["--run", str(run)])
    out = run / "prediction_comparison"
    for f in ("prediction_comparison.md", "metrics_per_seed.csv", "per_class.csv", "paired.csv",
              "confusion_vmd_pooled.csv", "confusion_uvmd_e2e_pooled.csv", "figures/confusion_matrices.png",
              "figures/loss_and_accuracy.png"):
        assert (out / f).exists(), f
    runs = pd.read_csv(run / "runs.csv").set_index(["pipeline", "seed"])
    ms = res["metrics"].set_index(["pipeline", "seed"])
    for key in ms.index:                                    # recomputed from predictions.csv == what the run logged
        assert ms.loc[key, "test_acc"] == pytest.approx(runs.loc[key, "test_acc"])
        assert ms.loc[key, "test_ce"] == pytest.approx(runs.loc[key, "test_ce"], rel=1e-4)
        assert ms.loc[key, "macro_f1"] == pytest.approx(runs.loc[key, "test_macro_f1"], abs=1e-9)
    for k in ("vmd", "uvmd_e2e"):                            # pooled confusion = the run's summed confusion
        a = pd.read_csv(run / f"confusion_{k}.csv", index_col=0).values
        b = pd.read_csv(out / f"confusion_{k}_pooled.csv", index_col=0).values
        np.testing.assert_array_equal(a, b)
    pr = res["paired"]
    n = pd.read_csv(run / "predictions.csv").query("pipeline == 'vmd' and seed == 0").shape[0]
    assert ((pr["both right"] + pr["only A right"] + pr["only B right"] + pr["both wrong"]) == n).all()


def test_missing_pipeline_is_reported(fake_run):
    _, run = fake_run
    with pytest.raises(ValueError, match="raw"):
        cp.main(["--run", str(run), "--a", "raw", "--b", "uvmd_e2e", "--out", str(run / "x")])


def test_extract_modes_from_any_parameter_file(fake_run, tmp_path):
    data, run = fake_run
    saved = np.load(run / "uvmd_e2e_modes_test_seed0.npz")
    out = tmp_path / "modes.npz"
    modes, omega = ex.main(["--params", str(run / "uvmd_e2e_params_N2_seed0.npz"), "--data", str(data), "--T", "256",
                            "--windows-per-file", "6", "--subset", "test", "--out", str(out)])
    m = np.load(out)
    assert m["modes"].shape[1:] == (4, 256) and str(m["subset"]) == "test"
    np.testing.assert_array_equal(m["window"], saved["window"])
    np.testing.assert_allclose(m["modes"], saved["modes"], atol=1e-4 * np.abs(saved["modes"]).max())   # same as the run's export
    # classic VMD, all windows
    ex.main(["--vmd", "--K", "4", "--alpha", "500", "--data", str(data), "--T", "256", "--windows-per-file", "6",
             "--subset", "all", "--out", str(tmp_path / "vmd.npz")])
    v = np.load(tmp_path / "vmd.npz")
    assert v["modes"].shape == (240, 4, 256) and "alpha" not in v.files
    with pytest.raises(ValueError, match="samples"):         # window length of the file must match --T
        ex.main(["--params", str(run / "uvmd_e2e_params_N2_seed0.npz"), "--data", str(data), "--T", "512",
                 "--out", str(tmp_path / "bad.npz")])


def test_training_curves_figures(tmp_path):
    from experiments import plot_training_curves as pc
    from experiments import run_cwru_e2e
    from tests.fake_cwru import write_fake_cwru
    data, run = tmp_path / "cwru", tmp_path / "run"
    write_fake_cwru(data, seconds=1.2)
    run_cwru_e2e.main(["--data", str(data), "--T", "256", "--quick", "--layers", "2", "--epochs", "3", "--out", str(run)])
    out = pc.main(["--run", str(run)])
    for f in ("training_loss", "validation_loss", "training_accuracy", "validation_accuracy", "overview_2x2"):
        assert (out / f"{f}.png").stat().st_size > 5000, f
    c = pd.read_csv(out / "curves.csv")
    assert set(c.panel) == {"train_loss", "val_loss", "train_acc", "val_acc"} and set(c.pipeline) == {"vmd", "uvmd_e2e"}
    assert c.value.between(0, 50).all()
