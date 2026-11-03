"""Mode-comparison statistics and the compare script, on fake CWRU data."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from experiments import compare_uvmd_modes as cm  # noqa: E402
from experiments import run_cwru_e2e  # noqa: E402
from tests.fake_cwru import write_fake_cwru  # noqa: E402


def test_similarity_statistics():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 4, 200))
    al, mt = cm.similarity(A, A)
    np.testing.assert_allclose(al, 1.0, atol=1e-9)
    np.testing.assert_allclose(mt, 1.0, atol=1e-9)
    al, mt = cm.similarity(A, A[:, ::-1])                      # same modes, different order
    assert np.all(mt > 0.999) and np.all(np.abs(al).mean(1) < 0.5)
    al, _ = cm.similarity(A, -A)
    np.testing.assert_allclose(al, -1.0, atol=1e-9)             # aligned correlation keeps the sign
    np.testing.assert_allclose(cm.energy_share(A).sum(-1), 1.0)
    K = 4
    on_grid = np.tile((0.5 / K) * np.arange(K), (3, 1))
    assert cm.grid_shift(on_grid, K).max() == 0
    assert cm.grid_shift(on_grid + 0.5 / K, K).min() == pytest.approx(1.0)


def test_compare_on_fake_run(tmp_path):
    data, run = tmp_path / "cwru", tmp_path / "run"
    write_fake_cwru(data, seconds=1.2)
    run_cwru_e2e.main(["--data", str(data), "--T", "256", "--quick", "--layers", "2", "--out", str(run),
                       "--epochs", "3", "--early-epochs", "1", "--pipelines", "uvmd_early", "uvmd_e2e"])
    res = cm.main(["--run", str(run), "--data", str(data), "--n-windows", "12"])
    out = run / "mode_comparison"
    for f in ("mode_comparison.md", "parameters.csv", "similarity.csv", "quality.csv", "per_window.csv",
              "figures/example_modes.png", "figures/mode_spectra.png", "figures/parameters_and_similarity.png"):
        assert (out / f).exists(), f
    st = res["similarity"].set_index(["A", "B"])
    row = st.loc[(cm.E2E_, cm.EARLY)]
    assert -1.0 <= row.aligned_mean <= 1.0 and 0.0 <= row.matched_mean_abs <= 1.0
    assert (res["quality"].recon_rel_err > 0).all()
    # the early UVMD was loaded from the very file the run saved -> reuse is exact
    again = pd.read_csv(out / "parameters.csv")
    early = np.load(run / "early_uvmd_params_N2_K4.npz")
    row0 = again[again.model == cm.EARLY].iloc[0]
    np.testing.assert_allclose([row0[f"alpha_{k}"] for k in range(4)], early["alpha"], rtol=1e-6)


def test_identical_parameters_give_identical_modes(tmp_path):
    """Feed the end-to-end parameters in as the 'early' ones: the two UVMDs must be indistinguishable."""
    data, run = tmp_path / "cwru", tmp_path / "run"
    write_fake_cwru(data, seconds=1.2)
    run_cwru_e2e.main(["--data", str(data), "--T", "256", "--quick", "--layers", "2", "--out", str(run),
                       "--epochs", "3", "--early-epochs", "1", "--pipelines", "uvmd_e2e", "uvmd_early"])
    res = cm.main(["--run", str(run), "--data", str(data), "--n-windows", "10",
                   "--early-params", str(run / "uvmd_e2e_params_N2_seed0.npz"), "--out", str(tmp_path / "same")])
    row = res["similarity"].set_index(["A", "B"]).loc[(cm.E2E_, cm.EARLY)]
    assert row.aligned_mean == pytest.approx(1.0, abs=1e-4) and row.matched_mean_abs == pytest.approx(1.0, abs=1e-4)
    p = res["parameters"].set_index("model")
    assert p.loc[f"{cm.E2E_} seed 0", "lambda_change_vs_early"] == pytest.approx(0.0, abs=1e-5)
