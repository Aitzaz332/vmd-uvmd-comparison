"""Statistics of the centre-frequency movement script (no data or torch needed)."""
import numpy as np

from experiments import omega_shift_all as osa


def test_shift_stats_in_grid_spacings():
    K = 5
    init = 0.5 / K * np.arange(K)
    om = np.tile(init, (4, 1))
    om[:, 2] += 0.5 / K              # mode 2 moves exactly one grid spacing in every signal
    om[:, 0] += 0.3                  # the DC mode is ignored
    s = osa.shift_stats(om, K)
    assert abs(s["mean_shift"] - 1 / (K - 1)) < 1e-12
    assert abs(s["max_mode_shift"] - 1.0) < 1e-12
    assert s["modes_moved"] == 1 and s["n_modes"] == K - 1 and s["n_signals"] == 4


def test_table_ratio_to_converged_and_markdown():
    K = 4
    init = 0.5 / K * np.arange(K)
    moved = lambda d: np.tile(init + np.r_[0, np.full(K - 1, d * 0.5 / K)], (3, 1))   # noqa: E731
    res = {"Synthetic": {"K": K, "omega": {osa.BASE: moved(1.0), "VMD-1": moved(0.5), "UVMD-1": moved(2.0)}}}
    df = osa.build_table(res)
    r = df.set_index("method")["share_of_converged"]
    assert abs(r[osa.BASE] - 1) < 1e-12 and abs(r["VMD-1"] - 0.5) < 1e-12 and abs(r["UVMD-1"] - 2.0) < 1e-12
    md = osa.markdown(df)
    assert "Synthetic (K=4)" in md and "2.00×" in md
