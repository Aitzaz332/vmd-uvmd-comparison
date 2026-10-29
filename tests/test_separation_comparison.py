"""Criteria and tables of experiments/separation_comparison.py (no benchmark data or torch needed)."""
import numpy as np
import pandas as pd

from experiments import separation_comparison as sc


def _sines(T=256, cycles=(8, 32)):
    t = np.arange(T)
    u = np.stack([np.sin(2 * np.pi * c * t / T) for c in cycles])
    return u.sum(axis=0), u


def test_clean_separation_scores_as_perfect():
    x, u = _sines()
    m = sc.signal_criteria(x, u)
    assert m["recon_rel_err"] < 1e-12
    assert m["ortho_index"] < 1e-9 and m["mode_overlap"] < 1e-9 and m["bandwidth_rel"] < 1e-6
    assert abs(m["energy_ratio"] - 1) < 1e-9
    assert abs(m["n_eff_modes"] - 2) < 1e-9


def test_all_in_one_mode_is_flagged_by_the_guard():
    x, _ = _sines()
    u = np.stack([x, np.zeros_like(x), np.zeros_like(x)])
    m = sc.signal_criteria(x, u)
    assert m["ortho_index"] == 0 and m["mode_overlap"] == 0       # looks perfect ...
    assert abs(m["n_eff_modes"] - 1) < 1e-9                       # ... but only one mode carries energy


def test_mixed_modes_score_worse():
    x, u = _sines()
    mixed = np.stack([u[0] + 0.5 * u[1], u[1] + 0.5 * u[0]])
    good, bad = sc.signal_criteria(x, u), sc.signal_criteria(x, mixed)
    assert bad["ortho_index"] > good["ortho_index"] and bad["mode_overlap"] > good["mode_overlap"]


def _df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    snr = np.linspace(5, 25, n)
    d = {"grouped_mean_snr_db": snr,
         "recon_rel_err": snr * 0.01 + rng.normal(0, 1e-4, n),           # grows with true quality: misleading
         "ortho_index": 1 / snr, "mode_overlap": 1 / snr ** 2, "bandwidth_rel": 1 / snr ** 3,   # shrink: good
         "energy_ratio": rng.normal(1, 0.3, n), "n_eff_modes": snr / 5}
    return pd.DataFrame(d)


def test_spearman_table_verdicts():
    _, out = sc.spearman_table(_df())
    v = dict(zip(out.key, out.verdict))
    assert v["ortho_index"] == v["mode_overlap"] == v["bandwidth_rel"] == "tracks truth"
    assert v["recon_rel_err"].startswith("misleading")
    assert v["n_eff_modes"].startswith("guard")
    assert v["energy_ratio"] == "no relation"


def test_training_effect_compares_distance_from_one_for_the_energy_ratio():
    rows = []
    for method, er in (("VMD-1", 0.60), ("UVMD-1", 0.55)):          # UVMD is farther from 1: worse, although 0.55 < 0.60
        for s in range(5):
            rows.append({"dataset": "D", "method": method, "signal": s, "recon_rel_err": 0.1, "ortho_index": 0.1,
                         "mode_overlap": 0.1, "bandwidth_rel": 0.1, "energy_ratio": er, "n_eff_modes": 2.0})
    md = sc.table_training_effect(pd.DataFrame(rows), ["D"], [1])
    last = md.splitlines()[-1]
    assert "1.12× (0%)" in last                                   # 0.45 / 0.40 = 1.125, UVMD never closer to 1


def test_main_table_marks_best_and_ignores_the_guard():
    agg = pd.DataFrame([
        {"dataset": "D", "method": "A", "n": 3, **{k: v for k in [c[0] for c in sc.CRITERIA] for v in [1.0]},
         **{c[0] + "_std": 0.1 for c in sc.CRITERIA}},
        {"dataset": "D", "method": "B", "n": 3, **{k: 2.0 for k in [c[0] for c in sc.CRITERIA]},
         **{c[0] + "_std": 0.1 for c in sc.CRITERIA}}])
    md = sc.table_main(agg, ["D"], {"D": 13})
    a_row = next(r for r in md.splitlines() if "| A |" in r)
    b_row = next(r for r in md.splitlines() if "| B |" in r)
    assert a_row.count("**") == 2 * 5       # best of 5 criteria with a direction; the guard column is never bold
    assert "**" not in b_row
