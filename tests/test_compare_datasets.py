"""The cross-dataset comparison builds a table and a figure from summary.json files (fake ones here)."""
import json

from experiments import compare_datasets as cd


def _run(folder, K, extra):
    base = {"recon_rel_err": 0.03, "ortho_index": 0.04, "energy_ratio": 0.9, **extra}
    summ = {
        "VMD (converged)": dict(base),
        "VMD-1": {**base, "recon_rel_err": 0.05, "ortho_index": 0.13},
        "UVMD-1": {**base, "recon_rel_err": 0.007, "ortho_index": 0.17, "alpha_learned": [600.0, 0.0, 700.0]},
    }
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps({"args": {"K": K, "layers": [1]}, "summary": summ}))


def test_table_and_figure_with_a_missing_dataset(tmp_path):
    _run(tmp_path / "syn", 13, {"grouped_mean_snr_db": 20.0, "mean_omega_err": 0.001})
    _run(tmp_path / "cwru", 13, {"oracle_fault_snr_db": 10.0, "envelope_hit": 0.7})
    out = tmp_path / "out"
    cd.main(["--synthetic", str(tmp_path / "syn"), "--cwru", str(tmp_path / "cwru"),
             "--traffic", str(tmp_path / "does_not_exist"), "--out", str(out)])
    md = (out / "comparison_table.md").read_text(encoding="utf-8")
    table, alpha = md.split("## Learned")
    assert "Synthetic (K=13)" in table and "CWRU (K=13)" in table and "LargeST" not in table
    assert "different K" not in md                              # same K, no warning
    assert alpha.count("| 1 |") == 2                           # one alpha below 5 in each trained model
    assert (out / "comparison.png").stat().st_size > 5000
    assert (out / "comparison_table.csv").exists()


def test_warns_when_k_differs(tmp_path):
    _run(tmp_path / "syn", 13, {"grouped_mean_snr_db": 20.0})
    _run(tmp_path / "cwru", 4, {"oracle_fault_snr_db": 10.0})
    cd.main(["--synthetic", str(tmp_path / "syn"), "--cwru", str(tmp_path / "cwru"),
             "--traffic", str(tmp_path / "none"), "--out", str(tmp_path / "out")])
    assert "different K" in (tmp_path / "out" / "comparison_table.md").read_text(encoding="utf-8")
