# End-to-end UVMD + 1D-CNN on CWRU (branch `e2e-uvmd-cnn`)

Question: does it help to train UVMD **through** a classifier, instead of on the reconstruction loss alone?

## Default comparison: VMD → CNN vs end-to-end UVMD → CNN
Defaults: K = 4, alpha = 500, converged VMD vs a **5-layer** UVMD trained by the CNN loss. Only `vmd` and `uvmd_e2e` run unless you pass `--pipelines`. The recon-trained UVMD is *not* retrained here (upstream training is slow, batch size 1); you supply yours for the mode comparison (below).
`alpha_k` is the bandwidth penalty of mode k: the mode filter is `1/(1+alpha_k (f - omega_k)^2)`, so larger alpha = narrower mode, alpha ~ 0 = the mode passes everything.

## Recommended evaluation: leave-one-load-out (`scripts/run_cwru_lolo_node7.sh`)
The default split (train 0-1 HP, test 3 HP) is dominated by two odd 3 HP recordings (files 172 and 225), which every method, including a raw-signal CNN, gets wrong; a time split shares recordings between train and test, so every method reaches ~100 %. Leave-one-load-out holds each load (0, 1, 2, 3 HP) out once (train/validation loads: `datasets/cwru.py::LOLO_FOLDS`), with the same CNN, K=4, N=5, alpha 500 and seeds in every fold.
```bash
ARGS="--quick" bash scripts/run_cwru_lolo_node7.sh   # smoke test, nothing committed
bash scripts/run_cwru_lolo_node7.sh                  # full run -> results/cwru_lolo_K4_N5, committed and pushed
```
Output: `lolo_summary.md/.png/.csv` (accuracy and loss per held-out load, VMD vs end-to-end UVMD) and, per fold, `test_<L>HP/` with `prediction_comparison/`, `training_curves/`, `modes_figures/` (1 modes of one window, 2 mode spectra, 3 centre frequencies, 4 alpha_k and loss over training; `alpha_table.csv`, `centre_frequencies.csv`) and the mode files `vmd_modes_test.npz`, `uvmd_e2e_modes_test_seed<s>.npz` of that fold's held-out load. `experiments/plot_modes_clean.py --run <folder>` redraws the mode figures of any run folder.

## The five pipelines available
One 1D-CNN (`e2e/cnn1d.py`), one training loop (`e2e/train.py::fit`), one split, one model-selection rule. Only the CNN input changes.

| key | input to the CNN | is the decomposition updated by a loss? |
|---|---|---|
| `vmd` | modes of classic ADMM VMD, run to convergence (alpha = `--alpha`, default 500, tau 0) | no, computed once |
| `vmd_n` | modes of VMD stopped after N iterations = untrained N-layer UVMD | no, computed once |
| `uvmd_early` | modes of the UVMD trained earlier on reconstruction only (upstream recipe) | no, frozen |
| `uvmd_e2e` | modes of a UVMD trained **together with the CNN** | yes: `CE + rec_weight * rel. spectral L1 residual` updates `alpha_k` and `lambda` |
| `raw` | the standardised window, 1 channel | – |

`vmd` and `uvmd_e2e` are the two cases asked for. The other three isolate the effect: `uvmd_e2e` and `vmd_n` start from the same modes and the same CNN initialisation (same seed), so any difference is what end-to-end training changed; `uvmd_e2e` and `uvmd_early` differ only in which loss trained the UVMD.

## Loss of the end-to-end UVMD
`loss = CE(logits, label) + rec_weight * mean_b( ||sum_k u_hat_k - f_hat||_1 / ||f_hat||_1 )`. The reconstruction term is the upstream UVMD loss divided by the L1 norm of the window's one-sided spectrum, so it is dimensionless (0 = perfect) and `--rec-weight` (default 0.1) has the same meaning for every window length. It never enters model selection: all pipelines pick the epoch with the best **validation accuracy** (ties: lower validation CE) and the best weights, UVMD parameters included, are restored before the test.

Only `alpha_raw` (K values, through softplus) and `lam` (N x 2T complex) of the UVMD are trainable, as in the paper. Their learning rates are separate (`--lr-alpha 2.0`, `--lr-lam 0.05`, CNN `--lr-cnn 1e-3`) because they live on very different scales; `0` freezes a parameter. **These defaults are not tuned**: on the synthetic stand-in data three of the four `alpha_k` moved down at close to the largest step Adam allows (about `--lr-alpha` per step), so on real data they can travel far; the per-epoch alpha is saved (`histories.json`, `figures/training_curves.png`) so you can see it. `--e2e-init early` starts from the recon-trained UVMD instead of the untrained one.

## Data
`datasets/cwru.py`: the 40-file subset, windows of T=2048 samples, 30 per file, each window standardised. Split `load` (default): train 0-1 HP, validate 2 HP, test 3 HP, no recording shared between sets. Labels `fault10` (default: normal + inner/ball/outer x 0.007/0.014/0.021 in) or `fault4`.

## Run (node7)
```bash
git fetch origin && git checkout e2e-uvmd-cnn && git pull
ARGS="--quick" bash scripts/run_cwru_e2e_node7.sh      # smoke run, nothing committed
bash scripts/run_cwru_e2e_node7.sh                     # K=4, N=5, alpha 500, 3 seeds; commits results/cwru_e2e_K4_N5 and pushes the branch
K=4 LAYERS=5 ALPHA=500 SEEDS="0 1 2 3 4" bash scripts/run_cwru_e2e_node7.sh
# also compare the modes with your recon-trained UVMD (same K, N, alpha; T=2048):
RECON=results/cwru/uvmd_params_N5.npz bash scripts/run_cwru_e2e_node7.sh
```
(Only with `--pipelines uvmd_early` or `--e2e-init early`: if no early parameter file is given, the early UVMD for that K and N is **trained first with the upstream recipe** (Adam lr 0.1, batch size 1, up to 100 epochs). That is the slow part (the earlier N=2 run took about 10 minutes on cuda). It is saved as `early_uvmd_params_N<N>_K<K>.npz` in the output folder and reused when the folder is reused.

Programs: `experiments/run_cwru_e2e.py`, `experiments/compare_predictions.py --run <out>` (loss / accuracy / confusion matrices, VMD vs e2e UVMD), `experiments/compare_uvmd_modes.py --run <out> --data <cwru> [--early-params <recon uvmd npz>]`, and `experiments/extract_uvmd_modes.py --params <npz> | --vmd --data <cwru> --out <npz>` (modes of ANY UVMD parameter file on the same windows, so yours and the e2e one are directly comparable).

Train your own recon-only UVMD for the comparison with the repo's `experiments/run_cwru.py` (K=4, alpha 500, 5 layers; check `--help` for the exact flags).

## Outputs (`results/cwru_e2e_K<K>_N<N>/`)
* `summary.md`, `summary.json`, `runs.csv`: test accuracy and macro-F1 per pipeline (mean ± std over seeds), validation accuracy, recon error, cost; per-class recall.
* `predictions.csv`: the predicted class of every test window, for every pipeline and seed. `confusion_<pipeline>.csv`: confusion matrices summed over seeds.
* `histories.json`, `figures/`: per-epoch curves, including the alpha of the end-to-end UVMD.
* `uvmd_e2e_params_N<N>_seed<s>.npz`: the trained end-to-end UVMD, in the upstream export format (`alpha`, `lambda`, `omega0`; plus `alpha_raw`).
* `mode_comparison/mode_comparison.md` (+ csv, `figures/`): modes of the end-to-end UVMD against the early UVMD, VMD-N and converged VMD on the same test windows: learned alpha, lambda change, centre frequencies, mode-by-mode correlation, energy share, reconstruction error, orthogonality, envelope-spectrum fault SNR.

* `prediction_comparison/prediction_comparison.md` (+ csv, `figures/confusion_matrices.png`, `figures/loss_and_accuracy.png`): test CE, accuracy, macro-F1, per-class P/R/F1, confusion matrices, paired right/wrong with McNemar.
* `training_curves/` (`experiments/plot_training_curves.py --run <out>`): slide-sized `training_loss.png`, `validation_loss.png`, `training_accuracy.png`, `validation_accuracy.png`, a 2x2 `overview_2x2.png` and `curves.csv`; mean over seeds with min-max band, dot = selected epoch. Training numbers are measured in eval mode (no dropout) so they are comparable with validation.
* `vmd_modes_test.npz`, `uvmd_e2e_modes_test_seed<s>.npz`: the test-window modes (n, K, T), centre frequencies, the input window, metadata, and for UVMD the alpha/lambda. Files are ~4 MB per 100 windows; check size before committing many.

## What was verified (`pytest`)
* `tests/test_uvmd_torch.py`: the batched UVMD equals `decomp/vmd_numpy.py` to 1e-15 in float64 (random alpha and complex lambda), and equals the upstream `Unfolding` bit for bit in float32 with random learned parameters; a batch equals one window at a time; gradients reach `alpha_raw` and `lam`; two deliberately wrong variants are caught.
* `tests/test_e2e.py`: the loss reaches UVMD and CNN; `fit` learns, restores the best weights and leaves a frozen UVMD untouched; a full run on fake CWRU files (all pipelines, early UVMD, e2e parameters in upstream format, mismatched parameters refused).
* `tests/test_compare_uvmd_modes.py`: similarity statistics; feeding identical parameters into both sides gives similarity 1.0.
* At run time, `run_cwru_e2e.py` re-checks the batched UVMD against the NumPy replica on the early parameters and stops if they disagree.

## Caveats
* Everything above was exercised on **synthetic stand-in files** (`tests/fake_cwru.py`) and on CPU. No real CWRU data and no GPU were available where this was written, so the CUDA path and the real-data accuracies are untested. Treat the first node7 run as the real test.
* Hyperparameters (learning rates, `rec_weight`, epochs 60, batch 32) are untuned starting points.
* The decomposition is non-causal (global FFT filters): fine for classifying a whole window offline, not for forecasting the samples after it.
* Test accuracy on 300 windows from 10 recordings is noisy; use several seeds and read the per-class recall.
