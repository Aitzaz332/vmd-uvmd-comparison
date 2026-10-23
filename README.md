# VMD vs UVMD comparison

This repo benchmarks classic **Variational Mode Decomposition (VMD)**, solved with ADMM, against
**UVMD**, the deep-unfolded VMD of the 2025 LUMS paper:

> O. Ahmad, L. Wesemann, F. Waschkowski, Z. Khalid. *Robust Spatiotemporal Forecasting Using Adaptive
> Deep-Unfolded Variational Mode Decomposition.* arXiv:2509.00703 (2025).
> Code: https://github.com/OsamaAhmad369/MAGN

UVMD is used **exactly as published**. `third_party/MAGN/uvmd.py` is the upstream file, unchanged and
hash-checked. `decomp/uvmd_magn.py` loads its model classes (`VMD_innerloop`, `VMD`, `Unfolding`)
straight from that file and copies its training procedure. See `third_party/SOURCES.md`.

## What UVMD is, in one paragraph
One UVMD layer is one ADMM iteration of VMD. The difference is that the bandwidth penalty of each mode
(`alpha_k`, through softplus) and the Lagrange multiplier of each layer (`lambda`, a complex vector of
length 2T) are learnable. They are trained by Adam to minimise the spectral reconstruction error
`sum |sum_k u_hat_k - f_hat|`. Training uses no ground-truth modes. The paper uses 1 to 2 layers and K = 13.
VMD runs until it converges, usually tens to hundreds of iterations. UVMD stops after N = 1 or 2 layers.

## Layout
```
third_party/MAGN/uvmd.py      upstream UVMD script (unchanged)
third_party/vmdpy/vmdpy.py    upstream vmdpy (MIT), reference VMD
decomp/uvmd_magn.py           loads the upstream classes; upstream training loop; helpers
decomp/vmd_numpy.py           VMD with memory O(TK) and fixed-iteration mode (tests show it equals vmdpy)
decomp/metrics.py             reconstruction, spectral, mode-recovery, orthogonality metrics
decomp/cost.py                analytic FLOP model and timing helpers
datasets/synthetic.py         dataset 1: signals built from 4 known modes plus noise
experiments/run_synthetic.py  dataset-1 benchmark (figures plus summary.md)
tests/                        equivalence and integrity tests
```

## Run
```bash
pip install -r requirements.txt          # torch: CPU or CUDA build from pytorch.org
python -m pytest -q                      # 1) checks that everything is wired correctly
python experiments/run_synthetic.py --quick   # 2) about a minute
python experiments/run_synthetic.py           # 3) full run: 200 signals, T=1024, K=4, N=1,2 layers
python experiments/run_synthetic.py --K 13    #    paper setting for K
python experiments/run_synthetic.py --device cuda:0
```
The results go to `results/synthetic/`: `summary.md`, `summary.json`, `per_signal.csv`, `cost.csv`,
the learned UVMD parameters, and `figures/`.

## Dataset 1: synthetic
Each signal is the sum of 4 known modes: a slow trend (≈0.001–0.004 cycles/sample), a pure tone
(0.040–0.050), an AM tone (0.095–0.110) and a linear chirp (0.22–0.24), plus white noise
(std 0.1 by default). The amplitudes, phases and frequencies are drawn again for every signal. The
upstream split is used: 70 / 15 / 15, `random_state=42`. UVMD trains on the training signals and every
method is scored on the same unseen test signals.

## Methods compared
| name | what it is |
|---|---|
| VMD (converged) | ADMM until `uDiff <= 1e-7` (max 499 iterations), alpha = 2000, tau = 0, DC mode, uniform init |
| VMD-n | the same solver stopped after exactly n iterations (a sweep over n) |
| UVMD-N untrained | the upstream `Unfolding` at its initial parameters. This is mathematically VMD-N, and a test checks it |
| UVMD-N | the upstream `Unfolding` trained with the upstream recipe (Adam lr 0.1, bs 1, 100 epochs, patience 10) |
| vmdpy package | the reference implementation, timed only |

## Metrics
* **Reconstruction:** RMSE, relative error, SNR (dB) of `sum_k u_k` against the input; the upstream
  spectral L1 loss; RMSE against the clean signal, which measures denoising.
* **Mode recovery** against the ground truth (Hungarian matching on |corr|): per-mode RMSE, NRMSE,
  correlation, SNR (dB), and centre-frequency error. Also a "grouped" SNR that works for any K (e.g. 13),
  and the energy share of spurious modes.
* **Mode quality:** orthogonality index and energy ratio.
* **Cost:** median wall-clock time per signal, iterations or layers, trainable parameters, analytic
  FLOPs per signal (`decomp/cost.py`), training time and training FLOPs.

## Notes on the upstream code (kept as-is, documented here)
* The paper writes the denominator as `1 + 2·softplus(alpha)·(w - w_k)^2` and the loss as squared L2.
  The code uses `1 + softplus(alpha)·(w - w_k)^2` and an L1 loss. This benchmark runs the code.
* `lambda_hat` has `layers + 1` rows, and the last row is never used. The parameter count printed
  upstream includes it.
* The upstream script saves the best-validation weights during training. It then overwrites them with
  the last-epoch weights before export. Upstream behaviour is the default here; `--restore-best` uses
  the best weights instead.
* Every step is a global FFT filter, so the decomposition is **non-causal**: a mode value at time t
  depends on later samples. This matters when the modes are used as forecasting inputs.
* vmdpy returns the second-to-last ADMM iterate. `decomp/vmd_numpy.py` returns the last one.
  `tests/test_vmd_numpy.py` checks that both give the same numbers.

## License
The upstream MAGN repository has no license file. Keep this repository private, or ask the authors
before publishing. vmdpy is MIT.
