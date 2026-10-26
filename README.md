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
datasets/cwru.py              dataset 2: CWRU bearing data loader (windows, labels, fault frequencies)
decomp/fault_metrics.py       envelope-spectrum fault metrics (no ground-truth modes needed)
experiments/run_cwru.py       dataset-2 benchmark
scripts/download_cwru.sh      downloads the 40-file CWRU subset
scripts/run_cwru_cluster.sh   full CWRU run on a GPU node (K=4 and K=13)
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

## Results — dataset 1 (synthetic, 30 unseen test signals, CPU, 2 threads)
Full tables: `results/synthetic/summary.md` (K=4) and `results/synthetic_K13/summary.md` (K=13, paper setting).

| K | method | rel. recon error | grouped mode SNR (dB) | ms / signal | training |
|---|---|---|---|---|---|
| 4 | VMD converged (~117 it) | 0.219 | 17.5 | 16.4 | – |
| 4 | VMD-1 / VMD-2 | 0.469 / 0.380 | 7.2 / 12.3 | 0.64 / 0.88 | – |
| 4 | UVMD-1 / UVMD-2 trained | 0.221 / 0.176 | 6.4 / 8.9 | 1.5 / 1.6 | 49 s / 72 s |
| 13 | VMD converged (~498 it) | 0.029 | 20.7 | 450 | – |
| 13 | VMD-1 / VMD-2 | 0.050 / 0.038 | 12.4 / 18.8 | 2.2 / 3.3 | – |
| 13 | UVMD-1 / UVMD-2 trained | 0.007 / 0.014 | 8.5 / 7.3 | 3.8 / 5.1 | 130 s / 231 s |

What the numbers show:
* **Reconstruction:** trained UVMD matches or beats converged VMD, using 1–2 layers instead of 100–500 iterations.
* **Mode recovery:** trained UVMD is *worse* than the untrained version, which is plain VMD with 1–2
  iterations. The loss rewards only reconstruction, so training lowers every alpha (2000 → ~500 at K=4),
  which widens the mode filters. The centre frequencies barely move in 1–2 layers. Modes absorb
  neighbouring content (orthogonality index 3–4× higher at K=4).
* **Cost:** UVMD inference costs ~10× (K=4) to ~120× (K=13) less than converged VMD. Training costs the same as roughly
  3,000 converged VMD decompositions at K=4, or ~300 at K=13.
* **Training length:** the training loss was still falling at epoch 100, so longer training may change these numbers.
* **Wall-clock times:** at the same iteration count, PyTorch-per-signal UVMD is slower than NumPy VMD
  (framework overhead, batch size 1). The FLOP counts in `cost.csv` are comparable across both.

## Dataset 2: CWRU bearing data
The 40 files used: normal baseline 97–100, plus the 12 kHz drive-end inner-race, ball and
outer-race @6:00 recordings at fault sizes 0.007″, 0.014″ and 0.021″, for motor loads 0–3 HP. Run
`bash scripts/download_cwru.sh ~/data/CWRU` to download them.

* The normal files are recorded at 48 kHz. The loader resamples them to 12 kHz.
* `99.mat` also contains the arrays of `98.mat`. The loader uses the key that matches the file number.
* `98.mat` has no RPM value, so the nominal speed for its load is used.
* Every recording is cut into non-overlapping windows of 2048 samples (171 ms); by default the first 30 of each are used.
  Each window is standardised.
* The split is by motor load: train on 0–1 HP, validate on 2 HP, test on 3 HP. No recording appears
  in more than one set. `--split time` splits each recording in time instead.

CWRU has no ground-truth modes, so the evaluation uses:
* the **reconstruction** metrics from dataset 1
* **envelope analysis of the mode with the highest kurtosis**: the harmonic SNR at 1–3× the
  fault frequency of the true fault (BPFI 5.4152, BPFO 3.5848, BSF 2.357 × shaft speed; SKF 6205), the hit
  rate (the true fault's frequency is the strongest of the three), and the best SNR across all modes
* the **raw signal** without decomposition, as a reference
* the same **cost** measurements as dataset 1

```bash
python experiments/run_cwru.py --data ~/data/CWRU --device cuda:0            # K=4
python experiments/run_cwru.py --data ~/data/CWRU --device cuda:0 --K 13 --out results/cwru_K13
bash scripts/run_cwru_cluster.sh ~/data/CWRU cuda:0                          # both, plus tests
```
Upstream UVMD processes one window at a time (batch size 1), so a GPU speeds it up less than you might expect.

### Results: dataset 2, K=4 (300 test windows at 3 HP, CPU)
Full tables: `results/cwru/summary.md`. Per-class numbers: `results/cwru/by_class.csv`.

| method | rel. recon error | energy ratio | ortho index | fault SNR (dB) | hit rate | oracle SNR (dB) | ms / window | training |
|---|---|---|---|---|---|---|---|---|
| raw signal (no decomposition) | – | – | – | 8.90 | 0.70 | – | – | – |
| VMD converged (~82 it) | 0.324 | 0.63 | 0.034 | 7.48 | 0.69 | 10.3 | 43.3 | – |
| VMD-1 / VMD-2 | 0.554 / 0.479 | 0.24 / 0.34 | 0.024 / 0.026 | 8.47 / 8.68 | 0.71 / 0.68 | 10.8 / 10.6 | 1.4 / 2.0 | – |
| UVMD-1 trained | 0.0004 | 0.83 | 0.083 | 8.64 | 0.70 | 10.1 | 2.9 | 275 s (66 ep.) |
| UVMD-2 trained | 0.0004 | 1.21 | 0.381 | 7.58 | 0.69 | 8.6 | 4.1 | 512 s (76 ep.) |

What the numbers show:
* **Reconstruction:** trained UVMD reconstructs the input almost exactly. It gets there by shrinking
  one mode's alpha to ≈0, which makes that mode an all-pass "residual" mode. The learned alphas were
  [220, 0, 127, 146] for N=1 and [257, 75, 0, 168] for N=2. The multipliers it learned close the
  remaining gap. The near-zero loss is therefore a degenerate solution, not better separation:
  at N=2 the modes overlap (energy ratio 1.21, orthogonality index 0.38).
* **Fault content:** no method beats the raw signal on hit rate. In fault SNR, only VMD-1, VMD-2 and
  UVMD-1 come within 0.4 dB of it, and converged VMD and UVMD-2 fall about 1.4 dB below it.
  The best mode (oracle SNR) is clearer than the raw signal for every method, but max kurtosis does
  not always pick that mode. Ball faults stay hard for every method (hit rate ≈0.3–0.4), as in the CWRU literature.
* **Cost:** a trained UVMD window costs 3–4 ms, against 43 ms for converged VMD. Training takes 4.5–8.5 minutes on CPU.

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
