# How to recreate this work

This guide has two parts:

* **Part A** reruns everything exactly as it was done, about 15 commands.
* **Part B** explains every step and decision, so you could rebuild it from scratch or change it.

---

## Part A: rerun everything

### A1. Environment (once, on node 7 or any machine)
```bash
git clone https://github.com/Aitzaz332/vmd-uvmd-comparison
cd vmd-uvmd-comparison
python -m venv .venv && source .venv/bin/activate      # or a conda env
pip install -r requirements.txt                         # numpy scipy pandas matplotlib sklearn tabulate torch pytest
# on the GPU node, install the CUDA build of torch that matches the driver, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### A2. Check that everything is wired correctly
```bash
python -m pytest -q        # expect: 15 passed
```
The tests check four things:
1. The two copied files (`uvmd.py` and `vmdpy.py`) are byte-identical to upstream.
2. Our VMD gives exactly the same output as `vmdpy`.
3. Untrained UVMD with N layers gives exactly the same output as VMD with N iterations.
4. The CWRU loader and fault metric behave correctly on fake data.

### A3. Dataset 1: synthetic
```bash
python experiments/run_synthetic.py --quick                         # ~10 s smoke test
python experiments/run_synthetic.py --device cuda:0                 # K=4  -> results/synthetic/
python experiments/run_synthetic.py --device cuda:0 --K 13 --out results/synthetic_K13
```

### A4. Dataset 2: CWRU
```bash
bash scripts/download_cwru.sh ~/data/CWRU                           # 40 .mat files, ~135 MB
python experiments/run_cwru.py --data ~/data/CWRU --quick           # ~40 s smoke test
python experiments/run_cwru.py --data ~/data/CWRU --device cuda:0                       # K=4
python experiments/run_cwru.py --data ~/data/CWRU --device cuda:0 --K 13 --out results/cwru_K13
# or all of A2 + A4 in one go:
bash scripts/run_cwru_cluster.sh ~/data/CWRU cuda:0
```
If node 7 has no internet access, download the files on your PC and copy them over:
`scp *.mat <user>@gpucluster2.st.lab.au.dk:~/data/CWRU/`.

To keep a long run alive after you log out, use `nohup python ... > run.log 2>&1 &` or `tmux`.

### A4b. Everything on node 7 in one command, results pushed to GitHub
```bash
H5=~/data/LargeST/sd_his_2019.h5 REG=sd bash scripts/run_all_node7.sh      # FULL=1 adds the full traffic run
```
It runs the tests, then the K=13 synthetic and CWRU runs and the bounded traffic run (only the ones whose
`results/<name>/summary.json` is missing), builds `results/comparison/`, and commits and pushes after each stage.
The machine needs a git login that can push to this repository.

### A5. What each run writes (in `--out`)
| file | contents |
|---|---|
| `summary.md` | tables and the full log. Read this first |
| `summary.json` | every averaged metric, the run arguments, and the environment |
| `per_signal.csv` / `per_window.csv` | one row per test signal and method |
| `by_class.csv` | CWRU only: metrics per class (normal / inner / outer / ball) |
| `cost.csv` | iterations or layers, parameters, FLOPs, ms per signal, training time |
| `uvmd_params_N*.npz` | learned UVMD parameters (alpha, lambda) |
| `figures/` | example decomposition, metric vs iterations, runtime, training curve |

The numbers will differ slightly between machines. Timings depend on the CPU or GPU, and GPU float
arithmetic is not bit-identical to CPU. The pattern of the results should not change.

---

## Part B: what was done and why

### B1. Finding the UVMD code
* Paper: Ahmad, Wesemann, Waschkowski, Khalid, *Robust Spatiotemporal Forecasting Using Adaptive
  Deep-Unfolded Variational Mode Decomposition*, arXiv:2509.00703 (LUMS / Maincode, 2025).
* The paper says the code is public but gives no link. The repository is
  **https://github.com/OsamaAhmad369/MAGN** (first author's account). The UVMD code is the single
  file `uvmd.py`. The rest of that repo is the traffic-forecasting network (ASTGCN), which we don't need.
* Pinned commit: `6bfae1bf3ebf5af4684382cd12ba23d28e3a6ae8`. That repo has **no license**, so keep yours private.

### B2. Using the upstream code without changing it
* The file is copied byte-for-byte to `third_party/MAGN/uvmd.py`. Its SHA-256 hash is checked by
  `tests/test_vendored_files.py`.
* Problem: `uvmd.py` is a script. Importing it immediately parses command-line arguments, loads
  `his.npz` and starts training.
* Solution (`decomp/uvmd_magn.py`): read the file as text, parse it with Python's `ast` module, keep
  only the `import` lines and the three model classes (`VMD_innerloop`, `VMD`, `Unfolding`), and run
  just those. The model we use is therefore the authors' code, not a re-typed copy.
* The training loop copies the upstream script: Adam with lr = 0.1, batch size 1 with shuffling,
  100 epochs, early stopping with patience 10 on validation loss. One upstream quirk is kept: the
  script ends with the last-epoch weights, not the best ones. The `--restore-best` flag uses the best
  weights instead.

### B3. The VMD baseline
* Reference: `vrcarva/vmdpy` (MIT license), copied unchanged to `third_party/vmdpy/vmdpy.py`.
* We also wrote `decomp/vmd_numpy.py`, which does the same arithmetic as vmdpy with three changes:
  1. It uses O(T·K) memory. vmdpy stores all 500 iterates.
  2. It can stop after **exactly n iterations** (`tol=-1, max_iter=n`). One UVMD layer is one VMD
     iteration, so "VMD-n" is the like-for-like baseline for an n-layer UVMD.
  3. It returns the **last** iterate. vmdpy returns the second-to-last, so after 1 iteration vmdpy
     returns all zeros.
* `tests/test_vmd_numpy.py` shows the outputs equal vmdpy's to 1e-10.
* Settings, the same as the paper and UVMD's initial values: alpha = 2000, tau = 0, a DC mode
  (mode 0 fixed at 0 Hz), centre frequencies initialised uniformly at ω_k = 0.5·k/K, tol = 1e-7,
  at most 499 iterations.

### B4. Why "untrained UVMD = VMD-N" matters
At initialisation UVMD has alpha_k = softplus(2000) = 2000 and lambda = 0. With those values one layer
is the same calculation as one VMD iteration with tau = 0. `tests/test_uvmd_magn.py` checks this for
N = 1, 2, 3. Any difference after training therefore comes from training alone.

### B5. Dataset 1: synthetic signals (`datasets/synthetic.py`)
* 200 signals of length T = 1024. Each is the sum of 4 known modes plus white noise (std 0.1).
  The frequencies are in cycles/sample:

  | mode | type | frequency |
  |---|---|---|
  | 0 | trend: two slow sines | 0.001–0.004 |
  | 1 | pure tone | 0.040–0.050 |
  | 2 | AM tone | 0.095–0.110 |
  | 3 | linear chirp | 0.22–0.24 |

  The amplitudes, phases and frequencies are drawn again for every signal (seed 0).
* Split: 70/15/15 with `train_test_split(random_state=42)`, the same as upstream. UVMD trains on
  140 signals, and every method is scored on the same 30 unseen signals.
* Metrics (`decomp/metrics.py`):
  * **Reconstruction:** RMSE, relative error ‖x − Σu‖/‖x‖, SNR, the upstream spectral L1 loss, RMSE
    against the clean signal, energy ratio, and orthogonality index.
  * **Mode recovery:** estimated modes are matched to the true modes with the Hungarian algorithm
    on |correlation|. Per mode we report NRMSE, correlation, SNR and centre-frequency error. A
    "grouped" SNR handles K > 4: each estimated mode is assigned to the true mode it projects onto
    most, and the assigned modes are summed.

### B6. Dataset 2: CWRU bearings (`datasets/cwru.py`)
* There are 40 files (`scripts/download_cwru.sh`):
  * normal baseline 97–100
  * 12 kHz drive-end inner race 105–108 / 169–172 / 209–212
  * ball 118–121 / 185–188 / 222–225
  * outer race @6:00 130–133 / 197–200 / 234–237

  That is fault sizes 0.007″, 0.014″ and 0.021″, each at loads 0–3 HP.
* We skip three groups:
  * 0.028″, because it has no outer-race recording
  * outer race @3:00 and @12:00, because they are missing for 0.014″
  * the 48 kHz drive-end and fan-end sets, kept for later
* File quirks the loader handles:
  1. Normal files are recorded at 48 kHz, so they are resampled to 12 kHz with `resample_poly`.
     Otherwise the sampling rate alone would give "normal" away.
  2. `99.mat` also contains the `X098_*` arrays, so the loader uses the key that matches the file number.
  3. `98.mat` has no RPM value, so the nominal speed for 1 HP (1772 rpm) is used.
* Each recording is cut into non-overlapping 2048-sample windows (171 ms). The first 30 per file are
  kept, giving 1199 windows. Each window is standardised to mean 0 and std 1.
* **Split by load:** train on 0–1 HP (599 windows), validate on 2 HP (300), test on 3 HP (300).
  No recording appears in two sets. Splitting windows at random would put neighbouring windows of
  one recording in both train and test.
* There are no true modes, so the metrics are reconstruction plus fault content
  (`decomp/fault_metrics.py`):
  1. Drop 16 samples at each window edge, because both methods leave a boundary spike in some modes.
  2. Pick the mode with the **highest kurtosis**, the most impulsive one.
  3. Take its **envelope spectrum**: |Hilbert transform|, remove the mean, apply a Hann window, FFT.
  4. Compute the fault frequencies from the shaft speed fr = rpm/60 and the SKF 6205 bearing:
     BPFI = 5.4152·fr, BPFO = 3.5848·fr, BSF = 2.357·fr.
  5. **Harmonic SNR** = 20·log10(peak within ±max(1.5 bins, 2%) of h·f ÷ median of the next 30 Hz on
     each side), averaged over harmonics h = 1, 2, 3.
  6. **Hit** = the true fault type has the highest SNR of the three.
  7. **Oracle SNR** = the best SNR over all modes, i.e. how much fault information the modes contain at all.
  8. The **raw signal** run through the same analysis is the reference.

### B7. Cost (`decomp/cost.py`)
* **Wall-clock:** median ms per signal, measured on the test set.
* **Analytic FLOPs:** each mode update costs about 19 FLOP per frequency bin, and VMD adds a
  convergence check of about 7 FLOP per bin. On top of that come one FFT of the mirrored signal
  (length 2T) and K inverse FFTs.
* **Training cost:** about 3 × the forward cost × signals × epochs.
* **Parameters:** K alphas plus (N+1) × 2T complex lambdas. Upstream allocates one lambda row it never uses.
* NumPy VMD and PyTorch UVMD are different frameworks, so their wall-clock times are not strictly
  comparable. The FLOP counts are.

### B8. Results so far
* **Synthetic:** trained UVMD reconstructs as well as converged VMD or better, but recovers the true
  modes **worse** than untrained UVMD. Training lowers alpha (2000 → ~500), which widens the mode
  filters, and in 1–2 layers the centre frequencies barely move.
* **CWRU (K=4):** trained UVMD reconstructs almost perfectly (relative error 4e-4). It does this by
  pushing one alpha to about 0, which turns that mode into a pass-everything residual. Its fault
  content stays about the same as VMD with 1–2 iterations. No method beats the raw signal on hit rate.
  UVMD costs 3–4 ms per window against 43 ms for converged VMD.

Full tables are in `README.md` and `results/*/summary.md`.

### B9. Things to know if you change something
* **Window length T:** UVMD learns one lambda vector of length 2T per layer, so a trained model only
  works on windows of exactly that length. Retrain if you change T.
* **K = 13** is the paper's setting. Use the grouped SNR (synthetic) or the oracle SNR (CWRU) to
  compare across K. The one-to-one metrics only score 4 of the 13 modes.
* **GPU:** upstream UVMD processes one signal at a time (batch size 1), so a GPU helps little. Its
  modules are rebuilt on every forward pass.
* **Causality:** both methods filter with a global FFT, so a mode value at time t uses samples after
  t. Don't feed these modes to a forecaster without checking for leakage.
* **Paper vs code:** the code uses `1 + softplus(α)(ω−ω_k)²` and an L1 loss. The paper writes
  `1 + 2·softplus(α)(…)` and an L2 loss. We run the code.
