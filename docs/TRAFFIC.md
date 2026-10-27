# Dataset 3: LargeST traffic (the UVMD paper's dataset)

This runs VMD and UVMD on the traffic data of Ahmad et al. 2025 (arXiv:2509.00703) the way the paper and the
authors' `uvmd.py` do, and compares the two decompositions.

## What the paper did (checked against the PDF and the authors' code)

| item | paper / authors' code | this repo |
|---|---|---|
| data | LargeST, year 2019, 15-minute flow | same |
| regions | GBA 2352, GLA 3834, SD 716 sensors (6902 total) | `--region gba/gla/sd` |
| signal | one signal per sensor, the whole year: 35,040 samples | same (`--length` shortens it) |
| normalisation | one global mean/std over all sensors, first 60% of time | same (`standardize_like_authors`) |
| split | sensors 70 / 15 / 15 %, `random_state=42`, batch size 1 | same (`split_nodes`) |
| UVMD | K=13, 1–2 layers, α=2000, H (lambda)=0, Adam lr 0.1, 100 epochs, early stopping, reconstruction loss only | the authors' code, unchanged |
| VMD baseline | iterative VMD (the paper gives no settings) | α=2000, τ=0, tol 1e-7, at most 499 iterations |

**Check that your setup is the paper's:** the number of trainable UVMD parameters at T=35,040, K=13 is
140,173 for 1 layer and 210,253 for 2 layers. These equal the "140.173K" and "210.253K" of the paper's
Tables I and II. The run prints it (`trainable parameters: …`).

## Step 1. Get the raw data (on node 7)
The data is on Kaggle (account needed): https://www.kaggle.com/datasets/liuxu77/largest
```bash
pip install kaggle                      # token: Kaggle -> Settings -> Create New Token -> ~/.kaggle/kaggle.json
mkdir -p ~/data/LargeST && cd ~/data/LargeST
kaggle datasets download -d liuxu77/largest && unzip largest.zip
find . -name "ca_his_raw_2019.h5" -o -name "ca_meta.csv"      # note where the two files ended up
```
You can also download in the browser and copy the zip with `scp`. Only `ca_his_raw_2019.h5` and `ca_meta.csv` are needed.

## Step 2. Build a region file
```bash
python scripts/prepare_traffic.py --raw <path>/ca_his_raw_2019.h5 --meta <path>/ca_meta.csv \
       --region sd --out ~/data/LargeST/sd_his_2019.h5
```
It prints the number of sensors (should be 716 for `sd`, 2352 for `gba`, 3834 for `gla`). This is the authors'
preprocessing: district filter (GBA = 4, GLA = 7, 8, 12, SD = 11), 5→15 minute averaging, rounding, missing → 0.
It may need about 10 GB of RAM for the raw file. If you already made `his.npz` with the authors' scripts,
pass that file as `--data` instead.

## Step 3. Run
```bash
python experiments/run_traffic.py --data demo --quick          # 30 s; checks the installation. Generated data, NOT traffic.
bash scripts/run_traffic_cluster.sh ~/data/LargeST/sd_his_2019.h5 sd cuda:0      # demo + a bounded real run
FULL=1 bash scripts/run_traffic_cluster.sh ~/data/LargeST/sd_his_2019.h5 sd cuda:0   # + the full paper protocol (hours)
bash scripts/run_traffic_paper_cases.sh ~/data/LargeST/sd_his_2019.h5 sd cuda:0      # window-length and K ablations
```
Start with SD (the smallest, 716 sensors).

### How long will it take?
Measured on a small CPU sandbox (2 threads) with generated data at the real length (35,040 samples, K=13):

| step | time per sensor |
|---|---|
| VMD to convergence (hit the 499-iteration cap) | 31 s |
| VMD, 1 iteration | 0.11 s |
| UVMD, 1 layer, decomposition (CPU) | 0.08 s |
| UVMD, 1 layer, training (forward + backward, CPU) | about 0.65 s per sensor per epoch |

These are not GPU timings. On SD the training set is 501 sensors, so 100 epochs on CPU would take about 9 hours;
a GPU should be much faster, but I have not measured it. This is why the bounded run (`--max-train-nodes 100
--max-test-nodes 30`) comes first. VMD at 31 s per sensor means 30 test sensors take about 15 minutes. If the
log says `0% converged`, VMD ran into the iteration cap on your data, and that is the time you are seeing.

## What you get (`results/traffic_<region>*/`)
* `summary.md`: all tables and the log. Read this first.
* `per_node.csv`, `summary.json`, `cost.csv`, `uvmd_params_N*.npz`
* `figures/`: `example_decomposition.png` (modes and their spectra for one sensor), `uvmd_training.png`
  (loss per epoch, paper Fig. 2), `learned_bandwidth.png` (learned α_k per mode, paper Fig. 2/3 right),
  `recon_err_vs_iterations.png`, `mode_overlap_vs_iterations.png`, `runtime.png`

### Metrics (no ground-truth modes exist for traffic)
* **Reconstruction:** relative error, SNR, the spectral L1 loss (the authors' code) and L2 loss (the paper's L_rec).
* **Mode structure** (`decomp/traffic_metrics.py`): mean bandwidth of the modes (cycles/day); `mode_overlap`
  (0 = modes at different frequencies, 1 = identical: the mode mixing the paper says α_k prevents); who owns the
  daily and weekly lines: `*_ownership` (1 = one mode holds the line, 1/K = split across all modes) and
  `*_capture` (≈1 = the line's power is kept).
* **Paired comparison:** for each N, the share of test sensors where trained UVMD-N beats VMD-N.
* **Cost:** time per sensor, extrapolated total for the region and for the paper's 6902 sensors, the speed-up
  over converged VMD, FLOPs, parameters, training time. If `--device cuda`, the trained UVMD is also timed on CPU.

## Paper claims you can now test
| claim (paper) | where to look |
|---|---|
| UVMD reconstructs the signal "perfectly" | `recon_rel_err`, `spec_l2` of UVMD-N vs VMD, `uvmd_training.png` |
| learned α_k differ by mode; some go to low values | `learned_bandwidth.png`, `alpha_learned` in `summary.json` |
| α_k prevents spectral overlap | `mode_overlap` and `*_ownership`, UVMD vs VMD |
| decomposition time falls from 267 min to 98.63 s on the whole LargeST (6902 sensors) | `cost.csv`: `est_min_6902_sensors` and `speedup_vs_VMD_converged` |
| shorter windows train faster but lose low-frequency content (Fig. 2, Table II case II) | `run_traffic_paper_cases.sh`: `--length 35040 17520 8760 4380` |
| K=13, N=1 is a good setting; K=30 over-decomposes (Fig. 3, case III) | the same script, `--K 3 6 9 13 15` |

Two things to keep in mind when reading these:
* The paper's own timing numbers give 267 min → 98.63 s, which is a factor of about 162, while the abstract says 250×.
* The training loss in the paper's Fig. 2 goes to about 0. On CWRU the same thing happened because one α went
  to ≈0 (that mode then passes every frequency and absorbs the residual). Check `learned_bandwidth.png` and the
  mode overlap on traffic before reading a near-zero loss as better separation.

## Caveats
* **Hardware:** VMD here is NumPy on CPU; UVMD runs on `--device`. The paper compared VMD with UVMD on its GPU.
  The CPU timing of the trained UVMD (`(CPU)` rows) gives a like-for-like comparison.
* **VMD settings:** the paper does not state its VMD settings, so the VMD baseline uses the same α (2000) as UVMD's
  start and a convergence tolerance of 1e-7.
* **Shorter windows:** I read "window length" as using only the first L samples of each sensor's year. The paper
  does not say whether it crops or splits the year into chunks.
* **Non-causal:** both methods filter with a global FFT over the whole year, so a mode value at time t depends on
  later samples, including the test period of a forecaster. This does not affect the comparison here, but it matters
  if the modes are used as forecasting inputs.
* **Not tested on the real data here:** I could not download LargeST in my environment. The pipeline was tested
  on generated data in the exact LargeST format (5-minute raw file, metadata, region files, the authors' `his.npz`).
