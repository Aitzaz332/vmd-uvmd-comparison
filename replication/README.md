# Replication of the MAGN paper (UVMD + ASTGCN on LargeST)

Goal: reproduce Table I of Ahmad et al., arXiv:2509.00703 (MAGN rows for GBA, GLA, SD) with **the authors' own,
unchanged code**, and compare the numbers with the paper.

* `magn/` is a byte-exact copy of https://github.com/OsamaAhmad369/MAGN at commit in `UPSTREAM_COMMIT.txt`
  (`6bfae1bf`). Hashes of all 18 files: `UPSTREAM_SHA256.txt`. Nothing in `magn/` was edited.
* `scripts/` only wraps it: data preparation, the run order from the upstream README, and the comparison.
* `results/<region>_N<layers>/` receives the logs, the trained UVMD (`model.pth`), `comparison.md`.

## Run on node7 (start with SD, the smallest)
```bash
git fetch && git checkout magn-paper-replication
# 1. environment (Python 3.8 and pandas < 2 are required by the upstream data script; install the CUDA torch you use first)
conda create -n magn python=3.8 -y && conda activate magn        # or any python3.8 venv
pip install torch                                                # pick the build matching node7's CUDA
pip install -r replication/requirements-magn.txt
# 2. data (Kaggle account; put kaggle.json in ~/.kaggle yourself, or download in a browser and scp the zip)
pip install kaggle && mkdir -p ~/data/LargeST && cd ~/data/LargeST && kaggle datasets download -d liuxu77/largest && unzip -q largest.zip && cd -
bash replication/scripts/prepare_data.sh ~/data/LargeST sd          # authors' notebooks + generate_data_for_training.py
# 3. UVMD (K=13, N=1) -> ASTGCN train + test -> comparison; commits and pushes results after each stage
tmux new -s magn
bash replication/scripts/run_region.sh sd cuda:0
```
Then `gla` and `gba` the same way (`prepare_data.sh ... gla`, `run_region.sh gla cuda:0`). Useful switches:
`K=...` and `LAYERS=...` (Table II rows), `SEEDS="2023 2024 2025"` (seed spread), `PUSH=0` (commit only), `BS=...`.
Table II / Fig. 3 on SD (K = 3, 6, 9, 15 with N=1, and K = 6, 13 with N=2): `bash replication/scripts/run_table2_sd.sh cuda:0`.

**Output: `results/PAPER_STYLE_REPORT.md`** (rebuilt after every stage by `scripts/paper_report.py`): Table I and Table II in the
paper's layout (Param as `x+y`, MAE/RMSE/MAPE at horizons 3, 6, 12 and average), the paper row directly above the replicated row
with relative differences, the data behind Fig. 2/3 (UVMD reconstruction loss per epoch, learned alpha_k per mode, ASTGCN validation
MAE per epoch) and timings. After a UVMD stage, `scripts/decomp_timing.py` times the decomposition for the paper's 98.63 s claim.

**Not reproducible with the unchanged upstream code** (shown as "not supported" in the report): Table II case I (one shared alpha;
upstream has one alpha per mode) and case II (shorter windows; `uvmd.py` has no length option and the parameter counts in the
paper, 70.093K = 2 x 2 x 17520 + 13, imply a UVMD built for the shorter length, but the paper does not say how the full year
is then decomposed). These two need small non-upstream additions with a stated interpretation; that is a separate step after the main run.
Rows for the other baselines of Table I (LSTM, D2STGNN, PatchSTG, RPMixer, RAGL, VMGCN, CA-VMGCN) are not rerun.

Disk and memory: the unfolded `his.npz` is uncompressed, about 5 GB (GBA), 8.6 GB (GLA), 1.6 GB (SD), and the raw
California file needs about 10 GB RAM to read. Allow roughly 40 GB of disk for all three regions.

### Sanity checks while it runs
* `data/<region>/2019/his.npz` must have shape (35040, N, 3) with N = 2352 / 3834 / 716 (`prepare_data.sh` asserts this).
* `uvmd.py` prints `model.params (M)`; the paper's "140.173K" is **0.140173** (N=1) and 0.210253 (N=2). Parameter
  count is `numel` of the complex lambda (counted as 1 per element), the allocation `2 x 70080` plus 13 alphas.
* `python replication/scripts/paper_report.py` rebuilds the report from the logs at any time.

## Where the upstream material is inconsistent (decisions taken, so the run is reproducible)
| item | paper / README / scripts | used here |
|---|---|---|
| ASTGCN batch size | paper: 48 (SD), 4 (GLA, GBA). README example: 4 (GLA). `experiments/astgcn/run.sh`: 17 / 45 / 64 (leftover from the LargeST baselines) | paper values; override with `BS=` |
| UVMD layers | paper main result N=1 | `LAYERS=1` |
| seed | default 2023 in `args.py` and `run.sh` | 2023 (add more with `SEEDS`) |
| UVMD loss | paper text: squared L2; code: L1 on the one-sided spectrum | the code |
| input channels | `--input_dim 15` (= K+2 in the scripts) = 13 modes + time-of-day + day-of-week; the raw flow is **not** an ASTGCN input, the 16th channel written by `uvmd.py` is zeros and unused | as upstream |
| `--years 2019`, window 12 -> 12, 60/20/20 split | same everywhere | as upstream |

The paper does not report seeds or standard deviations, so "matches the paper" cannot mean exact equality. The
comparison script reports relative differences and how many metrics fall within ±15% (a parameter, not a statistical test).

## Read this before using the results to choose a direction
In upstream `uvmd.py` each sensor's **whole year** (train, validation and test period together) goes through one global
FFT, a zero-phase filter and an inverse FFT. A mode value at time t therefore depends on samples after t, including the
values ASTGCN is asked to predict, and the trained UVMD reconstructs the signal almost perfectly (the paper says so).
The paper's MAE (about 0.6-1.4 against about 19-28 for the same ASTGCN with plain inputs) is the size one would
expect if that information reaches the forecaster. This is a hypothesis from reading the code; it has **not** been
measured on the real data yet. `scripts/leak_check.py` measures it in seconds on the trained UVMD (nudge one later sample,
see whether modes at t move). Run it after the main results are in (it needs the trained UVMD of the SD run):
```bash
python replication/scripts/leak_check.py --his replication/magn/data/sd/2019/his_raw.npz \
       --model replication/results/sd_N1/model.pth --layers 1
```
A faithful replication reproduces the paper's numbers whether or not the check is clean. If the check shows leakage,
the numbers describe a leaky pipeline, not forecasting skill. A causal control (decompose with only past samples at each
forecast origin) would be the next experiment, kept separate from this replica.

## Status
Written and syntax-checked in a sandbox without GPU, PyTorch or the LargeST data. Not yet run on node7. The log parser and report
generator were tested on synthetic logs; `prepare_data.sh`, `run_region.sh`, `run_table2_sd.sh`, `decomp_timing.py` and `leak_check.py` are untested.
