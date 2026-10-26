#!/usr/bin/env bash
# Full CWRU benchmark on a GPU node (e.g. AU gpucluster2 node7). Run from the repo root:
#   bash scripts/run_cwru_cluster.sh [data_dir] [device]
set -euo pipefail
DATA=${1:-$HOME/data/CWRU}
DEV=${2:-cuda:0}
[ "$(ls "$DATA"/*.mat 2>/dev/null | wc -l)" -ge 40 ] || bash scripts/download_cwru.sh "$DATA"
python -m pytest -q
python experiments/run_cwru.py --data "$DATA" --device "$DEV" --K 4  --out results/cwru     2>&1 | tee results/cwru_K4.log
python experiments/run_cwru.py --data "$DATA" --device "$DEV" --K 13 --out results/cwru_K13 2>&1 | tee results/cwru_K13.log
