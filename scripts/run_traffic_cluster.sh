#!/usr/bin/env bash
# Staged traffic run on a GPU node. Run from the repo root:
#   bash scripts/run_traffic_cluster.sh <region_h5> <region: gba|gla|sd> [device]
# Stage 1 checks the installation (demo data), stage 2 is a bounded real run, stage 3 the full paper protocol.
set -euo pipefail
H5=${1:?region h5, e.g. ~/data/LargeST/sd_his_2019.h5}
REG=${2:?region label (gba|gla|sd)}
DEV=${3:-cuda:0}
python -m pytest -q
python experiments/run_traffic.py --data demo --quick                                   # stage 1
python experiments/run_traffic.py --data "$H5" --region "$REG" --device "$DEV" \
    --max-train-nodes 100 --max-val-nodes 20 --max-test-nodes 30 --out results/traffic_${REG}_bounded   # stage 2
if [ "${FULL:-0}" = "1" ]; then                                                          # stage 3 (hours)
  python experiments/run_traffic.py --data "$H5" --region "$REG" --device "$DEV" --out results/traffic_${REG}_full
fi
