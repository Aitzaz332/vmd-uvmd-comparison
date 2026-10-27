#!/usr/bin/env bash
# The paper's ablations (Table II, Fig. 2, Fig. 3) on one region. Run from the repo root:
#   bash scripts/run_traffic_paper_cases.sh <region_h5> <region> [device]
# Bound the cost with CAPS, e.g.  CAPS="--max-train-nodes 100 --max-test-nodes 30"  (default below).
set -euo pipefail
H5=${1:?region h5}; REG=${2:?region label}; DEV=${3:-cuda:0}
CAPS=${CAPS:---max-train-nodes 100 --max-val-nodes 20 --max-test-nodes 30}
# Case II: window length 35040 (full year), 17520, 8760, 4380 samples; K=13, N=1,2
for L in 35040 17520 8760 4380; do
  python experiments/run_traffic.py --data "$H5" --region "$REG" --device "$DEV" --K 13 --length $L $CAPS \
      --out results/traffic_${REG}_length${L}
done
# Case III / Fig. 3: number of modes K (N=1,2 in every run)
for K in 3 6 9 15; do
  python experiments/run_traffic.py --data "$H5" --region "$REG" --device "$DEV" --K $K $CAPS \
      --out results/traffic_${REG}_K${K}
done
