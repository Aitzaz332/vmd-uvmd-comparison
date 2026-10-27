#!/usr/bin/env bash
# Run every stage that is still missing on a GPU node (AU gpucluster2 node7), build the cross-dataset
# comparison, and commit + push all outputs to GitHub. Each stage is saved as soon as it finishes,
# so a long run never loses results. Run from anywhere:
#
#   H5=~/data/LargeST/sd_his_2019.h5 REG=sd bash scripts/run_all_node7.sh
#
# Optional environment variables
#   CWRU=~/data/CWRU   folder with the 40 CWRU .mat files (downloaded if missing)
#   DEV=cuda:0         device for UVMD
#   FULL=1             also run the full traffic protocol (hours) and write results/comparison_full
#   FORCE=1            rerun stages whose results/<name>/summary.json already exists
#   PUSH=0             commit only, do not push
set -euo pipefail
cd "$(dirname "$0")/.."

CWRU=${CWRU:-$HOME/data/CWRU}
H5=${H5:?set H5 to the region file, e.g. H5=~/data/LargeST/sd_his_2019.h5}
REG=${REG:-sd}
DEV=${DEV:-cuda:0}
PUSH=${PUSH:-1}
FORCE=${FORCE:-0}

want() { [ "$FORCE" = "1" ] || [ ! -f "$1/summary.json" ]; }

save() {  # save "message": commit everything under results/ and push it
  git add results
  if git diff --cached --quiet; then echo "[save] nothing new for: $1"; return 0; fi
  git commit -q -m "$1"
  if [ "$PUSH" = "1" ]; then
    git pull --rebase -q && git push -q && echo "[save] pushed: $1" || echo "[save] PUSH FAILED for: $1 (committed locally)"
  fi
}

python -m pytest -q

if want results/synthetic_K13; then
  python experiments/run_synthetic.py --device "$DEV" --K 13 --out results/synthetic_K13 2>&1 | tee results/synthetic_K13.log
  save "node7: synthetic benchmark, K=13"
fi

if want results/cwru_K13; then
  [ "$(ls "$CWRU"/*.mat 2>/dev/null | wc -l)" -ge 40 ] || bash scripts/download_cwru.sh "$CWRU"
  python experiments/run_cwru.py --data "$CWRU" --device "$DEV" --K 13 --out results/cwru_K13 2>&1 | tee results/cwru_K13.log
  save "node7: CWRU benchmark, K=13"
fi

if want results/traffic_${REG}_bounded; then
  python experiments/run_traffic.py --data "$H5" --region "$REG" --device "$DEV" \
      --max-train-nodes 100 --max-val-nodes 20 --max-test-nodes 30 \
      --out results/traffic_${REG}_bounded 2>&1 | tee results/traffic_${REG}_bounded.log
  save "node7: LargeST traffic (${REG}), bounded run, K=13"
fi

python experiments/compare_datasets.py --traffic results/traffic_${REG}_bounded --out results/comparison
save "node7: cross-dataset comparison table and figure"

if [ "${FULL:-0}" = "1" ] && want results/traffic_${REG}_full; then
  python experiments/run_traffic.py --data "$H5" --region "$REG" --device "$DEV" \
      --out results/traffic_${REG}_full 2>&1 | tee results/traffic_${REG}_full.log
  save "node7: LargeST traffic (${REG}), full protocol, K=13"
  python experiments/compare_datasets.py --traffic results/traffic_${REG}_full --out results/comparison_full
  save "node7: cross-dataset comparison with the full traffic run"
fi

echo "done. Tables: results/comparison/comparison_table.md   Figure: results/comparison/comparison.png"
