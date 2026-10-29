#!/usr/bin/env bash
# Train UVMD with the separation-aware loss on the three datasets (GPU node, e.g. AU gpucluster2 node7), score it with the
# earlier criteria and push every finished stage to GitHub (branch separation-loss). Needs the benchmark results of the
# main branch in results/ (summary.json and uvmd_params_N*.npz of each dataset) and the data files, like run_all_node7.sh.
#
#   git fetch && git checkout separation-loss && git pull
#   bash scripts/run_sep_loss_node7.sh
#
# The data paths are read from the earlier runs' summary.json, so nothing has to be typed. Optional environment variables
#   DATASETS=synthetic,cwru,traffic   which datasets to run
#   VARIANTS=bw,bw+ortho,full         which loss variants
#   DEV=cuda:0   FORCE=1   PUSH=0
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=${DEV:-cuda:0}
DATASETS=${DATASETS:-synthetic,cwru,traffic}
VARIANTS=${VARIANTS:-bw,bw+ortho,full}
PUSH=${PUSH:-1}
FORCE=${FORCE:-0}
branch=$(git rev-parse --abbrev-ref HEAD)

save() {
  git add results/sep_loss
  if git diff --cached --quiet; then echo "[save] nothing new for: $1"; return 0; fi
  git commit -q -m "$1"
  if [ "$PUSH" = "1" ]; then
    git pull --rebase --autostash -q origin "$branch" 2>/dev/null || true
    git push -q origin "$branch" && echo "[save] pushed: $1" || echo "[save] PUSH FAILED for: $1 (committed locally)"
  fi
}

python -m pytest -q tests/test_uvmd_sep.py tests/test_separation_comparison.py

for ds in ${DATASETS//,/ }; do
  extra=""; [ "$FORCE" = "1" ] && extra="--force"
  python experiments/run_sep_loss.py --datasets "$ds" --variants "$VARIANTS" --device "$DEV" --quiet $extra \
      --out results/sep_loss 2>&1 | tee "results/sep_loss_${ds}.log"
  save "node7: separation-aware loss, $ds"
done

# one table and figure over everything that has been run (re-reads the saved models, no retraining)
python experiments/run_sep_loss.py --datasets "$DATASETS" --variants "$VARIANTS" --device "$DEV" --quiet \
    --out results/sep_loss 2>&1 | tee results/sep_loss_all.log
save "node7: separation-aware loss, comparison of all datasets"
echo "done. Table: results/sep_loss/sep_loss.md   Figures: results/sep_loss/comparison_bars.png, synthetic_truth_bars.png"
