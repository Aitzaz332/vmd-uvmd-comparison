#!/usr/bin/env bash
# End-to-end UVMD + 1D-CNN on CWRU, on a GPU node (AU gpucluster2 node7).
# Trains the same 1D-CNN on (a) converged-VMD modes (not updated) and (b) modes of a 5-layer UVMD trained end to end
# by the CNN loss (+ small reconstruction term); writes the loss / accuracy / confusion-matrix comparison, exports the
# modes, compares them with VMD (and with a recon-trained UVMD if RECON is set), then commits the results to the branch e2e-uvmd-cnn and pushes them. Run from anywhere:
#
#   git fetch origin && git checkout e2e-uvmd-cnn && git pull
#   bash scripts/run_cwru_e2e_node7.sh
#
# Optional environment variables
#   CWRU=~/data/CWRU       folder with the 40 CWRU .mat files (downloaded if missing)
#   DEV=cuda:0             device
#   K=4  LAYERS=5  ALPHA=500   modes, UVMD layers, initial / VMD bandwidth penalty
#   SEEDS="0 1 2"          one CNN (and one end-to-end UVMD) per seed
#   RECON=path/uvmd_params_N5.npz   UVMD trained only on reconstruction (run_cwru.py, same K/N/alpha) to compare modes with
#   LABELS=fault10         or fault4
#   OUT=results/cwru_e2e_K${K}_N${LAYERS}   output folder
#   ARGS="--quick"         extra arguments for run_cwru_e2e.py (e.g. a smoke run first, or --lr-alpha 0.5)
#   PUSH=0                 commit only, do not push
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=e2e-uvmd-cnn
[ "$(git branch --show-current)" = "$BRANCH" ] || { echo "run this on branch $BRANCH:  git checkout $BRANCH"; exit 1; }

CWRU=${CWRU:-$HOME/data/CWRU}
DEV=${DEV:-cuda:0}
K=${K:-4}
LAYERS=${LAYERS:-5}
ALPHA=${ALPHA:-500}
SEEDS=${SEEDS:-"0 1 2"}
LABELS=${LABELS:-fault10}
OUT=${OUT:-results/cwru_e2e_K${K}_N${LAYERS}}
PUSH=${PUSH:-1}

[ "$(ls "$CWRU"/*.mat 2>/dev/null | wc -l)" -ge 40 ] || bash scripts/download_cwru.sh "$CWRU"
python -m pytest -q tests/test_uvmd_torch.py tests/test_e2e.py tests/test_compare_uvmd_modes.py tests/test_compare_predictions.py

CMP=()
[ -n "${RECON:-}" ] && CMP=(--early-params "$RECON")
mkdir -p "$OUT"
python experiments/run_cwru_e2e.py --data "$CWRU" --device "$DEV" --K "$K" --layers "$LAYERS" --alpha "$ALPHA" --labels "$LABELS" \
    --seeds $SEEDS --out "$OUT" ${ARGS:-} 2>&1 | tee "$OUT/run.log"
python experiments/plot_training_curves.py --run "$OUT" 2>&1 | tee "$OUT/plot_curves.log"
python experiments/compare_predictions.py --run "$OUT" 2>&1 | tee "$OUT/compare_predictions.log"
python experiments/compare_uvmd_modes.py --run "$OUT" --data "$CWRU" --device "$DEV" ${CMP[@]+"${CMP[@]}"} 2>&1 | tee "$OUT/compare.log"

if [[ "${ARGS:-}" == *--quick* ]]; then
  echo "done (--quick smoke run: results are in $OUT and are NOT committed)."
  exit 0
fi

git add "$OUT"
if git diff --cached --quiet; then
  echo "[save] nothing new"
else
  git commit -q -m "node7: CWRU end-to-end UVMD + 1D-CNN (K=$K, N=$LAYERS, alpha=$ALPHA, $LABELS, seeds: $SEEDS)"
  if [ "$PUSH" = "1" ]; then
    git pull --rebase -q origin "$BRANCH" && git push -q origin "$BRANCH" && echo "[save] pushed to $BRANCH" \
      || echo "[save] PUSH FAILED (committed locally)"
  fi
fi
echo "done. $OUT/prediction_comparison/prediction_comparison.md  $OUT/summary.md   $OUT/mode_comparison/mode_comparison.md"
