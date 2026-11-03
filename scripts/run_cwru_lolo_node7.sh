#!/usr/bin/env bash
# Leave-one-load-out comparison on a GPU node (AU gpucluster2 node7): converged VMD -> 1D-CNN vs UVMD trained end to end
# with the same CNN, each motor load (0-3 HP) held out once. Writes per-fold results, the clean mode / alpha figures and a
# summary over folds, then commits them to the branch e2e-uvmd-cnn and pushes. Run from the repository folder:
#
#   git fetch origin && git checkout e2e-uvmd-cnn && git pull
#   ARGS="--quick" bash scripts/run_cwru_lolo_node7.sh      # smoke test, nothing committed
#   bash scripts/run_cwru_lolo_node7.sh                     # full run
#
# Optional environment variables
#   CWRU=~/data/CWRU       folder with the 40 CWRU .mat files (downloaded if missing)
#   DEV=cuda:0             device
#   K=4  LAYERS=5  ALPHA=500   modes, UVMD layers, VMD / initial UVMD bandwidth penalty
#   SEEDS="0 1 2"          one CNN (and one end-to-end UVMD) per seed and fold
#   FOLDS="0 1 2 3"        held-out test loads
#   OUT=results/cwru_lolo_K${K}_N${LAYERS}   output folder
#   ARGS="..."             extra arguments for run_cwru_e2e.py (e.g. --lr-alpha 0.5, --epochs 60, --quick)
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
FOLDS=${FOLDS:-"0 1 2 3"}
OUT=${OUT:-results/cwru_lolo_K${K}_N${LAYERS}}
PUSH=${PUSH:-1}

python - <<'PY'
import importlib.util, sys
missing = [m for m in ("torch", "numpy", "scipy", "sklearn", "pandas", "matplotlib", "tabulate", "pytest") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing Python packages: " + ", ".join(missing) + "  ->  pip install " + " ".join("scikit-learn" if m == "sklearn" else m for m in missing))
PY
[ "$(ls "$CWRU"/*.mat 2>/dev/null | wc -l)" -ge 40 ] || bash scripts/download_cwru.sh "$CWRU"
python -m pytest -q tests/test_uvmd_torch.py tests/test_e2e.py tests/test_compare_uvmd_modes.py tests/test_compare_predictions.py tests/test_lolo.py

mkdir -p "$OUT"
python experiments/run_cwru_lolo.py --data "$CWRU" --device "$DEV" --K "$K" --layers "$LAYERS" --alpha "$ALPHA" \
    --seeds $SEEDS --folds $FOLDS --out "$OUT" ${ARGS:-} 2>&1 | tee "$OUT/run.log"

if [[ "${ARGS:-}" == *--quick* ]]; then
  echo "done (--quick smoke run: results are in $OUT and are NOT committed)."
  exit 0
fi

git add "$OUT"
if git diff --cached --quiet; then
  echo "[save] nothing new"
else
  git commit -q -m "node7: CWRU leave-one-load-out, VMD vs end-to-end UVMD (K=$K, N=$LAYERS, alpha=$ALPHA, folds: $FOLDS, seeds: $SEEDS)"
  if [ "$PUSH" = "1" ]; then
    git push -q origin "$BRANCH" && echo "[save] pushed to $BRANCH" \
      || echo "[save] PUSH FAILED (committed locally): try  git pull --rebase origin $BRANCH && git push origin $BRANCH"
  fi
fi
echo "done. $OUT/lolo_summary.md  $OUT/lolo_summary.png   per fold: $OUT/test_<L>HP/{modes_figures,training_curves,prediction_comparison}"
