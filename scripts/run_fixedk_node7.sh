#!/usr/bin/env bash
# CWRU (K=4) and traffic (K=6): train VMD + UVMD with the original reconstruction-only loss at the small K the
# supervisor asked for, then draw the readable VMD | UVMD mode figures. Pushes after every dataset.
#
#   git fetch && git checkout fixed-k-modes && git pull
#   bash scripts/run_fixedk_node7.sh
#
# Data paths default to the ones used before (override with CWRU_DATA=... TRAFFIC_DATA=...).
# Optional: DATASETS=cwru,traffic  K_CWRU=4  K_TRAFFIC=6  DEV=cuda:0  PUSH=0
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=${DEV:-cuda:0}
DATASETS=${DATASETS:-cwru,traffic}
K_CWRU=${K_CWRU:-4}
K_TRAFFIC=${K_TRAFFIC:-6}
CWRU_DATA=${CWRU_DATA:-$HOME/data/CWRU}
TRAFFIC_DATA=${TRAFFIC_DATA:-$HOME/data/LargeST/sd_his_2019.h5}
PUSH=${PUSH:-1}
branch=$(git rev-parse --abbrev-ref HEAD)

save() {
  git add results/cwru_K${K_CWRU} results/traffic_K${K_TRAFFIC} results/*.log 2>/dev/null || true
  if git diff --cached --quiet; then echo "[save] nothing new for: $1"; return 0; fi
  git commit -q -m "$1"
  if [ "$PUSH" = "1" ]; then
    git pull --rebase --autostash -q origin "$branch" 2>/dev/null || true
    git push -q origin "$branch" && echo "[save] pushed: $1" || echo "[save] PUSH FAILED for: $1 (committed locally)"
  fi
}

for ds in ${DATASETS//,/ }; do
  case "$ds" in
    cwru)
      python experiments/run_cwru.py --data "$CWRU_DATA" --K "$K_CWRU" --device "$DEV" \
          --layers 3 5 --vmd-iters 3 5 --sweep-windows 20 --out results/cwru_K${K_CWRU} 2>&1 | tee results/cwru_K${K_CWRU}.log
      python experiments/mode_figures.py cwru --dir results/cwru_K${K_CWRU} --data "$CWRU_DATA"
      save "node7: CWRU K=${K_CWRU}, VMD vs UVMD modes" ;;
    traffic)
      python experiments/run_traffic.py --data "$TRAFFIC_DATA" --region sd --K "$K_TRAFFIC" --device "$DEV" \
          --max-train-nodes 100 --max-test-nodes 30 --layers 3 5 --vmd-iters 3 5 --sweep-nodes 3 \
          --out results/traffic_K${K_TRAFFIC} 2>&1 | tee results/traffic_K${K_TRAFFIC}.log
      python experiments/mode_figures.py traffic --dir results/traffic_K${K_TRAFFIC} --data "$TRAFFIC_DATA"
      save "node7: traffic K=${K_TRAFFIC}, VMD vs UVMD modes" ;;
    *) echo "unknown dataset $ds" ;;
  esac
done
echo "done. Figures: results/cwru_K${K_CWRU}/readable/  results/traffic_K${K_TRAFFIC}/readable/"
