#!/usr/bin/env bash
# Zero start (all centre frequencies start at 0) for CWRU K=4 and traffic K=6, then the comparison with synthetic.
# Needs the finished uniform-start runs in results/cwru_K4 and results/traffic_K6 (already on the branch).
#
#   git fetch && git checkout fixed-k-modes && git pull
#   bash scripts/run_zero_start_node7.sh
# Optional: DEV=cuda:0  PUSH=0  CWRU_DATA=...  TRAFFIC_DATA=...
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:0}
CWRU_DATA=${CWRU_DATA:-$HOME/data/CWRU}
TRAFFIC_DATA=${TRAFFIC_DATA:-$HOME/data/LargeST/sd_his_2019.h5}
PUSH=${PUSH:-1}
branch=$(git rev-parse --abbrev-ref HEAD)

save() {
  for p in results/cwru_K4_zeroinit results/traffic_K6_zeroinit results/zero_start_comparison; do
    if [ -e "$p" ]; then git add "$p"; fi
  done
  if git diff --cached --quiet; then echo "[save] nothing new for: $1"; return 0; fi
  git commit -q -m "$1"
  if [ "$PUSH" = "1" ]; then
    git pull --rebase --autostash -q origin "$branch" 2>/dev/null || true
    git push -q origin "$branch" && echo "[save] pushed: $1" || echo "[save] PUSH FAILED for: $1 (committed locally)"
  fi
}

python experiments/run_zero_start_real.py cwru --src results/cwru_K4 --data "$CWRU_DATA" --device "$DEV" 2>&1 | tee results/cwru_K4_zeroinit.log
save "node7: CWRU zero start vs uniform start"
python experiments/run_zero_start_real.py traffic --src results/traffic_K6 --data "$TRAFFIC_DATA" --device "$DEV" 2>&1 | tee results/traffic_K6_zeroinit.log
save "node7: traffic zero start vs uniform start"
python experiments/zero_start_compare.py
save "Zero start vs uniform start: synthetic, CWRU and traffic compared"
