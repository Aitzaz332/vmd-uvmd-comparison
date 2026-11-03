#!/usr/bin/env bash
# alpha_k and lambda over the training epochs of UVMD (uniform and zero start) for CWRU K=4 and traffic K=6.
#   git pull && bash scripts/run_param_trajectory_node7.sh
# Retrains UVMD twice per dataset (uniform start, zero start) with the settings of results/cwru_K4 and results/traffic_K6.
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:0}
CWRU_DATA=${CWRU_DATA:-$HOME/data/CWRU}
TRAFFIC_DATA=${TRAFFIC_DATA:-$HOME/data/LargeST/sd_his_2019.h5}
PUSH=${PUSH:-1}
branch=$(git rev-parse --abbrev-ref HEAD)
save() {
  git add results/param_trajectory
  if git diff --cached --quiet; then echo "[save] nothing new"; return 0; fi
  git commit -q -m "$1"
  if [ "$PUSH" = "1" ]; then
    git pull --rebase --autostash -q origin "$branch" 2>/dev/null || true
    git push -q origin "$branch" && echo "[save] pushed: $1" || echo "[save] PUSH FAILED for: $1 (committed locally)"
  fi
}
python experiments/param_trajectory.py run cwru --src results/cwru_K4 --data "$CWRU_DATA" --device "$DEV"
save "node7: alpha and lambda over epochs, CWRU"
python experiments/param_trajectory.py run traffic --src results/traffic_K6 --data "$TRAFFIC_DATA" --device "$DEV"
python experiments/param_trajectory.py plot
save "node7: alpha and lambda over epochs, traffic and plots"
