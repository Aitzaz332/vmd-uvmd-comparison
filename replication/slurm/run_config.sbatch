#!/bin/bash
#SBATCH --job-name=magn
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=replication/slurm_logs/%x-%j.out
# One MAGN configuration (UVMD with K modes and N layers, then ASTGCN) as a SLURM job on one RTX 4090.
# Submit from the REPO ROOT (the output path is relative to it, and the folder must exist):
#   mkdir -p replication/slurm_logs
#   sbatch --export=ALL,REG=sd,K=13,LAYERS=1 replication/slurm/run_config.sbatch
# It runs replication/scripts/run_isolated.sh (private folder under ~/magn_runs/, upstream code hash-checked), copies
# the logs into replication/results/<tag>/ and then commits and pushes them (best effort, serialised with a lock).
set -euo pipefail
: "${REG:?set REG=sd|gla|gba with --export=ALL,REG=...}"
K=${K:-13}; LAYERS=${LAYERS:-1}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
export PATH="$HOME/.conda/envs/magn/bin:$PATH"          # the 'magn' conda env, without needing conda init on the node
cd "$REPO"

echo "host=$(hostname) job=${SLURM_JOB_ID:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} REG=$REG K=$K N=$LAYERS"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
[ -x /usr/bin/time ] || { echo "/usr/bin/time is missing on this node, stop"; exit 1; }

K=$K LAYERS=$LAYERS bash replication/scripts/run_isolated.sh "$REG" cuda:0

TAG="${REG}_N${LAYERS}"; [ "$K" = "13" ] || TAG="${REG}_K${K}_N${LAYERS}"
python replication/scripts/paper_report.py --results replication/results > /dev/null || true
if [ "${PUSH:-1}" = "1" ]; then
  (
    flock -w 900 9
    git add replication/results
    if ! git diff --cached --quiet; then
      git commit -q -m "replication: $TAG (SLURM job ${SLURM_JOB_ID:-?}, $(hostname))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
      git pull --rebase -q && git push -q && echo "pushed $TAG"
    fi
  ) 9>"$HOME/.magn_git.lock" || echo "PUSH FAILED: results are in replication/results/$TAG, commit them by hand"
fi
echo "done: $TAG"
