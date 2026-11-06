#!/bin/bash
#SBATCH --job-name=magn_plain
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=replication/slurm_logs/%x-%j.out
# Plain ASTGCN (raw flow + tod + dow, no UVMD) on one region. Submit from the repo root:
#   sbatch --export=ALL,REG=sd replication/slurm/run_plain.sbatch
set -euo pipefail
: "${REG:?set REG=sd|gla|gba}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
export PATH="$HOME/.conda/envs/magn/bin:$PATH"
cd "$REPO"
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} REG=$REG"
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
bash replication/scripts/run_plain_astgcn.sh "$REG" cuda:0
echo "done: ${REG}_plain (commit replication/results/${REG}_plain by hand if the push is not set up)"
